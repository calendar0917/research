"""Fast supervised-teacher latent KSVD probe on an internal MolHIV scaffold fold.

This is a compression probe, not a terminal benchmark.  A small GINE teacher is
fit only on the outer-fit graphs, its node states are frozen, and KSVD/PCA/random
codes are used as matched graph-level compressed readouts.  Official valid/test
graphs are never loaded when a fold cache is supplied.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd


def seed_all(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(bundle, indices, torch, Data):
    out = {}
    for raw in indices:
        i = int(raw)
        g = bundle.graphs[i]
        src, dst, attrs = [], [], []
        for u, v in sorted(g.edges()):
            a = np.asarray(bundle.edge_feats[i][(u, v)], dtype=np.int64)
            src += [u, v]
            dst += [v, u]
            attrs += [a, a]
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(np.stack(attrs), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)
        out[i] = Data(
            x=torch.tensor(bundle.node_feats[i], dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([float(bundle.y[i])], dtype=torch.float32),
            graph_id=torch.tensor([i], dtype=torch.long),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=5)
    ap.add_argument("--teacher-hidden", type=int, default=32)
    ap.add_argument("--teacher-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--max-node-samples", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool

    t0 = time.time()
    seed_all(args.seed, torch)
    device = torch.device(args.device)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    z = np.load(args.fold_cache)
    fit_idx = np.asarray(z[f"fold_{args.fold}_train_indices"], dtype=np.int64)
    held_idx = np.asarray(z[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
    all_idx = np.concatenate([fit_idx, held_idx])
    data = make_data(bundle, all_idx, torch, Data)

    def loader(indices, shuffle, offset):
        gen = torch.Generator().manual_seed(args.seed + 1009 + offset) if shuffle else None
        return DataLoader(
            [data[int(i)] for i in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=gen,
        )

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            h = args.teacher_hidden
            self.atom = AtomEncoder(h)
            self.bond = BondEncoder(h)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(args.teacher_layers):
                mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(h))
            self.head = nn.Linear(h, 1)

        def forward(self, d, return_nodes=False):
            x = self.atom(d.x)
            e = self.bond(d.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, d.edge_index, e)))
            graph = global_mean_pool(x, d.batch)
            out = self.head(graph).view(-1)
            return (out, x) if return_nodes else out

    model = Teacher().to(device)
    pos = float(np.sum(bundle.y[fit_idx]))
    neg = float(len(fit_idx) - pos)
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        total = seen = 0
        for batch in loader(fit_idx, True, epoch):
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            seen += len(y)
        train_losses.append(total / max(seen, 1))
        print(f"teacher epoch={epoch:02d} loss={train_losses[-1]:.6f}", flush=True)

    @torch.no_grad()
    def extract(indices):
        model.eval()
        node_states, graph_logits, labels = {}, {}, {}
        for batch in loader(indices, False, 900):
            batch = batch.to(device)
            logits, nodes = model(batch, return_nodes=True)
            ptr = batch.ptr.detach().cpu().numpy()
            gids = batch.graph_id.view(-1).detach().cpu().numpy()
            for j, gid in enumerate(gids):
                node_states[int(gid)] = nodes[ptr[j] : ptr[j + 1]].cpu().numpy().astype(np.float64)
                graph_logits[int(gid)] = float(logits[j].cpu())
                labels[int(gid)] = int(batch.y[j].cpu())
        return node_states, graph_logits, labels

    node_states, graph_logits, labels = extract(all_idx)
    rng = np.random.default_rng(args.seed + 77)
    train_rows = [node_states[int(i)] for i in fit_idx]
    Y = np.concatenate(train_rows, axis=0)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    if len(Y) > args.max_node_samples:
        Y = Y[rng.choice(len(Y), size=args.max_node_samples, replace=False)]
    Ymat = Y.T
    D, _, info = ksvd(
        Ymat,
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        n_iter=args.ksvd_iter,
        seed=args.seed,
    )
    pca = PCA(n_components=args.n_atoms, random_state=args.seed).fit(Y)
    D_random = Y[rng.choice(len(Y), size=args.n_atoms, replace=len(Y) < args.n_atoms)].T
    D_random /= np.maximum(np.linalg.norm(D_random, axis=0, keepdims=True), 1e-12)

    def graph_features(dictionary, gid):
        x = node_states[int(gid)]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if dictionary is None:
            code = pca.transform(x).T
        else:
            code = np.stack([_omp(dictionary, row, args.sparsity) for row in x], axis=1)
        # Preserve local occurrence information without a neural readout.
        return np.concatenate([np.max(np.abs(code), axis=1), np.mean(code, axis=1)])

    def evaluate(kind):
        if kind == "pca":
            feats = np.stack([graph_features(None, i) for i in all_idx])
        elif kind == "ksvd":
            feats = np.stack([graph_features(D, i) for i in all_idx])
        elif kind == "random":
            feats = np.stack([graph_features(D_random, i) for i in all_idx])
        else:
            raise ValueError(kind)
        scaler = StandardScaler().fit(feats[: len(fit_idx)])
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=args.seed)
        clf.fit(scaler.transform(feats[: len(fit_idx)]), bundle.y[fit_idx].astype(int))
        prob = clf.predict_proba(scaler.transform(feats[len(fit_idx) :]))[:, 1]
        y = bundle.y[held_idx].astype(int)
        return {
            "auc": float(roc_auc_score(y, prob)),
            "ap": float(average_precision_score(y, prob)),
            "feature_dim": int(feats.shape[1]),
        }

    def distill(kind):
        if kind == "pca":
            feats = np.stack([graph_features(None, i) for i in all_idx])
        elif kind == "ksvd":
            feats = np.stack([graph_features(D, i) for i in all_idx])
        elif kind == "random":
            feats = np.stack([graph_features(D_random, i) for i in all_idx])
        else:
            raise ValueError(kind)
        scaler = StandardScaler().fit(feats[: len(fit_idx)])
        teacher_fit = np.asarray([graph_logits[int(i)] for i in fit_idx], dtype=np.float64)
        teacher_held = np.asarray([graph_logits[int(i)] for i in held_idx], dtype=np.float64)
        student = Ridge(alpha=1.0).fit(scaler.transform(feats[: len(fit_idx)]), teacher_fit)
        pred = student.predict(scaler.transform(feats[len(fit_idx) :]))
        y = bundle.y[held_idx].astype(int)
        return {
            "teacher_logit_mse": float(np.mean((pred - teacher_held) ** 2)),
            "teacher_logit_pearson": float(np.corrcoef(pred, teacher_held)[0, 1]),
            "teacher_logit_spearman": float(spearmanr(pred, teacher_held).statistic),
            "student_label_auc": float(roc_auc_score(y, pred)),
        }

    result = {
        "protocol_id": "molhiv-teacher-latent-ksvd-fast-probe-v1",
        "scope": "official-train internal scaffold fold only",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "fold": args.fold,
        "fit_graphs": int(len(fit_idx)),
        "heldout_graphs": int(len(held_idx)),
        "teacher": {
            "hidden": args.teacher_hidden,
            "layers": args.teacher_layers,
            "epochs": args.teacher_epochs,
            "train_losses": train_losses,
            "fit_auc": float(roc_auc_score(bundle.y[fit_idx], [1 / (1 + np.exp(-graph_logits[int(i)])) for i in fit_idx])),
            "heldout_auc": float(roc_auc_score(bundle.y[held_idx], [1 / (1 + np.exp(-graph_logits[int(i)])) for i in held_idx])),
        },
        "ksvd": {"n_atoms": args.n_atoms, "sparsity": args.sparsity, "iterations": args.ksvd_iter, **info},
        "probe": {name: evaluate(name) for name in ("ksvd", "pca", "random")},
        "distillation": {name: distill(name) for name in ("ksvd", "pca", "random")},
        "elapsed_sec": time.time() - t0,
        "config": vars(args),
    }
    result["probe_deltas"] = {
        "ksvd_minus_pca": result["probe"]["ksvd"]["auc"] - result["probe"]["pca"]["auc"],
        "ksvd_minus_random": result["probe"]["ksvd"]["auc"] - result["probe"]["random"]["auc"],
    }
    result["distillation_deltas"] = {
        "ksvd_minus_pca_spearman": result["distillation"]["ksvd"]["teacher_logit_spearman"] - result["distillation"]["pca"]["teacher_logit_spearman"],
        "ksvd_minus_random_spearman": result["distillation"]["ksvd"]["teacher_logit_spearman"] - result["distillation"]["random"]["teacher_logit_spearman"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
