"""Low-capacity occurrence residual on a frozen real-prototype MIL member.

This is a strict official-train-only scaffold-fold development runner.  It
loads predictions from an already trained stable graph-balanced real-prototype
MIL member and never changes that base model.  Frozen prototypes are rebuilt
from the audited caches, and per-prototype connected-support statistics supply
a small graph-level residual capped in logit space.

Controls:
  real      persistent prototype identity is retained;
  shuffled  prototype rows are permuted independently for every graph;
  no_id     prototype rows are sorted by occurrence mass, removing identity.

Official valid/test rows must be unencoded and are never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def sparse_positive_codes(z: np.ndarray, prototypes: np.ndarray, sparsity: int) -> np.ndarray:
    cosine = np.asarray(z, dtype=np.float32) @ np.asarray(prototypes, dtype=np.float32).T
    positive = np.maximum(cosine, 0.0)
    if sparsity < positive.shape[1]:
        idx = np.argpartition(positive, -sparsity, axis=1)[:, -sparsity:]
        codes = np.zeros_like(positive)
        rows = np.arange(len(positive))[:, None]
        codes[rows, idx] = positive[rows, idx]
    else:
        codes = positive.copy()
    empty = np.flatnonzero(codes.max(axis=1) <= 0.0)
    if len(empty):
        nearest = cosine[empty].argmax(axis=1)
        codes[empty, nearest] = 1e-6
    return codes.astype(np.float32, copy=False)


def occurrence_features(graph: Any, codes: np.ndarray) -> np.ndarray:
    """Return [prototype, 6] connected-support statistics.

    Columns are log component count, log active-node count, log largest
    component size, singleton fraction, log total coefficient mass, and log
    maximum component coefficient mass.
    """
    n_prototypes = int(codes.shape[1])
    out = np.zeros((n_prototypes, 6), dtype=np.float32)
    for proto in range(n_prototypes):
        active = np.flatnonzero(codes[:, proto] > 0.0)
        if len(active) == 0:
            continue
        unseen = set(int(x) for x in active.tolist())
        sizes: list[int] = []
        masses: list[float] = []
        while unseen:
            start = unseen.pop()
            stack = [start]
            component = [start]
            while stack:
                u = stack.pop()
                for raw_v in graph.neighbors(u):
                    v = int(raw_v)
                    if v in unseen:
                        unseen.remove(v)
                        stack.append(v)
                        component.append(v)
            comp = np.asarray(component, dtype=np.int64)
            sizes.append(int(len(comp)))
            masses.append(float(codes[comp, proto].sum()))
        size_np = np.asarray(sizes, dtype=np.float32)
        mass_np = np.asarray(masses, dtype=np.float32)
        out[proto] = np.asarray([
            np.log1p(len(sizes)),
            np.log1p(len(active)),
            np.log1p(size_np.max()),
            np.mean(size_np == 1),
            np.log1p(codes[active, proto].sum()),
            np.log1p(mass_np.max()),
        ], dtype=np.float32)
    return out


def transform_features(
    raw: np.ndarray, control: str, graph_indices: np.ndarray, shuffle_seed: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for feat, graph_i in zip(raw, graph_indices):
        if control == "real":
            ordered = feat
        elif control == "shuffled":
            permutation = np.random.default_rng(
                int(shuffle_seed) + 1_000_003 * int(graph_i)
            ).permutation(feat.shape[0])
            ordered = feat[permutation]
        elif control == "no_id":
            # A canonical, identity-free ordering.  It preserves the multiset of
            # occurrence shapes but not which fixed prototype produced a row.
            order = np.lexsort((
                -feat[:, 0], -feat[:, 2], -feat[:, 1], -feat[:, 4],
            ))
            ordered = feat[order]
        else:
            raise ValueError(f"unknown control: {control}")
        rows.append(ordered.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def prediction_arrays(base: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = base["results"]["fixed_random"][split]
    for key in ("graph_indices", "labels", "scores"):
        if key not in row:
            raise ValueError(f"base result lacks saved {split}.{key}")
    return (
        np.asarray(row["graph_indices"], dtype=np.int64),
        np.asarray(row["labels"], dtype=np.float32),
        np.asarray(row["scores"], dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--base-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--prototype-seed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controls", default="real,shuffled,no_id")
    ap.add_argument("--candidate-bank-size", type=int, default=256)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--residual-cap", type=float, default=0.25)
    ap.add_argument("--residual-epochs", type=int, default=20)
    ap.add_argument("--residual-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--residual-l2", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--shuffle-seed", type=int, default=314159)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    if not controls or len(set(controls)) != len(controls) or not set(controls) <= {"real", "shuffled", "no_id"}:
        raise ValueError(f"invalid controls: {controls}")
    if args.fold < 0 or min(args.candidate_bank_size, args.n_prototypes, args.sparsity) <= 0:
        raise ValueError("invalid fold/prototype configuration")
    if args.n_prototypes > args.candidate_bank_size or args.sparsity > args.n_prototypes:
        raise ValueError("invalid prototype/sparsity relation")
    if args.residual_cap <= 0 or args.residual_epochs <= 0 or args.hidden <= 0:
        raise ValueError("invalid residual configuration")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(f"missing training dependencies: {exc}") from exc

    t0 = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=False,
    )
    labels = np.asarray(bundle.y, dtype=np.float32)
    original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as f:
        fit_indices = np.asarray(f[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(f[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if not np.array_equal(np.asarray(f["original_indices"], dtype=np.int64), original):
            raise ValueError("fold cache original indices mismatch")
        if not np.array_equal(np.asarray(f["official_train_indices"], dtype=np.int64), official_train):
            raise ValueError("fold cache official train mismatch")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit/heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
        raise AssertionError("scaffold fold does not partition official train")

    with np.load(args.latent_cache, allow_pickle=False) as f:
        offsets = np.asarray(f["offsets"], dtype=np.int64)
        latents = np.asarray(f["latents"], dtype=np.float32)
        latent_original = np.asarray(f["original_indices"], dtype=np.int64)
        latent_train = np.asarray(f["train_indices"], dtype=np.int64)
    with np.load(args.token_cache, allow_pickle=False) as f:
        token_offsets = np.asarray(f["offsets"], dtype=np.int64)
        token_original = np.asarray(f["original_indices"], dtype=np.int64)
    if not np.array_equal(latent_original, original) or not np.array_equal(token_original, original):
        raise ValueError("cache original indices mismatch")
    if not np.array_equal(latent_train, official_train) or not np.array_equal(token_offsets, offsets):
        raise ValueError("cache split/offset mismatch")
    nontrain_rows = np.concatenate([
        np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents[nontrain_rows] != 0):
        raise AssertionError("official valid/test latents must remain unencoded")

    token_meta = json.loads(Path(args.token_cache).with_suffix(".json").read_text())
    latent_meta = json.loads(Path(args.latent_cache).with_suffix(".json").read_text())
    expected_fit_sha = sha256(fit_indices)
    if token_meta.get("dictionary_fit_indices_sha256") != expected_fit_sha:
        raise ValueError("token dictionary was not fit on this outer fold")
    if bool(token_meta.get("encoded_official_valid", True)):
        raise ValueError("token cache encoded official valid")
    if bool(latent_meta.get("official_valid_encoded", True)) or bool(latent_meta.get("official_test_encoded", True)):
        raise ValueError("latent cache encoded official valid/test")

    base = json.loads(Path(args.base_result).read_text())
    base_cfg = base.get("config", {})
    required_cfg = {
        "fold": args.fold,
        "prototype_seed": args.prototype_seed,
        "seed": 0,
        "max_graphs": args.max_graphs,
        "candidate_bank_size": args.candidate_bank_size,
        "n_prototypes": args.n_prototypes,
        "sparsity": args.sparsity,
    }
    for key, expected in required_cfg.items():
        if base_cfg.get(key) != expected:
            raise ValueError(f"base config mismatch for {key}: {base_cfg.get(key)} != {expected}")
    fit_graphs, fit_y, fit_base = prediction_arrays(base, "fit")
    held_graphs, held_y, held_base = prediction_arrays(base, "heldout")
    if not np.array_equal(fit_graphs, fit_indices) or not np.array_equal(held_graphs, heldout_indices):
        raise ValueError("base prediction order does not match fold cache")
    if not np.array_equal(fit_y, labels[fit_indices]) or not np.array_equal(held_y, labels[heldout_indices]):
        raise ValueError("base labels do not match dataset")
    saved_base_auc = float(base["results"]["fixed_random"]["heldout"]["auc"])
    if abs(roc_auc_score(held_y, held_base) - saved_base_auc) > 1e-7:
        raise ValueError("saved base AUC is not reproducible")

    # Rebuild exactly the graph-balanced fixed-random prototype bank.
    rng = np.random.default_rng(args.prototype_seed + 910_003 * (args.fold + 1))
    source_pos = rng.choice(len(fit_indices), size=args.candidate_bank_size, replace=False).astype(np.int64)
    source_graphs = fit_indices[source_pos]
    candidate_rows = np.asarray([
        rng.integers(int(offsets[i]), int(offsets[i + 1])) for i in source_graphs
    ], dtype=np.int64)
    candidate_bank = normalize_rows(latents[candidate_rows])
    selected = rng.choice(args.candidate_bank_size, size=args.n_prototypes, replace=False).astype(np.int64)
    prototypes = candidate_bank[selected]
    saved_selection = base["results"]["fixed_random"]["selection"]
    if saved_selection.get("candidate_indices") != selected.tolist():
        raise ValueError("reconstructed fixed-random selection differs from base")
    if saved_selection.get("prototype_sha256") != sha256(prototypes):
        raise ValueError("reconstructed prototype hash differs from base")

    all_graphs = np.concatenate([fit_graphs, held_graphs])
    raw_rows: list[np.ndarray] = []
    for j, raw_i in enumerate(all_graphs.tolist()):
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        codes = sparse_positive_codes(latents[lo:hi], prototypes, args.sparsity)
        raw_rows.append(occurrence_features(bundle.graphs[i], codes))
        if (j + 1) % 1000 == 0 or j + 1 == len(all_graphs):
            print(f"occurrence features {j + 1}/{len(all_graphs)}", flush=True)
    raw = np.asarray(raw_rows, dtype=np.float32)
    n_fit = len(fit_graphs)

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )
            self.gate = nn.Parameter(torch.zeros(()))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.zeros_(self.gate)

        def forward(self, features):
            raw_residual = self.net(features).view(-1)
            amplitude = args.residual_cap * torch.tanh(self.gate)
            return amplitude * torch.tanh(raw_residual)

    results: dict[str, Any] = {
        "base": {
            "fit_auc": float(roc_auc_score(fit_y, fit_base)),
            "heldout_auc": saved_base_auc,
            "fit_logits": fit_base.astype(np.float32).tolist(),
            "heldout_logits": held_base.astype(np.float32).tolist(),
            "fit_probabilities": sigmoid_np(fit_base).astype(np.float32).tolist(),
            "heldout_probabilities": sigmoid_np(held_base).astype(np.float32).tolist(),
        }
    }

    for control_i, control in enumerate(controls):
        features = transform_features(raw, control, all_graphs, args.shuffle_seed)
        fit_x_np, held_x_np = features[:n_fit], features[n_fit:]
        mean = fit_x_np.mean(axis=0)
        std = fit_x_np.std(axis=0)
        std = np.where(std >= 1e-5, std, 1.0).astype(np.float32)
        fit_x_np = np.clip((fit_x_np - mean) / std, -5.0, 5.0).astype(np.float32)
        held_x_np = np.clip((held_x_np - mean) / std, -5.0, 5.0).astype(np.float32)

        seed_all(args.seed, torch)
        model = ResidualMLP(fit_x_np.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.residual_lr, weight_decay=args.weight_decay
        )
        dataset = TensorDataset(
            torch.from_numpy(fit_x_np),
            torch.from_numpy(fit_base),
            torch.from_numpy(fit_y),
        )
        generator = torch.Generator().manual_seed(args.seed + 1000)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        history: list[dict[str, float]] = []
        for epoch in range(1, args.residual_epochs + 1):
            model.train()
            total = 0.0
            total_bce = 0.0
            seen = 0
            for x_batch, base_batch, y_batch in loader:
                delta = model(x_batch)
                bce = F.binary_cross_entropy_with_logits(base_batch + delta, y_batch)
                penalty = args.residual_l2 * delta.square().mean()
                loss = bce + penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                n = int(len(y_batch))
                total += float(loss.detach()) * n
                total_bce += float(bce.detach()) * n
                seen += n
            history.append({
                "epoch": epoch,
                "loss": total / max(seen, 1),
                "bce": total_bce / max(seen, 1),
                "gate": float(model.gate.detach()),
                "amplitude": float(args.residual_cap * torch.tanh(model.gate.detach())),
            })
            if epoch == 1 or epoch % 5 == 0 or epoch == args.residual_epochs:
                print(
                    f"{control} epoch={epoch:02d} loss={history[-1]['loss']:.6f} "
                    f"amplitude={history[-1]['amplitude']:+.5f}", flush=True,
                )

        model.eval()
        with torch.no_grad():
            fit_delta = model(torch.from_numpy(fit_x_np)).numpy().astype(np.float32)
            held_delta = model(torch.from_numpy(held_x_np)).numpy().astype(np.float32)
        fit_logits = fit_base + fit_delta
        held_logits = held_base + held_delta
        held_auc = float(roc_auc_score(held_y, held_logits))
        results[control] = {
            "fit_auc": float(roc_auc_score(fit_y, fit_logits)),
            "heldout_auc": held_auc,
            "heldout_delta_auc": held_auc - saved_base_auc,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "input_dim": int(fit_x_np.shape[1]),
            "gate": float(model.gate.detach()),
            "effective_amplitude": float(args.residual_cap * torch.tanh(model.gate.detach())),
            "fit_correction_mean": float(fit_delta.mean()),
            "fit_correction_std": float(fit_delta.std()),
            "heldout_correction_mean": float(held_delta.mean()),
            "heldout_correction_std": float(held_delta.std()),
            "heldout_correction_abs_max": float(np.abs(held_delta).max()),
            "fit_logits": fit_logits.astype(np.float32).tolist(),
            "heldout_logits": held_logits.astype(np.float32).tolist(),
            "fit_probabilities": sigmoid_np(fit_logits).astype(np.float32).tolist(),
            "heldout_probabilities": sigmoid_np(held_logits).astype(np.float32).tolist(),
            "history": history,
        }
        print(
            f"{control} heldout_auc={held_auc:.6f} delta={held_auc-saved_base_auc:+.6f} "
            f"params={results[control]['trainable_parameters']}", flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-frozen-realprototype-lowcapacity-occurrence-residual-scaffold-v1",
        "date": "2026-07-29",
        "hypothesis": (
            "connected prototype occurrence shape can add a small generalizable correction "
            "without replacing or retraining the stable real-prototype MIL predictor"
        ),
        "architecture": {
            "base": "previously trained stable graph-balanced fixed-random real-prototype MIL; fully frozen",
            "occurrence": "radius-1 connected component of positive top-k prototype support",
            "features_per_prototype": [
                "log_component_count", "log_active_node_count", "log_largest_component_size",
                "singleton_fraction", "log_total_coefficient_mass", "log_max_component_mass",
            ],
            "residual": "one hidden-layer graph-level MLP with zero-initialized scalar gate",
            "maximum_logit_correction": args.residual_cap,
            "supervised_message_passing_layers": 0,
        },
        "selection_policy": {
            "data": "8000-graph subset official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "epoch_policy": f"fixed {args.residual_epochs} residual epochs; no heldout epoch selection",
        },
        "config": vars(args),
        "fold": args.fold,
        "fit_graph_indices": fit_graphs.tolist(),
        "heldout_graph_indices": held_graphs.tolist(),
        "fit_labels": fit_y.astype(np.float32).tolist(),
        "heldout_labels": held_y.astype(np.float32).tolist(),
        "fit_indices_sha256": sha256(fit_indices),
        "heldout_indices_sha256": sha256(heldout_indices),
        "prototype_sha256": sha256(prototypes),
        "raw_occurrence_features_sha256": sha256(raw),
        "raw_occurrence_feature_shape": list(raw.shape),
        "results": results,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold": args.fold,
        "prototype_seed": args.prototype_seed,
        "heldout_auc": {name: row["heldout_auc"] for name, row in results.items()},
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
