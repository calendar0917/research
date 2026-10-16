"""Data preparation, training and prediction for the gate round."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import synthetic as S
from models import ARMS, GateModel, apply_shared_encoder_init, build_model, shared_encoder_init

RESULTS_DIR = Path(__file__).resolve().parents[1]
TRAIN_LOGS = RESULTS_DIR / "train_logs"
PRED_DIR = RESULTS_DIR / "predictions"


@dataclass
class GraphTensors:
    graph_id: str
    family: str
    split: str
    base_id: str
    pair_id: str
    bag_twin: str | None
    rule: str
    label: int
    meta: dict
    x: torch.Tensor
    rel_true: torch.Tensor
    rel_scrambled: torch.Tensor
    pair_ok: torch.Tensor
    patches: list
    true_dist: np.ndarray
    n_nodes: int = 0
    n_edges: int = 0


@dataclass
class TrainConfig:
    max_epochs: int = 600
    patience: int = 150
    lr: float = 3.0e-3
    batch_size: int = 16
    weight_decay: float = 0.0
    log_every: int = 50


def prepare_data() -> tuple[dict[str, GraphTensors], dict]:
    graphs = S.all_graphs()
    patches = {g.graph_id: S.extract_patches(g) for g in graphs}
    dist = {g.graph_id: S.anchor_distance_matrix(g, patches[g.graph_id]) for g in graphs}
    rel_true = {gid: S.relation_features(dist[gid], patches[gid]) for gid in patches}
    train_ids = [g.graph_id for g in graphs if g.split == "train"]
    rel_scr = S.scrambled_relation_features(dist, patches, train_ids)

    data: dict[str, GraphTensors] = {}
    for g in graphs:
        ps = patches[g.graph_id]
        p = len(ps)
        x = torch.tensor(np.stack([np.asarray(pt.feature, dtype=np.float32) for pt in ps]), dtype=torch.float32)
        rel_t = torch.tensor(rel_true[g.graph_id], dtype=torch.float32)
        rel_s = torch.tensor(rel_scr[g.graph_id], dtype=torch.float32)
        pair_ok = ~torch.eye(p, dtype=torch.bool)
        data[g.graph_id] = GraphTensors(
            graph_id=g.graph_id,
            family=g.family,
            split=g.split,
            base_id=g.base_id,
            pair_id=g.pair_id,
            bag_twin=g.bag_twin,
            rule=g.rule,
            label=g.label,
            meta=dict(g.meta),
            x=x,
            rel_true=rel_t,
            rel_scrambled=rel_s,
            pair_ok=pair_ok,
            patches=ps,
            true_dist=dist[g.graph_id],
            n_nodes=g.n,
            n_edges=len(g.edges),
        )
    manifest = {
        "n_graphs": len(data),
        "graphs_per_split": {s: sum(1 for d in data.values() if d.split == s) for s in ("train", "dev", "test")},
        "split_manifest": S.frozen_split_manifest(),
    }
    return data, manifest


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _rel(d: GraphTensors, relation_source: str) -> torch.Tensor:
    if relation_source == "true":
        return d.rel_true
    if relation_source == "scrambled":
        return d.rel_scrambled
    raise ValueError(relation_source)


@torch.no_grad()
def predict_graph(
    model: GateModel,
    d: GraphTensors,
    relation_source: str = "true",
    mode: str = "hard",
) -> tuple[float, float]:
    model.eval()
    logit, _ = model(d.x, _rel(d, relation_source), d.pair_ok)
    if mode == "soft":
        logit, _ = model(d.x, _rel(d, relation_source), d.pair_ok, mode="soft")
    return float(logit), float(torch.sigmoid(logit))


@torch.no_grad()
def collect_concepts(model: GateModel, d: GraphTensors) -> np.ndarray:
    model.eval()
    z = model.encoder(d.x)
    _, _, idx = model.vq.similarity(z)
    return idx.cpu().numpy()


def _balanced_accuracy(preds: Sequence[float], labels: Sequence[int]) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    yhat = (preds >= 0.5).astype(int)
    pos = labels == 1
    neg = ~pos
    acc_pos = (yhat[pos] == 1).mean() if pos.any() else float("nan")
    acc_neg = (yhat[neg] == 0).mean() if neg.any() else float("nan")
    return float(np.nanmean([acc_pos, acc_neg]))


def _graph_loss(model: GateModel, d: GraphTensors, y: torch.Tensor, relation_source: str):
    logit, info = model(d.x, _rel(d, relation_source), d.pair_ok)
    loss_task = F.binary_cross_entropy_with_logits(logit, y)
    loss_rec = F.mse_loss(info["recon"], d.x)
    loss = loss_task + model.cfg.rec_weight * loss_rec
    parts = {"task": float(loss_task), "rec": float(loss_rec)}
    if model.cfg.arm in ("C", "D"):
        z = info["z"]
        hard = info["hard"]
        commit = F.mse_loss(z, hard.detach())
        cb = F.mse_loss(z.detach(), hard)
        loss = loss + model.cfg.commit_weight * commit + model.cfg.codebook_weight * cb
        parts["commit"] = float(commit)
        parts["codebook"] = float(cb)
    return loss, parts


def train_arm(
    arm: str,
    seed: int,
    data: Mapping[str, GraphTensors],
    *,
    relation_source: str = "true",
    cfg: TrainConfig | None = None,
    model_kwargs: dict | None = None,
    verbose: bool = False,
) -> dict:
    cfg = cfg or TrainConfig()
    model_kwargs = model_kwargs or {}
    assert arm in ARMS
    torch.manual_seed(seed * 100 + 7)
    model = build_model(arm, seed, **model_kwargs)
    if arm in ("C", "D"):
        apply_shared_encoder_init(model, shared_encoder_init(seed, d=model.cfg.d, hidden=model.cfg.hidden, feat_dim=model.cfg.feat_dim))
    train_ids = sorted(k for k, d in data.items() if d.split == "train")
    dev_ids = sorted(k for k, d in data.items() if d.split == "dev")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(S._seed_from("train", arm, seed, relation_source))
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    best_dev_loss = float("inf")
    history: list[dict] = []
    t0 = time.time()

    def dev_metrics() -> tuple[float, float]:
        probs, labels, losses = [], [], []
        for gid in dev_ids:
            d = data[gid]
            logit, prob = predict_graph(model, d, relation_source, mode="hard")
            probs.append(prob)
            labels.append(1 if d.label > 0 else 0)
            with torch.no_grad():
                l, _ = _graph_loss(model, d, torch.tensor(float(d.label > 0)), relation_source)
            losses.append(float(l))
        return _balanced_accuracy(probs, labels), float(np.mean(losses))

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = rng.permutation(len(train_ids))
        accs: list[float] = []
        parts_acc: dict[str, list[float]] = {}
        for start in range(0, len(perm), cfg.batch_size):
            opt.zero_grad()
            batch = [train_ids[i] for i in perm[start : start + cfg.batch_size]]
            losses = []
            for gid in batch:
                d = data[gid]
                y = torch.tensor(float(d.label > 0), dtype=torch.float32)
                loss, parts = _graph_loss(model, d, y, relation_source)
                losses.append(loss)
                for k, v in parts.items():
                    parts_acc.setdefault(k, []).append(v)
            total = torch.stack(losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            accs.append(float(total))
        dev_acc, dev_loss = dev_metrics()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(accs)),
            "dev_balanced_accuracy": dev_acc,
            "dev_loss": dev_loss,
            **{f"train_{k}": float(np.mean(v)) for k, v in parts_acc.items()},
        }
        history.append(row)
        better = dev_acc > best_metric + 1e-12 or (abs(dev_acc - best_metric) <= 1e-12 and dev_loss < best_dev_loss)
        if better:
            best_metric = dev_acc
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % cfg.log_every == 0 or epoch == 1):
            print(f"  [{arm} seed{seed} {relation_source}] ep{epoch:4d} loss {row['train_loss']:.4f} dev_acc {dev_acc:.3f} best {best_metric:.3f}@{best_epoch}")
        if epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t0
    out = {
        "arm": arm,
        "seed": seed,
        "relation_source": relation_source,
        "best_epoch": best_epoch,
        "best_dev_balanced_accuracy": best_metric,
        "best_dev_loss": best_dev_loss,
        "epochs_run": len(history),
        "wall_seconds": elapsed,
        "train_config": asdict(cfg),
        "model_config": asdict(model.cfg),
        "parameter_breakdown": model.parameter_breakdown(),
        "n_parameters": model.n_parameters(),
        "history": history,
    }
    return {"model": model, "report": out}


def save_run(run: dict, tag: str) -> None:
    TRAIN_LOGS.mkdir(parents=True, exist_ok=True)
    torch.save(run["model"].state_dict(), TRAIN_LOGS / f"{tag}.pt")
    with open(TRAIN_LOGS / f"{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(run["report"], fh, indent=1, sort_keys=True)


def load_model(tag: str, *, state_path: Path | None = None) -> GateModel:
    with open(TRAIN_LOGS / f"{tag}.json", encoding="utf-8") as fh:
        report = json.load(fh)
    model = build_model(report["arm"], report["seed"], **{k: v for k, v in report["model_config"].items() if k != "arm"})
    model.load_state_dict(torch.load(state_path or TRAIN_LOGS / f"{tag}.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate_split(
    model: GateModel,
    data: Mapping[str, GraphTensors],
    split: str,
    relation_source: str = "true",
    mode: str = "hard",
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with torch.no_grad():
        for gid, d in sorted(data.items()):
            if d.split != split:
                continue
            logit, prob = predict_graph(model, d, relation_source, mode=mode)
            prob2, logit2 = None, None
            if mode != "soft":
                logit2, prob2 = predict_graph(model, d, relation_source, mode="soft")
            out[gid] = {
                "graph_id": gid,
                "family": d.family,
                "base_id": d.base_id,
                "pair_id": d.pair_id,
                "bag_twin": d.bag_twin,
                "rule": d.rule,
                "label": d.label,
                "logit": logit,
                "prob": prob,
                "soft_logit": logit2,
                "soft_prob": prob2,
            }
    return out
