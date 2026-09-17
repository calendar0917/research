"""Minimal faithful port of the frozen ZINC H96 architecture to OGBG-MolHIV.

This is a *first-round transfer audit*, not a MolHIV architecture search.  The
frozen ZINC design that is being ported is the T=2 weight-tied recurrent
pair--centre model (H96): a compact patch encoder, a shortest-path-conditioned
pair relation, a wide persistent centre state refreshed twice, and a single
graph-level readout.

What is re-used verbatim from the MolHIV pipeline
-------------------------------------------------
* atom / bond feature schema and the exact rooted patch certificate
  (``tracks/ksvd/experiments/luyin16/molhiv_patch_path_pooling.py``);
* the pre-built ``exact_rooted_{train,valid,test}.pkl`` record cache (all
  transforms are fit on the official train split only);
* the distance-conditioned pair relation and incident-pair centre aggregation;
* the official OGB scaffold split and the official ROC-AUC evaluator.

What is adapted
---------------
* the recurrent composition: the MolHIV one-shot centre update is replaced by
  the frozen ZINC T=2 weight-tied refresh (same shared ``pair_projection`` /
  ``relation_encoder`` / ``distance_gate`` / ``pair_encoder`` / ``center_update``
  applied twice);
* ``h_dim=96``, ``q_dim=16``, ``center_context_hidden=60``;
* the loss is the standard unweighted ``BCEWithLogitsLoss`` (no class weights,
  no focal loss);
* the graph head is the existing MolHIV binary head.

Discipline
----------
* model selection uses the official validation ROC-AUC only;
* the fixed rule is ``raw = best valid AUC checkpoint`` and
  ``Top-5 soup = equal-weight parameter average of the 5 highest-valid-AUC
  checkpoints (ties -> earliest epoch)`` -- pre-registered, no k/weight search;
* the official test split is loaded only after a freeze record is written.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_recurrent_pair_centre <stage>

Stages: ``data_sanity params smoke train train_queue freeze test report``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/molhiv_recurrent_pair_centre"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
CURVE_DIR = RESULTS_DIR / "curves"

RECORD_CACHE = (
    TRACK_ROOT
    / "results/luyin16/unified_relational_patch_molhiv_exact_rooted/record_cache"
)
DATA_ROOT = REPO_ROOT / "data/ogb"

PROTOCOL_VERSION = "molhiv_recurrent_pair_centre_v1"

SPLIT_SIZES = {"train": 32901, "valid": 4113, "test": 4113}

# Frozen H96-inspired architecture.
H_DIM = 96
Q_DIM = 16
TOKEN_WIDTH = 32
DROPOUT = 0.05
CENTER_CONTEXT_HIDDEN = 60
RECURRENCE_ROUNDS = 2

# Frozen training protocol (mirrors the ZINC optimized regime, adapted to a
# binary objective).  No class weights, no scheduler.
BATCH_SIZE = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
MAX_EPOCHS = 240
PATIENCE = 40
GRAD_CLIP = 5.0

SOUP_K = 5
SEEDS = (0, 1)

# Freeze the CUDA execution regime to deterministic algorithms (the model pools
# with ``index_add_``, atomic/non-deterministic on CUDA).
DETERMINISTIC = False


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class MolhivRecurrentPairCentreModel(mpp.PatchPathModel):
    """MolHIV patch--path model with the frozen ZINC T=2 weight-tied refresh.

    At ``recurrence_rounds=1`` the forward pass is bit-identical to
    ``mpp.PatchPathModel.forward`` (one pair value, one centre update, readout
    from the updated patch and that pair value).  At ``recurrence_rounds=2``
    the shared pair/centre modules are applied a second time to the updated
    centre state, exactly as in the frozen ZINC recurrent model:

        h0 = patch encoder
        q0 = Q(h0 endpoints);  h1 = h0 + U([h0, pool(q0)])
        q1 = Q(h1 endpoints);  h2 = h1 + U([h1, pool(q1)])
        readout = unary(h2) + pair_moments(q1) + global_encoder
    """

    def __init__(self, *args: Any, recurrence_rounds: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.center_update is None:
            raise ValueError("recurrent pair--centre requires center_context=True")
        if int(recurrence_rounds) < 1:
            raise ValueError("recurrence_rounds must be >= 1")
        self.recurrence_rounds = int(recurrence_rounds)

    def _pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        product = left * right
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)

    def forward(self, data: Data) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    self._patch_token_value(data),
                    self.parent_embedding(data.parent_token),
                ],
                dim=1,
            )
        )
        source = data.pair_index[0]
        target = data.pair_index[1]
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_value = None
        for _round in range(self.recurrence_rounds):
            pair_value = self._pair_value(patch, source, target, relation, gate)
            center_context = self._pool_pairs_to_centres(
                pair_value,
                source,
                target,
                data.pair_bucket,
                int(patch.shape[0]),
            )
            patch = patch + self.center_update(
                torch.cat([patch, center_context], dim=1)
            )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        assert pair_value is not None
        relation_readout = self._pool_pairs(
            pair_value, data.batch[source], data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        return self.head(
            torch.cat([unary, relation_readout, graph_hidden], dim=1)
        ).view(-1)

    def module_call_counts(self, data: Data) -> dict[str, int]:
        counts = {"pair_projection": 0, "pair_encoder": 0, "center_update": 0}
        handles = []

        def make_hook(name: str):
            def hook(_module, _inputs, _output):
                counts[name] += 1

            return hook

        for name in counts:
            module = getattr(self, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(make_hook(name)))
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self.forward(data)
        finally:
            for handle in handles:
                handle.remove()
            self.train(was_training)
        return counts


def build_model(
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    seed: int,
    patch_representation: str = "typed_lookup",
) -> MolhivRecurrentPairCentreModel:
    mpp._seed_everything(int(seed))
    return MolhivRecurrentPairCentreModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        patch_hidden=H_DIM,
        pair_hidden=Q_DIM,
        token_width=TOKEN_WIDTH,
        dropout=DROPOUT,
        center_context=True,
        center_context_hidden=CENTER_CONTEXT_HIDDEN,
        recurrence_rounds=RECURRENCE_ROUNDS,
        patch_representation=str(patch_representation),
    )


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_bundle():
    return load_molhiv(root=DATA_ROOT, with_features=False)


def _load_records(split: str) -> tuple[list[Any], dict[str, Any]]:
    return mpp._load_exact_rooted_record_cache(
        RECORD_CACHE, split, int(SPLIT_SIZES[split])
    )


def fit_train_transforms(records: Sequence[Any]) -> dict[str, Any]:
    """Fit tokenizer / vocabulary / standardizers on the official train split."""
    typed = mpp._fit_vocabulary(records, "typed_certificate", 32768, 1)
    parent = mpp._fit_vocabulary(records, "parent_certificate", 4096, 1)
    patch_std = mpp.Standardizer.fit(mpp._patch_matrix(records))
    context_std = mpp.Standardizer.fit(mpp._context_matrix(records))
    return {
        "typed_vocabulary": typed,
        "parent_vocabulary": parent,
        "patch_standardizer": patch_std,
        "context_standardizer": context_std,
    }


def encode_records(records: Sequence[Any], transforms: Mapping[str, Any]) -> list[Data]:
    return mpp._encode_records(
        records,
        transforms["typed_vocabulary"],
        transforms["parent_vocabulary"],
        transforms["patch_standardizer"],
        transforms["context_standardizer"],
    )


def data_sanity() -> dict[str, Any]:
    bundle = load_bundle()
    split = {name: np.asarray(bundle.split[name]) for name in ("train", "valid", "test")}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "ogbg-molhiv",
        "root": str(DATA_ROOT),
        "official_split": "OGB scaffold train/valid/test",
        "sizes": {name: int(values.size) for name, values in split.items()},
        "expected_sizes_match": all(
            int(split[name].size) == SPLIT_SIZES[name] for name in SPLIT_SIZES
        ),
        "positive": {
            name: int(bundle.y[split[name]].sum()) for name in split
        },
        "record_cache": str(RECORD_CACHE),
        "record_cache_exists": bool(
            all(
                (RECORD_CACHE / f"exact_rooted_{name}.pkl").exists()
                for name in ("train", "valid", "test")
            )
        ),
        "official_test_labels_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "data_sanity.json", payload)
    if not payload["expected_sizes_match"]:
        raise RuntimeError("MolHIV split sizes do not match the official scaffold split")
    return payload


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    model = build_model(32769, 4097, 0)
    breakdown = model.parameter_breakdown()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": {
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "token_width": TOKEN_WIDTH,
            "recurrence_rounds": RECURRENCE_ROUNDS,
            "weight_tied": True,
            "center_context_hidden": CENTER_CONTEXT_HIDDEN,
            "readout": "unary moments + distance-conditioned pair moments + global encoder",
            "loss": "BCEWithLogitsLoss (unweighted)",
            "head": "3-layer MLP (2h, h, 1)",
        },
        "total_params_example_vocab": _n_params(model),
        "parameter_breakdown_example_vocab": breakdown,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# smoke / training
# ---------------------------------------------------------------------------


def _valid_loader(graphs: Sequence[Data], seed: int):
    return mpp._make_loader(list(graphs), BATCH_SIZE, False, int(seed) + 91012)


def _train_loader(graphs: Sequence[Data], seed: int):
    return mpp._make_loader(list(graphs), BATCH_SIZE, True, int(seed) + 91011)


def smoke(device: str = "cuda") -> dict[str, Any]:
    """Tiny forward/backward smoke on a handful of graphs (no test)."""
    valid_records, _ = _load_records("valid")
    valid_records = valid_records[:256]
    transforms = fit_train_transforms(valid_records)
    graphs = encode_records(valid_records, transforms)
    del valid_records
    gc.collect()
    dev = torch.device(device)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    model = build_model(typed_size, parent_size, 0).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = _train_loader(graphs, 0)
    counts = model.module_call_counts(next(iter(loader)).to(dev))
    losses = []
    for step, batch in enumerate(loader):
        batch = batch.to(dev)
        target = batch.y.view(-1)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad()
        loss.backward()
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
        nonzero = any(
            p.grad is not None and float(p.grad.abs().max()) > 0.0
            for p in model.parameters()
        )
        assert finite and nonzero, "non-finite or zero gradient in smoke"
        optimizer.step()
        losses.append(float(loss.detach()))
        if step >= 3:
            break
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(dev),
        "cuda_available": bool(torch.cuda.is_available()),
        "n_graphs": int(len(graphs)),
        "steps": int(len(losses)),
        "losses": losses,
        "module_call_counts": counts,
        "params": _n_params(model),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def _topk(manifest: Sequence[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        manifest, key=lambda row: (-float(row["valid_auc"]), int(row["epoch"]))
    )
    return [dict(row) for row in ranked[:k]]


def _soup_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    states = [
        torch.load(Path(str(row["path"])), map_location="cpu", weights_only=True)
        for row in rows
    ]
    keys = list(states[0].keys())
    soup: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        soup[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    return soup


@torch.no_grad()
def _predict(
    model: nn.Module, state: Mapping[str, torch.Tensor], loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.eval()
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        targets.append(batch.y.view(-1).cpu().numpy())
        logits.append(model(batch).view(-1).cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(logits).astype(np.float64),
    )


def _auc(targets: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(targets, logits))


def train_seed(
    seed: int,
    device: str = "cuda",
    max_epochs: int | None = None,
    patience: int | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    max_epochs = int(MAX_EPOCHS if max_epochs is None else max_epochs)
    patience = int(PATIENCE if patience is None else patience)
    started = time.perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train requested CUDA but CUDA is unavailable")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    train_records, _ = _load_records("train")
    valid_records, _ = _load_records("valid")
    if train_limit is not None:
        train_records = train_records[: int(train_limit)]
    transforms = fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    train_data = encode_records(train_records, transforms)
    valid_data = encode_records(valid_records, transforms)
    del train_records, valid_records
    gc.collect()

    model = build_model(typed_size, parent_size, seed).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_loader = _train_loader(train_data, seed)
    valid_loader = _valid_loader(valid_data, seed)

    snapshot_dir = SNAPSHOT_DIR / f"seed{seed}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_epoch = 1
    stale = 0
    epoch_times: list[float] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(dev)
            target = batch.y.view(-1)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_loss = total_loss / max(seen, 1)
        valid_auc = mpp._evaluate_auc(model, valid_loader, dev)
        epoch_time = float(time.perf_counter() - epoch_started)
        epoch_times.append(epoch_time)
        snap_path = snapshot_dir / f"epoch_{epoch:03d}.pt"
        torch.save(copy.deepcopy(model.state_dict()), snap_path)
        manifest.append(
            {
                "epoch": int(epoch),
                "path": str(snap_path),
                "train_loss": float(train_loss),
                "valid_auc": float(valid_auc),
            }
        )
        curve.append(
            {"epoch": int(epoch), "train_bce": float(train_loss), "valid_auc": float(valid_auc)}
        )
        if valid_auc > best_auc:
            best_auc = float(valid_auc)
            best_epoch = int(epoch)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or valid_auc == best_auc:
            print(
                f"[molhiv-rpc seed{seed}] epoch={epoch:03d} bce={train_loss:.6f} "
                f"valid_auc={valid_auc:.6f} best={best_auc:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= patience:
            print(f"[molhiv-rpc seed{seed}] early_stop epoch={epoch}", flush=True)
            break

    top_rows = _topk(manifest, SOUP_K)
    best_row = _topk(manifest, 1)[0]
    raw_state = torch.load(Path(best_row["path"]), map_location="cpu", weights_only=True)
    soup_state = _soup_state(top_rows)
    valid_targets, raw_logits = _predict(model, raw_state, valid_loader, dev)
    _t, soup_logits = _predict(model, soup_state, valid_loader, dev)
    raw_valid_auc = _auc(valid_targets, raw_logits)
    soup_valid_auc = _auc(valid_targets, soup_logits)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    (CURVE_DIR / f"seed{seed}_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "device": str(dev),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if dev.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "parameters": _n_params(model),
        "typed_vocabulary_size_with_oov": int(typed_size),
        "parent_vocabulary_size_with_oov": int(parent_size),
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < max_epochs),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "raw_valid_auc_recomputed": float(raw_valid_auc),
        "soup_valid_auc": float(soup_valid_auc),
        "top5_epochs": [int(row["epoch"]) for row in top_rows],
        "top5_valid_auc": [float(row["valid_auc"]) for row in top_rows],
        "soup_parameters": int(sum(v.numel() for v in soup_state.values())),
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "wall_clock_s": float(time.perf_counter() - started),
        "curve": curve,
        "snapshot_manifest": manifest,
        "valid_targets": valid_targets.tolist(),
        "raw_valid_logits": raw_logits.tolist(),
        "soup_valid_logits": soup_logits.tolist(),
        "loss": "BCEWithLogitsLoss (unweighted)",
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"run_seed{seed}.json", summary)
    # Persist the frozen raw / soup selection states for the later one-shot test.
    torch.save(raw_state, RESULTS_DIR / f"raw_state_seed{seed}.pt")
    torch.save(soup_state, RESULTS_DIR / f"soup_state_seed{seed}.pt")
    return summary


def train_queue(seeds: Sequence[int], device: str) -> None:
    for seed in seeds:
        if (RESULTS_DIR / f"run_seed{seed}.json").exists():
            print(f"skip existing seed{seed}", flush=True)
            continue
        print(f"=== train molhiv-rpc seed{seed} device={device} ===", flush=True)
        train_seed(int(seed), device=device)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(torch.float32).cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def repro(
    seed: int, epochs: int, device: str, train_limit: int = 4096, run_tag: str = ""
) -> dict[str, Any]:
    """Short same-seed GPU reproducibility sanity on a fixed train subset."""
    global RESULTS_DIR, SNAPSHOT_DIR, CURVE_DIR
    base = RESULTS_DIR
    tag = str(run_tag or "default").replace("/", "_")
    saved = (RESULTS_DIR, SNAPSHOT_DIR, CURVE_DIR)
    root = base / "repro" / tag
    RESULTS_DIR, SNAPSHOT_DIR, CURVE_DIR = root, root / "snapshots", root / "curves"
    try:
        summary = train_seed(
            int(seed),
            device=device,
            max_epochs=int(epochs),
            patience=int(epochs),
            train_limit=int(train_limit),
        )
        state = torch.load(
            RESULTS_DIR / f"raw_state_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        state_hash = _state_sha256(state)
    finally:
        RESULTS_DIR, SNAPSHOT_DIR, CURVE_DIR = saved
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "gpu_reproducibility_sanity",
        "seed": int(seed),
        "epochs": int(epochs),
        "train_limit": int(train_limit),
        "run_tag": tag,
        "device": str(device),
        "best_valid_auc": float(summary["best_valid_auc"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "mean_epoch_time_s": float(summary["mean_epoch_time_s"]),
        "wall_clock_s": float(summary["wall_clock_s"]),
        "train_loss_curve": [float(row["train_bce"]) for row in summary["curve"]],
        "valid_auc_curve": [float(row["valid_auc"]) for row in summary["curve"]],
        "raw_state_sha256": state_hash,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "official_test_loaded": False,
    }
    _write_json(base / f"repro_{tag}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# freeze / one-shot test
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    lock_path = RESULTS_DIR / "official_test_unlock.json"
    if lock_path.exists():
        raise RuntimeError("refusing to change the frozen rule after test unlock")
    runs = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"run_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing run for seed{seed}; cannot freeze")
        runs[seed] = _read_json(path)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": {
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "token_width": TOKEN_WIDTH,
            "recurrence_rounds": RECURRENCE_ROUNDS,
            "weight_tied": True,
            "center_context_hidden": CENTER_CONTEXT_HIDDEN,
            "loss": "BCEWithLogitsLoss (unweighted)",
        },
        "seeds": [int(s) for s in SEEDS],
        "selection_metric": "official validation ROC-AUC",
        "raw_rule": "checkpoint with the highest validation ROC-AUC (ties -> earliest epoch)",
        "soup_rule": {
            "K": SOUP_K,
            "ranking": "highest validation ROC-AUC",
            "tie_rule": "earliest epoch",
            "aggregation": "equal-weight parameter average",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "validation": {
            str(seed): {
                "raw_valid_auc": float(runs[seed]["raw_valid_auc_recomputed"]),
                "soup_valid_auc": float(runs[seed]["soup_valid_auc"]),
                "best_epoch": int(runs[seed]["best_epoch"]),
                "epochs_run": int(runs[seed]["epochs_run"]),
                "parameters": int(runs[seed]["parameters"]),
                "mean_epoch_time_s": float(runs[seed]["mean_epoch_time_s"]),
            }
            for seed in SEEDS
        },
        "test_status": "not yet loaded",
        "platform": platform.platform(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


def test_eval(device: str = "cuda") -> dict[str, Any]:
    freeze_path = RESULTS_DIR / "architecture_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("refusing test: architecture_freeze.json missing")
    lock_path = RESULTS_DIR / "official_test_unlock.json"
    if lock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once")

    # Write the unlock record before the first (and only) test load.
    _write_json(
        lock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "frozen_before_test": True,
            "encoding": "transforms fit on official train only",
            "checkpoints": "pre-existing raw / Top-5 soup selection states",
            "official_test_loaded": True,
        },
    )

    dev = torch.device(device)
    train_records, _ = _load_records("train")
    transforms = fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    del train_records
    gc.collect()

    test_records, metadata = _load_records("test")
    test_data = encode_records(test_records, transforms)
    del test_records
    gc.collect()
    test_loader = mpp._make_loader(list(test_data), BATCH_SIZE, False, 0)
    targets = np.asarray([float(g.y.view(-1)[0]) for g in test_data], dtype=np.float64)

    rows = []
    raw_logits_by_seed: dict[int, np.ndarray] = {}
    soup_logits_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = build_model(typed_size, parent_size, seed).to(dev)
        raw_state = torch.load(
            RESULTS_DIR / f"raw_state_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        soup_state = torch.load(
            RESULTS_DIR / f"soup_state_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        t_raw, raw_logits = _predict(model, raw_state, test_loader, dev)
        t_soup, soup_logits = _predict(model, soup_state, test_loader, dev)
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        raw_logits_by_seed[seed] = raw_logits
        soup_logits_by_seed[seed] = soup_logits
        rows.append(
            {
                "seed": int(seed),
                "raw_test_auc": _auc(targets, raw_logits),
                "soup_test_auc": _auc(targets, soup_logits),
            }
        )

    raw_values = np.asarray([row["raw_test_auc"] for row in rows])
    soup_values = np.asarray([row["soup_test_auc"] for row in rows])
    raw_ensemble = np.mean(np.stack([raw_logits_by_seed[s] for s in SEEDS]), axis=0)
    soup_ensemble = np.mean(np.stack([soup_logits_by_seed[s] for s in SEEDS]), axis=0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
        "raw_test_mean": float(raw_values.mean()),
        "raw_test_std": float(raw_values.std(ddof=1)) if len(raw_values) > 1 else 0.0,
        "soup_test_mean": float(soup_values.mean()),
        "soup_test_std": float(soup_values.std(ddof=1)) if len(soup_values) > 1 else 0.0,
        "diagnostic_raw_2seed_ensemble_test_auc": _auc(targets, raw_ensemble),
        "diagnostic_soup_2seed_ensemble_test_auc": _auc(targets, soup_ensemble),
        "test_metadata": {k: v for k, v in metadata.items() if k != "records"},
        "official_test_loaded": True,
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "data_sanity": _maybe("data_sanity.json"),
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "smoke": _maybe("smoke.json"),
        "runs": {str(s): _maybe(f"run_seed{s}.json") for s in SEEDS},
        "architecture_freeze": _maybe("architecture_freeze.json"),
        "official_test_results": _maybe("official_test_results.json"),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "data_sanity",
            "params",
            "smoke",
            "train",
            "train_queue",
            "repro",
            "freeze",
            "test",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    torch.set_num_threads(4)
    _set_deterministic(DETERMINISTIC)
    if args.stage == "data_sanity":
        print(json.dumps(data_sanity(), indent=2, default=str), flush=True)
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "smoke":
        print(json.dumps(smoke(args.device), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(json.dumps(train_seed(args.seed, args.device), indent=2, default=str))
    if args.stage == "train_queue":
        train_queue([int(s) for s in args.seeds.split(",")], args.device)
    if args.stage == "repro":
        print(
            json.dumps(
                repro(args.seed, args.epochs, args.device, args.train_limit, args.run_tag),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "freeze":
        print(json.dumps(freeze(), indent=2, default=str), flush=True)
    if args.stage == "test":
        print(json.dumps(test_eval(args.device), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
