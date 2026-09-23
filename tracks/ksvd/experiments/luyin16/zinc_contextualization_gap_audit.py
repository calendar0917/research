"""CGA-v0 — Contextualization Gap Audit.

Existing-checkpoint **deterministic inference / export** plus train-only
representation audits.  No training, no architecture change, official ZINC test
never loaded.

Protocol: ``zinc_contextualization_gap_audit_v0``
Pre-registration: ``notes/zinc_contextualization_gap_audit_v0_preregistration.md``

Subcommands
-----------
``checkpoints``  verify + hash the checkpoints, write ``checkpoint_inventory.json``
``export``       deterministic state export for one model/seed
``audit``        run Audits A/B/C/D for one seed from the exported arrays
``verify``       repeat-export determinism gate
``full``         the whole audit for recurrent seed0/seed1 (+ auxiliary B-Null)

The heavy representation maths is done once per exported checkpoint; the
reported JSON artifacts are frozen evidence, and the per-patch / per-molecule
arrays are exported so the local analysis can regenerate every figure and
aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rec
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# paths / constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "zinc_contextualization_gap_audit_v0"
REPO_ROOT = Path(__file__).resolve().parents[4]  # .../research
TRACK_ROOT = REPO_ROOT / "tracks" / "ksvd"
RESULTS_DIR = TRACK_ROOT / "results" / "zinc_contextualization_gap_audit_v0"
RAW_DIR = RESULTS_DIR / "raw"
FIGURE_DIR = RESULTS_DIR / "figures"

RECURRENT_STATE_DIR = TRACK_ROOT / "results" / "compact_v4_recurrent_pair_centre" / "states"
BASELINE_STATE_DIR = RECURRENT_STATE_DIR
BNULL_STATE_DIR = TRACK_ROOT / "results" / "local_token_null" / "states"

K_NEIGHBOURS = 16
K_PROTOTYPES = 64
MIN_PROTOTYPE_N = 50
BOOTSTRAP_RESAMPLES = 10000
RANDOM_SEED = 20260923
Q_SUBSET_MOL = 128
EPS = 1.0e-8

ROOT_ATOM_BLOCK = slice(108, 136)  # root-atom one-hot inside the 146-D shell descriptor

BASELINE_RECORDED = {0: 0.14533376283763208, 1: 0.13805893784115325}
RECURRENT_RECORDED = {0: 0.13837560486892472, 1: 0.13343997858563672}
BNULL_RECORDED = 0.12687269969756015

PREREG_PATH = TRACK_ROOT / "notes" / "zinc_contextualization_gap_audit_v0_preregistration.md"


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()
        return text
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# checkpoint inventory
# ---------------------------------------------------------------------------


def checkpoint_spec() -> dict[str, dict[str, Any]]:
    """Checkpoint table: role -> seed -> path + recorded valid MAE + builder."""
    return {
        "recurrent": {
            "builder": "rec.build_recurrent",
            "record_dir": "results/compact_v4_recurrent_pair_centre",
            "states": {
                seed: RECURRENT_STATE_DIR / f"recurrent_seed{seed}_selection_state.pt"
                for seed in (0, 1)
            },
            "recorded_valid_mae": RECURRENT_RECORDED,
            "params": 82115,
            "training_commit": "a3515e76",
            "primary": True,
        },
        "baseline": {
            "builder": "rec.build_baseline",
            "record_dir": "results/compact_v4_recurrent_pair_centre",
            "states": {
                seed: BASELINE_STATE_DIR / f"baseline_seed{seed}_selection_state.pt"
                for seed in (0, 1)
            },
            "recorded_valid_mae": BASELINE_RECORDED,
            "params": 82115,
            "training_commit": "a3515e76",
            "primary": True,
        },
        "bnull": {
            "builder": "ltn.build_null",
            "record_dir": "results/local_token_null",
            "states": {0: BNULL_STATE_DIR / "lt_null_seed0_selection_state.pt"},
            "recorded_valid_mae": {0: BNULL_RECORDED},
            "params": 49343,
            "training_commit": "ltn",
            "primary": False,
        },
    }


def _require_checkpoints(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = checkpoint_spec() if spec is None else spec
    missing_primary: list[str] = []
    for role, info in spec.items():
        for seed, path in info["states"].items():
            if not Path(path).exists() and bool(info["primary"]):
                missing_primary.append(f"{role}:seed{seed}:{path}")
    if missing_primary:
        raise FileNotFoundError(
            "STOP_AND_REPORT_MISSING_ARTIFACT: missing checkpoint(s):\n"
            + "\n".join(missing_primary)
        )
    return spec


def _missing_auxiliary(spec: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for role, info in spec.items():
        if bool(info["primary"]):
            continue
        for seed, path in info["states"].items():
            if not Path(path).exists():
                missing.append(f"{role}:seed{seed}:{path}")
    return missing


def cmd_checkpoints() -> dict[str, Any]:
    spec = _require_checkpoints()
    inventory: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": PREREG_PATH.name,
        "preregistration_sha256": _sha256(PREREG_PATH) if PREREG_PATH.exists() else None,
        "max_new_full_training_runs": 0,
        "checkpoints": {},
    }
    for role, info in spec.items():
        role_entry: dict[str, Any] = {
            "builder": info["builder"],
            "record_dir": info["record_dir"],
            "parameters": info["params"],
            "training_commit": info["training_commit"],
            "primary": bool(info["primary"]),
            "seeds": {},
        }
        for seed, path in info["states"].items():
            path = Path(path)
            if not path.exists():
                role_entry["seeds"][str(seed)] = {
                    "path": str(path.relative_to(TRACK_ROOT)),
                    "present": False,
                    "recorded_valid_mae": float(info["recorded_valid_mae"][seed]),
                }
                continue
            role_entry["seeds"][str(seed)] = {
                "path": str(path.relative_to(TRACK_ROOT)),
                "present": True,
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
                "recorded_valid_mae": float(info["recorded_valid_mae"][seed]),
            }
        inventory["checkpoints"][role] = role_entry
    inventory["missing_auxiliary"] = _missing_auxiliary(spec)
    _write_json(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    return inventory


# ---------------------------------------------------------------------------
# model construction / loading
# ---------------------------------------------------------------------------


def _load_bnull():
    from tracks.ksvd.experiments.luyin16 import zinc_local_token_null as ltn

    return ltn


def build_model(role: str, seed: int) -> torch.nn.Module:
    if role == "recurrent":
        return rec.build_recurrent(int(seed))
    if role == "baseline":
        return rec.build_baseline(int(seed))
    if role == "bnull":
        ltn = _load_bnull()
        return ltn.build_null(int(seed))
    raise ValueError(f"unknown role {role!r}")


def load_model(role: str, seed: int) -> torch.nn.Module:
    spec = _require_checkpoints()
    path = Path(spec[role]["states"][seed])
    model = build_model(role, seed)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if hasattr(model, "capture_diagnostics"):
        model.capture_diagnostics = True
    return model


def _load_records() -> tuple[list[Any], list[Any], dict[str, Any]]:
    torch.set_num_threads(4)
    train_data, valid_data, audit = rec.load_encoded()
    return list(train_data), list(valid_data), dict(audit)


# ---------------------------------------------------------------------------
# deterministic export
# ---------------------------------------------------------------------------


def _export_recurrent_like(
    model: torch.nn.Module,
    dataset: Sequence[Any],
    *,
    batch_size: int = 128,
    with_targets: bool,
    q_subset_mol: int = 0,
) -> dict[str, Any]:
    """Run the checkpoint's own forward and capture the stored tensors."""
    loader = zpp._make_loader(list(dataset), batch_size, False, 0)
    keys = ("h0", "h1", "h2", "A0", "A1", "x", "g")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    sizes: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    yhats: list[np.ndarray] = []
    cq: list[np.ndarray] = []
    q_subsets: dict[str, list[np.ndarray]] = {k: [] for k in ("q0", "q1", "src", "dst", "bucket", "pg")}
    offset = 0
    graph = getattr(model, "unified_graph_width", None)
    model.eval()
    if hasattr(model, "capture_diagnostics"):
        model.capture_diagnostics = True
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            h0 = model.last_h0  # type: ignore[attr-defined]
            h1 = model.last_h1  # type: ignore[attr-defined]
            h2 = model.last_h2  # type: ignore[attr-defined]
            A0 = model.last_center_context0  # type: ignore[attr-defined]
            A1 = model.last_center_context1  # type: ignore[attr-defined]
            if h0 is None or h2 is None or A0 is None:
                raise RuntimeError("checkpoint forward did not expose h/A diagnostics")
            g_local = batch.batch.detach().cpu().numpy().astype(np.int64) + offset
            parts["h0"].append(h0.detach().float().cpu().numpy())
            parts["h2"].append(h2.detach().float().cpu().numpy())
            parts["h1"].append(h1.detach().float().cpu().numpy() if h1 is not None else h0.detach().float().cpu().numpy())
            parts["A0"].append(A0.detach().float().cpu().numpy())
            parts["A1"].append(A1.detach().float().cpu().numpy() if A1 is not None else A0.detach().float().cpu().numpy())
            parts["x"].append(batch.patch_cont.detach().float().cpu().numpy())
            parts["g"].append(g_local)
            n_graphs = int(batch.num_graphs)
            counts = torch.bincount(batch.batch.detach().cpu(), minlength=n_graphs).numpy()
            sizes.append(counts)
            if with_targets:
                ys.append(batch.y.view(-1).detach().cpu().numpy())
                yhats.append(prediction.view(-1).detach().cpu().numpy())
            q0 = model.last_q0  # type: ignore[attr-defined]
            q1 = model.last_q1  # type: ignore[attr-defined]
            pb = model.last_pair_batch  # type: ignore[attr-defined]
            if q0 is not None and q1 is not None and pb is not None:
                qdiff = torch.linalg.vector_norm(q1 - q0, dim=1).detach().cpu().numpy()
                pbg = pb.detach().cpu().numpy().astype(np.int64) + offset
                sums = np.bincount(pbg, weights=qdiff, minlength=n_graphs)
                cnts = np.bincount(pbg, minlength=n_graphs)
                cq.append(sums / np.maximum(cnts, 1))
                if offset < q_subset_mol:
                    q_subsets["q0"].append(q0.detach().float().cpu().numpy())
                    q_subsets["q1"].append(q1.detach().float().cpu().numpy())
                    q_subsets["src"].append(model.last_pair_source.detach().cpu().numpy())  # type: ignore[attr-defined]
                    q_subsets["dst"].append(model.last_pair_target.detach().cpu().numpy())  # type: ignore[attr-defined]
                    q_subsets["bucket"].append(model.last_pair_bucket.detach().cpu().numpy())  # type: ignore[attr-defined]
                    q_subsets["pg"].append(pbg)
            offset += n_graphs
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    out["sizes"] = np.concatenate(sizes).astype(np.int64)
    if with_targets:
        out["y"] = np.concatenate(ys).astype(np.float32)
        out["yhat"] = np.concatenate(yhats).astype(np.float32)
        out["cq"] = np.concatenate(cq).astype(np.float32) if cq else np.zeros(len(out["sizes"]), np.float32)
    if q_subset_mol and q_subsets["q0"]:
        out["q_subset"] = {k: np.concatenate(v, axis=0) for k, v in q_subsets.items()}
    return out


def _export_predictions(model: torch.nn.Module, dataset: Sequence[Any], *, batch_size: int = 128) -> dict[str, Any]:
    loader = zpp._make_loader(list(dataset), batch_size, False, 0)
    ys: list[np.ndarray] = []
    yhats: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            ys.append(batch.y.view(-1).detach().cpu().numpy())
            yhats.append(pred.view(-1).detach().cpu().numpy())
    y = np.concatenate(ys).astype(np.float32)
    yhat = np.concatenate(yhats).astype(np.float32)
    return {"y": y, "yhat": yhat, "mae": float(np.mean(np.abs(y - yhat)))}


def _raw_path(role: str, seed: int, split: str) -> Path:
    return RAW_DIR / f"{role}_seed{seed}_{split}.npz"


def save_export(role: str, seed: int, split: str, arrays: Mapping[str, np.ndarray]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(role, seed, split)
    np.savez(path, **{k: v for k, v in arrays.items() if not k.startswith("q_subset")})
    if "q_subset" in arrays:
        np.savez(
            RAW_DIR / f"{role}_seed{seed}_{split}_qsubset.npz",
            **arrays["q_subset"],  # type: ignore[arg-type]
        )
    return path


def load_export(role: str, seed: int, split: str) -> dict[str, np.ndarray]:
    path = _raw_path(role, seed, split)
    with np.load(path) as handle:
        return {k: np.array(handle[k]) for k in handle.files}


def cmd_export(role: str, seed: int, *, q_subset_mol: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    train_data, valid_data, _ = _load_records()
    model = load_model(role, seed)
    summary: dict[str, Any] = {
        "role": role,
        "seed": int(seed),
        "official_test_loaded": False,
        "train_molecules": len(train_data),
        "valid_molecules": len(valid_data),
    }
    if role == "baseline":
        valid = _export_predictions(model, valid_data)
        save_export(role, seed, "valid", {"y": valid["y"], "yhat": valid["yhat"]})
        summary["valid_mae"] = valid["mae"]
        summary["recorded_valid_mae"] = BASELINE_RECORDED[seed]
        summary["mae_abs_diff"] = abs(valid["mae"] - BASELINE_RECORDED[seed])
    else:
        valid = _export_recurrent_like(model, valid_data, with_targets=True, q_subset_mol=q_subset_mol)
        train = _export_recurrent_like(model, train_data, with_targets=False)
        save_export(role, seed, "valid", valid)
        save_export(role, seed, "train", train)
        mae = float(np.mean(np.abs(valid["y"] - valid["yhat"])))
        recorded = RECURRENT_RECORDED[seed] if role == "recurrent" else BNULL_RECORDED
        summary["valid_mae"] = mae
        summary["recorded_valid_mae"] = float(recorded)
        summary["mae_abs_diff"] = abs(mae - float(recorded))
        summary["train_patches"] = int(train["h0"].shape[0])
        summary["valid_patches"] = int(valid["h0"].shape[0])
    summary["wall_clock_s"] = float(time.perf_counter() - started)
    _write_json(RAW_DIR / f"{role}_seed{seed}_export.json", summary)
    return summary


# ---------------------------------------------------------------------------
# integrity gates
# ---------------------------------------------------------------------------


def _integrity_gates(role: str, seed: int) -> dict[str, Any]:
    train_data, valid_data, _ = _load_records()
    model = load_model(role, seed)
    recorded = RECURRENT_RECORDED[seed] if role == "recurrent" else BNULL_RECORDED
    checks: dict[str, Any] = {"role": role, "seed": int(seed), "official_test_loaded": False}

    # A. prediction reproduction on the full official valid split
    valid = _export_predictions(model, valid_data)
    target = float(recorded) if role != "baseline" else float(BASELINE_RECORDED[seed])
    checks["valid_mae"] = valid["mae"]
    checks["recorded_valid_mae"] = target
    checks["A_prediction_abs_diff"] = abs(valid["mae"] - target)
    checks["A_prediction_pass"] = bool(checks["A_prediction_abs_diff"] <= 1.0e-4)

    if role == "baseline":
        checks["passed"] = checks["A_prediction_pass"]
        return checks

    # B/C/D on a fixed first-128 valid batch, live tensors
    batch = next(iter(zpp._make_loader(valid_data[:128], 128, False, 0)))
    with torch.no_grad():
        model(batch)
    h0 = model.last_h0
    q0 = model.last_q0
    A0 = model.last_center_context0
    source = model.last_pair_source
    target_idx = model.last_pair_target
    bucket = model.last_pair_bucket
    assert h0 is not None and q0 is not None and A0 is not None

    # C. recompute q0 and A0 from the stored tensors
    relation = model.relation_encoder(batch.pair_relation)
    gate = 1.0 + torch.tanh(model.distance_gate(batch.pair_bucket))
    q0_recomputed = model._pair_value(h0, source, target_idx, relation, gate)
    A0_recomputed = model._pool_pairs_to_centres(q0, source, target_idx, bucket, int(h0.shape[0]))
    checks["C_q0_reconstruction_max_abs"] = float((q0_recomputed - q0).abs().max())
    checks["C_A0_reconstruction_max_abs"] = float((A0_recomputed - A0).abs().max())
    checks["C_reconstruction_pass"] = bool(
        checks["C_q0_reconstruction_max_abs"] <= 1.0e-5
        and checks["C_A0_reconstruction_max_abs"] <= 1.0e-5
    )

    # B. ordering: concatenated y of the valid loader equals the run record
    run_json = TRACK_ROOT / "results" / "compact_v4_recurrent_pair_centre" / "runs" / f"{role}_seed{seed}.json"
    if role == "bnull":
        run_json = TRACK_ROOT / "results" / "local_token_null" / "runs" / f"lt_null_seed{seed}.json"
    ordering_ok = None
    ordering_abs = None
    if run_json.exists():
        run = _read_json(run_json)
        recorded_targets = np.asarray(run.get("valid_targets", []), dtype=np.float32)
        if recorded_targets.size == valid["y"].size:
            ordering_abs = float(np.max(np.abs(recorded_targets - valid["y"])))
            ordering_ok = bool(ordering_abs == 0.0)
    checks["B_valid_targets_match_record"] = ordering_ok
    checks["B_valid_targets_max_abs"] = ordering_abs

    # single-molecule spot check: batched slice equals a lone forward
    single = next(iter(zpp._make_loader(valid_data[:1], 1, False, 0)))
    with torch.no_grad():
        model(single)
    single_h0 = model.last_h0
    n_first = int(single_h0.shape[0])
    checks["B_single_vs_batched_h0_max_abs"] = float(
        (single_h0 - h0[:n_first]).abs().max()
    )
    checks["B_state_ordering_pass"] = bool(checks["B_single_vs_batched_h0_max_abs"] <= 1.0e-6)

    # D. repeat export determinism on 256 valid molecules
    subset = valid_data[:256]
    first = _export_recurrent_like(model, subset, with_targets=True)
    second = _export_recurrent_like(model, subset, with_targets=True)
    det_diffs = {
        key: float(np.max(np.abs(first[key] - second[key])))
        for key in ("h0", "h1", "h2", "A0", "A1", "yhat")
    }
    checks["D_repeat_export_max_abs"] = det_diffs
    checks["D_determinism_pass"] = bool(all(v == 0.0 for v in det_diffs.values()))

    checks["passed"] = bool(
        checks["A_prediction_pass"]
        and checks["C_reconstruction_pass"]
        and checks.get("B_state_ordering_pass", False)
        and checks["D_determinism_pass"]
    )
    return checks


def cmd_verify() -> dict[str, Any]:
    results: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "official_test_loaded": False, "models": {}}
    for role in ("recurrent", "baseline", "bnull"):
        seed = 0
        if role == "baseline":
            model = load_model(role, seed)
            results["models"][f"{role}_seed{seed}"] = {"checked": "prediction-only", "skip": True}
            continue
        results["models"][f"{role}_seed{seed}"] = _integrity_gates(role, seed)
    _write_json(RESULTS_DIR / "export_integrity.json", results)
    return results


# ---------------------------------------------------------------------------
# representation maths
# ---------------------------------------------------------------------------


def _standardize(train: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0).astype(np.float32)
    scale = train.std(axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1.0e-8)] = 1.0
    return (train - mean) / scale, (valid - mean) / scale, mean, scale


def _knn(query: np.ndarray, reference: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """k nearest reference indices for every query row (Euclidean)."""
    q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32))
    r = torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32))
    r2 = (r * r).sum(dim=1)
    n = int(q.shape[0])
    out = np.empty((n, k), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, chunk):
            qc = q[start : start + chunk]
            d2 = r2.unsqueeze(0) - 2.0 * (qc @ r.t()) + (qc * qc).sum(dim=1, keepdim=True)
            _, idx = torch.topk(d2, k, dim=1, largest=False, sorted=True)
            out[start : start + chunk] = idx.numpy().astype(np.int64)
    return out


def _distances_to_neighbours(
    query: np.ndarray, reference: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """mean L2 distance from each query to its gathered neighbours."""
    q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32))
    r = torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32))[torch.from_numpy(idx)]
    with torch.no_grad():
        return torch.linalg.vector_norm(q.unsqueeze(1) - r, dim=2).mean(dim=1).numpy().astype(np.float32)


def _mean_of_neighbours(reference: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return reference[idx].mean(axis=1)


def _jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eq = a[:, :, None] == b[:, None, :]
    inter = eq.any(axis=2).sum(axis=1)
    union = a.shape[1] + b.shape[1] - inter
    return inter / np.maximum(union, 1)


def _kmeanspp(Xn: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(Xn.shape[0])
    centers = np.empty((k, Xn.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centers[0] = Xn[first]
    min_d2 = np.maximum(2.0 - 2.0 * (Xn @ centers[0]), 0.0)
    for j in range(1, k):
        total = float(min_d2.sum())
        if total <= 0.0:
            centers[j] = Xn[int(rng.integers(n))]
        else:
            prob = (min_d2 / total).astype(np.float64)
            prob = prob / prob.sum()
            idx = int(rng.choice(n, p=prob))
            centers[j] = Xn[idx]
        d2 = np.maximum(2.0 - 2.0 * (Xn @ centers[j]), 0.0)
        min_d2 = np.minimum(min_d2, d2)
    return centers


def _spherical_kmeans(
    X: np.ndarray, k: int, *, seed: int, max_iter: int = 100, tol: float = 1.0e-7
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm < 1.0e-8] = 1.0
    Xn = (X / norm).astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = _kmeanspp(Xn, k, rng)
    assign = np.full(Xn.shape[0], -1, dtype=np.int64)
    for _ in range(max_iter):
        sim = Xn @ centers.T
        new_assign = sim.argmax(axis=1).astype(np.int64)
        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = new_assign == j
            if bool(mask.any()):
                new_centers[j] = Xn[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        nrm = np.linalg.norm(new_centers, axis=1, keepdims=True)
        nrm[nrm < 1.0e-8] = 1.0
        new_centers /= nrm
        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        assign = new_assign
        if shift <= tol:
            break
    return assign, centers


def _participation_ratio(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    denom = float(np.sum(eig * eig))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(eig) ** 2 / denom)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _bootstrap_ci(
    values: Callable[[np.ndarray], float],
    n: int,
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    samples = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        samples[r] = values(idx[r])
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {
        "point": float(values(np.arange(n))),
        "ci_low": lo,
        "ci_high": hi,
        "resamples": int(resamples),
        "seed": int(seed),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
    }


def _rank_residual(target: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    rt = _rankdata(target)
    rc = _rankdata(covariate)
    design = np.stack([np.ones_like(rc), rc], axis=1)
    coef, *_ = np.linalg.lstsq(design, rt, rcond=None)
    return rt - design @ coef


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------


def audit_neighbour_divergence(
    train: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    train_h0 = train["h0"].astype(np.float32)
    train_h2 = train["h2"].astype(np.float32)
    valid_h0 = valid["h0"].astype(np.float32)
    valid_h2 = valid["h2"].astype(np.float32)
    train_delta = train_h2 - train_h0
    valid_delta = valid_h2 - valid_h0
    sigma_delta = float(np.sqrt(np.mean(np.sum((train_delta - train_delta.mean(0)) ** 2, axis=1))))

    X_train, X_valid, _, _ = _standardize(train["x"].astype(np.float32), valid["x"].astype(np.float32))
    H0_train, H0_valid, _, _ = _standardize(train_h0, valid_h0)
    H2_train, H2_valid, _, _ = _standardize(train_h2, valid_h2)

    spaces = {"H0": (H0_train, H0_valid), "X": (X_train, X_valid)}
    per_space: dict[str, Any] = {}
    for name, (ref, query) in spaces.items():
        idx = _knn(query, ref, K_NEIGHBOURS)
        d0 = _distances_to_neighbours(valid_h0, train_h0, idx)
        d2 = _distances_to_neighbours(valid_h2, train_h2, idx)
        ddelta = _distances_to_neighbours(valid_delta, train_delta, idx)
        per_space[name] = {
            "nn_index": idx,
            "d0": d0,
            "d2": d2,
            "dDelta": ddelta,
        }

    # neighbour retention H0 -> H2 (independent searches, k=16 both)
    nn_h0 = per_space["H0"]["nn_index"]
    nn_h2 = _knn(H2_valid, H2_train, K_NEIGHBOURS)
    jaccard = _jaccard_rows(nn_h0, nn_h2)

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
        }

    space_reports: dict[str, Any] = {}
    for name in spaces:
        d0 = per_space[name]["d0"]
        d2 = per_space[name]["d2"]
        ddelta = per_space[name]["dDelta"]
        amp = d2 / (d0 + EPS)
        positive = d0 > 1.0e-6
        amp_positive = d2[positive] / (d0[positive] + EPS)
        dnorm = ddelta / (sigma_delta + EPS)
        space_reports[name] = {
            "d0": _summary(d0),
            "d2": _summary(d2),
            "dDelta": _summary(ddelta),
            "amplification": _summary(amp),
            "amplification_d0_positive": _summary(amp_positive) if bool(positive.any()) else None,
            "frac_exact_h0_match": float(np.mean(d0 <= 1.0e-6)),
            "dDelta_over_sigma_delta": _summary(dnorm),
            "R_delta_median": float(np.median(ddelta) / (math.sqrt(2.0) * sigma_delta + EPS)),
        }

    # matched random control: same (root category, molecule-size decile)
    rng = np.random.default_rng(RANDOM_SEED)
    train_root = train["x"][:, ROOT_ATOM_BLOCK].argmax(axis=1).astype(np.int64)
    valid_root = valid["x"][:, ROOT_ATOM_BLOCK].argmax(axis=1).astype(np.int64)
    train_size = train["sizes"][train["g"]].astype(np.int64)
    valid_size = valid["sizes"][valid["g"]].astype(np.int64)
    edges = np.quantile(train_size, np.linspace(0.0, 1.0, 11))
    train_bin = np.clip(np.searchsorted(edges[1:-1], train_size, side="right"), 0, 9)
    valid_bin = np.clip(np.searchsorted(edges[1:-1], valid_size, side="right"), 0, 9)
    cells: dict[tuple[int, int], np.ndarray] = {}
    order = np.argsort(train_root * 10 + train_bin, kind="mergesort")
    keys = train_root * 10 + train_bin
    for key in np.unique(keys):
        cells[int(key)] = order[keys[order] == key]
    nq = valid_h0.shape[0]
    rand_idx = np.empty((nq, K_NEIGHBOURS), dtype=np.int64)
    for i in range(nq):
        key = int(valid_root[i]) * 10 + int(valid_bin[i])
        pool = cells.get(key)
        if pool is None or pool.size == 0:
            pool = np.arange(train_h0.shape[0])
        if pool.size >= K_NEIGHBOURS:
            rand_idx[i] = rng.choice(pool, size=K_NEIGHBOURS, replace=False)
        else:
            rand_idx[i] = rng.choice(pool, size=K_NEIGHBOURS, replace=True)
    rand_d0 = _distances_to_neighbours(valid_h0, train_h0, rand_idx)
    rand_d2 = _distances_to_neighbours(valid_h2, train_h2, rand_idx)
    rand_ddelta = _distances_to_neighbours(valid_delta, train_delta, rand_idx)

    # robust retention: NN dispersion relative to the matched random control
    h0_d2 = per_space["H0"]["d2"]
    h0_ddelta = per_space["H0"]["dDelta"]
    retention_d2_ratio = float(h0_d2.mean() / max(rand_d2.mean(), EPS))
    retention_ddelta_ratio = float(h0_ddelta.mean() / max(rand_ddelta.mean(), EPS))

    # exact-local-state subset: identical h0 -> what is the contextual spread?
    train_lut: dict[bytes, int] = {}
    for row in train_h0:
        key = row.tobytes()
        train_lut[key] = train_lut.get(key, 0) + 1
    duplicate_counts = np.asarray(
        [train_lut.get(row.tobytes(), 0) for row in valid_h0], dtype=np.int64
    )
    exact = per_space["H0"]["d0"] <= 1.0e-6
    exact_block: dict[str, Any] = {
        "n": int(exact.sum()),
        "fraction": float(np.mean(exact)),
        "train_exact_duplicate_count_median": float(np.median(duplicate_counts)),
        "train_exact_duplicate_count_mean": float(np.mean(duplicate_counts)),
        "train_exact_duplicate_count_ge16_fraction": float(np.mean(duplicate_counts >= K_NEIGHBOURS)),
    }
    if bool(exact.any()):
        exact_block.update(
            {
                "d2_mean": float(per_space["H0"]["d2"][exact].mean()),
                "d2_median": float(np.median(per_space["H0"]["d2"][exact])),
                "dDelta_mean": float(per_space["H0"]["dDelta"][exact].mean()),
                "dDelta_median": float(np.median(per_space["H0"]["dDelta"][exact])),
                "dDelta_over_sqrt2_sigma_delta_median": float(
                    np.median(per_space["H0"]["dDelta"][exact])
                    / (math.sqrt(2.0) * sigma_delta + EPS)
                ),
                "retention_dDelta_ratio": float(
                    per_space["H0"]["dDelta"][exact].mean() / max(rand_ddelta[exact].mean(), EPS)
                ),
            }
        )

    # d0-bin control on the primary H0 space
    d0_primary = per_space["H0"]["d0"]
    d2_primary = per_space["H0"]["d2"]
    ddelta_primary = per_space["H0"]["dDelta"]
    amp_primary = d2_primary / (d0_primary + EPS)
    bin_edges = np.quantile(d0_primary, np.linspace(0.0, 1.0, 11))
    bins = np.clip(np.searchsorted(bin_edges[1:-1], d0_primary, side="right"), 0, 9)
    d0_bin_report = []
    for b in range(10):
        mask = bins == b
        if not bool(mask.any()):
            continue
        d0_bin_report.append(
            {
                "bin": int(b),
                "n": int(mask.sum()),
                "d0_mean": float(d0_primary[mask].mean()),
                "amplification_mean": float(amp_primary[mask].mean()),
                "R_delta_median": float(
                    np.median(ddelta_primary[mask]) / (math.sqrt(2.0) * sigma_delta + EPS)
                ),
            }
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "k_neighbours": K_NEIGHBOURS,
        "sigma_delta": sigma_delta,
        "train_patches": int(train_h0.shape[0]),
        "valid_patches": int(valid_h0.shape[0]),
        "spaces": space_reports,
        "neighbour_retention_h0_h2": {
            **_summary(jaccard),
            "materially_below_one": bool(float(np.mean(jaccard)) <= 0.60),
            "tie_confounded_note": (
                "Jaccard is not diagnostic when the h0 space has many exact "
                "duplicates: the h0-NN and h2-NN sets can both lie inside the "
                "same duplicate group yet overlap little. See "
                "retention_dDelta_ratio and exact_local_state for the "
                "tie-robust measures."
            ),
        },
        "retention_robust": {
            "h0NN_d2_over_random_d2": retention_d2_ratio,
            "h0NN_dDelta_over_random_dDelta": retention_ddelta_ratio,
            "interpretation": "<1 means h0-neighbours stay closer / more similar in h2 than matched random patches",
        },
        "exact_local_state": exact_block,
        "random_control_rootcat_sizebin": {
            "d0": _summary(rand_d0),
            "d2": _summary(rand_d2),
            "dDelta": _summary(rand_ddelta),
        },
        "d0_bin_control": d0_bin_report,
        "arrays": {
            "H0_idx": per_space["H0"]["nn_index"],
            "X_idx": per_space["X"]["nn_index"],
            "H2_idx": nn_h2,
            "H0_d0": per_space["H0"]["d0"],
            "H0_d2": per_space["H0"]["d2"],
            "H0_dDelta": per_space["H0"]["dDelta"],
            "X_d0": per_space["X"]["d0"],
            "X_d2": per_space["X"]["d2"],
            "X_dDelta": per_space["X"]["dDelta"],
            "jaccard": jaccard,
            "rand_d0": rand_d0,
            "rand_d2": rand_d2,
            "rand_dDelta": rand_ddelta,
        },
    }
    return payload


def audit_local_predictability(
    train: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray], nn_h0: np.ndarray
) -> dict[str, Any]:
    train_h0 = train["h0"].astype(np.float32)
    train_h2 = train["h2"].astype(np.float32)
    valid_h0 = valid["h0"].astype(np.float32)
    valid_h2 = valid["h2"].astype(np.float32)
    train_delta = train_h2 - train_h0
    valid_delta = valid_h2 - valid_h0

    H0_train, H0_valid, _, _ = _standardize(train_h0, valid_h0)
    A0_train, A0_valid, _, _ = _standardize(train["A0"].astype(np.float32), valid["A0"].astype(np.float32))
    A1_train, A1_valid, _, _ = _standardize(train["A1"].astype(np.float32), valid["A1"].astype(np.float32))

    delta_bar = train_delta.mean(axis=0)
    mse_mean = float(np.mean(np.sum((valid_delta - delta_bar) ** 2, axis=1)) / valid_delta.shape[1])

    delta_hat_local = _mean_of_neighbours(train_delta, nn_h0)
    err_local = valid_delta - delta_hat_local
    mse_local = float(np.mean(np.sum(err_local ** 2, axis=1)) / valid_delta.shape[1])
    mae_local = float(np.mean(np.sum(np.abs(err_local), axis=1)) / valid_delta.shape[1])

    z0_train = np.concatenate([H0_train, A0_train], axis=1)
    z0_valid = np.concatenate([H0_valid, A0_valid], axis=1)
    nn_z0 = _knn(z0_valid, z0_train, K_NEIGHBOURS)
    delta_hat_ctx = _mean_of_neighbours(train_delta, nn_z0)
    mse_context = float(np.mean(np.sum((valid_delta - delta_hat_ctx) ** 2, axis=1)) / valid_delta.shape[1])

    z1_train = np.concatenate([H0_train, A0_train, A1_train], axis=1)
    z1_valid = np.concatenate([H0_valid, A0_valid, A1_valid], axis=1)
    nn_z1 = _knn(z1_valid, z1_train, K_NEIGHBOURS)
    delta_hat_ctx1 = _mean_of_neighbours(train_delta, nn_z1)
    mse_context_a1 = float(np.mean(np.sum((valid_delta - delta_hat_ctx1) ** 2, axis=1)) / valid_delta.shape[1])

    p_local = 1.0 - mse_local / mse_mean
    p_context = 1.0 - mse_context / mse_mean
    context_gain = (mse_local - mse_context) / mse_local if mse_local > 0.0 else float("nan")
    context_abs_gain = (mse_local - mse_context) / mse_mean
    p_context_a1 = 1.0 - mse_context_a1 / mse_mean
    context_gain_a1 = (mse_local - mse_context_a1) / mse_local if mse_local > 0.0 else float("nan")
    context_abs_gain_a1 = (mse_local - mse_context_a1) / mse_mean

    var_delta = float(np.mean(np.var(train_delta, axis=0)))
    var_h2 = float(np.mean(np.var(train_h2, axis=0)))

    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "k_neighbours": K_NEIGHBOURS,
        "mse_mean": mse_mean,
        "mse_local": mse_local,
        "mae_local": mae_local,
        "mse_context_z0": mse_context,
        "mse_context_z1": mse_context_a1,
        "P_local": p_local,
        "P_context": p_context,
        "P_context_A1": p_context_a1,
        "context_gain": context_gain,
        "context_gain_A1": context_gain_a1,
        "context_abs_gain": context_abs_gain,
        "context_abs_gain_A1": context_abs_gain_a1,
        "var_delta_perdim": var_delta,
        "var_h2_perdim": var_h2,
        "var_delta_over_var_h2": var_delta / var_h2 if var_h2 > 0.0 else float("nan"),
        "context_share_of_h2_variance": context_abs_gain * (var_delta / var_h2 if var_h2 > 0.0 else float("nan")),
        "local_insufficiency_relative": bool(context_gain >= 0.20),
        "local_compilable_absolute": bool(p_local >= 0.50 and context_abs_gain < 0.05),
        "local_compilable_strict_preregistered": bool(p_local >= 0.50 and context_gain < 0.05),
        "arrays": {"nn_z0": nn_z0, "nn_z1": nn_z1},
    }


def prototype_context_split(train: Mapping[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    train_h0 = train["h0"].astype(np.float32)
    train_h2 = train["h2"].astype(np.float32)
    train_delta = train_h2 - train_h0
    H0_train, _, _, _ = _standardize(train_h0, train_h0)
    assign, centers = _spherical_kmeans(H0_train, K_PROTOTYPES, seed=seed)

    global_cov = np.cov(train_delta, rowvar=False)
    global_trace = float(np.trace(global_cov))
    per_prototype: list[dict[str, Any]] = []
    for k in range(K_PROTOTYPES):
        mask = assign == k
        n_k = int(mask.sum())
        if n_k < MIN_PROTOTYPE_N:
            continue
        h0k = H0_train[mask]
        h2k = train_h2[mask]
        dk = train_delta[mask]
        mean_h0 = h0k.mean(axis=0)
        mean_d = dk.mean(axis=0)
        mean_h2 = h2k.mean(axis=0)
        cov_d = np.cov(dk, rowvar=False)
        trace_d = float(np.trace(cov_d))
        per_prototype.append(
            {
                "prototype": int(k),
                "n": n_k,
                "var_h0_perdim": float(np.mean(np.sum((h0k - mean_h0) ** 2, axis=1)) / h0k.shape[1]),
                "var_delta_perdim": float(np.mean(np.sum((dk - mean_d) ** 2, axis=1)) / dk.shape[1]),
                "var_h2_perdim": float(np.mean(np.sum((h2k - mean_h2) ** 2, axis=1)) / h2k.shape[1]),
                "trace_cov_delta": trace_d,
                "effective_contextual_rank": _participation_ratio(cov_d),
                "S_k": trace_d / global_trace if global_trace > 0 else float("nan"),
                "within_h0_radius": float(np.mean(np.linalg.norm(h0k - mean_h0, axis=1))),
                "within_delta_radius": float(np.mean(np.linalg.norm(dk - mean_d, axis=1))),
            }
        )
    ns = np.asarray([p["n"] for p in per_prototype], dtype=np.float64)
    sk = np.asarray([p["S_k"] for p in per_prototype], dtype=np.float64)
    weighted = float(np.sum(ns * sk) / np.sum(ns)) if ns.size else float("nan")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "K": K_PROTOTYPES,
        "seed": int(seed),
        "min_prototype_n": MIN_PROTOTYPE_N,
        "n_prototypes_used": len(per_prototype),
        "global_trace_cov_delta": global_trace,
        "weighted_mean_S_k": weighted,
        "median_S_k": float(np.median(sk)) if sk.size else float("nan"),
        "min_S_k": float(np.min(sk)) if sk.size else float("nan"),
        "max_S_k": float(np.max(sk)) if sk.size else float("nan"),
        "prototypes": per_prototype,
        "arrays": {"assign_train": assign, "centers": centers},
    }


def audit_task_link(
    valid: Mapping[str, np.ndarray], baseline_valid: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    n_graphs = int(valid["sizes"].shape[0])
    g = valid["g"].astype(np.int64)
    counts = np.bincount(g, minlength=n_graphs).astype(np.float64)
    h0 = valid["h0"].astype(np.float32)
    h1 = valid["h1"].astype(np.float32)
    h2 = valid["h2"].astype(np.float32)
    delta = h2 - h0
    delta1 = h1 - h0
    dA = np.linalg.norm(valid["A1"].astype(np.float32) - valid["A0"].astype(np.float32), axis=1)
    ch = np.bincount(g, weights=np.linalg.norm(delta, axis=1), minlength=n_graphs) / counts
    ch1 = np.bincount(g, weights=np.linalg.norm(delta1, axis=1), minlength=n_graphs) / counts
    cA = np.bincount(g, weights=dA, minlength=n_graphs) / counts
    cq = valid["cq"].astype(np.float64)
    y = valid["y"].astype(np.float64)
    yhat_rec = valid["yhat"].astype(np.float64)
    yhat_base = baseline_valid["yhat"].astype(np.float64)
    size = counts

    err_base = np.abs(y - yhat_base)
    err_rec = np.abs(y - yhat_rec)
    gain = err_base - err_rec

    n = n_graphs
    spearman = _spearman(ch, gain)
    pearson = _pearson(ch, gain)
    spearman_ci = _bootstrap_ci(
        lambda idx: _spearman(ch[idx], gain[idx]), n, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED
    )
    pearson_ci = _bootstrap_ci(
        lambda idx: _pearson(ch[idx], gain[idx]), n, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED + 1
    )

    q_edges = np.quantile(ch, [0.25, 0.5, 0.75])
    quart = np.searchsorted(q_edges, ch, side="right")
    quartile_gain = [float(gain[quart == q].mean()) for q in range(4)]
    q4_minus_q1 = quartile_gain[3] - quartile_gain[0]

    def _q4_minus_q1(idx: np.ndarray) -> float:
        c = ch[idx]
        gg = gain[idx]
        e = np.quantile(c, [0.25, 0.5, 0.75])
        qq = np.searchsorted(e, c, side="right")
        return float(gg[qq == 3].mean() - gg[qq == 0].mean())

    q4_ci = _bootstrap_ci(_q4_minus_q1, n, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED + 2)

    resid_ch = _rank_residual(ch, size)
    resid_gain = _rank_residual(gain, size)
    spearman_resid = _pearson(resid_ch, resid_gain)
    spearman_resid_ci = _bootstrap_ci(
        lambda idx: _pearson(_rank_residual(ch[idx], size[idx]), _rank_residual(gain[idx], size[idx])),
        n,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=RANDOM_SEED + 3,
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_molecules": n,
        "metrics": {
            "C_h": {"mean": float(ch.mean()), "median": float(np.median(ch))},
            "C_h1": {"mean": float(ch1.mean())},
            "C_q": {"mean": float(cq.mean())},
            "C_A": {"mean": float(cA.mean())},
            "err_base_mean": float(err_base.mean()),
            "err_rec_mean": float(err_rec.mean()),
            "gain_mean": float(gain.mean()),
            "gain_median": float(np.median(gain)),
        },
        "C_h_gain": {
            "spearman": spearman,
            "spearman_ci": spearman_ci,
            "pearson": pearson,
            "pearson_ci": pearson_ci,
        },
        "quartiles": {
            "edges": q_edges.tolist(),
            "mean_gain": quartile_gain,
            "q4_minus_q1": q4_minus_q1,
            "q4_minus_q1_ci": q4_ci,
        },
        "size_confound": {
            "n_atoms_is_n_patches": True,
            "spearman_C_h_size": _spearman(ch, size),
            "spearman_gain_size": _spearman(gain, size),
            "spearman_size_residualised": spearman_resid,
            "spearman_size_residualised_ci": spearman_resid_ci,
        },
        "task_link": bool(spearman > 0.0 and spearman_ci["ci_low"] > 0.0),
        "task_link_point_positive": bool(spearman > 0.0 and q4_minus_q1 > 0.0),
        "task_link_strict_definition": "Spearman > 0 and Spearman bootstrap 95% CI low > 0",
        "arrays": {
            "C_h": ch,
            "C_h1": ch1,
            "C_q": cq,
            "C_A": cA,
            "y": y,
            "yhat_rec": yhat_rec,
            "yhat_base": yhat_base,
            "gain": gain,
            "size": size,
        },
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_seed(seed: int, *, include_bnull: bool = True) -> dict[str, Any]:
    _require_checkpoints()
    export_recurrent = cmd_export("recurrent", seed, q_subset_mol=Q_SUBSET_MOL)
    export_baseline = cmd_export("baseline", seed)
    integrity = _integrity_gates("recurrent", seed)

    train = load_export("recurrent", seed, "train")
    valid = load_export("recurrent", seed, "valid")
    baseline_valid = load_export("baseline", seed, "valid")

    divergence = audit_neighbour_divergence(train, valid)
    predictability = audit_local_predictability(train, valid, divergence["arrays"]["H0_idx"])
    task_link = audit_task_link(valid, baseline_valid)
    prototypes = prototype_context_split(train, seed=seed)

    def _strip(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in payload.items() if k != "arrays"}

    _write_json(RESULTS_DIR / f"seed{seed}_neighbour_divergence.json", _strip(divergence))
    _write_json(RESULTS_DIR / f"seed{seed}_local_predictability.json", _strip(predictability))
    _write_json(RESULTS_DIR / f"seed{seed}_task_link.json", _strip(task_link))

    arrays_path = RESULTS_DIR / f"seed{seed}_patch_metrics.npz"
    np.savez(
        arrays_path,
        H0_d0=divergence["arrays"]["H0_d0"],
        H0_d2=divergence["arrays"]["H0_d2"],
        H0_dDelta=divergence["arrays"]["H0_dDelta"],
        X_d0=divergence["arrays"]["X_d0"],
        X_d2=divergence["arrays"]["X_d2"],
        X_dDelta=divergence["arrays"]["X_dDelta"],
        jaccard=divergence["arrays"]["jaccard"],
        rand_d0=divergence["arrays"]["rand_d0"],
        rand_d2=divergence["arrays"]["rand_d2"],
        rand_dDelta=divergence["arrays"]["rand_dDelta"],
        C_h=task_link["arrays"]["C_h"],
        C_h1=task_link["arrays"]["C_h1"],
        C_q=task_link["arrays"]["C_q"],
        C_A=task_link["arrays"]["C_A"],
        gain=task_link["arrays"]["gain"],
        size=task_link["arrays"]["size"],
        prototype=prototypes["arrays"]["assign_train"],
    )

    result: dict[str, Any] = {
        "seed": int(seed),
        "export": {"recurrent": export_recurrent, "baseline": export_baseline},
        "integrity": integrity,
        "divergence": _strip(divergence),
        "predictability": _strip(predictability),
        "task_link": _strip(task_link),
        "prototypes": _strip(prototypes),
    }

    if include_bnull:
        try:
            bnull_export = cmd_export("bnull", 0, q_subset_mol=0)
            bnull_train = load_export("bnull", 0, "train")
            bnull_valid = load_export("bnull", 0, "valid")
            bdiv = audit_neighbour_divergence(bnull_train, bnull_valid)
            bpred = audit_local_predictability(bnull_train, bnull_valid, bdiv["arrays"]["H0_idx"])
            result["bnull"] = {
                "export": bnull_export,
                "divergence": _strip(bdiv),
                "predictability": _strip(bpred),
            }
            _write_json(
                RESULTS_DIR / "bnull_seed0_auxiliary.json",
                {"export": bnull_export, "divergence": _strip(bdiv), "predictability": _strip(bpred)},
            )
        except Exception as exc:  # auxiliary only
            result["bnull"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _decision(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for seed, payload in per_seed.items():
        div = payload["divergence"]
        pred = payload["predictability"]
        task = payload["task_link"]
        retention_ddelta = div["retention_robust"]["h0NN_dDelta_over_random_dDelta"]
        divergence = bool(
            div["spaces"]["H0"]["R_delta_median"] >= 0.50 or retention_ddelta >= 0.60
        )
        insufficiency = bool(pred["context_gain"] >= 0.20)
        # absolute (tie-robust) compilability: the local state explains nearly
        # all of the shift, and context adds only a small absolute increment
        compilable = bool(
            pred["P_local"] >= 0.50
            and pred["context_abs_gain"] < 0.05
            and retention_ddelta <= 0.50
        )
        task_link = bool(task["task_link"])
        flags[str(seed)] = {
            "DIVERGENCE": divergence,
            "LOCAL_INSUFFICIENCY_RELATIVE": insufficiency,
            "LOCAL_COMPILABLE_ABSOLUTE": compilable,
            "locally_compilable_strict_preregistered": bool(
                pred["P_local"] >= 0.50
                and pred["context_gain"] < 0.05
                and div["neighbour_retention_h0_h2"]["mean"] >= 0.50
            ),
            "TASK_LINK": task_link,
            "R_delta_median_H0": div["spaces"]["H0"]["R_delta_median"],
            "mean_jaccard": div["neighbour_retention_h0_h2"]["mean"],
            "retention_dDelta_over_random": retention_ddelta,
            "retention_d2_over_random": div["retention_robust"]["h0NN_d2_over_random_d2"],
            "P_local": pred["P_local"],
            "P_context": pred["P_context"],
            "context_gain": pred["context_gain"],
            "context_abs_gain": pred["context_abs_gain"],
            "context_share_of_h2_variance": pred["context_share_of_h2_variance"],
            "spearman_C_h_gain": task["C_h_gain"]["spearman"],
            "spearman_C_h_gain_ci_low": task["C_h_gain"]["spearman_ci"]["ci_low"],
            "q4_minus_q1": task["quartiles"]["q4_minus_q1"],
        }
    seeds = sorted(flags)
    all_a = all(
        flags[s]["DIVERGENCE"]
        and flags[s]["LOCAL_INSUFFICIENCY_RELATIVE"]
        and flags[s]["TASK_LINK"]
        for s in seeds
    )
    all_div = all(flags[s]["DIVERGENCE"] for s in seeds)
    all_c = all(flags[s]["LOCAL_COMPILABLE_ABSOLUTE"] for s in seeds)
    if all_a:
        case = "A_CONTEXTUALIZATION_BOTTLENECK_SUPPORTED"
    elif all_c:
        case = "C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE"
    elif all_div and not any(flags[s]["TASK_LINK"] for s in seeds):
        case = "B_CONTEXT_EXISTS_BUT_TASK_LINK_UNSUPPORTED"
    else:
        case = "D_INCONCLUSIVE_CONTEXTUALIZATION_AUDIT"
    note = (
        "DIVERGENCE and LOCAL_COMPILABLE use the tie-robust retention "
        "dDelta_over_random; Jaccard is reported but is not decision-bearing "
        "because exact h0 duplicates make it non-diagnostic."
    )
    return {"flags": flags, "case": case, "criterion_note": note}


def cmd_full(*, include_bnull: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    inventory = cmd_checkpoints()
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in (0, 1):
        per_seed[seed] = run_seed(seed, include_bnull=include_bnull and seed == 0)
    integrity = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "max_new_full_training_runs": 0,
        "git_commit": _git_commit(),
        "seeds": {str(s): per_seed[s]["integrity"] for s in per_seed},
    }
    integrity["all_primary_pass"] = bool(
        all(per_seed[s]["integrity"].get("passed", False) for s in per_seed)
    )
    _write_json(RESULTS_DIR / "export_integrity.json", integrity)

    decision = _decision(per_seed)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "max_new_full_training_runs": 0,
        "git_commit": _git_commit(),
        "checkpoints": inventory["checkpoints"],
        "integrity": integrity,
        "seeds": {
            str(s): {
                "divergence": per_seed[s]["divergence"],
                "predictability": per_seed[s]["predictability"],
                "task_link": per_seed[s]["task_link"],
                "prototypes": per_seed[s]["prototypes"],
            }
            for s in per_seed
        },
        "decision": decision,
        "wall_clock_s": float(time.perf_counter() - started),
    }
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_results_summary(summary)
    try:
        make_figures(summary)
    except Exception as exc:  # figures are explanatory only
        summary["figure_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(RESULTS_DIR / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(summary: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(s) for s in summary["seeds"])

    # h0 distance vs delta_h distance (seed0 H0 space)
    div0 = summary["seeds"][str(seeds[0])]["divergence"]
    metrics0 = np.load(RESULTS_DIR / f"seed{seeds[0]}_patch_metrics.npz")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(metrics0["H0_d0"], metrics0["H0_dDelta"], s=2, alpha=0.2)
    ax.set_xlabel("h0 distance to 16 train NN")
    ax.set_ylabel("delta_h disagreement (dDelta)")
    ax.set_title(f"seed{seeds[0]}: local h0 distance vs contextual shift disagreement")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h0_distance_vs_delta_h_distance.png", dpi=140)
    plt.close(fig)

    # neighbour retention histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    for s in seeds:
        m = np.load(RESULTS_DIR / f"seed{s}_patch_metrics.npz")
        ax.hist(m["jaccard"], bins=40, alpha=0.5, label=f"seed{s}")
    ax.set_xlabel("Jaccard(NN16_h0, NN16_h2)")
    ax.set_ylabel("valid patches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "neighbour_retention_hist.png", dpi=140)
    plt.close(fig)

    # contextualization vs recurrent gain
    for s in seeds:
        m = np.load(RESULTS_DIR / f"seed{s}_patch_metrics.npz")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(m["C_h"], m["gain"], s=4, alpha=0.3)
        rho = summary["seeds"][str(s)]["task_link"]["C_h_gain"]["spearman"]
        ax.set_xlabel("C_h = mean ||h2-h0||")
        ax.set_ylabel("recurrent gain (err_base - err_rec)")
        ax.set_title(f"seed{s}: Spearman={rho:.3f}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"contextualization_vs_recurrent_gain_seed{s}.png", dpi=140)
        plt.close(fig)

    # gain by contextualization quartile
    fig, ax = plt.subplots(figsize=(5, 4))
    width = 0.35
    for i, s in enumerate(seeds):
        q = summary["seeds"][str(s)]["task_link"]["quartiles"]["mean_gain"]
        ax.bar(np.arange(4) + (i - 0.5) * width, q, width, label=f"seed{s}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylabel("mean recurrent gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gain_by_contextualization_quartile.png", dpi=140)
    plt.close(fig)

    # prototype contextual variance
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in seeds:
        protos = summary["seeds"][str(s)]["prototypes"]["prototypes"]
        ratio = [p["var_delta_perdim"] / (p["var_h0_perdim"] + EPS) for p in protos]
        ax.scatter([p["n"] for p in protos], ratio, s=8, alpha=0.6, label=f"seed{s}")
    ax.set_xscale("log")
    ax.set_xlabel("prototype size n_k")
    ax.set_ylabel("Var(delta|k) / Var(h0|k)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "prototype_context_variance.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary markdown
# ---------------------------------------------------------------------------


def _write_results_summary(summary: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# CGA-v0 — Contextualization Gap Audit — results summary")
    lines.append("")
    lines.append(f"Protocol `{PROTOCOL_VERSION}`; commit `{summary['git_commit'][:8]}`.")
    lines.append("`max_new_full_training_runs = 0`; `official_test_loaded = false`.")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    for seed, entry in summary["integrity"]["seeds"].items():
        lines.append(
            f"- seed{seed}: MAE {entry['valid_mae']:.8f} vs recorded "
            f"{entry['recorded_valid_mae']:.8f} (|Δ|={entry['A_prediction_abs_diff']:.2e}), "
            f"A={entry['A_prediction_pass']}, C={entry.get('C_reconstruction_pass')}, "
            f"B={entry.get('B_state_ordering_pass')}, D={entry.get('D_determinism_pass')}"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{summary['decision']['case']}`")
    lines.append("")
    lines.append("| seed | R_delta(H0) | ret.dDelta | mean J | P_local | P_context | context_gain | context_abs_gain | Spearman(C_h,gain) | Q4-Q1 gain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed, f in summary["decision"]["flags"].items():
        lines.append(
            f"| {seed} | {f['R_delta_median_H0']:.3f} | {f['retention_dDelta_over_random']:.3f} | "
            f"{f['mean_jaccard']:.3f} | "
            f"{f['P_local']:.3f} | {f['P_context']:.3f} | {f['context_gain']:.3f} | "
            f"{f['context_abs_gain']:.4f} | "
            f"{f['spearman_C_h_gain']:.3f} | {f['q4_minus_q1']:.4f} |"
        )
    lines.append("")
    lines.append("## Prototype context split (K=64)")
    lines.append("")
    lines.append("| seed | prototypes used | weighted mean S_k | median S_k |")
    lines.append("|---|---:|---:|---:|")
    for seed, entry in summary["seeds"].items():
        p = entry["prototypes"]
        lines.append(f"| {seed} | {p['n_prototypes_used']} | {p['weighted_mean_S_k']:.3f} | {p['median_S_k']:.3f} |")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CGA-v0 contextualization gap audit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("checkpoints")
    sub.add_parser("verify")

    export_p = sub.add_parser("export")
    export_p.add_argument("--role", required=True, choices=["recurrent", "baseline", "bnull"])
    export_p.add_argument("--seed", type=int, default=0)
    export_p.add_argument("--q-subset", type=int, default=0)

    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--seed", type=int, required=True)

    full_p = sub.add_parser("full")
    full_p.add_argument("--no-bnull", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "checkpoints":
        print(json.dumps(_jsonable(cmd_checkpoints()), indent=2)[:2000])
    elif args.command == "verify":
        print(json.dumps(_jsonable(cmd_verify()), indent=2)[:2000])
    elif args.command == "export":
        print(json.dumps(_jsonable(cmd_export(args.role, args.seed, q_subset_mol=args.q_subset)), indent=2))
    elif args.command == "audit":
        print(json.dumps(_jsonable(run_seed(args.seed)), indent=2)[:4000])
    elif args.command == "full":
        summary = cmd_full(include_bnull=not args.no_bnull)
        print(json.dumps(_jsonable(summary["decision"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
