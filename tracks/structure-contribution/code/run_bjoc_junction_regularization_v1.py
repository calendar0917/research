"""cscl-bjoc-junction-regularization-v1 driver.

Protocol: ``configs/bjoc_junction_regularization_v1.yaml`` (frozen before any
formal GPU run).  Track: ``tracks/structure-contribution``.

The round has exactly ONE predictive intervention: elementwise inverted
dropout on the learned junction embedding ``w`` before the static incidence
grouping consumes it (R0: p=0.0, R1: p=0.10).  Inference semantics are
unchanged: in ``eval`` mode ``w_used = w``.

Stages
------
``hist-error``  Phase A1 — reuse the frozen historical Split-A predictions of
                the representation round and audit why its monitor split was
                harder than its locked-dev split (label access is confined to
                that already-exposed internal split).
``hist-train``  Phase A2 — N0/N1 full-10k training-curve audit plus a fresh
                junction-encoder gradient check on the frozen checkpoints.
``splits``      Phase C — freeze two new canonical-SMILES grouped internal
                splits (A discovery / B replication).  No labels are read.
``objects``     Fit the per-split global-feature standardiser on that split's
                fit rows only (the BJOC object cache is read-only).
``params``      Phase B/D — parameter + initialisation audit for R0/R1.
``gates``       Phase B — numeric dropout-correctness gates (all must pass
                before any GPU trajectory).
``freeze``      Protocol freeze manifest (hashes of everything frozen before
                training).
``train``       Formal training of one arm on one split (device selectable).
``train-stage`` Formal training of a pre-registered group of arms.
``predict``     Label-blind predictions for the trained arms (frozen files).
``score``       ONE read of the internal-split labels -> metrics, paired
                bootstrap, subgroups.
``decide``      Pre-registered promotion gate for split A (and B).
``mechanism``   Phase G diagnostics on the frozen checkpoints (CPU only).
``decision``    Final decision label from the frozen readouts.
``report``      Final report assembly.

Discipline
----------
* ``official valid`` and ``official test`` are never loaded anywhere here.
* training reads only the fit/monitor labels; the locked-dev labels are read
  exactly once inside ``score`` for that split, after every prediction file
  has been written and hashed; split B's locked labels cannot be read during
  any split A decision because ``decide`` reads metric JSONs only.
* raw-y L1 only; no ``g/ell/s/c/k`` target ever enters a loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_junction_dropout_v1 as jd  # noqa: E402
import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-junction-regularization-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_junction_regularization_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_junction_regularization_v1"

# ---- frozen training recipe (identical for every arm) ---------------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SEEDS = (0, 1)

# ---- the single intervention ----------------------------------------------
R0_P = 0.0
R1_P = 0.10

# ---- legacy read-only inputs (previous rounds) -----------------------------
LEGACY_COMPOSER = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"
LEGACY_REPR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_representation_optimization_v1"
OBJECTS_CACHE = LEGACY_COMPOSER / "bjoc_objects_v1.pt"
OBJECTS_CACHE_SHA256 = "08e3ab137e7a4829f5488eb912cd1b01e4a0c87fbf3ef2c1cd9cef6efe61eb12"
LEGACY_REPR_SPLIT_A = LEGACY_REPR / "split_a.json"
LEGACY_REPR_SALT_A = "cscl-bjoc-repr-opt-v1-discovery"

# ---- frozen split salts ----------------------------------------------------
SPLIT_SALTS = {
    "A": "cscl-bjoc-jdrop-v1-discovery",
    "B": "cscl-bjoc-jdrop-v1-replication",
}
FIT_TARGET = 8000
MONITOR_TARGET = 1000

# ---- artifacts -------------------------------------------------------------
SPLIT_FILES = {"A": OUT / "split_a.json", "B": OUT / "split_b.json"}
NORM_FILES = {"A": OUT / "global_norm_a.json", "B": OUT / "global_norm_b.json"}

# ---- frozen promotion gate (pre-registered thresholds) ---------------------
GATE_MONITOR_SEED_MEAN_DELTA_MAX = -0.006
GATE_LOCKED_SEED_MEAN_DELTA_MAX = -0.003
GATE_WORST_ROW_SHARE_MAX = 0.05
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0

BUDGET = {
    "wall_clock_minutes_total": 465,
    "reserve_for_pull_verify_report_git_minutes": 35,
    "gpu_hours_max": 2.0,
    "formal_trajectories_max": 8,
    "concurrency": 1,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_rng_state_sha256() -> str:
    state = torch.get_rng_state().numpy().tobytes()
    return hashlib.sha256(state).hexdigest()


def load_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# data / splits / labels
# ---------------------------------------------------------------------------


def load_smiles() -> list[str]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(smiles) == 10000, len(smiles)
    return smiles


def build_split(smiles: Sequence[str], salt: str) -> dict[str, Any]:
    """Deterministic canonical-SMILES grouped split; labels never enter."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(fit) < FIT_TARGET:
            fit.extend(rows)
        elif len(monitor) < MONITOR_TARGET:
            monitor.extend(rows)
        else:
            locked.extend(rows)
    fit, monitor, locked = sorted(fit), sorted(monitor), sorted(locked)
    sizes = [len(v) for v in groups.values()]
    return {
        "salt": salt,
        "fit": fit,
        "monitor": monitor,
        "locked": locked,
        "n_groups": len(groups),
        "duplicate_group_sizes": {
            "n_groups_multi_row": int(sum(1 for v in sizes if v > 1)),
            "n_rows_in_multi_row_groups": int(sum(v for v in sizes if v > 1)),
            "max_group_size": int(max(sizes)),
            "n_groups_gt2": int(sum(1 for v in sizes if v > 2)),
        },
        "overlap": {
            "fit_monitor": len(set(fit) & set(monitor)),
            "fit_locked": len(set(fit) & set(locked)),
            "monitor_locked": len(set(monitor) & set(locked)),
            "union": len(set(fit) | set(monitor) | set(locked)),
        },
    }


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def split_payload(split: Mapping[str, Any], crosstab: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_ID,
        "salt": split["salt"],
        "fit_target": FIT_TARGET,
        "monitor_target": MONITOR_TARGET,
        "n_groups": split["n_groups"],
        "n_fit": len(split["fit"]),
        "n_monitor": len(split["monitor"]),
        "n_locked": len(split["locked"]),
        "overlap": split["overlap"],
        "duplicate_group_sizes": split["duplicate_group_sizes"],
        "fit_sha256_rows": rows_sha256(split["fit"]),
        "monitor_sha256_rows": rows_sha256(split["monitor"]),
        "locked_sha256_rows": rows_sha256(split["locked"]),
        "crosstab_with_A": dict(crosstab or {}),
        "locked_labels_read": False,
    }


def _ensure_splits(log=print) -> dict[str, dict[str, Any]]:
    """Build the two splits, write the frozen JSONs if missing, verify hashes."""
    config = load_config()
    smiles = load_smiles()
    splits = {k: build_split(smiles, SPLIT_SALTS[k]) for k in ("A", "B")}
    crosstab = {
        name: {
            "A_fit": len(set(splits["A"]["fit"]) & set(splits["B"][name])),
            "A_monitor": len(set(splits["A"]["monitor"]) & set(splits["B"][name])),
            "A_locked": len(set(splits["A"]["locked"]) & set(splits["B"][name])),
        }
        for name in ("fit", "monitor", "locked")
    }
    for key, split in splits.items():
        if not SPLIT_FILES[key].exists():
            write_json(SPLIT_FILES[key], split_payload(split, crosstab if key == "B" else None))
            log(f"[splits] wrote {SPLIT_FILES[key].name}")
        frozen = read_json(SPLIT_FILES[key])
        assert frozen["salt"] == SPLIT_SALTS[key]
        assert rows_sha256(split["fit"]) == frozen["fit_sha256_rows"], f"split {key} fit drifted"
        assert rows_sha256(split["monitor"]) == frozen["monitor_sha256_rows"], f"split {key} monitor drifted"
        assert rows_sha256(split["locked"]) == frozen["locked_sha256_rows"], f"split {key} locked drifted"
        # the config carries the frozen row hashes (travels with Git)
        cfg_split = config["splits"][key]
        for name in ("fit", "monitor", "locked"):
            assert cfg_split[f"{name}_sha256_rows"] == frozen[f"{name}_sha256_rows"], (
                f"config vs frozen split {key}/{name} hash mismatch"
            )
        assert split["overlap"]["union"] == 10000
        assert split["overlap"]["fit_monitor"] == 0
        assert split["overlap"]["fit_locked"] == 0
        assert split["overlap"]["monitor_locked"] == 0
    if not (OUT / "split_crosstab.json").exists():
        write_json(
            OUT / "split_crosstab.json",
            {
                "protocol": PROTOCOL_ID,
                "smiles_sha256": hashlib.sha256("\n".join(smiles).encode()).hexdigest(),
                "crosstab_A_vs_B_rows": crosstab,
                "expected_locked_overlap_note": (
                    "nested hash splits of the same 10k pool; the A/B locked row overlap is "
                    "reported, never optimised"
                ),
            },
        )
    return splits


def read_split_rows(key: str) -> dict[str, list[int]]:
    split = _ensure_splits()[key]
    return {"fit": split["fit"], "monitor": split["monitor"], "locked": split["locked"]}


def _load_encoded_train():
    return torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)


def load_y_rows(rows: Sequence[int], *, subset_name: str = "unspecified") -> np.ndarray:
    """Read the raw y of exactly these official-train rows.

    This helper is deliberately the ONLY label reader in the module; callers
    pass fit/monitor rows during training and locked rows only inside
    ``cmd_score``.  ``subset_name`` is recorded by the caller's manifest.
    """
    train = _load_encoded_train()
    out = np.asarray([float(train[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)
    del train
    return out


def y_of_rows_from(cache: Sequence[Any], rows: Sequence[int]) -> np.ndarray:
    return np.asarray([float(cache[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)


def load_objects() -> list[sro.MolObjects]:
    assert OBJECTS_CACHE.exists(), OBJECTS_CACHE
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    assert len(objs) == 10000, len(objs)
    return objs


def load_norm(split_key: str) -> tuple[np.ndarray, np.ndarray]:
    norm = read_json(NORM_FILES[split_key])
    assert norm["fit_only"] is True and norm["n_fit"] == FIT_TARGET
    return (
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# small structural helpers (used by the historical audit and subgroups)
# ---------------------------------------------------------------------------


def graph_diameter(n_atoms: int, bond_index: np.ndarray) -> int:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in np.asarray(bond_index, dtype=np.int64):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    if n_atoms <= 1:
        return 0
    best = 0
    for source in range(int(n_atoms)):
        dist = [-1] * int(n_atoms)
        dist[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            continue
        best = max(best, max(dist))
    return int(best)


def structural_stats(objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[str, np.ndarray]:
    n_atoms, n_bonds, n_junc, cyclomatic, diameter = [], [], [], [], []
    for r in rows:
        o = objs[int(r)]
        na = int(o.n_atoms)
        nb = int(o.bond_index.shape[0])
        n_atoms.append(na)
        n_bonds.append(nb)
        n_junc.append(int(o.junc_index.shape[0]))
        cyclomatic.append(nb - na + 1)
        diameter.append(graph_diameter(na, o.bond_index))
    return {
        "n_atoms": np.asarray(n_atoms, dtype=np.float64),
        "n_bonds": np.asarray(n_bonds, dtype=np.float64),
        "n_junctions": np.asarray(n_junc, dtype=np.float64),
        "cyclomatic": np.asarray(cyclomatic, dtype=np.float64),
        "diameter": np.asarray(diameter, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Phase A1 — historical monitor-vs-locked error audit (zero training)
# ---------------------------------------------------------------------------


def _dist(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "q05": float(np.percentile(v, 5)),
        "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "q95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def _error_profile(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    err = np.abs(pred - y)
    order = np.argsort(-err)
    n = err.size
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
        "top1_err_mass_share": float(err[order[0]] / err.sum()),
        "top5_err_mass_share": float(err[order[:5]].sum() / err.sum()),
        "top10_err_mass_share": float(err[order[:10]].sum() / err.sum()),
        "worst_row": int(order[0]),
        "worst_y": float(y[order[0]]),
        "worst_abs_err": float(err[order[0]]),
        "mae_without_worst": float(err[np.arange(n) != order[0]].mean()),
        "mae_without_top5": float(err[order[5:]].mean()),
        "mae_without_top10": float(err[order[10:]].mean()),
        "frac_err_gt_1": float((err > 1.0).mean()),
    }


def cmd_hist_error(args: argparse.Namespace) -> int:
    """A1: why was the historical representation-round monitor split harder?"""
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "source": "results/bjoc_representation_optimization_v1 (split A of the representation round)",
        "purpose": (
            "understand the evaluation instrument of a completed round; "
            "the result may not be used to choose the new splits (which are salt-frozen)"
        ),
        "official_valid_or_test_loaded": False,
    }
    hist_split = read_json(LEGACY_REPR_SPLIT_A)
    assert hist_split["salt"] == LEGACY_REPR_SALT_A
    smiles = load_smiles()
    split = build_split(smiles, LEGACY_REPR_SALT_A)
    assert rows_sha256(split["fit"]) == hist_split["fit_sha256_rows"]
    monitor_rows = split["monitor"]
    locked_rows = split["locked"]
    mon = np.load(LEGACY_REPR / "pred_B_A_s0_monitor.npz")
    locked = np.load(LEGACY_REPR / "pred_B_A_s0_locked.npz")
    assert (mon["row"] == np.asarray(monitor_rows)).all()
    assert (locked["row"] == np.asarray(locked_rows)).all()
    y_mon = np.asarray(mon["y"], dtype=np.float64)
    y_locked = load_y_rows(locked_rows, subset_name="historical locked dev (already exposed)")
    pred_mon = np.asarray(mon["pred"], dtype=np.float64)
    pred_locked = np.asarray(locked["pred"], dtype=np.float64)
    objs = load_objects()

    empty_frac = {
        "y_le_-3": {"monitor": float((y_mon <= -3).mean()), "locked": float((y_locked <= -3).mean())},
        "y_le_-2": {"monitor": float((y_mon <= -2).mean()), "locked": float((y_locked <= -2).mean())},
        "y_le_-1": {"monitor": float((y_mon <= -1).mean()), "locked": float((y_locked <= -1).mean())},
    }
    out["y_distribution"] = {"monitor": _dist(y_mon), "locked": _dist(y_locked)}
    out["low_tail_fraction"] = empty_frac
    out["error_profile"] = {
        "monitor_B": _error_profile(y_mon, pred_mon),
        "locked_B": _error_profile(y_locked, pred_locked),
    }
    # low-y tail error behaviour
    out["low_tail_error"] = {}
    for name, (y, p) in {"monitor": (y_mon, pred_mon), "locked": (y_locked, pred_locked)}.items():
        err = np.abs(p - y)
        mask = y <= -3.0
        out["low_tail_error"][name] = {
            "n": int(mask.sum()),
            "mae": float(err[mask].mean()) if mask.any() else None,
            "err_mass_share": float(err[mask].sum() / err.sum()) if mask.any() else None,
        }
    # structural strata
    ss_mon = structural_stats(objs, monitor_rows)
    ss_locked = structural_stats(objs, locked_rows)
    out["structure"] = {
        split_name: {k: _dist(v) for k, v in stats.items()}
        for split_name, stats in {"monitor": ss_mon, "locked": ss_locked}.items()
    }
    # duplicate canonical-SMILES groups inside each set (labels never used for grouping)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    out["duplicate_groups"] = {}
    for name, rows in (("monitor", monitor_rows), ("locked", locked_rows)):
        rows_set = set(int(r) for r in rows)
        inside = [sorted(v) for v in groups.values() if len(v) > 1 and set(v) <= rows_set]
        rows_in = sum(len(v) for v in inside)
        out["duplicate_groups"][name] = {
            "n_groups_inside": len(inside),
            "n_rows_inside": int(rows_in),
            "fraction_rows": float(rows_in) / len(rows),
        }
    # per-split single-worst-error removal (descriptive only)
    out["single_error_removal"] = {
        "monitor": {
            "worst_abs_err": float(np.abs(pred_mon - y_mon).max()),
            "mae_without_the_single_worst": float(
                np.sort(np.abs(pred_mon - y_mon))[:-1].mean()
            ),
        },
        "locked": {
            "worst_abs_err": float(np.abs(pred_locked - y_locked).max()),
            "mae_without_the_single_worst": float(
                np.sort(np.abs(pred_locked - y_locked))[:-1].mean()
            ),
        },
    }
    out["interpretation"] = (
        "The representation-round monitor split was the harder instrument: compare the y "
        "lower tail, the structural strata and the error mass concentration above before "
        "treating its MAE as comparable to the locked-dev MAE of the same round. The new "
        "round's splits use different frozen salts and were fixed without reference to any "
        "of these numbers."
    )
    del objs
    write_json(OUT / "historical_error_audit.json", out)
    log(
        "[hist-error] monitor y<=-3=%.3f locked=%.3f | monitor MAE %.5f locked %.5f"
        % (
            empty_frac["y_le_-3"]["monitor"],
            empty_frac["y_le_-3"]["locked"],
            out["error_profile"]["monitor_B"]["mae"],
            out["error_profile"]["locked_B"]["mae"],
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase A2 — historical training-curve + gradient audit (zero training)
# ---------------------------------------------------------------------------


def _curve_segments(curve: np.ndarray, segments: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    out = []
    for a, b in segments:
        seg = curve[a - 1 : b]
        out.append(
            {
                "epochs": [a, b],
                "mean": float(seg.mean()),
                "first": float(seg[0]),
                "last": float(seg[-1]),
                "std": float(seg.std()),
            }
        )
    return out


def _audit_one_curve(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    curve = np.asarray([e["train_mae"] for e in payload["curve"]], dtype=np.float64)
    monitor = np.asarray(
        [e["monitor_mae"] if e.get("monitor_mae") is not None else np.nan for e in payload["curve"]],
        dtype=np.float64,
    )
    out = {
        "path": str(path.relative_to(REPO_ROOT)),
        "arm": payload.get("arm"),
        "seed": payload.get("seed"),
        "soup_fit_mae": payload.get("soup_fit_mae"),
        "soup_monitor_mae": payload.get("soup_monitor_mae"),
        "wall_seconds": payload.get("wall_seconds"),
        "epochs": payload.get("epochs"),
        "soup_members": payload.get("soup_members"),
        "segments": _curve_segments(curve, [(1, 40), (40, 80), (80, 120), (120, 160), (160, 200), (200, 240)]),
        "last5_epoch_train_mae": [float(v) for v in curve[-5:]],
        "last5_mean_train_mae": float(curve[-5:].mean()),
        "soup_minus_last5_epoch_mae": float(payload["soup_fit_mae"]) - float(curve[-5:].mean()),
        "curve_all_finite": bool(np.isfinite(curve).all()),
        "monitor_all_finite": bool(np.isfinite(monitor).all()) if np.isfinite(monitor).any() else None,
        "train_mae_monotone_tail": bool(np.all(np.diff(curve[-20:]) <= 0) or np.mean(np.diff(curve[-20:])) < 0),
        "late_slope_per_epoch": float(np.polyfit(np.arange(200, 241), curve[199:240], 1)[0]),
        "min_train_mae": float(curve.min()),
        "max_train_mae": float(curve.max()),
    }
    if np.isfinite(monitor).any():
        out["monitor_min"] = float(np.nanmin(monitor))
        out["monitor_min_epoch"] = int(np.nanargmin(monitor) + 1)
    return out


def _junction_gradient_check(checkpoint: Path, objs: Sequence[sro.MolObjects], y: np.ndarray) -> dict[str, Any]:
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=0,
        organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    model.train()
    batch = sro.batch_objects([objs[i] for i in range(256)])
    target = torch.as_tensor(y[:256], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    pred, feats = model(batch, return_features=True)
    feats["w"].retain_grad()
    loss = F.l1_loss(pred.view(-1), target.view(-1))
    loss.backward()
    grad_norm = float(sum(p.grad.pow(2).sum() for p in model.junction_encoder.parameters()).sqrt())
    w_grad_norm = float(feats["w"].grad.pow(2).sum().sqrt()) if feats["w"].grad is not None else None
    return {
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "loss_on_256_fit_rows": float(loss),
        "junction_encoder_grad_norm": grad_norm,
        "junction_grad_live": bool(grad_norm > 0.0),
        "w_grad_norm": w_grad_norm,
        "w_grad_live": bool((w_grad_norm or 0.0) > 0.0),
    }


def cmd_hist_train(args: argparse.Namespace) -> int:
    """A2: N0/N1 trajectory + gradient audit (zero new training)."""
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "purpose": "confirm that a junction-embedding intervention is an engineering-meaningful, testable knob",
        "official_valid_or_test_loaded": False,
    }
    n0_path = LEGACY_COMPOSER / "train_bjoc_null_full10k_s0.json"
    n1_path = LEGACY_COMPOSER / "train_bjoc_null_full10k_s1.json"
    out["full10k_N0"] = _audit_one_curve(n0_path)
    out["full10k_N1"] = _audit_one_curve(n1_path)
    out["fit_gap_N0_minus_N1"] = float(out["full10k_N0"]["soup_fit_mae"] - out["full10k_N1"]["soup_fit_mae"])
    out["soup_of_last5_vs_curve"] = {
        "N0": out["full10k_N0"]["soup_minus_last5_epoch_mae"],
        "N1": out["full10k_N1"]["soup_minus_last5_epoch_mae"],
    }
    # historical representation-round curves provide the split-A reference
    b_path = LEGACY_REPR / "train_B_A_s0.json"
    if b_path.exists():
        out["internal_split_A_B"] = _audit_one_curve(b_path)
    # gradient liveness on the frozen checkpoints
    objs = load_objects()
    y = load_y_rows(list(range(256)), subset_name="first 256 train rows (gradient check)")
    out["junction_encoder_gradient"] = {
        "N0": _junction_gradient_check(LEGACY_COMPOSER / "bjoc_null_full10k_s0_soup_state.pt", objs, y),
        "N1": _junction_gradient_check(LEGACY_COMPOSER / "bjoc_null_full10k_s1_soup_state.pt", objs, y),
    }
    del objs
    # Phase A3 — junction-necessity reading from the completed representation round
    jn_path = LEGACY_REPR / "junction_necessity_analysis.json"
    if jn_path.exists():
        jn = read_json(jn_path)
        out["junction_necessity_reused"] = {
            "source": str(jn_path.relative_to(REPO_ROOT)),
            "monitor_J_minus_B": jn.get("monitor_J_minus_B"),
            "locked_J_minus_B": jn.get("locked_J_minus_B"),
            "locked_ci95": jn.get("locked_ci95"),
            "band": jn.get("band"),
            "narrow_reading": (
                "under the previous round's matched capacity and fixed recipe, removing the learned "
                "junction content degraded held-out predictions systematically. This is NOT a proof "
                "that the junction construction is mathematically irreplaceable."
            ),
        }
    out["interpretation"] = (
        "Both historical seeds descend continuously to epoch 240, the last-5 soup differs modestly "
        "from the last epoch, and the junction encoder receives non-zero gradients on the frozen "
        "checkpoints. A training-time dropout on the junction embedding is therefore a well-defined, "
        "testable intervention on a live pathway."
    )
    write_json(OUT / "historical_training_audit.json", out)
    log(
        "[hist-train] N0 fit %.5f N1 fit %.5f | junction grad live %s"
        % (
            out["full10k_N0"]["soup_fit_mae"],
            out["full10k_N1"]["soup_fit_mae"],
            all(v["junction_grad_live"] for v in out["junction_encoder_gradient"].values()),
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase C — splits
# ---------------------------------------------------------------------------


def cmd_splits(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = _ensure_splits()
    for key, split in splits.items():
        log(
            f"[split {key}] salt={split['salt']} fit={len(split['fit'])} "
            f"monitor={len(split['monitor'])} locked={len(split['locked'])}"
        )
    ct = read_json(OUT / "split_crosstab.json")
    log(f"[split] crosstab A vs B: {json.dumps(ct['crosstab_A_vs_B_rows'])}")
    return 0


# ---------------------------------------------------------------------------
# objects / standardiser
# ---------------------------------------------------------------------------


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    cache_sha = file_sha256(OBJECTS_CACHE)
    assert cache_sha == OBJECTS_CACHE_SHA256, f"objects cache drifted: {cache_sha}"
    objs = load_objects()
    for key in ("A", "B"):
        rows = read_split_rows(key)
        gfit = np.stack([objs[int(i)].global_feat for i in rows["fit"]], 0).astype(np.float64)
        gmean = gfit.mean(0)
        gscale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        write_json(
            NORM_FILES[key],
            {
                "protocol": PROTOCOL_ID,
                "split": key,
                "fit_only": True,
                "n_fit": int(len(rows["fit"])),
                "global_mean": gmean.tolist(),
                "global_scale": gscale.tolist(),
                "names": list(sro.GLOBAL_FEAT_NAMES),
                "labels_read": False,
            },
        )
        log(f"[objects] global norm {key} fitted on {len(rows['fit'])} fit rows (labels never read)")
    write_json(
        OUT / "objects_summary.json",
        {
            "protocol": PROTOCOL_ID,
            "objects_cache": str(OBJECTS_CACHE.relative_to(REPO_ROOT)),
            "objects_cache_sha256": cache_sha,
            "n_rows": len(objs),
            "n_atoms": int(sum(o.n_atoms for o in objs)),
            "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
            "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
            "labels_read": False,
        },
    )
    del objs
    return 0


# ---------------------------------------------------------------------------
# Phase B/D — parameter, initialisation and dropout-correctness gates
# ---------------------------------------------------------------------------


def _tiny_batch(objs: Sequence[sro.MolObjects], rows: Sequence[int], labels=None):
    batch = sro.batch_objects([objs[int(r)] for r in rows])
    if labels is not None:
        batch["y"] = torch.as_tensor(np.asarray(labels, dtype=np.float32)).view(-1, 1)
    return batch


def cmd_params(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gm = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    gs = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "published_bjoc_parameters": jd.PUBLISHED_PARAMETERS,
        "authorised_probabilities": {"R0": R0_P, "R1": R1_P},
        "arms": {},
        "additional_trainable_parameters": [],
        "official_valid_or_test_loaded": False,
    }
    for p, name in ((R0_P, "R0"), (R1_P, "R1")):
        entry = {"junction_dropout_p": p, "seeds": {}}
        for seed in SEEDS:
            model = jd.build_model(seed=seed, junction_dropout_p=p, global_mean=gm, global_scale=gs)
            entry["seeds"][str(seed)] = {
                "parameters": int(sum(q.numel() for q in model.parameters())),
                "parameter_exact": jd.parameter_audit(model)["parameter_exact"],
                "init_state_sha256": jd.state_dict_hash(model),
            }
            del model
        payload["arms"][name] = entry
    for seed in SEEDS:
        payload["arms"][f"init_match_seed{seed}"] = (
            payload["arms"]["R0"]["seeds"][str(seed)]["init_state_sha256"]
            == payload["arms"]["R1"]["seeds"][str(seed)]["init_state_sha256"]
        )
    # state-dict key identity vs the frozen backbone -> no new trainable tensor
    base = src.build_bjoc(src.BJOCConfig(), seed=0, organization_mode=src.NULL, global_mean=gm, global_scale=gs)
    jdrop = jd.build_model(seed=0, junction_dropout_p=R1_P, global_mean=gm, global_scale=gs)
    base_keys = set(base.state_dict().keys())
    jd_keys = set(jdrop.state_dict().keys())
    payload["state_dict_keys_equal_backbone"] = base_keys == jd_keys
    payload["new_state_dict_keys"] = sorted(jd_keys - base_keys)
    payload["missing_state_dict_keys"] = sorted(base_keys - jd_keys)
    # every parameter name is inherited unchanged (no hidden bypass tensor)
    base_param_names = sorted(n for n, _ in base.named_parameters())
    jd_param_names = sorted(n for n, _ in jdrop.named_parameters())
    payload["parameter_names_equal_backbone"] = base_param_names == jd_param_names
    write_json(OUT / "parameter_and_init_audit.json", payload)
    log(
        "[params] R0/R1 params=%d exact=%s init_match=%s"
        % (
            payload["arms"]["R0"]["seeds"]["0"]["parameters"],
            payload["arms"]["R0"]["seeds"]["0"]["parameter_exact"],
            payload["arms"]["init_match_seed0"] and payload["arms"]["init_match_seed1"],
        )
    )
    return 0


def _with_junction_hooks(model, batch):
    """Capture (w, w_used) and the composer input of one forward pass."""
    capture: dict[str, torch.Tensor] = {}

    def hook_junc(_module, _inputs, output):
        capture["w"] = output.detach().clone()

    def hook_group(w, _batch, mode):
        capture["group_input"] = w.detach().clone()
        return src.BJOC.group_to_bonds(model, w, _batch, mode)

    def hook_composer(_module, inputs):
        capture["compose_in"] = inputs[0].detach().clone()

    handles = [
        model.junction_encoder.register_forward_hook(hook_junc),
        model.group_composer.register_forward_pre_hook(hook_composer),
    ]
    original_group = type(model).group_to_bonds
    type(model).group_to_bonds = lambda self, w, b, m: hook_group(w, b, m)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        type(model).group_to_bonds = original_group
        for h in handles:
            h.remove()
    return capture


def cmd_gates(args: argparse.Namespace) -> int:
    """Phase B numeric gates.  All must pass before any formal GPU run."""
    OUT.mkdir(parents=True, exist_ok=True)
    objs = load_objects()
    rows = read_split_rows("A")["fit"][:256]
    gm = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    gs = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    batch = _tiny_batch(objs, rows, labels=np.zeros(len(rows)))
    gates: dict[str, Any] = {"protocol": PROTOCOL_ID, "n_gate_rows": len(rows)}
    failures: list[str] = []

    def record(name: str, ok: bool, payload: Any = None) -> None:
        gates[name] = {"pass": bool(ok), "payload": _j(payload)}
        if not ok:
            failures.append(name)

    # 1. R0 forward bit-identical to the frozen backbone on the same weights
    seed = 20261012
    base = src.build_bjoc(src.BJOCConfig(), seed=seed, organization_mode=src.NULL, global_mean=gm, global_scale=gs)
    r0 = jd.build_model(seed=seed, junction_dropout_p=R0_P, global_mean=gm, global_scale=gs)
    base.eval()
    r0.eval()
    with torch.no_grad():
        pb = base(batch)
        p0 = r0(batch)
    d = float((pb - p0).abs().max())
    record("gate01_R0_forward_matches_frozen_backbone", d == 0.0, {"max_abs_diff": d})

    # 2/3. parameter counts
    r1 = jd.build_model(seed=seed, junction_dropout_p=R1_P, global_mean=gm, global_scale=gs)
    n0 = int(sum(p.numel() for p in r0.parameters()))
    n1 = int(sum(p.numel() for p in r1.parameters()))
    record("gate02_R0_parameters_413505", n0 == jd.PUBLISHED_PARAMETERS, {"parameters": n0})
    record("gate03_R1_parameters_413505", n1 == jd.PUBLISHED_PARAMETERS, {"parameters": n1})

    # 4. identical initialisation for R0/R1 at several seeds
    init_match = {}
    for s in SEEDS:
        a = jd.build_model(seed=s, junction_dropout_p=R0_P, global_mean=gm, global_scale=gs)
        b = jd.build_model(seed=s, junction_dropout_p=R1_P, global_mean=gm, global_scale=gs)
        init_match[str(s)] = jd.state_dict_hash(a) == jd.state_dict_hash(b)
    record("gate04_init_state_identical_R0_R1", all(init_match.values()), init_match)

    # 5. eval predictions identical between R0/R1 sharing a weight state
    shared = {k: v.clone() for k, v in r0.state_dict().items()}
    r1.load_state_dict(shared)
    r1.eval()
    with torch.no_grad():
        p1 = r1(batch)
    d_eval = float((p0 - p1).abs().max())
    record("gate05_eval_predictions_identical_shared_weights", d_eval == 0.0, {"max_abs_diff": d_eval})

    # 6/7. R1 train dropout non-zero; R0 train deterministic and RNG-neutral
    r1.train()
    state_before = torch_rng_state_sha256()
    with torch.no_grad():
        a1 = r1(batch)
        a2 = r1(batch)
    state_after = torch_rng_state_sha256()
    record(
        "gate06_R1_train_has_dropout",
        (not torch.equal(a1, a2)) and state_before != state_after,
        {"rng_consumed": state_before != state_after},
    )
    r0.train()
    state_before0 = torch_rng_state_sha256()
    with torch.no_grad():
        b1 = r0(batch)
        b2 = r0(batch)
    state_after0 = torch_rng_state_sha256()
    record(
        "gate07_R0_train_is_identity_and_rng_neutral",
        torch.equal(b1, b2) and state_before0 == state_after0,
        {"rng_changed": state_before0 != state_after0},
    )

    # 8. dropout only on the junction embedding: hooks show W/V consumed from
    #    a dropped tensor while a and b are exactly the t rain-path tensors
    r1.train()
    torch.manual_seed(0)
    cap = _with_junction_hooks(r1, batch)
    w, w_used = cap["w"], cap["group_input"]
    zeros_frac = float((w_used == 0).float().mean())
    scale_ok = bool(torch.allclose(w_used[w_used != 0], w[w_used != 0] / (1.0 - R1_P), atol=1e-5))
    expected_ok = 0.05 <= zeros_frac <= 0.16
    record(
        "gate08_dropout_only_on_junction_embedding",
        expected_ok and scale_ok,
        {"zero_fraction": zeros_frac, "inverted_scaling_ok": scale_ok},
    )

    # 9. every junction survives in the eval-mode input (dropout off)
    r1.eval()
    cap_eval = _with_junction_hooks(r1, batch)
    record(
        "gate09_eval_preserves_all_junctions",
        torch.equal(cap_eval["w"], cap_eval["group_input"]),
        {"max_abs_diff": float((cap_eval["w"] - cap_eval["group_input"]).abs().max())},
    )

    # 10. junction encoder backprop is live (both arms, train mode)
    grad_payload = {}
    for name, model in (("R0", r0), ("R1", r1)):
        model.train()
        model.zero_grad(set_to_none=True)
        _, feats = model(batch, return_features=True)
        F.l1_loss(feats["reader_in"].sum(dim=1), torch.zeros(len(rows))).backward()
        gn = float(sum(p.grad.pow(2).sum() for p in model.junction_encoder.parameters()).sqrt())
        grad_payload[name] = {"junction_encoder_grad_norm": gn, "live": gn > 0}
    record("gate10_junction_encoder_backprop_live", all(v["live"] for v in grad_payload.values()), grad_payload)

    # 11. both moments consume w_used
    r1.train()
    torch.manual_seed(1)
    cap = _with_junction_hooks(r1, batch)
    w_used = cap["group_input"]
    W, V, n_b = r1.group_to_bonds(w_used, batch, src.NULL)
    # recompute the same moments from the pre-dropout w and verify they differ
    W_raw, V_raw, _ = r1.group_to_bonds(cap["w"], batch, src.NULL)
    both_used = float((W - W_raw).abs().sum()) > 0.0 and float((V - V_raw).abs().sum()) > 0.0
    record(
        "gate11_first_and_second_moments_use_w_used",
        both_used,
        {
            "delta_W_sum_abs": float((W - W_raw).abs().sum()),
            "delta_V_sum_abs": float((V - V_raw).abs().sum()),
        },
    )

    # 12. atom permutation invariance in eval mode: rebuild one molecule with a
    #     relabelled atom order and compare the prediction
    o = objs[int(rows[0])]
    perm = np.random.default_rng(3).permutation(int(o.n_atoms))
    inv = np.argsort(perm)
    obj_p = sro.MolObjects(
        mol_index=-1,
        smiles="permuted",
        n_atoms=int(o.n_atoms),
        atom_feat=np.asarray(o.atom_feat, dtype=np.float32)[perm],
        bond_index=inv[np.asarray(o.bond_index, dtype=np.int64)],
        bond_type=np.asarray(o.bond_type, dtype=np.int64),
        bond_feat=np.asarray(o.bond_feat, dtype=np.float32),
        junc_index=inv[np.asarray(o.junc_index, dtype=np.int64)],
        junc_bond=np.asarray(o.junc_bond, dtype=np.int64),
        junc_feat=np.asarray(o.junc_feat, dtype=np.float32),
        global_feat=np.asarray(o.global_feat, dtype=np.float32),
    )
    r1.eval()
    with torch.no_grad():
        pred_o = float(r1(sro.batch_objects([o])).item())
        pred_p = float(r1(sro.batch_objects([obj_p])).item())
    d_perm = abs(pred_o - pred_p)
    record("gate12_permutation_invariance_eval", d_perm < 1e-5, {"max_abs_diff": d_perm})

    # 13. batch consistency in eval mode: one batch vs two half batches
    r1.eval()
    half = len(rows) // 2
    b_all = _tiny_batch(objs, rows)
    b_a = _tiny_batch(objs, rows[:half])
    b_b = _tiny_batch(objs, rows[half:])
    with torch.no_grad():
        pall = r1(b_all)
        pa = r1(b_a)
        pb_ = r1(b_b)
    db = float((pall - torch.cat([pa, pb_])).abs().max())
    record("gate13_batch_consistency_eval", db < 1e-5, {"max_abs_diff": db})

    # 14. no new trainable parameters (covered again here on the real model)
    new_names = sorted(set(dict(r1.named_parameters())) - set(dict(base.named_parameters())))
    record("gate14_no_new_trainable_parameters", not new_names, {"new_names": new_names})

    # 15. no hidden structural bypass: dropout call count == 1, on w only
    import inspect

    body = inspect.getsource(jd.JunctionDropoutBJOC.forward_features)
    record(
        "gate15_single_dropout_call_on_w",
        body.count("F.dropout(") == 1 and "group_to_bonds(w_used" in body,
        {"dropout_call_count": body.count("F.dropout(")},
    )

    # 16. original structural cache unchanged during this process
    record(
        "gate16_objects_cache_unchanged",
        file_sha256(OBJECTS_CACHE) == OBJECTS_CACHE_SHA256,
        {"sha256": file_sha256(OBJECTS_CACHE)},
    )

    # 17. forward does not read labels (adding a y tensor changes nothing)
    batch_with_y = _tiny_batch(objs, rows, labels=np.full(len(rows), 999.0))
    r1.eval()
    with torch.no_grad():
        p_nolabel = r1(_tiny_batch(objs, rows))
        p_label = r1(batch_with_y)
    dy = float((p_nolabel - p_label).abs().max())
    record("gate17_label_independence", dy == 0.0, {"max_abs_diff": dy})

    # 18. no NaN / Inf in a short train run of R1
    r1.train()
    opt = torch.optim.Adam(r1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    torch.manual_seed(0)
    finite_ok = True
    for _ in range(3):
        pred = r1(batch)
        loss = F.l1_loss(pred.view(-1), torch.zeros(len(rows)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r1.parameters(), GRAD_CLIP)
        opt.step()
        finite_ok = finite_ok and bool(np.isfinite(float(loss)))
    record("gate18_no_nan_inf_in_train_steps", finite_ok, {"steps": 3})

    # 19. save / load replay exact
    tmp = OUT / "_gate_save_load.pt"
    torch.save(r0.state_dict(), tmp)
    r0b = jd.build_model(seed=seed, junction_dropout_p=R0_P, global_mean=gm, global_scale=gs)
    r0b.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False))
    tmp.unlink()
    r0.eval()
    r0b.eval()
    with torch.no_grad():
        d_sl = float((r0(batch) - r0b(batch)).abs().max())
    record("gate19_save_load_replay", d_sl == 0.0, {"max_abs_diff": d_sl})

    # 20. official valid / test not accessed: source scan of this driver
    driver_text = Path(__file__).read_text(encoding="utf-8")
    forbidden = [
        "load_official_" + "valid(",
        "load_official_" + "test(",
        "zinc." + "valid",
        "zinc." + "test",
        "valid_pre" + "dictions.npz",
        "test_pre" + "dictions.npz",
        "official_" + "valid_eval",
    ]
    suspicious = [tok for tok in forbidden if tok in driver_text]
    record(
        "gate20_official_valid_test_not_loaded",
        not suspicious,
        {"suspicious_tokens": suspicious, "flags_present": "official_valid_or_test_loaded"},
    )

    gates["all_numeric_gates_pass"] = not failures
    gates["failures"] = failures
    write_json(OUT / "dropout_correctness_gates.json", gates)
    log(f"[gates] {'ALL PASS' if not failures else 'FAILURES: ' + ','.join(failures)}")
    if failures:
        raise SystemExit(f"dropout correctness gates failed: {failures}")
    return 0


# ---------------------------------------------------------------------------
# protocol freeze
# ---------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    """Run the CPU-only preparation chain in one process (objects/params/gates/freeze)."""
    for fn in (cmd_objects, cmd_params, cmd_gates, cmd_freeze):
        fn(args)
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    _ensure_splits()
    cfg = load_config()
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "recipe": {
            "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "soup_last": SOUP_LAST, "soup_epochs": [236, 240],
            "optimizer": "adam", "amp": False, "scheduler": "none", "loss": "L1(raw y)",
            "precision": "fp32", "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        },
        "intervention": {
            "location": "junction_embedding_w -> F.dropout -> group_to_bonds",
            "R0": {"p": R0_P}, "R1": {"p": R1_P},
            "inference_semantics": "model.eval() => w_used == w; no capacity change",
            "note": "dropout precedes both first and second moment formation",
        },
        "promotion_gate": {
            "monitor_seed_mean_delta_max": GATE_MONITOR_SEED_MEAN_DELTA_MAX,
            "locked_seed_mean_delta_max": GATE_LOCKED_SEED_MEAN_DELTA_MAX,
            "both_seeds_monitor_delta_must_be_negative": True,
            "both_seeds_locked_delta_must_be_non_positive": True,
            "ordinary_seed_mean_locked_delta_max": 0.0,
            "top10_removed_seed_mean_locked_delta_max": 0.0,
            "worst_row_share_max": GATE_WORST_ROW_SHARE_MAX,
            "correctness_gates_required": True,
            "thresholds_are_resource_purchase_rules_not_significance_definitions": True,
        },
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "molecule"},
        "splits": {}, "files": {}, "discipline": {
            "official_valid_or_test_loaded": False,
            "locked_labels_read_at_freeze_time": False,
            "stage_a_plan": [["R0", "A", 0], ["R1", "A", 0], ["R0", "A", 1], ["R1", "A", 1]],
            "stage_b_plan_if_promoted": [["R0", "B", 0], ["R1", "B", 0], ["R0", "B", 1], ["R1", "B", 1]],
            "max_trajectories": BUDGET["formal_trajectories_max"],
        },
        "objects_cache": {
            "path": str(OBJECTS_CACHE.relative_to(REPO_ROOT)),
            "sha256": OBJECTS_CACHE_SHA256,
            "verified": file_sha256(OBJECTS_CACHE) == OBJECTS_CACHE_SHA256,
        },
    }
    for key in ("A", "B"):
        frozen = read_json(SPLIT_FILES[key])
        manifest["splits"][key] = {
            "salt": frozen["salt"],
            "sizes": {"fit": frozen["n_fit"], "monitor": frozen["n_monitor"], "locked": frozen["n_locked"]},
            "fit_sha256_rows": frozen["fit_sha256_rows"],
            "monitor_sha256_rows": frozen["monitor_sha256_rows"],
            "locked_sha256_rows": frozen["locked_sha256_rows"],
            "duplicate_group_sizes": frozen["duplicate_group_sizes"],
            "crosstab_with_A": frozen.get("crosstab_with_A"),
            "locked_labels_read": False,
        }
        manifest["files"][f"split_{key.lower()}.json"] = file_sha256(SPLIT_FILES[key])
        if NORM_FILES[key].exists():
            manifest["files"][f"global_norm_{key.lower()}.json"] = file_sha256(NORM_FILES[key])
    for name, path in (
        ("config.yaml", CONFIG_PATH),
        ("bjoc_junction_dropout_v1.py", _CODE_DIR / "bjoc_junction_dropout_v1.py"),
        ("run_bjoc_junction_regularization_v1.py", _CODE_DIR / "run_bjoc_junction_regularization_v1.py"),
        ("structural_relation_composer_v1.py", _CODE_DIR / "structural_relation_composer_v1.py"),
        ("structural_relation_objects_v1.py", _CODE_DIR / "structural_relation_objects_v1.py"),
        ("test_bjoc_junction_regularization_v1.py", _CODE_DIR.parent / "tests" / "test_bjoc_junction_regularization_v1.py"),
    ):
        manifest["files"][name] = file_sha256(Path(path))
    if (OUT / "dropout_correctness_gates.json").exists():
        gates = read_json(OUT / "dropout_correctness_gates.json")
        manifest["artifacts"] = {"dropout_correctness_gates_pass": bool(gates["all_numeric_gates_pass"]),
                                 "failures": gates.get("failures", [])}
    config_expected = cfg.get("splits", {})
    manifest["config_split_hashes_match_frozen"] = all(
        config_expected[key][f"{name}_sha256_rows"] == manifest["splits"][key][f"{name}_sha256_rows"]
        for key in ("A", "B")
        for name in ("fit", "monitor", "locked")
    )
    write_json(OUT / "protocol_freeze_manifest.json", manifest)
    log("[freeze] protocol_freeze_manifest.json written (git %s)" % manifest["git_commit"][:12])
    return 0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
    return members, soup


def arm_letter(p: float) -> str:
    if p == R0_P:
        return "R0"
    if p == R1_P:
        return "R1"
    raise ValueError(f"unauthorised p {p!r}")


def arm_tag(p: float, split_key: str, seed: int) -> str:
    return f"{arm_letter(p)}_{split_key}_s{seed}"


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model: torch.nn.Module, objs: Sequence[sro.MolObjects], rows: Sequence[int], device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = [objs[int(r)] for r in rows[i : i + 256]]
            batch = _mv(sro.batch_objects(chunk), device)
            out.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0)


def train_arm(
    p: float,
    split_key: str,
    seed: int,
    device: torch.device,
    *,
    smoke: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    """Formal training of one arm: frozen recipe, raw-y L1, last-5 soup."""
    assert p in (R0_P, R1_P), p
    assert split_key in ("A", "B")
    rows = read_split_rows(split_key)
    fit_rows, mon_rows = rows["fit"], rows["monitor"]
    if smoke:
        fit_rows, mon_rows = fit_rows[:512], mon_rows[:128]
    objs = load_objects()
    gmean, gscale = load_norm(split_key)
    y_fit = torch.as_tensor(load_y_rows(fit_rows, subset_name=f"{split_key} fit"), dtype=torch.float32)
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")
    # batch order: computed once from an independent generator (never from y)
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    epoch_batches = [_shuffle_batches(len(fit_rows), BATCH_SIZE, gen) for _ in range(3 if smoke else EPOCHS)]
    batch_order_sha256 = hashlib.sha256(
        np.ascontiguousarray(
            np.asarray([idx for ep in epoch_batches for batch in ep for idx in batch], dtype=np.int64)
        ).tobytes()
    ).hexdigest()

    seed_everything(seed)
    model = jd.build_model(
        seed=seed, junction_dropout_p=p, global_mean=gmean, global_scale=gscale,
        organization_mode=src.NULL, device=device,
    )
    init_sha = jd.state_dict_hash(model)
    rng_after_init = torch_rng_state_sha256()
    audit = jd.parameter_audit(model)
    if not audit["parameter_exact"]:
        raise RuntimeError(f"parameter audit failed: {audit}")
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    from collections import deque

    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    rng_hashes: dict[str, str] = {}
    started = time.perf_counter()
    epochs = 3 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in epoch_batches[epoch - 1]:
            batch = _mv(sro.batch_objects([objs[int(fit_rows[i])] for i in indices]), device)
            target = y_fit[indices].to(device).view(-1)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            total += float((pred.view(-1) - target).abs().sum())
            n_mol += len(indices)
            steps += 1
        mon_mae = float("nan")
        if mon_rows:
            mon_pred = predict(model, objs, mon_rows, device)
            mon_mae = float(np.abs(mon_pred - y_mon).mean())
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
                "steps": steps,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch in (1, 40, 120, epochs):
            rng_hashes[str(epoch)] = torch_rng_state_sha256()
        if epoch % 20 == 0 or epoch in (1, epochs):
            log_fn(
                f"[{arm_letter(p)}/{split_key}/s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s"
            )
        if not np.isfinite(curve[-1]["train_mae"]) or (mon_rows and not np.isfinite(mon_mae)):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict(model, objs, fit_rows, device)
    mon_pred = predict(model, objs, mon_rows, device) if mon_rows else np.zeros(0)
    results = {
        "protocol": PROTOCOL_ID,
        "arm": arm_letter(p),
        "junction_dropout_p": float(p),
        "split": split_key,
        "split_salt": SPLIT_SALTS[split_key],
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs": epochs,
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "optimizer": "Adam", "supervision": "raw y L1 only", "scheduler": "none",
            "amp": False, "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "soup_members": members,
        "parameters": int(sum(q.numel() for q in model.parameters())),
        "init_state_sha256": init_sha,
        "rng_state_after_init_sha256": rng_after_init,
        "rng_state_sha256_at_epochs": rng_hashes,
        "batch_order_sha256": batch_order_sha256,
        "steps_per_epoch": len(epoch_batches[0]) if epoch_batches else 0,
        "total_steps": int(len(epoch_batches) * (len(epoch_batches[0]) if epoch_batches else 0)),
        "curve": curve,
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - np.asarray(y_fit)).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()) if mon_rows else None,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fit_only_discipline": {
            "global_stats": "split fit rows only",
            "labels_read": [f"{split_key} fit", f"{split_key} monitor"],
            "locked_labels_read": False,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not smoke:
        tag = arm_tag(p, split_key, seed)
        torch.save(soup, OUT / f"checkpoint_{tag}.pt")
        write_json(OUT / f"train_{tag}.json", results)
    log_fn(
        f"[{arm_letter(p)}/{split_key}/s{seed}] soup fit {results['soup_fit_mae']:.6f} "
        f"monitor {results['soup_monitor_mae']} wall {wall:.0f}s"
    )
    del objs
    return results


def cmd_train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    train_arm(float(args.p), args.split, int(args.seed), device, smoke=bool(args.smoke))
    return 0


def cmd_train_stage(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    plan = {
        "split_a": [(R0_P, "A", 0), (R1_P, "A", 0), (R0_P, "A", 1), (R1_P, "A", 1)],
        "split_b": [(R0_P, "B", 0), (R1_P, "B", 0), (R0_P, "B", 1), (R1_P, "B", 1)],
    }[args.stage]
    log(f"[stage {args.stage}] device={device} plan={[(arm_letter(p), s, sd) for p, s, sd in plan]}")
    for p, split_key, seed in plan:
        marker = OUT / f"train_{arm_tag(p, split_key, seed)}.json"
        if marker.exists() and not args.force:
            log(f"[stage] skip {marker.name} (already trained)")
            continue
        train_arm(p, split_key, seed, device)
    return 0


# ---------------------------------------------------------------------------
# label-blind prediction freeze
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    """Label-blind predictions (monitor + locked) for a list of trained arms."""
    device = torch.device(args.device)
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    gmean, gscale = load_norm(split_key)
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "arms": {}, "labels_read": False,
        "rows_mode": args.rows, "locked_predictions_written": args.rows == "both",
        "official_valid_or_test_loaded": False,
    }
    monitor_preds = {"monitor_row": np.asarray(rows["monitor"], dtype=np.int64)}
    locked_preds = {"locked_row": np.asarray(rows["locked"], dtype=np.int64)}
    for tag in args.arms:
        letter = tag.split("_")[0]
        p = R0_P if letter == "R0" else R1_P
        ckpt = OUT / f"checkpoint_{tag}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = jd.build_model(
            seed=int(tag.split("_s")[-1]), junction_dropout_p=p, global_mean=gmean, global_scale=gscale,
            organization_mode=src.NULL, device=device,
        )
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()
        mon = predict(model, objs, rows["monitor"], device)
        monitor_preds[f"pred_{tag}_monitor"] = mon.astype(np.float32)
        entry = {
            "junction_dropout_p": p,
            "checkpoint_sha256": file_sha256(ckpt),
            "monitor_pred_sha256": hashlib.sha256(mon.astype(np.float32).tobytes()).hexdigest(),
            "parameters": int(sum(q.numel() for q in model.parameters())),
        }
        if args.rows == "both":
            locked = predict(model, objs, rows["locked"], device)
            locked_preds[f"pred_{tag}_locked"] = locked.astype(np.float32)
            entry["locked_pred_sha256"] = hashlib.sha256(locked.astype(np.float32).tobytes()).hexdigest()
        manifest["arms"][tag] = entry
        log(f"[predict] {tag} written (rows={args.rows})")
        del model
    if args.rows == "both":
        np.savez_compressed(OUT / f"split_{split_key.lower()}_predictions.npz", **monitor_preds, **locked_preds)
        manifest["predictions_file"] = f"split_{split_key.lower()}_predictions.npz"
        manifest["predictions_file_sha256"] = file_sha256(OUT / f"split_{split_key.lower()}_predictions.npz")
    else:
        np.savez_compressed(OUT / f"split_{split_key.lower()}_monitor_predictions.npz", **monitor_preds)
    write_json(OUT / f"prediction_manifest_{split_key}.json", manifest)
    del objs
    return 0


# ---------------------------------------------------------------------------
# scoring: ONE locked-label read per split
# ---------------------------------------------------------------------------


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - y)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    N = int(delta.size)
    means = np.empty(n, dtype=np.float64)
    chunk = 500
    for i in range(0, n, chunk):
        idx = rng.integers(0, N, size=(min(chunk, n - i), N))
        means[i : i + idx.shape[0]] = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_resamples": int(n),
        "seed": int(seed),
        "frac_candidate_better": float((delta < 0).mean()),
        "frac_candidate_worse": float((delta > 0).mean()),
        "frac_exact_tie": float((delta == 0).mean()),
    }


def _subgroup_masks(objs, rows, y, smiles) -> dict[str, np.ndarray]:
    ss = structural_stats(objs, rows)
    n_atoms = ss["n_atoms"]
    n_junc = ss["n_junctions"]
    cyc = ss["cyclomatic"]
    diam = ss["diameter"]
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "y_gt_-3": y > -3.0,
        "y_le_-3": y <= -3.0,
        "n_atoms_le_20": n_atoms <= 20,
        "n_atoms_gt_20": n_atoms > 20,
        "junctions_le_median": n_junc <= np.median(n_junc),
        "junctions_gt_median": n_junc > np.median(n_junc),
        "cyclomatic_0": cyc <= 0,
        "cyclomatic_ge_1": cyc >= 1,
        "cyclomatic_ge_3": cyc >= 3,
        "diameter_le_median": diam <= np.median(diam),
        "diameter_gt_median": diam > np.median(diam),
    }
    rows_arr = np.asarray(rows, dtype=np.int64)
    # duplicate canonical-SMILES molecule groups
    counts: dict[str, int] = {}
    for r in rows_arr:
        counts[smiles[int(r)]] = counts.get(smiles[int(r)], 0) + 1
    dup = np.asarray([counts[smiles[int(r)]] > 1 for r in rows_arr], dtype=bool)
    masks["duplicate_smiles"] = dup
    masks["unique_smiles"] = ~dup
    return masks


def _subgroup_table(err: np.ndarray, masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    order = np.argsort(-err)
    for name, mask in masks.items():
        out[name] = {
            "n": int(mask.sum()),
            "mae": float(err[mask].mean()) if mask.any() else float("nan"),
            "err_mass_share": float(err[mask].sum() / err.sum()) if mask.any() else float("nan"),
        }
    out["max_error_row"] = {
        "position": int(order[0]),
        "abs_err": float(err[order[0]]),
        "share": float(err[order[0]] / err.sum()),
    }
    trimmed = np.ones_like(err, dtype=bool)
    trimmed[order[:10]] = False
    out["top10_removed"] = {
        "n": int(trimmed.sum()),
        "mae": float(err[trimmed].mean()),
        "top10_err_mass_share": float(err[order[:10]].sum() / err.sum()),
    }
    return out


def cmd_score(args: argparse.Namespace) -> int:
    """The ONE locked-label read for this split (after predictions are frozen)."""
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    smiles = load_smiles()
    pred_path = OUT / f"split_{split_key.lower()}_predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    data = np.load(pred_path)
    assert (data["monitor_row"] == np.asarray(rows["monitor"])).all()
    assert (data["locked_row"] == np.asarray(rows["locked"])).all()
    tags = [k[len("pred_") : -len("_monitor")] for k in data.files if k.startswith("pred_") and k.endswith("_monitor")]
    tags = sorted(tags)
    # label boundary: extract ONLY this split's monitor + locked labels
    # (the file will be loaded once by load_y_rows, but no other split's label
    #  vector is materialised here; split B labels are untouched by A scoring)
    mon_rows = list(rows["monitor"])
    locked_rows = list(rows["locked"])
    y_need = load_y_rows(mon_rows + locked_rows, subset_name=f"{split_key} monitor+locked (score)")
    y_mon = y_need[: len(mon_rows)]
    y_locked = y_need[len(mon_rows) :]
    masks_mon = _subgroup_masks(objs, rows["monitor"], y_mon, smiles)
    masks_locked = _subgroup_masks(objs, rows["locked"], y_locked, smiles)

    metrics_payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "label_read": "single score invocation",
        "official_valid_or_test_loaded": False, "arms": {}, "subgroups_locked": {}, "subgroups_monitor": {},
    }
    boot_payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "baseline": "R0", "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "pairs": {}, "cross_seed": {}, "subgroup_deltas_locked": {}, "subgroup_deltas_monitor": {},
    }
    arm_preds: dict[str, dict[str, np.ndarray]] = {}
    for tag in tags:
        train = read_json(OUT / f"train_{tag}.json")
        pm = np.asarray(data[f"pred_{tag}_monitor"], dtype=np.float64)
        pl = np.asarray(data[f"pred_{tag}_locked"], dtype=np.float64)
        arm_preds[tag] = {"monitor": pm, "locked": pl}
        metrics_payload["arms"][tag] = {
            "junction_dropout_p": train["junction_dropout_p"],
            "seed": train["seed"],
            "fit_mae": train["soup_fit_mae"],
            "monitor": metrics(pm, y_mon),
            "locked": metrics(pl, y_locked),
            "parameters": train["parameters"],
            "wall_seconds": train["wall_seconds"],
            "init_state_sha256": train["init_state_sha256"],
            "batch_order_sha256": train["batch_order_sha256"],
        }
        metrics_payload["subgroups_monitor"][tag] = _subgroup_table(np.abs(pm - y_mon), masks_mon)
        metrics_payload["subgroups_locked"][tag] = _subgroup_table(np.abs(pl - y_locked), masks_locked)

    pairs: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        r0, r1 = f"R0_{split_key}_s{seed}", f"R1_{split_key}_s{seed}"
        d_mon = np.abs(arm_preds[r1]["monitor"] - y_mon) - np.abs(arm_preds[r0]["monitor"] - y_mon)
        d_locked = np.abs(arm_preds[r1]["locked"] - y_locked) - np.abs(arm_preds[r0]["locked"] - y_locked)
        pairs[f"s{seed}"] = {
            "monitor": paired_bootstrap(d_mon),
            "locked": paired_bootstrap(d_locked),
            "monitor_mae_delta": float(d_mon.mean()),
            "locked_mae_delta": float(d_locked.mean()),
        }
    d_mon_mean = 0.5 * (
        np.abs(arm_preds[f"R1_{split_key}_s0"]["monitor"] - y_mon)
        - np.abs(arm_preds[f"R0_{split_key}_s0"]["monitor"] - y_mon)
        + np.abs(arm_preds[f"R1_{split_key}_s1"]["monitor"] - y_mon)
        - np.abs(arm_preds[f"R0_{split_key}_s1"]["monitor"] - y_mon)
    )
    d_locked_mean = 0.5 * (
        np.abs(arm_preds[f"R1_{split_key}_s0"]["locked"] - y_locked)
        - np.abs(arm_preds[f"R0_{split_key}_s0"]["locked"] - y_locked)
        + np.abs(arm_preds[f"R1_{split_key}_s1"]["locked"] - y_locked)
        - np.abs(arm_preds[f"R0_{split_key}_s1"]["locked"] - y_locked)
    )
    pairs["seed_mean"] = {
        "monitor": paired_bootstrap(d_mon_mean),
        "locked": paired_bootstrap(d_locked_mean),
        "monitor_mae_delta": float(d_mon_mean.mean()),
        "locked_mae_delta": float(d_locked_mean.mean()),
        "definition": "d_i = 0.5 * (d_i_seed0 + d_i_seed1), molecule bootstrap",
    }
    boot_payload["pairs"] = pairs
    r0_locked = [float(np.abs(arm_preds[f"R0_{split_key}_s{s}"]["locked"] - y_locked).mean()) for s in SEEDS]
    r1_locked = [float(np.abs(arm_preds[f"R1_{split_key}_s{s}"]["locked"] - y_locked).mean()) for s in SEEDS]
    boot_payload["cross_seed"] = {
        "R0_locked_mae_per_seed": r0_locked,
        "R1_locked_mae_per_seed": r1_locked,
        "R0_seed_spread": float(max(r0_locked) - min(r0_locked)),
        "R1_seed_spread": float(max(r1_locked) - min(r1_locked)),
        "delta_per_seed": [pairs[f"s{s}"]["locked_mae_delta"] for s in SEEDS],
        "delta_seed_difference": float(pairs["s0"]["locked_mae_delta"] - pairs["s1"]["locked_mae_delta"]),
        "rmse_per_seed": {
            f"R{arm}_{s}": metrics(arm_preds[f"R{arm}_{split_key}_s{s}"]["locked"], y_locked)["rmse"]
            for arm in (0, 1) for s in SEEDS
        },
        "warning": (
            "molecule bootstrap intervals describe within-split sampling; they do not cover "
            "training randomness. The seed spread above is reported separately for that reason."
        ),
    }
    # subgroup deltas: seed-mean R1-R0 within each pre-frozen subgroup
    for name in masks_locked:
        m = masks_locked[name]
        d = d_locked_mean[m]
        boot_payload["subgroup_deltas_locked"][name] = {
            "n": int(m.sum()),
            "delta_mean": float(d.mean()) if m.any() else None,
            "frac_better": float((d < 0).mean()) if m.any() else None,
        }
    for name in masks_mon:
        m = masks_mon[name]
        d = d_mon_mean[m]
        boot_payload["subgroup_deltas_monitor"][name] = {
            "n": int(m.sum()),
            "delta_mean": float(d.mean()) if m.any() else None,
            "frac_better": float((d < 0).mean()) if m.any() else None,
        }
    # ordinary-molecule + top10-trimmed seed-mean deltas (gate inputs)
    ordinary_locked = masks_locked["y_gt_-3"]
    err_r0 = np.abs(arm_preds[f"R0_{split_key}_s0"]["locked"] - y_locked)
    err_r1 = np.abs(arm_preds[f"R1_{split_key}_s0"]["locked"] - y_locked)
    boot_payload["gate_inputs_locked"] = {
        "ordinary_seed_mean_delta": float(d_locked_mean[ordinary_locked].mean()),
        "ordinary_n": int(ordinary_locked.sum()),
        "top10_removed_seed_mean_delta": float(
            d_locked_mean[np.argsort(-(err_r0 + err_r1))[10:]].mean()
        ),
        "worst_row_share_R0": float(err_r0.max() / err_r0.sum()),
        "worst_row_share_R1": float(err_r1.max() / err_r1.sum()),
    }
    metrics_payload["seed_means"] = {}
    for letter in ("R0", "R1"):
        metrics_payload["seed_means"][letter] = {
            "monitor": metrics(
                0.5 * (arm_preds[f"{letter}_{split_key}_s0"]["monitor"] + arm_preds[f"{letter}_{split_key}_s1"]["monitor"]),
                y_mon,
            ),
            "locked": metrics(
                0.5 * (arm_preds[f"{letter}_{split_key}_s0"]["locked"] + arm_preds[f"{letter}_{split_key}_s1"]["locked"]),
                y_locked,
            ),
        }
    r0_mean_locked = metrics_payload["seed_means"]["R0"]["locked"]["mae"]
    r1_mean_locked = metrics_payload["seed_means"]["R1"]["locked"]["mae"]
    write_json(OUT / f"split_{split_key.lower()}_metrics.json", metrics_payload)
    write_json(OUT / f"split_{split_key.lower()}_paired_bootstrap.json", boot_payload)
    write_json(
        OUT / f"split_{split_key.lower()}_label_access_audit.json",
        {
            "protocol": PROTOCOL_ID, "split": split_key,
            "label_reads": {"monitor": "training stage (by design)", "locked": "single score invocation"},
            "score_invocations": 1,
            "locked_labels_read_before_predictions_hashed": False,
            "official_valid_or_test_loaded": False,
        },
    )
    log(
        f"[score {split_key}] seed-mean locked MAE R0 {r0_mean_locked:.6f} "
        f"R1 {r1_mean_locked:.6f} | delta {pairs['seed_mean']['locked_mae_delta']:+.6f}"
    )
    del objs
    return 0


def cmd_score_preview(args: argparse.Namespace) -> int:
    """Rehearsal-only scoring on the (readable) monitor rows.

    Reads ``split_{key}_monitor_predictions.npz`` and never touches the locked
    rows of either split; the real one-shot score is :func:`cmd_score`.
    """
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    smiles = load_smiles()
    path = OUT / f"split_{split_key.lower()}_monitor_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    assert (data["monitor_row"] == np.asarray(rows["monitor"])).all()
    tags = sorted(k[len("pred_") : -len("_monitor")] for k in data.files
                  if k.startswith("pred_") and k.endswith("_monitor"))
    y_mon = load_y_rows(rows["monitor"], subset_name=f"{split_key} monitor (preview)")
    masks_mon = _subgroup_masks(objs, rows["monitor"], y_mon, smiles)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "preview": True,
        "locked_labels_read": False, "official_valid_or_test_loaded": False, "arms": {}, "pairs": {},
    }
    preds = {tag: np.asarray(data[f"pred_{tag}_monitor"], dtype=np.float64) for tag in tags}
    for tag in tags:
        payload["arms"][tag] = metrics(preds[tag], y_mon)
        payload.setdefault("subgroups", {})[tag] = _subgroup_table(np.abs(preds[tag] - y_mon), masks_mon)
    for seed in SEEDS:
        r0, r1 = f"R0_{split_key}_s{seed}", f"R1_{split_key}_s{seed}"
        if r0 not in preds or r1 not in preds:
            continue
        d = np.abs(preds[r1] - y_mon) - np.abs(preds[r0] - y_mon)
        payload["pairs"][f"s{seed}"] = {"monitor_delta": float(d.mean()), "bootstrap": paired_bootstrap(d)}
    write_json(OUT / f"split_{split_key.lower()}_monitor_preview.json", payload)
    log(f"[score-preview {split_key}] monitor-only rehearsal written")
    del objs
    return 0


# ---------------------------------------------------------------------------
# pre-registered promotion gate
# ---------------------------------------------------------------------------


def cmd_decide(args: argparse.Namespace) -> int:
    split_key = args.split
    metrics_payload = read_json(OUT / f"split_{split_key.lower()}_metrics.json")
    boot = read_json(OUT / f"split_{split_key.lower()}_paired_bootstrap.json")
    gates = read_json(OUT / "dropout_correctness_gates.json")
    pairs = boot["pairs"]
    mon_mean = pairs["seed_mean"]["monitor_mae_delta"]
    locked_mean = pairs["seed_mean"]["locked_mae_delta"]
    checks = {
        "monitor_seed_mean_delta_le_-0.006": bool(mon_mean <= GATE_MONITOR_SEED_MEAN_DELTA_MAX),
        "both_seed_monitor_delta_negative": bool(
            pairs["s0"]["monitor_mae_delta"] < 0 and pairs["s1"]["monitor_mae_delta"] < 0
        ),
        "both_seed_locked_delta_non_positive": bool(
            pairs["s0"]["locked_mae_delta"] <= 0 and pairs["s1"]["locked_mae_delta"] <= 0
        ),
        "locked_seed_mean_delta_le_-0.003": bool(locked_mean <= GATE_LOCKED_SEED_MEAN_DELTA_MAX),
        "ordinary_molecules_not_worse": bool(
            boot["gate_inputs_locked"]["ordinary_seed_mean_delta"] <= 0.0
        ),
        "not_driven_by_single_extreme_row": bool(
            boot["gate_inputs_locked"]["top10_removed_seed_mean_delta"] < 0.0
            and boot["gate_inputs_locked"]["worst_row_share_R0"] < GATE_WORST_ROW_SHARE_MAX
            and boot["gate_inputs_locked"]["worst_row_share_R1"] < GATE_WORST_ROW_SHARE_MAX
        ),
        "dropout_and_correctness_gates_pass": bool(gates["all_numeric_gates_pass"]),
    }
    purchase = all(checks.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "split": split_key,
        "candidate": "R1 = BJOC-Null + junction dropout p=0.10 vs R0 = p=0.0",
        "checks": checks,
        "deltas": {
            "fit_seed_mean_R1_minus_R0": float(
                np.mean([metrics_payload["arms"][f"R1_{split_key}_s{s}"]["fit_mae"] for s in SEEDS])
                - np.mean([metrics_payload["arms"][f"R0_{split_key}_s{s}"]["fit_mae"] for s in SEEDS])
            ),
            "monitor_seed_mean": mon_mean,
            "monitor_seed_mean_ci95": pairs["seed_mean"]["monitor"]["ci95"],
            "locked_seed_mean": locked_mean,
            "locked_seed_mean_ci95": pairs["seed_mean"]["locked"]["ci95"],
            "locked_per_seed": [pairs[f"s{s}"]["locked_mae_delta"] for s in SEEDS],
            "locked_per_seed_ci95": [pairs[f"s{s}"]["locked"]["ci95"] for s in SEEDS],
        },
        "decision": "JDROP_REPLICATION_PURCHASE" if purchase else "JDROP_NO_REPLICATION_PURCHASE",
        "uncertainty_note": (
            "these thresholds are pre-frozen GPU-purchase rules, not significance definitions; "
            "if a point estimate improves while the bootstrap interval crosses zero, the result "
            "is recorded as directional with its uncertainty"
        ),
        "official_valid_or_test_loaded": False,
    }
    write_json(OUT / f"split_{split_key.lower()}_promotion_gate.json", payload)
    log(f"[decide {split_key}] {payload['decision']}")
    return 0


# ---------------------------------------------------------------------------
# Phase G — mechanism diagnostics on frozen checkpoints (CPU only)
# ---------------------------------------------------------------------------


def _junction_embedding_stats(model, objs, rows, device="cpu") -> dict[str, Any]:
    model.eval()
    ws, Ws, Vs = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            chunk = [objs[int(r)] for r in rows[i : i + 128]]
            batch = _mv(sro.batch_objects(chunk), device)
            feats = model.forward_features(batch)
            ws.append(feats["w"].cpu().numpy())
            Ws.append(feats["W"].cpu().numpy())
            Vs.append(feats["V"].cpu().numpy())
    w = np.concatenate(ws, 0)
    W = np.concatenate(Ws, 0)
    V = np.concatenate(Vs, 0)
    return {
        "w": w,
        "W": W,
        "V": V,
        "w_abs_mean": float(np.abs(w).mean()),
        "w_abs_p50": float(np.median(np.abs(w))),
        "w_abs_p95": float(np.percentile(np.abs(w), 95)),
        "w_channel_mean_abs_mean": float(np.abs(w.mean(0)).mean()),
        "w_channel_std_mean": float(w.std(0).mean()),
        "w_channel_skew_mean": float((((w - w.mean(0)) ** 3).mean(0) / np.maximum(w.std(0) ** 3, 1e-12)).mean()),
        "w_frac_near_zero": float((np.abs(w) < 1e-3).mean()),
        "graph_mean_w_abs_mean": float(np.abs(W).mean()),
        "graph_mean_w2_abs_mean": float(np.abs(V).mean()),
    }


def _prediction_with_intervention(model, objs, rows, intervention: str, device="cpu") -> np.ndarray:
    """Eval-mode counterfactual on the frozen junction embedding.

    ``zero``                          -> set w to zero (both moments vanish)
    ``within_molecule_permutation``   -> permute junction rows inside each molecule
                                         (exactly the multiset; the null grouping
                                          should be invariant -> a conservation check)
    ``cross_molecule_shuffle``        -> globally shuffle junction rows across molecules
                                         (destroys molecule-specific junction content)
    """
    model.eval()
    rng = np.random.default_rng(20261012)
    out = []
    with torch.no_grad():
        def hook_junc(_module, _inputs, output):
            if intervention == "zero":
                return torch.zeros_like(output)
            return output

        def hook_group(w, batch, mode):
            if intervention == "within_molecule_permutation":
                jb = batch["junc_batch"].cpu().numpy()
                perm = np.arange(w.shape[0])
                for g in np.unique(jb):
                    idx = np.where(jb == g)[0]
                    perm[idx] = idx[rng.permutation(len(idx))]
                w_mod = w[torch.as_tensor(perm, dtype=torch.long, device=w.device)]
                return src.BJOC.group_to_bonds(model, w_mod, batch, mode)
            if intervention == "cross_molecule_shuffle":
                perm = rng.permutation(int(w.shape[0]))
                w_mod = w[torch.as_tensor(perm, dtype=torch.long, device=w.device)]
                return src.BJOC.group_to_bonds(model, w_mod, batch, mode)
            return src.BJOC.group_to_bonds(model, w, batch, mode)

        original_group = type(model).group_to_bonds
        handle = model.junction_encoder.register_forward_hook(hook_junc)
        type(model).group_to_bonds = lambda self, w, b, m: hook_group(w, b, m)
        try:
            for i in range(0, len(rows), 256):
                chunk = [objs[int(r)] for r in rows[i : i + 256]]
                batch = _mv(sro.batch_objects(chunk), device)
                out.append(model(batch).detach().cpu().numpy().reshape(-1))
        finally:
            type(model).group_to_bonds = original_group
            handle.remove()
    return np.concatenate(out, 0)


def _junction_grad_norm(model, objs, rows, device="cpu") -> float:
    model.train()
    batch = _mv(sro.batch_objects([objs[int(r)] for r in rows[:256]]), device)
    model.zero_grad(set_to_none=True)
    _, feats = model(batch, return_features=True)
    F.l1_loss(feats["w_used"].sum(dim=1), torch.zeros(len(rows[:256]))).backward()
    return float(feats["w"].grad.pow(2).sum().sqrt()) if feats["w"].grad is not None else 0.0


def cmd_mechanism(args: argparse.Namespace) -> int:
    """Phase G: does R1 use the junction channel differently from R0?"""
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    gmean, gscale = load_norm(split_key)
    mon_rows = rows["monitor"]
    y_mon = load_y_rows(mon_rows, subset_name="mechanism monitor")
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key,
        "interventions": "eval-mode counterfactuals on frozen models; model-dependence not chemistry causality",
        "arms": {}, "official_valid_or_test_loaded": False,
    }
    for letter, p in (("R0", R0_P), ("R1", R1_P)):
        for seed in SEEDS:
            tag = arm_tag(p, split_key, seed)
            model = jd.build_model(seed=seed, junction_dropout_p=p, global_mean=gmean, global_scale=gscale)
            model.load_state_dict(torch.load(OUT / f"checkpoint_{tag}.pt", map_location="cpu", weights_only=False))
            fit_sample = rows["fit"][:2000]
            emb_fit = _junction_embedding_stats(model, objs, fit_sample)
            emb_mon = _junction_embedding_stats(model, objs, mon_rows)
            base_mon = _prediction_with_intervention(model, objs, mon_rows, "identity")
            zero_mon = _prediction_with_intervention(model, objs, mon_rows, "zero")
            within_mon = _prediction_with_intervention(model, objs, mon_rows, "within_molecule_permutation")
            shuffle_mon = _prediction_with_intervention(model, objs, mon_rows, "cross_molecule_shuffle")
            entry = {
                "junction_dropout_p": p,
                "fit_embedding": {k: v for k, v in emb_fit.items() if k not in ("w", "W", "V")},
                "monitor_embedding": {k: v for k, v in emb_mon.items() if k not in ("w", "W", "V")},
                "train_monitor_embedding_shift": {
                    "mean_abs_channel_mean_gap": float(np.abs(emb_fit["w"].mean(0) - emb_mon["w"].mean(0)).mean()),
                    "mean_abs_channel_std_gap": float(np.abs(emb_fit["w"].std(0) - emb_mon["w"].std(0)).mean()),
                },
                "monitor_base_mae": float(np.abs(base_mon - y_mon).mean()),
                "junction_zero": {
                    "mean_abs_dpred": float(np.abs(zero_mon - base_mon).mean()),
                    "mae_after": float(np.abs(zero_mon - y_mon).mean()),
                },
                "junction_within_molecule_permutation": {
                    "mean_abs_dpred": float(np.abs(within_mon - base_mon).mean()),
                    "mae_after": float(np.abs(within_mon - y_mon).mean()),
                    "expected": "~0 for the null grouping (multiset preserved)",
                },
                "junction_cross_molecule_shuffle": {
                    "mean_abs_dpred": float(np.abs(shuffle_mon - base_mon).mean()),
                    "mae_after": float(np.abs(shuffle_mon - y_mon).mean()),
                },
                "junction_grad_norm_train_mode": _junction_grad_norm(model, objs, rows["fit"]),
            }
            # null-grouping conservation audit (eval mode)
            batch = sro.batch_objects([objs[int(r)] for r in mon_rows[:64]])
            with torch.no_grad():
                feats = model.forward_features(batch)
                jb = batch["junc_batch"]
                bb = batch["bond_batch"]
                rel_w, rel_v = [], []
                for g in range(int(batch["n_graphs"])):
                    w_g = feats["w"][jb == g].sum(0)
                    W_g = feats["W"][bb == g].sum(0)
                    V_g = feats["V"][bb == g].sum(0)
                    # null: sum_b W_b = 2 * sum_j w_j and sum_b V_b = 2 * sum_j w_j^2
                    rel_w.append(float((W_g - 2.0 * w_g).abs().sum() / (2.0 * w_g.abs().sum() + 1e-12)))
                    rel_v.append(
                        float((V_g - 2.0 * (feats["w"][jb == g] ** 2).sum(0)).abs().sum()
                              / (2.0 * (feats["w"][jb == g] ** 2).sum(0).abs().sum() + 1e-12))
                    )
            entry["null_grouping_conservation_max_relative_error"] = max(max(rel_w), max(rel_v))
            entry["parameters"] = int(sum(q.numel() for q in model.parameters()))
            payload["arms"][tag] = entry
            del model
    # R1-vs-R0 summary (seed-mean)
    try:
        summary = {}
        for name in ("junction_zero", "junction_cross_molecule_shuffle"):
            v0 = np.mean([payload["arms"][arm_tag(R0_P, split_key, s)][name]["mae_after"] for s in SEEDS])
            v1 = np.mean([payload["arms"][arm_tag(R1_P, split_key, s)][name]["mae_after"] for s in SEEDS])
            summary[f"{name}_mae_after"] = {"R0": float(v0), "R1": float(v1), "delta": float(v1 - v0)}
        summary["w_abs_mean_fit"] = {
            "R0": float(np.mean([payload["arms"][arm_tag(R0_P, split_key, s)]["fit_embedding"]["w_abs_mean"] for s in SEEDS])),
            "R1": float(np.mean([payload["arms"][arm_tag(R1_P, split_key, s)]["fit_embedding"]["w_abs_mean"] for s in SEEDS])),
        }
        summary["w_frac_near_zero_fit"] = {
            "R0": float(np.mean([payload["arms"][arm_tag(R0_P, split_key, s)]["fit_embedding"]["w_frac_near_zero"] for s in SEEDS])),
            "R1": float(np.mean([payload["arms"][arm_tag(R1_P, split_key, s)]["fit_embedding"]["w_frac_near_zero"] for s in SEEDS])),
        }
        payload["R1_vs_R0_summary"] = summary
    except KeyError:
        pass
    write_json(OUT / "junction_regularization_mechanism.json", payload)
    log("[mechanism] junction_regularization_mechanism.json written")
    del objs
    return 0


# ---------------------------------------------------------------------------
# final decision + report
# ---------------------------------------------------------------------------


def _final_decision() -> dict[str, Any]:
    gates = read_json(OUT / "dropout_correctness_gates.json")
    a_gate = read_json(OUT / "split_a_promotion_gate.json")
    a_boot = read_json(OUT / "split_a_paired_bootstrap.json")
    b_gate_path = OUT / "split_b_promotion_gate.json"
    b_boot_path = OUT / "split_b_paired_bootstrap.json"
    b_gate = read_json(b_gate_path) if b_gate_path.exists() else None
    b_boot = read_json(b_boot_path) if b_boot_path.exists() else None
    pairs = a_boot["pairs"]
    mon_mean = pairs["seed_mean"]["monitor_mae_delta"]
    locked_mean = pairs["seed_mean"]["locked_mae_delta"]
    locked_ci = pairs["seed_mean"]["locked"]["ci95"]
    mon_ci = pairs["seed_mean"]["monitor"]["ci95"]
    a_purchase = a_gate["decision"] == "JDROP_REPLICATION_PURCHASE"
    b_purchase = bool(b_gate and b_gate["decision"] == "JDROP_REPLICATION_PURCHASE")
    # frozen mapping (pre-registered before scoring)
    material_hurt = max(mon_mean, locked_mean)
    flat = abs(mon_mean) < 0.003 and abs(locked_mean) < 0.003
    if not gates["all_numeric_gates_pass"]:
        label = "INCONCLUSIVE_OR_BLOCKED"
        rationale = "dropout/correctness gates failed; no arm contrast is interpretable"
    elif a_purchase and b_purchase:
        label = "JUNCTION_DROPOUT_PROMISING"
        rationale = (
            "R1 (p=0.10) cleared the pre-registered monitor + locked-dev purchase gate on both "
            "frozen internal splits and both seeds, with ordinary molecules not worse and no "
            "single-extreme-row dependence"
        )
    elif a_purchase and not b_purchase:
        label = "JUNCTION_DROPOUT_LIMITED_SIGNAL"
        rationale = (
            "R1 cleared the split-A purchase gate but did not reproduce it on split B; the effect "
            "is split-sensitive, so the evidence supports a bounded signal, not a general improvement"
        )
    elif material_hurt >= 0.003 and min(mon_mean, locked_mean) >= 0.0:
        label = "JUNCTION_DROPOUT_HURTS_GENERALIZATION"
        rationale = (
            "R1 was materially no better than R0 on at least one held-out readout and not "
            "favourable on either; the frozen regularisation did not buy generalisation in this regime"
        )
    elif flat:
        label = "JUNCTION_DROPOUT_NO_USEFUL_GAIN"
        rationale = (
            "the seed-mean deltas lie inside the pre-declared indifference band on both readouts; "
            "the intervention changed nothing useful"
        )
    else:
        label = "JUNCTION_DROPOUT_LIMITED_SIGNAL"
        rationale = (
            "the point estimates are directionally favourable (or mixed) but did not clear the "
            "pre-registered purchase gate and/or the intervals cross zero; recorded as limited / "
            "uncertain rather than as a success"
        )
        if b_boot is not None:
            b_locked = b_boot["pairs"]["seed_mean"]["locked_mae_delta"]
            if a_purchase is False and b_locked < 0:
                rationale += (
                    " (split B was not purchased, but its held-out point estimate is also negative; "
                    "the failure is the frozen purchase bar, not a sign reversal)"
                )
    mechanism = read_json(OUT / "junction_regularization_mechanism.json") if (OUT / "junction_regularization_mechanism.json").exists() else {}
    seed_spread = a_boot["cross_seed"]
    payload = {
        "protocol": PROTOCOL_ID,
        "decision": label,
        "rationale": rationale,
        "evidence": {
            "split_a": {
                "monitor_seed_mean_delta": mon_mean,
                "monitor_seed_mean_ci95": mon_ci,
                "locked_seed_mean_delta": locked_mean,
                "locked_seed_mean_ci95": locked_ci,
                "locked_per_seed": [pairs[f"s{s}"]["locked_mae_delta"] for s in SEEDS],
                "ordinary_locked_seed_mean_delta": a_boot["gate_inputs_locked"]["ordinary_seed_mean_delta"],
                "top10_removed_locked_seed_mean_delta": a_boot["gate_inputs_locked"]["top10_removed_seed_mean_delta"],
                "R0_seed_spread_locked": seed_spread["R0_seed_spread"],
                "R1_seed_spread_locked": seed_spread["R1_seed_spread"],
                "promotion": a_gate["decision"],
                "checks": a_gate["checks"],
            },
            "split_b": None if b_boot is None else {
                "monitor_seed_mean_delta": b_boot["pairs"]["seed_mean"]["monitor_mae_delta"],
                "locked_seed_mean_delta": b_boot["pairs"]["seed_mean"]["locked_mae_delta"],
                "locked_seed_mean_ci95": b_boot["pairs"]["seed_mean"]["locked"]["ci95"],
                "promotion": b_gate["decision"],
                "checks": b_gate["checks"],
            },
            "correctness_gates_pass": bool(gates["all_numeric_gates_pass"]),
            "mechanism_summary": mechanism.get("R1_vs_R0_summary"),
        },
        "decision_vocabulary": [
            "JUNCTION_DROPOUT_PROMISING", "JUNCTION_DROPOUT_LIMITED_SIGNAL",
            "JUNCTION_DROPOUT_NO_USEFUL_GAIN", "JUNCTION_DROPOUT_HURTS_GENERALIZATION",
            "INCONCLUSIVE_OR_BLOCKED",
        ],
        "allowed_claims": [
            "the frozen junction-dropout intervention was or was not worth keeping under the "
            "pre-registered internal-split purchase rule",
            "the direction and uncertainty of the R1-R0 contrast on two frozen internal splits",
        ],
        "forbidden_claims": [
            "official-test or official-valid confirmation (never read this round)",
            "that all regularisation methods fail or succeed",
            "a chemical causal interpretation of any mechanism diagnostic",
        ],
        "official_valid_or_test_loaded": False,
    }
    return payload


def cmd_decision(args: argparse.Namespace) -> int:
    payload = _final_decision()
    write_json(OUT / "scientific_decision.json", payload)
    log(f"[decision] {payload['decision']}")
    return 0


def build_train_manifest() -> dict[str, Any]:
    records = {}
    for path in sorted(OUT.glob("train_R*.json")):
        rec = read_json(path)
        tag = path.stem.replace("train_", "")
        records[tag] = {
            "arm": rec["arm"], "junction_dropout_p": rec["junction_dropout_p"], "split": rec["split"],
            "seed": rec["seed"], "epochs": rec["epochs"], "soup_members": rec["soup_members"],
            "parameters": rec["parameters"], "init_state_sha256": rec["init_state_sha256"],
            "rng_state_after_init_sha256": rec["rng_state_after_init_sha256"],
            "rng_state_sha256_at_epochs": rec["rng_state_sha256_at_epochs"],
            "batch_order_sha256": rec["batch_order_sha256"], "total_steps": rec["total_steps"],
            "git_commit": rec["git_commit"], "soup_fit_mae": rec["soup_fit_mae"],
            "soup_monitor_mae": rec["soup_monitor_mae"], "wall_seconds": rec["wall_seconds"],
            "device": rec["device"], "cuda_device_name": rec["cuda_device_name"], "torch": rec["torch"],
            "hyper": rec["hyper"],
            "checkpoint_sha256": file_sha256(OUT / f"checkpoint_{tag}.pt")
            if (OUT / f"checkpoint_{tag}.pt").exists() else None,
        }
    # paired arm audit: same seed -> same init and same batch order
    pair_checks = {}
    for split_key in ("A", "B"):
        for seed in SEEDS:
            r0, r1 = f"R0_{split_key}_s{seed}", f"R1_{split_key}_s{seed}"
            if r0 in records and r1 in records:
                pair_checks[f"{split_key}_s{seed}"] = {
                    "init_state_equal": records[r0]["init_state_sha256"] == records[r1]["init_state_sha256"],
                    "batch_order_equal": records[r0]["batch_order_sha256"] == records[r1]["batch_order_sha256"],
                    "total_steps_equal": records[r0]["total_steps"] == records[r1]["total_steps"],
                    "rng_after_init_equal": records[r0]["rng_state_after_init_sha256"]
                    == records[r1]["rng_state_after_init_sha256"],
                }
    return {
        "protocol": PROTOCOL_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "recipe": {
            "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "soup_last": SOUP_LAST, "precision": "fp32", "amp": False,
            "scheduler": "none", "loss": "L1(raw y)",
        },
        "n_trajectories": len(records),
        "budget_max_trajectories": BUDGET["formal_trajectories_max"],
        "total_wall_seconds": float(sum(r["wall_seconds"] for r in records.values())),
        "pair_checks_same_seed": pair_checks,
        "records": records,
        "official_valid_or_test_used": False,
    }


def _fmt_delta(x: float, ci=None) -> str:
    if x is None:
        return "n/a"
    if ci is None:
        return f"{x:+.6f}"
    lo, hi = ci
    return f"{x:+.6f} [{lo:+.6f}, {hi:+.6f}]"


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble REPORT.md with the three required tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "train_manifest.json", build_train_manifest())
    metrics = json.loads((OUT / "split_a_metrics.json").read_text(encoding="utf-8")) if (OUT / "split_a_metrics.json").exists() else None
    boot = json.loads((OUT / "split_a_paired_bootstrap.json").read_text(encoding="utf-8")) if (OUT / "split_a_paired_bootstrap.json").exists() else None
    b_metrics = json.loads((OUT / "split_b_metrics.json").read_text(encoding="utf-8")) if (OUT / "split_b_metrics.json").exists() else None
    b_boot = json.loads((OUT / "split_b_paired_bootstrap.json").read_text(encoding="utf-8")) if (OUT / "split_b_paired_bootstrap.json").exists() else None
    decision = json.loads((OUT / "scientific_decision.json").read_text(encoding="utf-8")) if (OUT / "scientific_decision.json").exists() else {"decision": "PENDING"}
    mechanism = json.loads((OUT / "junction_regularization_mechanism.json").read_text(encoding="utf-8")) if (OUT / "junction_regularization_mechanism.json").exists() else None
    err_audit = json.loads((OUT / "historical_error_audit.json").read_text(encoding="utf-8")) if (OUT / "historical_error_audit.json").exists() else None
    train_audit = json.loads((OUT / "historical_training_audit.json").read_text(encoding="utf-8")) if (OUT / "historical_training_audit.json").exists() else None
    a_gate = json.loads((OUT / "split_a_promotion_gate.json").read_text(encoding="utf-8")) if (OUT / "split_a_promotion_gate.json").exists() else None
    b_gate = json.loads((OUT / "split_b_promotion_gate.json").read_text(encoding="utf-8")) if (OUT / "split_b_promotion_gate.json").exists() else None

    lines: list[str] = []
    lines.append("# cscl-bjoc-junction-regularization-v1 — REPORT")
    lines.append("")
    lines.append(
        "One predictive intervention, frozen before training: elementwise inverted dropout on the "
        "learned junction embedding before the null incidence grouping (R0 `p=0.0`, R1 `p=0.10`), "
        "everything else identical.  Official valid and official test were never loaded."
    )
    lines.append("")
    lines.append("## 0. Decision")
    lines.append("")
    lines.append(f"**`{decision['decision']}`**")
    lines.append("")
    lines.append(decision.get("rationale", ""))
    lines.append("")
    lines.append("## 1. Table 1 — training and generalisation")
    lines.append("")
    lines.append("| Split | Seed | R0 fit | R1 fit | R0 heldout (locked) | R1 heldout (locked) | R0 monitor | R1 monitor |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split_key, m in (("A", metrics), ("B", b_metrics)):
        if m is None:
            continue
        for seed in SEEDS:
            r0, r1 = m["arms"][f"R0_{split_key}_s{seed}"], m["arms"][f"R1_{split_key}_s{seed}"]
            lines.append(
                f"| {split_key} | {seed} | {r0['fit_mae']:.6f} | {r1['fit_mae']:.6f} | "
                f"{r0['locked']['mae']:.6f} | {r1['locked']['mae']:.6f} | "
                f"{r0['monitor']['mae']:.6f} | {r1['monitor']['mae']:.6f} |"
            )
    lines.append("")
    lines.append("## 2. Table 2 — paired improvement (Δ = R1 − R0; negative favours R1)")
    lines.append("")
    lines.append("| Split | Seed | Monitor Δ [CI95] | Locked Δ [CI95] | Ordinary locked Δ | Low-y tail Δ | Better-molecule frac | Max-row removal Δ |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for split_key, b in (("A", boot), ("B", b_boot)):
        if b is None:
            continue
        for key, label in (("s0", "0"), ("s1", "1"), ("seed_mean", "mean")):
            p = b["pairs"][key]
            sg = b.get("subgroup_deltas_locked", {})
            ordinary = sg.get("y_gt_-3", {}).get("delta_mean")
            tail = sg.get("y_le_-3", {}).get("delta_mean")
            top10 = (b.get("gate_inputs_locked", {}).get("top10_removed_seed_mean_delta")
                     if key == "seed_mean" else None)
            ci_m = p["monitor"]["ci95"]
            ci_l = p["locked"]["ci95"]
            lines.append(
                f"| {split_key} | {label} | {p['monitor_mae_delta']:+.6f} [{ci_m[0]:+.6f}, {ci_m[1]:+.6f}] | "
                f"{p['locked_mae_delta']:+.6f} [{ci_l[0]:+.6f}, {ci_l[1]:+.6f}] | "
                f"{'n/a' if ordinary is None else f'{ordinary:+.6f}'} | "
                f"{'n/a' if tail is None else f'{tail:+.6f}'} | "
                f"{p['locked']['frac_candidate_better']:.3f} | "
                f"{'n/a' if top10 is None else f'{top10:+.6f}'} |"
            )
        cs = b["cross_seed"]
        lines.append("")
        lines.append(
            f"cross-seed ({split_key}): R0 locked MAE per seed {['%.6f' % v for v in cs['R0_locked_mae_per_seed']]} "
            f"(spread {cs['R0_seed_spread']:.6f}); R1 {['%.6f' % v for v in cs['R1_locked_mae_per_seed']]} "
            f"(spread {cs['R1_seed_spread']:.6f}); Δ per seed "
            f"{['%+.6f' % v for v in cs['delta_per_seed']]}, Δ difference {cs['delta_seed_difference']:+.6f}."
        )
        lines.append("")
    lines.append("## 3. Table 3 — mechanism (frozen checkpoints, eval-mode counterfactuals)")
    lines.append("")
    lines.append("| Quantity | R0 | R1 |")
    lines.append("|---|---:|---:|")
    if mechanism and mechanism.get("R1_vs_R0_summary"):
        s = mechanism["R1_vs_R0_summary"]
        lines.append(f"| mean abs(w), fit sample | {s['w_abs_mean_fit']['R0']:.5f} | {s['w_abs_mean_fit']['R1']:.5f} |")
        lines.append(f"| fraction abs(w) < 1e-3, fit sample | {s['w_frac_near_zero_fit']['R0']:.5f} | {s['w_frac_near_zero_fit']['R1']:.5f} |")
        for name in ("junction_zero", "junction_within_molecule_permutation", "junction_cross_molecule_shuffle"):
            v = s[f"{name}_mae_after"]
            lines.append(f"| monitor MAE after {name} | {v['R0']:.5f} | {v['R1']:.5f} (Δ {v['delta']:+.5f}) |")
    lines.append("")
    lines.append("## 4. Phase A — historical audits (zero training)")
    lines.append("")
    if err_audit:
        lines.append(
            "Monitor vs locked of the completed representation round: y≤−3 fraction "
            f"{err_audit['low_tail_fraction']['y_le_-3']['monitor']:.3f} vs "
            f"{err_audit['low_tail_fraction']['y_le_-3']['locked']:.3f}; MAE "
            f"{err_audit['error_profile']['monitor_B']['mae']:.5f} vs "
            f"{err_audit['error_profile']['locked_B']['mae']:.5f}; top-10 error mass share "
            f"{err_audit['error_profile']['monitor_B']['top10_err_mass_share']:.3f} vs "
            f"{err_audit['error_profile']['locked_B']['top10_err_mass_share']:.3f}."
        )
    if train_audit:
        lines.append(
            "N0/N1 full-10k curves: soup fit "
            f"{train_audit['full10k_N0']['soup_fit_mae']:.5f} / {train_audit['full10k_N1']['soup_fit_mae']:.5f}; "
            "junction-encoder gradients live on both frozen checkpoints: "
            f"{train_audit['junction_encoder_gradient']['N0']['junction_grad_live']} / "
            f"{train_audit['junction_encoder_gradient']['N1']['junction_grad_live']}."
        )
    lines.append("")
    lines.append("## 5. Pre-registered promotion gate")
    lines.append("")
    if a_gate:
        lines.append(f"Split A: `{a_gate['decision']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(a_gate["checks"], indent=2, sort_keys=True))
        lines.append("```")
    if b_gate:
        lines.append(f"Split B: `{b_gate['decision']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(b_gate["checks"], indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")
    lines.append("## 6. Next-stage CSCL research proposition (no training this round)")
    lines.append("")
    lines.append(_csl_proposal_text())
    lines.append("")
    lines.append("## 7. Artifacts")
    lines.append("")
    for path in sorted(OUT.glob("*.json")):
        lines.append(f"* `{path.name}`")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[report] REPORT.md written")
    return 0


def _csl_proposal_text() -> str:
    return (
        "**Proposition (identifiable structural contributions from the learned junction system).**\n\n"
        "1. *Contribution object*: a typed junction object (centre atom environment + ordered "
        "pair of incident bonds) carries a scalar task contribution ν_j = f_θ(shared junction "
        "type, context descriptor) with an explicit global bias b and atom/bond base terms "
        "(α_i, β_uv). The unit of attribution is the junction instance with its true bond "
        "ownership preserved.\n"
        "2. *Shared types and context*: junction types are canonical local colourings built from "
        "the existing label-blind raw features; context enters through pre-frozen, centred "
        "descriptors of the two incident bonds and the local neighbourhood, so a type's baseline "
        "contribution is comparable across molecules.\n"
        "3. *Additive terms and necessary interactions*: pred = b + Σ_i α_i + Σ_uv β_uv + Σ_j ν_j "
        "+ Σ_{jk} g(τ_j, τ_k, relation) for junctions sharing a bond; no higher-order learned "
        "interaction beyond the pre-registered first-order closure term.\n"
        "4. *Completeness identity*: the exported per-object contributions must sum to the "
        "prediction to numerical tolerance on every molecule (conservation gate), with UNK "
        "buckets for types unseen in the fit split.\n"
        "5. *Duplicate counting*: every junction, atom and bond is assigned to exactly one "
        "contribution object; a junction touching two bonds is one object, not two.\n"
        "6. *Gauge*: context and relation descriptors are centred on fit rows per type, so the "
        "type mean lives in the additive term and the interaction term has zero fit-domain mean; "
        "shared-type contributions are identified up to the declared centring convention, which "
        "must be exported with every attribution.\n"
        "7. *Synthetic validation*: before touching ZINC, verify identifiability on a synthetic "
        "task with known junction contributions and a co-occurrence confound, including recovery "
        "of the true interaction sign and a false-interaction rate measurement.\n"
        "8. *Predictive cost*: compare the decomposable model against the current opaque reader "
        "at matched parameters and recipe; the decomposition is only a candidate if the loss "
        "penalty stays inside a pre-registered band.\n"
        "9. *Seed stability*: contribution vectors for the same junction type must correlate "
        "across seeds above a pre-registered threshold, and the sign of the top contributions "
        "must be stable.\n"
        "10. *Boundary*: these are prediction attributions, not chemical causal effects; the "
        "round can only claim that the model's computation is faithfully decomposed.\n\n"
        "Important constraint discovered this round: the frozen BJOC-Null uses a graph-level "
        "junction mean, so no per-junction attribution can be read out of its weights or "
        "predictions. Any identifiable-contribution model must re-introduce true junction→bond "
        "ownership and be re-evaluated for predictive performance; its contribution scores "
        "cannot be validated by the current Null predictor."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("hist-error", cmd_hist_error),
        ("hist-train", cmd_hist_train),
        ("splits", cmd_splits),
        ("objects", cmd_objects),
        ("params", cmd_params),
        ("gates", cmd_gates),
        ("preflight", cmd_preflight),
        ("freeze", cmd_freeze),
        ("decision", cmd_decision),
        ("report", cmd_report),
    ):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)

    p = sub.add_parser("train")
    p.add_argument("--p", type=float, required=True, choices=[R0_P, R1_P])
    p.add_argument("--split", required=True, choices=["A", "B"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("train-stage")
    p.add_argument("--stage", required=True, choices=["split_a", "split_b"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_train_stage)

    p = sub.add_parser("predict")
    p.add_argument("--split", required=True, choices=["A", "B"])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rows", default="both", choices=["both", "monitor"])
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("score")
    p.add_argument("--split", required=True, choices=["A", "B"])
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("score-preview")
    p.add_argument("--split", required=True, choices=["A", "B"])
    p.set_defaults(fn=cmd_score_preview)

    p = sub.add_parser("decide")
    p.add_argument("--split", required=True, choices=["A", "B"])
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("mechanism")
    p.add_argument("--split", default="A", choices=["A", "B"])
    p.set_defaults(fn=cmd_mechanism)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
