"""SSRC Phase 2b shared constants, frozen-artifact audit and metrics.

Protocol: ``results/phase2b/PROTOCOL.md`` (``ssrc-phase2b-zinc-v1``).

Scope discipline
----------------
* Reuses the historical full-10k **P3/R3** checkpoints and predictions read-only.
* Trains only the two frozen Phase 2a arms (F, M), two seeds each, on the
  official ZINC-12k train 10,000 rows.
* Official valid is the development split: predictions are frozen and hashed
  before its labels are read.
* Official test is never forwarded or read unless the frozen valid promotion
  rule fires and exactly one finalist is locked.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (
    HERE,
    REPO / "tracks/structure-contribution/code",
    REPO / "tracks/structural-sufficiency/code/phase2a",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2a_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "ssrc-phase2b-zinc-v1"
TRACK = REPO / "tracks/structural-sufficiency"
OUT = TRACK / "results/phase2b"
CONFIG_PATH = TRACK / "configs/phase2b.yaml"

STRUCT_RESULTS = REPO / "tracks/structure-contribution/results"
F10K = STRUCT_RESULTS / "bjoc_full10k_pretraining_transfer_v1"
RECIPE = STRUCT_RESULTS / "bjoc_recipe_tuning_v1"
TERMINAL = STRUCT_RESULTS / "bjoc_r3_frozen_terminal_test_v1"

OBJECTS_TRAIN = F10K / "objects_train10k.pt"
OBJECTS_VALID = F10K / "objects_valid.pt"
PREP_TRAIN = F10K / "prep_train10k.json"
SSL_TEMPLATE = "pretrain_encoder_s{seed}.pt"
SPLIT_P3 = "soup_R3_seed{seed}.pt"
VALID_PRED_P3 = RECIPE / "predictions" / "valid_R3{seed}.npz"
TEST_PRED_P3 = TERMINAL / "predictions" / "test_P3{seed}.npz"
VALID_Y = RECIPE / "valid_y_read_once.npy"
TEST_Y = TERMINAL / "test_y_read_once.npy"
ENCODED_TRAIN = REPO / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

N_TRAIN = 10000
N_VALID = 1000
N_TEST = 1000
SEEDS = (0, 1)
ARM_KINDS = {"P3": M.KIND_NONE, "F": M.KIND_NODE_FFN, "M": M.KIND_LOCAL_MP}
NEW_ARM_KINDS = {"F": M.KIND_NODE_FFN, "M": M.KIND_LOCAL_MP}
CANDIDATES = ("F", "M")
EXPECTED_PARAMETERS = {"P3": 413505, "F": 463489, "M": 464257}

#: frozen hyper-parameters (protocol §6)
RECIPE_KEY = "R3"
EPOCHS = 360
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
STEPS_PER_EPOCH = 79
UPDATES = EPOCHS * STEPS_PER_EPOCH
GRAD_CLIP = 5.0
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5

#: frozen statistics
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261024
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = (1, 5, 10)
EXTREME_VALID_ROW = 172


# ---------------------------------------------------------------------------
# io / hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray, dtype: str = "float32") -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_json(path: Path | str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_once_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    write_json(path, payload)


def load_config() -> dict[str, Any]:
    import yaml

    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# frozen data
# ---------------------------------------------------------------------------


def verify_file(name: str, path: Path, expected: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing frozen artifact {name}: {path}")
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"hash mismatch for {name}: {got} != {expected}")
    return {"name": name, "path": str(path.relative_to(REPO)), "sha256": got}


def load_objects(path: Path, expected_sha: str) -> list[sro.MolObjects]:
    verify_file(path.name, path, expected_sha)
    objs = torch_load(path)
    return objs


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


_TRAIN_Y: np.ndarray | None = None


def load_train_y() -> np.ndarray:
    """Raw supervised targets for official-train rows 0..9999 (historical file)."""
    global _TRAIN_Y
    if _TRAIN_Y is None:
        enc = torch_load(ENCODED_TRAIN)
        if len(enc) != N_TRAIN:
            raise SystemExit(f"encoded_train has {len(enc)} rows, expected {N_TRAIN}")
        _TRAIN_Y = np.asarray([float(enc[i].y.reshape(-1)[0]) for i in range(N_TRAIN)], dtype=np.float64)
        del enc
    return _TRAIN_Y


def load_valid_y() -> np.ndarray:
    cfg = load_config()
    spec = cfg["frozen_inputs"]["valid_y"]
    verify_file("valid_y", REPO / spec["path"], spec["sha256"])
    y = np.load(REPO / spec["path"]).astype(np.float64)
    if y.shape != (N_VALID,):
        raise SystemExit(f"unexpected valid y shape {y.shape}")
    return y


def load_test_y() -> np.ndarray:
    cfg = load_config()
    spec = cfg["frozen_inputs"]["test_y"]
    verify_file("test_y", REPO / spec["path"], spec["sha256"])
    y = np.load(REPO / spec["path"]).astype(np.float64)
    if y.shape != (N_TEST,):
        raise SystemExit(f"unexpected test y shape {y.shape}")
    return y


# ---------------------------------------------------------------------------
# historical P3 artifacts
# ---------------------------------------------------------------------------


def p3_run(seed: int) -> dict[str, Any]:
    return read_json(RECIPE / f"train_R3_seed{seed}.json")


def p3_soup_path(seed: int) -> Path:
    return RECIPE / SPLIT_P3.format(seed=seed)


def p3_soup_sha(seed: int) -> str:
    return sha256_file(p3_soup_path(seed))


def p3_valid_pred(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = load_config()
    spec = cfg["frozen_inputs"][f"valid_pred_P3{seed}"]
    rec = verify_file(f"valid_pred_P3{seed}", REPO / spec["path"], spec["sha256"])
    with np.load(REPO / spec["path"]) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
        rows = np.asarray(z["row"], dtype=np.int64)
    if pred.shape != (N_VALID,) or not np.array_equal(rows, np.arange(N_VALID)):
        raise SystemExit(f"unexpected valid prediction layout for P3 seed {seed}")
    rec["array_sha256"] = sha256_array(pred)
    return pred, rec


def p3_test_pred(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = load_config()
    spec = cfg["frozen_inputs"][f"test_pred_P3{seed}"]
    rec = verify_file(f"test_pred_P3{seed}", REPO / spec["path"], spec["sha256"])
    with np.load(REPO / spec["path"]) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
        rows = np.asarray(z["row"], dtype=np.int64)
    if pred.shape != (N_TEST,) or not np.array_equal(rows, np.arange(N_TEST)):
        raise SystemExit(f"unexpected test prediction layout for P3 seed {seed}")
    rec["array_sha256"] = sha256_array(pred)
    return pred, rec


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def load_norm() -> dict[str, Any]:
    spec = load_config()["frozen_inputs"]["prep_train10k"]
    verify_file("prep_train10k", REPO / spec["path"], spec["sha256"])
    norm = read_json(REPO / spec["path"])
    if not norm.get("fit_only") or int(norm.get("n_fit", 0)) != N_TRAIN:
        raise SystemExit("standardiser is not the frozen train-10k-only one")
    return norm


def ssl_state(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config()
    spec = cfg["frozen_inputs"][f"pretrain_encoder_s{seed}"]
    rec = verify_file(f"pretrain_encoder_s{seed}", REPO / spec["path"], spec["sha256"])
    return torch_load(REPO / spec["path"]), rec


def build_arm(arm: str, seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """Rebuild the exact per-seed P3 supervision-phase init, then add F/M.

    The trunk is built by the frozen Phase 2a builder and must hash to the
    historical R3/P init for the seed; the seed-matched full-10k SSL encoder
    must then hash to the historical ``encoder_used_sha256``.
    """
    import torch

    import bjoc_masked_pretraining_v1 as mp
    import bjoc_perfmech_long_v1 as bp

    kind = ARM_KINDS[arm]
    norm = load_norm()
    base, model, info = M.build_local_model(kind, seed=seed, norm=norm, device="cpu")
    rec = p3_run(seed)["init"]
    checks = {
        "trunk_matches_base_init": bool(info["trunk_matches_base_init"]),
        "scratch_full_matches_r3": info["base_scratch_init_full_sha256"] == rec["init_full_sha256"],
        "scratch_group_matches_r3": info["base_scratch_init_group_state_sha256"] == rec["init_group_state_sha256"],
        "scratch_encoder_matches_r3": info["base_scratch_init_encoder_sha256"] == rec["init_encoder_sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"trunk init does not match the historical R3 record: {checks}")
    state, ssl_rec = ssl_state(seed)
    mp.load_encoder_state(model, state, "cpu")
    encoder_after = bp.encoder_group_hash(model)
    checks["encoder_used_matches_r3"] = encoder_after == rec["encoder_used_sha256"]
    checks["encoder_file_matches_record"] = ssl_rec["sha256"] == rec["pretrain_encoder_file_sha256"]
    if not all(checks.values()):
        raise SystemExit(f"SSL encoder does not match the historical R3 record: {checks}")
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != EXPECTED_PARAMETERS[arm]:
        raise SystemExit(f"parameter count {n_params} != {EXPECTED_PARAMETERS[arm]} for {arm}")
    audit = M.parameter_audit(model)
    if not audit["parameter_exact"]:
        raise SystemExit(f"parameter audit failed for {arm}: {audit}")
    model = model.to(torch.device(device))
    return model, {
        "arm": arm,
        "seed": int(seed),
        "parameters": n_params,
        "parameter_audit": audit,
        "init": info,
        "ssl": ssl_rec,
        "checks": checks,
        "encoder_used_sha256": encoder_after,
    }


def load_soup(model: Any, path: Path) -> dict[str, Any]:
    import torch

    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    soup = torch.load(path, map_location="cpu", weights_only=False)
    want = model.state_dict()
    if set(soup.keys()) != set(want.keys()):
        missing = sorted(set(want) - set(soup))
        extra = sorted(set(soup) - set(want))
        raise SystemExit(f"checkpoint key mismatch: missing={missing[:5]} extra={extra[:5]}")
    model.load_state_dict(soup, strict=True)
    return {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "keys": len(soup)}


def build_p3(seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    model, info = build_arm("P3", seed, "cpu")
    soup = load_soup(model, p3_soup_path(seed))
    cfg = load_config()
    expected = cfg["audit_expectations"]["p3_soup_sha256"][str(seed)]
    if soup["sha256"] != expected:
        raise SystemExit(f"P3 soup hash mismatch for seed {seed}")
    info = {**info, "soup": soup, "parameters": int(sum(p.numel() for p in model.parameters()))}
    return model.to(device), info


def predict(model: Any, objs: Sequence[sro.MolObjects], device: str | Any, batch_size: int = 256) -> tuple[np.ndarray, float]:
    import time

    import torch

    model.eval()
    out = []
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(objs), batch_size):
            batch = sro.batch_objects(list(objs[i : i + batch_size]))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(out, 0).astype(np.float64)
    return pred, time.perf_counter() - started


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def abs_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    e = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return {
        "n": int(e.size),
        "mae": float(e.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "median_ae": float(np.median(e)),
        "p90_ae": float(np.percentile(e, 90)),
        "p95_ae": float(np.percentile(e, 95)),
        "p99_ae": float(np.percentile(e, 99)),
        "max_ae": float(e.max()),
        "mean_squared_error": float(np.mean(e**2)),
    }


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n), d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "better_fraction": float((d > 0).mean()),
        "n_resamples": int(n),
        "bootstrap_seed": int(seed),
        "per_molecule_unit": True,
    }


def seed_averaged_delta(pred_cand: Mapping[int, np.ndarray], pred_ref: Mapping[int, np.ndarray], y: np.ndarray):
    """d_i = mean_s (|e_ref,s,i| - |e_cand,s,i|); mean(d) = seed-mean MAE gain."""
    y = np.asarray(y, dtype=np.float64)
    parts = [
        np.abs(np.asarray(pred_ref[s], dtype=np.float64) - y) - np.abs(np.asarray(pred_cand[s], dtype=np.float64) - y)
        for s in SEEDS
    ]
    return np.mean(np.stack(parts, 0), axis=0)


def mse_delta(pred_cand: Mapping[int, np.ndarray], pred_ref: Mapping[int, np.ndarray], y: np.ndarray):
    y = np.asarray(y, dtype=np.float64)
    parts = [
        (np.asarray(pred_ref[s], dtype=np.float64) - y) ** 2 - (np.asarray(pred_cand[s], dtype=np.float64) - y) ** 2
        for s in SEEDS
    ]
    return np.mean(np.stack(parts, 0), axis=0)


def top_removed_gain(d: np.ndarray, indices: Iterable[int]) -> float:
    drop = np.zeros(d.size, dtype=bool)
    drop[list(indices)] = True
    keep = ~drop
    return float(d[keep].mean())


def error_rank(pred: Mapping[int, np.ndarray], y: np.ndarray) -> np.ndarray:
    """Mean absolute error over seeds, descending order of magnitude."""
    y = np.asarray(y, dtype=np.float64)
    e = np.mean([np.abs(np.asarray(pred[s], dtype=np.float64) - y) for s in SEEDS], axis=0)
    return np.argsort(-e)


def subgroup_metrics(
    preds: Mapping[str, Mapping[int, np.ndarray]],
    y: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)
        entry: dict[str, Any] = {"n": int(idx.size)}
        for key, pred in preds.items():
            entry[key] = {
                "mae_seed0": float(np.abs(pred[0][idx] - y[idx]).mean()) if idx.size else None,
                "mae_seed1": float(np.abs(pred[1][idx] - y[idx]).mean()) if idx.size else None,
            }
            entry[key]["mae_seed_mean"] = (
                0.5 * (entry[key]["mae_seed0"] + entry[key]["mae_seed1"]) if idx.size else None
            )
        out[name] = entry
    return out


def cycle_proxy(objs: Sequence[sro.MolObjects]) -> tuple[np.ndarray, np.ndarray]:
    """``max(len(c) for c in nx.cycle_basis(G))`` per molecule (order-sensitive).

    This is a descriptive proxy for the label's long-cycle term, recomputed from
    the PyG bond graph.  It is never a model input.
    """
    import networkx as nx

    longest = np.zeros(len(objs), dtype=np.int64)
    for i, o in enumerate(objs):
        g = nx.Graph()
        g.add_nodes_from(range(int(o.n_atoms)))
        g.add_edges_from({(min(int(a), int(b)), max(int(a), int(b))) for a, b in o.bond_index.tolist()})
        basis = nx.cycle_basis(g)
        longest[i] = int(max((len(c) for c in basis), default=0))
    return longest, longest > 6


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSpec:
    key: str
    arm: str
    seed: int


def parse_seed_key(key: str) -> SeedSpec:
    arm = key[0]
    seed = int(key[1:])
    return SeedSpec(key=key, arm=arm, seed=seed)


def metric_block(preds: Mapping[str, Mapping[int, np.ndarray]], y: np.ndarray) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for key, pred in preds.items():
        per_seed[key] = {s: abs_metrics(pred[s], y) for s in SEEDS}
    seed_mean = {key: float(np.mean([per_seed[key][s]["mae"] for s in SEEDS])) for key in preds}
    return {"per_seed": per_seed, "seed_mean_mae": seed_mean}
