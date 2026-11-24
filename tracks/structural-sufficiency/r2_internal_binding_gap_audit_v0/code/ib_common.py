"""Shared, read-only plumbing for the R2 internal-binding gap audit.

No model is trained here.  The module only:

* puts the frozen parent code on ``sys.path``;
* re-verifies the parent source/data pins;
* loads the frozen bundles, best checkpoints and committed Dev predictions;
* reproduces the EGO2 ego packing with optional edge removal;
* provides the metric, scaffold-bootstrap and topology helpers used by the
  analysis scripts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
STRUCT = ROUND_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]

SR_ROUND = STRUCT / "structure_route_resolution_8h_v1"
SR_CODE = SR_ROUND / "code"
SR_RESULTS = SR_ROUND / "results"
R2_ROUND = STRUCT / "r2_sentence_replication_v1"
R2_CODE = R2_ROUND / "code"
GWIC_ROUND = STRUCT / "graph_word_incidence_composition_pilot_v0"
GWIC_CODE = GWIC_ROUND / "code"

for _p in (SR_CODE, R2_CODE, GWIC_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sr_common as SC  # noqa: E402
import sr_data as SD  # noqa: E402
import sr_gnn as SG  # noqa: E402
import sr_struct as SS  # noqa: E402
import r2_data as RD  # noqa: E402
import gwic_data as GD  # noqa: E402
import gwic_model as GM  # noqa: E402
import gwic_structure as GS  # noqa: E402
import gwic_tensor as GT  # noqa: E402

ANALYSIS_DIR = ROUND_DIR / "analysis"
SPLITS = ("D0", "D1")
CELLS = (("D0", 0), ("D0", 1), ("D1", 0))
RUNS = {
    ("D0", 0): {"R2": "R2_REF_D0_S0", "EGO2": "G_EGO2_5_D0_S0"},
    ("D0", 1): {"R2": "R2_REF_D0_S1", "EGO2": "G_EGO2_5_D0_S1"},
    ("D1", 0): {"R2": "R2_REF_D1_S0", "EGO2": "G_EGO2_5_D1_S0"},
}
EXPECTED_IDENTITY = {
    "D0": "60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e",
    "D1": "c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee",
}
EXPECTED_FIT = {
    "D0": "7edd3c68d946e9f3f7c5613993530dd44c2371956ee031582e123ef79c8bc6c0",
    "D1": "b9f74030ee0decc67005e10963e64d7da16fd70d9a6bf9790f4a1c4afcebebbc",
}
EXPECTED_EVAL = {
    "D0": "eacce8acde66acb6f0045813f8d6103381012ebab8181b7d314a2b66aa387193",
    "D1": "342c1de0ece164e9acba17ba17a48151c90235242e77481e41a1a9820b0380a6",
}
EXPECTED_SCALER = {
    "D0": {"mean": 2.185815826311436, "std": 1.2139664740467941, "n": 2856},
    "D1": {"mean": 2.151278011472791, "std": 1.2159778136886414, "n": 2856},
}
BOOT_SEED = 1703
N_BOOT = 5000
PRIMARY_FEATURES = ("closure_22", "reconvergence_2", "ring_frac")
ROOT_FEATURES = (
    "closure_11",
    "closure_22",
    "reconvergence_2",
    "n_nodes",
    "n_edges",
    "cycle_rank",
    "supports3",
    "supports4",
)


# ---------------------------------------------------------------------------
# small IO / identity helpers
# ---------------------------------------------------------------------------
def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path | str, payload: Any) -> Path:
    return SC.write_json(path, payload)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    return SC.write_csv(path, rows)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha(files: Sequence[str | Path] | None = None) -> str:
    if files is None:
        files = sorted(CODE_DIR.glob("ib_*.py"))
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in files):
        if not path.is_absolute():
            path = CODE_DIR / path
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is required by the round
        torch_version = "unknown"
        cuda = None
        cuda_available = False
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda,
        "cuda_available": cuda_available,
        "cpu_count": int(os.cpu_count() or 0),
        "device": "cpu",
        "git_commit": SC.git_commit(),
        "git_dirty_tracked": bool(SC.git_dirty()),
    }


def assert_inputs() -> dict[str, Any]:
    """Verify frozen sources, checkpoints, manifests and predictions exist."""
    pins = SC.assert_pins()
    frozen = RD.RC.assert_frozen_sources()
    details: dict[str, Any] = {"pins": pins, "frozen": frozen, "runs": {}}
    for cell, arms in RUNS.items():
        split, seed = cell
        for arm, run_id in arms.items():
            run_dir = SR_RESULTS / run_id
            manifest = run_dir / "manifest.json"
            checkpoint = run_dir / "state_best.pt"
            predictions = run_dir / "predictions_best_dev.npz"
            for path in (manifest, checkpoint, predictions):
                if not path.is_file():
                    raise FileNotFoundError(f"missing frozen input: {path}")
            details["runs"][run_id] = {
                "split": split,
                "seed": int(seed),
                "arm": arm,
                "manifest_sha256": sha256_file(manifest),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(predictions),
            }
    return details


def load_bundle(split: str) -> Any:
    if split not in SPLITS:
        raise KeyError(split)
    return SD.load_bundle(split)


def split_identity(bundle: Any) -> dict[str, Any]:
    spec = SD.split_identity(bundle)
    split = str(spec["split"])
    if str(spec["identity_sha256"]) != EXPECTED_IDENTITY[split]:
        raise ValueError(f"{split}: identity mismatch {spec['identity_sha256']}")
    if str(spec["fit_rows_sha256"]) != EXPECTED_FIT[split]:
        raise ValueError(f"{split}: Fit-row hash mismatch")
    if str(spec["eval_rows_sha256"]) != EXPECTED_EVAL[split]:
        raise ValueError(f"{split}: Dev-row hash mismatch")
    if int(spec["n_fit"]) != 2856 or int(spec["n_eval"]) != 504:
        raise ValueError(f"{split}: split sizes changed")
    scaler = {k: spec["scaler"][k] for k in ("mean", "std", "n")}
    if not np.allclose([scaler["mean"], scaler["std"]],
                       [EXPECTED_SCALER[split]["mean"], EXPECTED_SCALER[split]["std"]],
                       rtol=0.0, atol=1e-12):
        raise ValueError(f"{split}: target scaler mismatch {scaler}")
    if int(scaler["n"]) != 2856:
        raise ValueError(f"{split}: target scaler n mismatch")
    return spec


def run_dir(run_id: str) -> Path:
    return SR_RESULTS / run_id


def load_manifest(run_id: str) -> dict[str, Any]:
    return json.loads((run_dir(run_id) / "manifest.json").read_text(encoding="utf-8"))


def load_predictions(run_id: str, bundle: Any) -> dict[str, np.ndarray]:
    payload = np.load(run_dir(run_id) / "predictions_best_dev.npz", allow_pickle=False)
    out = {k: np.asarray(payload[k]) for k in payload.files}
    for key in ("row_id", "pred", "label", "pred_std"):
        if key not in out:
            raise KeyError(f"{run_id}: predictions missing {key}")
    rows = np.asarray(bundle.eval_rows, np.int64)
    label = np.asarray(bundle.labels[bundle.eval_rows], np.float64)
    if not np.array_equal(out["row_id"].astype(np.int64), rows):
        raise ValueError(f"{run_id}: prediction rows are not the frozen Dev rows")
    if not np.allclose(out["label"], label, rtol=0.0, atol=1e-9):
        raise ValueError(f"{run_id}: stored labels differ from the frozen bundle")
    if not np.isfinite(out["pred"]).all() or not np.isfinite(label).all():
        raise ValueError(f"{run_id}: non-finite Dev prediction or label")
    out["y"] = label
    out["rows"] = rows
    return out


def checkpoint(run_id: str) -> dict[str, Any]:
    import torch

    return torch.load(run_dir(run_id) / "state_best.pt", map_location="cpu",
                      weights_only=False)


def verify_checkpoint_identity(run_id: str, split: str) -> dict[str, Any]:
    ck = checkpoint(run_id)
    manifest = load_manifest(run_id)
    if int(ck["epoch"]) != int(manifest["best_epoch"]):
        raise ValueError(f"{run_id}: checkpoint epoch != manifest best_epoch")
    data = ck.get("data")
    if isinstance(data, dict) and "identity_sha256" in data:
        identity = str(data["identity_sha256"])
    elif isinstance(ck.get("split"), dict):
        identity = str(ck["split"]["identity_sha256"])
    else:
        raise ValueError(f"{run_id}: checkpoint has no split identity")
    if identity != EXPECTED_IDENTITY[split]:
        raise ValueError(f"{run_id}: checkpoint split identity mismatch {identity}")
    return {"run_id": run_id, "epoch": int(ck["epoch"]),
            "identity_sha256": identity}


def scaffold_map(split: str) -> dict[int, str]:
    mapping = SD.scaffolds(split)
    if len(mapping) != 3360:
        raise ValueError(f"{split}: scaffold map does not cover 3360 rows")
    return mapping


def clusters_for(rows: Sequence[int], scaffolds: Mapping[int, str]) -> np.ndarray:
    return np.asarray([str(scaffolds[int(r)]) for r in rows], dtype=object)


# ---------------------------------------------------------------------------
# metrics and paired scaffold bootstrap
# ---------------------------------------------------------------------------
def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64)
                                  - np.asarray(true, np.float64)) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64)
                                - np.asarray(true, np.float64))))


def point_metrics(y: np.ndarray, pred_r2: np.ndarray, pred_ego2: np.ndarray,
                  mask: np.ndarray | None = None,
                  clusters: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y, np.float64)
    pa = np.asarray(pred_r2, np.float64)
    pb = np.asarray(pred_ego2, np.float64)
    if mask is None:
        mask = np.ones(y.size, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    y = y[mask]
    pa = pa[mask]
    pb = pb[mask]
    delta = np.abs(pa - y) - np.abs(pb - y)
    out: dict[str, Any] = {
        "n_molecules": int(y.size),
        "rmse_r2": rmse(pa, y),
        "rmse_ego2": rmse(pb, y),
        "mae_r2": mae(pa, y),
        "mae_ego2": mae(pb, y),
        "gain_rmse": rmse(pa, y) - rmse(pb, y),
        "gain_mae": mae(pa, y) - mae(pb, y),
        "mean_delta": float(delta.mean()) if delta.size else None,
        "prop_ego2_better": float(np.mean(delta > 0)) if delta.size else None,
        "prop_r2_better": float(np.mean(delta < 0)) if delta.size else None,
        "prop_tie": float(np.mean(delta == 0)) if delta.size else None,
        "max_delta": float(delta.max()) if delta.size else None,
        "min_delta": float(delta.min()) if delta.size else None,
        "delta": delta,
    }
    if clusters is not None:
        c = np.asarray(clusters, dtype=object)[mask]
        out["n_clusters"] = int(np.unique(c).size)
    else:
        out["n_clusters"] = None
    return out


def _cluster_index(clusters: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    clusters = np.asarray(clusters, dtype=object)
    keys = sorted(set(str(x) for x in clusters.tolist()))
    mapping = {key: np.flatnonzero(clusters.astype(str) == key) for key in keys}
    return keys, mapping


def _bootstrap_indices(keys: Sequence[str], mapping: Mapping[str, np.ndarray],
                       n_boot: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out: list[np.ndarray] = []
    n_clusters = len(keys)
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, n_clusters, n_clusters)
        idx = [mapping[keys[int(i)]] for i in sampled.tolist()]
        out.append(np.concatenate(idx) if idx else np.zeros(0, np.int64))
    return out


def bootstrap_paired(y: np.ndarray, pred_r2: np.ndarray, pred_ego2: np.ndarray,
                     clusters: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict[str, Any]:
    y = np.asarray(y, np.float64)
    pa = np.asarray(pred_r2, np.float64)
    pb = np.asarray(pred_ego2, np.float64)
    clusters = np.asarray(clusters, dtype=object)
    keys, mapping = _cluster_index(clusters)
    draws = _bootstrap_indices(keys, mapping, n_boot, seed)
    gain_rmse = np.empty(n_boot, np.float64)
    gain_mae = np.empty(n_boot, np.float64)
    prop = np.empty(n_boot, np.float64)
    for i, idx in enumerate(draws):
        yy = y[idx]
        aa = pa[idx]
        bb = pb[idx]
        ra = np.sqrt(np.mean((aa - yy) ** 2))
        rb = np.sqrt(np.mean((bb - yy) ** 2))
        ea = np.abs(aa - yy)
        eb = np.abs(bb - yy)
        gain_rmse[i] = ra - rb
        gain_mae[i] = float(np.mean(ea - eb))
        prop[i] = float(np.mean(eb < ea))
    point = point_metrics(y, pa, pb, clusters=clusters)
    point.pop("delta", None)
    out = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_clusters": int(len(keys)),
        "gain_rmse": float(point["gain_rmse"]),
        "gain_mae": float(point["gain_mae"]),
        "prop_ego2_better": float(point["prop_ego2_better"]),
        "gain_rmse_ci": _ci(gain_rmse),
        "gain_mae_ci": _ci(gain_mae),
        "prop_ego2_better_ci": _ci(prop),
        "gain_rmse_sd": float(np.std(gain_rmse, ddof=1)),
        "gain_mae_sd": float(np.std(gain_mae, ddof=1)),
        "draws": {"gain_rmse": gain_rmse, "gain_mae": gain_mae, "prop": prop},
    }
    return out


def _ci(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"lo": None, "hi": None, "n": int(values.size)}
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n": int(values.size)}


def bootstrap_strata(y: np.ndarray, pred_r2: np.ndarray, pred_ego2: np.ndarray,
                     clusters: np.ndarray, masks: Mapping[str, np.ndarray],
                     contrasts: Sequence[tuple[str, str]],
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Joint scaffold-cluster bootstrap for disjoint or overlapping strata.

    Clusters are resampled over the whole Dev set; each stratum gain is evaluated
    on the molecules of that stratum inside the draw.  Contrasts are evaluated on
    the same draw, so shared clusters are preserved.
    """
    y = np.asarray(y, np.float64)
    pa = np.asarray(pred_r2, np.float64)
    pb = np.asarray(pred_ego2, np.float64)
    clusters = np.asarray(clusters, dtype=object)
    keys, mapping = _cluster_index(clusters)
    draws = _bootstrap_indices(keys, mapping, n_boot, seed)
    names = list(masks.keys())
    mask_arrays = {name: np.asarray(masks[name], dtype=bool) for name in names}
    draws_gain: dict[str, np.ndarray] = {
        name: np.full(n_boot, np.nan, np.float64) for name in names
    }
    draws_rmse: dict[str, np.ndarray] = {
        name: np.full(n_boot, np.nan, np.float64) for name in names
    }
    for i, idx in enumerate(draws):
        for name in names:
            sel = idx[mask_arrays[name][idx]]
            if sel.size < 2:
                continue
            yy = y[sel]
            aa = pa[sel]
            bb = pb[sel]
            draws_gain[name][i] = float(np.mean(np.abs(aa - yy) - np.abs(bb - yy)))
            draws_rmse[name][i] = (float(np.sqrt(np.mean((aa - yy) ** 2)))
                                   - float(np.sqrt(np.mean((bb - yy) ** 2))))
    strata: dict[str, Any] = {}
    for name in names:
        mask = mask_arrays[name]
        point = point_metrics(y, pa, pb, mask=mask, clusters=clusters)
        point.pop("delta", None)
        strata[name] = {
            **point,
            "gain_mae_ci": _ci(draws_gain[name]),
            "gain_rmse_ci": _ci(draws_rmse[name]),
        }
    contrast_out: dict[str, Any] = {}
    for a, b in contrasts:
        diff_mae = draws_gain[a] - draws_gain[b]
        diff_rmse = draws_rmse[a] - draws_rmse[b]
        point_a = strata[a]["gain_mae"]
        point_b = strata[b]["gain_mae"]
        contrast_out[f"{a}_minus_{b}"] = {
            "gain_mae": (None if point_a is None or point_b is None
                         else float(point_a - point_b)),
            "gain_rmse": (None if strata[a]["gain_rmse"] is None
                          or strata[b]["gain_rmse"] is None
                          else float(strata[a]["gain_rmse"] - strata[b]["gain_rmse"])),
            "gain_mae_ci": _ci(diff_mae),
            "gain_rmse_ci": _ci(diff_rmse),
            "n_valid_draws": int(np.isfinite(diff_mae).sum()),
        }
    return {"strata": strata, "contrasts": contrast_out,
            "draws_gain": draws_gain, "draws_rmse": draws_rmse,
            "n_boot": int(n_boot), "seed": int(seed), "n_clusters": int(len(keys))}


def dominance(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, np.float64)
    n = int(delta.size)
    if n == 0:
        return {"n_molecules": 0, "dominated": True}
    total_abs = float(np.abs(delta).sum())
    order = np.argsort(-np.abs(delta), kind="mergesort")
    max_share = float(np.abs(delta[order[0]]) / total_abs) if total_abs > 0 else 0.0
    k = max(1, int(math.ceil(0.05 * n)))
    top5_share = float(np.abs(delta[order[:k]]).sum() / total_abs) if total_abs > 0 else 0.0
    gain = float(delta.mean())
    after_max = float(np.delete(delta, order[0]).mean()) if n > 1 else float("nan")
    keep = np.ones(n, dtype=bool)
    keep[order[:k]] = False
    after_top5 = float(delta[keep].mean()) if keep.any() else float("nan")
    sign_preserved = (np.sign(after_max) == np.sign(gain)) if np.isfinite(after_max) else False
    retained = (abs(after_top5) / abs(gain)) if gain != 0 and np.isfinite(after_top5) else float("nan")
    return {
        "n_molecules": n,
        "max_abs_share": max_share,
        "top5pct_share": top5_share,
        "gain_mae": gain,
        "gain_after_max_abs": after_max,
        "sign_preserved_after_max_abs": bool(sign_preserved),
        "gain_after_top5pct": after_top5,
        "retained_fraction_after_top5pct": float(retained) if np.isfinite(retained) else None,
        "dominated": bool(
            (not sign_preserved)
            or (not np.isfinite(retained))
            or (abs(retained) < 0.5)
        ),
    }


# ---------------------------------------------------------------------------
# R2 model replay
# ---------------------------------------------------------------------------
def load_r2_model(run_id: str, bundle: Any):
    import torch

    ck = checkpoint(run_id)
    model = GM.load_model_from_state(*GD.atom_bond_dims(), ck["state"],
                                     RD.norm_of(bundle), use_words=False,
                                     config=ck["config"])
    model.eval()
    return model, ck


def build_r2_structs(bundle: Any, rows: Sequence[int]) -> dict[int, Any]:
    structs: dict[int, Any] = {}
    for row in rows:
        view = GD.molecule_view(bundle.raw, int(row))
        view["atom_scalars"] = GD.atom_scalars(view)
        structs[int(row)] = GS.build_molecule_struct(view)
    return structs


def predict_r2(model: Any, rows: Sequence[int], structs: Mapping[int, Any],
               raw: Mapping[str, Any], scaler: Mapping[str, float],
               batch_size: int = 64) -> np.ndarray:
    import torch

    model.eval()
    rows = np.asarray(rows, np.int64)
    pred = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = GT.collate(part.tolist(), structs, raw, "cpu")
            out = model(batch, arm="SENTENCE_BASE")
            pred[start:start + part.size] = (
                out["yhat_std"].double().numpy() * float(scaler["std"])
                + float(scaler["mean"])
            )
    return pred


# ---------------------------------------------------------------------------
# EGO2 model, ego packing and sensitivity
# ---------------------------------------------------------------------------
def load_ego2_model(run_id: str, bundle: Any):
    import torch

    ck = checkpoint(run_id)
    seed = int(ck["row"]["seed"])
    model, _info = SG.build_model("G_EGO2_5", *SD.feature_dims(), seed,
                                  SD.norm_of(bundle), ck["config"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def _adjacency(n_atoms: int, u: np.ndarray, v: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def ego_block_with_stats(st: Any, radius: int = 2, drop: str | None = None
                         ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Reproduce ``sr_gnn.ego_block`` with an optional non-tree edge removal.

    ``drop`` is ``None``, ``"11"`` (distance-1--distance-1 edges) or ``"22"``
    (distance-2--distance-2 edges).  Only the packed connectivity changes; the
    atom order and pointers are identical to the original ego packing.
    """
    if drop not in (None, "11", "22"):
        raise KeyError(drop)
    n = int(st.n_atoms)
    adj = _adjacency(n, st.bond_u, st.bond_v)
    atom: list[int] = []
    src: list[int] = []
    dst: list[int] = []
    bond: list[int] = []
    ptr: list[int] = [0]
    removed_per_root: list[int] = []
    total_per_root: list[int] = []
    included_per_root: list[int] = []
    for root in range(n):
        dist = SG._bfs(n, adj, root, radius)
        nodes = [root] + [x for x in range(n) if 0 < dist[x] <= radius]
        base = len(atom)
        pos = {int(a): base + i for i, a in enumerate(nodes)}
        atom.extend(nodes)
        removed = 0
        total = 0
        for bid, (a, b) in enumerate(zip(st.bond_u.tolist(), st.bond_v.tolist())):
            a = int(a)
            b = int(b)
            if a in pos and b in pos:
                total += 1
                da, db = int(dist[a]), int(dist[b])
                if drop == "11" and da == 1 and db == 1:
                    removed += 1
                    continue
                if drop == "22" and da == 2 and db == 2:
                    removed += 1
                    continue
                ia, ib = pos[a], pos[b]
                src.extend([ia, ib])
                dst.extend([ib, ia])
                bond.extend([bid, bid])
        ptr.append(len(atom))
        removed_per_root.append(removed)
        total_per_root.append(total)
        included_per_root.append(total - removed)
    block = {
        "atom": np.asarray(atom, np.int32),
        "src": np.asarray(src, np.int32),
        "dst": np.asarray(dst, np.int32),
        "bond": np.asarray(bond, np.int32),
        "ptr": np.asarray(ptr, np.int64),
    }
    stats = {
        "removed_per_root": np.asarray(removed_per_root, np.int64),
        "total_per_root": np.asarray(total_per_root, np.int64),
        "included_per_root": np.asarray(included_per_root, np.int64),
    }
    return block, stats


def build_ego_variants(structs: Mapping[int, Any], rows: Sequence[int]
                       ) -> tuple[dict[int, Any], dict[str, dict[int, Any]],
                                  dict[str, dict[int, Any]]]:
    """Return original structs, variant structs and per-molecule edge stats."""
    original: dict[int, Any] = {}
    variants: dict[str, dict[int, Any]] = {"11": {}, "22": {}}
    stats: dict[str, dict[int, Any]] = {"11": {}, "22": {}}
    for row in rows:
        st = structs[int(row)]
        SG.prepare_ranges(st)
        ref = SG.ego_block(st, 2)
        block0, stats0 = ego_block_with_stats(st, 2, None)
        for key in ("atom", "src", "dst", "bond", "ptr"):
            if not np.array_equal(block0[key], ref[key]):
                raise AssertionError(f"ego packing mismatch row={row} key={key}")
        original[int(row)] = st
        for drop in ("11", "22"):
            block, stt = ego_block_with_stats(st, 2, drop)
            for key in ("atom", "ptr"):
                if not np.array_equal(block[key], ref[key]):
                    raise AssertionError(f"variant node/pointer mismatch {row} {drop}")
            variant = copy.copy(st)
            variant.ego2 = block
            variants[drop][int(row)] = variant
            stats[drop][int(row)] = stt
    return original, variants, stats


def predict_ego2(model: Any, rows: Sequence[int], structs: Mapping[int, Any],
                 bundle: Any, batch_size: int = 16, return_roots: bool = False
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    model.eval()
    rows = np.asarray(rows, np.int64)
    pred = np.zeros(rows.size, np.float64)
    roots: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = SG.collate(part.tolist(), structs, "EGO2", "cpu")
            out = model(batch, return_h=bool(return_roots))
            pred[start:start + part.size] = (
                out["yhat_std"].double().numpy() * float(bundle.scaler["std"])
                + float(bundle.scaler["mean"])
            )
            if return_roots:
                roots.append(out["roots"].double().numpy().copy())
    return pred, (np.concatenate(roots, axis=0) if return_roots else None)


def root_row_ids(structs: Mapping[int, Any], rows: Sequence[int]) -> np.ndarray:
    parts = [np.full(int(structs[int(r)].n_atoms), int(r), np.int64) for r in rows]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


# ---------------------------------------------------------------------------
# label-blind R2 topology features
# ---------------------------------------------------------------------------
def _bfs_dist(n: int, adj: Sequence[Sequence[int]], source: int) -> np.ndarray:
    dist = np.full(n, -1, np.int16)
    dist[int(source)] = 0
    frontier = [int(source)]
    depth = 0
    while frontier:
        depth += 1
        nxt: list[int] = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = depth
                    nxt.append(int(y))
        frontier = nxt
    return dist


def _count_supports(root: int, allowed: set[int],
                    adj: Sequence[Sequence[int]]) -> tuple[int, int]:
    found: set[tuple[int, ...]] = set()
    stack: list[tuple[int, ...]] = [(int(root),)]
    while stack:
        cur = stack.pop()
        if len(cur) >= 3:
            found.add(tuple(sorted(cur)))
        if len(cur) >= 4:
            continue
        present = set(cur)
        for x in cur:
            for y in adj[x]:
                if y in allowed and y not in present:
                    stack.append(tuple(sorted(cur + (int(y),))))
    c3 = sum(1 for s in found if len(s) == 3)
    c4 = sum(1 for s in found if len(s) == 4)
    return c3, c4


def _components(nodes: set[int], edges: Sequence[tuple[int, int]]) -> int:
    parent = {x: x for x in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return len({find(x) for x in nodes})


def r2_topology(view: Mapping[str, Any]) -> dict[str, Any]:
    n = int(view["n_atoms"])
    bu = np.asarray(view["bond_u"], np.int64)
    bv = np.asarray(view["bond_v"], np.int64)
    adj = _adjacency(n, bu, bv)
    root_rows: list[tuple[int, ...]] = []
    for root in range(n):
        dist = _bfs_dist(n, adj, root)
        allowed = {int(x) for x in np.flatnonzero((dist >= 0) & (dist <= 2))}
        d1 = [int(x) for x in np.flatnonzero(dist == 1)]
        d2 = [int(x) for x in np.flatnonzero(dist == 2)]
        closure_11 = 0
        closure_22 = 0
        induced_edges: list[tuple[int, int]] = []
        for a, b in zip(bu.tolist(), bv.tolist()):
            a = int(a)
            b = int(b)
            if a in allowed and b in allowed:
                induced_edges.append((a, b))
                da, db = int(dist[a]), int(dist[b])
                if da == 1 and db == 1:
                    closure_11 += 1
                elif da == 2 and db == 2:
                    closure_22 += 1
        reconvergence_2 = sum(
            1 for x in d2
            if sum(1 for y in adj[x] if int(dist[y]) == 1) >= 2
        )
        n_nodes = len(allowed)
        n_edges = len(induced_edges)
        cycle_rank = n_edges - n_nodes + _components(allowed, induced_edges)
        supports3, supports4 = _count_supports(root, allowed, adj)
        root_rows.append((closure_11, closure_22, reconvergence_2, n_nodes,
                          n_edges, cycle_rank, supports3, supports4))
    arr = np.asarray(root_rows, np.float64)
    out: dict[str, Any] = {
        "row_id": int(view["row"]),
        "n_atoms": n,
        "n_bonds": int(view["n_bonds"]),
        "n_roots": n,
        "graph_cycle_rank": int(view["graph_cycle_rank"]),
        "graph_n_components": int(view["graph_n_components"]),
        "ring_frac": float(np.mean(np.asarray(view["in_cycle"], np.float64))),
    }
    for j, name in enumerate(ROOT_FEATURES):
        values = arr[:, j]
        out[f"{name}_mean"] = float(values.mean()) if values.size else 0.0
        out[f"{name}_nonzero_frac"] = float(np.mean(values > 0)) if values.size else 0.0
        out[f"{name}_total"] = float(values.sum())
    return out


def topology_rows(bundle: Any, rows: Sequence[int], side: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        view = GD.molecule_view(bundle.raw, int(row))
        rec = r2_topology(view)
        rec["split"] = str(bundle.split["name"])
        rec["side"] = str(side)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# stratum helpers
# ---------------------------------------------------------------------------
def fit_thresholds(fit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def median_nonzero(key: str) -> float | None:
        values = np.asarray([float(r[key]) for r in fit_rows], np.float64)
        values = values[values > 0]
        if values.size == 0:
            return None
        return float(np.median(values))

    ring = np.asarray([float(r["ring_frac"]) for r in fit_rows], np.float64)
    atoms = np.asarray([float(r["n_atoms"]) for r in fit_rows], np.float64)
    return {
        "closure_22_nonzero_median": median_nonzero("closure_22_mean"),
        "reconvergence_2_nonzero_median": median_nonzero("reconvergence_2_mean"),
        "ring_frac_median": float(np.median(ring)),
        "n_atoms_median": float(np.median(atoms)),
        "n_fit": int(len(fit_rows)),
    }


def assign_count_strata(values: np.ndarray, threshold: float | None,
                        prefix: str) -> dict[str, np.ndarray]:
    values = np.asarray(values, np.float64)
    zero = values == 0
    if threshold is None:
        low = np.zeros(values.size, dtype=bool)
        high = np.zeros(values.size, dtype=bool)
    else:
        low = (values > 0) & (values <= float(threshold))
        high = values > float(threshold)
    return {f"{prefix}_zero": zero, f"{prefix}_nz_low": low,
            f"{prefix}_nz_high": high}


def assign_median_strata(values: np.ndarray, threshold: float,
                         prefix: str) -> dict[str, np.ndarray]:
    values = np.asarray(values, np.float64)
    return {f"{prefix}_low": values <= float(threshold),
            f"{prefix}_high": values > float(threshold)}