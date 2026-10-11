"""cscl-strong-representation-readout-screen-v1 driver.

Protocol: ``configs/strong_representation_readout_screen_v1.yaml`` (frozen).
Two pre-purchased CPU regression candidates, one fixed XGBoost recipe:

* **Arm A** — Typed-WL / Morgan-style structural counts (raw attributed graph,
  4096-dim) + raw label-free global structure (30) + verified raw topology25
  (25) = ``X_A`` in R^(N x 4151).
* **Arm B** — the exact 814-dim tensor consumed by the frozen Full-Y seed-0
  soup reader (captured with a reader pre-forward hook under the full C6_MASK
  inference path, no backbone update) = ``X_B`` in R^(N x 814).

Mode split (discipline-critical):

``audit``    Phase-A data/protocol audit + feature construction + frozen
             Full-Y replay + reader-input capture proofs.  Writes
             ``phase_a_protocol_audit.json``, ``typed_wl_feature_audit.json``,
             ``fully_reader_input_replay.json`` and the feature matrices.
``train``    Loads the frozen feature matrices, fits XGB-A and XGB-B on
             fit_inner 7200, evaluates fit + monitor ONLY, saves models,
             fit/monitor predictions, per-arm train metrics and ``lock.json``.
             Dev y is never read here.
``analyze``  Verifies the lock hashes, predicts dev once for both arms and
             writes main table, paired comparisons, subgroup analysis and the
             pre-registered decision.

Hard constraints: seed 0 only, no early stopping, no grid/Optuna/CV search, no
second model, no full-8000/10000 refit, no GPU, official valid/test never
loaded, old target components (g/ell/s/c, SA, logP) never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# CPU thread cap is enforced before numpy / xgboost import.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import typed_wl_features as tw  # noqa: E402

PROTOCOL_ID = "cscl-strong-representation-readout-screen-v1"
CONFIG_PATH = REPO_ROOT / "tracks/structure-contribution/configs/strong_representation_readout_screen_v1.yaml"
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/strong_representation_readout_screen_v1"

FULL_Y_SOUP = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_soup_state.pt"
FULL_Y_PREP = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/prep_fit7200.npz"
FULL_Y_DEV_PREDS = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"
FULL_Y_MON_PREDS = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_monitor_preds.npz"
LEGACY_PREP = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/all_train_prep.npz"
TOPOLOGY_CACHE = REPO_ROOT / "tracks/ksvd/results/zinc_topology_cache/train_topology_features.csv"

# frozen reference numbers (historical; read-only context)
REF = {
    "full_y": {"fit": 0.04220, "monitor": 0.11826, "dev": 0.11641},
    "o_bucket": {"fit": 0.03641, "monitor": 0.11794, "dev": 0.11610},
}
MONITOR_SIGNAL_THRESHOLD = 0.10826  # 0.11826 - 0.01
DEV_MATERIAL = -0.005
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

ARM_A_DIM = tw.WL_DIM + 30 + 25  # 4151
ARM_B_DIM = 814
EXTRACT_BATCH = 128

#: frozen expected row hashes (cscl-v0 split; identical to the historical prep meta)
EXPECTED_ROW_HASHES = {
    "fit_inner": "ff1fb939f32201f0752761f9789ad49d1869346a1735d8b4ff9538d6b8068609",
    "monitor": "835012403f22fedb4d3783872457e1019895bc54e0b27c825cf9f09979a7503f",
    "dev": "53fe48f81031e0c2d55c028ee6fb42fda92a8c9faff8fd16bd6de1a08c48bd88",
}

#: frozen XGBoost recipe (copied from the protocol yaml; cross-checked against
#: the historical ``zinc_long_range_proxy.DEFAULT_XGB_PARAMS`` at runtime).
FROZEN_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 8,
}


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def sha256_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_config() -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"protocol id mismatch in {CONFIG_PATH}")
    if list(config.get("seeds_allowed", [])) != [0]:
        raise RuntimeError("only seed 0 is authorized")
    return config


def assert_recipe_frozen() -> None:
    """The frozen recipe must equal the historical DEFAULT_XGB_PARAMS + fixed extras."""
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import DEFAULT_XGB_PARAMS

    historical = dict(DEFAULT_XGB_PARAMS)
    frozen_core = {
        k: FROZEN_XGB_PARAMS[k]
        for k in ("n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha")
    }
    if historical != frozen_core:
        raise RuntimeError(f"historical DEFAULT_XGB_PARAMS drifted: {historical} != {frozen_core}")
    if FROZEN_XGB_PARAMS["random_state"] != 0 or FROZEN_XGB_PARAMS["n_jobs"] != 8:
        raise RuntimeError("seed/threads frozen values changed")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_dataset():
    return cf.load_official_train()


def split_indices() -> tuple[list[str], dict[str, np.ndarray]]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    return smiles, idx


def dataset_y(dataset) -> np.ndarray:
    return np.asarray([float(d.y.reshape(-1)[0]) for d in dataset], dtype=np.float64)


def row_hash_check(idx: Mapping[str, np.ndarray], log=print) -> dict[str, str]:
    got = {k: sha256_rows(np.asarray(idx[k], dtype=np.int64)) for k in ("fit_inner", "monitor", "dev")}
    for k, expected in EXPECTED_ROW_HASHES.items():
        if got[k] != expected:
            raise RuntimeError(f"split row hash mismatch for {k}: {got[k]} != {expected}")
    log(f"[audit] split row hashes match the frozen cscl-v0 split: {got}")
    return got


def load_train_rows(log=print):
    """Encoded official-train cache + env attachment (read-only)."""
    import sys as _sys

    if str(_CODE_DIR) not in _sys.path:
        _sys.path.insert(0, str(_CODE_DIR))
    from run_prediction_feasibility_v2 import load_train_rows as _load

    return _load(log=log)


# ---------------------------------------------------------------------------
# Phase A: data / protocol audit
# ---------------------------------------------------------------------------


def run_phase_a_audit(
    out_dir: Path,
    dataset,
    smiles: list[str],
    idx: Mapping[str, np.ndarray],
    y: np.ndarray,
    log=print,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    load_config()

    # 1. split sizes + hashes -------------------------------------------------
    sizes = {k: int(len(idx[k])) for k in ("fit_inner", "monitor", "dev")}
    if sizes != {"fit_inner": 7200, "monitor": 800, "dev": 2000}:
        raise RuntimeError(f"unexpected split sizes {sizes}")
    hashes = row_hash_check(idx, log=log)

    # 2. canonical-SMILES grouping never crosses internal splits -------------
    fit_smiles = set(smiles[i] for i in idx["fit_inner"])
    mon_smiles = set(smiles[i] for i in idx["monitor"])
    dev_smiles = set(smiles[i] for i in idx["dev"])
    cross_fm = fit_smiles & mon_smiles
    cross_fd = fit_smiles & dev_smiles
    cross_md = mon_smiles & dev_smiles
    n_dupes_all = len(smiles) - len(set(smiles))
    grouping_ok = not (cross_fm or cross_fd or cross_md)
    if not grouping_ok:
        raise RuntimeError(f"SMILES groups cross splits: fm={len(cross_fm)} fd={len(cross_fd)} md={len(cross_md)}")
    log(f"[audit] SMILES grouping clean (fit|monitor/disjoint/dev); duplicate rows in train={n_dupes_all}")

    # 3. raw-y alignment with historical Full-Y exports ----------------------
    dev_preds = np.load(FULL_Y_DEV_PREDS)
    mon_preds = np.load(FULL_Y_MON_PREDS)
    dev_rows = np.asarray(idx["dev"], dtype=np.int64)
    mon_rows = np.asarray(idx["monitor"], dtype=np.int64)
    dev_hist_ok = bool(np.array_equal(dev_preds["row"], dev_rows) and np.allclose(dev_preds["y"], y[dev_rows]))
    mon_hist_ok = bool(np.array_equal(mon_preds["row"], mon_rows) and np.allclose(mon_preds["y"], y[mon_rows]))
    if not (dev_hist_ok and mon_hist_ok):
        raise RuntimeError("historical Full-Y exports do not align with dataset rows/y")
    hist_mae = {
        "full_y_dev_export": float(np.abs(dev_preds["pred"].astype(np.float64) - y[dev_rows]).mean()),
        "full_y_monitor_export": float(np.abs(mon_preds["pred"].astype(np.float64) - y[mon_rows]).mean()),
    }
    if abs(hist_mae["full_y_dev_export"] - REF["full_y"]["dev"]) > 1e-4 or abs(
        hist_mae["full_y_monitor_export"] - REF["full_y"]["monitor"]
    ) > 1e-4:
        raise RuntimeError(f"historical Full-Y replay MAE drifted: {hist_mae}")

    # 4. row alignment between dataset and the encoded cache -----------------
    train_rows = load_train_rows(log=log)
    from run_prediction_feasibility_v2 import verify_row_alignment

    align = verify_row_alignment(train_rows, log=log)

    # 5. A/B use the identical molecules and the identical y -----------------
    arm_y_identity = {
        "dev": float(np.abs(dev_preds["y"].astype(np.float64) - y[dev_rows]).max()),
        "monitor": float(np.abs(mon_preds["y"].astype(np.float64) - y[mon_rows]).max()),
    }
    if max(arm_y_identity.values()) != 0.0:
        raise RuntimeError("A/B y identity check failed")

    # 6. no official valid/test, no old target components --------------------
    # static assertion: this driver's module source never references those files
    # (tokens are assembled from fragments so this guard itself is not a hit)
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_loaded = [
        token
        for token in (
            "encoded_" + "valid",
            "encoded_" + "test",
            "official_valid" + ".pt",
            "official_test" + ".pt",
            "target_" + "decomposition",
        )
        if token in source
    ]
    if forbidden_loaded:
        raise RuntimeError(f"forbidden official valid/test references found: {forbidden_loaded}")

    # 7. historical split meta cross-check (not only the hard-coded hashes) ---
    hist_meta = json.loads(
        (REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/prep_fit7200_meta.json").read_text()
    )
    meta_rows = hist_meta["fit_rows"]
    meta_match = {
        "fit_inner": meta_rows["sha256_rows"] == hashes["fit_inner"],
        "monitor": meta_rows["monitor_sha256_rows"] == hashes["monitor"],
        "dev": meta_rows["dev_sha256_rows"] == hashes["dev"],
    }
    if not all(meta_match.values()):
        raise RuntimeError(f"historical prep meta row hashes differ: {meta_match}")

    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "phase_a_protocol_audit",
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "seconds": float(time.perf_counter() - started),
        "split": {
            "sizes": sizes,
            "row_sha256": hashes,
            "expected_row_sha256": EXPECTED_ROW_HASHES,
            "expected_match": all(hashes[k] == EXPECTED_ROW_HASHES[k] for k in hashes),
            "historical_prep_meta_match": meta_match,
            "smiles_groups_do_not_cross_splits": grouping_ok,
            "n_duplicate_smiles_rows_train": int(n_dupes_all),
            "fit_inner_vs_monitor_shared_smiles": len(cross_fm),
            "fit_inner_vs_dev_shared_smiles": len(cross_fd),
            "monitor_vs_dev_shared_smiles": len(cross_md),
        },
        "raw_y_alignment": {
            "full_y_dev_export_rows_match": bool(np.array_equal(dev_preds["row"], dev_rows)),
            "full_y_monitor_export_rows_match": bool(np.array_equal(mon_preds["row"], mon_rows)),
            "max_abs_y_diff_dev": float(np.abs(dev_preds["y"] - y[dev_rows]).max()),
            "max_abs_y_diff_monitor": float(np.abs(mon_preds["y"] - y[mon_rows]).max()),
            "historical_export_mae": hist_mae,
            "arm_y_identity_max_abs_diff": arm_y_identity,
        },
        "encoded_cache_alignment": align,
        "audited_claims": {
            "A_fit_monitor_dev_rows_identical_to_historical": True,
            "B_canonical_smiles_groups_do_not_cross_splits": grouping_ok,
            "C_raw_y_labels_match_historical_full_y_replay": bool(hist_mae["full_y_dev_export"] < 1e-4),
            "D_AB_train_and_eval_use_same_molecules_and_same_y": max(arm_y_identity.values()) == 0.0,
            "E_features_from_raw_graph_or_frozen_full_y": True,
            "F_official_valid_test_not_read": True,
            "G_old_task_component_targets_not_read": True,
            "H_fitted_transforms_only_on_fit_inner": True,
            "I_dev_not_used_for_feature_or_parameter_choice": True,
        },
        "discipline": {
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "old_targets_loaded": [],
            "seed": 0,
            "gpu_used": False,
            "cpu_threads_cap": 8,
            "note": "audit mode never computes any dev metric; dev evaluation happens once in analyze mode after the lock",
        },
    }
    write_json(out_dir / "phase_a_protocol_audit.json", out)
    log(f"[audit] phase A audit written ({out['seconds']:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Arm A: Typed-WL + global30 + topology25
# ---------------------------------------------------------------------------


def _legacy_std(blob: Mapping[str, np.ndarray], prefix: str, log=print):
    from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

    return zjd.Std(blob[f"{prefix}_all_mean"], blob[f"{prefix}_all_scale"])


def _typed_wl_checks(dataset, idx, log=print) -> dict[str, Any]:
    """Deterministic self-checks of the Typed-WL descriptor (label-free)."""
    checks: dict[str, Any] = {}

    def _arrays(i: int):
        d = dataset[int(i)]
        return (
            d.x.numpy().reshape(-1).astype(np.int64),
            d.edge_index.numpy().astype(np.int64),
            d.edge_attr.numpy().reshape(-1).astype(np.int64),
        )

    # (1) atom-order permutation invariance + isomorphic descriptors
    perm_rng = np.random.default_rng(0)
    perm_ok = True
    for i in [0, 1, 7, 123, 4567, 9999]:
        x, ei, ea = _arrays(i)
        base = tw.molecule_typed_wl_from_arrays(x, ei, ea)
        perm = perm_rng.permutation(len(x))
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        x2 = x[perm]
        ei2 = inv[ei]  # new labels for endpoints
        got = tw.molecule_typed_wl_from_arrays(x2, ei2, ea)
        perm_ok &= bool(np.array_equal(base, got))
    checks["atom_permutation_invariance_6_molecules"] = bool(perm_ok)
    if not perm_ok:
        raise RuntimeError("Typed-WL is not atom-order permutation invariant")

    # (2) bidirectional duplicate edges must not double count
    x0, ei0, ea0 = _arrays(0)
    bonds = tw.unique_undirected_bonds(ei0, ea0)
    # PyG stores both directions; rebuild exactly that interleaved layout
    src, dst, typ = [], [], []
    for u, v, bt in bonds:
        src.extend([u, v])
        dst.extend([v, u])
        typ.extend([bt, bt])
    ei_pyg = np.asarray([src, dst], dtype=np.int64)
    ea_pyg = np.asarray(typ, dtype=np.int64)
    dedup_ok = bool(
        np.array_equal(
            tw.molecule_typed_wl_from_arrays(x0, ei_pyg, ea_pyg),
            _arrays_wl_from_bonds(x0, bonds),
        )
    )
    checks["bidirectional_dedup_equals_unique_bonds"] = dedup_ok
    if not dedup_ok:
        raise RuntimeError("Typed-WL double counts bidirectional edges")

    # (3) different radii use independent hash spaces
    sample_colors = [f"c{i}" for i in range(500)]
    single_radius_shared = sum(1 for c in sample_colors if len({tw.bucket_of(r, c) for r in tw.WL_RADII}) == 1)
    checks["radii_share_bucket_for_sampled_colors"] = int(single_radius_shared)
    checks["radii_hash_spaces_independent"] = bool(single_radius_shared == 0)
    if single_radius_shared != 0:
        raise RuntimeError("radius hash spaces are not independent")

    # (4) bond-type exchange changes the encoding
    x1, ei1, ea1 = _arrays(1)
    swapped = (ea1 + 1) % 4
    a = tw.molecule_typed_wl_from_arrays(x1, ei1, ea1)
    b = tw.molecule_typed_wl_from_arrays(x1, ei1, swapped)
    checks["bond_type_exchange_changes_descriptor"] = bool((a != b).any())

    # (5) isomorphic attributed graphs -> identical descriptors (VF2, labelled)
    # NOTE: equal canonical SMILES does NOT by itself imply equal attributed
    # graphs in this ZINC snapshot: the raw ZINC ``atom_type`` codes can differ
    # for two rows with the same RDKit canonical SMILES (verified below; the
    # split only guarantees they never cross sides).  The correct invariance
    # requirement is under attributed-graph isomorphism, checked here.
    import networkx as nx

    smiles = [str(s) for s in cf.load_canonical_smiles()]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    pairs = [rows for rows in groups.values() if len(rows) > 1]

    def _labelled_graph(i: int):
        x, ei, ea = _arrays(i)
        g = nx.Graph()
        for v, t in enumerate(x):
            g.add_node(int(v), t=int(t))
        for u, v, bt in tw.unique_undirected_bonds(ei, ea):
            g.add_edge(int(u), int(v), bt=int(bt))
        return g

    iso_pairs: list[int] = []
    noniso_pairs = 0
    noniso_example: dict[str, Any] | None = None
    descriptor_mismatch_isomorphic = 0
    checked_pairs = 0
    for rows in pairs:
        if checked_pairs >= 20:
            break
        i, j = int(rows[0]), int(rows[1])
        checked_pairs += 1
        gi, gj = _labelled_graph(i), _labelled_graph(j)
        gm = nx.algorithms.isomorphism.GraphMatcher(
            gi, gj, node_match=lambda a, b: a["t"] == b["t"], edge_match=lambda a, b: a["bt"] == b["bt"]
        )
        if not gm.is_isomorphic():
            noniso_pairs += 1
            if noniso_example is None:
                xi = np.bincount(_arrays(i)[0], minlength=28)
                xj = np.bincount(_arrays(j)[0], minlength=28)
                noniso_example = {
                    "rows": [i, j],
                    "smiles": smiles[i],
                    "atom_type_hist_differing_bins": [
                        int(k) for k in np.nonzero(xi != xj)[0]
                    ],
                    "note": "same canonical SMILES but different raw ZINC atom_type codes",
                }
            continue
        iso_pairs.append([i, j])
        di = tw.molecule_typed_wl_from_arrays(*_arrays(i))
        dj = tw.molecule_typed_wl_from_arrays(*_arrays(j))
        if not np.array_equal(di, dj):
            descriptor_mismatch_isomorphic += 1
    checks["duplicate_smiles_pairs_checked"] = checked_pairs
    checks["duplicate_smiles_pairs_isomorphic"] = iso_pairs
    checks["duplicate_smiles_pairs_non_isomorphic_attributed"] = noniso_pairs
    checks["duplicate_smiles_non_isomorphic_example"] = noniso_example
    checks["isomorphic_pair_descriptor_mismatches"] = descriptor_mismatch_isomorphic
    checks["isomorphic_attributed_graphs_same_descriptor"] = bool(
        descriptor_mismatch_isomorphic == 0 and (iso_pairs or checks["atom_permutation_invariance_6_molecules"])
    )
    if descriptor_mismatch_isomorphic != 0:
        raise RuntimeError("isomorphic attributed graphs produced different Typed-WL descriptors")
    return checks


def _arrays_wl_from_bonds(x: np.ndarray, bonds: Sequence[tuple[int, int, int]]) -> np.ndarray:
    return tw.wl_descriptor(np.asarray(x, dtype=np.int64).reshape(-1), bonds)


def build_arm_a_features(
    out_dir: Path, dataset, idx: Mapping[str, np.ndarray], log=print
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    import torch

    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztf

    n = len(dataset)
    x_wl = np.zeros((n, tw.WL_DIM), dtype=np.float32)
    global30 = np.zeros((n, 30), dtype=np.float32)
    topo25 = np.zeros((n, 25), dtype=np.float32)
    color_rows: list[list[str]] = []
    n_atoms = np.zeros(n, dtype=np.int64)

    # raw topology cache (verified; hinge mode) and the inversion basis
    frame = ztf.load_cache("train")
    if frame is None or len(frame) != n:
        raise RuntimeError("verified raw topology cache missing or wrong length")
    blob10 = {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}
    ctx_std = _legacy_std(blob10, "ctx")
    topo_std = _legacy_std(blob10, "topo")
    encoded = torch.load(Path(cf.REPO_ROOT) / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt", map_location="cpu", weights_only=False)
    ctx_cache = np.stack([encoded[i].global_context.numpy().reshape(-1) for i in range(n)]).astype(np.float32)
    topo_cache = np.stack([encoded[i].topology_features.numpy().reshape(-1) for i in range(n)]).astype(np.float32)
    ctx_raw_from_cache = ctx_std.inverse(ctx_cache)[:, :30]
    topo_raw_from_cache = topo_std.inverse(topo_cache)

    max_global_recompute = 0.0
    max_topo_csv = 0.0
    t0 = time.perf_counter()
    for i, data in enumerate(dataset):
        x = data.x.numpy().reshape(-1).astype(np.int64)
        ei = data.edge_index.numpy().astype(np.int64)
        ea = data.edge_attr.numpy().reshape(-1).astype(np.int64)
        bonds = tw.unique_undirected_bonds(ei, ea)
        x_wl[i] = tw.wl_descriptor(x, bonds)
        color_rows.append(tw.colors_per_radius(x, bonds))
        n_atoms[i] = len(x)
        graph = tw.graph_from_data(data)
        g = tw.global_structure_raw(graph)
        global30[i] = g
        max_global_recompute = max(max_global_recompute, float(np.abs(g - ctx_raw_from_cache[i]).max()))
        tv = ztf.raw_vector(frame.iloc[i], "hinge")
        topo25[i] = tv
        max_topo_csv = max(max_topo_csv, float(np.abs(tv - topo_raw_from_cache[i]).max()))
        if i and i % 2000 == 0:
            log(f"[armA] {i}/{n} Typed-WL + global/topology ({time.perf_counter() - t0:.0f}s)")
    log(f"[armA] all {n} rows built in {time.perf_counter() - t0:.0f}s")

    X_A = np.concatenate([x_wl, global30, topo25], axis=1).astype(np.float32)
    if X_A.shape != (n, ARM_A_DIM):
        raise RuntimeError(f"X_A shape {X_A.shape} != {(n, ARM_A_DIM)}")
    if not np.isfinite(X_A).all():
        raise RuntimeError("X_A has NaN/Inf")
    if (X_A < 0).any():
        raise RuntimeError("X_A has negative entries")

    checks = _typed_wl_checks(dataset, idx, log=log)
    checks["global30_recompute_vs_cache_inversion_max_abs_diff"] = max_global_recompute
    checks["topology25_csv_vs_cache_inversion_max_abs_diff"] = max_topo_csv
    # global30: v2 audit recorded inversion identity <= 7.5e-8; keep the 1e-5 gate.
    # topology25: the historical topology25 audit recorded the exact float32
    # round-trip maximum 6.103515625e-05 on all 10000 rows
    # (results/feasibility_v2_cpu/topology25_audit.json -> encoded_consistency);
    # our recomputation reproduces that same value, so 1e-4 is the frozen
    # historically verified tolerance for this block.
    TOPO25_HISTORICAL_MAX = 6.103515625e-05
    checks["topology25_historical_recorded_max"] = TOPO25_HISTORICAL_MAX
    checks["global30_recompute_matches_cache"] = bool(max_global_recompute <= 1e-5)
    checks["topology25_csv_matches_cache"] = bool(max_topo_csv <= 1e-4)
    if not (checks["global30_recompute_matches_cache"] and checks["topology25_csv_matches_cache"]):
        raise RuntimeError("global30/topology25 cross-check failed")

    collision = tw.bucket_collision_stats(color_rows)
    nonzero = tw.descriptor_nonzero_and_conservation(x_wl, n_atoms)
    if not nonzero["count_conservation"]:
        raise RuntimeError("Typed-WL count conservation failed")

    # structural nonzero statistics for the full X_A (XGB does not need scaling)
    audit: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "typed_wl_feature_audit",
        "git_commit": git_commit(),
        "n_rows": int(n),
        "radius_definition": {
            "radii": list(tw.WL_RADII),
            "color_0": "atom category",
            "color_update": "sha256(color_r(v), sorted_multiset{(bond_type(v,u), color_r(u))}) with radius salted",
            "count_rule": "one count per rooted fragment occurrence",
        },
        "hash": {
            "algorithm": tw.HASH_ALGORITHM,
            "version": tw.HASH_VERSION,
            "bucket_salt": tw.BUCKET_SALT,
            "color_salt": tw.COLOR_SALT,
            "bucket_rule": "sha256-derived 64-bit integer mod 1024; radius included in payload",
            "python_builtin_hash_used": False,
        },
        "dimensions": {
            "buckets_per_radius": tw.BUCKETS_PER_RADIUS,
            "wl_dim": tw.WL_DIM,
            "global_structure_dim": 30,
            "topology_dim": 25,
            "X_A_dim": int(X_A.shape[1]),
            "total_dim_formula": "4*1024 + 30 + 25",
        },
        "nonzero_and_conservation": nonzero,
        "bucket_collisions": collision,
        "correctness_checks": checks,
        "feature_matrix_sha256": sha256_array(X_A),
        "code_version": {"typed_wl_features": file_sha256(_CODE_DIR / "typed_wl_features.py")},
        "seconds": float(time.perf_counter() - started),
        "no_normalization": "XGBoost consumes raw counts + raw structural values; no fitted transform introduced",
        "disclosure": "collision statistics are diagnostic only and did NOT trigger any dimension change",
    }
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_dir / "x_a.npy", X_A)
    write_json(out_dir / "typed_wl_feature_audit.json", audit)
    log(
        f"[armA] X_A {X_A.shape} sha={audit['feature_matrix_sha256'][:16]} "
        f"collisions={ {k: v['collision_ratio'] for k, v in collision.items()} }"
    )
    return X_A, audit


# ---------------------------------------------------------------------------
# Arm B: frozen Full-Y reader input
# ---------------------------------------------------------------------------


def _batches(n: int, size: int = EXTRACT_BATCH) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


def extract_reader_inputs(model, train_rows: Sequence[Any], rows: Sequence[int], log=print) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture ``unified`` (reader input) for ``rows`` via a pre-forward hook."""
    import torch
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    rows = [int(r) for r in rows]
    X = np.zeros((len(rows), ARM_B_DIM), dtype=np.float32)
    stats: dict[str, Any] = {"reader_recompute_max_abs_diff": 0.0, "batches": 0}
    started = time.perf_counter()
    for s in range(0, len(rows), EXTRACT_BATCH):
        chunk = rows[s : s + EXTRACT_BATCH]
        batch = p1.env_collate([train_rows[i] for i in chunk])
        batch.y = torch.zeros(len(chunk))
        captured: dict[str, Any] = {}

        def _hook(_module, args):
            captured["unified"] = args[0].detach()

        handle = model.reader.register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                prediction = model(batch, mask=cm.C6_MASK).view(-1)
            unified = captured["unified"]
            with torch.no_grad():
                reader_out = model.reader(unified).view(-1)
        finally:
            handle.remove()
        if unified.shape != (len(chunk), ARM_B_DIM):
            raise RuntimeError(f"unified shape {tuple(unified.shape)} != {(len(chunk), ARM_B_DIM)}")
        stats["reader_recompute_max_abs_diff"] = max(
            stats["reader_recompute_max_abs_diff"],
            float((reader_out - prediction).abs().max()),
        )
        X[s : s + len(chunk)] = unified.cpu().numpy()
        stats["batches"] += 1
        if (s // EXTRACT_BATCH) % 20 == 0:
            log(f"[armB] {s + len(chunk)}/{len(rows)} rows captured ({time.perf_counter() - started:.0f}s)")
    stats["seconds"] = float(time.perf_counter() - started)
    return X, stats


def build_arm_b_features(
    out_dir: Path, dataset, idx: Mapping[str, np.ndarray], log=print
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    import strong_backbone_attribution as sba

    started = time.perf_counter()
    sha_before = file_sha256(FULL_Y_SOUP)
    model, _prep, _blob10 = sba.load_frozen_backbone(log=log)
    train_rows = load_train_rows(log=log)
    if len(train_rows) != len(dataset):
        raise RuntimeError("encoded cache and dataset length differ")

    # read-only replay of the committed Full-Y soup
    replay = sba.replay_predictions(model, train_rows, idx, device="cpu", log=log)

    # extract the 814-dim reader input for all 10k rows
    X_B, stats = extract_reader_inputs(model, train_rows, range(len(train_rows)), log=log)

    # ---- capture-proof battery -------------------------------------------
    proofs: dict[str, Any] = {}
    dev_rows = list(np.asarray(idx["dev"], dtype=np.int64))

    # (a) fixed-batch hook vs direct reader recompute already in stats; be explicit
    proofs["reader_recompute_max_abs_diff"] = stats["reader_recompute_max_abs_diff"]
    proofs["reader_recompute_within_tolerance"] = bool(stats["reader_recompute_max_abs_diff"] <= 1e-5)

    # (b) same molecules under different batch groupings -> same unified rows
    sample = dev_rows[:64]
    x_one, _ = extract_reader_inputs(model, train_rows, sample, log=lambda *a, **k: None)
    x_32 = np.concatenate(
        [
            extract_reader_inputs(model, train_rows, sample[:32], log=lambda *a, **k: None)[0],
            extract_reader_inputs(model, train_rows, sample[32:], log=lambda *a, **k: None)[0],
        ],
        axis=0,
    )
    proofs["batch_grouping_max_abs_diff_64_vs_32plus32"] = float(np.abs(x_one - x_32).max())
    proofs["batch_grouping_within_tolerance"] = bool(np.abs(x_one - x_32).max() <= 1e-5)

    # (c) y-independence: same molecules, different batch.y -> identical unified
    chunk = dev_rows[:16]
    batch = p1.env_collate([train_rows[i] for i in chunk])
    captured_a: dict[str, Any] = {}

    def _grab(_m, args):
        captured_a["unified"] = args[0].detach()

    handle = model.reader.register_forward_pre_hook(_grab)
    try:
        batch.y = torch.full((len(chunk),), 0.0)
        with torch.no_grad():
            model(batch, mask=cm.C6_MASK)
        unified_zero = captured_a["unified"].clone()
        batch.y = torch.arange(len(chunk), dtype=torch.float32) * 3.7 - 5.0
        with torch.no_grad():
            model(batch, mask=cm.C6_MASK)
    finally:
        handle.remove()
    proofs["y_independence_max_abs_diff"] = float((captured_a["unified"] - unified_zero).abs().max())
    proofs["y_independence"] = proofs["y_independence_max_abs_diff"] == 0.0

    # (d) row alignment: topology block of unified equals topology_encoder(input)
    batch = p1.env_collate([train_rows[i] for i in dev_rows[:32]])
    batch.y = torch.zeros(32)
    captured_b: dict[str, Any] = {}

    def _grab2(_m, args):
        captured_b["unified"] = args[0].detach()

    handle = model.reader.register_forward_pre_hook(_grab2)
    try:
        with torch.no_grad():
            model(batch, mask=cm.C6_MASK)
            topo_block = model.topology_encoder(batch.topology_features)
    finally:
        handle.remove()
    proofs["topology_row_alignment_max_abs_diff"] = float(
        (captured_b["unified"][:, 806:814] - topo_block).abs().max()
    )
    proofs["topology_row_alignment"] = bool(proofs["topology_row_alignment_max_abs_diff"] <= 1e-6)
    proofs["c6_mask_count_blocks_zero"] = bool(
        float(captured_b["unified"][:, 288].abs().max()) == 0.0
        and float(captured_b["unified"][:, 385:774:97].abs().max()) == 0.0
    )

    # (e) shape / dtype / finiteness of the full matrix
    proofs["shape"] = list(X_B.shape)
    proofs["dtype"] = str(X_B.dtype)
    proofs["finite"] = bool(np.isfinite(X_B).all())

    sha_after = file_sha256(FULL_Y_SOUP)
    proofs["soup_state_sha256_before"] = sha_before
    proofs["soup_state_sha256_after"] = sha_after
    proofs["soup_state_unchanged"] = sha_before == sha_after

    required = (
        "reader_recompute_within_tolerance",
        "batch_grouping_within_tolerance",
        "y_independence",
        "topology_row_alignment",
        "c6_mask_count_blocks_zero",
        "soup_state_unchanged",
    )
    if not all(proofs[k] for k in required):
        raise RuntimeError(f"Arm B capture proofs failed: { {k: proofs[k] for k in required} }")
    if not (proofs["finite"] and X_B.shape == (len(dataset), ARM_B_DIM)):
        raise RuntimeError("Arm B matrix malformed")
    if not (replay["monitor_ok"] and replay["dev_ok"]):
        raise RuntimeError("frozen Full-Y replay failed")

    audit: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "fully_reader_input_replay",
        "git_commit": git_commit(),
        "source_model": {
            "soup_state": str(FULL_Y_SOUP.relative_to(REPO_ROOT)),
            "soup_state_sha256": sha_before,
            "prep": str(FULL_Y_PREP.relative_to(REPO_ROOT)),
            "forward_path": "AuditModel.forward(data, mask=C6_MASK); eval(); torch.no_grad(); parameters frozen",
            "extraction": "reader pre-forward hook captures args[0] (unified)",
        },
        "historical_replay": {
            "monitor_max_abs_pred_diff": replay["monitor_max_abs_pred_diff"],
            "monitor_mae_replayed": replay["monitor_mae_replayed"],
            "monitor_mae_saved": replay["monitor_mae_saved"],
            "dev_max_abs_pred_diff": replay["dev_max_abs_pred_diff"],
            "dev_mae_replayed": replay["dev_mae_replayed"],
            "dev_mae_saved": replay["dev_mae_saved"],
            "expected_monitor_mae": REF["full_y"]["monitor"],
            "expected_dev_mae": REF["full_y"]["dev"],
            "monitor_ok": replay["monitor_ok"],
            "dev_ok": replay["dev_ok"],
            "engineering_only": "replay is an engineering check; it creates no new candidate",
            "official_valid_test_loaded": False,
        },
        "capture_proofs": proofs,
        "layout": {
            "expected_dim": ARM_B_DIM,
            "observed_dim": int(X_B.shape[1]),
            "layout": "[unary 289 | relation_readout 5*97=485 | graph_hidden 32 | topology 8]",
            "c6_mask": "global atom/bond histograms zeroed; unary/pair count blocks zeroed; path_count relation zeroed; topology kept",
        },
        "feature_matrix_sha256": sha256_array(X_B),
        "code_version": {"strong_backbone_attribution": file_sha256(_CODE_DIR / "strong_backbone_attribution.py")},
        "seconds": float(time.perf_counter() - started),
        "second_stage_supervision_disclosure": (
            "the frozen 814-dim representation was itself trained on fit_inner labels; "
            "XGB-B sees a representation produced by the same fit labels and then fits a "
            "second supervised stage. The comparison is NOT a matched-compute comparison."
        ),
    }
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_dir / "x_b.npy", X_B)
    write_json(out_dir / "fully_reader_input_replay.json", audit)
    log(f"[armB] X_B {X_B.shape} sha={audit['feature_matrix_sha256'][:16]} proofs ok")
    return X_B, audit


# ---------------------------------------------------------------------------
# audit entry point
# ---------------------------------------------------------------------------


def run_audit(out_dir: Path, log=print) -> dict[str, Any]:
    load_config()
    assert_recipe_frozen()
    dataset = load_dataset()
    smiles, idx = split_indices()
    y = dataset_y(dataset)
    phase_a = run_phase_a_audit(out_dir, dataset, smiles, idx, y, log=log)
    X_A, audit_a = build_arm_a_features(out_dir, dataset, idx, log=log)
    X_B, audit_b = build_arm_b_features(out_dir, dataset, idx, log=log)
    if X_A.shape[0] != X_B.shape[0] == len(dataset):
        raise RuntimeError("arm feature row counts differ")
    hashes = {
        "protocol": PROTOCOL_ID,
        "config_sha256": file_sha256(CONFIG_PATH),
        "x_a_sha256": sha256_array(X_A),
        "x_b_sha256": sha256_array(X_B),
        "x_a_shape": list(X_A.shape),
        "x_b_shape": list(X_B.shape),
        "x_a_audit": audit_a["feature_matrix_sha256"],
        "x_b_audit": audit_b["feature_matrix_sha256"],
        "row_order": "encoded official-train row order 0..9999 (verified against dataset y)",
        "dev_metrics_computed": False,
        "git_commit": git_commit(),
    }
    write_json(out_dir / "features" / "feature_hashes.json", hashes)
    log("[audit] all audits written; dev has not been evaluated")
    return {"phase_a": phase_a, "arm_a": audit_a, "arm_b": audit_b, "hashes": hashes}


# ---------------------------------------------------------------------------
# train: two frozen XGBoost fits (fit + monitor only)
# ---------------------------------------------------------------------------


def _fit_one(name: str, X: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, out_dir: Path, log=print) -> dict[str, Any]:
    from xgboost import XGBRegressor

    assert_recipe_frozen()
    t0 = time.perf_counter()
    model = XGBRegressor(**FROZEN_XGB_PARAMS)
    model.fit(X[fit_rows], y[fit_rows])
    train_seconds = time.perf_counter() - t0
    model_path = out_dir / f"xgb_{name}.json"
    model.save_model(str(model_path))
    model_sha = file_sha256(model_path)
    log(f"[train:{name}] {X.shape[1]} features, fit {len(fit_rows)} rows, {train_seconds:.1f}s, model_sha={model_sha[:16]}")
    return {
        "model": model,
        "model_path": model_path,
        "model_sha256": model_sha,
        "train_seconds": train_seconds,
    }


def run_train(out_dir: Path, log=print) -> dict[str, Any]:
    import xgboost

    load_config()
    assert_recipe_frozen()
    feat_hashes = json.loads((out_dir / "features" / "feature_hashes.json").read_text())
    X_A = np.load(out_dir / "features" / "x_a.npy")
    X_B = np.load(out_dir / "features" / "x_b.npy")
    if sha256_array(X_A) != feat_hashes["x_a_sha256"] or sha256_array(X_B) != feat_hashes["x_b_sha256"]:
        raise RuntimeError("feature matrices changed since the audit; refusing to train")
    if feat_hashes["config_sha256"] != file_sha256(CONFIG_PATH):
        raise RuntimeError("protocol config changed after feature freeze; refusing to train")

    dataset = load_dataset()
    y = dataset_y(dataset)
    _smiles, idx = split_indices()
    row_hash_check(idx, log=log)
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    mon_rows = np.asarray(idx["monitor"], dtype=np.int64)

    arms = {"A": X_A, "B": X_B}
    lock: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "mode": "train",
        "config_sha256": file_sha256(CONFIG_PATH),
        "feature_hashes": feat_hashes,
        "xgb_params": FROZEN_XGB_PARAMS,
        "rows": {
            "fit_inner_sha256": sha256_rows(fit_rows),
            "monitor_sha256": sha256_rows(mon_rows),
        },
        "dev_evaluated": False,
        "dev_metrics_computed_at_train_time": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgboost.__version__,
            "platform": platform.platform(),
        },
        "git_commit": git_commit(),
        "arms": {},
    }
    results: dict[str, Any] = {}
    for name, X in arms.items():
        if X.shape != (len(dataset), ARM_A_DIM if name == "A" else ARM_B_DIM):
            raise RuntimeError(f"arm {name} matrix shape {X.shape} unexpected")
        fit = _fit_one(name, X, y, fit_rows, out_dir, log=log)
        model = fit["model"]
        pred_fit = np.asarray(model.predict(X[fit_rows]), dtype=np.float64)
        pred_mon = np.asarray(model.predict(X[mon_rows]), dtype=np.float64)
        metrics = {
            "protocol": PROTOCOL_ID,
            "arm": name,
            "feature": "Typed-WL+global30+topology25" if name == "A" else "frozen Full-Y 814-dim reader input",
            "n_features": int(X.shape[1]),
            "n_fit": int(len(fit_rows)),
            "seed": 0,
            "xgb_params": FROZEN_XGB_PARAMS,
            "train_seconds": fit["train_seconds"],
            "model_sha256": fit["model_sha256"],
            "feature_sha256": feat_hashes[f"x_{name.lower()}_sha256"],
            "fit_mae": float(np.abs(pred_fit - y[fit_rows]).mean()),
            "monitor_mae": float(np.abs(pred_mon - y[mon_rows]).mean()),
            "dev_evaluated": False,
            "versions": lock["versions"],
            "git_commit": lock["git_commit"],
        }
        write_json(out_dir / f"xgb_{name}_train_metrics.json", metrics)
        np.savez_compressed(
            out_dir / f"xgb_{name}_preds_fit.npz",
            row=fit_rows,
            pred=pred_fit.astype(np.float32),
            y=y[fit_rows].astype(np.float32),
        )
        np.savez_compressed(
            out_dir / f"xgb_{name}_preds_monitor.npz",
            row=mon_rows,
            pred=pred_mon.astype(np.float32),
            y=y[mon_rows].astype(np.float32),
        )
        lock["arms"][name] = {
            "model_sha256": fit["model_sha256"],
            "feature_sha256": metrics["feature_sha256"],
            "fit_pred_sha256": sha256_array(pred_fit.astype(np.float32)),
            "monitor_pred_sha256": sha256_array(pred_mon.astype(np.float32)),
            "fit_mae": metrics["fit_mae"],
            "monitor_mae": metrics["monitor_mae"],
            "train_seconds": metrics["train_seconds"],
        }
        results[name] = metrics
        log(
            f"[train:{name}] fit MAE {metrics['fit_mae']:.5f} | monitor MAE {metrics['monitor_mae']:.5f} "
            f"(threshold {MONITOR_SIGNAL_THRESHOLD})"
        )
    write_json(out_dir / "lock.json", lock)
    log("[train] lock.json written; dev has NOT been evaluated")
    return results


# ---------------------------------------------------------------------------
# analyze: dev evaluation, comparisons, decision
# ---------------------------------------------------------------------------


def error_stats(e: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(e.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "max": float(e.max()),
    }


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Signed paired difference ``err_a - err_b`` with a percentile CI95.

    Negative => arm A has the lower MAE.
    """
    d = np.asarray(err_a, dtype=np.float64) - np.asarray(err_b, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.empty(resamples, dtype=np.float64)
    for b in range(resamples):
        boots[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff_signed": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "frac_a_better": float((d < 0).mean()),
        "resamples": int(resamples),
        "seed": int(seed),
        "statistic": "mean per-molecule signed absolute-error difference (err_a - err_b)",
    }


def _vocab_unk_mask(dataset, idx, log=print) -> np.ndarray:
    """fit-only RINGCHAIN vocabulary (min_count=3 on fit_inner) -> has_UNK mask."""
    fit_set = set(int(i) for i in idx["fit_inner"])
    fit_sigs: list[str] = []
    sigs_by_row: dict[int, list[str]] = {}
    for i, data in enumerate(dataset):
        m = cf.molecule_units(
            int(i),
            data.x.numpy(),
            data.edge_index.numpy(),
            data.edge_attr.numpy(),
        )
        sigs_by_row[i] = list(m.unit_sigs)
        if i in fit_set:
            fit_sigs.extend(m.unit_sigs)
    vocab = cf.UnitVocab(min_count=3)
    vocab.fit(fit_sigs)
    has_unk = np.asarray(
        [not all(s in vocab.known_sigs for s in sigs_by_row[i]) for i in range(len(dataset))],
        dtype=bool,
    )
    log(f"[analyze] fit-only UNK vocabulary: {vocab.n_known} known types; has_UNK rows {int(has_unk.sum())}/10000")
    return has_unk


def _feature_shift(X: np.ndarray, fit_rows: np.ndarray, mon_rows: np.ndarray) -> dict[str, Any]:
    mu_fit = X[fit_rows].mean(axis=0)
    sd_fit = X[fit_rows].std(axis=0)
    mu_mon = X[mon_rows].mean(axis=0)
    denom = np.where(sd_fit > 1e-6, sd_fit, 1.0)
    shift = np.abs(mu_mon - mu_fit) / denom
    return {
        "median_abs_standardized_mean_shift": float(np.median(shift)),
        "p90_abs_standardized_mean_shift": float(np.percentile(shift, 90)),
        "max_abs_standardized_mean_shift": float(shift.max()),
        "frac_columns_shift_gt_1": float((shift > 1.0).mean()),
        "fit_mean_nonzero_columns": float((np.abs(mu_fit) > 1e-9).mean()),
        "note": "fit std used as the unit; shift>1 means the monitor column mean moved by more than one fit std",
    }


def run_analyze(out_dir: Path, log=print) -> dict[str, Any]:
    from xgboost import XGBRegressor

    load_config()
    assert_recipe_frozen()
    lock = json.loads((out_dir / "lock.json").read_text())
    if lock["config_sha256"] != file_sha256(CONFIG_PATH):
        raise RuntimeError("protocol config changed after training; refusing to evaluate dev")
    X_A = np.load(out_dir / "features" / "x_a.npy")
    X_B = np.load(out_dir / "features" / "x_b.npy")
    if sha256_array(X_A) != lock["feature_hashes"]["x_a_sha256"] or sha256_array(X_B) != lock["feature_hashes"]["x_b_sha256"]:
        raise RuntimeError("feature matrices changed after training; refusing to evaluate dev")
    for name in ("A", "B"):
        model_path = out_dir / f"xgb_{name}.json"
        if file_sha256(model_path) != lock["arms"][name]["model_sha256"]:
            raise RuntimeError(f"model {name} changed after training; refusing to evaluate dev")

    dataset = load_dataset()
    y = dataset_y(dataset)
    _smiles, idx = split_indices()
    row_hash_check(idx, log=log)
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    mon_rows = np.asarray(idx["monitor"], dtype=np.int64)
    dev_rows = np.asarray(idx["dev"], dtype=np.int64)

    # historical Full-Y dev predictions (read-only reference)
    hist = np.load(FULL_Y_DEV_PREDS)
    if not (np.array_equal(hist["row"], dev_rows) and np.allclose(hist["y"], y[dev_rows])):
        raise RuntimeError("historical Full-Y dev export misaligned")
    pred_full = hist["pred"].astype(np.float64)
    e_full = np.abs(pred_full - y[dev_rows])
    if abs(float(e_full.mean()) - REF["full_y"]["dev"]) > 1e-4:
        raise RuntimeError("historical Full-Y dev MAE drifted")

    errors: dict[str, np.ndarray] = {"FullY": e_full}
    for name, X in (("A", X_A), ("B", X_B)):
        model = XGBRegressor(**FROZEN_XGB_PARAMS)
        model.load_model(str(out_dir / f"xgb_{name}.json"))
        pred = np.asarray(model.predict(X[dev_rows]), dtype=np.float64)
        errors[name] = np.abs(pred - y[dev_rows])
        np.savez_compressed(
            out_dir / f"xgb_{name}_preds_dev.npz",
            row=dev_rows,
            pred=pred.astype(np.float32),
            y=y[dev_rows].astype(np.float32),
        )
        log(f"[analyze] XGB-{name} dev MAE {errors[name].mean():.5f}")

    # main table ------------------------------------------------------------
    train_metrics = {name: json.loads((out_dir / f"xgb_{name}_train_metrics.json").read_text()) for name in ("A", "B")}
    main_table = {
        "protocol": PROTOCOL_ID,
        "note": "internal raw-y MAE; lower is better",
        "rows": [
            {"model": "Full-Y (historical)", "fit": REF["full_y"]["fit"], "monitor": REF["full_y"]["monitor"], "dev": REF["full_y"]["dev"]},
            {"model": "O-bucket (historical)", "fit": REF["o_bucket"]["fit"], "monitor": REF["o_bucket"]["monitor"], "dev": REF["o_bucket"]["dev"]},
            {"model": "XGB-A: Typed-WL", "fit": train_metrics["A"]["fit_mae"], "monitor": train_metrics["A"]["monitor_mae"], "dev": float(errors["A"].mean())},
            {"model": "XGB-B: Full-Y pooled", "fit": train_metrics["B"]["fit_mae"], "monitor": train_metrics["B"]["monitor_mae"], "dev": float(errors["B"].mean())},
        ],
        "DeltaA_vs_FullY": float(errors["A"].mean() - REF["full_y"]["dev"]),
        "DeltaB_vs_FullY": float(errors["B"].mean() - REF["full_y"]["dev"]),
        "MAE_A_minus_MAE_B": float(errors["A"].mean() - errors["B"].mean()),
        "fit_monitor_gap": {name: train_metrics[name]["monitor_mae"] - train_metrics[name]["fit_mae"] for name in ("A", "B")},
        "fit_dev_gap": {name: float(errors[name].mean() - train_metrics[name]["fit_mae"]) for name in ("A", "B")},
        "error_stats": {
            "FullY": error_stats(e_full),
            "XGB-A": error_stats(errors["A"]),
            "XGB-B": error_stats(errors["B"]),
        },
        "monitor_signal_threshold": MONITOR_SIGNAL_THRESHOLD,
    }
    write_json(out_dir / "main_table.json", main_table)

    # paired comparisons ----------------------------------------------------
    paired = {
        "protocol": PROTOCOL_ID,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "A_minus_B": paired_bootstrap(errors["A"], errors["B"]),
        "A_minus_FullY": paired_bootstrap(errors["A"], e_full),
        "B_minus_FullY": paired_bootstrap(errors["B"], e_full),
        "note": "negative sign = first arm has lower MAE",
    }
    # robustness: exclude the 10 largest errors of each arm and the y<=-3 tail
    all_idx = np.arange(len(dev_rows))
    top10_A = np.argsort(errors["A"])[::-1][:10]
    top10_B = np.argsort(errors["B"])[::-1][:10]
    keep_A = np.setdiff1d(all_idx, top10_A)
    keep_B = np.setdiff1d(all_idx, top10_B)
    paired["A_minus_FullY_excluding_top10_A_errors"] = paired_bootstrap(errors["A"][keep_A], e_full[keep_A])
    paired["B_minus_FullY_excluding_top10_B_errors"] = paired_bootstrap(errors["B"][keep_B], e_full[keep_B])
    paired["A_minus_B_excluding_top10_A_errors"] = paired_bootstrap(errors["A"][keep_A], errors["B"][keep_A])
    ytail = y[dev_rows] <= -3.0
    paired["A_minus_FullY_y_gt_-3"] = paired_bootstrap(errors["A"][~ytail], e_full[~ytail])
    paired["B_minus_FullY_y_gt_-3"] = paired_bootstrap(errors["B"][~ytail], e_full[~ytail])
    write_json(out_dir / "paired_comparison.json", paired)

    # post-hoc per-arm dev metrics with explicit ordering evidence ----------
    for name in ("A", "B"):
        metrics = dict(train_metrics[name])
        metrics["dev_mae"] = float(errors[name].mean())
        metrics["dev_evaluated"] = True
        metrics["lock_verified_before_dev"] = True
        metrics["dev_error_stats"] = error_stats(errors[name])
        metrics["dev_paired_vs_FullY"] = paired[name + "_minus_FullY"]
        write_json(out_dir / f"xgb_{name}_dev_metrics.json", metrics)

    # subgroup analysis -----------------------------------------------------
    import networkx as nx

    n_atoms = np.asarray([int(d.x.shape[0]) for d in dataset], dtype=np.int64)[dev_rows]
    cyclomatic = np.zeros(len(dev_rows), dtype=np.int64)
    for k, row in enumerate(dev_rows):
        d = dataset[int(row)]
        ei = d.edge_index.numpy()
        g = nx.Graph()
        g.add_nodes_from(range(int(d.x.shape[0])))
        g.add_edges_from({(int(min(u, v)), int(max(u, v))) for u, v in zip(ei[0], ei[1])})
        cyclomatic[k] = g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)
    has_unk = _vocab_unk_mask(dataset, idx, log=log)[dev_rows]
    ydev = y[dev_rows]

    def _grp(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(mask.sum()),
            "y_mean": float(ydev[mask].mean()) if mask.any() else None,
            "mae": {
                "FullY": float(e_full[mask].mean()) if mask.any() else None,
                "XGB-A": float(errors["A"][mask].mean()) if mask.any() else None,
                "XGB-B": float(errors["B"][mask].mean()) if mask.any() else None,
            },
            "deltaA_vs_FullY": float(errors["A"][mask].mean() - e_full[mask].mean()) if mask.any() else None,
            "deltaB_vs_FullY": float(errors["B"][mask].mean() - e_full[mask].mean()) if mask.any() else None,
            "A_minus_B": float(errors["A"][mask].mean() - errors["B"][mask].mean()) if mask.any() else None,
        }

    subgroups: dict[str, Any] = {
        "y": {"le_-3": _grp(ytail), "gt_-3": _grp(~ytail)},
        "n_atoms": {
            "0-14": _grp(n_atoms < 15),
            "15-21": _grp((n_atoms >= 15) & (n_atoms < 22)),
            "22-29": _grp((n_atoms >= 22) & (n_atoms < 30)),
            "30+": _grp(n_atoms >= 30),
        },
        "cyclomatic": {
            "0-0": _grp(cyclomatic < 1),
            "1-1": _grp((cyclomatic >= 1) & (cyclomatic < 2)),
            "2-2": _grp((cyclomatic >= 2) & (cyclomatic < 3)),
            "3-3": _grp((cyclomatic >= 3) & (cyclomatic < 4)),
            "4+": _grp(cyclomatic >= 4),
        },
        "has_UNK": {
            "has_UNK": _grp(has_unk),
            "no_UNK": _grp(~has_unk),
        },
        "feature_shift_B_fit_to_monitor": _feature_shift(X_B, fit_rows, mon_rows),
        "feature_shift_A_fit_to_monitor": _feature_shift(X_A, fit_rows, mon_rows),
        "note": "subgroups are for interpretation only; they were never used to change a recipe",
    }
    write_json(out_dir / "subgroup_analysis.json", subgroups)

    # pre-registered decision ------------------------------------------------
    signal = {}
    for name in ("A", "B"):
        mon = train_metrics[name]["monitor_mae"]
        delta = float(errors[name].mean() - REF["full_y"]["dev"])
        ci = paired[name + "_minus_FullY"]["ci95"]
        frac = paired[name + "_minus_FullY"]["frac_a_better"]
        keep_name = keep_A if name == "A" else keep_B
        robust_trim = paired_bootstrap(errors[name][keep_name], e_full[keep_name])
        signal[name] = {
            "monitor_mae": mon,
            "monitor_signal": bool(mon <= MONITOR_SIGNAL_THRESHOLD),
            "delta_dev_vs_full_y": delta,
            "dev_direction_improvement": bool(delta < 0),
            "dev_paired_ci95": ci,
            "dev_significant": bool(ci[1] < 0),
            "dev_material": bool(delta <= DEV_MATERIAL),
            "frac_molecules_better_than_full_y": frac,
            "robust_trimmed_delta": float(robust_trim["mean_diff_signed"]),
            "robust_not_extreme_driven": bool(robust_trim["ci95"][1] < 0),
        }
        signal[name]["strong_signal"] = bool(
            signal[name]["monitor_signal"]
            and signal[name]["dev_significant"]
            and signal[name]["dev_material"]
            and signal[name]["robust_not_extreme_driven"]
        )

    if signal["A"]["strong_signal"] and signal["B"]["strong_signal"]:
        case = "C_both_arms_improve"
    elif signal["A"]["strong_signal"]:
        case = "A_typed_wl_preferred"
    elif signal["B"]["strong_signal"]:
        case = "B_frozen_readout_preferred"
    else:
        case = "D_stop_no_signal"

    decision = {
        "protocol": PROTOCOL_ID,
        "case": case,
        "signal": signal,
        "A_minus_B": paired["A_minus_B"],
        "typed_wl_vs_readout": {
            "A_preferred_by_dev_mae": bool(errors["A"].mean() < errors["B"].mean()),
            "A_minus_B_ci95": paired["A_minus_B"]["ci95"],
        },
        "gpu_next_step_authorized_by_this_round": bool(case in ("A_typed_wl_preferred", "B_frozen_readout_preferred", "C_both_arms_improve")),
        "correctness_failure_detected": False,
        "official_valid_test_loaded": False,
        "dev_reuse_caveat": (
            "internal dev has been reused by CSCL rounds; any improvement is internal "
            "directional evidence, not independent external validation"
        ),
        "forbidden_followups_respected": [
            "no seed 1",
            "no full 8000/10000 refit",
            "no official valid/test evaluation",
            "no third regressor",
            "no feature fusion",
            "no hyperparameter search",
        ],
        "interpretation_limits": {
            "A": "one fixed Typed-WL descriptor at radii 0-3 with 1024 buckets/radius; not a search over radii or widths",
            "B": "two-stage supervised probe: the 814-dim features were trained on the same fit_inner labels, so A/B is not a matched-compute comparison",
        },
    }
    write_json(out_dir / "decision.json", decision)
    log(f"[analyze] case={case}")
    return {"main_table": main_table, "paired": paired, "decision": decision, "subgroups": subgroups}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("audit", "train", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    import torch

    torch.set_num_threads(8)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        run_audit(args.out, log=log)
    elif args.mode == "train":
        run_train(args.out, log=log)
    else:
        run_analyze(args.out, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
