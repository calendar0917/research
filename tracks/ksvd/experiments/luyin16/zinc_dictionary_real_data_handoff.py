"""Read-only small real-ZINC data handoff for the frozen Full reader.

This module is a **data-export utility**, not an experiment: it trains nothing,
fits nothing, scans no result tree, and never instantiates the official test
split.  It packages, per official split (train/valid only):

* the existing ``R`` reader-input cache (the exact ``814``-wide tensor the
  published Full soup reader consumes) together with ``y`` and ``p_base``;
* the raw PyG graph arrays (per-molecule node types and unique undirected
  physical bonds, local endpoints ``u < v``) recovered from the *original*
  graph, never from rooted patch occurrences;
* ``topo_model_input`` (the standardized 25-D tensor actually fed to
  ``topology_encoder``) and ``topo_raw`` (unstandardized ``raw_vector('hinge')``
  from the existing per-split topology CSV);
* ``canonical_group_id`` when an already-verified semantic-complete canonical
  certificate key exists (official train only here);
* the frozen reader parameters (``814 -> 39 -> 39 -> 1`` with two ReLUs) exported
  as plain NumPy arrays, plus the small ``topology_encoder`` state and the task
  dictionary ``D_L`` / ``V_L``.

Run (local CPU, <= 8 threads)::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_dictionary_real_data_handoff

All outputs land in ``tracks/ksvd/results/zinc_dictionary_real_data_handoff/``
(large ``*.npz`` / ``*.zip`` are git-ignored local evidence).
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/zinc_dictionary_real_data_handoff"
CHECKPOINT = (
    TRACK_ROOT
    / "results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt"
)
R_CACHE_DIR = TRACK_ROOT / "results/zinc_graph_dictionary_readout_v1/cache"
PREFLIGHT = TRACK_ROOT / "results/zinc_graph_dictionary_readout_v1/preflight.json"
TRAIN_PROBE = (
    TRACK_ROOT
    / "results/zinc_e2e_dictenv_typed_cycle_v1/probe/train_ring_probe.npz"
)
ZINC_ROOT = REPO_ROOT / "data/ZINC"

REFERENCE_VALID_MAE = 0.1191540920053958
READER_MAX_ABS_TOL = 1e-5
READER_MAE_TOL = 1e-6
#: split fingerprint of the official ZINC subset split (reused, never recomputed).
SPLIT_FINGERPRINT = "58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a"

FULL_BLOCK_WIDTHS = {"unary": 289, "relation": 485, "global": 32, "topology": 8}
assert sum(FULL_BLOCK_WIDTHS.values()) == 814


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_git_revision() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def load_reader_split(split: str) -> dict[str, np.ndarray]:
    path = R_CACHE_DIR / f"full_{split}_R.npz"
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def build_graph_arrays(
    dataset: Sequence[Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recover per-molecule raw graph arrays from the original PyG dataset.

    Each undirected physical bond is emitted once with local endpoints
    ``u < v``.  Bidirectional duplicates must carry identical bond types.
    """
    node_ptr = [0]
    atom_type: list[np.ndarray] = []
    edge_ptr = [0]
    edge_u: list[np.ndarray] = []
    edge_v: list[np.ndarray] = []
    edge_type: list[np.ndarray] = []
    n_nodes_total = 0
    n_edges_total = 0
    counts_once = 0
    counts_twice = 0
    counts_other = 0
    inconsistent_type = 0
    self_loops = 0
    out_of_range = 0
    for data in dataset:
        n = int(data.num_nodes)
        x = np.asarray(data.x).reshape(-1).astype(np.int32)
        if x.shape[0] != n:
            raise RuntimeError(f"x width {x.shape[0]} != num_nodes {n}")
        ei = np.asarray(data.edge_index)
        ea = np.asarray(data.edge_attr).reshape(-1).astype(np.int32)
        lo = np.minimum(ei[0], ei[1]).astype(np.int64)
        hi = np.maximum(ei[0], ei[1]).astype(np.int64)
        self_loops += int(np.sum(lo == hi))
        out_of_range += int(np.sum((lo < 0) | (hi >= n)))
        key = lo * np.int64(n) + hi
        unique_keys, _first_index, counts = np.unique(
            key, return_index=True, return_counts=True
        )
        # Bond-type consistency across duplicates of the same undirected key.
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        sorted_ea = ea[order]
        starts = np.searchsorted(sorted_key, unique_keys, side="left")
        seg_min = np.minimum.reduceat(sorted_ea, starts)
        seg_max = np.maximum.reduceat(sorted_ea, starts)
        inconsistent_type += int(np.sum(seg_min != seg_max))
        counts_once += int(np.sum(counts == 1))
        counts_twice += int(np.sum(counts == 2))
        counts_other += int(np.sum((counts != 1) & (counts != 2)))
        edge_u.append((unique_keys // np.int64(n)).astype(np.int32))
        edge_v.append((unique_keys % np.int64(n)).astype(np.int32))
        edge_type.append(seg_min.astype(np.int32))
        n_edges_total += int(unique_keys.shape[0])
        atom_type.append(x)
        n_nodes_total += n
        node_ptr.append(n_nodes_total)
        edge_ptr.append(n_edges_total)

    arrays = {
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "atom_type": np.concatenate(atom_type) if atom_type else np.zeros((0,), np.int32),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "edge_u": np.concatenate(edge_u) if edge_u else np.zeros((0,), np.int32),
        "edge_v": np.concatenate(edge_v) if edge_v else np.zeros((0,), np.int32),
        "edge_type": np.concatenate(edge_type) if edge_type else np.zeros((0,), np.int32),
    }
    report = {
        "n_graphs": int(len(node_ptr) - 1),
        "n_nodes_total": int(n_nodes_total),
        "n_undirected_edges_total": int(n_edges_total),
        "undirected_edge_multiplicity_1": int(counts_once),
        "undirected_edge_multiplicity_2": int(counts_twice),
        "undirected_edge_multiplicity_other": int(counts_other),
        "inconsistent_bond_type_keys": int(inconsistent_type),
        "self_loops": int(self_loops),
        "out_of_range_endpoints": int(out_of_range),
        "node_ptr_starts_at_zero": bool(arrays["node_ptr"][0] == 0),
        "node_ptr_monotone": bool(np.all(np.diff(arrays["node_ptr"]) >= 0)),
        "edge_ptr_starts_at_zero": bool(arrays["edge_ptr"][0] == 0),
        "edge_ptr_monotone": bool(np.all(np.diff(arrays["edge_ptr"]) >= 0)),
        "endpoints_within_graph": bool(out_of_range == 0),
        "all_edges_u_lt_v": bool(
            arrays["edge_u"].shape[0] == 0
            or np.all(arrays["edge_u"] < arrays["edge_v"])
        ),
        "bond_types_consistent": bool(inconsistent_type == 0),
    }
    return arrays, report


def reader_params(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "W1": state["reader.net.0.weight"].numpy().astype(np.float32),
        "b1": state["reader.net.0.bias"].numpy().astype(np.float32),
        "W2": state["reader.net.2.weight"].numpy().astype(np.float32),
        "b2": state["reader.net.2.bias"].numpy().astype(np.float32),
        "W3": state["reader.net.4.weight"].numpy().astype(np.float32),
        "b3": state["reader.net.4.bias"].numpy().astype(np.float32),
    }


def reader_forward(reader: dict[str, np.ndarray], R: np.ndarray) -> np.ndarray:
    h = np.maximum(R @ reader["W1"].T.astype(np.float64) + reader["b1"].astype(np.float64), 0.0)
    h = np.maximum(h @ reader["W2"].T.astype(np.float64) + reader["b2"].astype(np.float64), 0.0)
    return (h @ reader["W3"].T.astype(np.float64) + reader["b3"].astype(np.float64)).reshape(-1)


def topology_forward(enc: dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    h = np.maximum(
        z @ enc["W1"].T.astype(np.float64) + enc["b1"].astype(np.float64), 0.0
    )
    return h @ enc["W2"].T.astype(np.float64) + enc["b2"].astype(np.float64)


def main() -> int:
    torch.set_num_threads(8)
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "checkpoint": CHECKPOINT,
        "train_R_cache": R_CACHE_DIR / "full_train_R.npz",
        "valid_R_cache": R_CACHE_DIR / "full_valid_R.npz",
        "encoded_train": TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_train.pt",
        "encoded_valid": TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_valid.pt",
        "train_topology_csv": TRACK_ROOT / "results/zinc_topology_cache/train_topology_features.csv",
        "valid_topology_csv": TRACK_ROOT / "results/zinc_topology_cache/valid_topology_features.csv",
        "train_probe_npz": TRAIN_PROBE,
        "preflight_json": PREFLIGHT,
    }
    source_sha_before = {name: _sha256_file(path) for name, path in source_paths.items()}
    revision = _sha256_git_revision()

    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    reader = reader_params(state)
    topology_encoder = {
        "W1": state["topology_encoder.0.weight"].numpy().astype(np.float32),
        "b1": state["topology_encoder.0.bias"].numpy().astype(np.float32),
        "W2": state["topology_encoder.2.weight"].numpy().astype(np.float32),
        "b2": state["topology_encoder.2.bias"].numpy().astype(np.float32),
    }
    task_dict = {
        "D_L": state["local_dictionary_bridge.D_L"].numpy().astype(np.float32),
        "V_L": state["local_dictionary_bridge.V_L"].numpy().astype(np.float32),
    }
    np.savez_compressed(
        RESULTS_DIR / "reader.npz",
        **reader,
        topology_W1=topology_encoder["W1"],
        topology_b1=topology_encoder["b1"],
        topology_W2=topology_encoder["W2"],
        topology_b2=topology_encoder["b2"],
        D_L=task_dict["D_L"],
        V_L=task_dict["V_L"],
    )

    # -- encoded topology input (standardized) ---------------------------------
    train_enc, valid_enc, _audit = sdp.load_encoded()

    # -- raw topology (unstandardized hinge) from the existing split CSV -------
    train_raw_df = ztopo.load_cache("train")
    valid_raw_df = ztopo.load_cache("valid")
    if train_raw_df is None or valid_raw_df is None:
        raise RuntimeError("topology CSV cache missing")

    # -- cross-check CSV order against the frozen V4 record cache --------------
    from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
        load_train_valid_records,
    )

    train_records, valid_records = load_train_valid_records()

    # -- raw graphs (original PyG, never rooted patch occurrences) -------------
    train_ds = _load_zinc(ZINC_ROOT, "train")
    valid_ds = _load_zinc(ZINC_ROOT, "val")
    assert len(train_ds) == 10000 and len(valid_ds) == 1000

    # -- canonical group key for train (already verified, semantic complete) ----
    with np.load(TRAIN_PROBE, allow_pickle=False) as probe:
        train_group_ids = probe["group_ids"].astype(np.int64)

    bundles: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, Any] = {}
    for split, enc_list, ds, raw_df, records in (
        ("train", train_enc, train_ds, train_raw_df, train_records),
        ("valid", valid_enc, valid_ds, valid_raw_df, valid_records),
    ):
        cache = load_reader_split(split)
        n = int(cache["R"].shape[0])
        if not (len(enc_list) == len(ds) == len(raw_df) == len(records) == n):
            raise RuntimeError(
                f"{split} length mismatch: R={n} enc={len(enc_list)} ds={len(ds)} "
                f"csv={len(raw_df)} records={len(records)}"
            )
        y = cache["y"].astype(np.float64)
        # order alignment: encoded y vs R-cache y (encoded stores y per graph)
        enc_y = np.asarray(
            [float(d.y.reshape(-1)[0]) for d in enc_list], dtype=np.float64
        )
        enc_y_max_abs_diff = float(np.max(np.abs(enc_y - y)))

        topo_model_input = np.stack(
            [d.topology_features.numpy().reshape(-1) for d in enc_list], axis=0
        ).astype(np.float32)
        topo_raw = np.stack(
            [
                ztopo.raw_vector(raw_df.iloc[i], "hinge", input_width=25)
                for i in range(n)
            ],
            axis=0,
        ).astype(np.float32)
        record_topo = np.stack(
            [np.asarray(rec.topology_features, dtype=np.float32).reshape(-1) for rec in records],
            axis=0,
        )
        topo_raw_vs_records_max = float(np.max(np.abs(topo_raw - record_topo)))

        graph_arrays, graph_report = build_graph_arrays(ds)
        graph_report["y_max_abs_diff_encoded_vs_Rcache"] = enc_y_max_abs_diff
        graph_report["topo_raw_vs_record_cache_max_abs_diff"] = topo_raw_vs_records_max
        graph_report["n_nodes_per_graph_min"] = int(np.min(np.diff(graph_arrays["node_ptr"])))
        graph_report["n_nodes_per_graph_max"] = int(np.max(np.diff(graph_arrays["node_ptr"])))
        graph_report["n_edges_per_graph_min"] = int(np.min(np.diff(graph_arrays["edge_ptr"])))
        graph_report["n_edges_per_graph_max"] = int(np.max(np.diff(graph_arrays["edge_ptr"])))

        # topology encoder replay: R last 8 dims must equal encoder(topo_model_input)
        enc_out = topology_forward(topology_encoder, topo_model_input.astype(np.float64))
        topo_vs_R_last8 = float(np.max(np.abs(enc_out - cache["R"][:, -8:])))
        graph_report["topo_encoder_replay_vs_R_last8_max_abs_diff"] = topo_vs_R_last8

        p_replay = reader_forward(reader, cache["R"])
        graph_report["reader_replay_max_abs_diff"] = float(
            np.max(np.abs(p_replay - cache["p_base"]))
        )
        graph_report["reader_replay_valid_mae"] = (
            float(np.mean(np.abs(p_replay - y))) if split == "valid" else None
        )

        bundle = {
            "ids": cache["ids"].astype(np.int64),
            "y": y,
            "p_base": cache["p_base"].astype(np.float64),
            "R": cache["R"].astype(np.float32),
            "node_ptr": graph_arrays["node_ptr"],
            "atom_type": graph_arrays["atom_type"],
            "edge_ptr": graph_arrays["edge_ptr"],
            "edge_u": graph_arrays["edge_u"],
            "edge_v": graph_arrays["edge_v"],
            "edge_type": graph_arrays["edge_type"],
            "topo_model_input": topo_model_input,
            "topo_raw": topo_raw,
            "split": np.asarray(split),
            "official_test_loaded": np.asarray(False),
            "frozen_backbone": np.asarray(True),
            "checkpoint_sha": np.asarray(source_sha_before["checkpoint"]),
            "split_fingerprint": np.asarray(SPLIT_FINGERPRINT),
        }
        if split == "train":
            bundle["canonical_group_id"] = train_group_ids
        bundles[split] = bundle
        reports[split] = graph_report
        np.savez_compressed(RESULTS_DIR / f"{split}.npz", **bundle)
        if split == "train":
            np.savez_compressed(
                RESULTS_DIR / "train_reader.npz",
                R=bundle["R"],
                y=bundle["y"],
                p_base=bundle["p_base"],
                ids=bundle["ids"],
            )
        else:
            np.savez_compressed(
                RESULTS_DIR / "valid_reader.npz",
                R=bundle["R"],
                y=bundle["y"],
                p_base=bundle["p_base"],
                ids=bundle["ids"],
            )

    # -- acceptance ------------------------------------------------------------
    valid_mae = float(np.mean(np.abs(bundles["valid"]["p_base"] - bundles["valid"]["y"])))
    acceptance = {
        "n_train": int(bundles["train"]["R"].shape[0]),
        "n_valid": int(bundles["valid"]["R"].shape[0]),
        "R_width": int(bundles["train"]["R"].shape[1]),
        "checkpoint_sha_matches_this_round_cache": (
            str(source_sha_before["checkpoint"])
            == str(load_reader_split("train")["checkpoint_sha"])
        ),
        "reader_replay_max_abs_diff_train": reports["train"]["reader_replay_max_abs_diff"],
        "reader_replay_max_abs_diff_valid": reports["valid"]["reader_replay_max_abs_diff"],
        "reader_replay_within_1e-5": bool(
            reports["train"]["reader_replay_max_abs_diff"] <= READER_MAX_ABS_TOL
            and reports["valid"]["reader_replay_max_abs_diff"] <= READER_MAX_ABS_TOL
        ),
        "valid_mae": valid_mae,
        "reference_valid_mae": REFERENCE_VALID_MAE,
        "valid_mae_abs_diff": abs(valid_mae - REFERENCE_VALID_MAE),
        "valid_mae_within_1e-6": bool(abs(valid_mae - REFERENCE_VALID_MAE) <= READER_MAE_TOL),
        "valid_id172_y": float(bundles["valid"]["y"][172]),
        "valid_id172_p_base": float(bundles["valid"]["p_base"][172]),
        "valid_min_y": float(bundles["valid"]["y"].min()),
        "train_min_y": float(bundles["train"]["y"].min()),
        "topo_model_input_vs_R_last8_max_abs_diff": {
            "train": reports["train"]["topo_encoder_replay_vs_R_last8_max_abs_diff"],
            "valid": reports["valid"]["topo_encoder_replay_vs_R_last8_max_abs_diff"],
        },
        "official_test_loaded": False,
        "no_training_performed": True,
    }
    acceptance["passed"] = bool(
        acceptance["n_train"] == 10000
        and acceptance["n_valid"] == 1000
        and acceptance["R_width"] == 814
        and acceptance["checkpoint_sha_matches_this_round_cache"]
        and acceptance["reader_replay_within_1e-5"]
        and acceptance["valid_mae_within_1e-6"]
    )

    source_sha_after = {name: _sha256_file(path) for name, path in source_paths.items()}
    sources_unchanged = {
        name: bool(source_sha_before[name] == source_sha_after[name])
        for name in source_sha_before
    }

    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    split_fingerprint = str(
        preflight.get("split_fingerprint")
        or preflight.get("fingerprint", {}).get("split_fingerprint")
        or SPLIT_FINGERPRINT
    )

    export_sha = {
        name: _sha256_file(RESULTS_DIR / name)
        for name in (
            "reader.npz",
            "train.npz",
            "valid.npz",
            "train_reader.npz",
            "valid_reader.npz",
        )
    }

    manifest = {
        "revision": revision,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "small real ZINC data handoff for frozen Full reader (no training)",
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(REPO_ROOT)),
            "sha256": source_sha_before["checkpoint"],
            "tag": "SCALE-FULL-seed0_soup_state",
            "state_dict_parameters": int(
                sum(v.numel() for v in state.values())
            ),
        },
        "split_fingerprint": split_fingerprint,
        "official_order_rule": (
            "row i of every per-split array is official split position i, i.e. the "
            "order of the PyG ZINC subset processed file (train.pt/val.pt); ids = "
            "0..N-1.  R / y / p_base / graph arrays / topology share this order."
        ),
        "source_sha256": source_sha_before,
        "source_sha256_after": source_sha_after,
        "sources_unchanged": sources_unchanged,
        "export_sha256": export_sha,
        "fields": {
            "ids": {"shape": "[N]", "dtype": "int64", "meaning": "official split position"},
            "y": {"shape": "[N]", "dtype": "float64", "meaning": "penalized logP target"},
            "p_base": {
                "shape": "[N]",
                "dtype": "float64",
                "meaning": "published Full soup prediction on R (frozen)",
            },
            "R": {
                "shape": "[N,814]",
                "dtype": "float32",
                "meaning": "frozen Full reader input; float32 is lossless vs the float64 cache",
            },
            "node_ptr": {"shape": "[N+1]", "dtype": "int64", "meaning": "CSR node offsets"},
            "atom_type": {
                "shape": "[sum_n]",
                "dtype": "int32",
                "meaning": "original PyG data.x atom class 0..27 (one per node)",
            },
            "edge_ptr": {"shape": "[N+1]", "dtype": "int64", "meaning": "CSR edge offsets"},
            "edge_u": {
                "shape": "[sum_m]",
                "dtype": "int32",
                "meaning": "undirected physical bond endpoint (local, u<v)",
            },
            "edge_v": {
                "shape": "[sum_m]",
                "dtype": "int32",
                "meaning": "undirected physical bond endpoint (local, u<v)",
            },
            "edge_type": {
                "shape": "[sum_m]",
                "dtype": "int32",
                "meaning": "original PyG edge_attr bond class 0..3",
            },
            "topo_model_input": {
                "shape": "[N,25]",
                "dtype": "float32",
                "meaning": "standardized tensor fed to topology_encoder (train-only scaler)",
            },
            "topo_raw": {
                "shape": "[N,25]",
                "dtype": "float32",
                "meaning": "unstandardized zinc_topology_features.raw_vector('hinge')",
            },
            "canonical_group_id": {
                "shape": "[N] (train only)",
                "dtype": "int64",
                "meaning": "verified semantic-complete canonical certificate group key "
                "(pynauty_certificate+coloured_incidence_sequence)",
                "missing_on_valid": "no verified canonical cache for official valid exists",
            },
        },
        "topology_naming": {
            "topo_raw": "RAW zinc_topology_features.raw_vector(mode='hinge'), 25-D, "
            "unstandardized; from tracks/ksvd/results/zinc_topology_cache/{split}_topology_features.csv",
            "topo_model_input": "the actual standardized data.topology_features tensor "
            "(train-only Standardizer) consumed by model.topology_encoder in the encoded cache",
            "encoder": "Linear(25,16) -> ReLU -> Linear(16,8) = topology_encoder.0 / .2",
            "reader_input_layout": {
                "unary": [0, 289],
                "relation": [289, 774],
                "global": [774, 806],
                "topology": [806, 814],
            },
        },
        "reader": {
            "architecture": "R(814) -> Linear 814->39 -> ReLU -> Linear 39->39 -> ReLU -> Linear 39->1",
            "layer_order": ["reader.net.0 (W1,b1)", "ReLU", "reader.net.2 (W2,b2)", "ReLU", "reader.net.4 (W3,b3)"],
            "exported_arrays": ["W1", "b1", "W2", "b2", "W3", "b3"],
            "dtype": "float32",
        },
        "task_dictionary": {
            "D_L": [144, 288],
            "V_L": [288, 144],
            "bridge": {
                "steps": 16,
                "lambda1": 0.05,
                "lambda2": 0.01,
                "seed": 0,
                "coding": "sparse",
            },
            "note": "optional context; not part of the readout acceptance gate",
        },
        "source_cache_reuse": {
            "R": "tracks/ksvd/results/zinc_graph_dictionary_readout_v1/cache/full_{split}_R.npz",
            "graph_arrays": "original PyG ZINC subset processed files (train.pt/val.pt); test.pt never read",
            "topology_csv": "tracks/ksvd/results/zinc_topology_cache/{split}_topology_features.csv",
            "canonical_train": "tracks/ksvd/results/zinc_e2e_dictenv_typed_cycle_v1/probe/train_ring_probe.npz",
        },
        "graph_checks": reports,
        "acceptance": acceptance,
        "caveats": [
            "The Full backbone has already seen all official train labels; a train head "
            "split is therefore NOT an end-to-end OOF estimate.",
            "The official valid split has been reused repeatedly across prior rounds and "
            "can only serve as an exploratory screen.",
            "official test was never loaded or instantiated (official_test_loaded=False).",
            "No training, no dictionary/standardizer refit, no HPO, no seed1.",
        ],
        "known_oof_resources": {
            "status": "none identified for the Full model; not scanned",
            "note": "a raw-feature typed-cycle probe OOF diagnostic exists under "
            "tracks/ksvd/results/zinc_e2e_dictenv_typed_cycle_v1 (not a Full-model OOF and "
            "not used here)",
        },
        "wall_clock_seconds": None,
    }
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "accept_summary.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # -- packaging -------------------------------------------------------------
    def make_zip(name: str, members: Sequence[Path]) -> dict[str, Any]:
        path = RESULTS_DIR / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for member in members:
                zf.write(member, arcname=member.name)
        return {"path": name, "bytes": path.stat().st_size, "under_31MiB": path.stat().st_size < 31 * 1024 * 1024}

    packages = {}
    packages["A_reader_and_R"] = make_zip(
        "package_A_reader_R.zip",
        [
            RESULTS_DIR / "reader.npz",
            RESULTS_DIR / "train_reader.npz",
            RESULTS_DIR / "valid_reader.npz",
        ],
    )
    packages["B_graphs_topology"] = make_zip(
        "package_B_graphs_topology.zip",
        [RESULTS_DIR / "train.npz", RESULTS_DIR / "valid.npz"],
    )
    combined_members = [
        RESULTS_DIR / "reader.npz",
        RESULTS_DIR / "train.npz",
        RESULTS_DIR / "valid.npz",
        RESULTS_DIR / "manifest.json",
        RESULTS_DIR / "accept_summary.json",
    ]
    packages["combined"] = make_zip("package_zinc_handoff.zip", combined_members)

    manifest["packages"] = packages
    manifest["wall_clock_seconds"] = float(time.perf_counter() - started)
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({"acceptance": acceptance, "packages": packages}, indent=2))
    return 0 if acceptance["passed"] and all(sources_unchanged.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
