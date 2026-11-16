"""Runner ``sq-pilot-small-v1``: prepare / smoke / dataset_worker / aggregate.

One run = one stage.  The two parallel GPU workers are

    --set stage=dataset_worker --set dataset=ogbg-molesol   (pool res-gpu0)
    --set stage=dataset_worker --set dataset=ogbg-mollipo   (pool res-gpu1)

and each executes the frozen per-dataset sequence of the round brief:

    1 data + base bank + short smoke
    2 F_H1, E_H1            (usable performance coordinates first)
    3 A discovery + A_H1/A_H2
    4 R construction + R_H1/R_H2
    5 F_H2, E_H2
    6 G at both peak LRs
    7 L (C1 / NO_BOND_MEAN, fixed R3)
    8 native export, replay audit, case studies, summary

Every step persists its own JSON/npz so an interrupted worker resumes instead of
restarting, and so the aggregate stage never needs the GPU host again.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_ast as A
import sq_bank as B
import sq_common as C
import sq_data as D
import sq_explain as E
import sq_lcrc as L
import sq_search as S
import sq_xgb as X
from sq_common import Ledger, stage_timer

DEFAULT_CONFIG = C.TRACK_ROOT / "configs/sq-pilot-small-v1.yaml"
SPLITS = ("train", "valid", "test")


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Structural query learning pilot: supervised discrete query discovery + "
            "gradient-boosted head on official ogbg-molesol / ogbg-mollipo "
            "(F/R/A/E/G/L, test_policy=always, seed 0)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Dataset fingerprints computed from the committed label-free caches."""
    payload: dict[str, Any] = {"round_id": C.ROUND_ID, "cache_version": D.CACHE_VERSION}
    for dataset in C.DATASETS:
        entry: dict[str, Any] = {}
        path = D.cache_path(dataset)
        entry["graph_cache_sha256"] = C.sha256_file(path) if path.is_file() else None
        if path.is_file():
            cache = D.load_cache(dataset)
            entry["split_sizes"] = {
                split: int(D.split_indices(cache, split).size) for split in SPLITS
            }
            entry["split_hash"] = {
                split: C.sha256_array(D.split_indices(cache, split)) for split in SPLITS
            }
        for name, path_fn in (("ecfp", D.ecfp_path), ("lipo_raw", D.lipo_raw_path)):
            candidate = path_fn(dataset)
            entry[f"{name}_sha256"] = C.sha256_file(candidate) if candidate.is_file() else None
        payload[dataset] = entry
    return payload, C.sha256_json(payload)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def results_dir(dataset: str) -> Path:
    return C.RESULTS_DIR / dataset


def _write(path: Path, payload: Any) -> None:
    C.write_json(path, payload)


def _read(path: Path) -> Any:
    return C.read_json(path)


def _exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _save_predictions(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    rows: Mapping[str, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split in predictions:
        arrays[f"{split}_pred_raw"] = np.asarray(predictions[split], dtype=np.float64)
        arrays[f"{split}_y_raw"] = np.asarray(targets[split], dtype=np.float64)
        arrays[f"{split}_row_id"] = np.asarray(rows[split], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def _load_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    return {
        split: {
            "pred_raw": data[f"{split}_pred_raw"],
            "y_raw": data[f"{split}_y_raw"],
            "row_id": data[f"{split}_row_id"],
        }
        for split in SPLITS
    }


def _method_matrices(packed, records, cache):
    matrices = {
        split: B.design_matrix(packed[split], records, cache=cache, include_base=True)[0]
        for split in SPLITS
    }
    names = B.design_matrix(packed["train"], records, cache=cache, include_base=True)[1]
    return matrices, names


def _fit_preset(
    *,
    method: str,
    preset: str,
    dataset: str,
    matrices: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    rows: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    names: Sequence[str],
    device: str,
    out_dir: Path,
    ledger: Ledger,
    num_boost_round: int,
    save_booster: bool = False,
) -> dict[str, Any]:
    json_path = out_dir / f"method_{method}_{preset}.json"
    pred_path = out_dir / f"method_{method}_{preset}_pred.npz"
    if _exists(json_path) and _exists(pred_path):
        payload = _read(json_path)
        payload["resumed"] = True
        return payload
    with stage_timer(ledger, f"fit_{method}_{preset}", dataset=dataset, method=method, preset=preset):
        result = X.run_head(
            method,
            preset,
            matrices,
            targets,
            scaler,
            names,
            device=device,
            num_boost_round=num_boost_round,
        )
    payload = {
        "method": method,
        "preset": preset,
        "best": result.best,
        "final": result.final,
        "last20": result.last20,
        "seconds": result.seconds,
        "n_features": len(result.feature_names),
        "n_trees": int(result.booster.num_boosted_rounds()),
        "resumed": False,
    }
    _write(json_path, payload)
    _save_predictions(pred_path, result.predictions, targets, rows)
    C.write_csv(
        out_dir / f"curves_{method}_{preset}.csv",
        result.curves,
        columns=["method", "preset", "iteration", "split", "rmse_raw", "mae_raw"],
    )
    if save_booster:
        model_dir = C.ensure_dir(C.CACHE_DIR / "models" / dataset)
        result.booster.save_model(str(model_dir / f"{method}_{preset}.json"))
    return payload


def _finalize(method: str, presets: Sequence[str], out_dir: Path) -> dict[str, Any]:
    entries = [_read(out_dir / f"method_{method}_{preset}.json") for preset in presets]
    chosen = min(
        entries,
        key=lambda item: (
            item["best"]["valid_rmse_raw"],
            X.PRESET_ORDER.index(item["preset"]),
            item["best"]["iteration"],
        ),
    )
    return {
        "method": method,
        "selected_preset": chosen["preset"],
        "selected": chosen["best"],
        "final_round_selected": chosen["final"],
        "last20_selected": chosen["last20"],
        "per_preset": {
            entry["preset"]: {
                "best": entry["best"],
                "final": entry["final"],
                "last20": entry["last20"],
                "seconds": entry["seconds"],
                "n_features": entry["n_features"],
                "n_trees": entry["n_trees"],
                "resumed": entry.get("resumed", False),
            }
            for entry in entries
        },
        "fit_seconds_total": float(sum(entry["seconds"] for entry in entries)),
        "complete": True,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round requires test_policy=always; the control plane blocked official test access"
        )
    stage = str(config.get("stage", "dataset_worker"))
    ledger_dir = Path(context.artifact_dir)
    ledger = Ledger(ledger_dir / "ledger.jsonl")
    t0 = time.perf_counter()
    if stage == "prepare":
        metrics = _stage_prepare(config, ledger)
    elif stage == "smoke":
        metrics = _stage_smoke(config, ledger)
    elif stage == "dataset_worker":
        metrics = _stage_dataset_worker(config, ledger, Path(context.artifact_dir))
    elif stage == "aggregate":
        import sq_report

        metrics = sq_report.aggregate(config, ledger, output_dir=C.RESULTS_DIR)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "stage": stage, "wall_seconds": time.perf_counter() - t0}
    _write(ledger_dir / f"{stage}_summary.json", {**metrics, "ledger": ledger.summary()})
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[f"artifacts/{stage}_summary.json", "artifacts/ledger.jsonl"],
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _dataset_audit(dataset: str) -> dict[str, Any]:
    """Fingerprint the committed cache and re-verify every invariant the round needs.

    Deterministic and label-free apart from the scaler fit (Train labels only).
    Raises instead of reporting a number when an invariant is broken.
    """
    cache_path = D.cache_path(dataset)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"{dataset}: committed graph cache missing at {cache_path}. "
            "Build it from the official OGB data (see analysis/prepare_notes.md)."
        )
    cache = D.load_cache(dataset)
    audit: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": dataset,
        "cache_path": str(cache_path.relative_to(C.REPO_ROOT)) if cache_path.is_relative_to(C.REPO_ROOT) else str(cache_path),
        "cache_sha256": cache["cache_sha256"],
        "cache_sha256_verified": C.sha256_file(cache_path),
        "n_graphs": int(cache["row_ids"].size),
        "split_sizes": {split: int(D.split_indices(cache, split).size) for split in SPLITS},
        "split_hash": {split: C.sha256_array(D.split_indices(cache, split)) for split in SPLITS},
        "row_id_hash": {
            split: C.ordered_index_hash(D.split_indices(cache, split).tolist()) for split in SPLITS
        },
        "label_hash": {
            split: C.sha256_array(np.asarray(cache["labels"][D.split_indices(cache, split)])) for split in SPLITS
        },
        "label_range": {
            split: [
                float(np.min(cache["labels"][D.split_indices(cache, split)])),
                float(np.max(cache["labels"][D.split_indices(cache, split)])),
            ]
            for split in SPLITS
        },
        "n_atoms_total": int(cache["node_ptr"][-1]),
        "n_physical_bonds_total": int(cache["bond_ptr"][-1]),
        "n_multi_component_molecules": int((cache["graph_n_components"] > 1).sum()),
        "ring_bucket_counts": {str(b): int((cache["min_ring"] == b).sum()) for b in range(0, 10)},
        "official_evaluator_metric": D._evaluator_metric_name(dataset),
        "git_commit": C.git_commit(),
    }
    if audit["cache_sha256_verified"] != cache["cache_sha256"]:
        raise ValueError(f"{dataset}: cache checksum mismatch on disk")
    expected_sha = C.EXPECTED_CACHE_SHA256[cache_path.name]
    if audit["cache_sha256_verified"] != expected_sha:
        raise ValueError(
            f"{dataset}: cache {cache_path.name} sha256 {audit['cache_sha256_verified']} "
            f"!= protocol-pinned {expected_sha}"
        )
    expected = C.EXPECTED_SPLIT_SIZES[dataset]
    if audit["split_sizes"] != expected:
        raise ValueError(f"{dataset}: split sizes {audit['split_sizes']} != official {expected}")
    packed = D.packed_split(cache, "train")
    audit["physical_bond_dedup_check"] = {
        "n_physical_bonds": int(packed.n_bonds),
        "sum_graph_bonds": int(packed.graph_n_bonds.sum()),
        "matches": bool(int(packed.n_bonds) == int(packed.graph_n_bonds.sum())),
        "max_degree": int(packed.degree.max()) if packed.n_nodes else 0,
    }
    if not audit["physical_bond_dedup_check"]["matches"]:
        raise ValueError(f"{dataset}: physical bond count disagrees with the per-graph sum")
    vocab = D.train_vocabulary(cache)
    audit["train_vocabulary"] = {
        "element_values": vocab["element_values"],
        "n_bond_predicates": len(B.bond_predicates(vocab)),
        "degree_values": vocab["degree_values"],
        "ring_values": vocab["ring_values"],
    }
    scaler = C.train_target_scaler(cache["labels"][D.split_indices(cache, "train")])
    audit["train_target_scaler"] = scaler
    train_raw = cache["labels"][D.split_indices(cache, "train")].astype(np.float64)
    roundtrip = (train_raw - scaler["mean"]) / scaler["std"] * scaler["std"] + scaler["mean"]
    audit["target_norm_roundtrip_max_abs_error"] = float(np.abs(roundtrip - train_raw).max())
    if audit["target_norm_roundtrip_max_abs_error"] > 1e-5:
        raise ValueError("target standardisation is not reversible")
    out = C.ensure_dir(C.RESULTS_DIR / dataset)
    _write(out / "data_audit.json", audit)
    return audit


def _stage_prepare(config: Mapping[str, Any], ledger: Ledger) -> dict[str, Any]:
    datasets = _datasets(config)
    summary: dict[str, Any] = {"datasets": {}}
    for dataset in datasets:
        with stage_timer(ledger, "prepare_verify", dataset=dataset):
            audit = _dataset_audit(dataset)
            report: dict[str, Any] = {"audit": audit}
            for name, path in (("ecfp", D.ecfp_path(dataset)), ("lipo_raw", D.lipo_raw_path(dataset))):
                if not path.is_file():
                    raise FileNotFoundError(f"{dataset}: auxiliary cache missing at {path}")
                digest = C.sha256_file(path)
                if digest != C.EXPECTED_CACHE_SHA256[path.name]:
                    raise ValueError(f"{dataset}: {path.name} sha256 {digest} != pinned")
                report[name] = {"path": str(path), "present": True, "sha256": digest}
            summary["datasets"][dataset] = report
    summary["environment"] = C.environment_snapshot()
    summary["git_commit"] = C.git_commit()
    summary["n_datasets"] = len(datasets)
    return summary


def _datasets(config: Mapping[str, Any]) -> list[str]:
    dataset = config.get("dataset")
    if dataset in (None, "", "all"):
        return list(C.DATASETS)
    if dataset not in C.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {C.DATASETS}")
    return [str(dataset)]


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _stage_smoke(config: Mapping[str, Any], ledger: Ledger) -> dict[str, Any]:
    """Cheap end-to-end correctness probe (no scientific number is kept)."""
    dataset = _datasets(config)[0]
    cache = D.load_cache(dataset)
    packed = {split: D.packed_split(cache, split) for split in SPLITS}
    vocab = D.train_vocabulary(cache)
    masks = B.MaskCache()
    checks: dict[str, Any] = {}
    with stage_timer(ledger, "smoke_bank", dataset=dataset):
        records, stats = B.build_base_bank(packed["train"], vocab, cache=masks)
    checks["bank"] = {"n_queries": len(records), "stats": stats}

    disagreements = 0
    for graph in range(min(8, packed["train"].n_graphs)):
        sub = _single_graph(packed["train"], graph)
        start = int(np.concatenate([[0], np.cumsum(packed["train"].graph_n_atoms)])[graph])
        stop = start + int(packed["train"].graph_n_atoms[graph])
        for record in records[:6]:
            packed_mask = A.evaluate(packed["train"], record.ast)
            if not np.array_equal(packed_mask[start:stop], A.evaluate(sub, record.ast)):
                disagreements += 1
    checks["single_graph_vs_packed_disagreements"] = disagreements
    checks["single_graph_vs_packed_pass"] = disagreements == 0

    scaler = C.train_target_scaler(cache["labels"][D.split_indices(cache, "train")])
    targets = {
        split: cache["labels"][D.split_indices(cache, split)].astype(np.float64) for split in SPLITS
    }
    rows = {split: D.split_indices(cache, split) for split in SPLITS}
    y_std = {split: (targets[split] - scaler["mean"]) / scaler["std"] for split in SPLITS}
    old_rounds, old_proxy = S.MAX_OUTER_ROUNDS, S.PROXY_ROUNDS
    try:
        S.MAX_OUTER_ROUNDS = int(config.get("smoke_outer_rounds", 2))
        S.PROXY_ROUNDS = int(config.get("smoke_proxy_rounds", 16))
        with stage_timer(ledger, "smoke_discovery", dataset=dataset):
            discovery = S.discover_queries(
                train_packed=packed["train"],
                valid_packed=packed["valid"],
                test_packed=packed["test"],
                base_records=records,
                vocabulary=vocab,
                y_train_std=y_std["train"],
                y_valid_std=y_std["valid"],
                y_test_std=y_std["test"],
                scaler=scaler,
                device="cpu",
                cache=masks,
                time_limit_s=600.0,
            )
    finally:
        S.MAX_OUTER_ROUNDS, S.PROXY_ROUNDS = old_rounds, old_proxy
    checks["smoke_discovery"] = {
        "n_rounds": discovery["n_outer_rounds"],
        "n_discovered": len(discovery["discovered"]),
        "stop_reason": discovery["stop_reason"],
    }
    bank = list(records) + list(discovery["discovered"])
    matrices, names = _method_matrices(packed, bank, masks)
    with stage_timer(ledger, "smoke_head", dataset=dataset):
        head = X.run_head(
            "A",
            "H1",
            matrices,
            targets,
            scaler,
            names,
            device="cpu",
            num_boost_round=int(config.get("smoke_boost_rounds", 24)),
        )
    trees, replay = E.export_and_audit(
        booster=head.booster,
        feature_names=names,
        matrices=matrices,
        scaler=scaler,
        row_ids=rows,
        rounds=int(config.get("smoke_boost_rounds", 24)),
        max_train=32,
    )
    checks["native_replay"] = {key: value for key, value in replay.items() if key != "splits"}
    checks["official_metric_probe"] = D.official_metric(
        dataset, head.predictions["valid"], targets["valid"]
    )
    checks["pass"] = bool(
        checks["single_graph_vs_packed_pass"]
        and replay["pass"]
        and checks["bank"]["n_queries"] > 0
    )
    _write(C.RESULTS_DIR / dataset / "smoke.json", checks)
    return {"dataset": dataset, "smoke_pass": checks["pass"], **checks}


def _single_graph(packed, graph: int):
    node_ptr = np.concatenate([[0], np.cumsum(packed.graph_n_atoms)])
    bond_ptr = np.concatenate([[0], np.cumsum(packed.graph_n_bonds)])
    a0, a1 = int(node_ptr[graph]), int(node_ptr[graph + 1])
    b0, b1 = int(bond_ptr[graph]), int(bond_ptr[graph + 1])
    n = a1 - a0
    return A.PackedGraph(
        dataset=packed.dataset,
        split="single",
        n_graphs=1,
        n_nodes=n,
        node_graph=np.zeros(n, dtype=np.int32),
        atom_feat=packed.atom_feat[a0:a1],
        degree=packed.degree[a0:a1],
        min_ring=packed.min_ring[a0:a1],
        in_cycle=packed.in_cycle[a0:a1],
        n_bonds=b1 - b0,
        bond_u=packed.bond_u[b0:b1] - a0,
        bond_v=packed.bond_v[b0:b1] - a0,
        bond_feat=packed.bond_feat[b0:b1],
        graph_n_atoms=np.asarray([n], dtype=np.int32),
        graph_n_bonds=np.asarray([b1 - b0], dtype=np.int32),
        graph_n_components=np.asarray([packed.graph_n_components[graph]], dtype=np.int32),
        graph_cycle_rank=np.asarray([packed.graph_cycle_rank[graph]], dtype=np.int32),
        row_ids=np.asarray([packed.row_ids[graph]], dtype=np.int64),
        labels=np.asarray([packed.labels[graph]], dtype=np.float32),
        smiles=[packed.smiles[graph]],
    )


# ---------------------------------------------------------------------------
# dataset worker
# ---------------------------------------------------------------------------


def _stage_dataset_worker(config: Mapping[str, Any], ledger: Ledger, artifact_dir: Path) -> dict[str, Any]:
    dataset = _datasets(config)[0]
    out = C.ensure_dir(results_dir(dataset))
    audit = _dataset_audit(dataset)  # idempotent cache/split verification
    resume = bool(config.get("resume", True))
    num_rounds = int(config.get("num_boost_round", X.NUM_BOOST_ROUND))
    # Engineering-only budget overrides (absent in the formal configuration, so
    # the frozen module constants apply and are what the report records).
    if config.get("max_discovered"):
        S.MAX_DISCOVERED = int(config["max_discovered"])
    if config.get("max_outer_rounds"):
        S.MAX_OUTER_ROUNDS = int(config["max_outer_rounds"])
    xgb_device, xgb_status = X.resolve_backend(str(config.get("xgboost_device", "cuda:0")))
    summary: dict[str, Any] = {
        "dataset": dataset,
        "xgboost_device_requested": str(config.get("xgboost_device", "cuda:0")),
        "xgboost_device_used": xgb_device,
        "xgboost_device_status": xgb_status,
        "num_boost_round": num_rounds,
        "seed": int(config.get("seed", 0)),
        "cache_sha256": audit["cache_sha256"],
        "split_sizes": audit["split_sizes"],
        "test_access": "granted",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ---- 1. data, base bank ------------------------------------------------
    with stage_timer(ledger, "load_data", dataset=dataset):
        cache = D.load_cache(dataset)
        scaler = C.train_target_scaler(cache["labels"][D.split_indices(cache, "train")])
        targets = {
            split: cache["labels"][D.split_indices(cache, split)].astype(np.float64)
            for split in SPLITS
        }
        rows = {split: D.split_indices(cache, split) for split in SPLITS}
        packed = {split: D.packed_split(cache, split) for split in SPLITS}
        vocab = D.train_vocabulary(cache)
        masks = B.MaskCache()
        _write(
            out / "split_identity.json",
            {
                "dataset": dataset,
                "cache_sha256": cache["cache_sha256"],
                "split_hash": {split: C.sha256_array(rows[split]) for split in SPLITS},
                "row_id_hash": {
                    split: C.ordered_index_hash(rows[split].tolist()) for split in SPLITS
                },
                "split_sizes": {split: int(rows[split].size) for split in SPLITS},
                "train_target_scaler": scaler,
                "label_range": {
                    split: [float(targets[split].min()), float(targets[split].max())]
                    for split in SPLITS
                },
            },
        )
    with stage_timer(ledger, "base_bank", dataset=dataset):
        records_b, bank_stats = B.build_base_bank(packed["train"], vocab, cache=masks)
    _write(
        out / "base_bank.json",
        {
            "dataset": dataset,
            "stats": bank_stats,
            "queries": [record.to_dict() for record in records_b],
        },
    )
    y_std = {split: (targets[split] - scaler["mean"]) / scaler["std"] for split in SPLITS}

    # ---- 2. F_H1, E_H1 ----------------------------------------------------
    matrices_f, names_f = _method_matrices(packed, records_b, masks)
    _fit_preset(method="F", preset="H1", dataset=dataset, matrices=matrices_f, targets=targets,
                rows=rows, scaler=scaler, names=names_f, device=xgb_device, out_dir=out,
                ledger=ledger, num_boost_round=num_rounds)
    ecfp = D.load_ecfp(dataset)
    matrices_e: dict[str, np.ndarray] = {}
    names_e: list[str] = []
    for split in SPLITS:
        design, meta = D.ecfp_design_matrix(ecfp, cache, split)
        matrices_e[split] = design
        names_e = list(meta["names"])
    _write(out / "ecfp_meta.json", meta)
    _fit_preset(method="E", preset="H1", dataset=dataset, matrices=matrices_e, targets=targets,
                rows=rows, scaler=scaler, names=names_e, device=xgb_device, out_dir=out,
                ledger=ledger, num_boost_round=num_rounds)

    # ---- 3. A: discovery + A_H1/A_H2 --------------------------------------
    discovery_path = out / "discovery.json"
    if resume and _exists(discovery_path):
        discovery_payload = _read(discovery_path)
        records_a = list(records_b) + [_record_from_dict(item) for item in discovery_payload["discovered"]]
        summary["discovery_resumed"] = True
    else:
        with stage_timer(ledger, "discovery", dataset=dataset):
            discovery = S.discover_queries(
                train_packed=packed["train"],
                valid_packed=packed["valid"],
                test_packed=packed["test"],
                base_records=records_b,
                vocabulary=vocab,
                y_train_std=y_std["train"],
                y_valid_std=y_std["valid"],
                y_test_std=y_std["test"],
                scaler=scaler,
                device=xgb_device,
                cache=masks,
                time_limit_s=float(config.get("discovery_time_limit_s", S.DISCOVERY_TIME_LIMIT_S)),
            )
        records_a = list(records_b) + list(discovery["discovered"])
        discovery_payload = {
            key: value
            for key, value in discovery.items()
            if key not in ("bank", "base_records", "discovered")
        }
        discovery_payload["discovered"] = [
            record.to_dict() | {"ast_tree": E.ast_to_json(record.ast)}
            for record in discovery["discovered"]
        ]
        _write(discovery_path, discovery_payload)
        summary["discovery_resumed"] = False
    matrices_a, names_a = _method_matrices(packed, records_a, masks)
    for preset in ("H1", "H2"):
        _fit_preset(method="A", preset=preset, dataset=dataset, matrices=matrices_a, targets=targets,
                    rows=rows, scaler=scaler, names=names_a, device=xgb_device, out_dir=out,
                    ledger=ledger, num_boost_round=num_rounds, save_booster=True)
    summary["n_queries_A"] = len(records_a)
    summary["n_queries_B"] = len(records_b)
    summary["n_discovered"] = len(records_a) - len(records_b)
    summary["discovery"] = {
        key: discovery_payload.get(key)
        for key in ("stop_reason", "resource_truncated", "elapsed_seconds", "n_outer_rounds",
                    "proxy_seconds_total", "discovery_time_limit_s")
    }

    # ---- 4. R: complexity-matched random library --------------------------
    random_path = out / "random_library.json"
    if resume and _exists(random_path):
        random_payload = _read(random_path)
    else:
        with stage_timer(ledger, "random_library", dataset=dataset):
            built = S.build_random_library(
                discovered=[record for record in records_a if record.source == "discovered"],
                train_packed=packed["train"],
                vocabulary=vocab,
                existing_records=records_b,
                cache=masks,
                time_limit_s=float(config.get("random_time_limit_s", S.R_TIME_LIMIT_S)),
            )
        random_payload = {key: value for key, value in built.items() if key != "records"}
        random_payload["records"] = [
            record.to_dict() | {"ast_tree": E.ast_to_json(record.ast)} for record in built["records"]
        ]
        _write(random_path, random_payload)
    records_r = list(records_b) + [_record_from_dict(item) for item in random_payload["records"]]
    summary["random_library"] = {
        "status": random_payload["status"],
        "n_filled": random_payload["n_filled"],
        "n_requested": random_payload["n_requested"],
        "n_missing": random_payload["n_missing"],
        "all_shapes_match": random_payload["all_shapes_match"],
        "elapsed_seconds": random_payload["elapsed_seconds"],
    }
    matrices_r, names_r = _method_matrices(packed, records_r, masks)
    for preset in ("H1", "H2"):
        _fit_preset(method="R", preset=preset, dataset=dataset, matrices=matrices_r, targets=targets,
                    rows=rows, scaler=scaler, names=names_r, device=xgb_device, out_dir=out,
                    ledger=ledger, num_boost_round=num_rounds)

    # ---- 5. F_H2, E_H2 ----------------------------------------------------
    _fit_preset(method="F", preset="H2", dataset=dataset, matrices=matrices_f, targets=targets,
                rows=rows, scaler=scaler, names=names_f, device=xgb_device, out_dir=out,
                ledger=ledger, num_boost_round=num_rounds)
    _fit_preset(method="E", preset="H2", dataset=dataset, matrices=matrices_e, targets=targets,
                rows=rows, scaler=scaler, names=names_e, device=xgb_device, out_dir=out,
                ledger=ledger, num_boost_round=num_rounds)

    method_summaries = {method: _finalize(method, X.PRESET_ORDER, out) for method in ("F", "R", "A", "E")}
    _write(out / "methods_fra.json", method_summaries)

    # ---- 6. G -------------------------------------------------------------
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary["torch_device_used"] = device
    gine_path = out / "method_G.json"
    # A resumed stage is only trusted when *both* the record and its selected
    # predictions exist; otherwise G is refitted rather than half-reported.
    if not (resume and _exists(gine_path) and _exists(out / "method_G_pred.npz")):
        import sq_gine as G

        with stage_timer(ledger, "fit_G", dataset=dataset, device=device):
            gine = G.run_gine(
                packed=packed,
                scaler=scaler,
                device=device,
                epochs=int(config.get("gine_epochs", G.EPOCHS)),
            )
        C.write_csv(
            out / "curves_G.csv",
            gine["curves"],
            columns=["epoch", "peak_lr", "train_rmse_raw", "valid_rmse_raw", "test_rmse_raw",
                     "train_mae_raw", "valid_mae_raw", "test_mae_raw", "lr", "seconds"],
        )
        _save_predictions(
            out / "method_G_pred.npz",
            {split: gine["selected_predictions"][split]["pred_raw"] for split in SPLITS},
            targets,
            rows,
        )
        gine["curves"] = []
        _write(gine_path, gine)
    else:
        gine = _read(gine_path)
    summary["G"] = {
        "selected_peak_lr": gine["selected_peak_lr"],
        "selected": gine["selected"],
        "seconds_total": gine["seconds_total"],
        "init_hashes_identical": gine["init_hashes_identical"],
        "batch_order_identical": gine["batch_order_identical"],
        "n_parameters": gine["n_parameters"],
        "n_categories": gine["n_categories"],
    }

    # ---- 7. L -------------------------------------------------------------
    lcrc_path = out / "method_L.json"
    if not (resume and _exists(lcrc_path) and _exists(out / "method_L_pred.npz")):
        try:
            with stage_timer(ledger, "fit_L", dataset=dataset, device=device):
                lcrc = L.run_lcrc(
                    payload=cache,
                    dataset=dataset,
                    targets_raw=targets,
                    scaler=scaler,
                    device=device,
                    epochs=int(config["lcrc_epochs"]) if config.get("lcrc_epochs") else None,
                )
            _save_predictions(
                out / "method_L_pred.npz",
                {split: lcrc["selected_predictions"][split]["pred_raw"] for split in SPLITS},
                targets,
                rows,
            )
            _write(lcrc_path, lcrc)
        except Exception as exc:  # noqa: BLE001 - a refused baseline is recorded, not crashed
            lcrc = {
                "method": "L",
                "status": "BASELINE_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write(lcrc_path, lcrc)
    else:
        lcrc = _read(lcrc_path)
    summary["L"] = {
        "status": lcrc.get("status", "complete"),
        "selected": lcrc.get("selected"),
        "n_parameters": lcrc.get("n_parameters"),
        "seconds": lcrc.get("seconds"),
    }

    # ---- 8. export, replay, cases ----------------------------------------
    export_payload = _export_stage(
        dataset=dataset,
        out=out,
        packed=packed,
        records_a=records_a,
        records_b=records_b,
        matrices_a=matrices_a,
        names_a=names_a,
        scaler=scaler,
        targets=targets,
        method_summaries=method_summaries,
        config=config,
    )
    summary.update(export_payload)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["ledger"] = ledger.summary()
    summary["environment"] = C.environment_snapshot()
    summary["git_commit"] = C.git_commit()
    _write(out / "worker_summary.json", summary)
    _write(artifact_dir / "worker_summary.json", summary)
    return {
        "dataset": dataset,
        "n_queries_B": len(records_b),
        "n_queries_A": len(records_a),
        "A_valid_rmse": method_summaries["A"]["selected"]["valid_rmse_raw"],
        "A_test_rmse": method_summaries["A"]["selected"]["test_rmse_raw"],
        "F_test_rmse": method_summaries["F"]["selected"]["test_rmse_raw"],
        "R_test_rmse": method_summaries["R"]["selected"]["test_rmse_raw"],
        "E_test_rmse": method_summaries["E"]["selected"]["test_rmse_raw"],
        "G_test_rmse": gine["selected"]["test_rmse_raw"],
        "L_test_rmse": (lcrc.get("selected") or {}).get("test_rmse_raw"),
        "native_replay_pass": summary["native_replay"]["pass"],
        "xgboost_device_used": xgb_device,
        "torch_device_used": device,
        "random_library_status": random_payload["status"],
        "discovery_stop_reason": summary["discovery"]["stop_reason"],
        "wall_seconds": ledger.summary()["wall_seconds_total"],
    }


def _record_from_dict(item: Mapping[str, Any]) -> B.QueryRecord:
    if "ast_tree" not in item:
        raise ValueError(f"stored query {item.get('label')!r} has no ast_tree payload")
    ast = _ast_from_json(item["ast_tree"])
    return B.QueryRecord(
        ast=ast,
        label=item["label"],
        source=item["source"],
        outer_round=int(item["outer_round"]),
        train_support=int(item["train_support"]),
        train_gain=item.get("train_gain"),
    )


def _ast_from_json(payload: Mapping[str, Any]) -> tuple:
    op = payload["op"]
    if op == "ATOM_EQ":
        return A.ATOM_EQ(int(payload["atom_column"]), int(payload["value"]))
    if op in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
        return (op, int(payload["value"]))
    if op in A.NODE_KINDS:
        return (op,)
    if op == "AND":
        return A.AND(*[_ast_from_json(child) for child in payload["children"]])
    if op == "NBR":
        return A.NBR(
            _ast_from_json(payload["parent"]), _bond_from_json(payload["bond"]), int(payload["m"])
        )
    if op == "BRANCH":
        return A.BRANCH(
            _ast_from_json(payload["parent_p"]),
            _ast_from_json(payload["parent_q"]),
            _bond_from_json(payload["bond_p"]),
            _bond_from_json(payload["bond_q"]),
        )
    raise ValueError(f"unknown AST op {op!r}")


def _bond_from_json(payload: Mapping[str, Any]) -> tuple:
    if payload["op"] == "ANY":
        return A.ANY_BOND_PRED()
    return A.BOND_EQ(int(payload["bond_column"]), int(payload["value"]))


def _export_stage(
    *,
    dataset: str,
    out: Path,
    packed,
    records_a: Sequence[B.QueryRecord],
    records_b: Sequence[B.QueryRecord],
    matrices_a: Mapping[str, np.ndarray],
    names_a: Sequence[str],
    scaler: Mapping[str, float],
    targets: Mapping[str, np.ndarray],
    method_summaries: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    selected = method_summaries["A"]
    preset = selected["selected_preset"]
    rounds = int(selected["selected"]["iteration"])
    queries_path = out / "queries.json"
    _write(
        queries_path,
        E.export_queries(
            records_a,
            dataset=dataset,
            base_count=len(records_b),
            discovered_count=len(records_a) - len(records_b),
            feature_names=names_a,
            scaler=scaler,
            selected={
                "method": "A",
                "preset": preset,
                "iteration": rounds,
                "selection_rule": "minimum Val RMSE over H1/H2 and all rounds; ties H1 then earlier",
                "valid_rmse_raw": float(selected["selected"]["valid_rmse_raw"]),
                "test_rmse_raw": float(selected["selected"]["test_rmse_raw"]),
                "num_boost_round_max": int(config.get("num_boost_round", X.NUM_BOOST_ROUND)),
            },
        ),
    )
    model_path = C.CACHE_DIR / "models" / dataset / f"A_{preset}.json"
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    trees, replay = E.export_and_audit(
        booster=booster,
        feature_names=names_a,
        matrices=matrices_a,
        scaler=scaler,
        row_ids={split: np.arange(targets[split].size) for split in SPLITS},
        rounds=rounds,
    )
    _write(out / "native_trees.json", trees)
    _write(out / "replay_audit.json", replay)
    payload["native_replay"] = replay
    payload["tree_complexity"] = E.tree_complexity(trees)
    payload["native_export"] = {
        "selected_preset": preset,
        "selected_iteration": rounds,
        "queries_file": str(queries_path.relative_to(C.TRACK_ROOT)),
        "trees_file": str((out / "native_trees.json").relative_to(C.TRACK_ROOT)),
        "queries_file_bytes": int(queries_path.stat().st_size),
        "trees_file_bytes": int((out / "native_trees.json").stat().st_size),
    }
    if not replay["pass"]:
        raise RuntimeError(f"native replay failed tolerance: {replay['max_abs_error_overall']}")

    f_selected = method_summaries["F"]
    pred_f = _load_predictions(out / f"method_F_{f_selected['selected_preset']}_pred.npz")
    pred_a = _load_predictions(out / f"method_A_{preset}_pred.npz")
    cases = E.select_cases(
        pred_a=pred_a["valid"]["pred_raw"],
        pred_f=pred_f["valid"]["pred_raw"],
        target=targets["valid"],
        row_ids=pred_a["valid"]["row_id"],
    )
    masks = B.MaskCache()
    design_valid = matrices_a["valid"]
    explained = []
    for case in cases:
        row = int(case["row"])
        explained.append(
            E.explain_molecule(
                packed=packed["valid"],
                records=records_a,
                trees=trees,
                feature_names=names_a,
                design_row=E.design_row(design_valid, names_a, row),
                scaler=scaler,
                case=case,
                cache=masks,
            )
        )
    _write(out / "cases.json", {"dataset": dataset, "cases": explained})
    payload["cases"] = [
        {
            "kind": item["case"]["kind"],
            "row_id": item["case"]["row_id"],
            "target": item["case"]["target"],
            "pred_A": item["case"]["pred_A"],
            "pred_F": item["case"]["pred_F"],
            "total_path_length": item["tree_paths"]["total_path_length"],
            "reconstructed_raw_prediction": item["tree_paths"]["reconstructed_raw_prediction"],
        }
        for item in explained
    ]
    payload["costs"] = {
        "queries_file_bytes": int(queries_path.stat().st_size),
        "trees_file_bytes": int((out / "native_trees.json").stat().st_size),
        "n_queries": len(records_a),
        "n_trees": int(trees["n_trees"]),
        "n_nodes": int(payload["tree_complexity"]["n_nodes"]),
        "mean_tree_path_length": float(
            np.mean([item["tree_paths"]["mean_path_length"] for item in explained])
        ),
    }
    return payload
