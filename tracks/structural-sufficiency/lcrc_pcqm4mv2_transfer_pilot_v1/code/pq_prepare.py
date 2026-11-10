"""CPU preparation for the PCQM4Mv2 transfer pilot (data audit -> cache -> norms).

Stages (each idempotent and recorded):

``data``      official dataset audit, the frozen 300K/1M subsets, the fixed
              20,000-row Train monitor set and the Train+Valid label artifact.
``estimate``  disk / RAM projection from a measured smoke cache build.
``build``     sharded ``lipo_v1`` feature cache for the 1M Train subset and the
              full official Validation split, merged into mmap directories.
``norms``     the shared Train-300K feature and label standardisation.
``manifest``  hashes of every cache artifact.

Nothing here reads a hidden Test label: the label artifact is written from the
official Train/Valid rows only, with every other row NaN.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402

import feature_spec as FS  # noqa: E402

BYTES_PER_ATOM = 0  # filled by estimate() from the measured build, not assumed


def _summary_path() -> Path:
    return C.ANALYSIS / "pcqm_prepare_summary.json"


def _update_summary(key: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = _summary_path()
    summary = C.read_json(path) if path.exists() else {"schema": "lcrc-pcqm4mv2-prepare-v1", "steps": []}
    summary.setdefault("steps", [])
    summary["steps"] = [s for s in summary["steps"] if s.get("step") != key]
    summary["steps"].append(payload)
    summary["updated_utc"] = C.utcnow()
    C.write_json(path, summary)
    return summary


def step_data(*, workers: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    info = D.load_official()
    audit = D.data_audit(info)
    C.write_json(C.ANALYSIS / "pcqm_data_audit.json", audit)
    if audit["status"] != "OK":
        raise RuntimeError(f"dataset audit failed: {audit['gates']}")
    plan = D.subset_plan(info)
    permuted_rows = plan.pop("_permuted_rows")
    n_large = int(plan["subsets"]["PCQM_W2_1M"]["n"])
    rows_path = C.CACHE / "pcqm_subset_rows.npy"
    np.save(rows_path, permuted_rows[:n_large].astype(np.int64))
    perm_path = C.CACHE / "pcqm_permutation.npy"
    np.save(perm_path, permuted_rows.astype(np.int64))
    plan["subset_rows_file"] = C.track_rel(rows_path)
    plan["subset_rows_sha256"] = C.sha256_file(rows_path)
    plan["subset_rows_bytes"] = int(rows_path.stat().st_size)
    plan["permutation_file"] = C.track_rel(perm_path)
    plan["permutation_sha256"] = C.sha256_file(perm_path)
    C.write_json(C.ANALYSIS / "pcqm_subset_plan.json", plan)

    monitor = D.monitor_rows()
    positions = monitor.pop("_positions")
    mon_path = C.CACHE / "pcqm_monitor_positions.npy"
    np.save(mon_path, positions.astype(np.int64))
    monitor["positions_file"] = C.track_rel(mon_path)
    monitor["positions_file_sha256"] = C.sha256_file(mon_path)
    monitor["positions"] = [int(x) for x in positions.tolist()]
    C.write_json(C.ANALYSIS / "pcqm_monitor_plan.json", monitor)

    label_event = D.build_label_artifact(info, C.CACHE / "pcqm_labels_trainvalid.npy")
    C.write_json(C.ANALYSIS / "pcqm_label_artifact.json", label_event)

    payload = {
        "step": "data",
        "utc": C.utcnow(),
        "seconds": float(time.perf_counter() - t0),
        "dataset_audit": C.track_rel(C.ANALYSIS / "pcqm_data_audit.json"),
        "subset_plan": C.track_rel(C.ANALYSIS / "pcqm_subset_plan.json"),
        "subset_plan_status": plan["status"],
        "monitor_plan": C.track_rel(C.ANALYSIS / "pcqm_monitor_plan.json"),
        "label_artifact": label_event["artifact"],
        "label_artifact_sha256": label_event["artifact_sha256"],
        "split_sizes": audit["split_sizes"],
        "status": "OK",
    }
    _update_summary("data", payload)
    return payload


def step_estimate(*, workers: int) -> dict[str, Any]:
    """Build a small smoke cache, measure bytes per molecule, project the full build."""
    t0 = time.perf_counter()
    info = D.load_official()
    rows_1m = np.load(C.CACHE / "pcqm_subset_rows.npy").astype(np.int64)
    smoke_train = rows_1m[:6000].tolist()
    smoke_valid = np.asarray(info["split"]["valid"], dtype=np.int64)[:4000].tolist()
    shard_dir = C.CACHE / "shards_smoke"
    build = D.build_shards(info, rows=smoke_train, shard_dir=shard_dir, tag="pcqm_smoke_train", workers=workers)
    merge = D.merge_shards(
        sorted(shard_dir.glob("shard_*.pt")),
        FS.load_spec(C.ANALYSIS / "pcqm_spec.json"),
        out_dir=C.CACHE / "pcqm_smoke_train",
        split="train",
        tag="pcqm_smoke_train",
    )
    shutil.rmtree(shard_dir, ignore_errors=True)
    build_v = D.build_shards(info, rows=smoke_valid, shard_dir=shard_dir, tag="pcqm_smoke_valid", workers=workers)
    merge_v = D.merge_shards(
        sorted(shard_dir.glob("shard_*.pt")),
        FS.load_spec(C.ANALYSIS / "pcqm_spec.json"),
        out_dir=C.CACHE / "pcqm_smoke_valid",
        split="valid",
        tag="pcqm_smoke_valid",
    )
    shutil.rmtree(shard_dir, ignore_errors=True)
    per_mol = merge["cache_bytes"] / max(merge["n_molecules"], 1)
    n_train, n_valid = 1_000_000, int(len(info["split"]["valid"]))
    estimate_train = per_mol * n_train
    estimate_valid = merge_v["cache_bytes"] / max(merge_v["n_molecules"], 1) * n_valid
    free_bytes = shutil.disk_usage(C.CACHE).free
    total_estimate = estimate_train + estimate_valid
    payload = {
        "step": "estimate",
        "utc": C.utcnow(),
        "seconds": float(time.perf_counter() - t0),
        "smoke_train_molecules": int(merge["n_molecules"]),
        "smoke_valid_molecules": int(merge_v["n_molecules"]),
        "measured_bytes_per_molecule": float(per_mol),
        "measured_mean_atoms": float(build["n_atoms"]["mean"]),
        "measured_mean_s2_paths": float(build["s2_paths"]["mean"]),
        "projected_train1m_cache_bytes": float(estimate_train),
        "projected_valid_cache_bytes": float(estimate_valid),
        "projected_total_bytes": float(total_estimate),
        "projected_total_gb": float(total_estimate / 1e9),
        "shard_build_peak_bytes_estimate": float(total_estimate * 1.05),
        "free_disk_bytes_before_build": int(free_bytes),
        "free_disk_gb_before_build": float(free_bytes / 1e9),
        "headroom_factor": float(free_bytes / max(total_estimate * 2, 1)),
        "ram_peak_mb_build": C.peak_rss_mb(),
        "shards_are_removed_after_merge": True,
        "per_molecule_build_seconds": float(build["wall_seconds"] / max(build["n_rows_built"], 1)),
        "build": {"train": build, "valid": build_v},
        "merge": {"train": merge, "valid": merge_v},
        "status": "OK" if free_bytes > 3 * total_estimate else "INSUFFICIENT_DISK",
    }
    C.write_json(C.ANALYSIS / "pcqm_disk_ram_estimate.json", payload)
    _update_summary("estimate", {k: v for k, v in payload.items() if k not in ("build", "merge")})
    if payload["status"] != "OK":
        raise RuntimeError(f"insufficient disk: {payload['free_disk_gb_before_build']:.1f} GB free")
    return payload


def step_build(*, workers: int, chunk_size: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    info = D.load_official()
    spec = FS.lipo_v1_spec()
    rows_1m = np.load(C.CACHE / "pcqm_subset_rows.npy").astype(np.int64)
    valid_rows = np.asarray(info["split"]["valid"], dtype=np.int64)
    out: dict[str, Any] = {"step": "build", "utc": C.utcnow(), "caches": {}}
    for tag, rows in (("pcqm_train1m", rows_1m.tolist()), ("pcqm_valid", valid_rows.tolist())):
        shard_dir = C.CACHE / f"shards_{tag}"
        t_build = time.perf_counter()
        build = D.build_shards(
            info, rows=rows, shard_dir=shard_dir, tag=tag, workers=workers, chunk_size=chunk_size
        )
        merge = D.merge_shards(
            sorted(shard_dir.glob("shard_*.pt")), spec, out_dir=C.CACHE / tag, split=tag, tag=tag
        )
        C.write_json(C.ANALYSIS / f"{tag}_cache_merge.json", merge)
        shutil.rmtree(shard_dir, ignore_errors=True)
        out["caches"][tag] = {
            "rows": int(len(rows)),
            "rows_sha256": build["rows_sha256"],
            "build_seconds": float(time.perf_counter() - t_build),
            "n_atoms_total": merge["n_atoms_total"],
            "n_s2_paths": merge["n_s2_paths"],
            "cache_bytes": merge["cache_bytes"],
            "cache_sha256": merge["cache_sha256"],
            "max_s2_records_per_molecule": build["max_s2_records_per_molecule"],
            "shards_removed": True,
        }
        C.log(f"[prepare] {tag}: {merge['n_molecules']} molecules, {merge['cache_bytes']/1e9:.2f} GB")
    out["seconds"] = float(time.perf_counter() - t0)
    out["peak_rss_mb"] = C.peak_rss_mb()
    out["status"] = "OK"
    _update_summary("build", out)
    return out


def step_norms(*, workers: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
    cache = D.load_cache(C.CACHE / "pcqm_train1m")
    n_fit = min(int(s["train_size"]) for s in C.RUNS_TABLE.values())
    fit_rows = np.arange(n_fit, dtype=np.int64)
    norm = D.feature_norm_rows(cache, spec, fit_rows, source="pcqm_train1m cache rows 0..299999")
    C.write_json(C.ANALYSIS / "pcqm_feature_norm.json", norm)
    labels = np.load(C.CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    gids = np.asarray(cache["gids"], dtype=np.int64)[:n_fit]
    gap = labels[gids]
    lnorm = D.label_norm(gap)
    lnorm["gap_sha256"] = C.sha256_bytes(np.asarray(gap, dtype=np.float64).tobytes())
    z = D.to_z(gap, lnorm)
    lnorm["z_mean_check"] = float(z.mean())
    lnorm["z_std_check"] = float(z.std())
    lnorm["roundtrip_max_abs_error_eV"] = float(np.abs(D.from_z(z, lnorm) - gap).max())
    C.write_json(C.ANALYSIS / "pcqm_label_norm.json", lnorm)
    payload = {
        "step": "norms",
        "utc": C.utcnow(),
        "seconds": float(time.perf_counter() - t0),
        "n_fit_molecules": int(n_fit),
        "feature_norm_fingerprint": norm["norm_fingerprint"],
        "scalar_constant_indices": norm["scalar_constant_indices"],
        "global_constant_indices": norm["global_constant_indices"],
        "label_mean_eV": lnorm["mean_eV"],
        "label_std_eV": lnorm["std_eV"],
        "label_median_eV": lnorm["median_eV"],
        "z_mean_check": lnorm["z_mean_check"],
        "z_std_check": lnorm["z_std_check"],
        "roundtrip_max_abs_error_eV": lnorm["roundtrip_max_abs_error_eV"],
        "status": "OK",
    }
    # the constant-median baseline (no GPU)
    valid_cache = D.load_cache(C.CACHE / "pcqm_valid")
    valid_gids = np.asarray(valid_cache["gids"], dtype=np.int64)
    valid_gap = labels[valid_gids]
    median = float(np.median(gap))
    pred = np.full(valid_gap.shape, median, dtype=np.float64)
    evaluator = D._evaluator()
    report = {
        "model": "Constant median (Train 300K)",
        "baseline": "pred = median(gap) of subset_300k for every Validation molecule",
        "median_eV": median,
        "n_valid": int(valid_gap.size),
        "valid_mae_eV": float(evaluator.eval({"y_pred": pred, "y_true": valid_gap})["mae"]),
        "valid_mae_eV_numpy": float(np.mean(np.abs(pred - valid_gap))),
        "target_mean_eV": float(valid_gap.mean()),
        "params": 0,
        "gpu_hours": 0.0,
        "utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(C.ANALYSIS / "pcqm_constant_baseline.json", report)
    payload["constant_baseline_valid_mae_eV"] = report["valid_mae_eV"]
    _update_summary("norms", payload)
    return payload


def step_manifest(*, workers: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
    manifest: dict[str, Any] = {
        "schema": "lcrc-pcqm4mv2-cache-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "dataset": C.DATASET,
        "feature_spec": C.track_rel(C.ANALYSIS / "pcqm_spec.json"),
        "feature_spec_hash": spec.hash(),
        "labels": "Train + official Valid only; hidden rows are NaN and never indexed",
    }
    for key, rel in (("train1m", "pcqm_train1m"), ("valid", "pcqm_valid")):
        path = C.CACHE / rel
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        manifest[key] = {
            "path": C.track_rel(path),
            "cache_sha256": C.sha256_tree(path),
            "n_molecules": int(meta["n_molecules"]),
            "n_atoms_total": int(meta["n_atoms_total"]),
            "n_bonds_total": int(meta["n_bonds_total"]),
            "n_s1_codes": int(meta["n_s1_codes"]),
            "n_s2_paths": int(meta["n_s2_paths"]),
            "bytes": int(sum(p.stat().st_size for p in path.glob("*") if p.is_file())),
            "format": meta["format"],
            "labels_present": bool(meta.get("labels_present", True)),
        }
    manifest["subset_rows_sha256"] = C.sha256_file(C.CACHE / "pcqm_subset_rows.npy")
    manifest["subset_rows_bytes"] = int((C.CACHE / "pcqm_subset_rows.npy").stat().st_size)
    manifest["monitor_positions_sha256"] = C.sha256_file(C.CACHE / "pcqm_monitor_positions.npy")
    manifest["label_artifact_sha256"] = C.sha256_file(C.CACHE / "pcqm_labels_trainvalid.npy")
    manifest["total_bytes"] = int(
        manifest["train1m"]["bytes"] + manifest["valid"]["bytes"] + manifest["subset_rows_bytes"]
    )
    C.write_json(C.ANALYSIS / "pcqm_cache_manifest.json", manifest)
    payload = {
        "step": "manifest",
        "utc": C.utcnow(),
        "seconds": float(time.perf_counter() - t0),
        "manifest": C.track_rel(C.ANALYSIS / "pcqm_cache_manifest.json"),
        "total_cache_gb": float(manifest["total_bytes"] / 1e9),
        "status": "OK",
    }
    _update_summary("manifest", payload)
    return payload


STEPS = {
    "data": step_data,
    "estimate": step_estimate,
    "build": lambda workers: step_build(workers=workers, chunk_size=4000),
    "norms": step_norms,
    "manifest": step_manifest,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=sorted(STEPS) + ["all", "build-valid"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    C.log(f"[prepare] step={args.step} workers={args.workers} round={C.ROUND_ID}")
    done = []
    for name in (["data", "estimate", "build", "norms", "manifest"] if args.step == "all" else [args.step]):
        if name == "build-valid":
            continue
        payload = STEPS[name](workers=int(args.workers))
        done.append({"step": name, "status": payload.get("status")})
        C.log(f"[prepare] {name}: {payload.get('status')}")
    C.write_json(
        C.ANALYSIS / "pcqm_prepare_last_run.json",
        {"utc": C.utcnow(), "steps": done, "git_commit": C.git_commit()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
