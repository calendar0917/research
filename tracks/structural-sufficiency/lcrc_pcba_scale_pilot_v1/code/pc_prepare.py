"""PCBA preparation: data audit, label artifact, exact feature cache, cache audit.

Stages (each is idempotent and records hashes):

``audit``   official dataset + splits + Train/Valid label statistics
``cache``   chunked exact S2 cache build, merge, label-free cache audit,
            historical MolHIV S2 cross-check
``verify``  re-verify an existing cache on the host that will train (hash +
            label masking + structure); no RDKit required
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import pc_common as C  # noqa: E402
import pc_data as DATA  # noqa: E402


def stage_audit(*, force: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    info = DATA.load_official()
    C.log(f"[audit] official dataset loaded: {info['n']} graphs, {info['num_tasks']} tasks")
    split_payload = DATA.split_payload(info)
    C.write_json(C.ANALYSIS / "pcba_split.json", split_payload, compact=True)
    label_path = C.ANALYSIS / "pcba_labels_trainvalid.npy"
    event = DATA.build_label_artifact(info, label_path)
    audit = DATA.data_audit(info, label_path, event)
    audit["stage_seconds"] = float(time.perf_counter() - t0)
    C.write_json(C.ANALYSIS / "data_audit.json", audit)
    C.log(f"[audit] status={audit['status']} gates={audit['gates']}")
    if audit["status"] != "OK":
        raise RuntimeError(f"data audit failed: {audit['gates']}")
    return audit


def stage_cache(
    *, workers: int, chunk_size: int, sample: int, force: bool = False
) -> dict[str, Any]:
    t0 = time.perf_counter()
    info = DATA.load_official()
    build = DATA.build_shards(
        info,
        rows=list(range(int(info["n"]))),
        shard_dir=C.CACHE / "shards",
        workers=int(workers),
        chunk_size=int(chunk_size),
    )
    shard_paths = [C.REPO / s["path"] for s in build["shards"] if s["path"]]
    from mb2_data import frozen_spec
    from feature_spec import load_spec

    spec = load_spec(C.ANALYSIS / "pcba_spec.json")
    if spec.hash() != frozen_spec().hash():
        raise RuntimeError("cache spec hash is not the frozen lipo_v1 hash")
    merged = DATA.merge_shards(shard_paths, spec, out_dir=C.CACHE / "pcba_v1", split="all")
    C.log(f"[cache] merged {merged['n_molecules']} molecules, s2={merged['n_s2_paths']}")
    return stage_postmerge(build=build, merged=merged, spec=spec, sample=int(sample), t0=t0)


def stage_postmerge(
    *, build: Mapping[str, Any] | None = None, merged: Mapping[str, Any] | None = None,
    spec=None, sample: int, t0: float | None = None,
) -> dict[str, Any]:
    """Manifest + label-free / structure / S2-identity / historical audits."""
    from feature_spec import load_spec
    from mb2_data import frozen_spec

    t0 = time.perf_counter() if t0 is None else t0
    if spec is None:
        spec = load_spec(C.ANALYSIS / "pcba_spec.json")
        if spec.hash() != frozen_spec().hash():
            raise RuntimeError("cache spec hash is not the frozen lipo_v1 hash")
    if merged is None:
        prior_path = C.ANALYSIS / "cache_manifest.json"
        if prior_path.exists():
            prior = C.read_json(prior_path)["merged"]
            merged = {k: prior.get(k) for k in (
                "cache", "cache_sha256", "cache_bytes", "cache_files", "n_molecules",
                "n_s2_paths", "n_atoms_total", "n_bonds_total", "merge_seconds", "feature_spec_hash",
            )}
        else:  # resume after an interrupted merge: rebuild the summary from disk
            d = C.CACHE / "pcba_v1"
            meta = C.read_json(d / "meta.json")
            files = [f for f in d.glob("*") if f.is_file()]
            merged = {
                "cache": C.track_rel(d),
                "cache_format": meta["format"],
                "cache_sha256": C.sha256_tree(d),
                "cache_bytes": int(sum(f.stat().st_size for f in files)),
                "cache_files": len(files),
                "n_molecules": int(meta["n_molecules"]),
                "n_s2_paths": int(meta["n_s2_paths"]),
                "n_atoms_total": int(meta["n_atoms_total"]),
                "n_bonds_total": int(meta["n_bonds_total"]),
                "merge_seconds": None,
                "feature_spec_hash": meta["feature_spec_hash"],
                "summary_reconstructed_from_disk": True,
            }
            C.log("[postmerge] merged summary reconstructed from on-disk cache")
    if build is None:
        build_path = C.ANALYSIS / ("cache_manifest.json" if (C.ANALYSIS / "cache_manifest.json").exists() else "pcba_cache_build.json")
        build = C.read_json(build_path)["build"] if "manifest" in build_path.name else C.read_json(build_path)
    manifest = {
        "schema": "lcrc-pcba-cache-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "spec": {"path": C.track_rel(C.ANALYSIS / "pcba_spec.json"), "hash": spec.hash()},
        "build": build,
        "merged": merged,
        "labels": {
            "path": C.track_rel(C.ANALYSIS / "pcba_labels_trainvalid.npy"),
            "sha256": C.sha256_file(C.ANALYSIS / "pcba_labels_trainvalid.npy"),
        },
        "split": {
            "path": C.track_rel(C.ANALYSIS / "pcba_split.json"),
            "partition_sha256": C.read_json(C.ANALYSIS / "pcba_split.json")["partition_sha256"],
        },
    }
    C.write_json(C.ANALYSIS / "cache_manifest.json", manifest)

    if C.sha256_tree(C.CACHE / "pcba_v1") != merged["cache_sha256"]:
        raise RuntimeError("cache bytes changed between merge and audit")
    cache = DATA.load_cache(C.CACHE / "pcba_v1")
    rng = np.random.default_rng(0)
    sample_rows = sorted(int(x) for x in rng.choice(int(cache["n_molecules"]), size=int(sample), replace=False))
    audit = DATA.feature_cache_audit(cache, spec, sample_rows=sample_rows, build_summary=build)
    hist = DATA.historical_s2_crosscheck(cache, spec)
    del cache
    audit["historical_crosscheck"] = hist
    audit["gates"]["historical_s2_definition_unchanged"] = hist.get("status") == "OK"
    audit["stage_seconds"] = float(time.perf_counter() - t0)
    audit["status"] = "OK" if all(audit["gates"].values()) else "FAILED"
    C.write_json(C.ANALYSIS / "feature_cache_audit.json", audit)
    C.log(f"[cache] feature audit status={audit['status']} gates={audit['gates']}")
    if audit["status"] != "OK":
        raise RuntimeError(f"feature cache audit failed: {audit['gates']}")
    return {"build": build, "merged": merged, "audit_status": audit["status"]}


def stage_verify() -> dict[str, Any]:
    """Host-side verification of a transferred cache (no RDKit, no download)."""
    manifest = C.read_json(C.ANALYSIS / "cache_manifest.json")
    t0 = time.perf_counter()
    cache_path = C.REPO / manifest["merged"]["cache"]
    labels_path = C.REPO / manifest["labels"]["path"]
    cache_sha = C.sha256_tree(cache_path) if cache_path.is_dir() else C.sha256_file(cache_path)
    labels_sha = C.sha256_file(labels_path)
    from feature_spec import load_spec

    spec = load_spec(C.REPO / manifest["spec"]["path"])
    split = C.read_json(C.REPO / manifest["split"]["path"])
    test_rows = np.asarray(split["rows"]["test"], dtype=np.int64)
    labels = np.load(labels_path)
    cache = DATA.load_cache(cache_path)
    structure = DATA.cache_structure_audit(cache, spec)
    payload = {
        "schema": "lcrc-pcba-cache-host-verify-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "hostname": C.environment()["hostname"],
        "git_commit": C.git_commit(),
        "cache": C.track_rel(cache_path),
        "cache_sha256": cache_sha,
        "cache_sha256_matches_manifest": cache_sha == manifest["merged"]["cache_sha256"],
        "labels_sha256": labels_sha,
        "labels_sha256_matches_manifest": labels_sha == manifest["labels"]["sha256"],
        "spec_hash_matches_manifest": spec.hash() == manifest["spec"]["hash"],
        "split_partition_matches_manifest": split["partition_sha256"]
        == manifest["split"]["partition_sha256"],
        "n_molecules": int(cache["n_molecules"]),
        "n_molecules_matches_manifest": int(cache["n_molecules"]) == int(manifest["merged"]["n_molecules"]),
        "n_s2_paths": int(cache["path_ptr"][-1]),
        "n_s2_paths_matches_manifest": int(cache["path_ptr"][-1]) == int(manifest["merged"]["n_s2_paths"]),
        "labels_shape": list(labels.shape),
        "test_rows_all_nan": bool(np.isnan(labels[test_rows]).all()),
        "label_free_cache": DATA.cache_label_free(cache)["ok"],
        "structure_ok": structure["ok"],
        "s2_formula_identity": structure["formula_s2"]["ok"],
        "structure": structure,
        "verify_seconds": float(time.perf_counter() - t0),
    }
    del cache
    gates = [
        payload["cache_sha256_matches_manifest"],
        payload["labels_sha256_matches_manifest"],
        payload["spec_hash_matches_manifest"],
        payload["split_partition_matches_manifest"],
        payload["n_molecules_matches_manifest"],
        payload["n_s2_paths_matches_manifest"],
        payload["test_rows_all_nan"],
        payload["label_free_cache"],
        payload["structure_ok"],
        payload["s2_formula_identity"],
    ]
    payload["gates_ok"] = bool(all(gates))
    payload["status"] = "OK" if payload["gates_ok"] else "FAILED"
    C.write_json(C.ANALYSIS / "cache_host_verify.json", payload)
    C.log(f"[verify] status={payload['status']} sha_match={payload['cache_sha256_matches_manifest']}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("audit", "cache", "postmerge", "verify", "all")
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--sample", type=int, default=int(C.S2_AUDIT_SAMPLE))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("audit", "all"):
            stage_audit(force=args.force)
        if args.stage in ("cache", "all"):
            stage_cache(workers=args.workers, chunk_size=args.chunk_size, sample=args.sample, force=args.force)
        if args.stage in ("postmerge", "all"):
            stage_postmerge(sample=args.sample)
        if args.stage in ("verify", "all"):
            stage_verify()
        return 0
    except Exception as exc:  # noqa: BLE001 - recorded
        C.write_json(
            C.ANALYSIS / f"stage_{args.stage}_failure.json",
            {
                "stage": args.stage,
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "utc": C.utcnow(),
                "environment": C.environment(),
            },
        )
        C.log(f"STAGE FAILED {args.stage}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
