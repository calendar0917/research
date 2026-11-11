"""Full-Train ``lipo_v1`` cache: audit, rebuild, merge, prefix control.

Policy (PROTOCOL §2–§3): the full 3,378,606-molecule cache is a **reliable full
rebuild** in the frozen seed-42 permutation order, not a prefix injection.
The historical 1M cache is the independent control: after the merge, every
field of the new cache's first 1,000,000 rows must be bitwise equal to it.

A failed SMILES row blocks the build (no silent drop); strict vs fallback
parsing is recorded per row in the parse ledger.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fs_common as C
import fs_data as D

PCD = C.PQ_DATA.PCD
FS = C.PQ_DATA.FS

BUILD_SUMMARY = C.ANALYSIS / "fs_full_cache_build.json"
PARSE_LEDGER = C.ANALYSIS / "fs_full_cache_parse_ledger.json"
MERGE_SUMMARY = C.ANALYSIS / "fs_full_cache_merge.json"
PREFIX_REPORT = C.ANALYSIS / "fs_cache_prefix_control.json"
DATA_AUDIT = C.ANALYSIS / "fs_data_audit.json"

SHARD_DIR = C.CACHE / "shards_full"
VALUE_FIELDS = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_code",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "global",
    "n_components",
    "n_junctions",
    "n_ring_systems",
)


# ---------------------------------------------------------------------------
# stage 1 — official data identity + full row plan
# ---------------------------------------------------------------------------


def stage_data(*, verify_ogb: bool = True) -> dict[str, Any]:
    split_payload = D.load_split_dict(verify_with_ogb=verify_ogb)
    plan = D.full_row_plan(split=split_payload["split"])
    info = C.PQ_DATA.load_official()
    data_audit = C.PQ_DATA.data_audit(info)
    subset_plan = C.read_json(C.PARENT_ANALYSIS / "pcqm_subset_plan.json")
    payload = {
        "schema": "lcrc-pcqm4mv2-fullscale-data-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "installed_dataset": {
            "ogb_version": C.PQ.host_facts().get("ogb"),
            "n_total_rows": int(info["n"]),
            "release_files": info["release_files"],
            "raw_files": info["files"],
            "note": (
                "the installed OGB data version is authoritative; no row is added or "
                "dropped to match a public web-page total"
            ),
        },
        "split_identity": {k: v for k, v in split_payload.items() if k != "split"},
        "split_sizes": {k: int(v.size) for k, v in split_payload["split"].items()},
        "row_plan": {k: v for k, v in plan.items() if k != "_rows"},
        "parent_subset_plan": {
            "round": subset_plan["round_id"],
            "sample_seed": subset_plan["sample_seed"],
            "official_train_size": subset_plan["official_train_size"],
            "subset_1m_sha256": subset_plan["subsets"]["PCQM_W2_1M"]["rows_sha256"],
            "permutation_rows_sha256_full": subset_plan["permutation_rows_sha256_full"],
        },
        "data_audit": data_audit,
        "checks": {
            "split_sizes_ok": all(split_payload["size_checks"].values()),
            "row_plan_ok": plan["status"] == "OK",
            "row_plan_sha_matches_parent": (
                plan["rows_sha256"] == subset_plan["permutation_rows_sha256_full"]
            ),
            "data_audit_ok": data_audit["status"] == "OK",
            "official_splits_partition_dataset": bool(split_payload["partition_ok"]),
        },
    }
    payload["status"] = "OK" if all(payload["checks"].values()) else "FAILED"
    C.write_json(DATA_AUDIT, payload)
    C.log(f"[data] {payload['status']} splits={payload['split_sizes']} rows={plan['n_rows']}")
    if payload["status"] != "OK":
        raise RuntimeError(f"data audit failed: {payload['checks']}")
    return payload


# ---------------------------------------------------------------------------
# stage 2 — chunked feature build (all 3,378,606 rows)
# ---------------------------------------------------------------------------


def _sum_counts(results: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    width = max(len(res[key]) for res in results)
    total = np.zeros(width, dtype=np.int64)
    for res in results:
        total[: len(res[key])] += np.asarray(res[key], dtype=np.int64)
    return total.tolist()


def _concat_ints(results: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    parts = [np.asarray(res[key], dtype=np.int64) for res in results if len(res[key])]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def build_shards_full(
    *,
    rows: np.ndarray,
    smiles_all: Sequence[str],
    shard_dir: Path,
    tag: str = "pcqm_train_full",
    workers: int = 32,
    chunk_size: int = 4000,
    summary_path: Path | None = None,
    ledger_path: Path | None = None,
    failures_path: Path | None = None,
) -> dict[str, Any]:
    """Build every requested row, writing summaries into this round's analysis."""
    summary_path = Path(summary_path) if summary_path else BUILD_SUMMARY
    ledger_path = Path(ledger_path) if ledger_path else PARSE_LEDGER
    failures_path = Path(failures_path) if failures_path else C.ANALYSIS / "fs_full_cache_adapter_failures.json"
    spec = FS.lipo_v1_spec()
    spec_path = C.PARENT_ANALYSIS / "pcqm_spec.json"
    FS.save_spec(spec, spec_path)
    rows = [int(r) for r in rows]
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate rows in build selection")
    if min(rows) < 0 or max(rows) >= len(smiles_all):
        raise ValueError("row outside the official dataset")
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(shard_dir.glob("shard_*.pt")):
        stale.unlink()
    tasks = [
        (i // chunk_size, rows[i : i + chunk_size], [smiles_all[r] for r in rows[i : i + chunk_size]])
        for i in range(0, len(rows), chunk_size)
    ]
    C.log(f"[cache:{tag}] {len(rows)} rows in {len(tasks)} chunks, workers={workers}")
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if workers <= 1:
        C.PQ_DATA._init_worker(str(spec_path), str(shard_dir))
        for task in tasks:
            results.append(C.PQ_DATA._build_chunk(task))
            C.log(f"[cache:{tag}] chunk {results[-1]['shard']} done ({results[-1]['seconds']:.1f}s)")
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=mp.get_context("spawn"),
            initializer=C.PQ_DATA._init_worker,
            initargs=(str(spec_path), str(shard_dir)),
        ) as pool:
            for res in pool.map(C.PQ_DATA._build_chunk, tasks):
                results.append(res)
                if len(results) % 50 == 0:
                    C.log(f"[cache:{tag}] chunk {res['shard']}/{len(tasks) - 1} built")
    failures = [f for res in results for f in res["failures"]]
    fallback_rows = [r for res in results for r in res["fallback_rows"]]
    summary = {
        "schema": C.SCHEMA_BUILD,
        "round_id": C.ROUND_ID,
        "tag": str(tag),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "feature_spec": C.track_rel(spec_path),
        "feature_spec_hash": spec.hash(),
        "n_rows_requested": len(rows),
        "n_rows_built": int(sum(res["n_built"] for res in results)),
        "rows_sha256": C.index_hash(rows),
        "shard_size": int(chunk_size),
        "workers": int(workers),
        "wall_seconds": float(time.perf_counter() - t0),
        "parse_modes": {
            key: int(sum(res["modes"][key] for res in results))
            for key in ("strict", "unsanitized_fallback")
        },
        "n_fallback_rows": int(len(fallback_rows)),
        "n_failures": int(len(failures)),
        "n_atoms_total": int(_concat_ints(results, "n_atoms").sum()),
        "n_bonds_total": int(_concat_ints(results, "n_bonds").sum()),
        "s2_paths_total": int(_concat_ints(results, "s2_paths").sum()),
        "n_atoms": PCD._dist(_concat_ints(results, "n_atoms")),
        "n_bonds": PCD._dist(_concat_ints(results, "n_bonds")),
        "s2_paths": PCD._dist(_concat_ints(results, "s2_paths")),
        "max_s2_records_per_molecule": int(_concat_ints(results, "s2_paths").max()),
        "n_multi_component_molecules": int((_concat_ints(results, "n_components") > 1).sum()),
        "raw_vocabulary": {
            "atom_class_counts": {str(i): int(v) for i, v in enumerate(_sum_counts(results, "atom_class_counts"))},
            "bond_class_counts": {str(i): int(v) for i, v in enumerate(_sum_counts(results, "bond_class_counts"))},
            "charge_counts": _sum_counts(results, "charge_counts"),
            "h_counts": _sum_counts(results, "h_counts"),
            "stereo_counts": _sum_counts(results, "stereo_counts"),
        },
        "shards": [
            {
                "shard": int(res["shard"]),
                "path": C.track_rel(res["path"]),
                "sha256": res["sha256"],
                "n_built": int(res["n_built"]),
                "first_row": int(res["rows"][0]),
                "last_row": int(res["rows"][-1]),
                "seconds": float(res["seconds"]),
            }
            for res in results
        ],
        "status": "OK" if not failures else "BLOCKED_ADAPTER_FAILURES",
    }
    if failures:
        C.write_json(
            failures_path,
            {"schema": "lcrc-pcqm4mv2-fullscale-adapter-failures-v1", "failures": failures},
        )
        C.write_json(summary_path, summary)
        raise C.PQ_DATA.InputAdapterBlocked(
            f"{len(failures)} official SMILES rows failed the frozen adapter; see "
            f"{C.track_rel(failures_path)}"
        )
    C.write_json(
        ledger_path,
        {
            "schema": "lcrc-pcqm4mv2-fullscale-parse-ledger-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "git_commit": C.git_commit(),
            "tag": str(tag),
            "n_rows": len(rows),
            "n_strict": int(sum(res["modes"]["strict"] for res in results)),
            "n_unsanitized_fallback": int(sum(res["modes"]["unsanitized_fallback"] for res in results)),
            "n_failures": 0,
            "policy": (
                "one documented fallback (sanitize=False + UpdatePropertyCache(strict=False)) "
                "only after strict sanitisation failed; every affected row is listed, no row "
                "is dropped or renumbered"
            ),
            "fallback_rows": sorted(fallback_rows, key=lambda r: r["row"]),
        },
    )
    summary["parse_ledger"] = C.track_rel(ledger_path)
    C.write_json(summary_path, summary)
    C.log(
        f"[cache:{tag}] {summary['status']} built={summary['n_rows_built']} "
        f"strict={summary['parse_modes']['strict']} "
        f"fallback={summary['parse_modes']['unsanitized_fallback']} "
        f"{summary['wall_seconds']:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# stage 3 — merge + prefix control
# ---------------------------------------------------------------------------


def _compare_jsonl_prefix(new_path: Path, old_path: Path, n: int) -> dict[str, Any]:
    mismatches = 0
    first_mismatch = None
    with open(new_path, encoding="utf-8") as fh_new, open(old_path, encoding="utf-8") as fh_old:
        for i in range(int(n)):
            a = fh_new.readline()
            b = fh_old.readline()
            if not b:
                raise RuntimeError(f"{old_path} ended before {n} lines")
            if a != b:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {"line": int(i), "new": a[:200], "old": b[:200]}
    return {"scope": f"first {n} lines", "mismatch_lines": int(mismatches), "first_mismatch": first_mismatch}


def verify_prefix_control(*, full_dir: Path, old_dir: Path) -> dict[str, Any]:
    """Every field of the new cache's first 1M rows vs the historical cache."""
    t0 = time.perf_counter()
    old = D.load_cache(old_dir, with_smiles=False)
    new = D.load_cache(full_dir, with_smiles=False)
    n_old = int(old["n_molecules"])
    if n_old != 1_000_000:
        raise RuntimeError(f"control cache has {n_old} rows != 1,000,000")
    old_atoms = int(np.asarray(old["atom_ptr"])[-1])
    old_bonds = int(np.asarray(old["bond_ptr"])[-1])
    old_s1 = int(np.asarray(old["nbr_ptr"])[-1])
    old_s2 = int(np.asarray(old["path_ptr"])[-1])
    fields: dict[str, Any] = {}
    ok = True
    for name in ("atom_ptr", "bond_ptr", "nbr_ptr", "path_ptr"):
        a = np.asarray(new[name])
        b = np.asarray(old[name])
        limit = {
            "atom_ptr": n_old + 1,
            "bond_ptr": n_old + 1,
            "nbr_ptr": old_atoms + 1,
            "path_ptr": old_atoms + 1,
        }[name]
        same = bool(np.array_equal(a[:limit], b))
        fields[name] = {"compared": int(limit), "equal": same}
        ok = ok and same
    for name in VALUE_FIELDS:
        a = np.asarray(new[name])
        b = np.asarray(old[name])
        sliced = a[tuple(slice(0, s) for s in b.shape)]
        same = bool(sliced.shape == b.shape and np.array_equal(sliced, b))
        fields[name] = {"shape": list(b.shape), "equal": same}
        ok = ok and same
    gids_same = bool(np.array_equal(np.asarray(new["gids"])[:n_old], np.asarray(old["gids"])))
    ok = ok and gids_same
    smiles_cmp = _compare_jsonl_prefix(full_dir / "smiles.jsonl", old_dir / "smiles.jsonl", n_old)
    ids_cmp = _compare_jsonl_prefix(full_dir / "molecule_ids.jsonl", old_dir / "molecule_ids.jsonl", n_old)
    ok = ok and smiles_cmp["mismatch_lines"] == 0 and ids_cmp["mismatch_lines"] == 0

    # collated batches from the prefix, including the 1M boundary
    rng = np.random.default_rng(20261111)
    sample = np.unique(
        np.concatenate(
            [
                np.arange(0, 5),
                np.arange(n_old - 5, n_old),
                rng.choice(n_old, size=20, replace=False),
            ]
        )
    )
    batch_checks = []
    for size in (1, 2, 5, 128):
        rows = sample[: min(size, sample.size)].tolist()
        b_new = C.EW_COLLATE.collate_v2(new, rows)
        b_old = C.EW_COLLATE.collate_v2(old, rows)
        cmp = C.EW_COLLATE.compare_batches(b_new, b_old)
        batch_checks.append({"batch_size": len(rows), "rows": rows, "all_equal": cmp["all_equal"]})
        ok = ok and bool(cmp["all_equal"])
    report = {
        "schema": "lcrc-pcqm4mv2-fullscale-prefix-control-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "status": "OK" if ok else "FAILED",
        "new_cache": C.track_rel(full_dir),
        "control_cache": C.track_rel(old_dir),
        "control_rows": n_old,
        "fields": fields,
        "gids_equal": gids_same,
        "smiles": smiles_cmp,
        "molecule_ids": ids_cmp,
        "collated_batches": batch_checks,
        "all_equal": bool(ok),
        "seconds": float(time.perf_counter() - t0),
    }
    C.write_json(PREFIX_REPORT, report)
    return report


def stage_merge(*, keep_shards: bool = False, reuse_cache: bool = False) -> dict[str, Any]:
    build = C.read_json(BUILD_SUMMARY)
    if build.get("status") != "OK":
        raise RuntimeError(f"build status {build.get('status')} != OK")
    spec = FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json")
    shard_paths = [C.REPO / s["path"] for s in build["shards"]]
    full_dir = C.CACHE / D.FULL_CACHE_NAME
    if full_dir == C.PARENT_CACHE / "pcqm_train1m":
        raise RuntimeError("refusing to write over the historical 1M cache")
    if reuse_cache:
        if not (full_dir / "meta.json").is_file():
            raise FileNotFoundError(f"no merged cache to reuse: {full_dir}")
        meta = json.loads((full_dir / "meta.json").read_text(encoding="utf-8"))
        merged = {
            "n_molecules": int(meta["n_molecules"]),
            "n_atoms_total": int(meta["n_atoms_total"]),
            "n_bonds_total": int(meta["n_bonds_total"]),
            "n_s1_codes": int(meta["n_s1_codes"]),
            "n_s2_paths": int(meta["n_s2_paths"]),
            "feature_spec_hash": str(meta["feature_spec_hash"]),
            "merge_seconds": float(meta.get("merge_seconds", 0.0) or 0.0),
        }
        C.log(f"[merge:full] reusing existing merged cache {C.track_rel(full_dir)}")
    else:
        missing = [str(p) for p in shard_paths if not p.is_file()]
        if missing:
            raise RuntimeError(f"missing shards: {missing[:3]} (+{max(0, len(missing) - 3)})")
        C.log(f"[merge:full] {len(shard_paths)} shards -> {C.track_rel(full_dir)}")
        merged = C.PQ_DATA.merge_shards(
            shard_paths, spec, out_dir=full_dir, split="all", dataset=C.DATASET, tag="pcqm_train_full"
        )
    meta_path = full_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "round_id": C.ROUND_ID,
            "tag": "pcqm_train_full",
            "source": "official OGB-LSC pcqm4m-v2 raw data.csv.gz -> frozen lipo_v1 adapter",
            "row_plan": {
                "permutation": C.track_rel(C.PARENT_CACHE / "pcqm_permutation.npy"),
                "rows_sha256": build["rows_sha256"],
                "order": "seed-42 permutation of the official Train rows",
            },
            "build_summary": C.track_rel(BUILD_SUMMARY),
            "merge_note": "meta rewritten after merge for round provenance; arrays untouched",
        }
    )
    meta_path.write_text(_json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    tree_sha = C.sha256_tree(full_dir)
    files = sorted(p for p in full_dir.glob("*") if p.is_file())
    prefix = verify_prefix_control(full_dir=full_dir, old_dir=C.PARENT_CACHE / "pcqm_train1m")
    if prefix["status"] != "OK":
        raise RuntimeError(
            "BLOCKED_CACHE_PREFIX_MISMATCH: the rebuilt full cache does not reproduce the "
            f"historical 1M cache; see {C.track_rel(PREFIX_REPORT)}"
        )
    valid_manifest = C.read_json(C.PARENT_ANALYSIS / "pcqm_cache_manifest.json")
    manifest = {
        "schema": "lcrc-pcqm4mv2-fullscale-cache-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "cache_path": C.track_rel(full_dir),
        "cache_sha256": tree_sha,
        "cache_bytes": int(sum(f.stat().st_size for f in files)),
        "file_bytes": {p.name: int(p.stat().st_size) for p in files},
        "n_molecules": int(merged["n_molecules"]),
        "n_atoms_total": int(merged["n_atoms_total"]),
        "n_bonds_total": int(merged["n_bonds_total"]),
        "n_s1_codes": int(merged["n_s1_codes"]),
        "n_s2_paths": int(merged["n_s2_paths"]),
        "feature_spec_hash": merged["feature_spec_hash"],
        "rows_sha256": build["rows_sha256"],
        "labels_present": False,
        "uses_3d_geometry": False,
        "valid_tree_sha256": str(valid_manifest["valid"]["cache_sha256"]),
        "prefix_control": {
            "cache": C.track_rel(C.PARENT_CACHE / "pcqm_train1m"),
            "cache_sha256": str(valid_manifest["train1m"]["cache_sha256"]),
            "n_molecules": 1_000_000,
            "status": prefix["status"],
            "report": C.track_rel(PREFIX_REPORT),
            "fields_equal": bool(prefix["all_equal"]),
        },
        "build_summary": C.track_rel(BUILD_SUMMARY),
        "parse_ledger": C.track_rel(PARSE_LEDGER),
        "n_failures": int(build["n_failures"]),
        "n_fallback_rows": int(build["n_fallback_rows"]),
        "merge_seconds": float(merged["merge_seconds"]),
        "shards_removed": False,
        "status": "OK",
    }
    C.write_json(D.FULL_MANIFEST, manifest)
    merge_summary = {
        "schema": "lcrc-pcqm4mv2-fullscale-merge-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "merged": merged,
        "cache_sha256_after_meta_rewrite": tree_sha,
        "prefix_control_status": prefix["status"],
        "manifest": C.track_rel(D.FULL_MANIFEST),
    }
    C.write_json(MERGE_SUMMARY, merge_summary)
    if not keep_shards:
        import shutil

        freed = sum(p.stat().st_size for p in SHARD_DIR.glob("*.pt"))
        shutil.rmtree(SHARD_DIR)
        manifest["shards_removed"] = True
        manifest["shards_bytes_freed"] = int(freed)
        C.write_json(D.FULL_MANIFEST, manifest)
        C.log(f"[merge:full] shards removed ({freed / 1e9:.1f} GB)")
    C.log(
        f"[merge:full] OK rows={manifest['n_molecules']} cache={tree_sha[:12]} "
        f"prefix={prefix['status']} bytes={manifest['cache_bytes'] / 1e9:.2f} GB"
    )
    return manifest


def stage_check() -> dict[str, Any]:
    """Read-only summary of what exists (used before launching anything)."""
    out: dict[str, Any] = {"utc": C.utcnow(), "cache": C.track_rel(C.CACHE)}
    full_dir = C.CACHE / D.FULL_CACHE_NAME
    out["full_cache_exists"] = bool((full_dir / "meta.json").is_file())
    out["manifest_exists"] = D.FULL_MANIFEST.is_file()
    if out["full_cache_exists"]:
        cache = D.load_cache(full_dir)
        out["n_molecules"] = int(cache["n_molecules"])
        out["n_atoms"] = int(np.asarray(cache["atom_ptr"])[-1])
        out["n_s2_paths"] = int(np.asarray(cache["path_ptr"])[-1])
    out["shards_exist"] = SHARD_DIR.is_dir()
    if SHARD_DIR.is_dir():
        out["shard_files"] = len(list(SHARD_DIR.glob("shard_*.pt")))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("data", "build", "merge", "verify", "check", "all"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true", help="finalise an existing merged cache without re-merging")
    parser.add_argument("--no-verify-ogb", action="store_true")
    args = parser.parse_args(argv)

    if args.stage in ("data", "all"):
        stage_data(verify_ogb=not args.no_verify_ogb)
    if args.stage in ("build", "all"):
        audit = C.read_json(DATA_AUDIT)
        if audit.get("status") != "OK":
            raise RuntimeError("data audit missing or failed")
        plan = D.full_row_plan()
        info = C.PQ_DATA.load_official()
        summary = build_shards_full(
            rows=np.asarray(plan["_rows"], dtype=np.int64),
            smiles_all=info["smiles"],
            shard_dir=SHARD_DIR,
            workers=int(args.workers),
            chunk_size=int(args.chunk_size),
        )
        C.log(f"[build] {summary['status']} {summary['n_rows_built']} rows in {summary['wall_seconds']:.1f}s")
    if args.stage in ("merge", "all"):
        manifest = stage_merge(keep_shards=bool(args.keep_shards), reuse_cache=bool(args.reuse_cache))
        C.log(f"[merge] {manifest['status']} {manifest['cache_sha256'][:12]}")
    if args.stage in ("verify", "all"):
        full_dir = C.CACHE / D.FULL_CACHE_NAME
        prefix = verify_prefix_control(full_dir=full_dir, old_dir=C.PARENT_CACHE / "pcqm_train1m")
        manifest = C.read_json(D.FULL_MANIFEST)
        cache = D.load_cache(full_dir)
        plan = D.full_row_plan()
        identity = D.validate_full_cache_identity(cache, plan)
        audit = C.PQ_DATA.cache_structure_audit(cache, C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json"))
        out = {
            "schema": C.SCHEMA_VERIFY,
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "git_commit": C.git_commit(),
            "manifest_status": manifest["status"],
            "manifest_rows_sha256": manifest["rows_sha256"],
            "identity": identity,
            "prefix_control_status": prefix["status"],
            "structure_audit": audit,
            "cache_sha256_live": C.sha256_tree(full_dir),
            "status": "OK"
            if (
                prefix["status"] == "OK"
                and identity["status"] == "OK"
                and audit.get("status") in (None, "OK")
                and C.sha256_tree(full_dir) == manifest["cache_sha256"]
            )
            else "FAILED",
        }
        C.write_json(C.ANALYSIS / "fs_cache_verify.json", out)
        C.log(f"[verify] {out['status']} prefix={prefix['status']} sha={out['cache_sha256_live'][:12]}")
    if args.stage == "check":
        C.log(str(stage_check()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
