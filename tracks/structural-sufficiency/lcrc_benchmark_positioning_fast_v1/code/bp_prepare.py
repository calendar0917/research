"""Build label-free caches, folds/splits and precheck distributions."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import bp_common as C


def _run_tu(name: str) -> dict[str, Any]:
    import bp_tu as T

    t0 = time.perf_counter()
    try:
        stats = T.build_dataset(name)
        return {"dataset": name, "status": "CACHE_READY", "seconds": float(time.perf_counter() - t0), "stats": stats}
    except T.InputAdapterBlocked as exc:
        payload = {
            "dataset": name,
            "status": "INPUT_ADAPTER_BLOCKED",
            "reason": str(exc),
            "seconds": float(time.perf_counter() - t0),
        }
        C.write_json(C.ANALYSIS / f"{name}_blocked.json", payload)
        return payload
    except T.LargeGraphBlocked as exc:
        payload = {
            "dataset": name,
            "status": "RESOURCE_BLOCKED",
            "reason": "EXACT_S2_CACHE_BUDGET",
            "detail": str(exc),
            "seconds": float(time.perf_counter() - t0),
            "counts_file": C.track_rel(C.ANALYSIS / f"{name}_precheck_counts.json"),
        }
        C.write_json(C.ANALYSIS / f"{name}_blocked.json", payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = {
            "dataset": name,
            "status": "PREP_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": float(time.perf_counter() - t0),
        }
        C.write_json(C.ANALYSIS / f"{name}_blocked.json", payload)
        return payload


def _count_tu(name: str) -> dict[str, Any]:
    import bp_tu as T

    t0 = time.perf_counter()
    try:
        stats = T.precheck_counts(name)
        stats["seconds"] = float(time.perf_counter() - t0)
        C.write_json(C.ANALYSIS / f"{name}_precheck_counts.json", stats)
        return {"dataset": name, "status": "COUNTS_READY", "seconds": stats["seconds"], "s2_paths_total": stats["s2_paths_total"]}
    except Exception as exc:  # noqa: BLE001
        return {"dataset": name, "status": "COUNT_FAILED", "reason": f"{type(exc).__name__}: {exc}"}


def _run_mol(name: str) -> dict[str, Any]:
    import bp_mol as M

    t0 = time.perf_counter()
    try:
        stats = M.build(name)
        return {"dataset": name, "status": "CACHE_READY", "seconds": float(time.perf_counter() - t0), "stats": stats}
    except M.InputAdapterBlocked as exc:
        payload = {"dataset": name, "status": "INPUT_ADAPTER_BLOCKED", "reason": str(exc)}
        C.write_json(C.ANALYSIS / f"{name.replace('ogbg-', '')}_blocked.json", payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = {
            "dataset": name,
            "status": "PREP_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        C.write_json(C.ANALYSIS / f"{name.replace('ogbg-', '')}_blocked.json", payload)
        return payload


def _run_peptide() -> dict[str, Any]:
    import bp_peptides as P

    t0 = time.perf_counter()
    try:
        stats = P.build()
        return {"dataset": C.PEPTIDE_DATASET, "status": "CACHE_READY", "seconds": float(time.perf_counter() - t0), "stats": stats}
    except P.InputAdapterBlocked as exc:
        payload = {"dataset": C.PEPTIDE_DATASET, "status": "INPUT_ADAPTER_BLOCKED", "reason": str(exc)}
        C.write_json(C.ANALYSIS / "peptides_func_blocked.json", payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = {
            "dataset": C.PEPTIDE_DATASET,
            "status": "PREP_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        C.write_json(C.ANALYSIS / "peptides_func_blocked.json", payload)
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tu", nargs="*", default=[])
    parser.add_argument("--counts", nargs="*", default=[])
    parser.add_argument("--mol", nargs="*", default=[])
    parser.add_argument("--peptide", action="store_true")
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"generated_utc": C.utcnow(), "results": []}
    for name in args.counts:
        out["results"].append(_count_tu(name))
    for name in args.tu:
        out["results"].append(_run_tu(name))
    for name in args.mol:
        out["results"].append(_run_mol(name))
    if args.peptide:
        out["results"].append(_run_peptide())
    C.write_json(C.ANALYSIS / "prepare_summary.json", out)
    print(C.stable_json(out))
    return 0 if all(r.get("status") in ("CACHE_READY", "COUNTS_READY") for r in out["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
