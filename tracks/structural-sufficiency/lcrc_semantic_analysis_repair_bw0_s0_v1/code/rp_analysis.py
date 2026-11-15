"""One regenerating entry point for every repair artifact of this round.

``uv run python -m rp_analysis`` (or ``run_all()``) re-derives, from stored
predecessor artifacts plus the frozen caches, in a fixed order:

1. ``rp_source``   — input identity + independent metric recomputation
2. ``rp_support``  — corrected support / document-frequency statistics
3. ``rp_checks``   — focused checks on the repaired code paths
4. ``rp_explicit`` — corrected CPU Ridge / XGBoost coordinate
5. ``rp_replay``   — checkpoint replay, corrected gradient kernel, native export
6. ``rp_perform     — the two complete performance tables + bootstrap
7. ``rp_edit``     — corrected real edit response

It never launches a training run.  ``--sections`` selects a subset.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import rp_common as RP

SECTIONS: dict[str, Callable[[], Any]] = {}


def _register() -> None:
    import rp_checks as CK
    import rp_edit as ED
    import rp_explicit as EX
    import rp_init as IN
    import rp_perf as PF
    import rp_replay as RP2
    import rp_source as SC
    import rp_support as SU

    SECTIONS.update(
        {
            "source": SC.run,
            "support": SU.run,
            "checks": CK.run,
            "init": IN.run,
            "explicit": EX.run_all,
            "replay": RP2.run,
            "performance": PF.run,
            "edit": ED.run,
        }
    )


def run_all(sections: list[str] | None = None) -> dict[str, Any]:
    _register()
    names = list(sections or SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown sections {unknown}; known: {sorted(SECTIONS)}")
    results: dict[str, Any] = {}
    started = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        RP.log(f"[analysis] {name} ...")
        try:
            results[name] = SECTIONS[name]()
            results[name]["_seconds"] = round(time.perf_counter() - t0, 1)
        except Exception as exc:  # noqa: BLE001
            RP.log(f"[analysis] {name} FAILED: {type(exc).__name__}: {exc}")
            results[name] = {"error": f"{type(exc).__name__}: {exc}",
                             "_seconds": round(time.perf_counter() - t0, 1)}
    payload = {
        "schema": "lcrc-semantic-analysis-repair-analysis-run-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "seconds": round(time.perf_counter() - started, 1),
        "sections": results,
    }
    RP.write_json(RP.ANALYSIS / "analysis_run.json", payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="", help="comma-separated subset")
    args = ap.parse_args(argv)
    names = [s.strip() for s in args.sections.split(",") if s.strip()] or None
    payload = run_all(names)
    failed = [k for k, v in payload["sections"].items() if isinstance(v, dict) and "error" in v]
    print(f"sections={list(payload['sections'])} failed={failed} seconds={payload['seconds']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_all", "main", "SECTIONS"]
