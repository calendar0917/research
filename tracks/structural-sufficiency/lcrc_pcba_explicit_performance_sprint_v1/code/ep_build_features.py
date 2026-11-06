"""CLI for the two label-free feature caches (run on the remote CPU)."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _configure() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure()

import ep_common as C  # noqa: E402
import ep_features as F  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", required=True, choices=("physchem", "branch", "all"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ep-feature-build-v1",
        "round_id": C.ROUND_ID,
        "which": args.which,
        "workers": int(args.workers),
        "environment": C.environment(),
        "started_utc": C.utcnow(),
        "test_evaluated": False,
    }
    try:
        payload.update(F.run_builder(args.which, workers=int(args.workers)))
        payload["status"] = "COMPLETED"
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "FAILED"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["finished_utc"] = C.utcnow()
    C.write_json(out_path, payload)
    C.log(f"feature build status={payload['status']} -> {out_path}")
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
