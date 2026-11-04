"""Run the preflight gates and write ``analysis/gates.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

import wb_checks as CH
import wb_common as C


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "gates.json"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    gates = CH.run_all()
    # Preserve run/table checks when this is rerun after wb_analyze.py; otherwise
    # a preflight refresh would silently drop the formal-run gates.
    if out.exists():
        previous = C.read_json(out)
        for key in ("run_checks", "table_checks", "run"):
            if key in previous:
                gates[key] = previous[key]
        for key, value in previous.get("run_checks", {}).items():
            gates["checks"][f"run_{key}"] = bool(value)
        for key, value in previous.get("table_checks", {}).items():
            gates["checks"][f"tables_{key}"] = bool(value)
        gates["all_passed"] = bool(all(gates["checks"].values()))
    C.write_json(out, gates)
    C.log(f"gates all_passed={gates['all_passed']} -> {C.track_rel(out)}")
    for name, ok in gates["checks"].items():
        C.log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
