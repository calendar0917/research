"""Task-level concentration analysis of the epoch-40 per-task AP deltas.

Answers "are the gains broad or concentrated in a few assays?" without any
Test data.  Consumes the same run records as `ep_analyze.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _configure() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure()

import ep_common as C  # noqa: E402


def _load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _baseline(runs_dir: Path, historical: Path | None) -> tuple[list[float | None], list[int], list[int], str]:
    a0 = runs_dir / "A0" / "run.json"
    if a0.is_file():
        rep = _load(a0)["fixed_epoch_reports"]["40"]
        return list(rep["per_task_ap"]), list(rep["positive"]), list(rep["observed"]), "same_round_A0"
    if historical is None:
        raise SystemExit("no A0 run and no --historical-a0 given")
    rep = _load(historical / "run.json")["fixed_epoch_reports"]["40"]
    rows = [r for r in csv.DictReader(open(historical / "per_task_ap.csv")) if r["epoch"] == "40"]
    rows.sort(key=lambda r: int(r["task"]))
    tasks = [None if not r["ap"] else float(r["ap"]) for r in rows]
    if abs(float(rep["ap"]) - float(C.HISTORICAL_A0["valid_ap40"])) > 1e-9:
        raise RuntimeError("historical A0 run AP drifted")
    return tasks, [int(r["positive"]) for r in rows], [int(r["observed"]) for r in rows], "historical_phaseC_seed0_A"


def summarize(
    runs_dir: Path, historical: Path | None, out_path: Path
) -> dict[str, Any]:
    base, positive, observed, source = _baseline(runs_dir, historical)
    strata = np.asarray(
        ["lt5" if p < 5 else "s5_19" if p < 20 else "ge20" for p in positive], dtype=object
    )
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-explicit-performance-task-summary-v1",
        "round_id": C.ROUND_ID,
        "baseline": source,
        "baseline_valid_ap40": float(C.HISTORICAL_A0["valid_ap40"]) if source != "same_round_A0" else None,
        "arms": {},
        "test_evaluated": False,
    }
    for arm in ("A1", "A2", "A3", "COMBO"):
        path = runs_dir / arm / "run.json"
        if not path.is_file():
            continue
        rep = _load(path)["fixed_epoch_reports"]["40"]
        ap = np.asarray([np.nan if x is None else float(x) for x in rep["per_task_ap"]], dtype=float)
        b = np.asarray([np.nan if x is None else float(x) for x in base], dtype=float)
        ok = np.isfinite(ap) & np.isfinite(b)
        delta = np.where(ok, ap - b, np.nan)
        pos_delta = np.where(ok & (delta > 0), delta, 0.0)
        total_pos = float(pos_delta.sum())
        order = np.argsort(-pos_delta)
        top10 = float(pos_delta[order[:10]].sum())
        by_stratum = {}
        for s in ("ge20", "s5_19", "lt5"):
            m = ok & (strata == s)
            if not m.any():
                continue
            d = delta[m]
            by_stratum[s] = {
                "n_tasks": int(m.sum()),
                "mean_delta": float(np.nanmean(d)),
                "median_delta": float(np.nanmedian(d)),
                "n_improved": int((d > 1e-12).sum()),
                "n_degraded": int((d < -1e-12).sum()),
            }
        # Spearman correlation between delta and Validation support (observed).
        obs = np.asarray(observed, dtype=float)[ok]
        d_ok = delta[ok]
        if d_ok.size > 2 and np.std(d_ok) > 0 and np.std(obs) > 0:
            rho = float(np.corrcoef(np.argsort(np.argsort(d_ok)), np.argsort(np.argsort(obs)))[0, 1])
        else:
            rho = float("nan")
        out["arms"][arm] = {
            "n_scorable_tasks": int(ok.sum()),
            "mean_delta": float(np.nanmean(delta)),
            "median_delta": float(np.nanmedian(delta)),
            "n_improved": int((delta > 1e-12).sum()),
            "n_degraded": int((delta < -1e-12).sum()),
            "n_tied": int(((np.abs(delta) <= 1e-12) & ok).sum()),
            "improved_fraction": float((delta > 1e-12).sum() / max(1, ok.sum())),
            "total_positive_delta": total_pos,
            "top10_positive_delta_share": float(top10 / total_pos) if total_pos > 0 else None,
            "largest_gains": [
                {"task": int(i), "delta": float(delta[i]), "valid_positive": int(positive[i])}
                for i in order[:5] if delta[i] > 0
            ],
            "largest_regressions": [
                {"task": int(i), "delta": float(delta[i]), "valid_positive": int(positive[i])}
                for i in np.argsort(delta)[:5] if delta[i] < 0
            ],
            "by_stratum": by_stratum,
            "spearman_delta_vs_valid_observed": rho if np.isfinite(rho) else None,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--historical-a0", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = summarize(
        Path(args.runs_dir),
        Path(args.historical_a0) if args.historical_a0 else None,
        Path(args.out),
    )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("largest_gains", "largest_regressions", "by_stratum")} for k, v in out["arms"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
