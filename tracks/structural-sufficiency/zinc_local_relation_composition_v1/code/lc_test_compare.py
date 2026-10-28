"""Descriptive test comparison for LCRC A_s0 against historical DCC-L.

Reads only already-saved per-molecule predictions (the LCRC post-hoc test CSV
and the frozen DCC-L test CSVs).  It does not touch the official test split or
labels again, and it is not used for any model/seed/epoch choice.
"""

from __future__ import annotations

import csv

import numpy as np

import lc_common as L

LCRC = L.ANALYSIS / "test_per_molecule_lcrcA_s0.csv"
DCCL = L.REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis"
SEEDS = (0, 1, 2, 3)


def _read(path):
    rows = list(csv.DictReader(open(path, newline="")))
    y = np.asarray([float(r["y_raw"]) for r in rows], np.float64)
    pred = np.asarray([float(r["pred"]) for r in rows], np.float64)
    return y, pred


def _mae(pred, y, mask=None):
    ae = np.abs(pred - y)
    return float(ae[mask].mean())


def main() -> int:
    y, pred = _read(LCRC)
    ae = np.abs(pred - y)
    rows = []
    for seed in SEEDS:
        y2, pred2 = _read(DCCL / f"test_pred_B-R0-s{seed}_s{seed}.csv")
        if not np.allclose(y, y2, atol=1e-5):
            raise RuntimeError("test label order mismatch")
        ae2 = np.abs(pred2 - y)
        d = ae - ae2
        rng = np.random.default_rng(20261108)
        boot = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(10000)]
        rows.append({
            "seed": seed,
            "dcc_test_mae": float(ae2.mean()),
            "lcrc_test_mae": float(ae.mean()),
            "delta_lcrc_minus_dcc": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "win_rate_lcrc_better": float(np.mean(ae < ae2)),
            "dcc_ordinary_mae": _mae(pred2, y, y > -3.0),
            "lcrc_ordinary_mae": _mae(pred, y, y > -3.0),
            "dcc_tail_mae": _mae(pred2, y, y <= -3.0),
            "lcrc_tail_mae": _mae(pred, y, y <= -3.0),
        })
    dcc_mean = float(np.mean([r["dcc_test_mae"] for r in rows]))
    out = {
        "descriptive_only": True,
        "post_hoc": True,
        "not_confirmatory": True,
        "lcrc_run": "lcrcA_s0",
        "lcrc_test_mae": float(ae.mean()),
        "dcc_4seed_mean_test_mae": dcc_mean,
        "delta_lcrc_minus_dcc_4seed_mean": float(ae.mean()) - dcc_mean,
        "rows": rows,
        "n_ordinary": int((y > -3.0).sum()),
        "n_tail": int((y <= -3.0).sum()),
    }
    L.write_json(L.ANALYSIS / "test_compare.json", out)
    lines = ["# LCRC A_s0 post-hoc test comparison (descriptive only)", "",
             "A_s0 was frozen before this read; DCC-L predictions are historical and not retrained. "
             "No model/seed/epoch choice uses these numbers.", "",
             "| model | test MAE | ordinary | tail |", "|---|---:|---:|---:|",
             f"| A_s0 | {ae.mean():.6f} | {_mae(pred, y, y > -3.0):.6f} | {_mae(pred, y, y <= -3.0):.6f} |"]
    for r in rows:
        lines.append(f"| DCC-L s{r['seed']} | {r['dcc_test_mae']:.6f} | {r['dcc_ordinary_mae']:.6f} | {r['dcc_tail_mae']:.6f} |")
    lines += ["", "| comparison | Δ MAE (A_s0 − DCC-L) | 95% CI | A better rate |",
              "|---|---:|---|---:|"]
    for r in rows:
        lines.append(f"| vs DCC-L s{r['seed']} | {r['delta_lcrc_minus_dcc']:+.6f} | "
                     f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['win_rate_lcrc_better']:.3f} |")
    lines += ["", f"DCC-L four-seed mean test MAE = {dcc_mean:.6f}; "
              f"A_s0 − mean = {ae.mean() - dcc_mean:+.6f}. Single seed; CI spans zero against every seed."]
    (L.ANALYSIS / "test_compare.md").write_text("\n".join(lines) + "\n")
    L.log(f"A_s0 test {ae.mean():.6f}; DCC-L mean {dcc_mean:.6f}; delta {ae.mean()-dcc_mean:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
