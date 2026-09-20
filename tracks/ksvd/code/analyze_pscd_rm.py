#!/usr/bin/env python
"""PSCD-RM-v0 analysis — rate-performance plane + gain decomposition.

Reads the frozen local result directories (both git-ignored):

    results/pscd_rm/      (this round: FREQ-LOOSE, TM-RATE, rate match)
    results/pscd_tmdl/    (reused: FREQ, TM strong-reader summaries + audits)

and writes ``results/pscd_rm/ANALYSIS.json`` plus two figures.  Nothing here
retrains anything; it is a pure post-hoc summariser of the frozen artifacts.

Official ZINC ``test`` is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RM = REPO_ROOT / "tracks/ksvd/results/pscd_rm"
PARENT = REPO_ROOT / "tracks/ksvd/results/pscd_tmdl"


def _load(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _mae(summary: dict) -> dict:
    return {
        "best_valid_mae": summary["best_valid_mae"],
        "best_epoch": summary["best_epoch"],
        "top5_soup_valid_mae": summary["top5_soup_valid_mae"],
        "epochs_run": summary["epochs_run"],
        "train_mae_at_best": summary["train_mae_at_best"],
        "params": summary["params"]["total_trainable"],
    }


def build_table() -> list[dict]:
    summ_a = _load(RM / "SUMMARY_A.json")
    match = _load(RM / "rate_match.json")
    freq_s = _load(PARENT / "summary_BFREQ.json")
    tm_s = _load(PARENT / "summary_BTM.json")
    fl_s = _load(RM / "summary_BFREQLOOSE.json")
    tmrate = summ_a["stage_a_TM-RATE"]

    arms = [
        {"arm": "FREQ", "principle": "frequency", "rate_regime": "low / compressed",
         "R_occ": summ_a["freq_budget_target"]["R_occ"],
         "R_state": summ_a["freq_budget_target"]["R_state"], **_mae(freq_s)},
        {"arm": "TM", "principle": "task+MDL", "rate_regime": "high / loose",
         "R_occ": summ_a["tm_rates"]["R_occ"],
         "R_state": summ_a["tm_rates"]["R_state"], **_mae(tm_s)},
        {"arm": "FREQ-LOOSE", "principle": "frequency", "rate_regime": "high / matched to TM",
         "R_occ": match["R_occ_freqloose"], "R_state": match["R_state_freqloose"],
         **_mae(fl_s)},
        {"arm": "TM-RATE", "principle": "task+MDL", "rate_regime": "low / matched to FREQ",
         "R_occ": tmrate["target"]["R_occ"], "R_state": tmrate["target"]["R_state"],
         "best_valid_mae": None, "best_epoch": None, "top5_soup_valid_mae": None,
         "epochs_run": tmrate["n_learned"], "train_mae_at_best": None, "params": None,
         "completed": tmrate["completed"], "infeasible": tmrate["infeasible"]},
    ]
    return arms


def decompose(arms: list[dict]) -> dict:
    by = {a["arm"]: a for a in arms}
    out: dict = {}
    for metric in ("best_valid_mae", "top5_soup_valid_mae"):
        f = by["FREQ"][metric]
        t = by["TM"][metric]
        fl = by["FREQ-LOOSE"][metric]
        g_total = f - t
        g_decomp = f - fl
        g_task_high = fl - t
        out[metric] = {
            "MAE_FREQ": f, "MAE_TM": t, "MAE_FREQ_LOOSE": fl,
            "G_total": g_total, "G_decompress": g_decomp, "G_task_high": g_task_high,
            "decompress_frac": g_decomp / g_total if g_total else None,
            "task_high_frac": g_task_high / g_total if g_total else None,
        }
    # low-rate causal quantity is undefined because TM-RATE is infeasible
    out["delta_task_low_rate"] = None
    out["delta_task_low_rate_note"] = (
        "undefined: TM-RATE construction infeasible under the frozen 64-rule budget")
    return out


def plot_plane(arms: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"FREQ": "tab:gray", "TM": "tab:blue",
              "FREQ-LOOSE": "tab:orange", "TM-RATE": "tab:red"}
    for a in arms:
        c = colors[a["arm"]]
        if a["best_valid_mae"] is not None:
            ax.scatter(a["R_state"], a["best_valid_mae"], color=c, marker="o", s=70,
                       zorder=3, label=f"{a['arm']} (best valid)")
            ax.scatter(a["R_state"], a["top5_soup_valid_mae"], color=c, marker="^", s=55,
                       zorder=3)
            ax.annotate(a["arm"], (a["R_state"], a["best_valid_mae"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
        else:
            ax.scatter(a["R_state"], ax.get_ylim()[1], color=c, marker="x", s=80,
                       zorder=3)
            ax.annotate(f"{a['arm']}\n(infeasible)", (a["R_state"], 0.0),
                        textcoords="offset points", xytext=(6, 40), fontsize=8,
                        color=c)
    ax.set_xlabel(r"$R_{\rm state}$ = (occurrences + active ports) / atoms")
    ax.set_ylabel("valid MAE")
    ax.set_title("PSCD-RM-v0 rate-performance plane\n(circles: best valid, triangles: Top-5 soup)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = [
        (PARENT / "history_BFREQ.json", "FREQ", "tab:gray"),
        (PARENT / "history_BTM.json", "TM", "tab:blue"),
        (RM / "history_BFREQLOOSE.json", "FREQ-LOOSE", "tab:orange"),
    ]
    for path, name, c in series:
        if not path.exists():
            continue
        curve = _load(path)
        xs = [r["epoch"] for r in curve]
        ys = [r["valid_mae"] for r in curve]
        ax.plot(xs, ys, color=c, label=name, lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid MAE")
    ax.set_title("PSCD-RM-v0 Stage B valid curves (seed 0, identical reader)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    arms = build_table()
    decomp = decompose(arms)
    analysis = {"round": "PSCD-RM-v0", "arms": arms, "decomposition": decomp}
    with (RM / "ANALYSIS.json").open("w") as fh:
        json.dump(analysis, fh, indent=2, sort_keys=True, default=float)
    plot_plane(arms, RM / "rate_performance_plane.png")
    plot_curves(RM / "stageB_valid_curves.png")
    print(json.dumps(analysis, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
