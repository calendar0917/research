"""Round figures: (1) four-arm endpoint + contrast forest; (2) dictionary
task-division health map.  Runs off the round's own dev_eval / mechanism_health
JSONs; single split per panel (never stitching splits/targets into a curve)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def _load(pattern: str):
    matches = sorted(OUT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return json.loads(matches[-1].read_text())


def main() -> None:
    dev = _load("dev_eval_A_J_D_J_M_F_D_F_M.json")
    health = _load("mechanism_health.json")
    arms = dev["roster"]["arms"]
    table = dev["main_table"]
    gains = dev["gains"]
    colors = {"J_D": "#1f77b4", "J_M": "#7f7f7f", "F_D": "#d62728", "F_M": "#ff9896"}

    # ---- figure 1: four-arm endpoints + contrast forest ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    x = np.arange(len(arms))
    for i, arm in enumerate(arms):
        ax.errorbar(
            x[i] - 0.13, table[arm]["dev_G0_cal_mae"],
            yerr=[[0], [0]], fmt="o", color=colors[arm], capsize=3,
        )
        ax.errorbar(
            x[i] + 0.13, table[arm]["dev_overall_cal_mae"],
            yerr=[[0], [0]], fmt="s", color=colors[arm], alpha=0.55, capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("internal dev g-MAE (calibrated, soup 236-240)")
    lo = min(table[a]["dev_G0_cal_mae"] for a in arms)
    hi = max(table[a]["dev_overall_cal_mae"] for a in arms)
    ax.set_ylim(lo - 0.002, hi + 0.002)
    ax.set_title("Stage-1 endpoints (o = G0 k=0, s = overall)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    spec = [("G_J", "dict increment (joint)"), ("G_F", "dict increment (separate)"),
            ("C_D", "construction change (dict)"), ("C_M", "construction change (MLP)")]
    ypos = np.arange(len(spec))[::-1]
    for y, (name, label) in zip(ypos, spec):
        g = gains[name]
        ax.errorbar(g["G0_cal"]["point"], y, xerr=[[-(g["G0_cal"]["ci95"][0] - g["G0_cal"]["point"])],
                                                    [g["G0_cal"]["ci95"][1] - g["G0_cal"]["point"]]],
                    fmt="o", color="#333", capsize=4)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([l for _n, l in spec])
    ax.set_xlabel("dev G0 calibrated gain (95% paired bootstrap CI, 2000x)")
    ax.set_title("Contrasts (right = subtrahend better)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    inter = gains.get("I_derived")
    if inter:
        ax.errorbar(inter["G0_cal"]["point"], 0,
                    xerr=[[inter["G0_cal"]["point"] - inter["G0_cal"]["ci95"][0]],
                          [inter["G0_cal"]["ci95"][1] - inter["G0_cal"]["point"]]],
                    fmt="D", color="#9467bd", capsize=4)
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_yticks([0])
        ax.set_yticklabels(["I = G_F - G_J\n= C_D - C_M"])
        ax.set_xlabel("dev G0 calibrated interaction")
        ax.set_title("Construction x encoder interaction")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure_stage1_four_arm_effect.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: dictionary task-division health map ----------------------
    rows = []
    for arm in arms:
        h = health[arm]["epoch240"]
        lt = h["local_tensor"]
        bs = h["batch_stats"]
        rows.append({
            "arm": arm,
            "atoms_used": bs.get("structure_atoms_used", np.nan),
            "struct_nnz": bs.get("structure_code_nnz", np.nan),
            "injection_rms": h["injection_rms"],
            "drift_rel": lt.get("raw_drift_from_init", np.nan) / max(lt.get("init_norm", 1), 1e-9),
            "rel_recon": lt.get("rel_recon_structure"),
            "rank": h["root_code_effective_rank"],
        })
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    labels = [r["arm"] for r in rows]
    x = np.arange(len(rows))
    def bar(ax, key, title, ylabel, fmt="{:.3g}"):
        vals = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.bar(x, [0 if np.isnan(v) else v for v in vals],
               color=[colors[a] for a in labels], alpha=0.8)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
    bar(axes[0], "struct_nnz", "structure/tuple code nnz (of 64)", "mean nnz")
    bar(axes[1], "atoms_used", "dictionary atoms used (of 64)", "atoms")
    bar(axes[2], "injection_rms", "W_loc injection RMS (epoch 240)", "RMS")
    bar(axes[3], "drift_rel", "local tensor drift / init L2 (epoch 240)", "relative")
    fig.suptitle("Dictionary task division: mechanism health vs task role (split A, epoch 240 soup probes)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "figure_dictionary_task_division.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
