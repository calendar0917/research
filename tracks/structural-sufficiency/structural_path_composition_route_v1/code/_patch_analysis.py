"""Patch psc_analysis.py: ledger delegation, per-molecule bootstrap, final figures."""
import pathlib
import re

p = pathlib.Path(__file__).resolve().parent / "psc_analysis.py"
s = p.read_text()

# ---------------------------------------------------------------- ledger ----
start = s.index("def cmd_ledger(bundle: D.DataBundle) -> dict[str, Any]:")
end = s.index("def _predict_one(")
new_ledger = '''def cmd_ledger(bundle: D.DataBundle) -> dict[str, Any]:
    """Delegate to the v2 ledger (geometric S/C split, exact identity check)."""
    import psc_ledger

    lock = C.read_json(ANALYSIS / "selection_lock.json")
    cfg = C.load_config()
    out = psc_ledger.build(bundle, lock, cfg)
    C.write_json(ANALYSIS / "sum_edit_ledger.json", out)
    print(f"ledger: model={out['ledger_model']} identity_err="
          f"{out['identity_max_abs_std']:.2e} examples={len(out['examples'])} "
          f"alignable={out['n_alignable']}/{out['n_candidates']} "
          f"two-sided={out['n_two_sided']}")
    return out


'''
s = s[:start] + new_ledger + s[end:]

# ------------------------------------------------------- relative pair MAE --
s = s.replace("""        "sd_true": float(np.std(y)), "mean_abs_true": float(np.mean(np.abs(y))),
    }""", """        "sd_true": float(np.std(y)), "mean_abs_true": float(np.mean(np.abs(y))),
    }
    if out.get("mean_abs_true"):
        out["rel_mae"] = out["mae"] / out["mean_abs_true"]""")
s = s.replace("""                entry[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                            for k in ("mae", "rmse", "direction_accuracy")
                            if per_seed[0].get(k) is not None}""",
              """                entry[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                            for k in ("mae", "rmse", "direction_accuracy", "rel_mae",
                                      "mean_abs_true")
                            if per_seed[0].get(k) is not None}""")
s = s.replace("""                            sub[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                                      for k in ("mae", "rmse", "direction_accuracy")
                                      if per_seed[0].get(k) is not None}""",
              """                            sub[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                                      for k in ("mae", "rmse", "direction_accuracy", "rel_mae",
                                                "mean_abs_true")
                                      if per_seed[0].get(k) is not None}""")

# ------------------------------------------------- per-molecule bootstrap ---
s = s.replace('''    out["contrasts"] = contrasts''', '''    out["contrasts"] = contrasts
    # per-molecule error change of the champion against the reference
    champ_i = index.get(lock["champion"])
    ref_i = index.get(lock["ACRA_B_star"])
    if champ_i is not None and ref_i is not None:
        def per_mol(i: int) -> np.ndarray:
            return np.sqrt(np.nanmean(table[i], axis=0))
        diff = per_mol(ref_i) - per_mol(champ_i)  # >0 champion closer to truth
        boot = np.empty(reps)
        rng3 = np.random.default_rng(seed0 + 2)
        for r in range(reps):
            idx = rng3.integers(0, n, size=n)
            boot[r] = float(np.mean(diff[idx]))
        out["per_molecule"] = {
            "definition": "mean over molecules of |err_ref| - |err_champion| "
                          "(absolute error, positive = champion better)",
            "mean_abs_err_diff": float(np.mean(diff)),
            "median_abs_err_diff": float(np.median(diff)),
            "ci95": C.percentile_ci(boot),
            "share_molecules_champion_better": float(np.mean(diff > 0)),
            "share_molecules_tie": float(np.mean(np.abs(diff) < 1e-9)),
        }''')

# ---------------------------------------------------------------- figures ---
start = s.index("def cmd_figures(bundle: D.DataBundle) -> dict[str, Any]:")
end = s.index("# ---------------------------------------------------------------------------\n# entry point")
new_figs = '''def cmd_figures(bundle: D.DataBundle) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lock = C.read_json(ANALYSIS / "selection_lock.json")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    agg = summary["aggregate"]["D0"]
    boot = C.read_json(ANALYSIS / "bootstrap.json")
    fig_dir = ANALYSIS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # Fig 1: D0 whole-molecule accuracy by arm family (Q1)
    groups = [
        ("PSC SUM", ["PSC_SUM", "PSC_SUM_PAIR", "PSC_SUM_BAG", "PSC_SUM_BAG_PAIR"]),
        ("PSC POOL", ["PSC_POOL", "PSC_POOL_PAIR", "PSC_POOL_BAG", "PSC_POOL_BAG_PAIR"]),
        ("ACRA", ["ACRA_REF", "ACRA_PAIR"]),
        ("GINE", ["GINE_REF"]),
        ("CPU / fingerprint", ["PATH_COUNT_RIDGE", "PATH_COUNT_KRR", "MORGAN_COUNT_KRR",
                                "COUNT_ONLY_RIDGE"]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs, labels, colors = [], [], []
    x = 0.0
    ticks = []
    palette = {"PSC SUM": "#4c72b0", "PSC POOL": "#55a868", "ACRA": "#c44e52",
               "GINE": "#8172b2", "CPU / fingerprint": "#937860"}
    for gname, arms in groups:
        ticks.append((x - 0.5 + (len([a for a in arms if a in agg]) - 1) / 2, gname))
        for arm in arms:
            v = agg.get(arm)
            if not v:
                continue
            ax.bar(x, v["rmse_mean"], yerr=v["rmse_sd"], width=0.8,
                   color=palette[gname], alpha=0.9, capsize=2)
            ax.scatter([x] * len(v["rmse_seeds"]), v["rmse_seeds"], s=9, color="k", zorder=3)
            labels.append(arm.replace("PSC_", "").replace("_", "\\n"))
            xs.append(x)
            x += 1.0
        x += 0.6
    for name, ls in ((lock["ACRA_B_star"], "--"), ("GINE_REF", ":")):
        if name in agg:
            ax.axhline(agg[name]["rmse_mean"], ls=ls, lw=1, color="k", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    for pos, name in ticks:
        ax.text(pos, ax.get_ylim()[1] * 0.995, name, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("whole-molecule RMSE (label units)")
    ax.set_title("D0 (fit 2856 / held-out 504, scaffold-disjoint): 3-seed mean ± sd, dots = seeds")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_d0_arms.png", dpi=170)
    plt.close(fig)
    made.append("fig1_d0_arms.png")

    # Fig 2: pre-specified contrasts with paired bootstrap CIs (Q2/Q3/Q4 + Q1)
    con = boot["contrasts"]
    keys = [k for k in con if "champion_vs" not in k]
    champ_keys = [k for k in con if "champion_vs" in k]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    order = keys + champ_keys
    for row, k in enumerate(order):
        v = con[k]
        lo, hi = v["ci95"]
        col = "#4c72b0" if "champion_vs" not in k else "#c44e52"
        ax.plot([lo, hi], [row, row], color=col, lw=2)
        ax.plot([v["point"]], [row], "o", color=col, ms=5)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.02, 0.02, color="grey", alpha=0.15)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([con[k]["question"].split(" (")[0] + "\\n" + con[k]["candidate"]
                        .replace("PSC_", "") + " vs " + con[k]["reference"].replace("PSC_", "")
                        for k in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ RMSE = RMSE(reference) − RMSE(candidate), 95% paired-molecule bootstrap CI")
    ax.set_title("One-factor contrasts on D0 (grey band: ±0.02 practical-gain threshold)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_contrasts.png", dpi=170)
    plt.close(fig)
    made.append("fig2_contrasts.png")

    # Fig 3: where the errors live (Q5)
    upm = (C.read_json(ANALYSIS / "unique_pair_metrics.json")
           if (ANALYSIS / "unique_pair_metrics.json").is_file() else None)
    molnov = (C.read_json(ANALYSIS / "molecule_novelty.json")
              if (ANALYSIS / "molecule_novelty.json").is_file() else None)
    if upm and molnov:
        models = [lock["champion"], lock["ACRA_B_star"], "PATH_COUNT_RIDGE", "MORGAN_COUNT_KRR"]
        models = [m for m in dict.fromkeys(models) if m]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ax = axes[0]
        d = upm["splits"]["D0"]
        names = list(d["support_classes"])
        short = ["T seen", "T unseen\\nfrags seen", "T unseen\\n1 frag unseen",
                 "T unseen\\n2 frags unseen"]
        width = 0.8 / max(1, len(models))
        for j, m in enumerate(models):
            vals = [d["support_classes"][n].get(m, {}).get("mae") for n in names]
            ax.bar(np.arange(len(names)) + width * (j - (len(models) - 1) / 2), vals,
                   width=width, label=m.replace("PSC_", ""))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel("pair Δ MAE (cut-row weighted)")
        ax.set_title("Matched-pair error by support class (D0 held-out pairs)", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1]
        d = molnov["splits"]["D0"]
        names = list(d["buckets"])
        for j, m in enumerate(models):
            vals = [d["buckets"][n].get(m, {}).get("rmse") for n in names]
            ns = [d["buckets"][n]["n"] for n in names]
            ax.plot(range(len(names)), vals, "o-", label=m.replace("PSC_", ""))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\\nn={d['buckets'][n]['n']}" for n in names], fontsize=6.5)
        ax.set_ylabel("molecule RMSE")
        ax.set_title("Molecule error vs unseen-path-mass fraction $\\\\eta_{mol}$ (D0)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_where.png", dpi=170)
        plt.close(fig)
        made.append("fig3_where.png")

    # Fig 4: training curves + overfit sanity (capacity, not budget)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, color, label in ((lock["champion"], "#55a868", None),
                              (lock["R_star"], "#4c72b0", None),
                              (lock["ACRA_B_star"], "#c44e52", None),
                              ("GINE_REF", "#8172b2", None)):
        rid = run_id(arm, "D0", 0)
        path = RESULTS / rid / "curves.csv"
        if not path.is_file():
            continue
        import csv

        with path.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("eval_rmse")]
        ax.plot([int(r["epoch"]) for r in rows], [float(r["eval_rmse"]) for r in rows],
                label=arm, color=color, lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("D0 held-out RMSE")
    ax.set_title("Same 160-epoch recipe: PSC plateaus above ACRA/GINE", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_curves.png", dpi=170)
    plt.close(fig)
    made.append("fig4_curves.png")
    return {"figures": made}


'''
s = s[:start] + new_figs + s[end:]
p.write_text(s)
print("patched", p)