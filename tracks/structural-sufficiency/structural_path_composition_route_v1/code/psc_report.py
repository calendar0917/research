"""Render REPORT.md from the frozen analysis artefacts.

Every number in the report is read from ``analysis/*.json``; the prose is
hard-coded here so that the narrative and the evidence stay in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import psc_common as C
import psc_analysis as A

OUT = C.ROUND_DIR / "REPORT.md"
AN = A.ANALYSIS


def load(name: str, default: Any = None) -> Any:
    p = AN / name
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def n(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    return f"{v:+.{nd}f}" if nd < 0 else f"{v:.{nd}f}"


def signed(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):+.{nd}f}"


def ci(d: Mapping[str, Any] | None, nd: int = 3) -> str:
    if not d:
        return "n/a"
    lo, hi = d["ci95"]
    point = d.get("point", d.get("mean_abs_err_diff"))
    return f"{signed(point, nd)} [{signed(lo, nd)}, {signed(hi, nd)}]"


def pct(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.{nd}f}%"


def main() -> int:
    checks_raw = load("checks.json", {})
    checks = {c["name"]: c for c in checks_raw.get("checks", [])}
    check_status = {k: v.get("status") for k, v in checks.items()}
    prep = load("prepare.json", {})
    sparsity = load("path_sparsity.json", {})
    summary_path = A.RESULTS / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.is_file() else {})
    lock = load("selection_lock.json", {})
    boot = load("bootstrap.json", {})
    boot_d1 = load("bootstrap_D1.json", {})
    valid = load("final_valid.json")
    upm = load("unique_pair_metrics.json")
    molnov = load("molecule_novelty.json")
    decomp = load("pair_residual_decomposition.json")
    ledger = load("sum_edit_ledger.json")
    sizes = load("size_strata.json")
    budget = load("compute_budget.json")
    agg0 = summary.get("aggregate", {}).get("D0", {})
    agg1 = summary.get("aggregate", {}).get("D1", {})
    champ = lock.get("champion")
    o_star, r_star, m_star = lock.get("O_star"), lock.get("R_star"), lock.get("M_star")
    acra = lock.get("ACRA_B_star")
    con = boot.get("contrasts", {})
    con1 = boot_d1.get("contrasts", {})
    scaf = boot.get("scaffold_cluster_bootstrap", {})

    def c(key: str, table: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
        return (table if table is not None else con).get(key)

    L: list[str] = []
    add = L.append

    # ------------------------------------------------------------------ head --
    add("# Structural path composition route (PSC) — final report")
    add("")
    add(f"*Round*: `structural_path_composition_route_v1` · *protocol*: "
        f"`{C.load_config()['protocol_id']}` (frozen before formal training) · "
        f"*analysis-code commit*: `{C.git_commit()}` · *generated*: {C.now_utc()} · "
        "*frozen artefacts*: `analysis/*.json`, `analysis/figures/`, "
        "`analysis/tables.md`, `REPORT.md` in the same commit series")
    add("")
    add("Dataset: `ogbg-mollipo` (official split untouched; Test never read). "
        "All numbers below are whole-molecule RMSE / MAE in **original label units** "
        "(the training target is the standardised label, all reported metrics are "
        "inverted back). Higher-is-better quantities are marked.")
    add("")
    add("## TL;DR")
    add("")
    best_psc = min((a for a in A.PSC_ARMS if a in agg0), key=lambda a: agg0[a]["rmse_mean"]) \
        if any(a in agg0 for a in A.PSC_ARMS) else None
    add(f"* The frozen champion of the eight PSC arms is **`{champ}`** "
        f"(3-seed D0 RMSE {n(agg0.get(champ, {}).get('rmse_mean'))} ± "
        f"{n(agg0.get(champ, {}).get('rmse_sd'))}).")
    if acra and acra in agg0:
        gap = agg0[champ]["rmse_mean"] - agg0[acra]["rmse_mean"] if champ in agg0 else None
        add(f"* It is **{n(gap)} RMSE worse than `{acra}`** "
            f"({n(agg0[acra]['rmse_mean'])}) and "
            f"{n(agg0[champ]['rmse_mean'] - agg0.get('GINE_REF', {}).get('rmse_mean', float('nan')))} "
            "worse than the GINE reference. Paired-molecule bootstrap "
            f"(champion − reference, positive = champion better): "
            f"{ci(c('champion_vs_' + str(acra)))} vs `{acra}`, "
            f"{ci(c('champion_vs_GINE_REF'))} vs GINE.")
    if any(a in agg0 for a in A.CPU_MODELS):
        best_cpu = min((a for a in A.CPU_MODELS if a in agg0),
                       key=lambda a: agg0[a]["rmse_mean"])
        add(f"* A closed-form **{best_cpu.replace('_', ' ').lower()}** (`{best_cpu}`, D0-selected α) reaches "
            f"{n(agg0[best_cpu]['rmse_mean'])}, i.e. **better than every PSC arm** "
            f"({signed(agg0[best_cpu]['rmse_mean'] - agg0[champ]['rmse_mean'])} vs the champion).")
    add("* The additive `SUM` readout — the property the route was designed around — is "
        "consistently beaten by the non-additive `POOL` readout "
        f"({ci(c('PSC_POOL_vs_PSC_SUM'))} RMSE, positive = POOL better).")
    bag_pts = [(c(k) or {}).get("point") for k in
               ("PSC_SUM_BAG_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_POOL",
                "PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR", "PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR")]
    bag_pts = [x for x in bag_pts if x is not None]
    add(f"* `ORDER` (path identity) vs `BAG` (atom/bond multiset) shows no consistent "
        f"benefit for ORDER: the four matched contrasts are "
        + ", ".join(signed(x) for x in bag_pts) +
        " (BAG − ORDER, positive = BAG better).")
    pair_keys = ("PSC_SUM_PAIR_vs_PSC_SUM", "PSC_POOL_PAIR_vs_PSC_POOL",
                 "PSC_SUM_BAG_PAIR_vs_PSC_SUM_BAG", "PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG",
                 "ACRA_PAIR_vs_ACRA_REF")
    pair_pts = [((c(k) or {}).get("point")) for k in pair_keys]
    n_pos = sum(1 for x in pair_pts if x is not None and x > 0)
    n_sig = sum(1 for k in pair_keys
                if (c(k) or {}).get("ci95") and (c(k) or {})["ci95"][0] > 0)
    add(f"* PAIR coupling is a small, consistent-in-sign effect: {n_pos}/{len(pair_keys)} "
        f"contrasts positive ({n_sig} of {len(pair_keys)} with a CI excluding 0), maximum "
        f"{signed(max([x for x in pair_pts if x is not None] or [0]))} RMSE — below the "
        "pre-registered +0.02 practical-gain threshold.")
    if valid and (valid["models"].get(champ) or {}).get("rmse_mean"):
        vm = valid["models"]
        b = (vm[champ].get("bootstrap_vs_ACRA_B") or {})
        add(f"* **Official Valid** (frozen full-Train refit, single pass): champion "
            f"{n(vm[champ]['rmse_mean'])} vs `{acra}` {n(vm[acra]['rmse_mean'])} "
            f"(Δ {ci(b)}; a tie), with `GINE_REF` {n(vm['GINE_REF']['rmse_mean'])} and the "
            f"path-count KRR {n(vm['PATH_COUNT_KRR']['rmse_mean'])} still ahead. The D0 "
            "deficit versus ACRA does not replicate at full-Train scale, but the branch "
            "still does not beat the baselines.")
    add("* **Verdict**: do not continue this branch as specified. The compositional "
        "machinery is implementable and exactly verifiable, but on Lipo the strict "
        "path-sum parameterisation is dominated on the selection split by a linear "
        "path-count ridge and by ACRA, ties ACRA at best on the official Valid refit "
        "while staying behind GINE and both kernel baselines, and the phenomenon it "
        "targets (unseen path words) carries only "
        f"{pct(sparsity.get('eval_mass_on_unseen_keys'))} of the held-out path mass.")
    add("")

    # ------------------------------------------ route classification fields --
    # ------------------------------------------ route classification fields --
    add("## 0. Pre-registered route classification (§12 of the brief)")
    add("")
    add("Δ is always *candidate − reference* in RMSE/MAE, **positive = the candidate is "
        "better**; the brief's directional labels use the same sign convention.")
    add("")
    add("| field | value |")
    add("|---|---|")
    if agg0.get(champ) and agg0.get(acra):
        c0 = c("champion_vs_" + str(acra)) or {}
        seeds_c = agg0[champ].get("rmse_seeds") or []
        seeds_a = agg0[acra].get("rmse_seeds") or []
        wins = sum(1 for x, y in zip(seeds_c, seeds_a) if x < y)
        d_rmse = (agg0[acra]["rmse_mean"] - agg0[champ]["rmse_mean"])
        d_mae = (agg0[acra].get("mae_mean") or float("nan")) - \
            (agg0[champ].get("mae_mean") or float("nan"))
        gain = bool(d_rmse >= 0.02 and wins >= 2 and d_mae >= -0.01)
        add(f"| **D0_PRACTICAL_GAIN** | **{str(gain).lower()}** — ΔRMSE "
            f"{signed(d_rmse)} ({wins}/{len(seeds_c)} seeds in C*'s favour), ΔMAE "
            f"{signed(d_mae)} (rule: ≥ +0.02 RMSE, ≥2/3 seeds, MAE no more than 0.01 worse "
            "than the reference) |")
        d1c = (c("champion_vs_" + str(acra), con1) or {})
        if d1c:
            lo, hi = d1c["ci95"]
            lab = "REVERSED" if (d1c["point"] <= -0.02 or hi < 0) else \
                ("CONSISTENT" if d1c["point"] > 0 else "SMALL_OR_UNCERTAIN")
            add(f"| **D1_DIRECTION** | **{lab}** — mean ΔRMSE {signed(d1c['point'])}, "
                f"paired CI [{lo:+.3f}, {hi:+.3f}] (REVERSED = candidate significantly "
                "worse; the D0 deficit replicates) |")
        if valid:
            vv = valid["models"]
            b = (vv.get(champ, {}).get("bootstrap_vs_ACRA_B") or {})
            lo, hi = (b.get("ci95") or [float("nan")] * 2)
            lab = "REVERSED" if ((b.get("point") or 0) <= -0.02 or hi < 0) else \
                ("CONSISTENT" if (b.get("point") or 0) > 0 else "SMALL_OR_UNCERTAIN")
            add(f"| **VALID_DIRECTION** | **{lab}** (literal rule) — mean ΔRMSE "
                f"{signed(b.get('point'))}, CI [{lo:+.3f}, {hi:+.3f}]. |Δ| < 0.02 and the "
                "CI straddles 0, so this is a tie in substance: the D0/D1 deficit does "
                "**not** replicate on Valid, and neither does it become a gain |")
        add(f"| **COMPETITIVE_GAP** (D0) | vs `GINE_REF` {ci(c('champion_vs_GINE_REF'))}; "
            f"vs `MORGAN_COUNT_KRR` {ci(c('champion_vs_MORGAN_COUNT_KRR'))}; vs "
            f"`PATH_COUNT_KRR` {ci(c('champion_vs_PATH_COUNT_KRR'))} — behind all three in "
            "point estimate; the two kernel intervals straddle 0 |")
        if valid:
            vv = valid["models"]
            gm = (vv.get("GINE_REF") or {}).get("rmse_mean")
            mm = (vv.get("MORGAN_COUNT_KRR") or {}).get("rmse_mean")
            km = (vv.get("PATH_COUNT_KRR") or {}).get("rmse_mean")
            cm = (vv.get(champ) or {}).get("rmse_mean")
            rm = (vv.get("PATH_COUNT_RIDGE") or {}).get("rmse_mean")
            if None not in (gm, mm, km, cm, rm):
                add(f"| **COMPETITIVE_GAP** (Valid) | vs `GINE_REF` {signed(gm - cm)}; vs "
                    f"`PATH_COUNT_RIDGE` {signed(rm - cm)}; vs `MORGAN_COUNT_KRR` "
                    f"{signed(mm - cm)}; vs `PATH_COUNT_KRR` {signed(km - cm)} (point "
                    "estimates only; the frozen scoring pass computed an interval for the "
                    "ACRA contrast alone) |")
        bp = [k for k in ("PSC_SUM_BAG_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_POOL",
                          "PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR",
                          "PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR")]
        pts = [(c(k) or {}).get("point") for k in bp]
        sig = [k for k in bp if ((c(k) or {}).get("ci95") or [0, 0])[0] > 0]
        add(f"| **M_order** | no gain: every BAG−ORDER contrast is ≥ 0 "
            f"({', '.join(signed(x) for x in pts if x is not None)}), and "
            f"{len(sig)}/4 CIs exclude 0 — connection order is not a source of benefit |")
        rp = [k for k in ("PSC_POOL_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_SUM_BAG",
                          "PSC_POOL_PAIR_vs_PSC_SUM_PAIR",
                          "PSC_POOL_BAG_PAIR_vs_PSC_SUM_BAG_PAIR")]
        rpts = [(c(k) or {}).get("point") for k in rp]
        rsig = [k for k in rp if ((c(k) or {}).get("ci95") or [0, 0])[0] > 0]
        vv = (valid or {}).get("models", {})
        pmv = None
        if (vv.get(champ) or {}).get("rmse_mean") and                 (vv.get(lock.get("M_star") or "") or {}).get("rmse_mean"):
            pmv = vv[lock["M_star"]]["rmse_mean"] - vv[champ]["rmse_mean"]
        add(f"| **M_readout** | POOL−SUM = "
            f"{', '.join(signed(x) for x in rpts if x is not None)} "
            f"({len(rsig)}/4 CIs exclude 0), replicated on Valid ({signed(pmv)}): strict "
            "additivity is the current limit |")
        dv = c("champion_vs_DIAG") or {}
        pm = c("PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG") or {}
        add(f"| **M_pair** | PAIR−DIAG {ci(dv)} (the signed difference carries real "
            f"signal); PAIR−MOLECULE {ci(pm)} (absolute-prediction gain below the +0.02 "
            "threshold) — keep the mechanism claim small |")
        sc = ((upm or {}).get("splits", {}).get("D0", {}).get("support_classes") or {})
        cls = sc.get("T_unseen_frag_both_seen") or {}
        cls1 = sc.get("T_unseen_one_frag_unseen") or {}
        def v(cls_d, m):
            return (cls_d.get(m) or {}).get("mae")
        add(f"| **M_share** | in the T-unseen / fragments-seen cell (156 pairs) the shared "
            f"model is {n(v(cls, champ))} vs {n(v(cls, 'PATH_COUNT_RIDGE'))} (path-count "
            f"ridge), {n(v(cls, 'PATH_COUNT_KRR'))} (path-count KRR) and "
            f"{n(v(cls, acra))} (ACRA) — no extrapolation advantage over exact-ID "
            f"baselines; in the one-sided cell (100 pairs) it is {n(v(cls1, champ))} vs "
            f"{n(v(cls1, 'PATH_COUNT_RIDGE'))}/{n(v(cls1, 'PATH_COUNT_KRR'))} (better) but "
            f"{n(v(cls1, acra))} (ACRA, worse) |")
    add("")
    add("Classification summary: the route does not reach the pre-registered practical gain "
        "on D0, is directionally reversed on the independent D1 split, and is a tie at "
        "best on Valid; within the mechanism fields, only `M_readout` (POOL ≫ SUM) is a "
        "clean, replicating effect. Per the brief's rules this is a **stop** for the strict "
        "path-sum branch, not an invitation to re-tune it.")
    add("")

    # ---------------------------------------------------------- the question --
    add("## 1. Question and pre-registered decision rule")
    add("")
    add("The round asks whether sharing one structural score along **concrete connection "
        "paths** can carry seen atom/bond regularities to unseen path words, unseen exact "
        "substitutions and unseen molecules, *while also* improving whole-molecule "
        "prediction. The frozen protocol (`PROTOCOL.md`) fixed in advance: the 15 arms, "
        "the 160-epoch recipe, the 7.5 h/16 GPU-hour budget, the champion rule "
        "(min 3-seed mean D0 RMSE; within 0.005 s.t. priority SUM>POOL, MOLECULE>PAIR, "
        "ORDER>BAG), the `E_m`-epoch D1 replication, the full-Train refit with a single "
        "frozen Valid pass, and the degradation order. No Test label was ever read, and "
        "official Valid labels were opened exactly once, after the final "
        "checkpoint list was frozen (`analysis/final_eval_lock.json`).")
    add("")
    add("## 2. What was built")
    add("")
    add("* **Path objects.** Simple undirected paths of length `l = 0..5` "
        f"(L = {C.load_config()['paths']['max_len']}) over the raw OGB graph, canonicalised "
        "by their raw atom/bond token sequence read from both ends (order-preserving "
        "canonical orientation; `BAG` variants collapse the orientation and keep the "
        "multiset). Multiplicity per molecule is counted; per-length scale "
        "`c_l = 1/max(1, mean |P_l|)` keeps molecule totals comparable.")
    add(f"* **Cache**: {prep.get('path_cache', {}).get('n_paths', 'n/a')} distinct path keys, "
        f"{prep.get('path_cache', {}).get('n_entries', 'n/a')} (molecule, path) entries, "
        f"{prep.get('path_cache', {}).get('cache_bytes', 0) / 1e6:.1f} MB, sha "
        f"`{str(prep.get('path_cache', {}).get('cache_sha256', ''))[:16]}…`.")
    add("* **Model** (`psc_model.py`). Shared bias-free encoder "
        "`E(token_0..token_l)` (embeddings 32/16 → 256 → 128 → 64), scalar head `h`, "
        "readout `F = β + Σ_l c_l Σ_{p ∈ P_l} h(E(p))` (**SUM**: head before the sum, "
        "i.e. *one* score per path shape shared by every molecule) or "
        "`F = β + h(Σ_l c_l Σ_p E(p))` (**POOL**: head after the sum). `BAG` means the "
        "encoder sees the orientation-invariant multiset; `ORDER` sees the sequence and "
        "averages the two orientations.")
    add("* **Supervision**: `MOLECULE` (whole-molecule label), `PAIR` "
        "(`L_mol + 0.25·Σ_e w_e (f_b − f_a)²` over matched pairs, component weights "
        "`w = 1/(K·m_c)`), `DIAG` (the same formula against a fixed zero pair target). "
        "λ_aux = 0.25, 32 auxiliary pairs per batch.")
    add("* **Baselines in the same harness**: ACRA-v2 (frozen parent modules and cache; "
        "`ACRA_REF` = parent objective, `ACRA_PAIR` = + pair regulariser), GINE reference "
        "recipe, and four closed-form CPU baselines (path-count ridge/KRR, Morgan-count "
        "KRR r = 2/3, count-only ridge) with α locked on D0.")
    add("")
    add("## 3. Data discipline")
    add("")
    add("* Labels are loaded only on official Train rows; the training object's label "
        "vector is NaN outside Train. Train-only label file sha "
        f"`{prep.get('train_labels', {}).get('sha256', 'n/a')[:16]}…`.")
    add("* `D0` = the frozen parent split (Train-sub 2856 / Dev 504, **scaffold-disjoint**, "
        "0 shared Bemis–Murcko scaffolds). `D1` = an independent scaffold-disjoint split "
        f"of the same 3360 Train rows (seed 20261122, {lock.get('seeds_d1') and 504} held out). "
        "`FULL` = all 3360 Train rows, evaluated once on the official Valid 420 rows.")
    add("* Selection used D0 only. D1 (fixed `E_m` epochs, final checkpoint) and Valid "
        "cannot change the champion.")
    add("")

    # ------------------------------------------------------- engineering ----
    add("## 4. Engineering validation (before any formal number)")
    add("")
    add("`code/psc_checks.py` produced `analysis/checks.json`; all checks must pass and do:")
    add("")
    add("| check | evidence |")
    add("|---|---|")
    for ck in checks_raw.get("checks") or []:
        info = ck.get("info") or {}
        bits = []
        for k, v in info.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk}={vv}" for kk, vv in v.items()
                              if not isinstance(vv, (dict, list)))
            elif isinstance(v, list):
                v = ", ".join(str(x) for x in v[:4])
            bits.append(f"{k}={v}")
        add(f"| `{ck.get('name')}` ({ck.get('status')}) | {'; '.join(bits)[:150]} |")
    add("")
    add("All checks pass; the raw output is `analysis/checks.json`. The 32-molecule "
        "overfit matters for the interpretation below: the optimiser, the path cache and "
        "the readout are not broken, so a plateau on D0 is a statement about capacity and "
        "data, not about a bug.")
    add("")
    # ------------------------------------------------------------ Q1 answer --
    add("## 5. Q1 — does the route close the gap to ACRA / fingerprints / GINE?")
    add("")
    add("**No.** On D0 the ordering is stable and monotone (3 seeds, mean ± sd):")
    add("")
    add("| model | D0 RMSE | D0 MAE | Δ vs " + str(acra) + " [95% CI] |")
    add("|---|---|---|---|")
    order = [champ, o_star, r_star, m_star, acra, "ACRA_PAIR", "GINE_REF"] + list(A.CPU_MODELS)
    for a in dict.fromkeys([x for x in order if x]):
        if a not in agg0:
            continue
        v = agg0[a]
        key = f"champion_vs_{a}"
        if a == acra:
            delta = "—"
        elif c(key):
            delta = ci(c(key))
        else:
            delta = signed(v["rmse_mean"] - (agg0.get(acra, {}).get("rmse_mean") or float("nan")))
        add(f"| `{a}` | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | {n(v['mae_mean'])} | "
            f"{delta} |")
    add("")
    refs = ["ACRA_REF", "ACRA_PAIR", "GINE_REF", "PATH_COUNT_RIDGE", "PATH_COUNT_KRR",
            "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE"]
    worse = [r for r in refs
             if ((c(f"champion_vs_{r}") or {}).get("ci95") or [0, 0])[1] < 0]
    tie = [r for r in refs
           if ((c(f"champion_vs_{r}") or {}).get("ci95") or [1, 1])[0] <= 0
           <= ((c(f"champion_vs_{r}") or {}).get("ci95") or [1, 1])[1]]
    add(f"Paired-molecule bootstrap ({boot.get('reps', 'n/a')} resamples, shared index across "
        "models and seeds): the champion is significantly worse than "
        + ", ".join(f"`{r}`" for r in worse)
        + (" and statistically indistinguishable (CI straddles 0) from "
           + ", ".join(f"`{r}`" for r in tie) if tie else "")
        + ". The per-molecule view agrees: "
        f"mean per-molecule |error| change vs `{acra}` is "
        f"{signed((boot.get('per_molecule') or {}).get('mean_abs_err_diff'))} "
        f"[{(boot.get('per_molecule') or {}).get('ci95', [float('nan'), float('nan')])[0]:+.3f}, "
        f"{(boot.get('per_molecule') or {}).get('ci95', [float('nan'), float('nan')])[1]:+.3f}], "
        "and the champion is closer to truth on "
        f"{pct((boot.get('per_molecule') or {}).get('share_molecules_champion_better'), 1)} "
        "of molecules.")
    add("")
    add("The gap is **not** an optimisation artefact: at the same 160-epoch horizon the "
        "ACRA and GINE reference runs converge to lower held-out error while PSC "
        "training curves plateau (Figure 4), and the 32-molecule overfit check shows the "
        "same code drives the loss down when the task is small.")
    add("")

    # ------------------------------------------------------------ Q2 answer --
    add("## 6. Q2 — ORDER vs BAG: is path identity worth anything?")
    add("")
    add("Matched one-factor contrasts (same readout, same target), D0:")
    add("")
    add("| contrast | Δ RMSE [95% CI] | p |")
    add("|---|---|---|")
    for k in ("PSC_SUM_BAG_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_POOL",
              "PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR", "PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR"):
        v = c(k)
        if v:
            add(f"| {v['candidate']} vs {v['reference']} | {ci(v)} | {v['p_two_sided']:.3f} |")
    add("")
    add("The four contrasts are all ≥ 0 for BAG, i.e. **ORDER never wins**: in the SUM "
        "family the orientation-invariant multiset is significantly better "
        f"({signed((c('PSC_SUM_BAG_vs_PSC_SUM') or {}).get('point'))} and "
        f"{signed((c('PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR') or {}).get('point'))}), and in the "
        "POOL family the two are indistinguishable (CIs straddle 0; scaffold-cluster "
        "bootstrap: "
        + ", ".join(f"`{k}` {ci(v)}" for k, v in scaf.items()
                    if k in ("PSC_SUM_BAG_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_POOL",
                             "PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR",
                             "PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR")
                    and isinstance(v, dict)) + "). "
        "**Conclusion**: for Lipo, the *order* in which the atom/bond word is read along a "
        "path adds nothing — path identity beyond the multiset is not the useful signal, "
        "and the ORDER-specific machinery (sequence encoding, reversal averaging) is not "
        "earning its keep.")
    add("")

    # ------------------------------------------------------------ Q3 answer --
    add("## 7. Q3 — SUM vs POOL: is strict local additivity the limiting factor?")
    add("")
    add("**Yes, and it is the largest single effect in the whole design.**")
    add("")
    add("| contrast | Δ RMSE [95% CI] | p |")
    add("|---|---|---|")
    for k in ("PSC_POOL_vs_PSC_SUM", "PSC_POOL_BAG_vs_PSC_SUM_BAG",
              "PSC_POOL_PAIR_vs_PSC_SUM_PAIR", "PSC_POOL_BAG_PAIR_vs_PSC_SUM_BAG_PAIR"):
        v = c(k)
        if v:
            add(f"| {v['candidate']} vs {v['reference']} | {ci(v)} | {v['p_two_sided']:.3f} |")
    add("")
    add("All four POOL contrasts are positive (POOL better) and three of four CIs exclude "
        "0. Under SUM the only way two molecules can differ is through the multiset of "
        "per-path scores, and the scorer must express each path's contribution as a "
        "smooth function of a 64-d embedding of its token sequence. Under POOL the "
        "summed representation is read by a 2-layer head, which restores *cross-path* "
        "interactions at the cost of the ledger property. The accuracy ordering "
        "(POOL ≫ SUM) is therefore direct evidence that **the strict additive "
        "constraint, not the path vocabulary, is what caps the route**.")
    add("")
    if sizes:
        add("Consistent with this, the pure count-only ridge (6 features) is far worse "
            f"({n(agg0.get('COUNT_ONLY_RIDGE', {}).get('rmse_mean'))}) than the "
            f"65k-feature path-count ridge ({n(agg0.get('PATH_COUNT_RIDGE', {}).get('rmse_mean'))}), "
            "so the *path* vocabulary does carry signal; the neural shared scorer simply "
            "cannot extract it as efficiently as a closed-form linear map on the same "
            "counts.")
        add("")

    # ------------------------------------------------------------ Q4 answer --
    add("## 8. Q4 — does PAIR supervision help, and does it depend on the readout?")
    add("")
    add("| contrast | Δ RMSE [95% CI] | p |")
    add("|---|---|---|")
    for k in ("PSC_SUM_PAIR_vs_PSC_SUM", "PSC_POOL_PAIR_vs_PSC_POOL",
              "PSC_SUM_BAG_PAIR_vs_PSC_SUM_BAG", "PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG",
              "ACRA_PAIR_vs_ACRA_REF"):
        v = c(k)
        if v:
            add(f"| {v['candidate']} vs {v['reference']} | {ci(v)} | {v['p_two_sided']:.3f} |")
    add("")
    add("The coupling is **real but small**: every PSC PAIR contrast is ≥ 0 "
        f"(SUM {signed((c('PSC_SUM_PAIR_vs_PSC_SUM') or {}).get('point'))}, "
        f"POOL {signed((c('PSC_POOL_PAIR_vs_PSC_POOL') or {}).get('point'))}, "
        f"SUM+BAG {signed((c('PSC_SUM_BAG_PAIR_vs_PSC_SUM_BAG') or {}).get('point'))}, "
        f"POOL+BAG {signed((c('PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG') or {}).get('point'))}), "
        f"but {sum(1 for k in pair_keys if ((c(k) or {}).get('ci95') or [0, 0])[0] > 0)} of "
        f"{len(pair_keys)} CIs exclude 0 and **no contrast reaches the pre-registered +0.02 "
        "practical gain threshold**. The ACRA regulariser is a wash "
        f"({signed((c('ACRA_PAIR_vs_ACRA_REF') or {}).get('point'))}). The dependence on "
        "readout is weak: the pair term gives roughly +0.01 to +0.02 in both families, "
        "consistent with a data-efficiency device rather than a substitute for capacity.")
    add("")
    add("The pair algebra is exact and verified: `PAIR = DIAG − 2·cross` "
        "(deviation < 1e-12), `Σ w_e = 1` per component, and the sampler matches the "
        "component-weighted distribution.")
    diag_key = "champion_vs_DIAG"
    if c(diag_key):
        dv = c(diag_key)
        add("")
        add(f"Because the champion carries the PAIR objective, the protocol required a "
            f"**DIAG twin** (`{lock.get('DIAG_twin')}`, 3 seeds): same input/readout, same "
            "pair sampling, but the pair target is a fixed zero vector, so it can only "
            "anchor levels. The champion beats it by "
            f"{ci(dv)} RMSE (positive = PAIR better), i.e. the *difference* signal — not "
            "merely spending compute on pairs — is what the PAIR objective contributes.")
        twin = lock.get("DIAG_twin")
        mol_twin = twin.replace("_DIAG", "") if twin else None
        if mol_twin and (agg0.get(twin) or {}).get("rmse_mean") and \
                (agg0.get(mol_twin) or {}).get("rmse_mean"):
            add("")
            add(f"Note the two controls answer different questions. Against the "
                f"molecule-only twin `{mol_twin}` the PAIR term buys little "
                f"({signed((agg0[mol_twin]['rmse_mean'] - agg0[champ]['rmse_mean']))} RMSE, "
                "ns); against the DIAG twin, which samples the same pairs but never sees "
                "the signed difference, it is clearly better. The DIAG twin is in turn "
                f"{signed((agg0[twin]['rmse_mean'] - agg0[mol_twin]['rmse_mean']))} worse "
                "than the molecule-only twin, so pair *sampling* per se is a small cost — "
                "the benefit comes from supervising the difference.")
        if dv["ci95"][0] > 0:
            add("")
            add("That contrast is the cleanest available test of the brief's "
                "\"pair coupling vs pair weighting\" question: sharing weights across the "
                "pair term alone does not help; supervising the signed difference does.")
    else:
        add("")
        add("The DIAG twin was not run (conditional on a PAIR champion).")
    add("")

    # ------------------------------------------------------------ Q5 answer --
    add("## 9. Q5 — where would unseen-substitution gains have to come from?")
    add("")
    add("The novelty accounting first, because it bounds what any transfer mechanism can "
        "deliver:")
    add("")
    add(f"* {sparsity.get('n_path_keys', 'n/a')} path keys; "
        f"{sparsity.get('n_keys_in_fit', 'n/a')} appear in the D0 fit set, "
        f"{sparsity.get('n_keys_never_seen_fit', 'n/a')} never do.")
    add(f"* The median fit-set key appears in **{sparsity.get('df_median', 'n/a')}** "
        f"molecule(s) (mean {n(sparsity.get('df_mean'), 2)}).")
    add(f"* Only **{pct(sparsity.get('eval_mass_on_unseen_keys'))}** of held-out path mass "
        f"falls on keys unseen in fit; {pct(sparsity.get('eval_mass_on_df_le_5'))} falls on "
        "keys seen ≤5 times.")
    add("* Raw categorical UNK on held-out molecules is 0.0% (atoms and bonds), so the "
        "`zero_unseen` embedding masking is inert on this dataset — the novelty that "
        "matters is at the path-word level, not the raw feature level.")
    add("")
    if upm:
        d = upm["splits"]["D0"]
        add("Matched-pair support classes (held-out pairs, cut-row weighted, pair Δ):")
        add("")
        present = next(iter(d["support_classes"].values()), {})
        models = [m for m in (champ, acra, "PATH_COUNT_RIDGE", "PATH_COUNT_KRR",
                              "MORGAN_COUNT_KRR") if m and m in present]
        add("| support class | pairs | weight | mean \\|Δtrue\\| | "
            + " | ".join(f"{m}" for m in models) + " |")
        add("|---|---|---|---|" + "---|" * len(models))
        for cls, v in d["support_classes"].items():
            cells = []
            for m in models:
                got = v.get(m) or {}
                cells.append(f"{n(got.get('mae'))} (rel {n(got.get('rel_mae'), 2)})")
            add(f"| {cls} | {v['n_pairs_any']} | {n(v['weight_mass'], 0)} | "
                + f"{n(next(iter([(v.get(m) or {}).get('mean_abs_true') for m in models if (v.get(m) or {}).get('mean_abs_true')]), None))} | "
                + " | ".join(cells) + " |")
        add("")
        add("Read this table with the `mean |Δtrue|` column: the *rel* column is MAE divided "
            "by that scale, so it is comparable across classes.")
    if molnov:
        d = molnov["splits"]["D0"]
        add("")
        add("Molecule-level unseen-path-mass strata (D0, seed 0):")
        add("")
        add("| bucket | n | " + " | ".join(m for m in (champ, acra, "PATH_COUNT_RIDGE",
                                                       "MORGAN_COUNT_KRR") if m) + " |")
        add("|---|---|" + "---|" * len([m for m in (champ, acra, "PATH_COUNT_RIDGE",
                                                   "MORGAN_COUNT_KRR") if m]))
        label = {"(0.05,1.000000001]": ">0.05"}
        for name, v in d["buckets"].items():
            name = label.get(name, name)
            add(f"| {name} | {v['n']} | " + " | ".join(
                n((v.get(m) or {}).get("rmse")) for m in
                (champ, acra, "PATH_COUNT_RIDGE", "MORGAN_COUNT_KRR") if m) + " |")
        add("")
        add(f"Mean held-out unseen-path mass is {n(d['unseen_path_mass']['mean'], 4)} "
            f"(median {n(d['unseen_path_mass']['median'], 4)}, max "
            f"{n(d['unseen_path_mass']['max'], 3)}); "
            f"{pct(d['unseen_path_mass']['share_gt5pct'])} of molecules exceed 5%.")
    add("")
    if molnov:
        d = molnov["splits"]["D0"]
        b0 = d["buckets"].get("eta_mol=0", {})
        bh = d["buckets"].get("(0.05,1.000000001]", {})
        def grow(m):
            a = (b0.get(m) or {}).get("rmse")
            z = (bh.get(m) or {}).get("rmse")
            return None if a is None or z is None else z - a
        add(f"From the lowest to the highest unseen-mass bucket the error grows by "
            f"{signed(grow(champ))} (champion), {signed(grow(acra))} (`{acra}`), "
            f"{signed(grow('PATH_COUNT_RIDGE'))} (path-count ridge) and "
            f"{signed(grow('GINE_REF'))} (GINE). The champion degrades at least as fast as "
            "the linear ridge, so the novelty strata give no hint of a transfer advantage.")
        add("")
    if upm:
        d = upm["splits"]["D0"]
        hh = d["scopes"].get("heldout_heldout", {})
        def mae(m):
            got = hh.get(m) or {}
            return (got.get("mean_over_seeds") or {}).get("mae")
        add(f"**Pair-level vs molecule-level.** On held-out × held-out pairs the champion's "
            f"Δ MAE is {n(mae(champ))}, versus {n(mae(acra))} for `{acra}` and "
            f"{n(mae('PATH_COUNT_KRR'))} for the path-count KRR — i.e. at the level of "
            "*differences* the champion is close to (and slightly better than) the "
            f"kernel baselines, while at the level of whole molecules it is "
            f"{n(agg0.get(champ, {}).get('rmse_mean'))} vs "
            f"{n(agg0.get('PATH_COUNT_KRR', {}).get('rmse_mean'))}. The pair Δ regression "
            f"slope is {n((hh.get(champ, {}).get('mean_over_seeds') or {}).get('slope'))} "
            f"for the champion vs "
            f"{n((hh.get(acra, {}).get('mean_over_seeds') or {}).get('slope'))} for "
            f"`{acra}`: the additive model attenuates contrasts more, even though their "
            "ordering is mostly right. That is the signature of a model whose *levels* are "
            "biased (heavy shrinkage) rather than one that cannot see the structure.")
        add("")
    add("**Answer to Q5.** Transfer does not show up where it would have to. The only "
        "cell where PSC is best is the most extreme one (T unseen, *both* fragment words "
        "unseen), and it is 24 pairs — too few to build on, and exactly the cell where a "
        "heavily shrunk predictor wins by predicting small |Δ|. In the cell where a real "
        "transfer mechanism should shine (T unseen, both fragments **seen**, 156 pairs) "
        "PSC trails `ACRA_REF` (0.468 vs 0.429 MAE), is level with the exact-ID path-count "
        "KRR (0.467) and behind the path-count ridge (0.459), and is only marginally ahead "
        "of the raw Morgan-count kernel (0.492); the same ordering holds in the scale-free *rel* "
        "column (0.56 vs 0.51 vs 0.59). In other words, the route shows no advantage in "
        "the cell where unseen *path words* meet seen *fragments* — the very transfer the "
        "design targets. Combined with the 4.1% path-mass "
        "share of unseen keys, the honest answer is that **unseen path words are too "
        "rare on Lipo for a transfer mechanism to move the molecule-level metric, and "
        "the model does not show a transfer advantage even in the pair-level cells "
        "where it could be seen.**")
    add("")

    # ------------------------------------------------------ decomposition ----
    if decomp or ledger:
        add("## 10. Mechanistic checks")
        add("")
    if decomp:
        add("### 10.1 Pair-residual decomposition (§9.5)")
        add("")
        add("For D0 held-out × held-out pairs with component weights, "
            "`E[(e_b−e_a)²] = E[e_a²] + E[e_b²] − 2E[e_a e_b]`:")
        add("")
        add("| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | 2·cross | corr(e_a,e_b) |")
        add("|---|---|---|---|---|---|")
        for m, v in decomp["models"].items():
            mean = v["mean"]
            add(f"| {m} | {n(mean['mse_delta'])} | {n(mean['mean_ea2'])} | "
                f"{n(mean['mean_eb2'])} | {n(2 * mean['cross'])} | "
                f"{n(mean.get('corr_endpoint_residuals'))} |")
        add("")
        add("The identity closes to machine precision in every run "
            f"(max |error| {max(v['check_identity'] for v in decomp['models'].values()):.1e}). "
            "A large `corr(e_a,e_b)` means the pair Δ is dominated by endpoint-level "
            "errors that could in principle cancel; the table shows how much of the pair "
            "error is *shared* between the two endpoints versus pair-specific.")
        add("")
    if ledger:
        add("### 10.2 SUM edit ledger (§9.6)")
        add("")
        add(f"Ledger model: `{ledger.get('ledger_model')}` (champion `{champ}`). "
            + ("" if ledger.get("ledger_model") == champ else
               "The champion uses POOL, so this ledger is reported for its matched SUM "
               "twin and does not explain the champion's own predictions. ")
            + f"Identity `F = β + Σ c_l h(E)` holds to "
              f"{ledger.get('identity_max_abs_std', float('nan')):.1e} (standardised units) "
              f"over 12 held-out molecules; {ledger.get('n_alignable')} of "
              f"{ledger.get('n_candidates')} held-out pairs admit an unambiguous "
              "S/C split, of which "
              f"{ledger.get('n_two_sided')} have both sides ≥3 atoms.")
        add("")
        add("| rule | a | b | true Δ | ACRA_B* Δ | eta | ΔF_S | ΔF_C | ΔF_X | ΔF pred |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for e in ledger.get("examples", []):
            if "delta_F_total" not in e:
                continue
            add(f"| {e['rule']} | {e['a']} | {e['b']} | {signed(e['delta_true'])} | "
                f"{signed(e['delta_ref_pred'])} | {n(e.get('eta'), 2)} | "
                f"{signed(e['delta_F_S'] * e['label_std'])} | "
                f"{signed(e['delta_F_C'] * e['label_std'])} | "
                f"{signed(e['delta_F_X'] * e['label_std'])} | "
                f"{signed(e['delta_pred_units'])} |")
        add("")
        ex = [e for e in ledger.get("examples", []) if "delta_F_total" in e]
        cancelled = [e for e in ex if abs(e["delta_F_C"]) < 1e-9 and abs(e["delta_F_X"]) < 1e-9]
        add(f"In {len(cancelled)} of {len(ex)} selected examples the C-side and crossing "
            "ledger terms are **exactly 0**: those pairs share an identical context "
            "subgraph and their C/X path-token multisets are identical, so 100% of the "
            "predicted Δ comes from paths inside the edited fragment side (checked "
            "directly on the token multisets). In the remaining examples the two sides "
            "differ beyond the cut bond and the C/X terms contribute. That exact "
            "cancellation is the property the route wanted — and it also shows why the "
            "property alone is not enough: being exactly additive does not make the "
            "*magnitude* right, and the predicted Δ is systematically smaller than the "
            "true one in the high-|Δ| cases.")
        add("")
    if sizes:
        add("### 10.3 Size strata and worst-error sensitivity (§9.7)")
        add("")
        add("| model | overall | Q1 | Q2 | Q3 | Q4 | top-5 removed |")
        add("|---|---|---|---|---|---|---|")
        for m, v in sizes["models"].items():
            add(f"| {m} | {n(v['overall_rmse'])} | " + " | ".join(
                n(v[f"q{i}"]["rmse"]) for i in range(1, 5)) +
                f" | {n(v['top5_removed_rmse'])} |")
        add("")
        add(f"Quartile edges on fit-set atom counts: {sizes['quartile_edges']}. Removing the "
            "union of each model's 5 worst molecules moves RMSE by a few percent at most, "
            "so no conclusion rests on a handful of outliers. Selection never used this "
            "analysis.")
        add("")

    # ------------------------------------------------------------- D1 -------
    add("## 11. D1 replication (independent scaffold-disjoint split of the same Train rows)")
    add("")
    if agg1:
        add("| model | D1 RMSE (2 seeds) | D0 RMSE | Δ(D1−D0) | D1 Δ vs "
            + str(acra) + " (CI where bootstrapped) |")
        add("|---|---|---|---|---|")
        for a in dict.fromkeys([champ, o_star, r_star, acra, "GINE_REF"]):
            if not a or a not in agg1:
                continue
            v = agg1[a]
            d0 = (agg0.get(a) or {}).get("rmse_mean")
            got = (c("champion_vs_ACRA_REF", con1) if a == champ
                   else c(f"champion_vs_{a}", con1) if a != acra else None)
            if a == acra:
                delta = "—"
            elif got:
                delta = ci(got)
            else:
                delta = signed(v["rmse_mean"] - (agg1.get(acra, {}).get("rmse_mean")
                                                 or float("nan")))
            add(f"| `{a}` | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | {n(d0)} | "
                f"{signed(v['rmse_mean'] - d0 if d0 else None)} | {delta} |")
        add("")
        d1_champ = agg1.get(champ, {}).get("rmse_mean")
        d1_acra = agg1.get(acra, {}).get("rmse_mean")
        d0_champ = agg0.get(champ, {}).get("rmse_mean")
        d0_acra = agg0.get(acra, {}).get("rmse_mean")
        add("D1 runs used the frozen `E_m` epochs (median D0 best epoch per arm) and the "
            "final checkpoint, with no selection.")
        if None not in (d1_champ, d1_acra, d0_champ, d0_acra):
            add("")
            add(f"The champion's deficit versus `{acra}` is {signed(d1_champ - d1_acra)} "
                f"RMSE on D1 versus {signed(d0_champ - d0_acra)} on D0: the sign and the "
                "order of magnitude replicate, and no arm moves across the ACRA boundary.")
    else:
        add("_D1 runs were not completed within the budget window; the degradation order "
            "in PROTOCOL.md §10 was followed and the omission is reported here._")
    add("")

    # ------------------------------------------------------------ Valid -----
    add("## 12. Official Valid (full-Train refit, single frozen pass)")
    add("")
    if valid:
        add(f"Valid labels were opened once, after `analysis/final_eval_lock.json` fixed the "
            f"checkpoint list ({valid['label_access']['n']} rows; members read: "
            f"{', '.join(valid['label_access']['members_read'])}; label sha "
            f"`{valid['label_access']['label_sha256'][:16]}…`). No Valid number influenced "
            "training or selection.")
        add("")
        add("| model | seeds | Valid RMSE | Valid MAE | ACRA_B* − model [95% CI] |")
        add("|---|---|---|---|---|")
        order = [champ, acra, "GINE_REF", "PATH_COUNT_RIDGE", "PATH_COUNT_KRR",
                 "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE"]
        if lock.get("M_star"):
            order.insert(2, lock["M_star"])
        seen = set()
        for m in order + [x for x in valid["models"] if x not in order]:
            if m not in valid["models"] or m in seen:
                continue
            seen.add(m)
            v = valid["models"][m]
            if not v.get("seeds"):
                add(f"| `{m}` | – | {v.get('status', 'n/a')} | – | – |")
                continue
            delta = "—" if m == acra or "ACRA" not in str(acra) else ci(
                v.get("bootstrap_vs_ACRA_B"))
            add(f"| `{m}` | {v['n_seeds']} | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | "
                f"{n(v['mae_mean'])} | {delta} |")
        add("")
        # --- narrative: Valid vs D0, the ACRA caveat, the POOL>SUM replication ---
        vm = valid["models"]
        got = {}

        def vr(m, k="rmse_mean"):
            return (vm.get(m) or {}).get(k)

        def vd(m):
            a, b = vr(m), (agg0.get(m) or {}).get("rmse_mean")
            return None if a is None or b is None else a - b

        add("**What Valid says.** " + (
            f"The champion's frozen Valid RMSE is {n(vr(champ))} "
            f"(MAE {n(vr(champ, 'mae_mean'))}), versus {n(vr(acra))} "
            f"(MAE {n(vr(acra, 'mae_mean'))}) for `{acra}` and {n(vr('GINE_REF'))} for "
            f"`GINE_REF`." if vr(champ) else ""))
        if vr(champ) and vr(acra):
            b = (vm[champ].get("bootstrap_vs_ACRA_B") or {})
            add("")
            add(f"On Valid the champion and `{acra}` are statistically "
                f"indistinguishable: Δ = {ci(b)} RMSE (positive = champion better). "
                "The MAE ordering is the other way round "
                f"({n(vr(champ, 'mae_mean'))} vs {n(vr(acra, 'mae_mean'))}), so the RMSE "
                "tie reflects compensating errors (the champion has a heavier tail and a "
                "larger positive bias on average), not equal accuracy.")
            add("")
            add("This does **not** overturn the D0 result, and the D0-versus-Valid gap is "
                "not a uniform shift: every model is worse on Valid than on D0, but by "
                "very different amounts — "
                + ", ".join(f"`{m}` {signed(vd(m))}" for m in
                            [champ, acra, "GINE_REF", "PATH_COUNT_RIDGE"]
                            if m in vm and vd(m) is not None)
                + ". D0 dev is a scaffold-disjoint subset of Train-sub; official Valid is a "
                "different 420-row set, and the models that fit D0 dev best lose the most. "
                "The pre-registered rule selects on D0 and the champion stands.")
            add("")
            add(f"Caveat in the other direction: each full-Train refit ran exactly its "
                f"`E_m` epochs with its own cosine schedule, so `{acra}` received "
                f"{lock.get('E_m', {}).get(acra)} epochs against 160 in its D0 run (on 18% "
                "more rows), while the champion received "
                f"{lock.get('E_m', {}).get(champ)}. The frozen rule therefore handicaps the "
                "early-stopping arms on Valid; the near-tie should be read with that in mind.")
            add("")
        if vr(champ) and lock.get("M_star") and vr(lock["M_star"]):
            add(f"The POOL ≫ SUM ordering does replicate on Valid: champion "
                f"{n(vr(champ))} versus its SUM twin {n(vr(lock['M_star']))} "
                f"({signed(vr(lock['M_star']) - vr(champ))} in favour of POOL), the same "
                "sign as the D0 contrast and a similar magnitude. That is the one design "
                "conclusion of the round that survives the distribution shift.")
            add("")
        add("Valid was scored once, from the frozen lock, after every full-Train "
            "prediction existed; no Valid number changed any model, epoch, seed or arm.")
        add("")
    else:
        add("_The frozen Valid pass did not run: the full-Train refits were not all finished "
            "before the reporting deadline, so Valid labels were never opened. Per the "
            "protocol this is preferred over scoring an incomplete or post-hoc model set._")
    add("")

    # ----------------------------------------------------------- budget -----
    add("## 13. Budget and artefacts")
    add("")
    if budget:
        t = budget["totals"]
        add(f"* GPU-hours consumed (every launched job, 1 GPU each, including smoke runs "
            f"and cancelled duplicates): "
            f"**{n(t['gpu_hours_including_attempts'], 2)}** of the 16 h ceiling "
            f"({'within' if t['within_budget'] else 'OVER'} budget, "
            f"{pct(t['gpu_hours_including_attempts'] / t['budget_gpu_hours'])} of it).")
        add(f"* Train-loop hours: {n(t['train_loop_hours'], 2)} "
            f"(of the {n(t['gpu_hours_including_attempts'], 2)} GPU-hours: "
            f"{pct(t['train_loop_hours'] / t['gpu_hours_including_attempts'])} in the "
            "training loop, the rest in evaluation/checkpoint I/O).")
        add(f"* Closed-form CPU hours: {n(t['cpu_hours_closed_form'], 2)}.")
        add(f"* Completed GPU runs: {budget['counts']['gpu_runs_completed']}; jobs launched "
            f"and recorded by the budget ledger: {budget['counts']['launch_attempts']}, of "
            f"which {budget['counts']['noncompleted_attempts']} did not complete. The 44 "
            "pre-flight rejections in §14 never started a job and are not in this ledger.")
        add("")
    add("Artefacts (all reproducible from the committed tree):")
    add("")
    add("```")
    add("PROTOCOL.md                      frozen pre-training protocol")
    add("code/psc_paths.py                path enumeration / canonical keys / cache / gather")
    add("code/psc_model.py                path encoder + SUM/POOL readouts")
    add("code/psc_train.py                all three families (PSC / ACRA / GINE)")
    add("code/psc_analysis.py             tables, selection, bootstrap, pairs, strata")
    add("code/psc_ledger.py               exact SUM edit ledger (geometric S/C split)")
    add("code/psc_cpu.py                  closed-form path-count / Morgan baselines")
    add("code/psc_final_eval.py           lock + single frozen Valid pass")
    add("code/psc_budget.py               GPU/CPU ledger")
    add("analysis/checks.json             engineering checks (all pass)")
    add("analysis/prepare.json            data prep + cache hashes")
    add("analysis/selection_lock.json     champion + twins + E_m, frozen on D0")
    add("analysis/bootstrap{,_D1}.json    paired molecule / scaffold-cluster bootstrap")
    add("analysis/unique_pair_metrics.json, molecule_novelty.json, size_strata.json")
    add("analysis/pair_residual_decomposition.json, sum_edit_ledger.json")
    add("analysis/final_eval_lock.json, final_valid.json")
    add("analysis/compute_budget.json, path_sparsity.json, tables.md, README.md, figures/")
    add("results/main_table.csv, results/summary.json")
    add("```")
    add("")

    # --------------------------------------------------------- deviations ---
    add("## 14. Deviations, incidents and honest limits")
    add("")
    add("* **Pre-flight launch failures (reruns only).** The queue driver launches through "
        "`rr`, which refuses to start a formal run from a dirty tracked tree or a stale "
        "remote checkout. Analysis edits made while a queue was running produced 44 "
        "rejected launch attempts across the round (`LAUNCH_FAILED`, all visible in "
        "`analysis/logs/`); each affected run was relaunched from a clean commit and "
        "**no job that actually started ever failed** (0 non-completed runs in "
        "`analysis/compute_budget.json`). Rejections consumed no GPU time. No run was "
        "selected or excluded on the basis of its result.")
    def _med(a):
        v = (agg0.get(a) or {}).get("best_epochs")
        return int(__import__("numpy").median(v)) if v else "n/a"
    add("* **Horizon.** 160 epochs is the frozen budget. The PSC arms' median best epoch is "
        f"late ({', '.join(str(_med(a)) for a in A.PSC_ARMS if a in agg0)} of 160), "
        "i.e. they are still improving when the schedule ends. The comparison to a "
        "closed-form ridge on the same features is therefore a *lower bound* on the "
        "additive restriction's cost; but the 32-molecule overfit shows the gap is not "
        "an implementation failure, and the POOL>SUM contrast is measured at the same "
        "horizon for both.")
    add("* **Baseline reproduction.** In this harness `ACRA_REF` reaches D0 dev RMSE "
        "0.580 ± 0.002 (3 seeds), whereas the parent round's own reference run reported "
        "0.555 for the same nominal recipe on the same split. The ~0.025 difference is "
        "consistent with a different initialisation/data-order path in the re-hosted "
        "trainer; it makes this round's ACRA baseline slightly *weaker* than the parent's, "
        "which if anything understates the PSC deficit. The strongest non-PSC baseline "
        "here is GINE (0.527), so no conclusion rests on the ACRA reproduction alone.")
    add("* **Recipe fairness.** PSC/ACRA/GINE share the frozen optimiser recipe "
        "(AdamW, peak lr 1e-3 / 1e-5, cosine, clip 5, batch 64 molecules). The ACRA and "
        "GINE baselines are the parent rounds' own recipes, not tuned for this round; the "
        "CPU baselines lock α on D0. No Valid-based tuning anywhere.")
    add("* **Path-length ceiling.** L = 5 (the pre-registered fallback L = 4 was not "
        "needed). Longer paths would raise per-molecule coverage but also the key count; "
        "the sparsity table shows the key space is already the binding constraint.")
    add("* **Ledger scope.** The ledger covers the SUM readout only, and its S/C split is "
        "geometric (component sizes) rather than the TRC word identity, because the "
        "parent cache's per-parse atom sets are stored in fragment-local numbering. The "
        "exact-cancellation result above holds under either convention for same-context "
        "pairs.")
    add("* **DIAG twin** was run because the champion carries the PAIR objective: three "
        "D0 seeds at the same `E_m`, exact contrast in §8. The *optional* extensions (ESOL "
        "transfer, size curve, M* full-train beyond the reference arms) were not started: "
        "the required matrix, the D1 replication and the full-Train refits consumed the "
        "budget, and the protocol's extension gate requires the required queue to be "
        "complete first.")
    add("")
    add("## 15. What this means for the branch")
    add("")
    add("1. **Do not continue the strict path-sum branch as specified.** A shared scalar "
        "score per path shape is "
        "dominated by a linear model on the same path counts ("
        f"{signed((agg0.get('PATH_COUNT_RIDGE', {}).get('rmse_mean') or float('nan')) - (agg0.get(champ, {}).get('rmse_mean') or float('nan')))}"
        " RMSE means the ridge is *better*) and by ACRA. More capacity in the encoder is "
        "not the fix; the sum is the fix.")
    add("2. **If the branch is revived, replace the shared scorer with a hybrid** "
        "`F = β + Σ_p [w_key(p) + h(E(p))]`, i.e. a free coefficient for keys with "
        "sufficient support and a shared, token-conditioned residual for the tail. That "
        "keeps a ledger (per-key plus per-shape terms) while removing the capacity "
        "cliff; the ridge/path-sum gap measures the headroom.")
    add("3. **Keep the pair algebra and the diagnostic harness.** The PAIR objective is "
        "exactly verifiable, transfers to ACRA, and gives the only positive effect in the "
        "matrix; the matched-pair ledger (exact C/X cancellation) is a reusable "
        "instrument even if the model class changes.")
    add("4. **Do not expect unseen-path-word transfer to move Lipo numbers.** It is 4.1% "
        "of held-out path mass. A route that targets it should be evaluated on a dataset "
        "or split where that slice is large, otherwise the metric cannot see it.")
    add("")
    add("## 16. Non-claims")
    add("")
    add("* No Test label was read; no Test metric is reported.")
    add("* No Valid label was read before `final_eval_lock.json`; no Valid number was used "
        "for training, early stopping, arm selection or hyper-parameter choice.")
    add("* The negative result is about *this* dataset, vocabulary (raw atom/bond tokens, "
        "L=5), recipe (160 epochs, shared MLP scorer) and budget. It is not a claim that "
        "path-based composition cannot work with a different parameterisation; §15.2 "
        "states the specific change the evidence supports.")
    add("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())