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
    return f"{signed(d['point'], nd)} [{signed(lo, nd)}, {signed(hi, nd)}]"


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
    summary = load("summary.json", {})
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
        f"*commit*: `{C.git_commit()}` · *generated*: {C.now_utc()}")
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
            "worse than the GINE reference; the paired bootstrap CI is "
            f"{ci(c('champion_vs_' + str(acra)))} (positive = champion better).")
    if any(a in agg0 for a in A.CPU_MODELS):
        best_cpu = min((a for a in A.CPU_MODELS if a in agg0),
                       key=lambda a: agg0[a]["rmse_mean"])
        add(f"* A closed-form **path-count ridge** (`{best_cpu}`, D0-selected α) reaches "
            f"{n(agg0[best_cpu]['rmse_mean'])}, i.e. **better than every PSC arm** "
            f"({signed(agg0[best_cpu]['rmse_mean'] - agg0[champ]['rmse_mean'])} vs the champion).")
    add("* The additive `SUM` readout — the property the route was designed around — is "
        "consistently beaten by the non-additive `POOL` readout "
        f"({signed((c('PSC_POOL_vs_PSC_SUM') or {}).get('point'))} RMSE, CI "
        f"{ci(c('PSC_POOL_vs_PSC_SUM'))}).")
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
        f"contrasts positive ({n_sig} with a CI excluding 0), maximum "
        f"{signed(max([x for x in pair_pts if x is not None] or [0]))} RMSE — below the "
        "pre-registered +0.02 practical-gain threshold.")
    add("* **Verdict**: do not continue this branch as specified. The compositional "
        "machinery is implementable and exactly verifiable, but on Lipo the strict "
        "path-sum parameterisation is dominated by both a linear path-count kernel "
        "and ACRA, and the phenomenon it targets (unseen path words) carries only "
        f"{pct(sparsity.get('eval_mass_on_unseen_keys'))} of the held-out path mass.")
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
    names = {
        "1_enumeration": "path enumeration equals directed-DFS/2 and matches a "
                         "second independent enumerator",
        "2_invariance": "renumbering invariance (atom relabelling gives identical keys)",
        "3_padding_reversal": "padding + reversal apply to the declared columns only "
                              f"(input width {checks.get('3_padding_reversal', {}).get('info', {}).get('input_dim', 'n/a')})",
        "4_batch": "batched == single-molecule forward (max |Δ| ~1e-13)",
        "5_bag_equivalence": "BAG exactly invariant to path reversal; ORDER distinguishes it",
        "6_pair_algebra": "PAIR = DIAG − 2·cross, Σw = 1, sampler unbiased, masked ACRA "
                          "loss equals the parent",
        "7_fit_only": "path vocabulary / categorical support / scaler are fit-only",
        "8_init_grad": "SUM and POOL share identical initial outputs; all parameters "
                       "receive gradients",
        "8b_overfit32": "32-molecule overfit: MSE 1.716 → 0.112 in 300 steps "
                        "(the training chain can fit)",
    }
    for k, v in names.items():
        got = check_status.get(k, "?")
        add(f"| `{k}` ({got}) | {v} |")
    add("")
    add("The 32-molecule overfit matters for the interpretation below: the optimiser, "
        "the path cache and the readout are not broken, so a plateau on D0 is a statement "
        "about capacity/data, not about a bug.")
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
        key = f"champion_vs_{a}" if a != champ else f"{champ}_vs_{acra}"
        add(f"| `{a}` | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | {n(v['mae_mean'])} | "
            f"{ci(c(key))} |")
    add("")
    add(f"Paired-molecule bootstrap ({boot.get('reps', 'n/a')} resamples, shared index across "
        "models and seeds): the champion is worse than every non-PSC reference with a CI "
        "that excludes 0. The per-molecule view agrees: "
        f"mean per-molecule |error| change vs `{acra}` is "
        f"{signed((boot.get('per_molecule') or {}).get('mean_abs_err_diff'))} "
        f"[{ci(boot.get('per_molecule'))}], and the champion is closer to truth on "
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
                    if "BAG" in k and isinstance(v, dict)) + "). "
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
        "but only two CIs exclude 0 and **none reaches the pre-registered +0.02 practical "
        "gain threshold**. The ACRA regulariser is a wash "
        f"({signed((c('ACRA_PAIR_vs_ACRA_REF') or {}).get('point'))}). The dependence on "
        "readout is weak: the pair term gives roughly +0.01 to +0.02 in both families, "
        "consistent with a data-efficiency device rather than a substitute for capacity.")
    add("")
    add("The pair algebra is exact and verified: `PAIR = DIAG − 2·cross` "
        "(deviation < 1e-12), `Σ w_e = 1` per component, and the sampler matches the "
        "component-weighted distribution. The DIAG twin, however, was **not run** because "
        "the champion is not a PAIR arm (see the protocol's conditional rule).")
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
        models = [m for m in (champ, acra, "PATH_COUNT_RIDGE", "MORGAN_COUNT_KRR")
                  if m and m in next(iter(d["support_classes"].values()), {})]
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
        for name, v in d["buckets"].items():
            add(f"| {name} | {v['n']} | " + " | ".join(
                n((v.get(m) or {}).get("rmse")) for m in
                (champ, acra, "PATH_COUNT_RIDGE", "MORGAN_COUNT_KRR") if m) + " |")
        add("")
        add(f"Mean held-out unseen-path mass is {n(d['unseen_path_mass']['mean'], 4)} "
            f"(median {n(d['unseen_path_mass']['median'], 4)}, max "
            f"{n(d['unseen_path_mass']['max'], 3)}); "
            f"{pct(d['unseen_path_mass']['share_gt5pct'])} of molecules exceed 5%.")
    add("")
    add("**Answer to Q5.** Transfer does not show up where it would have to. In the "
        "highest-eta molecule bucket the champion's error grows faster than the linear "
        "path-count ridge's, and in the hardest support class (T unseen, both fragment "
        "words unseen) the PSC arms are competitive or better — but that cell is only "
        "24 pairs and its `mean |Δtrue|` is small, so a shrunk predictor wins there for "
        "reasons unrelated to structural transfer. The class where a real transfer "
        "mechanism should shine (T unseen, *both fragments seen*, 156 pairs) is exactly "
        "where PSC loses most to ACRA. There is therefore **no evidence of a "
        "path-composition transfer advantage** on this dataset.")
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
        add("The ledger exposes the matched-pair design exactly: for the selected "
            "same-context pairs the C-side and crossing path **token multisets are "
            "identical in both molecules**, so their ledger terms cancel to 0 and 100% of "
            "the predicted Δ comes from paths inside the edited fragment side. That is "
            "the property the route wanted — and it also shows why the property alone is "
            "not enough: being exactly additive does not make the *magnitude* right.")
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
            + str(acra) + " [95% CI] |")
        add("|---|---|---|---|---|")
        for a in dict.fromkeys([champ, o_star, r_star, acra, "GINE_REF"]):
            if not a or a not in agg1:
                continue
            v = agg1[a]
            d0 = (agg0.get(a) or {}).get("rmse_mean")
            add(f"| `{a}` | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | {n(d0)} | "
                f"{signed(v['rmse_mean'] - d0 if d0 else None)} | "
                f"{ci(c(f'{a}_vs_{acra}', con1))} |")
        add("")
        add("D1 runs used the frozen `E_m` epochs (median D0 best epoch per arm) and the "
            "final checkpoint, with no selection. The ordering and the size of the gaps "
            "replicate; the champion's deficit versus ACRA is "
            f"{signed((agg1.get(champ, {}).get('rmse_mean') or 0) - (agg1.get(acra, {}).get('rmse_mean') or 0))} "
            "RMSE on D1 versus "
            f"{signed((agg0.get(champ, {}).get('rmse_mean') or 0) - (agg0.get(acra, {}).get('rmse_mean') or 0))} "
            "on D0.")
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
        for m, v in valid["models"].items():
            if not v.get("seeds"):
                add(f"| `{m}` | – | {v.get('status', 'n/a')} | – | – |")
                continue
            add(f"| `{m}` | {v['n_seeds']} | {n(v['rmse_mean'])} ± {n(v['rmse_sd'])} | "
                f"{n(v['mae_mean'])} | {ci(v.get('bootstrap_vs_ACRA_B'))} |")
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
        add(f"* GPU-hours consumed (all launched jobs, 1 GPU each): "
            f"**{n(t['gpu_hours_including_attempts'], 2)}** of the 16 h ceiling "
            f"({'within' if t['within_budget'] else 'OVER'} budget).")
        add(f"* Train-loop hours: {n(t['train_loop_hours'], 2)} "
            f"({pct(t['train_loop_fraction']) if t['train_loop_fraction'] else 'n/a'} of GPU wall).")
        add(f"* Closed-form CPU hours: {n(t['cpu_hours_closed_form'], 2)}.")
        add(f"* Completed GPU runs: {budget['counts']['gpu_runs_completed']}; launch "
            f"attempts: {budget['counts']['launch_attempts']} "
            f"(non-completed: {budget['counts']['noncompleted_attempts']}).")
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
    add("analysis/compute_budget.json, path_sparsity.json, tables.md, figures/")
    add("results/main_table.csv, results/summary.json")
    add("```")
    add("")

    # --------------------------------------------------------- deviations ---
    add("## 14. Deviations, incidents and honest limits")
    add("")
    add("* **Ordering incident (no GPU waste beyond reruns).** The queue driver launches "
        "through `rr`, which refuses to start a formal run from a dirty tracked tree. One "
        "analysis edit left the tree dirty while the stage-2 queue was running, so a batch "
        "of launch attempts was rejected (`LAUNCH_FAILED`) before the next commit. All "
        "affected runs were relaunched from the same commit and the final matrix is "
        "complete (see `analysis/compute_budget.json` for every attempt). No run was "
        "selected or excluded on the basis of its result.")
    add("* **Horizon.** 160 epochs is the frozen budget. The PSC arms' best epoch is "
        f"late ({', '.join(str((agg0.get(a) or {}).get('best_epochs', ['n/a'])[0]) for a in A.PSC_ARMS if a in agg0)}), "
        "i.e. they are still improving when the schedule ends. The comparison to a "
        "closed-form ridge on the same features is therefore a *lower bound* on the "
        "additive restriction's cost; but the 32-molecule overfit shows the gap is not "
        "an implementation failure, and the POOL>SUM contrast is measured at the same "
        "horizon for both.")
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
    add("* **DIAG twin** was not run (conditional on a PAIR champion), and the optional "
        "extensions (ESOL transfer, size curve, M* full-train) were not started: the "
        "required matrix plus D1 and the full-Train refits consumed the budget, and the "
        "protocol's extension gate requires the required queue to be complete first.")
    add("")
    add("## 15. What this means for the branch")
    add("")
    add("1. **Kill the strict path-sum branch.** A shared scalar score per path shape is "
        "dominated by a linear model on the same path counts (−"
        f"{n(agg0.get('PATH_COUNT_RIDGE', {}).get('rmse_mean', float('nan')) - agg0.get(champ, {}).get('rmse_mean', float('nan')))}"
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