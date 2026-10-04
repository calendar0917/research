# EVIDENCE_SCOPE — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

Written before the new M_J dev score, to bound what existing evidence does and does not establish.

1. **The D_J vs B gain is not a confirmed breakthrough.** In
   `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`, D_J is active and exactly
   8-of-64 sparse per tuple, but its calibrated gain over B is only G0 `+0.001459` and overall
   `+0.001810` with CIs crossing zero; the J−I calibrated G0 gain is `+0.003575` with CI
   `[−0.000230, +0.007270]` also crossing zero. Neither the D_J>B nor the J>I calibrated claim is
   established. This round reads both as exploratory context only.

2. **The old posterior-bridge D_g/M_g comparison is not this round's contrast.** The completed
   `zinc_chemistry_dictionary_vs_mlp_seed0_v1` round did not resolve dictionary-vs-MLP at the
   posterior bridge. This round changes **only the local tuple encoder in front of fusion layer 1**;
   the posterior 144→288→144 MLP bridge is kept exactly as in D_J/B. No D_g/M_g conclusion is
   inherited or restated.

3. **Previous-round recipe wording is corrected by source/meta.** The real previous training ran
   training seed 0, batch 128, Adam lr 1e-3 with coupled WD 1e-5 and clip 5.0, 240 epochs /
   15,120 steps, soup epochs 236–240, and the IHT encoder starts from an all-zero code with 10
   steps. Text in older reports (e.g. "batch 32", "seed 20261004", "projected initial code") does
   not describe the actual run: 20261004 is the fold/frame/kappa/bootstrap seed, and the code is
   initialised to zeros. This round copies the verified recipe, not the prose.

4. **Zero-ablation proves dependence, not information share.** Zeroing the local channel after
   training degrades predictions drastically, but that intervention includes a large systematic
   output shift and breaks co-adapted downstream weights. It establishes that the network depends
   on the local term, not that the term "carries most of the effective information". This round
   adds a fit-mean replacement as a weaker intervention, and still does not claim a unique causal
   information share from it.

5. **Historical mixed-12000 label lineage stays closed.** No old `stage_refine` constant or
   `label_effective_cycle` / `train_cycle_audit_label` label column is refit, traced or reused.
   The new dependency chain reads the frozen `fit_only_targets.npz` (sha256 `e2adf5f2…`) that was
   already audited as refit on the 8000 fit rows only.

**Scope of this round**: exactly one new formal arm (M_J), one seed, one fold, 240 epochs; D_J and
B read-only; dev-only analysis; no official-valid/test; no ring head; no 10k confirmation. All
performance statements about this repeatedly used dev fold remain exploratory.
