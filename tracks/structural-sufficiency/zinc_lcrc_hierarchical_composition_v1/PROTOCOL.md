# PROTOCOL — LCRC-HC3-v1

**Round id:** `lcrc-hc3-v1`
**Track:** `tracks/structural-sufficiency/zinc_lcrc_hierarchical_composition_v1/`
**Status:** frozen before the first formal GPU run (2026-11-07). This file is the
pre-registration. Engineering resolutions and fixes are appended to
`TREE_LEDGER.md`, never silently folded into results.
**Upstream:** `zinc_local_relation_composition_v1` (frozen LCRC-A mathematics,
code, caches and train10k baselines), `zinc_compositional_research_tree_v1`
(CMA input semantics, Stage IV predictions), `zinc_secondhop_composition_pilot_v1`
(the old two-hop fixed-projection control).

---

## 0. Question and boundary

Can a **shared hierarchical three-bond composition operator** inside LCRC-A
simultaneously beat (i) a same-information direct encoder of the same paths and
(ii) the original strong LCRC-A, under official train10k label limits?

The round is **implementation, correctness, execution and honest reporting**.
It does not redesign the architecture, add models, tune widths/LRs/path length/
losses, or read official test.

Three arms, same frozen LCRC trunk and same recipe:

| arm | composition |
|---|---|
| **B** | frozen LCRC-A, no new composition |
| **H** | hierarchical: shared 3-atom fragment `F2`, then shared `F3` over two adjacent fragments |
| **F** | flat: one direct encoder of the whole four-atom / three-bond path |

---

## 1. Frozen structural interface (H and F share it)

Enumerate every **directed** simple three-bond path `(i,j,k,l)` for every
undirected physical centre bond `e={j,k}`, both centre orientations, four
distinct atoms. A path and its reverse are separate records. Per record store
the four atoms, the three physical bonds `(i-j, j-k, k-l)`, the centre bond
`j-k`, and the three chord types
`(type(i,k), type(j,l), type(i,l))` with
`type ∈ {0 no edge, 1/2/3 bond type + 1}`.

* `n_e` = number of **undirected** paths whose centre bond is `e`
  (`directed_per_bond / 2`, exact).
* `q_path ∈ R^64` per directed record (defined in §2/§3).
* `m_e = sum_oriented q_path / (2 * sqrt(1 + n_e))`, and `m_e ≡ 0` exactly when
  `n_e = 0`.
* `r_v = sum_{e incident v} m_e / sqrt(1 + degree_v)`.
* `W_comp: Linear(64,64)` with **zero weight and zero bias**.
* Injection: `h_v = LN(MLP(X_v) + W_comp(r_v))`, i.e. the projection is added
  immediately before the frozen `LocalContent.norm`. Nothing else in
  `LocalContent`, `f`, `rho`, the two-hop statistics or the trunk changes.

At step 0 `W_comp ≡ 0`, so H and F must reproduce the same-seed frozen LCRC-A
forward bit-for-bit. The projection is updated at step 1; the path encoder gets
a non-zero gradient from step 2 onward (its Jacobian is multiplied by the
still-zero `W_comp` at the first backward pass). The CPU gates check both.

---

## 2. H — hierarchical composition

* Three-atom fragments `(i,j,k)` and `(j,k,l)`:
  `F2: 484 -> 96 -> 48`, SiLU, input
  `[a_centre, a_outer1, a_outer2, b_centre-outer1, b_centre-outer2, onehot(chord)]`.
  `z_ijk = 0.5 [F2(i,j,k) + F2(k,j,i)]` (reversal-symmetric).
* Adjacent fragments sharing the centre bond:
  `F3: 252 -> 96 -> 64`, SiLU,
  input `[zL+zR, |zL-zR|, zL*zR, b_centre, r_chord]`, where
  `r_chord = [oh_ik + oh_jl, |oh_ik - oh_jl|, oh_il]` (12 dims, exactly the
  frozen CMA definition).
* `q_path = F3(...)`, then the shared §1 aggregation.

New parameters: `F2 51,216 + F3 30,496 = 81,712`, plus `W_comp 4,160`.
**Total 323,391.**

---

## 3. F — same-information direct encoding

For the same directed path, one call of the same encoder on
`[a_i, a_j, a_k, a_l, b_ij, b_jk, b_kl, onehot(chord_ik), onehot(chord_jl), onehot(chord_il)]`
(dimension **684**, `4*96 + 3*96 + 3*4`).
`Fflat: 684 -> 109 -> 64`, SiLU. The reversed path is encoded by the same
`Fflat` (bonds reversed, `chord_ik ↔ chord_jl`, `chord_il` unchanged) and
`q_path = 0.5 [Fflat(forward) + Fflat(reverse)]`, which is exactly
reversal-invariant.

Then the same `m_e`, `r_v`, `W_comp` and injection as H.

New parameters: `Fflat 81,705`, plus `W_comp 4,160`.
**Total 323,384.** H and F differ by exactly **7 parameters**.

F is **not** an average-then-nonlinearity CMA-MEAN control; each direction is
encoded first and only the two 64-d outputs are averaged.

---

## 4. Shared trunk and initialisation

* The trunk is the frozen `lc_models.py` (`atom_encoder`, `bond_encoder`,
  `LocalContent` statistics, `f`, `rho`, buffers). B is constructed through the
  frozen `build_model("A", seed)`.
* H/F construct the frozen `LCRC` under `torch.manual_seed(seed)` with the same
  RNG consumption, load the same seed SSL atom/bond encoders, replace the
  `LocalContent` instance by an identical subclass that accepts the injection
  vector (same state-dict keys and tensors), then draw `W_comp` and the path
  encoder under `seed + 771003`.
* H/F therefore share (a) the complete initial trunk with B, and (b) the same
  `W_comp` initial state with each other. Their path encoders use the same
  dedicated seed but different shapes, as required by the comparison.
* Hard gate: per-tensor equality with the frozen builder and the historical
  initial hashes `11c698ae…` (seed 0) / `d9422859…` (seed 1); step-0 H/F == B
  bit-for-bit for both seeds.

---

## 5. Data, split and recipes

* Official PyG ZINC `subset=True` train10k and valid1k. **Official test is never
  opened.** `load_cache("test")`, `cache_path("test")` and
  `structure.load_split("test")` raise.
* Internal split: frozen `bjoc_predictive_bottleneck_v1/split_a.json`
  Fit8k 8000 / Monitor 1000 / Locked 1000, row hashes re-verified on every load.
* Stage A/B: train on Fit8k only; select on Monitor. Fit8k-only normalisation
  (atom scalars and global33) for all three arms.
* Stage C: train on all official train10k; select on official valid. Use the
  historical full-train10k normalisation (`zinc_local_relation_composition_v1/
  analysis/norm.json`), so the historical LCRC-A checkpoint is an exactly
  matched baseline.
* Recipe, identical for B/H/F and both fits:
  AdamW, peak LR 1e-3, min LR 1e-5, 500-update linear warm-up then per-update
  cosine, weight decay 1e-5, batch 128, raw-y L1, gradient clip 5, no dropout,
  **exactly 47,400 optimizer updates**. Steps/epoch = 63 (Fit8k) or 79
  (train10k); epochs = ceil(47400 / steps), last epoch truncated. Full
  Monitor/valid every epoch; strict-minimum checkpoint, ties keep the earlier
  epoch.
* Data order: `torch.Generator().manual_seed(seed + 101)`, one `randperm` per
  epoch, same for all arms. Resume granularity is one epoch and is verified
  bit-exact.
* No label-derived value enters a forward batch; labels are used only in the
  loss and in analysis grouping.

---

## 6. Stages, purchase gates and stop rules

All deltas are `MAE(target) - MAE(reference)`; negative means the first arm is
better. Paired bootstrap 5000 resamples over rows, fixed seed; CIs are reported
but the purchase gates below are point-improvement gates.

### Stage A — Fit8k / Monitor1k, seed 0

Run B, H, F with the frozen recipe. Purchase gates (all must hold):

1. `Δ(H-B) full ≤ −0.004`
2. `Δ(H-F) full ≤ −0.003`
3. `Δ(H-B) ordinary ≤ −0.002` (`ordinary: y > −3`)
4. `Δ(H-B) unseen_ordinary ≤ −0.004` (radius-2 unseen vocabulary = Fit8k, same
   construction as the frozen SSRC audits)
5. **Not mainly fixed worst-10:** let `W10` be the 10 Monitor rows with the
   largest `|error|` of the frozen DCC-L Fit8k seed-0 best checkpoint (fixed
   list already recorded upstream). Require
   `Δ(H-B) on Monitor \ W10 ≤ −0.002` and the `W10` share of the net
   `AE(B) − AE(H)` mass `≤ 0.50`.
6. **Smoothing and best agree:** `W20(H) − W20(B) ≤ −0.004` and
   `W20(H) − W20(F) ≤ −0.003`, where `W20` is the minimum 20-epoch rolling
   median of the Monitor curve; `T100(H) − T100(B) < 0` and
   `T100(H) − T100(F) < 0`; `final(H) − final(B) ≤ 0` and
   `final(H) − final(F) ≤ 0`; and the best-checkpoint signs of gates 1–2 are
   reproduced by `W20` and `T100` (no sign reversal).

If any gate fails: **STOP**. No seed 1, no Stage C, no width/LR/path/loss
changes. The negative result is reported as-is.

### Stage B — Fit8k / Monitor1k, seed 1

Only if Stage A passes. Run B, H, F with the same recipe. Requirements:

* `Δ(H-B) full < 0` and `Δ(H-F) full < 0` (same direction as Stage A);
* `W20` deltas for H-B and H-F are also `< 0` (best and smoothing agree);
* no substantial reversal in the structure groups:
  `Δ(H-B) ordinary ≤ +0.001` and `Δ(H-B) unseen_ordinary ≤ +0.001`.

If these fail: **STOP**; Stage C is not purchased.

### Stage C — complete official train10k

Only if Stages A and B support the same proposition. Train H and F on all
train10k rows, select on official valid. B is the saved historical LCRC-A
same-seed full-train10k checkpoint (`lcrcA_s0`, `lcrcA_s1`), replayed and
verified against its saved valid predictions and cache hashes before use.
DCC-L is the mature external reference: `zinc_global_operator_search_v1/
analysis/paired_A_vs_dccl_per_molecule.npz` (4 seeds, same valid rows).

Support for Stage C is recomputed relative to the **full train10k structure**
(never the Fit8k vocabulary; see `ERRATUM_stage4_support.md`).

Report, for H vs F, H vs LCRC-A and H vs DCC-L (seed-wise and 4-seed mean):

* full / ordinary / unseen-ordinary on valid;
* the fixed extreme-row leave-outs: the 10 valid rows with the largest
  `|error|` of the historical LCRC-A seed-0 best checkpoint (fixed before any
  Stage C run) and valid row 172;
* best checkpoint, W20 rolling, T100 and final;
* tail (`y ≤ −3`, 65 rows) **separately from** the artificial ring penalty
  (`label_excess > 0`, 35 rows; source
  `tracks/ksvd/results/zinc_long_cycle_audit/valid_cycle_audit_label.csv`),
  including their overlap and the disjoint parts. The 65 tail rows and the 35
  penalty rows must never be pooled into one causal group.

Claim rules:

* **Performance breakthrough**: H beats LCRC-A on full **and** ordinary with
  paired-bootstrap 95% CI upper bound `< 0`, and beats the DCC-L seed mean on
  full **and** ordinary with the same property, and is not worse than LCRC-A on
  unseen-ordinary by more than `+0.001`.
* **Improvement over LCRC-A only**: H beats LCRC-A on full and ordinary but not
  DCC-L. Reported as an incremental improvement, not a breakthrough.
* **Hierarchical inductive-bias signal only**: H beats F on full/ordinary but
  does not beat LCRC-A. No performance-breakthrough claim and no automatic new
  model.
* Otherwise: no advantage, reported as such.

Official test is not read at any stage.

---

## 7. Budget

* Soft budget **6 A100·h**, hard cap **8 A100·h**, at most two GPUs in parallel.
* Every stage starts with a real **2-epoch timing** on the target fit/host; the
  projection uses `mean seconds/epoch × ceil(47400 / steps_per_epoch)`. Timing
  runs are tagged `_timing` and are never mixed with formal 47,400-update runs.
* Execution order: H and F in parallel on the two A100s, then B. Stage C: H and
  F in parallel, seed 0 first.
* Before each stage, compare `spent + projected` with the caps. If the hard cap
  would be exceeded, reduce the number of Stage C seeds (seed 0 only) or stop;
  never change the frozen model, recipe, gates or path length.
* Unspent budget is an acceptable outcome.

---

## 8. Reporting artefacts

`REPORT.md`, `DECISION.md`, `ERRATUM_stage4_support.md`, CPU gate JSON, the
three-arm parameter table, paired per-molecule comparisons, support-stratified
results, penalty-vs-tail decomposition, complete run/checkpoint provenance, and
`TREE_LEDGER.md`. The report must distinguish explicitly:

1. **Did the hierarchical composition prove more effective than the
   same-information flat encoder?** (H vs F)
2. **Did it actually improve the mature single model?** (H vs LCRC-A, H vs
   DCC-L)

These are separate questions and must not be conflated in the verdict.
