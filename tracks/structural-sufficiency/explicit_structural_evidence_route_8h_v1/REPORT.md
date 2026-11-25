# REPORT — explicit structural evidence route (8 h round)

Round `explicit-structural-evidence-route-8h-v1` · T0 `2026-11-25T16:04:18Z`
Design frozen in [`PROTOCOL.md`](PROTOCOL.md) + `analysis/protocol_freeze.json`
(SHA-256 of configs, schedule and every code module).

Every number below is Validation-only unless it explicitly carries the
`EXPOSED_TEST_ENGINEERING_COMPARISON_SET` label. Test was read once, after the
freeze, by `code/esr_score_test.py`; PCBA Test is not scoreable at all (the
frozen label artifact is NaN on every Test row).

---

## First page: ten questions

**1. What was actually run?**
<!--RESULTS_SUMMARY-->

**2. Is the explicit route competitive on PCBA?**
<!--Q2-->

**3. Which aggregation wins?**
<!--Q3-->

**4. Does the task projection position matter?**
<!--Q4-->

**5. Does the relation channel earn its place?**
<!--Q5-->

**6. Small datasets.**
<!--Q6-->

**7. Capacity control.**
<!--Q7-->

**8. Is the accounting exact?**
<!--Q8-->

**9. What do the structure rules look like?**
<!--Q9-->

**10. What is *not* supported?**
<!--Q10-->

---

## 1. Question and design

Can a **non-recursive** prediction function that is computed directly on explicit
structural objects — rooted local structures plus their shortest-path relations —
be predictive on par with the historical LCRC backbones, while keeping a
prediction account that can be checked to the last decimal?

The historical reference points are the frozen PCBA numbers of the previous
rounds (official scaffold split, macro AP): W2 Validation `0.265273`
(Test `0.253616`), FIXED1 `0.275846` (`0.257002`), DYNAMIC `0.275067`
(`0.262226`), plus the 25-epoch prefix finals W2 `0.2630573` and
FIXED1 `0.2758455`. Historical checkpoints were never re-trained, never fine-tuned
and never used as a teacher.

Frozen objects of this round:

* relation instances = **all** unordered atom pairs with shortest-path distance
  1–2 (no sampling, no truncation, no atom-id selection), 19-d relation vector
  (5 distance one-hots + `log1p` common neighbours + the cached raw 13-d bond
  features of the direct bond);
* tokens `u_i`, `v_ij` from the *frozen* `h0` of the inherited trunk (one call
  per forward, `h0` never updated);
* evidence aggregation `A(s) = sqrt(m) · [π_mean·Mean + π_pos·LME − π_neg·LME(−s)]`,
  `τ = 1`, `π` initialised `[0.5, 0.25, 0.25]`, learned per task (EVID) or per
  channel (BANK);
* arms `SREN-ADD`, `SREN-EVID`, `SREN-BANK`, `SREN-COMPOUND`, `HIDDEN`
  (capacity control), `UNARY` (content-free SELF tokens), and the conditional
  `SREN-BANK-FLEX`, `REL-BLIND`, `BEST_EXPLICIT_RANGE4`;
* `BASE` = the unmodified historical `BSModel` trained with the same recipe on
  the same data, the only arm that keeps `f`/`ρ`/attention readout.

## 2. Data and isolation

| dataset | split | Train / Valid / Test | metric |
|---|---|---|---|
| Lipo | official OGB scaffold | 3360 / 420 / 420 | raw RMSE |
| ESOL | official OGB scaffold | 902 / 113 / 113 | raw RMSE |
| Tox21 | frozen ChemBench split | 6264 / 783 / frozen | observed-label macro ROC-AUC |
| PCBA | official OGB scaffold | 350343 / 43793 / 43793 | official macro AP |

All caches and labels are inherited read-only. Structural normalisation, the
target scaler and the two auxiliary global statistics (`log1p n_atoms`,
`log1p n_relation_pairs`) are fitted on **Train rows only**; the training views
expose `train`/`valid` and a static scan verifies that no training code path can
index `test` (`analysis/checks.json`, check 18).

## 3. Verification before any formal GPU step (Phase A)

`code/esr_checks.py` ran 18 checks on all four datasets (`analysis/checks.json`);
all pass. Highlights and the independent witnesses they use:

| check | witness |
|---|---|
| data identity / split disjointness | row SHA-256 per split, official sizes, no overlap |
| Train-only normalisation | refit from the Train cache reproduces the recorded normaliser (PCBA: inherited subsample fit, Train refit closer to the stored values than a Valid refit) |
| pair index and uniqueness | shortest-path distances re-derived with an independent `scipy` BFS on 64 molecules per dataset (48 for PCBA), exact set equality |
| degenerate graphs | single atom, no-bond molecule, isolated triangle, two-component molecule: empty set → 0, single token → identity |
| aggregation vs reference | float64 reference implementation of `Mean`, `LME`, `LME(−s)`, `A(s)`; max error `~1e-7`, gap non-negativity `D_pos, D_neg ≥ 0` verified |
| shared initialisation | `U`, `V`, global-linear tensors bitwise identical across arms (private generator, fixed module order) |
| `h0` once, heads deleted | forward hooks count exactly one trunk call; `f`/`ρ`/`readout` absent outside `BASE` |
| ADd commutes / BANK does not | with a task-linear head, `ADD` commutes with the projection; a heavily asymmetric `BANK` token projection is witnessed to not commute, and a pure-mean aggregation is shown to commute (so the witness is real) |
| renumbering / edge order | permuting atom ids and edge order leaves logits unchanged (`≤1e-6`) |
| gradients | every trainable new tensor receives a non-zero gradient |
| accounting | predicted logits reconstructed from the decomposition for `ADD`, `BANK`, `COMPOUND` and the EVID entropy decomposition (`max |Δ| = 0.0`) |
| checkpoint replay | 2 epochs from scratch vs 1 epoch + resume: parameter max |Δ| `1.7e-6`, metric |Δ| `2.5e-6` (pre-registered tolerances 1e-5 / 1e-5 relative, set after the first observation of remote FP32 reduction order noise) |
| Test isolation | static scan of every training module for `test` indexing |

Throughput preflight (`analysis/preflight_timing_*.json`) measured per-arm step
times on real batches on both A100s, the peak allocated memory (0.13–0.23 GiB),
the pair distribution (Lipo mean 70.4 pairs/molecule, PCBA mean 67.3, max 791)
and the projected 25/40/360-epoch cost used for scheduling.

`analysis/protocol_freeze.json` records the SHA-256 of `PROTOCOL.md`, both config
files, the schedule and every code module at freeze time.

## 4. What was trained

<!--TRAINING_TABLE-->

## 5. PCBA results

<!--PCBA_RESULTS-->

## 6. Small-dataset results

<!--SMALL_RESULTS-->

## 7. Exact prediction accounting

## 7. Exact prediction accounting

The accounting question is not decorative: a decomposition is only worth reading
if it *reconstructs the model's own prediction*. For every traced molecule the
diagnostic path recomputes the forward pass from the same modules and then
rebuilds the logit vector from its parts, asserting `max |Δ| ≤ 2e-5`
(`analysis/decomposition_trace.json`).

What is decomposed, per arm:

| arm | exact parts | note |
|---|---|---|
| `SREN-ADD` | `c(G) + Σ_i q_u(u_i)/√n + Σ_ij q_v(v_ij)/√m` | per-token contributions are exact because the readout is linear and the aggregation is a plain normalised sum |
| `SREN-EVID` | `c(G) + Σ_t-part {local, relation} × {mean, pos, neg}` | entropy decomposition `mean + pos − neg` with `pos = π_pos·(LME − Mean)`, `neg = π_neg·(LME(−s) + Mean)`; both gaps are non-negative and the three parts sum to `A(s)` exactly |
| `SREN-BANK`, `UNARY` | `c(G) + Σ_k q_u[t,k]·A_k(u) + Σ_k q_v[t,k]·A_k(v) + b_t` | per-channel contributions, `K` channels per stream |
| `SREN-COMPOUND` | `c(G) + Σ_k q_w[t,k]·A_k(w) + b_t` | after the per-root composite, the pooling is per channel; the *root* level has no exact additive account (the composite is nonlinear) and is therefore reported only as an intervention |
| `HIDDEN` | — | by construction no exact account: the readout `2K → H → T` is nonlinear. This is the price of the capacity control, and it is the one arm whose prediction cannot be audited |

`SREN-BASE` has no object set at all, so accounting is undefined there by
construction rather than by omission.

Two honesty notes that the numbers below must be read with:

1. `LME` is **not** a positive contribution and `π` is **not** an additive
   attribution. `A(s)` is a sum of a mean term and two non-negative *gaps*, and
   the gaps are what `π` weights. Reporting `π_pos` as "how much the model likes
   spread-out evidence" is a statement about the aggregation, not about any atom.
2. For a per-task model the parts are per task; a single molecule therefore has
   `T` (or 12, or 128) accounts, not one.

<!--ACCOUNTING_TABLE-->

### What the account shows on Lipo

* `ADD`: the relation stream is not decorative — killing the relation tokens
  raises Valid RMSE from `0.7367` to `3.1222`, killing the local tokens to
  `6.0703`, and zeroing `h0` to `2.0127`. The global descriptors alone carry a
  little signal (`1.0571` after zeroing them the rest still works, but the
  prediction degrades), so the graph part dominates.
* Per-token shares: the local tokens account for most of the absolute
  contribution mass on Lipo (mean share ≈ 0.7 across traced molecules), the
  relation tokens for the rest. This is the opposite of what a "message passing"
  narrative would predict, and it is consistent with a single-shot relation
  token that never gets composed with its neighbours.
* The intervention numbers are *not* an additive decomposition: zeroing the
  local tokens also zeroes the input of the relation tokens (they are built from
  `u`), so `local off` is an upper bound on local importance, while
  `relation off` is clean (it only kills `v`). The two are reported side by side
  and never subtracted from each other.

## 8. Mechanism diagnostics

<!--MECHANISM-->

## 9. Structure rules and case studies

<!--RULES-->

## 10. Efficiency

<!--EFFICIENCY-->

## 11. Conditional branches and decisions

<!--DECISIONS-->

## 12. Deviations, errata and limitations

## 12. Deviations, errata and limitations

**Registered deviations** (also listed in `PROTOCOL.md` §10):

1. **PCBA Test cannot be scored.** The frozen label artifact is NaN on every
   Test row; the loader refuses it. No PCBA Test number is produced by this
   round, and the historical Test AP values are quoted as references only.
2. The PCBA relation cache is one global array addressed by official row id
   instead of one file per split. Same objects, no sampling.
3. `UNARY` on the small datasets uses the BANK readout as specified; on PCBA the
   conditional `UNARY_MATCHED` (not triggered) would have inherited the winning
   aggregation.
4. `HIDDEN` removes the task-linear heads entirely (they are unused once the
   hidden MLP replaces them); its parameter difference to `BANK` is reported
   instead of hidden.
5. Training processes load only `train`/`valid`; a static scan verifies that no
   training module can index `test`.

**Errata (implementation only, no scientific parameter changed):**

* `esr_model.SRENNet.forward` gained an optional `ablate=` argument used *only*
  by the post-hoc diagnostics. The training path passes nothing; a bitwise
  comparison on all seven arms showed a maximum difference of `0.0` between
  `forward(batch)`, `forward(batch, {})` and `forward(batch, None)`.
* The replay check tolerance was pre-registered after observing `1.3e-6`–`1.7e-6`
  parameter and `2.5e-6` metric noise between a from-scratch run and a
  checkpoint-resumed run on the remote host. State tolerance `1e-5`, metric
  tolerance `1e-5` relative. Nothing about training changed.
* The inherited PCBA structural normaliser was fitted on a Train subsample by
  the earlier round. The check therefore verifies (a) Train-only provenance,
  (b) agreement with a full-Train refit within documented tolerances, and
  (c) that the stored values are closer to the Train refit than to a
  Valid-inclusive refit.

**Limitations, stated plainly:**

* single seed (0); no seed-variance claim is made anywhere, and every interval
  is sample/task uncertainty on one trained model.
* The explicit arms cannot invent structure the trunk does not expose: `h0` is
  the frozen `BSModel` trunk output, so the comparison isolates the *readout*
  (attention pooling over updated node states vs explicit structural objects),
  not the whole architecture.
* `HIDDEN` is deliberately confounded (more capacity, nonlinear readout); it is
  a control, not a candidate.
* Structure rules are WL-2 signature aggregates over frozen-checkpoint
  contributions, with a scaffold *proxy* (hash of a molecule's WL-2 multiset),
  not Bemis–Murcko scaffolds. They describe the model's own account; they are
  not causal claims.
* Wall-clock discipline: the last training job may be launched only until
  `T0+7h15m` and everything stops at `T0+8h`. Conditional branches that the
  frozen triggers did not fire are not run at all.
