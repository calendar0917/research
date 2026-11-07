# LCRC PCBA Cross-atom Predictive Evidence (CPE) v1 — report

> **single-seed architecture exploration**
> **historical W2 baseline**
> **same-round paired P/S mechanism control**
> **not independently confirmed**

Round id `lcrc-pcba-cpe-v1`. Frozen design in [`PROTOCOL.md`](PROTOCOL.md),
operational guide in [`README.md`](README.md), numeric artifacts in
[`analysis/`](analysis/).

## 0. One-line result

Adding an explicit, permutation-invariant **cross-atom second-order evidence term**
to the frozen `W2` trunk — computed with `i != j` pair statistics and a
multiplicative gate on the existing per-atom evidence vectors — trains stably,
learns a large pair coefficient, and **does not improve official Validation
macro AP**: `P` reaches **0.260853** against the historical `W2` **0.261867**
(Δ `−0.001014`). The parameter- and compute-matched **same-atom control `S`**
reaches **0.261010**, i.e. the cross-atom statistic is **not** better than the
same-atom one either (Δ `P − S = −0.000157`). Pre-registered label:
**`NO_CLEAR_GAIN`**.

| arm | pair statistic | params | new params | Val AP@40 | Δ vs W2 | Δ P−S |
|---|---|---:|---:|---:|---:|---:|
| `W2` (historical, not retrained) | — | 994,324 | — | 0.2618669181 | — | — |
| `P` | cross-atom `(Σ_A Σ_B − Σ_AB)/(n(n−1))` | 1,027,732 | 33,408 | **0.2608529705** | −0.0010139477 | −0.0001572972 |
| `S` | same-atom `Σ_AB/n` | 1,027,732 | 33,408 | **0.2610102676** | −0.0008566505 | — |

Everything below is seed 0, one run per arm, from scratch, official split,
40 epochs, epoch 40 as the only primary number. Best-checkpoint values are
reported for diagnosis only and were never used for any selection.

---

## 1. Are the two arms parameter-identical?

Yes, exactly, and this is audited rather than asserted.

| quantity | `P` | `S` |
|---|---|---|
| total parameters | 1,027,732 | 1,027,732 |
| base `W2` parameters | 994,324 | 994,324 |
| this round's new parameters | 33,408 | 33,408 |
| epoch-0 base (`W2`) sub-state SHA-256 | `6a3df38f…` | `6a3df38f…` |
| epoch-0 full `state_dict` SHA-256 | `f37e3e17…` | `f37e3e17…` |
| per-parameter names / shapes / dtypes | identical sets | identical sets |
| epoch-0 tensor values (all 1,027,732) | bitwise equal | bitwise equal |

New tensors, identical in both arms: `evidence.linear_A.{weight,bias}`
`[128,128]`,`[128]`; `evidence.linear_B.{weight,bias}` `[128,128]`,`[128]`;
`evidence.alpha_A`, `alpha_B`, `alpha_pair` `[128]`. That is
2·(16,384+128) + 3·128 = 33,408.

The only difference between the two arms is a **non-parameter python attribute**
`evidence.pair_mode ∈ {"cross","self"}`, checked to be the sole differing entry
of the two model configs (`analysis/architecture_audit.json`,
`analysis/initialization_parity.json` → `all_passed: true`).

`architecture_audit.py` ran 34/34 checks, `initialization_parity` reported
`all_passed: true`, and the mechanism audit re-verified the shipped epoch-40
checkpoints (`all_passed: true`).

## 2. Are `W2` initialisation, batch order and dropout paired?

**Initialisation.** The new modules are constructed inside
`torch.random.fork_rng(devices=[])` from a **private CPU generator with seed 37**,
using the standard PyTorch `nn.Linear` uniform scheme
(`kaiming_uniform_(a=√5)` + `U(±1/√fan_in)`), with all three `alpha_*` set to
exactly zero. The global RNG is verified unchanged before/after construction, so
the `W2` trunk is drawn from exactly the historical seed-0 stream. Evidence:

- `W2` trunk prefix parameters are **bitwise identical** to the historical `W2`
  init fingerprint (`trunk_init_sha256` `5f7101a8…`, full `109756bd…`), and both
  arms' `W2` sub-state hashes are equal (`base_state_sha256` `6a3df38f…` for `P`
  and `S`).
- With `alpha_* = 0` the evidence branch contributes exactly zero, so the epoch-0
  forward pass must reproduce `W2`. It does: initial Validation AP
  `P` 0.02906309 / `S` 0.02906300 / `W2` (replayed and in the historical run.json)
  0.02906306–0.02906333; **max |ΔAP| = 6.7e-8**. Epoch-0 logits on CUDA differ by
  ≤3.8e-6 (float non-determinism in the reduction), well inside the 1e-4 gate.

**Batch order.** Same frozen plan, verified from the runs themselves:
`global_plan_sha256 = 06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`,
40-epoch consumed prefix `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58`
in both runs, `order_seed = 0`, `batch_size = 128`, `consumed_epochs = 40`.
`paired_training_audit.json` → `same_batch_plan: true`, `same_consumed_prefix: true`.

**Dropout / optimiser / schedule.** Identical code path, identical constants,
same seed 0, same `eval_every = 1`; `same_recipe: true`, `same_seed: true`,
`same_eval_cadence: true`. Only the *input* to the RNG differs — the evidence
modules consume no randomness at all, so per-step dropout masks are identical
between `P` and `S` as well.

## 3. Is the cross-atom term a real `i != j` computation?

Yes. For each molecule, over the `n` atom slots of the batch block,

```
sum_A  = Σ_i a_i          sum_B  = Σ_i b_i         sum_AB = Σ_i a_i·b_i
self   = sum_AB / n
cross  = (sum_A·sum_B − sum_AB) / (n·(n−1))        # n ≥ 2, masked to 0 at n = 1
```

`sum_A·sum_B` contains all `n²` ordered pairs; subtracting `sum_AB` removes the
`n` diagonal terms, and dividing by `n(n−1)` averages the remaining `n(n−1)`
off-diagonal products. So every atom pair `i ≠ j` contributes exactly once with
weight `1/(n(n−1))`, and **no atom contributes to its own cross term**. Both
statistics are `O(n)` (a sum reduction, not an `O(n²)` materialisation of pair
tensors). Both are computed in **both** arms — the arm only chooses which one
multiplies `alpha_pair` — so the compute and memory footprints are identical by
construction.

Independently verified in `analysis/architecture_audit.json`:

| check | result |
|---|---|
| vectorised `cross` vs explicit `O(n²)` double loop, 2 atoms | max abs diff 5.0e-8 |
| same, 3 and 5 atoms | 7.1e-8, 3.2e-8 |
| two atoms with `a=[1,−1]`, `b=[1,−1]` (co-occurrence ⇒ cross ≠ 0) | cross `0.5`, self `0` — anti-correlated pair is detected |
| graph with a single atom (`n = 1`) | cross exactly `0.0` (max abs 0.0) while self = 0.9123, so the zero is a mask, not a global zero |
| empty graphs in the batch (`n = 0`) | cross/self finite and zero, `n_empty` reported (2 in the sample) |
| swapping the A/B roles | cross, self, main-effect residual all invariant to 0.0 diff; the first-order term exchanges A↔B |
| atom renumbering / permutation of a real batch | logits invariant to 3.8e-6 (`P`,`S`,`W2` alike) |
| batch isolation (3 graphs of sizes 4/2/3 padded side by side) | `mean_A`, `cross`, `self` max abs diff **0.0** vs the same graphs run alone |
| `n_graphs`/`n_atoms` accounting | reported per forward |

Batch isolation matters because the reduction is over a padded block: the mask
is derived from the atom index `< n_i` of *that* graph, so padding and neighbours
contribute nothing (`0.0` difference, not merely small).

## 4. Did the implementation sneak in message passing, attention, or iteration?

No, and the code is auditable on this point:

- **One trunk encode.** `forward` calls `encode_h` once, then `post_local` once,
  then the evidence module once. `call_counts` measured by
  `count_module_calls` for one forward: `atom_encoder 1, bond_encoder 1, local 1,
  f 1, rho 1, evidence 1, evidence_linear_A 1, evidence_linear_B 1,
  readout 1, dropout 1`.
- **`dropout` count is 1** — the round does not re-enter the original dropout a
  second time, and the evidence module contains no `Dropout` at all.
- The evidence term is a **per-molecule scalar gate per task** applied to an
  atom-vector residual: `residual[m,t] = Σ_{i∈m} tanh(W_A h_i)[t]·α_A[t] +
  Σ_i tanh(W_B h_i)[t]·α_B[t] + stat[m]·α_pair[t]`. There is no cross-atom
  **tensor** multiplication, no QKV, no attention score, no atom re-encoding, no
  propagation along bonds, and no iteration — `i` and `j` interact only through
  the two scalar sums.
- No new motif, path, radius, S2 gate, loss term or regulariser was added; the
  loss is the unchanged masked BCE, and `D`/`E` are the unchanged `W2` heads.

## 5. Official Validation macro AP at epochs 10 / 20 / 25 / 30 / 40

| epoch | `W2` | `P` | `S` | `P−W2` | `S−W2` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0958246 | 0.0958627 | 0.0958466 | +0.000038 | +0.000022 |
| 2 | 0.1542072 | 0.1543867 | 0.1543625 | +0.000180 | +0.000155 |
| 3 | 0.1784280 | 0.1784855 | 0.1786705 | +0.000058 | +0.000243 |
| 5 | 0.2149818 | 0.2152603 | 0.2152872 | +0.000279 | +0.000305 |
| **10** | **0.2431892** | 0.2429183 | 0.2426133 | −0.000271 | −0.000576 |
| 15 | 0.2526963 | 0.2525408 | 0.2524198 | −0.000156 | −0.000277 |
| **20** | **0.2586349** | 0.2574365 | 0.2571246 | −0.001198 | −0.001510 |
| **25** | **0.2630573** | 0.2611670 | 0.2621906 | −0.001890 | −0.000867 |
| **30** | **0.2620222** | 0.2605183 | 0.2614570 | −0.001504 | −0.000565 |
| 35 | 0.2640862 | 0.2625668 | 0.2627107 | −0.001519 | −0.001375 |
| **40** | **0.2618669** | 0.2608530 | 0.2610103 | −0.001014 | −0.000857 |

Full 40-point curves: `runs/P/train_curve.csv`, `runs/S/train_curve.csv`,
`analysis/paired_comparison.json → ap_by_epoch`. Shape: `P` and `S` are ahead of
`W2` by ~1e-4 for epochs 1–5, cross below it between epochs 8 and 12, and stay
0.5–1.9e-3 below for the last 25 epochs. The gap is largest around epoch 25 and
narrows to ~1e-3 at epoch 40.

Best (diagnostic only, never used): `W2` e24 0.2652733 · `P` e35 0.2625668 ·
`S` e31 0.2633532. The spread between best and final (2.1e-3 / 1.7e-3 / 3.4e-3)
is much larger than the between-arm differences, i.e. endpoint noise of this
recipe is on the same order as everything this round set out to measure.

## 6. Does `P` beat the historical `W2` AP 0.2618669?

**No.** `P` = 0.2608530, Δ = **−0.001014**. The pre-registered bands put
`|Δ| < 0.002` in the "no clear band", so this is not a regression call either;
it is a null. For scale: the `W2` epoch-to-epoch standard deviation over the last ten epochs is 5.9e-4, so a
−1.0e-3 endpoint difference is inside run-to-run wobble —
and, importantly, it is **not** evidence of harm, only absence of gain.

## 7. Does `P` beat the same-atom control `S`?

**No.** `S` = 0.2610103, so `P − S` = **−0.000157**, i.e. ~19× below the +0.003
mechanism threshold and of the wrong sign. On the 107 high-support assays
`P − S` is +0.000601 (64 improved / 43 regressed) — again indistinguishable from
zero.

This is the informative null of the round: the cross-atom statistic is not
*worse* than the same-atom statistic, but it is not *better* either, at equal
parameters, equal data order, equal dropout stream, equal compute. Whatever the
pair gate contributes, it is not the off-diagonal vs diagonal distinction.

## 8. Do the ≥20-positive assays improve broadly?

No. 107 assays carry ≥20 Validation positives (task 45 has 0 positives and is
not scoreable, so 127 of 128 tasks are ranked).

| population | n | improved | regressed | mean Δ | median Δ |
|---|---:|---:|---:|---:|---:|
| all scoreable, `P−W2` | 127 | 60 | 67 | −0.001014 | −0.0000865 |
| support ≥20, `P−W2` | 107 | 52 | 55 | **+0.0000320** | −0.0000399 |
| support 5–19, `P−W2` | 18 | 8 | 10 | −0.005677 | −0.000697 |
| support <5, `P−W2` | 2 | 0 | 2 | −0.014999 | −0.014999 |
| all scoreable, `S−W2` | 127 | 59 | 68 | −0.000857 | −0.000251 |
| support ≥20, `S−W2` | 107 | 51 | 56 | −0.000569 | −0.000251 |
| all scoreable, `P−S` | 127 | 73 | 54 | −0.000157 | +0.000232 |
| support ≥20, `P−S` | 107 | 64 | 43 | **+0.000601** | +0.000480 |

The high-support mean is 3e-5 — *zero* to three decimals. The modest positive
headline of `P−S` (73/127 tasks improve) is offset by a few large losses, and it
does not survive the high-support filter. The support correlation of `P−W2`
deltas is only Spearman +0.098, i.e. no meaningful "better where data is
plentiful" trend.

Concentration of the positive deltas: `P−W2` top-5 assays hold 31.1% and top-10
hold 48.2% of the total positive mass; `S−W2` 37.8% / 58.3%; `P−S` 35.9% / 49.0%.
In other words the gains are a handful of assays, not a frontier shift.

Largest movers (`P−W2`): task 28 +0.0350 (9 pos), 34 +0.0290 (13), 30 +0.0172
(210), 19 +0.0170 (79), 112 +0.0152 (23); losses: 118 −0.0826 (8), 117 −0.0486
(9), 69 −0.0368 (19), 108 −0.0299 (1). Full table:
`analysis/per_task_deltas.csv`.

## 9. Do the main-effect and pair terms learn, and do they participate in prediction?

Yes to all four, on both arms, from the shipped epoch-40 checkpoints.

**Coefficients** (`‖α‖₂` over 128 tasks, epoch 40):

| arm | `‖α_A‖` | `‖α_B‖` | `‖α_pair‖` | mean `|α_pair|` | max `|α_pair|` | assays with `|α_pair| ≥ 1e-2` |
|---|---:|---:|---:|---:|---:|---:|
| `P` | 2.1325 | 2.1789 | **7.7200** | 0.6076 | 1.6416 | **128/128 (100%)** |
| `S` | 1.8964 | 2.1935 | **6.5457** | 0.5178 | 1.1549 | 127/128 (99.2%) |

The pair coefficient is ~3× larger than either main-effect coefficient, and it is
non-negligible for effectively every assay — no task switched the term off.

**Residual magnitudes on the full official Validation split** (43,793 molecules,
5,605,504 logit entries, mean absolute value):

| arm | `|base logits|` | `|main residual|` | `|pair residual|` | `|total residual|` | max `|pair residual|` |
|---|---:|---:|---:|---:|---:|
| `P` | 8.1817 | 0.1378 | **0.0870** | 0.2189 | 1.6303 |
| `S` | 8.1849 | 0.1045 | **0.1258** | 0.2149 | 1.0551 |

So the pair term carries ~1.1% (`P`) / ~1.5% (`S`) of the logit scale and ~40% /
~58% of the total evidence residual, and is non-zero on every molecule (max 1.63).

**The two statistics are not on the same scale.** On the fixed 256-molecule
diagnostic set, the same-atom statistic is intrinsically larger than the
cross-atom one: `S` `self` 0.3073 vs `S` `cross` 0.1272; in `P`, `self` 0.4491 vs
`cross` 0.1726 (both are computed inside both arms, only one is used). That is
expected — `a_i` and `b_i` share the atom embedding `h_i` and are strongly
correlated, whereas off-diagonal products average out. The frozen protocol
matched **parameter count**, not residual magnitude, and the realised pair
residuals differ by 45% (0.087 vs 0.126). Any "cross vs self" claim is therefore
a claim about *what the term is allowed to see*, not about equal-strength
perturbations.

**Ablation (diagnostic only).** Setting `alpha_pair := 0` at inference on the
trained `P` checkpoint:

| slice | AP with pair | AP without pair | Δ | valid BCE with / without |
|---|---:|---:|---:|---:|
| full official Validation (43,793) | 0.2608586 | 0.2600700 | **+0.0007885** | 0.0464818 / 0.0465384 |
| fixed 256 molecules | 0.4895482 | 0.4859154 | +0.0036328 | — |

99.999% of the logit entries change, mean |Δlogit| 0.0870, max 1.6303 — the term
is genuinely wired into the prediction. It buys +7.9e-4 AP and −5.7e-5 BCE on
Validation, which is ~1/6 of the 5e-3 threshold. (The full-split AP here is
0.2608586 vs the run's 0.2608530 because this audit ran on CPU; the audit
asserts reproduction within 1e-4 and the residual is 5.6e-6.)

Interpretation: the pair gate is *used* and is *not harmful*, but its marginal
contribution at this scale is far below what would move macro AP.

## 10. Train vs Validation

| arm | Train AP@40 | Train BCE@40 | Val AP@40 | Val BCE@40 |
|---|---:|---:|---:|---:|
| `W2` | 0.5818746 | 0.0252154 | 0.2618669 | 0.0465379 |
| `P` | 0.5836454 | 0.0251630 | 0.2608530 | 0.0464818 |
| `S` | 0.5844854 | 0.0251529 | 0.2610103 | 0.0465149 |

The extra capacity **does** buy a little Train AP (+0.0018 / +0.0026) and a
little Train BCE, and the Validation BCE is also slightly *better* than `W2`
(−5.6e-5 / −2.3e-5) while Validation AP is slightly worse. That is not the usual
over-fitting signature (train up, valid down, BCE up); it is a near-tie with the
ranking metric drifting the wrong way and the calibration metric drifting the
right way. Since the gains are 3–4 orders of magnitude below the decision
thresholds, the honest reading is "no effect", not "mild overfitting".

Instability check: 14 gradient-clipping events in each arm (out of 109,520
steps, ≈0.013%), i.e. 99.987% of steps stayed below the 5.0 norm, **0 non-finite
events**, and the `alpha` trajectories are smooth through the round (`P`
`‖α_pair‖`: 0.135 → 2.81 → 5.80 → 7.72 at epochs 1/10/20/40). Last-10-epoch
slope:
`P` −7.5e-5/epoch, `S` −5.0e-5, `W2` −9.2e-5 — all three slightly overfit after
epoch 30, `P`/`S` no worse than `W2`. No sign of pair-gate instability.

## 11. Extra cost

| quantity | `W2` | `P` | `S` | `P` vs `W2` |
|---|---:|---:|---:|---:|
| mean epoch time | 124.86 s | 128.78 s | 127.71 s | +3.1% |
| step p50 (mean over epochs) | 42.09 ms | 43.60 ms | 43.25 ms | +3.6% |
| peak allocated | 233.8 MB | 235.7 MB | 235.7 MB | +1.9 MB (+0.8%) |
| peak reserved | 424.0 MiB | 424.0 MiB | 424.0 MiB | equal (caching-allocator bucket) |
| total GPU-hours | 1.3949 | 1.4386 | 1.4268 | +0.0437 |

`P` and `S` differ in speed by 0.84% — pure run-to-run noise on a shared box
(same code path, same tensor shapes). The whole second-order reduction is
`O(n_atoms)` with a handful of extra elementwise kernels; the measured 3% is
consistent with the two 128×128 linears over ~26 atoms per molecule dominating
that branch, not with the pair algebra, and nothing of size `n_atoms²` is ever
materialised.

**Budget.** `analysis/gpu_ledger.json`:

| job | wall clock |
|---|---:|
| `cpe-audit` (`res-gpu1`) | 0.133 h |
| `cpe-p` (`res-gpu1`) | 1.441 h |
| `cpe-s` (`res-gpu1`) | 1.430 h |
| **charged total** | **3.004 h** |

Target was ≤4 h, hard cap ≤6 h → **`WITHIN_TARGET`**, 75% of target, 50% of the
hard cap, including the audit, the smoke test and one deployment. In-process GPU
time was 2.865 h. No retries were needed; nothing ran concurrently on GPU 1.

## 12. Is the route worth continuing?

The narrow answer: **not in this form**. Three facts block the "structural
sufficiency" reading that motivated the round.

1. The cross-atom term is real, learned, used, and *inert* for macro AP
   (+7.9e-4 in ablation, −1.0e-3 vs `W2` at the endpoint).
2. The matched same-atom control is statistically indistinguishable
   (`P − S = −1.6e-4`). So the round cannot attribute anything to *cross-atom*
   structure as such — a term that only ever sees one atom explains the data
   equally well. If atom-pair co-occurrence carried usable predictive signal at
   this scale, the cross arm should have beaten the self arm; it did not.
3. The high-support stratum — where a pair statistic has the most samples to
   work with — is exactly flat (+3e-5, 52/107). The apparent `P > S` on
   64/107 high-support assays (+6.0e-4 mean) is smaller than the epoch-to-epoch
   wobble of a single run (5.9e-4) and is not accompanied by a macro-AP gain.

What would change the verdict, in increasing order of cost: (i) a second seed to
establish whether |Δ| ~ 1e-3 is noise at all — necessary before any further
architecture claim on this track; (ii) a pair statistic that is *not* a bilinear
form of the existing per-atom evidence (the current term's information is
constrained to the span of `h_i`, so it can only reweight what the trunk already
knows — a plausible explanation for why both statistics do the same thing);
(iii) learning the pair interaction at the level of bond/path structure rather
than through two global sums, which is a different (and much more expensive)
object than what PROTOCOL §6 froze.

What this round *does* establish, and which is useful going forward: the
mechanism is implementable and auditable at 3% cost, it does not disturb the
frozen `W2` stream, it trains with no instability, and its contribution is
measurably small. That is a clean negative, obtained for 3 GPU-hours.

## 13. Provenance

| item | value |
|---|---|
| `P` commit | `840dcf5288ff` (freeze of protocol + code) |
| `S` commit | `d602342db8d0` (analysis-only delta; `cpe_{train,model,common}.py` byte-identical to the freeze) |
| checkout state at launch | clean (rr `git_dirty: false`, empty diff hash, both runs) |
| host | `a100-2` (`res`), `res-gpu1`, `CUDA_VISIBLE_DEVICES=1` |
| torch / python | 2.5.1+cu124, FP32 |
| data provenance | feature spec `7270f767…`, norm `6ce50f98…`, split `c872bd63…`, label artifact `5c3b5fbe…` |
| train / valid rows | 350,343 / 43,793, disjoint (checked) |
| batch plan | `06f437299fa451e4…`, consumed prefix `1a24dfca…`, order seed 0 |
| `P` checkpoint (epoch 40) | `5f0e5dc960ddbd5d134e5ac23bdbca94fd20ab6ade5b42d1e2fab5815e57c324` |
| `S` checkpoint (epoch 40) | `e5fc2d7bd418e7108e24dddc46f0a4638d73df3d060175cee96ed86d760262a1` |
| historical `W2` checkpoint | `7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a` |
| `P` best / last checkpoint | `62efb8fc…` (e35) / `86ca7574…` |
| `S` best / last checkpoint | `03b9fc23…` (e31) / `59953b20…` |
| curves | `ba61b1d2…` (`P`), `5e2b6395…` (`S`) |
| per-task CSVs | `67c44eb7…` (`P`), `6169c50f…` (`S`) |
| evidence diagnostics | `087499ed…` (`P`), `c44146c2…` (`S`) |

Note: `run.json → git_state.dirty` is `true` in both runs, because the earlier
audit job had written untracked `analysis/*.json` into the shared remote
checkout. The launch-time record (`meta.json → git_dirty: false`,
`git_diff_hash` = SHA-256 of the empty string) and the commit id are the
authoritative statements; no tracked file differed from the commit during
training.

**Official Test split: never used.** No Test label is read, no Test row is
indexed, no Test metric exists anywhere in this round. The label artifact stores
NaN for Test rows; the audit asserts that the indexing paths refuse them
(`test_labels_all_nan: true`, `test_rows_never_indexed: true`), and both
`run.json`s record `test_evaluated: false`. All numbers above are Validation.

## 14. Artifacts

```
PROTOCOL.md                     frozen design (18 pre-train gates)
README.md                       operational guide
REPORT.md                       this file
code/cpe_common.py              constants, frozen recipe, decision rules
code/cpe_model.py               pair statistics, evidence module, init, audits
code/cpe_train.py               from-scratch 40-epoch trainer
code/cpe_audit.py               gpu-preflight / gpu-sampling / init / architecture / smoke / mechanism
code/cpe_compare.py             post-run comparison and decision
tests/                          29 CPU tests (model + training contract)
analysis/architecture_audit.json          34/34 checks
analysis/initialization_parity.json       all_passed
analysis/evidence_mechanism_audit.json    coefficients, residuals, ablation
analysis/p_epoch40.json                   epoch-40 record
analysis/s_epoch40.json                   epoch-40 record
analysis/paired_comparison.json           AP, per-task, strata, cost
analysis/per_task_deltas.csv              127 rows × P−W2, S−W2, P−S
analysis/paired_training_audit.json       14 pairing checks
analysis/gpu_ledger.json                  3.004 h, WITHIN_TARGET
analysis/decision.json                    NO_CLEAR_GAIN
analysis/smoke.json                       1-epoch plumbing test (not a result)
analysis/audit_summary.json               aggregate of the post-run audit pass
analysis/gpu_preflight.json, gpu_sampling.json, rr_jobs.json
runs/P, runs/S                            git-ignored: checkpoints, curves, run.json
```

## 15. How to reproduce

```bash
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_cpe_v1/tests
rr run res cpe-p --pool res-gpu1 -- bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_train.py --model P --seed 0 --device cuda --eval-every 1 --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/P --soft-seconds 14400 --hard-seconds 18000"
rr run res cpe-s --pool res-gpu1 -- bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_train.py --model S --seed 0 --device cuda --eval-every 1 --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/S --soft-seconds 14400 --hard-seconds 18000"
uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_compare.py
```

## 16. 直接回答（中文摘要）

1. **两组参数量是否完全相同？** 是。`P`、`S` 均为 1,027,732（`W2` 994,324 + 新增
   33,408），参数名、形状、dtype、epoch-0 数值逐 bitwise 相同；唯一差异是
   非参数属性 `pair_mode`。`initialization_parity.json`、`architecture_audit.json`
   （34/34）均已验证。
2. **`W2` init、batch order、dropout 是否配对？** 是。新模块在
   `fork_rng` 内的私有 CPU generator(seed 37) 中构造，全局 RNG 不变，`W2` 子状态与
   历史 init 指纹 `5f7101a8…`/`109756bd…` 完全一致；epoch-0 全量 Valid AP 三者差异
   ≤6.7e-8；batch plan `06f43729…`、40-epoch 前缀 `1a24dfca…`、seed 0、每 epoch
   全量评价，两 run 相同；dropout 每 forward 只调用 1 次，两臂同一随机流。
3. **是否真实实现 `i != j`？** 是。`cross = (Σ_A Σ_B − Σ_AB)/(n(n−1))`，与显式
   `O(n²)` 双层循环最大误差 ≤1.6e-7；`n=1` 严格为 0 且无除零；A/B 反号精确反号；
   原子重编号不变；3 个不同大小图并批与单独跑差值为 **0.0**（无跨分子泄漏）。
4. **是否引入迭代消息传递 / 跨原子 attention / QKV / 重新编码 / 二次 dropout /
   沿键传播？** 均没有。每 forward 调用计数：`encode 1, local 1, evidence 1,
   dropout 1, readout 1`；`i`、`j` 只通过两个标量和耦合，无 `n²` 张量。
5. **AP10/20/25/30/40**：`P` = 0.242918 / 0.257437 / 0.261167 / 0.260518 /
   **0.260853**；`S` = 0.242613 / 0.257125 / 0.262191 / 0.261457 / **0.261010**；
   `W2` = 0.243189 / 0.258635 / 0.263057 / 0.262022 / **0.261867**。
6. **`P` 是否超过 `W2` 0.2618669？** 没有。Δ = **−0.001014**，落在预注册的
   “无清晰差异”带（|Δ|<0.002）内，既非提升也非显著退化。
7. **`P` 是否优于 `S`？** 没有。Δ = **−0.000157**（高支持 107 个 assay 上
   +0.000601，64 改善 / 43 退化），远低于 +0.003 的机制阈值。
8. **≥20 正例 assay 是否广泛改善？** 没有。107 个中 52 改善 / 55 退化，均值
   **+0.0000320**，中位数 −0.0000399；5–19 档均值 −0.005677；<5 档 −0.014999。
   `P−W2` 正增益的 top5 占 31.1%、top10 占 48.2%，收益集中在少数 assay。
9. **主效应与 pair 是否学习并参与预测？** 是。`‖α_A‖/‖α_B‖/‖α_pair‖`：`P`
   2.13/2.18/**7.72**，`S` 1.90/2.19/**6.55**；`|α_pair| ≥ 1e-2` 的 assay 比例
   128/128 与 127/128。全量 Valid 上 `|主效应残差|` `P` 0.1378、`S` 0.1045，
   `|pair 残差|` `P` 0.0870、`S` 0.1258，总残差 ≈0.22（logit 尺度 8.18 的 ~2.7%）。
   固定 256 分子上 CROSS/SELF 幅度：`P` 0.1726/0.4491，`S` 0.1272/0.3073（同原子
   统计量天然更大）。`P` 把 `alpha_pair` 置零的推理消融：全量 Valid AP
   0.2608586 → 0.2600700（Δ **+0.000789**），99.999% 的 logit 条目改变，
   说明该支路确实参与预测，但贡献仅 ~1/6 阈值。
10. **Train 与 Valid 的关系**：Train AP `W2` 0.581875 → `P` 0.583645 → `S`
    0.584485（小幅上升）；Valid AP 0.261867 → 0.260853 / 0.261010（小幅下降）；
    Valid BCE 反而略好（0.046538 → 0.046482 / 0.046515）。不是典型过拟合形态，
    更接近“排名指标持平略降、校准略好”。
11. **额外计算成本**：epoch +3.1%（124.86 → 128.78 s），step p50 +3.6%
    （42.09 → 43.60 ms），峰值显存 +1.9 MB（233.8 → 235.7 MB，reserved 持平），
    单臂 +0.044 GPU-h。`P`/`S` 之间速度差 0.84%（噪声）。总 GPU 时间 3.004 h
    （审计 0.133 + `P` 1.441 + `S` 1.430），`WITHIN_TARGET`（target 4 / cap 6）。
12. **这条路线是否值得继续？** 就当前形式而言，**否**。机制真实、可审计、低成本、
    数值稳定，但对宏 AP 无效；且与同原子对照无差别，说明“跨原子”本身没有被证明
    携带可用信号。若要继续，应先补第二个 seed 以确认 ~1e-3 的端点差是噪声；其次
    需要让 pair 项看到 `h_i` 之外的结构（当前是 `h_i` 的双线性型，只能重加权主干
    已知信息）；再其次才是键/路径级的交互，那是与本次冻结设计不同的、更贵的对象。
13. **GPU 时间/提交/报告路径/checkpoint**：3.004 GPU-h；`P` 提交 `840dcf5288ff`、
    `S` 提交 `d602342db8d0`（训练代码与冻结点逐字节一致）；报告
    `tracks/structural-sufficiency/lcrc_pcba_cpe_v1/REPORT.md`；checkpoint SHA256
    `P` = `5f0e5dc960ddbd5d134e5ac23bdbca94fd20ab6ade5b42d1e2fab5815e57c324`，
    `S` = `e5fc2d7bd418e7108e24dddc46f0a4638d73df3d060175cee96ed86d760262a1`。

## 17. What this report does not claim

- It is a **single-seed** comparison. Differences of ~1e-3 between arms are within
  the endpoint wobble of this recipe and are not significance tests; the
  pre-registered thresholds are resource-allocation rules, not inference.
- `W2` is a **historical** baseline from an earlier round, not re-run here; the
  fairness argument rests on the audited initialisation/order/dropout parity, not
  on a new `W2` run.
- The `P`/`S` control is a **mechanism** control (which statistic the gate sees),
  not a capacity control against a parameter-free baseline.
- Nothing here is **independently confirmed**: one host, one seed, one split.
- The official Test split was never evaluated, so no leaderboard-comparable number
  exists in this round.
