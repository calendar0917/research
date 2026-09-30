# Analysis — `e2e_dictenv_rolecorr_v1` (RoleCorr)

Round: **E2E-DictEnv-RoleCorr-v1** (`e2e_dictenv_rolecorr_v1`), study
`zinc-context-gap`, track `ksvd`.
Object under test: a **shared sparse dictionary over the within-patch
role↔attribute correspondence** `C ∈ R^536`
(`C_g = Σ_i (role_i − mean_g role)(attr_i − mean_g attr)^T` per node shell /
edge shellpair), replacing the pure-topology dictionary coordinate.
Pre-registration: [`e2e_dictenv_rolecorr_v1_preregistration.md`](e2e_dictenv_rolecorr_v1_preregistration.md)

Verdict: **`ROLE_CORR_NO_MATERIAL_GAIN` — the frozen-dictionary screening gate
did not fire and the round stops.**
`M_A(TOPO) = 0.123253705`, `M_B(CORR) = 0.126447222`,
`relative = (M_A − M_B)/M_A = −2.591 %` against a frozen `≥ +2 %` gate.
The correspondence coordinate *is* load-bearing inside arm B (zeroing it costs
+0.0355 MAE, shuffling it costs +0.0374 MAE) but it does not pay for the
structural capacity it displaces.

CPU only, `official_test_loaded = false` everywhere, one seed-0 trajectory per
arm, no test access (`test_access: blocked`).

---

## 0. Question

Beyond coarse chemical composition, does **which fine-grained structural role a
chemical property sits on** carry task value that a *shared* dictionary can
reuse across molecules? Concretely: freeze everything of the Sem108 / CSSD-C6
route (interface, bindings, 48-D environment, static composition, reader,
optimiser, 320-epoch horizon, Top-5 soup) and change **only** the dictionary
coordinate:

| arm | coordinate (all width 33) | dictionary |
|---|---|---|
| A `TOPO` | `[c~ ; IHT10(colnorm((I−UUᵀ)D_SDB), r)]` | frozen SDB `D_SDB ∈ R^{65×32}`, `s=8`, no labels used beyond the frozen artifacts |
| B `CORR` | `[c~ ; IHT10(D̄_S, r) (16) ; α_C (16)]` | `D_S ∈ R^{65×16}` (`s=4`) fitted on the train residual; `D_C ∈ R^{536×16}` (`s=4`) fitted on the train-scaled `C` |

Both arms keep `c~` (the frozen `q=1` common subspace coordinate, rms
5.082853) and therefore the same 33-wide binding (`W_A_S [33,96]`,
`W_E_S [99,48]`), the same trainable parameter count (95,629) and a **bit-identical
readout initialisation**.

The frozen screening gate was pre-registered as a ≥ 2 % relative soup-MAE
improvement of B over A on official valid, with unconditional stop (no width /
horizon / seed / regularisation rescue) if it fails.

## 1. Object construction, correctness and audit (all pre-training)

* 536-D object: node shells 1/2 → `2×7×28 = 392`, edge shellpairs
  `(0,1),(1,1),(1,2),(2,2)` → `4×9×4 = 144`; fixed order.
* `corr_raw_train.pt` 231,664 patches (10,000 molecules), `corr_raw_valid.pt`
  23,083 patches (1,000 molecules). Construction reads topology + atom/bond
  type only (label-free AST check passes: no `y/target/label` identifier in the
  guarded builders).
* Group-size accounting (fraction of patches whose group has < 2 members, i.e.
  the block is zero by construction): shell1 17.8 %, shell2 4.4 %,
  `(0,1)` 17.8 %, `(1,1)` 100 %, `(1,2)` 4.3 %, `(2,2)` 99.7 %. Non-zero object
  fraction 56.6 % (train) / 57.0 % (valid); the `(1,1)` and `(2,2)` edge blocks
  are effectively inert on ZINC radius-2 patches.
* Train-only two-layer scaling (no mean centring, per-coordinate RMS, zero-RMS
  coordinates masked, blocks renormalised to equal mean squared energy):
  node `w = 0.077152`, **224/392 node coordinates masked**; edge `w = 0.169031`,
  **109/144 edge coordinates masked** (only ~303 of 536 coordinates are live).
* Audit (24 train molecules, all patches): hand-reference max abs error `0.0`,
  node-relabelling invariance `2.22e-16`, endpoint/insertion-order invariance
  `0.0`, within-group attribute permutation changes the object in 259 checked
  patches while leaving every group size unchanged, 1,344 singleton/empty-group
  examples all exactly zero → `audit.passed = true`.
* Dictionaries (unlabeled K-SVD, `DICT_SEED 20260924`, 10 epochs, 231,664 rows):
  `D_S` final fit MSE `0.0028173` (152 s), `D_C` final fit MSE `1.109789`
  (460 s).
* Model gates (`correctness.json`, `all_passed = true`): coordinate width 33,
  common-coordinate reconstruction error `1.19e-07`, IHT `l0 = (4, 4)`,
  zero-`α_C` purity (only columns 17–32 change), zero-`α_S` purity,
  `TOPO` arm forward **bit-identical** to `sem.build_sem108_model`
  (`max abs diff 0.0`, dropout disabled for the gate), readout init
  bit-identical over 48 shared tensors (no mismatch keys), dictionaries frozen,
  reconstruction term gradient-inert. Smoke (8 epochs, 1,024 train / 512 valid)
  passed for both arms.

## 2. Primary result (official valid, Top-5 soup over the 320-epoch run)

| arm | best valid MAE | best epoch | soup valid MAE | members |
|---|---:|---:|---:|---|
| A `TOPO` | 0.128810 | 309 | **0.123254** | 282, 290, 295, 304, 309 |
| B `CORR` | 0.132879 | 313 | 0.126447 | 280, 292, 306, 313, 318 |

`relative improvement = −2.591 %` (gate `≥ +2 %`, fired `false`).

Curve summaries (epoch 1 / last / min / last-20 mean):

| arm | first | last | min | last-20 mean |
|---|---:|---:|---:|---:|
| `TOPO` | 0.719359 | 0.135955 | 0.128810 (309) | 0.139343 |
| `CORR` | 0.632212 | 0.140212 | 0.132879 (313) | 0.139150 |

Both arms converge to the same band; `CORR` trains marginally lower on train
(0.090 vs 0.093 at epoch 260) but never below `TOPO` on valid after epoch ~50.
Trainable parameters are identical (95,629 each), so this is not a capacity
artefact of the readout.

## 3. Frozen inference probes on the soup states

| probe | value | Δ vs `M_B` |
|---|---:|---:|
| `M_A` (TOPO soup) | 0.123254 | — |
| `M_B` (CORR soup) | 0.126447 | — |
| `M_A0` (`TOPO`, α₃₂ → 0) | 0.171613 | +0.048359 |
| `M_C0` (`CORR`, α_C → 0) | 0.161967 | **+0.035520** |
| `M_S0` (`CORR`, α_S → 0) | 0.165276 | +0.038829 |
| `M_Cshuf` (`CORR`, within-group attribute permutation, 5 seeds) | 0.141649 – 0.187199 | **mean +0.037430**, range +0.015202 … +0.060752 |

The candidate coordinate is *used*: zeroing or permuting `α_C` at inference
costs ~0.036–0.037 MAE, i.e. above the pre-registered directional (0.003) and
clear (0.010) mechanism thresholds. `α_C` also does not collapse: all 16 atoms
active, effective atom count 7.89, top1 share 0.53, exact `l0` mean 2.28
(`α_S`: 16/16 active, effective 13.83, top1 0.53, `l0` 4.0).

Per-molecule paired difference on official valid (A vs B, same ordering,
`per_molecule_errors.npz`):

* fraction of molecules where B is better: **0.4820**
* mean / median `|err_B| − |err_A|` = **+0.003194 / +0.002643**
* quantiles of the difference: p10 −0.10300, p25 −0.04447, p50 +0.00264,
  p75 +0.04408, p90 +0.10306

The loss is a small, broad shift rather than a handful of catastrophic
molecules: the median molecule is slightly worse under B, and the signed mean
difference is positive.

## 4. Reading of the result

1. **The correspondence object itself is not inactive.** `G_C0` and `G_Cshuf`
   are large and stable across five permutation seeds; the shuffled-object
   refit arm and the PCA control were therefore *not* authorised (the gate did
   not fire), so this round cannot separate "sparse dictionary over `C`" from
   "dense linear code of `C`". That question stays open, not refuted.
2. **Under the frozen budget the correspondence does not pay for the
   structural capacity it displaces.** Arm B halves the structural dictionary
   (32 atoms/`s=8` → 16 atoms/`s=4`) to make room for 16 `α_C` coordinates. The
   measured cost of that trade is ~2.6 % relative MAE on official valid. This is
   the honest scope of the negative: the round tested the *frozen coordinate
   budget*, not "the role↔attribute idea" in general.
3. The live-coordinate structure of `C` is thin on ZINC radius-2 patches: 224 of
   392 node coordinates and 109 of 144 edge coordinates are masked (zero RMS),
   and the `(1,1)`/`(2,2)` edge blocks are almost always empty. A future object
   would have to earn its keep on fewer than ~300 live coordinates.
4. The `TOPO` baseline reproduces the historical Sem108 soup band
   (`0.123254` vs the historical `0.123705`) under the frozen-dictionary
   protocol, so the comparison is against a healthy, current baseline — not a
   degraded straw man.

## 5. Provenance and limitations

* Frozen run command (reproducible, see `run_chain.sh`):
  `uv run research run zinc_e2e_dictenv_rolecorr_v1 --study zinc-context-gap --mode screen --purpose "RoleCorr-v1 primary screen (stage=primary)"`
* `test_access: blocked`; only the official train/valid splits are loaded; the
  official test split is never instantiated (`official_test_loaded = false` in
  every payload).
* Promoted run: `20260930-235241-20a4104e` (complete primary pipeline:
  cache → scaler → audit → dictionaries → correctness → smoke → train →
  interventions → analysis). The two trained soups were produced by the
  immediately preceding invocation `20260930-222706-25bb8cdf`, which failed
  *after* training because the frozen shuffle probes needed the (then missing)
  permuted valid caches; the completed run re-used those saved states. Model and
  training code are byte-identical between the two invocations; only the
  probe/analysis stage functions changed.
* `dirty_at_run: true`: the round's module, runner, config, preregistration and
  test file were authored before the run but committed after it. The run record
  snapshots their SHA-256 (e.g. module
  `2ff144bd26049afa3edbfa23f0b896f950afd3dbf4ef708ea0389059e268b826`, prereg
  `a58db539499316ab44339d0c73b28c0361a58c18547324ee7f8a1348beb6cef8`), so the
  executed content is recoverable; the process deviation is recorded here
  rather than hidden.
* One seed, one trajectory per arm; single frozen-screening comparison. No
  significance claim is made and no official-test read happened or is licensed
  by this result.
* Controls C (`CORR-SHUF`, 1) and D (`CORR-PCA`) exist in the runner but are
  gated behind the 2 % screen; they were never trained, so "B vs shuffled-object
  refit" and "B vs PCA16 of the same object" remain unmeasured.

## 6. Continue / stop

**Stop this round.** The frozen screening gate did not fire
(`−2.591 % < +2 %`), and the pre-registration forbids buying a second seed, a
wider/taller coordinate, a longer horizon or a re-tuned dictionary off this
result. A next round, if pursued, needs a **new pre-registration** and should
change exactly one of the remaining open questions, e.g.:

* a *matched* capacity comparison (keep the 32 structural atoms and append the
  16 `α_C` coordinates, width 49, versus the same-width pure-topology arm) so
  the "does `C` add anything on top of the full structural budget" question is
  separable from the "does `C` pay for structural capacity" question that this
  round answered negatively;
* or the correspondence *without* the sparse-dictionary claim (PCA/affine code
  of `C`, pre-registered as its own arm, since the C/D controls were never
  authorised here).

Both options must stay frozen-coordinate, seed-0, official-valid-only screens
unless a new pre-registration says otherwise.
