# RICE-CC-v0 — atom-conditioned coupling on the RICE sentence — report

Protocol: `PROTOCOL.md` (frozen before training, commit `78f56039`).
Task: `ogbg-mollipo`, D0 only (Fit 2856 / Dev 504, scaffold-disjoint), seed 0.
Arms: `CC_NULL` (bond-only `J_base`) and `CC_REAL` (chemistry-conditioned
`J_real`), both on the unchanged RICE-v0 rooted Möbius encoder.
Formal training revision: `78f56039` (clean tracked tree; see §10 for the
remote untracked-file note). Analysis/results revision: `3b5e6d3e`.
Checks: `analysis/checks.json`, **36/36 PASS**, recorded on `78f56039` with
`git_dirty=false`. Results: `results/CC_NULL_S0/`, `results/CC_REAL_S0/`.

## 0. Answer in one paragraph

Atom conditioning **does change the real bond potential**, but by a small amount,
and it **does not improve held-out Dev**. On the 504 Dev molecules, `CC_NULL`
reaches RMSE **0.598965** (MAE 0.465330) and `CC_REAL` reaches **0.599388**
(MAE 0.465718), so

```
Delta = RMSE(CC_NULL) - RMSE(CC_REAL) = -0.000423
95 % paired molecule bootstrap [-0.001744, +0.000964], 5000 resamples
fraction of resamples with Delta > 0: 0.2808
```

`CC_REAL` is slightly better on Fit (0.430629 vs 0.431363) but slightly worse on
Dev, so the change is not a held-out improvement. It also does not narrow the gap
to R2-SENTENCE D0 s0 (0.5506): `CC_REAL` is 0.0488 behind, slightly farther than
`CC_NULL` (0.0484 behind). The coupling is not a no-op:
99.85 % of Dev bonds have `J_real != J_base`, the mean `|delta|` is 0.001251
(0.83 % of the mean `|J_base|`), its maximum is 0.031851, and 68.7 % of the
within-bond-type variance of `delta` is explained by different endpoint element
pairs. The frozen decision rule gives **`NO_GAIN`** (Delta ≤ 0). The route stops
here: no D1, no second seed, no official Valid/Test.

## 1. Contract and execution

| item | value |
|---|---|
| code / protocol / config | commit `78f56039`, unchanged RICE-v0 encoder |
| arms | `CC_NULL` = `J_base`; `CC_REAL` = `J_real` |
| parameters | **129,198** in both arms; identical ledger and init hashes |
| sentence input | `κ(S,v)`, Möbius inversion sizes 1→4, in **both** arms |
| supports | same real radius-2 induced subgraphs, sizes 1–4, every internal bond |
| channels / states | 16 / 2, exact enumeration, no sampling, no message passing |
| recipe | AdamW 3e-4 → 1e-5, wd 1e-4, batch 64, 5 warmup epochs, cosine, 240 max, patience 45, min 80, clip 5, standardized whole-molecule MSE |
| data | D0 Fit 2856 / Dev 504, split `60ac13c0…`, Fit-only scaler 2.185815826311436 / 1.2139664740467941 |
| seed | 0 only |
| host | `res`, node `a100-2`, 2×A100-SXM4-40GB, driver 550.163.01 |
| regime | torch 2.5.1+cu124, CUDA 12.4, Python 3.12.14 |
| formal GPU cost | **0.4931 h** for both arms (0.2453 + 0.2479), budget 1.2 h |
| total GPU cost incl. preflight | **≈0.4942 h**, budget 1.5 h |
| peak VRAM | 237.9 MiB (`CC_NULL`), 238.9 MiB (`CC_REAL`) |

Runs: `CC_NULL_S0-20261123-215947-57a86746` (pool `res-gpu0`) and
`CC_REAL_S0-20261123-215947-b028c5e9` (pool `res-gpu1`). Both started from the
same commit, same seed, same Fit sample order and same recipe; the only
difference is the tensor entering `compute_phi`. Preflight
(`RICE_CC_PREFLIGHT_NULL/REAL`, 2 epochs on 256 Fit / 64 Dev molecules) completed
with the same path, 0.6–1.4 s/epoch, 237–238 MB peak VRAM.

`CC_NULL` reproduces the historical `RICE_CUMULANT` result: **0.598965** vs the
historical manifest **0.599373** (difference −0.000408; rounded reference
0.5994, difference −0.000435). The historical checkpoint replayed through the new
`CC_NULL` path matches the old source exactly (`pred_std` max diff 0.0) and the
recorded Dev RMSE to 2.8e-6. This pair is therefore comparable, not an
unexplained control drift.

## 2. What was computed

The RICE-v0 encoder is unchanged. For each physical bond with endpoints
`i = bond_index[:,0]`, `j = bond_index[:,1]`:

```python
h = tanh((u[..., 1] - u[..., 0]) / 2.0)          # [n_atoms, 16]
k = (J00 + J11 - 2 * J01) / 4.0                   # [n_bonds, 16]
delta = k * h[i] * h[j]                           # [n_bonds, 16]
J_real = (J00 + delta, J01 - delta, J11 + delta)
```

`CC_NULL` passes `J_base`; `CC_REAL` passes `J_real`. Nothing is detached, no
parameter is added, and the exact `Phi` / Möbius / sentence definitions are
unchanged. The conservative bounds are 2 / 2 / 4 for `|J_base|` / `|delta|` /
`|J_real|`; all checks use the new 4 bound for `J_real`.

Pre-training checks (`analysis/checks.json`, 36/36):

| group | what it pins | key result |
|---|---|---|
| A | `Phi = Σ κ` against independent float64 brute force, both potentials | max identity error 8.9e-16; vectorised 2.9e-15 / 2.7e-15 |
| B | disconnected support ⇒ `κ≈0`; `J_base=0` ⇒ `delta=0`, `κ_{≥2}=0` | 1.8e-15 / 1.6e-15; delta exactly 0 |
| C | triangle closure, 4-clique 6 slots, real 4-cycle order-4 κ | all pass; κ₄ max 0.034 |
| D | atom renumbering, bond reversal, bond order, batch order | 0 mismatches; max prediction diff 4.7e-8 |
| E | gradients for both arms; delta path; 20-step decrease | all finite/nonzero; `delta` grad norms `u`=0.857, `J_base`=0.470 |
| F | word census / vectorisation / cost | 1,124,230 words; one MLP call per module per batch; 0.34 s CPU step |
| G | historical equivalence, identical arms, endpoint conditioning, bounds, state-swap covariance | G1 all diffs 0; G2 replay exact; G3 129,198 params and identical state; G4 `delta` gap 0.0116; G5 endpoint symmetry 1.4e-17; G6 bounds 2/2/4; G7 covariance 0; G8 formula 0 |
| H | Fit/Dev identity, hashes, label isolation, Fit-only normalisation | 2856/504, disjoint, union 3360; all pinned hashes match; Valid labels NaN |

The state-swap check is deliberately factor-level only: flipping both latent
states in a channel leaves `delta` invariant and transforms `J_real` as
`R J_real R = R J_base R + delta·[[1,-1],[-1,1]]`. The downstream readout is not
required to be invariant, and no model-output equality is claimed.

## 3. Main result

| arm | pair potential | params | best epoch | epochs run | Fit RMSE | Fit MAE | Dev RMSE | Dev MAE |
|---|---|---|---|---|---|---|---|---|
| `CC_NULL` | `J_base` | 129,198 | 114 | 159 | 0.431363 | 0.324177 | **0.598965** | **0.465330** |
| `CC_REAL` | `J_real` | 129,198 | 114 | 159 | **0.430629** | **0.323829** | 0.599388 | 0.465718 |

`Delta = -0.000423`, 95 % CI `[-0.001744, +0.000964]`, fraction positive
0.2808. `CC_REAL` improves Fit by 0.000734 but worsens Dev by 0.000423, i.e. it
adds a little Fit capacity without held-out benefit. The two curves are nearly
indistinguishable:

| epoch | 20 Fit/Dev | 30 | 40 | 50 | 60 | 80 | 100 | 114 | 159 |
|---|---|---|---|---|---|---|---|---|---|
| `CC_NULL` | 0.905/1.004 | 0.756/0.839 | 0.675/0.732 | 0.618/0.716 | 0.581/0.672 | 0.512/0.649 | 0.475/0.622 | 0.431/0.599 | 0.385/0.624 |
| `CC_REAL` | 0.905/1.004 | 0.755/0.839 | 0.675/0.732 | 0.618/0.716 | 0.581/0.674 | 0.512/0.651 | 0.475/0.623 | 0.431/0.599 | 0.384/0.625 |

Both arms early-stop at epoch 159 with best epoch 114. There is no divergence
phase and no evidence of a representation-level gain. The complete per-epoch
curves are in `analysis/curves.csv` and each run's `curves.csv`.

## 4. Did conditioning change the real bond potential?

Yes, but weakly. Diagnostics are computed on the frozen `CC_REAL` best
checkpoint (`analysis/coupling.json`):

| quantity (Dev, 504 molecules, 15,023 bonds) | value |
|---|---|
| bonds with `J_real != J_base` (max channel diff >1e-8) | **99.85 %** |
| mean `|J_base|` over all bond/channel entries | 0.150454 |
| mean `|J_real|` | 0.151213 |
| mean `|delta|` | **0.001251** |
| max `|delta|` | **0.031851** |
| std of all `delta` entries | 0.004987 |
| positive / negative fraction of all `delta` entries | 0.5576 / 0.4424 |
| mean `|delta|` / mean `|J_base|` | **0.83 %** |
| mean `|k|` / max `|k|` | 0.114958 / 0.661526 |
| mean `|h|` / max `|h|` | 0.056599 / 0.259713 |
| fraction of bonds with per-bond `|delta|` mean >1e-3 | 49.75 % |
| non-degenerate rule (`max>1e-3`, `std>1e-4`, `frac>1 %`) | **PASS** |

The correction is non-degenerate by the pre-registered rule, but its scale is
limited by the learned unary contrast: `k` is not small (`|k|` mean 0.115), while
`h` is small (`|h|` mean 0.057), and `delta = k·h_i·h_j` multiplies two small
numbers. This is a property of the trained energy, not a detached or zero term.

Per bond type (OGB `bond_type` field; 0=single, 1=double, 2=triple,
3=aromatic), Dev:

| bond type | bonds | mean per-bond `|delta|` | std of per-bond signed delta | positive fraction | fraction of bonds with per-bond mean `|delta|>1e-3` |
|---|---|---|---|---|---|
| 0 single | 6,538 | 0.000462 | 0.000397 | 0.614 | 7.6 % |
| 1 double | 792 | 0.000323 | 0.000194 | 0.705 | 0.0 % |
| 2 triple | 46 | 0.001069 | 0.000721 | 0.848 | 63.0 % |
| 3 aromatic | 7,647 | **0.002024** | **0.001082** | 0.214 | 90.8 % |

The corresponding variances of the per-bond signed delta are 1.57e-7 (single),
3.76e-8 (double), 5.19e-7 (triple) and 1.17e-6 (aromatic); the overall
all-entry delta variance is 2.49e-5. Aromatic bonds carry the largest coupling
magnitude and are predominantly negative; single/double bonds are smaller and
predominantly positive. This is a descriptive split of a learned latent
correction, not a chemical causal claim.

### Endpoint-combination variation

Within one bond type, different endpoint atom combinations produce a large share
of the coupling variation. Nested variance decomposition of the per-bond signed
delta (mean over the 16 channels), Dev:

| level | fraction of total variance |
|---|---|
| between bond types | 25.9 % |
| between endpoint element pairs **within** a bond type | **68.7 %** |
| residual within an endpoint pair | 5.4 % |

Using the full 9-field atom category pair instead of the element index, the
within-bond-type endpoint share is **74.0 %**. Per bond type, the between-pair
fraction (element pair, pairs with ≥5 bonds) is 0.363 (single), 0.872 (double),
0.715 (triple), 0.993 (aromatic); the ranges of pair means are 7.2e-4, 6.9e-4,
1.7e-3 and 2.7e-3 respectively. So the endpoint conditioning is doing what the
formula specifies: the same bond type with different endpoint atoms gets a
different `J_real`. It simply does not translate into a Dev gain.

## 5. Frozen-checkpoint intervention (off-manifold)

Using the `CC_REAL` best weights and switching the pair formula to `J_base` at
evaluation time (`analysis/frozen_intervention.json`):

| quantity | value |
|---|---|
| Dev RMSE with `J_real` | 0.599382 (CPU replay; manifest 0.599388) |
| Dev RMSE switched to `J_base` | 0.602853 |
| RMSE change | **+0.003470** |
| mean / max absolute prediction change | 0.02089 / 0.13393 |

This is an **off-manifold frozen intervention**, not a retrained control. It
shows that the trained `CC_REAL` network's predictions do depend on `J_real`;
it does **not** show that `J_real` would help a retrained model, because the
retrained `CC_NULL` control is slightly better.

## 6. Historical coordinates

| model | Dev RMSE | relation to `CC_REAL` |
|---|---|---|
| GINE | 0.5266 | `CC_REAL` is +0.072788 behind |
| R2-SENTENCE D0 s1 | 0.5464 | +0.052988 behind |
| R2-SENTENCE D0 s0 | 0.5506 | +0.048788 behind |
| historical `RICE_CUMULANT` s0 | 0.599373 | `CC_REAL` is +0.000015 (essentially identical) |
| **`CC_NULL` s0** | **0.598965** | −0.000408 vs historical |
| **`CC_REAL` s0** | **0.599388** | +0.000015 vs historical |

The chemistry-conditioned arm does not narrow the gap to R2: compared with
`CC_NULL`, it is 0.000423 worse, so the descriptive R2 gap moves from 0.048365
to 0.048788. Against rounded historical 0.5994 the pair is
`CC_NULL` −0.000435 and `CC_REAL` −0.000012.

## 7. Frozen verdict

Rule from `PROTOCOL.md` §7, applied mechanically (`analysis/verdict.json`):

| criterion | threshold | value | pass |
|---|---|---|---|
| `Delta` | ≥ 0.025 | −0.000423 | no |
| bootstrap 95 % CI lower bound | > 0 | −0.001744 | no |
| `CC_REAL` Dev RMSE | ≤ 0.575 | 0.599388 | no |
| coupling non-degenerate | max/std/fraction rules | 0.0319 / 0.00499 / 0.140 | yes |
| all math/gradient/equivalence checks | all pass | 36/36 | yes |

Because `Delta ≤ 0`, the label is **`NO_GAIN`**, not
`WEAK_OR_INSUFFICIENT_SIGNAL`. The bootstrap CI includes zero, and the point
estimate favors the null arm; there is no positive signal to confirm on D1.

## 8. Answers to the required questions

**1. Did atom-conditioned coupling really change the physical bond potential?**
Yes. 99.85 % of Dev bonds have `J_real != J_base`, the correction is non-zero in
every bond-type group, and 68.7 % of the within-bond-type coupling variance is
attributable to different endpoint element pairs (74.0 % using full atom
categories). But the correction is small: mean `|delta|` is 0.00125, 0.83 % of
mean `|J_base|`, because the learned unary contrast `h` is small.

**2. Did it improve held-out Dev rather than only Fit?**
No. Fit RMSE improves by 0.000734, but Dev RMSE worsens by 0.000423; Delta is
−0.000423 with CI `[-0.001744, +0.000964]`. It is a small Fit-only perturbation.

**3. Did it narrow the gap to the strong R2 reference?**
No. `CC_REAL` is 0.048788 behind R2-SENTENCE D0 s0 (0.5506) and 0.072788 behind
GINE (0.5266). `CC_NULL` is 0.048365 behind R2, so the conditioned arm is
0.000423 farther away, not closer.

**4. Continue or stop?**
Stop this specific mechanism. The pre-registered verdict is `NO_GAIN`; the
no-parameter atom-conditioned multiplicative coupling does not improve D0 Dev.
Do not run D1, a second seed, official Valid/Test, RICE-v2 or any wider energy
on this result. A different coupling parameterization would be a new,
separately authorized experiment, not a continuation of this one.

## 9. Limitations

* **Single seed, single D0 split.** The bootstrap reflects 504-molecule Dev
  sampling only, not training-seed variance. The contrast is controlled across
  arms but neither absolute Dev number is an unbiased test estimate.
* **Best-Dev checkpoint selection** makes every Dev number slightly optimistic;
  the same rule is applied to both arms.
* **The coupling is small in this trained solution.** The negative result is
  about this exact multiplicative, parameter-free coupling and the unary
  contrast the model learns. It does not prove that every atom-conditioned
  pair potential must fail.
* **Frozen intervention is off-manifold.** The +0.00347 RMSE switch is not a
  retrained effect and does not replace the `CC_NULL` vs `CC_REAL` contrast.
* **No chemical causality.** The 16 channels and 2 latent states are
  permutation-non-identifiable; bond-type and endpoint-pair summaries describe
  the learned latent correction, not a chemical mechanism.
* **Historical replay device difference.** The historical `.npz` predictions
  differ from a CPU replay by at most 2.78e-4 standardized units (GPU vs CPU
  arithmetic); the old source replayed through the new `CC_NULL` path matches
  exactly, and the Dev RMSE differs by 2.8e-6.

## 10. Provenance and reproduction

* Frozen protocol/code/config: commit `78f56039`. Pre-training checks were run
  on that clean commit and record `git_commit=78f56039`, `git_dirty=false`.
* Results/analysis: commit `3b5e6d3e` (plus this report).
* Formal runs:
  * `CC_NULL_S0-20261123-215947-57a86746`, pool `res-gpu0`, `a100-2`,
    A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4.
  * `CC_REAL_S0-20261123-215947-b028c5e9`, pool `res-gpu1`, same regime.
* The manifests record `git_dirty=true` because the remote checkout contains
  untracked `.rr-untracked-backup-*` directories and pulled result directories.
  The `rr` job metadata records `git_dirty=false` and an empty tracked-diff hash
  for both runs; the historical RICE-v0 manifests show the same pattern.
  No tracked file was edited on the remote.
* The remote checkout had untracked, byte-identical copies of already-committed
  `word_owned_bilinear_response_pilot_v0` artifacts that blocked the Git
  deployment. They were moved to
  `.rr-untracked-backup-ricecc-20261123-215522/` on the remote after verifying
  their sha256 values matched the local tracked files; no content was changed.
* Preflight: `RICE_CC_PREFLIGHT_NULL` / `RICE_CC_PREFLIGHT_REAL`, 2 epochs,
  256 Fit / 64 Dev molecules, 0.6–1.4 s/epoch, peak VRAM 237–238 MB.
* Data: raw cache `fe002287…23c7c`, split `60ac13c0…559e`, labels
  `a44d29de…114c2`, Fit-only scaler 2.185815826311436 / 1.2139664740467941.
* Reproduce:

```bash
R=tracks/structural-sufficiency/rice_chemistry_conditioned_coupling_pilot_v0
uv run pytest -q $R/tests
uv run python $R/code/rice_checks.py
rr deploy res
rr run res CC_NULL_S0 --pool res-gpu0 --result $R/results/CC_NULL_S0 -- \
  uv run python $R/code/rice_train.py --arm CC_NULL --seed 0 --device cuda \
     --run-id CC_NULL_S0 --out $R/results/CC_NULL_S0
rr run res CC_REAL_S0 --pool res-gpu1 --result $R/results/CC_REAL_S0 -- \
  uv run python $R/code/rice_train.py --arm CC_REAL --seed 0 --device cuda \
     --run-id CC_REAL_S0 --out $R/results/CC_REAL_S0
rr pull CC_NULL_S0 && rr pull CC_REAL_S0
uv run python $R/code/rice_analyze.py --device cpu
```

## 11. Deliverables and stop condition

Delivered: `PROTOCOL.md`, `configs/rice.json`, `code/` (including
`old_rice_reference.py`), `tests/`, `analysis/checks.json`, `analysis/arms.csv`,
`analysis/curves.csv`, `analysis/summary.json`, `analysis/coupling.json`,
`analysis/frozen_intervention.json`, `analysis/verdict.json`, both run
manifests, complete curves, best/final checkpoints, Fit/Dev prediction CSV and
NPZ files.

The round stops here: two arms, one seed, one D0 split. No D1, no second seed,
no official Valid/Test, no hyper-parameter search, no wider energy, no RICE-v2.