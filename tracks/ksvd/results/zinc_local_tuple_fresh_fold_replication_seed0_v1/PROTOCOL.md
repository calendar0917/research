# PROTOCOL — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

Frozen **before** either formal trajectory started and before any new-dev score was read.
One question, two matched arms, one seed, one pre-fixed new fold, no search.

## 1. Question

Does the already-observed M_J-vs-B chemical-component (`g = y − c`) improvement survive on a
pre-fixed new 8000/2000 fold whose dev rows do not overlap the old dev and whose fit rows are not
the old fit?

Historically (old fold, official-train internal split, seed 0):

| model | fit overall cal | dev G0 cal | dev overall cal |
|---|---:|---:|---:|
| B | 0.029280 | 0.101875 | 0.103717 |
| D_J | 0.028155 | 0.100416 | 0.101907 |
| M_J | 0.030235 | 0.098216 | 0.099381 |

Old M_J−B G0 cal gain `+0.003659` CI `[−0.000438,+0.008602]`, overall cal `+0.004336` CI with
positive lower bound; raw point values same direction. The old M_J−D_J CIs all cross zero.

This round is a **replication of the B→M_J direction on a new fold**, not a new architecture
search and not an independent test.

## 2. New fold (fixed once, not selected)

```python
old_perm = np.random.default_rng(20261004).permutation(10000)
old_fit  = np.sort(old_perm[:8000]).astype(np.int64)
old_dev  = np.sort(old_perm[8000:]).astype(np.int64)
new_dev  = np.sort(np.random.default_rng(20261005).permutation(old_fit)[:2000]).astype(np.int64)
new_fit  = np.setdiff1d(np.arange(10000, dtype=np.int64), new_dev)
```

SHA256 of the int64 `tobytes()`:

| array | sha256 |
|---|---|
| new_fit | `2a21cb8771f6e24cfb4a5cc50602cf0b4390db9c4367f8bb910052f9f0aebbcb` |
| new_dev | `270ab4126b0f0413f1f6ff7ada2e9914bbaca91ae50cc65bbd8ca2673d8f9357` |
| old_fit | `7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9` |
| old_dev | `a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff` |

Frozen checks (all must hold, verified in `fresh_manifest.json`): sizes 8000/2000, disjoint,
cover 10000, `new_dev ⊂ old_fit`, `new_dev ∩ old_dev = ∅`, `|new_fit ∩ old_fit| = 6000`.
The new dev rows were trained on by the old models and the two fits overlap; the whole repository
may have seen these rows. **This is a partition-stability replication, not an unseen independent
test.** Canonical-group cross count (`canonical_group_id`, train-only handoff): 1 shared group
with 1 fit row and 1 dev row.

## 3. Fixed inputs and lineage

* Official-train only: `zftd.load_train_only()` (encoded train + env train). Official valid/test
  are never loaded, instantiated, predicted or scored (`official_valid_loaded=false`,
  `official_test_loaded=false` in every artifact).
* Raw train-only label fields `logP, SA, y_stored, cycle_score_gvae` are reconstructed from
  `train_cycle_audit_label.csv` (selected GVAE smiles line) × `gvae_full_properties.npz` × the
  train-only `train.npz` handoff. The mixed `formula_verification_per_molecule.csv` (12k rows
  incl. valid/test) is never opened.
* The reconstructed raw fields reproduce the historical all-train constants to ≤ 6.7e-16 and the
  old `k`/`c` exactly when fitted on the old fit mask (lineage anchor only; old `g`/`c` is never
  used as supervision).
* Body input standardizers: the encoded-cache all-train constants are used only to invert the
  cache, then every standardizer is refit on new fit rows only, verified equal to a direct
  `Std.fit` recomputation on the new fit rows (`prep_check.direct_new_fit_recompute`, all 0.0).
* Tuple structure (`root_base`, `node_sizes`, `pair_ptr`, `pair_t`, `pair_a`, `pair_wJ`,
  `pair_wI`, `root_atom`) is reused byte-for-byte from the committed `local_tuple_index.npz`
  (sha256 `4facb6ec77ff10130d49857bad461d16da0ee097495616ca4d4711de962b2c40`); it is all-10000-train
  raw structure. The old **fit-only scalers are never reused**.
* New phi scaler: fit-realised-tuple statistics on new-fit roots only (std floor 1e-3, L2-RMS
  scale), exactly the executable builder rule.
* New kappa sample: independent seed `20261004`, up to 8192 uniformly sampled new-fit roots
  (8192 taken), sorted; `kappa_D = 1/RMS(e_J, e_I)` on the new payload.
* `kappa_M = RMS(kappa_D · e_D_init) / RMS(e_M_init)` on the same fresh sample, with the
  untrained `D_loc_init` frame; no label, no dev, no multiplier search. Value:
  `kappa_D = 2.0631041526794434`, `kappa_M = 2.00007850651312`.

Frozen artifact hashes (written to `fresh_manifest.json`):

| artifact | sha256 |
|---|---|
| `fresh_fold.npz` | `941125505af5c769285adecddacd1882950b4010dc358b658df0b949fe014c75` |
| `fresh_targets.npz` | `2660ecbe16bdec0afb00929b3cd88a97582016e81ea251187e6995e2782d9037` |
| `fresh_tuple_payload.npz` | `9acf100136ec094a0498b83df08d4f8a4c7ca2558ec7d133cbb872e9fc09b5d2` |
| `fresh_prep.npz` | `5804dadb54e22b3ac3a514a86c02c6a68f227feaaba357cf650f7c387ec46312` |

New fit-only constants (OLS cycle-free bulk + Nelder–Mead snap, `mu_logP` fixed at
`2.4570953396190123`): `sigma_logP=1.434428173759835`, `sigma_SA=0.8327498022638992`,
`mu_SA=-3.1924626828735625`, `sigma_cycle=0.2885551506645267`, `mu_cycle=-1.3447189184664423e-05`.
New k groups: fit `k=0 7693, k=-1 269, k=-2 33, k≤-3 5`; dev `1935 / 56 / 7 / 2`.
`k = clip(round(eff·sigma_cycle+mu_cycle), −20, 0)`, `c=(k−mu_cycle)/sigma_cycle`, `g=y−c`.
A dev-only raw-field shuffle leaves the fit constants bit-identical
(`dev_label_shuffle_check.identical=true`).

## 4. Design (two arms, matched)

| arm | definition | params |
|---|---|---:|
| `B` | original compressed M_g skeleton; Sem110, posterior MLP bridge, static relations, topology/reader; **no** local-tuple path | 267,611 |
| `M` (M_J) | same B skeleton + the original M_J local tuple encoder and fusion first-layer injection | 297,499 |

M_J definition is unchanged: `x=[phi65; root_atom28; neighbour_atom28; bond4]`,
`A_bar = row_L2_normalize(A_raw[64,125], eps=1e−12)`, `f_M(x)=SiLU(x @ A_bar.T)`,
`e_M(v)=Σ_{t,a} pair_wJ(v,t,a)·f_M(x(v,a,t))`,
`preact = fusion0(Sem110) + W_loc[342,64]·(kappa_M·e_M)`. Per-tuple nonlinearity first, then real
`pair_wJ` aggregation; `d=0` roots are zero; no bias/new norm/second layer/gain/reconstruction
loss/extra dropout. `A_raw_init = D_loc_init.T`; `W_loc` exactly zero. No trained state (no old
soup, no D_J/I weights) enters either arm. Both arms are constructed from the same canonical
untrained seed-0 skeleton; all shared body/bridge tensors are byte-identical and the
post-construction RNG state is equal. Training starts from the same fresh seed-0 RNG state in both
arms (`seed_everything(0)` after construction).

What this round confirms is the **whole added local-interface scheme** (including added capacity
and optimisation effect), not an isolated attribution to correspondence, structure or a
dictionary.

## 5. Training recipe (both arms identical)

seed 0; Adam lr `1e-3`, coupled weight decay `1e-5`, global grad clip `5.0`; batch 128; 240
epochs; 15,120 steps; L1 on `g` at fit rows; fixed last-window (236–240) raw parameter soup;
FP32, no AMP/DDP, no early stopping, no reweighting, no auxiliary loss; C6 mask. Position schedule
`build_schedule(8000, 240, 0+101)`, frozen sha256
`7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`. The actual global graph-ID
stream is hashed separately and must be identical across arms (it need not match the old fold).

## 6. Evaluation and statistics (frozen)

* Raw soup only (last state described only). One fit-median bias per arm:
  `b = median(g_fit − p_raw_fit)`, `p_cal = p_raw + b`. No dev calibration, no double bias.
* Primary endpoint: new-dev G0 (k=0) calibrated g-MAE. Secondary: overall calibrated g-MAE.
  Raw, fit/dev gap, bias, the four mutually exclusive k groups and their contributions
  (`Σ|err|/N_dev`) are reported; group-contribution sum and group-gain sum identities are checked.
* `gain = MAE(B) − MAE(M)`; positive = M better. 1000 paired per-row bootstrap draws, fixed seed
  `20261005`; one shared index set per endpoint per draw for both arms and raw/cal. G0 resampled
  inside the G0 rows; overall inside all 2000 dev rows.
* Bootstrap witnesses: identical predictions → gain 0 and CI `[0,0]`; swapped arms mirror the CI;
  constant shift moves the point by exactly the shift within the absolute bound.
* One pre-fixed sensitivity: drop the unique dev row with maximal mean cal absolute error across
  the two arms (not a replacement for the main gate).
* Mechanism marked separately: HEALTHY / CHANNEL_COLLAPSED / IMPLEMENTATION_FAILED, from the
  training probes (A/W drift, task grads, root-code RMS/dead dims, injection RMS) plus a dev
  zero-ablation of the local path at the soup.

## 7. Frozen classification

Ordered; first match wins:

| category | frozen condition |
|---|---|
| INVALID / INCOMPLETE | input, label/prep, init/stream/replay invalid, or the two fixed trajectories not complete |
| PERFORMANCE_REPLICATED | new G0 cal gain ≥ 0.003 **and** new overall cal gain ≥ 0.003 **and** G0 cal paired 95% CI lower > 0 **and** G0 raw gain > 0 **and** overall raw gain > 0 (all five) |
| DIRECTIONAL_NOT_CONFIRMED | performance gate not fully passed, but new G0 cal and overall cal gains both > 0 |
| NOT_REPLICATED | any other complete valid result |

Mechanism class is reported alongside and does not replace the performance class. The new fold is
the only decision endpoint; old and new folds are shown side by side and are never pooled into a
"4000 independent samples" confirmation.

## 8. Stop rule

After the two fixed trajectories and the frozen analysis above, stop regardless of sign. No extra
seed, fold, encoder, capacity, WD/loss/optimiser/epoch/soup search, no incidence-permuted control,
no D_J/I re-open, no ring/tail rescue, no third trajectory. Only an exact restart of the same
trajectory from saved state/RNG within budget on failure, otherwise INCOMPLETE. A `g-MAE < 0.09`
(if reached) is only an internal chemical-component diagnostic, not an official-valid y-MAE; no
official-valid/test read is bought.
