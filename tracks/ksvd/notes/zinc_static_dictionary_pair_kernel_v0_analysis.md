# Analysis — ZINC static dictionary-conditioned pair kernel (SDPK-v0)

Round: **ZINC-static-dictionary-pair-kernel-v0** (SDPK-v0).
Preregistration: `tracks/ksvd/notes/zinc_static_dictionary_pair_kernel_v0_preregistration.md`
(frozen at commit `f0a778d`).
Implementation commit: `f0a778d`.
Device: remote `A100-SXM4-40GB` GPU 0.
Status: one seed-0 full run; seed-0 GO gate FAILS; round closed with **one**
full-training run and **no** seed 1.

Official ZINC **test was never loaded** in any stage
(`official_test_loaded=false` on every artifact).

## 0. What the round asked

> Promoting the learnable dictionary from a bypassable **local residual
> adapter** (`h_i = h0_i + gamma * delta_dict`) to the **core coordinate
> system of the occurrence-level static pair / relation kernel** — does that
> open a new absolute strict-static ZINC performance band?

No ablation, no sweep, no rescue.  The residual-dictionary / `gamma` concept is
deleted entirely.

## 1. Architecture actually implemented

```text
h_i     = LocalEncoder(x_i)                     # 48D, h_i = h0_i, no residual
s_i     = l2_normalize(Wq(h_i))                 # 48 -> 32
alpha_i = softmax((s_i @ D_norm^T) / tau)       # K = 64
c_i     = U(alpha_i)                            # 64 -> 16
```

Occurrence-level pair kernel (one evaluation per pair):

```text
u_i = pair_projection(h_i)                      # 48 -> 16
raw_sum, raw_diff, raw_prod
gate_ij = 1 + tanh(distance_gate(bucket))
m_ij = 1 + tanh(Wm([c_i+c_j, |c_i-c_j|, c_i*c_j, rel_ij]))   # 64 -> 16
conditioned_prod = raw_prod * gate_ij * m_ij
pair_input = [raw_sum|raw_diff|conditioned_prod|rel_ij|
              dict_sum|dict_diff|dict_prod]     # 112D
q_ij = pair_encoder(pair_input)                 # 112 -> 64 -> 16, ONCE
```

`q_ij` never updates `h_i`/`h_j`; the dictionary is a pure function of the
local state.  Selection/readout, tokenizer, topology, head, loss and split are
inherited from the strict-static S0 baseline.

### 1.1 Parameter audit (runtime)

| module | params |
|---|---|
| S0 (strict-static reference) | 66,228 |
| **SDPK-v0 total** | **74,996** |
| dictionary (`Wq` + `D` + `U` + `tau`) | 4,657 |
| dictionary gate `Wm` | 1,040 |
| pair encoder `112 -> 64 -> 16` | 8,400 |
| inherited small raw head | 4,135 |

74,996 `<= 82,000` (preferred) and `< 90,000` (hard).  No module was shrunk to
hit a round number.

### 1.2 Shared-tensor / baseline guard

* Every shared tensor (patch encoder except the widened pair input, global
  encoder, relation encoder, distance gate, pair projection, embeddings,
  topology encoder, head) is **bit-identical** to the S0 reference
  (`max_abs_diff = 0.0`); only the pair encoder input projection is genuinely
  widened (`64 -> 112`).
* The inherited S0 source is untouched by this round
  (`zinc_patch_path_pooling.py` sha256 `ee67a5f1…`, `zinc_static_dictionary_pair.py`
  sha256 `9bfcbbe5…`).
* **Cheap baseline replay**: the existing S0 seed-0 best checkpoint was loaded
  and re-evaluated on the full official valid split — valid MAE
  `0.14567433` versus the recorded `0.145674` (`abs diff 3.3e-7`), i.e. the
  reference baseline reproduces exactly.  No S0 retraining was spent.

## 2. Strict-static hard tests

All gates pass on real mini-batches (both locally and on the remote A100), and
in the targeted CPU test suite
(`tracks/ksvd/tests/test_zinc_static_dictionary_pair_kernel.py`, 15 passed).

| gate | result |
|---|---|
| A `center_context == False`, `center_update is None` | true |
| A forward with `_pool_pairs_to_centres` raising | succeeds |
| B `max abs h diff` under pair-relation mutation | `0.0` |
| B `max abs alpha diff` under mutation | `0.0` |
| B `max abs coord diff` under mutation | `0.0` |
| B prediction changes under mutation (non-vacuous) | true |
| C `pair_encoder` / `relation_encoder` / `dictionary` calls per forward | `1 / 1 / 1` |
| D pair-order permutation invariance | `1.01e-6` |
| E atoms / `Wq` / `U` / `Wm` / pair-encoder step-0 gradients | all `> 0`, finite |

Step-0 gradient norms on a real mini-batch:
atoms 0.093, `Wq` 0.184, `U` 2.199, `tau` 0.0059, `Wm` 0.095, pair encoder
13.18, relation encoder 3.75, head 26.93.

## 3. Seed-0 full run (240 epochs, no early termination)

```text
device              A100-SXM4-40GB GPU0
commit              f0a778dd07f5eeee4b2c8473115cdf980b45ac22
epochs run          240 / 240   (no early stop)
best valid MAE      0.14219318306347123 @ epoch 209
Top-5 soup          0.13973486851429334
soup members        [165, 180, 186, 196, 209]
member valid MAE    [0.146035, 0.145715, 0.143112, 0.144913, 0.142193]
valid MAE @ epoch 240  0.165984
parameters          74,996
GPU wall clock      888.1 s
elapsed wall clock  902 s (21:05:46 -> 21:20:48 +08:00)
peak GPU memory     174.3 MB
official test       never loaded
```

Note the run is clearly *not* still improving at the horizon: the best
checkpoint is epoch 209 and the final epoch is `0.1660`, i.e. the fixed 240-epoch
budget is sufficient and the best-checkpoint number is not truncated.

## 4. Seed-0 gate — FAIL

Reference (historical A100 strict-static S0 seed 0, **not** retrained):

```text
S0 best 0.145674   S0 Top-5 soup 0.140794
```

Pre-registered gate:

```text
soup <= 0.1328  AND  S0 soup - SDPK soup >= 0.008  AND  best improvement >= 0.006
```

Measured:

| quantity | value | requirement | pass |
|---|---|---|---|
| SDPK seed0 Top-5 soup | 0.139735 | `<= 0.1328` | **no** |
| soup improvement over S0 | **+0.001059** | `>= 0.008` | **no** |
| best-checkpoint improvement over S0 | **+0.003481** | `>= 0.006` | **no** |

Verdict: **`SDPK_V0_NO_STRONG_PERFORMANCE_SIGNAL`**.  Per the preregistration the
round stops here: no `K`/`rank`/`coord_dim`/`tau`/width change, no sparsity or
entropy term, no second dictionary layer, no residual, no dense control, no
rescue.  Seed 1 is **not** purchased.

### 4.1 Honest cross-round context

The improvement is also smaller than the previous round's *seed-0* residual
dictionary, which reached soup `0.133829` (S-Dict seed 0).  That residual
signal did not replicate in seed 1 (mechanism inert), so this is not a valid
like-for-like ranking; it is recorded only so the pair kernel is not
over-credited.  The SDPK-v0 soup `0.139735` sits essentially on the S0 seed-0
soup `0.140794` and above the S0 **seed-1** soup `0.136423` — i.e. this
architecture did **not** open a new band; it remains in the `0.13x–0.14x`
region.

## 5. Dictionary diagnostics (report only; never a GO gate)

Cheap inference diagnostics on the full official-valid split:

| diagnostic | SDPK-v0 seed0 best |
|---|---|
| active atoms (`mean mass > 1e-3`) | 58 / 64 |
| argmax-used atoms | 43 |
| effective atom count (`exp H`) | 9.49 |
| mean assignment entropy (max `ln 64 = 4.159`) | 2.251 |
| top-8 assignment mass | 0.824 |
| max average assignment mass | 0.0859 |
| dictionary coherence mean / max abs | 0.229 / 0.998 |
| `tau` init -> final | 0.200 -> 0.1498 |

Cheap intervention (no retraining):

| intervention | mean `|Δpred|` | max `|Δpred|` |
|---|---|---|
| replace `c_i` by graph-mean coordinate | 0.0517 | 1.278 |
| neutral dictionary (`dict_* = 0`, `m_ij = 1`) | 0.0591 | 1.779 |

**The dictionary is genuinely load-bearing in SDPK-v0.**  Unlike the previous
round's residual dictionary (which collapsed to `gamma ~ 4e-4`, ablation shift
`1e-5`), here the assignment sharpens (`tau` 0.20 -> 0.15), 43 atoms win an
argmax, and neutralising the dictionary moves predictions by ~0.06 MAE on
average (up to 1.78).  So the failure is **not** an inert-dictionary failure:
the dictionary is central and used, and yet absolute MAE does not improve.
This is the round's main mechanism finding.

## 6. Direct answers to the round's questions

1. **Strict no-message-passing contract?** Yes. `center_context=False`,
   `center_update=None`, forward succeeds with `_pool_pairs_to_centres` raising,
   `h_i`/`alpha_i`/`c_i` bit-identical under pair mutation.
2. **Does the dictionary enter the occurrence-level pair relation kernel?**
   Yes. `c_i` forms `dict_sum/diff/prod`, feeds the multiplicative gate `m_ij`
   that modulates `raw_prod * distance_gate`, and is concatenated into the
   112D pair input of the single pair encoder.
3. **Local residual / gamma removed?** Yes — no residual adapter, no `gamma`
   parameter anywhere (`gamma` name search returns empty).  `h_i = h0_i`.
4. **SDPK total parameters?** 74,996 (`<= 82,000` preferred).
5. **Seed-0 best / Top-5 soup?** best `0.142193 @209`; soup `0.139735`.
6. **Improvement over S0 seed 0?** best `+0.003481`; soup `+0.001059`.
7. **Seed-0 GO gate (`soup <= 0.1328`)?** No — and the delta gates also fail.
8. **Seed 1 purchased?** No.
9. **Seed-1 numbers?** Not run (not authorized).
10. **Seed-1 improvement?** Not applicable.
11. **Dictionary actually used?** Yes — `tau` 0.20 -> 0.150, 43 argmax-used
    atoms, neutral-dictionary intervention mean `|Δpred|` 0.059 (max 1.78).
12. **Hard tests all pass?** Yes (A–E, plus 15 targeted CPU tests).
13. **Full-training runs used?** **1** of the allowed 2.
14. **GPU / elapsed wall clock?** GPU `888.1 s`; elapsed `902 s`.
15. **Official test ever loaded?** No.
16. **New band or still `0.13x–0.14x`?** Still `0.13x–0.14x`; no breakthrough.

## 7. Interpretation

* Promoting the dictionary into the pair kernel does **not**, by itself, buy
  absolute strict-static MAE.  The seed-0 soup gain is `+0.0011`, an order of
  magnitude below the preregistered `0.008` threshold and inside the known
  seed-to-seed noise of the baseline (`S0` itself moved `0.0063` between seeds).
* The mechanism is nonetheless **alive and central** this time (unlike the
  previous residual dictionary), which decouples the two failure modes: the
  earlier round failed because the dictionary was bypassed; this round fails
  because a *used* dictionary-conditioned pair kernel does not add enough
  predictive information over the raw pair interaction.
* No rescue is attempted.  A matched dense pair-kernel control (to test whether
  the dictionary adds anything over generic capacity in the pair kernel) was
  deliberately out of scope for this performance-first round; it would be the
  natural *next* question only if a future round first observes a materially
  positive absolute effect.

## 8. Artifacts

* `results/zinc_static_dictionary_pair_kernel_v0/parameter_audit.json`
* `results/zinc_static_dictionary_pair_kernel_v0/initialization_match.json`
* `results/zinc_static_dictionary_pair_kernel_v0/integrity_gates.json`
* `results/zinc_static_dictionary_pair_kernel_v0/baseline_guard.json`
* `results/zinc_static_dictionary_pair_kernel_v0/runs/sdpl_seed0.json`
* `results/zinc_static_dictionary_pair_kernel_v0/curves/sdpl_seed0_curve.csv`
* `results/zinc_static_dictionary_pair_kernel_v0/states/sdpl_seed0_selection_state.pt`
* `results/zinc_static_dictionary_pair_kernel_v0/analysis.json`
* `results/zinc_static_dictionary_pair_kernel_v0/RESULTS_SUMMARY.md`
