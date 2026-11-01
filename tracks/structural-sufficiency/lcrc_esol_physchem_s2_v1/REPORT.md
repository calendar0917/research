# LCRC-A on ESOL — physicochemical descriptors on `rho` and a capacity-matched typed-S2-blind control

Round id `lcrc-esol-physchem-s2-v1`. Everything below is produced by
`code/run_esol2.py` and the artifacts it wrote into `analysis/`. No hand-edited
numbers. **No Test evaluation was run in this round.**

Two independent experiments on the *same* frozen ESOL partition:

* **H1** `ESOL-BASE` vs `ESOL-PHYSCHEM6` — does giving the global `rho` branch
  six explicitly recomputed physicochemical descriptors improve prediction?
* **H2** `ESOL-BASE` vs `ESOL-S2BLIND-E` — does the *typed* two-hop S2 path
  content carry predictive value that an S1-context network of matched
  effective capacity cannot replace?

---

## 1. One-paragraph verdict

**H1 — no measurable gain.** After repairing three real defects in the
`ESOL-PHYSCHEM6` data/init path (descriptor row alignment, double
standardisation, `rho` first-layer initialisation), the arm is provably
BASE-identical at step 0 (prediction difference `0.0`), reads the six
descriptors once into `rho`, and demonstrably uses them (setting the Validation
descriptors to the Train mean moves predictions by `0.50` log units on average
and worsens Validation RMSE by `0.10–0.24`). Yet the Validation result is
**UNRESOLVED**: ΔRMSE `+0.0108 / −0.0193 / −0.0067` (mean **−0.0051 ± 0.0151**,
2/3 seeds better) while ΔMAE is `−0.0102 / −0.0101 / −0.0054` (mean
**−0.0086 ± 0.0027**, 3/3 better) and Train RMSE is lower in 3/3 by
`0.031–0.062`. So: the descriptors carry information the model uses, but on this
113-molecule Validation split that information does not translate into a
Validation-RMSE improvement larger than seed noise.

**H2 — typed S2 content evidently matters here.** The typed-S2-blind,
capacity-matched control (effective parameters `247,049` vs BASE `247,106`, a
`−57` / `−0.023%` difference; typed-S2 tensors unreadable — proven by a
restricted batch, `0.0` bitwise invariance under three content perturbations and
an exactly-zero S2 input slot) is worse in **3/3 seeds**: ΔRMSE
`+0.0619 / +0.1184 / +0.1005` (mean **+0.0936 ± 0.0289**), ΔMAE
`+0.0378 / +0.0427 / +0.0762`. The degradation is spread over 5–11 scaffold
groups, not one molecule. Under the stated caveat (the control changes the local
computation path, and capacity matching is not representation matching) this is
evidence that, **on this split, recipe and control, real typed S2 path content
provides additional predictive value**.

---

## 2. Frozen inputs, reuse, and the BASE replay

Nothing in `lcrc_esol_quickstart_v1/` was written. The derived structural cache
is redirected into this round's `cache/` (git-ignored, rebuilt deterministically
on the GPU host).

| item | value |
|---|---|
| split | same frozen manifest, `split_seed = 20261103`, `47db06adfd4f…`, 902/113/113 |
| row hashes | Train `02eee26b…`, Validation `ee70104c…` — identical to v1's records |
| dataset | `delaney-processed.csv`, sha `8c06a76f0c64…` (re-hashed on read) |
| raw graphs | `4c704b49…` · physchem `b3be0948…` · split audit `8e473293…` |
| LCRC cache | content fingerprint `30f2bf97…` — rebuilt from the committed raw artifact, matched exactly |
| structure normaliser | fingerprint `ba7bf26e…` — identical to v1 |
| BASE checkpoints | reused read-only, sha `f5f9676c… / ea043afa… / 9f0604ef…` (pinned from the v1 manifest, whose own self-digest is verified) |
| BASE retrained | **no** (0 runs) |

**BASE replay** (`analysis/base_replay.json`, re-run on `res` as
`analysis/base_replay_remote.json`): all three checkpoints reloaded, Validation
recomputed from the shipped pipeline.

| seed | replayed Valid RMSE | recorded (v1) | abs diff |
|---:|---:|---:|---:|
| 0 | 0.72047509 | 0.72047514 | 1.0e-08 |
| 1 | 0.76699071 | 0.76699085 | 1.4e-07 |
| 2 | 0.66122155 | 0.66122158 | 2.6e-08 |

Train RMSE is reproduced to ≤ 4.1e-08 as well. The replay was executed both
locally (CPU) and on the A100 host before any new training.

---

## 3. P0 — what was actually broken, and what was repaired

`analysis/p0_gates.json` — **PASS, 26/26 checks**. Only the necessary checks are
in this round; the old round's 25-gate data/vocabulary audit is not repeated.

### 3.1 Descriptor row alignment (P0.1)

In v1, `views["physchem"]` was the full 1,128-row matrix in **CSV row order** and
`collate()` indexed it with the **cache-local** `rows` (0..901 / 0..112). So
position *i* received CSV row *i*'s descriptors — a different molecule.

| split | positions with the wrong CSV row | median \|MolWt\| error | max \|MolWt\| error | max \|MolLogP\| error |
|---|---:|---:|---:|---:|
| Train (902) | **899 (99.7%)** | 91.1 g/mol | 662.7 | 12.7 |
| Validation (113) | **113 (100%)** | 120.9 g/mol | 652.7 | 9.4 |

Repair: `views["physchem_train"]` / `["physchem_valid"]` are built in **cache
order** from `cache["molecule_ids"]`, and `collate()` now *raises* if the matrix
length differs from the cache — the old failure mode is structurally
impossible. Verified for **all 1,015 Train+Validation molecules**, not a sample:

* `cache.molecule_ids[i] == split_rows[i]` for every position;
* `physchem_<split>[i] == physchem_all[csv_row_i]` exactly;
* RDKit recomputed from `cache.smiles[i]` equals that row, max abs diff **0.0**;

Audit table: `analysis/p0_row_alignment.csv` (1,015 rows, with the per-row
recomputed descriptors and both diffs). The five duplicated SMILES groups
(213/976, 233/276, 450/1019, 680/1069, 703/822) all sit **inside Train**, as the
frozen split audit required, and their descriptor rows are identical.

### 3.2 Exactly one standardisation (P0.2)

v1 standardised in `esol_train._physchem_matrix()` **and** again inside
`LCRCPhyschem6.forward()`. The drift that caused is not cosmetic:

| quantity | value |
|---|---|
| mean \|single − double\| standardised units | 1.143 |
| max \|single − double\| | 7.392 |

Repair: `collate()` forwards **raw** RDKit values; `forward()` applies
`(raw − train_mean) / train_scale` exactly once, using registered Train-only
buffers. Numerical verification on real batches:

| check | result |
|---|---|
| batch rows vs descriptor artifact | `0.0` (float32) |
| model-seen standardised values vs manual `(raw−mean)/scale` | `3.5e-07` |
| same vs the *double*-standardised values | `5.573` (**discriminating**) |
| descriptor perturbation → `h`, `contrib`, `contrib_atom`, `mean_h`, `mean_h2` | bitwise **unchanged** |
| descriptor perturbation → `rho` / prediction | changed (informative path proven) |
| scaler fit | Train only (902 rows), `valid_or_test_used = false`, refit from the aligned matrix matches v1 to `< 1e-9` |
| access pattern | `physchem` read **once**, as the **last** batch key |

The model has `rho.lins[0].in_features = 177` and every other input width
identical to BASE (`local.mlp.lins[0].in_features = 529`), so the six descriptors
can only enter `rho`; the atom/bond encoders, S1/S2 aggregates, local MLP and
`f` are the frozen BASE modules.

### 3.3 `rho` first-layer initialisation (P0.3)

v1 copied only state-dict entries of equal shape, so the widened
`rho.lins[0].weight` (64×177) was *freshly random* instead of BASE's.

Repair: copy the 171 original columns **and** the bias bitwise from the
freshly-seeded BASE, set the 6 new columns to exactly zero, copy every other
module bitwise, keep dropout `0.10`.

| gate | result |
|---|---|
| shared state entries bitwise identical (34 entries + geometry) | yes |
| old 171 columns / bias bitwise equal to BASE | yes |
| new 6 columns exactly zero | yes |
| new parameters | 384 = 6 × 64 (247,106 → 247,490) |
| `model.eval()` predictions, 113 Validation molecules, init | max abs diff vs BASE **0.0** |
| gradient on the 6 new columns (train mode, real batch) | finite, norm 2.75, every column > 0 |
| after one AdamW step | new columns move (1.0e-04) |
| batch order, seeds 0/1/2 | replayed == v1 BASE hashes (`007413ff…`, `70f7db8a…`, `16182097…`); every run records `paired_with_v1_base = true` |
| dropout | 0.10 in all arms |

---

## 4. H1 — `ESOL-PHYSCHEM6`

Recipe: the frozen ESOL recipe, unchanged (AdamW 3e-4 → 1e-5 per-step cosine,
wd 1e-3, batch 128, 360 epochs, grad clip 5.0, dropout 0.10, MSE on the
Train-standardised target, single best-Validation-RMSE checkpoint, no soup/EMA/
SWA/SSL, no early stopping). Only `rho`'s input width changed.

### 4.1 Results (Validation only)

| model | seed | best epoch | Train RMSE | Valid RMSE | Valid MAE | Valid R² |
|---|---:|---:|---:|---:|---:|---:|
| ESOL-BASE | 0 | 172 | 0.4002 | 0.7205 | 0.5371 | 0.9016 |
| ESOL-BASE | 1 | 157 | 0.4316 | 0.7670 | 0.5685 | 0.8885 |
| ESOL-BASE | 2 | 159 | 0.4196 | 0.6612 | 0.4853 | 0.9171 |
| **BASE mean ± sd** | | 163 ± 8 | **0.4171 ± 0.0158** | **0.7162 ± 0.0530** | 0.5303 ± 0.0420 | 0.9024 |
| ESOL-PHYSCHEM6 | 0 | 292 | 0.3380 | 0.7312 | 0.5269 | 0.8986 |
| ESOL-PHYSCHEM6 | 1 | 157 | 0.3981 | 0.7477 | 0.5583 | 0.8940 |
| ESOL-PHYSCHEM6 | 2 | 159 | 0.3889 | 0.6545 | 0.4798 | 0.9188 |
| **PHYSCHEM6 mean ± sd** | | 203 ± 77 | **0.3750 ± 0.0324** | **0.7112 ± 0.0497** | 0.5217 ± 0.0395 | 0.9038 |

Paired differences (candidate − BASE, negative = candidate better):

| seed | ΔValid RMSE | ΔValid MAE | ΔTrain RMSE | Δbest epoch | ΔTrain/Valid gap |
|---:|---:|---:|---:|---:|---:|
| 0 | **+0.0108** | −0.0102 | −0.0622 | +120 | +0.062 |
| 1 | −0.0193 | −0.0101 | −0.0335 | 0 | −0.024 |
| 2 | −0.0067 | −0.0054 | −0.0307 | 0 | −0.003 |
| **mean ± sd** | **−0.0051 ± 0.0151** | **−0.0086 ± 0.0027** | −0.0421 | +40 | +0.012 |

### 4.2 Did the model use the descriptors?

Yes, unambiguously (`analysis/physchem_usage.json`, no training involved: the
Validation descriptors are set to the Train mean, i.e. standardised value 0):

| seed | mean \|Δprediction\| | max \|Δ\| | molecules with \|Δ\|>0.01 | Valid RMSE with true descriptors | with Train-mean descriptors |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.439 | 1.86 | 112/113 | 0.7312 | **0.8351** |
| 1 | 0.502 | 2.53 | 113/113 | 0.7477 | **0.8438** |
| 2 | 0.547 | 2.81 | 113/113 | 0.6545 | **0.8900** |

The induced shift is aligned with the label (corr ≈ −0.71…−0.77), i.e. the
descriptors are being used in an informative direction. This is a usage
diagnostic, not a causal claim and not a configuration signal.

### 4.3 Interpretation

Supported:

* the three defects were real, are repaired, and are now gated numerically;
* at step 0 the arm is exactly BASE (`0.0` prediction difference), so any
  difference is caused by the six columns and not by initialisation;
* the arm fits Train better in 3/3 seeds (ΔTrain −0.031…−0.062) and improves
  Validation **MAE** in 3/3 seeds (mean −0.0086, sd 0.0027 — the most consistent
  effect observed in this round);
* the descriptors are actually used, and used informatively.

Not supported:

* a Validation-RMSE improvement. The mean is −0.0051 with seed sd 0.0151; one
  seed is clearly worse (+0.0108), the group bootstrap interval of every seed
  crosses zero (pooled mean interval `[−0.055, +0.043]`, 61% of resamples
  favouring the candidate). The honest label is **UNRESOLVED / at most a small
  effect**, not "physchem descriptors help".
* any claim that the original S1/S2 representation "lacks" this information, or that the
  bottleneck is the `rho` branch: BASE already reaches 0.7162 without them, and
  the arm that is *given* the descriptors gains almost nothing on RMSE while
  fitting Train noticeably better.

Seed 0 is worth recording: the six columns delay the best epoch from 172 to 292
and buy 0.062 Train RMSE, but Validation is `+0.0108` worse — the extra input is
most useful for fitting, not for generalising, in that seed.

---

## 5. H2 — `ESOL-S2BLIND-E` (capacity-matched typed-S2-blind control)

Design: the validated control of
`lcrc_lipo_mechanism_generalization_ssl_v2/code/v2_models.py`, **imported, not
re-derived** (`v2_models.build_e`, `LCRCCapacityMatched`,
`LocalContentCapacityMatched`). On ESOL the frozen `lipo_v1` spec and
`DEFAULT_CONFIG` are the same, so the arithmetic ports exactly:

| quantity | value |
|---|---:|
| BASE parameters | 247,106 |
| typed-S2 module parameters removed (`path_emb`, `end_emb`, `path_u`, `path_v`, `joint_proj`, `close_emb`, `path_gate`) | 32,192 |
| original 200-wide S2 slot into the LocalMLP that stays **dead** (columns 320:520 of `mlp.lins[0].weight`) | 38,400 |
| S1-context replacement (320→135→200) | 70,535 |
| **nominal** parameters | 285,449 |
| **effective** parameters | **247,049** |
| Δ effective vs BASE | **−57 (−0.0231%)** |

Kept: atom features, bond features, S1 statistics, `atom_scalar` (which still
contains path counts and the closure ratio), the 42 global statistics, per-atom
contributions and the global `rho` readout. Removed: only the *typed* path
tensors `path_cat`, `path_end63`, `path_end`, `path_closure`, `path_graph`,
`path_atom`.

### 5.1 Is typed S2 really disconnected? (P2.2 / P2.3, CPU)

| gate | result |
|---|---|
| `RestrictedBatch` (raise on typed-S2 keys) forward | succeeds, **bitwise identical** to the unrestricted batch |
| predictions under `code_shift` / `closure_flip` / `path_end_shift` | max \|Δ\| **0.0**, bitwise identical in all three |
| activation entering the LocalMLP S2 slot | max abs **0.0** (exactly zero, all batches) |
| gradient on the dead columns | exactly **0.0** |
| gradient on the replacement network | all four parameters > 0 (9.7 … 59.1) |
| shared init vs this round's BASE | bitwise identical; E's reference BASE == our builder |
| message passing / attention | none (`iterative_hidden_message_passing = false`, no such modules) |
| **positive control** on the *trained* BASE s0 | `code_shift` mean \|Δy\| **0.620**, 112/113 molecules > 1e-3; `closure_flip` 0.150 / 110; `path_end_shift` 0.051 / 108 |

So the intervention removes exactly the input path that the trained BASE is
measurably using.

### 5.2 Results (Validation only)

| model | seed | best epoch | Train RMSE | Valid RMSE | Valid MAE | Valid R² |
|---|---:|---:|---:|---:|---:|---:|
| ESOL-S2BLIND-E | 0 | 117 | 0.5384 | 0.7823 | 0.5749 | 0.8840 |
| ESOL-S2BLIND-E | 1 | 206 | 0.4379 | 0.8854 | 0.6111 | 0.8514 |
| ESOL-S2BLIND-E | 2 | 181 | 0.4549 | 0.7617 | 0.5615 | 0.8900 |
| **E mean ± sd** | | 168 ± 46 | **0.4771 ± 0.0538** | **0.8098 ± 0.0663** | 0.5825 ± 0.0257 | 0.8751 |

Paired differences (E − BASE):

| seed | ΔValid RMSE | ΔValid MAE | ΔTrain RMSE | Δbest epoch |
|---:|---:|---:|---:|---:|
| 0 | **+0.0619** | +0.0378 | +0.1383 | −55 |
| 1 | **+0.1184** | +0.0427 | +0.0063 | +49 |
| 2 | **+0.1005** | +0.0762 | +0.0353 | +22 |
| **mean ± sd** | **+0.0936 ± 0.0289** | **+0.0522 ± 0.0209** | +0.0599 | +5 |

Worse in 3/3 on both metrics. The gap is ≈ 1.8× the between-seed sd of BASE
(0.053), and the three deltas share sign.

### 5.3 Interpretation

Permitted statement: **in the current ESOL split, recipe and this
capacity-matched control, the real typed S2 path content shows additional
predictive value.**

Not permitted (and not claimed): "S2 is the only reason", "S2 is proven
necessary". The control differs in more than its information source — it
removes the typed-path modules, moves the LocalMLP's input distribution (a
320-wide S1 context plus a zero slot instead of a 529-wide mixed input) and
injects a parameter-free residual into the hidden state. Equal *effective
parameter count* is not equal *representation*, *gradient scale* or *optimisation
difficulty*. E's Train RMSE is also higher in 3/3 seeds (+0.060 on average), so
part of the Validation gap is underfitting, not only missing information. The
`atom_scalar` path counts and closure ratio survive, so this is not "all two-hop
information removed".

---

## 6. Statistics, uncertainty, and what is not a winner

* Same-seed pairing is exact: all six runs reproduce the frozen BASE batch
  order (`paired_with_v1_base = true`), and BASE/PHYSCHEM6 share their
  initialisation by construction.
* Paired frozen-scaffold-group bootstrap (10,000 resamples, seed 20261103,
  88 Validation groups: 77 singletons, 6×2, 2×3, 1×5) — **descriptive only**,
  it does not model uncertainty across different scaffold partitions:

| arm | seed | observed ΔRMSE | 95% interval | fraction of resamples favouring candidate |
|---|---:|---:|---|---:|
| PHYSCHEM6 | 0 | +0.0108 | [−0.052, +0.076] | 0.38 |
| PHYSCHEM6 | 1 | −0.0193 | [−0.072, +0.029] | 0.77 |
| PHYSCHEM6 | 2 | −0.0067 | [−0.042, +0.023] | 0.66 |
| S2BLIND-E | 0 | +0.0619 | [−0.028, +0.149] | 0.09 |
| S2BLIND-E | 1 | +0.1184 | [−0.014, +0.267] | 0.04 |
| S2BLIND-E | 2 | +0.1005 | [+0.039, +0.159] | 0.002 |

* Is the change driven by a few molecules or groups? No, in either direction.
  For PHYSCHEM6 the single largest group carries 8–20% of the absolute ΔSSE and
  5–10 groups are needed for half of it; for E it is 11–25% and 5–11 groups.
  Molecule-level directions are near-split for PHYSCHEM6 (57/56, 57/56, 54/59)
  and clearly negative for E (52/61, 53/60, 48/65).
* No threshold rule was used ("improvement > 0.03 ⇒ mechanism holds" is not
  applied). The judgement combines effect size, sign consistency, seed spread
  and scaffold-group sensitivity.
* Discipline: the 113-molecule Validation split has been used by several rounds
  and is **not** a fresh confirmation set; molecule errors are correlated inside
  scaffolds; the old Test split was already read and analysed and is **not** used
  here for evaluation, selection or interpretation.

---

## 7. Answers to the two questions, in one line each

**H1: does explicit physicochemical information on `rho` help?** It is *used*
(0.44–0.55 log-unit prediction shift; +0.10…+0.24 Validation RMSE when the
descriptors are neutralised) but it does not produce a **Validation-RMSE**
improvement beyond seed noise: treat as `UNRESOLVED`, with a small, consistent
MAE/Train-RMSE gain and one clearly worse seed.

**H2: does typed S2 path content carry value beyond matched S1/local capacity?**
On this split and under this control, yes — `+0.094 ± 0.029` Validation RMSE in
3/3 seeds with the typed-S2 tensors provably unread, but the control also
changes the local computation path, so the claim stays at "additional predictive
value of the typed S2 content", not "S2 is necessary".

---

## 8. Execution, budget, provenance

| item | value |
|---|---|
| formal runs | **6** (PHYSCHEM6 ×3, S2BLIND-E ×3); BASE reused, 0 retrained |
| GPU | 6 × A100-SXM4-40GB on `res` (process backend, `res-gpu0` / `res-gpu1`), fp32 |
| training wall clock | 69.6–78.4 s per run, 444.7 s total → **0.1235 / 0.6 GPU·h** (`analysis/gpu_ledger.json`) |
| non-training jobs | 2 × CPU `preflight` on `res` (BASE replay, cache rebuild), no GPU |
| failures / retries | no training failure, no seed re-run, no run replaced. One **preflight** job (`esol2-preflight`) failed because the v1 `*.npz` prediction files are git-ignored and therefore absent on the GPU host; fixed by deriving BASE predictions from the committed checkpoints (`1276fc6a`) and re-run as `esol2-preflight2`, which passed. No formal run was affected. |
| torch / driver | 2.5.1+cu124 / 550.163.01 |
| smoke | local CPU 2-epoch smoke of both arms before deploy (`analysis/smoke/smoke.json`), never counted as a formal run |
| freeze | `analysis/protocol_freeze.json`, sha `22b79f167b5d…`, written before any new training, `p0_status = PASS`, BASE replay passed |
| P0 gates | `analysis/p0_gates.json`, **PASS 26/26** |
| new-checkpoint manifest | `analysis/manifest.json`, sha `d781342195c0…`, 6 arms, `test_evaluated = false`, `base_retrained = false` |
| post-run provenance | `analysis/post_run_provenance.json`: only `esol2/analysis` + `esol2/run_esol2` differ from the freeze bytes; every run's *training* code hash equals the freeze, and each `train.json` carries `test_labels_read = false` |
| Test | **no Test cache, no Test label, no Test metric** anywhere in this round |

### Artifacts

```
tracks/structural-sufficiency/lcrc_esol_physchem_s2_v1/
  REPORT.md
  configs/esol2.yaml
  code/{esol2_common,esol2_data,esol2_models,esol2_train,esol2_p0,esol2_analysis,run_esol2}.py
  analysis/p0_gates.json              analysis/p0_row_alignment.csv
  analysis/protocol_freeze.json       analysis/base_replay.json
  analysis/base_replay_remote.json    analysis/post_run_provenance.json
  analysis/valid_comparison.csv       analysis/paired_valid.csv
  analysis/paired_summary.json        analysis/physchem_usage.json
  analysis/per_molecule_valid.csv     analysis/manifest.json
  analysis/gpu_ledger.json            analysis/coverage.json
  analysis/runs/<arm>/{checkpoint.pt,curve.csv,train.json,diagnostics.json}
```

### Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_esol_physchem_s2_v1/code
uv run python run_esol2.py preflight     # BASE replay (local, CPU)
uv run python run_esol2.py p0            # 26 gates (needs RDKit, local)
uv run python run_esol2.py freeze        # only after P0 PASS and replay PASS
uv run python run_esol2.py smoke         # 2-epoch CPU smoke of both arms
rr deploy res && rr run res <tag> --pool res-gpu0 --result … -- \
  python .../run_esol2.py train --arm physchem6_s0 --device cuda:0 --pool res-gpu0
uv run python run_esol2.py analyse       # paired Validation comparison
uv run python run_esol2.py manifest      # freeze the six new checkpoints
uv run python run_esol2.py ledger        # GPU accounting
```

---

## 9. What this round does **not** buy

* no evidence that explicit physchem descriptors *harm* ESOL, and none that they
  *help* by a margin that matters at this sample size;
* no claim about ESOL Test performance of either new arm — the old Test split
  was already consumed by the previous round and would not be blind;
* no claim that typed S2 is "the" necessary signal: the control changes the local
  circuit and fits Train worse;
* no dependency of these conclusions on the old round's Test numbers, on scaffold
  difficulty, or on any post-hoc threshold.
