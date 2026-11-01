# LCRC-A seven-task MoleculeNet benchmark pilot — v2 report

Round: `lcrc-molnet-benchmark-pilot-v2` · protocol
`BENCHMARK_PROTOCOL.md`, frozen as `analysis/protocol_freeze.json`
(sha256 `b50137cdd81c…`) before the first training job.
Host `res` → `a100-2`, 2× A100-SXM4-40GB, driver 550.163.01,
torch 2.5.1+cu124, commit `62afa5a2` for every GPU run.

**One seed (0) per dataset. Nothing in this round is a multi-seed claim.**
Seven datasets, seven new GPU runs (three formal + four Validation-only probes),
**1.1478 A100 GPU-hours** of a 2.0 h cap, one Test read per new dataset.

---

## 1. What was executed

| stage | what | where it lives |
|---|---|---|
| P0 | source pinning, split integrity, label-free caches, Toy/duplicate/coverage/parameter/locality/numerics/target gates, Test-label isolation | `analysis/p0_audit.json` (**PASS** on all seven) |
| — | protocol freeze before training | `analysis/protocol_freeze.json` |
| P1 | three formal single-seed LCRC runs, 360 epochs, frozen R3 recipe | `analysis/runs/<dataset>_s0/` |
| — | checkpoint freeze (checkpoint sha256, frozen Validation metric, cache fingerprints, split-row hashes, code hashes) | `analysis/checkpoint_manifest.json` |
| P2 | frozen ECFP4 + 217 RDKit descriptors + XGBoost, CPU, **same row ids** | `analysis/baseline/`, `analysis/baseline_manifest.json` |
| P3 | four Validation-only recipe probes on FreeSolv/BACE | `analysis/validation_tuning_probe.csv` |
| Test | three one-shot reads (BBBP, Tox21, ogbg-molhiv) + seven baseline reads | `analysis/test_access_ledger.json` |
| — | tables, literature matrix, GPU ledger | `analysis/seven_dataset_table.csv`, `literature_matrix.csv`, `gpu_ledger.json` |

The four Sprint-01 datasets were **not** re-trained and **not** re-scored: their
committed artifacts are read read-only and their Test numbers are carried forward
with the label `EXPOSED_TEST_EXPLORATORY`.

Two launches were aborted before any training step: the first deployed revision
(`137f5a3b`) did not yet commit the label-free raw-attribute artifact, so the runs
exited with `FileNotFoundError` in the cache build. The artifact was committed
(`62afa5a2`), and the same jobs were relaunched; all seven formal runs and the one
timing smoke ran at `62afa5a2`. The aborted jobs are listed in
`analysis/rr_run_registry.json` with `exit_code: 1` and charged nothing.

The Test gate enforced, per dataset: frozen manifest membership, checkpoint
sha256, zero drift in the 14 scoring-critical modules, cache-fingerprint
reproduction, Validation-metric reproduction (≤1e-5; measured 0.0, 1.1e-16,
4.6e-06), curve-argbest = best epoch, Test row ids = the P0 split manifest, and
"no second read". All passed for all three datasets. The recorded Test metric
re-derives exactly from the stored predictions alone
(`mb2_eval.py --dataset <d> --verify`: diff 0.00e+00 three times).

## 2. Main table — frozen LCRC vs the matched classical baseline

Same Train/Validation/Test row ids, same splits, both frozen before the single
Test read. `Δ = LCRC − baseline`; for RMSE a negative Δ favours LCRC, for ROC-AUC
a positive Δ favours LCRC.

| dataset | split | metric | LCRC valid | LCRC **test** | XGB valid | XGB **test** | Δ (test) | rel. | better | exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| ESOL | random | RMSE ↓ | 0.5585 | **0.5226** | 0.6425 | 0.5281 | −0.0056 | +1.0 % | LCRC | EXPOSED_TEST_EXPLORATORY |
| Lipophilicity | random | RMSE ↓ | 0.7109 | **0.5727** | 0.6532 | 0.6477 | −0.0750 | +11.6 % | LCRC | EXPOSED_TEST_EXPLORATORY |
| FreeSolv | random | RMSE ↓ | 0.7594 | **1.0870** | 1.1331 | 1.0977 | −0.0107 | +1.0 % | LCRC | EXPOSED_TEST_EXPLORATORY |
| BACE | scaffold | ROC-AUC ↑ | 0.7761 | **0.8254** | 0.7065 | 0.7969 | +0.0284 | +3.6 % | LCRC | EXPOSED_TEST_EXPLORATORY |
| BBBP | scaffold | ROC-AUC ↑ | 0.9733 | 0.7110 | 0.9774 | **0.7208** | −0.0098 | −1.4 % | baseline | FIRST_TEST_READ |
| Tox21 | random | macro ROC-AUC ↑ | 0.8663 | 0.8353 | 0.8610 | **0.8521** | −0.0168 | −2.0 % | baseline | FIRST_TEST_READ |
| ogbg-molhiv | OGB scaffold | ROC-AUC ↑ | 0.8365 | 0.7524 | 0.8367 | **0.7624** | −0.0101 | −1.3 % | baseline | FIRST_TEST_READ |

Parameters: 247,106 per single-output dataset; **248,536** for Tox21 (the `f`/`ρ`
heads widen from 1 to 12 outputs, +1,430). The shared encoder is bit-identical
between the 1-output and 12-output model at the same seed
(`p0_audit.json` → `G6_frozen_model.parameter_identity`).

HIV uses the official `ogb.graphproppred.Evaluator("ogbg-molhiv")`; the Evaluator
and the sklearn cross-check agree to 0.0e+00 (0.7523880337588598 both).

Secondary Test metrics (record): ESOL MAE 0.3961; Lipophilicity MAE 0.4256;
FreeSolv MAE 0.7126; BACE PR-AUC 0.8516; BBBP PR-AUC 0.7212; Tox21 macro PR-AUC
0.4716; HIV PR-AUC 0.2114 (HIV Test positive rate 3.16 %, so PR-AUC is bounded low).

## 3. The Validation split is not a usable proxy everywhere — and that is a data fact, not a model excuse

Both models fall off a cliff from Validation to Test on BBBP: LCRC 0.9733 → 0.7110
(−26.2 points), XGBoost 0.9774 → 0.7208 (−25.7 points). Two unrelated model
families losing almost exactly the same amount points at the split, not at a model.
`analysis/split_representativeness.json` measures that directly, using one trivial
descriptor (molecular weight):

| dataset | positive/label rate train / valid / test | mean MW train / valid / test | \|AUC(MW) − 0.5\|·2 valid → test |
|---|---|---|---|
| BBBP | 0.822 / 0.549 / **0.525** | 327 / **454** / 379 | **0.71 → 0.23** |
| BACE | 0.426 / 0.556 / 0.605 | 474 / 498 / 506 | 0.22 → 0.52 |
| ESOL | −3.10 / −2.71 / −3.03 | 206 / 190 / 202 | 0.73 → 0.62 |
| Lipophilicity | 2.18 / 2.22 / 2.23 | 383 / 387 / 378 | 0.12 → 0.20 |
| FreeSolv | −3.82 / −3.66 / −3.80 | 135 / 145 / 160 | 0.28 → 0.24 |
| Tox21 | 0.075 / 0.077 / 0.078 | 278 / 267 / 269 | 0.36 → 0.36 |
| ogbg-molhiv | 0.037 / 0.020 / 0.032 | 367 / 402 / 363 | 0.37 → 0.45 |

On BBBP the Train split is 82 % positive while Valid and Test are ~53 % positive,
and molecular weight alone ranks the Validation labels at AUC 0.14 (i.e. 0.86
inverted) but only 0.39 on Test. The ChemBench BBBP scaffold split therefore has a
Validation split that is *structurally* unlike its Test split. Consequences:

* BBBP's Validation signal (0.97 for both models) is not evidence that any model
  generalises; it is evidence that "big molecules are inactive" holds in
  Validation and does not hold in Test.
* The frozen "best-Validation checkpoint" rule selects epoch 24 of 360 on BBBP,
  i.e. it picks a checkpoint using the least trustworthy Validation set of the
  seven. That is a property of the rule interacting with this split, and it is
  visible here precisely because the rule is frozen.
* **No result in this round was selected on Validation except the checkpoint.**

Where Validation does behave (Tox21, HIV, Lipophilicity, FreeSolv, ESOL), the
Validation→Test drop is 1–8 points and the ranking of LCRC vs the baseline is
mostly preserved; the one reversal is Lipophilicity, where Validation says the
baseline is ahead (+0.058 RMSE) and Test says LCRC is ahead by 0.075 RMSE. A
single Validation split cannot reliably rank two models at these sample sizes.

## 4. P3 — the FreeSolv/BACE regularisation probes (Validation only)

Four candidates, fixed before running, one parameter each, never scored on Test.

| dataset | config | dropout | weight decay | best epoch | valid (primary) | train (primary) | train−valid gap | Δ vs base | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FreeSolv | R3 base | 0.10 | 1e-3 | 116 | **0.7594 RMSE** | 0.6420 | 0.117 | — | base |
| FreeSolv | `fs_dropout00` | 0.00 | 1e-3 | 128 | 0.7893 | 0.5508 | 0.239 | +3.9 % worse | no improvement |
| FreeSolv | `fs_dropout20` | 0.20 | 1e-3 | 253 | 0.7946 | 0.4882 | 0.306 | +4.6 % worse | no improvement |
| BACE | R3 base | 0.10 | 1e-3 | 102 | **0.7761 ROC-AUC** | 0.9657 | 0.190 | — | base |
| BACE | `bace_dropout20` | 0.20 | 1e-3 | 68 | 0.7807 | 0.9480 | 0.167 | +0.6 % better | < 1 %: single-seed noise |
| BACE | `bace_wd1e2` | 0.10 | 1e-2 | 98 | 0.7758 | 0.9646 | 0.189 | −0.05 % | no improvement |

Reading: removing dropout makes FreeSolv *worse* on both splits of the gap (train
0.5508, valid 0.7893 — the model overfits harder and transfers worse); doubling it
also makes Validation worse. On BACE, doubling dropout is worth +0.005 ROC-AUC and
×10 weight decay is worth −0.0004, i.e. nothing. **The frozen regularisation is
already at a local optimum in the direction these probes could see**, and the
FreeSolv/BACE Test-vs-literature gap is not a dropout or weight-decay artefact.

## 5. Literature position (graded, never ranked as if the protocol matched)

Only one dataset in this round has an **exact split match** with a published number:

| dataset | this round (Test) | exact split match? | closest published | grade |
|---|---|---|---|---|
| ogbg-molhiv | 0.7524 | **YES** — official OGB row ids | OGB GIN+features+virtual node 0.7707 ± 0.0149 (10 runs); OGB GCN 0.7599 ± 0.0119 | **A** |
| BBBP | 0.7110 | no (ChemBench scaffold draw ≠ OGB/ GROVER draw) | GROVER-large 0.940 ± 0.019 (3 runs, its own scaffold split); OGB GIN 69.71 % (its own draw) | B/C |
| Tox21 | 0.8353 macro | no (ChemBench random split) | GROVER-large 0.831 ± 0.025 (scaffold, 3 runs); MoleculeNet GCN 0.829 (8014 molecules, random) | C |
| BACE | 0.8254 | no (published indices unknown) | GROVER-large 0.894 ± 0.028 (scaffold, 3 runs) | B |
| ESOL | 0.5226 RMSE | no | MoleculeNet MPNN 0.58, XGBoost 0.99 (random); GROVER D-MPNN 0.980 (scaffold) | B/C |
| Lipophilicity | 0.5727 RMSE | no | MoleculeNet XGBoost 0.799, GraphConv 0.655; GROVER-large 0.560 (scaffold) | B/C |
| FreeSolv | 1.0870 RMSE | no | MoleculeNet MPNN 1.15, XGBoost 1.74 (random); GROVER-large 1.544 (scaffold) | B/C |

The single honest cross-paper statement the data supports: **on `ogbg-molhiv`'s
official split, frozen LCRC-A at 0.7524 lands 1.8 points below the published
10-run GIN configuration (0.7707) and 0.8 points below the published 10-run GCN
(0.7599)** — same rows, same metric, different input featurisation and a single
seed. That is "competitive but not ahead", not "state of the art".

Every literature number is transcribed with its locator and the sha256 of the
fetched document (`analysis/literature_matrix.csv`, `analysis/literature_sources.json`);
33 rows: 2 grade A, 12 grade B, 19 grade C.

## 6. Findings

1. **Performance map (Test, single seed).** Regression: LCRC 0.5226 RMSE (ESOL),
   0.5727 (Lipophilicity), 1.0870 (FreeSolv). Classification: 0.8254 (BACE,
   scaffold), 0.7110 (BBBP, scaffold), 0.8353 macro (Tox21, random, 12 endpoints),
   0.7524 (ogbg-molhiv, official scaffold). Tox21 spans 0.711 (NR-ER) to 0.909
   (NR-AhR); all 12 endpoints were definable on Test, so the macro average has no
   silently dropped task.
2. **Against a cheap classical baseline on identical rows, LCRC wins 4/7 and loses
   3/7.** It wins the four Sprint-01 datasets (Lipophilicity decisively, by 11.6 %
   relative; ESOL, FreeSolv and BACE by 1.0–3.6 %) and loses all three new
   datasets by 1.3–2.0 % relative. The direction is consistent across the three
   fresh datasets, so this is a real (if small) pattern: on fingerprint-friendly
   classification tasks the frozen message-passing model does not earn its cost.
   On the four exposed datasets the comparison must stay descriptive.
3. **Train/Validation separation says little about Test.** HIV reaches 0.995 Train
   ROC-AUC and 0.8365 Validation, then 0.7524 Test: memorisation of Train
   scaffolds is not the whole story (Train/Valid/Test scaffolds are disjoint), yet
   the model still overfits structure-specific signal. BBBP shows the opposite
   pathology: near-perfect Validation but 0.711 Test, for both model families.
4. **Validation-based checkpoint selection is the only selection this round
   performs, and on BBBP it is provably unreliable** (Section 3). Every other
   pipeline choice was frozen before training.
5. **Tox21's masked multitask head works but is not free**: 248,536 parameters,
   macro Validation 0.8663 vs 0.8610 for per-endpoint XGBoost, and macro Test
   0.8353 vs 0.8521 — the classical baseline wins on the same observed cells.
6. **Data-property limits worth recording**: 17.05 % of Tox21 label cells are
   unobserved and are never treated as negatives; eight Tox21 salts need the one
   documented relaxed parse; scaffold overlap between Train and Validation is
   0.000 for BACE/BBBP/HIV but 0.52–0.95 for the random-split sets, which is the
   main reason FreeSolv and Lipophilicity look "easier" than BBBP/HIV.

## 7. Is it worth trying to tune FreeSolv/BACE further?

**No — not with this model, on these splits, in this form.** The evidence:

* The four single-parameter probes moved Validation by +0.6 % (BACE dropout 0.2),
  −0.05 % (BACE wd 1e-2), +3.9 % *worse* (FreeSolv dropout 0) and +4.6 % *worse*
  (FreeSolv dropout 0.2). The only improvement is inside single-seed noise, and
  on FreeSolv every probe is worse: dropping dropout and doubling it both hurt,
  so the frozen 0.10 is not a symptom of a mis-specified regularisation
  strength.
* FreeSolv Test 1.0870 vs Validation 0.7594 is a 43 % inflation, and 95 % of its
  Validation molecules have a scaffold already in Train: the dataset measures
  interpolation, and LCRC already beats the matched baseline there (1.0870 vs
  1.0977). The gap to published MPNN (1.15, random split) is inside one seed of
  noise; there is nothing to tune toward.
* BACE Test 0.8254 already exceeds the matched baseline (0.7969); the remaining
  distance to the published 3-run GROVER-large number (0.894) is 0.07, on a
  *different* scaffold draw, and a ±0.005 Validation effect cannot resolve a 0.07
  gap.
* Dimensions that would plausibly matter (feature vocabulary, S2 radius, readout)
  are architectural, not regularisation, and were explicitly out of scope. Tuning
  them on a single seed with 64–151 Validation molecules would mostly fit noise,
  and on BBBP it would fit a Validation split that is known to be unrepresentative.

## 8. Two recommendations (in order)

1. **Replace single-split, single-seed Validation with a repeated-split protocol on
   the three datasets whose Validation is demonstrably unreliable** (BBBP first:
   its Train/Valid/Test priors are 0.82/0.55/0.53 and a trivial descriptor's
   ranking strength flips 0.71 → 0.23). Concretely: 3 seeds × the 3 published
   ChemBench repeats (all already vendored and hashed here), report mean ± spread
   per dataset, and stop using one Validation epoch to pick a checkpoint. This
   costs ≈ 3× the current GPU time (≈ 3.4 h for the three new datasets) and is the
   only change that turns the 1–2-point LCRC-vs-XGBoost differences into
   statements that can be defended.
2. **Spend the next architecture cycle on the input featurisation, not on the
   message-passing depth.** The honest comparison here is "frozen LCRC with 15
   raw atom classes beats ECFP4+XGBoost on the four Sprint-01 tasks and loses on
   the three new ones", and `ogbg-molhiv` is the one place with an exact-split
   reference where LCRC (0.7524) sits below both published GCN (0.7599) and GIN
   (0.7707) that *do* use OGB's richer node/edge features. A single-seed,
   exact-split test of one alternative featurisation on `ogbg-molhiv` (the only
   dataset where the comparison is exact) is a cheaper, better-identified
   experiment than any further tuning of the four exposed datasets.

## 9. Threats to validity

* One seed per dataset, no confidence intervals, no significance tests. A
  difference of one or two points is an observation, not evidence.
* Four of the seven Test numbers were already exposed in Sprint 01; they are
  labelled `EXPOSED_TEST_EXPLORATORY` and were produced under the same frozen
  protocol, but they cannot be "first reads" again.
* The three new datasets' Test reads were single-shot at
  `analysis/runs/<dataset>_s0/test_eval.json`; the ledger records the timestamps,
  manifest hash and prediction hash of each. A one-line `UnboundLocalError` in the
  reporting tail of `mb2_eval.py` after `test_eval.json` was written (a stray
  `del` of an unbound name) was fixed afterwards; the numbers were re-derived from
  the stored predictions with diff 0.00e+00, so no Test data was read twice.
* `ogbg-molhiv` is scored on a 3.5 %-positive Test split; PR-AUC is the more
  informative secondary metric and ROC-AUC differences of ~0.01 are small in
  absolute terms.
* The matched baseline is one frozen XGBoost configuration, deliberately
  untuned — a tuned baseline would be a stronger opponent, so the LCRC advantage
  on the Sprint-01 tasks is an upper bound, not a lower bound.
* BBBP's ChemBench scaffold reproducibility is weak by construction: the three
  published repeats are byte-identical, so that dataset has one split draw, not
  three.

## 10. Budget accounting

| item | value |
|---|---|
| formal GPU runs this round | 7 (3 P1 + 4 P3) — limit 7 |
| charged GPU time | **1.1478 h** (training wall clock) of 2.0 h |
| charged GPU time, process wall clock (stricter) | 1.1614 h of 2.0 h |
| engineering smokes (not results) | 0.0059 h |
| Sprint-01 GPU time (read-only, not re-charged) | 0.1747 h |
| CPU baseline training, all seven datasets | 66.8 s total |
| Test inference (CPU) | < 1.5 s per dataset |

Per-run detail: BBBP 0.0384 h, Tox21 0.1789 h, `ogbg-molhiv` 0.8228 h (projected
0.86 h from a 2-epoch GPU smoke before launch, so the stop condition
"projected HIV run would exceed the remaining cap" was evaluated and not met),
FreeSolv probes 0.0118 + 0.0121 h, BACE probes 0.0417 + 0.0420 h.
`analysis/rr_run_registry.json` maps every experiment to its `rr` run id, node,
driver, torch/CUDA regime and commit.

## 11. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2
uv run python code/mb2_source.py --write && uv run python code/mb2_raw_artifact.py --write
uv run python code/mb2_p0.py && uv run python code/mb2_protocol_freeze.py
uv run python code/mb2_baseline.py --stage all && uv run python code/mb2_baseline.py --stage freeze

rr run res molnet-v2-train-BBBP     --pool res-gpu0 --gpus 1 --cpus 4 --mem 16G \
  --result tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/runs/BBBP_s0 \
  -- python code/mb2_train.py --dataset BBBP --device cuda
# Tox21 (res-gpu1), ogbg-molhiv (res-gpu0), and the four probes likewise

uv run python code/mb2_freeze.py
uv run python code/mb2_eval.py --dataset BBBP --device cpu        # then Tox21, ogbg-molhiv
uv run python code/mb2_eval.py --dataset BBBP --verify            # recompute from stored predictions
uv run python code/mb2_baseline.py --stage test
uv run python code/mb2_probe.py && uv run python code/mb2_literature.py && uv run python code/mb2_tables.py
uv run python code/mb2_split_audit.py && uv run python code/mb2_split_representativeness.py
uv run python -m pytest tests -q                                  # 49 tests
```
