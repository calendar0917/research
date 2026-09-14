# H96 T=2 recurrent pair--centre — faithful port to OGBG-MolHIV

Date: 2026-09-14
Protocol: `molhiv_recurrent_pair_centre_v1`
Code commit at run time: `481a7fa`; runs on the deterministic GPU regime
(`torch.use_deterministic_algorithms(True)`).

## Question

Does the frozen ZINC H96 configuration (T=2 weight-tied recurrent pair--centre,
`h=96, q=16, token_width=32, center_context_hidden=60`) transfer as a *molecular
inductive bias* to OGBG-MolHIV, or was the ZINC behaviour a ZINC-specific
capacity/allocation artefact?  Minimal faithful port: same architecture, no
MolHIV-specific head/feature/loss engineering.

## Protocol (pre-registered)

* Dataset: `ogbg-molhiv`, **official OGB scaffold split**
  (train 32,901 / valid 4,113 / test 4,113; positives 1,232 / 81 / 130).
* Records: pre-built `exact_rooted_{train,valid,test}.pkl` cache from the prior
  `unified_relational_patch_molhiv_exact_rooted` run.  All data-fitted transforms
  (typed/parent vocabularies, patch/context standardizers) are fit on the
  **official train split only**.
* Model: `MolhivRecurrentPairCentreModel` subclass of
  `mpp.PatchPathModel` (`T=1` reduces bit-identically to the base model);
  `1,076,589` params.
* Loss: `BCEWithLogitsLoss`, **unweighted**; Adam lr 1e-3, wd 1e-5, batch 128,
  240 epochs max, patience 40, grad clip 5.0; valid ROC-AUC every epoch.
* Selection (frozen before test): raw = highest-valid-ROC-AUC checkpoint
  (ties -> earliest epoch); Top-5 soup = equal-weight parameter average of the 5
  highest-AUC checkpoints, K/w fixed, no search.
* `architecture_freeze.json` written **before** any test read;
  `official_test_unlock.json` records `frozen_before_test: true`.

GPU smoke (deterministic): same-seed forward/backward bit-identical on GPU0/GPU1;
T=2 has 2 centre updates; empty-bucket/permutation invariants hold.

## Results

| seed | best epoch | raw valid AUC | soup valid AUC | epochs | test raw AUC | test soup AUC |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 0.814239 | 0.809505 | 56 | 0.769455 | 0.768080 |
| 1 | 61 | 0.849865 | 0.851092 | 101 | 0.755754 | 0.756110 |
| **2-seed mean** | | **0.832052** | **0.830299** | | **0.762605 ± 0.009688** | **0.762095 ± 0.008464** |
| 2-seed ensemble | | | | | **0.780048** | 0.775969 |

Both seeds early-stopped; canonical patience never boundary-pinned.  Top-5 soup
is neutral on test (raw mean 0.762605 vs soup 0.762095).

## Comparison (context only — different protocols)

| model | valid AUC | test AUC | params |
|---|---:|---:|---:|
| **this port (H96 RPC)** | 0.832 | 0.7626 (mean) / **0.7800** (ens.) | 1.08M |
| prior `MOLHIV_PATCH_PATH_POOLING_20260904` | 0.8028 | **0.7852** | ~1M |
| CIN-small (published) | — | 0.801 | — |
| GPS (published) | — | 0.788 | — |
| GIN (published) | — | 0.756 | — |

* Validation is clearly **higher** than the prior port (+0.029 mean; seed1
  +0.047), but official **test is lower** than the prior port
  (−0.023 on the 2-seed mean; even the 2-seed ensemble 0.7800 is below 0.7852).
* The valid→test gap is large and seed-dependent (seed0 −0.045, seed1 −0.094):
  the 4,113-graph / 81-positive valid split makes checkpoint selection noisy and
  seed1's high valid AUC does not survive on test.
* Test lands between GIN (0.756) and GPS (0.788)/CIN (0.801); it does not beat
  the prior patch-path pooling and is below the stronger published references.

## ZINC-specific-feature leakage audit

Every input channel of the ported ZINC pipeline is graph-observable and
label-free; no target-derived feature is carried over:

* patch/typed certificates and parent certificates — hashes of rooted
  neighbourhood structure (graph-observable);
* patch matrix — atom/bond/degree features;
* `pair_relation` — atom/bond types, distances, patch overlap, boundary
  intersection (graph-observable);
* global context (`global_all`, width 62) — global structure blocks +
  atom/bond histograms;
* topology features (`zinc_topology_features.py`) — cycle spectrum, longest
  simple cycle, MCB stats, cycle rank, hinge basis `ReLU(L−t)` for `t=3..10`
  (a generic integer ladder, **not** the ZINC target's `max(0, L−6)`);
* no `logP` / `penalized_logp` / synthetic-accessibility / `z_SA` term enters any
  input; the ZINC regression target is used only as a label;
  feature modules are marked `label_free: True`.

The MolHIV cache itself is built by `molhiv_patch_path_pooling` with the same
`label_free` input convention and its own (MolHIV) graph features.  So the port
tests the *architecture*, not a smuggled ZINC feature.

## Verdict

* The H96 recurrent pair--centre **transfers and trains stably** on MolHIV: it
  is competitive with the prior patch-path pooling on validation (better) and on
  test (slightly worse), and above GIN.
* It does **not** reproduce a transfer advantage: higher validation AUC does not
  convert into a higher official-test AUC, and the strict CIN/GPS references
  remain above it.
* Combined with Workstream A (no independent capacity axis; the smallest ZINC
  cell is best), the evidence points to **ZINC-specific allocation rather than a
  transferable molecular inductive-bias advantage**: spending parameters on the
  recurrent pair--centre does not buy a generalisation gain on a second,
  scaffold-split molecular benchmark once a fair selection protocol is used.
* Caveats: only 2 seeds; small valid split makes selection noisy; different
  training protocol from the published references.

Stop rule honoured: 2 seeds only, no MolHIV architecture search, no class
weighting, no test access before freeze.

## Files

- `experiments/luyin16/molhiv_recurrent_pair_centre.py`
- `tests/ksvd/tests/test_molhiv_recurrent_pair_centre.py` (7 pass)
- `results/molhiv_recurrent_pair_centre/`: `data_sanity.json`, `smoke.json`,
  `run_seed{0,1}.json`, `raw_state_seed*.pt`, `soup_state_seed*.pt`,
  `architecture_freeze.json`, `official_test_unlock.json`,
  `official_test_results.json`, `curves/`, `repro/`, `snapshots/`.
