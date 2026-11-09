# LCRC-BOC v1 — report

Round: `lcrc-pcba-boc-v1`  
Question: does a one-shot, learnable joint composition of bond-centred local
instances and ring-system instances, injected into the W2 **atom main
representation**, improve official `ogbg-molpcba` scaffold-split AP?  
Test role: **`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`**.  Test is not a blind
generalisation estimate and was not used for architecture or checkpoint
selection.

## Bottom line

**No evidence** by the pre-registered bands.  The primary arm, `BOC-FULL`
best-Validation, scores **0.254142** official Test AP.  That is

* **+0.000526** above W2 best-Valid (0.253617),
* **−0.001799** below RC-REAL best-Valid (0.255941),
* below the `0.258` no-evidence ceiling,
* and **−0.015807** below its own Validation AP (0.269950).

`BOC-BOND` (ring content exactly zero) scores 0.251666, i.e. **−0.001951** vs
W2.  The ring channel therefore adds +0.002476 Test AP over the separately
trained BOND arm at best-Valid, but this does not lift the joint model above W2
or RC-REAL, and it does not clear the engineering band.  At the fixed epoch-40
endpoint both arms are also below W2@40 (0.250986): FULL 0.250658, BOND 0.249054.
The two arms exhausted the round's budget; no further arms were run.

## Primary comparison

Official evaluator: `ogb.graphproppred.Evaluator("ogbg-molpcba")`.  The
independent per-task AP recomputation (sklearn per task, NaN masking, same
scored-task set) agrees to <1e-12; this is an implementation cross-check, not a
different metric.

| model | epoch | params | best Valid AP | Test AP (official) | Valid→Test | GPU h |
|---|---:|---:|---:|---:|---:|---:|
| W2 best-Valid | 24 | 994,324 | 0.265273 | 0.253617 | −0.011656 | 1.3949 |
| W2 @40 | 40 | 994,324 | 0.261867 | 0.250985 | −0.010882 | 1.3949 |
| RC-REAL best-Valid | 33 | 1,051,022 | 0.267718 | 0.255941 | −0.011777 | 1.9186 |
| RC-REAL @40 | 40 | 1,051,022 | 0.266950 | 0.256166 | −0.010784 | 1.9186 |
| **BOC-FULL best-Valid** | **29** | **1,352,596** | **0.269950** | **0.254142** | **−0.015807** | **2.4215** |
| BOC-FULL @40 | 40 | 1,352,596 | 0.264009 | 0.250658 | −0.013351 | 2.4215 |
| BOC-BOND best-Valid | 30 | 1,352,596 | 0.269908 | 0.251666 | −0.018241 | 2.3242 |
| BOC-BOND @40 | 40 | 1,352,596 | 0.267367 | 0.249054 | −0.018309 | 2.3242 |

Decision bands (engineering): ≥0.270 strong, 0.265–0.270 meaningful,
0.258–0.265 small, ≤0.258 no evidence.  **Primary band: `no_evidence`.**

Relative deltas (official Test AP):

| comparison | Δ Test | tasks improved / degraded (of 127 scored) | Δ Valid |
|---|---:|---:|---:|
| FULL best − W2 best | **+0.000526** | 69 / 58 | +0.004676 |
| FULL best − RC-REAL best | **−0.001799** | 63 / 64 | +0.002233 |
| BOND best − W2 best | −0.001951 | 71 / 56 | +0.004633 |
| FULL @40 − W2 best | −0.002959 | 67 / 60 | −0.001265 |
| FULL best − BOND best (ring effect) | **+0.002476** | — | +0.000044 |

The Validation ranking is misleading: FULL and BOND look clearly better than W2
on Validation (+0.0047 each), but the Test gain is essentially flat for FULL and
negative for BOND.  The composition increases the Valid→Test gap from ~0.0117
(W2/RC) to 0.0158–0.0183.  That is the main empirical finding: the extra
structural channels fit Validation better, but the improvement does not
transfer through the scaffold split.

## Per-task behaviour

The mean is not a uniform shift.  Of 127 scorable Test tasks, FULL beats W2 on
69 and loses on 58; the per-task distribution has large compensating tails.

Largest FULL−W2 gains (official Test AP):

| task | Test positives | FULL | W2 | Δ |
|---:|---:|---:|---:|---:|
| 77 | 10 | 0.2306 | 0.0553 | **+0.1752** |
| 27 | 16 | 0.2074 | 0.0719 | +0.1355 |
| 54 | 11 | 0.1070 | 0.0240 | +0.0830 |
| 109 | 78 | 0.2147 | 0.1405 | +0.0741 |
| 62 | 11 | 0.3232 | 0.2491 | +0.0741 |
| 17 | 21 | 0.1334 | 0.0666 | +0.0668 |

Largest FULL−W2 losses:

| task | Test positives | FULL | W2 | Δ |
|---:|---:|---:|---:|---:|
| 31 | 6 | 0.0316 | 0.3530 | **−0.3214** |
| 57 | 4 | 0.0141 | 0.1704 | −0.1562 |
| 108 | 7 | 0.1713 | 0.2913 | −0.1199 |
| 2 | 19 | 0.1377 | 0.2464 | −0.1087 |
| 32 | 35 | 0.0252 | 0.0883 | −0.0631 |
| 34 | 22 | 0.0200 | 0.0781 | −0.0581 |

Support strata (Test positives per task, FULL vs W2):

| stratum | n tasks | mean Δ | median Δ | improved |
|---|---:|---:|---:|---:|
| ≥20 positives | 108 | +0.003210 | +0.003872 | 61 |
| 5–19 positives | 18 | −0.006873 | −0.000115 | 8 |
| <5 positives | 1 | −0.156241 | −0.156241 | 0 |

Against RC-REAL the high-support stratum is also slightly positive (+0.002284,
55/108 improved), but the low/medium strata are negative; RC wins overall.
BOND shows the same pattern but a negative overall mean.  The low-support tasks
dominate the mean and are exactly where a single seed is least stable; the
per-task CSV is included so this is not hidden by the aggregate.

## Channel diagnostics

FULL ring content is active throughout training (diag batch `ring_nonzero =
32,640`, `r` RMS ≈ 0.845 at epoch 40; ring gradient L2 is non-zero at every
epoch, last-5 mean 0.0187).  Bond-channel gradient L2 last-5 mean is 0.0362;
fusion 0.1497; W2 trunk 0.2446.  BOND has exactly zero ring tensors, zero ring
gradients and zero ring-module calls, while its bond/fusion/W2 gradients remain
non-zero.  Both arms use 1,352,596 parameters; the BOND ring modules are
instantiated but bypassed, so this is not a parameter-count comparison.

Valid-only zeroing sensitivity (official Validation AP, best-Valid checkpoint):

| arm | both channels | ring zeroed | bond zeroed | both zeroed |
|---|---:|---:|---:|---:|
| FULL | 0.269952 | 0.217645 | 0.202970 | 0.157268 |
| BOND | 0.269908 | 0.269905 | 0.169700 | 0.169699 |

Zeroing the ring channel of the *trained FULL* model collapses Validation AP by
0.0523; zeroing the bond channel collapses it by 0.0670.  Both channels are
used.  A model trained from the start with ring zeros (BOND) compensates through
the bond channel and reaches the same Validation level, but both learned
solutions lose on Test relative to W2.  This rules out "the ring channel was
inactive" and also shows that the joint representation is not a free
improvement: the network can substitute one channel for the other on
Validation, yet neither substitution transfers better.

## Compute and cost

Formal runs (same commit, same frozen order plan, `res` A100-2, GPU1):

| arm | wall GPU h | epoch mean s | step p50 (formal) | peak reserved | clip events |
|---|---:|---:|---:|---:|---:|
| FULL | 2.4215 | 217.15 | ~73 ms | 629,145,600 B (600 MiB) | 14 |
| BOND | 2.3242 | 208.38 | ~73 ms | 627,048,448 B (598 MiB) | 14 |

Pre-run real-batch profile projected 1.81 h (FULL) and 1.45 h (BOND) for 40
epochs including Validation; actual was higher because the box was shared and
the measured step time rose from ~59 ms to ~73 ms.  Both arms stayed below the
3 h target and far below the 4 h stop.  Inference on the 43,793 Test rows:
FULL 0.140 ms/row (6.14 s), BOND 0.130 ms/row, W2 0.0935 ms/row, RC-REAL
0.108 ms/row; FULL is ~1.50× W2 and ~1.30× RC wall time for the same batch size.
The extra cost is small in absolute terms but the model is not accuracy-neutral:
it buys Validation AP, not Test AP.

Structural cache: 12,309,186 bond objects, 34,369,144 neighbour entries,
34,610,219 pair entries, 1,100,845 ring systems, 7,343,607 ring memberships,
23,424,393 intra-system pairs, 2,386,191 external substituents; 29 packed
arrays, 1,645,218,822 bytes, remote build 383.6 s (local 219.9 s).  No
truncation or sampling of ring systems; all training batches are vectorised
gather/scatter.

## Audits

Both pre-training audits passed and bind the same cache hash
`29a5051d…`:

* CPU structural audit: **16/16**.  Full-cache pointer/count invariants,
  independent Python re-enumeration of 2,000 spread rows, independent
  bridge-by-removal, exact ring BFS, external-substituent geometry, label-free
  proof.
* Small-batch GPU audit: **11/11**.  W2 bitwise seed-0 initialisation, 1,352,596
  params in band, centre-swap invariance, same-side order invariance, atom
  renumbering, two-molecule isolation, `FULL(ring_zero=True) == BOND` within
  2.9e-6, exact BOND ring zeros and zero ring calls, finite gradients and
  non-zero ring gradient in FULL.  Peak 85,983,232 B for the 16-molecule audit.
* Unit tests: 19 pass (structure + model), including hand-built chain, triangle,
  four-cycle with chord, benzene ortho/meta/para, fused naphthalene,
  bridge-connected ring systems, brute-force pair enumeration, cache roundtrip.
* Formal `boc_train.py` refuses to start without both audit JSONs and matching
  cache hash.

## What this does and does not show

* The one-shot bond object is a real, learnable structural channel and can fit
  Validation better than W2 (+0.0047).  But on the scaffold Test split that gain
  disappears for FULL and reverses for BOND.  The added representation is not
  transferring, despite the ring channel being active and necessary for the
  trained FULL model.
* The ring-system content adds +0.0025 Test over the separately trained BOND
  arm, but both are below W2 and RC-REAL.  It is not enough to call a meaningful
  improvement, and the absolute score is below the 0.258 no-evidence ceiling.
* The Test split is already exposed and this is a single seed.  The per-task
  tails (task 31 −0.321, task 77 +0.175) show that the aggregate is not a stable
  task-level story.  These results are descriptive engineering evidence, not a
  significance test, and do not justify a new architecture search.

## Pre-formal corrections and data gate

Two implementation bugs were found by the invariance tests and fixed **before**
any formal run: the batch pair-index offset (must use the bond's neighbour-block
start, not the molecule's start), and the reverse bond orientation's side counts
(`nL/nR`, `nLL/nRR` must swap for `z_vu`).  Both are covered by tests; neither
changes the frozen architecture or recipe.  The RC-REAL checkpoint hashes were
corrected to the actual run manifest values before the freeze.  The official
MolPCBA `graph-label.csv.gz` was absent on the compute host; it was vendored
(5.3 MB) and verified byte-for-byte (`ae9430cf…`) so that the inherited
`cc_data.read_official_test_labels` gate could run **after** the prediction
freeze.  The gate verified raw Train/Valid alignment and the frozen Test row
order before scoring.

## Artifacts

* `analysis/selected_checkpoint.json` — Validation-only selection, `test_read=false`.
* `analysis/test_prediction_manifest.json` — freeze seal
  `42e9a26c604de84d…`, `test_labels_read=false`, all checkpoint/logit hashes.
* `analysis/test_scores.json` — post-freeze official evaluator and per-task AP.
* `analysis/per_task_valid_test.csv` — per-task Test AP and deltas.
* `analysis/zeroing_sensitivity.json` — Valid-only channel zeroing.
* `analysis/profile.json` — real-batch profiling and projections.
* `analysis/structure_audit.json`, `analysis/model_audit.json` — 16/16 and 11/11.
* `analysis/comparison.json`, `analysis/report_tables.md` — generated tables.
* `runs/FULL`, `runs/BOND` — checkpoints, curves, RNG, training order (gitignored).
* `runs/freeze/test_predictions` — frozen Test logits (gitignored; hashes in the
  manifest).

Reproduce with `README.md`; the training recipe, order plan and Test policy are
in `PROTOCOL.md`.
