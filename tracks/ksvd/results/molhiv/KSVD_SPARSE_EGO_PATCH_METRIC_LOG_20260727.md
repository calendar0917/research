# KSVD MolHIV strict-scaffold continuation log — 2026-07-27

## Protocol

- Dataset subset: first 8,000 MolHIV graphs.
- Development only: official-train, 3-way Bemis–Murcko scaffold folds.
- This log reports fold 0, model seed 0 pilots only unless stated otherwise.
- Dictionary fitting uses only fold-specific inner-train graphs (`n=4108`).
- Official-valid evaluations: 0 for every run below.
- Official-test evaluations: 0 for every run below.
- Matched backbone initialization hash: `4a9fe0fa07dd73f1335f7b6e5658d61a1f429f2eeeb84b134796fa8e10044e70`.

Reference fold-0 AUC:

| model | AUC |
|---|---:|
| matched GINE | 0.752248 |
| historical raw-KSVD P0 motif slot | 0.759018 |
| historical raw-KSVD top-1 radius-2 ego | 0.760338 |

## 1. Full T=3 sparse ego occurrences

Each nonzero `(center, dictionary atom)` OMP coefficient becomes a typed radius-2 ego occurrence. Occurrence-to-node broadcast is coefficient-magnitude weighted and receiver normalized. It uses no ring/scaffold feature and adds no parameters relative to top-1 ego.

| configuration | AUC | delta vs GINE | result |
|---|---:|---:|---|
| scalar gate, scale=.25, warmup=10 | 0.748974 | -0.003274 | stop |
| zero-out output, scale=.05, warmup=10 | 0.752898 | +0.000650 | stop |

The scalar gate ended near `-1.3e-5`, so the full branch remained effectively dormant. Actively training it with a zero-output initialization recovered the GINE level but did not approach P0/top-1. Retaining all weak OMP atoms appears to dilute the useful dominant occurrence rather than recover missing semantics.

## 2. Train-only linear patch metrics

`build_molhiv_node_tokens.py` now supports `--patch-metric {raw,pca,pca_whiten}`. PCA mean/basis/scale are fit only on the fold-specific dictionary patch reservoir and frozen before encoding held-out graphs.

PCA64 retained 81.09% of centered train-patch variance.

| P0 dictionary input | AUC | delta vs GINE | delta vs raw P0 |
|---|---:|---:|---:|
| PCA-whitened 64-d | 0.746574 | -0.005674 | -0.012444 |
| PCA-projected 64-d | 0.754682 | +0.002434 | -0.004336 |

Whitening strongly harms the metric. PCA denoising preserves a small positive signal but is weaker than raw KSVD. Linear latent geometry is therefore not the current breakthrough.

## 3. No-ring complementary patch views

Audit of the historical raw 848×32 dictionary atom energy:

| block | dictionary energy |
|---|---:|
| unlabeled topology/WL block | 56.44% |
| labeled base chemistry/WL block | 31.51% |
| explicit ring statistics (13-d) | 1.07% |
| center atom one-hot | 7.47% |
| shell atom histograms | 1.59% |
| shell bond histograms | 1.61% |
| rooted shell counts | 0.31% |

Important conceptual audit: the historical 848-d patch vector does contain 13 manually computed cycle/aromatic/ring statistics, although their dictionary energy is only about 1.1%.

Two no-ring views were tested:

1. `rooted_topology`: unlabeled topology/WL plus center-relative shell node/edge counts (245-d), with all atom/bond labels and explicit ring statistics removed.
2. `center_residual_no_ring`: remove explicit ring statistics and center one-hot; condition on the exact center OGB atom tuple using train-only group means; KSVD models the normalized context residual (661-d).

| P0 patch view | AUC | delta vs GINE | delta vs raw P0 |
|---|---:|---:|---:|
| rooted topology | 0.753779 | +0.001531 | -0.005239 |
| center-conditioned no-ring residual | 0.745257 | -0.006991 | -0.013761 |

The full chemistry context is not merely redundant noise: removing it or residualizing it aggressively reduces performance. Conversely, the historical gain cannot yet be claimed to arise from a purely self-learned topology dictionary because the clean no-ring variants did not match raw P0.

## Decision

Do not expand any configuration in this log to fold 1/2. None exceeded raw P0, and none reached the predeclared fold-0 promotion target (`>=0.762`).

The next credible route should not be another hand-selected patch block, gate, or occurrence topology. It should replace the fixed handcrafted 848-d metric with a train-only nonlinear, label-free local encoder, followed by a genuine KSVD bottleneck and matched PCA/random controls. A suitable first version is masked local chemistry/context reconstruction with a 32–64-d latent, frozen before KSVD fitting. This directly tests whether semantic patch geometry—not downstream capacity—is the missing mechanism.

## 4. No-ring nonlinear metric and direct no-ring controls

The first no-ring feature-wise masked AE did not truly activate under the scalar-gated P0 model: its raw motif gate ended at `1.03e-4` (effective scale about `2.6e-5`). Forcing the branch to train with zero-output initialization was strongly negative.

| configuration | fold-0 AUC | delta vs GINE | delta vs full raw P0 |
|---|---:|---:|---:|
| no-ring AE64, scalar gate | 0.754568 | +0.002320 | -0.004450 |
| no-ring AE64, zero-output active branch | 0.739961 | -0.012287 | -0.019056 |

A patch audit found only `7.70%` of no-ring coordinates are nonzero. Independent 15% coordinate masking therefore hides only about 1.15% of all coordinates as nonzero signal and remains close to an identity autoencoder. A semantic-block masked AE was implemented, masking one complete block per sample among topology/WL, labeled chemistry/WL, center atom, shell atom, shell bond, and rooted counts. It still failed:

| configuration | fold-0 AUC | delta vs GINE |
|---|---:|---:|
| no-ring semantic block-AE64 + KSVD P0 | 0.743225 | -0.009023 |

The direct raw no-ring cache, which had previously been omitted, was then tested with matched dictionary families:

| no-ring raw P0 family | fold-0 AUC | delta vs GINE |
|---|---:|---:|
| KSVD | 0.755950 | +0.003702 |
| PCA | 0.750531 | -0.001717 |
| random patch | 0.759327 | +0.007079 |

Thus explicit ring statistics are not required for a positive P0-vs-GINE fold-0 signal, but the clean no-ring gain is not KSVD-specific because random patches outperform KSVD on this fold. No-ring top-1 radius-2 ego reached `0.755079`, also below raw P0.

## 5. Confidence-weighted ego and direct KSVD edge lifting

Two parameter-efficient KSVD-native topology variants were tested:

1. coefficient-dominance weighting of overlapping top-1 radius-2 motif egos;
2. direct GINE edge augmentation connecting within-graph nodes with the same top-1 KSVD atom, using a learned dictionary-atom edge embedding (39,430 trainable parameters).

| configuration | patch input | fold-0 AUC | result |
|---|---|---:|---|
| confidence-weighted top-1 ego | historical full | 0.751959 | stop |
| shared-atom augmented edges | raw no-ring | 0.667596 | stop |

The direct edge lift severely oversmooths/short-circuits molecular structure, while coefficient dominance removes useful averaging rather than filtering noise. Neither should be expanded or tuned further.

All runs above retained `official_valid_evaluations=0`, `official_test_evaluations=0`, and matched base initialization hash `4a9fe0fa07dd...`.

## Updated decision

Do not expand AE, semantic-block AE, no-ring P0, confidence weighting, or direct clique edge lifting to folds 1/2. The only remaining route with a materially different hypothesis is broader **external unlabeled dictionary coverage** (excluding every development graph and all official-valid/test graphs), followed by the already-established P0 controls. This targets unseen-scaffold coverage rather than adding another downstream fusion or hand-selected patch statistic.

## 6. Development-disjoint external official-train coverage

A strictly label-free external reservoir was added to `build_molhiv_node_tokens.py` via
`--external-official-train-patches`.  The pool is drawn only from full OGB official-train
molecules absent from the complete current `n=8000` development subset.  Each selected
external graph contributes one uniformly random center patch.  Selection does not access
HIV labels.

The fold-0 6k audit was:

- full official-train: 32,901 graphs;
- current development graphs in official-train: 6,400;
- eligible external official-train candidates: 26,501;
- selected external graphs/patches: 6,000 / 6,000;
- overlap with the complete current development subset: 0;
- overlap with official-valid: 0;
- overlap with official-test: 0;
- labels used: false.

The dictionary pool combined 6,000 fold-inner patches with 6,000 external patches.  All
families used the same 12,000-patch no-ring raw pool.

| configuration | fold | AUC | comparison |
|---|---:|---:|---|
| KSVD, external 6k | 0 | 0.758110 | +0.005862 vs GINE; +0.002160 vs no-ring KSVD |
| PCA, external 6k | 0 | 0.750907 | -0.001341 vs GINE |
| random patch, external 6k | 0 | 0.754718 | +0.002470 vs GINE |
| KSVD, external 6k | 1 | 0.622063 | -0.011473 vs GINE; stop |

Fold 0 is KSVD-specific under the matched cache (`KSVD > random > PCA`), but misses the
historical raw-P0 promotion line by 0.000908 and does not generalize to fold 1.  Its learned
motif-slot scalar gate is also effectively dormant (`-2.4e-6` on fold 0), so the fold-0
increase cannot be interpreted as a robust active external-coverage mechanism.

### External dosage check

To test whether the 1:1 external mixture diluted the fold-local dictionary, the external
pool was reduced from 6,000 to 1,500 while retaining 6,000 inner patches.

| configuration | fold | AUC | comparison |
|---|---:|---:|---|
| KSVD, external 1.5k | 1 | 0.642513 | +0.008977 vs GINE; avoids 6k collapse |
| KSVD, external 1.5k | 0 | 0.752171 | -0.000077 vs GINE; stop |

The smaller dose recovers fold 1 but removes the fold-0 gain.  This is split-dependent
trade-off rather than a stable coverage benefit, so no external-mixture sweep or fold-2 run
is justified.

### Core dictionary plus external residual atoms

A capacity-preserving variant was implemented via `--ksvd-external-residual-atoms`:
fit the original 32-atom core exclusively on the fold-inner 6k patches, reconstruct the
external patches with that core, and append eight KSVD atoms learned from normalized
external residuals.  The first 32 dictionary columns are bit-exact to the historical no-ring
fold-0 KSVD dictionary (`max_abs_diff=0`).  Total code dimension is 40 and the neural
parameter increase is only 128 parameters relative to D32 P0.

| configuration | fold | AUC | comparison |
|---|---:|---:|---|
| inner core32 + external residual8 | 0 | 0.745700 | -0.006548 vs GINE; stop |

Thus external atoms alter OMP support in a harmful way even when fold-local core capacity is
fully preserved.  Do not tune the residual atom count.

All runs in this section retained `official_valid_evaluations=0` and
`official_test_evaluations=0`.  The builder and runner pass `py_compile`.

## Updated decision after external coverage

Random external coverage, reduced-dose coverage, and core-plus-residual external atoms have
now tested the remaining coverage hypothesis.  The signal is not cross-fold stable, and the
active motif gates remain near zero.  Do not expand these variants to fold 2 or official
valid/test.  The evidence now points away from missing dictionary sample coverage and toward
the more fundamental issue that current P0 gains are dominated by optimization-trajectory
regularization rather than an actively used KSVD transport branch.

## 7. Fold-local masked-context SSL dictionary geometry

To move beyond the fixed handcrafted patch metric without introducing explicit ring or
scaffold features, a label-free fold-local encoder was added in
`build_molhiv_ssl_node_tokens.py`:

```text
raw atom/bond graph
-> 2-layer masked-context GINE trained only on the inner-train fold
-> frozen 64-d node context latent
-> KSVD / PCA / random-patch dictionary
-> T=3 sparse assignment to D=32 atoms
```

HIV labels are physically removed from the PyG objects before SSL.  Each cached node latent
is computed while that node's own atom features are masked.  The SSL encoder and all three
dictionary families are fit only on the corresponding fold-inner training graphs; the held-out
scaffold fold is encoding-only.  Official-valid and official-test graphs are never encoded or
sparse-coded.

### Representation-level dictionary diagnostic

The fold-0 held-out sparse reconstruction diagnostic gave:

| family | held-out reconstruction MSE | effective top-1 atoms | maximum top-1 share |
|---|---:|---:|---:|
| KSVD | 0.062546 | 23.91 | 0.1087 |
| PCA | 0.184518 | ~3.27 | ~0.616 |
| random patch | 0.109836 | 24.28 | 0.1026 |

KSVD had lower held-out MSE than PCA on `98.4573%` of nodes and lower MSE than random
patches on `73.8665%` of nodes.  Its mean MSE advantages were `0.121972` versus PCA and
`0.047291` versus random.  Fit-to-held-out top-1 assignment drift was also small
(`JS=0.001033` for KSVD, `0.000802` for random, `0.000275` for PCA).  Thus KSVD is genuinely
learning the best reconstructive dictionary and does not collapse on the held-out scaffold;
the bottleneck is downstream transport rather than dictionary fitting.

### Downstream transports that did not work

Matched fold-0 results were:

| transport | fold-0 AUC | conclusion |
|---|---:|---|
| original SSL KSVD motif slot | 0.757079 | positive vs GINE, but matched random is 0.757002 |
| fixed dictionary-SVD motif ID slot | 0.750495 | stop |
| reconstructed latent `X @ D.T` injection | 0.754578 | token gate stays near `1e-6`; stop |
| quotient motif-slot transition graph | 0.753686 | stop |
| learned rank-8 transition readout | 0.748654 | stop |
| full `Q^T A Q` transition matrix, KSVD | 0.742632 | stop |
| full `Q^T A Q` transition matrix, PCA | 0.749783 | stop |
| full `Q^T A Q` transition matrix, random | 0.738136 | stop |

A smaller D16/T2 SSL KSVD dictionary reached only `0.748685`, so reduced dictionary capacity
was also stopped.  These failures show that reconstruction quality by itself does not imply a
useful molecular task feature, and that high-capacity learned transports overfit this
small/imbalanced development setting.

## 8. Eight-parameter fixed geometry-transition readout

The most parameter-efficient transport tested was
`motif_geometry_transition_readout`:

```text
q_v = abs(x_v) / ||x_v||_1
fixed atom geometry = rank-8 SVD coordinates of dictionary columns
f_v = q_v @ fixed atom geometry
edge feature = f_u * f_v              # element-wise product
molecule feature = mean over edges
logit residual = <w, molecule feature>, w in R^8
```

Only the final rank-8 task vector is trainable.  Sparse assignment and dictionary geometry
are both required by the branch, but there is no additional GINE, MLP, explicit ring feature,
or scaffold feature.

Parameter counts:

```text
GINE base                         37,382
geometry-transition total        37,390
additional trainable parameters       8
historical raw KSVD P0 total      42,359
```

### Three strict scaffold folds, seed 0

| method | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| GINE | 0.752248 | 0.633536 | 0.781569 | 0.722451 |
| historical raw KSVD P0 | 0.759018 | 0.643425 | 0.781117 | **0.727853** |
| SSL geometry KSVD | **0.762281** | 0.645851 | 0.769396 | 0.725843 |
| SSL geometry PCA | 0.739776 | **0.654078** | **0.776872** | 0.723575 |
| SSL geometry random patch | 0.760281 | 0.653314 | 0.769173 | 0.727589 |

Paired KSVD geometry comparisons:

| comparison | fold deltas | mean delta | wins |
|---|---|---:|---:|
| vs GINE | +0.010033, +0.012315, -0.012173 | **+0.003392** | 2/3 |
| vs historical raw KSVD P0 | +0.003264, +0.002426, -0.011720 | -0.002010 | 2/3 |
| vs PCA geometry | +0.022506, -0.008227, -0.007476 | +0.002268 | 1/3 |
| vs random geometry | +0.002000, -0.007463, +0.000223 | -0.001746 | 2/3 |

The architecture passes the fold-0 promotion line and improves GINE on folds 0 and 1, but
fold 2 reverses the gain.  More importantly, random-patch geometry has the best mean among
the three SSL dictionary controls and nearly matches historical raw KSVD P0.  Therefore:

1. the eight-parameter edge-geometry statistic is a credible generic regularizer/feature;
2. the current three-fold evidence is **not KSVD-specific** despite KSVD's clearly superior
   reconstruction geometry;
3. this route does not beat the historical raw KSVD P0 mean and is not ready for
   official-valid or official-test evaluation;
4. a seed sweep is not justified yet: it would estimate variance around a configuration
   whose central matched-control claim has already failed.

Exact aggregate artifact:

```text
results/molhiv/sslctx64_geometry_transition_readout_r8_3fold_summary.json
```

All SSL caches and all nine geometry-transition runs retain:

```text
labels_used                 false
explicit_ring_features      false
official_valid_evaluations  0
official_test_evaluations   0
```

## Updated decision after SSL geometry transitions

Do not run official-valid/test and do not spend a five-seed budget on the current rank-8
geometry transition.  The strongest robust development result remains historical raw KSVD
P0 (`0.727853` mean), while the new eight-parameter route is scientifically useful mainly as
a diagnosis: the learned KSVD dictionary has superior representation-level reconstruction,
but the present supervised edge readouts do not convert that advantage into a stable
KSVD-specific MolHIV gain.

## 9. Signed and exact-SVD geometry corrections

The initial geometry readout used absolute sparse assignments and row-normalized dictionary
atom SVD coordinates.  Two mathematically motivated corrections were implemented as explicit
runner options:

```text
--motif-geometry-assignment {abs,signed}
--motif-geometry-atom-normalization {row,none}
```

`assignment=signed` preserves OMP coefficient signs.  `atom-normalization=none` keeps
`U_r S_r` unchanged, so signed coefficient aggregation is the truncated-SVD coordinate of
the reconstructed latent rather than a normalized atom-ID embedding.  Defaults remain
`abs,row`, preserving the original experiments exactly.

### Signed exact reconstruction geometry

Fold-0 matched results (`signed,none`, rank 8, eight trainable parameters):

| family | fold-0 AUC |
|---|---:|
| KSVD | 0.748701 |
| PCA | 0.748974 |
| random patch | 0.744298 |

Preserving reconstruction direction is harmful and falls below GINE.  The likely reason is
that edgewise element products of signed latent coordinates cancel across atoms/edges and
produce a difficult low-signal graph statistic.  This variant was stopped after fold 0.

### Absolute assignment with exact SVD atom energy

Keeping the robust absolute assignment while removing row normalization gave a promising
fold-0 separation:

| family | fold 0 | fold 1 |
|---|---:|---:|
| KSVD | **0.760575** | 0.629143 |
| PCA | 0.746664 | 0.640122 |
| random patch | 0.756012 | **0.640921** |
| GINE reference | 0.752248 | 0.633536 |
| raw KSVD P0 reference | 0.759018 | 0.643425 |

On fold 0, KSVD beats GINE by `+0.008327`, raw P0 by `+0.001557`, random by `+0.004563`,
and PCA by `+0.013911`.  However, fold 1 reverses every relevant claim: KSVD is `-0.004393`
versus GINE, `-0.014282` versus raw P0, `-0.011778` versus random, and `-0.010978` versus
PCA.  Fold 2 was therefore not run under the time-aware stopping rule.

This ablation confirms that dictionary leverage/retained-energy information can generate a
KSVD-specific gain on one scaffold partition, but that information is not stable across
scaffold partitions.  Neither signed reconstruction geometry nor unnormalized SVD geometry
is a viable continuation route.

All runs in this section use `37,390` trainable parameters and retain
`official_valid_evaluations=0`, `official_test_evaluations=0`.  The updated runner passes
`py_compile`.

## 10. Weakly supervised whole-graph hierarchical dictionary control

A graph-level hierarchy was tested to avoid copying each molecule label onto every node patch.
The frozen local sparse-code distribution was pooled into a whole-molecule descriptor, and a
second dictionary was fitted on balanced inner-train molecule descriptors augmented with a
small graph-label block (`alpha=0.3`).  All upper-dictionary families used the same descriptors,
atom count (`D=16`), sparsity (`T=3`), GINE initialization, and `37,399` trainable parameters.

Fold-0 results:

| upper dictionary family | fold-0 AUC | selected epoch |
|---|---:|---:|
| KSVD | 0.742137 | 6 |
| PCA | **0.752727** | 21 |
| random patch | 0.742488 | 16 |

The whole-graph supervision removes the most objectionable form of patch-label replication,
but it does not produce a KSVD-specific predictive gain.  KSVD is below both raw P0
(`0.759018`) and GINE (`0.752248`), while PCA is best.  The route was therefore stopped before
fold 1.

Artifacts:

```text
results/molhiv/discgraph_sslksvd_upperksvd_a03_d16t3_fold0_seed0.json
results/molhiv/discgraph_sslksvd_upperpca_a03_d16t3_fold0_seed0.json
results/molhiv/discgraph_sslksvd_upperrandom_a03_d16t3_fold0_seed0.json
```

All three runs have the same baseline initialization hash
`4a9fe0fa07dd73f1335f7b6e5658d61a1f429f2eeeb84b134796fa8e10044e70`
and retain `official_valid_evaluations=0`, `official_test_evaluations=0`.

## 11. Fixed rich sparse-code graph residuals

The strongest historical standalone KSVD readout retained the full distribution of sparse
coefficients rather than only maximum activation.  This readout was transferred to the GINE
model as a zero-initialized linear graph residual.  Per atom it contains:

```text
mean | max | top-3 mean | std | usage | q75 | q90 |
mean square energy | signed mean | winner frequency
```

No token projection is trained.  With `D=32`, this is a 320-dimensional fixed signature and
321 additional residual parameters.

### Raw D32/T3, fold 0

| family | fold-0 AUC |
|---|---:|
| KSVD | 0.752098 |
| PCA | 0.759858 |
| random patch | **0.775934** |

### SSL-context D32/T3, fold 0

| family | fold-0 AUC |
|---|---:|
| KSVD | 0.753094 |
| PCA | **0.767118** |
| random patch | 0.758146 |

The rich graph signature can improve prediction, but both experiments fail the matched-control
claim: random patch wins in raw space and PCA wins in SSL-context space.  Increasing readout
capacity therefore exposes generic patch-distribution information rather than a benefit from
KSVD dictionary learning.

Artifacts:

```text
results/molhiv/richcode_raw_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/richcode_raw_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/richcode_raw_fold0_randompatch_lr01wd1e3_seed0.json
results/molhiv/richcode_sslctx64_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/richcode_sslctx64_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/richcode_sslctx64_fold0_randompatch_lr01wd1e3_seed0.json
```

## 12. Clean D8/T2 no-ring replication of the historical rich setting

To test whether the D32 signature was unnecessarily large, a strict fold-0 cache was rebuilt
at the historical `D=8, T=2` setting:

```text
results/molhiv/node_tokens_n8000_a8_t2_noring_raw_scaffoldfit_fold0.npz
```

Protocol:

```text
radius                          2
max nodes                       8
patch view                      no_ring
patch metric                    raw
dictionary fit patches          6000
dictionary fit graphs           fold-0 inner train only (4108)
KSVD iterations                 4
families                        KSVD / PCA / random patch
official-valid encoded          false
official-test encoded           false
```

The cache took about 90 seconds to build.

### Ten-statistic rich residual, no reconstruction features

| family | fold-0 AUC |
|---|---:|
| KSVD | 0.746020 |
| PCA | **0.752341** |
| random patch | 0.738961 |

KSVD is `+0.007059` above random patch, which is a small family-specific signal, but it remains
`-0.006228` below GINE and below PCA.  It was not promoted.

### Exact rich residual plus OMP reconstruction distribution

Six unit-patch reconstruction statistics (`mean/std/q50/q75/q90/max`) and molecule node
count / log-count were appended.  This gives 88 fixed graph features and only 89 residual
parameters; total trainable parameters are `37,471`.

| family | fold-0 AUC | selected epoch | elapsed |
|---|---:|---:|---:|
| KSVD | 0.755372 | 23 | 30.31 s |
| PCA | **0.758817** | 23 | 30.68 s |
| random patch | 0.756723 | 23 | 30.30 s |

Reconstruction information raises KSVD by `+0.009352` relative to its no-reconstruction
variant, but it raises the controls as well.  KSVD remains below raw P0 by `-0.003646`, below
PCA by `-0.003444`, and below random patch by `-0.001351`.  Hence the exact historical rich
signature also fails the fold-0 promotion rule and no fold-1 cache will be built for this
variant.

Artifacts:

```text
results/molhiv/richcode_a8t2_noring_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/richcode_a8t2_noring_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/richcode_a8t2_noring_fold0_randompatch_lr01wd1e3_seed0.json
results/molhiv/richrecon_a8t2_noring_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/richrecon_a8t2_noring_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/richrecon_a8t2_noring_fold0_randompatch_lr01wd1e3_seed0.json
```

Every run in sections 10--12 retains the matched baseline initialization hash
`4a9fe0fa07dd73f1335f7b6e5658d61a1f429f2eeeb84b134796fa8e10044e70` and
`official_valid_evaluations=0`, `official_test_evaluations=0`.

## Updated route decision after rich-code controls

The repeated result is now strong enough to stop ordinary dictionary-size, sparse-code
summary, hierarchy, and GINE-fusion sweeps.  Sparse graph statistics can be useful, and KSVD
has the best reconstruction objective, but the predictive gains are repeatedly matched by
PCA or random patches.  The next experiment must change the supervision/representation
alignment rather than add parameters: graph-label weak supervision under a multiple-instance
assumption, where a positive molecule may contain only a few positive witness patches.  The
first time-aware screen will fit a bag-level soft-top-k witness scorer on fixed sparse codes
(and reconstruction novelty), with strict KSVD/PCA/random matched controls and no patch-label
replication.  Only a KSVD-specific fold-0 result will justify a negative-background dictionary
or additional folds.

## 13. Cross-fitted graph-level MIL novelty scorer

To avoid copying each molecule label onto every atom patch, a fixed bag-level scorer was fitted
on sparse-code instances.  Each instance contains normalized absolute assignments, OMP
reconstruction error, and sparse-code energy.  A soft-top/MIL pooling loss is trained only on
inner-train molecule labels.  Inner-train graph features are produced by three-fold
cross-fitting; held-out scaffold molecules use a scorer fitted on the full inner-train fold.
KSVD, PCA, and random-patch caches share the same scorer form and five fixed graph features.

| family | fold-0 AUC | selected epoch |
|---|---:|---:|
| KSVD | 0.744483 | 27 |
| PCA | 0.750918 | 23 |
| random patch | **0.755584** | 23 |

The bag-level formulation removes patch-label replication, but it does not create a
KSVD-specific gain.  KSVD is below both the fold-0 GINE baseline (`0.752248`) and both matched
controls.  The scorer route is stopped before fold 1.

Artifacts:

```text
results/molhiv/milnovelty_a8t2_noring_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/milnovelty_a8t2_noring_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/milnovelty_a8t2_noring_fold0_randompatch_lr01wd1e3_seed0.json
```

## 14. Negative background and positive residual-witness dictionaries

A second weakly supervised dictionary mechanism was tested.  A graph-balanced background
patch pool was drawn only from the `3956` negative inner-train molecules.  For each of the
`152` positive molecules, the two patches with the largest background reconstruction error
were retained, producing `304` positive residual witnesses.  A second `D=8, T=2` novelty
dictionary was fitted to normalized witness residuals.  The final graph signature contains
background error, joint background-plus-novelty error, absolute/relative reconstruction gain,
novelty-code energy, and per-atom novelty assignment statistics.  Official-valid/test graphs
were not vectorized or encoded.

### Family-specific background and novelty dictionaries

| family pipeline | fold-0 AUC | selected epoch |
|---|---:|---:|
| KSVD background + KSVD novelty | 0.746303 | 20 |
| PCA background + PCA novelty | **0.761317** | 25 |
| random background + random novelty | 0.739554 | 30 |

The positive-residual construction has predictive signal, but the signal is strongest for PCA,
not KSVD.  KSVD remains below GINE and historical raw P0.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a8_t2_noring_milnovelty_fold0.npz
results/molhiv/mildict_a8t2_noring_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/mildict_a8t2_noring_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/mildict_a8t2_noring_fold0_randompatch_lr01wd1e3_seed0.json
```

## 15. Corrected shared-PCA background matched control

To isolate the second-stage novelty dictionary, all methods were given the same PCA background
dictionary and therefore the same positive witness residual pool.  The first version also used
family-dependent numeric random seeds (`seed + family_index * 1009`), which was a small but real
matched-control confound.  The builder was corrected so every background family uses
`dict_seed=0` and every novelty family uses `dict_seed+1000003=1000003`.

Corrected cache:

```text
results/molhiv/node_tokens_n8000_a8_t2_noring_milnovelty_sharedpca_matchedseed_fold0.npz
```

The cache took `111.73 s` to build.  All three models have `37,447` trainable parameters
(`37,382` base plus `65` graph-residual parameters) and the same base initialization hash
`4a9fe0fa07dd73f1335f7b6e5658d61a1f429f2eeeb84b134796fa8e10044e70`.

| novelty dictionary family | fold-0 AUC | selected epoch | vs PCA | vs random |
|---|---:|---:|---:|---:|
| KSVD | **0.759136** | 25 | +0.001681 | +0.004697 |
| PCA | 0.757455 | 27 | -- | +0.003016 |
| random patch | 0.754439 | 27 | -0.003016 | -- |

The corrected seed removes the earlier apparent KSVD failure (`0.742349`) and makes KSVD the
best of the three matched novelty dictionaries.  This is a useful positive signal: on an
identical positive residual pool, sparse reconstruction can be slightly more predictive than
PCA or a sampled-patch basis.  It is not yet a promotion result, however:

- KSVD is only `+0.006888` above GINE and essentially tied with historical raw P0
  (`+0.000118`);
- the margin over PCA is only `0.001681`, too small for a single model seed/fold;
- it misses the preregistered fold-0 promotion threshold `0.762` by `0.002864`;
- the large sensitivity to a nominal dictionary seed means this mechanism needs stronger
  stability evidence before spending time on fold 1.

Therefore no fold-1 cache, additional model seeds, official-valid evaluation, or official-test
evaluation is run for this route.  The route is retained as mechanistic evidence, not as the
current candidate model.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a8_t2_noring_milnovelty_sharedpca_matchedseed_fold0.json
results/molhiv/mildict_sharedpca_matchedseed_a8t2_noring_fold0_ksvd_lr01wd1e3_seed0.json
results/molhiv/mildict_sharedpca_matchedseed_a8t2_noring_fold0_pca_lr01wd1e3_seed0.json
results/molhiv/mildict_sharedpca_matchedseed_a8t2_noring_fold0_randompatch_lr01wd1e3_seed0.json
```

Every run in sections 13--15 has `official_valid_evaluations=0` and
`official_test_evaluations=0`.  The caches record `official_valid_encoded=false` and
`official_test_encoded=false`.

## Updated route decision after graph-label weak supervision

The corrected control shows that KSVD is not intrinsically worse on the positive residual pool,
but the effect is too small and seed-sensitive to justify another ordinary readout or dictionary
hyperparameter sweep.  The next experiment must make graph-level discrimination alter the
sparse dictionary itself rather than fit a classifier or second dictionary after a frozen
reconstruction objective.  The lowest-cost next screen is an alternating bag-level procedure:
fit a cross-fitted MIL witness direction, use it to select a fixed graph-balanced set of
high-salience patches without copying graph labels onto every patch, refit the dictionary, and
repeat once.  KSVD/PCA/random must share the initial patch pool, witness budget, numeric seeds,
and downstream readout.  Only a clear KSVD-specific fold-0 gain will justify more alternations
or additional scaffold folds.

## 16. OOF-MIL witness selection followed by dictionary refit

The next route changes the dictionary training distribution rather than adding another GINE
branch.  Starting from the strict fold-specific raw `D=8, T=2`, radius-2, no-ring cache, the
cross-fitted graph-level MIL scorer ranks patches in every inner-train molecule using a model
that did not train on that molecule.  All positive inner-train molecules and a deterministic
equal number of negative molecules are retained, with the top two OOF-ranked patches from each
molecule.  KSVD/PCA/random are then fitted on the same graph-balanced witness pool and the
molecules are re-encoded.  No graph label is copied onto all patches.

The downstream model remains the three-layer matched GINE backbone plus the fixed rich sparse-code
and reconstruction signature.  At `D=8` this adds only `89` trainable graph-residual parameters:
`37,471` parameters in total.  Explicit ring features are absent.

### Fold 0, three model seeds on the KSVD-selected shared witness pool

| refit family | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| KSVD | **0.776362** | 0.736832 | 0.747793 | **0.753662** | 0.020408 |
| PCA | 0.745447 | 0.738812 | 0.753135 | 0.745798 | 0.007168 |
| random patch | 0.745298 | 0.741477 | **0.755012** | 0.747262 | 0.006978 |

Matched GINE scores are `0.752248 / 0.725943 / 0.745102`.  KSVD wins all three paired
model seeds over GINE, with a mean gain of `+0.012565`.  The large seed-0 value is not stable
near `0.78`, but the paired GINE improvement is not confined to seed 0.

Family-specific selector-to-dictionary controls were also run on fold 0:

| full pipeline | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| KSVD selector -> KSVD | **0.776362** | 0.736832 | **0.747793** | **0.753662** |
| PCA selector -> PCA | 0.751691 | 0.733248 | 0.746009 | 0.743650 |
| random selector -> random | 0.736533 | **0.742240** | 0.731738 | 0.736837 |

Thus the KSVD full pipeline has the best fold-0 three-seed mean by `+0.010012` over PCA and
`+0.016825` over random, although it wins only two of three individual seeds against each.

### Fold 1, three model seeds on the KSVD-selected shared witness pool

| refit family | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| KSVD | **0.656143** | 0.654113 | 0.650265 | 0.653507 | 0.002986 |
| PCA | 0.649600 | 0.659758 | **0.653413** | **0.654257** | 0.005131 |
| random patch | 0.635149 | **0.667044** | 0.647181 | 0.649791 | 0.016107 |

Matched GINE is `0.633536 / 0.647712 / 0.642442`; KSVD again wins all three paired seeds and
is extremely stable on this fold.  However, its mean is `0.000750` below PCA, so the final
dictionary family is not KSVD-specific on fold 1.

### Fold 2 seed-0 screen

The initial KSVD MIL scorer is much stronger on fold 2 (`0.769103`), but witness-only refitting
does not preserve a KSVD advantage:

| refit family | fold-2 seed-0 AUC |
|---|---:|
| KSVD | 0.775821 |
| PCA | 0.779166 |
| random patch | **0.780250** |
| matched GINE | 0.781569 |

This is evidence that replacing the whole background dictionary with a small witness dictionary
throws away useful common molecular structure.  The route is not promoted in witness-only form.

Key caches and artifacts:

```text
results/molhiv/node_tokens_n8000_a8_t2_noring_oofmil_witnessrefit_fold{0,1,2}.npz
results/molhiv/witnessrefit_richrecon_a8t2_noring_fold{0,1,2}_{ksvd,pca,randompatch}_seed*.json
results/molhiv/witnessrefit_fullpipeline_a8t2_noring_fold0_{pca,randompatch}_seed*.json
```

## 17. Background plus OOF-witness dual-bank dictionary with joint OMP

The fold-2 failure suggests retaining both distributions.  For each family, the original
train-only background dictionary (`D=8`) is concatenated with its dictionary fitted on the
KSVD-selected OOF witness pool (`D=8`).  Every official-train patch is then re-encoded by a
single joint `T=2` OMP solve over the resulting 16-atom dictionary.  This is not a two-GINE
architecture: both banks are frozen learned patch atoms, and the only predictive addition is
a zero-initialized linear residual over sparse-code/reconstruction summaries.

The mechanism has a direct interpretation:

- the background bank represents common local molecular structure;
- the witness bank represents graph-balanced, task-aligned local patches found by cross-fitted MIL;
- joint OMP decides whether each patch is better explained by common or witness atoms;
- graph-level atom usage and reconstruction distributions expose this allocation to a tiny
  residual head.

The builder now supports `--retain-initial-dictionary`.  Dictionary size rises from 8 to 16,
but dictionaries are frozen.  The model has only `37,551` trainable parameters (`37,382` base
plus `169` residual parameters), only 80 more trainable parameters than the witness-only model.

### Complete strict scaffold result: three folds by three model seeds

| fold | family | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---:|---|---:|---:|---:|---:|---:|
| 0 | KSVD | **0.772439** | 0.750580 | 0.746437 | **0.756485** | 0.013970 |
| 0 | PCA | 0.756651 | **0.751851** | 0.754104 | 0.754202 | 0.002402 |
| 0 | random patch | 0.752784 | 0.735347 | **0.759043** | 0.749058 | 0.012280 |
| 0 | GINE | 0.752248 | 0.725943 | 0.745102 | 0.741097 | 0.013602 |
| 1 | KSVD | 0.657586 | 0.642477 | **0.660692** | 0.653585 | 0.009744 |
| 1 | PCA | **0.661760** | 0.648023 | 0.643114 | 0.650966 | 0.009665 |
| 1 | random patch | 0.653710 | **0.680597** | 0.659949 | **0.664752** | 0.014072 |
| 1 | GINE | 0.633536 | 0.647712 | 0.642442 | 0.641230 | 0.007165 |
| 2 | KSVD | **0.783475** | 0.757963 | **0.794118** | **0.778518** | 0.018580 |
| 2 | PCA | 0.762405 | 0.761405 | 0.775629 | 0.766480 | 0.007940 |
| 2 | random patch | 0.781499 | **0.763463** | 0.773711 | 0.772891 | 0.009046 |
| 2 | GINE | 0.781569 | 0.762290 | 0.765936 | 0.769932 | 0.010242 |

Across the nine matched fold/seed cells:

| method | pooled mean |
|---|---:|
| KSVD dual bank | **0.729530** |
| random-patch dual bank | 0.728900 |
| PCA dual bank | 0.723882 |
| GINE | 0.717420 |

Paired KSVD comparisons:

```text
vs GINE          mean delta +0.012110, wins 7/9, median delta +0.018250
vs PCA           mean delta +0.005647, wins 4/9, median delta -0.001271
vs random patch  mean delta +0.000629, wins 6/9, median delta +0.001976
```

The positive conclusion is stronger than for the witness-only route: KSVD beats GINE on every
fold mean, by `+0.015388`, `+0.012355`, and `+0.008586` on folds 0/1/2 respectively.  It is
also the best method on fold-0 and fold-2 means.  The negative conclusion is equally important:
the pooled margin over random patch is only `0.000629`, and KSVD wins only four of nine cells
against PCA despite the higher mean.  Therefore this is currently a credible **GINE-beating
KSVD-centered candidate**, but not yet clean evidence that every gain comes from KSVD rather
than the task-aligned witness selection and dual-bank capacity.

The correct next control is not a larger model or another dictionary-size sweep.  It is an
end-to-end selector control for the dual-bank route: PCA selector -> PCA background+witness
banks and random selector -> random background+witness banks, initially on one predeclared
fold/seed and then expanded only if competitive.  Official-valid/test remain sealed.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundpluswitness_fold{0,1,2}.npz
results/molhiv/backgroundpluswitness_richrecon_a16t2_noring_fold{0,1,2}_{ksvd,pca,randompatch}_seed{0,1,2}.json
```

Audit over 48 witness-refit and dual-bank model JSONs found no violations.  Within every
fold/seed, the KSVD/PCA/random runs have the same matched base initialization hash.  Every
model has:

```text
official_valid_evaluations = 0
official_test_evaluations  = 0
official_valid_auc         = null
official_test_auc          = null
```

All three dual-bank caches also have zero official-valid/test token rows, no explicit ring
features, and `official_valid_encoded=false`, `official_test_encoded=false`.  Compilation of
the alternating builder, MIL builder, discriminative graph-code builder, and strict inner
runner succeeds.

### Fold-2 end-to-end selector controls for the dual-bank route

To isolate whether the apparent gain comes only from KSVD-selected witnesses, the selector,
background dictionary, witness dictionary, and final sparse encoding were replaced together.
The three pipelines are therefore fully family-specific:

```text
KSVD MIL selector   -> KSVD background + KSVD witness bank   -> joint OMP
PCA MIL selector    -> PCA background + PCA witness bank     -> joint OMP
random MIL selector -> random background + random witness bank -> joint OMP
```

The downstream architecture, model seeds, batches, initialization, `D=8+8`, `T=2`, and
training budget are matched.

The selector-stage fold-2 AUCs were:

| selector family | MIL AUC |
|---|---:|
| KSVD | 0.769103 |
| PCA | **0.777369** |
| random patch | 0.753164 |

Thus KSVD did not enter the final comparison with the best standalone selector AUC.
Nevertheless, the complete KSVD pipeline is best for every matched model seed:

| full pipeline | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| KSVD selector -> KSVD dual bank | **0.783475** | **0.757963** | **0.794118** | **0.778518** | 0.018580 |
| PCA selector -> PCA dual bank | 0.760308 | 0.757096 | 0.780154 | 0.765853 | 0.012489 |
| random selector -> random dual bank | 0.771844 | 0.755108 | 0.791371 | 0.772774 | 0.018149 |

Paired differences are:

```text
KSVD vs PCA     +0.023166 / +0.000867 / +0.013963
                 mean +0.012665, wins 3/3
KSVD vs random  +0.011631 / +0.002855 / +0.002747
                 mean +0.005744, wins 3/3
```

This is substantially cleaner KSVD-specific evidence than the shared-selector control:
KSVD wins all six paired comparisons despite PCA having the strongest standalone MIL scorer.
The result is still one scaffold fold, so it should be treated as a positive replication target,
not as permission to inspect official-valid/test.

Artifacts:

```text
results/molhiv/milnovelty_a8t2_noring_fold2_{pca,randompatch}_lr01wd1e3_seed0.json
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_{pca,random}fullpipeline_backgroundpluswitness_fold2.{npz,json}
results/molhiv/backgroundpluswitness_fullpipeline_a16t2_noring_fold2_{pca,randompatch}_seed{0,1,2}.json
```

Audit found zero violations.  All six new model JSONs have no official-valid/test evaluation,
`37,551` trainable parameters (`37,382 + 169`), and no ring token channels.  For each model
seed, KSVD/PCA/random have identical base and token-projection initialization hashes, loader
seed, and first three minibatches.  Both family-specific caches have exactly zero nonzero token
entries on official-valid and official-test graphs.

The next predeclared stress test is fold 1, the hardest scaffold fold and the fold where the
shared KSVD-selector random dictionary control was strongest.  Only the PCA and random
end-to-end controls are needed; the existing KSVD result is fixed.  No architecture or
hyperparameter change is allowed during this replication.

### Fold-1 predeclared stress replication of end-to-end selector controls

Fold 1 was chosen before running these controls because it is the hardest scaffold fold and the
shared-selector random control was strongest there.  No architecture, dictionary, optimizer,
or epoch-budget setting was changed from the fold-2 end-to-end control.

Selector-stage AUCs:

| selector family | MIL AUC |
|---|---:|
| KSVD | 0.639202 |
| PCA | 0.632673 |
| random patch | **0.646884** |

Complete family-specific dual-bank results:

| full pipeline | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| KSVD selector -> KSVD dual bank | 0.657586 | 0.642477 | **0.660692** | 0.653585 | 0.009744 |
| PCA selector -> PCA dual bank | 0.639004 | 0.646537 | 0.651821 | 0.645788 | 0.006442 |
| random selector -> random dual bank | **0.677640** | **0.649133** | 0.649551 | **0.658775** | 0.016339 |

Paired differences:

```text
KSVD vs PCA     +0.018582 / -0.004060 / +0.008870
                 mean +0.007798, wins 2/3
KSVD vs random  -0.020054 / -0.006656 / +0.011141
                 mean -0.005190, wins 1/3
```

This stress test is mixed rather than confirmatory.  KSVD remains better than PCA on mean and
still exceeds the matched GINE fold mean (`0.641230`) by `+0.012355`, but a fully random-patch
pipeline is stronger on fold 1 by `0.005190`.  Therefore the fold-2 3/3 KSVD-specific result is
real but not universal across scaffold regimes.  The current evidence supports a robust
**KSVD-centered GINE improvement**, not a claim that KSVD dominates every matched dictionary
control.

Artifacts:

```text
results/molhiv/milnovelty_a8t2_noring_fold1_{pca,randompatch}_lr01wd1e3_seed0.json
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_{pca,random}fullpipeline_backgroundpluswitness_fold1.{npz,json}
results/molhiv/backgroundpluswitness_fullpipeline_a16t2_noring_fold1_{pca,randompatch}_seed{0,1,2}.json
```

Audit again found zero violations.  All six model JSONs have sealed official-valid/test,
`37,551` trainable parameters, and no ring channels.  Per seed, all three families have matched
base/token-projection initialization hashes and minibatch order.  Both new caches have exactly
zero official-valid/test token entries.

### Fold-0 completion and pooled end-to-end selector evidence

The same frozen end-to-end control was completed on fold 0 so that KSVD specificity is not
judged from only the favorable fold 2 and unfavorable fold 1.

| full pipeline | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| KSVD selector -> KSVD dual bank | **0.772439** | **0.750580** | 0.746437 | **0.756485** | 0.013970 |
| PCA selector -> PCA dual bank | 0.754120 | 0.739967 | **0.750052** | 0.748046 | 0.007287 |
| random selector -> random dual bank | 0.751949 | 0.737894 | 0.749113 | 0.746319 | 0.007433 |

KSVD wins two of three seeds against each full-pipeline control, with mean gains of `+0.008439`
over PCA and `+0.010167` over random.

Across all nine matched fold/seed cells, now using fully family-specific selectors and dual
banks rather than a shared KSVD selector:

| full pipeline | pooled mean |
|---|---:|
| KSVD | **0.729530** |
| random patch | 0.725956 |
| PCA | 0.719896 |

```text
KSVD vs PCA     mean +0.009634, median +0.010613, wins 7/9
KSVD vs random  mean +0.003574, median +0.002855, wins 6/9
KSVD vs GINE    mean +0.012110, wins 7/9
```

This is the cleanest current conclusion.  The KSVD pipeline is best on pooled mean, wins the
majority of matched cells against both complete controls, and beats GINE on every fold mean.
Fold 1 prevents a claim of universal KSVD dominance, but the result is no longer explainable
only by using a KSVD selector for all dictionary families.

The full 3-fold audit found no errors: official-valid/test AUCs are null, evaluation counters
are zero, caches contain zero official-valid/test token entries, parameter counts are matched,
and per-seed initialization/minibatch fingerprints agree across all families.

Artifacts added:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_{pca,random}fullpipeline_backgroundpluswitness_fold0.{npz,json}
results/molhiv/backgroundpluswitness_fullpipeline_a16t2_noring_fold0_{pca,randompatch}_seed{0,1,2}.json
```

### Fixed-epoch stability audit

Best-epoch AUC is useful for development comparison but cannot be copied directly into a sealed
full-train evaluation protocol.  Epoch-wise aggregation over all nine KSVD fold/seed runs shows
that epoch 29 is the strongest fixed epoch:

```text
KSVD fixed epoch 29 pooled mean = 0.707170
GINE fixed epoch 29 pooled mean = 0.686430
paired delta                    = +0.020740
paired wins                     = 9/9
```

Epoch 29 is therefore the current predeclared full-train epoch candidate.  This result is more
conservative than the per-run selected KSVD mean (`0.729530`) and avoids relying on a noisy
single-fold early-stopping maximum.  No official-valid/test evaluation was performed.

## 18. Class-conditional OOF witness subdictionaries

A targeted Fold-1 screen tested whether the balanced positive/negative witness pool was causing
KSVD to learn only class-common variance.  The witness atom budget remained eight, but was split
into four atoms fitted on negative-bag witnesses and four atoms fitted on positive-bag witnesses.
These were concatenated with the same eight-atom background bank and encoded by the same joint
`T=2` OMP.  Parameter count remained exactly `37,551`; no explicit chemistry/ring feature was
added.

| Fold-1 seed-0 KSVD route | AUC |
|---|---:|
| shared balanced witness dictionary | **0.657586** |
| class-conditional 4-negative + 4-positive witness banks | 0.652515 |

The class-conditional split loses `0.005072` and does not address the Fold-1 random-pipeline
advantage.  Hard-assigning the small witness atom budget by bag label appears to reduce useful
shared statistical strength.  This route is stopped after the predeclared seed-0 screen; no
PCA/random expansion is justified.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_classconditional_backgroundpluswitness_fold1.{npz,json}
results/molhiv/backgroundpluswitness_classconditional_a16t2_noring_fold1_ksvd_seed0.json
```

The artifact audit passes: official-valid/test are unencoded and unevaluated, parameter count is
matched, and the cache has zero official-valid/test token entries.

## 19. Explicit background-to-witness reconstruction-gain residual

The 169-parameter rich readout exposes joint sparse-code statistics and joint reconstruction
error, but not the defining dual-bank quantity: how much the witness bank improves reconstruction
relative to background-only OMP.  A compact 64-feature KSVD-native residual was therefore added:
background error, joint error, absolute/relative gain, witness-code energy, per-witness-atom
usage, and graph size.  Features are computed only for official-train graphs and standardized
only on the current inner-train fold.

This reduces the model from `37,551` to `37,447` trainable parameters (`37,382 + 65`).

Fold-1 three-seed result:

| readout | seed 0 | seed 1 | seed 2 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| atom-wise rich + joint reconstruction | **0.657586** | 0.642477 | **0.660692** | 0.653585 | 0.009744 |
| compact background->witness gain | 0.657006 | **0.665459** | 0.648136 | **0.656867** | 0.008663 |

The gain readout improves the mean by `+0.003282`, lowers variance slightly, and uses 104 fewer
parameters, but it wins only one of three paired seeds.  It also remains `0.001908` below the
fully random-patch pipeline mean on Fold 1.  Under the existing promotion rule (`>= +0.003`
mean, at least `2/3` seed wins, and no variance degradation), it does not advance to Fold 0/2.
It is retained as a parameter-efficient mechanistic alternative, not as the primary candidate.

At fixed epochs it is somewhat more stable than the rich readout on Fold 1:

```text
fixed epoch 29: gain 0.627455 vs rich 0.621476
fixed epoch 30: gain 0.646146 vs rich 0.638370
```

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundpluswitness_gainfeatures_fold1.{npz,json}
results/molhiv/backgroundpluswitness_gainfeatures_a16t2_noring_fold1_ksvd_seed{0,1,2}.json
```

Audit passes with zero official-valid/test token and graph-feature entries, zero evaluations,
64 finite graph features, and the expected `37,447` parameter count.

## 20. Predeclared witness-dictionary initialization robustness audit

Before inspecting any new result, the primary dual-bank rich-reconstruction route is frozen and
only the KSVD witness-dictionary initialization is varied.  The background cache, OOF-MIL
selector, selected witnesses, `D=8+8`, joint `T=2` OMP, matched three-layer GINE, optimizer,
and scaffold folds remain unchanged.  Two additional deterministic dictionary seeds are used:

```text
2000003
3000003
```

The audit is restricted to Fold 1 (the hardest stress fold) and Fold 2 (the strongest KSVD fold).
For each dictionary seed, model seeds 0/1/2 are repeated with exactly 29 epochs; epoch 29 is the
previously selected fixed-epoch protocol and no per-run epoch selection is used for the robustness
conclusion.  No PCA/random rerun is required because this audit asks whether the fixed KSVD result
is unusually dependent on one KSVD initialization, not whether control-family numbers change.
Official-valid and official-test remain prohibited.

Predeclared interpretation:

- primary pass: the mean over the six new Fold/seed cells remains above the matched fixed-epoch
  GINE mean and does not collapse by more than 0.01 relative to the original dictionary seed;
- strong pass: KSVD beats matched fixed-epoch GINE in at least 5/6 new cells and both folds retain
  a positive mean delta;
- if initialization variance is material, the final protocol must fix one dictionary seed before
  any sealed evaluation rather than selecting it using held-out AUC.

### Result of the predeclared dictionary-seed audit

The fixed-epoch-29 results are:

| witness KSVD seed | Fold 1 seed0/1/2 | Fold 1 mean | Fold 2 seed0/1/2 | Fold 2 mean | pooled mean | delta vs GINE | wins |
|---|---|---:|---|---:|---:|---:|---:|
| 1000003 (original) | .612435 / .609733 / .642258 | .621476 | .783475 / .735874 / .754401 | .757916 | .689696 | +.021875 | 6/6 |
| 2000003 | .606670 / .603600 / .639966 | .616746 | .761545 / .720311 / .762042 | .747966 | .682356 | +.014534 | 4/6 |
| 3000003 | .580696 / .623555 / .645130 | .616460 | .756523 / .717851 / .750864 | .741746 | .679103 | +.011282 | 5/6 |

The matched fixed-epoch GINE means are `.602190` on Fold 1, `.733452` on Fold 2, and
`.667821` over the six Fold/model-seed cells.  Both new dictionary initializations retain a
positive mean delta on both folds.  Combined across the 12 new cells:

```text
new dictionary seeds mean          0.680729
matched GINE mean                  0.667821
paired mean delta                 +0.012908
paired wins                        9/12
drop from original dictionary seed 0.008967
```

Thus the combined predeclared primary criterion passes: the gain remains positive and the drop
from the original seed is below `0.01`.  Seed `3000003` satisfies the stated `5/6` strong-win
criterion with positive fold means; seed `2000003` is only `4/6`.  Across all three dictionary
seeds the fixed-epoch result is `0.683718` versus GINE `0.667821`, a `+0.015897` delta with
`15/18` paired wins.  The sample SD of the three six-cell dictionary-seed means is `0.005426`.

Interpretation: witness-dictionary initialization is material, especially on Fold-2 seed0/1,
but it does not explain away the KSVD gain.  The original seed is the strongest and must not be
chosen post hoc for that reason; it remains acceptable only because it was already fixed before
this audit.  A final protocol should either keep that predeclared seed or separately predeclare a
deterministic initialization rule.  It should not train several dictionaries and select the best
using held-out AUC.

Audit passes.  All 12 new models have `37,551 = 37,382 + 169` parameters, null official AUCs,
and zero official-valid/test evaluations.  The four new caches have exactly zero official-valid
and official-test token entries.  Within every fold/model-seed cell, the three dictionary seeds
have identical backbone and token-projection initialization hashes, loader seed, and first three
minibatches.  Selected witness hashes are also unchanged within each fold.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundpluswitness_fold{1,2}_dictseed{2000003,3000003}.{npz,json}
results/molhiv/backgroundpluswitness_richrecon_a16t2_noring_fold{1,2}_dictseed{2000003,3000003}_ksvd_seed{0,1,2}.json
results/molhiv/backgroundpluswitness_dictionary_seed_robustness_fixed_epoch29_summary.json
```

## 21. Predeclared deterministic-SVD warm-started KSVD audit

The dictionary-seed audit shows that the KSVD gain survives initialization changes, but the
six-cell fixed-epoch means still have sample SD `0.005426`.  The next targeted mechanism removes
the arbitrary random-column initialization while keeping the KSVD sparse-coding and atom-update
steps intact.

For the frozen dual-bank route, only the eight-atom witness bank is changed:

```text
witness matrix
-> exact deterministic SVD basis, with canonicalized signs
-> 0 / 1 / 4 KSVD atom-update iterations
-> concatenate the unchanged KSVD background D=8
-> one joint OMP with T=2
-> unchanged rich-reconstruction residual and three-layer GINE
```

The zero-step variant is the exact matched linear-subspace initializer control.  A positive
one/four-step result must therefore come from sparse assignment plus KSVD residual atom updates,
not from merely supplying PCA coordinates.  This is not an explicit ring/scaffold feature and
adds no trainable model parameter.

To control runtime, the predeclared screen is Fold 1 and Fold 2 with model seed 0 at fixed epoch
29.  The stronger of one/four update steps advances to model seeds 1/2 only if it satisfies both:

```text
mean over Fold1/Fold2 >= zero-step mean + 0.003
no individual fold is worse than zero-step by more than 0.010
```

A tie within `0.002` is resolved in favor of one update step.  The advanced configuration is
then judged over all six Fold/model-seed cells against the zero-step SVD control, the previously
fixed random-initialized KSVD seed `1000003`, and matched GINE.  No official-valid/test encoding
or evaluation is permitted.

### Deterministic-SVD seed-0 screen result and promotion

Fixed epoch 29, model seed 0:

| witness update steps | Fold 1 | Fold 2 | two-fold mean | delta vs zero-step |
|---:|---:|---:|---:|---:|
| 0 (SVD/PCA initializer control) | 0.607477 | 0.748837 | 0.678157 | -- |
| 1 KSVD update | **0.635418** | **0.755032** | **0.695225** | **+0.017068** |
| 4 KSVD updates | 0.610660 | 0.748423 | 0.679541 | +0.001384 |

One update step satisfies the predeclared promotion rule by a large margin: both folds improve,
the two-fold mean gains `+0.017068`, and there is no fold loss.  Four updates fail the required
`+0.003` mean improvement and are stopped.  In accordance with the rule written before the
screen, only the one-step and zero-step variants advance to model seeds 1/2.

### Completed six-cell deterministic-SVD confirmation

The promoted zero-step and one-step variants were completed for model seeds 0/1/2 on Fold 1
and Fold 2.  All numbers below are the predeclared fixed epoch 29, not per-run best epochs.

| route | Fold 1 seed 0/1/2 | Fold 1 mean | Fold 2 seed 0/1/2 | Fold 2 mean | pooled mean | pooled sample SD |
|---|---|---:|---|---:|---:|---:|
| deterministic SVD, 0 KSVD updates | .607477 / .625168 / .597340 | .609995 | .748837 / .717609 / .765177 | .743874 | .676935 | .075433 |
| deterministic SVD, 1 KSVD update | .635418 / .632581 / .595395 | .621131 | .755032 / .733134 / .712689 | .733618 | .677375 | .064613 |
| original random-init, 4 KSVD updates | .612435 / .609733 / .642258 | .621476 | .783475 / .735874 / .754401 | .757916 | **.689696** | .077107 |
| matched GINE | .579741 / .606218 / .620613 | .602190 | .781569 / .671257 / .747530 | .733452 | .667821 | .081346 |

Paired fixed-epoch comparisons:

| comparison | Fold 1 mean delta | Fold 2 mean delta | pooled delta | wins |
|---|---:|---:|---:|---:|
| SVD one-step - SVD zero-step | +.011136 | -.010256 | **+.000440** | 4/6 |
| SVD one-step - original random-init KSVD | -.000344 | -.024298 | **-.012321** | 2/6 |
| SVD one-step - matched GINE | +.018941 | +.000166 | +.009553 | 3/6 |
| SVD zero-step - matched GINE | +.007805 | +.010422 | +.009113 | 4/6 |
| original random-init KSVD - matched GINE | +.019285 | +.024464 | **+.021875** | **6/6** |

The large seed-0 one-step improvement does **not** replicate.  Seed 2 reverses the result,
including a `-.052489` paired change on Fold 2, leaving the six-cell one-step gain over the
zero-step control at only `+.000440`.  One deterministic KSVD update therefore cannot be claimed
to improve the SVD/OMP control.  It also remains `-.012321` below the already frozen
random-initialized KSVD route and wins only two of six paired cells against it.

The reconstruction diagnostic reinforces that reconstruction fit is not a sufficient selection
criterion.  On Fold 1, witness relative reconstruction error decreases from `.673089` at zero
steps to `.641836` at one and `.619118` at four; on Fold 2 it decreases from `.584449` to
`.531101` and `.496544`.  Four updates nevertheless failed the seed-0 predictive screen, and one
update failed to replicate.  Lower witness reconstruction error is not monotonically converted
into scaffold-fold ROC-AUC.

Interpretation and decision:

1. Deterministic initialization removes arbitrary dictionary-seed ambiguity, but the tested
   one-step variant does not preserve the original route's predictive strength.
2. The deterministic zero-step SVD plus joint sparse OMP control is already above matched GINE
   by `+.009113` on average, but only in four of six cells.  Thus witness selection, the dual-bank
   representation, and sparse assignment are useful candidates, while this audit does not isolate
   a stable benefit from the additional witness-bank atom update.
3. The original frozen random-initialized four-update KSVD remains the primary route: it is
   `+.021875` over GINE with `6/6` paired wins on these folds.  This is not a post-hoc dictionary
   seed choice; seed `1000003` was fixed before the initialization audit, and the separate
   dictionary-seed robustness experiment showed a positive aggregate gain across three seeds.
4. Do not promote deterministic one-step, do not select model seed 0 post hoc, and do not open
   official-valid/test on the basis of this result.

The full audit passes.  All 14 deterministic-SVD model artifacts (the 12 zero/one-step
confirmation runs plus two four-step screen runs) have null official AUCs, zero official
valid/test evaluations, and `37,551 = 37,382 + 169` parameters.  All six deterministic caches
have zero official-valid/test token entries and correct deterministic-SVD/iteration metadata.
Within each fold, original/zero/one/four-step caches use identical selected graphs and witness
matrices.  Within every fold/model-seed cell, original/zero/one-step models have identical
backbone and token-projection initialization hashes, loader seed, and first three minibatches.

Artifact:

```text
results/molhiv/backgroundpluswitness_svd_warmstart_fixed_epoch29_summary.json
```

## 22. Predeclared background-residual witness KSVD audit

The deterministic warm-start audit indicates that improving witness reconstruction alone is not
a reliable route to scaffold AUC.  The next experiment changes the *learning target* rather than
initialization or model capacity.  The hypothesis is that fitting the witness bank to raw patches
causes it to relearn common chemistry already represented by the background KSVD bank.

For every graph-balanced OOF-MIL witness patch `y`, compute the frozen background reconstruction
and its residual:

```text
a_bg = OMP(D_bg, y, T=2)
r     = y - D_bg a_bg
r_hat = r / ||r||_2
```

The eight-atom witness dictionary is fitted to the identical matrix of normalized residuals.  All
three controls use the same frozen KSVD background dictionary, selected witness identities,
residual matrix, atom budget, numeric dictionary seed, and downstream joint OMP:

```text
shared background KSVD D=8
+ residual KSVD / PCA / random-patch witness D=8
-> concatenate D=16
-> single joint OMP on the original patch, T=2
-> unchanged 169-parameter rich sparse-code + reconstruction residual
-> unchanged 3-layer GINE
```

This retains `37,551 = 37,382 + 169` trainable parameters and adds no explicit ring, scaffold, or
handwritten chemistry feature.  KSVD remains the mechanism that learns complementary residual
atoms; PCA and random-patch are matched witness-bank controls rather than full-pipeline changes.

### Stage-1 protocol and promotion rule

Before inspecting residual-dictionary results, Stage 1 is frozen to:

```text
Fold 1 + Fold 2
model seed 0
fixed epoch 29
KSVD iterations = 4, random-column initialization, dictionary seed = 1000003
families = residual KSVD / residual PCA / residual random-patch
```

The raw-witness comparator is the already frozen original KSVD route with the same background
KSVD, `D=8+8`, `T=2`, readout, model seed, and epoch.  Residual KSVD advances to model seeds 1/2
only if both conditions hold:

```text
mechanism condition:
  mean(Fold1,Fold2) >= raw-witness KSVD mean + 0.005
  and neither fold is below raw-witness KSVD by more than 0.010

KSVD-specific condition:
  residual-KSVD two-fold mean >= residual-PCA mean + 0.003
  and residual-KSVD two-fold mean >= residual-random-patch mean + 0.003
```

If either condition fails, the route stops after seed 0.  Per-run best epochs are diagnostic only;
all decisions use fixed epoch 29.  Official-valid/test patches remain unencoded and official
valid/test AUCs remain unevaluated.

### Stage-1 result: residual target does not satisfy KSVD promotion

Fixed epoch 29, model seed 0:

| witness bank fitted on background residuals | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| KSVD | 0.591710 | **0.785539** | 0.688625 |
| PCA | **0.636443** | 0.782640 | **0.709542** |
| random patch | 0.612846 | 0.761545 | 0.687195 |
| original raw-witness KSVD comparator | 0.612435 | 0.783475 | 0.697955 |

Predeclared comparisons:

```text
residual KSVD - raw-witness KSVD
  Fold 1         -0.020726
  Fold 2         +0.002065
  two-fold mean  -0.009330

residual KSVD - residual PCA
  Fold 1         -0.044734
  Fold 2         +0.002900
  two-fold mean  -0.020917

residual KSVD - residual random patch
  Fold 1         -0.021136
  Fold 2         +0.023995
  two-fold mean  +0.001429
```

Residual KSVD fails both preregistered conditions.  It does not improve the raw-witness KSVD
mean by `+0.005`; it instead loses `0.009330`, including a Fold-1 loss greater than the allowed
`0.010`.  It also trails the matched residual-PCA control by `0.020917`, rather than exceeding it
by `0.003`.  In accordance with the rule, model seeds 1/2 are not run and this route stops after
Stage 1.

The result separates two hypotheses.  Learning a complementary residual subspace may be useful:
residual PCA is the strongest seed-0 two-fold route in this screen.  However, the KSVD-specific
sparse reconstruction objective does not exploit that residual space reliably.  The residual
KSVD atoms are substantially non-orthogonal to each other (mean absolute off-diagonal coherence
`.1359` on Fold 1 and `.1005` on Fold 2), while the PCA residual atoms are orthogonal by
construction.  This geometry is diagnostic rather than a new selection criterion, but it is
consistent with the Fold-1 KSVD failure: an eight-atom, `T=2` sparse dictionary can specialize to
correlated residual modes that reconstruct selected witnesses without transferring across
scaffolds.

No claim is made from per-run best epochs.  Notably, Fold-2 KSVD reaches `0.785539` at epoch 29,
but the predeclared two-fold test prevents selecting that favorable fold in isolation.

Implementation and audit checks pass:

- the new default `raw_patch` path reproduced every array of the historical Fold-1 cache bitwise;
- KSVD/PCA/random residual controls use exactly the same selected witnesses, normalized residual
  matrix, eight-atom KSVD background, numeric dictionary seed, and downstream model/data order;
- all official-valid/test token entries are zero for all three families;
- all six model artifacts have null official AUCs and zero official-valid/test evaluations;
- every model has `37,551 = 37,382 + 169` parameters;
- original raw KSVD and all residual controls have identical backbone/token-projection
  initialization hashes, loader seed, and first three minibatches within each fold.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundplusresidualwitness_sharedksvd_fold{1,2}.{npz,json}
results/molhiv/backgroundplusresidualwitness_sharedksvd_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundplusresidualwitness_sharedksvd_stage1_fixed_epoch29_summary.json
```

## 23. Predeclared overcomplete KSVD with graph-balanced atom selection

The residual-target audit suggests that forcing KSVD to reconstruct a hand-isolated residual
subspace can discard transferable witness structure.  The next experiment instead preserves the
raw OOF-MIL witness patches and changes only the witness-bank capacity during dictionary fitting.
The hypothesis is that an eight-atom KSVD is too constrained to both discover diverse sparse
motifs and reserve atoms for the rare graph-level signal.  A larger candidate bank can discover
those modes, after which a graph-balanced supervised compression returns to the same eight-atom
runtime representation.

For each matched dictionary family:

```text
fit candidate witness dictionary D_candidate = 24 on identical raw OOF-MIL witnesses
encode the selected witnesses by OMP T_candidate = 3
average |code| within each graph before consulting its label
compute standardized positive-minus-negative graph-level usage for each atom
retain 4 atoms from the positive-associated end and 4 from the negative-associated end
shared train-only background KSVD D=8 + selected witness D=8
concatenate to D=16 and re-encode every permitted node by one joint OMP T=2
unchanged 169-parameter rich sparse-code + joint reconstruction residual
unchanged matched 3-layer GINE
```

Averaging atom usage within each graph prevents molecules with more patches from receiving more
weight, and the absolute-code statistic makes selection invariant to arbitrary dictionary-atom
signs.  The route uses only official-inner-train graph labels and cross-fitted witness scores;
official-valid/test patches remain unencoded.  It adds no explicit ring, scaffold, fingerprint,
or handwritten chemistry feature.  Final trainable parameters remain
`37,551 = 37,382 GINE + 169 KSVD residual`; the overcomplete bank exists only during offline
inner-train dictionary construction.

### Stage-1 protocol and promotion rule

Before inspecting any selected-atom model result, Stage 1 is frozen to:

```text
Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
raw witness target
shared background family = KSVD, D=8
candidate witness atoms = 24
candidate OMP sparsity = 3
retained witness atoms = 4 positive-associated + 4 negative-associated
final joint dictionary D=16, final OMP T=2
KSVD updates = 4, random-column initialization
numeric dictionary seed = 1000003
families = KSVD / PCA / random patch
```

The comparator is the already frozen original raw-witness KSVD route at model seed 0 and epoch 29:
Fold 1 `0.612435`, Fold 2 `0.783475`, two-fold mean `0.697955`.  Selected-atom KSVD advances to
model seeds 1/2 only if both preregistered conditions hold:

```text
mechanism condition:
  selected-atom KSVD two-fold mean >= original raw-witness KSVD mean + 0.005
  and neither fold is below original raw-witness KSVD by more than 0.010

KSVD-specific condition:
  selected-atom KSVD two-fold mean >= selected-atom PCA mean + 0.003
  and selected-atom KSVD two-fold mean >= selected-atom random-patch mean + 0.003
```

If either condition fails, this route stops after the six seed-0 cells.  Per-run best epochs are
strictly diagnostic; all promotion decisions use epoch 29.  No official-valid/test AUC will be
computed in this stage.

### Stage-1 result: graph-balanced atom selection does not transfer across folds

Fixed epoch 29, model seed 0:

| candidate-24 bank compressed to 8 atoms | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| KSVD | **0.618246** | 0.734803 | 0.676525 |
| PCA | 0.597270 | 0.748493 | 0.672881 |
| random patch | 0.600276 | **0.778854** | **0.689565** |
| original raw-witness KSVD comparator | 0.612435 | **0.783475** | **0.697955** |

Predeclared comparisons:

```text
selected-atom KSVD - original raw-witness KSVD
  Fold 1         +0.005811
  Fold 2         -0.048671
  two-fold mean  -0.021430

selected-atom KSVD - selected-atom PCA
  Fold 1         +0.020977
  Fold 2         -0.013689
  two-fold mean  +0.003644

selected-atom KSVD - selected-atom random patch
  Fold 1         +0.017971
  Fold 2         -0.044051
  two-fold mean  -0.013040
```

The route fails both promotion conditions.  Although selected-atom KSVD improves Fold 1 over the
original route by `0.005811` and clears the two-fold PCA margin by `0.003644`, it loses `0.048671`
on Fold 2, so its two-fold mean is `0.021430` below the frozen raw-witness KSVD.  It also trails
the matched selected-random-patch control by `0.013040`.  Model seeds 1/2 are therefore not run.

This is evidence against label-guided hard atom pruning as the next breakthrough.  The candidate
atoms do show nontrivial inner-train graph-level usage effects, and the implementation prevents
patch-count weighting, but the signed usage ranking is not scaffold-stable: it helps Fold 1 while
removing useful Fold-2 witness directions.  The Fold-2 random-patch value further indicates that
this screen can reward a favorable retained subspace without identifying a KSVD-specific
mechanism.  Increasing offline candidate capacity does not solve the core transfer problem when
it is followed by a hard eight-atom supervised bottleneck.

Per-run best epochs remain diagnostic only and were not used for this decision.  No
official-valid/test result was inspected.

Implementation and audit checks pass:

- the graph-balanced selector passed synthetic shape, uniqueness, 4-positive/4-negative budget,
  class-direction, and dictionary-sign invariance checks;
- with candidate selection disabled, a rebuilt historical Fold-1 cache reproduced every NPZ
  array bitwise and retained the historical protocol identifier;
- both selected-atom caches use the exact same selected witnesses as the original raw route and
  the exact same eight-atom KSVD background for KSVD/PCA/random controls;
- every candidate bank has 24 atoms, every retained witness bank has eight unique atoms split
  4/4 by association direction, and every final joint dictionary has 16 atoms;
- all official-valid/test token entries are zero;
- all six Stage-1 models have null official AUCs, zero official-valid/test evaluations, and
  `37,551 = 37,382 + 169` trainable parameters;
- within each fold, the original raw comparator and all three selected-atom models have identical
  backbone/token-projection initialization hashes, loader seed, and first three minibatches.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundplusselectedwitness_sharedksvd_fold{1,2}.{npz,json}
results/molhiv/backgroundplusselectedwitness_sharedksvd_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundplusselectedwitness_sharedksvd_stage1_fixed_epoch29_summary.json
```

## 24. Predeclared unpruned overcomplete OOF-witness bank

Section 23 cannot distinguish whether overcomplete witness discovery is unhelpful or whether the
failure is caused by supervised hard compression from 24 atoms back to eight.  The next screen
keeps the entire candidate witness bank and removes graph-label-based atom pruning altogether.
This is intentionally a single capacity intervention rather than another label-aware selector:

```text
identical graph-balanced OOF-MIL raw witnesses
shared train-only background KSVD D=8
KSVD / PCA / random-patch witness bank D=24
witness-bank fit sparsity T_fit=3
concatenate to final D=32
single joint node encoding OMP T_final=2
unchanged rich sparse-code + exact joint-reconstruction graph residual
unchanged matched 3-layer GINE
```

The builder now separates witness fitting sparsity from final joint encoding sparsity.  The new
argument defaults to the historical behavior, so existing routes remain unchanged.  Keeping all
24 witness atoms makes this screen the direct counterpart of Section 23's candidate dictionary,
while the deployment code remains sparse (`T=2`).

This route remains small: the rich residual expands from 168 to 328 fixed graph statistics, so
its trainable linear residual is expected to grow only from 169 to 329 parameters.  Expected total
trainable parameters are `37,711 = 37,382 + 329`, only 160 more than the frozen D16 route and still
far below CIN-scale models.  Offline node OMP is more expensive because the final dictionary has
32 rather than 16 columns, but no additional message-passing layer or explicit chemistry feature
is introduced.

### Stage-1 protocol and promotion rule

Before inspecting any unpruned-D24 result, Stage 1 is frozen to:

```text
Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
raw witness target
shared background family = KSVD, D=8
witness family atoms = 24
witness fit sparsity = 3
no graph-label atom selection or pruning
final joint dictionary D=32, final OMP T=2
KSVD updates = 4, random-column initialization
numeric dictionary seed = 1000003
families = KSVD / PCA / random patch
```

The comparator remains the frozen D8-background + D8-raw-witness KSVD route at epoch 29:
Fold 1 `0.612435`, Fold 2 `0.783475`, mean `0.697955`.  Unpruned-D24 KSVD advances to model seeds
1/2 only if both conditions hold:

```text
mechanism condition:
  unpruned-D24 KSVD two-fold mean >= D8-witness raw KSVD mean + 0.005
  and neither fold is below the D8-witness comparator by more than 0.010

KSVD-specific condition:
  unpruned-D24 KSVD mean >= unpruned-D24 PCA mean + 0.003
  and unpruned-D24 KSVD mean >= unpruned-D24 random-patch mean + 0.003
```

If either condition fails, the route stops after six seed-0 cells.  Decisions use epoch 29 only;
per-run best epochs are diagnostic.  Official-valid/test patches and AUCs remain untouched.

### Stage-1 result: retaining all 24 witness atoms removes the pruning collapse but does not improve KSVD

Fixed epoch 29, model seed 0:

| unpruned witness bank D=24, final D=32/T=2 | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| KSVD | 0.609160 | 0.776413 | 0.692787 |
| PCA | 0.592332 | 0.760806 | 0.676569 |
| random patch | **0.643956** | **0.780881** | **0.712418** |
| original D8-witness KSVD comparator | 0.612435 | **0.783475** | 0.697955 |

Predeclared comparisons:

```text
full-D24 KSVD - original D8-witness KSVD
  Fold 1         -0.003275
  Fold 2         -0.007061
  two-fold mean  -0.005168

full-D24 KSVD - full-D24 PCA
  Fold 1         +0.016828
  Fold 2         +0.015608
  two-fold mean  +0.016218

full-D24 KSVD - full-D24 random patch
  Fold 1         -0.034795
  Fold 2         -0.004468
  two-fold mean  -0.019631
```

The route fails both promotion conditions and stops after seed 0.  Keeping the full candidate bank
substantially repairs the supervised-pruning result (`0.692787` versus `0.676525` for selected-atom
KSVD), confirming that hard label-guided compression was harmful.  However, it remains below the
original eight-witness-atom KSVD on both folds and loses `0.005168` on average, so doubling the
final dictionary width and sparse-coding runtime is not justified.

There is a useful mechanistic split.  Full-D24 KSVD beats matched PCA on both folds by `0.016218`
on average, but loses to matched random patches on both folds by `0.019631`.  The exact same 24
candidate dictionaries were used here as in Section 23 before pruning, so this is not a refit or
seed discrepancy.  The result suggests that broad witness coverage is useful, while reconstruction-
optimized KSVD updates may collapse some of the diversity preserved by raw sampled witnesses.
Measured witness-bank coherence is high (original D8 KSVD mean absolute off-diagonal correlation
`0.3375/0.3986` on Folds 1/2; full-D24 KSVD `0.3007/0.3874`), which motivates a future
**diversity- or incoherence-controlled KSVD** mechanism rather than more atoms or supervised hard
selection.

Audit checks pass:

- `--witness-fit-sparsity=0` reproduced the historical Fold-1 cache bitwise; the new separation of
  fit sparsity and final OMP sparsity is backward compatible;
- each full D24 witness dictionary is bitwise identical (after float32 storage) to Section 23's
  corresponding pre-pruning candidate bank;
- all families share the exact same eight-atom KSVD background and witness identities;
- official-valid/test token entries are zero and official AUCs are null with zero evaluations;
- all six models have `37,711 = 37,382 + 329` parameters;
- within each fold, KSVD/PCA/random D32 models have identical initialization hashes and minibatch
  order; their base GINE initialization and loader order also match the D16 comparator.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a32_t2_noring_oofmil_backgroundplusfullwitness24_sharedksvd_fold{1,2}.{npz,json}
results/molhiv/backgroundplusfullwitness24_sharedksvd_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundplusfullwitness24_sharedksvd_stage1_fixed_epoch29_summary.json
```

## 25. Predeclared incoherence-controlled OOF-witness KSVD

The unpruned D24 screen suggests that wider witness coverage can help, but reconstruction-optimized
KSVD atoms remain strongly correlated and lose to the matched raw random-patch bank.  This section
therefore changes the **KSVD optimization itself**, rather than adding capacity, graph-label atom
selection, explicit chemistry features, or another GINE branch.

After every ordinary sequential K-SVD dictionary update, the unit-normalized witness dictionary
receives one projected gradient step on its off-diagonal Gram penalty:

```text
G_off = D^T D with diag(G_off) = 0
D <- D - eta * D G_off
column-normalize(D)
recompute sparse codes by OMP
```

The background KSVD remains the exact frozen train-only D=8 dictionary from the original cache.
Only the OOF-MIL witness KSVD uses this update.  PCA and random-patch controls use the same fixed
witness identities, dimensions, numeric dictionary seed, background dictionary, final joint OMP,
rich reconstruction residual, and downstream GINE initialization as KSVD.

### Stage-1 protocol and promotion rule

Before building or inspecting any incoherence-controlled MolHIV cache, Stage 1 is frozen to one
non-swept setting:

```text
Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
raw graph-balanced top-2 OOF-MIL witness patches
shared frozen background family = KSVD, D=8
witness family atoms = 8
witness fit sparsity = 2
KSVD updates = 4, random-column initialization
KSVD witness coherence step eta = 0.05
concatenate to final D=16
single joint node encoding OMP T=2
families = incoherent KSVD / PCA / random patch
numeric dictionary seed = 1000003
no ring, scaffold, fingerprint, or graph-label atom-pruning feature
```

The frozen original raw-witness KSVD comparator at model seed 0 and epoch 29 is Fold 1 `0.612435`,
Fold 2 `0.783475`, mean `0.697955`.  Incoherent KSVD advances to model seeds 1/2 only if both
conditions hold:

```text
mechanism condition:
  incoherent KSVD two-fold mean >= original KSVD mean + 0.005
  and neither fold is below original KSVD by more than 0.010

KSVD-specific condition:
  incoherent KSVD mean >= matched PCA mean + 0.003
  and incoherent KSVD mean >= matched random-patch mean + 0.003
```

Before model training, the dictionaries must pass a geometry/protocol gate: witness mean absolute
off-diagonal coherence must decrease on both folds relative to original KSVD; witness-pool OMP
reconstruction must not catastrophically worsen; witness graph/node identities and the frozen
background dictionary must remain identical; all official-valid/test token entries and evaluation
counts must remain zero.  If the geometry gate fails, no models are run.  If either Stage-1
promotion condition fails, seeds 1/2 are not run.  Official-valid/test AUCs remain untouched.

### Protocol incident before Stage-1 scoring

The first Fold-1 launcher omitted the runner's explicit `--selection-only` flag.  It therefore
performed three accidental official-valid evaluations after completing the inner-fold histories.
Those artifacts were immediately quarantined outside `results/molhiv`; their official-valid values
are excluded from every comparison, route decision, summary, and subsequent design choice.  No
official-test evaluation occurred.  All six Stage-1 cells are rerun from scratch with
`--selection-only`, and only artifacts with null official AUCs and zero official-valid/test
evaluation counters are admissible below.

### Stage-1 result: off-manifold incoherence pressure is harmful

Fixed epoch 29, model seed 0:

| eta=0.05 witness route | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| incoherent KSVD | 0.587939 | 0.764151 | 0.676045 |
| PCA | **0.623145** | **0.774393** | **0.698769** |
| random patch | 0.613581 | 0.766465 | 0.690023 |
| original raw-witness KSVD comparator | 0.612435 | **0.783475** | 0.697955 |

```text
incoherent KSVD - original KSVD mean = -0.021910
incoherent KSVD - PCA mean          = -0.022724
incoherent KSVD - random mean       = -0.013978
```

Both promotion conditions fail, so model seeds 1/2 are not run.  The geometry intervention itself
worked: mean witness coherence fell from `0.3375 -> 0.2342` on Fold 1 and `0.3986 -> 0.3009` on
Fold 2, while witness-pool reconstruction changed by only `+0.05%` and `+1.72%` respectively.
Nevertheless, downstream fixed-epoch AUC declined on both folds and KSVD lost to both controls.
This indicates that generic Gram orthogonalization removes or rotates task-useful patch directions
even when aggregate reconstruction remains acceptable.  The next diversity mechanism should keep
atoms on the empirical witness manifold rather than pushing learned atoms away from one another in
ambient feature space.

All six admissible model artifacts are fresh `--selection-only` reruns with null official AUCs,
zero official-valid/test evaluation counters, `37,551` parameters, and matched initialization and
minibatch order.  The three accidental official-valid evaluations described above remain excluded.
No official-test evaluation occurred.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundplusincoherentwitness_sharedksvd_fold{1,2}.{npz,json}
results/molhiv/backgroundplusincoherentwitness_sharedksvd_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundplusincoherentwitness_sharedksvd_geometry_audit.json
results/molhiv/backgroundplusincoherentwitness_sharedksvd_stage1_fixed_epoch29_summary.json
```

## 26. Predeclared max-min witness initialization followed by standard KSVD

Section 25 shows that diversity is not useful when imposed by an ambient-space Gram gradient.
The next route instead chooses a diverse set of **real OOF witness patches** as the initial atom
bank, then performs the unchanged four sequential K-SVD updates.  Starting from a seeded witness
column, each new initial atom minimizes its maximum absolute cosine similarity to already selected
columns.  No graph label, chemistry rule, ring feature, or validation statistic enters this
initialization.

An inner-train-only geometry pilot was used solely to freeze the update count before any model was
run.  Four standard KSVD iterations give the best reconstruction among the checked `1/2/4` update
counts while retaining substantially lower coherence than the historical random-column KSVD:

```text
Fold 1: coherence 0.3375 -> 0.2912, reconstruction 0.6087 -> 0.5880
Fold 2: coherence 0.3986 -> 0.2169, reconstruction 0.4784 -> 0.4936
```

The one setting predeclared for Stage 1 is therefore:

```text
Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
same raw graph-balanced top-2 OOF-MIL witness patches
shared frozen background KSVD D=8
witness KSVD D=8, T=2
seeded max-min absolute-cosine real-column initialization
standard KSVD iterations = 4
no incoherence gradient
final joint dictionary D=16, OMP T=2
numeric dictionary seed = 1000003
unchanged rich reconstruction residual + 3-layer GINE
```

The frozen original random-column KSVD comparator remains Fold 1 `0.612435`, Fold 2 `0.783475`,
mean `0.697955`.  Promotion requires:

```text
mechanism condition:
  max-min-init KSVD mean >= original KSVD mean + 0.005
  and neither fold is below original KSVD by more than 0.010

KSVD-specific condition:
  max-min-init KSVD mean >= matched PCA mean + 0.003
  and max-min-init KSVD mean >= matched random-patch mean + 0.003
```

PCA and random-patch witness dictionaries are mathematically unaffected by the KSVD initializer.
Their Section-25 model controls may be reused only if the new cache verifies their dictionaries,
node tokens, preprocessing inputs, model initialization, and minibatch order bitwise identical;
otherwise they must be rerun.  All model execution uses `--selection-only`; official-valid/test
remain excluded.  Seeds 1/2 run only if both promotion conditions pass.

### Stage-1 result

The max-min cache passed the predeclared protocol gate.  It retained the same witnesses and frozen
background KSVD as the original route; its protocol ID is distinct; all official-valid/test token
rows are zero; and all official evaluation counters remain zero.  PCA and random-patch dictionaries
and node-token arrays are bitwise identical to Section 25, so their admissible selection-only model
runs are reused.  Within each fold, all three families have the same base GINE initialization and
first three minibatches.

At the fixed, non-selected epoch 29 and model seed 0:

| family | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| max-min-init KSVD | 0.593294 | 0.774654 | 0.683974 |
| matched PCA | 0.623145 | 0.774393 | 0.698769 |
| matched random patch | 0.613581 | 0.766465 | 0.690023 |
| frozen original random-init KSVD | 0.612435 | 0.783475 | 0.697955 |

The max-min-init KSVD is `-0.019141/-0.008821` below the original KSVD on Fold 1/Fold 2 and
`-0.013981` on the two-fold mean.  It is `-0.014795` below matched PCA and `-0.006049` below matched
random patch on the mean.  It therefore fails both the mechanism and KSVD-specific promotion
conditions; model seeds 1/2 are not run.

This is a useful negative result: selecting diverse **real** witness columns improves unsupervised
atom geometry without the off-manifold rotation used in Section 25, but the downstream fixed-epoch
AUC still falls, especially on Fold 1.  Together, Sections 20, 21, 24, 25, and 26 indicate that
initialization, width, reconstruction quality, and coherence are not the present breakthrough
axis.  Further dictionary-initialization/coherence sweeps are stopped.  The next intervention should
instead test whether the deliberately tiny linear KSVD graph residual is under-expressive while
leaving the frozen primary dictionary and GINE backbone unchanged.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_backgroundplusmaxminwitness_sharedksvd_fold{1,2}.{npz,json}
results/molhiv/backgroundplusmaxminwitness_sharedksvd_geometry_audit.json
results/molhiv/backgroundplusmaxminwitness_sharedksvd_richrecon_fold{1,2}_ksvd_seed0.json
results/molhiv/backgroundplusmaxminwitness_sharedksvd_stage1_fixed_epoch29_summary.json
```

## 27. Predeclared additive nonlinear sparse-assignment readout

Sections 20--26 repeatedly changed dictionary geometry without improving the frozen primary route.
The surviving model, however, gives its 168-dimensional graph descriptor only a zero-initialized
linear map.  That map can express independent atom/statistic main effects, but it cannot directly
express KSVD-specific interactions such as background/witness co-usage, atom competition,
reconstruction error conditional on witness activation, or combinations of rare motif usage and
patch count.

The next intervention leaves the complete representation frozen and changes only this bottleneck.
The historical linear residual remains present, and a zero-output-initialized nonlinear correction
is added:

```text
logit = GINE(graph)
      + Linear_0(168 -> 1)
      + Linear_0(SiLU(Linear(168 -> 16)) -> 1)
```

Both output maps start at zero, so the initial predictor is exactly the matched GINE.  Retaining the
linear term makes this a strict extension of the successful readout rather than replacing it with a
harder-to-optimize MLP.  The 168 inputs remain fixed, interpretable summaries of sparse assignment
and reconstruction under the learned background+witness dictionary.  No extra message-passing
layer, GINE branch, explicit ring/scaffold/fingerprint feature, label-selected atom, or dictionary
change is introduced.

### Backward-compatibility gate

The implementation adds `--graph-residual-hidden`, with historical default `0`.  A one-epoch
Fold-1 replay at the default setting exactly reproduced the historical train loss, inner AUC, JK
gates, graph-residual norm/bias, base/token initialization hashes, and first three minibatches.
The default model remains `37,551 = 37,382 + 169` parameters.

### Stage-1 protocol and promotion rule

Before any nonlinear model result is inspected, the one setting is frozen to:

```text
Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
original frozen primary dual-bank caches
background dictionary D=8 + OOF-MIL witness dictionary D=8
joint OMP T=2
rich sparse-code + OMP reconstruction descriptor, dimension 168
historical linear residual retained
nonlinear correction hidden width = 16, SiLU, zero-initialized output
same 3-layer hidden-64 GINE-JK backbone
families = KSVD / PCA / random patch
selection-only; official valid and official test are not evaluated
```

The nonlinear model has only `40,272` trainable parameters: `37,382` GINE parameters plus `2,890`
readout parameters.  This is a `7.25%` increase over the frozen primary model and has negligible
message-passing/runtime overhead.

The frozen linear KSVD comparator is Fold 1 `0.612435`, Fold 2 `0.783475`, mean `0.697955` at
seed 0 and epoch 29.  Seeds 1/2 are promoted only if both conditions hold:

```text
mechanism condition:
  nonlinear KSVD mean >= linear KSVD mean + 0.005
  and neither fold is below linear KSVD by more than 0.010

KSVD-specific condition:
  nonlinear KSVD mean >= nonlinear PCA mean + 0.003
  and nonlinear KSVD mean >= nonlinear random-patch mean + 0.003
```

All three families use the same hidden width, base initialization seed, optimization settings, and
minibatch order within a fold.  Every output artifact must retain null official AUCs and zero
official-valid/test evaluation counters.

### Stage-1 result

All six hidden-16 runs are fresh `--selection-only` models with null official AUCs and zero
official-valid/test evaluation counters.  Within each fold, KSVD/PCA/random have identical base,
linear-residual, and nonlinear-residual initialization hashes and identical first three minibatches.
Each model has the expected `40,272 = 37,382 + 2,890` parameters.

At fixed epoch 29 and model seed 0:

| family | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| nonlinear KSVD | **0.616022** | 0.771321 | **0.693671** |
| nonlinear PCA | 0.616227 | 0.767918 | 0.692072 |
| nonlinear random patch | 0.601252 | **0.785616** | 0.693434 |
| frozen linear KSVD | 0.612435 | 0.783475 | 0.697955 |

Relative to the frozen linear KSVD, the nonlinear KSVD changes Fold 1/Fold 2 by
`+0.003586/-0.012153` and lowers the two-fold mean by `0.004284`.  Its mean is only `+0.001600`
over nonlinear PCA and `+0.000237` over nonlinear random patch.  It therefore fails both
predeclared promotion conditions, including the no-fold-regression guard on Fold 2.  Seeds 1/2 and
hidden-width sweeps are not run.

The result rejects the simple readout-capacity hypothesis.  A small nonlinear interaction map is
not harmful in parameter or runtime terms, but it is nearly family-neutral and does not improve the
frozen KSVD route.  Combined with the earlier rich-signature and transition-readout controls, this
indicates that adding generic supervised capacity after sparse coding is unlikely to create the
missing KSVD-specific scaffold transfer.

Artifacts:

```text
results/molhiv/backgroundpluswitness_nonlinear16_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundpluswitness_nonlinear16_stage1_fixed_epoch29_summary.json
```

## 28. Predeclared residual-learning-rate audit

The nonlinear screen exposes one untested optimization confound in all 134 historical rich
sparse-code/reconstruction runs: the graph residual was always optimized at only `0.1x` the GINE
learning rate (`1e-4` versus `1e-3`).  This convention originated from node-token injection, where a
slow token branch protects message passing from early perturbation.  It is not obviously appropriate
for the present zero-initialized **graph-level** residual, which cannot perturb any hidden GINE state
and whose linear weight norm is still growing late in training.

A single non-swept, moderate correction is frozen before running any new result:

```text
original primary KSVD caches, Fold 1 + Fold 2
model seed = 0
fixed decision epoch = 29
historical linear 168 -> 1 graph residual (graph_residual_hidden = 0)
base GINE learning rate = 1e-3
residual learning-rate scale = 0.3 (3e-4)
residual weight decay = 1e-3
all other architecture/data/optimizer settings unchanged
selection-only; official valid/test remain unevaluated
```

This changes neither representation nor parameter count (`37,551`).  It is deliberately a one-point
audit rather than a learning-rate sweep.  The route advances to matched PCA/random controls and
model seeds 1/2 only if its two-fold fixed-epoch mean exceeds the frozen `0.697955` linear-KSVD mean
by at least `0.005`, with neither fold below its frozen comparator by more than `0.010`.  Otherwise
further residual-LR tuning stops.

### Stage-1 result

Both runs are admissible selection-only artifacts with null official AUCs and zero evaluation
counters.  At fixed epoch 29:

| KSVD residual LR scale | Fold 1 | Fold 2 | two-fold mean |
|---|---:|---:|---:|
| historical `0.1` | **0.612435** | **0.783475** | **0.697955** |
| audited `0.3` | 0.609818 | 0.763858 | 0.686838 |

The moderate 3x residual LR loses `0.002617` on Fold 1, `0.019616` on Fold 2, and `0.011117` on
the mean.  Its epoch-29 residual weight norms grow to `0.353/0.349`, more than twice the historical
approximately `0.16` scale, while validation falls.  The branch was therefore not merely
undertrained; faster fitting increases scaffold overfit.  The predeclared mechanism gate fails, so
PCA/random controls, additional seeds, and further residual-LR tuning are not run.

Artifacts:

```text
results/molhiv/backgroundpluswitness_residuallr03_richrecon_fold{1,2}_ksvd_seed0.json
results/molhiv/backgroundpluswitness_residuallr03_stage1_fixed_epoch29_summary.json
```

## 29. Predeclared independently trained KSVD graph predictor

### Motivation

Sections 26--28 ruled out dictionary initialization/coherence, generic nonlinear readout capacity,
and a larger residual learning rate as immediate explanations for the remaining gap.  A remaining
failure mode is **optimization interference**: in the historical additive model, GINE and the
KSVD residual are optimized only through their summed logit, so the high-capacity backbone can
absorb, cancel, or make the sparse descriptor's signal redundant.  That makes the learned KSVD
term difficult to interpret as a standalone predictor even when the combined model improves.

This intervention retains exactly one GINE backbone and the frozen 168-dimensional rich
sparse-code/reconstruction descriptor.  It adds no parameter and no second GINE.  The two existing
predictors receive separate BCE objectives:

```text
L = BCE(GINE(graph), y) + BCE(Linear(KSVD descriptor), y)
```

The GINE objective has no computational dependence on the dictionary logit, and the dictionary
objective has no dependence on GINE states.  At inference, the predeclared fixed combination is:

```text
p_combined = 0.5 * sigmoid(GINE logit)
           + 0.5 * sigmoid(KSVD-only logit)
```

Thus the KSVD branch has a directly reportable standalone AUC and atom/statistic contributions,
while inference still uses only one GINE and one 168-to-1 linear map.  Parameter count remains
`37,551 = 37,382 + 169`.  No explicit ring, scaffold, fingerprint, or label-selected atom feature
is introduced.

### Backward-compatibility gate

The new CLI defaults are:

```text
--graph-residual-training joint
--graph-residual-combination logit_add
```

A fresh one-epoch Fold-1 replay under these defaults exactly reproduced the historical train loss,
inner AUC, JK gates, graph-residual norm/bias, base initialization hash, first three minibatches,
and parameter counts.  Therefore old commands retain the historical joint model exactly.

### Stage-1 protocol and promotion rule (frozen before independent results)

```text
families initially run = KSVD only
folds = Fold 1 + Fold 2
model seed = 0
decision epoch = exactly 29 (not best-epoch selection)
cache = original frozen background-D8 + OOF-witness-D8 shared-KSVD caches
coding = joint OMP T=2
readout = 168-dimensional rich sparse-code + reconstruction descriptor
KSVD predictor = linear 168 -> 1
training = independent BCE objectives with equal loss weight
inference = fixed 0.5/0.5 probability average
backbone = matched hidden-64, 3-layer GINE-JK
optimizer = historical LR 1e-3, dictionary-head LR scale 0.1, WD 1e-3
selection-only; official valid and official test are not evaluated
```

The mechanism is promoted to matched PCA/random-patch controls only if:

```text
independent combined KSVD mean >= original joint KSVD mean + 0.005
and neither fold is below original joint KSVD by more than 0.010
```

The frozen seed-0 epoch-29 joint KSVD comparator is Fold 1 `0.612435`, Fold 2 `0.783475`,
mean `0.697955`; therefore the required independent mean is at least `0.702955`, with fold floors
`0.602435` and `0.773475`.  If this gate passes, identically trained PCA and random-patch controls
are run, and seeds 1/2 are promoted only if KSVD also exceeds each matched control mean by at least
`0.003`.  All artifacts must retain null official AUCs and zero official-valid/test evaluations.

### Stage-1 result

The independently trained predictors remain exactly within the `37,551 = 37,382 + 169`
parameter budget.  At fixed epoch 29:

| predictor | Fold 1 | Fold 2 | mean |
|---|---:|---:|---:|
| independent GINE component | 0.612407 | 0.778236 | 0.695322 |
| independent KSVD-only component | 0.626137 | 0.717857 | 0.671997 |
| fixed probability average | 0.631739 | 0.768351 | 0.700045 |
| historical joint KSVD | 0.612435 | 0.783475 | 0.697955 |

The independent KSVD predictor is genuinely predictive on both held-out scaffold folds and is
stronger than GINE on Fold 1.  The fixed average improves Fold 1 by `0.019304`, demonstrating
complementarity, but loses `0.015124` on Fold 2, where GINE is much stronger.  Its mean gain over
the historical joint model is only `0.002090`, below the predeclared `0.005` requirement, and the
Fold-2 loss also exceeds the allowed `0.010`.  The mechanism gate therefore fails.  Matched
PCA/random controls and seeds 1/2 are not run for this probability-average rule.

This result does not reject independent KSVD training itself.  It specifically rejects an
uncalibrated equal probability average: it overweights the weaker sparse predictor on Fold 2.
The next cheapest identifiable question is whether the independently learned logit magnitude can
serve as its own fixed confidence scaling, without learning an ensemble coefficient.

Artifacts:

```text
results/molhiv/backgroundpluswitness_independentprobavg_richrecon_fold{1,2}_ksvd_seed0.json
results/molhiv/backgroundpluswitness_independentprobavg_stage1_fixed_epoch29_summary.json
```

## 30. Predeclared independent objectives with additive-logit inference

Section 29 established that the standalone KSVD descriptor carries held-out scaffold signal, but
an equal probability average is miscalibrated across folds.  The next intervention changes only
the **fixed inference algebra**:

```text
training:  BCE(GINE logit, y) + BCE(KSVD-only logit, y)
inference: combined logit = GINE logit + KSVD-only logit
```

No ensemble coefficient is learned or selected.  Because the two training objectives are still
independent, this retains the interpretability and non-interference properties of Section 29.
Unlike equal probability averaging, additive logits preserve the dictionary predictor's naturally
learned output scale, so a weak KSVD branch need not receive exactly half of the probability mass.
The model and parameter count are unchanged.

The protocol is frozen before results to the same two caches, Fold 1/Fold 2, seed 0, exact epoch 29,
and all historical optimizer/backbone settings.  KSVD alone is run first.  The same promotion gate
is retained: mean at least `0.702955`, Fold 1 at least `0.602435`, and Fold 2 at least `0.773475`.
Only on passing that gate are matched PCA/random controls considered.  Official valid/test remain
unevaluated.

### Stage-1 result

At fixed epoch 29, the additive-logit rule gives:

| predictor | Fold 1 | Fold 2 | mean |
|---|---:|---:|---:|
| independent additive-logit KSVD | 0.616941 | 0.779166 | 0.698054 |
| independent probability average | 0.631739 | 0.768351 | 0.700045 |
| historical joint KSVD | 0.612435 | 0.783475 | 0.697955 |

As required by the independent objectives, the complete training loss, GINE component AUC,
KSVD-only component AUC, JK gates, and sparse-head norm/bias trajectories are exactly identical to
Section 29; only inference algebra differs.  Additive logits repair most of the Fold-2 loss but
remove most of the Fold-1 gain.  The mean is only `+0.000099` over historical joint KSVD, far below
the promotion threshold.  Matched controls and more seeds are not run.

Together, Sections 29--30 show that the bottleneck is no longer GINE/KSVD gradient interference
alone.  The independently trained sparse head is predictive, but ordinary imbalanced BCE does not
produce a sufficiently scaffold-stable ranking for a fixed combination.

Artifacts:

```text
results/molhiv/backgroundpluswitness_independentlogitadd_richrecon_fold{1,2}_ksvd_seed0.json
results/molhiv/backgroundpluswitness_independentlogitadd_stage1_fixed_epoch29_summary.json
```

## 31. Predeclared metric-aligned pairwise KSVD-only objective

MolHIV is evaluated by ROC-AUC and has very few positives.  The independent KSVD head in Sections
29--30 was nevertheless trained with ordinary graph-level BCE, whose gradient is dominated by
negative calibration and whose learned bias has no effect on AUC.  The next intervention keeps the
GINE objective unchanged but trains the KSVD-only score with an all-positive/all-negative
within-minibatch pairwise logistic ranking loss:

```text
L = BCE(GINE logit, y)
  + mean_{positive i, negative j} softplus(-(s_ksvd[i] - s_ksvd[j]))
```

If a minibatch contains only one class, the sparse head falls back to its ordinary BCE for that
batch.  This directly asks the self-learned dictionary descriptor to rank active molecules above
inactive molecules, is aligned with the reported metric, adds no feature and no parameter, and
retains complete gradient independence from GINE.  Inference is predeclared as additive logits so
the learned sparse-score magnitude, rather than an arbitrary 50% probability share, controls its
influence.

Stage 1 is frozen to the original KSVD caches, Fold 1/Fold 2, seed 0, exact epoch 29, historical
head LR scale `0.1`, and `37,551` parameters.  Promotion requires both:

```text
ranking mechanism:
  pairwise standalone KSVD mean >= BCE standalone KSVD mean + 0.010
  and neither standalone fold loses more than 0.010

combined model:
  combined mean >= 0.702955
  and Fold 1 >= 0.602435
  and Fold 2 >= 0.773475
```

Matched PCA/random controls and seeds 1/2 are run only if both gates pass.  Official valid and test
remain unevaluated.

### Stage-1 result

At fixed epoch 29:

| objective | standalone Fold 1 | standalone Fold 2 | standalone mean | combined mean |
|---|---:|---:|---:|---:|
| independent BCE | 0.626137 | 0.717857 | 0.671997 | 0.698054 |
| pairwise logistic | 0.613666 | 0.701434 | 0.657550 | 0.700985 |

The pairwise objective increases the sparse-head norm substantially (`0.265/0.226` versus
`0.087/0.069`) but reduces standalone AUC on both folds.  The combined Fold 1/Fold 2 scores are
`0.622848/0.779122`; mean `0.700985` remains below `0.702955`.  Both the ranking-mechanism gate and
the combined-model gate fail, so no controls or additional seeds are run.  Noisy within-minibatch
positive-negative comparisons are not a useful replacement for the ordinary sparse-head BCE.

Artifacts:

```text
results/molhiv/backgroundpluswitness_independentpairwise_richrecon_fold{1,2}_ksvd_seed0.json
```

## 32. Predeclared convex balanced sparse-descriptor probe

The previous independent heads were optimized jointly in minibatches with GINE despite being a
convex 168-dimensional linear problem.  Section 28 showed that merely raising Adam's residual LR
causes joint overfit, while Section 31 showed that minibatch ranking is noisy.  The next diagnostic
removes both confounders: fit the KSVD-only graph predictor to numerical convergence with a single
regularized convex solver, completely outside GINE training.

The predeclared probe is:

```text
input = same fit-standardized 168 rich sparse-code + reconstruction features
classifier = L2 logistic regression
class weighting = balanced, computed from fit labels only
C = 0.1
solver = LBFGS, max_iter = 2000, tol = 1e-8
```

This is not a new handcrafted molecular feature: every input remains a dictionary-assignment or
KSVD reconstruction statistic already used by the frozen primary route.  It has exactly 169
trainable coefficients, admits direct per-atom/statistic contributions, and is far cheaper than a
GINE run.

Stage 1 is KSVD-only on Fold 1/Fold 2 using the original caches.  The route is worth integrating
with GINE only if its standalone mean is at least `0.690` and neither fold is below `0.620`; these
thresholds demand a material improvement over the minibatch-BCE standalone mean `0.671997` rather
than merely exploiting solver noise.  If it passes, matched PCA/random-patch probes with the exact
same solver are run before any GINE combination.  Official valid and test are never evaluated.

### Stage-1 and matched-control result

| family | Fold 1 | Fold 2 | mean |
|---|---:|---:|---:|
| KSVD | 0.667221 | 0.749117 | **0.708169** |
| PCA | 0.621695 | 0.735721 | 0.678708 |
| random patch | 0.665162 | 0.719763 | 0.692463 |

The convex probe passes its promotion gate and is strongly KSVD-specific: `+0.029461` over PCA
and `+0.015706` over random patch on mean AUC.  It also improves the prior minibatch-BCE KSVD-only
mean by `0.036172`.  This is the clearest new result since the frozen primary: the learned
assignment is useful, but its low-dimensional supervised readout should be solved as a regularized
convex problem rather than co-adapted with GINE.

Artifacts:

```text
results/molhiv/backgroundpluswitness_balancedconvexprobe_fold{1,2}_{ksvd,pca,randompatch}.json
results/molhiv/backgroundpluswitness_balancedconvexprobe_stage1_summary.json
```

## 33. Predeclared frozen convex KSVD probe plus one GINE

The promoted Section-32 KSVD probe is now fit on each fold's training graphs, installed into the
existing 168-to-1 sparse head, and frozen with respect to GINE optimization.  GINE is trained with
its ordinary BCE exactly as a standalone matched backbone.  Inference uses a fixed equal
probability average of the one GINE and one convex KSVD predictor.  There is no second GINE, no
learned ensemble weight, and no additional parameter; the total predictor remains 37,551
parameters and training runtime is approximately one GINE run plus a sub-second convex fit.

Stage 1 is Fold 1/Fold 2, seed 0, exact epoch 29, C=0.1, original KSVD caches.  Promotion requires
mean at least 0.702955 and fold floors 0.602435/0.773475.  Matched combined PCA/random controls are
run only if this gate passes.  Official valid/test remain unevaluated.


### Stage-1 and matched-control result

At the predeclared fixed epoch 29:

| family | Fold 1 | Fold 2 | mean |
|---|---:|---:|---:|
| **KSVD** | **0.670043** | **0.774011** | **0.722027** |
| PCA | 0.621603 | 0.750048 | 0.685825 |
| random patch | 0.662764 | 0.735192 | 0.698978 |

The fixed convex KSVD ensemble passes the mechanism gate and both matched-control margins.  Its
mean improvement is `+0.036202` over PCA and `+0.023049` over random patch.  Relative to the
historical joint KSVD seed-0 comparator (`0.697955`), it gains `+0.024072` while retaining one GINE,
one frozen 169-parameter dictionary readout, and `37,551` total predictor parameters.  The convex
probe itself remains exactly unchanged throughout GINE training.

All six artifacts retain null official-valid/test AUCs and zero official-valid/test evaluation
counters.

Artifacts:

```text
results/molhiv/backgroundpluswitness_fixedconvexprobavg_richrecon_fold{1,2}_{ksvd,pca,randompatch}_seed0.json
results/molhiv/backgroundpluswitness_fixedconvexprobavg_stage1_fixed_epoch29_summary.json
```

### Predeclared seed-robustness extension

Because the KSVD-specific Stage-1 margins pass, seeds 1 and 2 are authorized on Fold 1/Fold 2 for
KSVD only.  The cache, convex solver (`C=0.1`, balanced L2 logistic regression), one-GINE training,
fixed equal probability average, and exact decision epoch 29 are unchanged.  No epoch or ensemble
weight is selected per seed.

The route is considered robust enough to replace the historical primary only if the pooled six-run
mean is at least `0.700`, exceeds the matched standalone-GINE pooled mean by at least `0.015`, and
wins against its same-run GINE component on at least four of six fold/seed pairs.  Fold means are
reported separately to expose scaffold instability.  Official valid/test remain unevaluated.


## 34. Predeclared convex-probe additive-logit fusion

Section 33 shows that the convex KSVD predictor is seed-stable and complementary to GINE, but its
fixed equal probability average mixes a class-balanced probe probability with an imbalanced-BCE
GINE probability.  The next cheapest check changes only the inference algebra to additive logits:

```text
combined score = GINE logit + frozen balanced-convex KSVD logit
```

Training, cache, parameter count, probe `C=0.1`, and exact epoch 29 are unchanged, so this is a
calibration/fusion check rather than a new representation.  Stage 1 is seed 0 on Fold 1/Fold 2.
Promotion requires mean at least `0.725027` (Section-33 seed-0 mean plus 0.003), neither fold more
than 0.010 below Section 33 (`0.660043/0.764011`), and Fold 2 at least `0.773475`.  Only if all
conditions pass are seeds 1/2 run.  Official valid/test remain unevaluated.


### Seed-robustness result

At fixed epoch 29, the six KSVD ensemble scores are:

| seed | Fold 1 | Fold 2 | mean |
|---:|---:|---:|---:|
| 0 | 0.670043 | 0.774011 | 0.722027 |
| 1 | 0.669880 | 0.754216 | 0.712048 |
| 2 | 0.665728 | 0.782895 | 0.724311 |

The pooled ensemble mean is **0.719462**, versus `0.672710` for the same-run standalone GINE
components and `0.708169` for the deterministic convex KSVD probe.  The ensemble gains
`+0.046752` over matched GINE, wins five of six cells, and exceeds the historical joint-KSVD
pooled mean (`0.689696`) by `+0.029766`.  Fold means are `0.668551` and
`0.770374`.  All predeclared robustness conditions pass.

The top-20 absolute atom/statistic coefficients contain 10 witness-bank features on Fold 1 and 13
on Fold 2.  High-weight terms include sparse-code energy, upper-tail activation, winner frequency,
signed mean, and support frequency.  Because each fold learns its dictionary independently, raw
atom-index coefficient correlation is not interpreted as semantic stability; bank/statistic-level
participation is the valid explanation.

Artifacts:

```text
results/molhiv/backgroundpluswitness_fixedconvexprobavg_seed_robustness_fixed_epoch29_summary.json
results/molhiv/backgroundpluswitness_balancedconvexprobe_interpretability.json
```

### Section-34 Stage-1 result

At fixed epoch 29, additive logits score `0.657162` on Fold 1 and `0.789599` on Fold 2,
mean `0.723381`.  It improves Fold 2 but loses too much on Fold 1 and misses the
predeclared mean/fold gate.  The GINE training trajectories and component AUCs exactly match
Section 33; only inference algebra differs.  Seeds 1/2 are therefore not run, and fixed probability
averaging remains the promoted route.

Artifact:

```text
results/molhiv/backgroundpluswitness_fixedconvexlogitadd_stage1_fixed_epoch29_summary.json
```


## 35. Predeclared outer-fit-only scaffold-CV regularization

The convex KSVD probe is now the strongest single KSVD mechanism, but its `C=0.1` was fixed rather
than inferred from the available training scaffolds.  The next intervention tunes no molecular
feature and does not inspect the outer held-out fold.  For each outer Fold 1/Fold 2, its training
set is exactly the union of the other two cached scaffold groups.  Candidate regularization values

```text
C in {0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
```

are scored by two-way scaffold CV inside that outer-fit set: train on one retained scaffold group,
validate on the other, then reverse.  The highest mean-AUC C is selected with a smaller-C tie break;
only then is the probe refit on all outer-fit graphs and evaluated on the untouched outer fold.
All feature standardization remains outer-fit-only.

Stage 1 runs the standalone KSVD probe on Fold 1/Fold 2.  Promotion to GINE combination requires
outer mean at least `0.711169` (fixed-C mean plus 0.003), Fold 1 at least `0.662221`, and Fold 2 at
least `0.744117`.  Otherwise fixed `C=0.1` remains frozen.  Official valid/test remain unevaluated.


### Stage-1 result

Both outer-fit scaffold CVs select the strongest regularization, `C=0.003`.  Outer held-out AUC is
`0.643998` on Fold 1 and `0.740386` on Fold 2, mean `0.692192`.  This is
`-0.015977` below the fixed `C=0.1` probe.  The retained scaffold groups favor heavy shrinkage, but
that choice does not transfer to the third scaffold group; with only two inner groups the
regularization selector is too unstable.  The route fails before GINE integration, and fixed
`C=0.1` remains frozen.

Artifacts:

```text
results/molhiv/backgroundpluswitness_balancedconvexprobe_scaffoldcv_fold{1,2}_ksvd.json
results/molhiv/backgroundpluswitness_balancedconvexprobe_scaffoldcv_stage1_summary.json
```

A post-change one-epoch replay of the default historical joint path exactly reproduces the prior
Fold-1 seed-0 loss, AUC, JK gates, residual norm/bias, and initialization fingerprints.  The new
fixed-probe/scaffold-CV branches therefore did not alter default behavior.


## 36. Predeclared full-data scaling of the frozen convex-KSVD route

The n=8000 route is now frozen after six-run robustness.  The next intervention changes only data
scale: use all 41,127 MolHIV graphs while constructing development folds exclusively inside the
32,901 official-training graphs.  No architecture, explicit chemical feature, dictionary size,
sparsity, convex regularization, or fusion weight is tuned.

A three-way Bemis--Murcko `StratifiedGroupKFold` is generated with seed `20260726`.  Stage 1 uses
full-data Fold 1 and Fold 2, matching the prior untouched-fold convention.  For each outer fold:

```text
radius = 2, max_nodes = 8, no-ring raw topology patches
background dictionary = KSVD D=8, T=2, 4 iterations, 6000 graph-fit patches
OOF MIL selector = same 3-fold cross-fitted graph-label scorer
witnesses = graph-balanced top-2 patches
witness dictionary = KSVD D=8, T=2, 4 iterations
final encoding = concatenated D=16, joint OMP T=2
readout = 168-D rich sparse/reconstruction descriptor
classifier = balanced L2 logistic regression, fixed C=0.1
```

The first cost-controlled gate is the standalone 169-parameter convex KSVD probe.  Full-data GINE
integration is authorized only if the two-fold probe mean is at least `0.720` and neither fold is
below `0.680`.  If it passes, run one seed per fold with the frozen 3-layer GINE and fixed equal
probability average at epoch 29.  Additional seeds require a combined mean at least `0.740`, a mean
improvement of at least `0.010` over the same-run GINE components, and no fold below `0.700`.

Official valid/test are not encoded for prediction and are not evaluated.  Full-data runtime and
cache sizes are recorded explicitly before any expansion to more seeds or controls.


## 37. Full-data Stage-1 result: KSVD remains real, but the preregistered probe gate narrowly fails

The frozen Section-36 route was run on all `41,127` MolHIV graphs, with model development restricted
to Fold 1 and Fold 2 inside the `32,901` official-training graphs.  The two dictionaries and every
probe were fitted only from each fold's inner-training partition.  Official-valid/test AUCs remain
null with zero evaluations; the corresponding cache token blocks were audited to contain zero
nonzero entries for KSVD, PCA, and random-patch families.

A full-data handling bug in `run_molhiv_sparse_graph_probe.py` was found before scoring: unlike the
other runners, `--max-graphs 0` was passed literally to `load_molhiv`, producing an empty/mismatched
subset.  It now maps non-positive values to `None`; positive historical settings such as `8000` are
unchanged.

### Fixed `C=0.1` convex-probe results

| Family | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|
| KSVD | 0.691398 | 0.740742 | 0.716070 |
| PCA | 0.669223 | 0.736314 | 0.702768 |
| random patch | 0.685759 | 0.702596 | 0.694177 |

Matched full-data margins are:

```text
KSVD - PCA          = +0.013302
KSVD - random patch = +0.021893
```

Thus the self-learned sparse dictionary still has a substantive advantage over both matched linear
and random dictionaries after scaling.  The low Fold 1 also clears the preregistered per-fold floor
of `0.680`.  However, the two-fold KSVD mean is `0.716070`, which misses the predeclared `0.720`
mean gate by `0.003930`.  To preserve the gate's meaning, the full-data GINE/probability-average
integration is not launched from this result.

The complete KSVD path took approximately `1011.65 s` on Fold 1 and `1024.65 s` on Fold 2 including
initial cache construction, MIL selection, witness cache construction, and the final KSVD probe.
Each final three-family witness cache is about `12.1 MB`.

Artifact:

```text
results/molhiv/backgroundpluswitness_n41127_convexprobe_stage1_summary.json
```

Decision: the route is scientifically positive (KSVD beats both controls) but does not pass its
promotion threshold.  Do not post-hoc relax the threshold or run full-data GINE integration.  The
next intervention should address the actual scaling bottleneck: only `6000` background patches were
used despite `575k+` eligible inner-training node patches, representing only about `5.2k` of `23k`
fit graphs.  Any continuation should preregister a larger, graph-coverage-aware dictionary-fit pool
and screen Fold 1 first before paying the two-fold witness-encoding cost.


## 38. Predeclared dictionary-data scaling screen: 24k graph-capped background patches

Section 37 scaled the number of MolHIV graphs but left the background dictionary fit at only 6000
node-uniform patches.  On full-data Fold 1 this represented 5187 of 23150 fit graphs, so dictionary
data coverage did not scale with the dataset.  The next intervention changes only the unsupervised
background-patch sampling budget and coverage:

```text
full-data development Fold = 1 first
radius = 2, max_nodes = 8, no-ring raw patches
background patch pool = 24000 (previously 6000)
maximum candidate centers per graph = 8 (previously unlimited)
background dictionary = D8 KSVD, T2, 4 iterations
OOF-MIL and witness dictionary = frozen Section-36 settings
final dictionary = background D8 + witness D8, joint OMP T2
probe = frozen 168-D descriptor, balanced L2 logistic, C=0.1
matched controls = PCA and random patch throughout
```

This does not add a handcrafted molecular feature, increase atom count, change sparsity, or tune the
classifier.  The graph cap only prevents large molecules from dominating the unsupervised patch
reservoir; increasing the reservoir is intended to preserve substantially more fit-graph coverage.

The Fold-1 promotion rule is frozen before cache construction:

```text
KSVD Fold1 >= 0.705                         -> build and evaluate Fold2
KSVD Fold1 <  0.695                         -> stop
0.695 <= KSVD Fold1 < 0.705                 -> continue only if
                                               KSVD beats both matched controls
                                               and improves over the frozen
                                               Fold1 reference 0.691398 by >= 0.005
```

If Fold2 is authorized, the final two-fold gate remains the original `mean >= 0.720` and
`fold minimum >= 0.680`; it is not relaxed.  Official-valid/test remain unevaluated.


## 39. 24k graph-coverage result: more patches hurt KSVD and fail the Fold-1 stop gate

The predeclared Section-38 Fold-1 screen completed.  Increasing the background reservoir from 6000
to 24000 patches while capping candidate centers at eight per graph increased fit-graph coverage
from `5187 / 23150` to `15552 / 23150`.  Initial cache construction took `455.48 s`, versus
`527.13 s` for the node-uniform 6000-patch reference, because the candidate pool was smaller.

The frozen final probe results are:

| Family | 6000-patch reference | 24k graph-capped | Delta |
|---|---:|---:|---:|
| KSVD | 0.691398 | 0.672097 | -0.019300 |
| PCA | 0.669223 | 0.685261 | +0.016038 |
| random patch | 0.685759 | 0.699334 | +0.013576 |

Matched margins reverse sign:

```text
KSVD - PCA          = -0.013164
KSVD - random patch = -0.027237
```

The neural MIL screening metric did not expose the failure: its best inner AUC increased slightly
from `0.776591` to `0.778660`.  The promoted mechanism is the frozen convex dictionary descriptor,
however, and its KSVD Fold-1 AUC is `0.672097`, below the preregistered `0.695` immediate-stop
threshold.  Fold 2 is therefore not built.

This rules out the simple hypothesis that the full-data bottleneck is merely too little graph
coverage.  With only eight atoms, the more diverse 24k patch distribution likely increases the
number of modes the dictionary must represent; PCA and random prototypes benefit from broader
coverage, while four-iteration D8 KSVD no longer finds the discriminative sparse partition that was
present in the smaller reservoir.  This is an interpretation rather than a new selection result.
No post-hoc iteration/seed tuning is performed on this fold.

Official-valid/test AUCs remain null with zero evaluations, and their KSVD/PCA/random token blocks
were audited as all-zero.

Artifact:

```text
results/molhiv/backgroundpluswitness_n41127_p24000_graphcap8_fold1_screen_summary.json
```

Decision: stop the 24k graph-capped route.  Do not expand Fold 2.  Future work should not simply add
more patches; it needs a KSVD-specific way to preserve a compact, informative patch distribution
(e.g. learned coverage/coreset selection defined entirely inside the fit split) before another
full-data encoding run is justified.


## 40. Predeclared exploratory full-data ceiling audit: frozen KSVD probe plus one GINE

Section 37 did not pass its preregistered Stage-1 mean gate, so the following run is **not** a
promotion result and cannot retroactively change that decision.  Its purpose is narrower: quantify
whether the already-frozen KSVD mechanism plus a small GINE has enough development-fold ceiling to
justify further architecture work aimed at CIN-scale performance.

The audit uses the original 6000-patch full-data caches, not the failed Section-39 24k route:

```text
background KSVD D8 + OOF witness KSVD D8, joint OMP T2
168-D rich reconstruction descriptor
balanced L2 convex probe, fixed C=0.1, frozen
one independently optimized 3-layer h64 GINE
fixed 0.5/0.5 probability average
29 epochs, seed 0, Fold 1 and Fold 2
37,551 total predictor parameters
```

Only epoch 29 is reported; best-epoch selection is ignored.  Official-valid/test remain
unevaluated.  This audit is considered architecturally encouraging only if:

```text
combined two-fold mean >= 0.760
combined - same-run GINE mean >= 0.010
neither combined fold < 0.720
```

Failure means the present late-fusion family has insufficient headroom for a credible CIN target;
future work must move KSVD into a learned higher-order cell/message-passing mechanism rather than
continue probe/fusion tuning.

## 41. Fixed-epoch ceiling audit result: late fusion fails and underperforms the same-run GINE

The two Section-40 runs completed through the preregistered epoch 29.  The fixed 0.5/0.5
probability average does not reveal hidden CIN-scale headroom:

| Fold | GINE | frozen KSVD probe | fixed probability average |
|---|---:|---:|---:|
| 1 | 0.750902 | 0.691397 | 0.717627 |
| 2 | 0.770292 | 0.740742 | 0.769369 |
| mean | 0.760597 | 0.716070 | 0.743498 |

The combined mean is `-0.017099` below the same-run GINE mean, and Fold 1 is `0.717627`.
Consequently all three exploratory gates fail:

```text
combined mean >= 0.760                    false (0.743498)
combined - same-run GINE mean >= 0.010    false (-0.017099)
minimum combined fold >= 0.720            false (0.717627)
```

The best intermediate epochs are not substituted for epoch 29.  They were visible only as training
telemetry and are outside the frozen reporting rule.  Official-valid and official-test AUCs remain
null with zero evaluations in both artifacts.

Decision: stop tuning convex-probe/GINE late fusion.  The standalone KSVD signal is real relative to
its matched PCA/random-patch controls, but it is too correlated with and weaker than GINE to close
the gap by averaging predictions.  The next route must make sparse KSVD assignments participate in
message passing: dictionary atoms define data-learned motif/cell types, patch occurrences define
soft node-to-cell incidences, and alternating node-to-cell / cell-to-node propagation supplies a
higher-order interaction mechanism without explicit ring or scaffold features.

Artifact:

```text
results/molhiv/backgroundpluswitness_n41127_fixedconvexprobavg_richrecon_section40_summary.json
```

## 42. Predeclared learned-cell recurrent screen: move KSVD into higher-order propagation

Section 41 rules out late prediction fusion.  The next screen changes the computation graph rather
than the graph descriptor.  It contains no explicit ring, scaffold, or fingerprint feature.

Each atom-centered radius-2 patch is lifted to one learned cell.  Its boundary is the radius-2 ego
incidence, and its type is the signed/absolute normalized sparse assignment to the fold-local
background-plus-OOF-witness dictionary.  After every molecular GINE layer, the model alternates:

```text
boundary atoms -> KSVD-typed patch cell -> boundary atoms
```

Cell state is recurrent across the three alternations.  Overlapping cells therefore communicate
through shared boundary atoms.  This is a hypergraph/cell transport, not an independently trained
second GINE branch.  Graph-distance incidence weights are learned from a monotone initialization;
there are no molecularly handcrafted cell labels.  The branch uses a rank-16 bottleneck and is
expected to remain far below CIN-sized parameterization.

Fast architecture screen (official-train scaffold Fold 0 of the 8k development subset):

```text
cache = no-ring OOF-MIL background+witness D16/T2
backbone = matched h64/l3 GINE-JK
30 epochs, seed 0
cell gate = exact-GINE warmup for 5 epochs, then fixed wake at raw 0.1
cell scale = 0.25
families = KSVD / PCA / random patch / KSVD no-type-ID
```

Promotion to the 41,127-graph folds requires all of:

```text
KSVD best inner AUC >= 0.762
KSVD - matched GINE >= 0.008
KSVD beats PCA, random patch, and no-type-ID cell controls
```

This pilot may select its best inner epoch because it is explicitly an architecture triage screen.
Official-valid/test remain unevaluated.

## 43. Recurrent learned-cell result: generic hypergraph capacity dominates KSVD identity

The Section-42 fold-0 screen failed its promotion gate:

| family | best inner AUC | selected epoch |
|---|---:|---:|
| KSVD learned cells | 0.739095 | 27 |
| PCA learned cells | 0.748959 | 25 |
| random-patch learned cells | **0.754511** | 28 |
| KSVD cells without type ID | 0.737729 | 26 |
| matched historical GINE | 0.752248 | 30 |

KSVD misses the `0.762` threshold, is `-0.013153` below GINE, and loses to both matched
dictionary controls.  The model has `43,515` trainable parameters (`6,133` in the cell branch), so
the failure is not caused by a large parameter increase.  Repeatedly broadcasting every overlapping
radius-2 cell appears to oversmooth/dilute the specific sparse identity signal; random cells recover
generic extra capacity best.

Decision: do not scale this recurrent broadcast to full data.  Preserve learned cells but remove
the destructive cell-to-atom injection.  The next screen will let final GINE atom states populate
only OOF-witness-weighted KSVD cells and jointly read those cells at graph level.  This tests whether
higher-order witness aggregation is complementary without perturbing molecular message passing.
Official-valid/test remain unevaluated.

## 44. Predeclared OOF-witness learned-cell readout screen

The next fold-0 screen keeps the Section-42 cache/backbone/training budget but removes all
cell-to-atom broadcasts.  Final GINE atom states populate radius-2 cells; graph readout is computed
only over cells in proportion to their sparse mass on the OOF-witness half of the learned dual bank.
The cell graph vector is trained jointly with the molecular graph vector under one prediction head.
Thus this is feature-level higher-order aggregation, not post-hoc prediction averaging.

Controls use exactly the same architecture with PCA, random-patch, or removed dictionary-type ID.
The promotion gate is unchanged from Section 42 (`KSVD >= .762`, at least `+.008` over matched GINE,
and wins against all three controls).  Official-valid/test remain unevaluated.

## 45. Witness-cell readout result and return to the only replicated KSVD-specific transport

Section 44 also fails promotion:

| family | best inner AUC |
|---|---:|
| KSVD witness-cell readout | 0.754480 |
| PCA witness-cell readout | 0.748206 |
| random-patch witness-cell readout | **0.758131** |
| KSVD readout without type ID | 0.752047 |
| matched historical GINE | 0.752248 |

KSVD improves only `+0.002232` over GINE and `+0.002433` over its no-ID control, but misses `0.762`
and loses to random patch.  Removing destructive broadcast helps, but witness-bank mass alone still
admits a generic learned-patch explanation.

The only architecture in the project with replicated KSVD-specific evidence remains the original
atom-to-dictionary-slot-to-atom transport: on 8k, its frozen 3-fold x 3-seed study gave `+0.00370`
mean over matched GINE, 6/9 wins, and KSVD beat shuffled/no-ID controls on all three folds.  It has
not yet been tested with the stronger no-ring background-plus-OOF-witness dictionary on all 41,127
graphs.  This is therefore the next higher-value scale check, rather than another new cell variant.

Predeclared full-data Fold-1 screen:

```text
cache = original 6000-patch no-ring background D8 + OOF-witness D8, joint OMP T2
architecture = one h64/l3 GINE-JK with one atom->dictionary-slot->atom pass after layer 1
cell rank = 16; learned atom IDs; dense exact pilot backend
30 epochs; seed 0; warmup 10; gate scale .25
matched families = KSVD / PCA / random patch / KSVD no atom ID
```

Fold 2 is authorized only if KSVD Fold 1 is at least `0.755`, improves over the same-run/matched GINE
reference by at least `0.003`, and beats all three controls.  This screen is about whether KSVD cell
transport scales; it is not an official-test decision.  Official-valid/test remain unevaluated.

## 46. Full-data slot screen result and exploratory early-stopped KSVD refinement

The Section-45 Fold-1 runs were restarted sequentially after four concurrent dense-slot jobs
exceeded the practical memory budget near epoch 25.  The completed primary comparison is:

| family | best inner AUC | epoch |
|---|---:|---:|
| KSVD slot transport | 0.771514 | 26 |
| random-patch slot transport | **0.779741** | 26 |

KSVD clears the absolute `0.755` screen threshold and reaches the strongest full-data KSVD neural
score so far, but the predeclared control gate fails by `-0.008227`; Fold 2 is not authorized.
The unfinished concurrent PCA/no-ID telemetry is not used as a result.

This exposes a precise dictionary-side failure: the unchanged random witness prototypes create a
better slot partition for neural aggregation than the four-step reconstruction-optimized KSVD
witness atoms.  The next exploratory screen therefore treats KSVD iteration count as regularization,
not extra downstream capacity.  On 8k Fold 0, witness banks with one and two standard KSVD updates
will be compared with the existing four-update and zero-update/random endpoints under the same
slot architecture.  Only an intermediate KSVD model that exceeds both endpoints and `0.762` is
eligible for a full-data rebuild.  No official-valid/test evaluation is allowed.

## 47. Early-stopped witness KSVD result: reconstruction refinement is not the missing regularizer

Section 46 hypothesized that the full-data random-patch advantage might arise because four
reconstruction updates erase a useful task partition.  The preregistered 8k Fold-0 screen therefore
kept the witness pool, background bank, final D16/T2 OMP, motif-slot architecture, model seed, and
training budget fixed while changing only the number of witness-bank KSVD updates.

| witness dictionary endpoint | best inner-valid AUC |
|---|---:|
| 1 KSVD update | 0.742194 |
| 2 KSVD updates | 0.744720 |
| 4 KSVD updates | **0.752109** |
| zero-update random-patch endpoint | 0.751078 |

The intermediate endpoints fail the predeclared rule: neither exceeds both endpoints or reaches
`0.762`.  Performance increases monotonically from one to four KSVD updates, and the four-update
KSVD narrowly exceeds random patch by `0.001031` on this fold.  Thus ordinary early stopping is not
a useful dictionary regularizer and does not explain the full-data Fold-1 random-patch advantage.
No full-data one/two-update cache will be built.

The result sharpens the bottleneck.  The successful neural slot architecture already has enough
capacity to reach `0.779741` with sampled prototypes, while reconstruction-only KSVD reaches
`0.771514` on the same full-data fold.  The next experiment must change the KSVD objective itself:
fit sparse atoms to the identical cross-fitted OOF-MIL witness patches while adding a continuous
OOF witness-evidence consistency term.  Unlike the old failed patch-label augmentation, no molecule
label is copied to all of its patches; only OOF-selected witnesses receive the score produced by a
model that was not fitted on their molecule.  The chemistry block remains the deployed dictionary,
and matched PCA/random controls receive the identical augmented witness matrix.

Artifacts:

```text
results/molhiv/motifslot_oofbw_kiter1_fold0_ksvd_seed0.json
results/molhiv/motifslot_oofbw_kiter2_fold0_ksvd_seed0.json
results/molhiv/motifslot_oofbw_kiter4_fold0_ksvd_seed0.json
results/molhiv/motifslot_oofbw_kiter0random_fold0_seed0.json
```

## 48. Predeclared continuous OOF-witness-consistent KSVD screen

The next screen changes the offline sparse-dictionary objective, not the GNN/readout.  For each of
the same 608 graph-balanced OOF-MIL-selected witnesses on 8k Fold 0, define two continuous targets:
(1) its cross-fitted MIL instance score and (2) its within-molecule standardized score prominence.
Both are produced by a scorer that was not fitted on that molecule.  They are standardized over the
fit-only witness pool, clipped at three standard deviations, and appended during dictionary fitting:

```text
Y_aug = [Y_chem ; 0.20 * q_oof_score ; 0.20 * q_within_graph_prominence]
```

KSVD/PCA/random-patch controls receive the identical augmented matrix, atom count, T=2, four update
budget, initialization seed, and witness identities.  After fitting, the target rows are discarded,
the chemistry rows of every atom are renormalized, and only those chemistry atoms are used for all
node encodings.  Thus inference receives no score, molecule label, ring, scaffold, or fingerprint.
This differs from the old patch-label augmentation: graph labels are not copied to patches, and only
cross-fitted selected witnesses have consistency targets.

Stage A is one 30-epoch seed-0 motif-slot run on 8k Fold 0 with the unchanged h64/l3 GINE-JK,
10-epoch exact-GINE warmup, rank-16 slot bottleneck, and gate scale 0.25.  It advances to matched
PCA/no-ID controls only if KSVD reaches `0.758` and improves over the Section-47 four-update KSVD by
at least `0.004`.  A full-data Fold-1 build is authorized only if KSVD reaches `0.762`, beats the
unchanged random-patch endpoint by at least `0.004`, and beats both PCA and no-ID controls.  No
weight sweep is allowed on this fold.  Official-valid/test remain unencoded and unevaluated.

## 49. Continuous OOF-witness consistency fails Stage A

The Section-48 cache passed all leakage and matching audits.  Its two continuous targets have unit
fit-pool variance before clipping; the KSVD witness sparse codes explain `0.3193` of their training
variance, versus `0.2146` for PCA and `0.1578` for random patch.  Thus the augmented objective did
make KSVD assignments more aligned with the OOF witness evidence offline.

That alignment did not transfer to the neural scaffold fold.  The single preregistered KSVD
motif-slot run reached only `0.743313` (epoch 27), compared with `0.752109` for ordinary four-update
KSVD.  It misses the `0.758` Stage-A threshold and loses `0.008796`; PCA/no-ID controls and any weight
sweep are therefore not run.  Official-valid/test entries remain zero and official AUCs remain null.

This result rejects direct target-row augmentation for the current sparse slot model.  Even a
continuous, cross-fitted witness target can rotate reconstruction atoms toward a small noisy scorer
at the expense of the chemical partition needed by message passing.  The route is stopped.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_oofconsistency_w020_fold0.{npz,json,log}
results/molhiv/motifslot_oofconsistency_w020_fold0_ksvd_seed0.{json,log}
```

## 50. Predeclared dual-bank hierarchical pursuit screen

A direct assignment audit reveals a more concrete explanation for the full-data random-patch
advantage.  Under the current joint D16/T2 OMP on 8k Fold 0:

| family | nodes activating witness bank | witness coefficient-mass fraction | both banks active |
|---|---:|---:|---:|
| KSVD | 0.4653 | 0.3279 | 0.3245 |
| PCA | 0.6088 | 0.4345 | 0.3877 |
| random patch | **0.7683** | **0.4911** | **0.5659** |

KSVD background atoms reconstruct common chemistry well enough that joint OMP often spends both
coefficients before reaching the OOF-witness bank: `53.47%` of KSVD-coded nodes are background-only,
versus `23.17%` for random patch.  Consequently the supposedly task-selected witness cells are absent
on more than half of KSVD node incidences.  This is a sparse-inference bottleneck, not a GNN capacity
or dictionary-width bottleneck.

The next cache keeps every fitted dictionary atom fixed and changes only pursuit:

```text
background code: OMP(D_background, y, T=1)
residual:        r = y - D_background a_background
witness code:   OMP(D_witness, r, T=1)
final code:     [a_background ; a_witness]  # exactly two possible nonzeros
```

This is a two-stage KSVD decomposition with an interpretable common-chemistry then task-witness
residual, not a handcrafted feature.  KSVD/PCA/random controls use their matched background and
witness banks with identical one-plus-one budgets.  Dictionaries, selected witnesses, GINE,
motif-slot architecture, parameters, seed, and runtime budget are unchanged.

Stage A is one 30-epoch 8k Fold-0 KSVD run.  It advances to matched PCA/random/no-ID controls only if
it reaches `0.758` and improves over joint-OMP KSVD by at least `0.004`.  Full-data Fold 1 is
authorized only if KSVD reaches `0.762`, beats all controls, and retains at least `+0.004` over the
joint-OMP KSVD.  Official-valid/test remain untouched.

## 51. Hierarchical one-plus-one pursuit does not improve the KSVD slot model

The hierarchical cache behaved as designed: every nondegenerate node received one background and
one witness coefficient, raising KSVD witness activation from `0.4653` to `1.0000`.  However,
witness coefficient mass remained only `0.2454` because residual coefficients were usually smaller
than the background coefficient.

The preregistered KSVD motif-slot run reached `0.751222` at epoch 21.  This is `-0.000887` below the
joint-OMP KSVD comparator (`0.752109`) and misses the `0.758` gate, so PCA/random/no-ID controls and
full-data scaling are not run.  Forcing witness incidence is therefore not enough: the slot model
already weights incidences by coefficient magnitude, and weak residual witness assignments do not
supply a stronger partition.  The historical joint pursuit remains preferred.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_hierarchical1plus1_fold0.{npz,json,log}
results/molhiv/motifslot_hierarchical1plus1_fold0_ksvd_seed0.{json,log}
```

## 52. Predeclared empirical-anchor proximal KSVD screen

The strongest full-data slot result still comes from unchanged sampled witness prototypes
(`0.779741`), while ordinary KSVD improves reconstruction but reaches `0.771514`.  Early stopping,
ambient incoherence, max-min initialization, larger banks, and forced witness pursuit have all
failed.  The remaining precise hypothesis is that atom updates should denoise prototypes without
letting them abandon the empirical patch partition that benefits the neural slot model.

After every standard K-SVD dictionary sweep, each witness atom is sign-aligned to its own initial
sampled patch and receives a proximal anchor step:

```text
d_j <- normalize((1-rho) d_j + rho sign(<d_j,d_j^0>) d_j^0),  rho = 0.25
```

Sparse codes are recomputed after the proximal step.  This is still a learned KSVD dictionary: SVD
atom updates and sparse reconstruction determine 75% of every post-update direction, while the
regularizer keeps learned atoms near real OOF witnesses.  It adds no neural parameters or inference
features.  PCA/random controls remain unchanged on the identical witness pool.

One non-swept 8k Fold-0 setting is used: four updates, rho `0.25`, original joint D16/T2 OMP, and the
same 30-epoch motif-slot model.  It advances to controls only if AUC is at least `0.758` and at least
`0.004` above ordinary KSVD.  Full-data Fold 1 requires `0.762` plus wins over random/PCA/no-ID.
Official-valid/test remain untouched.

## 53. Empirical-anchor proximal KSVD fails Stage A

The rho-0.25 anchor behaved as intended offline: after four updates, mean absolute cosine to the
initial sampled witness atom was `0.8710` (minimum `0.6559`), while witness reconstruction remained
better than the random endpoint.  Matched PCA/random dictionaries and every corresponding node code
were bitwise unchanged from the historical cache.

The KSVD motif-slot run reached `0.748747` at epoch 23, `-0.003362` below ordinary KSVD and below the
`0.758` gate.  Controls and full-data scaling are not run.  Preserving sampled-prototype identity by
a fixed proximal atom constraint therefore does not recover the random-patch neural advantage.
Together with Sections 46--53, this stops dictionary-objective/initialization micro-variants on the
current D8 witness bank.

Artifacts:

```text
results/molhiv/node_tokens_n8000_a16_t2_noring_oofmil_anchoredksvd_rho025_fold0.{npz,json,log}
results/molhiv/motifslot_anchoredksvd_rho025_fold0_ksvd_seed0.{json,log}
```

## 54. Predeclared modest-width ceiling screen

The dictionary variants are now saturated, but the successful architecture is still extremely
small.  Before declaring the KSVD slot route incapable of CIN-scale development AUC, test one
capacity increase that preserves the mechanism: widen the *single* three-layer GINE and its existing
KSVD slot transport from hidden 64 to hidden 96.  No layer, second GINE branch, handcrafted feature,
dictionary change, or official split is added.  This tests whether h64 is underfitting the joint
atom/slot representation while retaining a compact model.

Stage A uses the original four-update background+witness D16/T2 cache on 8k Fold 0, 30 epochs, seed
0, rank-16 motif slot, 10-epoch warmup, and gate scale 0.25.  KSVD advances to same-width GINE and
random-patch controls only if it reaches `0.760`.  Full-data Fold 1 is authorized only if KSVD is at
least `0.765`, exceeds same-width GINE by `0.004`, and beats random patch.  This is one width point,
not a width sweep.  Official-valid/test remain untouched.

## 55. Modest h96 width raises the backbone ceiling but does not activate KSVD

The single preregistered Section-54 KSVD run completed on 8k Fold 0.  It reached `0.756857` at
epoch 28, a `+0.004749` increase over the h64 four-update KSVD slot run (`0.752109`), but below the
predeclared `0.760` control gate.  The model has `81,303` trainable parameters (`74,502` backbone,
`6,801` slot branch), still compact in absolute terms.

More importantly, the final effective motif-slot gate was only `-0.000298`.  Thus widening mostly
raises the ordinary GINE-JK capacity; it does not make the KSVD transport materially more active.
Matched h96 GINE/random/no-ID controls and full-data scaling are therefore not run under the
predeclared rule.  Width is not the missing KSVD mechanism, and further width/depth sweeps are
stopped.

Artifact:

```text
results/molhiv/motifslot_h96_oofbw_fold0_ksvd_seed0.{json,log}
```

## 56. Predeclared OOF-witness local-cell screen

Section 55 shows that width improves the ordinary backbone while the global dictionary-slot gate
remains effectively dormant.  Before replacing the offline KSVD pipeline, test a different use of
the already-frozen background+witness dictionary that has not been evaluated on this cache: each
atom-centered radius-2 patch becomes one local learned cell, typed by its dominant sparse KSVD atom,
with one atom->cell->atom pass after GINE layer 1.  Unlike the global slot, separate occurrences of
the same atom remain local; unlike CIN, no ring or handcrafted cell is supplied.  This is the
previously established top-1 ego transport applied to the stronger OOF-witness dictionary, not a
new capacity sweep.

Stage A is one h64/l3 GINE-JK, 30-epoch, seed-0 run on 8k Fold 0 using radius 2, scalar exact-GINE
initialization, warmup 10, and scale 0.25.  Matched random/PCA/no-ID controls are run only if KSVD
reaches `0.762`.  Full-data Fold 1 requires KSVD at least `0.765`, a `+0.004` gain over matched GINE,
and a win over random patch.  Official-valid/test remain untouched.

## 57. Unfiltered OOF-witness local cells fail Stage A

The Section-56 KSVD top-1 radius-2 local-cell model reached only `0.748742` at epoch 27, below the
`0.762` gate and below the current matched GINE reference (`0.752248`).  Thus simply preserving one
cell per atom does not make the background+witness dictionary useful; controls and full-data scaling
are not run.

An assignment audit gives one remaining non-micro local-cell hypothesis.  Only `33.34%` of fit nodes
are dominated by a witness-bank atom.  Requiring witness dominance at least `0.80` leaves about
`4.48` learned cells per molecule (17.73% of nodes), with nearly identical held-out statistics
(`4.41` cells/molecule, 17.34%).  This offers a data-learned sparse cell complex rather than the
oversmoothing all-center complex.  The next screen will implement this exact filter once, with no
threshold sweep.

Artifact:

```text
results/molhiv/motifego_oofbw_fold0_ksvd_seed0.{json,log}
```

## 58. Predeclared sparse witness-cell complex screen

The final local-cell screen retains only centered radius-2 occurrences whose dominant atom lies in
the OOF-witness bank (`atom index >= 8`) and whose sparse-code dominance is at least `0.80`.  The
threshold was fixed from the Section-57 assignment audit and will not be swept.  This removes common
background cells and weak ambiguous assignments, leaving roughly 4.5 data-learned cells per graph.
No ring/cycle/scaffold identity or label is supplied at neural inference.

To test the cell mechanism rather than permit another dormant scalar gate, the residual uses the
existing exact-GINE zero-output initialization, warmup 10, and conservative scale `0.05`.  The final
broadcast matrix is zero at initialization and receives direct gradients after warmup.  Stage A is
one h64/l3, 30-epoch, seed-0 KSVD run on 8k Fold 0.  Controls run only if AUC reaches `0.760` and is at
least `+0.004` over matched GINE.  Full-data Fold 1 additionally requires a win over matched random
patch and no-ID controls.  Official-valid/test remain untouched.

## 59. Sparse witness-cell complex is stable but not an accuracy breakthrough

The preregistered witness-only, dominance-0.80 local-cell run reached `0.752410` at epoch 23.  It
recovers the matched GINE level but misses the `0.760` gate and does not justify controls or scaling.
Sparse selection prevents the loss of the all-center cell model, but the local-cell computation does
not provide the missing CIN-scale gain.

The next test targets a different diagnosed defect in the only full-data architecture that reached
`.77`: the motif-slot branch uses one scalar gate for all 16 dictionary atoms, and that gate ends
near zero.  Background and witness atoms can demand opposite residual directions, causing gradient
cancellation.  Replace the single scalar by 16 zero-initialized atomwise gates inside the slot
broadcast.  This adds only 16 interpretable parameters and preserves exact-GINE initialization;
it is not a width, dictionary, or handcrafted-topology sweep.

Artifact:

```text
results/molhiv/motifego_sparsewitness_d080_fold0_ksvd_seed0.{json,log}
```

## 60. Predeclared atomwise-gated KSVD slot screen

The original global slot transport is retained, but its one scalar residual gate is replaced by one
zero-initialized gate per learned dictionary atom.  At initialization every atom gate is zero, so
the predictor is exactly matched GINE.  After the 10-epoch warmup, each atom can independently turn
on, turn off, or reverse its slot message.  This directly tests whether cancellation between common
background atoms and OOF-witness atoms caused the near-zero global gate.  It adds only 16 trainable
scalars and exposes their values for interpretation.

Stage A uses the original h64/l3, D16/T2 motif-slot setup on 8k Fold 0, 30 epochs, seed 0, rank 16,
and gate scale 0.25.  KSVD advances to matched random/PCA/no-ID atomwise controls only if it reaches
`0.760` and improves over matched GINE by at least `0.004`.  Full-data Fold 1 additionally requires
KSVD to beat random patch.  No gate sparsity or scale sweep is allowed.  Official-valid/test remain
untouched.

## 61. Atomwise slot gates reject the cancellation hypothesis

The atomwise-gated KSVD slot model reached only `0.744189` at epoch 21, below both the original
scalar-slot result (`0.752109`) and matched GINE (`0.752248`).  The 16 independent gates do move away
from zero, so the screen successfully removes global gradient cancellation, but this exposes a more
fundamental problem: actively injecting the present KSVD slot messages hurts held-out scaffold
ranking.  Controls and full-data scaling are not run.

Together, Sections 55--61 close the remaining low-cost architecture questions.  More backbone width,
all-center local cells, sparse witness-only cells, and atomwise slot selection do not convert the
offline reconstruction dictionary into CIN-scale predictive structure.  Further gate/radius/width
micro-sweeps are stopped.  The next credible attempt must change the learning formulation itself:
a task-adapted dictionary/code layer initialized by KSVD and constrained by local-patch
reconstruction, with matched frozen-KSVD and random-initialized controls.

Artifact:

```text
results/molhiv/motifslot_atomwise_oofbw_fold0_ksvd_seed0.{json,log}
```
