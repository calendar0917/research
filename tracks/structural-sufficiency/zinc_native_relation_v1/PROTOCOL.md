# PROTOCOL — zinc_native_relation_v1

Protocol id: `ssrc-zinc-native-relation-v1`
Track: `structural-sufficiency`
Status: **FROZEN before the first formal GPU training run.** This file, the
config, the code and the CPU checks are committed together. Any change to the
model, data contract, null construction, recipe, decision rule or budget
invalidates this protocol and requires a new id.

The official ZINC test split is **not** loaded by prep, checks, timing, training,
scoring or the main analysis. It is read once, at most, under the terminal rule
in §10.

## 0. Question and scope

Final performance target remains official ZINC subset raw-`y` **test** MAE
`< 0.060` (0.05x or lower). This round does not promise to reach it. It asks one
falsifiable question:

> With the original P3 pre-training computation graph and an unnormalised
> vector-magnitude main path preserved, does **one real structure-relation
> interaction** produce a repeatable improvement on unseen local environments
> and on overall valid error, relative to a same-round P3 reference and a
> parameter-identical shuffled-correspondence control?

Three propositions are kept separate:

* **H1 (R vs P):** candidate R improves overall and/or the fixed 225-molecule
  unseen+ordinary group relative to the same-round P3 reference P, repeatably
  across supervised seeds.
* **H2 (R vs N):** with identical parameters, initialisation, optimiser, data
  order, pair set and relation-label multiset, the correct relation-type ↔
  object-pair correspondence beats a fixed within-stratum label permutation N.
* **H3 (test):** the frozen single-model candidate reaches official test
  `< 0.060`. This is evaluated only if the terminal trigger fires.

`R − P` contains extra capacity/compute as well as interaction and therefore
cannot by itself establish H2. `R − N` controls parameters, initialisation,
optimiser, data order and the candidate-pair set; it does **not** test the value
of the pair set itself, and it does not prove that message passing is necessary.

This round does **not** re-run the old HR normalisation design. Nothing here can
claim that LayerNorm caused the old HR degradation, in either direction. It also
does not claim that preserving SSL plus one attention is a new contribution.

## 1. Data and frozen inputs

* `PyG ZINC(root=..., subset=True)`, official train 10,000 / valid 1,000.
* Inputs are the frozen P3 atom/bond/junction objects (`ad_train.pt`,
  `ad_valid.pt` from `zinc_arch_discovery_v1`) plus relation arrays between the
  tokens that already exist in P3. The P3 object cache is label-blind and its
  atom/bond/junction ordering is verified against the old HR cache.
* Atom feature 108-d, bond feature 7-d, junction feature 32-d, global-33. The
  global-33 standardiser is the original P3 train-only one
  (`ad_data.load_norm()`), used unchanged by all three arms.
* No SMILES, logP/SA, target cycle penalty, `y`-derived feature, row id,
  molecule id, split-local id or exact local-environment embedding.
* No new fitted statistic is introduced. The relation construction has no
  fitted parameters.
* Labels are raw `y`; the loss is raw-`y` L1. No target decomposition, oracle,
  external supervision or label-generator input.
* SSL initialisation:
  `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
  SHA-256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`.
  It is a train-only masked atom/bond reconstruction checkpoint with no labels.
  **Both supervised seeds 0 and 1 use this same checkpoint**, so the seed spread
  covers supervised training only, not SSL variance.
* No supervised P3/HR checkpoint is loaded. Historical checkpoints are used
  only for engineering comparison, never as a formal initialisation.

## 2. Tokens and relations

For every molecule, after the original P3 encoders produce
`(a0, b0, w0) = P3Encoders(batch)`, form one token per **existing** junction
and one per **original bond**:

```
token 0 .. n_junc-1            anchor = junction centre atom
token n_junc .. n_junc+nb-1    anchor = bond endpoints (unordered pair)
h0 = concat(w0, b0)            # junction first, bond second, both 96-d
```

No patch encoder, no new object family, no recursive hidden-state update.

Candidate pairs are the already-verified HR pair set: unordered token pairs with
clipped anchor distance `<= 3` **or** a shared biconnected ring block. Both
directions are expanded; every token also receives one self loop.

```
rel = 1 + 8*min(distance,3) + 4*same_ring_block + min(A^2,3)
self rel = 0
```

Distance is the P3 cached all-pairs graph distance; `A^2` is the unweighted
2-walk count. The relation construction does not read labels. The index mapping
from HR patch-to-root-bond to this bond token is verified by rebuilding the HR
relation arrays from the frozen P3 objects and comparing them row by row.

## 3. Models

All arms have no dropout, FP32, `<= 500,000` parameters.

### P — same-round P3 reference

`zinc_recipe_trunk_v2.code.ad2_models.P3NullModel`, instantiated verbatim:
encoders → Null `group_to_bonds` → original group composer → original reader.
Expected 413,505 parameters.

### R — native graph + one relation update

The first stage is exactly P3:

```
a0, b0, w0 = P3Encoders(batch)          # no LayerNorm/refine/relation before w0
h0 = concat(w0, b0)
u  = LayerNorm(h0 + role_embedding)     # only the relation branch is normalised
q, k, v = Wq u, Wk u, Wv u              # rank 16, no bias
score_ij = q_i . (k_j + e_rel_ij)/sqrt(16) + bias_rel_ij
alpha = segment_softmax_j(score_ij)      # segments are source tokens, fixed pairs
message_i = sum_j alpha_ij v_j
h1 = h0 + Wo message_i                   # no LayerNorm after h1
w1, b1 = h1[:, :96], h1[:, 96:]
```

Then the original P3 vector path is reused unchanged:

```
W, V, n_b = P3.group_to_bonds(w1, batch)
c = P3.group_composer([b1, W, V, log1p(n_b), original bond_feat])
z = [sum_b c, sum_a a0, sum_a a0^2, standardised global33]
y_hat = P3.reader(z)
```

There is one interaction round, no iterative object attention, no full pair
tensor, no zero-init learnable gate. `Wo` has weight std `1e-3`, bias 0.
`role_embedding` has only two rows (junction/bond), no molecule/row/local-key
embedding. `h1` is not normalised; the main path is not forced into a
normalised representation.

### N — shuffled correspondence control

Identical to R in architecture, parameters, initialisation, optimiser, data
order, candidate pairs, self loops and per-molecule relation frequency. The only
difference is the **relation-type ↔ object-pair correspondence**:

* within each molecule and each undirected token-role stratum (JJ / JB / BB),
  permute the labels of non-self pairs with a fixed RNG (`20261107`);
* both directions of a pair receive the same permuted label;
* self loops stay relation 0;
* if a random permutation is the identity for a group of size ≥ 2, a fixed
  cyclic shift is used instead;
* the mapping is cached once; it never reads `y`, predictions or errors and is
  not resampled per epoch.

The actual changed-label fraction, number of permutable groups, homogeneous
groups and per-group relation entropy are computed and reported. A global rel-id
rename is **not** used (it would be absorbable by permuting the embedding rows).

## 4. Initialisation and execution sharing

For each supervised seed:

1. Construct the common P3 modules under `torch.manual_seed(seed)`, in the
   original `P3NullModel` order: encoders, group composer, reader, buffers.
2. Load the shared SSL encoder above into the atom/bond/junction encoders.
3. Construct the relation module from an independent generator seeded by
   `80261107 + seed`; its construction does not consume the common RNG stream.
4. R and N therefore have identical relation-module initial values as well as
   identical common values.

P is built by the same common construction. The CPU gate compares
(a) our P state to the mature `ad2_models.build_model("P3N", ..., seed=0)`
state, (b) P/R/N common-state hashes, and (c) R/N relation-state hashes.

Training order is the mature trainer's `torch.randperm` with generator seed
`seed + 101`; the relation collate never consumes RNG. The order stream is
replayed and hashed in every run record. Shared initialisation does not imply
bitwise-identical CUDA trajectories; no "noise floor" is asserted.

## 5. Recipe (unchanged trainer)

`zinc_arch_discovery_v1.code.ad_train` with `RECIPE_VARIANTS["B"]`, unchanged:

* AdamW, peak LR `1e-3`, minimum `1e-5`, weight decay `1e-5`,
  betas `(0.9, 0.999)`, eps `1e-8`;
* 500-step linear warm-up then per-step cosine;
* gradient clip 5.0, batch 128, FP32, dropout 0;
* 600 epochs, 79 steps/epoch (last partial batch retained) = 47,400 updates;
* valid every epoch; selection `argmin(valid_mae)` with strict `<`, ties keep
  the earlier epoch;
* epoch 600 is a pre-declared secondary checkpoint. No soup, checkpoint
  average or seed-prediction average.

No AMP, reduced batch, truncated schedule or data subsampling is allowed for the
formal runs.

## 6. Mandatory implementation checks before formal training

All checks are CPU and written to `analysis/cpu_checks.json`; the run stops if
any gate fails.

1. **Native encoder replay:** on real data, R/N `a0,b0,w0` match P3 layer by
   layer (tolerance `1e-6`, actual value reported).
2. **Common state:** P/R/N common-state hashes equal; R/N relation-state hashes
   equal; SSL hash equal; parameter counts recorded.
3. **Bypass replay:** forcing the relation message to zero makes R/N predictions
   match P within `1e-6`. This is a check only, not a result arm.
4. **Invariances:** atom renumbering, bond-direction reversal, batch-vs-single
   and a synthetic no-bond / disconnected / one-bond batch; the null mapping is
   transported with the token pairs, not resampled by array position.
5. **Relation isolation:** every directed edge joins tokens of the same
   molecule; one self loop per token with rel 0; segment-softmax sums to 1;
   R/N have the same pair set and role frequencies.
6. **Gradients:** finite common and relation gradients; every relation tensor
   receives a non-zero gradient on a real train batch; the relation branch is
   not detached; real and null labels produce different relation gradients.
7. **Output semantics:** raw-`y` scalar output, L1 loss, no target decomposition
   and no row/molecule id input. Parameter total `<= 500,000`.

FP32 CPU replay tolerance is `1e-6`–`1e-5`; observed values are reported. An
implementation failure is fixed and re-checked, not reported as a mechanism
negative.

## 7. Metrics and fixed diagnostics

Every run records: full valid curve, best valid/epoch, final valid, eval-mode
train MAE at best/final, W20 and T100.

* `W20` = minimum over **every** contiguous 20-epoch window of the median valid
  MAE. It is not the best arbitrary 20 epochs.
* `T100` = median valid MAE over the last 100 epochs.

Fixed groups (definitions unchanged from earlier rounds):

* old four groups: seen/unseen × ordinary/tail = 710 / 225 / 39 / 26;
* ordinary 935, tail 65, tail without row 172, all without row 172
  (descriptive only);
* ring-rich (213) and ring-rich & unseen (47);
* per-molecule ΔAE, median ΔAE, better fraction, error contribution and
  prediction direction; tail bias and slope of prediction on `y`.

Row 172 is retained in every headline number. Removing it is an explicitly
labelled diagnostic, never a selection or gate value.

Molecule bootstrap is reported separately for each supervised seed. It describes
sampling uncertainty of fixed predictions, not training-seed variance; valid
molecules share radius-2 keys, so the resampling is not independent.

## 8. Frozen decision rules

Let `gain = MAE(reference) − MAE(candidate)`; positive means the candidate is
better. All thresholds use best-valid predictions unless stated.

* **H1 overall:** in both completed seeds, `R − P` gain `> 0`; the two-seed mean
  gain `>= 0.001`; and the two-seed mean W20 and T100 gains are each
  `>= −0.001` (no clear sustained reversal).
* **H1 unseen+ordinary:** in both seeds, the 225-group gain `> 0`; mean
  `>= 0.005`; and overall mean gain `>= −0.001` (small overall cost allowed).
* **H1 supported:** either of the above. A positive but smaller
  two-seed-consistent gain is reported as `SMALL_CONSISTENT_GAIN`, not silently
  promoted.
* **H2:** in both seeds, `R − N` best-valid gain `> 0`; mean `>= 0.001`; and
  W20/T100 mean gains `>= −0.001`.
* **H3/test trigger:** H1 overall supported, both seeds complete, and no
  sustained reversal. Otherwise the official test stays unread and H3 is
  reported as **not tested**, not as a failure of the method.
* The 0.002–0.003 overall and 0.01 unseen+ordinary scales are reporting
  references, not noise floors or significance tests.
* A gain concentrated in row 172, the top-10 P3B errors, or a few tail rows is
  reported as tail stability, not a general unseen-environment improvement.
* A consistent improvement smaller than the pre-registered band is reported as
  small; it is not called a confirmed mechanism or a near-0.05x result.

## 9. GPU budget and schedule

* Hard cap: **5.0 A100 GPU-hours total for both GPUs**, counting formal runs,
  timing/smoke, failures, retries and time a job holds a GPU. Wall clock target
  `<= 5 h`.
* Before formal training: measure one full step, full valid, checkpoint save and
  full-train eval for P/R/N on the actual host; extrapolate 600 epochs and add
  at least 15% margin. Record projected and actual cost.
* Stage 0: CPU checks and GPU timing.
* Stage 1 (mandatory): seed 0 P/R/N complete. P0 and R0 may run in parallel on
  the two GPUs, then N0. R/N are not dropped because R0 has not reached an
  absolute score.
* Stage 2: if measured budget supports it, complete seed 1 P/R/N. The default
  is all six runs. The stage-1 decision note is written before launching any
  seed-1 run and uses the full curve, overall and pre-specified groups.
* If six runs do not fit with margin: keep seed 0 three-arm completeness; then
  P1/R1 for performance replication; N1 next. If N1 is not run, H2 has one
  training seed and is not called confirmed. If P1/R1 cannot be paired, stop
  with seed-0 evidence and state the missing replication. No historical P3 is
  substituted for same-round P1.
* Do not start a full run projected to cross the cap or to be truncated.

## 10. Terminal test (conditional)

If and only if the H1-overall trigger fires with both seeded runs complete:

1. Freeze the single candidate R and the same-round P checkpoints/epochs; commit
   the valid conclusion and the test harness before reading test.
2. Read the official test once on CPU. At most P/R × seed 0/1 ×
   {best-valid, final} = 8 columns. No per-epoch or exploratory state is sent.
3. `best-valid` is the main table; `final` is a pre-declared selection-sensitivity
   table. Do not switch tables or average the two checkpoint choices.
4. Report each single model, the two-seed MAE mean, the same-selector `R − P`,
   the old four test groups, and whether `< 0.060` is reached.
5. The test is historically exposed; this is not a first blind read. Nothing is
   changed or selected from test feedback.

If the trigger does not fire or the second seed is missing, test is not read and
H3 is stated as **not tested**.

## 11. Deliverables

`PROTOCOL.md`, frozen config, init/data/relation-null/budget manifests, reusable
relation batch + unified P/R/N entry point, native replay / bypass / invariance
checks, per-run curves, best/final checkpoints and hashes, per-molecule valid
predictions, Chinese `REPORT.md`, `summary.json`, GPU ledger including failures,
and the track `STATE.yaml` update. Stop at the end of this scope.
