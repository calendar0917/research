# Pre-registration — ZINC E2E-DictEnv-Typed-Cycle-v1 (static typed chordless-cycle object, shared task dictionary)

Round: `zinc_e2e_dictenv_typed_cycle_v1` (Workstream Z, ZINC).
Audit start revision: `7fa46ab300b43c6457499e45408296c5a4bd4c5c`.
Handoff package (vendored):
`tracks/ksvd/experiments/luyin16/typed_cycle_reference/v1_20261002/`
(sha256 table in its `README.md`).

This note freezes **Phase A** (the cheap train-only object screen) and records
the **Phase B/C** plan that becomes authorised only if Phase A passes.  It is
written before the Phase A run.  No official valid/test is read in any phase;
the official ZINC test split is never instantiated.

## 0. Question and non-goals

Beyond the existing shell chemistry statistics (Sem108/Sem110), whether the
**composition and cyclic order of atoms and bond types inside one closed path**
carries task-relevant information for ZINC `penalized logP`; and, if so, whether
putting that static object into the *same* task dictionary as the node
environment (one graph-level readout) enters a better absolute-performance band.

Not in this round: message passing, attention, node/edge hidden-state write-back,
node/patch hidden-state write-back, full-ring-ID embeddings, per-ring label
filtering, a second K-SVD training, a second task dictionary, dense/random
controls, seed-1 confirmation, any widening of `K`/`d`, any hyper-parameter or
cycle-length sweep, and any rescue after a poor endpoint.  Historical Small
106,925 (soup valid 0.121058) and Full 408,651 (0.119154) numbers are unmatched
background only.

## 1. Phase A — train-only cheap property screen (frozen)

### 1.1 Data

- Official PyG ZINC `subset=True` **train** split only, 10,000 molecules,
  instantiated with `split="train"` (valid/test are never constructed).
- Split fingerprint and dataset fingerprint from
  `ksvd_research.runtime.fingerprints.zinc_fingerprints`.
- Verified `topology25` hinge cache loaded read-only from
  `tracks/ksvd/results/zinc_topology_cache/train_topology_features.csv`
  (never rebuilt silently).
- Group key: verified canonical molecule key
  `identity_incremental_information_audit._raw_zinc_certificate` (pynauty
  certificate + coloured incidence canonical sequence); duplicates share a
  group and cannot leak across folds.  Fallback (documented here): if the key
  were unavailable, fixed official row IDs would be used and the ungrouped-
  duplicate limitation recorded.

### 1.2 Objects

All **chordless simple cycles** of length 3..10 (chord iff a non-consecutive
ring pair has an extra edge), enumerated by
`typed_cycle_reference.chordless_cycles` (early chord pruning + ID-free
canonical dedup).  No cycle basis, no sampling, no top-N truncation.  Each
length-ℓ ring contributes all 2ℓ dihedral views; the view multiset is recorded
per graph.

### 1.3 Feature arms (label-free graph functions only)

- `X_base` (573): per-root raw Sem110 sum/sumsq (220) + per-root raw φ65
  sum/sumsq (130) + raw global62 + topology25 (87) + untyped ring length
  3..10 counts (8) + fixed 128-d atom-pair×bond-type counts (128).
- `X_typed` (1040): vendored `cheap_probe_features.ring_features` typed block
  (per-length atom/bond composition and second moments; cyclic bond-pair
  separation; fixed 128-d hash of **partial** typed patterns with ring length —
  no complete-ring token, no label-based vocabulary).

No neural feature trained on all 10k labels, no node `E`, no task code and no
prediction residual enters the screen.

### 1.4 Probe and gate

`cheap_property_probe.py`, fixed seed `20261002`, three outer grouped folds,
one fixed inner dev split per fold, ridge grid `lambda={1e-3,1e-2,1e-1}` with
mean-MSE + `lambda*||w||^2`; lambda chosen by inner MAE; outer predictions
assembled into OOF.  `asinh` plus fit-only mean/scale/rare-column removal.
Two arms differ only by the typed block.

**Purchase gate (frozen):** OOF absolute MAE gain `>= 0.003` **and**
improvement in `>= 2/3` outer folds.

- Pass -> `BUY_ONE_TYPED_CYCLE_SCREEN`, Phases B/C authorised.
- Fail -> `TYPED_CYCLE_OBJECT_SCREEN_STOP`, this round closes with the negative
  result; no B/C, no object-scope shrinkage presented as the original plan.

No additional lambda, repeated split, width or feature-subset search.

### 1.5 Resource decision

Phase A budget <= 45 min.  Object extraction and fold count are reported with
real timings; a budget failure is recorded as a resource stop, not silently
shrunk.

## 2. Phase B — the one formal model (only if Phase A passes)

Exact prototype: frozen Small (`LatentBridgeSEM108`, 106,925 params) with all
existing structure dictionary / K32 / s8 / C6 masks / node environment / static
pair / global channels / training loss unchanged.  Add one ring path:

- per length-ℓ ring, all 2ℓ dihedral views, each 338-d: position-filled
  10×28 atom one-hot + 10×4 outgoing-bond one-hot + 10 valid-position mask +
  8-d ring-length one-hot, zero-padded; bond at position `t` is
  `(vertex_t, vertex_{t+1 mod ℓ})`, re-read from the original graph bond table
  for reversed views.
- shared `ring_encoder = Linear(338,64) -> SiLU -> Linear(64,48)` (bias, no
  LayerNorm/dropout); nonlinear per view **before** the 2ℓ-view mean:
  `h_cycle = mean_g ring_encoder(view_g)`.
- same `local_dictionary_bridge(h_cycle)` (`D_L[48,96]`, `V_L[96,48]`,
  `lambda1=0.05`, `lambda2=0.01`, ISTA 16, detached step, RMS scaling) -> no
  second dictionary, no new reconstruction term.
- graph summary `[sum(E_cycle), sum(E_cycle^2), zero_count_slot]` (97 dims;
  acyclic graphs are exact zeros, count slot stays 0).
- old unified 302 + new ring_pool 97 -> the **same** reader `399 -> 13 -> 13 -> 1`.

Parameter budget: `106925 + 24816 + 1261 = 133002`; new task-dictionary params
= 0.  New reader columns initialise to zero; old reader columns/bias and later
layers copied; new layers use an isolated RNG.  Formal training starts from
scratch, never warm-started from an old checkpoint.

Data: cache the ring views once (per graph).  Fields
`ring_view_atom/bond/mask/length` (no offset), `ring_view_cycle` (+ cumulative
cycle count on batch), `ring_cycle_anchor` (routing only, + node ptr),
`ring_cycle_length`.  Reuse an independent wrapper around the old `env_collate`
without touching the old global offset whitelist.  Ring anchors are routing-only
and never encoder inputs.

### 2.1 Acceptance before any formal training (frozen)

1. exactly 133002 params; old params equal the same-seed Small; new reader
   columns zero -> real train batch32 plain/masked prediction delta <= 1e-5;
2. permutation invariance under raw-graph relabel, ring rotation/reversal,
   endpoint swap, graph order, cross-graph batching, acyclic/mixed batches;
   integer ring view sets exactly equal;
3. two full-graph counterexamples separated by the real repo input builder and
   PyTorch (old input collision retained as a documented difference if it
   holds); new representation separates the composition-equal / order-different
   six-ring example;
4. ring-only path passes the same single D/V; no raw-ring->reader or
   raw-ring->scalar bypass; no node/edge write-back; changing ring objects does
   not change node `E` at fixed state;
5. two-step cold-start gradient check: step-0 ring-MLP gradient may be zero
   while new reader columns have task gradient; after one Adam step the ring MLP
   and the shared D/V receive non-zero MAE-only gradient; no non-zero bypass
   added to "fix" the zero init;
6. finite gradients / finite ring output / finite ISTA; 24-step smoke passes;
7. `zero_ring` zeroes ring `E`/pool only after the shared bridge, never the
   shared bridge `zero_code`; node `E` identical under `zero_ring`;
8. official-test construction blocked.

## 3. Phase C — one full CPU screen (only if Phase A and B pass)

Config inherited from the frozen parent: seed 0, official train10k / valid1k,
CPU 8 threads, batch 128, Adam lr 0.001 / wd 1e-5 / clip 5, 320 epochs, original
structure reconstruction term, fixed C6, Top-5 parameter soup.  If the source
config disagrees, the real parent values are reported and frozen before
training.  A real full-epoch timing measurement must fit
`320 * steady_epoch * 1.15 + 15 min` inside the remaining total budget,
otherwise the round stops before formal training with predicted time recorded.

Absolute bands (first-line decision, not a significance test):

- soup valid `<= 0.115` -> `TYPED_CYCLE_PROMISING_ABSOLUTE_SIGNAL`;
- `0.115 < MAE <= 0.120` -> `TYPED_CYCLE_LIMITED_SIGNAL` (seal, no widening);
- `MAE > 0.120` -> `TYPED_CYCLE_STOP` (seal, no next same-family variant).

Inference-only diagnostics at the endpoint: same-protocol soup train/valid MAE
and gap; `zero_ring` ΔMAE and prediction RMS; fixed-seed 11/22 cross-molecule
same-length ring-object permutation (preserving per-graph counts/lengths and the
global multiset); ring vs node code non-zero rate / effective atoms / MAE-only
gradient.  Near-dense ring codes are called a soft-threshold task dictionary,
not a sparse-dictionary advantage.  Small/Full are never retrained; official
valid never drives a new ring-length/hash/regulariser/threshold choice.

## 4. Provenance / deliverables

Every phase records the real revision, dirty/diff, commands, run id, fitted-
cache train-only source, split fingerprint, device and budget.  Phase A output:
`train_ring_probe.npz`, `train_ring_probe_report.json`, `provenance.json`,
`screen.json`, `REPORT.md`, `DECISION.md`.  Conclusion-first Chinese report,
limitations and the next decision are delivered even on a stop.  Large
cache/checkpoint stay git-ignored; only code and small evidence are committed.