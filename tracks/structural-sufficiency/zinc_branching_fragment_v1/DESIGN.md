# DESIGN — zinc-branching-fragment-v1 (P3-BF)

**Status:** declared before the first formal GPU run. One candidate, one seed-0
incremental trajectory, best-valid selection, then one descriptive test read.

**Track:** `tracks/structural-sufficiency/zinc_branching_fragment_v1/`
**Base commit:** `d83d4ea` (DBT-PA closed)
**Budget:** ~60 minutes of incremental A100 work (prep + timing + training +
scoring), one GPU. No second arm, no second seed.

## 1. Proposition

Under the P3 computation (BJOC-Null composer + reader, `group_to_bonds(NULL)`
unchanged), does explicitly exposing the **joint typed configuration of short
rooted branching fragments** — with the full typed induced adjacency, the
extra edges included, and a true multiplicity — improve single-model
generalisation?

This is an exploratory hypothesis. Nothing here shows that raw108 collisions
are the only bottleneck, and no 0.05x outcome is promised. The round changes
the local input **and** runs a second training stage, and it has no P3
continuation control, so any improvement is first described as "this new
candidate is better under this training procedure", not as a causal estimate
of the fragment information.

## 2. Static objects (three families, exhaustive, label-free)

From the raw atom category, the real physical bonds and the bond category only:

* **S3** — root `v` plus three distinct neighbours `{u1,u2,u3}` (4 nodes).
* **S4** — root `v` plus four distinct neighbours `{u1,u2,u3,u4}` (5 nodes).
* **F4** — a neighbour `u` of `v`, then two distinct `{w1,w2}` from
  `N(u)\{v}`; node set `{v,u,w1,w2}` (4 nodes).

Every object keeps the **complete typed induced adjacency** on the selected
nodes: template edges plus every extra real edge, never erased. The root is
always at canonical position 0. Node identity inside a fragment is
canonicalised by taking the lexicographic minimum over all permutations of the
non-root nodes (at most 24) of the sequence
`atom categories (5 slots) + pair types in the 10-slot 5-node layout`, where
pair type `0 = no real edge`, `1..3 = real bond category`. Padding slots use
the explicit node/pair masks. A content key contains no original atom index,
split id, traversal order or temporary id.

Object semantics:

* one `(root, family, actual node set)` counts **once**;
* F4 generation orders / centres that produce the same node set are deduped;
* two different node sets with the same content key remain two **multiplicity**
  units (the weighted statistics keep the count, uniqueness is never
  substituted for multiplicity);
* families are counted independently even if one node set satisfies several;
* roots with no object of a family get the exact empty-set semantics (no
  fabricated token);
* degree ≤ 2 roots may have no objects; this is normal.

This is **not** a complete ego-graph encoding and no WL-completeness or
information-sufficiency claim is made.

## 3. Shared encoder φ and root statistics

Every unique content row is encoded by one shared continuous encoder; no
`nn.Embedding(fragment_id)` exists. Valid content unseen in train is encoded
from its raw content, never mapped to an OOV zero.

```
atom embedding     28 categories x 8        (same category space as P3 raw108)
pair embedding      4 categories x 4        (0 = no edge, 1..3 = bond)
input = [5 x 8 atom, 10 x 4 pair, node mask 5, pair mask 10, family one-hot 3]
      = 98
phi   = MLP(98 -> 64 -> 32), SiLU, dropout 0
```

Per `(root, family)`:

```
n      = sum of true multiplicities c
mu     = sum(c * phi) / n
sq     = sum(c * phi^2) / n
count  = log1p(n)
exists = 1(n > 0)
```

Empty family: all four entries are zero. The three families concatenate to
`s_v` of dimension `3 * (32 + 32 + 1 + 1) = 198`. No sampling, so there is no
train/valid multiplicity drift.

## 4. P3 extension (single model, everything trainable)

Starting point: **P3B_s0 best-valid checkpoint @377**
(`runs/xBssl_P3N_s0/best.pt`, valid 0.076223754, sha256
`4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f`).
No P3A, scratch, final/soup or raw SSL weights.

The first atom layer is implemented as two linear terms, one shared bias and
one shared activation:

```
h_v = SiLU(W_x raw108_v + b + W_s s_v)
a_v = head(h_v)                      # original 128 -> 96 layer
```

`W_x`, `b`, `head` and every other P3 tensor load exactly from P3B_s0.
`W_s: 198 -> 128` is zero-initialised. Then the original bond encoder,
junction encoder, Null composer and reader run unchanged; the original
global33, count/amplitude handling and Null grouping semantics are preserved.
There is no second prediction branch, no extra pooling and no lookup residual
on the final prediction.

Zero new columns make the initial function exactly the supervised P3B; φ is
normally initialised and receives gradient from the second step onward.
All parameters (inherited, `W_s`, embeddings, φ) are trainable and in the
optimizer.

Parameter count (real instantiation):
`413,505 (P3) + 8,656 (embeddings + φ) + 25,344 (W_s) = 447,505 <= 500,000`.
The cache stores only static content/indices/counts; no P3 encoder output is
cached.

## 5. Data and read discipline

Official PyG ZINC subset, train 10,000 / valid 1,000 / test 1,000. Raw-y L1.
Train labels only for optimisation; valid only for the fixed per-epoch
validation and best-valid checkpoint; test only after the checkpoint/recipe
freeze, as a descriptive read of an already exposed benchmark. The four
analysis groups (covered/unseen × ordinary `y>-3` / tail `y<=-3`) are
fixed before the run, never enter the model, sampling or inference, and row
172 is kept. No SA/logP/cycle component answers, no label-derived gate, no
external labels.

## 6. One incremental trajectory

Seed 0, batch 128, `drop_last=False`, 79 steps/epoch, raw-y L1, grad clip 5,
dropout 0, no aux loss, no resampling.

Two parameter groups of the same model:

| group | params | peak lr | min lr |
|---|---|---|---|
| inherited P3 (W_x, b, head, bond/junction, composer, reader) | 413,505 | 1e-4 | 1e-6 |
| new (`W_s`, atom/pair embeddings, φ) | 34,000 | 1e-3 | 1e-5 |

AdamW, weight_decay 1e-5, 200-update linear warm-up, then one shared relative
per-step cosine. Fresh optimizer; no old optimizer/scheduler state.

Four-epoch timing run first, from the same P3B initial state, recording full
valid every epoch. After timing the model is rebuilt from the same P3B state,
new modules re-seeded and a fresh optimizer is created. E is chosen before the
formal run as

```
E = min(600, floor((3600 - used_seconds - 180) / (1.15 * t_epoch)))
```

rounded down to a multiple of 10; if `< 100` the round is reported infeasible
instead of forcing a token run. No cosine reset, no horizon extension.

Epoch 0 (inherited P3B) is a candidate checkpoint; its valid is fixed first.
Best-valid uses strict `<`, ties keep the earlier epoch. Early stop (declared
before the run): a stopping-only reference starts at the epoch-0 valid and is
updated whenever the valid improves by `>= 0.0005` relative to it; after 100
epochs, 100 consecutive epochs without such a cumulative improvement stop the
run. Smaller improvements still update `best.pt`. Hitting E or the early-stop
rule is reported as such; a crash/manual stop is INCOMPLETE.

## 7. CPU gate before GPU

Required before any GPU work (this round, in `analysis/`):

* exact hand-graph counts (star, fork, path, extra closure) against an
  independent subset enumeration, plus a node-permutation check;
* real train/valid object counts, empty-set fractions, content kinds, maxima,
  cache size and extraction time;
* raw108-root-group refinement by the new multiplicity-aware canonical content
  multiset, with coverage and concrete examples;
* the declared A/B example (identical root raw108, F4 = 1 vs 0) plus an
  explicit check of whether the whole-graph bond/junction channels differ;
* train→valid new-content coverage (descriptive only).

GPU work starts only if the semantics/permutation/counting checks pass, real
branching objects exist, and the new encoding actually splits at least one
raw108-confounded real local configuration with a stated coverage. If not,
the round stops at the CPU fact and does not substitute another motif family.

## 8. Reporting

Report: what the encoding distinguishes and its real coverage; whether the
declared training/early-stop rule completed and the true incremental time;
best-valid / test / eval-train; differences vs P3B/L/DBT (reused predictions, no
retraining); whether the best point came from new training or epoch 0; and the
open confounds. Also: checkpoint provenance and hashes, total parameter count,
cache/content semantics, best/final/last locations, and reproduction commands.
