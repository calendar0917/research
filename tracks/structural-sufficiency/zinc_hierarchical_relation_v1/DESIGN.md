# DESIGN — zinc_hierarchical_relation_v1

Round: `zinc-hierarchical-relation-v1` (track `structural-sufficiency`).
Status: design draft, written **before** any GPU run. If the frozen protocol is
not eventually followed, the report must say so and must not redefine the gate.

## 0. Question and boundary

Can a **non-recursive** model whose input is a label-blind hierarchy of local
structural objects (atom → bond → junction → rooted induced patch), whose
fine-grained contribution is modulated by **train-only structural support**,
and whose graph readout uses vectors/second moments rather than additive object
scalars, reach official ZINC-12k **valid** MAE ≤ 0.073 (P3B_s0 = 0.076224)?
And does **one shot** of a shared low-rank pair interaction with real graph
relations (short paths, distance, ring/block membership, short walks) add a
structure-specific increment over the same hierarchy without relations?

This round is development + mechanism research. **Official test is never
loaded, predicted, or scored.** No claim about 0.05x is made unless a later,
separately frozen terminal protocol produces it. The primary reference is the
frozen `P3B_s0` best-valid MAE `0.076224` (same official valid split, same
10k/1k protocol), with historical seeds reported as context only.

## 1. What the repository already closed (absorbed, not repeated)

- `zinc_generalization_audit_v1`: the 0.076 platform is a **shared heavy-tail
  hard core**, not uniform error and not seed noise. 25.1 % of valid molecules
  carry a train-unseen radius-2 atom environment and carry 49.8 % of the
  absolute error. Global label quantity is not the confirmed main cause;
  local/compositional support is not excluded.
- `zinc_arch_exploration_v3`: full pair composition with an intermediate-atom
  contraction (`A_s0`) buys a real **tail** advantage (P3B worst-100 0.3590 →
  0.3053 valid) but loses on the covered/bulk molecules; native SSL on the new
  trunk and learned pair pooling do not fix that. Spectral triple invariants
  (`B_s0`) are a global average and are not localisable.
- `zinc_local_factorization_v1`: a factorised local-content side entrance into
  P3 is indistinguishable from its dense control; the factorisation costs
  compute, not parameters, and loses chord/ownership information.
- `zinc_shared_factor_supplement_v1`: the shared-state block is almost dead at
  the readout (zeroing it changes valid MAE by ~7e-6), so T vs N was decided by
  training noise. A new mechanism must enter the **main readout path**, not a
  side branch.
- `zinc_structural_object_tokens_v1`: “more object families + full-graph
  attention” under a single recipe underfits (train MAE 0.051 vs 0.0157) and is
  far worse; object coverage was not the bottleneck.
- `phase2b`: a real local-MP module adds +0.00306 valid (+0.00127 test, not
  confirmed); the gain is seed-fragile. Cost +12 % params, +38 % wall.
- `p3_bond_relation_bridge_v1`: a **zero-initialised gated** one-hop bond
  relation residual collapses (|γ| ~ 3e-4, kernel swap changes prediction by
  exactly 0). The failure is the fusion interface, not evidence that relations
  are useless.
- `bjoc_predictive_bottleneck_v1` and `bjoc_intrinsic_contribution_v1`: the
  exactly additive IJCM head cannot even fit the training relation
  (fit MAE 0.106 vs 0.037) and its residual is -0.95 correlated with the true
  pairwise interaction count; an interaction is needed, but an additive wider
  vocabulary is not the answer.

## 2. Why this round is not one of the five rejected mechanisms

| Rejected mechanism | This round |
|---|---|
| **IJCM pure additive object prediction** (each object emits one scalar; prediction is exactly `bias + Σ α + Σ β + Σ ν + Σ η`) | The readout pools **vectors** and **second moments** of four hierarchy levels plus **within-object cross moments** (`a_u ⊙ a_v` on bonds, `b_1 ⊙ b_2` on junctions, `b_root ⊙ p` on patches) through a small MLP. There is no per-object scalar contribution identity; the prediction is not a sum of independently emitted object scalars. |
| **P3 gated relation residual** (zero-init `γ`, one-hop bond–bond kernel added to `c`, gate collapses) | B has **no zero-init gate** and is **not** a residual on P3 output. A single shared-parameter bilinear interaction rewrites the junction/patch tokens that feed the main readout. Its score contains a real relation-type bias, so changing the relation assignment changes predictions by construction; a CPU gate requires a non-zero response. |
| **Full pair composition / multi-layer object attention** (all atom pairs get a pair state; a contraction over intermediate atoms and a per-pair nonlinear `H([X_ij,U_ij])`; or 4-layer full-graph attention over ~59 tokens) | B keeps **one** interaction round, **low rank r = 16**, **one** shared score/value projection, **no per-pair MLP**, **no intermediate-atom contraction**, no multi-layer token updates, and only graph-related pairs (distance ≤ 3 or shared ring block; ~900 pairs/molecule). This is a one-shot relational re-weighting, not a pair-composition trunk or a Transformer stack. |
| **Side-branch factorised structure features that the readout ignores** (T/N shared factor block, LF factorised local content) | The hierarchy tokens **are** the readout input; there is no separate side branch. Removing the patch level or the relation interaction must change the prediction (CPU sensitivity gates). Support gating multiplies the fine contribution *inside* the patch token, not a disconnected optional residual. |
| **Local correction on a frozen base residual** (ESR per-type lookup, P3+local-MP, P3 dual channel) | A/B are trained **from the seed-0 train-only masked-SSL encoders** with a fresh readout; no supervised P3 reader or P3 best checkpoint is loaded. There is no frozen base to correct. The atom/bond/junction encoders are fine-tuned; the hierarchy above them is new. |

## 3. Data contract (label-blind)

Allowed inputs, per molecule, all deterministic functions of the raw heavy-atom
graph (`x`, `edge_index`, `edge_attr`):

1. raw atom type (28-way one-hot) and raw bond type (4-way one-hot);
2. fixed local statistics used by the frozen BJOC object builder: degree, ring
   membership, minimum ring size, incident bond-type histogram, radius-1 and
   radius-2 **nested-WL type histograms**, 2-path/triangle counts, terminal and
   ring-system flags; bond ring flags; junction centre/endpoint/bond-type
   features; 33 label-blind global topology statistics;
3. **rooted induced patch structure** (new): for every bond the actual node set
   within graph distance ≤ 2 of either endpoint, the **induced edge set** among
   those nodes (bond type, endpoint distances to the root bond, root flag), and
   patch scalars (node/edge/chord/cycle counts, ring composition). This is a
   true induced subgraph, not a type histogram;
4. **relations** (new): clipped all-pairs graph distance, ring-block
   co-membership, and short-walk counts `A²`, `A³` between token anchors;
5. **train-only structural support**: a scalar count of a coarse
   **radius-2 nested-WL bond-centred key** over the 10k train molecules, plus
   its train log-count z-score, a quantile over unique train keys, an unseen
   flag and a clipped count. The key is stored only as four scalars; there is
   **no** learnable embedding for any full radius-2 key, no per-key lookup
   table, and no per-molecule or split-local id. On the first 200 valid
   molecules 13.6 % of patches have an unseen train key (13.8 % on all 1000),
   so the support feature actually varies on the hard part of the split.

Forbidden and absent: true logP/SA, target cycle penalty, any quantity derived
from `y`, official test statistics, split-local pattern ids, molecule row ids,
row-order features, SMILES strings inside the model. Standardisation (global
features, patch scalars, support log-counts) is fit on train only and applied
unchanged to valid.

## 4. Architecture A — hierarchical compositional encoding (HCE)

All token widths are `d = 96`.

- **Atom level**: frozen train-only masked-SSL atom encoder (108→128→96),
  fine-tuned, then LayerNorm.
- **Bond level**: frozen SSL bond encoder (295→160→96) on
  `[a_u+a_v, |a_u-a_v|, a_u⊙a_v, bond_feat]`.
- **Junction level**: frozen SSL junction encoder (608→160→96) on
  `[a_i+a_k, |a_i-a_k|, a_i⊙a_k, a_centre, b_1+b_2, |b_1-b_2|, junc_feat]`.
- **Shared coarse-to-fine refinement**: one `Refine` block (input
  `[h, Σchildren, Σchildren², role]` → 128 → 96, residual + LayerNorm) is
  applied **twice with the same weights** at the bond, junction and patch
  levels. Role embeddings distinguish the three roles; weights are shared
  across levels. This is the “shared” part: the hierarchy is not four
  independent encoders.
- **Rooted induced patch level** (new): per bond, a DeepSets-style encoder over
  the actual induced patch nodes and edges:
  `node = SiLU(W_n [a_x, dist_u one-hot, dist_v one-hot])` (32),
  `edge = SiLU(W_e [bond type, dist_u(x), dist_v(x), dist_u(y), dist_v(y),
  root flag])` (16)`;
  pooled node/edge sums and squared sums enter a patch MLP together with the
  root bond token, patch scalars and support features;
  `fine` is multiplied channel-wise by `sigmoid(W_s support)` (**support
  modulation of the fine contribution**), then the shared `Refine` block is
  applied twice with the root bond token as child context.
- **Readout**: per level (atom/bond/junction/patch) `sum`, `mean`, `mean(h²)`
  and `log1p(count)`; plus the three local cross moments
  (`mean(a_u⊙a_v)`, `mean(b_1⊙b_2)`, `mean(b_root⊙p)`); plus the 33 train-
  standardised global features. Small MLP `1477 → 96 → 96 → 1`. No additive
  scalar-object identity.
- **Parameters**: 438,585 (A) / 445,578 (B), both under the 500,000 budget.
  No recursive message passing, no attention stack, no whole-pattern lookup.

## 5. Architecture B — structural relational interaction (SRI)

B reuses A's modules, **shared initial values**, data order, recipe and
selection semantics. It adds, before the readout:

- tokens = junction tokens + patch tokens (~60/molecule);
- one shared low-rank (r = 16) query/key/value projection;
- pair set = graph-related pairs only: clipped anchor distance ≤ 3 **or**
  shared ring block (unoriented, expanded both directions; ~900/molecule);
- relation id = `(distance bucket 0–3) × (shared ring block) × (A² walk bucket
  0–3+ )`, plus a self relation id;
- `score_ij = q_i·(k_j + rel_emb[R_ij])/√r + b[R_ij]`;
- one segment-softmax per source token, one scatter-add of `v_j`;
- `h'_i = LayerNorm(h_i + W_o o_i)`, **no zero-init gate**;
- readout uses `h'` for junction/patch moments; atom/bond moments unchanged.
- Reports parameter/FLOP/wall difference vs A.

## 6. Null control C (prepared, not run formally)

Same parameters and compute as B. The relation-to-pair assignment is permuted
**within each molecule** among non-self pairs, preserving the per-graph
relation multiset and pair degree. CPU gate: C's predictions differ from B on
real molecules. C is only run on GPU if B passes the frozen promotion gate and
a second seed is bought under a new protocol.

## 7. Pre-registered gate (before GPU)

Frozen in `PROTOCOL.md` and `configs/hr_v1.json`:

1. candidate best-valid raw-y MAE **≤ 0.073**;
2. improvement over frozen `P3B_s0` after removing the **fixed** 10 valid rows
   with the largest `P3B_s0` absolute error is **≥ 0.002**;
3. fit capability is not materially behind P3: candidate eval-mode train MAE at
   best epoch **≤ 0.026** (P3B = 0.01468; 1.77×);
4. the gain is not one-molecule-driven (the drop-top-10 rule is the
   operational version of this).

If A and B both fail, the round closes; no seed 1, no depth/width/recipe/feature
grid, no test read. If one passes, only it is eligible for a second seed and a
parameter-matched mechanism control under a new pre-registration.

## 8. Pre-GPU CPU evidence (frozen under `analysis/`)

- `analysis/cpu_gates.json`: **all 12 gate groups pass**. Parameters A
  438,585 / B 445,578; shared-state hash identical; renumbering and edge-
  direction max prediction delta 0.0 or 4.8e-7; batch-vs-single <= 1.1e-6;
  empty/single-atom/ring/disconnected cases finite; train/valid support
  isolation exact; no parameter tensor has zero gradient on a 64-molecule
  batch; zeroing support changes A's prediction by 0.043, zeroing patch edge
  types by 0.022, zeroing junction features by 0.068, and real-vs-null
  relations changes B by 0.020; only embeddings are role (3) and relation
  (33); CPU 128-molecule fwd+bwd 0.058 s (A) / 0.107 s (B).
- `analysis/data_audit.json`: raw object cache is exact against the frozen
  builder; 3.86 % of 4,969 patches from 200 valid molecules sit in nested-WL
  keys that contain more than one **induced** patch signature (synthetic C6
  vs 2 x C3: identical 2-round WL colours, different induced triangle/path
  signatures), so the patch level is not a nested-WL restatement; support
  unseen fraction 13.8 % on valid; mean 59.3 tokens and 903 related pairs per
  molecule.
- `analysis/overfit_cpu.json`: 48 molecules, 400 steps, final eval-mode train
  MAE 0.132 (A) / 0.230 (B) vs constant baseline 1.005, so both arms can fit.

## 9. Honest expectation

The historical best *valid* number in this line is 0.076224 (P3B) and no
non-recursive trunk has beaten 0.073. The audit says most of the remaining
error sits on rare local environments; the patch level + support modulation is
aimed exactly there, while the shared hierarchy and the non-additive readout
target the bulk. B's relational re-weighting targets ring-rich long-range
composition. The gate may well fail; the deliverable is then a clean statement
of whether the failure is fit, generalisation, relation use, or missing
information — not a rescued hyper-parameter.
