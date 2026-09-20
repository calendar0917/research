# Pre-registration — PSCD-v0: Port-Structured Compositional Graph Dictionary

Round name: **PSCD-v0** (Port-Structured Compositional Graph Dictionary). Written
**before** any formal ZINC run, following the `remote-research-runner` skill.

Local code → remote A100 compute → local analysis. Official ZINC `test` is
**never** loaded, instantiated or referenced by any stage. All decisions use
official `train` + `valid` only. This note does **not** modify any historical
record.

---

## 0. The single question this round asks

Previous rounds (canonical coordinates, WG-ICSC, I-CRATE, AIOM, DOI, NPA, TIGD)
all assumed a graph sparse code is a **coefficient vector** `α_G ∈ R^K` (or a
soft variant). PSCD asks a different, more basic question:

> Can a molecular graph be represented as a small number of **learned subgraph
> primitives reused across graphs** plus an explicit, **port-aware composition
> graph** recording how those primitives are glued together?

The new graph sparse code is a structured object

```
G  <->  (D, C_G),   D = {d_1..d_K}  (shared motif dictionary),
                    C_G              (graph-specific composition graph)
```

where each atom `d_k = (V_k, E_k, X_k)` is a connected, attributed, typed-bond,
mesoscale subgraph — **not** a whole-molecule prototype, node embedding, fixed
graph vector or GNN latent.

Sparsity this round is defined by `|O_G| << |V_G|` (number of motif occurrences
vs atoms). **No** `||α||_1`, **no** whole-graph coefficient vector.

This is an **object-feasibility audit only**. No task-driven dictionary, no
end-to-end tokenizer, no Gumbel/hard-concrete, no motif attention, no
higher-capacity reader, no BRICS/chemistry predefined motifs, no GW/FGW.

## 1. Confirmed graph semantics (from the current repo loader)

| quantity | value |
|---|---|
| `x` (atom type) | integer categories `{0..20}` → **C_V = 21** |
| `edge_attr` (bond type) | integer categories `{1,2,3}` → **C_E = 3** |
| bond object | one undirected chemical bond (directed copies deduplicated) |
| added descriptors | **none** (no degree/ring/ECFP/hand-crafted chemistry) |

`B ∈ {0,1}^{n×m}` with `Σ_v B_ve = 2`. Asserted at run time.

## 2. Stage A — data-driven graph BPE (frozen)

Initial partition `P_G^{(0)} = {{v}: v ∈ V_G}` (one token per atom).

**Motif identity** is the internal attributed subgraph only (atom categories +
internal typed bonds). No external neighbours, graph id, attachment pattern,
degree, or label enter the type. Equality uses **exact** colored-graph
canonicalization (`pynauty` on the colored incidence graph, the repo's
`canonical_atom_order`); WL is never used as final identity.

**Merge candidate.** Two adjacent current tokens `o_i, o_j` with
`|V(o_i)| + |V(o_j)| <= 8` form a candidate with identity
`τ(o_i,o_j) = Canon(G[V(o_i) ∪ V(o_j)])` (the union **induced** subgraph type).

**Fixed budget (no sweep).** `64` merge rules, `max motif size = 8`.

Each round: (1) tally all legal adjacent-pair union types over the train
corpus; (2) pick the type with the largest occurrence count (ties → smallest
canonical key bytes); (3) add it to the dictionary/merge sequence; (4) apply it
in every train graph; (5) update partitions. Record per round: candidate
frequency, graph support, motif size, actual merges, remaining token count.
Types already introduced are excluded from later selection.

**Overlap determinism.** Candidates sharing a token are resolved by a
deterministic greedy maximal matching ordered by the canonical atom-set key
`(min ρ_G(V_o), sorted ρ_G(V_o))`, where `ρ_G` is the molecule's exact canonical
atom order. This is only a within-graph tie-break; it is not a cross-graph
coordinate.

**Valid tokenization is frozen.** After learning `r_1..r_64`, valid starts from
singletons and applies `r_1..r_64` in order. No new motif is added on valid, no
rule is changed by valid. A rule with no matching occurrence is skipped, so
there is no OOV: singletons are the fallback.

**Disjoint partition.** The final code is a disjoint partition
`V_G = V(o_1) ⊔ ... ⊔ V(o_M)`; motifs never share atoms.

## 3. Motif canonical form + occurrence mapping (frozen)

For every motif type `d_k` store the exact canonical attributed graph `d̂_k`
(canonical node-type sequence + internal typed bonds) and the automorphism
orbits `Aut(d_k)` (`pynauty.autgrp`, `|V_k| <= 8`).

For an occurrence `o_j`, enumerate attributed isomorphisms `ψ: d̂_{k_j} → o_j`
and keep the one minimising, lexicographically,
`[ρ_G(ψ(1)), ..., ρ_G(ψ(|V_k|))]` (backtracking in canonical-slot order with
occurrence atoms tried in increasing `ρ_G`; the first complete assignment is the
lex-min). This makes the occurrence→canonical port mapping invariant to input
atom labelling. **Permutation test required — not assumed.**

## 4. Joint port state, composition edge, decode (frozen)

Occurrence `o_j` has joint port state `A_j = {(p_r,t_r,e_r)}` over its external
chemical bonds (`p_r` = canonical vertex used as port, `t_r` = bond type,
`e_r` = composition connection). Motif type ≠ motif type + port state.

Each cross-motif bond is recorded fully as `(i, p_i, t, j, p_j)`
(source occurrence, source port, bond type, target occurrence, target port).

`Decode(D, C_G)`: for each occurrence copy `d̂_{k_j}` (atoms + internal typed
bonds) and add every composition edge as a chemical bond; compare exact colored
canonical keys: require `Ĝ ≅ G` for **100 %** of train (10 000) and valid
(1 000). Not 99.9 %.

## 5. Collision ladder (four representations)

| code | content |
|---|---|
| **A** motif bag | occurrence count per motif id only |
| **B** composition-no-port | occurrence types + inter-motif bond types, no endpoint ports |
| **C** independent-port | ports as independent `Aut`-orbit ids, no joint analysis |
| **D** full PSCD | full `(i, p_i, t, j, p_j)` |

Each is mapped to an exact colored-graph canonical key (multigraphs encoded by
subdividing bond/port vertices). Ground-truth equivalence is exact attributed
canonicalization. Report non-isomorphic-molecule confusions on train+valid.
Expected `A > B > C`, `D = 0`; **C may equal D** (not forced to fail).

## 6. Metrics (train / valid separately)

* Compression: `n_G`, `M_G`, `r_G = M_G/n_G` (mean/median/p5/p95), compression
  factor `n_G/M_G`.
* Motif size: occurrence-weighted mean/median/p95/histogram, fraction of atoms
  covered by motifs size ≥ 2, fraction of occurrences that are singletons.
* Vocabulary reuse: learned/used/unused-on-valid types, occurrence frequency,
  graph support + histogram, top-16/32/64 occurrence coverage.
* Port complexity: per occurrence `d_j` (external bonds), `p_j` (distinct port
  vertices), `p_j/|V(o_j)|`, port count vs motif size, joint port-state
  diversity per motif.
* MDL-style diagnostic (proxy, not full GraphMDL): dictionary cost (motif size,
  atom categories, internal typed bonds) + graph code cost (occurrence count,
  motif ids, composition connections, bond types, endpoint port positions) with
  empirical code lengths `L(x) = -log2 p(x)` estimated on train; compare
  `L_PSCD` to atom-level singleton encoding `L_atom`. Diagnostic only — the
  vocabulary is **not** tuned by it.

## 7. Stage A feasibility gate

* **A1 exactness** 100 % reconstruction, train+valid.
* **A2 invariance** representation exactly invariant under random relabel
  (500 train graphs × 20 relabels; recompute the full structured code +
  decode).
* **A3 compression** `mean(M_G/n_G) <= 0.55`.
* **A4 reuse** not >50 % of learned non-singleton motif types with total train
  occurrences `< 10` (vocabulary fragmentation).
* **A5 port complexity** not "large motifs almost all ports" (fail if the
  motif-size-weighted mean of `p_j/|V(o_j)|` over occurrences with `|V| >= 4`
  exceeds `0.75`).
* **A6 no collision** full PSCD code cannot confuse non-isomorphic species.

Any hard gate failure ⇒ **do not run Stage B**.

## 8. Stage B — representation capacity probe (only if all gates pass)

Frozen dictionary/merge sequence; no task-driven motif selection. Reader input
is the bipartite connection-node composition graph: motif-occurrence node
(feature motif id), connection node per inter-motif chemical bond (feature bond
type), incidence edge feature local port id.

Tiny reader: `d = 32`, `L = 2`, two rounds of sum message passing + SiLU; sum
pool over motif occurrences; head `Linear(32,32) → SiLU → Linear(32,1)`. No
attention/transformer/virtual node/hand-crafted descriptors/deep GNN.

Two matched controls: (1) **motif bag** — same motif embeddings, no composition
edges/ports, `z_G = Σ_j E_M(k_j)`, same head; (2) **raw atom graph** — identical
`d=32, L=2` message-passing form on the original atom graph (atom + bond
category embeddings), same head.

Protocol: train+valid only, seed 0; identical optimizer (Adam), LR `1e-3`, `wd=0`,
batch `128`, `max_epochs=300`, `patience=40`, grad-clip `5.0`, hidden 32, depth 2.
Record train/valid MAE, params, peak GPU, epochs.

Interpretation (matched, not absolute): Strong if
`MAE_comp <= MAE_raw + 0.03` and `mean(M_G/n_G) <= 0.55`; composition matters if
`MAE_bag − MAE_comp >= 0.02`; weak abstraction if `MAE_comp > MAE_raw + 0.08`
with train MAE also clearly higher.

## 9. Final answer and next decision

Final first line is one of the five frozen sentences (§36 of the brief); next
decision is one of
`proceed to task-driven dictionary selection` /
`refine the composition / port representation` /
`refine the motif discovery objective` /
`stop the compositional dictionary line`. This round does **not** start the next
phase.

## 10. Prohibited this round

task-supervised merge selection; end-to-end learned tokenizer; Gumbel /
hard-concrete; motif attention; higher-capacity reader; official test;
vocabulary-size sweep; max-motif-size sweep; chemistry predefined motifs;
BRICS as main method; dictionary sparse coefficient vector; GW/FGW.

## 11. Data-free correctness tests

Lock the invariants the audit depends on (no ZINC data / checkpoints / artifacts):
BPE merge determinism and permutation invariance on small synthetic molecules;
exact decode round-trip; lex-min occurrence mapping invariance; collision-ladder
ordering on a hand-built example; MDL proxy monotonicity sanity.
