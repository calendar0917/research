# PSCD-v0 — Port-Structured Compositional Graph Dictionary (ZINC)

**Question.** Can a molecular graph be coded as a small number of **learned
subgraph primitives reused across graphs** plus an explicit, **port-aware
composition graph** — the structured object `G <-> (D, C_G)` — rather than a
whole-graph coefficient vector `α_G ∈ R^K`? Stage A is the object-feasibility
audit (learn a data-driven graph-BPE dictionary from train only, tokenize train
+ valid with the frozen merge sequence, require exact decode / invariance /
compression / reuse / clean ports); Stage B is a matched tiny-reader capacity
probe (composition graph vs motif bag vs raw atom graph).

**Status.** Representation-object feasibility audit. No task-driven dictionary,
no learned tokenizer, no attention / transformer / deep GNN, no GW/FGW, no
vocabulary or max-size sweep. Official ZINC `test` was **never** loaded. `y` is
used **only** in the Stage-B capacity probe. This note does not modify any
historical record.

---

## 0. Provenance

| item | value |
|---|---|
| git commit | `cdae8b0fe6609ec8f2e347f82a97297ceee40263` |
| worktree at run | clean |
| splits used | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000** |
| official `test` | **never read / instantiated / referenced** |
| target `y` | used **only** in Stage B |
| execution | remote A100 host `res`, GPU 0 (`cuda`); Stage A CPU (pynauty) |
| analysis script | `tracks/ksvd/code/run_pscd_compositional_dictionary_audit.py` |
| preregistration | `tracks/ksvd/notes/pscd_preregistration.md` |
| tests | `tracks/ksvd/tests/test_pscd.py` (4 data-free tests, pass) |
| results | `tracks/ksvd/results/pscd_compositional/` (git-ignored) |
| reproduce | `uv run python tracks/ksvd/code/run_pscd_compositional_dictionary_audit.py --stage both --device cuda` |

Confirmed graph semantics: `C_V = 21`, `C_E = 3`; each undirected chemical bond
is one object (directed copies deduplicated); no added descriptors. Exact motif
identity uses `pynauty` on the **colored incidence graph** (the repo's
`canonical_atom_order`); WL is never used as final identity.

---

## 1. Stage A — data-driven graph BPE (frozen 64 rules, max size 8)

Initial partition = single atoms. Each round: tally all legal adjacent-token
union types over train, pick the most frequent (ties → smallest canonical key),
add it to the dictionary, apply it in every train graph. Types already
introduced are excluded from later selection; valid applies `r_1..r_64` in
order with **no** new motifs.

Vocabulary: **85 motif types** = 21 singleton atom types + **64 learned
non-singleton types**. Learned in 233 s (CPU).

Aggregate BPE effect (train): 158 636 deterministic merges; remaining tokens
169 731 → 73 028 across the 64 rounds. Round-1 rule (a 2-atom two-carbon edge)
fires 61 933 times with graph support 9 992/10 000 — i.e. the very first learned
primitive is reused in **every** molecule.

### 1.1 Gate A1 — exact reconstruction = 100 %

`Decode(D, C_G)` (copy each motif's canonical attributed graph, add every
composition edge `(i,p_i,t,j,p_j)`) compared to the original by exact colored
canonical key:

| split | exact | fraction |
|---|---|---|
| train (10 000) | 10 000 | **1.000000** |
| valid (1 000) | 1 000 | **1.000000** |

The mask-aware port mapping `(i,p_i,t,j,p_j)` reconstructs the graph **losslessly
and exactly**, including all non-singleton motifs and multi-bond / multi-port
occurrences. **PASS**.

### 1.2 Gate A2 — permutation invariance

500 train graphs × 20 random atom+bond relabelings = **10 000 checks**, each
re-running the full frozen tokenization + lex-min occurrence mapping + ports +
composition graph, and decoding:

* representation mismatches (full PSCD canonical key, before/after relabel): **0**
* decode failures: **0**

**PASS**.

### 1.3 Gate A3 — compression

`n_G` atom count, `M_G` motif-occurrence count, `r_G = M_G / n_G`:

| split | n_G mean | M_G mean | r_G mean | r_G median | p5 | p95 | compression n/M mean |
|---|---|---|---|---|---|---|---|
| train | 23.17 | 7.30 | **0.3198** | 0.3158 | 0.2353 | 0.4231 | 3.26× |
| valid | 23.08 | 7.32 | **0.3215** | 0.3182 | 0.2352 | 0.4211 | 3.15× |

`mean(M_G/n_G) = 0.320 ≤ 0.55`. Valid matches train with no drift (frozen
vocabulary). **PASS**.

### 1.4 Gate A4 — vocabulary reuse

| metric | value |
|---|---|
| learned non-singleton types | 64 |
| used on train | 64 (0 unused) |
| types with train occ < 10 | 1 (1.6 %) |
| graph-support range | 8 … 3 135 |
| top-16 / top-32 / top-64 occurrence coverage | 0.597 / 0.823 / 1.000 |

No vocabulary fragmentation (≪ 50 % low-frequency types). A 64-type vocabulary
explains all occurrence mass; the top 16 types cover ~60 %. **PASS**.

### 1.5 Motif-size statistics (occurrence-weighted)

| metric | train | valid |
|---|---|---|
| size mean / median / p5 / p95 | 3.17 / 3 / 1 / 7 | 3.16 / 3 / 1 / 7 |
| atom fraction covered by motifs size ≥ 2 | 0.9331 | 0.9325 |
| occurrence fraction still singleton | 0.2121 | 0.2130 |

Size histogram (train, occurrences): 1→469·(×scaled)… (full histogram in
`SUMMARY.json`): mostly size 2–7, with a tail at size 8. ~21 % of occurrences
are unmerged singleton atoms; 93 % of atoms live in a motif of size ≥ 2.

### 1.6 Port complexity audit

| metric | value |
|---|---|
| external bonds per occurrence `d_j` mean | 2.248 |
| distinct port vertices `p_j` mean / p95 | 1.817 / 3 |
| `p_j / |V(o_j)|` mean / median / p5 / p95 | 0.701 / 0.667 / 0.25 / 1.00 |
| `p_j / |V|` mean over occurrences with `|V| ≥ 4` | **0.471** |
| occurrences with `|V| ≥ 4` | 29 223 |
| fraction of `|V| ≥ 4` occurrences with "almost all atoms ports" (`p_j ≥ |V|-1`) | 0.194 |
| mean joint port-state diversity per motif type | 19.9 |

Large motifs are **not** mostly ports: over size-≥4 occurrences the mean port
fraction is 0.47, and only 19 % are near-fully-ported. Motifs have internal
cohesion (a size-6 or size-7 primitive has ~2 ports on average). **PASS**.

### 1.7 MDL-style proxy (diagnostic only, not tuned)

Empirical `L(x) = -log2 p(x)`, probabilities estimated on train; reported on
valid.

| quantity | bits |
|---|---|
| dictionary cost (all 64 learned motifs) | 1 267.75 (0.127 / graph amortised) |
| graph code mean (`occ count + motif ids + comp edges + bond types + ports`) | 75.32 |
| `L_PSCD` mean total | 75.44 |
| `L_atom` singleton encoding mean | 351.62 |
| **`L_PSCD / L_atom`** | **0.215** |

The structured code is ~4.6× shorter than the atom-level singleton encoding —
consistent with (and stronger than) the pure occurrence-count compression.

### 1.8 Collision ladder (train + valid, 11 000 molecules)

Exact colored canonical keys; ground-truth equivalence = exact iso classes.
Reported as pairs of **non-isomorphic** molecules sharing a code.

| code | n keys | colliding iso classes | colliding class pairs |
|---|---|---|---|
| A motif bag | 10 995 | 2 | **1** |
| B composition-no-port | 10 996 | 0 | 0 |
| C independent (Aut-orbit) port | 10 996 | 0 | 0 |
| D full PSCD | 10 996 | 0 | **0** |

`A ≥ B = C = D = 0`. The **only** observed confusion is a single bag collision;
once motif-level connectivity (B) is kept, ports are no longer needed *on this
corpus* to separate non-isomorphic ZINC molecules (C and D already clean, and
C = D because the relevant two-port configurations are in different Aut orbits
or the motif automorphism group is trivial). This is a **weaker** ladder
separation than the pre-registered *expectation* `A > B > C = 0`: on ZINC the
motif bag is already almost injective (a genuinely interesting negative
sub-result about collision-based arguments here). **Gate A6 (D = 0) PASSES**.

### 1.9 Dictionary visualisation (`results/pscd_compositional/dictionary_top20.{json,png}`)

Top learned primitives by train occurrence (no chemistry names assigned):

| mid | size | occ | support | valid occ | avg ports | structure (node cats; bond types) |
|---|---|---|---|---|---|---|
| 23 | 2 | 3 667 | 3 135 | 347 | 1.85 | {0,2} edge (t1) |
| 21 | 2 | 3 612 | 3 102 | 384 | 1.76 | {0,0} edge (t1) |
| 24 | 3 | 3 589 | 3 005 | 366 | 1.75 | 3-atom chain 0-0-0 (t1,t1) |
| 22 | 4 | 2 585 | 2 115 | 286 | 2.53 | 4-atom 0-tree (t1,t2) |
| 29 | 2 | 2 423 | 2 116 | 230 | 1.41 | {0,1} edge |
| 27 | 6 | 2 310 | 2 141 | 224 | 1.61 | 6-cycle, mixed t1/t2 (ring) |
| 30 | 3 | 2 224 | 2 064 | 232 | 1.94 | 0-1-2 chain |
| 25 | 3 | 2 045 | 1 827 | 191 | 2.26 | 0-0-2 chain |
| 26 | 2 | 1 851 | 1 624 | 179 | 1.51 | {0,4} edge |
| 32 | 3 | 1 762 | 1 630 | 203 | 1.90 | 0-0-1 chain |
| 33 | 7 | 1 334 | 1 289 | 116 | 2.10 | 7-atom ring, mixed t1/t2 |
| 35 | 5 | 1 640 | 1 566 | 145 | 2.46 | branched 5 atoms (t1/t2) |
| 28 | 4 | 1 632 | 1 541 | 148 | 2.87 | 0,0,0,2 4-tree |
| 40 | 7 | 937 | 924 | 91 | 1.49 | 7-atom ring variant (t2/t1/t3) |

The learned primitives are carbon-chain, carbon-ring, and heteroatom (C–N, C–O,
C=O-like) fragments — genuine mesoscale, cross-molecule-reused subgraphs. Common
joint port states are recorded per motif in the JSON.

---

## 2. Stage B — matched tiny-reader capacity probe

Only run because all Stage-A gates passed. Dictionary + merge sequence frozen.
Reader: `d = 32`, `L = 2` sum message passing + SiLU, sum-pool over motif
occurrences, head `Linear(32,32)→SiLU→Linear(32,1)`. Identical optimizer
(Adam, lr 1e-3, wd 0, batch 128, max 300 epochs, patience 40, clip 5), seed 0,
train 10 000 / valid 1 000, official test never loaded.

| reader | input | params | epochs | train MAE | **valid MAE** | peak GPU |
|---|---|---|---|---|---|---|
| **D** composition graph | motif-id nodes + bond-connection nodes + port features | 10 497 | 272 | 0.4884 | **0.5429** | 22 MB |
| **A** motif bag | motif ids only (same embeddings, no composition/ports) | 3 809 | 144 | 0.4822 | **0.5609** | 18 MB |
| **raw** atom graph | atom + bond category embeddings, same MP form | 10 241 | 165 | 0.2835 | **0.3113** | 24 MB |

Pre-registered matched interpretation (§8 of the prereg):

* **strong** (`comp ≤ raw + 0.03` and `M/n ≤ 0.55`): **False** (`comp − raw = +0.232`).
* **composition matters** (`bag − comp ≥ 0.02`): **False** — `bag − comp = 0.018`,
  just under the band. Keeping the relational composition instead of a bag gains
  only ~0.018 MAE under this reader.
* **weak abstraction** (`comp > raw + 0.08`, train MAE also clearly higher):
  **True** (`+0.232`; train 0.488 vs 0.284).

So the compressed composition code is **lossless and ~3× compressive (plus
4.6× under the MDL proxy)**, but under a matched tiny reader it is **~0.23 MAE
worse than the raw atom graph** — and the gap already exists on train, so the
abstraction **hides task-relevant local organisation**, it is not merely a
sample-efficiency / overfitting effect.

Interpretation caveat (recorded, not used to reverse the verdict): the raw
reader gets atom/bond category embeddings directly, whereas the composition
reader sees only a discrete motif **ID** per occurrence and must learn, from
task supervision, a task-relevant summary of each ≤8-atom primitive. The code is
information-theoretically lossless (A1), so the deficit is an *abstraction /
reader-capacity* limitation of a frequency-driven dictionary, not information
destruction.

---

## 3. Answers to the eight questions

**Q1 — does frequency-based merge learn cross-molecule reusable mesoscale
motifs?** **Yes.** 64/64 learned types are used; round-1 fires in 9 992/10 000
graphs; graph-support range 8…3 135; primitives are carbon chains/rings and
heteroatom fragments. Reuse is not an artefact of single-molecule repetition
(occurrence ≈ support for the top motifs).

**Q2 — is a molecule expressed by far fewer motif occurrences than atoms?**
**Yes.** `mean M_G/n_G = 0.32` (train) / 0.32 (valid); 93 % of atoms lie in a
motif of size ≥ 2; only 21 % of occurrences remain singletons.

**Q3 — do dictionary + composition code reconstruct the graph 100 %?**
**Yes.** 10 000/10 000 train and 1 000/1 000 valid, exact colored canonical key.

**Q4 — how much does the motif bag lose?** Almost nothing that is combinatorially
identifiable on ZINC: the bag collides for exactly **one** non-isomorphic pair in
11 000 molecules; the no-port composition code collides for **none**.

**Q5 — how much is lost by keeping motif-level connectivity but no ports?**
**Nothing measurable on this corpus**: B already has **0** non-isomorphic
collisions (C = D = 0). On ZINC, exact ports are not needed to make the code
injective — the ladder separation is much weaker than expected.

**Q6 — does automorphism-aware port structure remove the residual ambiguity?**
The independent-orbit (C) and exact-slot (D) codes are both collision-free and
**equal** here (C = D). The pre-registered section-13 concern (orbit ids
conflating adjacent vs separated joint arrangements) is real in general but does
**not** create a non-isomorphic collision on ZINC (the relevant β-adjacent vs
separated arrangements already lie in different orbits or the motif automorphism
is trivial). So C = D, as the preregistration allowed.

**Q7 — are ports so complex that they re-encode the graph?** **No.** External
bonds per occurrence `d_j = 2.25`; ports `p_j = 1.82` (p95 3); over size-≥4
occurrences the mean port fraction is 0.47. The MDL code is 0.215× the atom
encoding. Port complexity is **not** the failure mode.

**Q8 — under the same tiny reader, how much task capacity does the compressed
composition graph lose vs the raw atom graph?** `valid MAE 0.543 (comp) vs 0.311
(raw)`, i.e. **+0.232** (train 0.488 vs 0.284). The bag is 0.561, so composition
over the bag buys only +0.018. The dictionary abstraction is the bottleneck, not
the composition structure.

---

## 4. Final conclusion and decision

**Port-structured compositional dictionary is lossless, invariant, compressive
and genuinely reusable — but its frequency-driven abstraction is not
task-friendly enough under a matched tiny reader.**

**Final sentence (one of the five frozen):**

> **The compositional dictionary is lossless and compressive, but its abstraction is not task-friendly enough.**

**Next decision (one of the four frozen):**

> **`refine the motif discovery objective`**

Mapping to the frozen options:

* not *proceed to task-driven dictionary selection* — that requires the
  composition reader to be close to raw; it is `+0.232` away.
* not *refine the composition / port representation* — reconstruction is exact
  (A1) and ports are clean (A5); there is nothing to repair there.
* not *stop the compositional dictionary line* — the object itself is healthy on
  all four axes except task capacity, and that failure is localised to the
  frequency-only **choice** of primitives.
* **`refine the motif discovery objective`** — exact composition holds, and the
  deficit is that frequency-selected primitives hide task-relevant local
  organisation (comp and bag both ≈0.55 vs raw 0.31). The next round should
  decide which primitives are worth putting into the dictionary in a
  task-aware way (its own preregistration; this round does **not** start it).

### Revisit if

A task-coupled (but still port-structured, still exactly reconstructable)
dictionary-selection objective is pre-registered and the matched composition
reader moves to within `+0.03` of the raw reader while keeping
`mean(M_G/n_G) ≤ 0.55` and 100 % exact decode. If, under a task-aware dictionary,
the composition reader **still** stays ≥ 0.08 behind raw, then the ladder result
(a motif bag is nearly injective and composition adds ~0.018) plus this Stage-B
gap indicates the whole-graph compressed-code route is not task-friendly and the
line should be stopped rather than refined again.
