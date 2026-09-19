# AIOM — Attributed Incidence Operator Moments as a whole-graph dictionary domain (ZINC)

**Question.** Can a variable-size, permutation-free, attribute-carrying molecular
graph be mapped by an almost-learning-free deterministic lift into a common
finite-dimensional **linear** space that keeps enough structure and attribute
information for sparse dictionary learning to be a reasonable next step?

**Candidate lift.** Attributed Incidence Operator Moments (AIOM), with the
atom–bond incidence graph, its symmetric normalised operator
`S_G = D^{-1/2} A_G D^{-1/2}` and the **raw** one-hot object matrix `X_G`
(no learned projection `W`):

    M_t(G) = X_G^T S_G^t X_G ,  t = 0..8 ;   Phi_T(G) = [vech(M_0); ...; vech(M_T)].

**Status.** Representation analysis only. No dictionary, no ISTA/K-SVD, no
learned projection, no model training as a contribution, no architecture
candidate. This note does not modify `STATE.yaml`, any pre-registration or any
historical result. Official ZINC `test` was **never** loaded. `y` is used only
in the capacity probe (Test F) and its controls.

---

## 0. Provenance

| item | value |
|---|---|
| git commit (analysis start) | `1c47309046d16a105178a152b4d76372c5856a66` |
| worktree at start | clean (new analysis files untracked); branch `main` |
| splits used | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000** |
| official `test` | **never read / instantiated / referenced** |
| target `y` | used **only** in Test F |
| data source | `data/ZINC` (`torch_geometric.datasets.ZINC`, `subset=True`) |
| loader | `tracks/ksvd/experiments/luyin16/zinc_long_range_proxy._load_zinc` + `_data_to_graph` |
| exact canonicalization | reused `pynauty` 2.8.8.1 colored incidence (`canonical_atom_order`) |
| Python / numpy / torch | 3.12 / 2.1.3 / 2.5.1+cu124 (CPU run) |
| analysis script | `tracks/ksvd/code/run_aiom_representation_audit.py` |
| probe control | `tracks/ksvd/code/run_aiom_probe_control.py` |
| tests | `tracks/ksvd/tests/test_aiom.py` (8 data-free tests, pass) |
| results | `tracks/ksvd/results/aiom_representation/` (git-ignored) |
| reproduce | `env PYTHONPATH=. uv run python tracks/ksvd/code/run_aiom_representation_audit.py` (~223 s CPU) |

### 0.1 Confirmed graph semantics (from the loader, not textbook ZINC)

| quantity | value |
|---|---|
| atom categories `C_V` | **21** (`{0,…,20}`) |
| bond categories `C_E` | **3** (`{1,2,3}` → re-indexed `{0,1,2}`) |
| `C = C_V + C_E` | **24** |
| bond storage | each undirected bond stored as **two** directed rows in `edge_index` |
| directed-copy consistency | every bond's two copies have equal `edge_attr` (500/500 graphs) |
| self-loops | **0** (500-graph audit) |
| multi-edges | **0** (500-graph audit) |
| graph size | min 9, median 23, p95 31, max 37 (train+valid) |
| atom degree | min 1, median 2, max 4 |
| isomorphic duplicates | 4 duplicate-only groups; train has 9 997 iso classes for 10 000 graphs |

A bond object is one real chemical bond (deduplicated from the two directed
copies); `m` bonds ⇒ `N = n + m` incidence objects.

---

## 1. The lift, and what each moment order means

`B ∈ {0,1}^{n×m}`, `A_G = [[0,B],[Bᵀ,0]] ∈ R^{N×N}`,
`D_G = diag(A_G 1)`, `S_G = D_G^{-1/2} A_G D_G^{-1/2}`.
`X_G ∈ {0,1}^{N×C}`: atom rows activate only the first `C_V` channels; bond rows
only the last `C_E` channels (object type is already separated by the channel
block). `M_t = X_G^T S_G^t X_G ∈ R^{C×C}`; only the upper triangle is kept.

* **M₀ = XᵀX** is diagonal: the counts of each raw atom category and each raw
  bond category in the whole molecule (a bag-of-atom/bond-attributes statistic).
* **odd t (M₁, M₃, …)**: `S^t` is block off-diagonal (atom↔bond), so these
  blocks are weighted *atom-type ↔ bond-type* odd-hop interactions
  (`[M₁]_{c_a,c_b} = Σ_{v–e} 1/√(d_v d_e)` over incidences).
* **even t (M₂, M₄, …)**: `S^t` is block diagonal, so these are weighted
  *atom↔atom* and *bond↔bond* even-hop attributed relationships.

`{M_t}_{t=0..T}` is therefore a multiscale set of global attribute-response
moments of the incidence operator. It is **not** claimed to be an injective
graph representation; one purpose of this audit is to measure how much it loses.

**Why not the larger Krylov Gram.** With `H_r = S^r X`, `H_rᵀH_s = XᵀS^{r+s}X =
M_{r+s}` because `S` is symmetric; the full `[H_0..H_R]ᵀ[H_0..H_R]` contains
blocks that depend only on `r+s`. Studying `M_0..M_T` directly is the compact,
transparent form. With `C = 24`, each moment is `300`-D; `T = 8` gives `2700`-D,
which is trivially affordable.

---

## 2. Test A — strict permutation invariance

500 train graphs × 20 random atom + bond-object permutations = **10 000
checks**; float64 CPU reference.

| metric | value |
|---|---|
| checks | 10 000 |
| max abs err `max|Phi_8(G') - Phi_8(G)|` | **1.78 × 10⁻¹⁴** |
| target | < 10⁻¹⁰ → **PASS** |

The lift is *exactly* permutation invariant (the numerical residual is pure
floating-point round-off). The D4 negative control below reproduces this
(δ ≈ 10⁻¹⁶).

---

## 3. Test B — collision audit (10 000 train + 1 000 valid)

Ground truth = exact isomorphism class from the reused colored-incidence
`pynauty` canonical key (not a WL hash). Feature hashes from deterministic
float64 rounded at 10⁻¹⁰ and 10⁻¹² give identical results.

| T | feature dim | non-iso collision graphs | fraction | collision groups | non-iso colliding pairs | duplicate-only groups |
|---|---|---|---|---|---|---|
| 0 | 300 | 1 258 | **11.44 %** | 581 | 795 | 4 |
| 1 | 600 | 81 | 0.74 % | 40 | 42 | 4 |
| 2 | 900 | 2 | 0.018 % | 1 | 1 | 4 |
| 3 | 1200 | 2 | 0.018 % | 1 | 1 | 4 |
| 4 | 1500 | 0 | **0** | 0 | 0 | 4 |
| 6 | 2100 | 0 | **0** | 0 | 0 | 4 |
| 8 | 2700 | 0 | **0** | 0 | 0 | 4 |

The collision rate collapses rapidly: 11.4 % (`T=0`, the count baseline) →
0.74 % (`T=1`) → a single pair at `T=2,3` → **exactly zero non-isomorphic
collisions at `T ≥ 4`**. On this dataset AIOM `T=8` is a complete invariant of
the sampled attributed graphs.

**Near collisions.** Train-only standardisation; nearest *non-isomorphic*
neighbour per graph, standardised Euclidean distance:

| T | min | p1 | median | p95 |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.697 | 1.837 |
| 1 | 0.000 | 0.024 | 1.391 | 3.142 |
| 2 | 0.000 | 0.324 | 2.229 | 5.208 |
| 4 | 0.020 | 0.576 | 3.219 | 8.027 |
| 8 | 0.045 | 0.994 | 4.867 | 13.419 |

Nearest non-isomorphic pairs are saved in `near_collision_pairs.csv`.

---

## 4. Test C — intrinsic rank / redundancy

Train-only standardised `F_T ∈ R^{10000×d_T}`, full SVD.

| T | dim | numerical rank | eff. rank (entropy) | eff. rank (participation) | dims @90 % | dims @95 % | dims @99 % |
|---|---|---|---|---|---|---|---|
| 0 | 300 | 24 | 20.85 | 18.80 | 19 | 20 | 23 |
| 1 | 600 | 39 | 26.15 | 23.11 | 23 | 26 | 29 |
| 2 | 900 | 85 | 39.29 | 30.94 | 36 | 43 | 53 |
| 3 | 1200 | 99 | 38.71 | 30.16 | 35 | 44 | 56 |
| 4 | 1500 | 173 | 50.90 | 36.62 | 49 | 64 | 81 |
| 6 | 2100 | 294 | 63.78 | 42.35 | 66 | 87 | 113 |
| 8 | 2700 | 435 | 73.46 | 46.76 | 80 | 102 | 133 |

Higher orders **do** add independent directions (entropy rank 20.9 → 73.5;
99 %-energy dimension 23 → 133), so the lift is not purely repeating low-order
information. But the growth is heavily sublinear and the vector is very
redundant: at `T=8`, 99 % of the energy sits in 133 of 2700 coordinates and the
entropy-effective rank is 73. Doubling the dimension `T=4 → 8`
(1500 → 2700) raises the effective rank only 50.9 → 73.5.

---

## 5. Test D — controlled perturbation sensitivity (label-free)

500 train graphs; relative δ_T = `||Phi_T(G')-Phi_T(G)|| / (||Phi_T(G)||+ε)`;
mean over successfully constructed graphs. D3 two-switch construction rate
500/500; D2 bond-type swap 494/500 (needs two distinct bond types); D1 500/500.

| kind | T=0 | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|
| permutation (negative control) | 0 | 2.7e-17 | 7.2e-17 | 1.1e-16 | 1.3e-16 |
| atom-type substitution | 0.0571 | 0.0673 | 0.0657 | 0.0685 | 0.0710 |
| bond-type reassignment (multiset fixed) | **0.0000** | 0.0281 | 0.0307 | 0.0321 | 0.0311 |
| degree-preserving 2-switch | **0.0000** | 0.0072 | 0.0115 | 0.0135 | 0.0148 |
| random size-matched pair | 0.2087 | 0.2412 | 0.2460 | 0.2576 | 0.2676 |

ECDF / boxplot: `plots/perturbation_ecdf.png`.

Reading (theory-consistent): `T=0` (pure counts) is *exactly* blind to topology
and to bond-type placement, as predicted, and starts responding only once
`t ≥ 1` is added. The ordering is
permutation ≈ 0 ≪ two-switch < bond-type-swap < atom-substitution ≪ random
molecule. So AIOM **is** sensitive to degree-preserving rewiring, but the
response is small — ~18× smaller than for a genuinely different molecule of the
same size, and only ~2 slots of numerical noise above the permutation control
for the median 2-switch.

---

## 6. Test E — nearest-neighbour structural fidelity

Independent external metric: typed 3-round WL **subtree** histogram (atom
colour + `(neighbour colour, bond type)` refinement on the incidence graph),
cosine similarity. This is a similarity audit only, never an isomorphism proof.

| T | top-1 neighbour WL cosine | top-10 mean WL cosine | size-matched random control | Spearman(`d_Phi`, `d_WL`) |
|---|---|---|---|---|
| 0 | 0.9414 | 0.9370 | 0.9146 | 0.206 |
| 1 | 0.9419 | 0.9367 | 0.9146 | 0.197 |
| 2 | 0.9407 | 0.9351 | 0.9146 | 0.186 |
| 4 | 0.9405 | 0.9345 | 0.9146 | 0.179 |
| 8 | 0.9398 | 0.9337 | 0.9146 | 0.166 |

Φ-neighbours are consistently but only marginally more similar than
size-matched random molecules (+0.025–0.027 WL cosine). The Spearman
correlation between Φ-distance and WL-distance is ≈0.17–0.21 and *decreases*
with `T`. The Φ geometry is only weakly aligned with an independent structural
metric; the higher-order coordinates do not improve the correspondence.

---

## 7. Test F — dense capacity probe (the only `y` use)

Raw target (no normalisation; train mean 0.015, std 2.011; constant-train-mean
valid MAE = 1.478 is the trivial floor). Standardisation statistics from train
only; fixed protocol mirrors the ZINC protocol (Adam lr 1e-3, wd 1e-5, batch
128, L1, clip 5, max 240 epochs, patience 40 on valid). MLP =
`Linear(d,64)→SiLU→Linear(64,1)`.

| T | linear train | linear valid | **MLP train** | **MLP valid** | ridge valid |
|---|---|---|---|---|---|
| 0 | 0.5891 | 0.6083 | 0.5495 | 0.5663 | 0.6489 |
| 1 | 0.5408 | 0.5556 | 0.4667 | 0.4932 | 0.5989 |
| 2 | 0.5114 | 0.5214 | 0.4299 | 0.4717 | 0.5596 |
| 3 | 0.4734 | 0.4770 | 0.4152 | 0.4395 | 0.5249 |
| 4 | 0.4561 | 0.4607 | 0.3927 | **0.4308** | 0.5140 |
| 6 | 0.4581 | 0.4610 | 0.3902 | 0.4367 | 0.5198 |
| 8 | 0.4526 | 0.4643 | 0.3790 | 0.4459 | 0.5228 |

MLP seed 1/2 (stability): `T=2` valid 0.469 / 0.472; `T=8` valid 0.451 / 0.448.

**Probe-adequacy controls** (same fixed architecture/protocol, `y` only to
score; `probe_control_results.json`):

| input features | dim | MLP train | MLP valid |
|---|---|---|---|
| typed 3-round WL subtree histogram | 121 | 0.5194 | 0.6114 |
| AIOM `T=8` | 2700 | 0.3790 | 0.4459 |
| AIOM `T=8` + WL | 2821 | 0.3405 | 0.4713 |
| complete canonical descriptor (slot coords) | 6290 | 0.4177 | 0.6744 |

Additional sensitivity (`probe_sensitivity.json`): widening the hidden layer
(64/256/512) does **not** improve valid MAE (0.43–0.45 at `T=4,8`). Training the
MLP for 1500 epochs without early stopping drives the complete canonical
descriptor's **train** MAE to 0.114 (it can memorise), but AIOM `T=8`'s train MAE
floors at **0.285** and its valid MAE stays ≈0.49 — i.e. AIOM's ceiling is not a
probe-capacity artefact; the moment vector does not expose enough decodable
information even for the training set.

Plots: `plots/probe_linear_mae_vs_T.png`, `plots/probe_mlp_mae_vs_T.png`.

---

## 8. Answers to the seven required questions

* **Q1 — Invariance.** **Yes, exact.** `Phi_T(ΠG) = Phi_T(G)`; 10 000 checks,
  max abs error 1.78e-14 ≪ 1e-10.
* **Q2 — Collision.** **Yes, it collapses to zero.** Non-isomorphic exact
  collision fraction 11.44 % (`T=0`) → 0.74 % (`T=1`) → 0.018 % (`T=2,3`) →
  **0 % for `T ≥ 4`**; at `T=8` no two non-isomorphic graphs share a feature
  hash.
* **Q3 — Independent information.** Higher moments **do** add rank, but with
  strong redundancy and sublinear returns: entropy-effective rank 20.9 → 73.5,
  99 %-energy dims 23 → 133 over `T=0..8`; `T=4 → 8` doubles the dimension but
  raises the effective rank only 50.9 → 73.5.
* **Q4 — Structural sensitivity.** **Yes, but weak.** A degree-preserving
  2-switch (atom counts, bond-type multiset and degree sequence all fixed)
  changes `Phi` by δ = 0 at `T=0` and ≈0.015 at `T=8` — clearly above the
  permutation control (≈0) yet ~18× below a random size-matched molecule pair
  (0.27). Bond-type reassignment with fixed multiset gives δ ≈ 0.03.
* **Q5 — Geometry.** **Weak.** Φ-space top-1 non-isomorphic neighbours have WL
  cosine 0.940 vs 0.915 for size-matched random (gap +0.025); top-10 0.934;
  Spearman(`d_Phi`, `d_WL`) ≈ 0.17–0.21 and decreasing in `T`.
* **Q6 — Task-information adequacy.** **Weak / severe.** Small-MLP valid MAE
  bottoms at **0.431** (`T=4`) and is 0.446 at `T=8`; **train** MAE is 0.379 at
  `T=8` (floor 0.285 over long training) and never enters the 0.15–0.20
  promising band. → **> 0.20 ⇒ weak; train > 0.25 ⇒ severe information loss.**
* **Q7 — Order saturation.** Information grows `T=0 → 4` (MLP valid 0.566 →
  0.431; rank 20.9 → 50.9) and then **saturates and slightly reverses**
  (`T=6` 0.437, `T=8` 0.446). Saturation is at **T ≈ 4**; higher-order moments
  add rank but no task value.

---

## 9. Interpretation

**Facts directly supported by the data.**

* AIOM is an exact permutation-invariant whole-graph lift and is (on this
  dataset) essentially injective: zero non-isomorphic collisions for `T ≥ 4`.
* It carries real structural signal: rewiring and typed-bond placement change
  `Phi` above the exact permutation-noise floor, and effective rank grows with
  `T`.
* Its predictive ceiling with a generous dense readout is low: best valid MAE
  0.431 (`T=4`), and even the **train** MAE cannot be driven below ~0.29.
* Among fixed permutation-invariant whole-graph features tested, AIOM is the
  best (MLP valid 0.446 vs 0.611 for the WL subtree histogram and 0.674 for the
  complete canonical descriptor), yet all are far above the strong zone.
* Adding the WL histogram to AIOM does not improve valid MAE.

**My interpretation (not a direct fact).**

* AIOM passes every *formal* requirement — invariance, near-injectivity,
  monotone rank growth, structural sensitivity — but fails the decisive
  *readout-accessibility* requirement. The moments compress the typed graph into
  nine 24×24 matrices; the fine local configuration needed for the ZINC target
  is present but not linearly/MLP-decodable. A dense MLP is strictly more
  expressive than a linear sparse-dictionary readout, so its ceiling upper-bounds
  what a dictionary can recover.
* The probe protocol is fair: the same probe can memorise a complete descriptor
  (train 0.114), so AIOM's 0.29 train floor is a property of the lift, not of
  the probe. The reference ZINC GNNs (valid ≈0.12) are a different hypothesis
  class (learned message passing), so 0.15 is not literally attainable by a
  fixed-feature MLP; this caveat is stated explicitly and does not change the
  verdict that AIOM is insufficient for a *dictionary* domain.
* The non-monotone step `T=2 → 3` in effective rank and the declining
  Φ↔WL Spearman correlation with `T` suggest that the extra odd/even hop
  coordinates mostly add redundant, poorly-aligned directions rather than
  genuinely new structure.

---

## 10. Final

> **AIOM is not an adequate whole-graph dictionary domain.**

### Direct representation results

* **Dimensions:** `C_V=21`, `C_E=3`, `C=24`; 300 D/moment; `Phi_T` up to 2700 D
  at `T=8`.
* **Invariance:** exact, max abs err 1.78e-14 over 10 000 permutation checks.
* **Collisions:** non-isomorphic exact collision fraction 11.44 % (`T=0`) →
  0.74 % (`T=1`) → 0.018 % (`T=2,3`) → **0 % for `T ≥ 4`**.
* **Rank:** entropy-eff 20.9 → 73.5; numerical rank 24 → 435; 99 %-energy dim
  23 → 133 (`T=0..8`).
* **Perturbations:** permutation ≈ 0; degree-preserving 2-switch 0.000 → 0.015;
  bond-type reassignment 0.000 → 0.031; atom substitution 0.057 → 0.071; random
  pair 0.209 → 0.268.
* **NN fidelity:** top-1 WL cosine 0.940 vs 0.915 random; Spearman(`d_Phi`,
  `d_WL`) 0.17–0.21.

### Capacity probe

`T = 0,1,2,3,4,6,8` train/valid MAE (linear and MLP) as tabulated in §7; MLP
valid best **0.431** (`T=4`); `T=8` MLP train/valid **0.379 / 0.446**. Negative
results are reported as-is; no `T` reaches the 0.15–0.20 band.

### Next decision

> **`redesign the graph-to-dictionary lift`**

Rationale: AIOM has genuine, exactly permutation-invariant structural signal and
is collision-free, so the *idea* of a fixed operator-moment lift is not
disqualified — but the current moment family aggregates the typed local
configuration into a redundant, poorly-decodable object, and the dense readout
ceiling proves the bottleneck is the lift, not the solver. The next round should
redesign `Phi(G)` (operator choice, richer/different attribute probes, higher-
order or per-object moments, or a different aggregation) and re-run this same
audit **before** any sparse dictionary learning. Implementing a dictionary on
the current AIOM lift is not warranted.
