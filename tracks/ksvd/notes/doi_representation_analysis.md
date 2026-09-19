# DOI — Distributional Operator-Incidence Lift as a whole-graph dictionary domain (ZINC)

**Question.** AIOM failed with `Phi_T(G) = [vech(M_0); ...; vech(M_T)]`
(`M_t = X_G^T S_G^t X_G`). Is the failure mainly because `X^T S^t X` aggregates
object-level heterogeneity *before* any nonlinear representation? If we keep each
atom/bond object's multiscale operator response `r_i^(T) = [H_0[i]; ...; H_T[i]]`
(`H_t = S_G^t X_G`) and embed the *empirical distribution* of those responses —
plus the empirical atom–bond **incidence-pair** distribution — do we get a
clearly more task-readable object that is still permutation-invariant,
fixed-dimensional and dictionary-compatible?

**Candidate lift (DOI).**

    mu_V(G) = mean_{v in V} psi_V(r_v)
    mu_E(G) = mean_{e in E} psi_E(r_e)
    mu_I(G) = mean_{(v,e): B_ve=1} psi_I(q_ve) ,  q_ve = [r_v ; r_e]
    Phi_DOI(G) = [ mu_V ; mu_E ; mu_I ; log(1+n) ; log(1+m) ]

with three independent Gaussian kernels `k_V, k_E, k_I` (train-only
median-heuristic bandwidths) approximated by fixed-seed Random Fourier Features.
The operator `S_G = D^{-1/2} A_G D^{-1/2}` and the raw one-hot `X_G` are **exactly
the AIOM ones**, so AIOM → DOI differences come only from the aggregation.

**Status.** Representation audit only. No dictionary, no ISTA/K-SVD, no learned
projection/encoder, no model training as a contribution. Official ZINC `test` was
**never** loaded. `y` is used only in the capacity probe (Test F). This note does
not modify `STATE.yaml`.

---

## 0. Provenance

| item | value |
|---|---|
| git commit | `fe38e7f5af465236bf2b8d3cfca7f743f24d991e` |
| worktree at run | clean |
| splits used | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000** |
| official `test` | **never read / instantiated / referenced** |
| target `y` | used **only** in the capacity probe |
| loader / operator / perturbations / probe | reused from `run_aiom_representation_audit.py` |
| exact canonicalization | reused colored-incidence `pynauty` (`iso_key_hex`) |
| analysis script | `tracks/ksvd/code/run_doi_representation_audit.py` |
| tests | `tracks/ksvd/tests/test_doi.py` (9 data-free tests, pass) |
| results | `tracks/ksvd/results/doi_representation/` (git-ignored) |
| execution | remote A100 host `res`, CPU numpy + CPU torch probes, **1798 s** |
| reproduce | `uv run python tracks/ksvd/code/run_doi_representation_audit.py` |

Confirmed graph semantics from the loader: `C_V=21`, `C_E=3`, `C=24`; each
undirected bond stored as two directed copies with equal type; 0 self-loops; 0
multi-edges; response dims `T=0/2/4 → d=24/72/120`.

---

## 1. Fixed multiscale schedule

`T=4` is the primary (AIOM already saturated near `T=4`). Prefixes `T∈{0,2,4}`
are saved for the multiscale study; `T>4` was **not** tested. At `T=4` the code
selected `D=1024` (see §3), so `Phi_DOI ∈ R^{3·1024+2} = R^{3074}`.

---

## 2. Test A — strict permutation invariance (all three views)

500 train graphs × 20 random atom + bond-object permutations = **10 000 checks**,
float64 CPU reference, **recomputing** `r_v, r_e, q_ve, Phi` from the permuted
`Mol`.

| metric | value |
|---|---|
| checks | 10 000 |
| `max|Phi(ΠG) − Phi(G)|` full / object-only / incidence-only | **3.47 × 10⁻¹⁷** |
| target | < 10⁻¹⁰ → **PASS** |

DOI is exactly permutation invariant for all three views (pure float round-off).

---

## 3. Test B — RFF approximation sanity + label-free dimension selection

Exact Gaussian KME inner products on a fixed **train-only** calibration subset
(300 graphs, 2000 pairs). Candidate `D∈{512,1024,2048}`; the rule picks the
smallest `D` with **median abs error < 0.02 and p95 < 0.05 for all three
kernels**. `y` and valid are never involved.

| T | D selected | worst-kernel median abs err | worst-kernel p95 | min Spearman over V/E/I |
|---|---|---|---|---|
| 0 | 1024 | 0.0117 (E) | 0.0134 | 0.9960 |
| 2 | 2048 | 0.0141 (V) | 0.0213 | 0.9955 |
| 4 | 1024 | 0.0115 (E) | 0.0240 | 0.9947 |

All selected `D` pass comfortably; pairwise-kernel Spearman ≥ 0.9947. Plots:
`plots/exact_vs_rff_kernel.png`. **RFF approximation is adequate — it is not the
limiting factor.**

---

## 4. Response scaling & bandwidths (train-only, label-free)

RMS scaling `r̃_j = r_j/(s_j+ε)` with `s_j = sqrt(E_train[r_j²])`, separately for
atom and bond responses (no mean subtraction). `T=4`: `s_V` has 51/120 coordinates
exactly 0 (zero-variance channels safely kept at 0), `s_E` 69/120.

Median-heuristic bandwidths `σ = median‖r−r'‖` over ≤50 000 train pairs:

| T | σ_V | σ_E | σ_I |
|---|---|---|---|
| 0 | **0.000 → fallback 3.453** | **0.000 → fallback 2.296** | 2.296 |
| 2 | 4.073 | 3.110 | 4.871 |
| 4 | 4.888 | 4.107 | 6.287 |

`T=0` is the one degenerate case: the response is the raw one-hot object type,
and carbon is ~70 % of atoms, so >50 % of random pairs are *identical* → median
distance 0. This was reported and handled with a documented positive-pair-median
fallback; it is itself a finding that `T=0` is not a useful distributional object.

---

## 5. Test C — geometry / near-collision (train-only standardised)

Nearest **non-isomorphic** neighbour per graph (exact `pynauty` iso classes),
standardised Euclidean distance:

| rep | dim | min | p1 | p5 | median | p95 |
|---|---|---|---|---|---|---|
| AIOM T=4 | 1500 | 0.020 | 0.576 | 1.089 | 3.219 | 8.027 |
| Full DOI T=4 | 3074 | 0.223 | 4.393 | 6.840 | 16.709 | 37.599 |
| Object-only T=4 | 2050 | 0.184 | 3.420 | 5.453 | 13.515 | 30.824 |
| Incidence-only T=4 | 1026 | 0.125 | 2.652 | 4.104 | 9.660 | 21.204 |

DOI's minimum non-isomorphic distance is ~10× AIOM's (0.223 vs 0.020): the
distributional object **does not** put near-duplicate coordinates on distinct
graphs, and its non-isomorphic neighbours are far more separated. Geometry is
healthy (no artificial near-collisions). Nearest pairs:
`nearest_pairs_full.csv`.

---

## 6. Test D — controlled perturbation sensitivity (500 graphs)

Raw relative change `‖Φ(G')−Φ(G)‖/(‖Φ(G)‖+ε)` (mean):

| kind | AIOM T=4 | Full DOI T=4 | Object-only | Incidence-only |
|---|---|---|---|---|
| permutation (control) | ~0 | ~0 | ~0 | ~0 |
| atom-type substitution | 0.0693 | 0.0666 | 0.0546 | 0.0402 |
| bond-type reassignment | 0.0328 | 0.0111 | 0.0093 | 0.0065 |
| degree-preserving 2-switch | 0.0134 | 0.0092 | 0.0074 | 0.0056 |
| random size-matched pair | 0.2529 | 0.1015 | 0.0824 | 0.0612 |

Standardised `Δ_Φ` (train feature μ,σ), mean + random/rewiring ratio:

| rep | 2-switch (mean) | random (mean) | random / rewiring | relative random / rewiring |
|---|---|---|---|---|
| AIOM T=4 | 4.59 | 17.06 | **3.72** | 18.9 |
| Full DOI T=4 | 6.75 | 72.22 | **10.70** | 11.0 |
| Object-only T=4 | 5.40 | 57.61 | 10.66 | 11.1 |
| Incidence-only T=4 | 4.03 | 41.50 | 10.31 | 10.9 |

DOI *is* sensitive to degree-preserving rewiring (above the exact permutation
floor), but its rewiring relative response stays **~1 %** (0.92 % mean / 0.57 %
median) and is still dwarfed by random size-matched pairs — on the standardised
metric the random/rewiring gap actually **widens** (3.7× → 10.7×), while on the
relative metric it narrows only modestly (18.9× → 11.0×). **The distributional
lift did not fix AIOM's core topology-geometry problem.**

---

## 7. Test E — nearest-neighbour structural fidelity (independent metric)

External metric: typed 3-round WL subtree histogram cosine (never used to *define*
DOI similarity):

| rep | top-1 WL sim | top-10 mean WL sim | size-matched random | Spearman(d_Φ, d_WL) | top-10 overlap |
|---|---|---|---|---|---|
| AIOM T=4 | 0.9401 | 0.9343 | 0.9145 | 0.191 | 0.045 |
| Full DOI T=4 | 0.9405 | 0.9350 | 0.9145 | **0.303** | 0.048 |
| Object-only T=4 | 0.9406 | 0.9351 | 0.9145 | 0.303 | 0.047 |
| Incidence-only T=4 | 0.9404 | 0.9350 | 0.9145 | 0.300 | 0.047 |

DOI improves the Φ↔WL **distance correlation** (0.19 → 0.30) but the absolute
neighbour fidelity is essentially unchanged (top-1 0.940 → 0.940/0.941; top-10
0.934 → 0.935). The DOI geometry is a *little* better aligned with structure,
but the top neighbours are still only ~+0.026 WL cosine above size-matched
random — the same weak regime as AIOM.

---

## 8. Test F — multiscale contribution (Full DOI)

| T | D | dim | rewiring std Δ | top-1 WL | top-10 WL | MLP train / valid MAE |
|---|---|---|---|---|---|---|
| 0 | 1024 | 3074 | 0.948 | 0.9393 | 0.9349 | 0.4894 / 0.5200 |
| 2 | 2048 | 6146 | 7.712 | 0.9408 | 0.9354 | 0.3727 / 0.4029 |
| 4 | 1024 | 3074 | 6.749 | 0.9405 | 0.9350 | **0.3273 / 0.3865** |

Propagation is essential (T=0 is blind to rewiring and much worse on the probe),
and T=2→4 still improves the probe (valid 0.4029 → 0.3865) and lowers the train
error. But rewiring sensitivity and WL fidelity are already saturated by T=2.
This is a genuine multiscale gain, not a "best-T search".

---

## 9. Capacity probe — the only use of `y`

Same fixed probe as AIOM (`Linear(d,1)`; `Linear(d,64)→SiLU→Linear(64,1)`; Adam
lr 1e-3, wd 1e-5, batch 128, L1, clip 5, max 240 epochs, patience 40 on valid;
raw `y`; train-only standardisation). Trivial constant-train-mean valid MAE ≈ 1.48.

| rep | T | dim | linear train / valid | **MLP train / valid** |
|---|---|---|---|---|
| AIOM (frozen) | 4 | 1500 | 0.4579 / 0.4617 | 0.3956 / **0.4321** |
| Full DOI | 0 | 3074 | 0.5742 / 0.5784 | 0.4894 / 0.5200 |
| Full DOI | 2 | 6146 | 0.4528 / 0.4591 | 0.3727 / 0.4029 |
| **Full DOI** | **4** | **3074** | **0.4145 / 0.4185** | **0.3273 / 0.3865** |
| Object-only | 4 | 2050 | 0.4293 / 0.4363 | 0.3454 / 0.3975 |
| Incidence-only | 4 | 1026 | 0.4351 / 0.4366 | 0.3576 / 0.4056 |
| DOI-SUM (control) | 4 | 3072 | 0.4278 / 0.4383 | 0.3417 / 0.3937 |

* **Full DOI T=4 beats frozen AIOM T=4 by 0.0456 valid (0.4321 → 0.3865, −10.5 %)
  and 0.0683 train (0.3956 → 0.3273).** Real but modest.
* **Incidence adds a little over object-only**: full 0.3865 vs obj 0.3975
  (valid Δ = 0.011; train Δ = 0.018), and incidence-only (0.4056) is close.
* **Mean-embedding + separate mass beats the raw-sum control** only marginally
  (full 0.3865 vs SUM 0.3937, Δ = 0.007).
* **Ceiling text (probe ceiling): NOT triggered** — the improvement over AIOM is
  0.0456 < 0.05, so the pre-registered 256-wide Full-DOI diagnostic was not run.
* **Weak band.** Full DOI T=4 valid 0.3865 > 0.30 and train 0.3273 > 0.25, so
  under the pre-registered gates this is **Case C (Weak)**: DOI still has a
  serious information / readout bottleneck. The dense MLP is strictly more
  expressive than any linear sparse-dictionary readout, so this ceiling
  upper-bounds any dictionary on the same object.

> Numerical note: the `linear_ridge` control (normal-equation solve, α=1e-3) is
> unreliable for the DOI features (valid MAE 0.74–1.44) — `XᵀX` is very
> ill-conditioned on near-collinear RFF coordinates. It is a diagnostic only and
> is not used for any conclusion; the Adam-trained linear probe is the linear
> comparison.

---

## 10. Answers to the required questions

* **Q1 — Permutation invariance?** **Yes, exact.** 10 000 checks, max abs err
  3.47e-17 ≪ 1e-10, for full / object-only / incidence-only.
* **Q2 — Is RFF adequate (not approximation-noise-dominated)?** **Yes.** Every
  selected D meets median<0.02, p95<0.05 with pairwise-kernel Spearman ≥0.995
  (worst p95 0.024). Approximation does not explain the probe ceiling.
* **Q3 — Does keeping per-object response distributions significantly improve
  AIOM's graph geometry?** **Partially.** Separation is much healthier (min
  non-iso distance 0.020 → 0.223; no near-collisions) and Φ↔WL distance
  correlation rises 0.19 → 0.30, but absolute WL neighbour fidelity barely moves
  (top-1 0.940 → 0.940/0.941). Not a decisive geometry improvement.
* **Q4 — Does the explicit incidence-pair distribution add real endpoint-
  assignment sensitivity?** **Weakly.** Incidence-only is the *most* rewiring-
  sensitive view in relative terms (2-switch 0.0056) and adding it improves the
  probe by Δvalid 0.011 over object-only. But the increment is small and the
  rewiring response stays ~1 % relative.
* **Q5 — Who carries what?** Full (0.3865) < object-only (0.3975) < incidence-only
  (0.4056). Atom+bond object distributions carry most of the signal; the
  incidence-pair distribution adds a small but consistent extra amount.
* **Q6 — Does multiscale structure (T=0→2→4) bring clear gains?** **Yes.** T=0 is
  blind to rewiring (Δ=0.95, essentially the mass terms) and far worse on the
  probe (0.5200); T=2 fixes both; T=4 still improves the probe (0.4029 → 0.3865)
  although rewiring/WL fidelity saturate at T=2.
* **Q7 — Does Full DOI clearly break the AIOM 0.431 ceiling?** **Only modestly,
  not decisively.** Dense MLP valid 0.3865 vs 0.4321 (−0.046), train 0.3273 vs
  0.3956. It stays in the Weak band (>0.30), and no 256-wide ceiling test was
  authorised.
* **Q8 — If it still fails, what kind of failure is it?** Predominantly the
  **fixed-lift premise**, with distributional aggregation a real but secondary
  factor and the fixed operator not clearly the sole bottleneck. Evidence:
  (i) RFF is accurate and the geometry is healthy/non-collapsing, so the object
  is not under-resolved or colliding; (ii) a dense MLP — strictly stronger than
  any dictionary readout — still floors at train 0.327 > 0.25, so the *fixed,
  label-free, invariant* descriptor does not expose enough task-decodable
  information; (iii) switching the aggregation from global moments to per-object
  distributions + incidence pairs buys only ~0.05 valid, so aggregation alone is
  not the cure; (iv) rewiring sensitivity stays ~1 % relative → the fixed
  operator still does not encode topological assignment into an easily readable,
  task-aligned geometry.

---

## 11. Interpretation

**Supported by the data.**
* DOI is an exact permutation-invariant whole-graph lift, RFF-accurate, and its
  object/incidence views have no near-collar collisions (min non-iso distance
  ~0.22 vs AIOM 0.02).
* Per-object response distributions + the incidence-pair distribution carry
  *more* decodable task signal than AIOM's global moments (valid 0.432 → 0.387,
  train 0.396 → 0.327), and multiscale propagation is genuinely useful.
* But the dense readout ceiling is still 0.387 valid / 0.327 train: the
  distributional lift reduces but does not remove the readout bottleneck.
* The incidence term contributes only a small increment, and degree-preserving
  rewiring still moves Φ by ~1 % relative (random pairs remain 11× larger on the
  standardised metric).

**My interpretation (not a direct fact).** The distributional-lift idea is
*validated* — it improves the geometry and the readout meaningfully — but the
object is still a **fixed, label-free, invariant** function of the graph, and
that constraint is what caps the readout. The evidence (accurate RFF, healthy
geometry, but train MAE > 0.25 for a dense MLP) is exactly the pattern the
pre-registration flagged: this is not "not enough information" in the sense of a
lossy/colliding lift; it is that the **task geometry of a fixed invariant
distribution lift is still not decodable**. That points to allowing some
task-coupled structural transformation rather than adding more RFF dimensions.

---

## 12. Final conclusion

> **The fixed-lift premise is likely the bottleneck; a task-coupled structural
> lift is required.**

The distributional aggregation is a genuine improvement (AIOM 0.432 → DOI 0.387
valid, geometry repaired, RFF verified), but DOI remains in the Weak band with a
fixed, label-free lift, so more aggregation/RFF tuning inside the same premise is
unlikely to reach a dictionary-ready domain.

### Next decision (one)

> **`move to a task-coupled minimal structural lift`**

### What this round did **not** do (per pre-registration)

No dictionary, ISTA, K-SVD, learned RFF/bandwidth, GNN, learned diffusion,
higher `T`, higher-order incidence tuples or motif distributions; the next step is
not implemented here.

---

## 13. Machine-readable outputs (`results/doi_representation/`, git-ignored)

`SUMMARY.json`, `provenance.json`, `response_scaling.json`, `bandwidth_stats.json`,
`rff_metadata.json`, `rff_approximation.json`, `calibration_sample.json`,
`invariance.json`, `geometry_metrics.json`, `nearest_pairs_full.csv`,
`perturbation_summary.json`, `rewiring_ratio.json`, `multiscale.json`,
`nn_wl_metrics.json`, `probe_results.json`, `probe_ceiling.json`,
`doi_features_T{0,2,4}.npz`, `aiom_T4.npz`, `scales_T{0,2,4}.npz`, and
`plots/` (`exact_vs_rff_kernel`, `perturbation_ecdf`, `rewiring_vs_random`,
`wl_neighbor_fidelity`, `probe_mae_vs_T`, `mechanism_comparison_T4`,
`geometry_nn_distance`).
