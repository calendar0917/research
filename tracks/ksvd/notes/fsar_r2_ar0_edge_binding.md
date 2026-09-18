# FSAR-R2-AR0-EDGE — edge structural-role ↔ bond-type assignment residual (durable verdict)

Branch `exp/fsar-r2-ar0-edge-binding-zinc`.  Pre-registration:
`notes/fsar_r2_ar0_edge_binding_preregistration.md` (15 sections), committed at
`44660bb` **before** any formal run.  Formal revision `44660bb`.
**Official ZINC test never loaded** (`official_test_loaded = false` everywhere).

This is a first-order extension of the AR0 node round.  It is not a SOTA
attempt: the models are tiny (24,797 / 25,317 params) and the absolute MAE
levels (~0.44–0.47) are far from the repo's full-backbone numbers.

---

## A. Node-B mechanism (prerequisite, summary)

Full analysis: `notes/fsar_r2_ar0_node_binding_mechanism.md`, revision `6acfc45`.
No correctness failure was found and the AR0 verdict is unchanged.

* **Rank-1 caveat retracted.**  The AR0 "`C` effective rank 1.40 / top singular
  fraction 0.89" was an audit-slicing bug (`flattened_C[:, :64]`).  On the full
  flattened `C` the train-fit effective rank is `56.6`; on the model input `C~`
  it is `116.3`.  `audit()` is fixed on this branch.
* **Seeds learn the same function, not the same raw weights.**  Branch-output
  Pearson across seeds `0.9878–0.9922`; scaled-space `W_B` cosine `0.954–0.986`;
  raw-unit cosine `−0.63 … +0.91`.  Large raw-unit weights sit on near-zero-RMS
  coordinates and contribute `≈0`.
* **Not low-dimensional.**  PC1 explains `0.5–1.7 %` of the trained `B`
  (`k=10`: `~37 %` in-sample, `8–10 %` out-of-sample).
* **Contribution is interpretable and seed-stable**: `~94 %` in the pooled
  node/edge mean+std groups (root and size coordinates `~4 %` each);
  contribution-profile cosine across seeds `0.96–0.99`; dominated by N/O/C/S/F
  atom categories.
* **Frozen-`M0` diagnostic:** a linear `C` readout on a completely frozen `M0`
  soup recovers `70.1 % / 70.7 % / 66.7 %` of the full `MB` gain (seeds 0/1/2),
  so `≈2/3` of the node gain needs no base/B co-adaptation.
* **Witnesses:** 9 exact-`A`-key collision pairs in validation; the 5 largest
  `|Δb|` have `M0`-residual differences with the same sign as the `B`-branch
  difference (illustrative only).

This established that there was no correctness failure and authorised the edge
round (`preregistration §1`).

---

## B. Edge implementation

Branch `exp/fsar-r2-ar0-edge-binding-zinc`, formal commit `44660bb`.

* model/features: `experiments/luyin16/fsar_r2_ar0_edge.py`
* runner: `experiments/luyin16/zinc_fsar_r2_ar0_edge.py`
* tests: `tests/test_fsar_r2_ar0_edge.py` (**17/17 pass**)
* results: `results/fsar_r2_ar0_edge/` (gitignored)

**Edge role (exact).**  For each original undirected edge `e=(u,v)`:

```
psi_e = [ phi_u + phi_v , |phi_u - phi_v| ]   in R^130
```

endpoint-swap invariant and chemistry-free.  No bond type, atom type, typed
path, chemistry-aware descriptor, message passing or new motif enters `psi_e`.

**Bond attribute.**  `r_e = onehot(bond_type_e) in R^4`, dimension verified from
`data/ZINC/raw/bond_dict.pickle` (`NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3`);
PyG's two directed entries per bond are canonicalized to one undirected bond
(the existing `A` bond multiset is unchanged).

**Edge assignment residual.**

```
J_E = sum_e psi_e r_e^T
P_E = (1/m)(sum_e psi_e)(sum_e r_e)^T
C_E = J_E - P_E = sum_e (psi_e - psibar)(r_e - rbar)^T      # sum-centred
```

Train-only per-coordinate RMS scaling, no dataset-mean subtraction.  Effective
coordinates: `C_E 324/520`, `P_E 330/520`; `CE_rms_max 0.824`, `PE_rms_max
75.72`.  `psi` official feature audit: 130-D, effective rank `20.1`, 22
zero-std coordinates, no NaN/Inf.

**Models** (shared `F0` and node `B_V`; only the edge branch differs):

| variant | prediction | node B | edge branch | total params |
|---|---|---:|---:|---:|
| `BV` | `F0 + <W_B, C~_V>` | 1,820 | 0 | **24,797** |
| `BVE` | `BV + <W_E, C~_E>` | 1,820 | 520 | **25,317** |
| `BVEM` | `BV + <W_ME, P~_E>` | 1,820 | 520 | **25,317** |

`W_B`, `W_E`, `W_ME` bias-free, zero-init, no MLP/activation/bypass.
`dataset_dependent_vocabulary_params = 0`.  Shared-init identity verified
(base + `W_B` bit-identical across variants at the same seed); all three
predict exactly `F0 + node term` at step 0.

**Correctness tests (16 in the sanity gate + 1 extra).**  `sanity` =
**16/16 pass**: endpoint-swap invariance, chemistry purity, node relabel
invariance, permutation invariance of `S`/`A`/`C_V`/`P_E`, `C_E` sensitivity,
exact `mean_pi C_E = 0`, scalar `mean_pi <W_E,C_E> = 0`, undirected
canonicalization, train-only scaler, batched-vs-numpy equality, `W_E`/`W_ME`
gradient viability, no edge bypass, shared init identity, zero-init prediction
identity, parameter counts.  Remote `gradient_audit` (`--all --device cuda
--deterministic`): active path `25,317/25,317`, `W_E` and `W_ME` non-zero task
gradient.

**Synthetic controls (must pass before ZINC).**

* Positive (assignment-only, fixed 5-node tree with non-equivalent edge roles,
  all-`C`, one DOUBLE): `BV 0.2531`, `BVE **0.0173**`, `BVEM 0.3228` (3 seeds),
  `||W_E||≈0.05`.  **pass** — only `BVE` learns the real edge assignment.
* Negative (label from topology/atom/bond counts only, then assignment
  randomised): `BV 0.0206`, `BVE 0.0666`, `BVEM 0.0420`.  **pass** — no
  repeatable `BVE` advantage.

---

## C. Remote provenance

| item | value |
|---|---|
| local / remote commit | `44660bb` |
| host repo | `/home/hxy/cy/research`, branch `exp/fsar-r2-ar0-edge-binding-zinc` |
| GPU model | NVIDIA A100-SXM4-40GB ×2 |
| GPU 0 | unknown third-party task, 35.8 GB / 98 % — **never touched, never co-tenanted** |
| GPU 1 | ours; unrelated GSN process at ~0.54–0.56 GB recorded, never touched |
| CUDA / PyTorch / Python | 12.4 / 2.5.1+cu124 / 3.12.14 |
| determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed data seeds |
| execution | short trio concurrent on GPU 1; formal wave 1 `BV-s0 ∥ BVE-s0`, wave 2 `BVEM-s0 ∥ BV-s1`, wave 3 `BVE-s1 ∥ BVEM-s1`; seed 2 authorised by the gate but **stopped deliberately before completion** (time budget), so the verdict uses the two completed paired seeds |
| DDP | none; one process per job |
| our peak GPU memory | ~103 MB per formal run |
| protocol | Adam lr 1e-3, wd 1e-5, batch 128, L1, clip 5, no scheduler, max 240 epochs, patience 40, fixed Top-5 soup |
| `official_test_loaded` | **false** in every artifact |

**Cross-round control:** `BV` seed 0 and seed 1 reproduce the AR0 `MB` node
soups **exactly** (`0.4692853513918235`, `0.4587459604590549`), confirming the
node component is bit-identical between rounds.

---

## D. Formal edge results

Valid MAE, fixed equal-weight Top-5 soup (2 paired seeds):

| model | seed | best valid | Top-5 soup | best ep | node B norm | edge branch norm | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BV` | 0 | 0.477389 | 0.469285 | 58 | 0.7467 | 0.0000 | 274.5 |
| `BVE` | 0 | 0.459459 | **0.445958** | 59 | 0.7256 | 0.4691 | 275.7 |
| `BVEM` | 0 | 0.472997 | 0.463590 | 103 | 0.8666 | 0.6390 | 406.5 |
| `BV` | 1 | 0.464546 | 0.458746 | 149 | 1.0258 | 0.0000 | 529.2 |
| `BVE` | 1 | 0.458576 | **0.444375** | 91 | 0.8333 | 0.5396 | 378.5 |
| `BVEM` | 1 | 0.472059 | 0.460939 | 121 | 0.9142 | 0.7443 | 475.1 |

Paired deltas (positive = `BVE` better):

| seed | `BV − BVE` | `BVEM − BVE` | both > 0 |
| ---: | ---: | ---: | :---: |
| 0 | **+0.023328** | **+0.017632** | yes |
| 1 | **+0.014371** | **+0.016564** | yes |

Paired per-molecule bootstrap (5,000 resamples over the 1,000 valid molecules,
seed 20260918), 95 % percentile CI:

| seed | `BV − BVE` mean [CI] | `BVEM − BVE` mean [CI] | `P(Δ>0)` |
| ---: | --- | --- | :---: |
| 0 | 0.023339 [0.013024, 0.033619] | 0.017618 [0.007824, 0.027251] | 1.0 / 1.0 |
| 1 | 0.014430 [0.003825, 0.025392] | 0.016649 [0.007065, 0.026228] | 0.996 / 1.0 |

The seed-2 gate fired on both seeds 0 and 1 (`gate.json seed2_authorized =
true`); seed 2 was authorised but not completed (time budget), so the verdict
rests on the two completed paired seeds.  **The bootstrap is a within-seed
paired diagnostic and does not replace the 2-seed robustness check** (seed 1's
`Δ_edge` CI lower bound is only `+0.0038`, so the edge effect is clearly larger
and tighter on seed 0 than on seed 1 — a consistent-direction, moderate-size
effect, not a marginal one).

---

## E. Edge mechanism

**Branch liveness (soup states, full 1,000-molecule valid).**

| model | seed | `‖W_B‖` | node out std | `‖W_edge‖` | edge out std | edge zeros | eff. coords | edge task grad ‖·‖ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BV` | 0 | 0.777 | 0.673 | — | 0.000 | 0/0 | 0 | 0 |
| `BV` | 1 | 1.007 | 0.677 | — | 0.000 | 0/0 | 0 | 0 |
| `BVE` | 0 | 0.806 | 0.654 | 0.523 | 0.233 | 196/520 | 324 | 1.84 |
| `BVE` | 1 | 0.819 | 0.635 | 0.530 | 0.225 | 196/520 | 324 | 1.92 |
| `BVEM` | 0 | 0.847 | 0.670 | 0.626 | 0.338 | 190/520 | 330 | 0.73 |
| `BVEM` | 1 | 0.897 | 0.650 | 0.729 | 0.394 | 190/520 | 330 | 1.27 |

Every branch is live with non-zero task gradient.  **Zero-mask accounting:**
the exact-zero counts (`196 = 520 − 324` for `BVE`; `190 = 520 − 330` for
`BVEM`) are exactly the zero-RMS mask counts, so this is **not** learned
sparsity.  `BVEM`'s edge branch is *more* live (output std `0.34–0.39` vs
`0.23`) yet worse, so the `BVE` gain is not extra capacity.

**Evaluation-only bond-type permutation** (fixed topology, atom attributes,
atom assignment and bond-type multiset; permute bond types among undirected
edges; soup state):

| model/seed | Δ`S` | Δ`A` | Δnode `C_V` | Δ`P_E` | Δ`C_E` | actual MAE | `MAE(E[ŷ_π])` | `E[MAE(ŷ_π)]` | mean abs pred change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BVE` s0 | **0.0** | **0.0** | **0.0** | **0.0** | 49,419 | **0.4460** | 0.5149 | 0.5616 | 0.282 |
| `BVE` s1 | **0.0** | **0.0** | **0.0** | **0.0** | 47,805 | **0.4444** | 0.5090 | 0.5608 | 0.275 |

`S`, `A`, the node statistic and `P_E` are **exactly** invariant while `C_E`
and the prediction change; both permutation-MAE quantities are materially worse
than the actual MAE.  The trained `BVE` uses the real bond-type assignment.

**Does the real edge branch beat an equally-live matched marginal control?**
Yes: `BVEM` has the same parameter count, the same edge-role coordinates, a
*more* live edge branch, and a strictly weaker result on both seeds
(`+0.0176`, `+0.0166` in `BVE`'s favour).  The only difference is whether the
statistic is the assignment residual `C_E` or the assignment-independent
marginal `P_E`.

---

## F. Final verdict

**`EDGE_ASSIGNMENT_SUPPORTED`**

> Under the fixed radius-2 explicit pure-topology coordinates, and with the
> already-supported node assignment residual held fixed, the real edge
> structural-role ↔ bond-type assignment provides a stable predictive increment
> beyond the node baseline and beyond an exactly matched, assignment-independent
> edge marginal control on both formal paired seeds.

Scope, exactly: a **first-order linear** edge assignment residual on this fixed
`psi_e` basis, at this budget, on ZINC-12K official train/valid.  No SOTA /
general-binding claim.

### The first-order S / A / B picture

* `S` (pure topology, radius-2 explicit coordinate) and `A` (whole-graph
  attribute multiset) are the frozen marginals.
* `B_V` (node role ↔ atom attribute) is supported (AR0, 3 seeds).
* `B_E` (edge role ↔ bond type) is supported here (2 seeds, consistent
  direction).
* So a **complete first-order assignment picture** now exists: the two natural
  one-body assignments — node↔atom and edge↔bond — both carry real,
  seed-stable, control-beating information that the marginals cannot express.

### Is pair-level `B^(2)` needed?

Not yet demonstrated.  The pre-registered condition for even proposing `B^(2)`
is a **first-order collision witness**: attributed graphs with
`S(G1)=S(G2)`, `A(G1)=A(G2)`, `B_V(G1)=B_V(G2)`, `B_E(G1)=B_E(G2)` but
different target/systematic residual.  This round did **not** search for one
(it was explicitly out of scope) and no such witness is claimed.  `B^(2)` is
**not implemented**; it remains a discussion-only candidate to be justified by
a witness in a future round.

### Where does the ~0.12 gap to strong models most plausibly come from?

Both first-order assignments are real but small in absolute terms (`B_V`
≈ `+0.08`, `B_E` ≈ `+0.02` MAE), and the frozen-`M0` diagnostic showed the
node `C` readout is **high-dimensional but shallow**: it is a centred global
role↔attribute statistic, `~2/3` of it recoverable by a plain linear probe on a
frozen base.  That is consistent with the remaining gap being **not** an
assignment-order deficit but a **downstream compositional-processor** deficit:
the current model has no persistent, incidence-aware, higher-order object on
which to compute a compositional function.  This is a hypothesis, not a result
of this round; testing it requires its own pre-registered architecture round.
No next step is auto-implemented.

---

## G. What was explicitly not done

No official test; no pair-level node binding; no second-order attribute-pair
interaction; no radius > 2; no C4/C5 / RRWP / homomorphism; no attention /
Transformer; no message passing inside the primitive definition; no
width/depth/optimizer/wd sweep; no feature search; no `B-full` teacher or
distillation; no mixed descriptor; no seed hunting; no change of `A_exact`; no
validation-time change of the edge-role definition; no automatic model repair
after the result; seed 2 was authorised by the gate but stopped before
completion on time budget and is not part of the verdict.

## H. Reproduce

```
# Phase-1 node mechanism (local analysis + remote frozen-M0 diagnostic)
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism all --deterministic
# Phase-2 edge round (remote, deterministic A100)
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge preprocess
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge sanity
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge synthetic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge run --variant BVE --seed 0 --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge gate
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge decide
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge analyze
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge diagnostics --variant BVE --seed 0 --state soup
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge witness --variant BVE --seed 0 --state soup
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0_edge report
```

Artifacts: `results/fsar_r2_ar0_edge/` — `cache/`, `curves/`, `states/`,
`soup_states/`, `runs/`, `sanity.json`, `feature_audit.json`,
`parameter_accounting.json`, `gradient_audit_{BV,BVE,BVEM}.json`,
`synthetic_controls.json`, `soup_*`, `diagnostics_*`, `witness_*`, `gate.json`,
`analysis_paired_bootstrap.json`, `decision.json`, `FORMAL_RESULTS.md`; and
`results/fsar_r2_ar0/mechanism/` for the phase-1 analysis.
