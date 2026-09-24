# E2E-DictEnv-A1 — implementation note

Round `e2e_dictenv_a1` ("Invariant Attributed Dictionary Core").
Frozen pre-registration: `e2e_dictenv_a1_preregistration.md` (commit `bd3700a`).
Prior-artifact audit: `e2e_dictenv_a1_prior_artifact_audit.md` (commit `bd3700a`).

## Code

```
tracks/ksvd/experiments/luyin16/e2e_dictenv_a1.py        object + model + scaling (single source of truth)
tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_a1.py   runner: stages, gates, decisions, reports
tracks/ksvd/tests/test_e2e_dictenv_a1.py                 targeted CPU tests (38)
```

Nothing outside these files is modified: `e2e_dictenv_p1/p2_abs/v0`, `fsar_v2`,
`sdb_v0`, `zinc_patch_path_pooling` and the P1 runner are imported unchanged, so
the frozen rounds cannot be perturbed by A1.  `A1Model` re-states the P2-ABS H1
`forward` body (needed to make the inherited P1 intervention hooks effective
instead of silently swallowed) and a test asserts bit-identity with
`P2Model` on the same state.

## Objects

```
chi_REAL  = [phi65 ; vec(J^V) in R^308 ; vec(J^E) in R^60]   (433-D)
J^V_i     = sum_{v in P_i} b^V_iv q_v^T     b^V from fsar_v2._explicit_basis_for_patch (11)
J^E_i     = sum_{e in P_i} b^E_ie r_e^T     b^E same builder (15)
q_v       = onehot(atom_v, 28)              r_e = onehot(bond_e, 4)
chi_INDEP = [phi65 ; vec(P^V) ; vec(P^E)]
P^V_i     = (1/n_i) (sum_v b^V_iv)(sum_v q_v)^T
P^E_i     = (1/m_i) (sum_e b^E_ie)(sum_e r_e)^T   (exactly zero when m_i = 0)
```

Scaling is train-only and identical for both objects: per-coordinate RMS
(`sqrt(E[z^2]+1e-12)`, mask where `rms <= 1e-9`, masked scale set to 1.0 and
multiplied by 0), then block-energy normalisation `w_b = 1/sqrt(E[||block_b||^2]+1e-12)`.
Valid/test rows never enter the scaler, K-SVD, OMP or dictionary selection.

Budget inherited unchanged: K=32, s=8, H1 decoder, `d_e=48`, horizon 320,
`lambda_rec = 33.95873017865987`, Adam 1e-3 / wd 1e-5 / batch 128 / clip 5.0,
fixed Top-5 official-valid soup.  Parameters: TOPO 97,487; INDEP = REAL =
109,263 (94,160 local + 15,103 backend), i.e. REAL/INDEP are exactly
parameter-matched.

## Runner stages

```
cache scaler dictionaries omp-codes correctness assignment health continuity
posthoc-continuity omp-screen select-omp iht-diag formal mechanism liveness
specificity accounting decision report all
```

All stages are resumable (a stage skips work whose artifact already exists), so
an interrupted remote run can be relaunched without repeating finished work.
`all` walks the frozen gate order and stops at the first failure, always writing
`decision.json` + `REPORT.md`.

Every CUDA entry point asserts `CUDA_VISIBLE_DEVICES == "1"`; CPU stages (cache,
scaler, K-SVD, exact OMP, Gate 0, Stage 2) run without a device.  `official_test_loaded`
is `false` in every artifact and the shared P1 loader refuses `split == "test"`.

## Gate 0 evidence produced locally (CPU, official train/valid only)

* 19/19 correctness gates pass (layout, independent naive recomputation of
  `J^V`/`J^E`/`P^V`/`P^E`, edge order/endpoint-swap invariance, node relabel
  invariance of blocks *and* of the full H1 prediction, batching invariance,
  deterministic rebuild, finiteness, zero-edge convention, train-only scaler,
  official-test blocker, within-patch assignment semantics, exact top-s,
  once-only composition, no pair->centre, environment freeze, forbidden
  descriptor blocker, strict static contract).
* assignment semantics: `REAL` blocks change by up to 3.62, `INDEP` blocks are
  bit-invariant, marginals preserved, `phi` untouched.
* dictionary health (official train, exact OMP s=8, 20% held-out train rows):
  `TOPO` rec_hold 1.4e-05, `INDEP` 2.7e-02, `REAL` 1.0e-01 in the 2-epoch smoke
  fit (the formal run uses the frozen 10-epoch fits).
* isomorphic control in the continuity pool: mean coordinate distance 4.7e-09
  (float32 level) and code distance 1.7e-09, i.e. both objects are
  isomorphism-invariant.

## Continuity gate observation (smoke fits)

The frozen G0.3 gate is a TCCD-v0 transplant: are the 800 highest-cosine
*WL-identical, non-isomorphic* patch pairs nearer than size/root-atom-matched
random pairs (distance AUC in OMP code space, `>= 0.70`)?

Measured on official train with the frozen pool (`1500` patches, seed
`20260924+9`): the near stratum is entirely equal-size and chemically degenerate
(mean patch size 5.35, range 3-9).  x-space AUC: `REAL` 0.5625, `INDEP` 0.6412,
chemistry-blind `TOPO` 0.4706 (`< 0.5`); code-space AUC with the smoke
dictionaries: `REAL` 0.4975, `INDEP` 0.6375.  The graded similarity correlation
between WL cosine and coordinate distance is only rho = 0.32.

The round's verdict is therefore decided by Gate 0, and the formal run records
it.  No rescue is allowed by the pre-registration; the post-hoc diagnostics
(`continuity_posthoc.json`) are recorded for the *next* pre-registration, which
must fix the stratum (see the analysis note).

## Local validation

* `pytest tracks/ksvd/tests/test_e2e_dictenv_a1.py` — 38 passed.
* Plus `test_e2e_dictenv_p1.py` and `test_e2e_dictenv_p2_abs.py` — 54 passed
  in total, i.e. the frozen rounds are unaffected.
* Full stage smoke on the real ZINC cache with short horizons (2 epochs) and
  2-epoch K-SVD fits: every stage ran; `cache` train 30 s / valid 3 s,
  `scaler` 7 s, `dictionaries` 480 s (2 epochs), exact OMP codes 62 s for all
  six arm/split pairs, `correctness` 18 s, `continuity` 3 s, 320-epoch remote
  training expected at the historical P2-ABS cost (~1,300-2,200 s per arm on
  the A100).
