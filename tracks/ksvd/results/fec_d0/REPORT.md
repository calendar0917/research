# FEC-D0 — Within-Coarse-Role Residual Structural Dictionary Audit — REPORT

**Verdict: `FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL`**

- commit `ca2e947013263a2590c7ad4edffff03a46e6b830` (dirty=True)
- regime: **LABEL-FREE**; no property training, no official valid/test, no FEC-S1 retrain, no task coupling
- official_train molecules used: 10000; FIT 8000 / HOLDOUT 2000 (SPLIT_SEED 20260922)
- official_valid_loaded=false; official_test_loaded=false; targets_loaded=false

## Frozen object

- basis: FSAR explicit rooted node basis `b^V_iv in R^11` (`fsar_v2._explicit_basis_for_patch`), pure topology
- coarse role: `shell(i,v) in {0,1,2}`; exact shell coordinates deleted (0,1,2,3)
- residualized coordinate `phi^perp in R^7` with FIT-only per-shell `mu_s, sigma_s`
- dictionary: ONE shared `D in R^{7x16}`, `K=16`, `s=4` (OMP exact top-s)

## Gates

| gate | value | threshold | pass |
|---|---:|---:|---|
| Gate 0 correctness | 7/7 | all | True |
| Gate 1 reconstruction R_rec | 0.0038 | <= 0.8 | True |
| Gate 2 active FIT | 16/16 | >= 12 | True |
| Gate 2 active HOLDOUT | 16/16 | >= 12 | True |
| Gate 2 active Jaccard | 1.0000 | >= 0.75 | True |
| Gate 2 usage Spearman | 1.0000 | >= 0.7 | True |
| Gate 2b oracle G_O | 0.0244 | >= 0.03 | False |
| semantic F1_D - F1_R | -0.0008 | >= 0.03 | False |
| semantic retained G_D/G_O | 1.0000 | >= 0.7 | True |

## Structural facts (report-only)

- per-shell residual rank: [2, 4, 4]
- shell-deterministic kept coordinates: [5, 8]
- rank(D_learned)=7, rank(D_random)=7, rank(holdout residual)=4
- probe convergence caveat: The frozen probe fixes max_iter=200 (no HPO). Some lbfgs fits emitted ConvergenceWarning. This caveat does not change the decisive evidence: per-shell residual rank <= 4 (shell 0 rank 2), D == O identity, and the near-degenerate degree baseline are structural, not optimisation artefacts.

## Interpretation

The audited rooted node basis, after the coarse shell role is removed and the
residual per-shell standardized, has rank 2-4 and is dominated by the
within-patch degree. A `K=16` dictionary is therefore massively overcomplete:
it reconstructs the residual almost exactly (`R_rec` trivially passes) and the
sparse code is a linear reparametrisation of the dense residual (`F1_D == F1_O`).
The dense basis carries only `G_O=0.0244` macro-F1 beyond the degree baseline,
below the pre-registered 0.03, so the frozen verdict fires. This does not
authorise any property experiment.
