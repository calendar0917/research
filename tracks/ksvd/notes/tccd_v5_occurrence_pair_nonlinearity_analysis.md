# TCCD-v5 — Occurrence-Preserving Pair Nonlinearity Audit

Preregistration: `notes/tccd_v5_occurrence_pair_nonlinearity_preregistration.md`.
Preregistration commit: `539ee07`.
Formal implementation commit: `1ec96ffd2abf0e9e2690d4fa7faf0ddb734bb1bb`.
Formal result: `results/tccd_v5/stageA_seed0.json`.

## Verdict

**STOP after Stage A failure.** The frozen screen shows that a shared nonlinear
pair transform adds useful capacity relative to BASE, but placing that same
nonlinearity before pair aggregation does not materially outperform placing it
after pair aggregation. The registered causal placement hypothesis therefore
fails.

## Execution integrity

- Formal compute: remote A100 **GPU1** only.
- Seed: `0`.
- Split: fixed internal `8000/2000`, split seed `20260922`.
- Base: exact TCCD-v2 PrototypeREL best checkpoint,
  `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`.
- Checkpoint SHA-256:
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`.
- Pair set: all unordered distinct occurrence pairs `i<j`; no self-pairs and no sampling.
- Pair count: `2,668,346` total; mean `266.8346` per graph; range `36..666`.
- Relation count: `5`, in exact TCCD-v2 order `[R_int, R_b0, R_b1, R_b2, R_geo]`.
- Pair descriptor width: `197 = 3*64 + 5`.
- Pair MLP: `Linear(197,64) -> ReLU -> Linear(64,16)`; `13,712` parameters.
- PRE and POST initialization checksum matched:
  `9fbedb92ced44449ade422b152684e3706f34fb943c32b5bfd64979290e8487f`.
- Official ZINC test: never loaded (`official_test_loaded: false`).

## Gate 0

**PASS** on GPU1. The implementation verified:

- pair-swap invariance: maximum descriptor delta `0.0`;
- graph relabel invariance for PRE and POST branches;
- pair-enumeration-order invariance;
- batching invariance;
- exact linear-phi equivalence, PRE vs POST maximum delta
  `1.19e-7`;
- nonlinear PRE/POST separation on a heterogeneous synthetic pair set,
  maximum delta `0.1641543`;
- Stage-A gradients reach the pair MLP and reader;
- no target access or official-test access in pair construction;
- fixed zero output for graphs with no unordered pairs.

## Stage A — frozen PrototypeREL screen

The exact frozen TCCD-v2 assignments and base representation were reused. BASE,
POST and PRE used the same lightweight Adam reader protocol; PRE-SHUFFLE was an
evaluation-only intervention that permuted assignments only inside the PRE pair
branch while leaving the base `C^T R C` representation real.

| arm | best internal-dev MAE | Top-5 soup MAE | feature width |
|---|---:|---:|---:|
| BASE | `0.287915766` | `0.278363913` | `10,464` |
| POST | `0.260826916` | `0.252233326` | `10,480` |
| PRE | `0.261728227` | `0.251418143` | `10,480` |
| PRE-SHUFFLE | `0.262061745` | `0.251755565` | evaluation-only |

Primary Top-5 soup deltas:

```text
delta_place  = MAE_POST - MAE_PRE   = 0.000815183
delta_add    = MAE_BASE - MAE_PRE   = 0.026945770
delta_shuffle= MAE_SHUFFLE - MAE_PRE= 0.000337422
```

The preregistered Stage-A failure threshold is `0.005` for any one delta.
`delta_place` and `delta_shuffle` both fail directly. Therefore no paired seed
is authorized and Stage B is blocked. The best-checkpoint deltas agree on the
same interpretation: placement delta `-0.000901312`, additive delta
`0.026187539`, and shuffle delta `0.000333518`.

## Scientific interpretation

The pair branch itself has practical predictive value relative to BASE:
`delta_add = 0.02695` MAE. However, PRE does not materially beat POST, and the
assignment-location shuffle barely changes the PRE prediction. Thus this round
does **not** support the claim that the TCCD-v2 ceiling is caused by applying a
shared pair nonlinearity after occurrence aggregation rather than before it.

The correct narrow conclusion is:

> Adding one shared nonlinear transform over complete unordered prototype
> pairs improves the frozen TCCD-v2 control, but moving that transform before
> pair pooling does not materially improve over the matched post-aggregation
> placement, and the evaluation-only assignment shuffle does not show material
> use of occurrence-to-relation alignment.

This result does not prove that occurrence information is irrelevant. It only
rules out this specific one-level global PRE-vs-POST intervention as the
material rescue under the frozen TCCD-v2 interface.

## Stopped work

Because Stage A failed, the following were not run:

- Stage B POST-E2E;
- Stage B PRE-E2E;
- paired seed 1;
- vocabulary diagnostics for an E2E model;
- official-valid training;
- official-test evaluation.

No vocabulary-health conclusion is authorized for TCCD-v5 because the E2E
prototype vocabulary was never trained in this round.

## Next hypothesis

The next justified representation hypothesis is **center-preserving composition**:
retain the centre occurrence while aggregating its pair interactions, for example
`u_i = mean_j phi(c_i,c_j,r_ij)` followed by centre-aware pooling. This is a new
round and requires a new preregistration; it is not authorized by this closure.

Evidence:

- `tracks/ksvd/notes/tccd_v5_occurrence_pair_nonlinearity_preregistration.md`
- `tracks/ksvd/results/tccd_v5/gate0.json`
- `tracks/ksvd/results/tccd_v5/stageA_seed0.json`
- `tracks/ksvd/code/tccd_v5.py`
- `tracks/ksvd/code/run_tccd_v5.py`
- `tracks/ksvd/tests/test_tccd_v5.py`
