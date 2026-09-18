# Local 16-D patch-token channel necessity on the strong ZINC backbone

Protocol: `local_token_null_v1` (pre-registration:
`notes/local_token_null_preregistration.md`, committed before any run).
Study: `zinc-context-gap`. Regime: deterministic A100, ZINC-12K official
train/valid, **official test never loaded**.
Training revision for both candidates: `b31d1bc`.

## Question

In the strong mixed ZINC backbone, is the molecule-dependent local 16-D
patch-token channel (`e_patch in R^16`) actually necessary, or do the other
backbone paths already provide almost all of the predictive power?

This round does **not** re-invest the released parameters: the parameter drop is
the experimental variable.

## The three references share one downstream backbone (49,343 params)

`A0` = `cd.build_cell("A", seed)` (learned typed lookup), `B-Bag` =
`sbpe.build_candidate` (shared bag), `B-Full` = `sspe.build_candidate` (shared
structural message passing). They differ **only** in the local-token generator:

| Reference | Local-token generator | Generator params | Total params |
|---|---|---|---|
| A0 | `typed_embedding` (exact certificate lookup) | 36,420 | 85,763 |
| B-Bag | `SharedBagPatchEncoder` | 35,168 | 84,511 |
| B-Full | `SharedStructuralPatchEncoder` | 35,152 | 84,495 |
| Null | *none* (`e_patch = 0`) | 0 | **49,343** |
| Constant | one trainable `c in R^16` | 16 | **49,359** |

Downstream backbone (identical in all five): `patch_encoder` 15,232 +
`parent_embedding` 256 + `pair_projection` 1,024 + `relation_encoder` 1,360 +
`distance_gate` 80 + `pair_encoder` 5,328 + `center_update` 17,824 +
`global_encoder` 3,136 + `topology_encoder` 552 + `head` 4,551 = **49,343**.

The token is produced by `PatchPathModel._patch_token_value` and enters the
downstream patch encoder in the `torch.cat([patch_cont, patch_context, e_patch,
parent_token, ...])` of `PatchPathModel.encode` and
`PatchPathRecurrentPairCentreModel._encode_core`.

## Stage A -- frozen knockout (eval-only)

Forced `e_patch` to an exact zero, to the train-side mean token, and to a
within-molecule permutation, on the frozen reference soups. Seed means across
seed0/seed1:

| Reference | original | Zero-16 | Delta zero | Constant-16 | Delta const |
|---|---|---|---|---|---|
| A0 (learned lookup) | 0.126368 | 0.322299 | **+0.195931** | 0.274810 | +0.148442 |
| B-Bag (shared bag) | 0.123806 | 0.124100 | **+0.000294** | 0.124136 | +0.000331 |
| B-Full (shared structural) | 0.118972 | 0.121506 | **+0.002534** | 0.121582 | +0.002610 |

Recomputed original MAE reproduces the recorded references to `<=5e-8`.

**Interpretation.** The learned-lookup reference A0 is catastrophically
sensitive, but A0's generator has mean-token norm 0.44/0.36, i.e. it is a real
high-rank categorical memory. The two strongest references are essentially
insensitive: B-Bag's mean token norm is only 0.016/0.002 and B-Full's structural
encoder is near rank-1 (effective rank 1.107/1.202, top singular fraction
0.979/0.955), so its "molecule-dependent" token is already almost a constant.
This is a frozen-intervention screening result with distribution shift, not a
causal claim; the A0 catastrophe is the expected signature of freezing a large
learned table, and the retrained A2 zero-slot control already reached 0.1217.

## Stage B/C -- retraining (seed0 only)

| Model | params | best epoch | raw best valid | Top-5 soup valid | wall |
|---|---|---|---|---|---|
| B-Null | 49,343 | 237 | 0.1268726 | **0.1230275** | 5229 s |
| Constant-16 | 49,359 | 225 | 0.1262599 | **0.1205154** | 5112 s |

Matched seed0 fixed Top-5 soup references:

| Comparison | delta |
|---|---|
| Null - B-Full (0.119818) | **+0.003210** |
| Null - A0 (0.124704) | -0.001677 |
| Null - B-Bag (0.127382) | -0.004354 |
| Constant - B-Full (0.119818) | **+0.000697** |
| Constant - Null | -0.002512 |
| Constant - A0 (0.124704) | -0.004189 |

B-Null removes 36,420 / 35,168 / 35,152 parameters (41.6-42.5 % of each
reference) and lands at 0.123028, *better* than A0 and B-Bag seed0 and only
+0.0032 above B-Full. Constant-16 adds 16 trainable parameters and lands at
0.120515, within +0.0007 of the strongest reference.

Mechanism witness (post-training):
`typed_embedding` and `structural_encoder` absent in both candidates;
`patch_encoder` input width unchanged (170); Null token is exactly zero;
Constant learns a small graph-wide vector (norm 0.00919, per-dim |values| <=
0.005); all ten remaining major backbone groups receive non-zero task gradient;
forward finite; `official_test_loaded = false`.

## Result pattern

This is **Pattern 1 with Pattern 2's mechanism**: `Null ~ Constant ~ B-Full`
(0.1230 / 0.1205 / 0.1198, all within ~0.003). The molecule-dependent local 16-D
channel is essentially redundant for this backbone; whatever it contributes is
a small shared trainable offset (16 params recover it), which is exactly why the
shared structural encoder trained to near rank-1.

## Connection to FSAR-C1 (0.182 valid soup)

FSAR-C1 is a different architecture (parameter-matched incidence processor, 88,643
params) at seed0 soup 0.182050, +0.062232 above B-Full seed0. This round shows
that the strong mixed backbone keeps its performance while the entire
molecule-dependent local 16-D patch-token family (35-36k params) is deleted.
Therefore the FSAR-C1 residual gap can **no longer** be attributed to "missing a
sophisticated local 16-D patch embedding". The more plausible remaining gap
sources are the other strong-backbone paths: shell-conditioned mixed chemistry
(`patch_cont`), radius-1 parent typed context, mixed pair relation, the
pair-centre / co-occurrence computation, global/topology channels and the
readout. This round does not identify which one; it only re-ranks the
hypotheses.

## What was not done

No official ZINC test; no multi-seed upfront; no parameter re-investment; no
backbone widening; no new structural feature / radius / RRWP / cycles / pair
network / attention; no hyperparameter or optimizer sweep; no rescue after a
result. Frozen knockout is reported as a screen, never as a retrained causal
conclusion.

## Artefacts

`results/local_token_null/`:
`stage_a_{a0,b-bag,b-full}_seed{0,1}.json`,
`stage_a_frozen_knockout.json`, `stage_a_gate.json`,
`runs/lt_null_seed0.json`, `runs/lt_constant_seed0.json`,
`soup_lt_null_seed0.json`, `soup_lt_constant_seed0.json`,
`witness_lt_null_seed0.json`, `witness_lt_constant_seed0.json`,
`parameter_accounting` via `params`, `sanity.json`, `decision.json`,
`report.json`, `curves/`, `snapshots/`.
