# Pre-registration: ZINC local 16-D patch-token channel necessity

Protocol version: `local_token_null_v1`
Date: 2026-09-19
Track: `zinc-context-gap` (ZINC cell A, deterministic A100 regime)
Status: pre-registered before any Stage A evaluation / Stage B training.

## Question

> In the strong mixed ZINC backbone, is the molecule-dependent local 16-D
> patch-token channel actually necessary, or do the other backbone paths already
> provide almost all of the predictive power?

This is a **channel-necessity / backbone ablation** question, not "which 16-D
encoder is best".

## The three references share ONE downstream backbone

`A0` (learned typed lookup / Cell-A), `B-Bag` (shared bag encoder) and `B-Full`
(shared structural encoder) all instantiate the **same** cell-A geometry and the
**same** downstream backbone. They differ **only** in the local-token generator:

| Reference | Builder | Local-token generator | Generator params | Total params |
|---|---|---|---|---|
| A0 | `cd.build_cell("A", seed)` | `typed_embedding` (learned row per exact rooted certificate) | 36,420 | 85,763 |
| B-Bag | `sbpe.build_candidate(seed)` | `SharedBagPatchEncoder` (connectivity-free bag) | 35,168 | 84,511 |
| B-Full | `sspe.build_candidate(seed)` | `SharedStructuralPatchEncoder` (radius-2 edge-aware MP) | 35,152 | 84,495 |

Shared downstream backbone = **49,343** params:

`patch_encoder 15,232` + `parent_embedding 256` + `pair_projection 1,024` +
`relation_encoder 1,360` + `distance_gate 80` + `pair_encoder 5,328` +
`center_update 17,824` + `global_encoder 3,136` + `topology_encoder 552` +
`head 4,551`.

## Exact code location of the 16-D tensor

The local token is produced by `PatchPathModel._patch_token_value`
(`zinc_patch_path_pooling.py`) and enters the downstream patch encoder in the
`torch.cat([...])` of

* `PatchPathModel.encode` (`zinc_patch_path_pooling.py`, ~line 2513), and
* `PatchPathRecurrentPairCentreModel._encode_core`
  (`zinc_compact_v4_recurrent_pair_centre.py`, ~line 468),

as the third block:

```python
e_patch = self._patch_token_value(data)
...
patch = self.patch_encoder(torch.cat([
    data.patch_cont,
    data.patch_context,
    e_patch,                              # <-- local 16-D channel
    self.parent_embedding(data.parent_token),
    *structural_blocks,
    *attribute_blocks,
], dim=1))
```

`e_patch` has shape `[n_patches, 16]`.

## Paths that stay unchanged in every intervention / candidate

`patch_cont` (shell-conditioned continuous chemistry), `patch_context`,
`parent_token` typed radius-1 context, `pair_relation` + `relation_encoder`,
`pair_bucket` + `distance_gate`, `pair_projection` / `pair_encoder`,
`center_update` (T=2 weight-tied recurrent pair--centre), `global_encoder`,
topology hinge, the fixed small raw head, and the canonical training protocol
(Adam lr 1e-3, wd 1e-5, batch 128, 240 epochs, patience 40, L1, clip 5.0, fixed
Top-5 checkpoint soup).

## Stage A (eval-only frozen knockout) -- first priority

No new training. For each frozen reference soup (seed0 required, seed1 if the
checkpoint is already present):

* A1 `e_patch := 0` for every patch;
* A2 `e_patch := c` with `c = mean(e_patch)` over the **official-train** split
  under the frozen generator (never fitted on valid targets);
* A3 (optional, cheap) permutation of patch tokens within each molecule.

Backbone weights frozen; tensor shape unchanged; no input column removed; no
retrain; all other paths unchanged. Record original valid MAE, intervention
valid MAE, `Delta MAE`.

Interpretation is a **cheap screening / experiment-economy gate only** (frozen
interventions have distribution shift). Frozen thresholds:

* worst `Delta > 0.13` (~0.12 -> > 0.25): `CATASTROPHIC_STOP`, do not retrain;
* worst `Delta >= 0.06` and `<= 0.13`: `MODERATE`, retrain Null is worth it;
* worst `Delta < 0.02`: strongly supports B-Null seed0.

## Stage B (B-Null retraining) -- only if Stage A is not catastrophic

One new architecture: `patch_representation="null"`.

* keeps the 16-D slot (`e_patch = zeros(16)` for every patch);
* instantiates **no** local-token generator (no `typed_embedding`, no
  `structural_encoder`); the generator parameters are genuinely absent;
* downstream patch-encoder input shape unchanged;
* identical downstream backbone, identical training protocol.

Only **B-Null seed0** is trained. No seed1 upfront, no sweep, no parameter
re-investment, no new bypass, no official test.

Pre-registered matched reference (seed0 fixed Top-5 soup): A0 `0.124704`,
B-Bag `0.127382`, B-Full `0.119818`.

Gate (experiment economy, not a statistical claim):

* Null `<= 0.13`: strong -> local 16-D channel may be unnecessary; authorize
  Stage C;
* `0.13 < Null <= 0.15`: some contribution, far below parameter share;
  authorize Stage C;
* `0.15 < Null <= 0.20`: substantial -> analyze frozen-vs-retrain difference,
  do not auto-run Stage C;
* Null `> 0.20`: STOP, do not rescue by adding features / tuning.

## Stage C (Constant-16 retraining) -- only if authorized

`patch_representation="constant"`: one trainable graph-wide `c in R^16` (16
params) fed to every patch of every molecule. Everything else identical to
B-Null. Only **Constant-16 seed0** is run.

Core question: does the slot need only a trainable bias-like degree of freedom,
or does patch-specific information matter?

## Forbidden this round

official ZINC test; multi-seed upfront; parameter-matching re-investment;
widening the backbone after deleting the token; new structural features; new
radius; RRWP / LapPE / cycles as new architecture input; pair/triangle network;
attention; hyperparameter / optimizer sweep; ad-hoc rescue after a bad B-Null;
editing tracked files remotely; treating frozen knockout as a retrained causal
conclusion.

## Deliverables

Parameter accounting; frozen original / Zero / Constant evaluation; B-Null seed0
result (if authorized); Constant-16 seed0 result (if authorized); matched
B-Full / B-Bag / A0 references; official-test status `false`; mechanism
diagnostics (intervention effective, downstream shape unchanged, no hidden
local-token path, removed generator params absent, all remaining groups receive
non-zero gradient); and the connection to the FSAR-C1 `0.182` residual gap.
