# FSAR seed0 case verdict — strict factorization is real but information-starved

Branch `exp/factorized-structure-attribute-relational-zinc`.
Architecture/training commit `cf192e4`; final (eval-only) commit `1f84090`.
Stage-0 audit: `notes/zinc_current_backbone_information_flow.md`.
Implementation status: `notes/fsar_implementation_status.md`.
Official ZINC test never loaded (`official_test_loaded = false` in every JSON).

## Question

If every MIXED path that bypasses factorization is removed from the strong
recurrent patch–centre backbone, and each atom is strictly factorized into
`A_v` (attribute marginal; cannot see topology), `S_v` (topology only; cannot see
chemistry) and `B_v` (the only channel that pairs a role with the attribute of
the *same* node/edge, patch-internally centered), with a topology-only
relational substrate (`T=2` recurrent pair–centre core, chemistry-free 15-D
relation) and `g = Pool(h^T) + topology hinge`, does the factorized system still
reach the mixed backbone band?

## Pre-registered gate

`A seed0 soup <= 0.135` (brief section 31). **FAILED by +0.02292** →
`status = STOP_A_GUARD`, seed1 **not** authorized.

## Seed0 results (deterministic A100, Top-5 soup, valid MAE; lower is better)

| mode | params | best valid | best ep | soup | Δ vs previous |
|---|---:|---:|---:|---:|---:|
| A | 56,617 | 0.160617 | 177 | **0.157917** | — |
| SA | 77,609 | 0.156987 | 208 | **0.151617** | A→SA 0.006299 |
| SAB | 95,209 | 0.132160 | 222 | **0.130913** | SA→SAB 0.020705 |
| SAM (capacity control) | 95,153 | 0.147073 | 232 | **0.134182** | SA→SAM 0.017435 |

Frozen matched references, seed0 soup: B-Full 0.119818, A2/cell-A 0.121694,
B-Bag 0.127382. SAB sits +0.011095 / +0.009219 / +0.003531 above them; it is
**not** competitive with the strong mixed backbone.

## Guard-failure diagnosis: architecture is functional, the A channel is starved

Zeroing the `T=2` relational core of the trained A checkpoint (last
`center_update` layer → 0, so `h² = h⁰`) sends valid MAE 0.1606 → **0.9067**
(Δ +0.746). Zeroing the topology hinge channel sends it 0.1606 → **0.3529**
(Δ +0.192). Both components carry the prediction and are not broken.

The starvation is representational: 32-D `A` marginals pooled to a single
mean/std over the patch (no shell resolution, no typed histogram, by
construction) simply contain far less than the removed `patch_cont` 146-D shell
descriptor + `patch_context` + `global_context`. This is the brief's
"accidental information starvation", not a relational-core/pooling defect.

## Mechanism — B is genuinely used

Corrected eval-only context-only shuffle (centre atom fixed, per brief section
27; the inherited FSAB shuffle also permuted the centre, which invalidated the
first read):

* witness: `A_stable` true (1.53e-8), `S_stable` true (0.0), `B_responds` true
  (0.3637), prediction mean |Δ| 0.9970.
* interventions on the full 1,000-molecule valid split (true MAE 0.13216):
  no-A +2.2552, no-S +0.2536, no-B **+0.4467**, no-A_self +2.4126,
  no-A_ctx +0.1945, attribute shuffle +0.8323.
* channel diagnostics: A/S/B norms 3.31/1.89/2.68, cross-molecule std
  0.50/0.27/0.50, effective rank 11.8/6.8/16.3; node-init contributions
  A 7.63 / B 5.32 / S 4.17; all gradients non-zero.

## Capacity control — most of the SAB gain is capacity, not binding

The pre-registered conditional control `SAM` adds only a marginal-only MLP
`F_M([A_v,S_v])` (reads no aligned pair) with matched parameter count
(95,153 vs SAB 95,209, −56).

* SA→SAB total gain **0.020705**
* SA→SAM capacity gain **0.017435** (84.2 % of the total)
* SAM→SAB aligned-binding residual **0.003270** (15.8 %)

SAB still beats SAM by 0.00327 > 0.001 gate → `capacity_confound_excluded =
true`, but the naive "SA→SAB = binding effect" reading would be ~5× overstated.
The binding channel's *specific* contribution is real but small; its *removal*
costs a lot only because the marginal-only substitute is not an identical
function class.

## Decision (`decision.json`)

```
status: STOP_A_GUARD
case: A_guard_failed__SAB_binding_increment
a_guard_pass: false
mechanism_supported: true
performance_supported: true
capacity_confound_excluded: true
seed1_authorized: false
deltas: {A_to_SA: 0.0062994, SA_to_SAB: 0.0207048, SAB_to_SAM: 0.0032696}
```

Per the pre-registered guard, **do not promote, do not buy seed1, do not widen
channels / sweep capacity, do not open the official test**.

## What was learned

1. Strict A/S/B factorization with a topology-only relational core is
   *implementable and functional*: the core and topology channels carry the
   prediction, and B is measurable and necessary at the checkpoint.
2. At the inherited ~32-D-per-channel budget and node-state-only pooling, the
   factorized A channel is too weak: A soup 0.158 vs the 0.135 guard.
3. The apparent SA→SAB binding gain is ~84 % generic capacity; only ~0.0033 is
   aligned structure–attribute binding. Any future claim about "binding" must
   be made against a marginal-only capacity control, not against SA alone.
4. Parameter accounting is clean: 0 dataset-dependent vocabulary parameters;
   nested A 56,617 < SA 77,609 < SAB 95,209 (≤ 200 k).

## Files

* code: `experiments/luyin16/fsar.py`, `experiments/luyin16/zinc_fsar.py`
* tests: `tests/test_fsar.py` (10/10 pass local + remote)
* results: `results/fsar/` (gitignored; pulled locally)
* audit: `notes/zinc_current_backbone_information_flow.md`
* this note; records `claims/` + `decisions/` 20260916

## Provenance note

A/SA/SAB were trained at `cf192e4`. Commit `1f84090` changed only (a) the
eval-only witness shuffle to keep the centre fixed and (b) added the SAM mode
and the richer `decide()`. A/SA/SAB parameterization is unchanged — the pulled
checkpoints load and re-soup to the identical values under `1f84090`.
