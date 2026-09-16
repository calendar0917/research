# FSAR Stage 0/1 status + GPU blocker

Branch `exp/factorized-structure-attribute-relational-zinc`.
Formal commit `902cd2a` (Stage-0 audit + FSAR implementation + correctness tests
+ runner).  This note records what is done and the one blocking condition.

## Stage 0 — information-flow audit (complete)

`notes/zinc_current_backbone_information_flow.md`.  Every path that reaches the
prediction graph `R` of the current strong model (B-Full / cell-A recurrent) is
classified `A` / `S` / `B` / `MIXED` with a raw-tensor trace:

* MIXED bypasses found: `patch_cont` (typed shell histogram), `patch_context`
  (radius-3 typed histogram), the typed local token (all four encoders),
  `parent_token`, the mixed columns of `pair_relation` (`path_bond_mean`,
  `adjacent`), `global_context` (global typed histogram), the derived
  `center_context` / unary / pair readouts, `direct_token_readout`, the v6
  `attribute_encoder`, the ring-context conditioning.
* The only pre-existing pure-`S` path is `topology_features` (hinge, 25D:
  cycle spectrum / MCB / hinge ladder — reads no atom or bond type).
* **FSAB is not a factorized backbone**: it replaced only the 16D local token
  (`e_patch`) while #1/#4/#5a/#6 stayed MIXED, which is exactly why the token
  could be annihilated with no loss.

FSAR handling plan (authorised to implement): remove every MIXED path; rewrite
`pair_relation` to a 15D topology-only descriptor; keep `pair_bucket` and the
topology-only hinge; `g = Pool(h^T) + topology hinge`; pair information reaches
the readout only through the `T=2` centre updates.

## Stage 1 — implementation + tests (complete, verified remote)

* `experiments/luyin16/fsar.py` — strict `A`/`S`/`B` channel encoder
  (input-access enforced by construction), topology-only relation builder,
  `T=2` recurrent pair–centre core, shared head, nested modes `A`/`SA`/`SAB`
  in one implementation; evaluation-only channel interventions.
* `experiments/luyin16/zinc_fsar.py` — runner stages `params preprocess sanity
  smoke train soup diagnostics witness interventions decide report`.
* `tests/test_fsar.py` — 10 architecture correctness gates.

### Parameter accounting (zero dataset-dependent vocabulary params)

| mode | A enc | S enc | B enc | node init | relation core | head | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 5,264 | 0 | 0 | 12,480 | 27,376 | 10,945 | **56,617** |
| SA | 5,264 | 16,896 | 0 | 16,576 | 27,376 | 10,945 | **77,609** |
| SAB | 5,264 | 16,896 | 13,504 | 20,672 | 27,376 | 10,945 | **95,209** |

`topology_channel` 552 is shared; nested ordering `A < SA < SAB` holds; budget
≤ 200k holds.  (Relation-core includes the 64-wide centre update.)

### Correctness gates (all pass locally and on the remote checkout at `902cd2a`)

`no_mixed_bypass`, `s_chemistry_invariance` (exact 0.0), `a_assignment_invariance`,
`b_assignment_sensitivity` (A and S unchanged, B moves), `relation_purity`
(chemistry-free), `node_relabel_invariance`, `batch_invariance`,
`gradient_viability`, `parameter_accounting`.

Remote CPU smoke SAB: loss finite and decreasing
(1.31638 → 1.30556 → 1.29809), output not constant, cross-molecule channel std
A 6.86e-2 / S 1.83e-2 / B 2.36e-3, all gradient blocks nonzero, 95,209 params,
`official_test_loaded = false`.

### Deleted vs the current strong backbone

`patch_cont`, `patch_context`, typed `e_patch` lookup / shared / bag encoders,
`parent_token`, `global_context`, direct token readout, v6 `attribute_encoder`,
ring-context conditioning, and the chemistry columns of `pair_relation`.
No typed certificate, no vocabulary table, no mixed descriptor exists in the
FSAR module.

## Blocker — GPU co-tenancy not authorised

Both A100s are occupied (checked at the time of writing):

```
GPU 0: pid 117912, 35,780 MiB, 100 % util
GPU 1: pid 230846,    564 MiB
       pid 232854,  5,620 MiB, 99 % util
```

Per the operator rule ("do not co-tenant on an occupied GPU without explicit
authorisation"), the Stage-2 GPU smoke and the Stage-3/4/5 formal training are
**not launched**.  FSAR peaks well under 1 GiB (FSAB, a larger model, peaked at
478 MiB on A100), so a GPU-1 co-tenant run would be technically safe, but it
still requires explicit authorisation.

### Prepared formal commands (in order, once a GPU is free/authorised)

```
preprocess                                  # already done remotely (cache built)
smoke   --device cuda --deterministic --mode SAB
train   --mode A   --seed 0 --device cuda --deterministic
soup    --mode A   --seed 0
train   --mode SA  --seed 0 --device cuda --deterministic
soup    --mode SA  --seed 0
train   --mode SAB --seed 0 --device cuda --deterministic
soup    --mode SAB --seed 0
diagnostics   --mode SAB --seed 0
witness       --mode SAB --seed 0
interventions --mode SAB --seed 0
decide
report
```

Long runs use `launch_remote.sh` + `wait_remote.sh` (SSH-drop safe).  Stage 3
(A seed0) is gated by the pre-registered guard `A seed0 soup ≤ 0.135`; if it
fails, all further training stops and the relational core / pooling is audited
first.

Official ZINC test remains locked (`official_test_loaded = false` in every
produced JSON; no test split is loaded anywhere in this branch).
