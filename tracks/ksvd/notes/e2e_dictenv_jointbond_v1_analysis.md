# Analysis — `e2e_dictenv_jointbond_v1` (JointBond-v1)

Round: **E2E-DictEnv-JointBond-v1** (`e2e_dictenv_jointbond_v1`), study
`zinc-context-gap`, track `ksvd`.
Candidate: **JointBond-v1 / key-level joint structure–semantics fusion**.
Parent: `SEM108Model` (`e2e_dictenv_sem108_v1.py`) @ `da2af28`.
Pre-registration:
[`e2e_dictenv_jointbond_v1_preregistration.md`](e2e_dictenv_jointbond_v1_preregistration.md)
(sha256 prefix `7384eca4a61a`).
Prior-artifact audit:
[`e2e_dictenv_jointbond_v1_prior_artifact_audit.md`](e2e_dictenv_jointbond_v1_prior_artifact_audit.md).

Verdict: **`JOINTBOND_NO_NEW_TASK_BAND` — the branch is exactly inert at the
endpoint, so this round is a negative mechanism result, not a performance
result.**

* `M_S = 0.12793026330223073` (Top-5 soup, seed 0, 320 epochs) → above the
  `0.123` band boundary and `+0.004225` **worse** than the historical
  (unmatched) parent soup `0.123704927947314`.
* `G_branch_off = 0.0` **exactly**: removing `j_uv` changes *no* prediction
  bit. Same for `G_joint_alpha0`, and for the branch-local residual-code /
  atom-type row shuffles at all three seeds (`G = 0.0`).
* Diagnosis: the 4224-parameter branch was **numerically sub-resolution from
  initialisation** (`F`-last `Normal(0, 0.01)` gating a multiplicative
  `b·B` term → branch response `2.2e-2` vs a parent edge response of `O(1)`,
  prediction influence `5.96e-06` already at epoch 0) and was then driven to
  the float32 **denormal floor** (`|w| <= 7.06e-38` for every branch tensor)
  by L2 weight decay against a gradient that is `0` in float32 from epoch 20
  onwards. At the soup state the branch output is exactly `0`.

CPU only, `official_test_loaded = false` everywhere, exactly one seed-0
trajectory, no baseline retrain, no sweep, no rescue. Formal-run commit
`a27f2ddb69fb8b5f25266e9dc10bfaed48df9a49` (promoted run
`20260930-100245-e164e7e7`, `records/runs/20260930-100245-e164e7e7.json`).

---

## 0. Question

The parent round (`e2e_dictenv_sem108_v1`) established that the *direct
shell-resolved interface* is strongly load-bearing while the **node-level**
dictionary binding collapses, and that the surviving increment is carried by
the **edge** channel (`G_edge = +0.038245`, `G_node = 0.0`).

This round asked one falsifiable question: **if the structure code and the
atom/bond semantics are bound at the *key* (per-bond-endpoint) level instead of
being pooled through the node path, does the fusion become load-bearing and
move the ~0.13 task band?**

Exactly one architectural change over the frozen Sem108 parent: a 48-D
per-bond residual

```
alpha_v = coord[v, common_dim:]              # 32-D residual code (common excluded)
q_v     = one_hot(dict_atom[v], 28)          # the *actual* endpoint's atom type
b_uv    = one_hot(env_bond_type[uv], 4)
h_v     = (alpha_v @ A) * (q_v @ C)          # 16-D
t_uv    = [h_u + h_v ; |h_u - h_v| ; h_u * h_v]      # 48-D
j_uv    = F(t_uv) * (b_uv @ B)               # 48-D, added BEFORE the shellpair index_add_
edge_response_uv = parent_edge_response_uv + j_uv
```

`A: 32->16`, `C: 28->16`, `B: 4->48` bias-free; `F = 48->32->48`, `SiLU`,
bias-free (no gate, no bias, no norm, no extra layer). `A/C/B/F0`
Kaiming-uniform, `F2 ~ Normal(0, 0.01)` once; the branch is initialised
*after* the parent so the parent's RNG stream is untouched.

---

## 1. What stayed frozen

Parent object `CSSD-Sem108` (`SEM108Model`), `cm.H1_CONFIG` (`d_e=48`, `K=32`,
`s=8`, `lambda=33.95873017865987`, `horizon=320`, `dict_kind=sdb32`),
`cm.C6_MASK` (`output_to_perf_core_only=False`; `cm.c6_equivalence_check()`
true), train-only common subspace `q1`, residual dictionary `Dbar_perp`,
IHT-10, node/edge shell definitions, pair relation, distance bucket, global and
topology branches, moment pooling, reader, Adam `lr=1e-3`,
`weight_decay=1e-5`, grad-clip `5.0`, batch 128, 8 CPU threads, split
train 10000 / valid 1000, Top-5 soup protocol, seed 0, 320 epochs. Nothing was
retuned.

The parent parameters are **bit-identical** to `sem.build_sem108_model` with the
same `torch.manual_seed(0)` stream (`correctness.json` → `J3`, and the
parent-equivalence forward `J4`: `branch_off_prediction_max_abs_diff = 0.0`).

The only parent-file modification is one hook, `_edge_response_delta`
(default `None` → bit-identical parent forward), plus the `_edge_env_parts` /
`_edge_env_slots` factoring that lets the delta be added on exactly the slots
the parent would sum. `J0_parent_interface_reused` re-checks the frozen
interface (`interface_dim 110`, `sem_dim 108`, `fusion_in 446`, `fusion_hidden
114`).

---

## 2. Parameter budget (closed form, one shot)

| | params |
|---|---|
| parent | 97709 |
| candidate | 101933 |
| new | **4224** (relative `0.04323`) |
| closed form | `A 32·16 + C 28·16 + B 4·48 + F 48·32 + 32·48 = 512+448+192+1536+1536` |

Deliberately **un-matched** (the pre-registration refuses a matched control at
this stage; the parameter contract only requires no parent width change).

---

## 3. Correctness gates J0–J13 (all PASS) and three pre-training gate fixes

All 14 gates pass on the formal-run commit:

| gate | key evidence |
|---|---|
| J0 parent interface reused | frozen dims/`c6_equivalence_check` true |
| J1 prereg phase A | frozen sha, bands, stop rule present |
| J2 parameter contract | `new = 4224`, `total = 101933` exactly |
| J3 parent shared bit-identical | 45/45 shared tensors `torch.equal` |
| J4 parent equivalence + edge path | branch-off prediction `0.0`; branch-on vs off `4.17e-06` |
| J5 endpoint correspondence | both-swap `0.0`; single-swap `2.70e-04`; 44.9 % of bonds change |
| J6 synthetic endpoint pairs | both-swap `0.0`; atom-only `1.67e-03` |
| J7 residual-code zero | alpha-zero branch output `0.0`, predictions **exactly** equal branch-off; common coordinate not read; no bias/norm |
| J8 gradients | grads reach `A/C/B/F` and `D`; branch and dictionary both move |
| J9 routing / batch offsets | routed delta == manual float64 reference `4.8e-09`; two-graph offset invariance |
| J10 parent edge shuffle | branch-off under shuffle == parent under shuffle `0.0`; shuffled vs unshuffled delta `2.47e-04` |
| J11 frozen parent reproduction | soup MAE reproduced **exactly** `0.123704927947314` (`bit_identical: true`) |
| J12 relabel invariance | `1.75e-05 < 1e-4` |
| J13 official-test blocker | official test never loaded |

Three **gate-definition** bugs were found and fixed *before any training*; no
architecture, weight, mask or threshold changed, and the corrected gates are
strictly stronger than the first draft:

1. `J0` reported `passed=None` (dict key was `"pass"`, the harness reads
   `"passed"`).
2. `J7` was written as `diff(alpha-zero, branch-off) > 0.0`, which is the wrong
   direction: with the residual code zeroed the branch output is exactly `0`,
   so the alpha-zero model **must** equal the branch-off model bit-for-bit. The
   gate now requires exact equality (`zeroed_alpha_is_exactly_branch_off:
   true`) **and** that branch-on differs from branch-off
   (`prediction_diff_branch_on_vs_off = 4.17e-06`), so it can no longer pass
   vacuously.
3. `J10`/`J12` compared forwards while the model was left in `train()` mode by
   the preceding gradient probe, so the backend dropout mask differed between
   the two sides of the comparison (`0.066`/`0.079` artefacts). Both gates now
   force `eval()` first.

Focused tests: `tracks/ksvd/tests/test_e2e_dictenv_jointbond_v1.py` (12 tests,
data-free synthetic dictionary/subspace) pass; the frozen parent's own 14 tests
still pass after the hook refactor.

---

## 4. Smoke (trainability only)

64 optimizer steps (8 epochs × 1024 graphs): finite, gradients on, best valid
`0.680372`. The smoke also records the only two live branch measurements of the
whole round:

| | init | after 64 steps |
|---|---|---|
| branch response norm | `2.219e-02` | `2.657e-03` |
| branch response absmax | `2.816e-04` | `5.514e-05` |
| `prediction_diff(on, off)` | `5.96e-06` | `5.48e-06` |

A branch response of `1e-4..1e-2` against a parent edge response of `O(1)`, and
a readout influence of `~6e-06` against a task scale of `~0.14`, is already the
float32 resolution floor: **the branch was numerically sub-resolution before
training started.**

---

## 5. Training — the single seed-0 trajectory

* device CPU, 8 threads, 320 epochs, wall `2334.2 s` (`7.29 s/epoch`)
* best valid MAE `0.1319279600828304` @ epoch `309`
* Top-5 soup `M_S = 0.12793026330223073`, members `[259, 271, 290, 309, 311]`
* final valid `0.13936922863771906`
* reconstruction term healthy throughout (`rec ~ 4e-05 .. 5e-04`,
  `rec_full ~ 0.976`), dictionary moves (`Dgrad ~ 1.8e-02..5.0e-02`,
  `movement 5.024`), 32/32 atoms active, `effective_atoms 19.57`,
  `top1_share 0.795`, `valid_reconstruction_relative 0.97696`

Band: `JOINTBOND_NO_NEW_TASK_BAND` (`M_S > 0.123`).

Historical (unmatched, read-only) references:
`SEM108 parent 0.123705`, `T1 0.125765`, `FINAL-CLEAN-C6 0.128499`,
`CSSD-q1 0.130028`, `RNDB 0.133117`, `control-lr1e3 0.127428`.  The candidate
soup is better than only two of these six.

**The `+0.004225` gap to the parent soup must not be attributed to the branch.**
`G_branch_off = 0.0` exactly proves the endpoint branch contributes nothing, so
this candidate at the soup state *is* a Sem108 parent; the gap is a
trajectory-level (seed/RNG/early-branch-perturbation) difference between two
single unmatched runs.  Equally, the round gives no evidence that the branch
helps.

---

## 6. Frozen interventions (no retraining)

All probes are inference-time interventions on the single frozen soup state;
they are not retrained controls.

Branch probes (this round; seeds `101/202/303` where applicable):

| probe | `M` | `G = M - M_S` |
|---|---|---|
| branch off (`j_uv := 0`) | `0.12793026330223073` | **`0.0`** (gate `0.003`) |
| residual code zeroed in the branch | `0.12793026330223073` | **`0.0`** |
| branch-local residual-code row shuffle | `0.12793026330223073` ×3 seeds | **`0.0`** |
| branch-local atom-type row shuffle | `0.12793026330223073` ×3 seeds | **`0.0`** |

Parent probes (re-run under this round's model/soup; seeds `101..505`):

| probe | `G` |
|---|---|
| residual dictionary zeroed (`G_dict0`) | `+0.045849` |
| edge assignment shuffle (`G_edge`) | `+0.026822` (range `+0.025286 .. +0.028306`) |
| node assignment shuffle (`G_node`) | **`0.0`** |
| semantic-root row shuffle (`G_sem_shuffle`) | `+0.573315` (range `+0.566254 .. +0.578367`) |

Pre-registered decision rule:
`band in {further_confirmation, candidate_signal} AND G_branch_off >= 0.003` →
`buy_matched_control = False` **on both clauses**.  No matched control, no
second seed, no rescue was purchased.

---

## 7. Why the branch died (mechanism diagnosis)

This is the substantive finding of the round.

**(a) Sub-resolution at birth.**  `F` ends in a `Normal(0, 0.01)` linear layer
*and* its output is multiplied by `b·B`.  The branch response was already
`2.2e-2` (absmax `2.8e-04`) at initialisation, and its influence on the
per-molecule prediction was `5.96e-06`.  The first gradient probe recorded
`jointA = 1.73e-06` at epoch 1 — i.e. the loss gradient entering `A` was at
float32 accumulation noise, so there is no numerically viable path for the
branch to grow out of a near-zero gate.

**(b) The optimizer then annihilated it.**  From epoch 20 onwards the
diagnostic gradient was `joint_A = 0.00e+00` *exactly* at every probe.  With
`weight_decay=1e-5` L2 added to an (effectively) zero gradient, Adam's
normalised step is ≈ `lr·sign(w)`; the branch weights therefore decayed
linearly and then multiplicatively to the float32 denormal floor.  At the soup
state:

| tensor | absmax | absmean |
|---|---|---|
| `joint_A.weight` | `7.008e-38` | `3.19e-38` |
| `joint_C.weight` | `7.055e-38` | `3.22e-38` |
| `joint_B.weight` | `7.020e-38` | `2.75e-38` |
| `joint_F.0.weight` | `7.032e-38` | `3.23e-38` |
| `joint_F.2.weight` | `7.051e-38` | `3.23e-38` |

Denormal `joint_A` underflows to exactly `0` in the matmul, hence `h = 0`,
`t = 0`, `SiLU(0) = 0`, `j = 0` exactly — which is *why* all six branch
probes read `0.0` and why `pred` is bit-identical with and without the branch.

**(c) The channel was redundant.**  The parent's *edge* channel is alive and
already carries the same information: `W_E_C absmax 0.636`, `W_E_S absmax
0.564` (vs node-level `W_A_C`/`W_A_S` at `7.05e-38`), `G_edge = +0.0268`,
`G_sem_shuffle = +0.5733`.  Binding the atom type and the bond type at the key
level re-injects a signal the frozen edge response already receives, from a
near-zero multiplicative gate, with no unique residual to fit.

**(d) The collapse repeats a pattern already in the record.**  In this same
system the *node*-level binding matrices (`W_A_C`, `W_A_S`, `node_encoder.0`)
sit at the identical denormal scale `7.05e-38`, and the parent round already
documented `G_node = 0.0` exactly.  The JointBond branch fell into the same
attractor: **in this architecture, any added channel whose loss gradient does
not exceed float32 resolution at its initialisation is deleted by Adam's L2
decay, and "add a small gated residual" is exactly such a channel.**  This is a
property of the optimizer/scale regime, not of the
structure–semantics-binding idea as such (the same round shows the *live* edge
channel is what carries binding here).

---

## 8. Case classification

| pre-registered band | condition | this round |
|---|---|---|
| further confirmation | `M_S <= 0.120` | no |
| candidate signal | `0.120 < M_S <= 0.123` | no |
| **no breakthrough** | `M_S > 0.123` | **`0.127930`** |

`endpoint_correspondence_read`: `not_established` (the branch-off and both
branch-local shuffles are all exactly `0.0`; the J5/J6 swap-contrast numbers
only establish the *shape* of the response, which is never used).

---

## 9. Limitations

* One seed, one trajectory, no contemporaneous baseline, no matched control,
  un-matched `+4224` parameters.  Nothing here measures the branch's causal
  cost — it measures that its causal *benefit* is exactly zero.
* The `+0.004225` gap to the historical parent soup is a between-trajectory
  difference, not a mechanism effect (see §5); the round cannot say whether the
  early (pre-denormal) branch perturbation helped or hurt the shared
  dictionary, because the two runs differ in more than the branch.
* The two causes in §7 — a design-time gate that is too small
  (`F2 ~ Normal(0,0.01)` × `b·B`) and the L2-decay/denormal attractor — were not
  separated by a controlled probe (that would require a no-decay or larger-gate
  variant, which the round forbade).  The recorded evidence supports
  "sub-resolution at init, then annihilated", not a specific split between the
  two.
* `G_branch_off = 0.0` is measured on a frozen soup state; early-branch effects
  on the shared parameters are invisible to it.

---

## 10. Next-round shapes (recorded only; nothing authorized or implemented)

1. **Do not iterate on this branch** (wider `F`, larger init, `tanh` gate,
   LayerNorm, bias, per-endpoint re-scaling): each would be an
   architecture-rescue attempt on a channel that was provably redundant and
   numerically dead.
2. If key-level binding is to be tested again, it needs a **matched,
   multi-seed, contemporaneous** design in which the branch is *not* gated by a
   near-zero multiplicative factor (e.g. the increment enters additively with
   unit-scale init), plus a mandatory liveness gate
   (`|w|_max > 1e-3` and `G_branch_off >= 0.003`) measured *before* any task
   interpretation.
3. The more valuable follow-up is optimizer-level and is shared with the
   parent's dead node binding: a preregistered diagnostic of the
   Adam-L2-decay/denormal attractor (e.g. no-weight-decay on residual channels,
   or an explicit denormal guard) that measures how many channels in the frozen
   Sem108 parent are already at the `7e-38` floor and whether the dead node
   binding can be revived by that single change.  This is a *diagnostic*, not a
   performance round.

`official_test_loaded = false`; the official test was not read, and the test
closure is not spent by this round.
