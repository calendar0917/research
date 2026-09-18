# Factorized Structure–Attribute Binding (FSAB) on ZINC patches

Branch `exp/factorized-structure-attribute-binding-zinc`.
Scientific commit `74dc5e5`; reporting-only follow-up `a8b138b` (see §12).
Results: `results/factorized_binding_encoder/`.
Records: `records/claims/claim-factorized-structure-attribute-binding-collapse-20260916.yaml`,
`records/decisions/decision-factorized-structure-attribute-binding-stop-20260916.yaml`.

**Verdict: Case E — the binding channel collapsed. STOP. Not promoted. Seed 1 not
purchased. Official ZINC test never loaded.**

---

## 1. Question

Can one strong molecular model **explicitly factorize** the information inside the
radius-2 rooted patch — used only as a *rooted local reference frame*, not as a
motif — into three separated quantities

* `S_v` — topology only (root flag, root distance, untyped adjacency),
* `A_v` — attribute marginals (atom/bond category statistics grouped by patch
  membership),
* `B_v` — centered structure–attribute **binding** (the only place where a
  structural role meets the attribute of the *same* node/edge),

and still match the frozen B-Full / recurrent reference on ZINC, under a
≤120k-parameter budget with **zero dataset-dependent vocabulary parameters**?

The separation is enforced by **input access**, not by naming or by an auxiliary
loss: S must not be able to see chemistry, A must not be able to see the root
flag / distance / degree / role / `src` / `dst` / adjacency, and B is the only
channel that sees the correspondence between a role and an attribute.

## 2. Pre-registered architecture (exactly one)

`FactorizedStructureAttributeBindingEncoder`
(`experiments/luyin16/structural_patch_encoder.py`), registered as
`patch_representation="factorized_binding"` in `zinc_patch_path_pooling.py`.

| stream | computation | output |
|---|---|---|
| **S** | `role = root_embedding(root) + distance_embedding(dist) + topology_base`; exactly 2 rounds of edge-aware message passing on the **untyped** directed edge list with `scatter_mean`; permutation-invariant pooling `[root_state; mean; std]` → MLP | `S_v` ∈ R¹⁶ |
| **A** | `atom_mlp(atom_embedding(atom))` and `bond_mlp(bond_embedding(bond))` pooled per patch by `struct_patch` / `struct_edge_patch` into `[atom_mean; atom_std; bond_mean; bond_std]` → MLP | `A_v` ∈ R¹⁶ |
| **B** | per-node centered role `r_u − mean(r)` and centered attribute `a_u − mean(a)`; node binding `p_u ⊙ q_u`; per-edge centered low-rank binding; pooled | `B_v` ∈ R¹⁶ |

Fusion is the pre-registered linear combination plus a residual MLP:

```
z_v = W_S S_v + W_A A_v + W_B B_v + b
e_v = z_v + MLP(z_v)          # output dim 16, identical interface to B-Full
```

Center is subtracted **patch-internally** (the finite-patch analogue of
`P(S,A) − P(S)P(A)`), so B is a genuine centered binding rather than a
re-encoding of the marginals. No attention, no gate, no softmax, no top-k, no
learned support selection, no motif vocabulary, no bypass, no raw typed token.

**Only the local token encoder changes** relative to B-Full: radius-2 patches,
relation construction, T = 2 tied recurrent pair–centre, global/topology
channels, head, optimizer, schedule and Top-5 soup are inherited bit-exactly.

## 3. Information-access table and purity / invariance audit

| channel | is allowed to read | is forbidden to read | observed |
|---|---|---|---|
| **S** | `struct_root`, `struct_dist`, `struct_src`, `struct_dst` (types discarded) | `struct_atom`, `struct_bond`, `struct_edge_patch`, certificate | `S_ignores_attributes = true`, shuffle-A changes S by exactly **0.0** |
| **A** | `struct_atom`, `struct_bond`, `struct_patch`, `struct_edge_patch` | root flag, distance, degree, role, `src`, `dst`, adjacency, propagation state | `A_ignores_topology = true`, shuffle-S changes A by **exactly 0.0** |
| **B** | role `r_u` and attribute `a_u` of the *same* node/edge (node and edge correspondences), centered in-patch | — | edge-attribute shuffle leaves A unchanged and changes B |

Verified by 30/30 sanity checks (local **and** remote checkout), including
synthetic-patch constructions where each check has a non-trivial expected sign:

* `node_relabel_invariant`, `batch_invariant` (max diff 5.96e-08),
* `shuffle_S_exactly_unchanged` = true (0.0),
* `shuffle_A_unchanged_within_float_noise` (2.76e-07),
* `shuffle_B_changes` (9.15e-03 at init),
* `S_ignores_attributes` / `A_reacts_to_attributes` / `A_ignores_topology`,
* `edge_shuffle_A_unchanged` and `edge_shuffle_B_changes`,
* AST audit: `fsab_has_no_loss_calls`, `no_attention_calls`,
  `no_auxiliary_loss_calls`, `no_distillation_calls`, `train_loss_is_l1_only`.

The loss used for training is **only** the ZINC regression L1/MAE. Shuffling
appears only as an evaluation diagnostic; no distillation, contrastive,
shuffle-training, reconstruction, role/atom-prediction, orthogonality, HSIC, MI,
rank or sparsity term exists anywhere in the encoder or the runner (enforced by
an AST call-name audit, not by string matching).

## 4. Parameters

| block | params |
|---|---|
| structure stream S | 16,368 |
| attribute stream A | 5,264 |
| binding stream B | 8,592 |
| fusion | 1,856 |
| **FSAB encoder total** | **32,080** |
| downstream (inherited, frozen design) | 49,343 |
| **candidate total** | **81,423** |
| dataset-dependent vocabulary params | **0** |
| inherited `parent_embedding` (reported separately) | 256 |

`candidate_typed_state_keys = []` — the inherited `typed_embedding` is deleted
at construction, so no vocab-sized lookup exists. Budget ≤ 120,000 respected
with 38,577 to spare. References: B-Bag 84,511 (encoder 35,168), B-Full 84,495
(encoder 35,152), A2 85,740, Cell A 85,763.

Initialization is ordinary (no tiny-residual or zero-final-layer trick), and the
optimizer / weight decay / schedule are inherited unchanged.

## 5. Tests and sanity

* `tests/test_factorized_binding_encoder.py` — **17 tests, all pass** (relabel
  and edge-order invariance, S attribute purity, A marginal purity, binding
  witness, attribute-permutation null on an all-distinct-role patch, orbit
  symmetry between structurally equivalent nodes, edge binding, batch
  invariance, gradient viability, AST no-attention / no-auxiliary-loss audit,
  parameter accounting).
* 469 related repository tests pass (all `zinc`/`patch`/`compact_v4`/
  `structural`/`explicit` selections) — no regression from the shared-model edit.
* Sanity stage: 30/30 checks on both the local tree and the remote checkout at
  the same commit.

## 6. GPU smoke and memory

`smoke --device cuda --deterministic` on A100-SXM4-40GB: loss finite and
decreasing; `structure_alive`, `attribute_alive`, `binding_alive`,
`output_not_constant` all true; output effective rank 3.27; peak
262,795,264 B (251 MiB). Formal seed0 peak 501,506,048 B (478 MiB) — no memory
risk, and no OOM possible for co-tenant jobs.

## 7. Formal protocol and run

Inherited protocol verbatim: Adam, lr 1e-3, weight decay 1e-5, batch 128,
gradient clip 5.0, max 240 epochs, patience 40, no scheduler, single stage,
checkpoint = best official-valid MAE, fixed equal-weight Top-5 soup,
`torch.use_deterministic_algorithms(True)`.

One seed0 run, physical GPU 1, commit `74dc5e5`:

| quantity | value |
|---|---|
| best valid MAE | 0.1259851760864258 @ epoch 234 |
| epochs run | 240 (no early stop) |
| wall clock | 3378.97 s (56.3 min), 14.08 s/epoch |
| Top-5 epochs | 234, 230, 216, 235, 204 |
| Top-5 valid | 0.1259852, 0.1277871, 0.1294376, 0.1296174, 0.1297518 |
| **Top-5 soup valid** | **0.12295305345853558** |
| soup improvement over best checkpoint | 0.0030321 |
| peak GPU memory | 501,506,048 B |

## 8. Results and matched comparisons (seed 0, seed0-vs-seed0)

| reference | seed0 soup | FSAB seed0 − reference |
|---|---|---|
| B-Bag | 0.12738177864899625 | **−0.0044287** (better) |
| B-Full | 0.11981802638241788 | **+0.0031350** (worse) |
| A2 | 0.12169302585240742 | **+0.0012594** (worse) |
| pre-registered guard (B-Bag seed0 + 0.002) | 0.12938177864899625 | **−0.0064287 → PASS** |

So the **performance guard passed** — by a wide margin, and nominally better
than the matched B-Bag seed0 soup. On performance alone this would have
authorised seed 1. The mechanism gate is what stops it.

## 9. Mechanism diagnostics (decisive)

At the selected checkpoint (epoch 234) the token encoder is **bit-constant**:

| channel | norm mean | norm std | effective rank | top singular fraction |
|---|---|---|---|---|
| S | 0.0057257 | 9.31e-10 | 0.0 | — |
| A | 0.0111738 | 1.86e-09 | 0.0 | — |
| B | 0.0068811 | 2.33e-09 | 0.0 | — |
| e (token) | 0.0206094 | **0.0** | 0.0 | — |

The norm spread is float32-rounding level (`eps × mean`). Weighted channel
contributions fell from `W_S·S 0.646 / W_A·A 0.221 / W_B·B 0.233` at init to
`1.21e-4 / 5.66e-4 / 2.63e-4`. The epoch curve shows the monotone decay:
`|S|` 0.635 (ep1) → 0.246 (ep20) → 0.038 (ep140) → 0.006 (ep240);
`|B|` 0.328 → 0.295 → 0.086 → 0.0042.

Eval-only witness and interventions agree exactly:

| diagnostic | value |
|---|---|
| `witness` S / A / B / e / pred mean abs delta | **0.0** for every channel |
| `B_delta_above_noise` | **false** |
| `interventions` valid MAE true | 0.1259851778735756 |
| `delta_no_S` | −2.068e-08 |
| `delta_no_A` | −1.615e-07 |
| `delta_no_B` | −6.992e-08 |
| `delta_shuffle_B` (3 repeats) | **0.0**, all repeats identical |

Removing S, removing A, removing B, or shuffling the attribute assignment
changes the prediction by nothing at all. `channels_alive = false`.

### 9.1 Independent verification (init-vs-trained A/B on identical molecules)

Because "the encoder is exactly constant" is a strong claim, it was re-measured
outside the runner's own diagnostic path, on 200 valid molecules at batch size
one, with a **fresh** model as the control on the same data:

```
FRESH   zero_param_frac=0.0015   S norm_std=2.88e-02  cross_mol_std=1.397e-02
                                 A norm_std=1.17e-02  cross_mol_std=7.639e-03
                                 B norm_std=1.61e-03  cross_mol_std=1.950e-03
TRAINED zero_param_frac=0.9391   S norm_std=4.67e-10  cross_mol_std=0.0
                                 A norm_std=9.34e-10  cross_mol_std=0.0
                                 B norm_std=4.67e-10  cross_mol_std=0.0
                                 e norm_std=0.0       cross_mol_std=0.0
```

Per-molecule vectors at the trained checkpoint (inputs differ wildly — atom
counts 231/111/115/163/144, atom-category sums 141/27/160/180/116):

```
mol0 y=0.630 pred=0.574 S=[9.67e-05, 3.023e-04, -5.490e-04, -4.67e-05]
mol1 y=2.466 pred=2.492 S=[9.67e-05, 3.023e-04, -5.490e-04, -4.67e-05]
mol2 y=1.351 pred=1.603 S=[9.67e-05, 3.023e-04, -5.490e-04, -4.67e-05]
```

Identical to the last digit. The **model still predicts differently per
molecule** (0.574 / 2.492 / 1.603 / 1.487 / 1.028), so this is not a broken
forward pass: the inherited channels carry the signal and the FSAB token is inert.

### 9.2 Parameter annihilation

**93.9 % of the encoder's 32,080 parameters are exactly zero** at the selected
checkpoint (0.15 % at init). The runner's `diagnostics` reports 81.8 % because it
averages each parameter tensor's zero fraction unweighted by tensor size; both
statistics describe the same annihilation.

This is the **same mechanism already documented in this track** for the Z1/ASB
cell, BCE and EOR: primitive/channel parameters receive a vanishing task
gradient once the residual path fits, and Adam's weight decay then erases them
exponentially. The monotone 100× decay of `|S|`, `|A|`, `|B|` with
`gS, gA, gB → 1e-5` across epochs 140–240 is that trajectory.

## 10. Interpretation

The factorization was **implemented and externally valid** — S/A/B purity,
relabel and edge-order invariance, batch invariance and the binding witness all
hold, and the channel statistics are alive at initialization. It is the
**optimization** that discards it: the inherited downstream can fit the target
from the global/topology/centre/relation channels alone, so the local token
becomes redundant, the fusion weights for S/A/B decay, gradients to the encoder
vanish, and weight decay finishes the job.

Consequently the guard-passing number **is not evidence for the factorization**.
The trained artifact is effectively the inherited Cell-A-family downstream plus
a learned constant token offset; its seed0 soup (0.122953) sits inside the
B-Bag per-seed spread (B-Bag seed0 0.127382 vs seed1 0.120229, spread 0.0072),
and it is worse than both A2 (0.121693) and B-Full (0.119818) seed0 soups. No
claim about structure–attribute binding, about S/A/B complementarity, or about
factorized local computation can be made from this run.

The result is therefore a **replication, in a third independent architecture, of
the same collapse mechanism** — which is durable information, not a clean
refutation of the factorization hypothesis (the hypothesis was not cleanly
tested, exactly as in the BCE case).

## 11. Verdict

Pre-registered decision mapping (Case E definition: the binding channel
collapses), evaluated on the matched per-seed references:

```
status                = SEED0_ONLY
case                  = E_binding_channel_collapsed
seed0_guard_pass      = true    (0.122953 ≤ 0.129382)
channels_alive        = false
mechanism_ok          = false
seed1_authorized      = false
official_test_loaded  = false
```

* **Do NOT promote.**
* **Do NOT buy seed 1** — the study is guard- and mechanism-gated, and the
  mechanism is exactly dead; a second seed cannot rescue it.
* **Do NOT** sweep widths, hidden sizes, activation, rounds, `s/a/b/attr` dims,
  fusion width, pooling, optimizer or weight decay.
* **Do NOT** open the official ZINC test (valid-only by design; a test read was
  never pre-registered).

## 12. Deviations from the brief

1. **GPU co-tenancy (authorised).** At launch time both A100s were running other
   users' formal jobs. The operator explicitly authorised sharing after the
   memory audit (FSAB peaks at 478 MiB). Seed0 ran on physical GPU 1 alongside
   an unrelated 0.56 GB job; `pandawei`'s 20.8 GB job exited mid-run.
2. **Reporting-only fix after the run (`a8b138b`).** The seed0-only branch of
   `decide()` returned the guard flag but neither the pre-registered Case label
   nor the seed0-vs-seed0 reference deltas, so the mechanism-gate failure
   produced an unclassifiable payload. The fix adds those reporting fields and
   uses the matched per-seed references; **no selection, guard, threshold,
   training or architecture logic changed**, and `decide()` was re-run to
   regenerate `decision.json`. The scientific commit for the trained artifact
   remains `74dc5e5`.
3. **No GPU0/GPU1 bit-identical repro was purchased.** The verdict does not ride
   on any numeric margin: the mechanism signal is *exact* (channel std 0.0,
   witness deltas 0.0, 93.9 % of parameters exactly zero), not threshold-adjacent.
   `--deterministic` was used throughout and the environment is pinned. This
   follows the BCE STOP precedent.
4. **Witness resolution.** `witness` uses 64 valid molecules (one batch) with 3
   shuffle repeats; `interventions` covers the full 1,000-molecule valid split
   with 3 repeats, which is the stronger of the two and agrees exactly.
5. **`dead_parameter_fraction` definition.** Reported two ways (§9.2) because the
   runner's statistic is a per-tensor unweighted mean while the independent
   probe measures the global parameter fraction.

Not deviated: no remote tracked-checkout edits (all remote work ran from the
pushed commit via `uv run`), single-variable comparison, L1-only training loss,
no bypass, zero dataset-dependent vocabulary parameters, official test locked in
every produced JSON.

## 13. Provenance

* Scientific commit `74dc5e5` (encoder + registration + runner + 17 tests);
  reporting commit `a8b138b`.
* Remote checkout `/home/hxy/cy/research` fast-forwarded to the pushed branch;
  remote runs recorded `remote_commit=74dc5e5…` and `remote_commit=a8b138b…`.
* A100-SXM4-40GB (physical GPU 1), torch 2.5.1+cu124, CUDA 12.4, Python 3.12.14,
  numpy 2.1.3, deterministic algorithms enabled.
* Run store `/home/hxy/.research_runs/fsab-seed0.{log,meta,pid,exit}` (exit 0).
* Results pulled to `tracks/ksvd/results/factorized_binding_encoder/`.
* `official_test_loaded: false` in `parameter_accounting`, `sanity`, `smoke`,
  `soup_fsab_seed0`, `diagnostics`, `witness_seed0`, `interventions_seed0`,
  `runs/fsab_seed0` and `decision`.

## 14. Recorded repair direction (NOT authorised)

The only repair that follows mechanically from §9.2 is to stop the annihilation:
give the S/A/B parameters a non-vanishing task gradient (e.g. exclude the FSAB
encoder's primitive/fusion parameters from weight decay, or bound the fusion
weights away from zero), and verify with a frozen witness that the channels stay
alive to convergence. This changes optimization semantics, so it requires its own
fresh pre-registration and its own frozen witness. It is **recorded only** and is
**not authorised** here.

The stronger read of the accumulated evidence — B-Full near rank-1, EOR inner
collapse, ESB connectivity-free tie, BCE composition collapse, and now FSAB
factorization collapse — is that *any* additional local structural channel on
this dataset/downstream is discarded by the optimizer because the inherited
channels already suffice. The next move should be paradigm-level, not another
local patch representation.
