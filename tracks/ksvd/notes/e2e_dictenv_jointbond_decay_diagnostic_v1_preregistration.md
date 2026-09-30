# Pre-registration — `e2e_dictenv_jointbond_decay_diagnostic_v1`

Round: **E2E-DictEnv-JointBond-Decay-Diagnostic-v1**, study `zinc-context-gap`,
track `ksvd`, reference result commit `aa9485b946738ce73f6e51ecc3c62b5e7f33e842`
(JointBond-v1).

This is a **short diagnostic** (2 x 200 optimizer steps), not a performance
round.  It exists because JointBond-v1 could only report that the added branch
ended **exactly inert** (all `joint_*` tensors at the float32 denormal floor
`7.0e-38`, `G_branch_off = 0.0`, gradient `0.00e+00` from epoch 20) without
isolating the contributions of **initialisation scale**, **multiplicative
parameterisation**, **coupled L2 weight decay** and **competition with the
parent path**.

Frozen **before** any run.  Everything below is a commitment; nothing may be
tuned after seeing results.

---

## 0. The one question

> With the parent model **frozen**, the input **fixed** and the **same
> non-zero initialisation**, can the new branch learn a training residual at
> all — and does Adam's **coupled L2 weight decay** suppress that?

Falsification is symmetric: if the NO-WD arm cannot fit the fixed batch either,
the dead endpoint in JointBond-v1 is **not** explained by weight decay, and the
round must say so.

---

## 1. Frozen parent

* Checkpoint: `tracks/ksvd/results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt`
  (the **original Sem108 soup**, do **not** use the JointBond-v1 soup).
* Canonical state hash (the runner's `audit.state_sha256` over sorted keys):
  `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a`
  — must be re-verified at load time and recorded.
* 49 parent keys, **no** `joint_*` keys; reference soup valid MAE
  `0.123704927947314` (C6 mask).
* Frozen: `D` (`sdb32`, sha `b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd`,
  65 x 32), the train-fit common subspace `q1`
  (`results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json`),
  `W_A_S/W_A_C/W_E_S/W_E_C`, `node_encoder`, `edge_encoder`, `fusion`,
  `pair_projection`, `relation_encoder`, `distance_gate`, `pair_encoder`,
  `global_encoder`, `topology_encoder`, `reader`, `U`, `common_rms`.
* **No parent parameter is ever updated** (the parent is excluded from both
  optimizers; verified by construction *and* by a bit-identical parent state
  check after the run).  No parent retraining, no parent re-initialisation.

## 2. Fresh branch initialisation (v1 architecture, exactly once)

* Build **one** `jb.JointBondModel` with `jb.build_jointbond_model(D, seed=0, q1)`
  (its own `torch.manual_seed(0)`), then load the Sem108 soup into the parent
  keys with `strict=False` (expected: `missing = 5 joint keys`,
  `unexpected = []`), then `copy.deepcopy` for the second arm.
* Branch architecture **unchanged from v1**: `A: 32->16` (Kaiming-uniform),
  `C: 28->16` (Kaiming-uniform), `B: 4->48` (Kaiming-uniform),
  `F = Linear(48,32,bias=False) -> SiLU -> Linear(32,48,bias=False)`
  (first layer Kaiming-uniform, last layer `Normal(0, 0.01)`), 4224 params.
  No width, bias, scaling, normalisation, gate or architecture change.
* **Never** start from the dead `joint_*` weights; the runner asserts the fresh
  branch is alive (per-tensor absmax `> 1e-6`) and its hash differs from the
  JointBond-v1 soup `joint_*` hash.
* One build -> deep copy, so **both arms start from bit-identical full state**
  (asserted by canonical full-state hash equality).

## 3. Two arms — the only difference

Both arms start from identical weights, identical data, identical order,
CPU only, 8 threads, and are run **sequentially** with a fresh optimizer:

| arm | optimizer | lr | weight_decay |
|---|---|---|---|
| **WD** | `torch.optim.Adam` (**coupled L2**, not AdamW) | `1e-3` | `1e-5` |
| **NO-WD** | `torch.optim.Adam` | `1e-3` | `0.0` |

* Optimizer parameter set: **only the 5 branch tensors** (4224 params).
  Verified by identity (ids), count and numel.
* "Coupled L2, not AdamW" is verified empirically before training, not assumed:
  a one-parameter probe with `grad = 0` and `weight = 1` must move by
  approximately `-lr` under `Adam(wd=1e-5)` (coupled L2: the decay enters the
  gradient) and by `-lr*wd` under `AdamW`; the same probe with `wd=0` must not
  move at all.
* Remaining Adam defaults unchanged (`betas=(0.9,0.999)`, `eps=1e-8`,
  `amsgrad=False`), identical (zero) initial optimizer state in both arms.
* `clip = 5.0` on the **global** parameter set, exactly as the frozen protocol
  does (`clip_grad_norm_(model.parameters(), 5.0)`), before `optimizer.step()`.
  Both the pre-clip global norm and the pre-clip branch-only norm are recorded
  so the applied clip coefficient is auditable.
* Dropout is off: the model stays in `eval()` for the whole diagnostic.
  `eval()` is *not* `no_grad()`: the training forward and `loss.backward()` run
  with autograd enabled so gradients flow **through** the frozen parent into
  the branch.  Measurement forwards use `torch.no_grad()`.

## 4. Data (official train only)

* `p1run.load_split("train")` -> official ZINC train, 10000 graphs.
  **The official valid and test splits are never loaded** (`official_test_loaded = false`).
* `torch.randperm(10000, generator=manual_seed(0))` (frozen seed 0):
  * **fit batch** = the first 128 indices, **probe batch** = the next 128
    indices (disjoint, no overlap).
  * The actual index lists, the collated-batch fingerprints (sha256 over the
    canonical bytes of `dict_phi / dict_atom / anchor / patch_cont / y / env_*`),
    the train cache path + sha256 and the runner's version/config hashes are
    written to `data.json`.
* The **same** fit batch is repeated for all 200 steps of each arm
  (400 steps total, 200 per arm).  The probe batch is **observation only**:
  never used in a loss, never used for selection, no early stopping, no
  checkpointing, no soup.

## 5. Objective

`loss = L1(prediction.view(-1), y.view(-1))` on the fit batch, C6 mask
(`cm.C6_MASK`), exactly the task term of the frozen protocol.  The
reconstruction term is a constant with respect to the branch under a frozen
parent (`rec` does not involve the branch and the dictionary is frozen), so it
is **not** used and is not evidence about branch learning.

## 6. Recorded steps

Telemetry at steps `0, 1, 5, 20, 50, 100, 200`, where step `k >= 1` is the
state **after** the k-th update and step 0 is the initial state:

**A. Learning.** `fit` and `probe`: `mae_on`, `mae_off` (branch disabled via
`jb.JointBondMask(joint_branch_off=True)`), `G_branch = mae_off - mae_on`,
`mean|pred_on - pred_off|`, `max|pred_on - pred_off|`, and the Pearson
correlation / sign-agreement between `pred_on - pred_off` and the parent
residual `y - pred_off` (a learned branch should start uncorrelated and become
positively correlated; the probe value is reported as *observation*, since the
probe batch is still training data).

**B. Branch response (per real bond occurrence).** RMS of the new branch
response `j_uv`, RMS of the parent edge response `ue`, their ratio, the
fraction of nonzero `j_uv` entries, and explicit zero-denominator flags
(`parent_rms == 0`, `branch_rms == 0`) — no epsilon-padded ratio is reported as
if it were a real ratio.

**C. Per new parameter tensor** (`joint_A.weight`, `joint_C.weight`,
`joint_B.weight`, `joint_F.0.weight`, `joint_F.2.weight`):
* weights: `absmax`, float64 norm, original float32 norm, nonzero fraction,
  counts of normal / subnormal / zero entries, and the flag
  `float32_norm_zero_but_float64_norm_nonzero`;
* the **actual float32 gradient used by this step's update** (recorded after
  `backward()` and **before** the optimizer adds its L2 term): `absmax`,
  nonzero fraction, original float32 norm, float64 norm, `all_zero_float32`
  flag, normal/subnormal/zero counts, and the same float32/float64 norm flag.
  A zero float32 norm is reported as *"the float32 norm underflowed"*, never as
  *"every gradient element is zero"*;
* the reference decay term `||1e-5 * w||` (float64) and its ratio to the task
  gradient norm (reported as `null` with an explicit flag when the task
  gradient norm is `0`; the NO-WD arm records the same reference value but is
  explicitly marked as **not applied**);
* the realised update: `(w_after.double() - w_before.double())` -> float64 norm,
  absmax, and relative update `||dw|| / ||w_before||` (`null` + flag if the
  denominator is 0), plus `weight_changed`;
* runtime dtype and `torch.finfo(dtype)` (`eps`, `tiny`, `max`) are recorded
  once per tensor record.

A per-step CSV (`step, loss, pre-clip global/branch grad norms, clip
coefficient, branch post-clip norm, joint weight norm float64, per-step update
norm`) is written for both arms; every step, not only the logged ones.

## 7. Pre-training correctness gates (all must pass before step 1)

* `D0` preregistration frozen (sha256 captured) and unchanged at run time.
* `D1` parent == Sem108 soup: 49 keys, canonical hash
  `7a721984...`, no `joint_*` key in the checkpoint.
* `D2` fresh branch alive: each tensor absmax `> 1e-6`; joint hash != v1 dead
  joint hash.
* `D3` both arms bit-identical at init (canonical full-state hash equality).
* `D4` step-0 predictions identical across arms (fit and probe), bit-wise.
* `D5` branch-off == frozen Sem108 parent prediction, bit-wise (separately
  instantiated `sem.build_sem108_model` + soup).
* `D6` initial branch response is nonzero (nonzero fraction `> 0`, RMS `> 0`).
* `D7` optimizer holds exactly the 5 branch tensors (identity + numel).
* `D8` coupled L2 confirmed empirically (Adam vs AdamW probe), `wd=0` -> no
  decay update.
* `D9` one throwaway optimizer step on a copy: parent state bit-identical,
  branch params changed.
* `D10` fit/probe index sets disjoint, inside the official train split,
  fingerprints recorded; only `train` was loaded.
* `D11` the training forward carries a gradient graph to the branch
  (`loss.requires_grad` and every branch tensor has `.grad` after `backward()`).
* `D12` reporting self-test: a synthetic subnormal float32 tensor is classified
  as subnormal, and its float32 norm (0) is distinguished from its float64 norm
  (> 0).

Post-run: the parent sub-state of **both** arms must be bit-identical to the
starting parent state (canonical hash equality).

## 8. Interpretation rules (fixed before the run)

Let `d_fit(arm) = mae_on_fit(step 200) - mae_on_fit(step 0)` (negative =
improvement on the fixed fit batch), and `d_probe` similarly.

* **"learns"** := `d_fit <= -1e-3`.
* **WD suppression** := NO-WD learns (as defined) **and**
  `d_fit(WD) - d_fit(NO-WD) >= 5e-4` (WD gives up at least 5e-4 MAE).
* **Both learn** := both arms satisfy `d_fit <= -1e-3`.
* **Neither learns** := both arms have `d_fit > -1e-3`.
* **Survival only** := NO-WD does not learn but keeps strictly larger weight
  norms / nonzero fractions than WD.

Reading:
1. NO-WD lowers the fit loss and WD clearly suppresses it → **decay-suppression
   effect under a frozen parent** (supports the JointBond-v1 reading, within
   this restricted setting).
2. Both arms learn → weight decay did **not** prevent learning here; the
   JointBond-v1 death then also involves the full-training dynamics
   (parent co-adaptation, moving inputs, 320-epoch horizon, soup averaging,
   dropout/train-mode).
3. NO-WD only keeps the weights nonzero without improving the loss → **survival
   only**, no effectiveness claim.
4. Neither arm learns → the trainability problem at this
   parameterisation/scale is **not** solved, and this diagnostic does **not**
   prove the structure–semantics information is redundant.

Explicit limits to be repeated in the analysis: a fixed 128-graph batch
improvement is **residual fitting, not generalisation**; the probe batch comes
from the **official train split** and is therefore not a generalisation measure;
freezing the parent changes the original training environment, so this round
cannot fully explain the from-scratch death; `eval()` (no dropout) is also a
deviation from the original `train()` loop.

## 9. Forbidden in this round

* any full training, second seed, matched control, soup, or performance claim;
* any architecture change (width, bias, gate, norm, scaling, additive form);
* initialisation search (no larger `F2` init, no zero-init); no `tanh`/linear
  gate; no denormal guard; no optimizer change other than the two arms above;
* reviving or retraining the parent node binding; no JointBond-v2;
* reading official valid or official test; nothing may be selected from them;
* changing the logged steps, the fit/probe indices, or the loss after the fact.

## 10. Deliverables

Stage runner + control-plane entry, pre-training gates, `data.json`,
`correctness.json`, `trace_{arm}.csv`, `records_{arm}.json`,
`decay_diagnostic.json`, `REPORT.md`, a concise analysis note, research records
(claim + decision) and a `STATE.yaml` entry.
