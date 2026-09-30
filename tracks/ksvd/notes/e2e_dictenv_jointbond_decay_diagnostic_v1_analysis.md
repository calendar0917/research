# Analysis — `e2e_dictenv_jointbond_decay_diagnostic_v1` (JointBond-Decay-Diagnostic-v1)

Round: **E2E-DictEnv-JointBond-Decay-Diagnostic-v1**, study `zinc-context-gap`,
track `ksvd`.  Reference result commit `aa9485b946738ce73f6e51ecc3c62b5e7f33e842`
(JointBond-v1).
Pre-registration:
[`e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md`](e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md)
(sha256 prefix `019d295cfd2b`, frozen before the run).
Promoted run `20260930-114703-8e2b5e5e`.

**Verdict: `WD_SUPPRESSES_BUT_BOTH_LEARN`.**

Under the frozen Sem108 soup parent, a fresh v1 JointBond branch with the same
architecture, initialisation and scale **does learn**: fit MAE on the repeated
128-graph batch falls `0.0581025 -> 0.0300958` (WD, `wd=1e-5`) and
`-> 0.0292343` (NO-WD, `wd=0`), i.e. `d_fit = -0.0280 / -0.0289`, and the
branch's own contribution (`G_branch = MAE_off - MAE_on`) rises from
`-3.14e-05` to `+0.0280 / +0.0288`.  Adam's coupled L2 weight decay
(`1e-5`) **slows** this by `8.6e-4` MAE (`3.0 %` of the improvement) and keeps
the branch weight norm lower (`6.76` vs `8.45`) — it does **not** prevent it.
So the JointBond-v1 death is **not** explained by "weight decay kills the
branch" and **not** by "this parameterisation cannot learn".

---

## 0. The one question

With the parent **frozen**, the input **fixed** and the **same non-zero**
initialisation, can the added branch learn a training residual — and does
Adam's **coupled L2** weight decay suppress that?

---

## 1. Frozen setting (everything except the two optimizers is shared)

| | |
|---|---|
| parent | `SEM108-seed0_soup_state.pt`, canonical state sha `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a`, 49 keys, soup valid MAE `0.123704927314` |
| branch init | one `build_jointbond_model(D, seed=0, q1)` (`A/C/B/F0` Kaiming-uniform, `F2 ~ Normal(0,0.01)`), joint state sha `0a9d7ae7…` != dead v1 soup joint state sha `2e1d23b4…`; per-tensor absmax `0.030 … 0.500` |
| arms | `Adam(lr=1e-3, weight_decay=1e-5)` vs `Adam(lr=1e-3, weight_decay=0)`, fresh identical optimizers, only the 5 branch tensors (4224 params), global `clip=5` |
| coupled L2 | verified empirically (`D8`): zero-grad/`w=1` probe moves `-9.9897e-04 ~ -lr` under `wd=1e-5`, `0.0` under `wd=0`; AdamW moves `0.0` (a `wd`-sized `-1e-8`, i.e. below the float32 step at `w=1`) |
| data | official train only; seed-0 permutation; fit = first 128 graphs, probe = next 128 (disjoint), never reused for optimisation; fit fingerprint `59fc4820…`, probe `c3033fb5…`, train cache sha256 `2d647981…` |
| protocol | `eval()` (dropout off) with autograd on, C6 mask, `L1(pred, y)` objective, 200 steps per arm on the same repeated fit batch, CPU 8 threads |
| gates | 15/15 pass (`D0`–`D12` incl. `D1b`); both arms bit-identical at init; step-0 predictions bit-identical; branch-off == frozen parent **bit-wise**; parent state bit-identical after **both** arms |
| wall clock | `29.7 s` for the 2 x 200 steps |

---

## 2. Learning (steps 0, 1, 5, 20, 50, 100, 200)

| step | fit MAE (WD) | fit MAE (NO-WD) | `G_branch` fit (WD) | `G_branch` fit (NO-WD) | corr(pred-diff, residual) WD | NO-WD | branch/parent RMS WD | NO-WD |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.058103 | 0.058103 | `-3.14e-05` | `-3.14e-05` | 0.083 | 0.083 | 0.0047 | 0.0047 |
| 1 | 0.058009 | 0.058008 | `+6.21e-05` | `+6.28e-05` | 0.072 | 0.074 | 0.0047 | 0.0048 |
| 5 | 0.057583 | 0.057576 | `+4.88e-04` | `+4.95e-04` | 0.030 | 0.030 | 0.0070 | 0.0077 |
| 20 | 0.053566 | 0.053519 | `+4.51e-03` | `+4.55e-03` | 0.025 | 0.023 | 0.0559 | 0.0583 |
| 50 | 0.049832 | 0.049704 | `+8.24e-03` | `+8.37e-03` | 0.142 | 0.151 | 0.1002 | 0.1058 |
| 100 | 0.040823 | 0.040277 | `+1.73e-02` | `+1.78e-02` | 0.453 | 0.464 | 0.1577 | 0.1724 |
| 200 | 0.030096 | 0.029234 | `+2.80e-02` | `+2.88e-02` | 0.596 | 0.608 | 0.2752 | 0.3106 |

* `d_fit`: `-0.0280067` (WD) / `-0.0288682` (NO-WD); `d_probe`:
  `+0.0019259` / `+0.0022133`.
* `mae_branch_off` stays exactly `0.0580711` in both arms at every step (the
  frozen parent's own fit MAE — the parent cannot move).
* The branch response grows from `0.47 %` to `27.5 / 31.1 %` of the parent edge
  response RMS; `joint` weight norm `6.17 -> 6.76 / 8.45`.
* **Probe (observation only, official-train data): the branch does not help.**
  Probe MAE *rises* `0.0569302 -> 0.0588561 / 0.0591435` and
  `G_branch_probe` is `-1.97e-03 / -2.26e-03` at step 200, while
  corr(probe) is only `0.189 / 0.201`.  What the branch learns is
  **batch fitting**, not a transferable response — and it slightly hurts a
  disjoint same-split batch.
* Both arms are **still improving** at step 200 (`min` at the last step; `15 %`
  of the improvement happens in the last 50 steps), so the WD/NO-WD gap is a
  *rate* difference at a fixed step count, not a converged ceiling difference.

## 3. Weight decay: small, real, not blocking

* `d_fit` difference `d_fit(WD) - d_fit(NO-WD) = +8.615e-04` (`3.0 %` of the
  NO-WD improvement) — above the pre-registered `5e-4` suppression threshold
  but far below "prevents learning".
* Final joint weight norm `6.764` (WD) vs `8.448` (NO-WD); branch/parent RMS
  ratio `0.275` vs `0.311`; corr `0.596` vs `0.608`.
* Reference decay `||1e-5 * w||` vs the task gradient at the **first** step:
  `F2 0.0009`, `C 0.046`, `A 0.078`, `B 0.203` — the decay term never
  dominates the task gradient.  By step 200 the task gradients grew ~100x
  (`|g| 2.9e-04 … 4.2e-03` -> `6.2e-03 … 1.2e-01`) and every decay/task ratio
  fell below `0.008`.
* Global clip coefficients were `0.367 … 1.000` (median `0.66-0.67`) in both
  arms, i.e. the frozen protocol's global clip scaling was active and identical
  in kind for the two arms; branch-only clip norms are recorded alongside.

## 4. Numerics (what has to be tightened)

* The first per-tensor task gradients are `2.0e-04 … 4.2e-03` in float32; the
  float64 norms of the same float32 tensors agree to ~7 significant digits and
  the smallest of them (`joint_B 2.0e-04`) is ~`4e32` times the smallest
  *normal* float32 value.  **A small gradient is not a float32 precision
  failure**, and the earlier JointBond-v1 `1.73e-06` first-epoch `dL/dA` must
  be described as *small*, not as *below float32 resolution*.
* **No tensor** showed the "float32 norm `0` but float64 norm `> 0`" pattern —
  not for weights, not for gradients, in either arm at any logged step.  That
  reporting rule (`D12` self-test: a subnormal tensor gives float32 norm `0.0`,
  float64 norm `5.877e-39`) never had to be invoked; all weights and gradients
  here are normal numbers (`weight_counts` = all-normal for every tensor).
* JointBond-v1's endpoint observations (branch output exactly `0.0`, all
  `joint_*` tensors at the `7e-38` denormal floor, `dL/dA = 0.00e+00` from
  epoch 20) remain valid *as observations of that trajectory*; they do **not**
  support "the branch parameterisation is untrainable", "the structure–semantics
  signal is redundant/uninformative", or "weight decay prevents learning".
  The correct reading is narrower: in that from-scratch trajectory the
  branch's endpoint gradient path was exactly zero, while here — same
  parameterisation, same initialisation, frozen parent — it is small but real
  and usable.
* Consequently "the branch died because `wd=1e-5` decayed it" needs the
  qualifier *conditional on the from-scratch state*, in which the gradient path
  is exactly `0` rather than merely small.  "Adam's coupled L2 overcame a
  sub-resolution gradient" is the narrow statement the previous round can
  support; the broader statement is refuted.

## 5. Answers to the four required questions

1. **Can the new branch learn an effective response and lower the fit loss?**
   **Yes.** Fit MAE `-0.0280 (WD) / -0.0289 (NO-WD)` (`~48-50 %`), `G_branch`
   `-3.1e-05 -> +0.028`, branch/parent edge-response RMS `0.005 -> 0.28-0.31`,
   and the correlation between the branch's prediction shift and the parent's
   residual `0.08 -> 0.60`.  The identical architecture/init/scale is trainable
   once the parent is frozen.
2. **What discernible effect does weight decay have?** A small, real,
   non-blocking retardation: `+8.6e-4` MAE at 200 steps (`3.0 %`), lower weight
   norm (`6.76` vs `8.45`), lower branch response (`0.275` vs `0.311` of the
   parent), same sign of learning in both arms.  Decay/task ratios are
   `<= 0.203` at the first step and `< 0.008` by step 200.  Because both arms
   are still improving, this is "slower per step", not "cannot".
3. **Which claims have to be tightened?**  See §4: (i) "the parameterisation
   cannot learn / the signal is redundant" — not supported; (ii) "the gradient
   was below float32 precision" — must be "small but representable", and no
   float32/float64 norm discrepancy occurred here; (iii) "weight decay
   suppressed the branch" — true only as a *conditional* statement about a
   state whose endpoint gradient is exactly zero, not as a general property of
   this branch.
4. **Is a new full experiment worth it?**  **Not for this branch.**  The
   diagnostic removes the two hypotheses that would justify a rescue
   (untrainable parameterisation; decay-dominated learning): with a frozen
   parent the branch optimises happily, and `wd=1e-5` only slows it.  At the
   same time a *successfully fitted* branch **does not transfer** even to a
   disjoint same-split probe batch (`G_branch_probe = -2.0e-03 / -2.3e-03`,
   probe MAE worse than at step 0), so "make the branch trainable" is not a
   promising route to a task-band gain.  Combined with the previous round's
   finding that this key-level channel is redundant with the still-live edge
   channel (`G_edge +0.0268`, `W_E_S 0.564`, `W_E_C 0.636`) while the
   node-level binding is dead at the same denormal floor, the motivated next
   step remains the **optimizer-level attractor diagnostic** recorded in
   `decision-e2e-dictenv-jointbond-v1-stop-branch-inert-no-task-band-20260930`
   (how many Sem108 channels already sit at `7e-38`; does a no-decay-on-residual
   -channels or denormal-guard variant revive the dead node binding).  Nothing
   is authorized or started here.

## 6. Limits (what this round cannot say)

* Fitting one repeated 128-graph batch is **residual fitting, not
  generalisation**; the probe batch is official **train** data, so even the
  probe numbers are not a generalisation measure.
* The frozen parent changes the original training environment: no
  co-adaptation, constant branch inputs, dropout off (`eval()`), L1 only, and
  only the branch in the optimizer.  This round therefore **cannot** fully
  explain the from-scratch JointBond-v1 death; it only removes two candidate
  explanations.
* 200 steps, one seed, one fit batch, one pair of arms, no architecture
  variation (as pre-registered).  Both arms are still improving, so the
  suppression magnitude is step-count dependent.
* The global clip (`0.367 … 1.0`) scales the branch updates as the frozen
  protocol does; a branch-only clip would give different step sizes.  Both
  norms are recorded, neither is interpreted as a mechanism.
* Implementation disclosures (all before any optimizer step, no design change):
  the `D5` "unmasked" sub-check first compared the branch-**on** forward (fixed
  to the branch-off mask; the pre-registered masked comparison
  `branch-off == frozen parent` passed and is the one reported), and the `D8`
  payload lacked its `passed` key; the first control-plane attempt failed while
  assembling metrics (`KeyError: optimizer_numel`, no training outputs used),
  was fixed and re-run, and the promoted run reproduces the same numbers.

## 7. Artifacts

* pre-registration `notes/e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md`
* stage runner `experiments/luyin16/zinc_jointbond_decay_diagnostic_v1.py`
* control-plane entry `src/ksvd_research/runners/zinc_jointbond_decay_diagnostic_v1.py`
* tests `tests/test_e2e_dictenv_jointbond_decay_diagnostic_v1.py` (6 data-free tests)
* results `results/e2e_dictenv_jointbond_decay_diagnostic_v1/`:
  `preflight.json`, `correctness.json` (15 gates), `data.json`,
  `records_wd.json`, `records_nowd.json`, `trace_wd.csv`, `trace_nowd.csv`,
  `decay_diagnostic.json`, `REPORT.md`
* promoted run `records/runs/20260930-114703-8e2b5e5e.json`
  (artifacts also copied into `runs/2026/09/30/20260930-114703-8e2b5e5e/artifacts/`)
