# SBCI Fit-vs-Generalization Failure Triage

**Track:** `ksvd` · **Protocol:** `sbci_fit_generalization_triage_v1`
**Module:** `tracks/ksvd/experiments/luyin16/zinc_sbci_fit_generalization_triage.py`
**Results:** `tracks/ksvd/results/sbci_fit_generalization_triage/`
**Tests:** `tracks/ksvd/tests/test_sbci_fit_generalization_triage.py`
**Official valid:** never loaded. **Official test:** never loaded.
**Gradient updates:** zero. Every number below comes from `model.eval()` forward
passes over the already-saved N3600 / I0 / T0 checkpoints.

**Verdict: `FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE` — no next architecture
family authorized.** SBCI does **not** lose the ≥0.008 of D3600 training fit
that a clean over-constraint failure would require, and it does **not** fit
comparably enough for a clean adequate-fit/wrong-bias verdict either. The four
fit deficits are small but point in opposite directions (RAW better for SBCI,
SOUP worse for SBCI), which is exactly the mixed case the pre-registered rules
send to "inconclusive" rather than to a new design.

---

## 1. Motivation

The `compact-v4-SBCI` run at N=3600 seed0 is a clean sample-efficiency NO-GO:
SOUP probe MAE **0.182924** vs the reused `compact-v4-smallhead` baseline
**0.176633** (`delta_SE = -0.006291`, RAW `-0.006215`). The SBCI note (sec. 27)
hypothesised that the constraint was "too strong at the tested scale", but that
hypothesis was never tested against **training-set fit**. A worse probe MAE
alone cannot distinguish two very different redesigns:

* the candidate function class is too restrictive to even approximate the
  required function, or
* the candidate fits the training data but encodes the wrong inductive bias.

This audit measures deterministic D3600 training fit for both frozen models and
routes the next architecture decision. It re-judges nothing about the NO-GO.

## 2. Why the SBCI NO-GO is already final

`compact_v4_sbci_sample_efficiency.md` (and `records/decisions/decision-compact-v4-sbci-nogo-20261007.yaml`)
established the NO-GO: the primary SOUP estimator is significantly worse
(paired bootstrap 95% CI `[-0.012760, -0.000176]`), the direction is negative
rather than sub-threshold, and the pre-registered seed1 / N7200 / N1800 / seed2-3
runs were correctly not purchased. That decision is unchanged here. This audit
only asks **why** the already-final failure happened, so that the next design
does not chase the wrong mechanism.

## 3. Two possible failure mechanisms

| | Hypothesis A | Hypothesis B |
|---|---|---|
| name | approximation / over-constraint failure | wrong inductive bias despite adequate fit |
| condition | SBCI cannot fit D3600 comparably | SBCI fits D3600 comparably, probe worse |
| next family | **Shared Structural Operator Dictionary (SSOD)** | **Direct Shared Structural Potentials (DSP)** |
| keep shared-basis idea? | yes, restore cross-channel expressivity | no, abandon latent-basis/operator sharing |
| shortening credit assignment? | no | yes |

The pre-registered thresholds are `dF_min, dF_late >= +0.008` for A,
`|dF_min|, |dF_late| <= 0.003` **and** RAW/SOUP selected deficits `<= +0.003`
for B, with probe degradation `>= +0.005` required for B and no
optimization-convergence warning for either.

## 4. Protocol comparability

`triage_protocol_compatibility.json` is **comparable = true** with all 16 checks
passing. Both runs are N=3600 / I=0 / T=0; the same D3600 indices
(`sha256 = 3be9f7e1…`), the same 800-selection and 2000-probe index hashes, and
the same subset salt `compact-v4-sample-efficiency-nested-subset-v1-20260912`;
identical optimizer (Adam), lr `1e-3`, wd `1e-5`, batch 128, L1, gradient clip
5.0, `max_optimizer_steps=13,680`, `eval_interval=57`, `patience=40`, and the
same K=5 equal-weight arithmetic-mean soup rule. Both rebuilt initial state
hashes match the frozen run manifests exactly
(baseline `63f2cecb…`, SBCI `f3479439…`), and the **20 shared tensors** are
bit-identical (`max_abs_diff = 0.0`, `hash_match = true`). If any of this had
failed the audit would have stopped as `INVALID COMPARISON`.

## 5. RAW selected training fit

`train_eval_raw.json`, recomputed from the frozen RAW checkpoint under
`model.eval()` over the full 3,600-molecule D3600 set:

| | step | select-800 | D3600 train MAE |
|---|---:|---:|---:|
| baseline RAW | 3,762 | 0.209662 | **0.073954** |
| SBCI RAW | 3,534 | 0.215511 | **0.072016** |
| `dF_raw = SBCI - baseline` | | | **-0.001938** |

Paired bootstrap over the 3,600 molecules (B=2000, seed 20260912): mean
`-0.001938`, 95% CI `[-0.004562, +0.000603]` (includes zero). On the RAW
estimator SBCI fits the training set slightly **better**, not worse.

## 6. SOUP training fit

`train_eval_soup.json`, recomputed from the frozen K=5 SOUP states:

| | member steps | D3600 train MAE |
|---|---|---:|
| baseline SOUP | 3762, 5985, 4788, 6042, 5073 | **0.048170** |
| SBCI SOUP | 3534, 3420, 5016, 5130, 4788 | **0.052724** |
| `dF_soup = SBCI - baseline` | | **+0.004554** |

Paired bootstrap 95% CI `[+0.002693, +0.006696]`, excludes zero
(`P(>0)=1.0`). On the primary SOUP estimator SBCI fits the training set
significantly **worse**. `dF_raw` and `dF_soup` therefore **conflict in sign** —
the first of the two mixed-evidence signals.

## 7. Best achieved training fit (diagnostic only)

`best_train_fit.json`, computed over **all 208 saved selection-evaluation
snapshots** (106 baseline, 102 SBCI), each re-evaluated on D3600 in eval mode.
The recomputed per-snapshot train MAE matches the stored curve exactly
(`max|diff| = 0.0`).

| | F_min | step | eval |
|---|---:|---:|---:|
| baseline | **0.055960** | 5,985 | 105 |
| SBCI | **0.059002** | 5,301 | 93 |
| `dF_min` | **+0.003042** | | |

This is far below the +0.008 approximation gate. Per the ticket, `F_min` is a
**diagnostic only**: it never selects a deployed checkpoint and never changes
the SBCI decision (Test 10).

## 8. Late-trajectory fit

`late_trajectory_fit.json`, median of each run's own last 10 selection events
(the grids are identical up to step 5,814, so the matched-grid window gives the
same answer):

| | last-10 median |
|---|---:|
| baseline | 0.069631 |
| SBCI | 0.068679 |
| `dF_late` | **-0.000952** |

The per-eval MAE is strongly oscillatory (residual std ≈ 0.030 for both models).
SBCI's window is `[0.0590, 0.1144, 0.0820, 0.1360, 0.0703, 0.0648, 0.0641,
0.1202, 0.0608, 0.0671]`. By median SBCI is marginally better, the second mixed
signal.

## 9. Optimization convergence

`convergence_diagnostic.json`. The **literal** pre-registered rule
(raw OLS slope of `F(e)` on the last 10 evals `< -5e-5` per selection-evaluation
interval, while stopped by patience/budget) **fires**: SBCI slope
`-0.002014`/eval. However that slope is a noise artifact:

* slope std error `0.003259`, `t = -0.618`, bootstrap 95% CI
  `[-0.008627, +0.003827]` — **includes zero**;
* the **monotone best-so-far** training fit improved only `+0.000993` over the
  last 10 SBCI evals, versus `+0.008651` for the baseline: SBCI's best training
  fit had essentially **plateaued**;
* the last-5 slope is `+0.00014` (rising), and the negative last-10 OLS sign is
  produced by two early spikes (0.1144, 0.1360) rather than a sustained decline;
* the selection-800 slope is `-0.000619`, also not a clean continuation signal.

The ticket defines the warning as *"SBCI still clearly continues to decline"*;
a statistically insignificant, spike-driven slope is not "clearly declining".
The effective `convergence_warning` therefore requires a **significant**
negative slope and is **False**; `literal_threshold_fired = true` is recorded
honestly. Either way the conclusion does not change: the fit gates already fail
both A and B, and both cases route to "no architecture".

## 10. Probe degradation

`probe_degradation()` recomputes the locked 2,000-molecule probe arrays:
baseline SOUP **0.176633**, SBCI SOUP **0.182924**, `delta_probe = +0.006291`
(RAW `+0.006215`). Paired SOUP bootstrap 95% CI `[-0.012760, -0.000176]`,
excluding zero. This is an **input** to the triage (B4 met), not a re-judgement.

## 11. Generalization-gap caution

`generalization_gap.json`: baseline gap `0.128464`, SBCI gap `0.130200`,
`delta_gap = +0.001737`. Because SBCI's SOUP training fit is worse, part of the
apparently larger generalization gap is just a worse training fit. The ticket
explicitly forbids reading a smaller gap as better regularisation, and the
absolute train-fit deficit — not the gap — is the primary diagnosis.

## 12. Approximation-vs-bias decision

`triage_decision.json`. The four fit deficits and the gates:

| quantity | value | A gate | B gate |
|---|---:|---|---|
| `dF_raw` | -0.001938 | `> +0.005` ✗ | `<= +0.003` ✓ |
| `dF_soup` | **+0.004554** | `> +0.005` ✗ | `<= +0.003` **✗** |
| `dF_min` | **+0.003042** | `>= +0.008` ✗ | `|·| <= 0.003` **✗** (by 4.2e-5) |
| `dF_late` | -0.000952 | `>= +0.008` ✗ | `|·| <= 0.003` ✓ |
| probe degradation | +0.006291 | — | `>= +0.005` ✓ |
| optimization warning | false | none ✓ | none ✓ |

* **Case A fails**: no fit loss reaches +0.008; the largest is `dF_soup = +0.0046`.
* **Case B fails**: `dF_soup = +0.00455 > +0.003` and `|dF_min| = 0.003042 > 0.003`,
  while `dF_raw` and `dF_late` have the opposite sign.
* The pattern — RAW better but SOUP worse, `dF_min` larger than `dF_late`,
  selected-estimator directions in conflict — is precisely the mixed zone the
  ticket (§21) sends to `FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE`.

**Decision: `FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE`.**

## 13. Implication for SSOD

`next_architecture_family.json` does **not** authorize the Shared Structural
Operator Dictionary. SSOD would require evidence that the diagonal shared-basis
factorization removed useful context-dependent expressivity, i.e. a material
training-fit deficit. The strongest fit deficit is `+0.0046`, well under the
`+0.008` gate, and on RAW the candidate actually fits better. There is no
approximation-failure license to restore cross-channel operators.

## 14. Implication for DSP

The audit also does **not** authorize Direct Shared Structural Potentials. DSP
would require a clean adequate-fit result (`|dF| <= 0.003` on best, late and
selected estimators, plus worse probe). SBCI narrowly misses that: `dF_min` is
`+0.00304` and `dF_soup` is `+0.00455`. It would be a "manufactured" conclusion
to round those to zero. The latent-basis prior is therefore *not* exonerated,
but the evidence is not strong enough to make abandoning it the pre-registered
design choice either.

## 15. What is and is not proven

**Proven (N3600 / I0 / T0, official-train development environment only):**

* both models were compared under an exact, verified, index-identical protocol;
* the reconstructions are exact: 208/208 snapshot train MAEs match the stored
  curves to `0.0`, and both frozen initial-state hashes and all 20 shared tensors
  match;
* SBCI does not lose ≥0.008 of D3600 training fit — it is **not** an
  approximation/over-constraint failure on this evidence;
* SBCI does not cleanly fit comparably either — its SOUP training fit is
  `+0.0046` worse (CI excludes zero) while its RAW fit is `+0.0019` better;
* the late-training decline is statistically indistinguishable from noise
  (bootstrap CI includes zero) and the best-so-far envelope had plateaued;
* SBCI did learn a non-trivial function: the frozen SBCI-only module weight
  norms are non-zero (`phi` 4.57, `psi` 3.95, `centre_composer` 2.91, `head`
  2.89) and all four gradient traces stay alive through training
  (`parameter_utilization.json`); no representation geometry was computed;
* no gradient update, no rescue, no new seed, no N7200/N1800, no official
  valid/test access.

**Not proven / not claimed:**

* that either SSOD or DSP is the right next family;
* anything about seed1, N7200, N1800 or official valid/test;
* that SBCI simply needed more training (the best-so-far plateau argues against
  it, and no resume is authorized);
* that "shared/low-rank structure" is disproven in general — only this specific
  K=16 minimal factorisation is closed.

## 16. Final architecture-family authorization

```
decision_case            FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE
case_if_literal_rule     OPTIMIZATION_AMBIGUOUS   (also authorizes no architecture)
authorized_for_design    false
family                   null
full_training_authorized false
zero_new_grad_updates    true
sbci_decision_unchanged  true
official_valid_used      false
official_test_loaded     false
```

No SSOD, no DSP, no SBCI rescue. The triage's own decision rule (§21/§29) is
explicit: "如果 neither A nor B … 不要 SSOD; 不要 DSP; 不要 SBCI rescue." A future
re-opening needs a genuinely different pre-registered hypothesis with its own
frozen witness, not a re-reading of these four small, sign-conflicting deficits.

---

## Q1–Q16

* **Q1** baseline N3600 RAW D3600 train MAE = **0.073954** (step 3,762).
* **Q2** SBCI N3600 RAW D3600 train MAE = **0.072016** (step 3,534).
* **Q3** `dF_raw = SBCI - baseline` = **-0.001938** (SBCI better; CI
  `[-0.004562, +0.000603]`).
* **Q4** baseline SOUP D3600 train MAE = **0.048170**.
* **Q5** SBCI SOUP D3600 train MAE = **0.052724**.
* **Q6** `dF_soup` = **+0.004554** (SBCI worse; CI `[+0.002693, +0.006696]`).
* **Q7** baseline best-achieved train MAE `F_min` = **0.055960** (step 5,985).
* **Q8** SBCI best-achieved train MAE `F_min` = **0.059002** (step 5,301).
* **Q9** `dF_min` = **+0.003042**.
* **Q10** `dF_late` (median of each run's last 10 events) = **-0.000952**.
* **Q11** SBCI train curve essentially converged before stopping: **yes** —
  best-so-far improved only `+0.000993` in the last 10 evals (baseline
  `+0.008651`); the literal slope is noise (bootstrap CI includes zero).
* **Q12** optimization warning: **no** (literal `-5e-5` threshold crossed only
  by an insignificant, spike-driven slope).
* **Q13** probe degradation (SOUP) = **+0.006291** (RAW `+0.006215`).
* **Q14** generalization-gap difference `delta_gap` = **+0.001737**
  (descriptive only).
* **Q15** failure type = **`FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE`** (not a
  clean approximation failure; not a clean adequate-fit/wrong-bias result).
* **Q16** next architecture family authorized: **no — none**
  (`authorized_for_design = false`).

---

## Artifact map

* Lock / inventory: `audit_protocol_lock.json`, `baseline_run_inventory.json`,
  `sbci_run_inventory.json`, `triage_protocol_compatibility.json`.
* Fit: `train_eval_raw.json`, `train_eval_soup.json`,
  `train_abs_err_{raw,soup}_{baseline,sbci}_N3600_I0T0.npy`,
  `snapshot_train_fit.csv`, `snapshot_train_fit.summary.json`,
  `best_train_fit.json`, `late_trajectory_fit.json`,
  `convergence_diagnostic.json`, `generalization_gap.json`,
  `parameter_utilization.json`.
* Decision: `triage_decision.json`, `next_architecture_family.json`,
  `answers_q1_q16.json`, `final_decision.json`.
* Tests: `tracks/ksvd/tests/test_sbci_fit_generalization_triage.py`.
