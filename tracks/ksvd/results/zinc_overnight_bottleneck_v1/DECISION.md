# DECISION — zinc_overnight_bottleneck_v1

**Protocol** `zinc-overnight-bottleneck-v1` · **commit** `8657343edc62` · **date** 2026-10-03
**Decision class: `FAIL_TARGET / DIAGNOSIS_DELIVERED`.** Close both directions; do not open the
conditional combination; do not spend a second seed on any of these candidates.

---

## 1. What was decided

| decision | ruling | frozen basis |
|---|---|---|
| Direction **N** (2×2 node-collapse rescue) | **CLOSE** | no arm passes the N-gate; best gain −0.001207, all gains negative |
| Direction **T** (additive topology readout) | **CLOSE** | T-gate fails all three clauses; gain −0.004548 |
| Conditional combination **C** | **NOT RUN** | pre-registered precondition (both directions pass) is false |
| Second-seed confirmation | **NOT RUN** | pre-registered precondition (a frozen candidate passes its seed-0 gate) is false |
| ≥0.09 calibrated-valid target | **NOT REACHED** | best arm is the control itself, 0.111206 |
| Bottleneck question | **ANSWERED (negative)** | the node-channel collapse is not the valid-MAE bottleneck |

## 2. Mechanism attribution — how far the causal chain is actually established

**Established (directly observed, single-factor-controlled):**

* The control node channel dies under training. Two fixed **train** sentinel batches, eval-mode only:
  node-slot RMS 2.418e-4 (ep0) → 9.95e-6 (ep10) → 6.8e-7 (ep20) → 2.5e-11 (ep40) → **exactly 0**
  (ep80, 160, 240), `node_out` column-wise dead from ep20, and the node gradients go to 0 with it.
* The collapse needs **both** a tiny initial node-slot amplitude **and** node-specific weight decay.
  The 2×2 is unambiguous: N0 dies; N1 (unit amplitude, WD on) survives at RMS 1.36; N2 (original
  amplitude, WD off) survives and *grows 8×* to 2.0e-3; N3 (both) survives. Either single factor
  prevents collapse ⇒ the failure is a **multiplicative interaction**, not one culprit.
* The channel is *recoverable* and *used* when alive: with N1 the node output std is O(0.6), node
  gradients are O(1), and the dependency probe shows the task prediction changes when the node path is
  ablated much more than in the dead control. So the collapse is not "the module is unused"; it is
  "the module is switched off by the optimiser early".
* The additive-topology arm does what it claims: the topology response is provably independent of the
  other coordinates (swap-delta equals the scalar head delta to 6e-8, and is identical across 8
  different `other` vectors, whereas the shared reader's delta varies by 0.0596 across the same
  vectors).

**Not established (and explicitly not claimed):**

* The exact optimisation dynamics that drive the slot tensor to *exactly* zero (e.g. whether the
  gradient signal is below the effective decay threshold from epoch ~10, or an Adam second-moment
  artefact). The 2×2 identifies the *sufficient* factor pair; it does not decompose the per-epoch
  dynamics inside the interaction.
* Any statement about the official **test** split — it was never touched.
* Any significance statement. One seed per intervention arm; control seed spread is ±0.0029.

**Competing explanations that were tested and eliminated** (see `AUDIT.md` §3 "竞争解释的证据边界"):

| candidate explanation | status |
|---|---|
| A. Amplitude only (init too small) | **eliminated as sole cause** — N2 (original amplitude, no WD) survives |
| B. Node weight decay only | **eliminated as sole cause** — N1 (unit amplitude, WD on) survives |
| C. The node channel is the accuracy bottleneck | **eliminated** — all four rescues are ≥ control |
| D. The topology channel is gated illegitimately by the other coordinates | **confirmed as a mechanism** (see T0 check), but **it is not the accuracy bottleneck** — removing the gate makes MAE worse |
| E. A single catastrophic row (id172) dominating the mean | **eliminated as an artefact** — removing id172 leaves every gain negative (`gain_without172` < 0 for all arms) |

## 3. The decisive evidence: recovery costs accuracy rather than buying it

| arm | raw train MAE | eval gap | G0 MAE (965 rows) | G1 MAE (34 rows) | cal valid MAE | Δ vs control |
|---|---|---|---|---|---|---|
| `N0_s0` (control, channel dead) | 0.038990 | 0.073846 | **0.087130** | 0.247161 | **0.111206** | — |
| `N1_s0` (channel alive, unit scale) | 0.037299 | 0.078311 | 0.090646 | **0.156479** | 0.112710 | −0.001504 |
| `N3_s0` (channel alive, both rescuers) | **0.033356** | **0.088442** | 0.098850 | 0.182762 | 0.121682 | −0.010476 |
| `T0_s0` (topology ungated) | 0.044438 | 0.073757 | 0.088412 | 0.297215 | 0.115754 | −0.004548 |

The pattern is monotone in the *strength* of the rescue: train error falls (0.0390 → 0.0373 → 0.0334),
the train/valid gap widens (0.0738 → 0.0783 → 0.0884), the 965-row bulk group G0 gets worse, and the
calibrated valid MAE gets worse. The recovered node channel is **capacity that overfits the training
split**, and because G0 contributes 0.084 of the 0.111 total, its degradation swamps the G1
improvement. The control's own seed spread (0.002874) exceeds every rescue's effect size.

## 4. Consequences for the original bottleneck hypothesis

The working hypothesis was: *"the node channel collapses, therefore the model is stuck at ≈0.109;
re-open the channel and MAE will fall toward 0.09."* Both halves are now decided:

* First half is **true**: the collapse is real and fully characterised (2×2 interaction, exact-zero
  trajectory, ~350× signal-to-bias deficit at init).
* Second half is **false**: re-opening the channel does not reduce MAE. The 0.109 → 0.09 gap is
  therefore **not** explained by the node-channel collapse.

This is a useful falsification, but it means the 0.019 gap to target remains unattributed. What the
overnight evidence does narrow down, from the error budget and the arm behaviour, is that the residual
error lives in the **965-row bulk group G0** (contribution ≈0.084 of 0.111), and that all four
interventions that touched the *structural* interface (node binding, topology ungating) traded G1
improvement for G0 degradation. The gap therefore looks like a **bulk-group generalisation** problem
(train MAE 0.039 vs valid MAE 0.111, gap 0.074), not a broken-channel problem.

## 5. Single next action

**Do not iterate on node amplitude, node weight decay, or the topology readout.** The next
pre-registered experiment should attack the **G0 generalisation gap** directly: hold the architecture
and the node channel exactly at the control configuration, hold the recipe, and change exactly one
*regularisation / capacity-allocation* knob whose effect can be predicted before the run — with the
frozen control seed spread (±0.0029, two seeds) as the effect-size floor, and with the primary metric
restricted to `gain_without172` on G0. Any future gain claim must clear 0.0029 on at least two seeds.

Concretely, the cheapest honest next step is a **two-seed control-only replication at 240 epochs with a
different data-ordering seed**, to pin down the seed-spread floor before any further candidate is
funded.

## 6. What must not be re-done

* Re-running N1/N2/N3/T0 hoping the sign flips — the effect is negative and inside seed noise.
* Reporting a gain against the *historical* 0.109042 anchor as if it were the matched control.
* Any use of the official test split to rescue the target.
* Any second-seed confirmation of a candidate that failed its seed-0 gate.