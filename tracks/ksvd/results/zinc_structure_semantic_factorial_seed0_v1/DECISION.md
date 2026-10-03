# DECISION — zinc-structure-semantic-factorial-seed0-v1

Single seed (0), four fresh arms, train-inner dev only. The decision below is
about the **current canonical Full** and is deliberately narrower than
"dictionaries help / do not help".

## 1. Design decision

**Re-examine the explicit slot-binding interface against the Sem108 interface
before spending anything else on it.** Do not add reader capacity, do not scan
`kappa`/weight-decay/amplitudes, do not try to "revive" the node channel.

Evidence:

1. The node binding is dead in **all four** arms (weights denormal zero, slot
   variance 0). The node half of the interface is functionally unidentifiable
   in the current Full, so no node-factor decision can be made from this round
   — only recorded.
2. The edge binding is dead in `S_M`, `D_J`, `D_M` and alive only in `S_J`.
   Fresh training with the marginal operator (`S_M`) therefore realizes
   "binding off", and that arm is **better** on G0 than the correspondence arm
   (`C_S = −0.004981`, CI `[−0.009020, −0.000740]`). A model that lets the edge
   channel collapse generalises at least as well as a model that keeps it.
3. Conditional on `S_J`'s trained weights, the live paired edge interface is
   used and is better than its marginal substitute (`+0.010596` G0 raw). So the
   interface is not inert when alive; it is simply **redundant with the
   retained Sem108/marginal/topology bypasses and not worth keeping alive**
   under the current objective.
4. The coding factor gives no consistent benefit: `G_J`/`G_M`/`I` are
   inconclusive in both cal and raw on G0 and overall, with point estimates
   neutral-to-dense-favouring on G0 cal. The only coding×binding structure is
   survival (sparse+paired keeps the edge channel; dense does not), which is a
   training-dynamics fact, not a generalisation gain.

The concrete follow-up question (not executed, not authorized here) is the
**division of labour** between Sem108 (which already carries shell-conditioned
semantic statistics) and the explicit binding channel: which one is to be
retained, and what keeps the binding channel from being driven to zero. That is
an interface/regularisation/architecture question, not a capacity or
hyper-parameter search.

## 2. Mapping to the pre-registered observation table

| observed | row that applies | reading |
|---|---|---|
| J/M difference is absorbed by learned collapse in the real model; node dead everywhere, edge dead 3/4 | "collapse / constant semantics absorbed the channel" | The interface's claim is not realised in the current Full. Evidence stops at interface deactivation; do not scan amplitude/WD and do not write "correspondence does not matter". |
| `S_M` not worse than `S_J`, channels alive for `S_J` | "M not worse, channels alive, operator difference real" | Confirms the interface-labour re-examination; cannot be upgraded to "all structure–semantics combining is useless" because `S_J`'s live interface *is* functional (`+0.0106` swap). |
| `G_J`/`G_M` no sparse advantage; `I` inconclusive | "C_S/C_D positive but no sparse advantage" family, but here `C_S` is negative | Do not claim sparse-dictionary specificity; defer it until an alive interface exists to test on. |

## 3. What this round closes

* **Closed (this seed/dev):** "the current explicit slot correspondence provides
  an arm-level generalisation gain beyond the retained marginals." `C_S` is
  negative with a CI excluding 0 on the primary endpoint; `C_D` and `I` do not
  rescue a positive reading.
* **Closed (mechanism):** "the node binding is an active generalisation channel
  in the current Full." It is dead in every arm.
* **Closed (this seed/dev):** "the sparse code gives a generalisation advantage
  through the binding interface." No raw/cal-consistent positive signal.
* **Closed:** "the discrepancy is a reader-capacity or topology problem." No
  capacity/schedule change was made; the arms differ only in coding/binding, and
  the decisive difference is channel survival.

## 4. What remains unknown

* Whether the collapse is seed-specific (one seed only). The current Full's
  node death was already known to be unstable; this round shows the edge death
  is the common outcome and only sparse+paired resists it.
* Whether a deliberately kept-alive edge interface would produce a positive
  `C_S`/`C_D` on G0. The frozen swap shows the interface has value *when it
  exists*, but no arm in this round both keeps it alive and trains a comparable
  function.
* Whether sparse vs dense matters conditional on an alive interface at the same
  arm level. `I` is inconclusive.
* Whether any of this changes on `k=−1`/`k≤−2` tails (the M arms are worse
  there; `D_M` is much better on `k=−1`, `D_J` better on the extreme tail).
  Single-digit cell sizes; not a decision basis.

## 5. Not done, by rule

No seed 1, no full-10k confirmation, no warm fork, no `kappa` scan, no reader
capacity, no loss/regulariser change, no new features, no tail-head work.
`S_M`'s numeric pass of the candidate bar is reported but **not** promoted,
because its mechanism is interface collapse rather than a better marginal
operator.
