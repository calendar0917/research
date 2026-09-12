# Canonical Late-Readout Adaptation Confirmation

> **Question.** After canonical compact-v4-hinge joint training finishes, does
> freezing the final graph representation and separately adapting the *existing*
> MAE head improve **official-valid** MAE, and — if so — is the gain from
> "representation stabilization + head adaptation" or merely from "some extra
> optimisation steps"?
>
> **Verdict (2026-09-11).** **FULL CONTINUATION BETTER (Case C) — HEAD-ONLY
> FREEZING NO-GO.** Under the leak-free preregistered protocol the head-only
> branch **did not improve** official valid (`Delta_A = -0.00298`), while the
> matched full-model continuation improved it substantially
> (`Delta_C = +0.00909`). The decisive mechanism contrast
> `Delta_F = MAE(C) - MAE(E) = -0.01207` is strongly negative. The OOF
> late-readout signal **did not transfer** to the canonical fixed-budget,
> valid-free adaptation protocol. Official test was never loaded.
>
> Module: `experiments/luyin16/zinc_canonical_late_readout_adaptation.py`
> Results: `results/canonical_late_readout_adaptation/`
> Tests: `tests/test_canonical_late_readout_adaptation.py` (17 tests)

---

## 1. Motivation

Several representation / statistic / function-family routes have terminated
NO-GO. The only surviving positive local signal was a **frozen-representation
late-refit** effect observed in the OOF audit
(`notes/graph_head_refit_capacity_decomposition.md`): on a frozen compact-v4
`R`, re-fitting the *existing* head (warm continuation) improved the
outer-heldout MAE by `+0.0054` with 5/5 folds positive. That signal was
measured on **OOF folds of the official-train universe with a held-out
selection set**, not under the canonical leak-free train→valid protocol.

This stage performs the one missing test: take the canonical selected
validation checkpoint and ask whether a **leak-free** head-only adaptation can
reproduce the gain on official valid.

## 2. What level of the modeling stack is being tested

This is a **training-dynamics / representation-readout optimisation-coupling**
question, *not* any of:

- missing structural information;
- representation capacity;
- graph pooling;
- an alternative regression-function family;
- head width / depth.

Formally, joint training produces `(phi*, theta*)` with representation
`R_{phi*}(x)`. The experiment asks whether

```
theta_adapt = argmin_theta  sum_train |y - f_theta(R_{phi*}(x))|
```

generalises better than `theta*`. Representation, head architecture and loss
are held **fixed**; the intervention is the **optimisation schedule**.

## 3. OOF evidence

From the previous stage (`results/graph_head_refit_capacity_decomposition/`,
2 backbone seeds x 5 folds, frozen `R`):

| head | pooled outer-heldout MAE |
|---|---:|
| H0 original jointly-trained | 0.175031 |
| S-small scratch (302→13→13→1) | 0.170280 |
| L-large scratch (302→64→32→1) | 0.170534 |
| **L-warm original head continuation** | **0.169661** |

`Delta_W = H0 - L-warm = +0.005369` (95% CI [+0.00414, +0.00660], P(>0)=1.0,
5/5 folds positive). This is the strongest, most stable local signal in the
track so far.

## 4. Why capacity was rejected

`Delta_cap = MAE(L-scratch) - MAE(S-scratch) = +0.000255` (95% CI
[-0.00097, +0.00150], P(>0)=0.658, 2/5 folds); on **raw** inputs the sign
reverses (`Delta_cap_raw = -0.003264`). Small-head capacity is therefore not
the mechanism. The surviving hypothesis is that the jointly-trained head is
**under-adapted** to the final frozen `R`.

## 5. Joint-training optimisation hypothesis

If the representation keeps drifting faster than the readout can track it
(optimisation-timescale mismatch), then at the end of joint training the head
may be systematically behind the representation. Freezing `R` and letting the
head catch up would then help. The alternative — that *any* extra optimisation
helps — is exactly what the full-continuation control (C) tests.

## 6. Why full continuation is required

`B0` vs `E` alone cannot distinguish "freezing the representation helps" from
"extra steps help". The decisive comparison is therefore `C` vs `E` under an
identical budget, identical fresh-optimizer reset, identical minibatch order,
identical loss, and identical normalisation. Only `Delta_F = MAE(C) - MAE(E)`
supports a freeze-specific mechanism. **This stage's `Delta_F` is strongly
negative, so the mechanism is rejected.**

## 7. Leakage-safe protocol

- official **train** only for all parameter updates;
- official **valid** used **once**, after the fixed budget, for evaluation only;
- official **test never loaded**;
- the adaptation loop has **no valid loader argument** (enforced by test);
- `K*` fixed from the *previously completed* OOF audit before any canonical
  valid number was seen.

## 8. Adaptation-budget preregistration

`adaptation_protocol_lock.json` was generated **before** the canonical run. It
records the source commit, the OOF fingerprints, `K*`, the optimiser, LR, weight
decay, batch size, data-order seed, global torch seed, the standardisation rule,
the trainable-parameter sets, and all decision thresholds.

`K*` is the deterministic integer median of the L-warm OOF best-selection
epochs (`fold_results.csv`, column `best_epoch_lwarm`):

```
[11, 34, 42, 58, 79, 108, 131, 295, 312, 327]  -> median 93.5 -> K* = 94
```

(rule `round_half_up(median)`, recorded in the lock). Official valid was never
consulted to choose the horizon. Caveat recorded in the lock: the OOF fit set
was 7200 molecules (≈15 batches/epoch) while the canonical train set is 10000
(≈20 batches/epoch); `K*` is an **epoch** budget, so the canonical run takes
about 43% more optimizer steps than the median OOF run. This is the
preregistered rule and was not changed after seeing results.

## 9. Starting-point equivalence (hard gate)

All three branches start from the same canonical selected checkpoint (same
state SHA; identical tensor fingerprint). The normalisation is a fixed affine
`z = (R - mu) / max(sigma, 1e-6)` fit **once** on official-train `R`, and the
head's first layer is reparameterised exactly (`W' = W diag(scale)`,
`b' = b + W mu`) so that `head'(z) == head(R)`.

| quantity | value |
|---|---:|
| float64 algebraic identity (256 train samples) | 7.1e-14 |
| float32 deployment, official train (max abs) | 1.9e-6 |
| float32 deployment, official valid (max abs) | 9.5e-7 |

B0 recomputed official-valid MAE equals the canonical JSON value bit-for-bit
(`0.17006561887910357` on seed 0). Gate passed.

## 10. B0 / C / E definitions

| branch | trained parameters | forward |
|---|---|---|
| **B0** stop | none | canonical checkpoint on raw `R` |
| **C** full continuation | all 99,613 model params | whole model in train mode |
| **E** head-only adaptation | 8 head tensors (Linear + LayerNorm affine), R frozen | backbone eval (R fixed), head dropout active |

Common: L1 loss, Adam(lr=1e-3, wd=0), batch 512, deterministic mini-batch
order (seed 0), global torch seed 0, 94 epochs, same fixed standardisation.

## 11. Seed 0

| quantity | value |
|---|---:|
| B0 stop | **0.170066** |
| C full continue | **0.160976** |
| E head-only | **0.173042** |
| `Delta_A = B0 - E` | **-0.002977** |
| `Delta_C = B0 - C` | **+0.009090** |
| `Delta_F = C - E` | **-0.012067** |

Seed-0 status: **FULL_CONTINUATION_BETTER / CLEAR_NO_GO** for head-only
(`Delta_A < +0.0015` and `MAE(C) < MAE(E) - 0.0015`).

## 12. Replication decision

Per the preregistered rules, a seed-0 `Delta_A < +0.0015` is a **CLEAR NO-GO**
and seed 1 is **not** run (§33/§35). The full-continuation control additionally
satisfies the §32 "head-only freezing no-go" condition. Seed 1 was therefore
**not executed**.

## 13. Seed 1 if run

Not run. `pooled_seed_results.csv` contains seed 0 only.

## 14. Representation-drift diagnostics

Fixed probe = first 2000 official-train molecules.

| branch | normalised L2 drift | mean cosine | max abs drift | R frozen? |
|---|---:|---:|---:|---|
| C | 0.1801 | 0.9873 | 63.9 | no |
| E | 0.0000 | 1.0000 | 0.0 | **yes** |

Head movement `||Δθ||/||θ||`: C = 0.052, E = 0.080. Valid prediction movement
`mean |Δŷ|`: C = 0.098, E = 0.073.

Interpretation (descriptive, **not** causal): in C the representation moves
substantially while valid improves; in E the representation is exactly frozen
yet valid worsens. This does **not** support "the final representation is being
perturbed by joint optimisation"; here joint optimisation is what helps.

Training-only curves (eval-mode, no valid used for selection):

| epoch | C train MAE | E train MAE |
|---:|---:|---:|
| 1 | 0.1079 | 0.1000 |
| 24 | 0.0835 | 0.1037 |
| 70 | 0.0821 | 0.0922 |
| 94 | 0.0833 | 0.0893 |

E lowers training error but raises valid error — i.e. the head-only adaptation
**negatively transfers** in this protocol.

## 15. What is and is not proven

**Proven (within this protocol):**

- The OOF late-refit gain does **not** transfer to a canonical, leak-free,
  fixed-budget, valid-free head-only adaptation on the *same* architecture.
- Full continuation with the *same* fresh-optimizer budget **does** improve
  official valid by `+0.00909`, so the extra optimisation signal is real here.
- Freezing the representation is **actively worse** than continuing to adapt it
  (`Delta_F = -0.01207`); head-only freezing is a no-go.

**Not proven:**

- That the frozen-representation refit is worthless in general — it is
  worthless **under this fixed-budget, no-selection canonical protocol** where
  the OOF gain relied on an inner selection set.
- That representation drift was the cause of anything (the drift here is
  associated with *improvement*, not degradation).
- Anything about MSE, FM/CatBoost/XGBoost, head width, or structural features —
  none were tested.

**Implementation caveat (recorded honestly).** An early exploratory version did
not seed the global torch RNG, so the head dropout masks varied between runs
(E landed in 0.1688–0.1730 across unseeded reruns). The final protocol seeds the
global RNG (0, recorded in the lock); the canonical run is **bit-reproducible**
(two independent serial 4-thread runs produced identical C and E MAE). The
conclusion is robust in the sense that **no** observed E realisation reached the
`+0.003` advance threshold, and C always dominated E. But the seed sensitivity
of the E branch is itself a warning that the head-only signal is fragile at this
budget.

## 16. Implications for MAE readout training

For fixed `R`, L1 regression estimates the conditional median `median(y | R)`.
The experiment does not show that this refit is a productive late step under the
canonical protocol; it shows the opposite at a fixed budget. The large positive
effect is instead in **continuing to train the whole model**: this canonical
run's early-stopped selected checkpoint (epoch 53, valid `0.170066`) can be
improved to `0.160976` by a fresh-optimizer continuation, which is a statement
about the **canonical training budget / early-stopping**, not about a frozen
readout.

## 17. Next-step authorization

- **Not authorized:** formalising "two-phase freeze + head adaptation" as a
  method candidate; extending the head-only budget / lowering its LR /
  layerwise LR / unfreezing one backbone layer / adding MSE control / FM /
  CatBoost / XGBoost / small-head replacement.
- **Not authorized:** opening official test.
- **Separate future audit (if the project wants to use the gain):** the
  full-continuation gain is a training-budget/early-stopping question. It should
  be studied as its own hypothesis (e.g. matched-step full continuation vs the
  canonical early-stopped checkpoint), not folded into the late-readout branch.
- Reopening representation capacity, FM/CatBoost, or structural-feature searches
  during this experiment is forbidden and was not done.

## 18. Final verdict

**FULL CONTINUATION BETTER (Case C) — HEAD-ONLY FREEZING NO-GO.**

The canonical late-readout adaptation branch is **not supported**. Freezing the
final compact-v4 representation and adapting only the existing MAE head does not
improve official-valid MAE; matched full-model continuation improves it by
`+0.00909`. The OOF late-refit gain did not transfer to the leak-free canonical
protocol.

---

## Q1–Q17

- **Q1** Canonical compact-v4 graph head real structure? `302 → 64 → 32 → 1`
  with `LayerNorm(64)` and `Dropout(0.05)`; 21,633 params
  (21,505 linear + 128 LayerNorm). Verified from the real state dict.
- **Q2** Why is this training dynamics, not representation capacity?
  Representation, head architecture and loss are held fixed; the only
  intervention is which parameters are trainable / for how long.
- **Q3** OOF late-refit evidence? `Delta_W = +0.005369` (95% CI
  [+0.00414,+0.00660], P=1.0, 5/5 folds).
- **Q4** Why is the capacity hypothesis downweighted? `Delta_cap = +0.000255`
  (CI across 0, 2/5 folds) and sign reversal on raw inputs.
- **Q5** Fixed adaptation budget `K*`? **94 epochs**, deterministic
  `round_half_up(median(best_epoch_lwarm))` over the 10 OOF runs; read from
  `fold_results.csv`, recorded in `adaptation_protocol_lock.json`.
- **Q6** B0 seed 0 valid MAE? **0.17006561887910357** (bit-equal to the canonical
  run).
- **Q7** C seed 0 valid MAE? **0.1609755826992332**.
- **Q8** E seed 0 valid MAE? **0.17304227268049727**.
- **Q9** seed 0 `Delta_A`? **-0.00297665**.
- **Q10** seed 0 `Delta_F`? **-0.01206669**.
- **Q11** Does plain full continuation improve? **Yes**, `Delta_C = +0.00909004`.
- **Q12** Is seed 1 run? **No** (seed 0 is a clear negative / full-continuation
  better).
- **Q13** Did seed 1 replicate? **N/A**.
- **Q14** How much does `R` drift in C? Normalised L2 `0.1801`, mean cosine
  `0.9873`, max abs `63.9` on a fixed 2000-molecule train probe.
- **Q15** How much head adaptation in E? Head `||Δθ||/||θ|| = 0.080`; valid
  prediction movement `mean |Δŷ| = 0.073`; `R` exactly unchanged.
- **Q16** Does head-only improvement exceed the matched extra-training control?
  **No** — it is far worse (`Delta_F = -0.01207`).
- **Q17** Final category? **FULL-CONTINUATION BETTER** (Case C), i.e. the
  head-only freezing branch is a NO-GO.

---

## Reproduction

```
uv run python -m tracks.ksvd.experiments.luyin16.zinc_canonical_late_readout_adaptation all
uv run pytest tracks/ksvd/tests/test_canonical_late_readout_adaptation.py
```

Output: `tracks/ksvd/results/canonical_late_readout_adaptation/`.

**Execution note.** All canonical numbers were produced **serially**, one
training process at a time, with `torch_threads=4`, per
`notes/reproducibility_cpu_determinism.md`; the two branches are
bit-reproducible across independent serial reruns.
