# PSCD-RM-v0 — Rate-Matched Task-Driven Dictionary Audit (analysis)

**Question.** PSCD-TMDL-v0 improved the identical strong reader when the
dictionary-selection rule was changed from frequency to task+MDL
(`Δ_dictionary = MAE_FREQ − MAE_TM ≈ 0.058–0.067`), but the task+MDL dictionary
was also materially **less compressive** (dynamic-state ratio `1.392×` vs
`0.888×` raw). Was the gain from *better task boundaries*, or simply from
*retaining more fine-grained states*?

**Verdict (first line).**

> **Rate-matched task dictionary construction is infeasible under the current
> 64-rule merge-only parameterisation.**

with the co-primary empirical finding that, at a **strongly rate-matched** high
compression regime, the task dictionary keeps only a marginal advantage over
frequency:

> `Δ_task^{high-rate} = MAE_FREQ-LOOSE − MAE_TM = 0.00919 (best) / 0.00808
> (soup)` — i.e. **≈86 % of the original PSCD-TMDL gain is a decompression
> effect**, not a task-selection effect.

**Next decision.**

> **`analyze merge-only parameterization before further dictionary learning`** —
> the decisive low-rate arm cannot be built because the frozen first-order MDL
> eligibility rejects exactly the high-merge rules a frequency-level budget
> requires.

---

## 0. Provenance

| item | value |
|---|---|
| round | PSCD-RM-v0 (`notes/pscd_rm_preregistration.md`) |
| code | `tracks/ksvd/code/run_pscd_rm_dictionary.py`, analysis `analyze_pscd_rm.py` |
| tests | `tracks/ksvd/tests/test_pscd_rm.py` (6 pass, data-free) |
| commits | `faf8c49` (prereg + code + tests), `a4d1eb2` (invariance fix + regression test) |
| execution | remote A100 host `res`; Stage A CPU (`rm2-stageA`, `a4d1eb2`), Stage B GPU 0 (`rm-BFREQLOOSE`, `faf8c49`) |
| frozen inputs | `dictionary_FREQ.pkl` md5 `bbb26189…`, `dictionary_TM.pkl` md5 `8c151be6…`, `split_indices.json` md5 `c4a21aa1…` (all unchanged from PSCD-TMDL-v0) |
| splits | canonical train 10 000 / official valid 1 000; **official test never loaded**; 9 000 discovery + 1 000 internal-monitor (only discovery drives merges) |
| reader | PSCD-SC-v0 `StrongPortOperator`, **192 257 params**, seed 0, unchanged |
| results | `tracks/ksvd/results/pscd_rm/` (git-ignored) |

**Commit-mixing note.** The Stage-A dictionaries/audits were regenerated at
`a4d1eb2`; the Stage-B reader summary was produced at `faf8c49`. The two
commits differ **only** in the `invariance_check` diagnostic (a relabel-index
bug), which cannot affect the dictionary, its tokenisation, or `stage_b_train`.
The FREQ-LOOSE dictionary content was verified **bit-identical** across the two
commits (`sequence`, `vocab` ids/keys/sizes/orbits, `k*` all equal; only the
embedded `provenance.git_commit` differs). Stage B was therefore not re-run.

## 1. Rate definition (validated against PSCD-TMDL-v0)

$$
R_{\rm occ}=\frac{\sum_G|O_G|}{\sum_G|V_G|},\qquad
R_{\rm state}=\frac{\sum_G|O_G|+\sum_G|P_G^{\rm active}|}{\sum_G|V_G|},
$$

an active port being a distinct `(occurrence, canonical slot)` pair
(`precompute_graph`'s `n_port`). Recomputed on the 9 000 discovery split:
FREQ-64 `(0.3156, 0.8887)`, TM-64 `(0.5551, 1.3923)` — matching the recorded
`0.315 / 0.888` and `0.555 / 1.392`.

## 2. Control H — FREQ-LOOSE (strong rate match)

The frequency prefix curve is monotone; `k* = argmin_k |R_state^F(k) −
R_state^TM| = 5`.

| quantity | FREQ-LOOSE (`k*=5`) | TM |
|---|---:|---:|
| `R_state` | **1.4046** | 1.3923 |
| `R_occ` | **0.5556** | 0.5551 |
| `|ΔR_state|` | 0.0124 (≤ 0.03 ✓) | — |
| `|ΔR_occ|` | 0.0004 (≤ 0.05 ✓) | — |
| match | **strong rate match** | — |

So the frequency algorithm stopped after **5 merges** lands essentially exactly
on the task+MDL dictionary's occurrence **and** dynamic-state budget.

Structural audit for FREQ-LOOSE (`audit_freqloose.json`):

* exactness `Decode(D,C_G) == G` **100 %** (discovery 9 000 + monitor 1 000);
* invariance random relabel **0 / 1 000** mismatches;
* `n_learned = 5`, all 5 used, **0 low-support (<45)**, **0 unused**, median
  graph support **7 277**, motif type sizes `{2:2, 3:2, 4:1}`;
* ports/occurrence 1.528, port/size 0.917; occurrence/atom 0.5606, MDL ratio
  0.250.

**Same dynamic-state budget, very different dictionary.** At matched
`R_state ≈ 1.40`, FREQ-LOOSE uses **5 coarse high-support motifs** (median
support 7 277, 0 unused) while TM uses **64 fine low-support motifs** (median
support 140, 6 low-support, 2 unused).

## 3. Control L — TM-RATE: infeasible under the frozen method

Task+MDL discovery under the hard representation-rate constraint stopped after a
single merge. At `r = 1` the method is **forced**: `support_ok = 36`,
`mdl_ok = 7`, `rate_ok = 8`, **admissible = 1** (the only candidate satisfying
support + `ΔL<0` + both rate floors was the size-2 motif `occ=10120`, task score
`0.0675`, `g_occ=0.0485`, `g_state=0.0971`). Then at `r = 2`:

| diagnostic | value |
|---|---:|
| `support_ok` / `mdl_ok` / `rate_ok` / **admissible** | 41 / 7 / 9 / **0** |
| `R_occ` (current) | 0.9515 |
| rate floor `need_occ = D_occ/m` | **0.01009** (≈ 2 103 merges) |
| best `g_occ` among MDL-compressive candidates | **0.00252** (526 merges) |
| best `g_state` among MDL-compressive candidates | 0.00505 |

No candidate is simultaneously **MDL-compressive** and **compressive enough**
for the frequency budget. Per the pre-registration this is recorded as

> `rate-matched task dictionary construction infeasible under the frozen 64-rule
> budget`

and the method is **not relaxed** (no support / `β` / `K` / max-size / extra-step
change).

**Mechanistic diagnostic (why infeasible).** The binding constraint is the
**frozen first-order MDL eligibility**, not the rate inequalities themselves
(which had 9 satisfiers). Verifying `candidate_mdl_and_rates` against
PSCD-TMDL-v0's `mdl_delta_candidate` (bit-identical `ΔL` and merge count for all
41 step-2 candidates), the first-order two-part code **rejects exactly the
high-merge rules** the frequency dictionary uses: e.g. a candidate with
**51 753 merges** has `ΔL = +129 998` (non-compressive), whereas the compressive
set tops out at **526 merges**. Technical cause: a freshly merged occurrence is
a new motif id whose ports are absent from the current `port_lp` table and are
charged the default 3 bits each, so large merges that create many ports accrue a
cost penalty — which is also why `D_TM` itself is less compressive than
`D_freq`. This reproduces PSCD-TMDL-v0's own `n_compressive` counts (7 at rounds
1–2) and is unchanged by the parent's MDL forced-fallback rule (the fallback
pool is itself the low-merge candidates, still below the rate floor).

## 4. Stage B — strong reader (the four comparison points)

Identical 192 257-param `StrongPortOperator`, seed 0, protocol identical to
PSCD-TMDL-v0. FREQ and TM are reused; **only FREQ-LOOSE is newly trained**
(TM-RATE cannot enter Stage B — infeasible).

| arm | dictionary principle | rate regime | `R_occ` | `R_state` | best valid | best epoch | Top-5 soup | train@best | epochs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **FREQ** | frequency | low / compressed | 0.3156 | 0.8887 | 0.27078 | 113 | 0.25331 | 0.13184 | 153 |
| **TM** | task+MDL | high / loose | 0.5551 | 1.3923 | **0.20374** | 230 | **0.19517** | 0.08517 | 240 |
| **FREQ-LOOSE** | frequency | high / matched to TM | 0.5556 | 1.4046 | 0.21293 | 220 | 0.20325 | 0.09249 | 240 |
| **TM-RATE** | task+MDL | low / matched to FREQ | 0.3156 | 0.8887 | — | — | — | — | 1 (infeasible) |

At the **same** `R_state`/`R_occ` as TM, FREQ-LOOSE reaches 0.21293 (best) /
0.20325 (soup): only **0.00919 / 0.00808** worse than TM — **below the
pre-registered `0.02` "negligible" band** and a small fraction of the `0.067`
raw gap.

## 5. Decomposition of the original TM gain

$$
G_{\rm total}=MAE_{\rm FREQ}-MAE_{\rm TM},\quad
G_{\rm decompress}=MAE_{\rm FREQ}-MAE_{\rm FREQ-LOOSE},\quad
G_{\rm task,high}=MAE_{\rm FREQ-LOOSE}-MAE_{\rm TM}.
$$

| metric | `G_total` | `G_decompress` | `G_task,high` | decompress share | task share |
|---|---:|---:|---:|---:|---:|
| best valid | 0.06704 | 0.05785 | 0.00919 | **86.3 %** | 13.7 % |
| Top-5 soup | 0.05814 | 0.05006 | 0.00808 | **86.1 %** | 13.9 % |

This is an **empirical decomposition**, not a strict additive causal identity;
it is reported because FREQ-LOOSE and TM are strongly rate-matched
(`|ΔR_state| = 0.012`, `|ΔR_occ| = 0.0004`).

`Δ_task^{low-rate} = MAE_FREQ − MAE_TM-RATE` is **undefined** (no TM-RATE
dictionary exists).

The rate–performance plane is `results/pscd_rm/rate_performance_plane.png`
(x = `R_state`, y = valid MAE; circles best valid, triangles Top-5 soup; TM-RATE
annotated at its target rate as infeasible). It shows FREQ-LOOSE and TM nearly
coincident at `R_state ≈ 1.40` and both far better than FREQ, while the low-rate
target point at `R_state = 0.889` could not be constructed.

## 6. Answers to the eight questions (§35)

* **Q1 — how much of the TM gain is explained by "less compression"?** **Most of
  it.** ≈86 % of the pre-registered `Δ_dictionary` is recovered by the *pure
  frequency* dictionary at matched rate (`G_decompress/G_total = 0.863 / 0.861`).
* **Q2 — at the same high-rate regime, is TM still better than FREQ-LOOSE?**
  **Only marginally** (`Δ_task^{high-rate} = 0.00919` best / `0.00808` soup),
  below the `0.02` negligible band.
* **Q3 — can the 64-step task dictionary meet the frequency-level budget?**
  **No.** TM-RATE is infeasible at round 2 under the frozen 64-rule method.
* **Q4 — how different is the matched-rate task vocabulary from frequency?**
  Cannot be assessed as intended: only 1 of 64 TM-RATE rules could be selected.
  The high-rate comparison is instead stark: at matched `R_state`, frequency uses
  5 coarse motifs vs TM's 64 fine motifs.
* **Q5 — is matched-rate TM reuse healthy?** N/A (no completed TM-RATE). Note
  the *matched-rate frequency* control is maximally healthy (median support
  7 277, 0 unused) — reuse is not the binding problem for frequency.
* **Q6 — in the low-rate regime, `MAE_TM-RATE < MAE_FREQ`?** **Undefined**
  (no TM-RATE dictionary).
* **Q7 — does the task gain change with representation rate?** **Yes**: at the
  loose (TM) rate it looks large (`0.067`), at matched frequency rate it is
  ≈`0.009`. The "task gain" is essentially the decompression effect.
* **Q8 — enough evidence to move from diagnostic to formal task-driven
  dictionary learning?** **No.** The low-rate control cannot be built under the
  merge-only parameterisation, and the high-rate control shows the residual task
  effect is marginal.

## 7. Final conclusion and next decision

**First line (frozen choice):**

> **Rate-matched task dictionary construction is infeasible under the current
> 64-rule merge-only parameterisation.**

Co-primary finding (must accompany the first line):

> At a **strong rate match** to TM, the pure-frequency dictionary (FREQ-LOOSE)
> is only `0.009` worse than the task+MDL dictionary, i.e. **≈86 % of the
> PSCD-TMDL-v0 gain is weaker abstraction / higher representation rate**, not
> task-selected boundaries.

**Next decision (frozen choice):**

> **`analyze merge-only parameterization before further dictionary learning`**

The infeasibility is caused by the frozen first-order MDL eligibility rejecting
the high-merge rules a frequency-level budget needs; the next round must
analyse whether the **merge-only, irreversible BPE parameterisation** (and its
MDL eligibility proxy) is what prevents a genuinely task-adapted dictionary,
rather than increasing `K`.

## 8. Caveats

* Single seed (0) for FREQ-LOOSE; the effect `0.009 < 0.02` is at/below the
  negligible band, so no seed-1 paired confirmation was triggered (the
  pre-registration only triggers seed 1 for a moderate `0.02–0.04` effect).
* The `Δ_task^{high-rate}` decomposition assumes FREQ-LOOSE and TM are
  interchangeable rate controls; they are strongly matched but not identical
  (`|ΔR_state| = 0.012`).
* The mechanistic claim (first-order MDL eligibility rejects high-merge rules)
  is a diagnostic of the frozen proxy, not a redesign; no proxy change was made.
* Official test never loaded; all numbers are valid-side.
* `analyze_pscd_rm.py` figures are regenerated locally from the pulled
  git-ignored artifacts; `ANALYSIS.json` records the arm table and decomposition.
