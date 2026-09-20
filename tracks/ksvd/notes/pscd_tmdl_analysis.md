# PSCD-TMDL-v0 — Task + MDL Motif Discovery Diagnostic (analysis)

**Question.** If reader, ports, vocabulary size and motif size are frozen, does
changing **only the dictionary-selection rule** from frequency to task+MDL
materially improve held-out prediction?
`Δ_dictionary = MAE_FREQ − MAE_TM`.

**Verdict (first line).**

> **Task supervision materially improves the reusable compositional dictionary.**

with a **mandatory caveat**: the task+MDL dictionary is substantially **less
compressive** than the frequency control (dynamic-state ratio `1.39×` vs
`0.89×` raw), so the predictive gain is entangled with reduced compression and
the "reusable / compressive" half of that claim is **not** cleanly established.

**Next decision.**

> **`strengthen dictionary regularization before further task coupling`** — the
> decisive follow-up is a **compression-matched control**: hold the
> occurrence/dynamic-state budget equal to the frequency arm and ask whether the
> task-selection gain survives.

---

## 0. Provenance

| item | value |
|---|---|
| round | PSCD-TMDL-v0 (`notes/pscd_tmdl_preregistration.md`) |
| code | `tracks/ksvd/code/run_pscd_tmdl_dictionary.py` |
| tests | `tracks/ksvd/tests/test_pscd_tmdl.py` (5 pass, data-free) |
| commits | `3b77b4b` (prereg + code + tests), `aed5796` (stageB `--artifact`), `0839151` (artifact tag `TM`) |
| execution | remote A100 host `res`; Stage A CPU, Stage B GPU 0 (FREQ) / GPU 1 (TM) |
| splits | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000**; official `test` never loaded |
| dictionary split | canonical train → **9 000 discovery + 1 000 internal-monitor** (seed 0, `split_indices.json`) |
| reader | PSCD-SC-v0 `StrongPortOperator`, **192 257 params**, unchanged |
| seed | 0 |
| results | `tracks/ksvd/results/pscd_tmdl/` (git-ignored) |

## 1. Stage A — three dictionaries (cheap Ridge probe)

All three start from the same singleton partition, 64 merge steps, max size 8;
all reconstruct **100 %** (`Decode(D, C_G) == G`, discovery + monitor).

| arm | discovery Ridge MAE | monitor Ridge MAE | disc−mon gap | motif/atom | MDL `ratio_total_over_atom` | support median | low-support `<45` | unused motifs | dynamic state /raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D_freq** | 0.5955 | 0.5909 | −0.0046 | 0.320 | 0.212 | 506 | 1 | 0 | 0.888 |
| **D_task** | **0.5400** | **0.5391** | −0.0010 | 0.490 | 0.256 | 266 | 0 | 0 | 1.317 |
| **D_TM** | 0.5813 | 0.5911 | **+0.0098** | 0.559 | 0.232 | 140 | 6 | 2 | 1.392 |

Cheap-probe read: **task-only is best and generalises; D_TM ≈ D_freq on monitor
with a small positive (overfit-leaning) discovery gap; D_TM's dictionary is the
least reused and the least compressive.** The cheap probe alone would predict
"no useful signal" (Δ<0.02). It does **not** predict the Stage-B outcome — the
decisive difference appears only under the strong reader.

`D_TM` MDL-eligibility stats at full scale: `f_graph≥45` candidate count 36–183
per step; compressive (`ΔL<0`) candidates 0–142 per step; **only 3 / 64 steps**
had no `ΔL<0` candidate (forced MDL-best fallback). So the task selector did have
a real (≤32) pool to choose from — the constraint was not vacuous.

### Dictionary similarity (F vs TM), §15

| quantity | value |
|---|---:|
| identical positional merge rules | 0 / 64 |
| shared rule keys | 7 |
| identical final motif types | 7 of 64 each |
| Jaccard(motif type sets) | **0.058** |
| freq-only / tm-only types | 57 / 57 |

Top task-scored motifs (by `S_task`) include size-2 `C#C`-like triple-bond motifs
(task 0.17, support 8254) and several low-support size-4 subgraphs with large MDL
gain; top-frequency motifs are the usual high-count carbon skeletons. No
chemistry labels are assigned. The two dictionaries are **almost disjoint**.

## 2. Stage B — decisive strong-reader test (§16–21)

Identical architecture / hidden size / depth / optimizer / budget / checkpoint
rule / seed / data for both arms; **only the frozen dictionary differs**.

| reader | dictionary | best valid MAE | best epoch | Top-5 soup | train @ best | train−valid gap | epochs run | params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **B-FREQ** | `D_freq` | 0.27078 | 113 | 0.25331 | 0.13184 | 0.1389 | 153 (early stop) | 192 257 |
| **B-TM** | `D_TM` | **0.20374** | 230 | **0.19517** | 0.08517 | **0.1186** | 240 | 192 257 |

$$
\Delta_{\rm dictionary}^{\rm best} = 0.0670,\qquad
\Delta_{\rm dictionary}^{\rm soup} = 0.0581 .
$$

**Matched-budget / matched-epoch checks** (FREQ early-stopped at 153, TM ran 240):

| comparison | FREQ | TM | Δ |
|---|---:|---:|---:|
| best valid up to epoch 153 (matched budget) | 0.2708 | 0.2185 | +0.0523 |
| epoch 100 | 0.3114 | 0.2555 | +0.0559 |
| epoch 150 | 0.2864 | 0.2453 | +0.0411 |

The gap is present from the first 50 epochs, is stable across the whole budget,
and **is not an artefact of B-TM running more epochs**. `B-TM` fits train *and*
valid better than `B-FREQ`, and its train−valid gap is **smaller** — a genuine
improvement, not a variance/overfit effect.

Raw reference (same regime, PSCD-SC-v0): strong raw `zinc-b-full` ≈ 0.12–0.14.
TM narrows the PSCD/raw gap from ≈0.13 (freq) to ≈0.07 — roughly halved, not
closed.

## 3. Gates (§21)

* **Strong positive** — `Δ ≥ 0.04` ✓ (0.067 best / 0.058 soup / 0.052 matched);
  `MAE_TM ≤ 0.22` ✓ (0.204 / 0.195); train−valid gap smaller ✓ (0.119 < 0.139).
* **Reuse** — *partially met*: `D_TM` median support 140 (vs 506), 6 low-support
  motifs and 2 unused (vs 1 / 0). Reuse is weaker, not collapsed.
* **MDL/compression** — *partially met*: MDL code ratio 0.232 vs 0.212 (near),
  but **dynamic-state ratio 1.392 vs 0.888** — the TM reader processes ≈1.6× the
  nodes of the frequency reader (no dynamic compression vs raw).

⇒ the pre-registered **prediction** gate fires decisively; the
**reuse/compression** conditions are only partially met. This is the central
caveat and the reason the next step is a **compression-matched control**, not
formalisation yet.

## 4. Compression confound (explicit)

`D_TM` `occ/atom = 0.555`, `(occ+ports)/atom = 1.392`; `D_freq` `0.315` / `0.888`.
The task-guided dictionary keeps **more** structure, so part of the 0.067 gain
could be "less abstraction / more retained information" rather than
"task-relevant abstraction". The cheap linear probe does **not** track this
monotonically (`D_TM` has far more tokens than `D_freq` yet identical cheap-probe
MAE), which argues the strong-reader gain is not *purely* a token-count effect —
but it cannot rule the confound out. A frequency dictionary with a matched
occurrence/dynamic-state budget is required to separate the two.

## 5. Answers to the eight questions (§28)

* **Q1 — are the frequency and task+MDL motifs clearly different?** **Yes, almost
  disjoint**: Jaccard 0.058, 7/64 shared types, 0 identical positional rules.
* **Q2 — does task-only show rare / memorisation motifs?** **No**: 0 low-support
  (<45) motifs, 0 unused, median support 266 — but it is the **least compressive**
  (motif/atom 0.490, port/size 0.895). Its cheap-probe monitor MAE is the best
  (0.539).
* **Q3 — does the MDL constraint suppress degeneration?** **Only partially /
  opposite**: `D_TM` has *more* low-support motifs (6) and unused motifs (2) than
  `D_freq` (1, 0) — because an early high-support motif can lose support as later
  merges absorb it — although `D_TM` compresses better than task-only
  (MDL ratio 0.232 vs 0.256).
* **Q4 — does the discovery residual gain generalise to internal-monitor?** On the
  cheap probe, **only partially**: `D_TM` discovery 0.5813 vs monitor 0.5911
  (gap +0.0098, overfit-leaning), whereas `D_freq` has a −0.0046 gap. The strong
  reader, by contrast, shows a *smaller* train−valid gap for TM.
* **Q5 — under the identical strong reader, is `MAE_TM < MAE_FREQ`?** **Yes,
  decisively** (0.204 vs 0.271 best; 0.195 vs 0.253 soup).
* **Q6 — is the improvement from the dictionary, not params/reader?** **Yes by
  construction**: reader class, hidden size, depth, optimizer, budget, seed and
  data are identical (192 257 params each); only the frozen dictionary differs.
* **Q7 — does the task-driven dictionary narrow the PSCD strong gap?** **Yes**:
  PSCD/raw gap ≈0.13 → ≈0.07 (approximately halved), still `>0.04` from raw.
* **Q8 — enough to formalise task-driven dictionary learning?** **Not yet**:
  the signal is strong and real, but the compression/reuse confound must be
  removed by a matched-compression control before formalising an
  alternating/end-to-end objective.

## 6. Final conclusion and next decision

**First line (frozen choice):**

> **Task supervision materially improves the reusable compositional dictionary.**

— recorded **with the mandatory qualifier** that the task+MDL dictionary is also
materially less compressive (state 1.39× vs 0.89× raw) and less reused
(median support 140 vs 506), so "reusable/compressive" is only partially
supported and the gain is entangled with reduced compression.

**Next decision (frozen choice):**

> **`strengthen dictionary regularization before further task coupling`**

The required follow-up is a **compression-matched control**: constrain the
task+MDL discovery so its occurrence / dynamic-state budget matches
`D_freq` (or add an explicit reuse/graph-support penalty to the MDL objective),
then re-run the identical strong reader. If the task-selection gain survives at
matched compression, formalise task-driven dictionary learning; if it vanishes,
the effect was compression, not task relevance.

## 7. Caveats

* `B-FREQ` early-stopped at epoch 153 (patience 40) while `B-TM` ran the full 240
  — the **protocol is identical**; the length differs only because TM's valid
  curve kept improving. Matched-budget and matched-epoch comparisons confirm the
  gap (`+0.052` at epoch 153, `+0.041` at epoch 150).
* Single seed (0). `Δ = 0.067 > 0.04`, so the pre-registration authorises seed-0
  direction; a paired seed-1 confirmation is deferred to the compression-matched
  follow-up.
* The eligibility MDL proxy is a **two-part uniform-symbol code**, not the
  PSCD-v0 empirical-entropy `mdl_proxy` (which cannot express a positive merge
  gain; see preregistration §6a). The corpus-level proxy is still reported as the
  compression measure.
* Official test never loaded; results are valid-side only.

## 8. Artifacts

`results/pscd_tmdl/`: `audit_{freq,task,tmdl}.json`, `rounds_{freq,task,tmdl}.json`,
`dictionary_similarity.json`, `SUMMARY_A.json`, `summary_B{FREQ,TM}.json`,
`history_B{FREQ,TM}.json`, `stageB_valid_curves.png`, `dictionary_*.pkl`,
`B{FREQ,TM}_top5_soup.pt`.
