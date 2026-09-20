# PSCD-SC-v0 — Strong-Capacity Port-Operator Reader Audit (analysis)

**Question.** PSCD-R-v0 localised the historical PSCD-v0 failure to the *reader*
(a batching defect + an opaque, port-collapsed reader) at a **tiny** budget
(`d=32, L=2, ≤6000` steps), where the frozen code was read at/ below the raw atom
graph but never left the ~0.32 band. This round freezes the dictionary and asks
the **converged-capacity** question: with a **strong** port-operator reader
(`d=64`, `L_outer=4`, alternating inter-/intra-motif operator, full canonical
budget), does the fixed PSCD representation follow the raw graph down into a
`0.1–0.2` MAE band?

**Verdict (first line).**

> **The fixed frequency dictionary imposes a real predictive abstraction ceiling;
> motif discovery must become task-coupled.**

**Next decision.**

> **`replace frequency motif discovery with task+MDL discovery`** — and, as a
> required reporting correction, **stop claiming computational compression** of
> the frozen operator (`r_state = 0.89`, message count `0.90×` raw).

---

## 0. Provenance

| item | value |
|---|---|
| round | PSCD-SC-v0 (`notes/pscd_sc_preregistration.md`) |
| code | `tracks/ksvd/code/run_pscd_strong_reader.py` |
| tests | `tracks/ksvd/tests/test_pscd_strong_reader.py` (5 pass) + in-run data-free gate |
| frozen artifact | `results/pscd_compositional/stageA.pkl` (PSCD-v0, md5 `b180e9695fb4f0a2cb366338bd58f459`) |
| commit | `34bd87a` (Arm C / Arm B), `f167269` (compression fix) |
| execution | remote A100 host `res` (Arm A GPU 1, Arm C GPU 0, Arm B/metrics CPU) |
| splits | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000**; official `test` never loaded |
| seed | 0 |
| results | `tracks/ksvd/results/pscd_strong_reader/` (git-ignored) |

**Execution note (material caveat).** Both training runs were **stopped early at
the user's explicit request**: Arm A at epoch 140/240 (patience 40) and Arm C at
epoch 170/240. Neither reached its canonical stop, so neither materialised a
Top-5 soup or a saved `history_*.json`; the numbers below are the **best-valid
checkpoint** read from the (fully captured) run logs `arm_a_run.log` /
`arm_c_run.log`. This is called out wherever it matters.

## 1. Freeze gate + data-free tests (all pass, before any training)

| gate | result |
|---|---|
| `Decode(D, C_G) == G` | train **10 000/10 000**, valid **1 000/1 000** (exact colored-canonical key) |
| A batch invariance | max|Δ| `1.3e-7` (individually vs batched) — no repeat of the PSCD-R collate defect |
| B molecule permutation invariance | max|Δ| `1.2e-7` (random atom relabel + refrozen code) |
| C motif automorphism invariance | max|Δ| `1.1e-8` (8 same-orbit port pairs) |
| D empty / single / multi-port | finite |
| E port-routing sensitivity | max|Δ| `0.0168` > 0 (not a port mean) |

No vocabulary / merge rule / partition / port / composition / decoder changed.

## 2. Arm A — canonical strong raw (B-Full), execution regime

Re-run of the repo's canonical strong architecture **`zinc-b-full`** (shared
structural patch encoder, **84 495** params; reused builder/config/protocol:
Adam lr 1e-3 / wd 1e-5 / batch 128 / clip 5 / L1 / max_epochs 240 / patience 40,
cached patch graphs), seed 0, GPU 1.

| source | best valid MAE |
|---|---:|
| Arm A fresh run, best checkpoint (**epoch 119**, truncated run) | **0.1364** |
| Arm A fresh run valid at epoch 140 (train 0.110) | 0.1384 |
| `zinc-b-full` historical canonical 2-seed Top-5 soup | 0.118972 |

So the execution-regime strong **raw ceiling is ≈ 0.12–0.14**. Arm A is well
below the `> 0.16` baseline-reproduction stop rule ⇒ **baseline reproduction
OK; performance interpretation is admissible.**

## 3. Arm B — decode oracle (same Arm-A model + weights)

The **same** canonical B-Full model and weights were fed `G` and `Decode(C_G)`
(decoded graph processed through the *same* record + encoding + structural
pipeline, not a clone), on a deterministic **CPU** reference, for **512 train +
512 valid** graphs:

| split | max&#124;Δŷ&#124; | mean | p95 | frac < 1e-5 |
|---|---:|---:|---:|---:|
| valid | **3.58e-6** | 2.96e-7 | 9.54e-7 | **1.00** |
| train | **3.58e-6** | 3.15e-7 | 9.54e-7 | **1.00** |

`max|Δŷ| < 1e-5` on both splits ⇒ **the PSCD artifact hides no data loss**: the
lossless isomorphism is not just combinatorial, it is computationally invisible
to a strong model. No decode-oracle model was trained.

## 4. Arm C — strong compressed port-operator reader (primary)

Frozen `(D, C_G)` → **StrongPortOperator** (`d=64`, `L_outer=4`, `L_motif=2`,
`L_intra=2`), canonical protocol, seed 0, GPU 0, **truncated at epoch 170/240**.

### 4.1 Learning curve (best-valid checkpoint)

| epoch | train MAE | valid MAE | best valid |
|---:|---:|---:|---:|
| 1 | 1.071 | 0.864 | 0.864 |
| 30 | 0.306 | 0.361 | 0.356 |
| 60 | 0.196 | 0.305 | 0.301 |
| 90 | 0.153 | 0.298 | 0.278 |
| 120 | 0.126 | 0.278 | 0.270 |
| 150 | 0.104 | 0.273 | 0.268 |
| 170 | **0.096** | 0.263 | **0.263** |

The valid curve **plateaus** (~2–5×10⁻⁴ / epoch over epochs 130–170) while the
train curve keeps falling → the remaining budget is **memorisation, not
generalisation**. Extrapolating the plateau to the full 240-epoch budget lands
near ~0.24–0.25, i.e. still **> 0.22** with a **gap > 0.08**.

### 4.2 Arm C vs Arm A

| reader | params | best valid | train @ best |
|---|---:|---:|---:|
| Arm A raw (B-Full), truncated | 84 495 | **0.136** | 0.110 |
| Arm C frozen PSCD port-op, truncated | 192 257 | **0.263** | 0.096 |

**Arm C train MAE is *lower* than Arm A train MAE (0.096 vs 0.110) while Arm C
valid is 0.127 *worse*.** The strong PSCD reader fits the training set even
better than raw and still fails to transfer — a *representation-induced
generalisation/abstraction* bottleneck, not a capacity or optimisation shortfall.

## 5. Compression and dynamic compute (*measured*)

Train (10 000 graphs) / valid (1 000):

| quantity | train | valid | ratio to raw |
|---|---:|---:|---:|
| `N_raw = |V_G|` | 231 664 | 23 083 | 1.00 |
| occurrences `|O_G|` | 73 028 | 7 316 | **0.315** |
| active ports `Σ_o |P_o^active|` | 132 671 | 13 338 | 0.573 |
| `N_PSCD = |O_G| + Σ|P_o|` | 205 699 | 20 654 | **0.888** |
| inter-motif messages | 164 096 | 16 482 | 0.329 |
| intra-motif port-pair messages | 283 877 | 28 560 | 0.569 |
| raw edge messages | 498 558 | 49 692 | 1.00 |
| PSCD messages (inter+intra) | 447 973 | 45 042 | **0.899** |
| wall time / epoch | ~4.65 s | | ~2 150 graphs/s |

The headline **`M_G/n_G = 0.315` occurrence ratio is *not* the dynamic-state
ratio**: once the *port* states required for port-resolved computation are
counted, `N_PSCD/N_raw = 0.888`, and the message count is `0.90×` the raw edge
message count. **The frozen operator is barely compressed and is *not* cheaper in
dynamic compute than the raw graph.** Peak GPU memory was not captured (the run
was stopped before it could be reported); both models are tiny (< 1 GB) and never
OOM. This is Gate E territory and is reported as a co-finding.

## 6. Parameter discipline (§15) — flagged

| bucket | params |
|---|---:|
| motif structural encoder (shared, 2 layers) | 34 816 |
| inter-motif MLPs (4 layers) | 66 048 |
| intra-motif MLPs (4 layers) | 49 664 |
| transfer descriptor ψ (once) | 16 512 |
| shared state maps `W_C,W_S,W_G,W_P,W_D` | 20 800 |
| bond lookup + head | 4 417 |
| **total** | **192 257** |

`P_PSCD / P_raw = 2.28×` — above the `1.5×` ideal and above the `2×` design
check. The overshoot is **structural** (the brief freezes `d=64`, `L_outer=4`, and
the `Linear→SiLU→Linear` message MLPs); it is **reported, not corrected by
hidden-size tuning**. The overshoot *favours PSCD*, so the Arm C failure is
robust to it.

## 7. Scaling curve (tiny → strong)

`results/pscd_strong_reader/scaling_curve.png` (best valid MAE vs reader
capacity; B-Full historical soup as a dotted reference).

| regime | raw | frozen PSCD | gap (PSCD − raw) |
|---|---:|---:|---:|
| tiny (`d=32, L=2`) | 0.3363 | 0.3246 | **−0.012** (PSCD slightly *better*) |
| strong (`d=64, L=4`) | 0.1364 | 0.2630 | **+0.127** |

Raw improves by **0.20** MAE with capacity; frozen PSCD improves by only **0.06**
and **does not follow the raw curve**. The gap *grows* by an order of magnitude.

## 8. Gate evaluation (frozen §23–27)

* **Gate A (strong success)** — fails (`0.263 ≫ 0.16`).
* **Gate B (intermediate)** — fails (`0.263 > 0.22`).
* **Gate D (optimisation failure)** — the run was truncated, but valid had
  **plateaued** on the frozen representation while train kept falling, so the
  residual budget buys memorisation; this is **not** interpreted as pure
  undertraining. A strict "must reach the exact canonical stop" reading leaves a
  small optimisation caveat.
* **Gate C (abstraction ceiling)** — **fires**: raw ~0.12–0.14, PSCD `0.263 >
  0.22`, gap `0.127 > 0.08`, and the reader fits train *better* than raw while
  generalising far worse ⇒ a real representation/task abstraction bottleneck.
* **Gate E (compression failure)** — **also fires**: `r_state = 0.89`,
  messages `0.90×` raw.

Seed-1 confirmation is **not** triggered (gap `0.127 > 0.08`; the brief forbids
extra seeds merely to confirm an obvious failure).

## 9. Answers to the eight questions

* **Q1 — canonical strong raw in this regime?** `≈0.12–0.14` (fresh Arm A best
  `0.136` @epoch 119, truncated; historical canonical B-Full soup `0.119`).
* **Q2 — does Decode keep the same strong prediction?** **Yes**: same Arm-A
  model+weights, `max|Δŷ| = 3.6e-6` on 512+512 graphs, 100 % `< 1e-5`.
* **Q3 — fixed-PSCD strong-reader ceiling?** **≈0.26** (best valid `0.263`,
  truncated at epoch 170 with a plateauing valid curve; an upper bound on the
  converged value, likely ~0.24–0.25).
* **Q4 — do raw and PSCD improve together tiny→strong?** Both improve, but
  asymmetrically: raw `0.336→0.136` (−0.20), PSCD `0.325→0.263` (−0.06).
* **Q5 — PSCD/raw gap with capacity?** **Grows**: `−0.012` (tiny, PSCD slightly
  better) → `+0.127` (strong).
* **Q6 — is dynamic intra-motif transfer necessary?** **Not answered**: the
  ablation gate (Arm C valid `≤ 0.20`) did not fire, so the no-transfer ablation
  was **not run**.
* **Q7 — are PSCD dynamic state / message counts smaller?** **Barely**: dynamic
  state `0.888×` raw, messages `0.899×` raw. Occurrence count is `0.315×`, but
  the required port states erase the saving.
* **Q8 — enough evidence for task-coupled dictionary learning?** **Yes**, but
  the evidence points at the *frequency objective*, not merely at a reader: the
  frozen frequency dictionary is a genuine bottleneck (predictive **and**
  compression), so the next step is **task + MDL motif discovery**.

## 10. Caveats

* **Truncated runs** (user-requested stop): Arm A epoch 140/240, Arm C epoch
  170/240; no Top-5 soup, no saved histories, no peak-memory row, single seed.
  The valid curves had effectively plateaued, so the direction is robust, but the
  numbers are **upper bounds** on the converged PSCD ceiling.
* Arm B used the **historical canonical B-Full seed-0 state** (`sspe_seed0`),
  because the fresh Arm A run was stopped before saving a `selection_state.pt`;
  the model class/weights are the canonical Arm-A instance.
* `P_PSCD = 2.28 × P_raw` (favours PSCD; the failure is therefore not a
  parameter-starvation artifact).
* Comparison is same-host, same-protocol, seed 0; raw is a regime reference, not
  a benchmark claim.

## 11. Final conclusion and next decision

**Final first line (frozen choice):**

> **The fixed frequency dictionary imposes a real predictive abstraction ceiling;
> motif discovery must become task-coupled.**

**Next decision (frozen choice):**

> **`replace frequency motif discovery with task+MDL discovery`**

with the mandatory co-statement:

> the frozen port operator is **not** a demonstrated computational compression
> (`N_PSCD/N_raw = 0.89`, message count `0.90×` raw); do not claim the historical
> ~3.2× occurrence-count compression as dynamic compression.

Do **not** keep strengthening the port-operator reader, and do **not** yet
proceed to task-coupled dictionary learning on top of the *frequency* dictionary
— the frozen frequency objective itself must be coupled to the task (and MDL).

### Revisit if

A **fully converged** (exact canonical stop, ≥240 epochs) strong port-operator
run with Top-5 soup lands `≤ 0.22` and closes the raw gap to `≤ 0.08`, while the
dynamic-state ratio stays `≳ 0.85`; or if a reader that exposes **atom-resolved**
intra-motif states (not only canonical ports) collapses the gap without
exceeding `N_raw` dynamic states. Either would move the verdict toward Gate B and
re-open "one more port-operator mechanism revision".
