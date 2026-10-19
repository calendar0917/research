# Post-hoc exploratory — frozen-mode error under C/G weights

**Does restoring the uniform (P3-Null) context mapping under the already-trained
C/G weights reduce raw-y MAE, and does it recover P3?**

| field | value |
|---|---|
| protocol_id | `zinc-bond-conditioned-context-attention-v1` |
| posthoc_id | `posthoc-frozen-mode-error-v1` |
| status | **POST_HOC_EXPLORATORY** — proposed after the original valid scores were seen |
| nature | same-weight intervention diagnostic; **not** a new model, **not** independent confirmation, **not** a blind test |
| inputs | frozen mechanism predictions, original formal valid predictions, cached valid labels |
| computation | CPU only; no training, no forward pass, no GPU, no official test |
| fixed endpoints | 6 (A–F); no interpolation, no lambda/threshold/checkpoint/seed search |
| original valid read | `valid_read.lock` unchanged: `read_count = 1`, `reread = false` |
| this post-hoc access | additional read of the **same cached valid labels**; recorded in `valid_label_access.json` |

**Gain definition used everywhere:**
`gain = MAE(reference) - MAE(candidate)`; **positive means the candidate is
better** (lower error).

Endpoint letters A–F are analysis endpoints, not training arms. In particular,
endpoint **F** is *G-trained weights × uniform mode*, not the F/FFN arm.

---

## 0. Answer first

**No. Restoring uniform does not improve; it is much worse. It does not recover
P3 either.**

* **D vs B** (C weights, uniform vs C mode): mean gain **−0.042248**
  (CI95 [−0.064454, −0.029009]); both seeds negative
  (−0.048105 / −0.036392). Uniform is worse on 73.3% of molecules.
* **F vs E** (G weights, uniform vs G mode): mean gain **−0.044327**
  (CI95 [−0.066258, −0.030631]); both seeds negative
  (−0.051005 / −0.037649). Uniform is worse on 71.6% of molecules.
* **D vs A** (C weights uniform vs P3): mean gain **−0.049700**
  (CI95 [−0.076563, −0.033819]).
* **F vs A** (G weights uniform vs P3): mean gain **−0.050369**
  (CI95 [−0.077337, −0.034448]).
* Uniform MAE is 0.129544 (C weights) / 0.130213 (G weights), versus P3
  0.079844. That is about **1.62–1.63× P3 MAE**, not a recovery.
* The same-weight G-mode under C weights is also slightly worse than C mode
  (**C vs B** = −0.000610, CI95 [−0.001082, −0.000139]) and still worse than
  P3 (**C vs A** = −0.008063, CI95 [−0.014316, −0.003035]).

**Reading.** The trained C/G networks are strongly co-adapted to their learned
context. Forcing the uniform context at inference is an off-training-distribution
intervention and destroys error. This is evidence *against* “just shrink the
attention at inference” as a rescue path, and it provides **no positive evidence**
for a residual/gating training follow-up. The original negative result is
strengthened, not rescued.

**Boundary.** This is post-hoc, two training instances, same frozen weights.
The CIs describe molecule-level sample fluctuation given these two instances;
they are **not** a confidence statement over training randomness. Mode switching
is an intervention diagnostic, not a chemical causal explanation.

---

## 1. Fixed endpoints

| endpoint | weights | context mode | meaning |
|---|---|---|---|
| **A** | P3 | P3 Null | reference (original P3) |
| **B** | C-trained | C | original formal C |
| **C** | C-trained | G | same C weights, marginalised G mode |
| **D** | C-trained | uniform | same C weights, restored P3-Null mapping |
| **E** | G-trained | G | original formal G |
| **F** | G-trained | uniform | same G weights, restored P3-Null mapping |

By code, the `uniform` branch computes the same W/V mapping as the frozen P3
Null grouping (up to float32 reduction order). D/F are therefore the
same-weights “restore uniform” interventions: the junction embeddings and all
other weights remain C/G-trained; only the W/V reduction is switched to uniform.

---

## 2. Main table — raw-y MAE (single models, no prediction ensemble)

| endpoint | weights · mode | seed 0 | seed 1 | **seed mean** |
|---|---|---:|---:|---:|
| A | P3 · Null | 0.079101498 | 0.080586400 | **0.079843949** |
| B | Cw · C | 0.087476989 | 0.087115184 | **0.087296086** |
| C | Cw · G | 0.087504923 | 0.088308170 | **0.087906547** |
| D | Cw · uniform | 0.135581546 | 0.123506728 | **0.129544137** |
| E | Gw · G | 0.085182374 | 0.086589861 | **0.085886118** |
| F | Gw · uniform | 0.136187171 | 0.124238511 | **0.130212841** |

Seed aggregation is per-seed MAE first, then the arithmetic mean of the two
seed MAEs. No predictions were averaged.

---

## 3. Fixed comparisons

All CIs are molecule-level paired bootstrap, 5,000 resamples, seed 20261019,
same resampling index for both seeds (the per-molecule delta is averaged across
seeds before resampling). “Better fraction” is the fraction of molecules on
which the candidate has lower absolute error.

| comparison | seed 0 gain | seed 1 gain | **mean gain** | 95% CI | better fraction |
|---|---:|---:|---:|---|---:|
| **D vs B** (Cw uniform vs C mode) | −0.048104557 | −0.036391544 | **−0.042248050** | [−0.064454427, −0.029009097] | 0.267 |
| **D vs A** (Cw uniform vs P3) | −0.056480048 | −0.042920328 | **−0.049700188** | [−0.076562738, −0.033819361] | 0.251 |
| **F vs E** (Gw uniform vs G mode) | −0.051004797 | −0.037648649 | **−0.044326723** | [−0.066257927, −0.030631317] | 0.284 |
| **F vs A** (Gw uniform vs P3) | −0.057085673 | −0.043652111 | **−0.050368892** | [−0.077337499, −0.034447585] | 0.284 |
| **C vs B** (Cw G mode vs C mode) | −0.000027934 | −0.001192986 | **−0.000610460** | [−0.001082290, −0.000139085] | 0.442 |
| **C vs A** (Cw G mode vs P3) | −0.008403425 | −0.007721771 | **−0.008062598** | [−0.014315552, −0.003035069] | 0.433 |

Every fixed comparison is negative on both seeds and has a CI entirely below
zero. The two uniform interventions lose 0.042–0.050 MAE; the same-weight
marginalisation is nearly neutral but slightly negative; the C-weight G-mode
still loses 0.008 MAE against P3.

---

## 4. Robustness — only the original fixed groups

No label-driven slicing and no new masks were introduced. `ordinary` is
`y > −3`; `without row 172` is the original extreme-row exclusion; the
`drop P3-worst-10` mask is the original selection
`error_rank(P3, y)[:10]`, fixed from the original P3 prediction and labels
(rows 172, 917, 753, 659, 260, 403, 447, 224, 31, 267). It was not reselected
from the new predictions.

### 4.1 Seed-mean MAE by group

| group | n | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1000 | 0.079843949 | 0.087296086 | 0.087906547 | 0.129544137 | 0.085886118 | 0.130212841 |
| ordinary (y > −3) | 935 | 0.068089790 | 0.072427423 | 0.073115393 | 0.103315769 | 0.070832851 | 0.103927969 |
| drop P3-worst-10 (original mask) | 990 | 0.065935273 | 0.070606533 | 0.071308771 | 0.102504897 | 0.069464122 | 0.103105915 |
| without row 172 | 999 | 0.074688736 | 0.079845970 | 0.080491726 | 0.112457614 | 0.078386893 | 0.113295770 |

### 4.2 Mean gains by group

| group | D vs B | D vs A | F vs E | F vs A | C vs B | C vs A |
|---|---:|---:|---:|---:|---:|---:|
| all | −0.042248050 | −0.049700188 | −0.044326723 | −0.050368892 | −0.000610460 | −0.008062598 |
| ordinary | −0.030888346 | −0.035225979 | −0.033095118 | −0.035838180 | −0.000687970 | −0.005025604 |
| drop P3-worst-10 | −0.031898364 | −0.036569624 | −0.033641793 | −0.037170642 | −0.000702239 | −0.005373498 |
| without row 172 | −0.032611643 | −0.037768877 | −0.034908877 | −0.038607034 | −0.000645756 | −0.005802990 |

The sign pattern is identical in every fixed group: uniform restoration is much
worse, and it does not approach P3. The original drop-10 C/G/F−P3 gains were
reproduced exactly with the recovered mask, which confirms the mask matches the
original selection.

---

## 5. Interpretation and boundaries

**What is supported.** Under these trained weights, restoring uniform increases
MAE substantially on both seeds and in every fixed robustness group. The
same-weight G marginalisation is also not a rescue. The most parsimonious
reading is **co-adaptation**: the C/G encoders, composer and reader adapted to
the learned context, so a forced return to the Null context is far off their
training distribution. The result supports the interpretation rule “if uniform
is worse, attention and the rest of the network are jointly adapted; simply
shrinking attention does not solve the problem.”

**What is not supported.** This does **not** show that the learned reweighting
is beneficial, and it does not show that the original C/G gap is caused by the
reweighting operation alone. It also does not give positive evidence for a
residual/gating training follow-up. The original REPORT’s causal phrasing
(“the loss comes from reweighting itself”) should be narrowed; see
`ERRATUM_SUGGESTIONS.md`.

**Boundaries.**

* **Post-hoc:** the analysis was proposed after the original official-valid
  scores were seen. It is exploratory and cannot serve as independent
  confirmation of any new model.
* **Two training instances:** seed 0 and seed 1 are the only instances. The
  bootstrap CIs describe molecule-level sample variation conditional on these
  two trained models; they do not cover training-seed randomness.
* **Same-weight intervention:** switching the context mode at inference can be
  off the training distribution. The result is an intervention diagnostic, not
  a chemical causal claim and not a claim about what would happen if the model
  were trained with uniform context from the start.
* **Label access:** the original frozen record remains `read_count = 1`,
  `reread = false`; this post-hoc analysis additionally read the **same cached
  valid labels**. The track must not be described as having only one
  valid-label use in total. See `valid_label_access.json`.
* **No leakage / no compute:** official test was never loaded; no GPU, no
  training, no inference, no threshold/lambda/checkpoint/seed search. No frozen
  artefact was rewritten.

---

## 6. Recommendation

**No reason to continue this replacement-attention branch.** The same-weight
uniform intervention is much worse than both original modes and P3, and the
same-weight G marginalisation does not recover P3 either. There is no positive
signal here to justify another attention variant, and “try more parameters” is
not a recommendation of this analysis.

Do not start new training from this post-hoc. If the track ever wants to test a
zero-initialised, strongly regularised gated correction, that is a **separate
pre-registered hypothesis** with a fresh confirmation endpoint, not a
continuation justified by this result. The present evidence points to stopping
the branch and keeping the frozen P3 Null reference.

---

## 7. Input verification and artefacts

All 20 frozen prediction arrays used here (P3 ×2, official C/G/F ×6, mechanism
C/G mode cells ×12) passed file SHA256 + array SHA256 verification; every
`pred` is length 1000 and finite, and every `row` is exactly `0..999`, matching
the original label/P3 order. The C/G/F official and mechanism `.npz` arrays are
not committed in the result directory (round convention); they were read from
the verified `.rr/pulled` copies of the frozen compute-host outputs. The P3
arrays came from their historical frozen paths. All hashes were checked against
the frozen records before scoring.

Key checks:

| check | result |
|---|---|
| valid label array SHA256 (float32) | `72a9cfc3…` matches `valid_metrics` and `valid_read.lock` |
| valid label array SHA256 (float64) | `8f6ee636…` matches `valid_metrics` and `valid_read.lock` |
| P3 per-seed MAE vs original score | 0.07910149778419873 / 0.08058639967715135, diff **0.0** |
| C weights, C mode vs official C | max abs diff **1.907e−6** (s0) / **2.861e−6** (s1), ≤ 1e−5 |
| G weights, G mode vs official G | max abs diff **2.861e−6** (s0) / **3.815e−6** (s1), ≤ 1e−5 |
| original six contrasts vs `valid_metrics` | max abs diff **0.0** |
| masks `all` / `ordinary` / `without 172` | row hashes match `valid_metrics` |
| drop-P3-worst-10 | original rows above; original C/G/F−P3 drop-10 gains reproduced exactly |
| `valid_read.lock` | unchanged `read_count = 1`, `reread = false` |
| official test | never loaded |

Artefacts in this directory:

```
REPORT.md                 this report
metrics.json              machine-readable endpoints, comparisons, robustness, checks
input_manifest.json       input paths, file/array hashes, row/finite checks
valid_label_access.json   explicit record of the additional post-hoc label access
ERRATUM_SUGGESTIONS.md    documentation-only corrections; frozen files untouched
analyze.py                CPU-only, read-only verification + scoring script
```

Reproduce with:

```bash
uv run python tracks/structural-sufficiency/results/zinc_bond_conditioned_context_attention_v1/posthoc_frozen_mode_error_v1/analyze.py --check-only
```

The script imports no model code and never constructs a model. It refuses to
continue if any frozen array is missing or any hash/row/finiteness check fails;
it does not fall back to retraining or re-inference.

---

## 8. Bottom line

Restoring the uniform context under the trained C/G weights **does not improve
MAE** — it adds about 0.042–0.044 MAE against the model’s own mode and about
0.050 MAE against P3 — so it does **not recover P3**. The result is consistent
with strong co-adaptation between the learned attention context and the rest of
the network, and inconsistent with “inference-time uniform restoration” as a
fix. The original negative result stands; this attention branch should not be
continued.
