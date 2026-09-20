# Pre-registration — PSCD-RM-v0: Rate-Matched Task-Driven Dictionary Audit

Round name: **PSCD-RM-v0** (Rate-Matched). Written **before** the formal remote
run, following the `remote-research-runner` skill (local code → remote A100 →
local analysis). It does not modify any historical record.

Official ZINC `test` is **never** loaded, instantiated or referenced. `y` is used
only by the internal Ridge selector and the Stage-B readers. Splits are the
repository's canonical PyG ZINC `subset=True`: **train 10 000 / valid 1 000**.

---

## 0. The single question

PSCD-TMDL-v0 changed **only** the dictionary-selection rule (frequency →
task+MDL) and improved the identical strong reader by
`Δ_dictionary = MAE_FREQ - MAE_TM ≈ 0.058–0.067`, but the task+MDL dictionary
was also materially **less compressive** (dynamic-state ratio `1.392` vs `0.888`
raw; occurrence/atom `0.555` vs `0.315`). This round removes that confound:

> **At equal representation rate, does task supervision still learn a better
> graph dictionary?**

$$
\Delta_{\text{task}}^{\text{low-rate}} = MAE_{\text{FREQ}} - MAE_{\text{TM-RATE}}.
$$

## 1. What this round does NOT re-litigate / forbids

* No change to reader architecture (PSCD-SC-v0 `StrongPortOperator`, 192 257
  params reused verbatim), port definition, composition representation,
  vocabulary budget, max motif size, target preprocessing, optimizer protocol or
  official valid/test discipline.
* No `β` / `K` / max-size sweep; no hand-coded chemistry; no overlapping motifs;
  no Gumbel / differentiable tokenizer; no new target descriptors; no extra
  strong-reader variants.
* Official test never loaded. Official valid never used to choose a dictionary
  merge or select a candidate.
* Two new arms only: **FREQ-LOOSE** (decompression control) and **TM-RATE**
  (rate-matched task dictionary). The original FREQ and TM dictionaries are
  **reused** from PSCD-TMDL-v0, not re-run.

## 2. Data discipline (frozen, identical to PSCD-TMDL-v0)

* Canonical train = 10 000, official valid = 1 000.
* Deterministic split of the canonical train (**seed 0**): **9 000 discovery** +
  **1 000 internal-monitor** (`split_indices.json`, md5 unchanged from
  PSCD-TMDL-v0). No re-randomisation.
* Dictionary learning uses only the 9 000 discovery graphs (structure + labels).
  The 1 000 internal-monitor graphs participate in **no** merge decision; they
  are post-hoc only.
* Official valid is touched only after the dictionary is frozen and the Stage-B
  reader has finished training.

## 3. Frozen inputs (provenance)

* `dictionary_FREQ.pkl` (frequency trajectory, 64 rules) — md5
  `bbb26189d1757c2762d8b28818ee1bb8`.
* `dictionary_TM.pkl` (task+MDL dictionary, 64 rules) — md5
  `8c151be6814d3cf2ae52d5689bb9c302`.
* Split `split_indices.json` md5 `c4a21aa1dc3f874f52995c0480341dee`.
* Code / reader / tests from `run_pscd_strong_reader.py` (PSCD-SC-v0) and
  `run_pscd_tmdl_dictionary.py` (PSCD-TMDL-v0) reused unchanged.

## 4. Rate definition (exact, matches PSCD-SC-v0 `r_state`)

For a frozen dictionary tokenisation of the discovery corpus:

$$
R_{\rm occ}=\frac{\sum_G |O_G|}{\sum_G |V_G|},
\qquad
R_{\rm state}=\frac{\sum_G |O_G| + \sum_G |P_G^{\rm active}|}{\sum_G |V_G|},
$$

where an **active port** is a distinct `(occurrence, canonical slot)` pair
(`precompute_graph`'s `n_port`). Verified against PSCD-TMDL-v0: FREQ-64
discovery `(R_occ, R_state) = (0.3156, 0.8887)`; TM-64 discovery
`(0.5552, 1.3923)`, matching the recorded `0.315 / 0.888` and `0.555 / 1.392`.

## 5. Control H — FREQ-LOOSE (pure decompression control)

For every prefix `k = 0 … 64` of the **frozen frequency trajectory**, retokenise
the 9 000 discovery graphs (incrementally; prefix `k` equals
`tokenize_frozen(sequence[:k])`) and record `R_occ^F(k)`, `R_state^F(k)`.

$$
k^\*=\arg\min_k \left| R_{\rm state}^{F}(k)-R_{\rm state}^{\rm TM}\right| .
$$

State ratio is the primary matching quantity; `R_occ^F(k*)` vs `R_occ^TM` is
reported. Match quality:

* `|R_state^F(k*) − R_state^TM| ≤ 0.03` ⇒ **state-matched**;
* additionally `|R_occ^F(k*) − R_occ^TM| ≤ 0.05` ⇒ **strong rate match**;
* otherwise the nearest prefix is still run and reported as an **approximate
  decompression control** (no interpolation / new tokenisation).

FREQ-LOOSE keeps its **true** dictionary: exactly the first `k*` frequency merge
rules, in order, with the same motifs. No motif is added, removed or reordered.

## 6. Control L — TM-RATE (rate-matched task dictionary)

Fresh task-driven dictionary from singletons with `K = 64` merge steps,
`max motif size = 8`, and the PSCD-TMDL-v0 selector unchanged (exact motif
identity, Ridge residual selector, `f_graph ≥ 45`, `ΔL_MDL < 0`).

### 6a. Per-step true corpus-level rate gain

At step `t` simulate each admissible candidate `c` on exactly the discovery
graphs it touches, and compute the real corpus-level reductions

$$
g_{\rm occ}(c)=\frac{\Delta|O|(c)}{\sum_G|V_G|},\qquad
g_{\rm state}(c)=\frac{\Delta|O|(c)+\Delta|P^{\rm active}|(c)}{\sum_G|V_G|},
$$

with `Δ` the *reduction* (before − after). These are computed exactly (re-tokenise
+ re-code the touched graphs); no motif-size / frequency approximation.

### 6b. Hard rate constraint

With `m = 64 − t + 1` steps remaining and

$$
D_{\rm occ}=\max(0, R_{\rm occ}^{\rm cur}-R_{\rm occ}^{\rm target}),\quad
D_{\rm state}=\max(0, R_{\rm state}^{\rm cur}-R_{\rm state}^{\rm target}),
$$

a candidate is admissible iff

$$
f_{\rm graph}(c)\ge 45,\qquad \Delta L_{\rm MDL}(c)<0,\qquad
g_{\rm occ}(c)\ge \frac{D_{\rm occ}}{m},\qquad
g_{\rm state}(c)\ge \frac{D_{\rm state}}{m}.
$$

**No `β`.** The targets are recomputed on the same 9 000 discovery split:

$$
R_{\rm state}^{\rm target}=R_{\rm state}^{\rm FREQ64,disc}\approx0.889,\qquad
R_{\rm occ}^{\rm target}=R_{\rm occ}^{\rm FREQ64,disc}\approx0.316.
$$

### 6c. Selection

Among admissible candidates, `c* = argmax S_task(c)` with the unchanged
PSCD-TMDL-v0 task score (fixed Ridge `α=1` on discovery-9000, standardised
candidate occurrence feature, `|mean(residual · z̃)|`). Deterministic tie-break
within 1 % relative `S_task`: (1) larger `g_state`, (2) larger MDL gain,
(3) larger graph support, (4) larger occurrence frequency, (5) canonical key.
Eligibility uses only discovery statistics; internal/official-valid MAE never
select a candidate.

### 6d. Infeasibility is a legitimate result

If at some step **no** candidate satisfies all constraints simultaneously, the
construction is declared

> `rate-matched task dictionary construction infeasible under the frozen
> 64-rule budget`,

diagnostics are recorded (`n_support_ok`, `n_mdl_ok`, `n_rate_ok`, best achieved
`g_occ` / `g_state`, `D` / `need`), and the method is **not** relaxed (no support
relaxation, no `β`, no `K`, no extra merges, no size change).

## 7. Stage-A structural audit

For **both** new dictionaries:

* **exactness** — `Decode(D, C_G) ≅ G` on discovery **and** internal-monitor
  (must be 100 %);
* **invariance** — random relabelling of discovery graphs, 0 mismatch;
* **reuse** — median / mean graph support, low-support (`< 45`) motifs, unused
  motifs, occurrence frequency, top-coverage;
* **motif sizes** — occurrence size distribution + type size histogram;
* **ports** — active ports/occurrence, port/size, dynamic-state ratio;
* **compression** — occurrence/atom, `mdl_proxy` ratio;
* **dictionary similarity** — FREQ-LOOSE vs TM-RATE and TM vs TM-RATE
  (shared rule keys, identical final types, Jaccard).

## 8. Stage B — strong-reader test

Entered (a) for FREQ-LOOSE always, and (b) for TM-RATE **only if** it is exact,
invariant, completes 64 steps and passes the rate gate (§9).

* Reader = PSCD-SC-v0 `StrongPortOperator` **unchanged** (192 257 params).
* Identical architecture / hidden size / depth / optimizer / budget / checkpoint
  rule / seed 0 / Top-5 soup rule for both arms; canonical B-Full protocol
  (Adam lr 1e-3, wd 1e-5, batch 128, clip 5, L1, max 240 epochs, patience 40).
* Reader training uses the **full 10 000** canonical train graphs; evaluation on
  the official valid 1 000. Official test never loaded.

## 9. Rate gate for TM-RATE

Required on the discovery split:

$$
R_{\rm state}^{\rm TM-RATE}\le R_{\rm state}^{\rm target}+0.02,\qquad
R_{\rm occ}^{\rm TM-RATE}\le R_{\rm occ}^{\rm target}+0.02.
$$

Absolute differences to target are reported; a difference `> 0.05` precludes a
claim of strict rate matching.

## 10. Gates (frozen)

* **Strong positive** — TM-RATE rate-matched, `Δ_task^{low-rate} ≥ 0.04`,
  `MAE_TM-RATE ≤ 0.22`, reuse not catastrophically collapsed ⇒
  *task supervision learns a better compositional graph dictionary under a
  matched structural representation budget*.
* **Moderate positive** — `0.02 ≤ Δ_task^{low-rate} < 0.04` ⇒ run seed 1 as a
  paired confirmation of FREQ vs TM-RATE; if direction is consistent,
  *task coupling provides a real but moderate matched-rate dictionary gain*.
* **Decompression-dominated failure** — `|Δ_task^{low-rate}| < 0.02` **and**
  `|MAE_FREQ-LOOSE − MAE_TM| < 0.02` ⇒ *the apparent task-driven gain is
  primarily explained by weaker abstraction / higher representation rate*.
* **Mixed result** — `Δ_task^{high-rate} ≥ 0.04` but `Δ_task^{low-rate} < 0.02`
  ⇒ *task-selected boundaries are useful only in a less-compressed regime; the
  current sparse-rate requirement conflicts with predictive structure* (not a
  simple failure).
* **Rate-construction failure** — 64-rule TM-RATE cannot reach the FREQ target
  ⇒ *under the current merge-only parameterisation, task-guided motif discovery
  cannot satisfy the frequency dictionary's sparsity budget*.

Definitions of the causal quantities:

$$
\Delta_{\rm decompress}=MAE_{\rm FREQ}-MAE_{\rm FREQ-LOOSE},
\qquad
\Delta_{\rm task}^{\rm high\text{-}rate}=MAE_{\rm FREQ-LOOSE}-MAE_{\rm TM},
$$

$$
G_{\rm total}=MAE_{\rm FREQ}-MAE_{\rm TM},\quad
G_{\rm decompress}=MAE_{\rm FREQ}-MAE_{\rm FREQ-LOOSE},\quad
G_{\rm task,high}=MAE_{\rm FREQ-LOOSE}-MAE_{\rm TM}.
$$

The decomposition into `G_decompress` / `G_task,high` is reported as an
**empirical decomposition**, not a strict additive causal identity, and only
when FREQ-LOOSE and TM rates are well matched.

Both **best valid** and **Top-5 soup** are reported; the primary claim names
which one it uses.

## 11. Rate–performance plane

Report a plane with x-axis `R_state` and y-axis valid MAE containing at least
FREQ, FREQ-LOOSE, TM, TM-RATE. This is the primary figure — it shows whether
task distortion improves at matched representation rate.

## 12. Claim discipline

If positive, the accurate statement is **task-coupled discrete graph dictionary
learning**, not differentiable end-to-end dictionary learning. Even if TM-RATE
matches, **do not** claim large computational compression: FREQ's own dynamic
state is `0.889×` raw (`≈11 %` state reduction). The correct phrase is
**matched structural representation budget**, never `3.2×` compute compression
(occurrence-count compression and actual dynamic-state compression are reported
separately).

## 13. Final conclusion first line (choose exactly one)

> **Task supervision learns a better compositional graph dictionary under a
> matched structural representation budget.**
> / **Task coupling provides a modest but real matched-rate dictionary gain.**
> / **Task-selected boundaries help only when the representation is allowed to
> remain less compressed.**
> / **The apparent task-driven gain is primarily explained by weaker
> abstraction / higher representation rate.**
> / **Rate-matched task dictionary construction is infeasible under the current
> 64-rule merge-only parameterisation.**
> / **Inconclusive due to implementation/training integrity failure.**

## 14. Next decision (choose exactly one)

`formalize alternating task-driven graph dictionary learning` /
`run one reader-gradient dictionary-selection diagnostic` /
`reformulate the problem as task-aware rate-distortion without strong sparsity` /
`analyze merge-only parameterization before further dictionary learning` /
`reconsider dictionary-as-primary representation` /
`fix experiment integrity before interpretation`.

This round does **not** implement the next stage.

## 15. Provenance

Code `tracks/ksvd/code/run_pscd_rm_dictionary.py`; tests
`tracks/ksvd/tests/test_pscd_rm.py`; results (git-ignored)
`tracks/ksvd/results/pscd_rm/`. Reader protocol / `StrongPortOperator` and the
task+MDL selector, candidate enumeration and MDL proxy are reused unchanged from
`run_pscd_tmdl_dictionary.py` (PSCD-TMDL-v0), which reuses
`run_pscd_strong_reader.py` (PSCD-SC-v0) and
`run_pscd_compositional_dictionary_audit.py` (PSCD-v0).
