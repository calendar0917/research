# REPORT — SSRC Phase 1e: ZINC label-mechanism audit and route decision

- Track: `tracks/structural-sufficiency`
- Protocol: `ssrc-phase1e-zinc-v1` (`notes/phase1e_protocol.md`, sha256 `bfa8d9f3…`, status `FROZEN_PRE_GPU`)
- Frozen Phase 1d training commit: `6e26ca9` (`0f2112e` = Phase 1d per-molecule table)
- Code commit for this round: `9a96ad0` (baseline runner introduced at `c84d5de`); artefact manifest: `results/phase1e/run_manifest.tsv`
- Compute: CPU-only for A1–A4 and B1; **one** conditional GPU job for B2 (`ssrc-p1e-zinc-baseline-s0`), occupancy **0.607 A100 GPU-h** on `res`/a100-2 against a 2.0 h cap.
- Status: **COMPLETE**. Direction: **A**.

> This is a **post-hoc audit of a Monitor already used by Phases 1b/1c/1d**. It is
> not a confirmation set, and no result below is a blind test. The official ZINC
> valid/test splits and the internal Reserved 1,000 were not read.

---

## Answer in one paragraph

The large-ring "structural" effect that Phase 1c/1d chased is, on ZINC,
**almost entirely the benchmark's artificial cycle penalty**. The target is not
"number of cycles with >6 atoms" as the PyG docstring claims; it is
`−max(0, longest_cycle_in_a_NetworkX_Paton_basis − 6)` computed in one particular
RDKit-canonical atom order, and it is **not permutation-invariant**. On the
Monitor, the 40 RDKit-large-ring molecules and the 43 label-penalised molecules
coincide (purity 1.0, recall 0.93), the penalised 4% carry 29–55% of every
SSRC arm's absolute error, and the cycle term alone is **68% of the label magnitude**
on that group. Once the penalty is accounted for, **no SSRC arm beats the plain
`G_ref` baseline anywhere** — not on the 96% complement, not on the large-ring
tail. The Phase 1d "large-ring rescue" was a win of `R1_size` over a *worse
sibling* (`R_incidence`), never over `G_ref`. A conditional route built without
training reproduces 0.1922 but is still 0.027 MAE worse than `G_ref`. Finally,
a strictly isolated strong baseline on the same split scores **0.0911** against
`G_ref`'s 0.1653, so the whole family is not competitive either. **Direction A:
stop optimizing this architecture branch; move to real measured properties, and
if the relational question is still of interest, test it on a competitive
backbone (Phase C design).**

---

## 1. Q1 — Is the large-ring grouping a faithful proxy for the real cycle penalty?

**Yes on Monitor, with a documented label-side defect that is smaller than the
artefact itself.**

### 1.1 What the target actually is

`results/phase1e/label_audit.json` reconstructs the generator from the upstream
GVAE script (sha256 `9ff67492…`):

```
cycle   = -max(0, max(len(c) for c in nx.cycle_basis(nx.Graph(adj)) - 6)
y       = (logP-μ)/σ + (SA-μ)/σ + (cycle-μ)/σ
```

- The cycle statistic is the **NetworkX Paton fundamental basis** — not the
  minimum cycle basis, not `GetSymmSSSR`, not the longest simple cycle.
- It is computed in **one fixed atom order** (the label generator's RDKit
  canonical SMILES order) on a graph the model never sees ordered. A basis is
  order-dependent, so the label is **not a permutation-invariant graph function**.
- The PyG dataset docstring ("number of cycles with >6 atoms") does **not**
  describe this computation. That is a benchmark documentation defect, recorded
  here and not fixed in this round.

### 1.2 Agreement between the grouping and the penalty

| set | label-penalised ∩ RDKit-large | label-only | RDKit-large-only | neither |
|---|---:|---:|---:|---:|
| Monitor (1,000) | **40** | 3 | **0** | 957 |
| official train (10,000) | 339 | 33 | **0** | 9,628 |

- Monitor purity of the RDKit-large group **1.000**; recall of the label penalty
  **0.930**. The 3 missed rows are `1238, 1077, 2052`.
- There are **zero** RDKit-large molecules the label does not penalise, so the
  Phase 1d binary split never puts a non-penalised molecule into the "large
  ring" group on this Monitor.

### 1.3 Label reconstruction

Snapping the residual to the integer lattice (Nelder–Mead on `σ_c, μ_c`) recovers
`σ_c = 0.288375`, `μ_c = −0.048159`, and agrees with the independent KSVD audit
table on **10,000/10,000** rows. Snap error: p95 `0.0194` y-units, max `0.373`.
Recovered excess agrees with the canonical-order basis on 9,983/10,000,
with the stored PyG order on 9,980/10,000, and with the order-independent
RDKit SSSR maximum on 9,966/10,000.

### 1.4 The defect that stays

Under 30 random atom permutations, **26 of the 372** penalised training molecules
change their penalty level; **4 of the 43** Monitor molecules do
(`9864, 2052, 4235, 2334`). That is 9.3% of the penalised Monitor rows, above
the 5% unresolved-mechanism threshold frozen in §2 of the protocol. The artefact
is therefore large **and** irreducibly order-dependent; part of the remaining
error on this subset is not learnable from an unordered graph at all.

Penalty structure on Monitor: 957 rows at level 0, 37 at level 1, 6 at level 2.
On the penalised group the mean cycle component is **−3.785** y-units against a
mean `|y_without_cycle|` of **1.748** — the artificial term is **68.4%** of the
label magnitude.

---

## 2. Q2 — Do the performance gaps relate to the artificial label term?

**Yes: the label-penalised 4% of the Monitor carries 29–55% of every SSRC arm's
absolute error, and no arm is better than `G_ref` in any group.**

Monitor MAE by group (`error_attribution.json`; 957 complement / 43 penalised):

| arm | all | complement | penalised | penalised error share | `cycle_offset_capture` |
|---|---:|---:|---:|---:|---:|
| `G_ref` | **0.1653** | **0.1219** | **1.1301** | 0.294 | 0.790 |
| `R1_size_cap` | 0.1935 | 0.1383 | 1.4211 | 0.316 | 0.662 |
| `R1_size` s0 | 0.2366 | 0.1658 | 1.8111 | 0.329 | 0.545 |
| `M_keep_edge` | 0.2370 | 0.1532 | 2.1022 | 0.381 | 0.455 |
| `R1` | 0.2602 | 0.1722 | 2.2185 | 0.367 | 0.459 |
| `R_incidence` | 0.2797 | 0.1383 | 3.4253 | 0.527 | 0.097 |
| `S_cap_size` | 0.2895 | 0.1376 | 3.6690 | 0.545 | 0.042 |
| `S0` | 0.3173 | 0.1727 | 3.5352 | 0.479 | 0.066 |

Every arm-vs-`G_ref` paired contrast is **positive in every group**, including
the RDKit-large group that Phase 1d reported as an `R1_size` win:

| arm | complement Δ | penalised Δ | RDKit-large Δ |
|---|---:|---:|---:|
| `R1_size_cap` | +0.0164 [+0.0033,+0.0303] | +0.2909 [−0.1096,+0.7010] | +0.2862 [−0.1313,+0.7208] |
| `R1_size` s0 | +0.0439 [+0.0261,+0.0629] | +0.6809 [+0.1853,+1.1802] | +0.7219 [+0.1891,+1.2471] |
| `R_incidence` | +0.0164 [+0.0050,+0.0267] | +2.2951 [+1.7870,+2.7930] | +2.4730 [+1.9903,+2.9383] |

Read this against the Phase 1d statement "`R1_size` is much better on the 40-molecule
large-ring subgroup (+1.751 MAE)": that +1.751 is `R1_size − R_incidence`. **Against
`G_ref`, `R1_size` is 0.72 MAE worse on exactly those 40 molecules**, and so is every
other arm. The "large-ring rescue" rescues a control from itself; it does not
reach the baseline.

`cycle_offset_capture` (fraction of the artificial offset a model reproduces) is
0.79 for `G_ref` and 0.66 for `R1_size_cap`: even a plain GINE learns to imitate
much of the penalty, because ring size is inferable from structure. A model that
did *not* imitate it would be scored worse on this target.

Practical consequence: **any ZINC `logP_SA_cycle` number in this track is a
mixture of property prediction and a hidden ring-size task**, weighted by
whichever 4% of the split happens to carry the penalty. See §9 for what that
means for previously reported numbers.

---

## 3. Q3 — What does the zero-training conditional routing diagnostic give?

**0.1922 Monitor MAE — a 0.001 improvement over `R1_size_cap` alone and still
0.027 worse than `G_ref`. It is a diagnostic, not a method.**

Frozen rule (`zero_training_routing.json`), evaluated without any training or
label-based selection:

```
if RDKit large ring (large_cycles_gt6 ≥ 1)  -> R1_size_cap   (40 molecules)
else                                        -> R_incidence   (960 molecules)
```

| quantity | value | 95% CI |
|---|---:|---|
| routed MAE | **0.1922** | — |
| routed − `G_ref` | **+0.0269** | [+0.0073, +0.0477] |
| routed − `R1_size_cap` | −0.0013 | [−0.0156, +0.0111] |
| routed − `R_incidence` | −0.0875 | [−0.1213, −0.0560] |
| label-oracle route (43 → cap) | 0.1935 | — |

The interesting part is the complement: routing *away* from `R_incidence` on
ordinary molecules makes them worse (+0.0164 [+0.0050, +0.0267] vs `G_ref`),
exactly as Phase 1d's incidence result predicted. A per-molecule FLOPs lower
bound of 719.7 M = **0.607× `G_ref`** is achieved only in the algebra; the
batched implementation cannot realise it, and the single `R1_size_cap` model is
2.10× `G_ref`. **The combine-and-route idea is arithmetically attractive and
empirically pointless**: it buys 0.001 MAE over its own best member while
remaining worse than the plain baseline.

Status label carried in the artefact:
`POST_HOC_DIAGNOSTIC_ON_A_REUSED_MONITOR`, `not_a_new_model_result: true`.

---

## 4. Q4 — Unified performance reference, and is `R1` competitive?

### 4.1 Three protocol classes, never merged

`reference_audit.json` keeps the four tables separate and states that no number
may be subtracted from, ratioed with, or ranked against a number in another
table.

**Class 1 — internal, SSRC Fit-8000 / Monitor-1000 (this protocol):**

| model | Monitor MAE |
|---|---:|
| `P3E` strong baseline (B2) | **0.0911** |
| `G_ref` | 0.1653 |
| `R1_size_cap` | 0.1935 |
| `R1_size` s1 / s0 | 0.2810 / 0.2366 |
| `R_incidence` | 0.2797 |
| `S0` | 0.3173 |

**Class 2 — BJOC internal split-B (different Monitor molecules; 793/1000 of its
Fit rows are SSRC Monitor rows, so it cannot be rescored here):** seed-mean
P3 0.114942, P0 0.115948, B3 0.131529, B0 0.129920; Monitor-B has 121/1000
molecules in common with SSRC Monitor. Comparable only to itself.

**Class 3 — exposed official ZINC-subset test:** `BJOC-P3` **0.069822**
(0.069304 / 0.070340 across two seeds), `P0` 0.073957. **Exposed**: this test was
read on 2026-10-03 by the KSVD track. It is not a fresh measurement and it is not
comparable to a Monitor number.

**Class 4 — published official ZINC-12k (cited, not re-verified):** ESA+ 0.052,
CSA-rings 0.056, MoSE+GRIT 0.056, GRIT 0.059, SPSE+GRIT 0.059, GPM 0.064,
GraphGPS ≈0.070, GSN ≈0.101.

The distance between class 1 and class 4 is the honest measure of how far this
project's *internal* protocol sits from the literature; the distance between
class 3 and class 4 is the only like-for-like comparison the repo has, and it is
an exposed one.

### 4.2 Contamination audit — no historical strong checkpoint is usable

| checkpoint | supervised rows | Monitor overlap | usable |
|---|---|---:|---|
| BJOC-P3 full-10k / BJOC-P0 full-10k | all official train | 1000/1000 | no |
| BJOC-P3 split-B (internal 8k/1k/1k) | split-B Fit | 793/1000 | no |
| BJOC masked-pretraining split-A arms | split-A Fit | 817/1000 | no |
| BJOC predictive-bottleneck N8/O8 | split-A Fit | 804/1000 | no |

**0 of 5** historical strong checkpoints can be scored on the SSRC Monitor
without label contamination. The only way to get a matched reference was to run
one.

### 4.3 B2 — the one conditional GPU baseline actually run

Justification per protocol §3: a frozen, two-seed-replicated recipe exists
(BJOC-Null + masked atom/bond SSL + R3 360-epoch cosine + last-5 soup, 413,505
params), projected cost 0.595 GPU-h < 2.0 h cap, and strict label isolation is
enforceable in code. Isolation contract enforced: objects, standardiser, SSL and
supervised batches use **SSRC Fit 8,000 only**; Monitor predictions were written
and hashed **before** the single label read; no early stopping; no model
selection; parameter-count assertion (413,505) and encoder-hash assertion pass.

| | Monitor MAE | vs `G_ref` | 95% CI |
|---|---:|---:|---|
| `P3E` strong baseline | **0.0911** | **−0.0742** | [−0.0965, −0.0545] |
| 957-molecule complement | 0.0773 | −0.0446 | [−0.0557, −0.0346] |
| 43 penalised | 0.3974 | −0.7328 | [−1.1214, −0.3810] |
| 40 RDKit-large | 0.2298 | −0.7847 | [−1.1984, −0.4141] |

Wall clock 1635 s SSL + 524 s supervised; total job 2184 s = 0.607 GPU-h.

**B3 label: `R1_IS_NOT_COMPETITIVE`** — the best SSRC arm is 0.074 MAE worse with
a CI wholly below zero, and the gap survives on the ordinary 96% (0.045), where
ring-size descriptors are nearly constant.

**Mandatory caveat.** This is a *competitiveness* statement, not a controlled
architecture comparison. The B2 arm receives structure-only global descriptors
(`cyclomatic_rank`, `ring_atom_fraction`, `ring_size7_frac`, `ring_size8plus_frac`,
junction fractions) that the SSRC arms never see, plus SSL pretraining. Its
93% capture of the artificial cycle offset is partly by construction. What it
legitimately establishes is: **on this split, the R1 family is far from the
strongest thing the repo can already build.**

---

## 5. Q5 — A4 classification

Applying the frozen rules of `notes/phase1e_protocol.md` §2 mechanically
(`decision.json`):

### `BENCHMARK_CYCLE_ARTIFACT_DOMINATES` — **FIRED**

| condition | threshold | value | pass |
|---|---|---:|---|
| RDKit-large group left unpenalised | ≤ 5% | 0/40 = 0.0% | ✅ |
| complement gap `R1_size_cap − G_ref` | < 0.02 | **0.0164** | ✅ (narrow) |
| penalised share of absolute error (max over arms; rule needs ≥1 arm) | ≥ 25% | **54.5%** (`S_cap_size`); `R1_size_cap` 31.6%, `G_ref` 29.4% | ✅ |

### `STRUCTURAL_SIGNAL_BEYOND_CYCLE_REMAINS` — **NOT FIRED**

No SSRC arm beats `G_ref` by ≥0.02 with a CI wholly below zero in either the
complement or the penalised group. `R1_size_cap` on the penalised group has a
positive *point* estimate (+0.291) whose CI crosses zero; it is recorded and not
upgraded. (The B2 strong baseline would satisfy this rule — it is −0.045 on the
complement and −0.73 on the penalised group — but it is a different information
regime and answers B3, not A4. Mixing it in would be a category error.)

### `UNRESOLVED_LABEL_MECHANISM` — **FIRED** (criterion a only)

| condition | threshold | value | pass |
|---|---|---:|---|
| permutation-unstable penalised fraction | ≥ 5% | 4/43 = **9.3%** | ✅ |
| reconstruction error > 1.0 fraction | ≥ 5% | 0% | ❌ |
| proxy miss fraction **and** missed-error share | ≥ 5% both | 7.0% / 4.9% | ❌ |

**Combined classification:**
`BENCHMARK_CYCLE_ARTIFACT_DOMINATES` **(primary)** +
`UNRESOLVED_LABEL_MECHANISM` **(secondary, order-dependence of the cycle basis)**.
Primary cause of the Phase 1c/1d large-ring effect: the artificial `−cycles`
term in the target, not a relational mechanism.

---

## 6. Q6 — Route decision

### **Direction A** — move to real measured properties / conditional structural computation.

Triggered because the artefact rule fired and nothing survived beyond it. Two
constraints carry forward, both recorded in `decision.json`:

1. Report the order-dependent cycle basis as a **benchmark defect**, not a
   modelling failure. Any future ZINC claim must exclude the penalised subset or
   report it separately, and must never quote an aggregate that mixes them.
2. The Phase C test must run on a **competitive** backbone, not on `G_ref` or
   `R1`, because §4.3 shows the family is 0.074 MAE behind a baseline the repo
   can already build.

Direction B (adopt a mature backbone as base) is *not* selected as the route
because it is a means, not a research question; Direction C is not selected
because no out-of-label-rule structural signal was demonstrated; Direction D is
not selected because the round produced a decisive, reproducible explanation
rather than an unresolved one.

---

## 7. Phase C — chosen task and design (design only, no training)

Full design: `results/phase1e/phase_c_design.md`.

**Task: Lipophilicity**, frozen `ssrc-phase1a-lipo-v1` scaffold split
**Fit 2352 / Monitor 504 / Reserved 504**, measured `exp` label, no artificial
term, existing in-repo structure and label caches, zero new downloads.

Because of §4.3 the design is **two-tier**:

- **Tier 1 (≈1.3 GPU-h, 4 arms × 5 seeds)** — family-internal: `G_ref` (H1),
  `S0` (H1, matched), `R1` (H2), new `D8` = 8-layer GINE with residual +
  jump-knowledge (H3). Fixed 200 epochs, last-5 soup, no early stopping.
- **Tier 2 (≈0.3 GPU-h, 2 readouts × 5 seeds)** — decisive: freeze a BJOC trunk
  pretrained on lipo Fit structures only, then compare `RO_pool` (H1) against
  `RO_rel` (H2, one-shot endpoint-pair relation readout) at matched parameters.
  This is the only version of the H2 question whose null is interpretable.

Preregistered thresholds, seed-sign requirements and falsification rules are in
the design file; no lipophilicity training was run in Phase 1e.

---

## 8. Reproducibility

```bash
# CPU stages (no GPU, no new labels)
PYTHONPATH=. .venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_label_audit.py
PYTHONPATH=. .venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_analysis.py
PYTHONPATH=. .venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_reference_audit.py
PYTHONPATH=. .venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_decision.py

# tests
.venv/bin/python -m pytest tracks/structural-sufficiency/code/phase1e/tests -q   # 16 passed

# the single GPU job (conditional Stage B2)
rr deploy res
rr run res ssrc-p1e-zinc-baseline-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/results/phase1e/baseline \
  -- .venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_baseline.py run --device cuda

# manifest
.venv/bin/python tracks/structural-sufficiency/code/phase1e/phase1e_manifest.py
```

Key hashes:

| artefact | sha256 |
|---|---|
| protocol `notes/phase1e_protocol.md` | `bfa8d9f38bfe3a266b32ceda1fb1b58f4d8c44a4f7619646457a94496b6cd186` |
| split `split_zinc_phase1b.json` | `fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49` |
| labels `zinc_labels_fit_monitor.json` | `631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78` |
| B2 Monitor predictions | `fc698ccc82dba024f73baa4604e27443bc25d74c452ccd38b8685f2a6aa4d1b1` |
| B2 soup | `94ba0f5f14d87169610af8f3abdbc577ca30b0685e322ac4af896fb52eb530fd` |

All other artefact hashes are in `run_manifest.tsv`.

Environment: `.venv/bin/python` (torch 2.5.1+cu124, RDKit via the lipo group,
numpy available only inside the venv). The upstream `sascorer`/`fpscores` modules
are loaded with an explicit `fpscores` path; without it the import silently
picks up a different file.

---

## 9. What this changes, and what it does not

**Changes:**

- Phase 1d's `PAIRWISE_INTERACTION_PROMISING` / `LARGE_RING_TAIL_SPECIFIC` must be
  read as **relative to a worse sibling arm**. No SSRC arm is better than
  `G_ref` on the large-ring tail. The Phase 1d non-claim "the large-ring subgroup
  is partly label-mechanism-coupled because the target includes −cycles" is now
  quantified: purity 1.0, 29–55% of error share, 68% of label magnitude,
  9.3% permutation-unstable.
- **Any aggregate ZINC `logP_SA_cycle` number in this track is protocol-mixed**
  with a hidden ring-size task. Historical numbers (e.g. `R1` 0.260161,
  `G_ref` 0.165290) remain valid as *measurements of that target*, but they are
  not clean property-prediction results and must not be presented as such.
- The efficiency obstacle from Phase 1d is now dominated by a competitiveness
  obstacle: 2.10× FLOPs *and* +0.074 MAE against an isolated baseline.

**Does not change:**

- No claim that relational/structural computation is useless in general. The
  Lipophilicity evidence remains a single-seed, Monitor-selected screening in
  which every contrast CI crosses zero (Phase 1b).
- No claim about official ZINC test performance, no SOTA comparison, no
  cross-protocol ranking.
- No retraction of Phase 1b's ZINC relation gain `S0 → R1` (+0.057, replicated
  across two seeds): that contrast is within the SSRC family and is untouched by
  this audit. What changes is that it does not translate into a win over
  `G_ref`, and that part of the tail advantage it does have is label artefact.

---

## 10. Non-claims

1. Post-hoc audit of a Monitor reused by Phases 1b/1c/1d; not a confirmation set.
2. `BENCHMARK_CYCLE_ARTIFACT_DOMINATES` is a statement about **ZINC's
   `logP_SA_cycle` target and the SSRC arm family on it**, not about SSRC as a
   research direction.
3. The conditional routing number is a diagnostic on frozen predictions, not a
   method, and its FLOPs figure is a theoretical lower bound.
4. The B2 baseline is not a controlled architecture comparison (different
   information access, different pretraining); it establishes competitiveness
   only.
5. `UNRESOLVED_LABEL_MECHANISM` rests on a single criterion (9.3% permutation
   instability) with n = 4 molecules; it is a defect flag, not a mechanism proof.
6. The cycle penalty was never used as a model input and no label compensation
   was applied anywhere in this round.
7. Phase C is a design, not a result.
