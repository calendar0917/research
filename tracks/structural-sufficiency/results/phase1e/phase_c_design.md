# Phase 1e — Phase C task selection and preregistration draft

Status: **DESIGN ONLY. No lipophilicity training was run in Phase 1e.**
Parent protocol: `ssrc-phase1e-zinc-v1`.

This file is the deliverable of Phase C: *choose one* non-ZINC-cycle-penalty task
and design a test that can distinguish **H1 local statistics**,
**H2 one-shot relational composition** and **H3 longer-range iteration**.

Two facts from Phase 1e shape the design and were not known when Phase C was
first sketched:

1. **The ZINC large-ring axis is an artefact** (A1–A4). Any task whose target
   contains an artificial structural term is disqualified.
2. **The SSRC arm family is not competitive** (B2). A strictly isolated strong
   baseline scores **0.0911** Monitor MAE on the SSRC split against **0.1653**
   for the best SSRC arm, and it also wins on the 957-molecule complement
   (0.0773 vs 0.1219). A null result inside a family that is 0.06–0.10 MAE
   behind a known baseline on the same rows is uninterpretable — it cannot
   distinguish "relational composition does not help" from "this family cannot
   reach the signal at all".

The design below therefore has two tiers, and the decisive one runs the H2
contrast on the competitive backbone rather than inside the patch family.

---

## 1. Rules

1. The target must be a **measured property**, never a formula that contains an
   artificial structural term.
2. Split, labels and metric must already exist in the repo and be frozen.
3. Every arm is scored on the **same rows** with the **same budget**.
4. No Monitor-based early stopping, no per-arm epoch selection, no seed
   cherry-picking: fixed epoch count, fixed last-k checkpoint soup.
5. Monitor is read **once**, after all predictions are frozen and hashed.
6. Each tier contains a local-statistics control, a one-shot relational arm and,
   where the architecture permits it, an iterative/longer-range arm.

## 2. Candidates considered

| Candidate | Why not chosen |
|---|---|
| ZINC `logP_SA_cycle` (any variant) | the exact defect Phase 1e documented; the cycle term is an order-dependent artificial target |
| ZINC `logP_SA` without cycle | not a frozen repo target; regenerating it re-opens the label pipeline mid-track |
| Any ring-count / cycle-length target | the target would be the structure feature under test |
| Synthetic relational probes (parity, subgraph matching) | no application reference; already covered by Phase 0 |
| BBBP / Tox21 / ESOL | label imbalance or tiny effect sizes; no frozen SSRC split in-repo; new downloads required |
| **Lipophilicity, frozen `ssrc-phase1a-lipo-v1` scaffold split** | **chosen** |

## 3. Chosen task

**Lipophilicity (MoleculeNet / DeepChem), 4,200 rows**, target = measured `exp`.

- Official DeepChem 2.8.0 scaffold split 3360/420/420. Only the **official train
  3,360** rows are used. The official valid/test labels were already read by an
  earlier, unrelated track experiment (`bjoc_cross_task_lipophilicity_v1`), so
  they are **exposed** and are not used here at all.
- Internal split, structure-only and label-blind, frozen at
  `results/phase1a/split_phase1a.json`
  (sha256 `9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6`):
  **Fit 2352 / Monitor 504 / Reserved 504**, scaffold-grouped, zero cross-side
  scaffold overlap (725 / 421 / 422 groups).
- Fit+Monitor labels only: `results/phase1a/labels_fit_monitor.json`
  (sha256 `0260a3909eec30900df646bc760e65360315f4a1398c7092c7c72d04a4e404f7`,
  2,856 labels; no Reserved, no official valid/test).
- Structure caches already built: `results/phase1b/prep_lipo_manifest.json`
  (patch cache for the SSRC arms) and
  `tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/`
  (`objects_train.pt`, `vocab_v1.json`, `standardizer_v1.json` for the BJOC
  object pipeline).

### Why this task can separate H1/H2/H3

- The scaffold split puts **unseen scaffolds** in Monitor: functional-group
  statistics transfer, while composing a novel scaffold is exactly what should
  fail if H2 is not real.
- There is **no artificial cycle term**: a large-ring molecule is not
  automatically hard, so the Phase 1e artefact cannot masquerade as signal.
- The Phase 1b train≈0.21 / Monitor≈0.55 gap is a genuine distribution shift.

### Existing evidence, and why it is not a confirmation

| source | setting | numbers |
|---|---|---|
| Phase 1b, 1 seed/arm, Monitor early stopping | SSRC arms, lipo | G_ref 0.5644, R1 0.5521, S0 0.5610, M2 0.5470, M1 0.5716 |
| `bjoc_cross_task_lipophilicity_v1`, 5 seeds | BJOC backbone, **official** split, raw scale | test MAE 0.5433 (SSL) / 0.5768 (scratch); RMSE 0.7344 / 0.7751 |

Neither is usable as Phase C confirmation: the first is single-seed and selected
on Monitor (spread 0.025 ≈ seed noise); the second uses the **exposed official
split**, so its test number can never be a fresh measurement here.

## 4. Preregistered design

### Tier 1 — family-internal H3 test (cheap, 4 arms × 5 seeds)

Purpose: decide whether the *iteration* hypothesis is worth anything **before**
spending on the decisive tier, and give the H2 contrast a same-family reference.

| arm | role | H | params |
|---|---|---|---|
| `G_ref` | 2-layer GINE on the raw graph | H1 (local, higher capacity) | 447,751 |
| `S0` | radius-2 patch encoder + pooling, no cross-patch interaction | H1 (local, lower capacity) | 59,011 |
| `R1` | one-shot relational composition on the patch multiset | H2 | 97,731 |
| `D8` | GINE, **8** layers, residual + jump-knowledge readout, width 96 | H3 | pinned at implementation, ≤ 500k |

`D8` is the only new model code in tier 1 and must be a depth-only change
against `G_ref` (same width, features, residual + concatenated per-layer
pooling). Its parameter count is recorded before any run and may not be tuned
after seeing Monitor.

Fixed recipe: batch 64, Adam 1e-3 → 1e-5 cosine, wd 1e-5, clip 5, **200 fixed
epochs** (≈ 7,400 steps), prediction = mean of the last 5 epoch checkpoints.
Seeds 0–4, identical seed→init map across arms. Cost ≈ **1.3 GPU-h**.

### Tier 2 — decisive H2 test on a competitive backbone (2 readouts × 5 seeds)

Purpose: ask whether one-shot relational composition adds value **on top of the
representation that is already competitive**, which is the only version of the
H2 question that a null result can settle.

Base: the frozen BJOC object pipeline (`Atom→Bond→Junction`, Null composer, no
message passing) with the frozen masked-SSL recipe, pretrained **on lipo Fit
2,352 structures only** (no labels, no ZINC transfer). The trunk is then frozen.

| readout | role | H |
|---|---|---|
| `RO_pool` | size-aware global pooling of atom/bond/junction embeddings + MLP | H1 |
| `RO_rel` | same trunk, same parameter budget, plus one-shot endpoint-pair relation `phi(h_u+h_v, |h_u-h_v|, h_u*h_v, e_uv)` read out once per bond, then aggregated | H2 |

Both readouts are trained on Fit only, at matched parameter count (±3%), same
fixed 200 epochs and last-5 soup. The trunk is not fine-tuned, so the contrast
is purely "does a one-shot relational readout extract more from the *same*
frozen representation". Cost ≈ **0.3 GPU-h** (one SSL pretrain ≈ 8 min on 2,352
molecules, plus ten short readout fits).

H3 is deliberately **not** tested in tier 2: the BJOC trunk has no iteration by
construction, so an iteration arm there would be a different architecture rather
than a controlled contrast. H3 belongs to tier 1.

### 4.1 Statistics

- Per-seed paired bootstrap over Monitor rows (5,000 resamples, seed 20261022).
- Pooled seed-mean with a seed-level bootstrap as a cross-check.
- A contrast counts only if the 95% interval **and** a ≥4/5 seed-sign agreement
  both point the same way.

### 4.2 Frozen decision rules

| observation | conclusion |
|---|---|
| tier 1 `D8 − G_ref` CI upper < 0 and ≥4/5 seeds negative, and tier 2 `RO_rel − RO_pool` does not clear its bar | **H3 supported** — the gain needs iteration, not one-shot composition |
| tier 2 `RO_rel − RO_pool` CI upper < 0 and ≥4/5 seeds negative | **H2 supported** — one-shot relational composition carries real signal on a competitive base |
| tier 1 `R1 − G_ref` and `R1 − S0` both CI upper < 0, while tier 2 does not clear its bar | **H2 family-specific** — the effect exists but does not survive a stronger base |
| no contrast clears its bar | **H1 sufficient on this task** — no evidence that cross-patch relational computation adds value here |
| conflicting seed signs or CIs straddle 0 | **UNDERPOWERED / UNRESOLVED** — no directional claim; the branch is closed for lack of a testable effect |

Tier 2 is only run if tier 1 does not already produce a decisive H3 result; the
tiers are ordered, not combined. A "supported" label is a statement about *this*
scaffold split at n=2,352, not about molecular property prediction in general.

### 4.3 Cost and gates

- Tier 1 ≈ 1.3 A100 GPU-h; tier 2 ≈ 0.3 A100 GPU-h; total ≈ **1.6 GPU-h** if both run.
- Gates before any GPU run: parameter-count assertions; per-arm batch-key
  assertions (each arm sees only its declared inputs); a `D8` depth-only diff
  against `G_ref`; a `RO_rel`/`RO_pool` parameter-match check; a CPU smoke of
  2 epochs per arm; and a check that no official valid/test or Reserved label
  enters any reader.
- Go/no-go: Phase C is worth running only if the route decision lands on
  Direction A *and* the relational question is judged still open. If the route
  is B, Phase C is superseded; if D, Phase C is cancelled.

## 5. Threats to validity, stated in advance

1. **n = 504 Monitor rows.** A 0.02 MAE difference is at the edge of resolution;
   the rules therefore require an interval *and* a seed-sign pattern.
2. **Scaffold shift dominates.** If the gap is driven by scaffold novelty rather
   than relational structure, every arm fails together. That is informative
   (H1 sufficient) but does not generalise to in-distribution prediction.
3. **Tier 2 freezes the trunk.** A null there means "this frozen representation
   does not expose the signal to a one-shot readout", not "the signal does not
   exist in the data".
4. **Tier 1 capacity asymmetry.** `R1` has 4.6× fewer parameters than `G_ref`;
   a win is conservative for H2, a loss is not automatically evidence against
   H2 — hence the `S0` contrast.
5. **Tier 2 differs from the B2 baseline in one respect:** the B2 strong
   baseline is supervised end-to-end on ZINC, while tier 2 freezes a lipo-only
   SSL trunk. Tier 2 is therefore not a re-run of B2 and its numbers must not be
   compared with the B2 Monitor MAE.
6. **One dataset, one split.** No cross-task replication is proposed here.
