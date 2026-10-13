# BJOC Performance & Mechanism — Long-Horizon v1 — REPORT

Protocol `cscl-bjoc-performance-mechanism-long-v1` · frozen code revision commit
`3bbd9ee76413` (preregistration; later commits only add analysis/chain scripts) ·
config `configs/bjoc_performance_mechanism_long_v1.yaml` (`sha256 fada448e3ca3…`)
· driver `code/run_bjoc_performance_mechanism_long_v1.py` (`sha256 620ffa1e10ce…`)
· training module `code/bjoc_perfmech_long_v1.py` (`sha256 cb9796cb45f6…`) ·
note `notes/bjoc_performance_mechanism_long_v1.md` (preregistration, deviations).

**Scope.** Performance-first round on BJOC-Null (413,505 params; C1 438,081).
Stage A fills the missing `Scratch + 360ep cosine` (B3) cell on the official train
10,000. Stage B repeats the {scratch, SSL} × {R0 240-const, R3 360-cos} 2×2 on a
new grouped internal split (8k/1k/1k, Fit-only preprocessing, Fit-only SSL
encoder). Stage C tests a bond-composition second-moment readout (`Σc_b²`) after
a CPU gate. Stage D (second split) and Stage E (atom-only vs atom+bond masked
SSL) are conditional purchases.

**Discipline.** Official test was never loaded, forwarded or read in any form;
the driver has no test-data path. Official valid is an exposed development split
and is only used descriptively. Monitor/Locked labels of split B were each read
exactly once, after every prediction was frozen and hashed. All GPU work ran on
`res` **GPU1 only** (GPU0 was occupied by an unrelated external job all round);
matched arms share the same device and regime.

---

## A. One-line conclusion

**`SSL_GAIN_PERSISTS_UNDER_R3` + `SSL_GAIN_ROBUST_UNDER_R3` + `RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS`:
on both the full-10k exposed valid and the new grouped split, masked-SSL
initialisation is the only large, seed-consistent performance component
(−0.014 … −0.020 MAE, surviving every recipe and both monitor/locked blocks),
while the 360-epoch cosine recipe (R3) is a small, seed-unstable,
selection-conditioned effect (−0.0037 exposed valid, −0.0010 Monitor, −0.0028
Locked, never both seeds positive and never CI-significant) that is *not*
replicated on the new split; the recipe × SSL interaction is likewise small
(−0.0018 … −0.0026) and seed-heterogeneous. The second-moment readout (C1) is
directionally positive but below the pre-registered purchase threshold
(−0.0019 Monitor, `C1_GAIN_WEAK_OR_MIXED`), and atom-only masked SSL transfers
as well as joint atom+bond (`BOND_OBJECTIVE_NEUTRAL`, +0.0020, single seed).
Round cost 3.04 of 7.0 GPU-h.**

---

## B. Core tables

### B1. Stage A — full-10k 2×2 on official valid (exposed dev, n = 1000, paired)

| Init | Recipe | Seed 0 | Seed 1 | Seed-mean |
|---|---|---:|---:|---:|
| scratch | R0 240 const | 0.100172 | 0.098232 | 0.099202 |
| SSL | R0 240 const | 0.084786 | 0.082313 | 0.083550 |
| scratch | **R3 360 cos (B3, new)** | **0.097775** | **0.096714** | **0.097244** |
| SSL | R3 360 cos (P3) | 0.079101 | 0.080586 | 0.079844 |

| Contrast | Δ seed-mean | CI95 | Per-seed |
|---|---:|---|---|
| `G_SSL_R0 = P0−B0` | −0.01565 | [−0.02105, −0.00990] | −0.01539 / −0.01592 |
| `G_SSL_R3 = P3−B3` | −0.01740 | [−0.02479, −0.00997] | −0.01867 / −0.01613 |
| interaction `G_SSL_R3−G_SSL_R0` | −0.00175 | — | −0.00329 / −0.00021 |
| recipe (SSL) `P3−P0` | −0.00371 | [−0.00639, −0.00101] | −0.00568 / −0.00173 |
| recipe (scratch) `B3−B0` | −0.00196 | [−0.00689, +0.00326] | −0.00240 / −0.00152 |

Decision vocabulary: `G_SSL_R3 ≤ −0.005` and CI upper < 0 →
**`SSL_GAIN_PERSISTS_UNDER_R3`**. B3 closes only ~14 % of the P3 gap; the
360-cosine recipe is *not* what made the SSL arm strong. Ordinary (y > −3) Δ
−0.0193, tail (n=65) Δ +0.0103 — the historical tail trade-off of SSL is
reproduced on this split.

### B2. Stage B — new grouped split (Fit 8,000 / Monitor 1,000 / Locked 1,000)

Preregistration split salt `…-B`; `duplicate_groups_cross_sides = 0`. **Overlap
disclosure:** 6,432 of the 8,000 Fit-B rows come from the historical split-A fit
block (1,568 from A monitor/locked); 782 Monitor and 786 Locked rows were
historical split-A *fit* rows (i.e. present in historical full-10k training),
while 95/123 Monitor and 99/115 Locked rows overlap A monitor/locked. The
split's value is a *matched* partition, not fresh data; see §E.

**Monitor (read once, n = 1000, paired, seed-mean of single models):**

| Contrast | Δ | CI95 | Per-seed | ordinary | tail | top-10 removed |
|---|---:|---|---|---:|---:|---:|
| `P3−P0` (recipe, SSL) | **−0.00101** | [−0.00411, +0.00191] | −0.00496 / **+0.00295** | −0.00240 | +0.01667 | −0.00040 |
| `B3−B0` (recipe, scratch) | +0.00161 | [−0.00230, +0.00542] | −0.00061 / +0.00383 | +0.00249 | −0.00958 | +0.00259 |
| `P3−B3` (SSL under R3) | **−0.01659** | [−0.02207, −0.01104] | −0.01744 / −0.01574 | −0.01788 | −0.00023 | −0.01698 |
| `P0−B0` (SSL under R0) | **−0.01397** | [−0.01903, −0.00870] | −0.01309 / −0.01486 | −0.01299 | −0.02647 | −0.01416 |
| interaction (R3−R0 SSL gain) | −0.00261 | — | −0.00434 / −0.00088 | | | |

Gates: `recipe_effect = FAIL` (seed 1 positive), `recipe_effect_scratch = FAIL`,
`ssl_effect_r3 = PASS`, `ssl_effect_r0 = PASS`. Decision
**`RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS`** → Stage D skipped per the frozen
purchase rule. The SSL gain is the robust effect and it is slightly *larger*
under R3 than under R0 on this split too.

### B3. Final Locked table (split B, read once after all choices were frozen)

| Arm | Seed 0 | Seed 1 | Seed-mean | RMSE | median | p90 |
|---|---:|---:|---:|---:|---:|---:|
| B0 scratch R0 | 0.110946 | 0.115653 | 0.113300 | 0.298374 | 0.066774 | 0.211768 |
| B3 scratch R3 | 0.107855 | 0.112969 | 0.110412 | 0.263897 | 0.067343 | 0.210598 |
| P0 SSL R0 | 0.097056 | 0.090156 | 0.093606 | 0.249366 | 0.055154 | 0.179599 |
| P3 SSL R3 | 0.090440 | 0.091241 | 0.090841 | 0.235368 | 0.049267 | 0.180649 |
| A1 atom-only SSL R3 (s0) | 0.091454 | — | 0.091454 | 0.226296 | 0.050539 | 0.184859 |

| Contrast | Δ | CI95 | Per-seed |
|---|---:|---|---|
| `P3−P0` | −0.00277 | [−0.00647, +0.00039] | −0.00662 / +0.00108 |
| `B3−B0` | −0.00289 | [−0.01096, +0.00298] | −0.00309 / −0.00268 |
| `P3−B3` | **−0.01957** | [−0.03098, −0.00964] | −0.01741 / −0.02173 |
| `P0−B0` | **−0.01969** | [−0.02683, −0.01375] | −0.01389 / −0.02550 |
| `A1−P3` | +0.00101 | [−0.00402, +0.00623] | +0.00101 (s0) |

Locked confirms the Monitor pattern: SSL gain ≈ 0.02 MAE and CI-significant on
both recipes; recipe gain ≈ −0.003 with a seed-sign flip and CI crossing 0.

### B4. Stage C — `Σc_b²` second-moment readout (C1, seed 0, Monitor)

CPU gate passed all nine conditions (`identity` 1.04e-7; algebraic
non-identifiability; linear probe test R² median 0.619 of `c_sq` from the frozen
reader inputs; non-degenerate; 100 % non-zero new-column gradients; one-step
prediction change 1.36; permutation/endpoint/batch invariance; no forbidden
modules; 438,081 params). GPU: Monitor Δ(C1−P3) = **−0.00194** — passes
ordinary-not-worse (−0.00115), top-10-removed (−0.00060) and RMSE
(−0.00163 better) but misses the pre-registered ≤ −0.003 seed-0 gate →
**`C1_GAIN_WEAK_OR_MIXED`**; seed 1 not purchased, C1 excluded from the locked
table. Direction is real but too small to claim at this scale.

### B5. Stage E — atom-only vs atom+bond masked SSL (A1, seed 0, Monitor)

A1 uses the atom-only encoder (`c35927aeceaf…`) with the same R3 recipe and
15,120/7,560-update budget discipline. Monitor Δ(A1−P3) = **+0.00197**
(ordinary +0.00104, tail +0.01373); Locked Δ = +0.00101 [−0.00402, +0.00623] →
**`BOND_OBJECTIVE_NEUTRAL`**: at this scale the bond masking objective is not
needed for the transfer gain (single seed, CI ±0.005).

### B6. Budget

| Stage | GPU-h |
|---|---:|
| A (B3F s0+s1 + valid forward) | 0.386 |
| B (2× SSL 0.914; 8 arms 0.981) | 1.895 |
| C/E training (atom-only SSL, A1, C1) | 0.753 |
| prediction freeze | 0.004 |
| D | 0.000 (skipped by gate) |
| **total** | **3.04 / 7.0** |

All on `res` GPU1 (A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, node
a100-2, `CUDA_VISIBLE_DEVICES=1`). Ledger:
`results/bjoc_performance_mechanism_long_v1/gpu_budget_ledger.json`.

---

## C. Direct answers

**Q1 (SSL × recipe interaction after 360ep cosine).** The missing B3 cell is
0.097244 (seed-mean), i.e. scratch+R3 recovers only 0.0020 of the SSL arm's
0.0174 gap; the interaction is −0.00175 (per-seed −0.00329 / −0.00021) — the
SSL gain *does not shrink* under R3, it is marginally larger, driven almost
entirely by seed 0. The new-split Monitor reproduces this: interaction −0.00261
(per-seed −0.00434 / −0.00088) with `ssl_effect_r3` passing all gates. So the
historical "recipe and pretraining are substitutes" reading is not supported;
pretraining dominates the 2×2 in both data regimes.

**Q2 (R3 cross-split replication).** Failed. Official valid: P3−P0 = −0.00371
(seed 1 only −0.00173). New split Monitor: −0.00101 with seed 1 **positive**;
Locked: −0.00277 with seed 1 positive. For scratch, R3 is +0.00161 on Monitor
and −0.00289 (CI crossing 0) on Locked. Verified decision:
`RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS`, Stage D skipped. R3 remains a
seed-0-flavoured candidate, not a transferable training phenomenon at this
scale.

**Q3 (generalisation bottleneck).** The supervised fit/serve gap is now the
binding constraint: Fit-B soup MAE 0.001–0.04 vs Locked 0.090–0.113, and the
only lever that moved Locked by a meaningful amount was the pretraining
initialisation (−0.02). Recipe optimisation of the supervision phase moved the
needle by ≤ 0.003 (seed-unstable), and a strictly richer readout (C1) by
−0.002. The bottleneck is representation/init and data regime, not scheduler or
readout capacity at this model size.

**Q4 (low-complexity readout upgrade).** C1 adds a second moment of the composer
output to the reader for +5.9 % parameters with zero change at init and is
directionally better on Monitor (all secondary gates pass) but misses the
purchase threshold (−0.00194 vs −0.003). No RMSE penalty (it improves RMSE by
0.0016). This is a "weak or mixed" signal, not a demonstrated gain.

**Q5 (paper/attribution implications).** Attribute the BJOC performance
narrative to the **masked-SSL initialisation** (large, replicated across two
data regimes, two recipes, two seeds, and both monitor/locked blocks), not to
the recipe search: the R3 selection is a seed-unstable ≤ 0.003 effect once
tested outside the exposed valid split, and the recipe×SSL interaction does not
show substitution. The second-moment readout and the atom-only ablation bound
the "readout capacity" and "objective detail" axes as small at this scale, which
is directly useful negative evidence for the paper's ablation table. The
tail-vs-ordinary trade-off of SSL should be reported explicitly (ordinary Δ
≈ −0.019, tail can be +0.010 on official valid, but −0.019 on the locked block).

---

## D. Discipline and deviations

1. **Official test never used** (`official_test_loaded: false` in every record;
   driver has no test path; tests assert no test-file access).
2. **Frozen-input audit**: all reused artefacts re-hashed on the remote before
   use (`input_audit.json`), including the Stage-A initialisation parity audit
   (`stage_a_audit.json`: B3F init == P-round scratch init per seed, R3 recipe
   constants and 360-epoch batch order identical, both seeds OK).
3. **Read-once**: `monitor_read_B.lock` and `locked_read_B.lock` record row
   count, y hash, alignment (max |Δ| = 0 vs raw dataset) and the manifest hash;
   predictions were hashed into `prediction_manifest_B.json` before each read.
4. **Deviation D1** (recorded before the Monitor read): the Stage-E purchase
   rule ("read the Monitor SSL gate, then decide") conflicts with "all
   predictions frozen before the first label read". Resolution: atom-only SSL +
   A1 were trained seed-0-only *blind* before the read (+0.62 GPU-h), and the
   arm is kept out of the main table if the gate would have skipped it; the gate
   passed, so A1 is reported. (The alternative — training after the read —
   cannot prove predictions were label-blind.)
5. **Deviation D2**: `locked-freeze`/`locked-score` in the frozen driver assume
   both seeds for every arm, incompatible with the seed-0-only A1; a standalone
   `code/locked_finalize.py` mirrors them with the same manifest checks and
   one-shot label read (no driver change, no metric change).
6. **Split-overlap disclosure** (B2) and the failed prediction-freeze attempts
   (~20 s GPU total, retried; ledger records them as uncounted).
7. Selection pressure: Stage A/B/D/E decisions were fixed in the config before
   GPU time; the only post-hoc decision is C1's weak-gain classification, which
   is mechanical from the frozen gate.

## E. Limitations

- Split B is a new partition, not new data: ~78 % of its Monitor/Locked rows
  were in the historical full-10k training set. Within-split contrasts are
  matched, but absolute MAEs are not comparable to the official-valid numbers.
- Two seeds only; Monitor and Locked blocks are 1,000 molecules (paired
  bootstrap CIs are molecule-level, not seed-level). Recipe-effect estimates
  smaller than ~0.003–0.006 are not resolvable with this design.
- C1 was purchased for seed 0 only (by rule) and evaluated on a 1,000-molecule
  Monitor block; `C1_GAIN_WEAK_OR_MIXED` is a bound, not a refutation.
- A1 is a single seed; `BOND_OBJECTIVE_NEUTRAL` cannot exclude a ±0.005 effect.
- CPU/probe evidence for C1 (linear probe R² median 0.62 with centered residual
  1.03) mixes collinearity and heavy tails; the honest statement is "linear
  recovery of `c_sq` from the frozen reader inputs is far from exact".

## F. Reproduce

Commit chain: `3bbd9ee` (preregistration) → `0b18252` (Stage A + ledger + D1
note) → `f15a41c`/`abf37f5`/`e6dac9d`/`477097d`/`fff8bf8`/`fa8c47a` (analysis
and chain scripts). All remote jobs were launched with `rr` (no manual GPU
commands):

```bash
# remote CPU prep (split B, objects, freeze, audits)
rr run res longv1-prep --pool res-local --result <OUT>/{protocol_freeze,input_audit,split_B,objects_manifest_B,stage_a_audit,prep_fit_B}.json \
  -- python tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py prep --tags B
# Stage A (GPU1)
rr run res longv1-b3-s0 --pool res-gpu1 --result <OUT>/soup_B3F_s0.pt … -- python … stage-a-train --seed 0 --device cuda
# Stage B SSL + arms
rr run res longv1-ssl-s0  --pool res-gpu1 … -- python … stage-b-ssl  --tag B --seed 0 --mode atom_bond --device cuda
rr run res longv1-arms-s0 --pool res-gpu1 … -- bash code/remote_stage_b_arms_chain.sh 0
# Stage C/E training, prediction freeze, one-shot reads
rr run res longv1-ec     --pool res-gpu1 … -- bash code/remote_stage_ec_chain.sh
rr run res longv1-predict --pool res-gpu1 … -- bash code/remote_stage_predict_chain.sh
rr run res longv1-monitor-score --pool res-local … -- python … stage-b-score-monitor --tag B
rr run res longv1-ce-score --pool res-local … -- bash -c '… stage-c-score-monitor --tag B && … stage-e-score-monitor --tag B'
rr run res longv1-locked2 --pool res-local … -- python code/locked_finalize.py --tag B --mode freeze && … --mode score
```

## G. Next steps (not started)

1. Given Q3, test one representation-side change with the remaining budget:
   Fit-B SSL with a different mask/target mix or a longer SSL budget is a more
   promising lever than any supervision-recipe or readout change.
2. If recipe work is revisited, run enough seeds (≥ 4) to resolve a 0.003 effect
   before investing GPU time in candidate selection.
3. C1: the +5.9 %-parameter reader is not dead — a seed-1 purchase would settle
   whether −0.002 is real, but only if it costs ≤ 0.2 GPU-h.
