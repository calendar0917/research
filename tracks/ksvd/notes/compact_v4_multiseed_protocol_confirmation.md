# Compact-v4-hinge multi-seed confirmation under the fixed benchmark protocol

**Status:** FINAL — 2026-09-09
**Stage:** Protocol Correction + Multi-seed Confirmation (confirmation stage; **no
architecture/optimizer/loss/feature change** — models frozen exactly at their seed-0
GO versions)
**Question:** Under the literature-comparable evaluation protocol, is
compact-v4-hinge's seed-0 gain stable across seeds vs compact-v2?
**Verdict:** **GO (not Strong GO)** — mean paired test improvement +0.0092 ≥ 0.008,
v4-hinge better on 4/4 seeds, valid and test improvements agree in direction on 4/4
seeds; gain is concentrated in the long-cycle subgroups (B/C 4/4) while the bulk
group (A) is a mean +0.0031 with **2/4 seed-level sign flips (−0.0068 worst)** →
report as real-but-small bulk gain with instability, not a clean bulk win.
**Main secondary finding:** the earlier seed-0 refit-robustness regression
(+0.0041 vs v2 under train+valid refit) is confirmed as a *different protocol's*
result, not a contradiction of the benchmark-protocol gain.

---

## 1. Why this stage

After the v4 round the track had a split verdict: v4-hinge is very strong on
validation (+0.0141) and on the *frozen selection checkpoint* test (+0.0204,
post-hoc measurement) but regressed under the internal "official" terminal
protocol (train+valid refit → test, −0.0041). Before any further model work, two
things had to be fixed:

1. **Which evaluation protocol is the benchmark's?** The internal terminal protocol
   (refit on train+valid, then test) is *not* the protocol used by the ZINC
   literature. Using it as "official test" both misnamed the measurement and
   created a false obstacle for the v4 direction.
2. **Is the seed-0 gain real?** All v2/v4 numbers were seed-0 only. Per the stage
   mandate: *freeze the architecture, fix the protocol, measure whether the effect
   survives across seeds.* No model/topology/loss/optimizer/scheduler/epoch/batch
   variable was touched.

## 2. Protocol audit (verified against primary sources, 2026-09-09)

### 2.1 What the public ZINC benchmark / cited literature actually does

Sources checked: Dwivedi et al., *Benchmarking Graph Neural Networks* (arXiv:2003.00982v4,
published version) incl. its framework code
(`graphdeeplearning/benchmarking-gnns`, `main_molecules_graph_regression.py`), and
Bodnar et al., *Weisfeiler and Lehman Go Cellular* (CIN, arXiv:2106.12575v2, §5.2/E.4);
plus the repo's own literature note `docs/literature/deep/cin.md`.

| protocol element | VERIFIED benchmark protocol (ZINC-12k) |
|---|---|
| splits | **fixed official splits** 10,000 train / 1,000 valid / 1,000 test (PyG `ZINC(subset=True)` = the benchmarking-gnns split of the 12k GVAE-derived subset). Same split object our runs use (raw-file + index SHA-256 fingerprint identical). |
| fitting scope | **train split only**; validation is evaluation/monitoring only |
| validation role | LR scheduling (ReduceLROnPlateau on val loss: patience 10–20, factor 0.5, stop at lr 1e-5) and/or checkpoint selection / early stopping |
| test evaluation | **single evaluation of the frozen model state on test** (benchmarking-gnns evaluates test at the LR-floor stop; several papers/models additionally store the best-validation state; CIN reports "test MAEs at the time of early stopping") |
| train+valid refit before test | **NO** — none of the checked sources refit on train+validation before test |
| repetitions | mean ± std over repeated initializations: **4 runs / 4 seeds** (Dwivedi et al. published tables; 10 in CIN) |
| budget | 100k-parameter class for the fair-comparison rows (CIN-small ≈ 100k) |

Caveat (kept honest): the checkpoint policy inside "validation selection" differs in
detail between sources (best-val state vs LR-floor stop state), and the frameworks
do not share our exact optimizer/epoch schedule. The *structural* facts — fixed
split, train-only fitting, validation-based model state choice, one test evaluation,
no train+valid refit, multi-seed mean±std — are common to all checked sources.

### 2.2 The internal convention that must be renamed

The `zinc-context-gap` terminal mode retrains the selected epoch budget on
**train+validation** (vocabularies/standardizers refit on train+valid) and then
evaluates test once. This is **NOT** literature-standard; it is an internal
distribution-expansion / robustness protocol. Earlier notes call it "official
test" / "official terminal refit/test"; that wording is deprecated by this stage.

### 2.3 Fixed naming (protocol addendum 2026-09-09, `protocols/zinc-context-gap.yaml`)

```
VERIFIED BENCHMARK PROTOCOL  =  Primary benchmark protocol
    train only -> validation selection -> frozen checkpoint -> single test eval
    measured_as: test_with_selection_checkpoint_mae
    used for: literature comparison, model ranking, mean ± std, GO/NO-GO
    seeds: [0, 1, 2, 3]

INTERNAL REFIT ROBUSTNESS PROTOCOL  =  Secondary refit-robustness protocol
    train+valid -> refit (selected epoch budget) -> single test eval
    measured_as: test_after_train_valid_refit_mae
    used for: robustness / distribution-expansion analysis ONLY
    (formerly "official refit/test"; that wording is retired)
```

Tooling: terminal runs can now measure the primary protocol in-run via
`evaluation.selection_checkpoint_test=true` (frozen best-valid state, train-only
fits, evaluated on official test once; additive, default off) and may skip the
refit phase via `evaluation.skip_train_valid_refit=true` (default off). Commit
`0a61b23`; both flags are measurement plumbing — they do not alter any training
path (seed-0 guard proves bit-identity, §5).

## 3. Frozen models (exactly the seed-0 GO versions; nothing re-chosen)

| | Model A: compact-v2 | Model B: compact-v4-hinge |
|---|---|---|
| base config | `configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml` | `configs/luyin16/zinc_compact_v4_topology_hinge.yaml` |
| config hash (seed 0, +measurement flag) | `d0fbaa28a9499518…` | `06bdd24c7d44e574…` |
| full config hash per seed | `9d95f474…`(s1) `0df673da…`(s2) `aea37bd0…`(s3) | `eb1a8b6d…`(s1) `a9d98b54…`(s2) `14ed98be…`(s3) |
| protocol hash (all runs) | `52b50e0bdd50dbce…` (protocol file after 2026-09-09 addendum) | same |
| git commit (all runs) | `0a61b231b853ce52…` (dirty: pre-existing untracked learning/ docs) | same |
| params (selection phase) | 98,549 | 99,613 |
| model delta | — | only GlobalTopologyEncoder (25→16→8) + head input +8 (topology_mode=hinge, hidden 16, out 8, generic ReLU(L−k) ladder; thresholds/topology features untouched) |
| per-run resolved config | in each run dir `config.resolved.yaml` / `legacy_input.yaml` | same |

Nothing else changed: topology feature set, hinge basis, hidden dims, head sizes,
optimizer (Adam 1e-3, wd 1e-5), scheduler (none), epochs 60, batch 128, patience
12, L1 loss, vocabulary/standardizer scope rules — all identical to the frozen
seed-0 runs. No new model variant was created.

## 4. Determinism policy (this stage executed it)

All 8 runs executed **serially** (one training process at a time; run intervals
below never overlap), same machine, same environment
(Python 3.12.14, torch 2.5.1+cu124 CPU, numpy 2.1.3, networkx 3.4.2, sklearn 1.5.2,
PyG 2.6.1, `torch_threads=4`, machine otherwise idle). Full policy + the ≥3-way
concurrency divergence evidence are recorded in
`notes/reproducibility_cpu_determinism.md` (adopted as repository policy: promoted
runs are serial; concurrency variance must never mix into seed variance).

Execution log (strictly non-overlapping; guard runs first):

| start → end (2026-09-09) | run | runtime |
|---|---|---|
| 19:31:41 → 19:41:57 | v2 s0 guard | 614.1 s |
| 19:44:45 → 19:54:34 | hinge s0 guard | 586.5 s |
| 19:55:30 → 20:00:22 | v2 s1 | 289.0 s |
| 20:03:20 → 20:08:54 | hinge s1 | 332.0 s |
| 20:11:21 → 20:16:55 | v2 s2 | 330.9 s |
| 20:19:18 → 20:24:49 | hinge s2 | 329.1 s |
| 20:27:22 → 20:32:55 | v2 s3 | 330.5 s |
| 20:35:16 → 20:40:50 | hinge s3 | 331.9 s |

## 5. Seed-0 guard (required before multi-seed): PASSED, bit-identical

| check | v2 guard `20260909-193141-6962889e` | hinge guard `20260909-194445-182c7021` |
|---|---|---|
| valid trace vs canonical run | 60/60 epochs bit-identical to `20260907-194818-604fa0f5` | 60/60 epochs bit-identical to `20260909-171303-95a4f542` |
| best valid MAE / epoch | 0.18415821571176638 @56 ✓ | 0.17006561887910357 @53 ✓ |
| refit test (secondary protocol) | 0.1353615188403055 ✓ | 0.13944621286727488 ✓ |
| selection-checkpoint test (new in-run primary measurement) | **0.15428431200794876** — matches the post-hoc ad-hoc value 0.154284 recorded in the v4 note §14.2 (tool cross-validation) | **0.13390139845572413** — matches ad-hoc 0.133901 |

Both guards therefore reproduce the canonical numbers exactly, and the new in-run
measurement reproduces the previously ad-hoc selection-checkpoint test values —
the tooling change is measurement-transparent. Multi-seed proceeded.

## 6. Per-seed results (Primary benchmark protocol)

Primary test = frozen validation-selected checkpoint, train-only fits, single
official-test evaluation (`test_with_selection_checkpoint_mae`). Valid = best
validation MAE of the same run. Seed-0 rows are the guard runs.

| seed | v2 valid | v2 test | v2 epoch | v4 valid | v4 test | v4 epoch | **Δtest (v2−v4)** | Δvalid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.184158 | 0.154284 | 56 | 0.170066 | 0.133901 | 53 | **+0.020383** | +0.014093 |
| 1 | 0.184569 | 0.145225 | 38 | 0.163167 | 0.132006 | 48 | **+0.013219** | +0.021402 |
| 2 | 0.175910 | 0.145472 | 59 | 0.174149 | 0.144614 | 56 | **+0.000858** | +0.001761 |
| 3 | 0.186737 | 0.139294 | 53 | 0.170421 | 0.137019 | 60 | **+0.002275** | +0.016316 |

Per-seed record fields (seed / best_valid_epoch / best_valid_mae / test_mae /
params / training_time / run_id / git_commit / config_hash) are stored in each run
manifest (`runs/2026/09/09/<run_id>/manifest.json`) and in the promoted records
(`records/runs/<run_id>.json`); run ids:

- v2: `20260909-193141-6962889e` (s0), `20260909-195530-666da952` (s1),
  `20260909-201121-e3e57f05` (s2), `20260909-202722-a1dd7a00` (s3)
- v4-hinge: `20260909-194445-182c7021` (s0), `20260909-200320-34b347bf` (s1),
  `20260909-201918-45fbe48d` (s2), `20260909-203516-04e62a28` (s3)

Seeds [0,1,2,3] fixed in advance (protocol seed list extended from [0]); no seed
was replaced or added after looking at results. Test was evaluated once per seed
after freeze; nothing was changed from test feedback.

## 7. Mean ± std and paired improvement

### Test (selection checkpoint, primary protocol)

| model | mean test MAE | std (across seeds) |
|---|---:|---:|
| compact-v2 | **0.146069** | 0.006177 |
| compact-v4-hinge | **0.136885** | 0.005552 |

Paired improvement Δᵢ = MAE_v2 − MAE_v4 (positive = v4 better):

| stat | value |
|---|---:|
| mean Δ | **+0.009184** |
| std Δ | 0.009287 |
| median Δ | +0.007747 |
| min / max | +0.000858 / +0.020383 |
| seeds where v4 beats v2 | **4/4** |
| paired t (n=4) | 1.98 (two-sided p = 0.142; 95% CI ±0.0148) |

### Validation (same runs, same pairing)

| stat | value |
|---|---:|
| v2 mean valid | 0.182844 ± 0.004759 |
| v4 mean valid | 0.169451 ± 0.004578 |
| mean paired Δvalid | **+0.013393** |
| std / median Δvalid | 0.008336 / +0.015204 |
| seeds where v4 beats v2 (valid) | **4/4** |
| paired t (n=4) | 3.21 (p = 0.049) |

Directional agreement: valid and test improvements agree in sign on all 4 seeds
(effect is not a selection artifact of one split). Statistical significance is
reported but not used as the decision criterion (n=4; CI includes 0 on test).

## 8. Subgroup confirmation (official test, selection checkpoints)

Groups (long-cycle audit label-excess definition, test n = 948 / 44 / 8):
A = no long-cycle (excess 0), B = mild (excess 1), C = extreme (excess ≥ 2).
Per-seed group MAE (Δ = v2 − v4, positive = v4 better):

| seed | v2 A | v4 A | ΔA | v2 B | v4 B | ΔB | v2 C | v4 C | ΔC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1366 | 0.1247 | +0.0119 | 0.3545 | 0.2038 | +0.1507 | 1.1431 | 0.8379 | +0.3052 |
| 1 | 0.1342 | 0.1221 | +0.0120 | 0.2941 | 0.2733 | +0.0209 | 0.6374 | 0.5268 | +0.1105 |
| 2 | 0.1293 | 0.1343 | −0.0050 | 0.3032 | 0.2350 | +0.0682 | 1.1918 | 0.8723 | +0.3195 |
| 3 | 0.1218 | 0.1286 | −0.0068 | 0.2843 | 0.2353 | +0.0491 | 1.4114 | 0.5938 | +0.8176 |

Cross-seed summary:

| group | n/test | v2 mean MAE | v4 mean MAE | mean subgroup Δ | std Δ | v4 wins |
|---|---:|---:|---:|---:|---:|---:|
| A ordinary | 948 | 0.1305 | 0.1274 | **+0.0031** | 0.0103 | 2/4 |
| B mild | 44 | 0.3090 | 0.2368 | **+0.0722** | 0.0558 | 4/4 |
| C extreme | 8 | 1.0959 | 0.7077 | **+0.3882** | 0.3017 | 4/4 |

Answer to the stage question: v4's average gain is **not only** tail — but it is
most *reliable* in the tail (B/C 4/4; C's per-seed Δ is 0.11–0.82). The bulk group
A shows mean improvement (+0.0031, within the +0.002 bulk-safety margin if read as
v4−v2 = −0.0031) but **flips sign across seeds** (s0/s1 +0.012, s2/s3 −0.005/−0.007;
worst single-seed A regression 0.0068). Under the bulk safety gate
("ordinary molecules must not clearly worsen"; flag as tradeoff only if *consistently*
worse > 0.005), A is not consistently worse → **gate passes with an explicit
instability note**: bulk behavior is seed-dependent; the two seeds with bulk
regression (s2, s3) are exactly the seeds with the smallest overall gain.

Valid subgroup (diagnostic, n = 965/30/5) behaves similarly: ΔA +0.0023 ± 0.0103
(2/4 wins), ΔB +0.168 ± 0.096 (4/4), ΔC +1.233 ± 0.875 (4/4).

## 9. Mechanism evidence (no new training): shuffle + capacity controls

Reused from the frozen seed-0 round (`notes/compact_v4_global_topology_channel.md`
§16/§22): shuffled-topology valid MAE 0.4084 vs clean 0.1701 (gain fully depends on
the true topology pairing); capacity_control (equal-capacity zeros) Δ ≈ 0.000 → the
gain is not added capacity. The 4 new hinge runs re-recorded the shuffle diagnostic
per seed (valid shuffled 0.3965–0.4084 vs clean 0.1632–0.1741), i.e. the mechanism
check reproduces on every seed without any extra training; no new capacity-control
runs were made (per stage rules §22).

## 10. Refit-robustness caveat (secondary protocol; seed-0 only)

Under the secondary train+valid refit protocol the seed-0 numbers stand exactly as
recorded: v2 refit test 0.1353615 vs v4-hinge refit test 0.1394462 (**v4 worse by
+0.0041**; guard runs re-verified both bit-identically). The stage deliberately did
**not** re-run refit for seeds 1–3 (§18): refit robustness is a separate question
and must not be mixed into the benchmark-protocol verdict, and no refit-based
tuning was performed. The two conclusions are separate:

1. Under the benchmark-comparable primary protocol, v4-hinge improves v2
   (mean paired Δtest +0.0092, 4/4 seeds).
2. Under train+valid refit, seed-0 shows v4-hinge unstable/regressed (train+valid
   refit puts the validation-only extreme tail into the training set; see v4 note
   §14.2/§18 for the mechanism).

Conclusion (1) is about the model's benchmark-comparable performance; conclusion
(2) is about its stability under a distribution-expansion protocol. They do not
cancel each other out, and (2) was not used to overrule (1).

## 11. Verdict (pre-registered rules §16)

| rule | requirement | measured | met? |
|---|---|---|---|
| Strong GO | mean Δ ≥ 0.010 and ≥3/4 seeds | +0.00918, 4/4 | **no** (mean short of 0.010) |
| **GO** | mean Δ ≥ 0.008 and majority seeds | +0.00918 ≥ 0.008, 4/4 | **yes** |
| Mild/unstable | 0.003 ≤ mean < 0.008 or strong sign flips | — | no (mean ≥ 0.008, 0 sign flips) |
| NO-GO | mean < 0.003 or majority worse | — | no |

**Verdict: GO (with instability caveats).** v4-hinge's benchmark-protocol gain
survives across 4 seeds (4/4 positive; mean +0.0092; valid agrees 4/4), but the
effect size is seed-dependent (seed 0–1 ≈ +0.013–0.020, seed 2–3 ≈ +0.001–0.002),
the 95% CI of the mean includes zero, and the bulk group A flips sign across seeds.
This is therefore **not** a Strong GO and not yet a "clean bulk improvement" claim.

---

## Q1–Q12 (final answers)

**Q1. 公开 ZINC benchmark 的 primary evaluation protocol 到底是什么？**
固定官方 split（10k/1k/1k，benchmarking-gnns 划分）；只在 train 上训练；validation
只用于模型状态选择（LR schedule / early stop / best-val checkpoint）；test 在冻结的
模型状态上**一次**评估；**没有** train+valid refit 再 test；重复 4（Dwivedi 发表版
主表）或 10（CIN）次初始化，报 mean±std。（来源：arXiv:2003.00982v4 + benchmarking-gnns
代码 + arXiv:2106.12575v2 E.4；repo 内 `docs/literature/deep/cin.md` 一致。）

**Q2. 当前 internal train+valid refit 是否属于 literature-standard protocol？**
**不属于。** 没有任何 checked 来源做 train+valid refit 后再 test。它是 repo 内部的
robustness/distribution-expansion 协议。

**Q3. 从现在开始 repository 应该把哪套 protocol 作为 primary？**
`train → validation selection → frozen checkpoint → single test`（本 note §2.3 的
Primary benchmark protocol，`test_with_selection_checkpoint_mae`），用于排名与
GO/NO-GO；refit 协议降级为 secondary（`zinc-context-gap.yaml` addendum 已写入）。

**Q4. 哪些 runs 被固定为 benchmark-comparable？**
8 个新 canonical runs（§6，seeds 0–3 × v2/hinge，全部 promoted 到
`records/runs/`）；此前 v4 note §14.2 的 post-hoc selection-test 数字（0.154284 /
0.133901）被本阶段 in-run 测量逐位复现。旧的 terminal runs（如
`20260907-194818-604fa0f5`、`20260909-171303-95a4f542`）保留为 **refit-robustness
记录**，不再称 "official test"。

**Q5. seed=0 guard 是否重新逐位复现？**
是：v2 与 v4-hinge 的 60-epoch valid traces 与 canonical runs 逐位一致，best
valid/epoch 与 refit test 全部逐位复现（§5）；新的 selection-checkpoint test 测量
同时复现了此前 ad-hoc 数字（0.15428431200794876 / 0.13390139845572413）。

**Q6. v2 multi-seed test mean ± std 是多少？**
0.146069 ± 0.006177（selection-checkpoint test，n=4）。

**Q7. v4-hinge multi-seed test mean ± std 是多少？**
0.136885 ± 0.005552。

**Q8. paired mean improvement 是多少？**
+0.009184（std 0.009287；median +0.007747；paired t = 1.98，p = 0.142）。

**Q9. 4 个 seed 中多少个 v4 优于 v2？**
**4/4**（test 与 valid 都是 4/4，方向一致）。

**Q10. ordinary / mild / extreme subgroup 的平均 improvement 各是多少？**
A（ordinary，n=948/seed）：**+0.0031** ± 0.0103（2/4 seeds）；B（mild，n=44）：
**+0.0722** ± 0.0558（4/4）；C（extreme，n=8）：**+0.3882** ± 0.3017（4/4）。

**Q11. v4 是否满足预注册 GO（mean ≥ 0.008 且 majority seeds improve）？**
满足 → **GO**（mean +0.0092 ≥ 0.008；4/4 seeds）。未达 Strong GO（mean < 0.010），
且注意 seed 2/3 增益小、A 组 sign flip → 结论带 instability caveat。

**Q12. 下一步 primary recommendation？**
**A. post-v4 residual audit**（GO 达成后的唯一推荐；先冻结 v4-hinge 作为新
baseline，重新观察其 residual 中是否出现新的 rarity / OOV / pair composition /
chemistry / objective signal，再决定是否开新机制方向）。在单独的后续 stage 之前，
**不要**堆 topology features、改 loss、修 refit、做更大模型。可选并行（非本 stage
范围）：refit stability study（seeds 1–3 的 refit 尚未跑）——但不得覆盖上面的主
结论；若 seed 1–3 的 refit 同样翻转，才考虑 refit-phase 机制研究。

---

## Files / evidence index

- Runs (local truth, git-ignored): `runs/2026/09/09/` — manifests + configs +
  traces + per-molecule valid/test predictions (JSON artifacts) for all 8 runs.
- Promoted records: `records/runs/20260909-{193141-6962889e,194445-182c7021,
  195530-666da952,200320-34b347bf,201121-e3e57f05,201918-45fbe48d,
  202722-a1dd7a00,203516-04e62a28}.json`
- Aggregated machine-readable summary:
  `tracks/ksvd/results/luyin16/ZINC_COMPACT_V4_MULTISEED_CONFIRMATION_20260909.json`
  (per-seed + paired + subgroup stats; produced by
  `experiments/luyin16/zinc_multiseed_protocol_analysis.py`)
- Protocol naming: `protocols/zinc-context-gap.yaml` (2026-09-09 addendum)
- Determinism policy: `notes/reproducibility_cpu_determinism.md`
- Tooling (additive measurement; training paths untouched): commit `0a61b23`
  (`zinc_patch_path_pooling.py`, runner metrics, protocol yaml)
- Seed-0 mechanism/control evidence (reused, not re-run): v4 note §16/§22
