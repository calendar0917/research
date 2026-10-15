# REPORT — bjoc-mmp-feasibility-v1 (Phase 0 feasibility audit)

Protocol: `bjoc-mmp-feasibility-v1` · Frozen 2026-10-15 · `labels_read = false` at freeze
Artifacts: `PROTOCOL.md`, `summary.json`, `data_integrity.json`, `phase0a_coverage.json`,
`phase0a_manifest.json`, `phase0a_pairs_{P0,P0s,P1}.parquet`, `phase0b_analysis.json`

---

## 结论摘要（中文）

**Q1. 可用的单点 MMP 数量与独立单位数？**
主方案 P0（核心 ≥5 重原子、片段 1–10 重原子）在 9,996 条可核实分子上得到
**5,039 个单点替换对**（去重前 5,074 个切割实现），对应 4,015 个无序分子对、
3,247 个分子、2,449 个不同核心、2,332 个 Murcko 骨架（其中 4,534 对跨骨架）。
P0s（核心 ≥8、片段 ≤6）110 对 / 92 分子对 / 178 分子。P1（核心 ≥3、无片段上限、
每核心至多 200 对）125,112 对（未截断理论值 2,892,616），92,304 分子对、
9,828 分子、9,566 核心、7,797 骨架。数据边界：10,000 行中 4 行因提交 SMILES 与
PyG 图仅差构型（constitutional isomer，上游 pynauty 拓扑匹配固有歧义）在**读标签前**
冻结排除，其余 9,996 行构型/原子/键数全部一致，无解析失败。

**Q2. 跨骨架的编辑规则可复用性？**
几乎为零。P0 中 5,039 对产生 5,039 条不同的严格规则 `(R,R′)`：
**support ≥2 个不同上下文的规则为 0**，support ≥2 骨架组的也为 0。P0s 同样为 0。
最宽松的 P1 中 2,892,612 个不同规则（2,892,616 个实例）里只有 **4 条** support=2，且这 4 条全部来自
同 4 个分子（3662/7669/1171/675）构成的闭环，没有独立复制；加附件环境约束后
support ≥2 的规则为 **0**。粗粒度类别（环境+元素数差）确有重复（74,612 类中
16,489 类 ≥2），但类内 Δy 标准差 1.1–1.3（y 总体 std≈2.0），符号一致率仅 0.78，
不足以支撑规则复用。结论：在该冻结单切定义与官方训练集内，**跨上下文规则复用
不存在，跨骨架效应迁移无法被估计**（不是"覆盖太小"，而是"规则不重复"）。

**Q3. 上下文差异是否超过组成/已知目标公式？**
不。ZINC-12k 的 y = 归一化(logP) + 归一化(−SA) + 归一化(−环惩罚)。只读重算三项后，
P0 的 Δy 与公式 Δ 的 **R² = 0.99934**（corr 0.99967），残差中位数 0.0069，低于
逐分子重构噪声下限 0.0134；OLS 系数 ≈（logP 1.011、−SA 0.997、−环惩罚 0.985）。
公平的组成基线 R² = 0.614（不含公式项）。因此观测到的局部替换 Δy 就是目标公式本身，
**没有可测的、超出公式的上下文偏离**。规则均值基线因 100% 测试规则在训练折外而退化
（fallback=0，R²=−0.013），与"规则不可复用"互为印证。

**Q4. ZINC-12k 的 penalized logP 是否适合作为"局部结构变化效应"研究的主对象？**
不适合。原因：(1) 标签是结构的确定函数（logP、SA、环罚），Δy 与公式 R²≈0.999，
研究"效应"等于研究已知公式；(2) 该目标为合成代理量，非实验测量性质；(3) 冻结单切
规则下跨上下文复用为零，核心问题在该数据上**不可识别**，而非"效应弱"。因此不能把
ZINC-12k MMP 对当作真实化学取代效应的证据。

**Q5. 是否有足够理由进入 Phase 1，还是停止 ZINC 转向实验测量性质？**
**停止 ZINC（NO_GO_ON_ZINC），不进入 Phase 1**（A 覆盖不足 + B 公式主导）。
若仍要研究"局部替换效应的跨骨架迁移"，需要另立协议并用**实验测量性质**（例如
lipophilicity 类数据集）重新注册；本审计不读取任何此类数据、不启动新实验。

**最终可证伪判断。** 设 H："官方 ZINC-12k train 中存在同一精确单切替换 `R→R′`
出现在 ≥2 个不同核心/骨架上下文，且 Δy 含有超出目标公式的上下文依赖。"
观测结果直接否证 H 的前提与信号：P0 重复规则 = 0（P1 亦仅 4 条且同源分子闭环），
公式 R² = 0.99934、残差 ≤ 重构噪声。若未来数据在同等冻结规则下给出 ≥50 条
support ≥2 规则且公式 R² < 0.90，本判断即被推翻——这是可复核、可反驳的。

---

## 1. Scope and goals

The question was whether a single legitimate local replacement `R → R′` in a
shared context `C` yields a property delta `Δy = y(C+R′) − y(C+R)` that is
reusable across scaffolds, or whether there is a reliable learnable context
dependence. Phase 0 asks only whether official ZINC-12k data and this target
support that question at all. No training, no model modification, no Phase 1;
CPU only (~95 s total wall time, deterministic).

Data boundary (hard): only PyG `ZINC(subset=True, split="train")` (10,000
molecules) + official-train raw `y` + the strictly aligned committed SMILES
table. Official valid/test, internal Monitor/Locked labels, ZINC-Full and
external labels were never read; Phase 0A code cannot even call the label
loader (unit-tested with a patched loader that raises).

## 2. Data integrity (label-free)

* 10,000 rows checked one-by-one against RDKit reconstructions of the PyG
  graphs at constitution level.
* 9,996/10,000 rows match exactly (same heavy-atom constitution); atom-count
  and bond-count mismatches = 0; RDKit parse failures = 0; verdict
  `PASS_WITH_FROZEN_EXCLUSIONS`.
* 4 rows (941, 2210, 6071, 9766) are constitutional isomers of the PyG graphs
  (the committed SMILES is stereo-, not constitution-identical; upstream
  matching is topology-only). Excluded **before any label read**, recorded in
  the manifest; this is a data-integrity limit, not a modelling choice.
* Independent tiered checker (hand-written SMILES parser + kekulization + VF2)
  on 2,000 deterministically sampled rows: 1,967 tier-1, 32 tier-2, 1 none
  (the known row 941); 11 rows with hand-kekulizer warnings that still verify.
* Stereo audit: 6,324/10,000 rows carry stereo descriptors, 981 have unassigned
  stereo centres, 643 double-bond stereo. Fragments/pairs are constitution-level
  (stereo stripped); no pair can be a stereo-only edit (0 such pairs).

## 3. Phase 0A — frozen structure coverage (no labels)

Extraction (frozen in `PROTOCOL.md`): one acyclic SINGLE-bond cut, one dummy per
side, both sides connected, variable fragments ≥1 heavy atom; direction fixed by
lexicographic canonical fragment key (`R` = smaller, `row_r` holds `R`);
multiple cut planes producing the same replacement deduplicated with
`pre_dedup_variants`; Murcko scaffold per row; attachment environment per pair.

| regime | definition | pairs | molecule pairs | molecules | cores | scaffolds | cross-scaffold pairs | repeated strict rules (≥2 contexts) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **P0** (primary) | core ≥5, frag 1–10 | 5,039 | 4,015 | 3,247 | 2,449 | 2,332 | 4,534 (90.0%) | **0** |
| P0s | core ≥8, frag ≤6 | 110 | 92 | 178 | 108 | 120 | 61 | **0** |
| P1 | core ≥3, cap 200/core | 125,112 | 92,304 | 9,828 | 9,566 | 7,797 | 123,682 | **4** (all from the same 4 molecules) |

P1 uncapped incidence count is exact: 2,892,616 candidate pairs; 255 cores hit the
cap. Exact rule support counts distinct cores; scaffold reuse counts distinct
scaffold-pair groups per rule (a single occurrence is not reuse).

Frozen thresholds and verdict (fixed before the label read): distinct molecules
≥1,000 ✓, pairs ≥3,000 ✓, scaffolds ≥300 ✓, rules with support ≥2 contexts ≥50 ✗
(0), rules with ≥2 scaffold-pair groups ≥20 ✗ (0) → **`MMP_COVERAGE_INSUFFICIENT`**.
Note this is not "no MMP coverage": within-pair local edits are broad (5,039
pairs across 2,332 scaffolds). It is the exact cross-context *repetition* that is
absent, and repetition is precisely what the research question requires.

## 4. Phase 0B — label analysis (after the freeze was verified)

Freeze re-checked before reading labels: module/pair-table/SMILES-table hashes
match `phase0a_manifest.json` (`labels_read = false`).

Δy on raw ZINC scale (no normalisation/filtering), P0: mean 0.221, median 0.226,
std 1.963, mean |Δy| 1.501, 55.9% positive / 44.1% negative / 0 zeros.
P1: mean 0.149, std 2.192.

Target formula audit (read-only recomputation with the GVAE `sascorer_gvae.py`
and its `fpscores.pkl.gz`; Crippen logP; NetworkX `cycle_basis` cycle penalty):

| quantity (P0) | community constants | dataset constants |
| --- | ---: | ---: |
| R²(d_formula, Δy) | 0.99934 | 0.99938 |
| residual median | 0.0069 | 0.0006 |
| per-molecule reconstruction median | 0.0067 | 0.0013 |
| noise floor (2×recon median, ≥0.01) | 0.0134 | 0.0100 |

OLS of Δy on the three scaled components: intercept −0.0006, coef logP 1.011,
coef −SA 0.997, coef −cycle 0.985, R² 0.9994. Residuals are at
per-molecule reconstruction noise; there is no measurable beyond-formula signal.

Fair baselines (5-fold leave-Murcko-scaffold-pair-out, seed 20261015; unknown-rule
fallback = 0 frozen): P0 zero R² = −0.013, global-mean R² = −0.001,
rule-mean/fallback-zero R² = −0.013 (100% of test rules unseen), composition-only
OLS R² = 0.614 (MAE 0.859 vs zero 1.501). The composition baseline is the honest
structural reference; the formula audit is a near-oracle diagnostic and is never
presented as a learned result.

Cross-scaffold statistics: only P1 has repeated rules (4 rules, 2 occurrences
each, mean Δy 3.52/0.92 with within-rule std 0.101, signs consistent), and all
four involve one 4-molecule closed set. No rule reaches support ≥3 anywhere.
Coarse classes (env + element delta) repeat descriptively (P1: 74,612 classes,
16,489 with support ≥2, max 59) but are not exact rules: within-class Δy std
1.1–1.3 and sign consistency 0.78, on a y-scale of std 2.0.

## 5. Pre-registered decisions

* **A. `MMP_COVERAGE_INSUFFICIENT`** — thresholds frozen, repetition checks fail.
* **B. `TARGET_FORMULA_DOMINATED`** — R² 0.999 ≥ 0.90 and residual median
  0.0069 ≤ 0.0134 noise floor; not softened.
* **C. `NO_GO_ON_ZINC`** — formula-dominated. `GO_TO_BASELINE_VALIDATION` was
  conditional on sufficient coverage **and** plausible context dependence; both
  fail, and it was never reachable here.

## 6. Honest limitations

* The conclusion is about **official ZINC-12k train under this exact frozen
  single-cut definition** (single acyclic single bond, constitution-level,
  core-size regimes). It does not claim that all molecular editing is
  infeasible, nor that other fragmentation rules (multi-cut, ring-cut, mmpdb
  indexing over larger data) could not create rule repetitions.
* 4 rows are excluded by a frozen upstream data-integrity limit; they cannot
  change any verdict (0.04% of rows, excluded before labels).
* The formula reconstruction uses RDKit 2026.09.1 and the audit's SA code; the
  label-generation code is not guaranteed identical, which is why per-molecule
  reconstruction quality defines the noise floor and both constant sets are
  reported. Residuals sit at that floor; the near-oracle audit is not proof of
  exact equality, but it bounds any beyond-formula signal to ≲0.01 y-units.
* Statistics units are reported as raw pairs / molecule pairs / molecules /
  scaffolds / rules; no pair-level bootstrap or naive significance testing was
  done. Coarse-class analyses are exploratory and threshold-free.

## 7. What this does and does not say

It says: on ZINC-12k, a single local replacement is almost never repeated in a
second context, and where a delta can be measured it equals the known target
formula; therefore cross-scaffold transfer of local-replacement effects cannot
be estimated here, and ZINC MMP pairs must not be used as evidence about real
chemical substituent effects. It does not say: that local structural effects
are unlearnable in principle, or that no dataset can support Phase 1. If the
question is pursued later, re-register a protocol against an experimentally
measured property (e.g. lipophilicity-class data) with the same
freeze-then-label discipline; this audit does not read or start anything.

## 8. Reproduction and artifacts

```bash
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --verify-data
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --phase0a
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --phase0b
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --summary
uv run pytest -q tracks/structure-contribution/tests/test_bjoc_mmp_feasibility_v1.py  # 14 passed
```

Frozen hashes (also in `phase0a_manifest.json` / `summary.json.artifact_hashes`):
module `c390ec728af42f974cd893e0acd482623269a42771ac2a57cf00abcb2c0e6000`,
SMILES table `04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff`,
PyG train.pt `e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b`,
pair tables `51ee4083…` (P0), `d6ec81d4…` (P0s), `cbf0a43e…` (P1).
