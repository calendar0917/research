# REPORT — `tsb-lipo-feature-ladder-v1`

树模型 + 显式图统计：Lipophilicity 五臂特征阶梯（XGBoost，CPU，第一轮）。
所有数字来自 `analysis/tables/`（由 `report` stage 生成）与
`analysis/test/test_summary.json`；本文件只做转述，不重新计算。

- Track：`tracks/tree-statistics-boundary`
- 协议：`tsb-lipo-feature-ladder-v1`（见 `PROTOCOL.md`）
- 执行日期：2026-11-04
- 设备：本机 CPU（16 核，XGBoost `n_jobs=4`），无 GPU
- 总 CPU 墙钟：**83.2 s**（20 个 run，含调试失败）；正式 stage 合计 50.7 s

---

## 1. 本轮实际做了什么

| 阶段 | run | 内容 | 时长 |
|---|---|---|---|
| S0 | `20261104-193731-3f819450` | 干净重建全部标签无关特征块 + 3 套 Train-only WL 词表 | 6.6 s |
| P0 | `20261104-193914-3b67c806` | 11 项 CPU 正确性门槛（无 fit） | 11.2 s |
| P1 | `20261104-194100-5d76aa27` | 15 个完整 Train 模型（3 划分 × 5 臂，N=3360） | 7.7 s |
| P2 | `20261104-194122-5d86fed8` | 30 个模型（N=1000、N=2000） | 13.0 s |
| P0 | `20261104-194203-779836c3` | 11 项门槛（含 45 个 fit 的行号/标签对齐） | 11.2 s |
| P3a | `20261104-194308-864f304c` | Test 前冻结：哈希 15 个模型/预测/配置/代码/特征 | 0.05 s |
| P3b | `20261104-194319-30d83aa4` | 一次性 Test 评分（terminal，`test_access=granted`） | 0.30 s |
| Report | `20261104-194404-876236b1` 等 | 表 A–D 与小节 10 的事后修正 | < 1 s |

共同配置：XGBoost `reg:squarederror`, `hist`, CPU, 500 trees, depth 5, lr 0.03,
subsample 0.8, colsample 0.8, λ=5, `random_state=0`, `n_jobs=4`。
无 Optuna / 网格 / 早停 / 学习率搜索。

## 2. P0 正确性门槛（正式训练前）

11/11 全部通过（`analysis/p0/p0_gates.json`，`all_passed=true`）：

```
01_published_splits                    PASS  3 划分 3360/420/420，行号哈希与历史一致，两两不交，并集 4200
02_label_free_features                 PASS  把 exp 换成 N(0,1) 噪声后 240 分子的 G0/WL/Morgan 特征逐位不变
03_vocab_from_train_only               PASS  每个划分每一轮的词表都能只用官方 Train 精确重建
04_wl2_bin_equals_wl2_count_columns    PASS  列名/列序/稀疏模式一致，BIN 为 0/1 且与计数同支撑
05_wl3_low_rounds_equal_wl2            PASS  WL3 = WL2（rounds 0-2 逐元素相同）+ 恰好一个 round-3 块
06_permutation_invariance              PASS  随机原子重编号 + 完整端点翻转，200 分子特征不变
07_cache_reload_identical              PASS  G0 全体 / WL counts+binary 全体 / Morgan 200 行与全新计算逐元素一致
08_no_nan_and_explicit_policy          PASS  15 个臂×划分矩阵全有限；OOV 桶存在；解析失败为硬错误；0 例失败
09_row_and_label_alignment             PASS  45 个 fit 的 Train 行号/标签哈希/预测行号/标签全部对齐
10_model_roundtrip                     PASS  模型 .ubj 存盘重载后预测逐位一致
11_synthetic_wl_boundary               PASS  见 §8
```

pytest 侧同一实现：`uv run pytest -q tracks/tree-statistics-boundary/tests` →
**20 passed**（其中 cache 相关 9 项标记 `slow`，无 cache 时自动 skip）。

## 3. 表 A — 完整 Train（N=3,360）性能

原始标签尺度，RMSE/MAE 越低越好。`gap` = Validation RMSE − Train RMSE。

| 划分 | 臂 | Train RMSE | Valid RMSE | Test RMSE | Valid MAE | Test MAE | gap | 训练 s | 特征装配 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random122 | G0 | 0.5640 | 0.8332 | 0.8642 | 0.6455 | 0.6702 | 0.2692 | 0.22 | 0.008 |
| random122 | WL2_BIN | 0.4737 | 0.6912 | **0.7029** | 0.5278 | 0.5461 | 0.2175 | 0.39 | 0.012 |
| random122 | WL2_COUNT | 0.4597 | **0.6825** | 0.7187 | 0.5230 | 0.5547 | 0.2228 | 0.49 | 0.012 |
| random122 | WL3_COUNT | 0.4580 | **0.6815** | 0.7257 | 0.5218 | 0.5572 | 0.2235 | 0.58 | 0.016 |
| random122 | MORGAN_COUNT | 0.4803 | 0.7190 | 0.7486 | 0.5506 | 0.5803 | 0.2387 | 0.67 | 0.011 |
| random123 | G0 | 0.5713 | 0.8549 | 0.8185 | 0.6529 | 0.6466 | 0.2836 | 0.22 | 0.005 |
| random123 | WL2_BIN | 0.4677 | 0.7368 | 0.6509 | 0.5482 | 0.5020 | 0.2691 | 0.39 | 0.012 |
| random123 | WL2_COUNT | 0.4554 | 0.7397 | 0.6535 | 0.5508 | 0.5034 | 0.2842 | 0.48 | 0.012 |
| random123 | WL3_COUNT | 0.4549 | **0.7328** | 0.6550 | 0.5453 | 0.5059 | 0.2779 | 0.58 | 0.017 |
| random123 | MORGAN_COUNT | 0.4829 | 0.7405 | 0.6809 | 0.5620 | 0.5321 | 0.2576 | 0.67 | 0.011 |
| random124 | G0 | 0.5776 | 0.8870 | 0.8470 | 0.6962 | 0.6666 | 0.3094 | 0.23 | 0.005 |
| random124 | WL2_BIN | 0.4771 | **0.6859** | **0.6372** | 0.5255 | 0.5030 | 0.2088 | 0.39 | 0.011 |
| random124 | WL2_COUNT | 0.4637 | 0.6939 | 0.6430 | 0.5309 | 0.5060 | 0.2302 | 0.48 | 0.012 |
| random124 | WL3_COUNT | 0.4593 | 0.6901 | 0.6449 | 0.5304 | 0.5053 | 0.2308 | 0.58 | 0.016 |
| random124 | MORGAN_COUNT | 0.4823 | 0.7222 | 0.7001 | 0.5585 | 0.5473 | 0.2399 | 0.66 | 0.010 |

三次划分的均值 ± 标准差（**注意：三套划分使用同一批 4,200 个分子，只换了随机
划分，因此这不是三个独立外部数据集上的均值/标准差**）：

| 臂 | 列数 | Train RMSE | Valid RMSE | Test RMSE | Valid MAE | Test MAE | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 76 | 0.5710±0.0055 | 0.8583±0.0221 | 0.8432±0.0189 | 0.6649 | 0.6611 | 0.2874 |
| WL2_BIN | 1152/1151 | 0.4728±0.0039 | 0.7046±0.0228 | **0.6637±0.0283** | 0.5338 | 0.5170 | 0.2318 |
| WL2_COUNT | 1152/1151 | 0.4596±0.0034 | 0.7054±0.0247 | 0.6717±0.0335 | 0.5349 | 0.5214 | 0.2458 |
| WL3_COUNT | 1665/1664 | 0.4574±0.0018 | **0.7015±0.0225** | 0.6752±0.0360 | 0.5325 | 0.5228 | 0.2441 |
| MORGAN_COUNT | 2124 | 0.4818±0.0011 | 0.7273±0.0095 | 0.7099±0.0285 | 0.5570 | 0.5532 | 0.2454 |

Test RMSE 在 random123/124 上普遍低于 Validation（两套划分的分子难度不同），
random122 上 Test 略高于 Validation。**Test 不用于任何选择**，此处仅按预注册
要求对全部 15 个模型统一报告。

## 4. 表 B — 结构信息的增量（Validation RMSE 差值）

delta = RMSE(前) − RMSE(后)；RMSE 越低越好，**负值表示前者更好**。
每个模型对照给出固定种子 1000 次逐分子配对 bootstrap 的 95% 区间
（`seed=20261104`；这是**分子抽样**区间，不是训练种子方差）。

| 对照 | 划分 | delta RMSE | 95% CI | 三划分方向一致 |
|---|---|---:|---|---|
| `WL2_COUNT − WL2_BIN`（计数 vs 仅出现） | random122 | −0.0087 | [−0.0200, +0.0043] | — |
| | random123 | +0.0029 | [−0.0083, +0.0133] | — |
| | random124 | +0.0080 | [−0.0054, +0.0219] | — |
| | **均值** | **+0.0007** | — | **否** |
| `WL3_COUNT − WL2_COUNT`（更大局部范围） | random122 | −0.0010 | [−0.0108, +0.0080] | — |
| | random123 | −0.0068 | [−0.0159, +0.0024] | — |
| | random124 | −0.0038 | [−0.0121, +0.0055] | — |
| | **均值** | **−0.0039** | — | **是（3/3 为负）** |
| `WL2_COUNT − G0`（局部模式身份） | random122 | −0.1507 | [−0.2038, −0.0914] | — |
| | random123 | −0.1152 | [−0.1719, −0.0619] | — |
| | random124 | −0.1931 | [−0.2426, −0.1441] | — |
| | **均值** | **−0.1530** | — | **是（3/3 为负，且 3/3 CI 不含 0）** |
| `MORGAN_COUNT − WL2_COUNT`（不同局部编码） | random122 | +0.0365 | [−0.0015, +0.0748] | — |
| | random123 | +0.0008 | [−0.0224, +0.0235] | — |
| | random124 | +0.0283 | [+0.0007, +0.0518] | — |
| | **均值** | **+0.0219** | — | **是（3/3 为正，即 Morgan 更差）** |

观察（只陈述方向与差值，不做机制判断）：

- 加入 WL（round 0–2）相对纯 `G0` 的验证误差下降幅度最大，且三个划分方向一致、
  区间不含 0。
- 计数 vs 二值化的差别在 ±0.01 以内，符号在两个划分上相反（122 上计数更好，
  123/124 上二值更好），三划分方向不一致。
- round 3 在三个划分上都略降低验证误差（0.001–0.007），但三个 95% 区间都跨 0。
- Morgan count 相对 `WL2_COUNT` 在三个划分上都更差，幅度 0.001–0.037，只有
  random124 的区间不含 0。

## 5. 表 C — 训练样本量曲线（Validation RMSE）

三档是同一条 `sha256("20261104:<row id>")` 顺序的嵌套前缀；WL 词表始终来自完整
官方 Train（3,360 行）的**无标签图**，因此这是"有标签样本量"曲线，不是
"只有 N 张图可用"的实验。

三次划分均值：

| 臂 | N=1,000 | N=2,000 | N=3,360 |
|---|---:|---:|---:|
| G0 | 0.9503 | 0.8878 | 0.8583 |
| WL2_BIN | 0.7785 | 0.7308 | 0.7046 |
| WL2_COUNT | 0.7830 | 0.7299 | 0.7054 |
| WL3_COUNT | 0.7836 | 0.7280 | 0.7015 |
| MORGAN_COUNT | 0.8212 | 0.7640 | 0.7273 |

逐划分：

| 臂 | 划分 | N=1,000 | N=2,000 | N=3,360 |
|---|---|---:|---:|---:|
| G0 | random122 | 0.9101 | 0.8697 | 0.8332 |
| G0 | random123 | 0.9680 | 0.8979 | 0.8549 |
| G0 | random124 | 0.9728 | 0.8959 | 0.8870 |
| WL2_BIN | random122 | 0.7586 | 0.7186 | 0.6912 |
| WL2_BIN | random123 | 0.8143 | 0.7719 | 0.7368 |
| WL2_BIN | random124 | 0.7626 | 0.7021 | 0.6859 |
| WL2_COUNT | random122 | 0.7554 | 0.7088 | 0.6825 |
| WL2_COUNT | random123 | 0.8252 | 0.7667 | 0.7397 |
| WL2_COUNT | random124 | 0.7684 | 0.7142 | 0.6939 |
| WL3_COUNT | random122 | 0.7617 | 0.7173 | 0.6815 |
| WL3_COUNT | random123 | 0.8244 | 0.7636 | 0.7328 |
| WL3_COUNT | random124 | 0.7647 | 0.7030 | 0.6901 |
| MORGAN_COUNT | random122 | 0.8026 | 0.7560 | 0.7190 |
| MORGAN_COUNT | random123 | 0.8493 | 0.7837 | 0.7405 |
| MORGAN_COUNT | random124 | 0.8118 | 0.7522 | 0.7222 |

- 所有臂在 1,000 → 2,000 的改善都大于 2,000 → 3,360。
- 在每一档 N、每个划分上，四个含局部模式的臂都优于 `G0`。
- 在每一档 N 上，Morgan count 的验证误差都高于三个 typed-WL 臂。

## 6. 表 D — 结构覆盖与运行成本

### 6.1 各轮词表覆盖率与 OOV（每轮 512 token 上限 + 1 个 OOV 桶）

| 划分 | round | 列数 | Train OOV | Valid OOV | Test OOV | Valid 行含 OOV | Valid 非零列比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random122 | 0 | 50 | 0.00000 | 0.00000 | 0.00000 | 0.0000 | 0.820 |
| random122 | 1 | 513 | 0.04160 | 0.04239 | 0.04448 | 0.4810 | 0.951 |
| random122 | 2 | 513 | 0.45778 | 0.46454 | 0.47295 | 1.0000 | 0.990 |
| random122 | 3 | 513 | 0.72235 | 0.72324 | 0.74057 | 1.0000 | 0.969 |
| random123 | 0 | 49 | 0.00000 | 0.00009 | 0.00009 | 0.0024 | 0.878 |
| random123 | 1 | 513 | 0.04091 | 0.04073 | 0.05106 | 0.4476 | 0.951 |
| random123 | 2 | 513 | 0.45661 | 0.45612 | 0.48710 | 1.0000 | 0.988 |
| random123 | 3 | 513 | 0.71959 | 0.73515 | 0.74973 | 1.0000 | 0.951 |
| random124 | 0 | 49 | 0.00000 | 0.00000 | 0.00009 | 0.0000 | 0.837 |
| random124 | 1 | 513 | 0.04111 | 0.04240 | 0.04734 | 0.4976 | 0.940 |
| random124 | 2 | 513 | 0.45709 | 0.47288 | 0.46803 | 0.9976 | 0.981 |
| random124 | 3 | 513 | 0.72033 | 0.74579 | 0.73706 | 1.0000 | 0.959 |

- round 0：Train 只出现 48–49 种原子签名，少于 512，故保留全部（列数 49/50），
  Train OOV ≈ 0；Valid/Test 极少数行含未见签名。
- round 1：OOV ≈ 4% 的 token 出现次数，约 45–50% 的 Valid 行至少有一个 OOV token。
- round 2：OOV ≈ 46–49%，几乎所有行都触发 OOV。
- round 3：OOV ≈ 72–75%，所有行都触发 OOV。

这直接限制了 `WL3_COUNT` 的增量可解释性：round 3 的大部分 token 质量落在 OOV 桶里。
这是"每轮最多 512 个 token"这一冻结规则的直接后果，本轮不修改。

### 6.2 成本

| 项 | 值 |
|---|---|
| 一次性 tokenization（4,200 分子 × rounds 0–3） | 2.78 s |
| 特征 stage 总时长（含 G0、Morgan、3 套词表、全部缓存写盘） | 6.23 s |
| 其中 G0 | 0.32 s |
| 其中 Morgan（4,200 分子） | 0.35 s |
| 单臂/单划分特征装配（hstack） | 0.005–0.017 s |
| 单模型训练（500 trees, depth 5） | 0.22 s（G0）→ 0.67 s（Morgan/2124 列） |
| 15 个完整 Train 模型训练合计 | 7.03 s |
| 30 个小样本模型训练合计 | 12.01 s |
| 特征 stage 峰值 RSS | 1,658 MB |
| P1 stage 峰值 RSS | 720 MB |
| P2 stage 峰值 RSS | 720 MB |
| P0 门槛峰值 RSS | 1,516 MB |
| 模型文件大小 | 1.15–1.29 MB / 个（共 45 个，53 MB，git-ignored） |
| Morgan 桶冲突 | 平均 49.18 raw environments / 48.51 buckets，平均 0.66 个冲突桶/分子；1,994/4,200 行至少 1 个冲突 |

内存为 `ru_maxrss` 进程高水位（MB），是进程级而非单任务级测度。

## 7. 预算

- 上限：90 分钟 CPU 墙钟。
- 实际：**83.2 s**（20 个 run，含 7 个调试失败 run；正式 stage 合计 50.7 s）。
- 未使用远端计算，未申请 GPU。

## 8. 表达能力边界（诊断，不是预测模型）

`PROTOCOL.md` §7 的合成例子：两个 24 元环，环原子 0 = N、原子 k = O（k = 8 或 12）。
N–O 图距离 8 vs 12，但 rooted typed-WL round 0–3 的 token 多重集、`G0` 向量、
Morgan radius-2 计数向量三者都**完全相同**。因此本轮五个臂对这对图给出同一个
特征向量：它们都无法表达"两个标记原子之间的距离"这一非局部统计。

控制实验：12 元环上 d=4 vs d=6 可以被 WL 区分（`tests/test_tsb_synthetic.py`）。

## 9. Test 使用记录与历史暴露

- 本轮 Test 只被读取**一次**：run `20261104-194319-30d83aa4`（`mode=terminal`,
  `test_access=granted`），15 个模型全部报告，见 `analysis/test/test_summary.json`
  （sha256 `66b97ad1…9b29`）。
- 预注册冻结：`analysis/freeze/pre_test_freeze.json`
  （sha256 `58afd451340ebc0c57d1fe5903ff54fee2a7726ec7e713b2e519db13bdaa1970`），
  指针 `analysis/freeze/freeze_pointer.json` 与之一致；`test_not_read=true`。
- 一次 terminal 运行 `20261104-194252-514c5178` 在读取任何 Test 标签**之前**
  因配置哈希不匹配失败（见 §10），未产生 Test 数字。
- **Test 暴露**：Lipophilicity Test 标签在本仓库更早的 structural-sufficiency
  Sprint 01 / repeat 轮次中已被读取。本轮所有 Test 数字标记
  `EXPOSED_PUBLIC_TEST`，不是首次盲测，也不能被表述为新的盲测证据。
- Test 未被用于选择臂、调参、增加实验或修改协议。

## 10. 执行中的真实异常与修正（全部已记录）

| 时间点 | 现象 | 处理 | 对结果的影响 |
|---|---|---|---|
| 特征首次构建 | RDKit 2026.09.1 的 `GetMorganFingerprint` 不接受 `nBits` | 改用 `GetHashedMorganFingerprint(mol, 2, nBits=2048)` | 无正式结果；重建缓存 |
| 特征首次构建 | round-0 词表只有 48–49 个 token，< 512，导致不同划分的臂宽度相差 1 列 | 按规则"取频数最高的 512 个"字面执行（不足则全取），允许划分间宽度差 1；同一划分内所有臂/样本量列完全一致 | 无；已在表 A/B/C 标注列数 |
| gate 04 | 误把 `G0`（非 0/1）也纳入 BIN 的 0/1 检查 | 只检查 WL 子块 | 无；重跑 P0 |
| gate 09 | 字段名 `suffix` vs `split_suffix`、预测路径结构不符 | 修正 gate 读取逻辑 | 无；重跑 P0 |
| freeze v1 → v2 | 科学配置哈希把运行键 `stage` 也算进去，terminal 运行无法对齐 | v2 把 `stage`/`rebuild_features` 排除在科学配置之外；**在任何 Test 标签读取之前**重新冻结 | 无 Test 泄漏；freeze v1 仍保留在 run `20261104-194231-436224f9` |
| 表 B 首版 | 对照元组解包顺序错误，表为空 | 修正后重建（不重训） | 无模型影响 |
| 表 D 首版 | 缓存 manifest 的 coverage 是全体 4,200 行聚合，三个 role 数字相同 | 在 `report` stage 直接从冻结 CSR 重算 per-role OOV/非零列 | **Test 评分之后**修改了 `code/tsb_tables.py`（仅分析代码）；记录在 `analysis/post_freeze_edits.json` |

`analysis/post_freeze_edits.json` 明确列出：冻结后**只有** `tsb_tables.py`
的哈希变化；没有重建任何特征/词表/缓存，没有重训或重新评分任何模型，
没有 Test 预测或指标变化。由于 `test` stage 会校验代码哈希，现在重跑 `test`
会被（正确地）拒绝——任何后续轮次必须新建预注册，而不是重放本轮 Test。

## 11. 只读外部参照（不属于本协议，不可与臂混比）

仓库既有的 `ECFP4 binary + 217 RDKit 描述符 + XGBoost`（各轮各自的 Train-only
中位数填补与常数列删除）：

| 划分 | Valid RMSE | Test RMSE |
|---|---:|---:|
| random122 | 0.6532 | 0.6477 |
| random123 | 0.6909 | 0.6260 |
| random124 | 0.6353 | 0.6033 |
| 均值±std | 0.6598±0.0232 | 0.6257±0.0181 |

它**不是**本轮的一个臂：输入块不同（二值 ECFP4 + 物理化学描述符）、预处理不同、
维度不同（2265）。本轮没有用它做任何选择，也没有把两者之差解释成"纯信息增益"。

## 12. 限制与不可比因素

1. **维度不同**：臂宽 76 / 1151–1152 / 1664–1665 / 2124。`MORGAN_COUNT` 与
   `WL2_COUNT` 的差异同时包含"编码方式"和"列数"，不能当作纯信息差。
2. **词表上限 512 造成的 OOV**：round 2/3 的 OOV 质量占比约 46%/72%。
   `WL3_COUNT` 的增量因此被系统性压低，本轮数字不能回答"更大的 WL 半径是否有用"。
3. **三套划分不是独立数据集**：同一批 4,200 分子，仅随机划分不同；均值/标准差
   不能当作跨数据集泛化。
4. **单一种子**：每个 (划分, 臂, N) 只训练一次（`random_state=0`），没有训练种子
   重复；bootstrap 只刻画分子抽样，不刻画训练随机性。
5. **超参数未搜索**：500 trees / depth 5 / lr 0.03 是沿用历史基线的固定配方；
   不能断言某个臂在"最优"超参数下的相对位置。
6. **G0 的边界**：G0 只包含图上的组成/拓扑计数；它故意不含任何物理化学描述符。
   因此 G0 的低性能不等于"全局统计无用"，只等于"这套受控 G0 计数不够"。
7. **Morgan 哈希冲突**：2048 桶平均每分子 0.66 个冲突桶，1,994/4,200 行至少 1 个。
8. **Test 已暴露**：见 §9。
9. `configs/feature_spec.json` 中 WL 块 `dim` 字段的记录缺陷（恒为 1）：
   真实维度在 `arms[*].per_split[*].n_columns` 与词表 manifest 中。该字段不参与
   计算；为保持预注册文件哈希不变，本轮不修改（见 `PROTOCOL.md` §10）。

## 13. 复现

```bash
cd /home/calendar/code/research
export RESEARCH_TRACK_ROOT=tracks/tree-statistics-boundary
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/tree-statistics-boundary/code
export RESEARCH_EXTRA_RUNNERS=tsb_runner

uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "feature build"  --set stage=features
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "P0 gates"       --set stage=p0
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "P1 full Train"  --set stage=p1
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "P2 sample size" --set stage=p2
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "freeze"         --set stage=freeze
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "Test"           --set stage=test --mode terminal
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "tables"         --set stage=report

uv run pytest -q tracks/tree-statistics-boundary/tests
```

产物：

| 路径 | 内容 |
|---|---|
| `analysis/p0/features_manifest.json` | 特征块、词表、哈希、覆盖率、计时 |
| `analysis/p0/p0_gates.json` | 11 项门槛结果与证据 |
| `configs/feature_spec.json` | 五臂列定义（预注册冻结） |
| `analysis/fits/*.json` | 45 个 fit 的配置、行号、标签哈希、指标 |
| `analysis/models/*.ubj` | 45 个模型（git-ignored，哈希在 fit 记录中） |
| `analysis/predictions/*.npz` | Train/Valid 预测（行号 + 预测 + 真值） |
| `analysis/freeze/` | Test 前冻结清单与指针 |
| `analysis/test/` | 15 个 Test 记录、预测与 summary |
| `analysis/tables/` | 表 A–D CSV/JSON + 外部参照 |
| `analysis/post_freeze_edits.json` | 冻结后分析代码修改与未变项核验 |
| `runs/2026/11/04/*` | 控制面本地执行现场（git-ignored） |
| `records/runs/*` | promote 后的不可变科学记录（git-tracked） |
