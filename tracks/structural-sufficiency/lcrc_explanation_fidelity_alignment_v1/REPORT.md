# LCRC 解释忠实性与结构贡献对齐（`lcrc-explanation-fidelity-alignment-v1`）

预注册协议：`PROTOCOL.md`；协议冻结哈希 `f9e283bdb38b7e790bd2bac4f37f38a2fcd05a80fa7144a476324be9892e6040`；代码提交 `fb561973ec15b1614e9ae086a86901e644772a36`。

## 0. 结论速览

**预注册状态标签**： `LOCAL_FAITHFUL_AND_RELEVANT`、`NO_LCRC_SPECIFIC_ADVANTAGE`

本轮只做冻结模型上的解释方法比较，没有训练新架构、没有新 seed、没有重选 checkpoint。主 split 是官方 Valid（420 分子），Test（420）只做方向复审，固定 128 个 Train 分子只作描述。

**主要事实**

- B（LOCAL）在 Valid 上同时通过两项：oracle enrichment `0.0318` [0.0270, 0.0367]，忠实性 advantage `0.0745` [0.0476, 0.1019]；Test 同号（enrichment `0.0340`）。
- LOCAL 的忠实性优势是同族中最小的过线者（A `0.2107`、C `0.3303`、E `0.2816`）。
- C（MASK50）比 A（GRAD）多捕获 `0.0113` 的外部质量（CI `[0.0061, 0.0168]`），低于预注册的 0.02 改进阈值；C 平均选中 5.6 根键、A 为 4.0 根，k=6 内两者只差 0.003，说明差异主要是“选得更大”。
- 跨模型外部对齐：最强 LCRC（`B_LCRC_LOCAL`）减最强 GINE（`F_GINE_MASK50`）= `-0.0004` `[-0.0070, 0.0062]` → 没有 LCRC 特有优势。
- 拓扑基线 D 在 LCRC 上比随机子图**更不**影响预测（advantage `-0.1035`），在 GINE 上却为正；同一批子图在两族上的预测依赖方向不同。
- 在 LOCAL 的 420 个可评分子里，外部对齐与预测依赖同时为正的只有 `46.2%`，“仅 oracle”`29.5%` > “仅忠实性”`12.4%`。

## 1. 冻结资产与身份复核

协议冻结哈希 `f9e283bdb38b7e790bd2bac4f37f38a2fcd05a80fa7144a476324be9892e6040`，代码提交 `fb561973ec15b1614e9ae086a86901e644772a36`，环境：python `3.12.15` / torch `2.5.1+cu124` / cuda `False`。

| 检查 | 结果 |
| --- | --- |
| cache_identity | PASS |
| asset_identity | PASS |
| split_identity | PASS |
| checkpoint_identity_and_replay | PASS |
| wc_glop_graph_closure | PASS |
| wc_glop_main_scope_closure | PASS |
| wc_glop_pair_closure | PASS |

执行阶段部署的提交：`aa2c0890`（fields；2 个阶段）、`fb561973`（pool, select；5 个阶段）；`protocol_freeze.json` 冻结了执行时的代码哈希。本报告由提交 `fc16e874d52d4daf9d9f05f2319aa54856c7a326` 的分析代码生成。相对冻结时被改动的是 `lfa_aux.py`, `lfa_common.py`, `lfa_gates.py`, `lfa_main.py`, `lfa_manifest.py`, `lfa_report.py`。这些改动只在聚合/门禁/报告层（`aggregation_replay` 门禁、Train-128 分层过滤、缺失计数、artifact 清单），不触及候选生成、选择分数或模型前向；`pool_main/pool_aux` 与四个 select 记录的 SHA-256 与执行时一致（第 12 节）。

| checkpoint | epoch | valid RMSE（身份记录） | sha256 | 身份+重放 | 重放 valid RMSE |
| --- | --- | --- | --- | --- | --- |
| L_WC_LOGP | 333 | 0.1144 | 4522a6361909… | PASS | 0.1144 |
| T_WC_LOGP | 355 | 0.1033 | 5e812d7e4581… | PASS | 0.1033 |
| L_lipophilicity | 124 | 0.7069 | 9ef542e59196… | PASS | 0.7069 |
| G0_lipophilicity | 183 | 0.6161 | 04282be531bd… | PASS | 0.6161 |

WC_LOGP 原子贡献闭合：图级 `max_abs_error = 0.00000000`；主范围 968 个分子（闭合 968，最大误差 0.0000000000）；823 对端点（最大误差 0.0000000000）全部闭合。

## 2. 正确性门禁

| 门禁 | 结果 | 错误 |
| --- | --- | --- |
| `aggregation_replay` | PASS |  |
| `all_ones` | PASS |  |
| `dose_path` | PASS |  |
| `finite_difference` | PASS |  |
| `gine_batched_reference` | PASS |  |
| `label_blind` | PASS |  |
| `oracle_permutation` | PASS |  |
| `pool_identity` | PASS |  |
| `renumbering` | PASS |  |
| `weight_semantics` | PASS |  |

总门禁：**全部通过**。门禁失败会停止受影响的方法；本轮没有任何门禁被跳过或放宽。

剂量路径闭合（m=1 → 0，沿 64 步线性路径积分梯度）：示例 lcrc `wc` 分子闭合误差 `0.00001633`（容差 `0.0040`），中点 m=0.5 与 `DOSE_GATE` 一致。

## 3. 共享候选池

| scope | 分子数 | 平均候选数 | 最小 | 最大 | sha256 |
| --- | --- | --- | --- | --- | --- |
| aux | 704 | 185.7 | 113 | 229 | `c18cf56ce43e…` |
| main | 968 | 188.2 | 109 | 231 | `38090b675d92…` |

四个生成器（LCRC-GRAD / LCRC-LOCAL / GINE-GRAD / 拓扑 `n_s2_affected`）用同一套冻结 beam 搜索产生候选，合并后按物理键 ID 元组去重；两个 GPU worker 只读这一份池文件，并在入口校验 SHA-256。候选生成过程不读取标签、编辑区、配对信息或 oracle 贡献（见门禁 `label_blind`）。

## 4. 外部结构贡献对齐（WC_LOGP oracle）

`capture(S) = Σ_{v∈atoms(S)} |c_v| / Σ_v |c_v|`；`oracle_enrichment = capture(Top1) − median(capture(16 个同键数、同原子数、连通随机子图))`。这是本轮唯一与模型无关的外部结构贡献口径。

### Valid（主）

| 方法 | n | oracle 可评 | 平均 k | capture 均值 | enrichment 均值 | 95% CI（均值） | 中位数 | IQR | >0 比例 | =0 比例 | Top5% 占比 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 420 | 419 | 4.00 | 0.2060 | 0.0106 | [0.0056, 0.0155] | 0.0067 | [-0.0236, 0.0423] | 57.5% | 0.5% | 25% | OK |
| B LCRC-LOCAL | 420 | 420 | 4.76 | 0.2564 | 0.0318 | [0.0270, 0.0367] | 0.0301 | [0.0006, 0.0627] | 75.7% | 1.0% | 18% | OK |
| C LCRC-MASK50 | 420 | 418 | 5.62 | 0.2776 | 0.0218 | [0.0169, 0.0268] | 0.0174 | [-0.0125, 0.0548] | 66.5% | 0.7% | 21% | OK |
| D TOPOLOGY（LCRC 评价） | 420 | 420 | 6.00 | 0.2777 | 0.0009 | [-0.0039, 0.0057] | 0.0004 | [-0.0293, 0.0316] | 50.5% | 0.0% | 30% | OK |
| D TOPOLOGY（GINE 评价） | 420 | 420 | 6.00 | 0.2777 | 0.0009 | [-0.0039, 0.0057] | 0.0004 | [-0.0293, 0.0316] | 50.5% | 0.0% | 30% | OK |
| E GINE-GRAD | 420 | 420 | 4.98 | 0.2658 | 0.0302 | [0.0252, 0.0352] | 0.0254 | [-0.0041, 0.0652] | 70.7% | 0.5% | 19% | OK |
| F GINE-MASK50 | 420 | 418 | 5.97 | 0.3039 | 0.0317 | [0.0270, 0.0367] | 0.0271 | [-0.0021, 0.0637] | 73.7% | 0.5% | 19% | OK |

### Test（复审）

| 方法 | n | oracle 可评 | 平均 k | capture 均值 | enrichment 均值 | 95% CI（均值） | 中位数 | IQR | >0 比例 | =0 比例 | Top5% 占比 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 420 | 419 | 4.06 | 0.2109 | 0.0147 | [0.0097, 0.0198] | 0.0150 | [-0.0173, 0.0496] | 60.4% | 0.7% | 23% | OK |
| B LCRC-LOCAL | 420 | 420 | 4.92 | 0.2623 | 0.0340 | [0.0286, 0.0393] | 0.0348 | [0.0027, 0.0648] | 76.4% | 0.5% | 17% | OK |
| C LCRC-MASK50 | 420 | 419 | 5.69 | 0.2778 | 0.0212 | [0.0163, 0.0262] | 0.0179 | [-0.0064, 0.0506] | 69.7% | 0.5% | 23% | OK |
| D TOPOLOGY（LCRC 评价） | 420 | 420 | 6.00 | 0.2819 | 0.0090 | [0.0041, 0.0141] | 0.0075 | [-0.0262, 0.0385] | 55.0% | 0.2% | 27% | OK |
| D TOPOLOGY（GINE 评价） | 420 | 420 | 6.00 | 0.2819 | 0.0090 | [0.0041, 0.0141] | 0.0075 | [-0.0262, 0.0385] | 55.0% | 0.2% | 27% | OK |
| E GINE-GRAD | 420 | 419 | 5.06 | 0.2698 | 0.0341 | [0.0289, 0.0399] | 0.0273 | [0.0013, 0.0660] | 75.7% | 0.5% | 22% | OK |
| F GINE-MASK50 | 420 | 420 | 5.97 | 0.3063 | 0.0367 | [0.0316, 0.0419] | 0.0322 | [0.0053, 0.0652] | 78.1% | 0.5% | 20% | OK |

### 固定 128 Train（描述）

| 方法 | n | oracle 可评 | 平均 k | capture 均值 | enrichment 均值 | 95% CI（均值） | 中位数 | IQR | >0 比例 | =0 比例 | Top5% 占比 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 128 | 127 | 3.81 | 0.1942 | 0.0026 | [-0.0084, 0.0142] | -0.0058 | [-0.0386, 0.0366] | 42.5% | 0.0% | 37% | OK |
| B LCRC-LOCAL | 128 | 128 | 4.80 | 0.2438 | 0.0158 | [0.0045, 0.0272] | 0.0156 | [-0.0121, 0.0565] | 65.6% | 0.0% | 22% | OK |
| C LCRC-MASK50 | 128 | 128 | 5.52 | 0.2728 | 0.0134 | [0.0029, 0.0251] | 0.0120 | [-0.0308, 0.0522] | 57.8% | 0.8% | 30% | OK |
| D TOPOLOGY（LCRC 评价） | 128 | 128 | 6.00 | 0.2895 | 0.0108 | [0.0016, 0.0210] | 0.0049 | [-0.0194, 0.0414] | 57.8% | 0.0% | 34% | OK |
| D TOPOLOGY（GINE 评价） | 128 | 128 | 6.00 | 0.2895 | 0.0108 | [0.0016, 0.0210] | 0.0049 | [-0.0194, 0.0414] | 57.8% | 0.0% | 34% | OK |
| E GINE-GRAD | 128 | 128 | 4.76 | 0.2670 | 0.0341 | [0.0244, 0.0443] | 0.0281 | [0.0000, 0.0650] | 75.0% | 0.8% | 23% | OK |
| F GINE-MASK50 | 128 | 128 | 5.95 | 0.3170 | 0.0376 | [0.0283, 0.0475] | 0.0330 | [0.0004, 0.0692] | 75.0% | 0.8% | 21% | OK |

缺失说明：oracle enrichment 记为缺失的 (分子, 方法) 记录为 Valid（主） 5 条、Test（复审） 3 条、固定 128 Train（描述） 1 条；其中 9/9 条的 `control_status = INSUFFICIENT_RANDOM`（Top-1 太大，找不到同键数、同原子数的**不同**连通子图，`n_controls = 0`）；`capture_top1` 本身缺失（零分母）的有 0 条。按预注册规则这些记为 missing，没有放宽匹配条件，也没有用 0 代替。

固定 128 个 Train 分子只作描述性对照（它们参与过四轮之前的训练），CI 与 Test 一样不作为独立确认；把它们列出来是为了检查 Valid/Test 的排序是否在这批分子上也出现。

## 5. 模型预测忠实性

`delta0 = pred(m=1) − pred(m_S=0)`（S 上键的门控置 0，其余键保持 1）；`fidelity_advantage = |delta0(Top1)| − median(|delta0(16 个随机对照)|)`，全部在标准化尺度（target std）上报告。`half_over_full = |delta0.5| / |delta0|` 检验半剂量一致性；`n_s2_affected` 是同一子图影响的 S2 路径数。LCRC 与 GINE 的门控语义不同，**忠实性只在同族内比较**。

### Valid（主）

| 方法 | 可评 | 平均 k | advantage（std） | 95% CI | \|delta0\|（std） | 95% CI | 半/全剂量中位数 | 同号比例 | n_s2_affected 中位数 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 419 | 4.00 | 0.2107 | [0.1863, 0.2342] | 0.5148 | [0.4884, 0.5412] | 0.5154 | 92.1% | 16.0 | OK |
| B LCRC-LOCAL | 420 | 4.76 | 0.0745 | [0.0476, 0.1019] | 0.4307 | [0.4026, 0.4597] | 0.3137 | 90.0% | 20.0 | OK |
| C LCRC-MASK50 | 418 | 5.62 | 0.3303 | [0.3002, 0.3594] | 0.7253 | [0.6967, 0.7523] | 0.4670 | 98.8% | 24.0 | OK |
| D TOPOLOGY（LCRC 评价） | 420 | 6.00 | -0.1035 | [-0.1288, -0.0808] | 0.3060 | [0.2842, 0.3254] | 0.4093 | 83.8% | 34.0 | OK |
| D TOPOLOGY（GINE 评价） | 420 | 6.00 | 0.1516 | [0.1285, 0.1728] | 0.6352 | [0.6072, 0.6616] | 0.4383 | 96.9% | 34.0 | OK |
| E GINE-GRAD | 420 | 4.98 | 0.2816 | [0.2579, 0.3061] | 0.6850 | [0.6545, 0.7147] | 0.5567 | 96.9% | 24.0 | OK |
| F GINE-MASK50 | 418 | 5.97 | 0.4494 | [0.4286, 0.4722] | 0.9300 | [0.9054, 0.9555] | 0.5071 | 99.8% | 26.0 | OK |

### Test（复审）

| 方法 | 可评 | 平均 k | advantage（std） | 95% CI | \|delta0\|（std） | 95% CI | 半/全剂量中位数 | 同号比例 | n_s2_affected 中位数 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 419 | 4.06 | 0.2085 | [0.1852, 0.2334] | 0.5300 | [0.5030, 0.5582] | 0.5145 | 93.6% | 17.0 | OK |
| B LCRC-LOCAL | 420 | 4.92 | 0.0948 | [0.0689, 0.1211] | 0.4487 | [0.4200, 0.4778] | 0.2818 | 88.8% | 20.0 | OK |
| C LCRC-MASK50 | 419 | 5.69 | 0.3510 | [0.3236, 0.3787] | 0.7597 | [0.7323, 0.7848] | 0.4479 | 99.0% | 24.0 | OK |
| D TOPOLOGY（LCRC 评价） | 420 | 6.00 | -0.0885 | [-0.1118, -0.0617] | 0.3283 | [0.3068, 0.3510] | 0.4069 | 86.0% | 34.0 | OK |
| D TOPOLOGY（GINE 评价） | 420 | 6.00 | 0.1674 | [0.1443, 0.1912] | 0.6345 | [0.6054, 0.6637] | 0.4412 | 93.3% | 34.0 | OK |
| E GINE-GRAD | 419 | 5.06 | 0.2888 | [0.2647, 0.3137] | 0.6937 | [0.6614, 0.7260] | 0.5463 | 96.4% | 24.0 | OK |
| F GINE-MASK50 | 420 | 5.97 | 0.4713 | [0.4509, 0.4928] | 0.9454 | [0.9200, 0.9716] | 0.4803 | 99.8% | 28.0 | OK |

### 固定 128 Train（描述）

| 方法 | 可评 | 平均 k | advantage（std） | 95% CI | \|delta0\|（std） | 95% CI | 半/全剂量中位数 | 同号比例 | n_s2_affected 中位数 | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 127 | 3.81 | 0.1738 | [0.1282, 0.2194] | 0.4835 | [0.4333, 0.5355] | 0.5662 | 90.6% | 16.0 | OK |
| B LCRC-LOCAL | 128 | 4.80 | 0.0492 | [0.0012, 0.0949] | 0.4036 | [0.3518, 0.4530] | 0.3133 | 84.4% | 18.0 | OK |
| C LCRC-MASK50 | 128 | 5.52 | 0.2990 | [0.2502, 0.3476] | 0.7275 | [0.6761, 0.7797] | 0.4498 | 96.9% | 24.0 | OK |
| D TOPOLOGY（LCRC 评价） | 128 | 6.00 | -0.0819 | [-0.1244, -0.0412] | 0.3619 | [0.3232, 0.3993] | 0.3689 | 84.4% | 32.0 | OK |
| D TOPOLOGY（GINE 评价） | 128 | 6.00 | 0.2005 | [0.1598, 0.2413] | 0.6759 | [0.6253, 0.7269] | 0.4346 | 99.2% | 32.0 | OK |
| E GINE-GRAD | 128 | 4.76 | 0.2481 | [0.2043, 0.2921] | 0.6531 | [0.5948, 0.7105] | 0.5471 | 96.9% | 24.0 | OK |
| F GINE-MASK50 | 128 | 5.95 | 0.4553 | [0.4190, 0.4894] | 0.9495 | [0.9056, 0.9914] | 0.4713 | 100.0% | 26.0 | OK |

### S2 结构分解与 LCRC 局部/rho 分解（Valid，中位数）

| 方法 | n | n_s2_internal | n_s2_boundary | n_s2_affected | delta_local（std） | delta_rho（std） | max 闭合误差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 420 | 6.0 | 10.0 | 16.0 | 0.3271 | 0.0221 | 0.00000100 |
| B LCRC-LOCAL | 420 | 10.0 | 8.0 | 20.0 | 0.2361 | 0.0469 | 0.00000110 |
| C LCRC-MASK50 | 420 | 12.0 | 12.0 | 24.0 | 0.6001 | 0.0641 | 0.00000115 |
| E GINE-GRAD | 420 | 10.0 | 14.0 | 24.0 | — | — | — |
| F GINE-MASK50 | 420 | 12.0 | 14.0 | 26.0 | — | — | — |

`delta_local + delta_rho = delta0` 是 LCRC 的门控分解（`delta_rho` 为 GINE 不适用，故留空）；闭合误差是逐分子的 `||delta0 − delta_local − delta_rho||`，全部远小于目标尺度。`n_s2_*` 说明同一组键在 LCRC 内部命中的 S2 路径类型，不是因果量。

## 6. 同一分子上：外部对齐与预测依赖是否落在同一结构

| 方法 | 两项均可评 | 两项均 >0 | 仅 oracle | 仅忠实性 | 两项都未过 | 两项均 >0 比例 |
| --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 419 | 210 | 31 | 132 | 46 | 50.1% |
| B LCRC-LOCAL | 420 | 194 | 124 | 52 | 50 | 46.2% |
| C LCRC-MASK50 | 418 | 241 | 37 | 113 | 27 | 57.7% |
| D TOPOLOGY（LCRC 评价） | 420 | 70 | 142 | 68 | 140 | 16.7% |
| D TOPOLOGY（GINE 评价） | 420 | 173 | 39 | 139 | 69 | 41.2% |
| E GINE-GRAD | 420 | 279 | 18 | 87 | 36 | 66.4% |
| F GINE-MASK50 | 418 | 308 | 0 | 106 | 4 | 73.7% |

“仅 oracle” 表示该方法的 Top-1 子图捕获了更多外部原子贡献，但在模型预测上并不比随机子图更敏感；“仅忠实性” 相反。两者大量分离即说明“模型依赖的结构”和“真实结构贡献所在”不是同一组键。

## 7. 方法间配对比较与按候选规模分层

| split | 配对差（oracle enrichment） | n | 均值差 | 95% CI | >0 比例 |
| --- | --- | --- | --- | --- | --- |
| Valid（主） | A LCRC-GRAD − B LCRC-LOCAL | 419 | -0.0213 | [-0.0280, -0.0146] | 39.4% |
| Valid（主） | A LCRC-GRAD − C LCRC-MASK50 | 418 | -0.0113 | [-0.0168, -0.0061] | 44.0% |
| Valid（主） | A LCRC-GRAD − E GINE-GRAD | 419 | -0.0196 | [-0.0254, -0.0140] | 39.1% |
| Valid（主） | A LCRC-GRAD − F GINE-MASK50 | 417 | -0.0212 | [-0.0274, -0.0151] | 36.5% |
| Valid（主） | B LCRC-LOCAL − C LCRC-MASK50 | 418 | 0.0102 | [0.0031, 0.0171] | 53.6% |
| Valid（主） | B LCRC-LOCAL − E GINE-GRAD | 420 | 0.0016 | [-0.0054, 0.0087] | 48.3% |
| Valid（主） | B LCRC-LOCAL − F GINE-MASK50 | 418 | -0.0004 | [-0.0070, 0.0062] | 48.8% |
| Valid（主） | C LCRC-MASK50 − E GINE-GRAD | 418 | -0.0084 | [-0.0144, -0.0022] | 45.0% |
| Valid（主） | C LCRC-MASK50 − F GINE-MASK50 | 416 | -0.0098 | [-0.0158, -0.0035] | 43.5% |
| Valid（主） | E GINE-GRAD − F GINE-MASK50 | 418 | -0.0017 | [-0.0069, 0.0036] | 49.0% |
| Test（复审） | A LCRC-GRAD − B LCRC-LOCAL | 419 | -0.0191 | [-0.0259, -0.0119] | 38.9% |
| Test（复审） | A LCRC-GRAD − C LCRC-MASK50 | 419 | -0.0065 | [-0.0119, -0.0012] | 47.3% |
| Test（复审） | A LCRC-GRAD − E GINE-GRAD | 418 | -0.0192 | [-0.0252, -0.0136] | 34.7% |
| Test（复审） | A LCRC-GRAD − F GINE-MASK50 | 419 | -0.0218 | [-0.0282, -0.0154] | 35.1% |
| Test（复审） | B LCRC-LOCAL − C LCRC-MASK50 | 419 | 0.0127 | [0.0066, 0.0190] | 56.3% |
| Test（复审） | B LCRC-LOCAL − E GINE-GRAD | 419 | -0.0003 | [-0.0081, 0.0071] | 49.9% |
| Test（复审） | B LCRC-LOCAL − F GINE-MASK50 | 420 | -0.0027 | [-0.0093, 0.0038] | 49.0% |
| Test（复审） | C LCRC-MASK50 − E GINE-GRAD | 418 | -0.0131 | [-0.0190, -0.0075] | 39.5% |
| Test（复审） | C LCRC-MASK50 − F GINE-MASK50 | 419 | -0.0154 | [-0.0204, -0.0097] | 37.7% |
| Test（复审） | E GINE-GRAD − F GINE-MASK50 | 419 | -0.0026 | [-0.0073, 0.0023] | 48.0% |

### 按候选键数 k 分层（Valid）

| 方法 | k | n | oracle 可评 | enrichment 均值 | 中位数 | >0 | advantage 均值 | advantage >0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 2 | 122 | 122 | 0.0006 | 0.0020 | 52.5% | 0.1046 | 76.2% |
| A LCRC-GRAD | 3 | 70 | 70 | 0.0092 | 0.0079 | 55.7% | 0.1878 | 87.1% |
| A LCRC-GRAD | 4 | 47 | 47 | 0.0136 | -0.0008 | 48.9% | 0.1836 | 76.6% |
| A LCRC-GRAD | 5 | 46 | 45 | 0.0038 | 0.0066 | 55.6% | 0.1596 | 71.1% |
| A LCRC-GRAD | 6 | 135 | 135 | 0.0216 | 0.0151 | 66.7% | 0.3449 | 88.9% |
| B LCRC-LOCAL | 2 | 57 | 57 | 0.0371 | 0.0378 | 84.2% | 0.0104 | 49.1% |
| B LCRC-LOCAL | 3 | 62 | 62 | 0.0390 | 0.0371 | 83.9% | 0.0841 | 72.6% |
| B LCRC-LOCAL | 4 | 31 | 31 | 0.0513 | 0.0537 | 80.6% | 0.1048 | 61.3% |
| B LCRC-LOCAL | 5 | 43 | 43 | 0.0478 | 0.0476 | 83.7% | 0.1376 | 65.1% |
| B LCRC-LOCAL | 6 | 227 | 227 | 0.0227 | 0.0181 | 69.2% | 0.0718 | 55.5% |
| C LCRC-MASK50 | 2 | 10 | 10 | -0.0439 | -0.0472 | 0.0% | 0.0657 | 70.0% |
| C LCRC-MASK50 | 3 | 11 | 11 | -0.0098 | -0.0023 | 45.5% | 0.1163 | 63.6% |
| C LCRC-MASK50 | 4 | 21 | 21 | 0.0294 | 0.0305 | 76.2% | 0.3292 | 81.0% |
| C LCRC-MASK50 | 5 | 44 | 42 | 0.0186 | 0.0083 | 71.4% | 0.3216 | 85.7% |
| C LCRC-MASK50 | 6 | 334 | 334 | 0.0247 | 0.0236 | 68.0% | 0.3464 | 85.9% |
| D TOPOLOGY（LCRC 评价） | 2 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（LCRC 评价） | 3 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（LCRC 评价） | 4 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（LCRC 评价） | 5 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（LCRC 评价） | 6 | 420 | 420 | 0.0009 | 0.0004 | 50.5% | -0.1035 | 32.9% |
| D TOPOLOGY（GINE 评价） | 2 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（GINE 评价） | 3 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（GINE 评价） | 4 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（GINE 评价） | 5 | 0 | 0 | — | — | — | — | — |
| D TOPOLOGY（GINE 评价） | 6 | 420 | 420 | 0.0009 | 0.0004 | 50.5% | 0.1516 | 74.3% |
| E GINE-GRAD | 2 | 41 | 41 | 0.0192 | 0.0156 | 73.2% | 0.1934 | 80.5% |
| E GINE-GRAD | 3 | 51 | 51 | 0.0146 | 0.0087 | 54.9% | 0.2450 | 86.3% |
| E GINE-GRAD | 4 | 37 | 37 | 0.0364 | 0.0337 | 78.4% | 0.2760 | 83.8% |
| E GINE-GRAD | 5 | 38 | 38 | 0.0460 | 0.0500 | 73.7% | 0.2752 | 92.1% |
| E GINE-GRAD | 6 | 253 | 253 | 0.0318 | 0.0306 | 71.9% | 0.3051 | 88.1% |
| F GINE-MASK50 | 2 | 0 | 0 | — | — | — | — | — |
| F GINE-MASK50 | 3 | 1 | 1 | 0.0133 | 0.0133 | 100.0% | 0.3302 | 100.0% |
| F GINE-MASK50 | 4 | 2 | 2 | 0.0333 | 0.0333 | 100.0% | 0.3348 | 100.0% |
| F GINE-MASK50 | 5 | 7 | 7 | 0.0192 | -0.0212 | 42.9% | 0.2622 | 85.7% |
| F GINE-MASK50 | 6 | 410 | 408 | 0.0320 | 0.0274 | 74.0% | 0.4535 | 99.3% |

分层用来检查均值是否由少数大 k（更原子、更容易覆盖任何区域）的分子主导；若 enrichment 只在 k 较大时为正，就不能把它读成“小片段定位成功”。

- 小片段上差距最大：k=2 时 B 的 enrichment 为 `0.0371`（>0 比例 `84.2%`），A 只有 `0.0006`（`52.5%`）；B 在 k=2–5 都稳定在 `0.0390`–`0.0513` 一线，到 k=6 才回落到 `0.0227`（k=6 的 227 个分子多是大候选，任何方法都容易命中）。
- MASK50 在小片段上**反选**：k=2 时 C 的 enrichment 为 `-0.0439`，>0 比例 `0.0%`；它靠 k=6 的 334 个分子（`0.0247`）把总均值拉回正数。这与“未归一化的 |Δy| 偏好大子图”一致。
- 所以第 4/5 节的汇总均值必须与这张表一起读：LOCAL 的优势是真的出现在最小的候选上，而 C/F 的汇总优势主要来自 k=6。

**规模组成**（Valid）：A LCRC-GRAD 平均 k `4.00`、k=6 占 `32.1%`；B LCRC-LOCAL 平均 k `4.76`、k=6 占 `54.0%`；C LCRC-MASK50 平均 k `5.62`、k=6 占 `79.5%`；D TOPOLOGY（LCRC 评价） 平均 k `6.00`、k=6 占 `100.0%`；D TOPOLOGY（GINE 评价） 平均 k `6.00`、k=6 占 `100.0%`；E GINE-GRAD 平均 k `4.98`、k=6 占 `60.2%`；F GINE-MASK50 平均 k `5.97`、k=6 占 `97.6%`

这说明 C/F 的汇总 enrichment 有很大一部分来自“选更大的子图”，而不是同尺寸内更准；A 的选择集中在较小 k，但小 k 上的外部对齐接近 0。D 在每个分子上都选了 k=6 候选（分数 `n_s2_affected` 随键数单调），因此 D 只能读作“最大子图基线”，不能读作一个独立的小片段发现器。

## 8. LCRC 与 GINE：同一外部 oracle 下的比较

| 方法 | enrichment 均值 | 95% CI | capture 均值 | 可评 |
| --- | --- | --- | --- | --- |
| A LCRC-GRAD | 0.0106 | [0.0056, 0.0155] | 0.2060 | 419 |
| B LCRC-LOCAL | 0.0318 | [0.0270, 0.0367] | 0.2564 | 420 |
| C LCRC-MASK50 | 0.0218 | [0.0169, 0.0268] | 0.2776 | 418 |
| E GINE-GRAD | 0.0302 | [0.0252, 0.0352] | 0.2658 | 420 |
| F GINE-MASK50 | 0.0317 | [0.0270, 0.0367] | 0.3039 | 418 |

外部对齐最强 LCRC 方法 `B_LCRC_LOCAL` 与最强 GINE 方法 `F_GINE_MASK50` 的分子级配对差：均值 `-0.0004`，95% CI `[-0.0070, 0.0062]`，n = 418。

注：跨模型**不比较** `|delta0|` 的绝对值——两个族的门控读的是不同消息通道，原始响应尺度不可比；跨模型结论只来自与模型无关的 oracle capture/enrichment。

## 9. 辅助：823 对 Lipophilicity 描述性诊断

固定 823 对（1646 个端点、231 个共享端点分量）来自上一轮已冻结的配对目录；本轮不新增配对、不按标签筛对、不新增 pair-aware 解释器。Top-1 与编辑区的重叠只用来说明“解释落在哪里”，编辑区本身是结构差异，不是因果性质真值。

### Top-1 与编辑区的重叠（相对同尺寸随机子图的 advantage）

| 切片 | 方法 | 端点 | 分量 | jaccard(Top1) | jaccard(随机) | adv | adv CI 下界 | atom recall adv | bond recall adv | 边界命中 adv | 支持度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全部 | A LCRC-GRAD | 1646 | 231 | 0.0900 | 0.0785 | 0.0116 | -0.0105 | 0.0102 | 0.0051 | 0.0241 | OK |
| 全部 | B LCRC-LOCAL | 1646 | 231 | 0.1399 | 0.0754 | 0.0645 | 0.0355 | 0.0793 | 0.0891 | 0.0278 | OK |
| 全部 | C LCRC-MASK50 | 1646 | 231 | 0.0652 | 0.0767 | -0.0115 | -0.0252 | -0.0190 | -0.0272 | 0.0002 | OK |
| 全部 | D TOPOLOGY | 1646 | 231 | 0.0686 | 0.0796 | -0.0110 | -0.0275 | 0.0022 | -0.0890 | 0.1465 | OK |
| 全部 | E GINE-GRAD | 1646 | 231 | 0.0980 | 0.0755 | 0.0225 | 0.0017 | 0.0548 | 0.0437 | 0.0906 | OK |
| 全部 | F GINE-MASK50 | 1646 | 231 | 0.0832 | 0.0735 | 0.0098 | -0.0097 | 0.0277 | 0.0210 | 0.0473 | OK |
| Train–Train | A LCRC-GRAD | 1582 | 228 | 0.0922 | 0.0776 | 0.0146 | -0.0074 | 0.0131 | 0.0069 | 0.0273 | OK |
| Train–Train | B LCRC-LOCAL | 1582 | 228 | 0.1435 | 0.0745 | 0.0690 | 0.0393 | 0.0849 | 0.0942 | 0.0313 | OK |
| Train–Train | C LCRC-MASK50 | 1582 | 228 | 0.0674 | 0.0763 | -0.0088 | -0.0232 | -0.0147 | -0.0242 | 0.0067 | OK |
| Train–Train | D TOPOLOGY | 1582 | 228 | 0.0669 | 0.0792 | -0.0122 | -0.0294 | -0.0007 | -0.0914 | 0.1392 | OK |
| Train–Train | E GINE-GRAD | 1582 | 228 | 0.0994 | 0.0744 | 0.0251 | 0.0041 | 0.0580 | 0.0467 | 0.0904 | OK |
| Train–Train | F GINE-MASK50 | 1582 | 228 | 0.0843 | 0.0720 | 0.0122 | -0.0075 | 0.0309 | 0.0242 | 0.0479 | OK |
| 含非 Train 端点 | A LCRC-GRAD | 64 | 11 | 0.0364 | 0.0993 | -0.0629 | -0.1014 | -0.0618 | -0.0388 | -0.0547 | LOW_SUPPORT |
| 含非 Train 端点 | B LCRC-LOCAL | 64 | 11 | 0.0510 | 0.0993 | -0.0482 | -0.0888 | -0.0586 | -0.0385 | -0.0586 | LOW_SUPPORT |
| 含非 Train 端点 | C LCRC-MASK50 | 64 | 11 | 0.0109 | 0.0876 | -0.0767 | -0.0943 | -0.1231 | -0.1027 | -0.1602 | LOW_SUPPORT |
| 含非 Train 端点 | D TOPOLOGY | 64 | 11 | 0.1097 | 0.0895 | 0.0202 | -0.0431 | 0.0752 | -0.0298 | 0.3262 | LOW_SUPPORT |
| 含非 Train 端点 | E GINE-GRAD | 64 | 11 | 0.0615 | 0.1022 | -0.0408 | -0.0893 | -0.0230 | -0.0293 | 0.0957 | LOW_SUPPORT |
| 含非 Train 端点 | F GINE-MASK50 | 64 | 11 | 0.0575 | 0.1089 | -0.0515 | -0.0806 | -0.0515 | -0.0569 | 0.0322 | LOW_SUPPORT |
| 配对预测正确 | A LCRC-GRAD | 990 | 164 | 0.0932 | 0.0770 | 0.0161 | -0.0087 | 0.0143 | 0.0141 | 0.0245 | OK |
| 配对预测正确 | B LCRC-LOCAL | 990 | 164 | 0.1308 | 0.0755 | 0.0553 | 0.0209 | 0.0664 | 0.0792 | 0.0150 | OK |
| 配对预测正确 | C LCRC-MASK50 | 990 | 164 | 0.0642 | 0.0774 | -0.0132 | -0.0279 | -0.0217 | -0.0258 | -0.0079 | OK |
| 配对预测正确 | D TOPOLOGY | 990 | 164 | 0.0642 | 0.0799 | -0.0157 | -0.0325 | -0.0084 | -0.0944 | 0.1232 | OK |
| 配对预测正确 | E GINE-GRAD | 856 | 158 | 0.1043 | 0.0778 | 0.0265 | 0.0057 | 0.0626 | 0.0451 | 0.1133 | OK |
| 配对预测正确 | F GINE-MASK50 | 856 | 158 | 0.0811 | 0.0755 | 0.0057 | -0.0148 | 0.0193 | 0.0105 | 0.0427 | OK |
| 配对预测错误 | A LCRC-GRAD | 656 | 131 | 0.0853 | 0.0806 | 0.0047 | -0.0194 | 0.0040 | -0.0085 | 0.0235 | OK |
| 配对预测错误 | B LCRC-LOCAL | 656 | 131 | 0.1537 | 0.0754 | 0.0783 | 0.0417 | 0.0987 | 0.1040 | 0.0472 | OK |
| 配对预测错误 | C LCRC-MASK50 | 656 | 131 | 0.0668 | 0.0757 | -0.0089 | -0.0273 | -0.0148 | -0.0293 | 0.0124 | OK |
| 配对预测错误 | D TOPOLOGY | 656 | 131 | 0.0753 | 0.0791 | -0.0038 | -0.0228 | 0.0182 | -0.0810 | 0.1815 | OK |
| 配对预测错误 | E GINE-GRAD | 790 | 141 | 0.0911 | 0.0729 | 0.0182 | -0.0072 | 0.0465 | 0.0422 | 0.0662 | OK |
| 配对预测错误 | F GINE-MASK50 | 790 | 141 | 0.0855 | 0.0713 | 0.0142 | -0.0096 | 0.0367 | 0.0324 | 0.0524 | OK |

### 结构组成：Top-1 子图有多少落在可变区 / 连接边界 / 共同核心

| 方法 | 平均 k | 原子:可变区 | 原子:边界 | 原子:核心 | 键:可变区 | 键:连接边界 | 键:核心 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A LCRC-GRAD | 3.34 | 0.0612 | 0.0879 | 0.8868 | 0.0349 | 0.0445 | 0.9207 |
| B LCRC-LOCAL | 3.35 | 0.1142 | 0.1147 | 0.8365 | 0.0820 | 0.0554 | 0.8626 |
| C LCRC-MASK50 | 5.31 | 0.0376 | 0.0607 | 0.9240 | 0.0175 | 0.0270 | 0.9555 |
| D TOPOLOGY | 6.00 | 0.0266 | 0.0767 | 0.9141 | 0.0048 | 0.0262 | 0.9690 |
| E GINE-GRAD | 3.73 | 0.0642 | 0.1015 | 0.8707 | 0.0261 | 0.0563 | 0.9176 |
| F GINE-MASK50 | 5.32 | 0.0528 | 0.0720 | 0.9037 | 0.0272 | 0.0345 | 0.9383 |

### 预测表现与配对 Δy

| split / model | n | RMSE | MAE |
| --- | --- | --- | --- |
| test gine | 420 | 0.6800 | 0.5092 |
| test lcrc | 420 | 0.7293 | 0.5324 |
| train gine | 3360 | 0.2552 | 0.1942 |
| train lcrc | 3360 | 0.2391 | 0.1782 |
| valid gine | 420 | 0.6161 | 0.4670 |
| valid lcrc | 420 | 0.7069 | 0.5079 |

| 模型 | 配对数 | 平均 \|Δŷ−Δy\| | RMSE(Δ) | 方向准确率 | \|Δy\|≥0.1 方向准确率 | 平均 \|Δy\| |
| --- | --- | --- | --- | --- | --- | --- |
| GINE | 823 | 0.2866 | 0.3727 | 84.9% | 89.3% | 0.6323 |
| LCRC | 823 | 0.2584 | 0.3565 | 86.8% | 90.9% | 0.6323 |

### 配对 Δ 贡献的局部性（为什么这里不报“oracle 捕获”）

冻结的配对 Δ 贡献在 1646 个端点中有 1642 个**只**落在编辑原子上（平均非零原子占比 `9.4%`，编辑区之外平均 `0.0024` 个原子）。在这些端点上，“子图捕获了多少 Δ 贡献”与“子图是否碰到编辑原子”几乎同义，所以本报告不把端点 capture 当作独立结构验证；对外部贡献的独立检验只在主实验（WC_LOGP 全分子原子贡献 + 匹配随机子图）上做。

说明：`含非 Train 端点` 只有 64 个端点，来自 43 个分子 / 11 个分量，记 `LOW_SUPPORT`；`两端均非 Train` 为空，按预注册规则记 `NO_SUPPORT`。这些 CI 只作描述。“配对预测正确/错误”的阈值是固定 `|Δŷ−Δy| ≤ 0.25`，且按模型族分别判定，所以 LCRC 与 GINE 的端点分母不同。

## 10. 五个问题的回答

### Q1. LOCAL_LCRC 的外部定位更强，它同时具备模型预测忠实性吗？

- Valid：LOCAL 的 oracle enrichment 均值 `0.0318`，95% CI `[0.0270, 0.0367]`；Test 均值 `0.0340`。
- 同族忠实性：`fidelity_advantage(std) = 0.0745`，95% CI `[0.0476, 0.1019]`；半/全剂量中位数 `0.3137`，|delta0|(std) 均值 `0.4307`。
- 结论：LOCAL 既在外部结构贡献上优于随机，也在模型预测依赖上优于同尺寸随机子图，两项同时成立。
- 幅度对比：LOCAL 的忠实性优势远小于同族 GRAD（`0.2107`）和 MASK50（`0.3303`）。也就是说 LOCAL 只是**过线**，不是“预测依赖最强”的方法。
- 半剂量检查：`|delta0.5| / |delta0|` 中位数 `0.3137`，半剂量效应明显**小于**全剂量（次线性）。这意味着用 m=0.5 的响应来近似 m=0 的删除效应会系统性低估，C 的“先 0.5 筛选、再 0 评价”也因此存在剂量错配。

### Q2. GRAD 与 MASK50 的差异是否说明梯度近似限制了结构发现？

- 外部对齐：A(GRAD) `0.0106` [0.0056, 0.0155]，C(MASK50) `0.0218` [0.0169, 0.0268]；配对差 C−A 均值 `0.0113`，95% CI `[0.0061, 0.0168]`。
- 同族忠实性：A `0.2107` [0.1863, 0.2342]，C `0.3303` [0.3002, 0.3594]。
- C 的 MASK50 选择在中位数 `191` 个候选上做筛选；选择分数（m=0.5）与真实忠实性评测（m=0）分属不同剂量，两者同族相关不能当作独立验证。
- 规模混淆：C 的平均 k 为 `5.62`，A 为 `4.00`；MASK50 分数没有按 |S| 归一化，更大的子图天然产生更大的 `|Δy|`。在 k=6 内比较时 C 对 A 的 enrichment 差只有 `0.0032`，远小于汇总的 `0.0113`。
- 结论：C 的选择在汇总上确实优于 A（配对差 `0.0113`，CI 排除 0），但幅度低于预注册的 `0.0200` 阈值，且其中大部分可由“C 平均选了更大的子图”解释（k=6 内差 `0.0032`）。因此 `FINITE_MASK_IMPROVES_ALIGNMENT` 不成立，也不能据此断言梯度近似是结构发现的主要限制。

### Q3. 在同一外部 oracle 下，LCRC 与 GINE 谁更强？有支持吗？

- LCRC 侧：A `0.0106`、B `0.0318`、C `0.0218`；GINE 侧：E `0.0302`、F `0.0317`。
- 最强 LCRC vs 最强 GINE 的配对差 `-0.0004`（CI 下界 `-0.0070`），n = 418。
- 结论：在**与模型无关**的外部 oracle 上，没有检测到 LCRC 相对 GINE 的配对优势（B 与 E/F 的配对差 CI 都跨 0）。跨模型的**忠实性**本来就不在本轮可比范围内（门控语义不同），所以本轮只能说“LCRC 没有外部定位优势”，不能说“LCRC 更不忠实”。

### Q4. 外部贡献捕获与模型预测依赖，是否经常落在不同结构上？

- LOCAL 在 Valid 上两项均可评的 `420` 个分子中：两项均 >0 `194`、仅 oracle `124`、仅忠实性 `52`、都未过 `50`。
- 两项同时为正的比例为 `46.2%`；也就是说，多数分子的“外部贡献所在”和“模型最依赖的结构”并不重合。
- 分离的方向不对称：仅 oracle 占 `29.5%`，仅忠实性占 `12.4%`。LOCAL 的问题主要是“结构贡献被覆盖了，但模型并不特别依赖它”，而不是反过来。
- 作为对照，F（GINE-MASK50）的两项均为正比例是 `73.7%`，而它几乎不选小片段（97.6% 在 k=6）；所以“两项重合”本身也可以由“总是选大子图”买到，不能单独当作定位能力。

### Q5. 下一步：改搜索方法、改结构表示/读出，还是停止 LCRC 特定解释性主张？

- 停止把“LCRC 特有读出/解释”当作优势来源：外部 oracle 上没有 LCRC 相对 GINE 的优势，唯一同时通过两项的 LOCAL 是一个**与模型无关**的局部输出启发式。
- 若继续，优先级是（1）改**选择目标**：尺寸归一化的剂量/删除响应，而不是未归一化的 `|Δy|`（本轮 C/F 的汇总优势主要来自选大子图）；（2）再考虑结构表示/读出层面的改动；（3）不再单独优化 LCRC 的梯度/局部输出读出。
- 本轮不支持“需要新架构”或“LCRC 解释路线整体失败”这类更强结论：LOCAL 的过线说明模型侧确实存在可被小片段捕获的预测依赖，只是它与外部结构贡献只在约一半分子上重合。

**按本轮预注册规则，到此停止**：不自动开始训练、不自动进入架构搜索。

## 11. 局限与不能从本轮推出的结论

- 只有 968 个主实验分子（Valid 420 + Test 420 + 固定 128 Train），Test 只用于方向复审，不是独立的大样本验证。
- 候选池被冻结在“2–6 根物理键、beam 12、四个生成器”的搜索空间内；池外的子图没有被评估，因此“某方法不忠实”只在本池语义内成立。
- oracle 是 WC_LOGP 的原子加性贡献，不是因果真值；capture 高只说明与该加性分解一致。
- `m=0` 是完全移除门控，`m=0.5` 是半剂量；两者都不是 Shapley 值，也不保证可加。
- 823 对辅助实验的编辑区是结构差异，不是性质变化的因果位置；非 Train 端点只有 11 个，配对结论只作描述。
- 本轮没有训练任何新模型，也没有更换 checkpoint；所有结论都条件于这四个冻结模型。

## 12. 复现与文件清单

| 阶段 | 命令 | 输出 |
| --- | --- | --- |
| fields-lcrc | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-lcrc` | `results/fields/lcrc_*.jsonl + fields_manifest_lcrc.json` |
| fields-gine | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-gine` | `results/fields/gine_*.jsonl + fields_manifest_gine.json` |
| pool | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh pool` | `results/pool/pool_{main,aux}.jsonl + pool_manifest.json` |
| select-lcrc-main | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-lcrc-main` | `results/select/select_lcrc_main.jsonl + select_manifest_lcrc_main.json` |
| select-lcrc-aux | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-lcrc-aux` | `results/select/select_lcrc_aux.jsonl + select_manifest_lcrc_aux.json` |
| select-gine-main | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-gine-main` | `results/select/select_gine_main.jsonl + select_manifest_gine_main.json` |
| select-gine-aux | `bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-gine-aux` | `results/select/select_gine_aux.jsonl + select_manifest_gine_aux.json` |

| 文件 | sha256 |
| --- | --- |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/aux_endpoint_results.csv` | `1652b9adfb9fe638da8a85674b04c7652fd881680bc4eaaabea63d980670ee0a` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/aux_pair_predictions.csv` | `0fb57e40942b027d91163035cc5d5220129af9e7a59e173587b35517946c2986` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/aux_paired_differences.csv` | `92644f8c357b7a2f366a1313c1d5320a1f273513c48ca80f2aff78b9c0f4aebf` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/aux_summary.json` | `363ee9e63ea4d924a9ec3fe0bd087545c241f189667431586969bf86f526b106` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/decision.json` | `fdb61e29baef12a976120e0f3157c2ba9ee9bc6d2725e8de51ebb32ab7e69ef8` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/gates.json` | `e15cdc98dd7c4e2168cd3468e4759437a7c7d899a266e83a067a8583c2256f0a` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/identity.json` | `bd6d04b68c145ef4a67e2a54036eccdbb4c2528198b7b94a3016a67594728f03` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/lipo_replay.json` | `77edb51be00e83b47ba1c47ff55045ba73cfc657bc430c9e86a9581dcad1320c` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/main_concentration.csv` | `e87e523a5f94c9d6e01c8b449f10ffc32fcc0ec8b1ccdb5bd3a84b2c2250bf2c` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/main_molecule_results.csv` | `760e5abbb22609ad4d0c5757a48773d6f2ed3d6b5b00e5bf9735047f0c7a1289` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/main_paired_differences.csv` | `b4dc6d0dbe4cf307bf42ad4dd097f9095ac2beb116e3e4ca523b23c915e8df8b` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/main_stratified_by_k.csv` | `b310ba5f1b65b090a91fff4ef72ea54ced5765d4e694652109babdf85a6a37c1` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/main_summary.json` | `5c250c1612494aee5126a78f98d3387b6f571755904cbaebb38bbabacbf47baa` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/analysis/protocol_freeze.json` | `b2c85c99cf43b739479797ada61352b91bfc0a09e0dabb5e18027722788484cb` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/gine_lipo_aux.jsonl` | `7377c7a7b5d59b559ff816f86c0a756e1ebcb4e1bce089e12580f76b9eff4923` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/gine_lipo_main.jsonl` | `22b07eed30664acb4cfaa8f9047b763982c2c4594f3a73c62ab2d063964ad25e` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/gine_wc_aux.jsonl` | `d86f8c2356705f109f94848c8af08ae6939c435774944856c340a881d1781a2b` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/gine_wc_main.jsonl` | `60ed4471feaf48f372d9011d655d3490c1a05a82b7489257e181589f43756cb7` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/lcrc_lipo_aux.jsonl` | `333b0fca336ee5e250dbf6760f8679b82b4beb597a14f3bbe33f3f63c40c5d8e` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/lcrc_lipo_main.jsonl` | `ac6f0232a5bd321a29519ed0ce692694ad5e0cf2b11a485b3e8060140edc15b1` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/lcrc_wc_aux.jsonl` | `a94cf121971aa3a3037d016a227607b267abeb0ecea6cfe4a1238d742aeee5cc` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields/lcrc_wc_main.jsonl` | `3b509e033fc84c2ab4d62a71bc4df6ccf76f0e82100e55e68a829b133fc69b3b` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/pool/pool_aux.jsonl` | `c18cf56ce43ea92a9dd036c58d5d3533a96674398af1f9c659f5de6fc1fc6aad` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/pool/pool_main.jsonl` | `38090b675d928586034adb14900da5e225728415115539c1c9d6e04bf872f58a` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/select/select_gine_aux.jsonl` | `856294de7d42a4accc49d07a6080a1998a394884de1fbc5fa66795b74b4f8f83` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/select/select_gine_main.jsonl` | `77a0e643ff13e20f9491db50edb1651077f80ebc31d0f488406cdab55d67c3c4` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/select/select_lcrc_aux.jsonl` | `33143a715995970615ea46852c73ab6d73cffc6666325b0bdad1ddf5bcaca688` |
| `tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/select/select_lcrc_main.jsonl` | `abdb1bccd3b83c6dbc17659c2a3f34185066fe6097943d5c4aeff659e456867e` |

本地复现（CPU，无 GPU）：

```bash
uv run python code/lfa_prepare.py            # 身份 + 重放 + WC_LOGP 闭合
uv run python code/lfa_gates.py             # 实现门禁
uv run python code/lfa_pred.py              # 冻结 Lipophilicity 预测
uv run python code/lfa_main.py              # 主实验聚合 + 决策
uv run python code/lfa_aux.py               # 823 对描述性诊断
uv run python code/lfa_report.py            # 本报告
uv run pytest tests -q                      # 结构不变量
```
