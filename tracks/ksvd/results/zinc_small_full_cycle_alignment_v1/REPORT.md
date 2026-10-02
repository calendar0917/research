# zinc_small_full_cycle_alignment_v1 — Small/Full 标定比较与环惩罚分解对齐

Date: 2026-10-02 · 只读/描述性 · **官方 test 从未实例化** ·
证据 `tracks/ksvd/results/zinc_small_full_cycle_alignment_v1/`。

**底座：保留 Full。优先验证：现有 `topology25 → topology8 → reader` 全局拓扑通道对
环惩罚样本（G1）与 id172 的输入保留与预测响应。证据强度：中等**——单 seed、两个 soup
分别在各自 valid 上选模、训练协议不同，口径已统一但**不是**匹配因果对照。

## 1. Small/Full 统一 train-median 标定

对象（均已核对身份）：

| | Small | Full |
|---|---|---|
| checkpoint | `e2e_dictenv_latent_bridge_v1/…/LATENT-BRIDGE-seed0_soup_state.pt` | `e2e_dictenv_scale_v1/…/SCALE-FULL-seed0_soup_state.pt` |
| sha256 | `b14c92af…c8e54` | `17f5fcc3…574eb` |
| 参数量 | 106,925（`LatentBridgeSEM108`, m=1） | 408,651（`LatentScaleSEM108`, m=3）→ **3.82×** |
| soup 成员 | `[285,296,303,307,309]`，有效 raw 0.1210583 | `[268,276,289,299,317]`，有效 raw 0.1191541 |
| replay 校验 | raw valid `0.12105831989174476`（逐位复现） | handoff R 复现 `0.119154089` |

标定 `b_m = median_train(y − pred_raw_m)`（每模型各自，valid 不参与）：
Small `b=−0.008670`，Full `b=−0.034288`。checkpoint 未折叠 bias（raw 与 calibrated 分列）。

| | raw train | cal train | raw valid | cal valid |
|---|---:|---:|---:|---:|
| Small | 0.055133 | 0.054486 | 0.121058 | **0.120999** |
| Full | 0.045397 | 0.032950 | 0.119154 | **0.115066** |

**统一标定后 Full 相对 Small 的 gain**：raw `+0.001904` → calibrated **`+0.005934`**
（≥0.003 资源门槛；排除 id172 后 `+0.004496`，仍为正）。标定**放大了**表面差距而非消除：
Full 的 train-median bias 大（−0.0343 vs −0.0087），标定把 Full 从 0.1192 拉到 0.1151、
Small 几乎不动。**若只看 raw，Full 的 3.8× 参数只值 0.0019（低于门槛）；标定后为 0.0059。**

## 2. 环惩罚分组对齐（标签机制，非候选环值）

用 `zinc_long_cycle_audit/*_cycle_audit_label.csv` 的 `label_effective_cycle_snapped`
（标签实际使用的反向工程值；无惩罚=0，有惩罚<0）。身份核对：1000/1000 行、
`subset_index` 零基连续、`molecule_id == "valid:%04d"`、`target` 与 handoff `y`
max diff `3.6e-15`；取值集合 `{0,−1,−2,−6}`。**UNKNOWN=0**，各组之和覆盖全部 1000 行。
id172 = `valid:0172`（row 172），`y=−20.3405`，`label=−6`，label 环项 `−20.8100`，
`y_without_cycle=+0.4695`（与旧审计一致）。

各组 calibrated valid（`signed = pred−y`；贡献 `=Σ|err|/1000`）：

| 组 | n | Small MAE | Small 贡献 | Full MAE | Full 贡献 | Full signed 均值 |
|---|---:|---:|---:|---:|---:|---:|
| G0（无环惩罚，非 id172） | 965 | 0.09815 | 0.09471 | 0.08939 | **0.08626** | +0.0032 |
| G1（有环惩罚<0，非 id172） | 34 | **0.17426** | 0.00592 | 0.29070 | 0.00988 | +0.0134 |
| G172 | 1 | 20.35988 | 0.02036 | 18.91766 | 0.01892 | +18.918 |
| 合计 | 1000 | 0.120999 | 0.120999 | 0.115066 | 0.115066 | |

**Full 相对 Small 的配对贡献差**：G0 **+0.008450**（Full 好）、G1 **−0.003959**（Full 差）、
G172 +0.001442（Full 好）。即 Full 的优势全部来自普通分子，**在 34 个环惩罚分子上反而退化**。
train 上相反：Full G1 MAE `0.0948` vs Small `0.2923`——Full 在 train 环样本拟合更好、
valid 更差，是**环样本过拟合/泛化退化**，不是“不会表示”。

新头 A/B/C（上轮冻结）相对 Full 的校准 valid 变差，主要落在 **G0**（贡献 −0.0050…−0.0070），
其次 G1（−0.0006…−0.0028），id172 贡献可忽略（−0.0001…−0.0005）。

## 3. 到 0.09 的缺口

Full cal valid `0.115066` → 需减 **0.025066**（到 0.08 需 0.035066）。
算术预算（其余不变、完美预测某组，**非可实现收益**）：G0 可减 0.086264、G1 0.009884、
G172 0.018918（**id172 单行占所需缺口的 75%**）；G1+G172 = 0.028802 ≥ 0.025066。
按池子可减比例：G1/G172 需移除约 87%，G0 需移除约 29%。二者都重要，见 §4 取舍。

## 4. 决策与下一步

**底座 = Full**：calibrated gain 0.0059 ≥ 0.003，且在 G0 有明确分组优势（+0.0085 贡献）。
单 seed、非匹配训练协议，故只用于资源决策，不主张因果。若后续证伪，退回 Small 保留 Full 比较。

**优先方向 = 现有全局拓扑通道（选项 2），不做新特征。** 取舍理由：本轮“到目标
0.09 的缺口”数值上集中在环惩罚样本（G1+G172 ≥ 缺口；G172 单行占 75%），且 Full 在 G1
**主动退化**（train 更好、valid 更差）；相比之下容量/读出方向（capacity_localization_v1/v2、
CODE、P1/P2、A/B/C）已多次关闭，而 `topology25 → topology8 → reader` 对环样本的**实际利用**
从未定位。G0 仍是最大绝对池（0.0863），是回退方向。id172 是单行极端外推（需预测 −20.34
而 train 该段极少），很可能不可约——因此本检查先区分“标签机制存在”与“模型实际利用”，
不承诺收益。

**下一步定位检查（只写设计，本轮不启动）：**
- 可证伪假设：在冻结 Full 上，`reader` 的预测对 `topology25` 中环严重度统计的响应在 G1
  与 G0 上不可区分（即该通道有效信号在 `topology8` 已丢失或被 reader 忽略）。
- 对照/操作：固定 checkpoint，仅对 valid 的 G1 vs 匹配 G0 子集做输入置换/置零
  （不训练、不改输入分布之外的任何标量），测量 `topology8` 保留与 reader 输出变化；
  Small 作参照。
- 预算：只读 CPU 前向，<5 分钟；固定标定（各自 train-median）、固定末轮参数。
- 通过条件：若 `topology8` 对环严重度的保留在 G1 显著低于 G0 或置换不改变输出
  → 通道丢失，值得提局部 pilot；否则关闭该假设，回退 G0 泛化方向。
- 停止条件：无 >0.003 量级差分响应即停止，不再列 head/模块清单。

## 5. 边界、缺失与耗时

**缺失/未完成**：Small 只有 seed0 发布 soup；两模型训练协议（latent-bridge vs scale FULL）、
选模（各自 top-5-by-valid，同一 1000 行）、硬件/轮数不同 → **口径已对齐，但不是匹配消融**。
旧 compact-v2 oracle 的 `+0.0117/+0.0247` 是旧模型/旧目标的算术上限，**不是当前 Full 的收益预测**，
不能推出“53% 误差不可达”；环基顺序敏感也不能单独推出不可约误差。旧表
`compact_v2_prediction`/`residual` 未使用，所有误差从本轮模型预测重算。

**边界**：单 seed 历史比较仅用于资源决策；未访问 official test、未读 test 标签或逐图预测；
未拟合任何 head/scaler/oracle，唯一拟合量为各模型 train-median 标量 bias；未更新 backbone。

**耗时**：起 `12:46:17Z`，deadline `13:06:17Z`，`13:01:17Z` 停止新计算。Small replay
16.4 s，对齐分析 <1 s，定位阅读约 14 min。全部已完成。

**复现：**
```
uv run python -m tracks.ksvd.results.zinc_small_full_cycle_alignment_v1.replay_small
uv run python -m tracks.ksvd.results.zinc_small_full_cycle_alignment_v1.align_groups
```
产物：`summary.json`、`group_table.csv`、`valid_per_graph_aligned.csv`（1000 行逐图，
列含 group / small,full raw+calibrated / A,B,C × 2 seed calibrated）、`small_replay.json`。