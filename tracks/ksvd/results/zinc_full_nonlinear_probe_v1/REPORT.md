# zinc_full_nonlinear_probe_v1 — 限时非线性条件探针与尾部复核

Date: 2026-10-02 · 冻结父 Full soup（sha `17f5fcc3…574eb`）· **官方 test 从未实例化** ·
protocol `tracks/ksvd/protocols/zinc-full-nonlinear-probe-v1.yaml` ·
run `tracks/ksvd/runs/2026/10/02/20261002-200428-11b0b325`（44.1 s）·
证据 `tracks/ksvd/results/zinc_full_nonlinear_probe_v1/`。

## 0. 决策（先给结论）

**两方向都未买到改善 → 本轮停止，不购买下一个 pilot。** 三个具体配置
（`[R]`、`[R,z]`、`[R,q(R)]` 的 `64→32` SiLU MAE 头）都**低于**父模型门槛，且
B 相对 A/C 无增益。**只关闭这三个配置**；**不能**据此宣布 R 信息已充分、节点信息无用、
或整类非线性读出无效。

## 1. 对象与身份核对（最低限度）

| 项 | 结果 |
|---|---|
| R 缓存身份 | handoff `train_reader/valid_reader.npz`，`checkpoint_sha=17f5fcc3…`、`split_fingerprint=58c69506…` 均与父 soup 一致 |
| reader replay | 用 `reader.npz` 从 R 复现预测：train max diff `5.1e-6`、valid `9.4e-7`；valid raw MAE `0.119154089` vs 发布 `0.119154092`（差 `3e-9`） |
| 数据行序 | 直接用 handoff 的 official split 位置序；`p_base` 对回 `valid_per_graph.csv::pred_raw`（max diff `5e-7`） |
| z 定义 | 复用审计 P2：`Σ_occurrence outer(coord_33, onehot(atom_28))`，图级、**无 shell 权重**、无新统计量 |
| 标定 | 每头各自 `b=median_train(y−pred)`，`pred+b`；不复用父 bias，不用 valid |

**恒定列冻结规则**（训练前记录）：train `std<1e-8` 的列丢弃。R：814→**598**（丢 216，
与 C6 屏蔽的全局直方图/count 通道一致）；z：924→**383**（丢 541）；`q(R)`：383/383。
`z` 非恒定维 383 ≤ 1024，直接使用，无 PCA。

## 2. 非线性头结果（3 配置 × 2 固定 seed）

冻结配置：`input→64→32→1`、SiLU、线性输出、无 dropout；全体 train 均匀 MAE；AdamW
lr `1e-3`、wd `1e-4`（bias 不 decay）、batch 256、clip 5。测速 `0.068 s/epoch` →
**epoch 数在看 valid 前冻结为 `min(150, ⌊80/spe⌋)=150`**，末轮参数评估。
A/B/C 共用同一初始化的 R 输入块与更深层（同 seed 同 generator）。

| 配置 | in_dim | seed0 cal_valid | seed1 cal_valid | 两 seed 均值 gain vs 父 |
|---|---:|---:|---:|---:|
| 父 Full（基线） | — | — | — | `0.11506585458567133` |
| **A `[R]`** | 598 | 0.121917 | 0.121338 | **−0.006531** |
| **B `[R,z]`** | 981 | 0.122062 | 0.124858 | **−0.008394** |
| **C `[R,q(R)]`** | 981 | 0.123491 | 0.122842 | **−0.008090** |

配对 gain（valid，正=候选更好）：

| 比较 | seed0 | seed1 | 均值 | 判定 |
|---|---:|---:|---:|---|
| A − 父 | −0.00679 | −0.00627 | −0.00653 | 失败 |
| B − 父 | −0.00700 | −0.00979 | −0.00839 | 失败 |
| **B − A** | −0.00021 | −0.00352 | **−0.00186** | 无节点增益 |
| **B − C** | +0.00143 | −0.00204 | **−0.00030** | 方向不稳、未过门槛 |

**不是 `PROBE_UNDERPOWERED`：** 新头在 train 上**比父模型更好**（A raw train MAE
`0.0325/0.0272` vs 父 eval-train `0.04540`；cal train `0.0283/0.0252` vs 父 `0.03295`），
却在 valid 上更差。即该 `64→32` 头**拟合能力足够、泛化被父的 39 维瓶颈更好**，不是训练不足。
数值全部有限、末段 batch L1 `0.016–0.021`、参数确实更新（train loss 下降）。分组方向也不支持
候选：A 的 `id%5` 五组仅 0–1/5 改善、最低十分位更差（−0.008…−0.016）。

## 3. 尾部复核（父模型 calibrated，`signed error = pred − y`）

| `y<−3` | n | MAE | signed mean | signed median | OLS slope `pred~y` |
|---|---:|---:|---:|---:|---:|
| train | 716 | 0.0660 | −0.0190 | +0.0058 | **1.034** |
| valid | 65 | 0.5068 | +0.3471 | +0.0722 | **0.287** |
| **valid 排除 id172** | 64 | **0.2191** | +0.0569 | +0.0699 | **0.962** |

**关键：id172（`y=−20.34`）几乎单独制造了“尾部回缩”。** 排除它后 valid 尾斜率
从 0.287 回到 0.962（≈train 1.034），系统性 signed bias 从 +0.347 降到 +0.057；
但 MAE 仍为 0.219（train 0.066 的 3.3×）。按**固定 train 尾 target 四分位**分箱
（每箱 train 179 行）：bin0 `[−42.0,−5.24]` valid 12 行 MAE 2.05（id172 主导；排除后 11 行
MAE 0.516、slope 0.877）；bin1 `[−5.24,−4.14]` n17 MAE 0.189 slope 0.988；bin2
`[−4.14,−3.42]` n18 MAE 0.176 slope 0.845；bin3 `[−3.42,−3.00]` n18 MAE 0.109 slope 1.343。
valid 每箱仅 17–18 行、目标上限 −20.34（train 到 −42.04），**不作覆盖率归因**，只报数量。

**误差预算：** 最低十分位阈值 `−2.5415`，valid 93 行、MAE 0.3812，占绝对误差和 **30.8%**，
对总 MAE 贡献 **0.03545**；**完全修好该区，其余不变仍为 0.07962**。train p5 尾部
（`y<−3.55`）44 行 MAE 0.684、贡献 0.03010、排除 id172 后 0.260。中央/尾部分别报告，
id172 已含在尾部，**不重复相加**。注意审计文字里的 “signed −0.381” 符号与本次 `pred−y`
约定相反，本文统一为 `pred−y`。

## 4. 节点死亡与 soup 成员

磁盘上只有 `raw_state`（单一非 soup 端点）、`final_state`、`soup_state`，**5 个 soup 成员
[268,276,289,299,317] 未保存**。三者中 `W_A_S`、`W_A_C`、`node_encoder.0.weight` 的 absmax
与 norm **逐个字节相同**（absmax≈`7.05e-38`，norm=0.0）；状态间 Frobenius 差
raw↔soup 15.4、final↔soup 36.5、raw↔final 41.1。→ 死亡**不是** 5 成员平均抵消的产物
（抵消会留下非零成员）：单个已训练端点 `final` 本身已死。逐成员全检因文件缺失标
**未完成/未知**。本轮不复活节点。`raw_state` 是初始化还是训练端点未在预算内从代码确认，
但这不影响“非平均抵消”结论。

## 5. Valid/Test 差值口径

当前上下文中**没有**“同一模型、同一 checkpoint/选模协议、同数据规模与单位”的成对
valid/test 汇总（control-plane records 均为 `official_test_loaded=false`；legacy 长程
factorial 是不同架构，不可比）。因此用户观察的 **valid 比 test 高 ~0.02 标“当前对象未验证”**，
停止查找。**不采用 `test=valid−0.02` 固定换算。** 阶段目标 valid `0.09` / 进一步 `0.08`
只作为**目标口径**，不因本轮探针调整；本轮未加载 test、未读 test 标签或逐图预测。

## 6. 五问直答

1. **尾部回缩在排除 id172、按相近 target 区间比较后是否仍成立？**
   **斜率回缩不成立**：多由 id172 单行造成（0.287→0.962）。但**尾部绝对误差仍升高**
   （排除后 MAE 0.219 vs train 0.066），且 valid 尾箱样本很小、目标范围更窄。
   即“尾部泛化差”弱化为“尾部绝对误差大 + 单行极端失败”，不能简单归因为覆盖或架构。
2. **支持读出改动还是节点信息入口改动？**
   都**不支持**。A（读出改动）两 seed 均 −0.0065；B 相对 A（−0.00186）与同宽 C
   （−0.00030）均无正增益。证据是**冻结 R 上、固定 3 配置 × 2 seed** 的负结果，
   头部训练充分（train 优于父），边界为：不覆盖“不同读出结构/容量/正则、端到端训练、
   节点复活后 R 改变”的情形。
3. **有没有理由购买下一个小 pilot？**
   **没有。** 唯一值得记的线索是：新头 train 更低而 valid 更差，指向**读出的正则/瓶颈**而非
   “缺信息”。若未来再买，应先解决“读出的泛化”，但本轮不追加拟合、不放宽门槛。
4. **valid 0.08–0.09 目标与 valid/test 观察哪些有记录支持？**
   支持：父 calibrated valid `0.115066`、id172 单行贡献 `0.0189`、最低十分位贡献 `0.03545`
   （残 0.07962）。**未验证**：valid 比 test 高 ~0.02（无同协议成对汇总），故
   `0.09/0.08` 仅为目标，不作为已完成口径。
5. **耗时/完成度/边界：** 总 30 min 内完成；首次工具调用 `11:58:04Z`，deadline `12:28:04Z`，
   `12:23Z` 停止新计算。6/6 拟合完成（A/B/C × seed 0,1，各 150 epoch，`6.0–7.9 s`），
   单次远低于 90 s 硬上限，拟合总耗时 ~44 s。
   （另有一次 0.4 s 即因脚本入参顺序 bug 中止的本地 run `20261002-200317-00d0b4ed`，在任何
   拟合完成前终止，未计入；已修复后重跑。）未完成项：5 个 soup 成员逐检查（文件缺失）、
   成对 valid/test 汇总（不存在）。**全程未访问 official test、未训练 backbone、未改旧协议。**

## 7. 复现

```
uv run research run zinc_full_nonlinear_probe_v1 --study zinc-context-gap \
  --purpose "frozen-Full nonlinear head probe: A=[R] B=[R,z] C=[R,qR], 2 seeds"
# run: tracks/ksvd/runs/2026/10/02/20261002-200428-11b0b325  (44.1 s)
```
产物：`nonlinear_probe.json`（全量指标/分组/尾部/成员检查）、`valid_paired.csv`
（逐图 `y`、父与 3×2 头的 raw/calibrated 预测）、`node_member_check_supplement.json`；
脚本 `nonlinear_probe.py`；runner `src/ksvd_research/runners/zinc_full_nonlinear_probe_v1.py`（单阶段，无 stage 分支）。