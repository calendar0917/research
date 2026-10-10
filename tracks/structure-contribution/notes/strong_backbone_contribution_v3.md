# cscl-v3 — Strong-Backbone Structural Contribution Learning (SB-CSCL)

Track: `structure-contribution` · 日期: 2026-10-11 · 协议:
`configs/strong_backbone_contribution_v3.yaml`（冻结于任何正式 GPU 训练之前）
代码: `code/strong_backbone_attribution.py`（归属核心）、`code/run_cscl_v3.py`（S/O 头 + 训练/分析）
测试: `tests/test_cscl_v3.py`（14 项）+ 既有 81 项全部保持通过

> 研究问题（本轮）：能否利用已验证的强非 GNN 局部结构编码（冻结 Full-Y seed-0，
> dev MAE 0.11641），把最终预测改造为**跨分子共享结构单元、上下文修正、单元间
> 关系及显式全局拓扑贡献的严格加和**，同时尽量保留预测能力？

三参照设计：**F** = 冻结 Full-Y（0.11641，不重训）；**O** = 同一冻结特征 +
不透明读出（新训 seed 0）；**S** = 同一冻结特征 + 结构化贡献读出（新训 seed 0）。

---

## 0. 勘误（上一轮统计代码）

`run_prediction_feasibility_v2.py::run_compare` 的配对 bootstrap 统计量误为
`np.abs(e_full[smp] − e_or[smp]).mean()`（恒正的逐分子误差距离均值）。已修复为有符号差
`(e_full[smp] − e_or[smp]).mean()` 并重算：Full-Y − O-rich = −0.21565，正确 95% CI
**[−0.23064, −0.20123]**（原存储区间 [+0.24133, +0.26895] 被报告手工变号为
[−0.26895, −0.24133]，并非有符号差的 bootstrap）。定性结论不变。原始文件未覆盖；
勘误记录：`results/feasibility_v2_cpu/paired_vs_orich_erratum.json` +
`notes/prediction_feasibility_v2.md` 头部勘误框 +
`test_compare_statistic_is_signed_difference` 回归钉。本轮全部配对 bootstrap 使用
同一正确定义（`run_cscl_v3.paired_bootstrap`）。

## 1. Phase A：映射与重建验证（全部通过，`results/cscl_v3/phase_a_checks.json`）

### 1.1 映射证据链（root → 原子 → RINGCHAIN 单元）

1. **`dict_phi` 行序 = 原子序**：`zinc_e2e_dictenv_p1.prepare_env` 逐分子断言
   `n == num_nodes == graph.n == phi.shape[0]` 且 `node_types == molecule.atom_idx`；
   `fsar_r2_ar0.build_phi` 按 `graph.nodes`（= 0..n−1 = PyG 原子序）每原子一行。
   实测：25 个抽样分子**从原始行重建 phi 与缓存逐位相等（max|d| = 0.0）**。
2. **`dict_atom[i] == x[i]`**：全部 10000 分子 0 不匹配。
3. **pair 覆盖**：`_graph_record` 对所有 center 对 (i<j) 生成 pair；逐分子断言
   `len == n(n−1)/2` 且 `source < target`（每分子、每 batch 内执行）。
4. **batch 拼接**：PyG `Data.__inc__` 对含 "index" 的键返回 `num_nodes` ⇒
   `pair_index` 逐图偏移；`env_collate` 以 `ptr` 偏移 occ/bond 字段。
   实测：同一分子在不同 batch 组合下聚合量一致（相对误差 < 1e-4，测试钉住）；
   原分组提取与缓存**逐位相等**（确定性代码路径）。
5. **原子重标号**：对真实分子做原子置换，经完整冻结编码器重提取：单元签名多重集
   不变、逐类型 E 聚合一致（<1e-3）、重建审计正常（测试钉住）。E 的输入字段
   （dict_phi/patch_cont/anchor/occ/bond/pair 表）全部随置换同步，缺一即失败。

### 1.2 池化重建（对 640 行全新提取审计；全量提取审计同量级）

| 检查 | 最大误差 |
|---|---:|
| 单元级 ΣE、ΣE² 重建 unary first/second 块 | 6.9e-7（相对） |
| 单元内 + 跨单元 per-bucket Σpv、Σpv² 重建 pair 池 | 7.1e-7（相对） |
| log1p(count)（原子计数与 per-bucket pair 计数，由总量重建） | 2.4e-7（绝对） |
| C6 掩码后 reader 实际输入 vs 参考池化（first/second 块） | 0.0（逐位） |
| C6 掩码的 count 块确认为 0 | 0.0 |

结论：**冻结 reader 消费的 pooled 表示可由单元/关系局部统计精确重建**；count 由
总量做 log1p（不局部相加）；每个原子恰好归属一个单元；每个 pair 恰好归属一次
（单元内或某无序单元对）；物理 inter-unit 键与划分的 `inter_relations` 逐键一致
（有键单元对必出现在跨单元 pair 中）。

### 1.3 冻结 soup 回放

CPU 重放 dev/monitor：max|Δpred| = 1.9e-6 / 1.7e-6（float32 设备差），dev MAE
0.11641 与保存值一致（容差事先定为 atol 5e-3 / MAE 1e-3）。checkpoint 未做任何修改。

## 2. 模型（S/O，同一冻结特征）

S（SB-CSCL，118,168 参数）：
`pred = b + Σ_u[α(t_u) + δ(z_u, c_u)] + Σ_{(u,v)} γ(z_u, z_v, r_uv) + g_ctx + g_topo`

- `z_u` = [mean E, rms E, mean/rms 单元内 pair value]（384 维，冻结值）；
- `c_u` = 单元描述子 36 维，按 fit-only 同类型均值中心化；
- `α` = 每类型 1 标量共享表（fit-only 词表 746 known + UNK 桶 = 761 id，未见签名走
  (kind, size-bucket) UNK）；
- `γ` 输入 = [z_u+z_v, |z_u−z_v|, z_u·z_v, pair 值矩, 距离桶直方图, 物理键直方图,
  log1p 计数, 中心化 relation 均值]（1273 维，端点对称，lo<hi 确定性去重）；
- `g_ctx`/`g_topo` = 逐特征加性 bank（线性 + 3 个冻结 fit 分位铰结，30+25 特征），
  逐特征贡献可导出；
- C6 信息域一致：不用全局原子/键直方图、不用图级 count 块；单元大小属于局部结构
  （Full-Y anchor 同类信息），如实披露。

O（matched opaque，138,753 参数，比值 1.17，披露非严格容量匹配）：
同 collated 单元/关系行；类型经 16 维 embedding 进不透明单元编码器；DeepSets 式
mean+sum 池化 + 全局 MLP → 图级读出。**无加和结构**。

S/O 输入信息对应表（协议 yaml `head_architecture`）：两者读取完全相同的
z_u/ctx/pair 聚合与 ctx30/topo25；唯一设计差异 = 类型通道（S: 1 标量/类型且约束
δ 类型均值；O: 16 维/类型自由使用）。**O−S 差不能全部解释为"解释性代价"**。

### 可辨识性（research_core §5 对齐）

- δ 惩罚 = batch 内**逐类型均值**的加权平方和（类型平均贡献归 α）；
- γ 惩罚 = batch 均值平方（关系平均贡献归 b）；
- 全局通道仅读 ctx/topo，与局部项参数完全分离；b 固定为截距；
- 设计修订记录：Phase C 首次 CPU rehearsal（batch 均值版惩罚）显示 δ̂–α* 混淆
  0.63——不满足 research_core "δ 在类型典型上下文处为 0" 的要求，遂在正式 GPU
  训练前改为逐类型中心化惩罚（v0 已验证的做法）。两种版本的 rehearsal 数字均
  保留在 §4，作为"可辨识性—性能"权衡的证据，不作为正式成绩。

## 3. 合成已知真值任务（`results/cscl_v3*/synthetic_s0.json`）

在真实冻结特征上构造 y_syn = Σα\*(t) + 0.8·(类型中心化 attach 偏差) + Σγ\*(真值对)
+ 0.05 噪声（seed 0；α\* 由签名哈希确定，混合正负；3 个高频真交互对 + 1 个共现
混淆类型对，真值 γ\* 无）。固定种子 0，CPU。

逐类型惩罚版（正式设计）：α 符号恢复 **1.00**、Spearman **1.00**；δ–真值相关
**1.00**、δ̂–α\* 混淆 **0.021**；γ̂ 真值对 +0.860/−0.707/+0.499（真值 +0.9/−0.7/+0.5，
方向全对）；混淆对 γ̂ = **0.047 ≈ 0**；fit 中未见过的单元对上 |γ̂| = 0.028 vs 可见对
0.090（无大伪交互）。batch 均值版：α 方向 1.00/ρ 0.94，但 δ̂–α\* 混淆 0.63。

**合成证据**：在噪声水平内，S 的三项分解可以恢复已知真值、区分真交互与共现、
并对未见组合保持小贡献——分解不是恒等式上的任意分摊。

## 4. CPU rehearsal（工程验证，非正式成绩；240 epochs，seed 0，CPU）

| 臂 | dev MAE（primary best-monitor） | 备注 |
|---|---:|---|
| F（冻结） | 0.11641 | 不重训 |
| O | 0.13842 | 与正式 GPU 协议相同 |
| S（batch 均值惩罚，弃用） | 0.16823 | S−O +0.030；δ̂–α\* 混淆 0.63 |
| S（逐类型惩罚，正式） | 0.19749 | S−O +0.059 [0.050, 0.069]；混淆 0.021 |

加和恒等式残差 1.1e-6（导出值独立重建）；通道方差占比（正式版 dev）：unit 0.129 /
rel 0.413 / g_ctx 0.011 / g_topo 0.132 —— 全局通道（合计 ~14%）未吞掉预测，但
**关系项方差占比最高**，归因解读时须考虑 δ 受约束后的贡献转移。

## 5. 正式 GPU 实验（res / GPU1 / seed 0）

（GPU 运行后回填：见 §6 与 results/cscl_v3/。）

## 6. 边界与声明

- 单 seed 0；bootstrap 为分子重采样区间，非跨 seed 稳定性。
- 冻结编码器由 fit_inner raw y 训练（feasibility-v2），S/O 消费其表示——披露。
- F 的预测仅用于配对比较（analyze），从不作为 S/O 的输入或监督（AST 测试钉住）。
- rooted patch 可跨单元边界：α/δ 主张**按结构单元锚定的预测贡献**，非片段内部
  独占化学作用；真实 ZINC 分数只称"模型预测归因"。
- 0.11641 是内部 dev 单 seed 开发证据；本轮不访问 official valid/test。
