# CSCL-v0 协议（预注册）

Track: `structure-contribution` · protocol_id: `cscl-v0`
冻结日期: 2026-10-10（本文件在**任何标签参与的开发比较之前**提交）
性质：**本轮决策标准**，非普适统计阈值。

## 0. 一句话

在不使用 GNN/Transformer 消息传递的前提下，检验"跨分子共享结构单元 +
真实单元间关系的显式加性/成对分解"能否在 ZINC y-回归上达到接近不透明
对照的性能，且其贡献分解通过完整性/稳定性/合成真值三层评价。

## 1. 数据与划分（冻结）

- 数据：canonical PyG ZINC `subset=True`，`data/ZINC`；**仅加载 official train
  10000**。official valid 本轮不加载（如需背景参照须显式记录）；**official test
  禁读**（历史已消耗）。
- 监督：仅原始 `y`。不使用 `g/ell/s/c` 辅助标签、冻结 Q 头或任何历史模型输出。
- 内部划分：official-train 10000 → **fit 8000 / dev 2000**。
  - 分组键：committed canonical SMILES 表
    `tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz`
    （row i = official-train subset_index i；provenance 见同目录 json）。
    同 SMILES 的行不可跨 fit/dev（该表实测仅 4 个重复组）。
  - 划分算法：`seed=20261010`；组按 `sha256("cscl-v0|" + smiles) % 1000`
    排序 + 行内稳定排序，贪心填 fit 至 8000 行（与 ksvd 轨 basis-reuse 分组法
    同型），其余进 dev。划分只看 SMILES，不看 y。
- **fit 内部再分**：fit 8000 → fit_inner 7200 / monitor 800（同一算法，
  `seed=20261011`，同分组约束）。early-stop、best-epoch、Top-5 soup、
  标准化参数、类型词表、XGBoost 全部只用 fit_inner/monitor；**dev 只在
  各臂训练完成后各评分一次**（dev 是本轮臂间比较集，不是选择集）。
- 特征/词表/统计全部 train-only（fit_inner ∪ monitor 内按角色区分，见上）。

## 2. 结构单元划分（冻结，含预定义后备方案）

主方案 **RINGCHAIN-v0**（确定性、标签盲、非重叠、保环）：

1. 原子归属：计算 biconnected components（block 切割）。
   - 环 block（边数 ≥3 的 2-connected block）经并查集按共享原子合并为
     **环系单元**（覆盖稠环/螺环/桥环）；
   - 不属于任何环 block 的原子按连通性组成 **链单元**（含全链分子 =
     单链单元、连接两个环系的链 = 一个链单元）。
2. 键归属：单元内键（两端同单元）/ 单元间键（两端不同单元）。由原子划分
   唯一决定，无键丢失或重复；单元间键在关系集合中**无向计一次**。
3. 单元类型签名：单元内部图（原子类型 28 类 + 键类型 4 类 + 单元种类位
   {环系, 链}）上做 4 轮 WL 颜色细化，签名 = (kind, size, 排序后的最终颜色
   多重集 hash)。签名对原子重标号不变；**不同构单元可能极小概率碰撞**
   （WL 局限，如实记录，不做逐点 canonical 化）。
4. 词表（train-only）：fit_inner 上出现次数 ≥3 的签名 → 类型 id；
   其余 → 按 (kind, size 桶) 的 UNK 桶（UNK_RING / UNK_CHAIN，size 桶边界
   1,2,3,4,6,8,10,∞）。dev 未知签名落入同规则 UNK。UNK 桶可比较、可导出。

**预算定的后备方案（若主方案过粗，自动切换并记录）**：环系保持整体，但链
单元在 degree≥3 的分支原子处切开（每个分支段为一个单元，分支原子归其
"最小编号邻环/邻段"一侧），以增大词表粒度与复用率。触发条件见 §8。
不进行划分参数网格搜索。

## 3. 模型（冻结）

记单元 i：类型 t_i，类型嵌入 E_u[t_i]∈R^16；描述子 c_i∈R^d（单元内原子/键
直方图、size/log-size、环数、内键数 + 外部上下文：跨单元键数、环/链邻居数、
是否末端），按类型中心化（fit_inner 上每类型均值 μ_t；UNK/低频用全局均值）。

- **A (additive)**：`ŷ = b + Σᵢ [ wαᵀE_u[tᵢ] + wδᵀMLPδ([E_u[tᵢ]; cᵢ−μ_{tᵢ}]) ]`
- **B (relational，主方法)**：A + 关系项。关系集合 = 所有单元间键（无向，
  每单元对 ≤1 关系槽）。关系特征 r_ij = 两单元间键类型直方图(4) + log1p
  键数(1)，fit_inner 中心化(ν)。端点按类型 id 排序，
  `γᵢⱼ = wγᵀMLPγ([E_ru[tlo]; E_ru[thi]; rᵢⱼ−ν])`，
  `ŷ = b + Σᵢ(αᵢ+δᵢ) + Σ₍ᵢⱼ₎γᵢⱼ`。
- **C (shuffled-relation)**：与 B 同架构同参数量；将全体关系槽的输入向量
  在同一 split 内整体随机置换（seed 固定，train/eval 同一置换，保留每分子
  槽数与全局边际，破坏真实对应）。目的：机制对照，非合理分子生成。
- **D (opaque)**：同输入（类型嵌入表 + 描述子 + 关系特征），DeepSets 型
  图级回归：h_u=MLP([E[t];c])，p=MLP([h_i;h_j;r])，图向量=[mean/sum h,
  mean/sum p, log counts] → MLP 头。无非线性全局旁路以外的结构约束，
  容量与 B 相近（记录参数量表）。
- **E (XGBoost)**：图级扁平特征（单元类型计数向量 + 单元描述子聚合
  [mean/std/sum] + 关系类型对计数 + 全图原子/键直方图 + size 标量），
  xgboost 2.1.3 默认参数 + early stopping on monitor；同一 fit/dev。
  定位：非 GNN 参考线，不与 A–D 严格参数匹配（如实记录信息差异）。

统一训练配置：hidden=64、一层 MLP、SiLU；AdamW lr=1e-3、wd=1e-5、
batch=128、≤300 epochs、monitor 上 early-stop patience=30；
可辨识性惩罚 `λδ·Σₜ(mean_{fit,type=t} δ)² + λγ·(mean_fit γ)²`，λ=0.01；
y 标准化（fit_inner 统计），MAE 汇报回原始 y 单位；Top-5 monitor checkpoints
权重平均 = soup；**主指标 = soup dev MAE**（best-epoch dev MAE 作副表）。
同 seed 下 A/B 的 unary 路径共享初始化（同一起点构造），C/D 用同 seed 独立初始化。

## 4. 预算与种子（冻结）

- 冒烟：seed 0 全臂，限 ≤1 GPU 小时，只检查实现与量级，不作结论。
- 正式：seeds {0,1} 配对（同划分同置换规则），GPU1 (`res`, pool `res-gpu1`,
  进程内 `cuda:0`)。不追加超参搜索；不因结果追加种子。

## 5. 指标与比较规则（冻结）

- 主表：各臂 × seed 的 fit/dev y-MAE(soup)、参数量、训练时间、GPU 型号/显存。
- 配对差（同一 seed 同一分子配对，bootstrap 2000 次重采样分子，95% CI）：
  A−B、C−B、D−B、XGB−B（正 = B 更差）。
- 种子不确定性与分子重采样不确定性分开汇报（seed 间差 vs bootstrap CI）。

**性能容忍（本轮决策标准）**：以 dev MAE 计，
- COMPETITIVE：`MAE_B − MAE_D ≤ +0.010` 且 `MAE_B − MAE_XGB ≤ +0.010`；
- DEGRADED：`> +0.020`（任一）；
- 两者之间：WARN 带，需两 seed 方向一致且 CI 半宽 < 0.010 才可判 COMPETITIVE，
  否则记"不可判"。
- 理由：内部 dev y-MAE 量级预期 0.15–0.20（历史背景），+0.010 ≈ 相对 5%；
  这是本轮工程容忍，不是统计学普适阈值。

**关系价值判据**：关系建模有增量价值 = (A−B) 与 (C−B) 的配对均值 > 0、
方向在两 seed 一致、且至少一个的 95% CI 不含 0；否则记"无证据"。
不把 CI 跨零解释成"两法等价"。

## 6. 解释评价（冻结，独立于性能结论）

1. **计算完整性**：每分子 `pred = b + Σ(α+δ) + Σγ`，容差 1e-5（fp32 累加）；
   导出表：分子ID、单元ID、类型、原子/键索引、α、δ、关系端点与 γ、预测。
2. **稳定性**：原子重标号不变（1e-6）；无向关系端点互换不变；batch 组成/
   顺序不变；同型单元跨分子同一类型 id；两 seed 贡献分布（α 按类型、γ 分布）
   相关性与符号一致率；UNK 处理一致性。
3. **合成真值**：按 ZINC 划分器生成 ~4000 个合成分子（真实结构 + 人造 y）：
   `y = Σ_t θ_t·count_t + Σ_{(t,t')} θ_{tt'}·count对 + ε`，θ 有正有负，
   含强共现但无交互的单元对（伪交互陷阱）与真交互对。检验：方向恢复率、
   真交互识别、伪交互误报、拟合模型向新组合外推。
4. **可辨识性审计**：训练后 per-type mean|δ̄_t|、mean γ（应≈0 by 惩罚），
   以及去除惩罚的对照（λ=0，仅冒烟规模）看分工是否塌缩。

## 7. 停止条件（冻结）

- 划分审计失败（§8 触发且后备方案仍不达标）→ 停，不训练。
- 任一核心正确性测试失败且无法修复 → 停，不出正式数字。
- `res` GPU1 不可用（以 `rr doctor` 为准）→ 完成 CPU/合成/冒烟后停，
  记录阻塞证据，不换服务器、不静默降级 CPU 正式训练。
- 正式结果按 §5 判 DEGRADED 且 B 弱于 XGB → 记录负结果，建议停止当前
  方法形态（不自动扩大搜索救活）。

## 8. 训练前 CPU 数据审计（标签盲，先行）

统计并落盘 `results/cscl_v0_units_audit/`：
单元数/大小分布、每分子单元数、覆盖检查（原子 100%、键唯一归属）、类型词表
大小、类型复用率（top-k 类型覆盖的单元比例、singleton 类型比例）、dev→fit
UNK 率、每分子关系数分布、零关系分子比例。

**切换后备方案的触发线（预注册）**：fit_inner 词表中 singleton 类型（仅 1 次
出现）占比 > 30%，或零关系分子占比 > 50%（主方法关系项无用武之地），或
单元大小分布 P95 > 15 个原子（过粗）。触发即切 RINGCHAIN-v0 分支后备并
记录；后备后仍触线 → 停止，报告阻塞。

## 9. 与历史数字的关系（冻结）

历史 dev ≈0.120/0.119（ksvd RAW/DICT）、0.182（hier-relation）、
official valid ≈0.119/test ≈0.081（分解模型）均为**不同协议**的历史背景：
辅助标签、不同划分、已消耗 test。本轮不与之混表比较，只作量级参照。
