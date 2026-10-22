# DESIGN — zinc_structural_object_tokens_v1（两臂单模型，seed 0）

**状态：预注册（在正式 GPU 训练之前写定）。**
Track `structural-sufficiency`；本轮代码 `tracks/structural-sufficiency/zinc_structural_object_tokens_v1/`。
旧 BF 轮（commit `784d0ad`，报告 `5255cda`）只做统计 erratum，训练/预测不覆盖。

## 1. 边界

- 数据：official PyG ZINC subset，train 10,000 / valid 1,000 / test 1,000。
  train 标签训练；每 epoch 全量 valid；strict `<` 选 best（相同取早）；冻结后两臂各一次
  descriptive test。官方 test 在本仓库已被多次读取，本轮不声称盲测。
- 目标：<=500,000 推理参数，raw-y test MAE 进入 0.05x。当前没有证据保证达到。
- 不做：递归局部邻居 hidden-state 更新、真实 SA/logP/cycle 分量、y 反推、推理 gate、
  whole-pattern 查表、VQ/codebook、teacher、ensemble、平均预测、第三候选、seed 1、
  消融或网格。
- 允许：原始图静态统计/对象提取；完整分子范围的全局 attention；静态输入 cache
  （只含内容/成员/静态关系，不含正在训练的 hidden state 或 φ 输出）。
- 复用：P3 数据 cache（`ad_{train,valid}.pt`）、BF 的 S3/S4/F4 定义与 canonical key
  （A 臂直接复用 `bf_data`）、seed-0 train-only masked-SSL 预训练文件
  （sha256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`）。
- 时间：A 约 25 分钟，B 约 30–40 分钟，两卡并行，目标约一小时收口（合计 ~1–1.2 A100-h）。

## 2. A_joint_BF

**唯一变量：训练起点 + 完整训练方式。模型与静态内容与旧 BF 相同（447,505 参数）。**

- 结构：P3 原模块（atom/bond/junction encoder、Null composer、reader）完全不变；
  BF 的 S3/S4/F4（root+3 / root+4 / root-u-{w1,w2}，完整 typed induced adjacency，
  root 固定在 canonical slot 0，不同实际节点集合保留 multiplicity，family 独立，
  空集为精确空集）经 φ(98→64→32) 聚合成 198 维 `s_v`，由 `W_s∈R^{128×198}`
  零初始化进入 atom encoder 第一层：`SiLU(W_x raw108 + b + W_s s_v)`。
- 初始化（顺序固定，`build_model_a`）：
  1. `torch.manual_seed(0)`；按历史顺序构造 P3 核心（与 `A2.build_model("P3N", seed 0)`
     的构造序列一致）；
  2. 只加载 seed-0 train-only masked-SSL 的 atom/bond/junction encoders（sha256 核对）；
  3. 追加 `w_s`（零）与 BF branch（正常随机初始化）。
  **禁止**加载监督 P3B `best.pt`/`final.pt`，**禁止**从旧 BF epoch 100 续跑。
  `ot_checks` 校验：22 个 P3 参数张量 + 2 个 buffer 与历史 `BJOC(seed0)+load_encoder`
  逐 tensor 位级相同；与监督 P3B 的 composer/reader 明显不同；`W_s` 精确为零。
- recipe（P3 recipe B，单组 LR）：AdamW，全模型 peak lr **1e-3**、min **1e-5**、wd **1e-5**；
  batch 128、`drop_last=False`（79 updates/epoch）；warmup 500 updates、per-step cosine；
  **固定 600 epochs / 47,400 updates**；raw-y L1、clip 5、dropout 0。无 patience、无早停、
  无两组 LR。只在非有限值/真实故障时中止。
- 计时：先跑 3–5 个完整 epoch（含全量 valid），记录 steady epoch time，然后重建
  完全相同的初始模型/optimizer/RNG 再跑 600 epochs。若 `steady*600 > 3600 s`，
  记录 `status=INCOMPLETE` 并停止（不把 600 epochs 拖成数小时）。
- epoch 0（SSL 起点 P3，`W_s=0`）作为候选点记录（它不是监督 P3B）。

## 3. B_object_tokens

**主要性能尝试：结构对象成为主干 token，与 atom token 一起进入同一预测网络。**

### 3.1 三类物理对象 / 五种 token

| token | 定义 | 数量（train） |
|---|---|---:|
| A | 每个原子恰好一个 token（覆盖全部原子） | 231,664 |
| S3 | BF rooted：root + 3 个不同邻居（4 节点） | 82,568 |
| S4 | BF rooted：root + 4 个不同邻居（5 节点） | 3,104 |
| F4 | BF rooted：root-u-{w1,w2}（4 节点） | 247,704 |
| C | 原图全部长度 3–8 的 **chordless** 简单环，每个实际环一次 | 27,808 |

- 每个 `(root, family, 实际节点集合)` 是一个 token；相同 canonical 内容但不同节点集合
  **不合并**（multiplicity 保留）。不同 family 各自存在。S3/S4/F4 内容直接复用
  `bf_data.canonical_key`（root 固定 slot 0，完整 typed induced adjacency）。
- 环：`chordless` = 环上非相邻节点之间没有额外边；穷举长度 3–8（无 SSSR、无
  cycle_basis、无采样、无按 atom ID 截断）。同一实际环按 node set 去重（chordless 环的
  node set 唯一）。canonical 内容 = 所有 `2L` 个旋转/反向实际序列上 typed 内容的字典序
  最小；`bond[i]` 始终连接 `seq[i]` 与 `seq[(i+1)%L]`（atom/bond 数组一起旋转/反向）。
  环没有 root。无环图是精确空集，不造 dummy token。
- token 类型 id 固定 `A=0, S3=1, S4=2, F4=3, C=4`（5 类）。atom index 只用于 gather、
  membership、距离等置换等变操作，不是可学习身份。
- 数量核对：train/valid 的 S3/S4/F4 总 occurrence 与每个 `(molecule, family, root)`
  计数与冻结 BF cache 精确一致（`analysis/cpu_checks.json`）。

### 3.2 编码与 token 初始化（宽 96）

- atom encoder：raw108 →128 →96（复用 P3 结构与 train-only SSL atom_encoder 权重，
  26336 参数，监督阶段正常 fine-tune）。P3 的 bond/junction encoder 与 composer/reader
  **不创建**。
- 共享 raw 类别 embedding（各一张表）：atom 28×8、pair 4×4（0=无边、1..3=真实 bond）。
  padding 乘 mask。
- branch 内容 φ：`[5×atom_emb(8), 10×pair_emb(4), node mask 5, pair mask 10, family one-hot 3]`
  = 98 →64 →32，SiLU，dropout 0。
- ring 内容 φ：`[8×atom_emb(8), 8×沿环 bond_emb(4), node mask 8, edge mask 8, len one-hot 6]`
  = 118 →64 →32，SiLU，dropout 0。
- 内容 id 只是地址（不等于 `Embedding(fragment_id)`）；新内容也由原始内容编码。
- 令 `a_v` 为 live atom encoder 输出，所有均值都在当前图内对真实成员计算：
  - `h_A(v) = Linear96→96(a_v)`
  - `h_F(o) = MLP228→96→96([a_root, mean_{u∈members} a_u, φ_branch32, log|members|, family_oh3])`
  - `h_C(o) = MLP129→96→96([mean_{u∈ring} a_u, φ_ring32, log L])`
  - 各 MLP 只有中间层 SiLU；bias 计入。加对应 5×96 类型 embedding。

### 3.3 静态关系 + 四层全局 attention

每个 token 有成员集合 `S_o`（atom token 为单原子）。静态关系（只从原始图预计算）：

1. 有序 token type 对 5×5 = 25 类；
2. `|S_i∩S_j|` 0..8 = 9 类；
3. `min_{u∈S_i,v∈S_j} dist(u,v)`：0..8 精确、9 = 有限且 ≥9、10 = 不连通 = 11 类；
4. 仅 atom-atom 的直接 bond：0=不是 atom-atom、1=atom-atom 无边、2..4=bond 1..3 = 5 类；
5. root 关系：0=至少一方无 root、1=都有但不同、2=同 root = 3 类。

每类关系一张 `(n_cat × 4 heads)` 表，五项相加为每对 token 的加性 logit bias；四层共享
（212 参数）。不构造 DBT 式 `[B,H,T,T,head_dim]` relation value。

主干：4 × pre-LN block，width 96、heads 4、FFN 192、SiLU、dropout 0.05；
`LN→QKV→full-graph attention(+bias)→out→residual`，`LN→FFN→residual`；带 bias 线性层，
每 block 两次 LN，无额外 final LN。attention 只屏蔽 padding 与不同分子；真实 token 对
全部可交互；不改局部邻接/共享成员 hard mask；padding query 输出与 residual 清零。
关系只进入加性 bias。

### 3.4 读出

atom / branch(S3+S4+F4 合并) / cycle 三组最终表示分别取
`sum / 固定 train 平均真实对象数`（下限 1）与 `mean`（空集精确 0），拼 6×96=576；
加标准化 global33 与三组真实对象数的 `log1p` → 612。
Reader：612→96→48→1，SiLU、dropout 0.05，raw-y 输出。pooled 组不做逐分子 LayerNorm，
没有第二预测器或 P3 shortcut。

### 3.5 参数（实际实例化计数）

| component | params |
|---|---:|
| SSL atom encoder | 26,336 |
| 共享 raw embedding | 240 |
| branch φ | 8,416 |
| ring φ | 9,696 |
| 三类 token 投影 | 62,400 |
| 类型 embedding | 480 |
| 4 × Transformer block | 299,136 |
| 关系表 | 212 |
| reader | 63,553 |
| **合计** | **470,469** |

### 3.6 recipe 与日程

- seed 0；只加载 SSL atom encoder（sha256 核对），不加载监督 P3B / DBT / 旧 BF。
- AdamW，全模型 peak lr 1e-3、min 1e-5、wd **1e-4**；batch 128（79 updates/epoch）、
  warmup 500、per-step cosine；raw-y L1、clip 5；encoder/内容 MLP dropout 0，
  Transformer/Reader dropout 0.05。无 aux / EMA / soup / 重采样。数值精度保持 fp32 默认。
- 先跑 4 个完整 epoch（含全量 valid、真实 collate 与关系读取）计时；随后重置全部
  权重/optimizer/RNG。
- `E_B = min(600, floor((2400 − 已用 GPU 秒 − 120)/(1.15·t_epoch)))`，向下取整到 10 的倍数；
  **只由速度决定**，不按计时 valid 选择 recipe。记录 `E_B` 与 `79·E_B` 后跑完整短 cosine。
  `E_B < 300` 时先优化实现；仍不满足则交付 A 结果 + B 的瓶颈与代码，不偷跑数小时。
- 训练中不重置/压缩/延长 cosine；故障则标 `INCOMPLETE` 并保留恢复状态。

## 4. 必要的窄检查（`code/ot_checks.py`，11 项）

A：初值 == SSL-start P3（非监督 P3B）、`W_s=0` 时 epoch-0 前向与同初值 P3 位级相同、
全部参数可训练且在同一 optimizer、φ 首步梯度 0 并在第 2 步打开。

B：原子 token 覆盖（每原子恰好一个）、S3/S4/F4 与 BF 的 occurrence 及
`(molecule, family, root)` 计数精确一致、`(family, root, node set)` 无重复而相同内容
不同节点集合保留、手工 star/fork/path/6-ring/带 chord 6-ring/K4/5-ring 与独立
subset 穷举一致、环 canonical 内容在旋转/反向下不变且能被独立排列枚举复现、
`atom/bond` 对齐（反向 atom + 正向 bond 不会被视为 canonical）、原子重编号后内容/
对象数/关系/预测不变、batch 重组与跨图隔离、空 branch/空 cycle、共享 raw embedding
只有一份、参数计数、所有有效模块梯度非零（含真实分叉与环的 batch）、静态 cache
无 `requires_grad`/`grad_fn`、LR/updates/checkpoint 重载语义。

不跑 KSVD 全量回归、不为普通分析生成新审批文件。

## 5. 冻结与 test

- 两臂各自 best-valid 后写 `analysis/<arm>/freeze.json`（含 best sha256、epoch、valid、
  初值来源/hash、参数、recipe、cache hash）；两臂都完成后统一读 test。
- test 只做 descriptive benchmark：不改 checkpoint/epoch/recipe/对象家族，不触发第三模型，
  不因 test 切换 final。若发现实现错误则披露并标旧 run 无效。
- 复用 P3B/L/DBT 已保存预测；主表含全部参数、初始化、best-valid、best epoch、
  eval-mode train@best、final-valid、test MAE、实际 epochs/updates/GPU 时间。
- 固定四组 covered/unseen × ordinary(y>-3)/tail(y<=-3) 只作分析；保留 row 172 与全部分子；
  不做新统计工程，单 seed 结果不写成机制确认。

## 6. 停止规则

- A 改善 → 新候选值得继续；不能证明上一轮失败一定由早停造成。
- B 改善 → 整体 token 系统值得继续；不把收益单独归给环、参数分配、bias 或 attention。
- 两臂都没赢 → 停止本轮，不自动开第三 arm/seed 1/更多环长/宽度/预训练网格。
- 完整短日程失败只约束本配置与预算，不外推“显式结构无用”或“0.07 是理论上限”。
- 区分信息覆盖、训练拟合与泛化；CPU 门控通过不是性能证据。
- 微小差异如实写“没有新的性能水平”，不追加数小时。
