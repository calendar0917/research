# DESIGN — zinc_local_factorization_v1

Track `structural-sufficiency`。**两臂、单模型、seed 0**：在完全相同的 P3 载体、完全相同的
S3/S4/F4 局部对象、相同的内部置换平均、相同的 198 维接入口、相同的初始公共权重、相同的
数据顺序、recipe 和 horizon 下，只比较**局部内容函数 `phi` 的参数化**：

- **D = LF-Dense**：普通内容 MLP，`98 -> 80 -> SiLU -> 32`，10512 个独有参数，全模型 449,601；
- **F = LF-Factorized**：共享原子/键投影 `A(8->32)`、`B(4->32)` + 角色表 `R_atom[5,32]`、
  `R_pair[10,32]` + 逐通道 1/2/3 阶矩 `z1/z2/z3` + `114 -> 64 -> SiLU -> 32` 头，
  10368 个独有参数，全模型 449,457。

两者只差 144 个参数（0.032%）。本轮**不**加第三臂、不扫 rank/阶数/模式/初始化/读出、
不加 seed 1、不新开预训练、不再扫 SAM/正则化。历史 P3B / L / A_joint_BF / 2026-10-22 的
P3-Control 与 P3-SAM 只作描述性参照，不重训。

## 问题

已有事实：P3 已跨原子共享 MLP；BF 联合训练（`A_joint_BF`，valid 0.076731 / test 0.070067）
在增加同一批模式内容后没有出现新的性能水平；固定 rho=0.05 的 SAM 明显变差（已关闭）。
因此本轮问：**在结构内容与载体不变时，把「每个模式一个完整展开向量 + 普通 MLP」换成
「共享因子 + 角色调制 + 低阶乘积统计」，能否改善有限样本泛化？**

这是局部编码实验，不包装成新主干，也不声称新增结构信息。允许负结果；负结果只关闭
"这个 rank 32 / 最高三阶 / 这个接入方式"的组合，不否定所有因子化或组合共享。
乘法/张量交互已有先例（Eijkelboom et al., PMLR 221），本轮的 primitive 不是新的。

## 固定不变的部分（两臂共享）

- official PyG ZINC subset：train 10,000 / valid 1,000 / test 1,000，raw-y L1，无变换/重加权。
- P3 载体：`ad2_models.P3NullModel`（413,505 参数）+ 198→128 无 bias、零初始化的 `W_s`：
  `a_v = Linear128→96(SiLU(Linear108→128(raw108_v) + W_s s_v))`；Null grouping、composer、
  reader、global33 完全沿用。
- 初值：`torch.manual_seed(0)` 按原构造顺序生成 P3 核心，只加载 train-only masked-SSL 的
  atom/bond/junction encoder（`pretrain_encoder_s0.pt`，sha256 `006fabc1…bfa97`）；
  composer/reader 用 seed0 随机初值。两臂复制同一份公共 state_dict；`W_s` 精确为 0；
  内容 embedding 由独立 `CONTENT_SEED=101` 生成一份并复制；D/F 的 `phi` 分别由
  `PHI_SEED=202/303` 生成（独立 `torch.Generator`，不影响公共流与数据顺序 RNG）。
  禁用监督 P3B `best.pt`/`final.pt` 与任何已训练 checkpoint 作为训练起点。
- 局部对象（`bf_data` 审计版）：S3 = root+3 邻居（4 节点）、S4 = root+4 邻居（5 节点）、
  F4 = v-u 与 u 的另外两个邻居（4 节点）；完整 typed induced adjacency（含额外 chord）；
  `(root, family, 实际节点集合)` 计一次，同内容不同集合保留 multiplicity；family 独立；
  无对象是精确空集。`pattern id` 只是静态内容缓存地址，不是可学习 embedding，且是
  split-local（跨 split 只按内容 key 对齐）。
- 内容布局：5 个 atom 槽（词表 28）、10 个按字典序的无序 pair（0 = 真实无边，1..3 = 键型）、
  node_mask 5、pair_mask 10、family one-hot 3。padding 的 pair 用 mask 清零，不与"无边"混类。
- **内部置换平均**：对 n 个真实节点，固定 root 在 slot 0，枚举其余 n−1 个槽的全部排列
  （4 节点 6 个、5 节点 24 个），atom 与所有 pair 一起重排，先算同一个共享 `phi`，再平均
  32 维输出：`phi_bar(p) = mean_pi phi(pi·p)`。这是同一模型内部的对称化，不是集成；
  每个物理对象仍只贡献一次 multiplicity，6/24 个排列不被当成 6/24 个对象。静态索引
  缓存 + 批向量化；`phi` 每次 forward 都带 autograd 重算，不缓存训练中的隐状态。
- 聚合：每个 root、每个 family 内 occurrence-weighted `mean(phi_bar)`（32）、
  `mean(phi_bar²)`（32）、`log1p(真实 occurrence count)`（1）、existence（1），共 66；
  三 family 共 198。**平方发生在内部置换平均之后**；空 family 全 0。
- 训练：seed 0；batch 128（`drop_last=False`，79 updates/epoch）；全部参数单组 AdamW，
  peak 1e-3 / min 1e-5 / wd 1e-5，betas (0.9,0.999)、eps 1e-8、warmup 500 + per-step cosine、
  clip 5、dropout 0、fp32；每 epoch 全量 valid，strict `<` 选 best（相等保留更早）；
  无早停、无 EMA/SWA/soup、无两组 LR、无辅助任务。两臂逐 epoch 的 batch order 由同一个
  独立 RNG 生成，order hash 必须逐 epoch 相同。

## 只在本轮出现的差异

| | D | F |
|---|---|---|
| 每排列输入 | masked atom emb 40 + pair emb 40 + node_mask 5 + pair_mask 10 + family 3 = 98 | 15 个 32 维字段（5 atom、10 pair）由 `A/B` + 角色表 + mask 得到 |
| 组合规则 | 两层 MLP | 逐通道 `z1,z2,z3`（1/2/3 阶 elementary symmetric moment） |
| 输出头 | `98→80→32` | `114→64→32`（`z` 96 + masks 15 + family 3） |
| 独有参数 | 10,512 | 10,368 |

`F` 的 `m = sum(node_mask) + sum(pair_mask)`（4 节点 10、5 节点 15），
`z2=(p1²−p2)/(m(m−1))`、`z3=(p1³−3p1p2+2p3)/(m(m−1)(m−2))`，`m<k` 时 `z_k=0`；
分母只用真实字段数，padding 在 `p1/p2/p3` 之前就已被 mask 清零。角色表随置换一起变换，
不对应分子绝对 atom ID。

## 时间与判定

- 4 个真实计时 epoch（含完整对称化、两阶统计、backward、全量 valid、日志）后重置初值、
  optimizer 与 RNG 再正式训练；计时 valid 不用于选架构/rank/LR/horizon。
- 若 `1.15*t_slow*600+120 ≤ 3600 s` 则 E=600；否则两臂共同 `E=min(600, 10*floor((3000/(1.15*t_slow))/10))`；
  `E<300` 是停止条件（交付实现与瓶颈），不得擅自开数小时任务。
- 0.05x 目标是 test MAE 落入 `[0.050, 0.060)`；不把恰好 0.050 当唯一目标算差距。
- 判定：F 相对 D 若在多个 epoch 维持清楚的 valid 优势并在冻结 test 同向改善 → 值得继续；
  若只有约 1e-3 的单次差异 → 弱信号；若 train 更差但 valid/test 更好 → 可能是有益约束；
  若 train 与 held-out 都明显更差 → 关闭该 rank/阶数/接入组合；若接入口长期不起作用 →
  该接入方案未提供足够干预，**不能**据此说内容完全无用。

official test 在历史轮次已多次暴露；本轮两臂各在冻结 best 之后读取一次，只作描述，
不声称独立盲测，也不参与任何选择。四个 covered/unseen × ordinary/tail 分组只作事后描述，
保留 row 172 与全部极端值。
