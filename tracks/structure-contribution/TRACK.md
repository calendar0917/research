# Track: 跨分子结构贡献与上下文交互 (CSCL)

| 项 | 内容 |
|----|------|
| slug | `structure-contribution` |
| 状态 | `active` |
| 创建 | 2026-10-10 |
| 主线位置 | 独立新路线；与 `tracks/ksvd` 平行，选择性继承其数据/特征基础设施，不继承其研究命题 |

## 问题（一句话）

能否构造一种**跨分子共享、结构上可对齐**的单元—关系表示，在不依赖 GNN/Transformer
消息传递的情况下实现有竞争力的分子性质预测，同时对局部结构贡献与上下文交互给出
**忠实、稳定、可检验**的分解？(CSCL: Contextual Structural Contribution Learning)

## 范围

- 做：
  - 确定性、保环、非重叠的分子结构单元划分与跨分子类型对齐；
  - 显式单元贡献（基础 + 上下文修正）与单元间关系贡献的可辨识分解；
  - 不以 GNN/Transformer/消息传递为核心的预测机制；
  - 贡献分解的计算完整性、稳定性与合成真值检验。
- 不做：
  - 不以证明 K-SVD/CSSD 必要性为中心（那是 `ksvd` 轨的历史命题，已停）；
  - 不使用 radius-2 重叠 ego patch 作为解释单元；
  - 不引入隐藏的全局非线性旁路；
  - 第一版不使用旧模型的 `g/ell/s/c` 辅助标签或冻结 Q 头（y-only）；
  - 本轮不声称化学因果机制——只声称模型预测归因。

## 协议 / 评估

- 是否复用全局图分类 strict：否（本轨自建 `cscl-v0` 协议，见 `notes/v0_protocol.md`）
- `protocol_id`：`cscl-v0`（只增不改语义）
- 主表角色：fit/dev 内部划分上的 y-MAE + 配对差（B−A/B−C/B−D/XGB）；
  official valid 仅作背景参照，official test 本轮禁读。

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | done（2026-10-10，research_core.md 已固化） |
| 精读 / 概念 | done-lite（prior_work_positioning.md，多处待核对） |
| 最小实验 | v0 进行中 |
| 交付 | — |

## 入口

| 路径 | 用途 |
|------|------|
| `notes/research_core.md` | **权威研究核心**（问题/假设/可反驳条件） |
| `notes/v0_protocol.md` | v0 预注册协议（先冻结后看结果） |
| `notes/prior_work_positioning.md` | 与已有工作的边界 |
| `code/` | 实现与跑法 |
| `configs/` | 配置 |
| `results/` | 本轨结果（JSON + 报告，本轨唯一数字源） |
| `tests/` | 正确性测试（划分/归因/不变性） |

## 禁止

- 与其它轨结果混表横比（除非单独 deliverable 写清协议）
- official ZINC test 参与任何本轮选择（已被历史消耗，禁读）
- GNN/Transformer 作为核心预测机制
- 用未解释的全局旁路掩盖可解释分解的代价

## 下一步

1. CSCL-v0 实现 + 测试（code/、tests/）
2. CPU 数据审计 → seed-0 冒烟 → GPU1（`res`）正式 A/B/C/D/XGB 对照
3. 按 v0_protocol.md 冻结的判断规则写报告与科研判断
