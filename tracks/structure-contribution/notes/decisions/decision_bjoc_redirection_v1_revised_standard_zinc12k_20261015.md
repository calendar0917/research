# 决定 — bjoc_predictive_research_redirection_v1 修订：标准 ZINC-12k 主协议 → `NO_JUSTIFIED_NEW_EXPERIMENT`

日期: 2026-10-15 · Track: `structure-contribution` · round: `bjoc_predictive_research_redirection_v1`（修订）
依据: `results/bjoc_predictive_research_redirection_v1/REVISION_01_standard_zinc12k_protocol_and_decision.md`
（原始三份文档 commit `22629ae`，**未修改**）；本决定不执行任何训练/GPU。

## 背景

研究者明确选择**标准 ZINC-12k、无额外预训练语料**作为主评估协议：监督与结构 SSL 都只允许
官方 train 10k 分子。原提案把 H-COVER（SSL 语料 10k→50k/220k，需 ZINC-Full）作为主线推荐，
并要求把该实验列入下一步；该推荐需要修订。

## 决定

1. **主协议 = 标准 ZINC-12k**：监督标签与 SSL 结构均只用官方 train 10k；official valid 为暴露 dev
   （单次冻结读取）；official test / Locked 不读；**外部无标签结构不进入主协议**。
2. **H-COVER 降级**为"额外无标签数据场景下的独立备选假设"，不再是主线；**本轮不授权**其 50k/220k
   GPU 实验（设计文本保留在 `EXPERIMENT_PLAN.md`，状态 = parked）。
3. **凡使用外部无标签数据得到的成绩，不得与常规 ZINC-12k 排行榜/基线视为同等数据条件的比较**；
   必须作为单独协议预注册、与同语料同算力对照比较、并在报告中显式标注数据条件差异。
4. **修订结论：`NO_JUSTIFIED_NEW_EXPERIMENT`** —— 标准协议下不存在证据充分、与已有方法有明确区别、
   且真正改变模型可用预测信息的新机制，值得一次 GPU 实验。

## 关键依据

* **证据重读（有标签 vs 无标签）**：`N8−N4 = −0.037304` 证明的是**有标签分子数**是最有效杠杆；
  `P−B = −0.0156 / −0.0207` 证明的是**同一批 10k 结构**的 SSL 再利用有效；曝光已饱和
  （96.8 次/分子、bond acc ≈0.995）。原提案用前者为"扩大无标签覆盖"提供正面先验，推理已更正。
  标准协议下 SSL 语料已覆盖全部可用结构，**"扩大覆盖"不是一个可操作变量**。
* **分布压缩评估（允许的问题）**：B1/B2/B3 证实表示确实压缩（归属盲区、有效秩 27.9、
  最近不同分子 reader 距离 3.35 vs Fit p01 14.26）。本轮新增只读检查（只读 official-train 标签，
  无模型前向）显示 20 组最近不同分子对的 `|Δy|` 中位数 **0.374**、均值 0.411、最大 1.051，
  随机对基线中位数 1.639、均值 2.108 —— 即当前测得的压缩主要把**目标相近**的结构压在一起；
  未发现"压缩导致标签不一致"的证据。其任何修复手段（二阶矩/池化/注意力）都属已被多次匹配否定的族，
  且与标准做法无区别、信息上接近冗余（`atom_sq` R² 0.927；junction moments 使 composer 秩 32.4→27.9）。
  工件：`b3_near_pair_train_label_check.json`。
* **关闭族清单**（不得重开）：Junction ownership/incidence（O/S/GLRC/Organized/D/E 六次匹配对照均未取得稳健增益：O−N +0.00301 跨零、O−S −0.00199 跨零、GLRC +0.0012 且 Locked 翻符号、Organized 0.108399 vs N 0.103423、D −0.0028 seed 翻转、E −0.0017）、
  固定签名/词表（INCONCLUSIVE 关线）、结构加性残差/校准（STRC 永久停止，Route B 已拒）、
  pair/距离通道（DLP、cross-bond、RW 多步、Soft5、三体）、SSL 目标搜索（含 TA-SSL）、
  配方搜索（R3 选择条件化后不追加）、尾部加权/目标变换、集成作为主张。

## 明确未决定 / 不执行

* 不修改 `notes/research_core.md`、`STATE.yaml`、历史结果与原始三份文档；
  核心条款修订以 `REVISION_01` §7 的 `PROPOSED_RESEARCH_CORE_UPDATE` 形式待研究者确认。
* 不自动执行任何确认性训练（新内部 split 复现或第二分子端点）——那是有效性审计，
  需单独授权与预注册，不是新机制。
* 不把分布压缩评估升级为实验：它是诊断，且当前结论为"无害"。

## 重新开启决策的条件

1. 未暴露 split/端点上出现系统性、标签相关的表示失败（如近邻对错分且不能由标签噪声解释），
   并伴随一个区别于已关闭族与标准池化做法的机制设计；
2. 研究者授权把"额外无标签数据"作为独立协议立项（H-COVER 或其它数据侧假设），
   并同意与主协议分开报告；
3. 出现新的评估端点/数据条件，使 10k 标签限制可被合法绕过，且候选机制能给出先验行为预测。

## 数据纪律声明

本轮未训练、未运行 GPU、未改模型/代码、未读 official valid/test/Locked 或任何新评价标签、
未改写历史结果。唯一新增读取为 official-train 标签（训练标签）用于上述只读一致性检查，
已记录源工件 sha256；`research_core.md` 与 `STATE.yaml` 未被改动。
