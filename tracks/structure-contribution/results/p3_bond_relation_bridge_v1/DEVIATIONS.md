# cscl-p3-bond-relation-bridge-v1 — 执行偏差披露

本文件记录**在标签读取之后**才发现、无法通过 amendment 修正的执行偏差。它不修改任何
冻结文件，只如实披露“实际做了什么”和“结论因此受到什么限制”。

## D-1（重要）split F 实际上等于历史 split E，不是新的内部划分

**事实**：`split_F.json` / `split_F.json` 的 `fit_rows` / `monitor_rows` / `locked_rows`
与 `results/bond_relation_attribution_v1/split_E.json` **逐行完全相同**
（fit 7,996 / monitor 1,000 / locked 1,000；行哈希 `b73ed7f7…` / `77e79463…` / `7015fb01…`
在两轮一致）。

**原因**：`cmd_split` 调用
`bra.build_identity_split(rows, keys)` 时**没有传入** `salt=pb.SPLIT_SALT`，
而该函数的默认参数是历史常量 `bra.SPLIT_SALT = "cscl-bond-relation-attribution-v1"`。
因此分组顺序沿用了上一轮的盐值，贪心填充得到完全相同的三个区。
`split_F.json` 中记录的 `"salt": "cscl-p3-bond-relation-bridge-v1"` 是**错误的元数据**
（它写的是本轮常量，而不是实际参与构造的盐值）。该 JSON 已冻结，不做改写。

**影响**：

1. 本轮 Locked/Monitor 与上一轮 `bond_relation_attribution_v1` 是**同一批分子**，
   因此本轮**不是**独立复现，两轮数字也不能当作两份独立证据合并。
2. 任务要求“使用新的冻结内部划分 F”**未被满足**；这是本轮最主要的执行偏差。
3. 训练集（Fit）也与上一轮相同。本轮所有模型都只在这 7,996 行上训练，
   Locked/Monitor 的标签在 `score-holdout` 之前不可达（`LabelGuard` 计数为 0，
   单次读取锁 `holdout_read.lock` 记录了唯一一次读取），因此
   **臂间对照（P3R vs P3 vs P3U）本身仍然是无标签泄漏的配对比较**；
   受损失的是“跨轮独立复现”这一性质，而不是本轮三臂比较的内部有效性。
4. 没有、也不能用“换一个盐值重跑”来补救：新的 1,000 行 Locked 里约 80% 会落在
   本轮已经用于训练的 Fit 行上（因为 Fit 覆盖了 9,996 行中的 7,996 行），
   那将是训练集评估，比现在更糟。本轮单次读标签也已用掉。

**为什么没有在训练前发现**：CPU 门禁（G1–G8）检查的是核/参数/前向/梯度/评测代码，
没有任何一条门禁对比“新划分与历史划分是否不同”；`split` 阶段只断言了尺寸与
`cross=0`，没有断言“与 E 不同”。这是门禁设计的缺口，已在本文件中记录；
未来同类轮次应加入“新 split 与所有历史 split 的行集合不相交（或至少不完全相同）”断言。

**对判定的影响**：`CASE_D_NO_GAIN_ON_P3` 依赖的是同一轮内三臂的配对差值
（同 Fit、同 SSL、同 batch order、同配方、同评测分子），不依赖划分是否“新”。
但它的证据强度应被表述为“在 split E 这批分子上的两 seed 配对测量”，
而不是“在新留出集上的独立复现”。

## D-2 预算口径

`gpu_budget_ledger.json` 只累计 SSL 与监督训练（1.9896 GPU-h / 4.0）。
预测、换核诊断、评分、post-hoc 为短时 CPU/GPU 前向（合计 < 5 min），未计入 ledger；
`finalize` 的 evidence manifest 中如实记录了这个口径。
