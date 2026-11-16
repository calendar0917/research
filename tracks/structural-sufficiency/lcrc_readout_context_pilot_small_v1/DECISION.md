# DECISION — lcrc-readout-context-pilot-small-v1

状态：DEVELOPMENT 可行性实验，单 seed，两个官方小任务。Test 数字为 `DEVELOPMENT_TEST`。

## 已记录的判读

| 任务 | 轴 | 判定 |
|---|---|---|
| ogbg-molesol | A. 回溯 | TRACE_PASS |
| ogbg-molesol | B. 性能保留 | PERFORMANCE_LOSS_THIS_RECIPE |
| ogbg-molesol | C. 上下文增量 | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-molesol | D. 对 G 的竞争力 | WITHIN_5PCT_OF_THIS_GINE (A) |
| ogbg-mollipo | A. 回溯 | TRACE_PASS |
| ogbg-mollipo | B. 性能保留 | PRESERVED_POINT_ESTIMATE / HORIZON_CONSISTENT |
| ogbg-mollipo | C. 上下文增量 | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-mollipo | D. 对 G 的竞争力 | BEHIND_THIS_GINE |

## 一句话结论

**这套显式上下文化读出可以做到逐项回溯，但在本配方下它没有带来可用的性能收益：在 molesol 上替换 C1 读出造成了超出容差的损失，在 mollipo 上只是打平，而 C 相对 B 的上下文增量在 Val/Test 之间变号。停止扩大这一读出形式。**

## 逐条依据

1. **回溯（A 轴）通过。** B/C 的每个 Val/Test 预测都能由导出的分子级与 root 级项在容差内重建（最大 5.5e-7 原始单位）；32/32 个固定输入干预可由账目变化重建；`a_k×2/c_k÷2` 尺度等价精确；选中 checkpoint 重载后预测与账目一致。这部分与性能好坏无关。
2. **性能保留（B 轴）在本轮配方下失败于 ogbg-molesol。** C1 的后处理池化头（`rho`）仍然带来加性 root + 单项 global 拿不回来的精度；在 ogbg-mollipo 上 C 与 A 打平。
   措辞限定在本架构/本配方（R3, peak 3e-4, 360 epoch, 单 seed），不是对全部可回溯架构的否定；也不是「加性读出一定不行」的一般结论。
3. **上下文增量（C 轴）方向不稳定**（ogbg-molesol: last20_test +21.35%、last20_valid -8.19%、selected_test +20.50%、selected_valid -8.67%；ogbg-mollipo: last20_test -2.06%、last20_valid +8.53%、selected_test -1.56%、selected_valid +8.20%）。按预冻结规则记 `SPLIT_OR_SELECTION_SENSITIVE`：**不宣称上下文增益**。注意这与「机制没被使用」不同——第 6 节显示交互项在数值上确实被使用（RMS 达 0.19–1.05 个目标标准差、根间差异明显），只是没有产生两个 split 同向的收益。
4. **对 G 的竞争力（D 轴）**：只有 ogbg-molesol 的臂 A 落在冻结 GINE 的 5% 带内。G 的输入通路不同、LR 预设更强（1e-3 对 3e-4），所以这是距离标记，不是归因，也不是任何方向的 SOTA 声明。
5. **在 ogbg-mollipo 上三臂全部落后冻结 GINE**（Val / G：ogbg-mollipo 1.14–1.25×）。这一条必须和上面的「保留成功」一起报告：即使 C 保住了 A，它仍不在 G 的 5% 带内。

## 本轮下一步（以及不做什么）

* 本轮到此结束：两个任务、一个 seed、不换更友好的配置重跑。
* 真实成本 0.659 A100-h（软预算 2、硬预算 3），没有作废重跑。
* **不**追加第二个 seed、**不**加宽 `a_k`、**不**引入第三个上下文字段：这些都需要新的预注册，
  而且当前证据不支持它们会改变结论。
* 保留下来的是**机制与账目代码**（共享 trunk、预算匹配读出、账目/干预/尺度等价门禁、分析驱动），
  它们与具体读出形式解耦，可直接用于后续任何「可回溯读出」问题。
* 若以后仍要问上下文问题，新轮次必须预先声明：(i) 一个上下文本该有用的任务；(ii) 固定形式的交互 + 冻结 `h` 的基线，用来把「数值进去了」与「结构被读得不一样」分开；(iii) 多 seed 的稳定性预算，而不是单 seed 点估计。

分支 `lcrc-readout-context-pilot-small-v1`，正式轨迹执行 commit ee72310d9290，freeze hash `5a6d48aaef012aac…`。
