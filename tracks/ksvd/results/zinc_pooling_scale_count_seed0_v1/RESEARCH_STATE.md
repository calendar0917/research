# RESEARCH_STATE — 下一轮入口（2026-10-05，pooling scale/count round 后）

本页是研究流程整理，不替代 `tracks/ksvd/results/*/DECISION.md`。所有数字来自已核验摘要；
未重算历史，也未读取 official-valid/test。目的：下一轮开工只读这一页 + 冻结的账本，减少
重读历史与无效支线。**开工先讲主要效应、解释改变与购买决定；合规篇幅不等于科学成功。**

## 0. 本轮（`zinc_pooling_scale_count_seed0_v1`）一句话

分类 **`NO_CANDIDATE`**（冻结规则 5：N-vs-C 0/5、N-vs-M 0/5、C-vs-M 3/5）。同信息、
同参数量、同 init/RNG/schedule 下，把 unary/pair 矩从 sum 改为按实际 n 归一化的 mean
（`N`）明显更差：四个端点全部 CI 分离为负（G0 cal −0.011631 [−0.015971,−0.007671]；
overall cal −0.011797），并同样弱于只读来源 `M`。仅恢复 count 可见性的 `C` 在**校准**端点
上优于 `M`（G0 cal +0.003722、overall cal +0.003560，CI 下界>0），但 raw 端点为负
（G0 −0.001053、overall −0.001525），增益来自 fit-median bias（C bias −0.031017 vs M
+0.013944），且该比较为跨 regime（本轮本地 CPU vs 来源 A100），不构成可买的增量。
`C` 达到本轮最好校准值 G0 cal 0.093661 / overall cal 0.095743（仍 >0.09，且是内部
`g`-MAE，不是 official `y`）。细节见本目录 `REPORT.md` / `DECISION.md` / `EXECUTION.md`。

## 1. 结构绑定死亡（原因已定位，救活不买性能）

固定 2×2 的`zinc_zero_binding_baseline_seed0_v1`（seed0，一条 N0 轨迹，train-inner dev）：
Phase A 证明 `S_M` 预测对结构绑定路径完全独立（等价模型 max|diff| 1.9e-6），可作
“推理无结构绑定”的干净参照；N0（首步起绑定置零）在预注册阈值上 **INCONCLUSIVE**
（G0 cal 差 −0.002613，CI [−0.006588,+0.001113]；overall +0.004569）。决定：保留压缩
`S_M` 工作参照，不买第二 seed，不扫幅度/WD/容量，不再做 rescue 轨迹。**“通道活着”
不是性能目标。**

## 2. 旧任务字典桥（码稠密；匹配 MLP 无可靠优势）

`zinc_task_dictionary_and_cycle_witness_seed0_v1`：主 gate **INCONCLUSIVE（无字典优势）**，
CPU witness 判定为真实输入 aliasing。新 dev overall cal：`D 0.122960` vs `M 0.117032`
（gain −0.005928，CI [−0.014148,+0.001230]，负值利 MLP）；G0 cal 为平局
（`0.104300` vs `0.103305`）。唯一稳健差异是拟合集重尾容量：fit `k<=-3` cal MAE `3.962`(D)
vs `1.150`(M)。结论限定为“这个 16 步/288 原子 ISTA 字典 + 这个折”，不是“所有字典无用”。

## 3. 当前 local tuple 字典/MLP（机制健康；新折未确认方向收益；冻结家族）

`zinc_local_tuple_dictionary_vs_mlp_seed0_v1`（旧内部折）：primary encoder 比较
**INCONCLUSIVE**，方向 M_J（同参数 SiLU 投影）优于 IHT 字典（四个 dev 端点方向一致，
CI 全部跨零；fit 上字典更好）；性能 gate 4/5，G0 cal CI 含零。
`zinc_local_tuple_fresh_fold_replication_seed0_v1`（预注册新折）：`M` vs `B` 四点方向
保持（G0 cal +0.000549，overall cal +0.000888），但冻结五位 gate 只过 2/5（两个 raw 方向
项），G0 cal CI [−0.003438,+0.004754] 跨零且宽于 ±0.003；旧折 +0.003659 未复现。
机制 HEALTHY（A drift 0.713、W_loc 7.46、0/64 dead、injection RMS 0.594、zero-ablation
0.0993→0.5841）。决定：冻结该 local tuple 家族，不再加编码器/折/seed/阈值；工作参照保持
`M`；**机制健康 ≠ 泛化增量**。

## 4. cycle 极尾（输入 aliasing / helper 顺序依赖与覆盖问题；与主体化学分账）

`zinc_long_cycle_audit`（12k 含 valid 的诊断，只读摘要）：极端负 `y` 由目标定义中的
long-cycle penalty 主导（y<−10 的 12 条全部 label cycle<0）；但 `label_vs_gvae_order_mismatch
= 19`，`stored_order_label_flip` 示例存在；置换敏感 34/655（5.2%），其中 extremes 34/387
（8.8%）、非极端 0/268；min-basis 在 2 例上不变但丢 penalty。决策记录为
“GO – GLOBAL TOPOLOGY (with benchmark-artifact caveat)”。**当前 `g`/G0 误差不能全归给环；
cycle 线与其 benchmark-artifact 需要单独账。**

## 5. 主体化学 `g`（不同折约 0.10；无已证主瓶颈）

`g = y − c` 主体化学分量在不同折上约 0.10（fresh-fold `M` overall cal 0.099304，G0 cal
0.097383；G0 贡献 0.094218，占 overall 的 94.9%）。旧折 `M` overall cal 0.099381、
G0 cal 0.098216。**目前没有被实验证明的主瓶颈**：G0 是 k=0 样本组，不是“global 模块”的
别名；G0 误差占比高只说明误差质量集中，不说明 global 机制是原因。本轮只补上“聚合尺度/
计数可访问性”的一个判别。

## 6. 科研流程为什么可能低效（用本轮前的账本）

1. **比较对象漂移**：`y` vs `g`、旧 Full/压缩骨架、冻头 vs 端到端、旧折 vs 新折，不能拼成
   同一条进度；主表必须绑定同一输出目标、同一骨架、同一 fold 与同一训练条件。
2. **阴性范围过窄却被当作换方向依据**：结构绑定、字典桥、local tuple 的阴性各自只关闭一组
   固定配置，但历史上被读成“换赛道”；不同微干预没有积累成同一个架构因果问题。
3. **通道活跃/输入可重构/消融承重都不等于可泛化增量**。机制健康是必要条件，不是购买理由。
4. **单 seed 的行 bootstrap 不能替代训练随机性**；未过 gate 的小点值不应持续购买新支线。
5. **历史索引、广播、bootstrap、标定/标签来源错误需要一次性公共回归保护**，不能每轮复制
   实现再重新审计；本轮的 pooling 参考检查、count witness、replay、witness 都是可复用模板。
6. **合规/溯源是必要条件，不是科学成功**；下一轮开头先讲主要效应、解释改变和购买决定。

## 7. 新工作规则（自本轮起生效）

* 固定一个工作参照（当前：fresh-fold `M`，推理无结构绑定）、一个主指标/诊断目标
  （内部 dev `g`-MAE，G0 cal 为主端点）和一个现用 fold（fresh-fold 8000/2000, seed0）。
  新结果必须标注折/目标/条件，不按最漂亮口径选择。
* 一条方向最多一次探索 + 有实质信号后一次确认；达到次数即冻结，重开必须有新的判别证据，
  不能只换系数/阈值/名字。
* 每轮开工前写清：竞争解释、哪些结果会改变决策、预算与停止条件（含“无候选即停”）。
* 保留性能目标与 luyin19 主张的关系表：冻结接口的局部正结果属于骨架改进，不证明字典核心/
  correspondence；字典要回主线必须对匹配非字典对照买到可靠增量。
* 不每轮自动生成“唯一下一审计”；无候选时结束该支线，只列一个未解决问题及所需证据。
* 现 dev 已被多轮使用；若买到明显候选，下一轮才设计一次冻结的确认/正式验证。official-valid
  与 test 继续不读。
