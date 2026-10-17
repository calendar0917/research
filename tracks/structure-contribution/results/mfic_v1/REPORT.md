# MFIC v1 最终报告：CPU 廉价信号通过，GPU Monitor 失败

协议：`cscl-bjoc-mfic-v1`
Track：`tracks/structure-contribution`
Split：V（7,996 Fit / 1,000 Monitor / 1,000 Locked，salt `cscl-bjoc-mfic-v1`）
最终 GPU 提交：`6eabb8b0b367fd5e9aa0a61a72ed6503f975c4eb`
日期：2026-10-17
最终决策：**`MFIC_NO_USEFUL_GAIN`**

## 0. 一句话结论

真实物理邻接矩阵的解析矩阵函数 `K = sum_{k>=2} 0.65^k S^k` 在 **CPU 固定 16 维 PCA-q +
Ridge** 探针中提供了非常强、四折一致的廉价信号（MAE -0.1008），但把这个 `K` 放进可学习的
`q` 投影和 158,721 参数 Reader 后，MFIC-R 在 Monitor 上 **比 P3 差 +0.004644**
（95% CI [+0.001032, +0.008585]，两 seed 同为正），RMSE 退化 +0.0370，tail 退化 +0.0255。
真实 `K` 相对 content-null `K_U` 只有 -0.002276 的方向性改善，未达到预注册的 -0.003，
且 CI 跨 0。Monitor 关键条件 8/10 失败，按协议停止；**Locked 标签未读**，没有 full10k、
official test 或任何 alpha/PCA/rank/Reader rescue。

CPU 结论仍然成立：`K` 里有 Base 之外的结构信号；GPU 结论是：**该信号没有转化为这个
one-shot 神经模型的 held-out 收益，反而伴随明显的训练集过拟合。**

## 1. CPU 科学购买门槛（通过）

完整记录见 `REPORT_CPU.md`。摘要：

- 4 折 identity-grouped Ridge，固定 `alpha=1`；每折只用训练折原子拟合 16 维 PCA；
- CPU-Base 68 维；CPU-Matrix = 68 + 136 维 `upper_tri(Q^T K Q / max(1,n))`；
- 平均 MAE：Base **0.496257** → Matrix **0.395422**，Δ = **-0.100835**；
- 4/4 折改善（-0.108719 / -0.113480 / -0.081242 / -0.099899）；
- trimmed-10 Δ = -0.092055；ordinary Δ = -0.090967；tail Δ = -0.228014；
- 平均 60.7% 分子变好；16 维 PCA 平均解释率 94.3%；
- 数学门全 PASS；模型审计 20/20 PASS；无 official valid/test、无模型预测、无 Monitor/Locked
  标签进入 CPU 探针；
- `decision_cpu.json`：`MATRIX_FEATURE_CHEAP_SIGNAL`，6/6 条件通过。

CPU 探针与模型的关键差别必须明确：探针使用 **固定 PCA q** 和强正则 Ridge；模型使用
**可学习 q_proj**、158,721 参数 Reader 和 22,680 次 raw-y L1 更新。CPU 通过只购买 GPU
实验，不购买“模型会受益”的结论。

## 2. GPU Stage B 配置与公平性

三臂、两 seed：

| 臂 | 结构 | 参数 | Reader 参数 |
|---|---|---:|---:|
| P3 | 旧 321 维输入（含 `sum a_i^2` 96 维） | 413,505 | 148,481 |
| MFIC-R | 361 维输入，C 由真实 `K` 计算 | 425,297 | 158,721 |
| MFIC-U | 361 维输入，C 由 content-null `K_U` 计算 | 425,297 | 158,721 |

- Fit-only masked SSL：7,560 updates；seed0 1,612.1 s，seed1 1,657.2 s；
- 同 seed 三臂共享 SSL encoder checkpoint、公共模块初始权重、R/U 全初始化、batch order、
  22,680 updates、raw-y L1、R3 配方、last-5 soup；
- 独立校验：三臂 shared hash 相等、R/U full init 相等、encoder 与 SSL checkpoint 相等、
  batch order 与独立重算相等、每臂 updates = 22,680；
- 训练在 `res` 上执行：seed0 GPU0，seed1 GPU1；所有远端操作走 `rr`；
- 没有使用 official valid/test；SSL/训练/预测期间 Monitor/Locked 标签被 `LabelGuard` 守护；
- 预测在标签读取前冻结：12 个预测文件 + 6 个 soup + 2 个 SSL checkpoint + 4 个
  participation 文件；`prediction_set_sha256 = 5fd6b1bc83c0204e3424570b1573efbd6fca9358f779de34b08064191e35baee`；
- 预测冻结时间 18:35:06 UTC；Monitor 标签读取时间 18:37:23 UTC；顺序正确。

## 3. Monitor 结果（n=1,000）

| 指标 | P3 | MFIC-R | MFIC-U |
|---|---:|---:|---:|
| MAE（seed 均值） | **0.087845** | 0.092489 | 0.094765 |
| MAE seed0 / seed1 | 0.087752 / 0.087939 | 0.092463 / 0.092516 | 0.094110 / 0.095421 |
| RMSE | **0.206378** | 0.243333 | 0.252498 |
| ordinary（y>-3） | **0.078231** | 0.081592 | 0.082714 |
| tail（y<=-3, n=58） | **0.243997** | 0.269480 | 0.290496 |
| Fit MAE（soup） | 0.001178 / 0.001016 | **0.000698 / 0.000811** | 0.000788 / 0.000888 |

主比较：

| 比较 | Δ MAE | 逐 seed | 95% CI | ordinary | tail | top-10-removed | RMSE Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| **R − P3** | **+0.004644** | +0.004711 / +0.004577 | [+0.001032, +0.008585] | +0.003361 | +0.025482 | +0.003229 | +0.036955 |
| R − U | -0.002276 | -0.001647 / -0.002905 | [-0.005456, +0.000393] | -0.001122 | -0.021017 | -0.000605 | -0.009165 |
| U − P3 | +0.006920 | +0.006358 / +0.007482 | [+0.002646, +0.011927] | +0.004483 | +0.046499 | +0.004840 | +0.046120 |

注意：R−U 的负号表示 R 比 U 好；其点估计 -0.002276 未达到预注册的 -0.003，CI 上界
+0.000393 > 0。方向性上真实 `K` 略优于 content-null，但不能据此声称“真实拓扑绑定贡献”
成立；更不能把 R 说成优于 P3。

### Monitor 购买门槛

| # | 条件 | 结果 |
|---:|---|---|
| 1 | R−P3 ≤ -0.004 | **FAIL**（+0.004644） |
| 2 | 两 seed R−P3 均 <0 | **FAIL**（均为正） |
| 3 | R−P3 配对 CI 上界 <0 | **FAIL**（+0.008585） |
| 4a | ordinary 改善 | **FAIL**（+0.003361） |
| 4b | top-10-removed 改善 | **FAIL**（+0.003229） |
| 5a | RMSE 退化 ≤ +0.005 | **FAIL**（+0.036955） |
| 5b | tail 退化 ≤ +0.005 | **FAIL**（+0.025482） |
| 6a | R−U ≤ -0.003 | **FAIL**（-0.002276） |
| 6b | 两 seed R−U 均 <0 | PASS |
| 7 | participation mean ≥0.0025 且 max ≥0.02 | PASS |

**8/10 失败 → `MFIC_NO_USEFUL_GAIN`。** `decision.json` 记录 `locked_gate=null`、
`locked_labels_read=false`；`locked_read.lock` 不存在。

## 4. 结构特征参与度（不是失败原因）

label-free K↔K_U swap：

| zone / seed | mean\|Δpred\| | max\|Δpred\| | zero-K mean |
|---|---:|---:|---:|
| Monitor s0 | 0.047480 | 0.647060 | 0.134248 |
| Monitor s1 | 0.049054 | 0.645740 | 0.131359 |
| Locked s0 | 0.048931 | 0.559042 | 0.148195 |
| Locked s1 | 0.051222 | 0.656115 | 0.148387 |

矩阵被模型实际消费，且响应远高于门槛。失败不是“tensor 没用”，而是**用错了/过拟合了**。

## 5. 机制诊断（不构成 rescue）

1. **CPU 信号没有迁移到可学习模型。** 固定 PCA-q + Ridge 的强正则估计器能从 K 的 136 维
   二次型中提取 -0.1008 的 MAE 信号；MFIC-R 的 q 是学出来的、Reader 容量更高、训练
   22,680 步后 Fit MAE 降到 0.000698（P3 0.001178），但 Monitor 更差。典型“训练集更好、
   held-out 更差”的过拟合特征。
2. **不是单纯的容量/参数问题。** MFIC-R 与 MFIC-U 参数完全相同；U 也在 Monitor 上比 P3
   差 +0.006920。R 比 U 好 -0.002276，说明真实拓扑有小的方向性作用，但不足以抵消整个
   二次型 Reader 带来的泛化代价。
3. **不是 sum(R) 泄漏或 K 未消费。** 数学门禁止 `sum(R)` 特征；参与度检查显示 K 对预测
   有大响应；模型审计显示 K 只改变 C 块，且 C 通道有非零梯度。
4. **不能从本实验推出“K 没有信息”。** 正确结论是：在 freeze 的 one-shot 架构、q 学习方式、
   L1 配方和 22,680 更新预算下，**K 的廉价信号没有变成 held-out 收益**。任何“加正则/
   降 q 维/换 Ridge/改 alpha/加 rank”的想法都是新的、未授权的 rescue，本协议禁止执行。

## 6. 预算与运行

| 项目 | 时间 |
|---|---:|
| stage-inputs | 0.38 min |
| smoke（首次，stop） | 0.25 min |
| smoke-long | 0.61 min |
| SSL seed0 / seed1 | 26.87 / 27.62 min |
| 训练 P3 s0/s1 | 8.96 / 10.35 min |
| 训练 MFIC-R s0/s1 | 11.85 / 13.09 min |
| 训练 MFIC-U s0/s1 | 11.96 / 13.27 min |
| 预测 s0/s1 | 0.03 / 0.03 min |
| 失败重试（D1 + hashcheck） | 0.08 / 0.12 min |
| **合计** | **2.091 GPU-h / 3.0** |

所有远端运行通过 `rr`；`res` GPU0 跑 seed0、GPU1 跑 seed1。首次 smoke 的预算 stop 与随后
的 `smoke-long` 重新估计已完整记录在 `DEVIATIONS.md`；正式训练没有被缩短。

## 7. 纪律与外部声明

- Split V 是同一 official-train 10k 暴露池的重新抽样，**不是独立盲测**；与 U Fit 重叠
  6,421 行（80.3%）。不得把本结果称为独立 test。
- official valid/test 从未加载；`official_valid_loaded=false`、`official_test_loaded=false`。
- Monitor 标签只读一次（`monitor_read.lock`）；Locked 标签未读；没有 full10k、没有
  official test、没有 `test 0.06` 声称。
- 不引用 Batatia et al. ICLR 2024 的 ZINC 0.063 作为本模型结果；该论文模型含 GCN 层，
  本模型是非迭代、非 Attention、无 GNN 的解析矩阵函数组合，二者不可直接对比。
- 没有 alpha / PCA 维数 / kernel rank / Ridge alpha / Reader 宽度 / seed 的 rescue 搜索。

## 8. 证据文件

- CPU：`REPORT_CPU.md`、`cpu_signal.json`、`cpu_folds_V.json`、`cpu_gate.json`、
  `decision_cpu.json`、`math_gates.json`、`model_audit.json`。
- GPU：`smoke_V.json`、`smoke_long_V.json`、`staged_inputs_V.json`、
  `ssl_V_s{0,1}.json`、`train_*_V_s{0,1}.json`、`train_summary_V_s{0,1}.json`、
  `prediction_manifest_V.json`、`monitor_score_V.json`、`monitor_gate_V.json`、
  `decision.json`、`monitor_read.lock`、`gpu_budget_ledger.json`、
  `evidence_manifest_gpu.json`。
- 预测与 soup 共 12 + 6 个文件；完整哈希见 `prediction_manifest_V.json` 和
  `evidence_manifest_gpu.json`。
- `DEVIATIONS.md` 记录 D1 代码哈希比较 bug、D2 smoke-long 预算重估、D3 重试计入。

## 9. 下一步

按预注册协议，**没有自动下一步**。`MFIC_NO_USEFUL_GAIN` 关闭本轮路线；不执行 full10k、
不读 official test、不做 alpha/PCA/rank/Reader rescue。若未来要研究“为什么 CPU 的 K 信号
不能迁移到可学习 q”，需要新的预注册和用户明确授权；不能把本轮的负结果改写成机制成功。
