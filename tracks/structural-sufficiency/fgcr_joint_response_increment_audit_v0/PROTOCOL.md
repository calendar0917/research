# FGCR 联合结构响应增量验证 — 冻结协议

Round id: `fgcr-joint-response-increment-audit-v0`

Directory: `tracks/structural-sufficiency/fgcr_joint_response_increment_audit_v0/`

本文件在运行前冻结。只做冻结 checkpoint 的重放、精确响应分解、局部参考替换和
描述性统计；不训练、不微调、不调超参数、不评分 Test。

---

## 0. 问题

历史 FGCR 模型（`full_graph_cavity_response_pilot_v0`，best-Valid epoch 96，
Valid RMSE 0.7128937193）实际使用的是全局联合响应中的独特结构信息，还是主要
使用了可由局部计算获得的响应？

把历史完整响应拆成：

```
H_FULL      = logZ - sum_i logsumexp(u_i)
H_IND       = sum_{(i,j) in E} logsumexp_{s,t}(log p_i(s) + log p_j(t) + psi_ij(s,t))
H_INCREMENT = H_FULL - H_IND

ETA_FULL_i      = posterior_logodds_i - unary_logodds_i
ETA_STAR_i      = sum_{j in N(i)} [m_{j->i}(1) - m_{j->i}(0)]
                  m_{j->i}(s) = logsumexp_t(log p_j(t) + psi_ij(s,t))
ETA_INCREMENT_i = ETA_FULL_i - ETA_STAR_i
```

其中 `log p_i = log_softmax(u_i)`。`ETA_STAR` 使用独立邻居边缘化，是局部星形
因子模型的精确根节点响应，**使用邻居信息，但不是全图联合推断**。

## 1. 资产与数据

只读复用：

```
full_graph_cavity_response_pilot_v0/code/fgcr_model.py
full_graph_cavity_response_pilot_v0/code/fgcr_inference.py
full_graph_cavity_response_pilot_v0/results/seed0/best.pt
tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz
```

数据：官方 Lipo Train3360 / Valid420。Test 的 row id、特征、标签、预测和指标
均不访问、不计算。junction-tree plan 只在 Train+Valid 的 3780 张图上构建。

若 checkpoint 或缓存缺失/哈希不符：立即写 `checks.json` 的
`status=MISSING_ARTIFACT`，不重训。若墙钟超过 1200 s：写
`status=RESOURCE_LIMIT`，不增加训练任务。

## 2. 重放

加载 best checkpoint，`model.eval()`，`torch.inference_mode()`，batch 大小
256（历史 eval 设置）。完整 Valid/Train forward。要求：

```
|Valid RMSE - 0.7128937193209034| <= 1e-5
```

并对照 `results/seed0/predictions_best.npz` 的 Train/Valid 预测做最大绝对差检查。

## 3. 四个冻结替换条件

共享同一 checkpoint、同一原子/边势、sentence MLP、LayerNorm、readout、标量
bias；不重新拟合任何归一化参数。

| 条件 | H 输入 | eta 输入 |
|---|---|---|
| FULL | H_FULL | ETA_FULL |
| LOCAL_H | H_IND | ETA_FULL |
| LOCAL_ETA | H_FULL | ETA_STAR |
| LOCAL_BOTH | H_IND | ETA_STAR |

实现方式：仅替换 `fgcr_model.forward` 内部从一次精确推断得到的响应张量，
`forward` 的其余代码、head 和参数完全不动。`eta_norm` 原样使用。

报告每个条件在 Train/Valid 的 RMSE、MAE、相对 FULL 的逐分子预测变化 RMS、
配对平方误差差、2000 次配对分子 bootstrap 区间，以及去掉双方共同平方误差最大
5% 后的 RMSE 敏感性。

## 4. 必要核验

1. 单边图 FULL = 局部参考；2. 多个不相连单边图同上；3. 零边势时 H、eta 及
增量全为零；4. 链、星、环、多组分、孤立原子对穷举枚举；5. 原子重编号、边方向
反转、边存储顺序不变；6. 每条物理边只计一次；7. `FULL = LOCAL + INCREMENT`
数值重建；8. 历史原始 forward 预测复现。

小图检查用 float64 穷举，容差 1e-9；真实数据重建与替换目标用 1e-5 容差。

## 5. 统计与判读

诊断（Train/Valid 分别）：

- H/ETA 的 FULL、LOCAL、INCREMENT 通道均值、标准差、RMS；
- 增量相对完整响应的 RMS 尺度；
- `eta_norm` 前后的响应变化与增量尺度；
- 分子间增量的分位数、近零比例、常数偏移 vs 跨分子变化；
- 增量与原子数、键数、局部响应的 Pearson/Spearman 相关；
- 预测变化与增量大小的相关。

允许类别：

- `JOINT_DEPENDENCE_SIGNAL`：去除联合增量后 Valid 明显退化，且不是单个困难
  分子造成；
- `LOCAL_SUFFICIENT_FOR_FROZEN_MODEL`：局部参考基本保持性能；
- `LOCAL_REPLACEMENT_IMPROVES`：冻结替换降低官方预测误差；
- `INCONCLUSIVE`：明显的输入分布变化、资产/实现/预算问题。

不把冻结替换的性能变化解释为重新训练后的模型性能；不把较小增量尺度直接等同于
不重要；不把较大预测变化直接等同于泛化收益。若证据混合，保留矛盾并以主要
通道/敏感性结果说明。

## 6. 停止

输出 `REPORT.md`、`checks.json`、`main_table.csv`、
`increment_statistics.json`、逐分子预测对照、`provenance.json` 后立即停止。
不启动新训练，不设计下一代架构。
