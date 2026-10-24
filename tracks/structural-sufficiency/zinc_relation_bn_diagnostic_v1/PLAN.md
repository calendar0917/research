# `zinc_relation_bn_diagnostic_v1` — fixed checkpoint 的 BN 训练/推理状态诊断

## 问题

上一轮 `zinc_relation_composition_probe_v1`（source commit `06cd7fd`，训练提交
`7464c0d30441`）的两个主体在 eval 模式下 train MAE（D_best 0.0236 / S_best 0.0535）远高于
它们在训练时的在线 train MAE（~0.0087/0.0091），同时 valid 停在 0.082–0.099。
本轮只回答一件事：**eval 模式的误差里有多少来自 BN 推理统计（running_mean/var）与固定权重不匹配，
用 train-only 重新估计统计量能否恢复 train/valid 表现。**

## 四份固定权重与四种状态

D_best(ep566)、D_final(ep600)、S_best(ep557)、S_final(ep600)，全部 supervised seed0，
来自 `zinc_relation_composition_probe_v1/runs/RC_{D,S}_s0/{best,final}.pt`（按 manifest 定位，
不凭文件名猜内容；file sha256 / state sha256 / param hash / BN buffer hash 全部登记）。

| 状态 | BN 统计 | dropout | 用途 |
|---|---|---|---|
| O | 原 running buffers（eval） | off | 重放源结果（batch 256） |
| B | 当前 batch 统计（`F.batch_norm(x,None,None,w,b,training=True)`，不写 buffer） | off | train-only 诊断 |
| T | 同 B | 原 attention dropout 0.2 | train-only，**门控**：仅当 B 解释不到 (online−O) 差距的 50% 时跑一次 |
| C | train-only **逐层**重估的 running buffers | off | 唯一校准方案，之后统一 eval |

隔离规则：不建优化器、不 backward、不改任何可学习参数；只在派生副本里改 BN 的
`running_mean/running_var`；源 checkpoint / 源报告不改写；**official test 不构造、不加载、不评分**；
valid 只用于最后一次评估（不参与任何统计量、顺序、批次、checkpoint 或方法选择）。

## C 的估计过程（唯一方案，无扫描）

1. 一次完整 train 前向用 hooks 记录真实 BN 调用顺序与入口计数（预期 11 个 BN：前 3 层
   `bn_e→bn_h1→bn_h2`，第 4 层 `bn_h1→bn_h2`），并核对计数 = 真实 bond token / pair 数
   （padding 不进统计）。
2. 按调用顺序逐个 BN：把前序 BN 换成已校准统计、其余保持 O；给目标 BN 挂 `forward_pre_hook`，
   用 **FP64 Chan/Welford 流式合并**（按真实条目数加权）收集 mean 与中心二阶矩；写入
   `running_mean=mean`、`running_var=M2/(n−1)`（PyTorch 无偏口径），cast 回原 dtype；
   `num_batches_tracked` 不动。
3. 全部完成后，再用完整 C 在 train 上过一遍，重新测量每个 BN 的真实全局矩并与写入值对比
   （自洽检查），再做统一 eval 评估 train/valid。

判据：如果某个 BN 的入口分布依赖更晚的 BN，自洽检查会报错；这是"逐层足够"的验证方式。

## 判读规则（先写死，避免事后挑赢家）

按 §6 的 A–E 五种情形报告：B 是否恢复 train、C 是否同时改变 train/valid。
恢复比例一律用同范围、同 checkpoint 的固定权重数字计算，不用在线 MAE 混算。
