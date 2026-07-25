# 共享字典：是一开始就定的吗？

**不是。** 时间线：

| 阶段 | 字典策略 | 原因 |
|------|----------|------|
| 初版 `definition.md` / 烟测 | **每图独立** \(D_g\) | 实现简单；luyin3 也讨论过每图系数 |
| followup / 首次下游 | 仍每图 \(D\) | MUTAG/合成上结构≈噪声 |
| **fixes 实验** | 改为 **共享 \(D\)**（仅 train fold 学习） | 诊断发现：patch-pool 已有信号，per-graph KSVD 丢了可比性 |

结论写在 `results/FIXES_SUMMARY.md`：合成三角任务 per-graph ~0.55 → shared ~0.86。

**当前默认推荐**：跨图 **共享字典**（CV 时 train-only），写进协议 `shared_dict_train_only`。  
每图字典仅作消融/对照，不作主方法。
