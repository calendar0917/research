# DECISION — `zinc_error_mechanism_audit_v1`

- **日期**: 2026-10-27
- **范围**: CPU-only error-mechanism audit; official test never loaded; no training.
- **证据**: `REPORT.md`, `analysis/decomposition.json`, `analysis/audit.json`.

## 1. 停止环特征搜索

**STOP**：不再为 ZINC 强模型提出 exact longest simple cycle、minimum cycle basis
max/total/mean、mean max-basis 或任何新的全局环尺度。

理由（按权重）：

1. `global33` 已含池化的环原子 minimum-ring 直方图
   (`ring_size3_frac`…`ring_size8plus_frac`)；每原子 one-hot 在 `atom_feat`，不在
   `global33`。所谓“缺每原子最小环特征”在表示层不成立。
2. 精确最长/MCB 尺度基本可由 `global33` 重建（valid R² 0.943/0.892/0.999；
   `mean_maxbasis` 0.747），大部分可由现有描述重建；MCB total 几乎是 `global33` 的确定性函数（R² 0.999）。
3. 置换不变的 broad long-ring 组（exact longest ≥7）占 50.8% AE，但 per-row MAE
   只比非 long 高 1.6×；它主要是组大，不是“长环难”。与 penalty 对齐的窄代理
   （RDKit/MCB >6）31 行里 31 行就是 penalty，不能作为独立结构证据。
4. 严格 train-internal OOF（compact-v4-hinge，**不是** DCC-L）显示：加这些环尺度
   的全部表面增益都来自 372 个 penalty 行；在 9,628 个无惩罚行上
   ΔR² = +0.00006、ΔSpearman = −0.0008，跨折符号翻转。没有稳定的条件预测信息。
5. 环惩罚本身只占 18.7% 总 AE；其中 cycle-attributable 0.00944，单行 valid:0172
   占 0.00509，去掉后只剩 0.00434；4/35 行是 basis-order sensitive（label 是众数）。
   为一个 0.004–0.009 上限、且大部分不可迁移的切片再买特征，不符合证据门槛。

**NOT**: 不把本轮的负结果改写成“换一种环/长环/层次环/环组合特征再试”；不提出新的
结构算子；不因为预算未花完而训练。

## 2. 下一次正式 GPU 实验

**唯一候选**：`zinc_dccl_bottleneck_v1` §E 已记录的 **10k→12k 标签数量斜率点**。
它检验的假设是：“4k→8k 的 −0.038 斜率是局部小数据效应，还是会在 10k 之外继续？”
这区分“剩余误差 = 共享函数估计方差（标签数受限）”与“该家族在官方协议下已到结构天花板”。

- 冻结 DCC-L `B-R0`，forward 不变，同 47,400 updates、同 SSL/R0、Fit-only 标准化；
- 10k（官方 train）vs 12k（+2k 同分布 ZINC，逐分子 WL + 精确图同构排除 valid/test 同构）；
- 2 条轨迹，~0.75 A100·h；official test 不读；
- 预注册判定：`Δlocked(12k−10k) ≤ −0.010` → 标签数量问题；`≥ −0.004` → 4k→8k
  斜率是局部的，家族以 0.0663 收尾；两者之间 → 只补 11k 点。

**约束**：外部标签会改写 benchmark 口径，本 track 历史上禁止。若这条不可接受，正确
动作是**停止**，不买任何新 GPU 实验；不要用一个新的环/结构方案替代。

## 3. 不做什么

- 不读 official test；不重训 DCC-L/BOSI/ZBO；不改
  `zinc_block_ownership_v1/v2`、`zinc_generalization_audit_v1`、`results/phase1e`、
  `zinc_dccl_bottleneck_v1` 的历史记录。
- 不用 official valid 的事后相关性声称“加环特征可泛化”；valid 是已暴露开发集。
- 不把“标签数量斜率点”包装成新架构方向；它是数据/估计器侧诊断。
