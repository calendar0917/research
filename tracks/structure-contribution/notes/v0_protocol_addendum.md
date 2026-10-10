# cscl-v0 协议附录:解释对象 = best-epoch 检查点(非 soup)

日期: 2026-10-10(synthetic 阶段发现,先于任何正式 GPU 结果)
状态: 对 `v0_protocol.md` §3/§6 的**澄清**,不是超参或判据变更。
      主性能指标保持 soup dev MAE 不变。

## 发现

CSCL-v0 的归因依赖类型 embedding 的几何:`α(t)=wαᵀE_u[t]`,`γ` 由
`[E_ru[tlo];E_ru[thi];r]` 经 MLP 读出。Top-5 monitor checkpoints 的**权重
平均(soup)会把不同 epoch 的 embedding 表逐元素平均**;不同 epoch 的
embedding 之间存在旋转/重排自由度,平均后几何错位——

- 合成真值任务上,soup 模型的 α-θ spearman 从 **0.85 塌缩到 ≈0**,
  真交互对 vs 陷阱对的 |γ| 区分(0.091 vs 0.034,θ_pair-γ spearman 0.98)
  **完全消失**(0.0704 vs 0.0702);
- 同时 soup dev MAE(0.283)略好于单模型(0.31):soup 只有益于性能,
  有害于解释。

## 决定(冻结)

1. **主性能指标 = soup dev MAE**(v0_protocol §3 不变);副表报
   best-epoch dev MAE。
2. **解释/归因分析对象 = best-epoch 检查点**(monitor 最优单模型)。
   贡献导出、合成真值诊断、seed 稳定性检验均在其上进行。
3. 该选择的含义(如实记录):归因结果属于一个具体训练实例;跨 seed
   稳定性检验(§6.2)就是为它设计的,若 best-epoch 模型的归因本身不稳,
   会在该检验中暴露。

## 证据

- `results/cscl_v0_synthetic/`(soup 路径的诊断,spearman≈0)
- 本附录的诊断记录:单模型 α-θ spearman 0.848(λ=0.01)/0.488(λ=0),
  real|γ| 0.0906 vs trap 0.0342,θ_pair-γ spearman 0.983
- 机制:embedding 跨 epoch 旋转自由度 + 逐元素平均不可对齐
