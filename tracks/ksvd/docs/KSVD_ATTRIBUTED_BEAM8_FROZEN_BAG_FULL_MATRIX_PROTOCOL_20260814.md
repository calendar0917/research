# Attributed Beam8 graph-conditioned frozen BAG full dual-axis matrix

> 日期：2026-08-14  
> 状态：split3/4 × model0/1/2 压力网格通过后冻结；split1/2 × model1/2 结果不可见。

## 1. 固定矩阵

- split seeds：0/1/2/3/4；
- model seeds：0/1/2；
- 每个 split×model cell 3 outer folds；
- 总计 15 cells、45 fold units；
- 复用所有已冻结结果，只新增 split1/2 × model1/2 四个 cells；
- 模型、Beam8 INIT dictionary、BAG construction、strict inner checkpoint 与 frozen residual
  配置完全沿用上一协议。

## 2. Final gate

完整 45-fold 矩阵要求全部满足：

1. BAG−GINE mean `≥+0.5pt`；
2. 至少 30/45 folds 为正；
3. 至少 4/5 split-seed means 为正；
4. 至少 2/3 model-seed means 为正；
5. 最差 split-seed mean 不低于 `−0.5pt`；
6. 最差 model-seed mean 不低于 `−0.5pt`；
7. 至少 10/15 split×model cell means 为正。

通过后，将 graph-conditioned frozen BAG residual 定为 Beam8 在 Mutagenicity 上唯一稳定的
分类接入机制；仍不恢复 localized binding、atom gate、cross-attention 或普通 KSVD updates。
失败则停止 Beam8 分类扩展。
