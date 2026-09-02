# ZINC centered conditional residual + SVD16

日期：2026-08-30  
协议：`luyin16-zinc-centered-conditional-residual-svd16-screen-v1`  
状态：small-slice no-go；未加载 test

## 问题

conditional joint v2 已证明 structure--attribute binding 存在，但约 2K 维条件
直方图未稳定超过 typed marginal。本轮只提取超出两侧 marginals 的绑定残差：

```text
R = P(signature, attribute) - P(signature)P(attribute)
```

对训练残差减去 train mean，再使用 train-only randomized TruncatedSVD 压到固定
16 维。true 和每个图内 attribute-shuffle control 各自拟合同预算、train-only 的
SVD，随后拼接到稳定 baseline `global+typed raw`。固定 XGBoost 和 model seeds
`0/1/2`；不做维度扫描、Optuna 或 test 评估。

严格晋级门槛预先固定为：两个非重叠 `2000/200` 切片均满足：

1. residual SVD 相对 typed raw 改善 `>=0.01 MAE`；
2. true 相对三次 shuffle 的平均间隔 `>=0.01 MAE`。

任一条件失败即停止，不运行 `5000/500` 或 full。

## 结果

| slice | global+raw | + true residual SVD16 | true 对 raw 增益 | mean true-shuffle gap | SVD explained variance |
|---|---:|---:|---:|---:|---:|
| A：train 0:2000 / valid 0:200 | 0.73813 | 0.71246 | +0.02567 | +0.04408 | 0.70635 |
| B：train 5000:7000 / valid 500:700 | 0.56485 | 0.56640 | -0.00155 | +0.02089 | 0.71122 |

三次 true-shuffle 间隔：

- A：`0.05933 / 0.03029 / 0.04262`；
- B：`0.02501 / 0.01868 / 0.01900`。

SVD16 将原始 `1856/1984` 维残差压至 16 维，并保留约 71% 方差。因而 B 的
失败不能简单解释为压缩完全破坏残差信号。

## 判定

### binding detection：PASS

两个切片的 true residual 都稳定优于所有 shuffled residual。结合 joint v1、
conditional joint v2，本阶段已经用三种统计方式重复确认：局部结构与 atom/bond
属性并非独立，其对应关系可被检测和低秩压缩。

### stable target increment：NO-GO

slice B 没有超过 typed raw，严格 gate 失败；因此不扩大数据、不扫 SVD 维度和
XGBoost 参数，也不接 K-SVD。slice A 的 residual 模型跨 seed 标准差为 `0.0224`，
进一步说明单切片正增益不应升级为主结论。

## 路线结论

当前矛盾已收敛为：

```text
structure--attribute dependence exists
        ≠
dependence supplies stable incremental information for the ZINC target
```

typed marginal 是更稳定的预测表示；unsupervised binding residual 更适合作为
诊断量，而不是当前下游主特征。普通 K-SVD 即使能进一步压缩或重建这些残差，
也没有证据表明会自动产生任务对齐，因此此处不投入 K/T 或稀疏度搜索。

ZINC 的后续重点应回到已有正证据更强的方向：radius-3 提升所指向的采样覆盖和
跨 patch 长程关系。下一轮应改变“局部对象之间如何连接”，而不是继续增加局部
属性联合统计。

原始结果：

- `ZINC_CONDITIONAL_RESIDUAL_SVD16_SLICE_A_20260830.json`
- `ZINC_CONDITIONAL_RESIDUAL_SVD16_SLICE_B_20260830.json`
