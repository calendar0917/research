# ZINC radius-2 / radius-3 multiscale 快筛

日期：2026-08-30  
协议：`luyin16-zinc-r2-r3-multiscale-screen-v1`  
状态：weak complementarity；不晋级 full；未加载 test

## 问题与协议

在 fixed XGBoost 下比较：

```text
global + typed radius-2 raw       (max_nodes=12)
global + typed radius-3 raw       (max_nodes=20)
global + radius-2 raw + radius-3 raw
```

radius-2/3 均使用全原子中心。cap12 下 r2 几乎无截断，cap20 下 r3 也近乎无截断，
因此比较主要反映感受野而不是 patch cap。固定 model seeds `0/1/2`，不做 Optuna、
K-SVD 或 test 评估。

小切片 gate 预设为：两个非重叠 `2000/200` 切片中，multiscale 均比单独 r3
改善 `>=0.01 MAE`。通过后只扩大到两个 `5000/500` 终止复核。

## 结果

| 阶段 | slice | global+r2 | global+r3 | global+r2+r3 | multi 对 r3 增益 | multi 对最佳单尺度增益 |
|---|---|---:|---:|---:|---:|---:|
| 2000/200 | A | 0.73813 | 0.75960 | 0.72747 | +0.03213 | +0.01066 |
| 2000/200 | B | 0.56485 | 0.57043 | 0.55936 | +0.01107 | +0.00549 |
| 5000/500 | A | 0.67419 | 0.66777 | 0.65276 | +0.01501 | +0.01501 |
| 5000/500 | B | 0.55329 | 0.53190 | 0.52719 | +0.00471 | +0.00471 |

multiscale 在四个切片均为最优，并且两个大切片的每个 model seed 都优于 r3。
但四片中只有两片相对最佳单尺度改善达到 `0.01`；大切片 B 仅改善 `0.00471`，
未通过稳定晋级门槛。

覆盖审计：

- r2 平均 pair coverage 约 `0.494--0.500`；
- r3 平均 pair coverage 约 `0.695--0.702`；
- r2/r3 truncation 均接近 0。

## 判定

### radius-3：保留为大样本主单尺度

在 2000 图训练量下，r3 的高维 readout 不如 r2；到 5000 图时，r3 在两片均
超过 r2，增益为 `0.00643/0.02139`。这说明更大 radius 提供了有价值但更吃样本
的对象上下文，与 full factorial 的 radius-3 正增益一致。

### r2+r3：弱互补，不晋级 full

四片方向一致说明 r2 与 r3 并非完全冗余；但增益大小未跨切片达到预设 `0.01`。
因此不使用当前 valid 结果继续做 10K/1K、Optuna 或 test。multiscale 只登记为
future candidate，不升级为主结论。

## 本轮采样结论

1. 全中心采样保留；
2. ZINC 大样本 baseline 使用 radius-3 typed raw，并保留 cap20 避免截断；
3. radius-2+3 存在弱互补，但证据不足以增加主 pipeline 复杂度；
4. 简单 long-distance pair relation 已 no-go，因此 radius-3 增益主要解释为
   局部对象上下文扩大，而非 pair histogram；
5. 暂不继续 radius-4、随机游走、多尺度权重或 fusion 调参。

原始结果：

- `ZINC_MULTISCALE_RADIUS_SLICE_A_20260830.json`
- `ZINC_MULTISCALE_RADIUS_SLICE_B_20260830.json`
- `ZINC_MULTISCALE_RADIUS_5000_SLICE_A_20260830.json`
- `ZINC_MULTISCALE_RADIUS_5000_SLICE_B_20260830.json`
