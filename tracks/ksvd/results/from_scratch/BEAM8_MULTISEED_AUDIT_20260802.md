# Beam8 continuous cover：3-seed robustness 审计

> 日期：2026-08-02  
> s10/o3/m1.5；72 fixed graphs；RAW comparison。

## 1. 判定

**PASS_BEAM8_THREE_SEED_ROBUSTNESS**

| seed | target edge | Beam8 edge | edge gain | target RMSE | Beam8 RMSE | RMSE reduction | pair gain | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 900101 | 0.7650 | 0.8734 | 0.1083 | 0.3054 | 0.2205 | 0.2781 | -0.0317 | True |
| 900102 | 0.7660 | 0.8745 | 0.1085 | 0.3045 | 0.2192 | 0.2803 | -0.0317 | True |
| 900103 | 0.7650 | 0.8741 | 0.1092 | 0.3055 | 0.2195 | 0.2817 | -0.0327 | True |

Passed seeds：`3/3`；invariants：`True`。
