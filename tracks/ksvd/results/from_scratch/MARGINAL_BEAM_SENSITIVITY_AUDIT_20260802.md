# Marginal candidate beam：cost–coverage sensitivity 审计

> 日期：2026-08-02  
> s10/o3/m1.5；72 graphs；wall-clock sampling cost。

## 1. 判定

**SMALL_BEAM_SUFFICIENT**

Selected knee：`B8_R1`。

## 2. Cells

| cell | sec/graph | edge cover | pair cover | RAW full RMSE | edge gain | RMSE reduction | effect gate | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TARGET | 0.0253 | 0.7650 | 0.5605 | 0.3054 | - | - | - | True |
| B8_R1 | 0.2794 | 0.8734 | 0.5288 | 0.2205 | 0.1083 | 0.2781 | True | True |
| B16_R1 | 0.5050 | 0.8803 | 0.5298 | 0.2139 | 0.1152 | 0.2996 | True | True |
| B32_R1 | 0.9541 | 0.8844 | 0.5307 | 0.2102 | 0.1193 | 0.3118 | True | True |
| B32_R2 | 1.8486 | 0.8845 | 0.5309 | 0.2101 | 0.1195 | 0.3118 | True | True |

Pareto cells：`['TARGET', 'B8_R1', 'B16_R1', 'B32_R1', 'B32_R2']`。

## 3. 边界

- wall-clock 只在当前机器和实现内比较，不外推绝对部署延迟。
- sensitivity 只回答 beam search effort；不重新扫描 patch size、overlap 或 KSVD capacity。
