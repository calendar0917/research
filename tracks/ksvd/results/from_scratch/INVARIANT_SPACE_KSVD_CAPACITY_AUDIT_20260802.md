# Invariant-space KSVD capacity 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_INVARIANT_SPACE_CAPACITY_PROTOCOL_20260802.md`  
> 判定：`ADOPT_INVARIANT_KSVD_CAPACITY`

| cell | dict scalars | code scalars/graph | TRUE RMSE | vs raw invariant | vs BAG | vs SHUFFLED | cosine | cell decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K16_T3 | 352 | 52.96 | 0.09945 | -0.2027 | 0.2963 | 0.1314 | 0.9673 | ADOPT_INVARIANT_SPACE_KSVD_TOKEN |
| K16_T4 | 352 | 70.61 | 0.10644 | -0.1467 | 0.3555 | 0.1338 | 0.9722 | ADOPT_INVARIANT_SPACE_KSVD_TOKEN |
| K24_T3 | 528 | 52.96 | 0.13625 | 0.0923 | 0.3053 | 0.1231 | 0.9506 | INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY |
| K24_T4 | 528 | 70.61 | 0.14758 | 0.1832 | 0.3216 | 0.1436 | 0.9628 | INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY |
| K32_T3 | 704 | 52.96 | 0.16162 | 0.2957 | 0.2996 | 0.1307 | 0.9462 | INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY |
| K32_T4 | 704 | 70.61 | 0.15776 | 0.2648 | 0.2981 | 0.1347 | 0.9534 | INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY |

Passing cells：`['K16_T3', 'K16_T4']`。
Selected cell：`K16_T3`。
