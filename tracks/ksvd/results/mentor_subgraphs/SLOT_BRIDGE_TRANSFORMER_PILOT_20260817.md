# 槽位—共享节点桥 Transformer 首轮结果

> 协议：`tracks/ksvd/docs/KSVD_MENTOR_SLOT_BRIDGE_TRANSFORMER_PROTOCOL_20260817.md`  
> 门槛：`True`

| 分支 | RMSE | F1 |
|---|---:|---:|
| NO_BRIDGE | 0.396059 | 0.783691 |
| TRUE_BRIDGE | 0.348479 | 0.848205 |
| SHUFFLED_BRIDGE | 0.395022 | 0.788697 |

- 正确桥相对无桥改善：`12.0135%`，赢 `3/3` 折；
- 正确桥相对打乱桥改善：`11.7826%`，赢 `3/3` 折；
- 目标是完整 8 节点局部块邻接，而不是统计描述。
- 本轮不使用图标签、KSVD、GIN/GINE 或源节点编号嵌入。
