# protocol_id 登记（只增不改语义）

| protocol_id | 用途 | 指标 | 划分 / 汇报 |
|-------------|------|------|-------------|
| `stage0-sample-v0` | 采样过程评估 | edge_cover, edge_repeat, mean_\|S\|, hit_cap_rate, … | 合成/单图；多种子 mean±std |
| `stage1-ksvd-smoke-v0` | 向量化+KSVD 烟测 | + recon_rel, atoms_used | 同上；KSVD 超参进 config |
| `ogb-molhiv-v0` | 主下游对标 | ROC-AUC | OGB scaffold；test @ best val；≥10 seeds |
| `tud-paper-cite` | **仅引用** CIN/GIN 文 | Acc | Xu 10-fold max val；**禁止自跑混入主表当 strict** |

新增协议：复制行改 id，旧 id 永不改定义。
