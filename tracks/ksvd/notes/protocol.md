# protocol_id 登记（只增不改语义）

| protocol_id | 用途 | 指标 | 划分 / 汇报 |
|-------------|------|------|-------------|
| `stage0-sample-v0` | 采样过程评估 | edge_cover, edge_repeat, mean_\|S\|, hit_cap_rate, … | 合成/单图；多种子 mean±std |
| `stage1-ksvd-smoke-v0` | 向量化+KSVD 烟测 | + recon_rel, atoms_used | 同上；KSVD 超参进 config |
| `ogb-molhiv-v0` | 主下游对标 | ROC-AUC | OGB scaffold；test @ best val；≥10 seeds（烟测 ≥3） |
| `molhiv-struct-probe-v0` | 结构-only / 池化消融探针 | ROC-AUC（LR on \(s_G\)） | 同 scaffold；train 上 fit D+LR，val 选 pool，test 汇报；**非端到端 GNN** |
| `graph-level-solid-v0` | 合成图级闭环 | Acc / 过程指标 | C4 等；见 `GRAPH_LEVEL_SOLID_SUMMARY` |
| `pool-ablation-v0` | mean/max/attn 池化 | Acc | 合成 C4；无 molhiv 依赖 |
| `tud-paper-cite` | **仅引用** CIN/GIN 文 | Acc | Xu 10-fold max val；**禁止自跑混入主表当 strict** |

新增协议：复制行改 id，旧 id 永不改定义。

## `ogb-molhiv-v0` 细则（锁）

| 项 | 值 |
|----|-----|
| 数据 | `ogbg-molhiv`（OGB） |
| 划分 | scaffold split（官方） |
| 指标 | ROC-AUC |
| 选模 | epoch/超参见 val AUC 最大；**test 只报一次 @ best val** |
| 种子 | 正式 ≥10；探针 ≥1–3 |
| 结构通道 | CoverageRW → 共享 KSVD → readout；D **仅 train** |
| 融合 | 双通道属性 ‖ 结构；禁止 atom 拼进字典 |
| 结果路径 | `results/molhiv/` |
