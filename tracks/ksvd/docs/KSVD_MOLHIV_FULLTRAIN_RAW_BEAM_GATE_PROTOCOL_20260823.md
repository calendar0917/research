# MolHIV 全量强基线上的 RAW Beam8 条件增量 gate

> 日期：2026-08-23  
> 状态：执行前冻结  
> 数据边界：仅 `ogbg-molhiv` official-train 的内部 Bemis--Murcko scaffold folds；official-valid/test 不编码、不训练、不评估。

## 1. 要回答的唯一问题

在一个足够强的全量分子分类骨干中，**未压缩的** Beam8 局部 patch 内容是否还带来依赖真实
patch--bond 对齐的分类信息？

这是压缩之前的必要门槛。若 RAW branch 不成立，K-SVD 不可能通过压缩创造这部分信息；若 RAW
成立而 K-SVD 不成立，断点才可归因于压缩目标。

此前结果不能替代本 gate：

- `KSVD_MOLHIV_BEAM8_NODE_ENDPOINT_ROUTE_20260815.md` 只在 8k 开发子集上测试 GINE
  endpoint 注入；
- `KSVD_MOLHIV_CIN_BEAM8_FULLTRAIN_PILOT_20260816.md` 在全量数据上测试的是**训练期**
  Beam auxiliary，而不是 inference-time patch branch；
- 既有 task-adapted K-SVD 实验以 dictionary-primary MIL 为主，不能回答强 CIN 条件下的
  RAW patch 上限。

## 2. 冻结模型与数据

- 骨干：现有 `CIN-small-compatible`，2 层、hidden 48、dropout 0.5、150 epochs；
- 覆盖：typed canonical Beam8，`s=8/o=2`、retained beam 8、`EDGE100`；
- 注入：第一层 CIN 后，patch 内部 typed-bond / canonical-pair / endpoint-slot / position / completion
  role 组成的 patch state 回写对应 bond cell；
- 不使用 K-SVD、dictionary、额外预训练、官方 valid/test 或结果后调参；
- 第一单元：full official-train internal scaffold fold 1、model seed 0。这与已完成
  full-train CIN auxiliary pilot 使用同一 fold，便于区分“数据规模”与“机制”的影响。

## 3. matched matrix

| variant | 保留 | 破坏 | 作用 |
|---|---|---|---|
| `cin` | CIN | 全部 Beam 输入 | 强基线 |
| `cin_beam8_bond` | patch 内容、endpoint binding、bond role | 无 | RAW TRUE candidate |
| `cin_beam8_bond_shuffled` | patch 内容、每条 bond 的 role 和边际 | patch→bond binding | 对齐是否必要 |
| `cin_beam8_bond_bag` | 每图完整 patch-content multiset | 所有局部 binding | 是否只是图级 patch 统计 |
| `cin_beam8_bond_no_patch` | endpoint role/容量 | patch content | patch 内容是否必要 |

`BAG` 通过每图 patch state 的均值广播到该图全部 bond cells 实现；它与 TRUE 共用所有
可训练参数、初始化、数据顺序和 patch multiset，但不访问 endpoint incidence、canonical
pair/slot 或 completion role。因此它不保留任何 patch-to-bond 绑定或由覆盖频次产生的局部
定位线索。

## 4. 决策门槛

主指标为 held-out scaffold ROC-AUC；AP 作为方向一致性诊断。

阶段 A（fold 1, seed 0）只有全部满足才扩展：

1. `TRUE - CIN >= +0.005` AUC；
2. `TRUE - SHUFFLED >= +0.003` AUC；
3. `TRUE - BAG >= +0.003` AUC；
4. AP 不相对 CIN、SHUFFLED 和 BAG 同时为负；
5. TRUE branch 的 `beam_bond_fusion_norm > 0`，排除分支未训练。

阶段 B：仅在 A 通过后，固定所有超参并扩展至 folds 0/2、seed 0。要求三个 fold 的平均
`TRUE-CIN >= +0.005`，至少 2/3 folds 为正，且 TRUE 同时超过 shuffled 与 BAG。

阶段 C：仅在 B 通过后，扩展三个 model seeds。要求 6/9 paired cells 正、平均
`TRUE-CIN >= +0.005`，并持续满足 binding / BAG specificity。

任何阶段失败都停止 MolHIV 上的当前 Beam8→分类路线；不以更多层、attention、K/T、loss
weight 或官方 valid/test 补救。

## 5. K-SVD 仅在 RAW gate 通过后进入

若阶段 C 通过，保持 endpoint/CIN/readout 不变，只将 RAW patch state 替换为压缩表示：

1. PCA（matched latent dimension）；
2. random real-prototype sparse code；
3. frozen K-SVD code；
4. task-aligned K-SVD code。

第 4 项必须同时优化 graph classification 与 patch-latent reconstruction，并以
PCA/random/frozen K-SVD 作为 matched controls。若 RAW 增量不能被 K-SVD 保留，结论是
“重构式压缩切断了任务信号”；只有 task-aligned K-SVD 恢复并超过这些 controls 时，它才是
MolHIV 强分类方法的一部分。

## 6. 执行命令

```bash
PY=/home/calendar/.conda/envs/gsn-official/bin/python
$PY -m code.run_molhiv_cin_beam8 \
  --fold-cache results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --max-graphs 0 --fold 1 --seed 0 --epochs 150 --batch-size 128 \
  --hidden 48 --layers 2 --dropout 0.5 --lr 1e-4 \
  --variants cin,cin_beam8_bond,cin_beam8_bond_shuffled,cin_beam8_bond_bag,cin_beam8_bond_no_patch \
  --output results/molhiv/cin_raw_beam_gate_fulltrain_fold1_seed0_20260823.json

$PY -m code.summarize_molhiv_fulltrain_raw_beam_gate \
  --input results/molhiv/cin_raw_beam_gate_fulltrain_fold1_seed0_20260823.json \
  --output results/molhiv/cin_raw_beam_gate_fulltrain_fold1_seed0_20260823_summary.json
```

该命令不读取 official-valid/test；输出中必须保留两者为零的审计字段。
