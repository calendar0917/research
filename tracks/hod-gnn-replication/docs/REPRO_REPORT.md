# HOD-GNN 基线复现审计 — GPS × MOLHIV

> 协议：`hod-gnn-replication-v1` · `paper-seeds-provisional`
> 证据所在：`results/kaggle/v16/`（seed 0,1）、`results/kaggle/v22/`（seed 2,3）
> 另见：`docs/PROTOCOL.md`、`docs/DEVIATIONS.md`、`docs/SOURCE_LEDGER.md`

## 结果

| seed | best epoch | valid AUC | **test AUC** | 状态 | 来源 |
|------|-----------|-----------|--------------|------|------|
| 0 | 43 | 0.81136 | **0.78850** | success | v16 |
| 1 | 19 | 0.80730 | **0.77454** | success | v16 |
| 2 | 63 | 0.81982 | **0.78382** | success | v22 |
| 3 | 44 | 0.83384 | **0.76889** | success | v22 |

**聚合（n=4）：`0.7789 ± 0.0089`**，论文 `0.788 ± 0.0101` → **判定 reproduced**

判定按 `consolidate_results.py` 规则：
`combined = sqrt(repro_std² + paper_std²) = sqrt(0.0089²+0.0101²) = 0.0134`
`|mean − paper| = |0.7789 − 0.7880| = 0.0091 ≤ 0.0134` → `reproduced`

## 评估协议

- **协议**：`paper-seeds-provisional`（论文未披露 seed，使用 [0,1,2,3] 临时替代；不冒充严格论文复现）
- **指标**：ROC-AUC，越高越好（config `metric_best: auc`）
- **选优**：按**验证集** AUC 选 best epoch，再取该 epoch 的 test AUC（全程不看 test 选模）
- **数据**：OGB `ogbg-molhiv` 官方 scaffold split；`split_hash = feda8af4…321179`；train/valid/test = 32901/4113/4113
- **代码/commit**：GraphGPS 官方仓 `https://github.com/rampasek/GraphGPS.git` @ `28015707cbab7f8ad72bed0ee872d068ea59c94b`
- **配置**：`configs/GPS/ogbg-molhiv-GPS+RWSE.yaml`（max_epoch=100, warmup=5, gt.layers=10, n_heads=4, dim_hidden=64, lr=1e-4, RWSE PE）
- **批量/偏差**：`batch_size=32`（官方 config）；论文附录 molhiv 写 128 → 记录在 DEVIATIONS.md
- **环境**：Kaggle Tesla P100-16GB；torch 1.13.0+cu117；python 3.12.13；约 3h/seed

## 逐 seed 原始命令（等效）

```bash
python main.py --cfg configs/GPS/ogbg-molhiv-GPS+RWSE.yaml wandb.use False seed {0,1,2,3}
```

## 过程备注

- v16（2026-08-24）跑了 seed 0,1（`run_state.json: args.seeds=[0,1]`）。
- v10−v14 曾有 4-seed 的 molhiv 日志，但 `best_epoch` 全为 1、耗时 ~250s，是**未训练起来的坏运行**，弃用。
- 本次 docs 成稿当日补充运行 v22（2026-09-11）：seed 2,3，`run_id=20260911T155609Z-p43`，`status=complete`。
- 合并 v16 + v22 registry 得到完整 4-seed。

## 最终判定（GPS / MOLHIV）

- **reproduced**（n=4，`0.7789 ± 0.0089`，覆盖论文 `0.788 ± 0.0101`）
- 注意：`batch_size=32` vs 论文 128 的 deviation 未消除，结论基于官方 config。