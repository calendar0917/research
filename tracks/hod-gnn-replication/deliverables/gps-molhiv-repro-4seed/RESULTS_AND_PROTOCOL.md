# 结果与实验协议 — GPS × MOLHIV

## 1. 结果

### 1.1 逐 seed

| seed | best epoch | valid ROC-AUC | test ROC-AUC | 来源 |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 43 | 0.81136 | 0.78850 | v16 |
| 1 | 19 | 0.80730 | 0.77454 | v16 |
| 2 | 63 | 0.81982 | 0.78382 | v22 |
| 3 | 44 | 0.83384 | 0.76889 | v22 |

### 1.2 聚合与判定

| 项 | 值 |
|---|---|
| 复现 mean ± std (n=4) | **0.7789 ± 0.0089** |
| 论文 | **0.788 ± 0.0101** |
| `combined = sqrt(std² + paper_std²)` | 0.0134 |
| `|mean − paper|` | 0.0091 |
| **判定** | **reproduced**（0.0091 ≤ 0.0134） |

判定规则来自 `code/consolidate_results.py`：
```
combined = sqrt(repro_std² + paper_std²)
reproduced       if |mean − paper| ≤ combined
partial          if |mean − paper| ≤ 2·combined
not_reproduced   otherwise
```

## 2. 实验协议

| 项 | 内容 |
|---|---|
| 协议 ID | `hod-gnn-replication-v1` |
| seed 协议 | `paper-seeds-provisional`（论文未披露 seed，用 `[0,1,2,3]` 临时替代，不冒充严格论文复现） |
| 方法 | GraphGPS（GPS）——官方代码，未重写 |
| 数据集 | OGB `ogbg-molhiv`，官方 scaffold split |
| split | `split_hash = feda8af43ac4b55b656ba2e02727d03560e898f3edad160a615797f6ad321179`；train/valid/test = 32901/4113/4113 |
| 任务 / 指标 | 二分类 / **ROC-AUC，越高越好** |
| 选模 | config `metric_best: auc` → **按验证集 AUC 选 best epoch**，再取该 epoch 的 test AUC；全程不看 test 选模 |
| 聚合 | 跨 seed `mean ± stdev(ddof=1)` |
| 代码 | GraphGPS `rampasek/GraphGPS` @ `28015707cbab7f8ad72bed0ee872d068ea59c94b` |
| 配置 | `configs/GPS/ogbg-molhiv-GPS+RWSE.yaml`（见包内 `code/config/`） |
| 超参 | `max_epoch=100`, `num_warmup_epochs=5`, `gt.layers=10`, `n_heads=4`, `dim_hidden=64`, `gnn.head=san_graph`, `base_lr=1e-4`, `scheduler=cosine_with_warmup`, `batch_size=32`, RWSE PE（`times_func: range(1,17)`, `dim_pe=16`） |
| 环境 | Kaggle GPU，Tesla P100-PCIE-16GB；python 3.12.13；torch 1.13.0+cu117；cuda 11.7；GraphGPS 自带 `.venv310` |
| 耗时 | ≈2.9–3.1 h / seed |

### 2.1 逐 seed 命令（等效）

```bash
# 官方仓库固定 commit
git clone https://github.com/rampasek/GraphGPS.git
git -C GraphGPS checkout 28015707cbab7f8ad72bed0ee872d068ea59c94b

for seed in 0 1 2 3; do
  python main.py --cfg configs/GPS/ogbg-molhiv-GPS+RWSE.yaml wandb.use False seed $seed
done
```

### 2.2 结果如何从日志得出

`code/collect_gps_results.py` 解析 GraphGPS 逐 epoch 输出：
- 每个 epoch 打印 `train: {...}` / `val: {...}` / `test: {...}` 三个 dict（AUC 字段 `auc`）
- 按 `val.auc` 最大选 best epoch
- 取该 epoch 的 `test.auc` 作为该 seed 的 test 指标，写入 `registry.jsonl`

在日志中可直接看到，例如 `logs/gps-molhiv-2.log` 末尾：
```
> Epoch 99: ... | Best so far: epoch 63   val_auc: 0.8198   test_auc: 0.7838
```

## 3. 已知偏差（`docs/DEVIATIONS.md`）

| 项 | 论文 | 本次 | 说明 |
|---|---|---|---|
| `batch_size` (molhiv) | 128（论文附录） | **32**（GraphGPS 官方 config） | 保留官方基线配置；论文 0.788 可能基于 128，差异已登记，未消除 |
| seed | 未披露 | `[0,1,2,3]` 临时替代 | 协议命名 `paper-seeds-provisional`，不代表严格论文复现 |
| GPU | 未披露 | Tesla P100-16GB | — |

## 4. 过程说明

- v16（2026-08-24）只跑了 seed 0,1（`run_state_v16.json: args.seeds=[0,1]`）。
- 更早的 v10–v14 曾有 4-seed 日志，但 `best_epoch` 全为 1、耗时仅 ~250s，属未训练起来的坏运行，**已弃用，不作为结果**。
- v22（2026-09-11）补跑 seed 2,3（`run_state_v22.json: args.seeds=[2,3]`, `status=complete`）。
- 合并 v16 + v22 的 molhiv 记录得到本表 4 seed。
