# code/ 说明

本目录是 GPS × MOLHIV 复现的**实验编排与结果解析代码**。
模型本身是 GraphGPS 官方代码，未修改。

| 文件 | 作用 |
|---|---|
| `kaggle_run.py` | Kaggle 执行器（本次 v22 实际运行版本）：装环境 → clone 官方仓到固定 commit → 按矩阵训练 → 解析日志 → 写 `registry.jsonl` / `paper_vs_reproduced.csv` |
| `kaggle_run_v16.py` | 同上，v16 运行时的历史版本（对照用） |
| `collect_gps_results.py` | 从单个 GraphGPS 训练日志解析每个 epoch 的 train/val/test，按 val AUC 选 best epoch，写 `registry.jsonl` |
| `consolidate_results.py` | 合并多来源 registry，按 `mean ± std` 聚合，并按论文值给出 reproduced/partial/not_reproduced 判定 |
| `run_gps_molhiv_seeds.sh` | 本地 CPU 串行跑 molhiv seed 0–3 的脚本（备查；本次结果来自 Kaggle GPU） |
| `make_notebook.py` | 生成/更新 Kaggle notebook |
| `notebook/hod-gnn-replication.ipynb` | 实际在 Kaggle 上运行的 notebook（v22） |
| `config/ogbg-molhiv-GPS+RWSE.yaml` | GraphGPS 官方 config 原文件（commit `28015707…`） |

## 关键入口约定

- 数据：OGB `ogbg-molhiv`，官方 scaffold split（由 GraphGPS 自动下载）
- 指标：ROC-AUC；`metric_best: auc` → 验证集选 best epoch
- 种子：`--protocol paper-seeds-provisional`，seed 由 `--seeds` 指定（或默认 0–3）

## 运行（Kaggle notebook 内）

```
--method gps --dataset molhiv --seeds 2 3 --protocol paper-seeds-provisional --resume
```

本地 CPU 版见 `run_gps_molhiv_seeds.sh`（100 epoch ≈ 6.4h/seed，仅供无 GPU 时使用）。

## 与官方代码的关系

`kaggle_run.py` 只做“编排 + 解析”，训练命令等价于在官方仓内执行：

```bash
python main.py --cfg configs/GPS/ogbg-molhiv-GPS+RWSE.yaml wandb.use False seed <N>
```

唯一对官方代码的兼容性处理记录在 `../docs/DEVIATIONS.md` 与 `../docs/SOURCE_LEDGER.md`。
