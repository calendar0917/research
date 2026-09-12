# GPS × MOLHIV 复现结果交付包（4 seed）

> 交付日期：2026-09-12
> 实验协议：`hod-gnn-replication-v1` / `paper-seeds-provisional`
> 目的：请师兄核对 GPS 基线在 `ogbg-molhiv` 上的复现结果与协议。

## 一句话结论

GPS 在 MOLHIV 上 4 个 seed 复现：**mean test ROC-AUC = 0.7789 ± 0.0089（n=4）**，
论文值 **0.788 ± 0.0101** → 判定 **reproduced**。

## 结果

| seed | best epoch | valid AUC | test AUC |
|:---:|:---:|:---:|:---:|
| 0 | 43 | 0.81136 | 0.78850 |
| 1 | 19 | 0.80730 | 0.77454 |
| 2 | 63 | 0.81982 | 0.78382 |
| 3 | 44 | 0.83384 | 0.76889 |
| **mean ± std (n=4)** | — | — | **0.7789 ± 0.0089** |

论文 `0.788 ± 0.0101`（`|diff| = 0.0091 ≤ sqrt(0.0089²+0.0101²) = 0.0134`）→ **reproduced**。

## 包内文件

```
.
├── README.md                     # 本文件
├── RESULTS_AND_PROTOCOL.md       # 结果表 + 完整实验协议 + 已知偏差
├── VERIFY.md                     # 逐步核对指南（师兄用）
├── SHA256SUMS                    # 全部文件校验和
├── code/
│   ├── README.md
│   ├── kaggle_run.py             # 实际执行器（v22 运行）
│   ├── kaggle_run_v16.py         # 执行器（v16 运行，历史）
│   ├── collect_gps_results.py    # 从日志解析结果 → registry.jsonl
│   ├── consolidate_results.py    # 聚合 + 判定 reproduced/partial/not_reproduced
│   ├── run_gps_molhiv_seeds.sh   # 本地 CPU 串行脚本（备查）
│   ├── make_notebook.py          # 生成 Kaggle notebook
│   ├── notebook/hod-gnn-replication.ipynb
│   └── config/ogbg-molhiv-GPS+RWSE.yaml   # 官方 config（pinned commit 内原文件）
├── logs/
│   ├── gps-molhiv-{0,1,2,3}.log  # GraphGPS 原始逐 epoch 训练日志
│   ├── console_v16_run.log       # v16 会话控制台日志
│   ├── console_v22_run.log       # v22 会话控制台日志
│   └── events_v16.jsonl          # v16 阶段/心跳事件
├── registry/
│   ├── registry_v16.jsonl        # seed 0,1 原始记录
│   ├── registry_v22.jsonl        # seed 2,3 原始记录
│   ├── registry_4seed.jsonl      # 合并后的 4 seed 记录（唯一数字源）
│   ├── gps-molhiv-4seed-perseed.csv
│   ├── paper_vs_reproduced_{v16,v22,4seed}.csv
│   └── run_state_{v16,v22}.json  # 每轮实际命令 args
└── docs/
    ├── PROTOCOL.md               # 轨内协议
    ├── DEVIATIONS.md             # 与论文的偏差登记
    └── SOURCE_LEDGER.md          # 官方代码来源登记
```

## 数据来源

| seed | 来源 | 日期 | run_id |
|---|---|---|---|
| 0, 1 | Kaggle `calendar917/hod-gnn-replication` version 16 | 2026-08-24 | `20260824T135339Z-p45` |
| 2, 3 | Kaggle 同 notebook version 22 | 2026-09-11 | `20260911T155609Z-p43` |

原始训练仓库：GraphGPS 官方 `https://github.com/rampasek/GraphGPS.git` @ commit
`28015707cbab7f8ad72bed0ee872d068ea59c94b`（GPU：Tesla P100-16GB）。

核对方式见 `VERIFY.md`；协议细节见 `RESULTS_AND_PROTOCOL.md`。
