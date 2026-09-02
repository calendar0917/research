# code/

本轨实现入口。

- `run_gps.py` — GPS 基线运行入口
- `run_graphvit.py` — GraphViT 基线运行入口
- `run_subgraph_policy.py` — Full/Random/Policy-Learn 基线运行入口（共用代码，切换 policy 配置）
- `audit_splits.py` — 数据集 split 校验
- `collect_run.py` — 单次运行结果收集
- `summarize_results.py` — 结果汇总与审计
- `kaggle_run.py` — Kaggle/Linux 单脚本执行器（结构化日志、心跳、单元级 resume）
- `kaggle_ctl.sh` — Kaggle CLI 推送、状态轮询和结果同步

Kaggle 操作和断点策略见 [`docs/KAGGLE_RUNBOOK.md`](../docs/KAGGLE_RUNBOOK.md)。

外部官方仓可放 `code/vendor/` 或子模块，并在 TRACK 写清环境。
