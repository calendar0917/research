# TRACK — LCRC 瓶颈诊断（public random122 / scaffold122, seed 0）

## Charter

在**固定学习过程**下，用一组预先冻结的干预，直接测量 LCRC 主线（C1 / K / chem8 / 修正
D-MPNN）在本任务上的可修复性能限制来自哪一类：**选择（H_SEL）**、**优化预算（H_OPT）**、
**监督样本量/覆盖（H_DATA）**，还是**冻结表示的读出利用（H_USE）**。原始输入是否缺少任务
相关信息（H_REP）**本轮不直接检验**，也不靠排除前三项宣布它成立。

本轮不是模型搜索，不产出新 SOTA；成功结果是一张有直接证据的路线诊断。

## 入口

```bash
uv run python code/run.py prepare      # 嵌套子集 / 共用 norm / 缓存指纹（CPU）
uv run python code/run.py smoke        # 2-epoch 冒烟（单独计费，不进结果）
uv run python code/run.py train --model C1_R1 --dataset Lipophilicity --fit 25
uv run python code/run.py analyze      # 选择/数据预算/读出三份机器量
uv run python code/run.py report       # REPORT.md + 3 张核心图 + run_manifest
uv run python -m pytest tests -q       # 不变量回归（快、CPU）
```

远端执行统一走 `rr`（`code/lb_dispatch.py` → `rr run res --pool res-gpu0/res-gpu1`）。

## 冻结的科学常量

见 `PROTOCOL.md`、`configs/diagnosis.yaml`、`code/lb_common.py`。任何训练开始前的改动都
必须同步三者；训练期间的代码快照不可变。

## 只读来源

`lcrc_performance_route_8h_public122_s0_v1`（修正口径 + 代码）、
`lcrc_lipo_model_dissection_v1`、`lcrc_molnet_benchmark_pilot_v2`、
`lcrc_cross_task_preparation_v1`、`lcrc_pcba_frozen_readout_probe_v1`、
`lcrc_lipo_oracle_task_relevance_public122_s0_v1`。

## 结论

见 `REPORT.md` 首页（先答 H_SEL / H_OPT / H_DATA / H_USE）与
`analysis/synthesis/{selection,data_budget,readout}.json`。
