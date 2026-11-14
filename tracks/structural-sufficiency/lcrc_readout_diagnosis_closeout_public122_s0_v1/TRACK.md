# TRACK — 读出诊断收尾（public random122 / Lipophilicity, seed 0）

状态：**DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP**（研究者看过上一轮结果后设计，不称独立确认）。

## Charter

1. 恢复原计划的冻结表示来源：`LIP_C1_R1_F100` 的 E1 `R_SEL` 状态（update=8165，来自
   `best_R_SEL.pt`），验证可恢复并能精确重放，替换上游“改用 u0_snapshot(9720)”的来源。
2. 在该来源上重跑三条读出探针 `P_WARM / P_POST / P_PRE`（科学规格不变），并**直接测量**
   额外通道是否被训练、是否形成可测响应，从而区分“读出没有可兑现收益”与“PRE/POST 干预
   没有形成有效差异”。

不重跑主实验、不搜索新架构、不采购 BACE、不加主干联合训练。

## 入口

```bash
uv run python code/run.py source-audit --device cpu   # 来源恢复 + 身份化 h 缓存（eval）
uv run python code/run.py smoke --device cpu
uv run python code/run.py probe --rule P_WARM --device cuda --out-dir runs/P_WARM
uv run python code/run.py analyze
uv run python code/run.py report
uv run python -m pytest tests -q
```

远端统一走 `rr`（`code/rc_dispatch.py` → `rr run res --pool res-gpu0/res-gpu1`）。

## 只读来源

`lcrc_bottleneck_diagnosis_public122_s0_v1`（交付 commit `7e668a9a`）：读取其
`best_R_SEL.pt`、`grid_predictions.npz`、`analysis/synthesis/*.json`，导入其
`lb_common / lb_data / lb_train / lb_selection` 并记录源码 hash。上游目录只读。

## 关键产物

- `analysis/source_audit.json`：来源/缓存/重放/逐分子对齐/交叉核对。
- `analysis/{results,learning_path,comparisons,budget}.json` + `analysis/figures/`。
- `REPORT.md`：开头四句回答（来源一致性、通道响应、收益、方向）。
