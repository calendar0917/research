# lcrc_explanation_fidelity_alignment_v1

同一批冻结 LCRC/GINE 预测模型上，自动连通子图发现方法能否**同时**做到

1. 模型预测忠实性（移除所选子图真的改变预测），与
2. 独立化学结构贡献定位（WC_LOGP 原子加性贡献被所选子图捕获）。

主实验：WC_LOGP 模型，官方 Valid 420 + Test 420（+ 固定 128 Train 描述）；
辅助：固定 823 对 Lipophilicity 描述性诊断。无新训练、无新 seed、无 checkpoint 重选。

- 协议：`PROTOCOL.md`
- 代码：`code/lfa_*.py`；远端入口：`run_remote.sh <stage>`
- 测试：`uv run pytest tests -q`
- 结果：`results/`（原始记录，`rr pull` 拉回）与 `analysis/`（汇总、门禁、决策）
- 报告：`REPORT.md`；决策：`DECISION.md`

## 远端执行（`rr`）

```bash
rr doctor res
rr deploy res
rr run res lfa-fields-lcrc --pool res-gpu0 --gpus 1 --cpus 8 --mem 24G --time 02:00:00 \
  --result tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields \
  -- bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-lcrc
rr run res lfa-fields-gine --pool res-gpu1 --gpus 1 --cpus 8 --mem 24G --time 02:00:00 \
  --result tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/fields \
  -- bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-gine
rr run res lfa-pool --pool res-local --gpus 0 --cpus 8 --mem 24G --time 02:00:00 \
  --result tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/results/pool \
  -- bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh pool
rr run res lfa-select-lcrc-main --pool res-gpu0 ... -- ... select-lcrc-main
rr run res lfa-select-lcrc-aux  --pool res-gpu0 ... -- ... select-lcrc-aux
rr run res lfa-select-gine-main --pool res-gpu1 ... -- ... select-gine-main
rr run res lfa-select-gine-aux  --pool res-gpu1 ... -- ... select-gine-aux
```

本地聚合（拉回后，CPU）：

```bash
uv run python code/lfa_prepare.py && uv run python code/lfa_gates.py
uv run python code/lfa_pred.py && uv run python code/lfa_main.py && uv run python code/lfa_aux.py
uv run python code/lfa_manifest.py && uv run python code/lfa_report.py
```
