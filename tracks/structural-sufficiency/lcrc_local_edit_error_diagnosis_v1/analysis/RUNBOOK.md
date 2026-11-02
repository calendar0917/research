# RUNBOOK — 本轮执行状态与恢复入口

Round: `lcrc-local-edit-error-diagnosis-v1`
最后更新：2026-11-02（Asia/Shanghai）

## 状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 配对构造 | **DONE** | 509 对 / 502 参与分子 / 211 类变换；`analysis/pairs.csv`、`pair_mappings.json`、`core_classes.json`、`data_manifest.json` |
| P0 划分 | **DONE**，`status=OK` | `analysis/component_split.json`；fit/select/diagnose/audit = 2016/336/504/504 行，配对 301/50/84/74 |
| 结构 cache + Fit 归一化 | **DONE（本地）** | `cache/`（gitignored，可重建）；`cache/manifest.json` |
| CPU 正确性闸门（pre） | **DONE** 10/10 | `analysis/correctness_gates.json` |
| 6 个训练单元 | **未启动**（用户暂停） | 代码、冻配、run matrix 已就绪 |
| P1 成对回溯 | 代码就绪，**待真实 checkpoint** | `code/edit_trace.py`（已用 2-epoch 冒烟 checkpoint 跑通并核验对称性） |
| P2 探针 / P3 Audit | **未实现** | `code/edit_probe.py` 待写 |
| 报告 / 决策 | 未写 | `REPORT.md`、`DECISION.md` |

`git commit 46788bdf` 已推送到 `origin/main`。

## 恢复步骤（一次会话内可完成）

```bash
D=tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1

# 0) 预检。注意：上一轮部署失败是因为**远端** checkout 里另一条 track 的
#    tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/analysis/gates.json
#    被修改（不是本轮文件）。rr 拒绝覆盖脏的 tracked 工作树。
#    处理办法：在远端把该文件还原/提交（属用户决定），不要动本轮文件。
rr doctor res

# 1) 部署本轮 commit（需要远端工作树干净）
rr deploy res

# 2) 建 cache（每主机一次，约 10 s）
rr run res edit-prep --result $D/cache --result $D/analysis \
  -- uv run python $D/code/edit_prepare.py

# 3) 短 CUDA 冒烟（2 epochs，写入 runs/smoke_*，不产生结果）
rr run res edit-smoke --pool res-gpu0 \
  -- uv run python $D/code/edit_train.py --model C1 --seed 0 --smoke

# 4) 三个 wave，每 pool 至多 1 个进程，总并发 <= 2
rr run res edit-c1-s0 --pool res-gpu0 --result $D/runs/edit_test_s0_C1 \
  -- uv run python $D/code/edit_train.py --model C1 --seed 0
rr run res edit-a0-s0 --pool res-gpu1 --result $D/runs/edit_test_s0_A0 \
  -- uv run python $D/code/edit_train.py --model A0 --seed 0
# ... A0:s1 / C1:s1 -> C1:s2 / A0:s2（见 configs/run_matrix.json 的 waves）

# 5) 拉回 checkpoint 与曲线
rr pull edit-c1-s0 && rr pull edit-a0-s0 && ...

# 6) post 阶段核验（原子重编号 / 反号 / batch 拆分在真实权重上）
uv run python $D/code/edit_audit.py --stage both

# 7) P1 回溯（CPU，约 1 min/单元）
uv run python $D/code/edit_trace.py --models C1 A0 --seeds 0 1 2

# 8) P2 探针 + 冻结 + 一次性 Audit + bootstrap/影响度/闭环（待实现）
uv run python $D/code/edit_probe.py --stage fit        # Diagnose 拟合 + CV -> probe_freeze.json
uv run python $D/code/edit_probe.py --stage audit      # 一次性读 Audit 标签并评分
uv run python $D/code/edit_probe.py --stage report     # bootstrap / influence / cycles
```

## 预算记账

硬上限 4.0 A100 GPU·h。单 run 估算：Fit 2016 行 → 16 steps/epoch × 360 epochs；
上一轮 3360 行同配方约 0.088 GPU·h/run，本轮约 0.05 GPU·h/run，6 个单元约
0.3 GPU·h，加冒烟与推理预计 < 0.5 GPU·h。已用：**0 GPU·h**。
执行后必须把逐 run GPU 小时与 `rr` job id 写入 `analysis/gpu_ledger.json`。

## 已知必须如实报道的实现细节

1. 本轮独立重算得到 509 对 / 211 类变换，设计期预筛为 501 对 / 228 类；参与分子
   数 502 完全一致。差异来自 RDKit 重建与"每对唯一定义"的消解序，未放宽任何冻结条件。
2. 442/509 对的核心自同构群非平凡 → 逐原子热点多数不唯一，群体统计只用求和量。
3. `dL_var` 与 `dR` 是规约依赖的（113/509 对的变量片段重原子数不同）。
4. `analysis/label_access_ledger.json` 记录的是真实访问历史，含开发期多次
   Fit/Select/Diagnose 读取；**Audit 标签尚未被读取**（`audit_labels_accessed=false`）。
