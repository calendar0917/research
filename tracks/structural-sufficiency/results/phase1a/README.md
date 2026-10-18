# results/phase1a — SSRC Phase 1a

Protocol: `ssrc-phase1a-lipo-v1`  
Protocol document: `../../notes/phase1a_protocol.md`  
Config: `../../configs/phase1a.yaml`

## Files

| 路径 | 内容 | Git |
|---|---|---|
| `split_phase1a.json` | 冻结的内部 Fit/Monitor/Reserved + n=253 子集（structure-only，无标签） | tracked |
| `labels_fit_monitor.json` | 仅 Fit+Monitor 的 2,856 条标签；无 Reserved/valid/test | tracked |
| `CPU_GATES.md` | GPU 前 CPU 门禁摘要 | tracked |
| `REPORT.md` | R1+R2 GPU screening 结果、复现性、判定 | tracked |
| `run_manifest.tsv` / `run_manifest_r2.tsv` | R1 / R2 各 18 条 run 的 rr provenance | tracked |
| `analysis_summary.json` / `analysis_summary_r2.json` | R1 / R2 汇总（无逐行标签） | tracked |
| `analysis_replicates.json` | R1+R2 pooled 与 run-to-run 对比 | tracked |
| `cpu_gates.json` | 完整门禁 JSON | local evidence (ignored) |
| `runs/phase1a/`, `runs/phase1a_r2/` | 每条轨迹的完整 JSON（曲线、预测、参数、成本） | local evidence (ignored; pulled from `res`) |

## 纪律

- 官方 valid/test 与内部 Reserved 的标签/评分**未读**，不进入本目录任何主表。
- 模型/阈值/口径在 GPU 前冻结；GPU 后不改。
- R1/R2 是同一冻结设计的两个完整 replicate；R2 用于检验 `git_dirty` 与
  run-to-run 非确定性，不用于结果驱动的 model selection。
- 本目录数字不与其它 track 的结果混表。
