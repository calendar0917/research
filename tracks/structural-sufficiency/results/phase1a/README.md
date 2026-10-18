# results/phase1a — SSRC Phase 1a

Protocol: `ssrc-phase1a-lipo-v1`  
Protocol document: `../../notes/phase1a_protocol.md`  
Config: `../../configs/phase1a.yaml`

## Files

| 路径 | 内容 | Git |
|---|---|---|
| `split_phase1a.json` | 冻结的内部 Fit/Monitor/Reserved + n=253 子集（structure-only，无标签） | tracked |
| `CPU_GATES.md` | GPU 前 CPU 门禁摘要 | tracked |
| `REPORT.md` | GPU screening 结果与判定（GPU 后生成） | tracked |
| `cpu_gates.json` | 完整门禁 JSON | local evidence (ignored) |
| `runs/` | 每条轨迹的完整 JSON（曲线、预测、参数、成本） | local evidence (ignored; pulled from `res`) |

## 纪律

- 官方 valid/test 与内部 Reserved 的标签/评分**未读**，不进入本目录任何主表。
- 模型/阈值/口径在 GPU 前冻结；GPU 后不改。
- 本目录数字不与其它 track 的结果混表。
