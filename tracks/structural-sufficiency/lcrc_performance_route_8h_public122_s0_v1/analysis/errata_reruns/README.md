# `analysis/errata_reruns/` — 逐字节副本

本目录是 ERRATA E4 两次新训练的**原样副本**（`cp`，未修改一个字节），因为工作区里 `runs/` 被 `.gitignore`
忽略、旧 35 条 run 的权重在轮末已删盘，而这两条新证据不应只存在于本机磁盘上。

| 子目录 | 对应工作区路径 | 内容 |
| --- | --- | --- |
| `DMPNN_LR1E3_FIX/` | `runs/DMPNN_LR1E3_FIX__Lipophilicity/` | `train.json`、`curve.csv`（100 epoch）、`predictions_best.npz`（best-Valid epoch 的预测） |
| `DMPNN_LR3E4_FIX/` | `runs/DMPNN_LR3E4_FIX__Lipophilicity/` | 同上 |

未复制 `best.pt`（各 1.3 MB，二进制；工作区仍保留，若需要长期保存应另行归档）。

校验方式（与工作区副本逐字段一致，可用 `analysis/dmpnn_schedule_audit.json` 交叉核对）：

```bash
sha256sum analysis/errata_reruns/*/* 
python -c "import json;d=json.load(open('analysis/dmpnn_schedule_audit.json'));\
print({k:v['metrics_best']['test']['rmse'] for k,v in d['observed_runs'].items() if k.endswith('FIX')})"
# -> {'DMPNN_LR1E3_FIX': 0.56694..., 'DMPNN_LR3E4_FIX': 0.58468...}
```

两条 run 都从 commit `57a4777d` 启动，`provenance.code_hashes["code/lcpr_dmpnn.py"] = f6994af0…`
（含基准 lr 修复）；它们与 legacy 两条的区别**仅**为 `optimizer_base_lr: peak_lr → 1e-4`。
详见 `ERRATA.md` §5。
