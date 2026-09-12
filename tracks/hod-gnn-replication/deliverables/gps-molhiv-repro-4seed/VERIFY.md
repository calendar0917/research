# 核对指南（VERIFY.md）

面向核对者（师兄）。全部操作在解压后的包根目录执行。

## 0. 校验文件完整性

```bash
sha256sum -c SHA256SUMS
```

## 1. 看结果总表

```bash
cat registry/gps-molhiv-4seed-perseed.csv
cat registry/paper_vs_reproduced_4seed.csv
```

预期：
```
seed,best_epoch,valid_auc,test_auc,...
0,43,0.81136,0.7885,...
1,19,0.8073,0.77454,...
2,63,0.81982,0.78382,...
3,44,0.83384,0.76889,...
```

## 2. 从原始日志复核 best epoch 与 test AUC

每个 seed 的日志末尾会打印 best epoch 的 val/test，例如：

```bash
for s in 0 1 2 3; do
  echo "=== seed $s ==="
  grep "Best so far: epoch" logs/gps-molhiv-$s.log | tail -1
done
```

`Best so far: epoch N ... val_auc: V test_auc: T` 应逐一对应上表。
（日志中 `Best so far` 由 GraphGPS 按验证集最优维护，等价于 `metric_best: auc`。）

## 3. 复核聚合与判定（独立复算）

```bash
python3 - <<'PY'
import json, statistics
vals=[json.loads(l)['test_metric'] for l in open('registry/registry_4seed.jsonl') if l.strip()]
m=statistics.mean(vals); s=statistics.stdev(vals)
paper_m, paper_s = 0.7880, 0.0101
comb=(s**2+paper_s**2)**0.5
verdict='reproduced' if abs(m-paper_m)<=comb else ('partial' if abs(m-paper_m)<=2*comb else 'not_reproduced')
print(f"mean={m:.4f} std={s:.4f} n={len(vals)} |diff|={abs(m-paper_m):.4f} combined={comb:.4f} -> {verdict}")
PY
```

预期：`mean=0.7789 std=0.0089 n=4 |diff|=0.0091 combined=0.0134 -> reproduced`

## 4. 确认协议与代码来源一致

```bash
python3 - <<'PY'
import json
for s in [json.loads(l) for l in open('registry/registry_4seed.jsonl') if l.strip()]:
    print(s['seed'], s['code_repo'], s['code_commit'], s['split_hash'], s['metric_name'])
PY
```

应全部为：
- repo `https://github.com/rampasek/GraphGPS.git`
- commit `28015707cbab7f8ad72bed0ee872d068ea59c94b`
- split_hash `feda8af43ac4b55b656ba2e02727d03560e898f3edad160a615797f6ad321179`
- metric `roc_auc`

并核对 `code/config/ogbg-molhiv-GPS+RWSE.yaml` 与官方 commit 内文件一致。

## 5. 看每轮实际执行参数

```bash
cat registry/run_state_v16.json   # args.seeds = [0,1]
cat registry/run_state_v22.json   # args.seeds = [2,3], status = complete
```

## 6.（可选）重跑验证

按 `RESULTS_AND_PROTOCOL.md` §2.1，克隆官方仓到固定 commit、装好官方环境后：

```bash
python main.py --cfg configs/GPS/ogbg-molhiv-GPS+RWSE.yaml wandb.use False seed 0
```

用包内 `code/collect_gps_results.py` 从新日志解析，应得到相近（非逐位相同，GPU 训练有随机性）的 test AUC。

## 7. 需要注意的点

- 这是 `paper-seeds-provisional`：论文未披露 seed，用 `[0,1,2,3]` 替代。
- `batch_size=32` 用的是 GraphGPS 官方 config，论文附录 molhiv 为 128（见 `docs/DEVIATIONS.md`）。
- 早期 v10–v14 有 4-seed 日志但 `best_epoch=1`、未训练起来，已弃用；本包不含它们。
