# LCRC-PW v1 — 运行与复现

Round: `lcrc-pcba-pw-v1`（定义见 `PROTOCOL.md`，在训练前冻结）。所有命令在仓库根目录执行。
远程主机 `res`（单机 2×A100-40GB，`rr` process backend）。训练用 pool `res-gpu1`，
**必须等 `rc-blind` 结束并释放 GPU1 之后**；`res-gpu0` 只在确认无既有进程时使用。

---

## 0. 本地测试（CPU，无需 GPU 与大内存）

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_pw_v1/tests
```

55 个用例，覆盖：距离桶定义/互斥/覆盖、同分量枚举与暴力 BFS 一致、单原子图零 pair、
subset cache 往返与 `require_full` 拒绝、批量 gather 与逐行 gather 一致、collate 只增加
pair 载荷、参数算术与逐位 W2 初始化、全局 RNG 不被额外推进、zero-init 逐位等价、
pair token 交换逐位不变（尾部非零时 logits 亦不变）、无 pair 分子残差严格为 0、
segment softmax 归一、单次 encode/无原子写回、所有新增参数梯度、brute-force vs batched、
注意力熵有界、配方/order plan/Test 标签隔离继承性。

---

## 1. 关系索引（一次性，纯 CPU，只读结构）

```bash
uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_audit.py build-pairs
```

输出 `runs/pair_cache/`：`pair_i int16`、`pair_j int16`、`pair_bucket uint8`、
`pair_ptr int64`、`meta.json`（逐桶计数、SHA256、`labels_read: false`、
5,000 行纯 Python BFS 自检 + 200 行与 RC 轮 distance cache 的交叉核对）。
该步骤不打开标签文件。

---

## 2. 训练前正确性检查（目标设备）

```bash
rr run res pw-checks --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_pw_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_audit.py checks --device cuda
```

写 `analysis/preflight.json`（11 项，全部带数值证据）；任一项失败退出码 4。正式训练
`pw_train.py` 在启动时会先复算这份检查，`runs/PW/run.json` 里记录其 SHA256，训练中途
改代码会被 checkpoint 的 schema/round 校验拒绝。

---

## 3. GPU profiling（正式训练前）

```bash
rr run res pw-profile --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_pw_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_audit.py profile \
     --device cuda --steps 25 --warmup 8 --valid-seconds 8
```

写 `analysis/profile.json`：同一批真实数据的 W2 / PW 训练 step p50/p90、相对开销、
按 40 epochs × 2,738 steps + 40 次 Validation + 一次 Train AP 推算的 GPU-hour 与预算判定
（目标 2.5 h，硬上限 3.0 h）。

---

## 4. 正式训练（单模型，seed 0，40 epochs）

```bash
rr run res pw-v1 --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_pw_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_train.py \
     --device cuda --epochs 40 --seed 0 --pair-cache runs/pair_cache \
     --out runs/PW --soft-seconds 9000 --hard-seconds 10800
```

`runs/PW/` 下：`best_valid.pt`、`epoch40.pt`、`last.pt`、`run.json`（哈希、参数量、
GPU-hour、环境、preflight SHA）、`train_curve.csv`、`per_task_ap.csv`、`training_order.json`。

---

## 5. 官方 Test 评分（freeze → score）

```bash
rr run res pw-freeze --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_pw_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_test.py freeze --device cuda

rr run res pw-score --pool res-local --result tracks/structural-sufficiency/lcrc_pcba_pw_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_test.py score
```

`freeze` 只写 Test logits 与哈希（含 Validation AP 复算），不打开任何 Test 标签；
`score` 验证 seal 与全部哈希后才读官方标签。预注册 checkpoint：
`PW_best_valid`、`PW_epoch40`、`W2_best_valid_epoch24`、`W2_epoch40`。

---

## 6. 注意力熵与对比表

```bash
uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_audit.py query-entropy --device cuda
uv run python tracks/structural-sufficiency/lcrc_pcba_pw_v1/code/pw_compare.py
```

写 `analysis/query_entropy.json`、`analysis/comparison.json`、`analysis/comparison_table.csv`；
`REPORT.md` 里的每个数字都应能从这三个文件追到。
