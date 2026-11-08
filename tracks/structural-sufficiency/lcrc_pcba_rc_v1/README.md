# LCRC-RC v1 — 运行与复现

Round: `lcrc-pcba-rc-v1`（见 `PROTOCOL.md`）。所有命令在仓库根目录执行。

远程主机 `res`（单机 2×A100-40GB，`rr` process backend）：
`REAL` → pool `res-gpu1`，`BLIND` → pool `res-gpu0`。

---

## 0. 本地测试（不需要 GPU，不需要大内存）

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_rc_v1/tests
```

覆盖：距离桶定义与互斥/覆盖、不相连分量的哨兵、三角索引逆变换、T_d 对称性与和、
空桶严格为 0、T_all = Σ w_d T_d、REAL/BLIND 对照（均匀角色 → 相同；角色—距离重绑定 →
REAL 变、BLIND 不变）、原子重编号不变性、跨分子隔离、CPU 逐对枚举 vs batched 实现、
参数与初始化身份、zero-init 前向等价、梯度路径、训练配方/损失/order plan 继承性。

---

## 1. 结构距离缓存（一次性，纯 CPU，只读结构）

```bash
uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_audit.py build-cache
```

输出 `runs/dist_cache/`（`pair_bucket.npy` uint8 152.7 MB + `pair_ptr.npy` int64 +
`meta.json`，含逐桶计数、SHA256 与一次 0.2% 抽样 BFS 自检），并写
`analysis/distance_cache_build.json`。该步骤**不打开标签文件**（`labels_read: false`），
因此 Test 图进入缓存时没有携带任何 Test 标签。

---

## 2. 训练前正确性检查（GPU）

```bash
rr run res rc-checks --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_audit.py checks --device cuda
```

写 `analysis/preflight.json`；任一项失败时退出码 4。

---

## 3. GPU 成本 profiling（正式训练前）

```bash
rr run res rc-profile --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_audit.py profile \
     --device cuda --steps 25 --warmup 8 --valid-seconds 8 --train-eval-seconds 60
```

写 `analysis/profile.json`：REAL / BLIND / W2 的同 batch 真实训练 step 时间
（p50/p90）、相对 W2 的开销、以及按 40 epoch × 2738 step + 40 次 Validation +
一次 Train AP 推算的 GPU-hour 与预算判定。

---

## 4. 正式训练（每臂一次，seed 0，40 epoch）

```bash
# RC-BLIND（GPU0）
rr run res rc-blind --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/BLIND \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_train.py \
     --model BLIND --device cuda \
     --out-dir tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/BLIND \
     --soft-seconds 10000 --hard-seconds 12600

# RC-REAL（GPU1，主候选）
rr run res rc-real --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_train.py \
     --model REAL --device cuda \
     --out-dir tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL \
     --soft-seconds 10000 --hard-seconds 12600
```

产物（每臂）：`run.json`、`train_curve.csv`、`per_task_ap.csv`、`training_order.json`、
`rng_state.pt`、`checkpoint_best.pt`（best-Validation，被选中）、
`checkpoint_epoch40.pt`、`checkpoint_last.pt`（可断点续训）。

断点续训（仅工程恢复用，协议禁止为了结果重跑搜索）：

```bash
... rc_train.py --model REAL --out-dir .../runs/REAL \
    --resume tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL/checkpoint_last.pt \
    --soft-seconds ... --hard-seconds ...
```

---

## 5. 官方 Test 预测冻结 + 评分

```bash
# 1) 冻结（只写 Test logits 与哈希，不读 Test 标签）
rr run res rc-test-freeze --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_test.py freeze --device cuda

# 2) 评分（校验封条后，通过继承的 cc_data 门读取官方 Test 标签）
#    远端 checkout 故意不带 data/ogb/*/raw/graph-label.csv.gz，所以这一步在本地跑：
rr pull rc-test-freeze                      # 拉回冻结的 Test logits 与 manifest
uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_test.py score
```

产物：`analysis/test_predictions/*.npy`、`analysis/test_prediction_manifest.json`、
`analysis/selected_checkpoint.json`、`analysis/test_scores.json`、
`analysis/per_task_valid_test.csv`。

---

## 6. 比较与报告数字

```bash
uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_compare.py
```

写 `analysis/comparison.json` 与 `analysis/comparison_table.csv`（对比表、差值、
高支持任务子集、稀疏任务主导性、分支活动量、推理开销）。`REPORT.md` 的数字全部来自
这些 JSON/CSV。

---

## 7. 推理开销

```bash
rr run res rc-inference --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_rc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rc_v1/code/rc_audit.py inference --device cuda
```

写 `analysis/inference_timing.json`（REAL / BLIND / W2 在相同 Validation 行上的
rows/s 与相对 W2 倍数）。
