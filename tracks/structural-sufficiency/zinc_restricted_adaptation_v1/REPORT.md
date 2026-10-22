# REPORT — zinc-restricted-adaptation-v1 (DBT-PA)

Track `structural-sufficiency`, round `zinc-restricted-adaptation-v1`.
Design: `DESIGN.md`; config: `configs/pa_v1.json`; checks:
`analysis/cpu_checks.json`; curve: `analysis/curves.png`; freeze:
`analysis/test_freeze.json`; test: `analysis/test_report.json`; groups:
`analysis/groups.md`.

**Status: complete.** One candidate, one seed-0 short schedule (`E = 310`,
48,670 updates), best-valid selection on the full 1,000-molecule valid split,
then one descriptive test read of the frozen checkpoint. The run was not
early-stopped and the cosine horizon was not changed.

---

## 0. 一句话结论

冻结 P3 局部 SSL 栈、只在 `a/b/w` 后加 rank-8 适配器的 DBT-PA，在本轮
310-epoch / 48,670-update 短日程下 **明显失败**：best-valid `0.140224 @287`，
test `0.107659`。同一 epoch 的原 DBT 参考 valid 是 `0.101307`，原 DBT 在
前 310 epoch 内的最好 valid 是 `0.085079 @299`（两者 cosine 不同，只能作短期
进度参考）。新模型没有任何 0.05x 证据；即使 test 比 valid 容易，也没有接近
现有参照。剩余 90 epoch 时 valid 已稳定在 `0.141` 附近，继续到 310 只是完成
声明日程，不是等待翻盘。

---

## 1. 模型、初始化与参数

DBT-PA = 原 DBT 的全部 token / 关系 / 4 层全图 attention / reader，加上：

* 冻结的 P3 `atom_encoder` / `bond_encoder` / `junction_encoder`；
* 三个独立的 `x + U SiLU(D x)` 适配器，`D: 96→8` 默认初始化，
  `U: 8→96` weight/bias 零初始化。

局部基座来自 **seed-0 train-only masked-SSL**
`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
(sha256 `006fabc11efede...`)，不是监督 DBT 的 `best.pt`；三个 encoder 的
tensor 与 SSL 文件逐项相等。原 DBT 模块先按 seed 0 构造，适配器最后追加，
因此原模块的初始随机状态与原 DBT 相同（check: 107 个非 encoder 公共
tensor，max abs diff `0.0`）。

| item | value |
|---|---:|
| inference parameters (all) | **457,443** |
| frozen local stack | 202,048 |
| trainable (adapters + DBT downstream) | 255,395 |
| adapters | 3 × 1,640 = 4,920 |
| cap | 500,000 (headroom 42,557) |

冻结栈不在 optimizer 中，`requires_grad=False`，且 `model.train()` 后仍被强制
`eval()`；训练数步前后 encoder hash 完全不变，适配器和 DBT trunk 的 hash
改变。适配器的 `D` 在第一步梯度恰好为零（`U=0` 的预期行为），第 2/3 步
非零；`U` 第一步非零。零初始化适配器时，输出与“同一 SSL 基座 + 同一随机
DBT 下游、无适配器”逐元素相同（max diff `0.0`），不是拿监督 DBT 输出比较。

## 2. 冻结特征缓存

缓存只存基座最终输出 `a0/b0/w0`（float32，逐图），不含标签、不含适配后
状态。key 包含 `dbt-pa-feat-v1`、SSL sha256、encoder state sha256、图结构
fingerprint。train/valid 缓存与直接基座前向在抽样图上一致：

| split | a max diff | b max diff | w max diff | pass |
|---|---:|---:|---:|---|
| train (8 rows) | 0.0 | 0.0 | 1.907e-06 | yes |
| valid (4 rows) | 0.0 | 5.960e-07 | 1.431e-06 | yes |
| test (4 rows, after freeze) | — | — | — | yes |

缓存构建本身只用 **0.509 s (train) + 0.044 s (valid)**；prep 墙钟 15 s，
其中包含首次构建缺失的 compact train object cache。test 缓存在 best-valid
freeze 之后才构建，并再次通过抽样核验。

## 3. 时间测量与冻结 recipe

先做 4 个完整 epoch 的计时（含全量 valid 前向和真实数据装载）：

| item | value |
|---|---:|
| steady epoch (epoch 2–4 mean) | 8.4867 s |
| safety estimate (×1.15) | 9.7597 s |
| spent before formal (wall) | 334 s |
| final-scoring reserve | 180 s |
| **E** | **310** |
| steps/epoch | 157 |
| **total updates** | **48,670** |

公式：`E = min(600, floor((3600 − 334 − 180) / 9.7597))`，向下取整到 10。
E 只由速度决定，没有看计时阶段的 valid MAE。正式 run 重置模型/optimizer/RNG，
cosine horizon 固定为 `310×157 = 48,670`，未重置、未改 horizon。

实际时间：

| phase | seconds | note |
|---|---:|---|
| prep | 15 | feature cache + compact object cache + checks |
| timing | 49.3 wall / 34.6 train loop | 4 epochs |
| formal training | **2,612.4** | 0.7257 GPU-h; 310/310 epochs |
| terminal test | 10 | freeze + test cache + score |
| round wall (GPU jobs) | **≈3,144** | 52.4 min, under 60 min |

缓存并没有把训练变成高速：稳态 `8.49 s/epoch` 仅比原 DBT 的 `8.85 s/epoch`
快约 4%。冻结后省掉的 encoder 前向/反向很小，下游 DBT、关系 gather 和
valid 前向仍占主导；60 分钟预算因此只允许 `E=310`，不是靠提速把日程拉到
500+。这是本轮的实际计算—表征权衡，不能把“训练更快”当作泛化结果。

## 4. 主结果

Valid 是 official ZINC valid，raw-y MAE；test 是已经暴露的 official test。
DBT-PA 的 checkpoint 在 test 之前 freeze（sha256 `7314b092...`）。

| model | params | best valid | best ep | eval-train@best | test MAE | updates | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| P3B_s0 (reused) | 413,505 | 0.076223754 | — | — | **0.069864841** | — | reused | — |
| L_s0 (reused) | 494,193 | 0.078537152 | — | — | **0.066266555** | — | reused | — |
| DBT_s0 (reused) | 452,523 | 0.077768402 | 949 | 0.015964 | 0.069401494 | 235,500 | reused | 3.6918 |
| **DBT-PA_s0 (this round)** | **457,443** | **0.140224125** | **287** | 0.028934 | **0.107658585** | **48,670** | complete | **0.7257** |

Paired descriptive contrasts (Δ = AE(DBT-PA) − AE(ref); negative = PA better;
10,000 bootstrap resamples over molecules):

| split | ref | mean ΔAE | 95% CI | PA better on |
|---|---|---:|---|---:|
| valid | P3B_s0 | +0.064000 | [+0.037800, +0.108172] | 31.3% |
| valid | L_s0 | +0.061687 | [+0.040586, +0.096398] | 30.3% |
| valid | DBT_s0 | +0.062456 | [+0.038230, +0.096117] | 30.9% |
| test | P3B_s0 | +0.037794 | [+0.030885, +0.044864] | 30.2% |
| test | L_s0 | +0.041392 | [+0.035159, +0.047689] | 30.5% |
| test | DBT_s0 | +0.038257 | [+0.031197, +0.045735] | 32.1% |

所有 CI 都排除 0，方向一致为更差。新模型 test 比 valid 低
`0.03257`；参照的 test−valid 分别是 P3B `−0.00636`、L `−0.01227`、
DBT `−0.00837`。test 更容易，但没有改变结论。

## 5. 训练曲线与近期 valid

* best-valid `0.140224 @287`；final valid `0.141026`；final eval-train `0.033049`。
* 最近 20 epoch valid mean `0.141182`，min `0.140452`；最近 50 epoch mean
  `0.141357`；best 附近 ±20 epoch mean `0.141228`。best 不是孤立噪声点，
  曲线在 `~0.141` 平台。
* eval-train@best `0.028934` 明显高于参照 DBT 的 `0.015964`（不同日程，仅
  描述），说明这个受限模型同时 **训练拟合更低 + valid 更差**，不是单纯
  过拟合；更像是冻结基座 + 短日程下的欠拟合/优化受限。

历史 DBT 参考（**cosine 不同，不是严格因果对照**）：

| reference | value |
|---|---:|
| DBT valid at same epoch 310 | 0.101307 |
| DBT best valid within first 310 epochs | 0.085079 @299 |
| DBT last-20 valid mean within first 310 | 0.094714 |
| DBT-PA − DBT at same epoch | +0.039719 |
| DBT-PA best − DBT first-310 best | +0.055145 |

## 6. 固定四组误差

分组定义：radius-2 content key 相对于 train；`signature_unseen` = 至少一个
原子签名在 train 中未出现；tail = `y <= -3`。标签分组只用于分析，不进入
模型、采样或推理门控。row 172 保留在所有主数字中；去 row 172 只作敏感性
描述（PA valid `0.120082`，仍远差于参照 `0.0707–0.0747`）。

完整逐组 `n / MAE / sum AE / 总误差份额 / DBT 与 PA 差值` 见
`analysis/groups.md`；摘要：

| split | group | n | P3B | L | DBT | PA | PA−DBT |
|---|---|---:|---:|---:|---:|---:|---:|
| valid | covered_ordinary | 710 | 0.047468 | 0.044864 | 0.042996 | 0.078503 | +0.035507 |
| valid | covered_tail | 39 | 0.116284 | 0.225481 | 0.258636 | 0.784161 | +0.525525 |
| valid | unseen_ordinary | 225 | 0.133940 | 0.130773 | 0.141320 | 0.182115 | +0.040795 |
| valid | unseen_tail | 26 | 0.301922 | 0.325623 | 0.206050 | 0.497270 | +0.291219 |
| test | covered_ordinary | 705 | 0.047252 | 0.043744 | 0.048998 | 0.081736 | +0.032739 |
| test | covered_tail | 51 | 0.128495 | 0.076441 | 0.086733 | 0.175218 | +0.088485 |
| test | unseen_ordinary | 218 | 0.114921 | 0.114230 | 0.118929 | 0.154758 | +0.035829 |
| test | unseen_tail | 26 | 0.190233 | 0.254859 | 0.173397 | 0.283117 | +0.109720 |

PA 在 valid/test 的四个组里全部更差，包括 `covered_ordinary`（最大组）；
因此这不是只由 row 172、tail 或 unseen 子群造成的局部现象。`covered_tail`
的 valid 大差距里 row 172 贡献很大（PA AE `20.262`），但去掉它整体仍然
输 `0.045–0.049` MAE；test 的 row 172 是 ordinary 且 PA 也只差 `0.000024`
地影响总体，不能解释主结果。

## 7. 解释：哪些是证据，哪些还不能区分

**证据支持的：**

* 在本轮单 seed、310 epoch、固定 cosine、冻结全部局部权重的约束下，
  “保留 SSL 局部表示、只做后置 rank-8 监督适配”没有改善单模型泛化；
  它在同 epoch 和成熟参照上都明显更差，且四个固定组一致更差。
* 这个受限模型不是简单的过拟合：train@best 比 DBT 成熟 best 更高，valid
  却更差；更像是冻结基座使下游无法获得足够可分的局部特征，短日程又不足以
  补偿。
* 适配器确实被训练了（hash 改变、`D` 在 `U` 移动后获得梯度），但 4,920
  个参数没有把冻结基座“救回来”。
* 本轮没有产生任何 0.05x 证据；test `0.107659` 距离目标还很远。

**还不能区分的：**

* 更长的训练是否会让 DBT-PA 接近 DBT。当前曲线在 `0.141` 平台、cosine
  已到 `3e-6`，继续同 recipe 的收益预计很小；但这仍是外推。
* 失败应归因于“冻结”本身、适配器位置/容量/学习率，还是短日程优化，单
  seed + 单 recipe 不能分离。没有做 adapter-off ablation，也没有 rank/
  placement 网格，因此不能把结果写成“保留 SSL 表示必然有害”。
* 原 DBT 的同 epoch 参考使用不同 cosine（1,500-epoch schedule），不是严格
  的适配器因果对照；它能说明“同预算下这个候选落后”，不能单独估计冻结的
  因果效应。
* official test 已在仓库中多轮暴露；本轮是冻结 checkpoint 的描述性复读，
  不是新的盲测。

## 8. 下一决定

**停止。** 不买 seed 1、不做 rank 网格、不加第二种 adapter、不改输出分解、
不把 310 epoch 延长到数小时来“收口”。本轮的负结果足以说明这个具体受限
候选在当前时间预算下不值得继续。

若未来还要碰这个方向，下一笔时间应先回答一个更窄的优化问题：冻结基座
是否让下游欠拟合（例如 adapter 容量/位置或 adapter 学习率），而不是直接
再跑更长日程。那需要新的 pre-registration，不能把本轮结果包装成已确诊的
泛化机制。

## 9. 复现

实现 commit：`3855dea1b693`（formal train + test）；分析脚本另在本轮
`code/` 中。关键命令：

```bash
# 1. model + cache checks (GPU)
rr run res pa-prep --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/cache \
  -- python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py prep --device cuda:0

# 2. timing (4 epochs)
rr run res pa-timing --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/runs \
  -- python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py timing --device cuda:0 --epochs 4

# 3. E from measured speed
uv run python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py plan --spent-seconds 334

# 4. formal short run (E=310, 48,670 updates)
rr run res pa-train-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/runs \
  -- python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py train --device cuda:0 --epochs 310

# 5. freeze + test
rr run res pa-test --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis \
  --result tracks/structural-sufficiency/zinc_restricted_adaptation_v1/runs \
  -- python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py test --device cuda:0

# 6. local tables / curve
uv run python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py groups
uv run python tracks/structural-sufficiency/zinc_restricted_adaptation_v1/code/dbt_pa_runner.py analysis
```

## 10. Artifacts and hashes

| artifact | path | sha256 |
|---|---|---|
| best checkpoint | `runs/pa_s0/best.pt` | `7314b0928aa987f592986abcb06fafdddc39c07d15900f4c3de0be2247a90b1d` |
| final checkpoint | `runs/pa_s0/final.pt` | `dc7aafe832dfd0dc252ff4358e7cbf3882a3ad1ffb00a144d5406d9b0057bea7` |
| last/recovery | `runs/pa_s0/last.pt` | `3276c62631e2f248fcf8861cc43c717bda3bd294a7b003a266dbcd76c543a597` |
| valid predictions (best) | `runs/pa_s0/predictions/valid_best.npz` | `6d160fb5ec14420aa5789f40687c3d9087a9ee620f7aee260920186fece8a023` |
| test predictions (best) | `runs/pa_s0/predictions/test_best.npz` | `aaeb4c775788d6f3e172f467bd1e2cc5b2cd8358ab596e3fec79a2c38de736ee` |
| curve | `runs/pa_s0/curve.csv` | `36380dd16d22bc02a56efe949d91fea26685c2ef5b881e515c152b47930d2386` |
| SSL base | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt` | `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97` |
| encoder state | 457,443-param model | `86a3c2a26c1b721acaa829b82202f2c69f25531995514c82ad8a8f947a1da759` |
| train feature cache | `cache/dbt-pa-feat-v1-006fabc11e-86a3c2a26c-f7a58b6684_train.pt` | `6596e6186b03d698b9a26ce7bc258c66e776902bd09e96edb7d08f9efe0c2994` |
| valid feature cache | `cache/dbt-pa-feat-v1-006fabc11e-86a3c2a26c-d764aa738f_valid.pt` | `ab94d8acf92ff570b416f4581699d747b0bef50fd905d78a981d39a1ebdbc277` |

`runs/` and `cache/` are git-ignored (local/remote artifacts); `analysis/`
contains the frozen JSON/CSV/PNG and the full group table. The test freeze was
written before any test label was read; the test feature cache was built after
the freeze.

## 11. Non-claims

* Single seed, single short schedule, single adapter rank; not a training-seed
  replication and not a final architecture verdict.
* The official test split is historically exposed; this is not a blind test.
* The same-epoch DBT curve uses a different cosine schedule; it is a short-run
  progress reference, not a causal estimate of freezing.
* The four-group label split is analysis-only; no routing or gating is proposed.
* `0.140224` / `0.107659` are not SOTA, not a 0.05x result, and not evidence
  that preserving pretrained representations is generally harmful.
