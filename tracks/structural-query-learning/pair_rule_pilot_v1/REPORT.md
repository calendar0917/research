# Lipo 显式结构关系小试验｜sq-lipo-pair-rule-v1

官方 OGB `ogbg-mollipo`，官方 scaffold Train/Val/Test = 3360/420/420，单训练 seed=0，180 epoch，`test_policy: always`（Test 为本轮 DEVELOPMENT_TEST）。本文件由 `stage=aggregate` 从已拉回的冻结产物自动生成；所有数字均为 raw logD 单位。所有模型状态只由 Val RMSE 选择，Test 仅作为 DEVELOPMENT_TEST 随同一状态报告，不参与任何选择。

## 运行与协议

- 训练代码 commit：`27a25b4bf83fa6857159c2ee194c7723316264b8`；聚合代码 commit：`18d0024ef40ab8a1104e334344fcd5c5992bee7b`
- 协议：`sq-lipo-pair-rule-v1`，protocol hash `970bd76e274b87876fd509955e6f3d632ab7de56591239e205f0c5e37d8862ed`
- rr run：prepare `sq-pair-prep-v2-20261118-102303-ed36fb05`，C1 `sq-pair-c1-v2-20261118-102436-331ce910`，rules `sq-pair-rules-v2-20261118-102436-fde5d663`
- 设备：C1 GPU0 `cuda:0`；MLP GPU1 `cuda:0`；三条加性臂 CPU（BLAS ≤4 线程）
- 冻结缓存：graph `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c`；lipo raw `3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3`
- 规则规模：B=203，S2=247，真实关系槽=512，共同列=1136，扩展列=1024

## 运行状态

| stage | rr run | state | node | CUDA_VISIBLE_DEVICES | wall s |
| --- | --- | --- | --- | --- | --- |
| prepare | sq-pair-prep-v2-20261118-102303-ed36fb05 | completed | a100-2 | 1 | 42.7804 |
| worker_c1 | sq-pair-c1-v2-20261118-102436-331ce910 | completed | a100-2 | 0 | 344.1043 |
| worker_rules | sq-pair-rules-v2-20261118-102436-fde5d663 | completed | a100-2 | 1 | 116.2243 |

三个远端阶段均通过 rr 进程后端在 `res` 的 A100 上执行；prepare/rules 使用 GPU1，C1 使用 GPU0，两条 worker 并行。

## 预定结论

- 关系轴：**UNRESOLVED_BINDING**
- 性能轴：**NOT_NEAR_C1**
- `PROMISING_PILOT`：**no**
- 瓶颈线索：**COMPRESSION** — same weights without the 32-group budget lower Val RMSE by at least 0.02

关系轴的关键数：Val 覆盖 420/420；REAL-32 在 Val 的活跃关系贡献最大绝对值 4.1830 logD；Val gain vs BASE-32 = 0.0153，vs NULL-32 = 0.0471；Test gain vs BASE-32 = 0.0273，vs NULL-32 = 0.0476。
性能轴：REAL-32 相对 live C1 的 gap：Val 0.3200，Test 0.2737 logD；规则重放通过 = yes。

就数字本身：REAL-32 的 Val 比 BASE-32 好 0.0153（95% CI [-0.0321, 0.0670]），比 NULL-32 好 0.0471（[0.0093, 0.0849]）；Test 对应 gain 为 0.0273 与 0.0476。也就是说，实际锚点对计数相对**边际 NULL**有可辨认的增量，但相对共同输入 BASE 的增量没有达到预定的 0.02、其 Val 区间也跨 0，因此关系轴只能记为 `UNRESOLVED_BINDING`，不能记为 `BINDING_SIGNAL`。
性能轴没有接近 live C1：REAL FULL 的 Val 0.7952 / Test 0.8673，MLP 的 Val 0.7793 / Test 0.7740，C1 的 Val 0.7126 / Test 0.7164；live C1 与历史 0.7052 的差为 0.0074，未触发 `WEAK_LIVE_REFERENCE`。

## 预检

- `preflight.json` pass = **yes**，失败项 0。
- 行/划分：4200 行覆盖 0…4199，三划分互斥，标签逐项一致，Train-only 标准化往返最大误差 0.00000000。
- 全图计数：4200 张图全部满足 `Σ_d H_d = T`、`Σ_t C_t = T`、有限 d≥3 表与距离边际一致；最大误差 0。
- 重标号不变性：8 个官方 Train 行全部通过；18 环节点正对照通过；NULL `T=4,H_d=2,C_t=2` 两列为 `5/6, 1/6`；组近端与 FULL/-32 偏置复算通过。

## 主表（各臂最小 Val RMSE 状态）

| arm/readout | epoch | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | params / active groups | fit s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE-FULL | 180 | 0.6613 | 0.8058 | 0.8831 | 0.6246 | 0.6852 | 268/568 groups | 19.8231 |
| BASE-32 | 180 | 0.9775 | 1.0480 | 1.0174 | 0.8335 | 0.8248 | 32/568 groups | 19.8231 |
| NULL-FULL | 71 | 0.6599 | 0.8529 | 0.8619 | 0.6412 | 0.6823 | 729/1080 groups | 22.4242 |
| NULL-32 | 180 | 1.0087 | 1.0797 | 1.0376 | 0.8640 | 0.8378 | 32/1080 groups | 22.4242 |
| REAL-FULL | 158 | 0.5703 | 0.7952 | 0.8673 | 0.6250 | 0.6664 | 471/1080 groups | 31.0019 |
| REAL-32 | 173 | 0.9575 | 1.0326 | 0.9901 | 0.8267 | 0.7999 | 32/1080 groups | 31.0019 |
| MLP | 57 | 0.1437 | 0.7793 | 0.7740 | 0.5915 | 0.5839 | 284929 | 37.5796 |
| C1 | 106 | 0.2684 | 0.7126 | 0.7164 | 0.4918 | 0.5151 | 228674 | 328.1614 |

## 固定 epoch 180 表

| arm/readout | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | params / active groups |
| --- | --- | --- | --- | --- | --- | --- |
| BASE-FULL | 0.6613 | 0.8058 | 0.8831 | 0.6246 | 0.6852 | 268/568 groups |
| BASE-32 | 0.9775 | 1.0480 | 1.0174 | 0.8335 | 0.8248 | 32/568 groups |
| NULL-FULL | 0.6367 | 0.8740 | 0.8439 | 0.6289 | 0.6655 | 729/1080 groups |
| NULL-32 | 1.0087 | 1.0797 | 1.0376 | 0.8640 | 0.8378 | 32/1080 groups |
| REAL-FULL | 0.5698 | 0.7955 | 0.8685 | 0.6262 | 0.6670 | 471/1080 groups |
| REAL-32 | 0.9589 | 1.0408 | 0.9826 | 0.8349 | 0.7962 | 32/1080 groups |
| MLP | 0.0469 | 0.7842 | 0.7694 | 0.5995 | 0.5851 | 284929 |
| C1 | 0.2132 | 0.7292 | 0.7429 | 0.5025 | 0.5280 | 228674 |

## 配对 cluster bootstrap（2000 次，seed 1703）

| split | state | comparison | delta (reference−focal) | CI95 low | CI95 high | clusters | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid | best | REAL-32_vs_BASE-32 | 0.0153 | -0.0321 | 0.0670 | 420 | 420 |
| valid | best | REAL-32_vs_NULL-32 | 0.0471 | 0.0093 | 0.0849 | 420 | 420 |
| valid | best | REAL-32_vs_C1 | -0.3200 | -0.4361 | -0.1992 | 420 | 420 |
| valid | best | REAL-FULL_vs_MLP | -0.0160 | -0.0661 | 0.0384 | 420 | 420 |
| valid | epoch180 | REAL-32_vs_BASE-32 | 0.0071 | -0.0388 | 0.0547 | 420 | 420 |
| valid | epoch180 | REAL-32_vs_NULL-32 | 0.0389 | 0.0014 | 0.0762 | 420 | 420 |
| valid | epoch180 | REAL-32_vs_C1 | -0.3116 | -0.4294 | -0.1754 | 420 | 420 |
| valid | epoch180 | REAL-FULL_vs_MLP | -0.0113 | -0.0572 | 0.0384 | 420 | 420 |
| test | best | REAL-32_vs_BASE-32 | 0.0273 | -0.0110 | 0.0644 | 419 | 420 |
| test | best | REAL-32_vs_NULL-32 | 0.0476 | 0.0092 | 0.0859 | 419 | 420 |
| test | best | REAL-32_vs_C1 | -0.2737 | -0.3680 | -0.1846 | 419 | 420 |
| test | best | REAL-FULL_vs_MLP | -0.0933 | -0.1567 | -0.0327 | 419 | 420 |
| test | epoch180 | REAL-32_vs_BASE-32 | 0.0348 | -0.0025 | 0.0719 | 419 | 420 |
| test | epoch180 | REAL-32_vs_NULL-32 | 0.0550 | 0.0172 | 0.0945 | 419 | 420 |
| test | epoch180 | REAL-32_vs_C1 | -0.2397 | -0.3388 | -0.1445 | 419 | 420 |
| test | epoch180 | REAL-FULL_vs_MLP | -0.0991 | -0.1639 | -0.0398 | 419 | 420 |

`delta = RMSE(reference) - RMSE(focal)`；对 REAL-32 的三行，正数表示 REAL-32 更好；`REAL-FULL_vs_MLP` 正数表示 REAL FULL 更好。

Bemis–Murcko 簇：valid 420 簇 / 420 行，解析失败 0；test 419 簇 / 420 行，解析失败 0。

## 规则复核与解释

- REAL-32 导出 `rules.json`：epoch 173，32 个活跃规则组；model hash `9f4c65364d1823af…`。
- 官方指标从保存的 raw 预测独立重算：48 组，最大 RMSE 绝对差 0.000000000000，最大 MAE 绝对差 0.000000000000，行 ID 与标签逐项一致 → **yes**。
- 独立规则重放（从原图重算共同项与关系项）：Val 最大绝对误差 0.00000000，Test 0.00000000；容差 1e-6 → **yes**。
- 单项减一检查（Val 最小官方 row ID）：row 950，规则 `REL::TP|5,5,1,1,0|7,5,0,0,0|4`，K 1→0；预测差 0.09534682，导出函数单项差 0.09534682，最大误差 0.0000000000 → **yes**。

| split | 有入库关系命中的分子比例 | REAL-32 关系项非零的分子比例 | 非零关系规则数中位数 | 非零规则总数中位数 | 关系项 \|贡献\| 和中位数 |
| --- | --- | --- | --- | --- | --- |
| valid | 1.0000 | 0.9095 | 3.0000 | 8.0000 | 0.5136 |
| test | 1.0000 | 0.9095 | 3.0000 | 8.0000 | 0.4822 |

升序前 20 个 Val 分子的逐规则解释写入 `analysis/explanation_cases.jsonl`；样例计数、锚点对 / S2 路径 / 根 ID 都在该文件中。

## 瓶颈线索

| comparison | FULL / reference Val RMSE | -32 / focal Val RMSE | difference |
| --- | --- | --- | --- |
| 同权重 FULL vs -32（REAL-32 选中 epoch） | 0.7959 | 1.0326 | 0.2367 |
| 各自选中状态 REAL FULL / REAL-32 | 0.7952 | 1.0326 | 0.2374 |

- REAL FULL vs MLP：Val 差（FULL−MLP）0.0160，MLP 相对 C1 的 gap：Val 0.0666，Test 0.0576。
- REAL FULL 相对 C1：Val gap 0.0826；REAL-32 相对 C1：Val gap 0.3200。

## 历史背景（不参与本轮判定）

| 对象 | 旧轮 Val RMSE | 旧轮 Test RMSE |
| --- | --- | --- |
| `sq-pilot-small-v1` Lipo A | 0.7911 | 0.7525 |
| `sq-pilot-small-v1` Lipo C1 | 0.7052 | 0.7139 |
| `sq-pilot-small-v1` Lipo GINE | 0.6091 | 0.6456 |

本轮门槛使用**本次现场重训的 C1**，不是 0.7052。历史数字只说明旧轮的量级；若本轮 C1 明显弱于旧轮，已在预定结论中附记 `WEAK_LIVE_REFERENCE`。

## 资源账本

| item | seconds |
| --- | --- |
| prepare wall | 42.7804 |
| C1 worker wall | 344.1043 |
| rules worker wall | 116.2243 |
| aggregate wall (local) | 3.3530 |
| remote wall (first prep start → last worker finish) | 438.7376 |
| GPU card reservation (prepare + C1 + rules) | 503.1090 |
| GPU hours reservation | 0.1398 |
| CPU additive solve s | 73.2492 |

预算：60 分钟远端墙钟 / 2 A100·小时；本轮远端墙钟 7.31 分钟，占卡 0.140 A100·h。CPU 求解时间单独列出。

## 产物与证据

- `analysis/summary.json`：本轮结构化摘要、两轴判定、全部数字与运行 provenance。
- `analysis/rules.json`：REAL-32 最佳 Val 状态的 32 个规则组、尺度、raw 系数与 bias；`model_hash`/`candidate_hash` 可追溯。
- `analysis/replay_audit.json`：独立重放的最大绝对误差与单项减一检查。
- `analysis/metric_recheck.json`：48 个 model/state/split 组合的官方指标重算与记录值差异。
- `analysis/bootstrap.csv`：2000 次配对 cluster bootstrap 的全部比较。
- `analysis/explanation_cases.jsonl`：升序前 20 个 Val 分子的 top-3 逐规则贡献与支持对象。
- `analysis/resource_ledger.json`：阶段墙钟、占卡时间、CPU 求解时间。
- 远端原始产物：`.rr/pulled/<run_id>/result/tracks/structural-query-learning/pair_rule_pilot_v1/runs/{prep,c1,rules}`。

## 不确定性与边界

- 单一训练 seed=0；本轮不提供多 seed 稳定性结论。MLP 与 C1 的容量/优化器差异保留在表中，不能把 MLP 的差异单独归因于读出。
- bootstrap 的对象是当前固定预测在留出分子上的不确定性（按 Bemis–Murcko 簇重抽），不是重新训练模型的不确定性。
- 关系轴只在当前 512 个 Train 过滤后的槽位、180 epoch、该 lambda 和该 32 项预算下成立；词表外的结构关系没有被检验。
- 结论限于本轮求解条件与单种子范围；如果性能轴为 `NOT_NEAR_C1`，不能外推为“显式关系无用”，只能说明当前读出/预算/词表组合未达到 0.04 logD 门槛。

