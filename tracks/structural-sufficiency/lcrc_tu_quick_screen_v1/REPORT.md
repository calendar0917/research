# LCRC TU quick screen v1 — report

**Result label**: `EXPLORATORY_SINGLE_SEED_RANDOM_811`  
**Round**: `lcrc-tu-quick-screen-v1`  
**Status**: 12/12 formal runs completed, all integrity checks passed  
**Code commit**: `52a94d5591d27e1b5edd09e7915eb23865c06d7a`  
**Compute**: `res` (process backend, 2×NVIDIA A100-SXM4-40GB), driver 550.163.01, torch 2.5.1+cu124 (cuda 12.4), cuDNN 90100  

单 seed（`training_seed=0`）、单固定 8:1:1 分层划分（`split_seed=0`）的探索性筛查。
12 个正式 run = 4 数据集 × 3 模型，全部完成；没有多 seed、交叉验证、调参、消融、早停、
EMA/SWA/soup、增强、SSL 或额外 loss。

## 1. 结果（Test 只在 checkpoint 哈希写盘后评估一次）

| dataset | model | params | best epoch | Train acc | Valid acc | Valid CE | Test acc | Test macro-F1 | Test CE | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| NCI1 | LCRC-FULL | 229,544 | 93 | 0.9951 | 0.8467 | 1.2366 | 0.8248 | 0.8247 | 1.3458 | 442 |
| NCI1 | LCRC-C1 (no b_mean) | 211,112 | 86 | 0.9891 | 0.8394 | 1.1721 | 0.8540 | 0.8539 | 1.1271 | 521 |
| NCI1 | GIN+global | 98,055 | 93 | 0.9824 | 0.8467 | 0.7086 | 0.8297 | 0.8297 | 0.7348 | 373 |
| PROTEINS | LCRC-FULL | 210,404 | 24 | 0.7539 | 0.8018 | 0.5240 | 0.7679 | 0.7602 | 0.4570 | 159 |
| PROTEINS | LCRC-C1 (no b_mean) | 191,972 | 59 | 0.7955 | 0.7928 | 0.4892 | 0.7679 | 0.7524 | 0.4543 | 176 |
| PROTEINS | GIN+global | 95,943 | 3 | 0.7438 | 0.8198 | 0.5867 | 0.7232 | 0.7003 | 0.5601 | 112 |
| IMDB-BINARY | LCRC-FULL | 218,524 | 10 | 0.7887 | 0.7000 | 0.6469 | 0.6700 | 0.6643 | 0.6643 | 146 |
| IMDB-BINARY | LCRC-C1 (no b_mean) | 200,092 | 38 | 0.8612 | 0.7300 | 0.7788 | 0.7100 | 0.7064 | 0.9143 | 146 |
| IMDB-BINARY | GIN+global | 96,839 | 84 | 0.8100 | 0.7500 | 0.5130 | 0.6800 | 0.6779 | 0.6068 | 168 |
| IMDB-MULTI | LCRC-FULL | 218,654 | 19 | 0.5708 | 0.5333 | 0.9579 | 0.5200 | 0.5064 | 1.0240 | 169 |
| IMDB-MULTI | LCRC-C1 (no b_mean) | 200,222 | 68 | 0.6117 | 0.5400 | 1.0777 | 0.4933 | 0.4737 | 1.4017 | 186 |
| IMDB-MULTI | GIN+global | 96,904 | 96 | 0.6058 | 0.5267 | 1.0889 | 0.5267 | 0.5061 | 1.0399 | 153 |

`wall s` 是 `rr` 进程内计时（预处理读缓存 + 200 epoch 训练 + 每 epoch Valid 评估 + 一次 Test）。
所有 run 都跑满 200 epoch（`epochs_run=200`），没有早停。
12 个 run 合计 wall 45.9 min。

### 1.1 Test accuracy 与配对差值

| dataset | n_test | majority | FULL | C1 | GIN | FULL−C1 | FULL−GIN | C1−GIN |
|---|---|---|---|---|---|---|---|---|
| NCI1 | 411 | 0.4988 | 0.8248 | 0.8540 | 0.8297 | -0.0292 | -0.0049 | +0.0243 |
| PROTEINS | 112 | 0.5982 | 0.7679 | 0.7679 | 0.7232 | +0.0000 | +0.0446 | +0.0446 |
| IMDB-BINARY | 100 | 0.5000 | 0.6700 | 0.7100 | 0.6800 | -0.0400 | -0.0100 | +0.0300 |
| IMDB-MULTI | 150 | 0.3333 | 0.5200 | 0.4933 | 0.5267 | +0.0267 | -0.0067 | -0.0333 |

### 1.2 Test macro-F1

| dataset | FULL | C1 | GIN |
|---|---|---|---|
| NCI1 | 0.8247 | 0.8539 | 0.8297 |
| PROTEINS | 0.7602 | 0.7524 | 0.7003 |
| IMDB-BINARY | 0.6643 | 0.7064 | 0.6779 |
| IMDB-MULTI | 0.5064 | 0.4737 | 0.5061 |

### 1.3 Test confusion matrices

* **NCI1**（行 = 真类，列 = 预测）
  * FULL: [[164, 42], [30, 175]]
  * C1: [[169, 37], [23, 182]]
  * GIN: [[172, 34], [36, 169]]
* **PROTEINS**（行 = 真类，列 = 预测）
  * FULL: [[53, 14], [12, 33]]
  * C1: [[57, 10], [16, 29]]
  * GIN: [[56, 11], [20, 25]]
* **IMDB-BINARY**（行 = 真类，列 = 预测）
  * FULL: [[40, 10], [23, 27]]
  * C1: [[30, 20], [9, 41]]
  * GIN: [[30, 20], [12, 38]]
* **IMDB-MULTI**（行 = 真类，列 = 预测）
  * FULL: [[15, 17, 18], [8, 30, 12], [5, 12, 33]]
  * C1: [[12, 20, 18], [5, 33, 12], [8, 13, 29]]
  * GIN: [[13, 20, 17], [4, 35, 11], [6, 13, 31]]

### 1.4 选定 epoch 与选择轨迹

| dataset | model | best epoch | Valid acc @best | Valid CE @best | Train acc @best | Train CE @best |
|---|---|---|---|---|---|---|
| NCI1 | FULL | 93 | 0.8467 | 1.2366 | 0.9951 | 0.0162 |
| NCI1 | C1 | 86 | 0.8394 | 1.1721 | 0.9891 | 0.0316 |
| NCI1 | GIN | 93 | 0.8467 | 0.7086 | 0.9824 | 0.0584 |
| PROTEINS | FULL | 24 | 0.8018 | 0.5240 | 0.7539 | 0.4986 |
| PROTEINS | C1 | 59 | 0.7928 | 0.4892 | 0.7955 | 0.4269 |
| PROTEINS | GIN | 3 | 0.8198 | 0.5867 | 0.7438 | 0.6251 |
| IMDB-BINARY | FULL | 10 | 0.7000 | 0.6469 | 0.7887 | 0.4488 |
| IMDB-BINARY | C1 | 38 | 0.7300 | 0.7788 | 0.8612 | 0.2603 |
| IMDB-BINARY | GIN | 84 | 0.7500 | 0.5130 | 0.8100 | 0.3446 |
| IMDB-MULTI | FULL | 19 | 0.5333 | 0.9579 | 0.5708 | 0.8853 |
| IMDB-MULTI | C1 | 68 | 0.5400 | 1.0777 | 0.6117 | 0.7844 |
| IMDB-MULTI | GIN | 96 | 0.5267 | 1.0889 | 0.6058 | 0.8064 |

## 2. 正确性验证（都在正式 run 之前完成）

* 单元测试 9 项通过（CPU、无数据）：特征构造、清洗计数、划分、共享初始化、异构 batch、GIN 传播。
* 结构 gates：`analysis/gates.json`，`all_passed=True`，14 项，包含
  划分互斥/完备、Train-only 归一化审计、手算链/三角/孤立点/不连通图/未见标签→UNK/度数桶、
  节点重编号不变性（特征/global/logits）、batch 拆分-合并一致、FULL/C1 共享初始化
  （非切片张量逐位相等、恰好删除 96 个输入列、参数差 18,432、FULL 的 b_mean 置零 ≡ C1 logits）、
  FULL/C1/GIN 每个可训练张量梯度可达 + S2 content gate。
* CUDA smoke：`analysis/smoke_cuda.json`，`all_passed=True`，device `cuda:0`，3.9s；三模型前向/反向/保存/重载 logits 一致，
  未读 Test。
### 2.1 汇总阶段完整性校验（每个 run 都过）

`analysis/summary.json` 里 12/12 个 run 的 `all_ok=true`，每格 14 项：

* `best_epoch_matches_curve`
* `checkpoint_file_hash`
* `checkpoint_hash_matches_selection`
* `curve_rows_equal_epochs`
* `deterministic_seeded`
* `reload_valid_recheck`
* `seed_zero`
* `split_seed_zero`
* `status_completed`
* `test_evaluated_after_selection`
* `test_gids_unique_complete`
* `test_labels_aligned`
* `test_metrics_match_predictions`
* `test_prediction_rows`

其中 `test_metrics_match_predictions` 从保存的 Test logits 重算 accuracy / macro-F1 / CE，
与 `run.json` 报告值逐位或 1e-5 内一致；`checkpoint_file_hash` 与 `checkpoint_hash_matches_selection`
把 selection.json、checkpoint 文件、磁盘文件哈希三方对齐。

* 跨主机一致性：本地与 `res` 各自从官方 raw 文件重建，`feature_spec_hash`、`norm_fingerprint`、
  `partition_sha256`、清洗计数、节点类型数、S2 路径数全部逐项相同（见 §3 表）。

## 3. 数据与输入（官方 raw，无手工清洗）

| dataset | graphs | classes | 8:1:1 | node types K | atom_feat | S2 paths | directed→physical edges | duplicates merged | self-loops dropped |
|---|---|---|---|---|---|---|---|---|---|
| NCI1 | 4110 | 2 | 3288/411/411 | 37 | 66 | 383,116 | 265,506→132,753 | 132,753 | 0 |
| PROTEINS | 1113 | 2 | 890/111/112 | 4 | 33 | 499,732 | 162,088→81,044 | 81,044 | 0 |
| IMDB-BINARY | 1000 | 2 | 800/100/100 | 18 | 47 | 2,782,108 | 193,062→96,531 | 96,531 | 0 |
| IMDB-MULTI | 1500 | 3 | 1200/150/150 | 18 | 47 | 3,056,474 | 197,806→98,903 | 98,903 | 0 |

| dataset | feature_spec_hash | norm_fingerprint | partition_sha256 |
|---|---|---|---|
| NCI1 | `d13f9ed7eab6fff1…` | `0f2dfffb2f309065…` | `b8ecf9feee86ad1f…` |
| PROTEINS | `5c3e1cecc0523218…` | `77451b0024dbb0fa…` | `68d9490aa5e7a8bc…` |
| IMDB-BINARY | `808cd3f0bf13d63e…` | `71fee4c90485cce9…` | `b1cacabd5a2c4fef…` |
| IMDB-MULTI | `3ce8a5200952ee1e…` | `a3a6c66feba93a37…` | `624b69aee1a7cf19…` |

官方 raw 文件 sha256（`analysis/data_manifest.json` 的 `raw_manifest`）：

* `NCI1` ← `https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip`（raw_dir `data/TUD/NCI1/NCI1/raw`）
  * `NCI1_A.txt` 3,235,898 B `2c028eda59a5fe96e2265a63d9ac236901bd9a7dc658498be13cd704a703076d`
  * `NCI1_graph_indicator.txt` 585,267 B `ffe5acfa754b057d5ce0dc725c5aadb631628161886cc040d38c719635257dc5`
  * `NCI1_graph_labels.txt` 8,220 B `fae44681e70113f496256ae87eaac33b7836e3a97fe8e023933b84fde6a5c8a2`
  * `NCI1_node_labels.txt` 245,932 B `dc0a22b6cbff0c939de00373e823977ba32b358f560e4288d7602f7b51407ee2`
  * `README.txt` 2,477 B `1866d918409a22531ef00ff7b0f72200c3a06722342c23c5e3299cebb5cd8757`
* `PROTEINS` ← `https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip`（raw_dir `data/TUD/PROTEINS/PROTEINS/raw`）
  * `PROTEINS_A.txt` 2,022,990 B `4c4b33e272fc95cac6d27ed6d5d12b9a852c8610e91fff59f8f0dbdd5a20df67`
  * `PROTEINS_graph_indicator.txt` 169,673 B `8d4ff1e852a5c28561e9bffb44cf9490bc552b13a3b6156dd12dc7f44e9afc4a`
  * `PROTEINS_graph_labels.txt` 2,226 B `c0ef9810c440775252872318d0c8adc385588e14bc4c47a2c34aa96ee3c7cdcd`
  * `PROTEINS_node_labels.txt` 86,942 B `f758e8ad3d6e3c026b50833d75f6f90735b94eb3345e4243707f495ca8afea8d`
  * `PROTEINS_node_attributes.txt` 478,190 B `54eef829abb37e458a5fbf3a3c581e48e5810e220463c2cd951856b9ceb08f07`
  * `README.txt` 2,221 B `e9c4ed2c9a4059ddfdc2eb7330ebecfc6c8d42dabd5ce2ee67f28aed0db49da7`
* `IMDB-BINARY` ← `https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip`（raw_dir `data/TUD/IMDB-BINARY/IMDB-BINARY/raw`）
  * `IMDB-BINARY_A.txt` 4,596,012 B `23c8964bab82f81469271186d721131714a265d0ca8c44ae048118da1d0c4092`
  * `IMDB-BINARY_graph_indicator.txt` 76,924 B `415cc3ba25b8c647dcf34d3de034f675ee070616f3b803be939cb26773489f26`
  * `IMDB-BINARY_graph_labels.txt` 2,000 B `63a44411fd6342b9c4429828c16e144a2a11ef5244e25bb60f0ffa8b3ee2ffce`
* `IMDB-MULTI` ← `https://www.chrsmrrs.com/graphkerneldatasets/IMDB-MULTI.zip`（raw_dir `data/TUD/IMDB-MULTI/IMDB-MULTI/raw`）
  * `IMDB-MULTI_A.txt` 4,702,980 B `fb538b23758d78df199272301e57dc74ad5754d6c3d79d5ed0a343193ddf4ec6`
  * `IMDB-MULTI_graph_indicator.txt` 82,760 B `0274532a0e47dad9cdd21c08995c2265e43040a2c2a3b36c62fdb790994571d3`
  * `IMDB-MULTI_graph_labels.txt` 3,000 B `613c2357b6b996f6f04b91697a99da87774820f9d08bf22813992f4c7afda855`

## 4. 执行记录

所有正式 run 都在 `res` 上以 `rr` 启动，commit `52a94d5591d27e1b5edd09e7915eb23865c06d7a`，2 卡各跑一个任务（`CUDA_VISIBLE_DEVICES`
0/1 交替），任意时刻最多 2 个训练任务。完整 registry（含 prep/gates/smoke）见
`analysis/run_registry.json`。

| experiment | run_id | pool | CUDA_VISIBLE_DEVICES | state | exit |
|---|---|---|---|---|---|
| lcx-prep | `lcx-prep-20261102-200003-05d56632` | res-local | — | completed | 0 |
| lcx-gates | `lcx-gates-20261102-200345-b0171d02` | res-local | — | completed | 0 |
| lcx-smoke | `lcx-smoke-20261102-200547-dd6c8e2d` | res-gpu0 | 0 | completed | 0 |
| lcx-NCI1-FULL | `lcx-NCI1-FULL-20261102-200742-fa81f8be` | res-gpu0 | 0 | completed | 0 |
| lcx-NCI1-C1 | `lcx-NCI1-C1-20261102-200754-d9399d92` | res-gpu1 | 1 | completed | 0 |
| lcx-NCI1-GIN | `lcx-NCI1-GIN-20261102-202305-cd10b78e` | res-gpu0 | 0 | completed | 0 |
| lcx-PROTEINS-FULL | `lcx-PROTEINS-FULL-20261102-202326-21fa2d91` | res-gpu1 | 1 | completed | 0 |
| lcx-PROTEINS-C1 | `lcx-PROTEINS-C1-20261102-202854-64c96f2b` | res-gpu1 | 1 | completed | 0 |
| lcx-PROTEINS-GIN | `lcx-PROTEINS-GIN-20261102-203604-800b9b66` | res-gpu0 | 0 | completed | 0 |
| lcx-IMDB-BINARY-FULL | `lcx-IMDB-BINARY-FULL-20261102-203623-dd07c199` | res-gpu1 | 1 | completed | 0 |
| lcx-IMDB-BINARY-C1 | `lcx-IMDB-BINARY-C1-20261102-204217-6ccf7fc3` | res-gpu0 | 0 | completed | 0 |
| lcx-IMDB-BINARY-GIN | `lcx-IMDB-BINARY-GIN-20261102-204238-3fa22b9b` | res-gpu1 | 1 | completed | 0 |
| lcx-IMDB-MULTI-FULL | `lcx-IMDB-MULTI-FULL-20261102-204653-0bc4d122` | res-gpu0 | 0 | completed | 0 |
| lcx-IMDB-MULTI-C1 | `lcx-IMDB-MULTI-C1-20261102-204715-823e792a` | res-gpu1 | 1 | completed | 0 |
| lcx-IMDB-MULTI-GIN | `lcx-IMDB-MULTI-GIN-20261102-205226-a8013784` | res-gpu0 | 0 | completed | 0 |

代码状态：12 个 run 的 `run.json` 都记录 `git_commit=52a94d5591d27e1b5edd09e7915eb23865c06d7a`、`git_dirty=true`；
dirty 项只有当时已被 prep/gates 任务写过的证据文件：
* `M tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/analysis/gates.json`
* `?? tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/analysis/smoke_cuda.json`
即代码与 commit 一致，差异只在 `analysis/` 下的 JSON 证据（`gates.json` 被重写、`smoke_cuda.json` 新建），
没有任何 source/config/test 文件被修改。

### 4.1 参数量与初始化共享

| dataset | FULL params | C1 params | diff | FULL init hash | C1 init hash | C1 ← FULL init |
|---|---|---|---|---|---|---|
| NCI1 | 229,544 | 211,112 | 18,432 | `440f3bdb5b2c…` | `3ab2cf6060d3…` | `440f3bdb5b2c…` |
| PROTEINS | 210,404 | 191,972 | 18,432 | `3d291cb58385…` | `09adb31cbf7c…` | `3d291cb58385…` |
| IMDB-BINARY | 218,524 | 200,092 | 18,432 | `bcb248cbfeb3…` | `2cf727208cf4…` | `bcb248cbfeb3…` |
| IMDB-MULTI | 218,654 | 200,222 | 18,432 | `5f8e8ede3aaf…` | `1c184248bd6c…` | `5f8e8ede3aaf…` |

C1 的 `full_init_state_hash` 与同数据集 FULL 的 `init_state_hash` 相同，参数差恰为 `b_mean` 的
96×192 = 18,432 个权重。

## 5. 解释（范围内可说的话）

* 三个模型都明显高于 majority 基线；NCI1 0.82–0.85、PROTEINS 0.72–0.77、
  IMDB-BINARY 0.67–0.71、IMDB-MULTI 0.49–0.53。
* `FULL−C1`（Test acc）为 −0.029 / 0.000 / −0.040 / +0.027，均值 −0.011：本轮没有看到
  `b_mean` 块带来一致收益；两个 IMDB 上方向相反。这与该 block 在分子任务上的既有 dissection
  结论（C1 ≈ FULL）一致，但**本轮不是对该结论的检验**：单 seed、每格只有一个数。
* `C1−GIN` 为 +0.024 / +0.045 / +0.030 / −0.033，均值 +0.017：LCRC 主体与固定 GIN 参照在这个
  量级上互有胜负，没有形成稳定排序。
* Test 与 Valid 的排序并不总一致（例：IMDB-BINARY Valid 上 GIN 0.75 > C1 0.73 > FULL 0.70，
  Test 上 C1 0.71 > GIN 0.68 > FULL 0.67），说明 100/150 图的 Valid 选择本身带 ±0.04–0.05 的
  波动，单格差值小于约 0.05 时不应解读方向。
* NCI1 上三个模型的 Valid CE 在选定 epoch 都明显高于 Train CE（FULL 0.016 vs 1.237、
  C1 0.032 vs 1.172、GIN 0.058 vs 0.709），是 sum-readout + 200 epoch 无早停的过自信表现；
  它不影响本轮的 accuracy 排序，但说明这些 run 的 CE 不能当作校准质量来读。
* PROTEINS 的 GIN 在 epoch 3 就被选中（Valid 111 图，最差值 CE 0.587）：它的 5 层残差 GIN
  在这个小数据上很快过拟合，Valid 早期即达峰。该 run 仍按固定规则完整训练 200 epoch 并只取
  epoch 3 的权重，不是早停。
* `dropout_calls_observed` 只统计 LCRC 主干作用在 `h` 上的那一次 dropout（FULL/C1 每 step 一次，
  数目 = steps × 2 ≈ 2×total_updates）。GIN 的 dropout 在层内以 functional 形式调用，不计入该计数器；
  三个模型的 dropout 强度都是 0.10（GIN 每层一次），见 `recipe.dropout`。

**不能说**：本轮结果不能支持或否定"结构充分性"假设，不能给出方差估计、显著性、
数据集间泛化结论，也不能外推到其它 split/seed。

## 6. 偏离与基础设施记录

* 远端 `res` 的 checkout 没有这四个数据集的 raw 文件；按官方 `chrsmrrs.com` 归档下载到本地后，
  用 `rsync` 把**未修改的官方 raw 文件**放到 `res:data/TUD/<NAME>/<NAME>/raw/`。远端 prep 从这些
  raw 文件重建缓存，raw sha256、spec/norm/partition 哈希与本地完全一致（§3），因此不构成数据偏离。
* `res-2`（Slurm）未使用：其 checkout 没有 TU 数据，且需要额外的驱动/离线环境验证；本轮 12 个 run
  在 `res` 的双卡上顺序成对完成，未触及并发上限。
* 未发生失败重跑、超时、OOM 或作业取消；15 个 `rr` 作业全部 `exit_code=0`。
* 没有修改任何既有 track、冻结产物或 `data/` 下的文件。

## 7. 未做（明确列出）

* 多 seed / 交叉验证 / 重复划分；
* 超参数或结构搜索、消融（除固定的 FULL vs C1 vs GIN 三格外没有第四格）；
* 早停、EMA/SWA/weight soup、数据增强、SSL 预训练、额外正则或 loss；
* 其它数据集、其它节点/边属性方案、其它划分比例；
* 校准、OOD、可解释性或化学语义分析。

