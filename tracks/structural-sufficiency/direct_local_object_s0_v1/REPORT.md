# REPORT — 直接完整局部对象，独立预测，s0（ogbg-mollipo, seed 0）

Protocol: `PROTOCOL.md`（训练前冻结），config `configs/dlo_s0.json`，
protocol id `direct_local_object_s0_v1`。
**Decision: `DIRECT_LOCAL_PREDICTION_PROMISING_SCOPE_UNRESOLVED`**。
实际执行：两个 arm 均完成共同 240 epoch，单 seed，官方 3360/420/420；
无 OOM；总 GPU 时间约 0.67 GPU·h。

T0 = `2026-11-21T21:23:58+08:00`。最终训练于 22:19:57 结束，早于 T0+105 min
（23:08:58）。全部训练在 `res`（a100-2）执行：`DIRECT_S56` 使用
`res-gpu1`（CUDA_VISIBLE_DEVICES=1），`MATCHED_W34` 使用 `res-gpu0`
（CUDA_VISIBLE_DEVICES=0），驱动 550.163.01，torch 2.5.1+cu124。

## 1. 直接回答

### 1.1 性能水平（best-Valid 权重，原始标签单位）

| arm | 参数 | best ep | horizon | Train RMSE | Valid RMSE | Test RMSE | Valid MAE | Test MAE | wall | GPU·h | max VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DIRECT_S56` | 329 705 | 112 | 240 | 0.2487 | **0.7387** | 0.7753 | 0.5406 | 0.5945 | 1208 s | 0.335 | 939 MB |
| `MATCHED_W34` | 329 705 | 101 | 240 | 0.4043 | **0.6567** | 0.7247 | 0.4980 | 0.5429 | 1110 s | 0.308 | 239 MB |

Final epoch（240）三划分：

| arm | Train RMSE | Valid RMSE | Test RMSE |
|---|---:|---:|---:|
| `DIRECT_S56` | 0.1389 | 0.7690 | 0.8156 |
| `MATCHED_W34` | 0.2673 | 0.6908 | 0.7425 |

只读旧参照（同数据、同轮次固定坐标；未重训，prediction 已核对 row_id 与数值）：

| 参照 | Valid RMSE | Test RMSE |
|---|---:|---:|
| `WORD_PREFIX` | 0.69909 | 0.68906 |
| `BAG_CONT` | 0.69237 | 0.69804 |
| `GINE_RAW_REF` | 0.61015 | 0.67220 |

### 1.2 是否优于匹配小对象？

**没有。小对象 `MATCHED_W34` 在两个划分上都更好。**

* Valid：`e_L − e_S = 0.738724 − 0.656651 = +0.08207`（S56 更差）。
* Test：`t_L − t_S = 0.775328 − 0.724698 = +0.05063`（S56 更差）。
* 分子级配对 bootstrap（1 000 次，best 权重）：Valid 中位数 **+0.0835**
 ，95% 区间 **[+0.0324, +0.1375]**，99.9% 重采样为正；Test 中位数
  **+0.0523**，95% 区间 **[−0.0029, +0.1039]**，96.9% 为正。
  这只是选定权重下的样本组成敏感性，不是跨 seed 或选择不确定性。

因此本轮的 5/6 原子大对象没有显示相对 3/4 原子对象的优势；`MATCHED_W34`
是相对更好的小对象参照。

### 1.3 与 GINE 的差距

* `MATCHED_W34` Valid 落后 GINE **0.04650**，Test 落后 **0.05250**。
* `DIRECT_S56` Valid 落后 GINE **0.12858**，Test 落后 **0.10313**。

即使 `MATCHED_W34` 的 Valid 0.6567 已低于 0.67，它仍不是“追平强参照”；Test
上它也比 GINE 差 0.05 以上。

### 1.4 是否值得继续？

按冻结门槛，`min(e_L,e_S)=0.65665 <= 0.67`，但 `e_S−e_L=−0.08207`，不满足
“大对象有前景”的条件，所以结论是
**`DIRECT_LOCAL_PREDICTION_PROMISING_SCOPE_UNRESOLVED`**：直接局部对象模型有
一定预测潜力，但优势不能归因于 5/6 对象；小对象版本才是这个信号所在。

同时附 **`SPLIT_CONFLICT`**：`MATCHED_W34` 相对 `WORD_PREFIX` 的 Valid 改善
为 **−0.04244**（更好），但 Test 恶化 **+0.03563**（更差，超过 0.03）。
因此“小对象直接模型”不能作为稳健路线；本轮不追加 seed、配方矩阵或
count-only 训练，任务到此结束。

## 2. 三个核心差值

### 2.1 Valid（best-Valid 权重）

| 差值 | 值 | 解释 |
|---|---:|---|
| `e_L − e_S` | **+0.08207** | S56 比匹配的 W34 差 |
| `e_L − e_P` | **+0.03963** | S56 比旧词袋 WORD_PREFIX 差 |
| `e_L − e_G` | **+0.12858** | S56 比 GINE 差 |

补充：`e_S − e_P = −0.04244`（W34 比旧词袋好），
`e_S − e_G = +0.04650`（W34 比 GINE 差）。

### 2.2 Test（同一 best-Valid 权重）

| 差值 | 值 | 解释 |
|---|---:|---|
| `t_L − t_S` | **+0.05063** | S56 比 W34 差 |
| `t_L − t_P` | **+0.08627** | S56 比旧词袋差 |
| `t_L − t_G` | **+0.10313** | S56 比 GINE 差 |

补充：`t_S − t_P = +0.03563`（W34 的 Test 比旧词袋差），
`t_S − t_G = +0.05250`（W34 的 Test 比 GINE 差）。

S56 的 Train RMSE 明显更低（0.2487 vs W34 的 0.4043），但 Valid/Test 都更差；
这是当前模型/配方的泛化差异，不能据此推断“5/6 对象缺少信息”，也不能推断
小对象一定更优。本轮不再扩实验解释。

## 3. 结构、帧和覆盖（实际全量）

新缓存：`data/dlo_objects_s0.npz`，sha256
`bab7894d488a2724d8795aa44ba899153d3bde09ccd8143de2bcbf43a61c6dcc`，
214 356 069 bytes（204.4 MiB）。原始缓存 sha256
`fe002287...93123c7c`；旧 `swf_base` / `swf_sent` 的 sha256 分别为
`5931fe04...c633b22b` / `31af04d4...a16538`，均与旧轮记录一致。

| 项目 | `DIRECT_S56` | `MATCHED_W34` |
|---|---:|---:|
| 核心对象 | 3 316 287 | 1 593 359 |
| 补齐对象（完整连通分量） | 3（全部 size 1） | 3（全部 size 1） |
| 对象总数 | 3 316 290 | 1 593 362 |
| 原始 frames | 5 060 215 | 2 101 795 |
| 去重后 frame 输入 | 516 790 | 69 816 |
| frame 去重率 | 89.79% | 96.68% |
| frames / object | 1.5259 | 1.3191 |
| 对象 / 分子（Train 均值） | 780.4（bucket0 374.3 + bucket1 406.1） | 376.4（bucket0 123.5 + bucket1 252.8） |
| Train / Valid / Test 对象数 | 2 621 983 / 342 912 / 351 395 | 1 264 566 / 162 942 / 165 854 |
| 原子覆盖 | 113 568 / 113 568 | 113 568 / 113 568 |
| 未覆盖原子 | 0 | 0 |
| 大分量（≥5）无 S56 句 | 0 | — |

补充事实：

* 核心行数与旧缓存逐行一致：S56 3 316 287（旧 `mol_sent_ptr` 全量），W34
  1 593 359（旧 `word_n ∈ {3,4}` 全量）；`core_duplicates = 0`。
* 全部 4 200 个分子都至少有一个对象；没有删除分子、没有按标签补对象。
* 对象补齐只发现 3 个 size-1 分量；没有 size 2–4 的 W34/S56 小分量需要补。
* 全局 frame 去重比设计阶段估计的“约 1.44 个/对象”更强，因为原子属性只有
  76 种唯一 9 列组合，完整 frame 输入在世界层面高度重复；对象行本身没有去重。
* 每个对象的 frame 权重 `1/n_frames` 之和最大误差 4.4e-16（S56）/1.1e-16
  （W34，缓存构建阶段 float64 检查；存储为 float32）。
* 大对象 arm 的 5/6 对象、小对象 arm 的 3/4 对象均按定义保留；没有以
  cover/frame-pair 次数重复计权。

## 4. 检查（训练前，全部通过）

`analysis/checks.json`（远程 GPU 版本，elapsed 9.87 s）：

| 检查 | 结果 |
|---|---|
| 官方 split、row_id=arange、Train-only scaler、分子内 bond 编号 | pass |
| 旧句/词核心计数、padding 截断、核心/补齐覆盖 | pass（S56/W34 未覆盖原子均为 0） |
| LUT vs brute force，k=1..5 全量 + k=6 固定 515 个 mask 抽样 | pass |
| 重编号不变性，2 个合成图 + 3 个真实分子 × 2 arms，FP64 | 最大绝对差 **1.04e-16** |
| 同上 GPU FP32 | 最大绝对差 **5.96e-8** |
| 单分子 vs 乱序多分子 batch | **6.71e-8** |
| scatter vs 显式对象循环 | **7.15e-7** |
| 真实 Train batch 反向：atom/bond/frame encoder/graph head | 梯度有限且非零 |
| optimizer 是否更新 frame encoder | 是，最大参数变化 **1.00e-3** |
| 参数 ledger | 329 705（atom_emb 1 392 + atom_mlp 6 752 + bond_emb 104 + bond_mlp 1 328 + frame_encoder 154 752 + graph_head 165 377） |

两 arm 的可训练参数初始 hash 相同：
`3d8d8cb1064efc9ca7908a5e40c348ec2fae6a79ff260b0771580e8998a24a99`。
`run.json` 里的 `init_state_hash` 还包含 arm-specific 的 `mu_bucket` buffer，
所以两个值不同；这是固定常量 buffer，不是训练初始参数，已在 manifest 中注明。

## 5. 固定权重依赖检查：h_o 置零

在各自 best-Valid 权重上，把对象输出 `h_o` 全部置零、只保留四个结构计数
（`log1p(n0), log1p(n1), log1p(n_atoms), log1p(n_bonds)`），不做重训：

| arm | 正常 Valid RMSE | 置零后 Valid RMSE | ΔRMSE | 预测 RMS 正常 | 预测 RMS 置零 | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| `DIRECT_S56` | 0.7387 | 1.8113 | **+1.0726** | 2.5272 | 0.8646 | 0.342 |
| `MATCHED_W34` | 0.6567 | 2.2402 | **+1.5835** | 2.4237 | 0.3302 | 0.136 |

预测对对象路径有强依赖，不是“几乎只看计数”。这只是一个使用依赖检查，
不是 count-only 的收益对照，也没有启动 count-only 训练。

## 6. 执行、预算和偏差

| 阶段 | 时间 | 说明 |
|---|---|---|
| T0 | 21:23:58 | 本地 |
| 数据/缓存 prepare + checks | 21:48:23–21:49:57 | 远程 CPU/GPU，prepare 约 63 s，checks 9.9 s |
| stage A（第一次） | 21:53:49–21:54:26 | 发现输出路径拼接错误，未用于结果 |
| stage A 重跑（唯一工程重跑） | 21:56:59–21:57:35 | 结果与第一次逐 epoch 相同 |
| stage B（共同 240 epoch） | 22:00:04–22:19:57 | S56 20m8s，W34 18m31s |
| 最终交付 | 约 22:30 | 早于 23:08:58 |

* 成本估计：S56 epoch 3–5 中位 4.81 s → 240 epoch 估计 24.0 min；W34 4.64 s →
23.2 min；均 ≤45 min，所以 **240 epoch 分支被采用，未切 160**。
* 预算：总 GPU 时间约 **0.67 GPU·h**（含 stage A 重跑；上限 2 GPU·h）；
  总墙钟约 66 min（上限 120 min）；无 OOM。远端主机 503 GB RAM；缓存构建
  峰值本地约 1.5 GB，GPU 峰值显存 939 MB（S56）/239 MB（W34）。
* 偏差 1（唯一工程重跑）：`run_dir` 最初把 repo-relative `--out` 当作
  `results/` 的子目录，stage A 写到了嵌套路径；修复为绝对/仓库相对/
  results 名称三种解析后，重跑 stage A。模型、配方、数据、seed 均未改变，
  重跑结果逐 epoch 相同。
* 偏差 2（记录口径）：`init_state_hash` 含 arm-specific `mu_bucket` buffer；
  可训练参数 hash 已单独记录并相同，见 §4。
* 没有新 seed、ESOL、LR/宽度扫描、长过拟合检查、旧轮救援实验；没有使用
  GINE/指纹/预训练/蒸馏/零初始化残差/旧词袋权重；没有按 Test 改配方或挑 epoch。
* 旧 WORD_PREFIX 的 613 033 是实际参与计算/优化的参数；旧表 808 810 是包含
  未用模块的总分配参数。新两 arm 均为 329 705。

## 7. 结果解释与停止决定

* **大对象假设不成立（本轮范围内）**：`DIRECT_S56` 在 Valid 和 Test 都差于
  匹配的 3/4 对象，且 Valid 差 0.082，bootstrap 方向一致；同时它也差于旧词袋
  和 GINE。不能把任何改善归因于“完整 5/6 原子”。
* **小对象直接模型有信号但不稳健**：`MATCHED_W34` Valid 0.6567，低于 0.67，
  且比旧词袋 Valid 好 0.042；但 Test 0.7247，比旧词袋差 0.036、比 GINE 差
  0.052。`SPLIT_CONFLICT` 已记录，不能称为稳健路线。
* **没有达到继续大投入的门槛**：本轮冻结门槛把它归为
  `DIRECT_LOCAL_PREDICTION_PROMISING_SCOPE_UNRESOLVED`，而不是
  `LARGER_OBJECT_PROMISING`；按任务书，本轮结束，不自动回词组合、不加新机制、
  不加 seed/数据集。
* 若未来仍要研究，唯一合理的研究建议是：**从相同完整局部输入出发，设计一个
  从头训练的可复用词函数模型，并与这个 `MATCHED_W34` 直接参照比较**；
  这不是已完成的创新，也不能把本参照的预测/隐藏向量接成教师或外挂。

## 8. 文件

```
PROTOCOL.md                        冻结协议 + 偏差
configs/dlo_s0.json                全部维度/配方/预算
code/dlo_common.py                 路径、身份、seed、指标、OGB 维度
code/dlo_frames.py                 精确 canonical LUT + 完整 frame 生成
code/dlo_prepare.py                对象/小分量补齐 + frame 去重缓存
code/dlo_cache.py                  任意乱序 batch 的显式映射打包
code/dlo_model.py                  474→…→516→1 固定网络
code/dlo_train.py                  stage-a / continue、三划分逐 epoch、checkpoint
code/dlo_checks.py                 数据/LUT/重编号/batch/梯度检查
code/dlo_analyze.py                主表、差值、bootstrap、manifest
analysis/main_table.csv            主表（新两 arm + 只读旧参照）
analysis/summary.json              决策、差值、bootstrap、干预、split conflict
analysis/structure_stats.json      对象/帧/覆盖/缓存 hash
analysis/checks.json               训练前检查
analysis/run_manifest.json         配置/缓存/检查/rr provenance
analysis/rr_provenance.json        每次 rr 运行的 pool/GPU/驱动/时间
results/lipo_DIRECT_S56_s0/        curves, run.json, state_best/final, predictions
results/lipo_MATCHED_W34_s0/       同上
data/dlo_objects_s0.npz            大缓存（本地保留，gitignore）
data/dlo_cache_stats.json          缓存统计（本地保留，gitignore）
```

`state_*.pt`、`predictions_*.npz`、大缓存按任务要求本地保留并 gitignore；
git 只提交代码、配置、协议、报告、曲线/run 记录和轻量 analysis。

## 9. 本报告不声称的内容

* 不声称“完整 5/6 原子对象”在一般情况下无效；只声称在本轮的完整对象编码、
  固定 MLP/池化/读出和 240 epoch 下，它没有优于匹配的 3/4 对象。
* 不声称直接局部模型整体失败；小对象参照在 Valid 上确实有信号，但 Test
  反向，且只有一个 seed/一个划分。
* 不声称 bootstrap 是跨 seed 或模型选择不确定性；它只反映固定权重下的样本组成。
* 不声称两个 arm 的函数类容量完全等价：网络/参数/更新相同，但对象分布、
  bucket 计数归一化和 padding 列不同。
* 不解释 Train 拟合差异的机制（Adam、宽度、初始化、对象分布均未做归因实验）。
