# REPORT — `lcrc-pcba-rie-frozen-pilot-v1`

在**原始 LCRC-A（seed 0，epoch 40）完全冻结**的前提下，检验显式 motif–motif 拓扑距离关系
是否提供超出“相同 motif 组成 + 相同容量 + 相同监督配方”的 MolPCBA Validation AP 增量。
单 subset、单 seed、单宽度、8 epoch、一次运行；未做任何超参数搜索。

| field | value |
| --- | --- |
| status | `COMPLETED` |
| decision label | **`NO_RELATION_SIGNAL`** |
| source checkpoint | `tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A/checkpoint_epoch40.pt` |
| source SHA256 | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` |
| 参数（A / 头部 / 合计） | 287,644 / **8,784** / 296,428 |
| AP_A（本轮复现） | 0.2463553008（对登记值 0.2463560961 的绝对差 **7.95e-07**） |
| AP_REAL（epoch 8） | **0.2469585547** |
| AP_SHUFFLE（epoch 8） | **0.2454476301** |
| D_base = REAL − A | **+0.0006032539** |
| D_ctrl = REAL − SHUFFLE | **+0.0015109245** |
| D_ge20（107 个正例 ≥20 的 assay） | **+0.0005490662** |
| GPU 预算 | **0.0912 A100 GPU-hour**（target 0.5 / hard 1.0） |
| 代码 commit（正式训练 job） | `71db82d57538177bf64668a6fe3d1c2e27b2864e` |
| Test 是否被评估 | **否**（43,793 行 Test 标签有限值计数 = 0，行交集 = 0） |

---

## 0. 十二个必答问题

1. **用的是真正的原始 LCRC-A seed0 epoch40 吗？**
   是。文件 SHA256 与冻结值一致；payload 为 `lcrc-pcba-no-mp-checkpoint-v1` / round
   `lcrc-pcba-no-mp-sprint-v1` / arm `A` / variant `C1 + attention_pool` / seed 0 /
   epoch 40 / global step 109,520（= 40×2738）/ n_outputs 128 / 57 个 state tensor；
   `strict=True` 加载后与 checkpoint 逐 tensor **bitwise** 相等；参数 287,644 =
   245,184（base）+ 42,460（new）。20 项身份检查全部通过（`analysis/checkpoint_replay.json`）。

2. **官方 Validation AP 复现了吗？**
   是。本轮在同一 cache、同一 OGB `Evaluator` 下重放得到 **0.2463553008**，与运行登记表的
   0.2463560961 相差 **7.95e-07**，与指令值 0.2463560956 相差 7.95e-07，远小于 1e-4 容差。
   冻结前向使用 CUDA `index_add_`（atomic）归约，因此该判据是容差判据；单 batch 重抽取的
   logits 复现差 1.14e-05、关系特征差 2.4e-06 / 3.3e-06（`gpu_row_correspondence.json`）。

3. **8 类 motif 在 MolPCBA 上覆盖如何？**
   Train 350,343 / Valid 43,793 行；平均 **12.73 / 13.17** 个 motif，**59.7 / 76.0** 个合法 pair
   每分子；有至少一个合法 pair 的分子占 **99.69% / 99.68%**；落在多个距离 bin 的占
   **97.82% / 97.23%**。全库 20,918,875（Train）+ 3,326,408（Valid）个 pair。
   家族计数（Train）：Acceptor 1,276,416、Aromatic 808,203、Carbonyl 423,978、Donor 553,012、
   Halogen 185,299、Hydrophobe 1,118,235、NegIonizable 19,555、PosIonizable 76,490。
   Train 有 **2** 个 RDKit 解析失败（`CC(=O)O[AlH3]OC(C)=O`、`CCCCO[AlH3](OCCCC)OCCCC`，
   铝配合物），保留行、关系特征严格为零、监督样本不删；Valid 0 失败。

4. **距离是真实物理键图上的吗？特征是否真的只依赖真实图？**
   是。512 个真实分子（13,350 原子）上，自实现全对 BFS 与
   `scipy.sparse.csgraph.shortest_path(directed=False, unweighted=True)` 的**最大绝对差 = 0**；
   同一 512 分子的 cache 键图与 RDKit 图的原子数、键集合、原子顺序**全部一致**（0 处不符）。
   另外 18 项不变量全通过：motif 顺序置换差 **0.0**、原子重编号差 **1.9e-06**（容差 1e-5）、
   pair 端点交换差 **0.0**、重复 motif 去重差 **0.0**、跨连通分量 pair 数 **0**、
   无 motif / 全短距离 / 双分量三类退化情形关系张量严格为 **0**。

5. **SHUFFLE 对照是否严格只改“哪一对对应哪个距离”？**
   是。同一 motif 集合、同一 pair 集合、同一 165 维 pair 特征、同一每 bin pair 数量，只用
   `SHA256("rie-distance-shuffle-v1|<official_row_id>")` 派生的 PCG64 在分子内置换 bin。
   审计确认：每分子每 bin 计数**逐 bin 相等**；REAL≠SHUFFLE 的分子占比 Valid **96.97%**、
   Train 97.61%，远高于 15% 的“干预不足”下限；张量改变只发生在多 bin 分子上；
   零张量 ⟺ 无合法 pair（双向成立）。

6. **REAL 是否优于原 A？**
   只高 **+0.000603**（0.2469586 vs 0.2463553，相对 +0.245%），比预注册的 weak 门槛
   （+0.002）低 3.3 倍、比 promising 门槛（+0.005）低 8.3 倍。Valid BCE 从
   0.04577261 降到 0.04571470（两臂都降，REAL 略多）。

7. **REAL 是否优于等容量 SHUFFLE？**
   是，**+0.001511**（0.2469586 vs 0.2454476），方向正确且逐 epoch 单调；但这只有 promising
   门槛（+0.003）的一半。注意 SHUFFLE **低于** A −0.000908：无意义距离分配时，这个小监督头
   会把原 A 的排序**弄坏**；于是“REAL 比 SHUFFLE 好”里约 60% 来自“SHUFFLE 掉下去”，
   只有约 40% 来自“REAL 升上去”。

8. **在正例充分的 assay 上优势是否同样存在？**
   部分存在但更弱。107 个正例 ≥20 的 assay 上 D_ge20 = **+0.000549**，中位数只有
   **+0.0000435**；其中 **54/107** 改善（门槛要求 ≥60）。5–19 正例的 18 个 assay 上
   REAL−A = +0.001007、REAL−SHUFFLE = +0.005157（噪声更大，且 SHUFFLE−A = −0.004150）。

9. **改善是广泛的还是集中在少数 assay？**
   集中。127 个可评分任务中 REAL 比 A 好 **66** 个、差 **61** 个；正向增量合计 +0.3861，
   负向合计 −0.3095，净值 +0.0766。前 5 个任务占正向质量的 **45.7%**，前 10 个占 **60.1%**。
   最大改善 task 112（+0.0739，仅 23 个正例；REAL 0.3886 / SHUFFLE 0.3602），
   其次 task 1（+0.0304，70 正例）、task 31（+0.0283，6 正例）、task 69（+0.0252，19 正例）；
   最大退化 task 27（−0.0470，16 正例）、task 11（−0.0251，53 正例）、
   task 40（−0.0227，85 正例）。REAL 比 SHUFFLE 好 **85/127**（比 A 好的只有 66/127），
   说明“控制对照”这一层的一致性明显强于“绝对提升”这一层。

10. **成本与预算？**
    GPU 合计 **0.0912 h**：唯一一次冻结前向 0.0478 h（forward 34.76 s，全程 172 s，
    11,338 分子/s 前向、2,288 分子/s 端到端）、REAL 头训练 0.0221 h（79.5 s，其中纯训练
    15.2 s）、SHUFFLE 0.0212 h（76.1 s，13.2 s）、smoke 8.1e-05 h；峰值显存 132 MB（抽取）。
    CPU 结构构建全量 **190.2 s**（Train 163.4 s + Valid 26.8 s，12 workers），构建峰值
    RSS 2.66 GB。头部推理增量 **17,568 FLOPs / 分子**（8,784 MACs），可训练参数为
    A+头部总量的 **2.96%**。
    *实现说明*：1024 分子计时探针的并行外推（0.68 h）无效——探针只有 1 个 chunk，
    被单个 worker 串行执行；真实并行构建是 190 s。

11. **冻结判定是什么？**
    **`NO_RELATION_SIGNAL`**。预注册规则：`D_base < +0.002` 或 REAL 对 SHUFFLE 无明确优势
    → NO_RELATION_SIGNAL。本轮 D_base = +0.000603 < 0.002，且 D_ctrl = +0.001511、
    D_ge20 = +0.000549 都远低于各自门槛（+0.003），改善任务数 54 < 60。
    所有前置门槛（checkpoint 身份、AP 复现、18 项不变量、SHUFFLE 保留性/干预度、
    配对初始化 15 项、test isolation）**全部通过**，因此这是有效阴性结果，
    不是 `INCOMPLETE` / `SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH` /
    `INSUFFICIENT_RELATION_INTERVENTION`。

12. **是否授权/启动完整 LCRC-RIE 联合训练？**
    **不授权，也没有启动。** 四项 PROMISING 条件一个都没满足。本轮只交付冻结探针与结论。

---

## 1. Protocol compliance

| constraint | observed |
| --- | --- |
| source checkpoint | 原 A，seed 0，epoch 40，未重训、未替换、未用“最接近”版本 |
| SSL / A1 / A2 / A3 / G / V | 均未使用 |
| motif 家族 | 固定 8 类（RDKit ChemicalFeatures 6 + Carbonyl/Halogen SMARTS 2），未增删 |
| 距离定义 | 物理键图最短路径，`d<3` 丢弃，跨分量排除，bin `3–4 / 5–6 / >=7` |
| pair 特征 | 恰好 165 维（64 + 64 + 36 + 1） |
| 关系张量 | `[3,165]`，`sum/sqrt(1+n)`，无消息传递、无 attention、无迭代 |
| 可训练参数 | **8,784** = 165×16 + 48×128（两组各一，均为 8,784） |
| REAL / SHUFFLE | 同一 motif、pair、pair 特征、每 bin 计数、初始化、配方、batch 顺序 |
| seed / epochs / batch | 0 / 8 / 1024（343 步/epoch，合计 2,744 步） |
| optimizer | AdamW，lr 3e-4 constant，wd 1e-3，betas (0.9,0.999)，eps 1e-8，clip 5.0，FP32 |
| dropout / scheduler / early stop | 0 / none / false |
| loss | masked BCE + 1e-5·mean\|task_weight\| |
| 评价点 / 主比较 | 0/2/4/8；**只看 epoch 8** |
| 超参数搜索 / 多 seed | 无 / 无 |
| Test | 从未加载、预测、评分或调参 |
| 运行预算 | 0.0912 GPU-h（≤0.5 target，≤1.0 hard），GPU1 only（`CUDA_VISIBLE_DEVICES=1`） |

### 1.1 冻结后仅修复实现 bug（不改协议数学）

| commit | 问题 | 影响 |
| --- | --- | --- |
| `ed11f7db` | `merge_structures` 把每个 chunk 的 `motif_atom_ptr` 整段拼接，172 个 chunk 产生 171 个重复边界，首个 chunk 之后的 motif 全部切错原子 | 第一次 structures job（`rie-structures-20261107-133829`）产物被判无效，**在任何 feature 抽取前**丢弃并重建；新增 3-chunk 回归测试 |
| `f4ba81cd` | `load_structures` 只返回 ndarray，`n_molecules` 丢失 | audit 的 SHUFFLE 段 KeyError |
| `15a83411` | smoke 在 `inference_mode` 内构图做 backward | smoke 崩溃；改为在块外基于 detach 的 logits 建图 |
| `c41fdc1c` | `rr deploy` 的 `uv sync` 只装默认组，rdkit 被移除 | 需要 RDKit 的 job 在 job 内重装 lipo 组 |
| `b9a29b0b` | SHUFFLE 保留性 gate 读错嵌套 key | 新增合成 payload 的 gate 测试（含 15% 下限） |
| `71db82d5` | `verify_feature_cache` 用文件哈希比对 manifest 的内容哈希；preflight 的 “recipe/plan 相同” 是硬编码 True | 改为内容哈希与真实的 AdamW audit / 首 batch 行哈希比对 |

特征缓存于 `ed11f7db` 生成，之后所有 commit 均未触碰 `rie_extract.stage_features` /
`rie_features` 的关系数学；`71db82d5` 的 `verify_feature_cache` 重新校验了全部内容哈希并通过，
因此证据链未断裂。

---

## 2. 源 checkpoint 身份与 AP 复现

见问题 1、2。补充：`state_dict_bitwise_equal_to_checkpoint = True`、`strict_load_missing/unexpected = []`；
冻结后 `requires_grad_all_false`、`no_gradients`、`parameters_unchanged`、`still_eval` 全为 true；
抽取前后参数指纹相同（`c48a103a…`）。Train macro AP 重放 **0.4196760**。

---

## 3. 图与关系构造的可核验性

| 审计 | 规模 | 结果 |
| --- | --- | --- |
| 距离 vs scipy | 512 分子 / 13,350 原子 | 最大绝对差 **0** |
| cache 键图 vs RDKit | 512 分子 | 原子数/键集合/原子顺序 0 处不符 |
| motif 顺序 | 96 分子 | 0.0 |
| 原子重编号 | 96 分子 | 1.9e-06（≤1e-5） |
| pair 端点交换 | 96 分子 | 0.0 |
| 重复 motif 去重 | 96 分子 | 0.0 |
| 跨分量 pair | 全部退化用例 | 0 个 |
| 无 motif / 全短程 / 双分量 | 3 类 | 关系张量严格 0 |
| 官方 AP 一致性 | Valid + Train | OGB Evaluator 与 `pc_metrics` 独立实现差 ≤1e-9 |
| 行对应 | Train/Valid | cache 行、结构行与官方 split 行完全一致（含 sha256） |
| Test 隔离 | — | Test 标签有限值 0；行交集 0；未写任何 Test 预测 |

RMS 归一化只用 REAL Train 拟合（350,343 行 × 495 列）：min 0.00751 / median 2.584 /
max 8.096，**没有列被 floor**；REAL 与 SHUFFLE、Train 与 Valid 共用同一 scale，
不做均值中心化（零 pair → 严格零张量）。

---

## 4. 训练设置与配对性

* 初始 state SHA256 `13d3e18cd485799cc6ecd6dc6a71f2b05f20439872763492718849bc9f963bc9`
  （两臂相同，且与 preflight 记录一致）；`task_weight` 精确全零，
  `shared_projection` 由 `Generator(manual_seed=0)` 生成，不推进全局 RNG。
* Epoch 0 两臂 logits 与冻结 A **逐位相等**（2048 行，`max_abs_delta = 0.0`）。
* 首次更新时 `shared_projection` 梯度精确为零（零残差初始化使然），`task_weight` 梯度非零；
  step 1 之后两者梯度均非零且有限（15 项 preflight gate 全通过）。
* 顺序 plan 相同：`ac1dd991d325126c2e1780867e059f561b3147b7da8cc4b886334e8e956dd3de`；
  AdamW audit 相同；第一 batch 行哈希相同；两臂唯一差异是关系输入
  （`relation_inputs_differ = True`）。
* 最终 state 不同：REAL `c8f1586b…`，SHUFFLE `94c239ee…`；
  REAL `\|W\|_1 = 136.95`、mean\|w\| = 0.02229、shared L2 = 3.520；
  SHUFFLE `\|W\|_1 = 125.66`、mean\|w\| = 0.02045、shared L2 = 3.212；
  两者 task weight 非零比例都是 100%（没有塌缩成稀疏解）。

---

## 5. 结果

### 5.1 epoch 曲线（官方 Validation macro AP）

| arm | ep0 | ep2 | ep4 | ep8 |
| --- | ---: | ---: | ---: | ---: |
| REAL | 0.2463553 | 0.2455856 | 0.2464663 | **0.2469586** |
| SHUFFLE | 0.2463553 | 0.2452656 | 0.2454813 | 0.2454476 |

两臂都在 epoch 2 下探；REAL 在 epoch 4 恢复到 A 之上并继续上行，SHUFFLE 再未回到 A。
REAL 在 ep2/4/8 都高于 SHUFFLE（+0.00032 / +0.00099 / +0.00151），差异随训练单调增大——
方向与“真实距离关系提供了一点可用信息”一致，但幅度只有一个千分点的量级。

### 5.2 三项比较

| 量 | 值 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| D_base = REAL − A | **+0.000603** | weak +0.002 / promising +0.005 | 未达 weak |
| D_ctrl = REAL − SHUFFLE | **+0.001511** | promising +0.003 | 达一半 |
| D_ge20 = REAL − A（107 assay） | **+0.000549** | promising +0.003 | 未达 |
| REAL 优于 A 的任务数 | 66 / 127 | promising ≥60（ge20） | 54 / 107 |
| REAL 优于 SHUFFLE 的任务数 | 85 / 127 | — | — |
| SHUFFLE 优于 A 的任务数 | 48 / 127 | — | SHUFFLE 整体更差 |

### 5.3 分层与集中度

| stratum | n | REAL−A mean / median | REAL−SHUFFLE mean | SHUFFLE−A mean |
| --- | ---: | --- | ---: | ---: |
| ge20 | 107 | +0.000549 / +0.0000435 | +0.001094 | −0.000545 |
| 5–19 | 18 | +0.001007 / +0.000647 | +0.005157 | −0.004150 |
| <5 | 2 | −0.000129 | −0.009 | +0.0089 |

逐任务 REAL−A：max **+0.0739**（task 112）、min **−0.0470**（task 27）、sum +0.0766；
正向质量前 5 / 前 10 任务占比 **45.7% / 60.1%**。逐任务权重 L2 与 |ΔAP| 的相关只有 0.161。

### 5.4 为什么是 `NO_RELATION_SIGNAL` 而不是 `WEAK_OR_MIXED_SIGNAL`

weak 标签要求 `D_base ≥ +0.002`；本轮 +0.000603 只有该门槛的 30%。虽然 D_ctrl 为正且
逐 epoch 单调，但 60% 的 D_ctrl 来自 SHUFFLE 相对 A 的下降，而不是 REAL 的实际增益；
高支持度 assay 上的增益中位数仅 +4e-5，改善任务数 54 < 60。
按预注册规则必须给出 NO_RELATION_SIGNAL，不能因为“方向都对”就上调标签。

---

## 6. 成本台账（`analysis/gpu_ledger.json`）

| entry | GPU 秒 | GPU 小时 | 峰值显存 |
| --- | ---: | ---: | ---: |
| 冻结 A 抽取（394,136 分子，唯一一次 forward+aux） | 172.3 | 0.04785 | 132 MB |
| REAL 头训练（8 epoch + 4 次评价） | 79.5 | 0.02209 | 60 MB |
| SHUFFLE 头训练 | 76.1 | 0.02115 | 60 MB |
| smoke（8 分子 + 头部梯度） | 0.29 | 0.00008 | 18.5 MB |
| **合计（阶段墙钟上界）** | — | **0.0912** | — |

CPU：结构构建 190.2 s / 12 workers（峰值 RSS 2.66 GB）；特征作业设备时间 172.3 s
（抽取 124.5 s = Train 107.2 s + Valid 17.3 s，RMS 拟合 17.3 s，其余为 cache 装载与 AP 重放）；
audit 作业 42 s（smoke 0.3 s + 18 项审计 + GPU 行对应抽查）；preflight 14 s；compare 15 s。窗口内 GPU1 独占，GPU0 未被使用
（`CUDA_VISIBLE_DEVICES=1`，显存 132 MB，未干扰外部占用）。

---

## 7. 本轮**不能**声称什么

* 不能声称“显式结构关系在 MolPCBA 上无效”。
  证据只是：单 subset、单 seed、8 epoch、只有 8,784 参数的线性读出、单宽度（48 维），
  在 +0.002 的预注册分辨率下看不到效应。
* 不能声称“加一个小监督头没用”。REAL 确实比 A 高 0.0006、比 SHUFFLE 高 0.0015，
  且 BCE 下降；但无法把这点差异从“多 8,784 个参数的读出层 + 单 seed 噪声”里分离出来。
* 不能把本轮的 Validation 当作独立确认集：该 split 已被历史选型反复使用。
* 不能把 54/107 或 66/127 当作显著性证据（无多重比较校正、无 seed 方差）。
* 任何结果都不自动授权完整 LCRC-RIE 联合训练、超参数搜索、更多 seed 或更多 motif 家族。

若要把结论推向“有/无”，最少需要：3 个 seed × 同一冻结 A（估 3×0.043 = 0.13 GPU-h）、
以及一个**参数量匹配的非关系对照**（例如把关系张量替换成同形状的 motif 组成统计量，
没有距离分箱），用来把“读出层容量”与“距离关系”彻底分开。

---

## 8. 复现命令与产物

```bash
R=tracks/structural-sufficiency/lcrc_pcba_rie_frozen_pilot_v1
rr deploy res --pool res-gpu1
rr run res --pool res-gpu1 rie-structures --result $R/runs --result $R/analysis -- \
  bash -c "uv sync --frozen --group lipo -q && python $R/code/rie_extract.py --stage structures --workers 12 --chunk-size 2048"
rr run res --pool res-gpu1 rie-features --result $R/runs --result $R/analysis -- \
  bash -c "python $R/code/rie_extract.py --stage features"
rr run res --pool res-gpu1 rie-audit --result $R/runs --result $R/analysis -- \
  bash -c "uv sync --frozen --group lipo -q && python $R/code/rie_extract.py --stage smoke && python $R/code/rie_audit.py --with-gpu-spotcheck"
rr run res --pool res-gpu1 rie-train-preflight --result $R/runs --result $R/analysis -- \
  bash -c "uv sync --frozen --group lipo -q && python $R/code/rie_train.py --stage preflight"
rr run res --pool res-gpu1 rie-train --result $R/runs --result $R/analysis -- \
  bash -c "uv sync --frozen --group lipo -q && python $R/code/rie_train.py --stage formal"
rr run res --pool res-gpu1 rie-compare --result $R/analysis -- \
  bash -c "python $R/code/rie_compare.py"
```

| run id | 作用 |
| --- | --- |
| `rie-structures-20261107-135421-cabafc3d` | 结构（最终有效版） |
| `rie-features-20261107-135930-3eb233c8` | 唯一一次冻结前向 + 特征 + RMS + AP 复现 |
| `rie-audit-20261107-142103-7097be5a` | smoke + 18 项审计 + GPU 行对应 |
| `rie-train-preflight-20261107-144153-392ed936` | 配对齐性预检（15 gates） |
| `rie-train-20261107-144613-db624118` | REAL / SHUFFLE 正式 8 epoch |
| `rie-compare-20261107-145149-fb286e21` | 比较、逐任务 CSV、ledger、decision |

远端恢复路径（git-ignored，未入库）：
`/home/hxy/cy/research/tracks/structural-sufficiency/lcrc_pcba_rie_frozen_pilot_v1/runs/`
（`structures/` 全量结构、`features/` 冻结 logits + 4 份关系张量 + RMS、`heads/{real,shuffle}/`、
`smoke/`）。关键哈希：

| artifact | sha256 |
| --- | --- |
| checkpoint | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` |
| head 初始 state | `13d3e18cd485799cc6ecd6dc6a71f2b05f20439872763492718849bc9f963bc9` |
| REAL 最终 head | `c8f1586ba2c681d71ff16c30faf1585738baab00ef2ca34c3c7df387b875b82a` |
| SHUFFLE 最终 head | `94c239eefdc30b7c7cdfe8731bcb3285d808b3efdd022c4a2ad8686920934fab` |
| valid base logits（内容） | `8f7297eb4c47f9d971ee3191a81220c48ff44cacf0c225492c6c5ed5d2f8868b` |
| valid REAL 关系（归一化文件） | `41122be88ade9ef5cb703fc98058fdd22411d729006735cf8b5a1cb3e0f5fa39` |
| valid SHUFFLE 关系（归一化文件） | `d7efa9b6f2a50371c146bddc6d4dc0dac3b10cb1eb7299cae0c62fa29fb17407` |
| RMS scale | `25684391c5a76cbe248f621956a4386dcb52c8e6bcf4edc4b6fc077f0104de41` |
| order plan | `ac1dd991d325126c2e1780867e059f561b3147b7da8cc4b886334e8e956dd3de` |
| per-task CSV | `b0b042f9dd64819988b31e75644a233dc3e8646bd2ae4055181c76d4131ae4c0` |

---

## 9. 交付物

```
PROTOCOL.md  README.md  REPORT.md
code/rie_common.py rie_motifs.py rie_features.py rie_extract.py
     rie_model.py rie_train.py rie_compare.py rie_audit.py
tests/test_rie_pilot.py            （28 项，CPU-only，~1.1 s）
analysis/checkpoint_replay.json    analysis/motif_coverage.json
analysis/feature_manifest.json     analysis/relation_invariance_audit.json
analysis/shuffle_control_audit.json analysis/gpu_row_correspondence.json
analysis/paired_initialization.json analysis/epoch8_comparison.json
analysis/per_task_deltas.csv       analysis/gpu_ledger.json
analysis/decision.json
```
