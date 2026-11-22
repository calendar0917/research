# CSOI-v1 — 上下文条件结构对象交互 pilot 报告

协议 `contextual-structural-object-interaction-pilot-v1`（`PROTOCOL.md`，训练前冻结）。
数据 `ogbg-mollipo`（官方 scaffold split），seed 0，单轮。
所有数字来自 `analysis/*.json` 与 `results/*.json|csv`；本文件不重算指标，只引用记录。

---

## 0. 结论（先给判决）

1. **主假设不成立：把细粒度对象 token 集合保留到交互之后再池化，没有比提前压成一个片段嵌入更好。**
   主对比 B→C（`CSOI_ADD` → `CSOI_GLOBAL`，唯一差别是加入内容注意力交互）在 Valid 整分子 RMSE 上
   **Δ = −0.0150**（B 0.7196 → C 0.7346，负数表示交互臂更差），配对 bootstrap（2000 次，420 个 Valid 分子）
   95% CI **[−0.0640, +0.0323]** 跨 0 → 按冻结判据是 **no evidence**（且点估计方向为负）。
   没有达到 weak 门槛（0.0121），更没有达到 `delta_task`（0.0303）。

2. **次级对比（关系偏置）也没有证据。** C→D（`CSOI_GLOBAL` → `CSOI_REL`）：Δ = **+0.0074**
   （D 0.7272 更好一点），CI **[−0.0258, +0.0391]**，跨 0 → no evidence。

3. **这次 pilot 在统计上没有能力分辨协议设定的效应量。** 主对比的 bootstrap 标准误 ≈ 0.0246，
   双侧 95% 最小可检测差值 ≈ **0.048**，而协议设定的 `delta_task` 是 **0.0303**。
   在 420 个 Valid 分子、单 seed、Valid 上 90.8% 的解析是"见过的 S + 没见过的 C"的条件下，
   分辨率不足是设计层面的既成事实，不能把 "no evidence" 读成 "效应为零"。

4. **在"见过的上下文"上交互确实更强，但在"没见过的上下文"上优势反转**——这是本轮最有信息量的结果：

   | 配对脚本范围 | A `TRC_FACTOR` | B `CSOI_ADD` | C `CSOI_GLOBAL` | D `CSOI_REL` | E `GINE_RAW` |
   |---|---|---|---|---|---|
   | Train 内部配对（n=4646，全在样本内） | **0.227** | 0.408 | 0.312 | 0.335 | 0.276 |
   | Train×Valid 交叉配对（n=434，一侧在样本外） | **0.477** | **0.450** | 0.548 | 0.519 | **0.407** |
   | 纯 Valid 配对（n=15，全在样本外） | 0.647 | 0.626 | **0.593** | 0.694 | **0.459** |

   （表内为配对 δ 的 MAE。B→C 在样本内从 0.408 降到 0.312，越到样本外就越反过来：
   434 对上 0.450 → 0.548。即交互分支**记住了**上下文相关响应，而不是学会可迁移的交互。）

5. **机制诊断显示交互分支确实在工作，但不是这个任务需要的那种工作方式。**
   - 注意力**没有**坍塌到单个 token：每查询 token 的中位键数 289，熵 p50 ≈ 3.1 nats，
     有效 token 数 p50 ≈ 29.6（`CSOI_GLOBAL`）/ 35.5（`CSOI_REL`），约为上下文的 **13–15%**，
     p90 也只有 57–74；逐头最大权重 p50 仅 0.19/0.17，说明是**稀疏但分散**的读取。
   - 交互项幅度**很大**：`f(S|C) − f_additive(S|C)` 的逐分子平均绝对值 0.688（`CSOI_GLOBAL`）
     / 0.696（`CSOI_REL`）个训练尺度单位（≈0.83 / 0.84 目标单位），系统性均值 −0.28 / −0.17。
     也就是说交互分支把预测搬动了很多，但方向在未见上下文上不对。
   - 关系偏置是真被用上的：`CSOI_REL` 的 bias 标准差 0.76、|bias| 均值 0.59；
     **冻结权重把 bias 输出置零**后 Valid RMSE 从 0.7272 变到 0.7456（差 0.018，变差），
     说明学到的偏置本身有小幅正贡献，但不足以把 C→D 的整分子差值推出噪声。

6. **能力检查通过：交互机制确实会条件化于 C。** 合成 context-flip 任务上，
   `CSOI_ADD` 在纯加性规则 `R_ADD` 上归一化 RMSE 0.013（学得会），
   在翻转规则 `R_FLIP` 上 **1.005（等价于预测均值，相关系数 0.03）**；
   `CSOI_GLOBAL`/`CSOI_REL` 在 `R_FLIP` 上分别是 0.180 / 0.313（相关 0.984 / 0.951）。
   所以第 4 条的"记住了"不是实现缺陷，而是**优化到的解**。

7. **外部 GINE 坐标仍然显著更好**（本轮再次确认）：E 0.6406 vs B 0.7196（Δ = −0.0790，
   CI [−0.1414, −0.0174]）、vs C −0.0940、vs D −0.0866。这一条与父轮报告一致，
   说明"结构响应"这一族在 Valid 上没有竞争力，不是本轮引入交互才变差的。

8. **条件扩展没有触发。** Lipo 主对比不是 positive（既非 strong 也非 weak），
   按冻结规则 **不运行 ESOL**（`summary.json → decision.esol_extension_allowed = false`）。

9. **预算与执行。** 3 个本轮训练的臂共 **1.708 A100 GPU·h**（B 0.236 + C 0.580 + D 0.893；
   上限 5），2 卡并行，墙钟 **1.18 h**（10:57:42 → 12:08:07，上限 6 h）。13 项工程门全部通过；horizon 由 remote timing smoke
   （20.6 s/epoch）固定为 160 epoch，未缩减、未按臂调整。

---

## 1. 冻结结构与本轮改动（只做实现，不改父轮）

- 父轮 `task_response_composition_s0_v1` **只读复用**：切分枚举、parse 表、属性表、
  距离桶、train-only 标定器与配对目录全部走父轮模块（`csoi_common.check_signature()`
  记录 6 个父模块的 sha256；本轮 commit 见 `results/run_manifest.json`）。
- 缓存结构同一性用 **content digest** 校验：
  `46439b91dfd760255be4c53256f12687959d203b6740e55c0213d66476168825`，
  与父轮缓存数组内容逐元素一致（npz zip 时间戳使文件 hash 不可复现，故不用文件 hash）。
  `dataset_key="lipo"` 是显式字段：父轮 `load_raw_cache()` 对任何非 `"lipo"` 的字符串都会
  静默回落到 ESOL 缓存（本轮加了 raw-size 断言与 digest 断言把它变成硬错误）。
- 数据规模：4200 分子 / 33438 parses / 29283 片段 / 71004 帧 / 70794 个唯一对象
  （`analysis/structure_stats.json`）。对象按大小：k=1:76, k=2:1113, k=3:11545, k=4:58060；
  距离桶占比 [0.034, 0.086, 0.124, 0.133, 0.622, 2e−6]；对象重数 p50=1, p99=5。
- 片段 token 数（**就是交互的 key 预算**）：p50 358 / p90 520 / p95 567 / **p99 712** / max 1399，
  全库 10,362,794 个 token 条目。每 parse 的键数：S 侧 p50 1 / p99 125，C 侧 p50 366 / p99 700；
  全库注意力条目 **190.4 M**（Train 151.1 M / Valid 19.9 M）。
- **token 上限决策（协议 §5）**：p99 = 712 ≫ 128，且冻结的 128-token 分层抽样会在
  **91.1%** 的片段上保留不到 95% 的重数质量（远超 5% 的 blocker 门槛），因此**不做任何截断**，
  改用 **分块 ragged 注意力**（`attention_entry_chunk = 262144`）处理完整 token 集合。
  `analysis/token_stats.json` 记录 guard 全部统计量与"guard 未用于训练"的说明；
  13 项门里的 `guard` 检查断言：guard 触发、cap 确定性、抽样确实不足、训练使用全量。

## 2. 工程门（13 项，训练前必须全绿）

`code/csoi_checks.py` → `analysis/checks.json`，全部 `ok=true`：

| 检查 | 钉住的事实 |
|---|---|
| `inputs` | 33438 个 parse 的 S/C 原子集合、切键端点、桥键、帧内容（对 20 个分子独立重建子缓存逐帧比对） |
| `renumber` | 原子重标号不变性 |
| `tokens` | token 质量与片段重数一致；按片段键独立重建后条目一致 |
| `branches` | `CSOI_ADD` 组合双差恒为 0；`CSOI_GLOBAL` ≡ `CSOI_REL`（bias 置零时）；`dtot` 表语义 |
| `attention` | 快速分块实现 ≡ 逐 parse 朴素实现（**8.3e−17**）；chunk 大小不变性（0.0） |
| `batching` | batch ≡ 单条（**2.8e−17**）；分子顺序不变 |
| `batch` | 真实 mini-batch 损失有限、三个模块组梯度均非零 |
| `boundary` | split 互斥（3360/420/420）、标定器与配对与采样器只用 Train、训练视图里没有 Test |
| `init` | 三臂共享模块初始化哈希完全相同；GLOBAL 与 REL 的注意力初始化相同 |
| `params` | 69,080 / 130,969 / 132,973；ADD 的交互参数恰为 0；REL−GLOBAL = 2,004 = 关系分支 |
| `loss` | 逐分子归一化（每分子权重 1，与切数无关）、配对 δ 语义、τ² 与采样加权公式一致 |
| `guard` | 128-token guard 确定性、确实不足 95% 质量、训练使用全量 token |
| `determinism` | 同配置逐位一致；初始化不受全局 torch RNG 影响；换 model seed 会改变预测 |

**门抓到的真实 bug（训练前修掉）**：快速分块注意力的 softmax 归一化按 **parse** 分段，
而不是按 **query token** 分段——于是 `f(S|C)` 会依赖该片段有多少个 S 对象。
朴素参考实现当时差 0.105，逐项定位到 `den[seg]` 的分段粒度，改为 `seg = qs` 后
`vs_naive_reference_max_abs = 8.3e−17`。这个 bug 不会被双差/分支等效检查抓到，
只会被独立参考实现抓到——是"朴素参考"这一项门的价值所在。

## 3. 主表（Valid 整分子，Test 一律未读）

`results/main_table.csv`：

| arm | 角色 | 参数 | best epoch | Valid RMSE | Valid MAE | Pearson | R² | GPU·h |
|---|---|---|---|---|---|---|---|---|
| A `TRC_FACTOR_CUTRESP` | 父轮复用 | 154,983 | 100 | **0.7096** | **0.5393** | 0.815 | 0.660 | 0.193 |
| B `CSOI_ADD` | 本轮训练 | 69,080 | 147 | 0.7196 | 0.5631 | 0.807 | 0.651 | 0.236 |
| C `CSOI_GLOBAL` | 本轮训练 | 130,969 | 135 | 0.7346 | 0.5646 | 0.798 | 0.636 | 0.580 |
| D `CSOI_REL` | 本轮训练 | 132,973 | 113 | 0.7272 | 0.5576 | 0.802 | 0.643 | 0.893 |
| E `GINE_RAW_REF` | 外部坐标 | 387,974 | 128 | **0.6406** | **0.4674** | 0.851 | 0.723 | 0.070 |

训练集 RMSE（过拟合信号）：A 0.222 / B 0.573 / C 0.367 / D 0.422 / E 0.295。
交互臂把训练集拟合得好得多（C 比 B 低 0.21），但 Valid 反而更差——这是"记住而非泛化"的直接证据。

### 判据对比（配对 bootstrap，2000 次，单位：Valid 整分子 RMSE 差；正 = 后者更好）

| 对比 | 类型 | Δ | 95% CI | 判据 |
|---|---|---|---|---|
| B→C（交互 vs 加性） | **主对比** | −0.0150 | [−0.0640, +0.0323] | **no evidence** |
| C→D（关系偏置 vs 无） | 次级（仅当 B→C ≥ weak 才解释） | +0.0074 | [−0.0258, +0.0391] | no evidence |
| A→B（父轮 FACTOR vs 本轮加性） | 标定 | −0.0101 | [−0.0578, +0.0355] | no evidence |
| E→B / E→C / E→D | 外部坐标 | −0.0790 / −0.0940 / −0.0866 | 全部不含 0 | 结构族显著更差 |

`delta_task = 0.025·σ_train = 0.0303`；主对比最小可检测差 ≈ 0.048（由 bootstrap SE 反推）。
**结论：这一轮无法判定"保留 token 交互是否更优"，只能记录点估计为负、且设计分辨率不足。**

## 4. 支持分层：Valid 几乎全是"没见过的上下文"

`results/support.json`（支撑集合只在 Train 上定义）：

- Valid 3140 个 parse 中：S 见过 95.1%，**(C, 切键) 只见 5.8%**，三者组合 **0.0%**。
- 逐分子多数层：0（见过 S+见过 C）133 个 parse、1（见过 S+未见 C）2852、2（未见 S+见过 C）48、
  3（都未见）107。也就是说 **90.8% 的 Valid 解析落在"见过词、没见过上下文"**。

| 分层（分子多数票） | A | B | C | D | E |
|---|---|---|---|---|---|
| 0 见过 S / 见过 C（n=133 parse） | 0.3616 | 0.3513 | **0.2721** | 0.3945 | 0.4019 |
| 1 见过 S / 未见 C（n=2852，90.8%） | 0.7121 | **0.7232** | 0.7390 | 0.7314 | 0.6452 |
| 3 都未见（n=107） | 0.8729 | 0.7791 | 0.7652 | **0.6648** | **0.3194** |

两个可读的事实：(a) 在"见过上下文"的小分层里 `CSOI_GLOBAL` 最好（0.2722 vs 加性 0.3513），
与配对表在样本内的方向一致，但 `CSOI_REL`（0.3945）反而比加性差；
(b) 在"都未见"的分层里，GINE 是 0.3194 而显式结构臂是 0.66–0.87——结构族的泛化缺口集中在
**未见对象类型**上，而不是未见上下文（第 1 层 0.71–0.74 各臂差别不大）。
分层样本量小（133 / 2852 / 48 / 107 个 parse），只作方向性证据。

## 5. 配对层面（`results/pair_metrics.json`）

冻结的配对目录**只在 Train 内部构建**（4646 对 / 3399 transforms / 458 components），
因此"两个分子都在 Valid"的配对只有 **15** 对；与 Test 有任何关联的 675 对**全部丢弃、标签未读**。
三个范围分别报告：

| 范围 | n | 指标 | A | B | C | D | E |
|---|---|---|---|---|---|---|---|
| Train 内部（样本内） | 4646 | δMAE / δPearson / 方向 | **0.227** / 0.954 / 0.883 | 0.408 / 0.827 / 0.796 | 0.312 / 0.906 / 0.832 | 0.335 / 0.893 / 0.836 | 0.276 / 0.921 / 0.851 |
| Train×Valid 交叉（一侧样本外） | 434 | δMAE / δPearson / 方向 | 0.477 / 0.823 / 0.811 | **0.450** / 0.858 / 0.823 | 0.548 / 0.722 / 0.760 | 0.519 / 0.753 / 0.776 | **0.407** / 0.876 / 0.829 |
| 纯 Valid（全样本外） | 15 | δMAE / δPearson / 方向 | 0.647 / 0.804 / 0.600 | 0.626 / 0.747 / 0.667 | **0.593** / 0.743 / 0.667 | 0.694 / 0.636 / 0.733 | **0.459** / 0.872 / 0.667 |

"同 transform、不同 component"的 1695 对（更接近可迁移的替换）δMAE：
A 0.220 / C 0.288 / D 0.310 / E 0.278 / B 0.360——顺序与整分子一致（A、E 最好）。

**读法**：交互分支在**见过的**上下文上把 δ 误差从 0.408 压到 0.312（−24%），
但一旦另一侧分子来自未见上下文，误差从 0.450 反转成 0.548（+22%）。
这正是"整分子 Valid 上 no evidence、样本内配对上看得到收益"的统一解释。

## 6. 交互机制诊断（`results/interaction_diagnostics.json`）

| 诊断 | B `CSOI_ADD` | C `CSOI_GLOBAL` | D `CSOI_REL` |
|---|---|---|---|
| 每查询 token 键数 p50 | — | 289 | 289 |
| 注意力熵（nats）p50 / p90 | — | 3.11 / 3.79 | 3.40 / 4.10 |
| 有效 token 数 p50 / p90 | — | 29.6 / 56.9 | 35.5 / 73.5 |
| 有效 token / 键数（均值） | — | 0.133 | 0.155 |
| 逐头最大权重 p50 | — | 0.19 | 0.17 |
| 交互项 \|f−f_add\| 逐分子（尺度单位） | 0（构造上） | 0.688 | 0.696 |
| 交互项均值 | 0 | −0.280 | −0.169 |
| 关系 bias 标准差 / \|bias\| 均值 | — | — | 0.760 / 0.592 |
| 梯度范数（首/末 epoch）交互组 | 0 / 0 | 0.58 / 0.73 | 0.60 / 0.77 |
| 梯度范数（首/末）token 编码器组 | 0.07 / 3.64 | 0.32 / 1.36 | 0.33 / 1.63 |
| 冻结权重去 bias 后 Valid RMSE | — | — | 0.7272 → **0.7456** |

三个可读结论：

1. **注意力是稀疏但分散的**（有效 token ~13–15% 的上下文），没有坍塌到 argmax，
   也没有均匀摊平。关系偏置让注意力略微更分散（35.5 vs 29.6 有效 token）。
2. **交互项不是装饰**：它把预测搬动 0.69 个尺度单位，并且训练集拟合显著变好。
   所以"交互没有帮助"不是因为交互没学到东西，而是学到的东西在未见上下文上不成立。
3. **编码器在训练末段仍在快速漂移**（`grad_token_encoder` 从 0.07→3.64，B 臂最明显），
   说明 160 epoch 内共享编码器与加性头的联合优化远未稳定；这是"单 seed + 固定 horizon"
   设计下的已知限制，不是本轮可以事后调的东西。

> 诊断过程中的一处自查：最初把每查询熵除以键数再取指数，得到"有效 token ≈ 1.01"的假坍塌结论；
> 同时捕获的 `alpha_max` p50 = 0.19 与之矛盾，定位到分块捕获的 `qs` 是 chunk 内局部索引、
> 且熵不该再除以键数。修正后为表中的数字（代码里保留了 chunk 偏移 `q0` 的记录）。

## 7. 合成 context-flip 能力检查（`analysis/synthetic_flip.json`）

在**真实 token 机制**上构造三条规则，同一架构、同一预算（1536 例、400 步、seed 0、A100 单卡）：

| 规则 | 定义 | `CSOI_ADD` | `CSOI_GLOBAL` | `CSOI_REL` |
|---|---|---|---|---|
| `R_ADD` | y = a(S)（上下文无关） | 0.013 | 0.011 | 0.014 |
| `R_FLIP` | y = a(S)·sign(g(C))（换 C 翻号） | **1.005**（corr 0.03） | **0.180**（corr 0.98） | 0.313（corr 0.95） |
| `R_REL` | y = a(S)·sign(bucket(S)−bucket(C)) | 0.849（corr 0.54） | 0.280（corr 0.96） | 0.328（corr 0.95） |

（数值为相对"预测均值"的归一化 Valid RMSE，越低越好。翻转规则的两侧样本按中位数平衡，a(S) 已中心化。）

- **加性臂学不会非加性规则**（1.005 ≈ 完全失败），**交互臂学得会**（0.18 / 0.31）——
  证明交互路径在实现上是有效的，第 4 节的"记住"不是 bug。
- 关系翻转规则上 `CSOI_GLOBAL` 也不差于 `CSOI_REL`：内容注意力本就能读取两侧桶信息，
  显式关系偏置不是表达关系翻转的必要条件。这与第 0 节第 2 条（关系偏置无证据）互相印证。

## 8. 父轮表示的压缩诊断（`results/compression_diagnostics.json`）

用父轮 `FACTOR` 检查点取出全部 29283 个片段嵌入（128 维；前 8 个主成分只解释
0.093/0.047/0.042/…/0.028 的方差，说明这 128 维确实被铺开使用），
对配对 δ 做**按 transform 分组的 5 折 out-of-fold** 岭回归探针（4646 对，Test 相关配对全部排除）：

| 特征块 | 维数 | OOF R² |
|---|---|---|
| 压缩后的两侧嵌入 `[h_C, h_S(b)−h_S(a)]` | 256 | **0.659** |
| + 平均 S 嵌入 | 384 | 0.654 |
| + 显式乘积基 `h_S ⊙ h_C` | 512 | 0.638 |
| + 切键 one-hot | 518 | 0.638 |
| 只用切键 one-hot | 6 | −0.001 |

**结论是负面的但重要**：压缩表示已经线性携带了约 66% 的配对 δ 方差，
而**显式线性乘积基没有增加任何信息**（0.638 < 0.659，纯特征数惩罚）。
也就是说：父轮 128 维片段嵌入并没有丢"线性可读的交互信号"；
如果交互信息丢了，它是以非线性方式丢的，用线性探针查不出来。
这条诊断**不支持**"提前压缩丢失信息"这一动机假设的线性版本，必须如实记录。

## 9. 真实配对个案（`results/cases.json`）

自动规则：在"两个分子都在 Valid"的 15 对上，按交互臂（C 或 D）相对 `CSOI_ADD` 的
|δ 误差| 改善量排序，取前 20；不足时补入 Valid×Train 交叉对（标注 `scope`）。
不做人工挑选，正反两侧都列（`*_wins` / `*_loses`）。

由于 Valid 内配对只有 15 对，本节只作定性示例：交互臂在若干"同 transform、不同 C"的配对里
把 δ 误差压到接近 0，同时在别的配对里把符号弄反（与第 5 节的方向反转一致）。
个案不作为判决依据。

## 10. 预算、执行与产物

- 训练：`res`（a100-2，process backend，`res-gpu0` / `res-gpu1`，2 卡并行），
  驱动 550.163.01 / torch 2.5.1+cu124；全部经 `rr deploy` + `rr run`，无手工 SSH/sbatch/nohup。
- GPU：**1.708 h**（本轮训练的 B/C/D 三臂合计；A、E 是父轮复用，其 0.193/0.070 h 不计入本轮预算；
  上限 5 h）。墙钟 **1.18 h**（上限 6 h）。
  per-arm：B 14.1 min / C 34.8 min / D 53.6 min。
- 本地 CPU：缓存构建、13 项门、结构/token 统计、配对与诊断分析、报告。
- horizon：remote timing smoke 实测 20.6 s/epoch（最重的 `CSOI_REL`），按
  `min(160, floor(budget·3600/(3·epoch_s)))` 得 **160**，未缩减，三臂一致。
- 机器记录：`results/{main_table.csv, summary.json, support.json, pair_metrics.json,
  interaction_diagnostics.json, cases.json, compression_diagnostics.json, run_manifest.json}`、
  `analysis/{checks.json, structure_stats.json, token_stats.json, timing.json, synthetic_flip.json}`。
- **Test 从未读取**：训练器 `RunData` 只暴露 train/valid，`manifest.json` 记
  `test_policy: "no test labels read in this round"`；分析器只白名单读取 `train_*`/`valid_*`
  数组键，并在 675 个涉及 Test 的配对上直接丢弃。

## 11. 这一轮说明了什么、没说明什么

**说明了**

1. 在这套切分/枚举/预算下，**把对象 token 保留到交互之后再池化，没有带来整分子 Valid 收益**；
   点估计方向为负（−0.015），且本设计的分辨率（≈0.048）本来就覆盖不到协议设的 0.0303。
2. 交互分支**在样本内的上下文中确实更强**（配对 δMAE 0.408→0.312、同 transform 子集 0.360→0.288、
   见过上下文的 Valid 分层 0.351→0.272），但**在未见上下文上反转**（交叉配对 0.450→0.548）。
   这与"Valid 上 90.8% 的解析是未见上下文"的结构事实一致。
3. 显式关系偏置（桶对 + 总距离 + 切键）**不是**关系翻转的必要条件（合成任务里内容注意力同样能学），
   真实数据上它的整分子效应落在噪声内，但冻结权重去 bias 显示 bias 本身贡献约 0.018 RMSE。
4. 父轮的 128 维片段嵌入**没有丢线性可读的交互信号**（乘积基探针不增益）——
   "提前压缩丢信息"这个动机在本数据上找不到线性证据。

**没说明**

1. 不能说"交互无用"：单 seed、420 个 Valid 分子、CI 宽度 ±0.048，任何 <0.048 的效应都不可判。
2. 不能外推到 ESOL 或其它 split 比例：按冻结规则 ESOL 未运行。
3. 不能把配对层面的样本内优势当作可用收益：它没有转化为整分子验证集收益。

## 12. 若继续（不在本轮范围内，仅记录方向）

- **先解决分辨率**：多 seed（≥3）或用配对/分层加权的验证统计，把最小可检测差压到 0.03 以下，
  否则任何形式的交互都只能得到 no evidence。
- **把交互的正则化/容量对齐加性臂**：交互臂参数是加性臂的 1.9 倍却共用同一 horizon 与 lr，
  训练集 RMSE 0.367 vs 0.573 说明它抢了容量；等参数预算的对照才能回答"是不是交互本身没用"。
- **在训练分布内评估上下文相关响应**（例如按"见过的 (S,C,bond)"分层重采样验证集），
  才能把"记住 vs 泛化"从描述性观察变成可证伪的判据。

---

### 复现

```bash
# 本地（CPU）：数据、门、统计、合成任务、分析
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_stats.py
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_checks.py
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_synthetic.py --n 1536 --steps 400 --device cuda   # 或 --device cpu
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_analyze.py
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_diag.py
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_compress.py
.venv/bin/python tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/code/csoi_manifest.py

# 远端（A100）：每个臂一条
rr deploy res
rr run res CSOI_ADD_LIPO_R0    --pool res-gpu1 --result <round>/results/CSOI_ADD_LIPO_R0 \
   -- python <round>/code/csoi_train.py --run-id CSOI_ADD_LIPO_R0 --arm CSOI_ADD --dataset lipo --epochs 0 --device cuda
rr run res CSOI_GLOBAL_LIPO_R0 --pool res-gpu0 --result <round>/results/CSOI_GLOBAL_LIPO_R0 \
   -- python <round>/code/csoi_train.py --run-id CSOI_GLOBAL_LIPO_R0 --arm CSOI_GLOBAL --dataset lipo --epochs 0 --device cuda
rr run res CSOI_REL_LIPO_R0    --pool res-gpu1 --result <round>/results/CSOI_REL_LIPO_R0 \
   -- python <round>/code/csoi_train.py --run-id CSOI_REL_LIPO_R0 --arm CSOI_REL --dataset lipo --epochs 0 --device cuda
rr pull CSOI_ADD_LIPO_R0 && rr pull CSOI_GLOBAL_LIPO_R0 && rr pull CSOI_REL_LIPO_R0
```

`rr` 的运行前后元数据（host / node / pool / driver / torch / commit / exit code）逐臂抄录在
`results/run_manifest.json → runs[].rr`。
