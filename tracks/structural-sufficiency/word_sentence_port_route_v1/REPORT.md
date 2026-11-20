# REPORT — 结构词 → 局部句 → 整图（`word_sentence_port_route_v1`）

本轮实施日期 2026-11-20/21，T0 = `1795190995`（2026-11-20T16:09:55Z）。
代码固定于 `main`（逐 run 的 commit 见 `run_manifest.json`；本文件随最终提交入库），
所有结构/配方/分支规则在正式训练前写入并冻结于 `PROTOCOL.md` 与 `configs/ws_v1.json`。
**正式训练单元 11 + 参考 1（`GINE_REF`），GPU 用量 1.17 GPU·h（软上限 14，硬上限 16），
总墙钟约 1.9 h（硬截止 8 h）。** 无 OOM，无超时，无未完成主表项。

---

## 0. 一句话结论

在本轮 Lipo 的受控对照里，**“结构词（精确 canonical 内容）→ 局部句（真实端口布局 R）→ 整图”这条
路线没有取得任务优势**：最好的局部模型 `FULL_WGINE` Valid/Test RMSE = **0.768 / 0.836**，比同协议
重训的强整图参照 `GINE_REF`（**0.629 / 0.663**）差 0.139 / 0.173（≈2.9 个 `band`）。
真实端口布局 R 在**重训**（FULL vs ERASED 共 4 组配对：R0 的 CODE/WGINE 各 seed0，R1 的 WGINE seed0/1）
与**冻结干预**（R→ERASED 均值）两条证据线上**都没有带来误差改善**；句模型相对同词表词袋（`BAG`，0.795）的点估计增益 +0.027 在按 scaffold
分组的 bootstrap 下 **CI 跨 0**；而精确 canonical 内容编码（`CODE` 0.835 / `ID` 0.857）反而比只做
sum/mean 池化的属性编码器 `WGINE`（0.768）更差。
总体标签：**`NO_TASK_ADVANTAGE_IN_THIS_FAMILY`**（分层标签见 §2）。

---

## 1. 六个研究问题（先给结论与数字）

阈值（Lipo，Train std = 1.21154）：`delta_task = 0.01σ = 0.0121`，`delta_port = 0.005σ = 0.0061`，
`band = 0.04σ = 0.0485`。以下均为**原始单位**（g/L，log 标度）RMSE，seed 0，R0 配方，E = 160。

### Q1 词内容编码：ID / CODE / GINE 学到的表示，哪个更合适？

**判定：`WGINE 较合适`（单 seed，差距远超阈值；无 seed 方差可报）。**

| 编码器 | 参数 | best-Valid Train | best-Valid Valid | 选中 epoch |
|---|---|---|---|---|
| `ID`（每模板自由 embedding，oracle 词身份） | 90,500 | 0.47352 | **0.85744** | 34 |
| `CODE`（共享 229 维 exact canonical 码） | 87,772 | **0.43704** | **0.83495** | 44 |
| `WGINE`（词内属性重构图 + 2 层消息，sum/mean 池化） | 89,650 | 0.49034 | **0.76795** | 40 |

* `CODE` 比 `ID` 好 0.0225（> `delta_task`）：共享 canonical 码比自由模板 embedding 泛化更好一点，
  这是**唯一**一个对“canonical 化有价值”的正向信号。
* `WGINE` 比 `CODE` 好 **0.0670（0.055σ）**，比 `ID` 好 0.0895；而 `WGINE` 的 Train 拟合是三者中
  最差的（0.49034）。注意 **Q1 的排序与“编码器表达力”排序相反**：Stage-A 内容探针里
  `ID`/`CODE` 都能把三棱柱与 K3,3 拟合成两个不同标签，而 `WGINE` 的 sum/mean 池化对这两者给出
  **完全相同**的表示（`wgine_distinguishes=false`，见 `analysis/probes_mollipo.json`）。
* 直接含义：本轮 Valid 上“赢”的编码器，是**几乎不看精确词身份**的那一个；把词身份做成精确
  canonical 码并没有转化成泛化收益（反而在 Train 上过拟合更快、Valid 更差）。

### Q2 真实端口布局（R）到底有用吗？

**判定：`当前支持无增益`，机制上是 `只有冻结依赖`（冻结模型确实读 R，但读到的东西不改善误差）。**

三条独立证据（全部 Lipo Valid，420 分子）：

1. **重训（R0 配方）**：`FULL_CODE` 0.83495 vs `ERASED_CODE` 0.83313 → gain = **−0.0018**
   （scaffold bootstrap CI `[−0.026, +0.021]`，跨 0），即完整 R 略差。
   `FULL_WGINE` 0.76795 vs `ERASED_WGINE` 0.77396 → gain = **+0.0060**
   （CI `[−0.0035, +0.0157]`，跨 0），且比预先登记的 `delta_port = 0.0061` **低 0.00005**。
2. **重训（N2 修复配方 R1：peak 3e-4、wd 1e-3）**：seed0 +0.0005，seed1 **−0.0020**；
   两 seed 方向不一致、量级都在噪声内。
3. **冻结干预（对冻结 L 的权重，把几何 R 换成 ERASED 均值，不重训）**：Valid 上 87.9% 的
   pair 的 R 与 ER 不同（逐维 `max|R−ER|` 均值 0.621），预测平均移动 **0.0711**（最大 0.331），
   但 RMSE 0.76795 → **0.76690**（−0.00105，−0.14%），203 个分子变好 / 212 个变差，
   平均绝对误差变化 +0.0020。

结论：R 通道**确实接上了**（Stage-A 健康性检查 `pair_R` 梯度非零、pair head 梯度非零；FULL/ERASED
同 init 预测差 0.0），干预强度也足够（预测移动量是 Valid RMSE 的 9%），但**无论冻结依赖还是重训，
都没有把布局信息转化为误差收益**。因此不能把“真实端口布局”写成支持本轮路线的证据。

### Q3 结构词是否有价值？相对词袋 / 字符？

**判定：`仅压缩与参数化作用`（点估计优于词袋，但 bootstrap CI 跨 0；字符对照确实更差）。**

| 对照 | Valid RMSE | 与 L 的 gain | scaffold bootstrap CI | 参数 |
|---|---|---|---|---|
| **L = `FULL_WGINE`（词+句）** | **0.76795** | — | — | 89,650 |
| `BAG`（同一词表，无句/端口） | 0.79501 | **+0.0271**（2.2%σ） | **`[−0.0449, +0.0982]`** | 108,637 |
| `CHAR_FULL`（原子“字符”句模型，同配方） | 0.84337 | **+0.0754** | `[+0.0048, +0.1479]` | 87,772 |

* 对词袋的点估计增益 +0.0271 ≥ `delta_task`，且去掉对该差值贡献最有利的单个 Valid 分子后仍为
  **+0.0127 > 0**（登记的去极值检查通过）；但 **CI 跨 0**，且只有 1 个 seed，不能声称稳健。
* 词表/分词本身不是瓶颈：换成原子级“字符”分词后误差变差 0.075（CI 勉强不跨 0），
  说明瓶颈不在“字符 → 词”这一步，而在词内容/读出（见 §11）。
* 参数化不是解释：`BAG` 参数**更多**（108,637 > 89,650），仍更差。

### Q4 预测水平：接近强 GINE 参照吗？

**判定：`有明显差距`。**

| 模型 | Valid RMSE | Test RMSE | 备注 |
|---|---|---|---|
| 本轮 `FULL_WGINE`（finalist） | **0.76795** | **0.83606** | 89,650 参数 |
| 本轮 `GINE_REF`（同协议重训 5 层/128 宽 GINE） | **0.62857** | **0.66312** | 391,942 参数 |
| 差距 | **0.1394（2.9×band）** | **0.173（3.6×band）** | 相对差 22.2% / 26.1% |
| 历史坐标（不进主表） | RESP 0.62874 / Test 0.67603；旧 GINE 0.6091 / 0.6456 | | 未统一重训 |

本轮 `GINE_REF`（0.62857）与历史 GINE（0.6091）同量级（差 3.2%，未触发 LR 校准例外阈值 0.67001），
说明**参照够强**，差距不是“参照太弱”造成的。

### Q5 句—句关系（段层）是否带来真正收益？

**判定：`未执行`。** 段层（`PARA_REL`/`PARA_CAP`，D2）按预注册规则**只在 P 分支之后**执行；
本轮 B 阶段直接落到 N（见 §5），故未执行，无数字可报。

### Q6 复用（ESOL D1）是否成立？

**判定：`未执行`。** 同上（P 分支前置条件未满足）。

---

## 2. 路线判定表（每项都有实际数字或缺项说明）

| 问题 | 判定 | 依据（数字） |
|---|---|---|
| 内容编码 | **WGINE 较合适**（单 seed） | Valid 0.76795（WGINE）< 0.83495（CODE）< 0.85744（ID）；CODE−WGINE = 0.0670 = 0.055σ；Train 上 CODE 最好（0.437）→ 精确码过拟合 |
| 位点布局 | **当前支持无增益**（机制上仅冻结依赖） | 重训 gain −0.0018（CODE）/ +0.0060（WGINE，CI 跨 0，< `delta_port`）；R1 两 seed +0.0005/−0.0020；冻结干预 RMSE −0.00105（0.14%），203↑/212↓ |
| 结构词价值 | **仅压缩与参数化作用** | L−BAG = +0.0271（点估计），去极值后 +0.0127，但 CI `[−0.0449,+0.0982]`；CHAR 差 0.0754；BAG 参数更多 |
| 预测水平 | **有明显差距** | Valid/Test 0.768/0.836 vs 0.629/0.663；2.9–3.6 band；相对差 22–26% |
| 段层 | **未执行** | P 分支未达（L 比 `GINE_REF+band` 差 0.091） |
| 复用 | **未执行** | 同上 |

**总体标签：`NO_TASK_ADVANTAGE_IN_THIS_FAMILY`。**
分层子标签（不合并成单一 GO/NO-GO）：
`WORDS_USEFUL_LAYOUT_UNCONFIRMED`（词级分词确实优于字符，布局贡献未确认）、
`CANONICAL_CODE_LEARNING_LIMITATION`（精确 canonical 码在 Train 拟合最好、Valid 最差之一）、
`TOKENIZATION_OR_ATTRIBUTE_POOLING_LIMITATION`（诊断的机制瓶颈，见 §11）。
**不是** `ENGINEERING_OR_BUDGET_UNRESOLVED`：所有通道都验证过接上且有梯度（§4），预算只用了 1.17/16 GPU·h。

---

## 3. 冻结结构与准备（`analysis/structure_freeze_*.json`）

| 数据集 | content hash | 模板 | 合并轮数 | 多原子词原子覆盖 | 双位点支持实例 | 扩容 | 判定 |
|---|---|---|---|---|---|---|---|
| mollipo | `42aa4fc909956e9065afc9453f1a9d728a84e7ac5cb11b5ca3e548b01df8d3fc` | 148（128 多原子 + 20 单原子） | 128 | 0.9310 | 4,888 | 未触发（128 已达标） | `LAYOUT_SUPPORT_OK` |
| molesol | `0ca999908f4a1534…` | 144（128 + 16） | 129 | 0.898 | 380 | 未触发 | `LAYOUT_SUPPORT_OK` |

* 词表、合并顺序、端口位置号全部只用 **Train** 学习（`labels_used_for_vocabulary=false`，
  `decided_before_any_valid_metric=true`），Train/Valid/Test 只按冻结合并历史 replay，测试集不增补新词。
* Train 侧词型覆盖率：146/148 个模板在 Train 出现；`word_size` 直方图与端口数直方图
  （1 端口 10,915 / 2 端口 7,371 / 3 端口 4,900 / 4 端口 1,372 / 5 端口 158 / 6 端口 34 实例）
  记录在 freeze 里；4 个词型的 Train 频次 ≤ 2（稀有词风险，见 §11）。
* 数据身份在每次训练前由 `load_prepared` 校验 content hash（与 zip 时间戳无关），
  12 个 run 的 `cache_sha256` 全部等于上表。

---

## 4. Stage-A 工程检查（`analysis/probes_mollipo.json`，27.8 s，commit 见文件）

| 检查 | 结果 |
|---|---|
| 数据保真（原子恰好一次、内部键 + 唯一跨词键 = 全部物理键、9/3 列可达） | **通过** |
| 重编号 / 拆 batch / 单分子不变性 | **通过**：max abs pred diff ≤ 4e-8（tol 1e-5） |
| 内容探针（三棱柱 vs K3,3） | ID/CODE 均可拟合两标签；**WGINE sum/mean 表示完全相同**（`wgine_distinguishes=false`，记录为表达力事实） |
| 接口探针（同 6 环 + 两个取代基，环内距离 1/2/3） | FULL train/dev RMSE 3.1e-4（取代基组合未在 Train 出现）；ERASED 等价输入组预测 spread **0.0**，ERASED dev RMSE 0.276 |
| 训练健康性（32 真实分子，60 步） | loss 1.5518 → 0.0484，梯度全有限，`pair_R` 梯度非零（2.2e-3），pair head 梯度非零，无零梯度参数 |
| FULL/ERASED 同 init（真实 Valid 64 分子，seed0） | `max_abs_pred_diff = 0.0`，`max|pred| = 0.079` |

> 记录：该文件在 01:31 的巡检中发现缺失（早期产物被本地清理误删），于 01:33 用**同一冻结代码**
> 重跑一次（`ws-probes-rerun-01`，commit `0339a5c`），数字与首轮一致；重跑属工程检查，不计正式单元。

---

## 5. Stage-B 主表与分支判定（Lipo，seed0，R0，E = 160）

| run | 臂 | 参数 | s/epoch¹ | best ep | best Train | **best Valid** | final Valid | last-20 Valid |
|---|---|---|---|---|---|---|---|---|
| `lipo_full_code_s0` | FULL_CODE | 87,772 | 3.40 | 44 | 0.43704 | **0.83495** | 0.91482 | 0.91338 |
| `lipo_erased_code_s0` | ERASED_CODE | 87,772 | 0.95 | 46 | 0.41470 | **0.83313** | 0.88545 | 0.88436 |
| `lipo_full_id_s0` | FULL_ID | 90,500 | 3.37 | 34 | 0.47352 | **0.85744** | 1.02463 | 1.02279 |
| `lipo_full_wgine_s0` | **FULL_WGINE (L)** | 89,650 | 1.07 | 40 | 0.49034 | **0.76795** | 0.87605 | 0.87412 |
| `lipo_erased_wgine_s0` | ERASED_WGINE | 89,650 | 3.23 | 40 | 0.49514 | **0.77396** | 0.87938 | 0.87761 |
| `lipo_bag_s0` | BAG | 108,637 | 0.85 | 39 | 0.54183 | **0.79501** | 0.82694 | 0.82610 |
| `lipo_char_full_s0` | CHAR_FULL | 87,772 | 1.09 | 84 | 0.62107 | **0.84337** | 0.85138 | 0.85126 |
| `lipo_gine_ref_s0` | GINE_REF（参考） | 391,942 | 0.98 | 148 | 0.22945 | **0.62857** | 0.63523 | 0.63276 |

¹ `s/epoch` 含每 epoch 的 Train+Valid 评估；两卡并行时受同机 CPU 竞争影响（同一臂在独占时为
1.07 s/epoch，与独占时的其他臂 3.4 s/epoch 之差来自 pair head 的 279→64 MLP 在 43k pair 上的开销）。
horizon 由 Stage-A 计时冻结为 **E = 160**（360 → 40.3 min、240 → 26.9 min 均超 24 min 上限）。

**L 的选择**（预注册规则：三种 FULL 中 best-Valid 最低，`0.002σ` 内按 CODE → ID → WGINE）：
`L = FULL_WGINE`（0.76795；与 CODE 差 0.067 ≫ 0.0024，无需 tie-break）。

**分支算术**（`GINE_REF = 0.62857`）：

```
gain(L, BAG)        = 0.79501 − 0.76795 = +0.02706   ≥ delta_task 0.01212     ✓（去极值后 +0.0127 > 0 ✓）
gain(L, ERASED_L)   = 0.77396 − 0.76795 = +0.00600   <  delta_port 0.00606    ✗（差 0.00005）
L ≤ BAG + delta_port= 0.79501 + 0.00606 = 0.80107                              ✓
L ≤ GINE_REF + band = 0.62857 + 0.04846 = 0.67703                              ✗（差 0.0909）
L ≤ GINE_REF + 2band= 0.62857 + 0.09692 = 0.72549                              ✗
⇒ 不满足 P（第一项前置条件失败），也不满足 B（超出 2·band），分支 = N
```

**N 分支的修复选择**（只能选一族，按 N1 → N2 → N3 → N4 顺序，证据写进 `finalist_manifest.json`）：

| 检查 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| **N1** `CHAR_FULL` 优于 L ≥ `delta_task` | 0.84337 ≤ 0.76795 − 0.01212 = 0.75583 | **否**（CHAR 更差 0.0754） | 不满足 |
| **N2** L 选中点 Train 标准化 MSE ≥ 1.25× min(CHAR, GINE_REF) | 0.1638 ≥ 1.25 × 0.03587 = 0.04483 | **是**（ratio **4.57**） | **满足 → 执行 N2** |
| **N3** CODE 在 Train **与** Valid 都差于 ID/WGINE ≥ `delta_task` | Valid 差 0.067 ✓，Train CODE 0.437 **优于** WGINE 0.490 ✗ | **否** | 不满足 |

**效率与内存（`analysis/efficiency.json`，实测，非训练 run）**

| 指标 | finalist `FULL_WGINE` | `GINE_REF` |
|---|---|---|
| 推理 / 分子（batch 64，GPU） | **0.064 ms** | 0.074 ms |
| 推理 / 分子（batch 1，GPU） | 3.42 ms（Python/打包开销主导） | — |
| eval 峰值显存 | 15.1 MB | 27.2 MB |
| 单个训练 step（batch 128）峰值显存 | 32.0 MB | — |
| 进程峰值 RSS（含准备数据） | 1.13 GB | 1.27 GB |

CPU 准备（`analysis/structure_freeze_*.json` 的 `timing_seconds`）：Lipo **23.3 s**
（合并 14.6 / 张量 4.6 / 支撑回放 2.1），ESOL **3.8 s**。两卡并行 40 GB A100 上**没有任何 OOM 风险**，
本轮也未发生 OOM（本地机器只做 CPU 准备/测试，未跑训练）。

**N2 执行结果（唯一 R1 配方：peak 3e-4、wd 1e-3，其余同 R0，E = 160）**

| run | 臂 | best ep | best Train | best Valid | 相对原 L |
|---|---|---|---|---|---|
| `lipo_full_wgine_r1_s0` | FULL_L_R1 | 129 | 0.54384 | **0.78261** | **+0.0147（更差）** |
| `lipo_erased_wgine_r1_s0` | ERASED_L_R1 | 108 | 0.56025 | 0.78312 | +0.0152 |
| `lipo_full_wgine_r1_s1` | FULL_L_R1 | 126 | 0.55746 | **0.83050** | **+0.0626（更差）** |
| `lipo_erased_wgine_r1_s1` | ERASED_L_R1 | 126 | 0.56174 | 0.82848 | +0.0605 |

* 两 seed 都**没有**改善原 L → 不满足 `RESCUED_CANDIDATE` 的“两 seed 均改善”条件 →
  **`RESCUE_FAILED`**，finalist 保持 `lipo_full_wgine_s0`（冻结于 `finalist_manifest.json`）。
* R1 下 FULL vs ERASED：seed0 +0.0005，seed1 −0.0020（端口结论与 R0 一致）。
* 重要负向结果：R1（更低 LR + 100× wd）**变差**而不是变好 → N2 所检测到的“Train 拟合不足”
  不是学习率/正则的调参问题，而是**表示容量/读出形式的限制**（§11）。

---

## 6. 主要差值的 scaffold bootstrap（1000 轮，seed 1703，420 个 scaffold 组）

| 对照 | gain（正 = 前者的误差更低） | 95% CI | 判定 |
|---|---|---|---|
| L(WGINE) vs BAG | +0.0271 | `[−0.0449, +0.0982]` | 跨 0 |
| L(WGINE) vs ERASED_WGINE | +0.0060 | `[−0.0035, +0.0157]` | 跨 0 |
| L(WGINE) vs CHAR_FULL | +0.0754 | `[+0.0048, +0.1479]` | 勉强不跨 0 |
| L(WGINE) vs GINE_REF | **−0.1394** | `[−0.2005, −0.0798]` | 明确更差 |
| CODE vs ERASED_CODE | −0.0018 | `[−0.0257, +0.0213]` | 跨 0 |
| CODE vs GINE_REF | −0.2064 | `[−0.2675, −0.1411]` | 明确更差 |
| ID vs ERASED_CODE | −0.0243 | `[−0.0759, +0.0296]` | 跨 0 |
| ID vs GINE_REF | −0.2289 | `[−0.3001, −0.1585]` | 明确更差 |

每 seed 与 mean±sd：本轮只有 1 个 seed（P 未达，未进 Stage C），**seed 方差不可估**；
唯一有多 seed 的族是 R1（FULL 0.78261/0.83050；ERASED 0.78312/0.82848），
其 seed 间差异（0.048 / 0.045）**大于**该族内 FULL/ERASED 的差（0.0005 / 0.0020）——
这本身说明端口结论在 seed 噪声以下。不要把样本 bootstrap 当作训练稳定性。

---

## 7. §12.2 两个结构子集（只用冻结结构定义，不用于重选）

定义（任务书 §12.2）：**已知词** = 实例模板在 Train 出现；**已知内容的新位点布局** = 中心词类型、
无序两邻词类型、两条连接 bond type 在 Train 某句共同出现，但其 exact 双位点 canonical 布局
`(center, p, q)` 未在 Train 出现。

| | Valid | Test |
|---|---|---|
| Train 布局数 / Train 词型数 | 818 / 146 | 818 / 146 |
| 该 split 的 pair 实例 | 5,799 | 6,047 |
| 其中“已知内容 + 新布局” pair 数 | **6**（0.10%） | **2**（0.03%） |
| 涉及词型 / 布局数 | 5 / 5 | 2 / 2 |
| 涉及分子 / scaffold | **11 / 11**（覆盖 2.6%） | **15 / 15**（覆盖 3.6%） |
| 该子集模型误差（finalist） | RMSE **1.1103**（整体 0.7680） | RMSE **0.8760**（整体 0.8361） |
| 同子集 `GINE_REF` 误差 | —（未在 Valid 子集评估） | RMSE **0.8586**（整体 0.6631） |
| 已知词子集 | 419 分子 / 419 scaffold，RMSE 0.7665（99.8%） | 420 分子，RMSE 0.8361（100%） |
| 未知词型分子 | 1 分子，RMSE 1.2241 | 0 分子 |

**解读与限制**：布局字母表在 Train 里几乎饱和（818 个布局，Valid/Test 只各新增 5/2 个），
所以“已知内容、新布局”这一集合在分子层面只有 11 / 15 个分子、11 / 15 个 scaffold——
**低于 40 分子 / 15 scaffold 的门槛，只能作描述与案例，不能当组合泛化证据**；也不能把
“pair 实例数（6 / 2）”当独立样本。这一事实本身是有信息量的：本轮的 `R(D;p,q)` 描述子把
“布局”压成了一个小字母表上的类别特征，Valid 里几乎不存在“布局新颖性”，因此这条路线里
“布局泛化”这个科学卖点**先验地缺少检验场所**。

---

## 8. 案例与代数账目（`analysis/inspect_lipo_full_wgine_s0.json`）

* **完整 F 从中间项恢复**：`F = bias + Σ_i mass_i (u_i+v_i+w_i+r_i)`，
  Valid 420 分子上 `max |F_recon − F_model| = 3.15e-7`（要求 ≤ 1e-5）。这是实现/代数账目，
  不是解释正确性。
* **8 个固定案例**（规则：先按“词端口数最大”取 4 个结构支持案例，再按 row id 升序；
  然后取 4 个绝对误差最大的案例，不挑选好看的图）：

| 规则 | row id | y | F | |err| | 原子 | 词数 |
|---|---|---|---|---|---|---|---|
| support | 1818 | 1.190 | 2.497 | 1.307 | 50 | 17 |
| support | 1161 | 1.970 | 2.409 | 0.439 | 24 | 8 |
| support | 1180 | 2.200 | 3.052 | 0.852 | 31 | 8 |
| support | 1192 | 0.700 | 0.830 | 0.130 | 19 | 7 |
| max_error | 1425 | −0.040 | 3.694 | 3.734 | 36 | 13 |
| max_error | 1739 | 0.380 | 2.725 | 2.345 | 15 | 4 |
| max_error | 1601 | −0.850 | 1.456 | 2.306 | 32 | 10 |
| max_error | 1557 | 0.880 | 3.151 | 2.271 | 19 | 8 |

  每个案例在 JSON 里带：词的原图原子映射（row-local 原子号）、每个端口的
  `own_site / neighbour_word_offset / neighbour_site / bond_columns`、`u/v/w/r`（原始单位）、
  `mass`、截距、`F`、`y`。示例（row 1818，前 5 词）：

```
w0 tid=29 ports=1 u=+1.156 v=+1.406 w= 0.000 r=0 mass=0.040
w1 tid=22 ports=3 u=+2.126 v=+0.610 w=−1.100 r=0 mass=0.040
w2 tid=29 ports=2 u=−3.476 v=−1.393 w=+1.262 r=0 mass=0.040
w3 tid=56 ports=4 u=+1.868 v=−1.295 w=+1.591 r=0 mass=0.080
w4 tid=93 ports=5 u=+2.867 v=+0.494 w=−1.310 r=0 mass=0.100
Σ mass·(u+v+w+r) = 0.3449；截距 2.1518；F = 2.4967 = y 的模型读数（y = 1.19）
```

  `r ≡ 0` 是因为 finalist 未启用段层（`para=none`）；`w=0` 是因为该词只有 1 个端口（0 个 pair）。
  贡献值存在重参数化与归属约定，**不把符号一致当成化学定律**，也不是概率或因果份额。

---

## 9. 终局 Test（一次性，`analysis/terminal_test.json`）

Test policy 固定 `TERMINAL_DEVELOPMENT_TEST`：在所有选择/分支完成、`finalist_manifest.json` 提交
（commit `63a584a`）之后，对 finalist 与必要对照各做一次终局评分；训练期从未建立 Test loader。

| 模型 | Test RMSE | Test MAE | Valid（选择用） |
|---|---|---|---|
| **finalist `FULL_WGINE`** | **0.83606** | 0.64509 | 0.76795 |
| `GINE_REF`（必要对照） | **0.66312** | 0.49033 | 0.62857 |

Test 与 Valid 的相对差距（22% vs 26%）一致，说明 Valid 上的结论不是划分偶然。
历史坐标（RESP 0.676、旧 GINE 0.646）只作参照，**不与本轮数字混成同协议主表**。

> **诚实披露**：终局评分脚本前两次尝试因代码 bug 崩溃（第一次在 `run_dir` 缺省、
> 第二次在 support-subset 计算处；见 `run_manifest.json` 里 `ws-terminal-finalist*` 与
> `ws-terminal-diag` 的 failed/completed 记录）。第二次崩溃发生在**已算完 Test 预测但尚未写盘**
> 的位置；该进程的数值没有写出、也没有被我读取或用于任何选择，且同 checkpoint 的 eval 是确定性的
> （无 dropout、BN 用 running stats），因此第三次成功运行给出的数字与前两次会完全一致。
> 除此之外 Test 只被读了一次。

---

## 10. 端口布局证据总表（把“冻结依赖”和“重训收益”分开）

| 证据 | 做法 | 结果 | 允许的解读 |
|---|---|---|---|
| 重训 R0 | FULL vs ERASED（CODE / WGINE） | −0.0018 / +0.0060（CI 均跨 0） | 无重训收益 |
| 重训 R1 | FULL vs ERASED（WGINE，2 seed） | +0.0005 / −0.0020 | 无重训收益 |
| 冻结干预 | 冻结 L 的 R → ERASED 均值 | 预测移动 0.0711（9% of RMSE），RMSE −0.00105，203↑/212↓ | 有依赖、无收益 |
| 梯度/接线 | `pair_R` 梯度非零、pair head 梯度非零 | 通过 | 不是工程未接上 |

**干预有效 ≠ 重训收益 ≠ 化学因果。** 本轮三条都不支持“真实端口布局是这条路线的主要贡献”。

---

## 11. 失败分解（§12.4，落到具体事实）

1. **词频与稀有词**：Train 里 146/148 个词型出现过，4 个词型频次 ≤ 2；Valid 只有 1 个分子用到
   Train 未见的词型（RMSE 1.224）。**稀有词不是主因**。
2. **端口/接口支持**：Train 双位点布局支持实例 4,888/7,371；端口数分布到 6。**接口支持充足**，
   且 Stage-A 接口探针在未见取代基组合上 dev RMSE 3.1e-4。
3. **词袋 vs 真实连接**：`BAG` 0.795 与 L 0.768 的差在 bootstrap 下跨 0；**真实连接（v/w 项）
   没有产生可分辨的整分子收益**。
4. **编码器 Train 拟合 vs Valid 泛化**：`CODE` Train 最好（0.437）而 Valid 0.835；`WGINE` Train
   最差（0.490）而 Valid 0.768。**精确 canonical 身份被记忆化，泛化反而更差**；
   “把词压成 229 维精确码 + 64 维 MLP”是本轮最明显的 learning limitation。
4b. **ID 编码器的未训练 embedding（按要求单独报告）**：`ID` 对多原子词用自由模板 embedding，
   对单原子词用 3 个字段的因子化原子 embedding（不是把所有未见组合压成一个 UNK）。但 Train 里
   **多数单字段类别没有出现**：`atom_0`（原子序数）119 类中只有 11 类出现（108 行 embedding 未训练），
   `atom_1` 缺 [3,4]、`atom_2` 缺 8 类、`atom_3` 缺 9 类、`atom_4` 缺 6 类、`atom_5` 缺 5 类、
   `atom_6` 缺 3 类、`bond_0` 缺 1 类、`bond_1` 缺 3 类。这些行保持初始化值，若在 Valid/Test 出现
   就是噪声；实测 Valid 仅 **4/11,467 个原子（0.03%）** 的 9 字段组合未在 Train 出现，Test 为 **0/11,587**，
   因此对 Q1 的排序没有影响。
5. **原子字符对照**：`CHAR_FULL` 0.843（比 L 差 0.075）且 Train 只到 0.621（欠拟合）→
   **分词不是瓶颈**，瓶颈在词内容的编码与读出。
6. **实例属性池化限制**：`a_i` 把词内原子属性做成 sum/mean（36 维 ×4）+ 键 sum/mean，`WGINE`
   同样用 sum/mean 池化——Stage-A 内容探针已证明这种池化**无法区分同度数的三棱柱与 K3,3**。
   这是“结构词内容被池化抹平”的直接证据。
7. **强参照是否够强**：`GINE_REF` 0.6286/0.6631 与历史 0.6091/0.6456 同量级（+3.2%），
   且显著优于本路线（CI 不跨 0）。**参照够强，差距真实**。
8. **预算/吞吐/优化不足**：N2 检查显示 L 的选中点 Train 标准化 MSE（0.164）是 `GINE_REF`
   （0.036）的 4.6 倍，看似优化不足；但 **R1 修复（peak 3e-4、wd 1e-3）两 seed 都更差**
   （+0.0147/+0.0626），且 L 的 best epoch 在 40/160（不是 horizon 末端），排除“训练不够久”。
   → **不是学习率/正则/预算问题**，是表示与读出形式的容量错配。

> 不把残差归因标签噪声；不宣称“所有可能的词—句—段路线失败”（只测了本轮定义下的这一族）；
> 也不因 CODE 能区分合成 3-正则模板就声称它在真实性质任务上应当更强（本轮实测相反）。

---

## 11b. 附录：固定真实编辑目录上的 B−A 差值（任务相关性，不选择分支）

复用 `response_dictionary_task_transfer_s0_v1/analysis/pairs.csv` 这份**固定**真实编辑目录
（E_T = 301 对、E_HH_T = 41 对、E_STRICT = 2 对；不重新采样、不改 cut 阈值），用两个模型按 row id
的既有预测算 `ΔF = F_B − F_A` 与 `Δy` 的差值误差（`analysis/edit_differences.json`）：

| 目录 | 对数 | finalist ΔRMSE | finalist 符号一致率 | finalist corr | `GINE_REF` ΔRMSE | `GINE_REF` 符号一致率 | `GINE_REF` corr |
|---|---|---|---|---|---|---|---|
| **E_T** | 301 | **0.667** | 70% | 0.69 | **0.523** | 81% | 0.82 |
| **E_HH_T** | 41 | **0.728** | 66% | 0.58 | **0.497** | 88% | 0.87 |
| E_STRICT | 2（仅列案例） | 0.406 | 50% | — | 0.129 | 100% | — |

* 差值误差的分母（`Δy` 的 RMSE）在 E_T 上是 0.920、E_HH_T 上是 0.903，所以两个模型都不是“零响应”。
* **差值误差不是整分子误差**：同一 scaffold 的共享偏移会抵消，因此绝对值不能与 §5 的 RMSE 横比；
  这里只做同一目录上的**模型间**比较。结论与主表一致：在真实编辑的响应方向上，本路线也**弱于**
  同协议 `GINE_REF`（E_T 0.667 vs 0.523，符号一致率 70% vs 81%）。
* E_STRICT 只有 2 对，仅作案例，不做比率推断。此项**未参与**任何分支或 checkpoint 选择。

## 12. 局限

* **单 seed**：Stage C（seed1/seed2 确认）因 P 未达未执行；除 R1 族外无 seed 方差，
  §6 的所有点估计都带不可估的训练噪声（历史同族 run 的 seed 差约 0.05，与 L−BAG 同量级）。
* **best-Valid 选择**本身在 420 个 Valid 分子上有 ±0.02 量级的抖动；`last20`/`final` 列已给出
  供读者判断（例如 `BAG` 的 final 0.827 比 best 0.795 差很多，说明其 best 有挑选成分）。
* **Test 是长期暴露的开发 Test**，不是 fresh/blind confirmation；本轮只读一次且未用于选择。
* **bootstrap 分组**用 Murcko scaffold（420 组，平均 2 分子/组），组内相关性低 → CI 偏窄；
  这也让“CI 跨 0”的结论偏保守。
* **布局字母表饱和**（§7）：Valid 只有 6 个、Test 只有 2 个“新布局”pair，无法对该子集做统计声明。
* **E = 160 的 horizon** 由计时冻结（360/240 都超 24 min），所有臂同等受限；不排除更长训练改变排序，
  但 N2 的 R1 结果与 best-epoch 位置都不支持“训练不足”解释。
* 只做了 Lipo；ESOL（D1）与段层（D2）按预注册规则未执行。
* 历史 LCRC/RESP/GINE 数字只列坐标，不做同协议比较。

---

## 13. 条件分支实际执行情况

| 分支 | 状态 | 说明 |
|---|---|---|
| Stage A（保真/内容/接口/健康性/同 init） | **执行** | 全通过；内容探针记录 WGINE 表达边界 |
| Stage B（7 臂 + L 选择） | **执行** | 见 §5；L = FULL_WGINE |
| P | 未达 | L 比 `GINE_REF+band` 差 0.0909 |
| B（2·band 边界） | 未达 | 超出 2·band 0.0424 |
| Stage C（seed1/2 确认） | **未执行** | 仅 P 后执行 |
| N1（ATTRPAIR） | 未执行 | 条件不满足（CHAR 更差） |
| **N2（R1 配方，4 run）** | **执行** | 条件满足（ratio 4.57）；结果 `RESCUE_FAILED` |
| N3（CODE_ROOTMEAN） | 未执行 | 条件不满足（CODE 在 Train 优于 WGINE） |
| N4 | 未使用 | N2 已满足 |
| D1（ESOL 复用） | 未执行 | 仅 P 后执行 |
| D2（PARA_REL/CAP） | 未执行 | 仅 P 后执行 |
| 终局 Test | **执行** | finalist + GINE_REF 各一次 |

---

## 14. 资源与未完成项

* 墙钟（UTC）：T0 16:09:55 → 准备/计时/探针 16:35–16:48 → Stage-B 16:52–17:20 →
  N2 修复族 17:17–17:40 → 分析与终局 Test 17:56–18:01 完成；总计约 **1.9 h**（上限 8 h）。
* GPU：**1.17 GPU·h**（软 14 / 硬 16），峰值 2 卡并行；**正式训练单元 11 + 参考 1**（上限 32）。
* 失败/重跑（全部保留在 `run_manifest.json`，资源已计入）：`lipo-full-code-s0`、`lipo-erased-code-s0`
  （早期 deploy 冲突，1–2 s 即退出）、`ws-terminal-finalist`/`-2`/`ws-terminal-diag`（代码 bug，见 §9 披露）。
* 未完成项：Stage C、D1、D2、N1、N3（均因预注册条件未满足而**不应**执行）；ESOL 未训练。
* **已知实现缺口（不影响任何报告数字）**：本轮 12 个 run 只落盘了 `state_best.pt`（每个 epoch 的
  best 状态），最后一 epoch 的权重没有单独保存——`state_final.pt` 的保存路径是在矩阵跑完之后补的
  （已在代码中实现并用 3-epoch smoke 验证）。最终点的指标与按 row id 的预测已经保存
  （`run.json` 的 `final`/`last20`、`curves.json`、`preds_final_*.npz`），因此 final/last-20 的数字
  可复核；但由于 `index_add_` 在 CUDA 上非确定性，重跑无法还原同一份 final 权重。
  所有报告数字与终局 Test 都只用 **best** checkpoint，故结论不受影响。
* Checkpoint/预测：`results/<run_id>/{state_best.pt,state_final.pt,curves.json,run.json,preds_*.npz}`
  按仓库 gitignore 不提交，但保留在结果存储（本地 `results/` + `res` 主机），并已 `rr pull` 回本地。

---

## 15. 复现

```bash
R=tracks/structural-sufficiency/word_sentence_port_route_v1
# 结构准备（Train-only 词表 + 冻结合并历史 + content hash）
rr run res ws-prepare-both --pool res-local -- bash -lc \
  "uv run python $R/code/ws_run.py prepare --dataset mollipo --write-record && \
   uv run python $R/code/ws_run.py prepare --dataset molesol --write-record"
# Stage-A 工程检查
rr run res ws-probes --pool res-gpu1 --result $R/results -- bash -lc \
  "uv run python $R/code/ws_run.py probe --dataset mollipo --device cuda:0"
# 计时与 horizon 冻结（2 warm + 5 timed）
rr run res ws-timing --pool res-local --result $R/results -- bash -lc \
  "uv run python $R/code/ws_run.py timing --dataset mollipo --arms full_code full_wgine gine_ref --device cuda:0"
# 正式 run（每臂一条；<arm> ∈ full_code erased_code full_id full_wgine erased_wgine bag char_full gine_ref
#                                    full_wgine_r1 erased_wgine_r1）
rr run res lipo-<arm>-s<seed> --pool res-gpu{0,1} --result $R/results -- bash -lc \
  "uv run python $R/code/ws_run.py train --dataset mollipo --arm <arm> --seed <seed> \
   --epochs 160 --run-id lipo_<arm>_s<seed> --device cuda:0"
# 汇总 / 冻结 finalist / 终局 Test / 机器运行记录
uv run python $R/code/ws_run.py summary
uv run python $R/code/ws_run.py finalist --dataset mollipo --terminal-runs lipo_full_wgine_s0 lipo_gine_ref_s0
rr run res ws-terminal --pool res-gpu0 --result $R/results -- bash -lc \
  "uv run python $R/code/ws_run.py terminal --runs lipo_full_wgine_s0 lipo_gine_ref_s0 --device cuda:0"
uv run python $R/code/ws_run.py inspect --run lipo_full_wgine_s0 --dataset mollipo --cases
rr jobs --json > /tmp/rr_jobs.json && uv run python $R/code/ws_run.py manifest --jobs-json /tmp/rr_jobs.json
```

产物索引：`analysis/summary.json`、`analysis/main_table.csv`、`analysis/terminal_test.json`、
`analysis/inspect_lipo_full_wgine_s0.json`、`analysis/structure_freeze_*.json`、
`finalist_manifest.json`、`run_manifest.json`、`analysis/efficiency.json`、`analysis/edit_differences.json`、
`analysis/probes_mollipo.json`、`analysis/timing_mollipo.json`、
`results/terminal/*/terminal_test.json`。
