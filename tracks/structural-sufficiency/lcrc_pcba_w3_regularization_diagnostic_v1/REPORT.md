# REPORT — LCRC-W3 正则化诊断 v1

**轮次**：`lcrc-pcba-w3-regularization-diagnostic-v1`
**问题**：历史 LCRC-W3 在 MolPCBA 上增加参数后 Train AP 显著提高、Validation AP 停滞，
这是否至少部分源于**正则化配方没有适配模型规模**？
**执行**：两个固定 run（`W3_WD10`、`W3_DO20`），历史 LCRC-A-W3（2,120,844 参数）结构完全不变，
seed 0，40 epochs，每个 epoch 训练结束后对 **Train / Validation / Test 三个完整划分**做全量推理与评分
（120 个 epoch×split 记录，无缺失）。
**最终判定**：两个 run 都是 **`NO_CLEAR_GAIN`**（按 `PROTOCOL.md` §10 的预注册规则与 §12.1 的固定算式）。

> 一句话结论：把 weight decay 放大 10×（0.001→0.010）或把 dropout 翻倍（0.10→0.20），
> 都没有把 W3 的 Validation 拉起来——Best Valid AP 分别 **+0.00047 / −0.00017**，
> Valid AP@40 分别 **+0.00021 / +0.00197**，都远低于本轮预注册的 +0.005 / +0.003 门槛。
> 同时 Train AP@40 下降 0.011（WD10）与 0.042（DO20），
> 即**正则化确实抑制了拟合，但没有转化为可测的 Validation 泛化**。
> 单 seed、单旋钮、两个取值，不足以据此继续投入正则化配方调优；也不足以证明配方已经用尽。

---

## 0. 主结果表

| Model | Params | Best epoch | Train AP@best | Valid AP best | Test AP@best | Train AP@40 | Valid AP@40 | Test AP@40 | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **W3 historical** (read-only) | 2,120,844 | 25 | *n/a* | 0.263670 | 0.251822¹ | 0.705498 | 0.259004 | 0.249720¹ | 1.388 |
| **W3_WD10** (dropout 0.10, wd 0.010) | 2,120,844 | **19** | 0.575921 | **0.264135** | 0.254055 | 0.694325 | **0.259214** | 0.249636 | 2.181 |
| **W3_DO20** (dropout 0.20, wd 0.001) | 2,120,844 | **19** | 0.556474 | **0.263498** | 0.257691 | 0.663519 | **0.260978** | 0.255280 | 1.958 |
| W2 historical (reference only) | 994,324 | 24 | *n/a* | 0.265273 | 0.253617 | 0.581875 | 0.261867 | 0.250986 | 1.395 |

¹ 历史 W3 的 Test AP 在本轮以**一次只读推理**补齐（`analysis/historical_test.json`，checkpoint SHA-256
校验一致，Valid AP 复算误差 3.3e-6 / 7.5e-8）。这不是新的基线训练。
历史 W3/W2 的 `Train AP@best` 在历史记录中不存在（只记录了 epoch-40 Train AP），标记缺失，未重训。

**对照差值（Validation 为主指标）**

| 比较 | ΔBest Valid | ΔValid@40 | ΔBest vs W2 | Δ@40 vs W2 | 标签 |
|---|---:|---:|---:|---:|---|
| `W3_WD10` − 历史 W3 | **+0.000465** | **+0.000210** | −0.001138 | −0.002653 | `NO_CLEAR_GAIN` |
| `W3_DO20` − 历史 W3 | **−0.000173** | **+0.001974** | −0.001776 | −0.000889 | `NO_CLEAR_GAIN` |
| `W3_DO20` − `W3_WD10` | −0.000638 | +0.001764 | — | — | — |

Test（`EXPOSED_TEST_EXPLORATORY`，**不参与任何 checkpoint 选择**）：`W3_WD10` Test@best +0.00223、
Test@40 −0.00008；`W3_DO20` Test@best **+0.00587**、Test@40 **+0.00556**（均相对历史 W3）。

---

## 1. 五个问题

### Q1. 更强 weight decay（0.001 → 0.010）是否确实提高了 W3 的 Validation 泛化？

**OBSERVED（实测）**

| 指标 | 历史 W3 | W3_WD10 | Δ |
|---|---:|---:|---:|
| Best Valid AP (epoch) | 0.263670 (25) | 0.264135 (19) | **+0.000465** |
| Valid AP@40 | 0.259004 | 0.259214 | **+0.000210** |
| Valid BCE@40 | 0.049079 | 0.048574 | −0.000505 |
| Train AP@40 | 0.705498 | 0.694325 | **−0.011173** |
| Test AP@best / @40 | 0.251822 / 0.249720 | 0.254055 / 0.249636 | +0.002233 / −0.000084 |
| 超过历史 Best 的 epoch | — | 只有 epoch 19（1 个 epoch，最长连续 1） | — |

- 逐任务（Valid，best epoch，相对历史 W3 e25）：127 个可评分任务中 65 改善 / 62 变差，
  mean Δ +0.00047、median Δ +0.00016；正支撑 ≥20 的 107 个任务里 55 改善（mean +0.00149），
  稀疏层（5–19 正例，18 个任务）mean −0.00557。
  正增益最集中的 10 个任务贡献了全部正增量的 51%。
- 20–40 epoch 的 Validation 斜率 −8.5e-5/epoch；Test 斜率 −1.7e-4/epoch。
- 末 10 个 epoch 的 Validation AP 标准差 0.00124（历史 W3 为 0.00088），
  即 +0.0005 的差异**小于本轮的 epoch 间噪声**。

**INTERPRETATION**：不能认为 wd=0.01 提高了泛化。它连“明显抑制拟合”都没有做到：
Train AP@40 只降了 0.011。原因是 AdamW 的解耦衰减在本配方下预算很小——
40 epoch 的 LR 积分 Σlr = 17.003（109,520 步），
`wd=0.010` 对应总乘性收缩 `exp(−0.010×17.003) ≈ 0.844`（约 15.6%），
`wd=0.001`（即历史配方）只有 1.7%。也就是说“10× weight decay”在本 schedule 下
仍然只是一个十几 % 量级的权重收缩，而不是强正则。

**NOT ESTABLISHED**：不能从这一组数据推断“weight decay 对 W3 无效”。
只测了一个取值（0.010），且该取值对应的衰减预算本身很小。

### Q2. 更高 dropout（0.10 → 0.20）是否确实提高了 W3 的 Validation 泛化？

**OBSERVED（实测）**

| 指标 | 历史 W3 | W3_DO20 | Δ |
|---|---:|---:|---:|
| Best Valid AP (epoch) | 0.263670 (25) | 0.263498 (19) | **−0.000173** |
| Valid AP@40 | 0.259004 | 0.260978 | **+0.001974** |
| Valid BCE@40 | 0.049079 | 0.047838 | −0.001241 |
| Train AP@40 | 0.705498 | 0.663519 | **−0.041979** |
| Train AP@best epoch | — | 0.556474（历史 best epoch 无该记录） | — |
| Test AP@best / @40 | 0.251822 / 0.249720 | 0.257691 / 0.255280 | **+0.005869 / +0.005559** |
| 超过历史 Best 的 epoch | — | **0 个** | — |

- 逐任务（Valid@40，相对历史 W3 e40）：127 个任务中 **71 改善 / 56 变差**，
  mean Δ +0.00197、median Δ +0.00164；正支撑 ≥20：**61/107 改善，mean +0.00602、median +0.00239**；
  稀疏层（5–19，18 个任务）mean **−0.02396**（8/18 改善）——宏观均值被稀疏任务拉回。
- 逐任务（Valid，best epoch）：68 改善 / 59 变差，mean −0.00017；≥20 层 mean +0.00245（57/107）。
- 20–40 epoch 的 Validation 斜率 +5.9e-5/epoch（基本持平），Test 斜率 −4.7e-5/epoch。
- 末 10 个 epoch 的 Validation AP 标准差 0.00081，Range 0.0025；
  +0.00197 的 @40 差异约为该噪声的 2.4σ（单 seed，不能当显著性）。

**INTERPRETATION**：dropout 翻倍确实压低了拟合（Train AP@40 −0.042，Valid CE 也下降），
但 **Best Valid AP 没有提高（−0.00017，且从未超过历史 W3 的峰值）**；
唯一的正向是 “@40 时更高、更稳”，而且这个正向主要来自 ≥20 正例的 assay，
被 18 个稀疏 assay 的平均 −0.024 抵消了一部分。
Test 侧的改善（+0.0056）大于 Valid 侧的改善，这在方向上是“没有把 Validation 过拟合掉”，
但 Test 在本轮是探索性暴露，不能用于结论或选择。

**NOT ESTABLISHED**：不能声称 dropout 0.20 提升了泛化；也不能排除 0.15 或 0.25 更好（本轮不搜索）。

### Q3. 增益是否覆盖大部分有充足正例的 assay？

**不覆盖。**

| run | 比较点 | 支撑 ≥20 正例（107 个） | 支撑 5–19（18 个） | 支撑 <5（2 个） |
|---|---|---:|---:|---:|
| W3_WD10 | Valid best | mean +0.00149，median +0.00016，**55/107 改善** | mean −0.00557，9/18 | mean +0.00019，1/2 |
| W3_WD10 | Valid@40 | mean +0.00090，median +0.00179，61/107 | mean −0.00359，7/18 | mean −0.00267，0/2 |
| W3_DO20 | Valid best | mean +0.00245，median +0.00099，57/107 | mean −0.01618，9/18 | mean +0.00331，2/2 |
| W3_DO20 | Valid@40 | mean +0.00602，median +0.00239，**61/107 改善** | mean −0.02396，8/18 | mean +0.01878，2/2 |
| W3_DO20 | Test best | mean +0.00363，median +0.00395，60/108 | mean +0.02723，10/18 | mean −0.13739，0/1 |

- 两个 run 在 ≥20 支撑层的改善任务数都**没有过半**（55/107、57/107；@40 时 61/107 刚过半）。
- 稀疏 assay（Valid 正例 5–19）在两个 run 里都是**净负**，DO20 在 @40 时 −0.024，
  说明“macro AP 的小幅提升”并不是全体 assay 普遍改善。
- 正增量高度集中：前 10 个任务占全部正增量的 51%（WD10）/ 46%（DO20）；
  逐任务 Δ 与 Valid 支撑度的 Spearman 仅 +0.054（WD10 best）、+0.208（WD10 @40）、
  −0.085（DO20 best）、+0.042（DO20 @40），没有系统性“只有大 assay 受益”的模式，
  但也没有“普遍改善”。

### Q4. 更强正则化主要造成拟合下降、Validation 改善，还是两者同时发生？

**OBSERVED**

| run | Train AP@40 Δ | Valid AP@best Δ | Valid AP@40 Δ | Valid BCE@40 Δ | Train−Valid gap@40（共同 127 任务） |
|---|---:|---:|---:|---:|---:|
| W3_WD10 | −0.0112 | +0.0005 | +0.0002 | −0.0005 | 0.4353（历史 W3 0.4468） |
| W3_DO20 | −0.0420 | −0.0002 | +0.0020 | −0.0012 | 0.4030（历史 W3 0.4468） |

- WD10：**两者都没有明显发生**——拟合只降 0.011，Validation 只升 0.0002。
- DO20：**拟合下降为主**（Train −0.042，best epoch 时 Train AP 比历史 W3 e40 低 0.149），
  Validation 只在 epoch-40 端点上小幅上升（+0.0020），峰值没有上升。
  gap 收窄 0.044，全部来自 Train 下降，而非 Valid 上升。
- 两个 run 的 Validation 曲线形状都变了：峰值更早（e19 vs 历史 e25）、峰值后更平（DO20 20–40 斜率 +5.9e-5/epoch，
  历史 W3 的 20–40 斜率 −2.0e-4/epoch，末 10 epoch 标准差 0.00088），但**峰值高度没有提高**。

**INTERPRETATION**：本轮“更强正则化”的效果是**以牺牲拟合为代价换取后期更平稳的曲线**，
而不是提高可达的 Validation 峰值。按预注册规则（§10 + §12.1）：
`PEAK_ONLY_OR_UNSTABLE` 需要 `d_best ≥ +0.002`（WD10 只有 +0.00047，DO20 为负），
`FIT_SUPPRESSION` 需要 `Train@40` 下降 ≥0.05（DO20 只有 0.042），
所以两个 run 都落在 `NO_CLEAR_GAIN`。

### Q5. 这次实验是否支持继续围绕 1M–2M 的无迭代 LCRC 做训练配方优化？

**就“单旋钮正则化配方”而言：没有足够证据继续投入。**

支持这一判断的实测事实：

1. 两个最直接的单旋钮（weight decay ×10、dropout ×2）对 Validation 的影响是
   +0.0005/+0.0002 与 −0.0002/+0.0020，**全部低于本轮预注册的 +0.005/+0.003 门槛**，
   也基本落在 epoch 间噪声（末 10 epoch std 0.0006–0.0012，历史 W3 0.00088）之内。
2. 它们改变拟合的能力也不对称：wd=0.010 的衰减预算只有 ~15.6% 权重收缩（Σlr=17.0），
   Train 只降 0.011；dropout 0.20 把 Train 降了 0.042，但**峰值 Valid 反而低了 0.0002**。
3. 有支撑的 assay 没有普遍改善（≥20 支撑层 55–61/107 改善），稀疏 assay 净负。
4. Test 侧唯一的正向信号（DO20 Test@best +0.0059）方向与 Valid 不完全一致，
   且 Test 在本轮是 `EXPOSED_TEST_EXPLORATORY`，**不能作为继续投入的依据**。

**不能确认的**：
- 这**不能**证明“LCRC 的配方已经没有空间”。本轮只测了两个取值、一个 seed；
  wd 的取值在真实衰减预算意义上仍然很弱（见 §Q1），dropout 也只测了 0.20。
- 这**不能**回答 1M–2M 容量区间是否值得继续投入；本轮没有比较架构，
  也没有改变数据、长度（40 epoch）或优化器。
- 任何“再调一调就能上去”的说法都缺乏证据；反过来“已经到顶”同样缺乏证据。
  本轮可支持的结论仅是：**在历史 W3 配方附近做单旋钮正则化微调，收益小于本任务的噪声与预注册门槛。**

---

## 2. 曲线与分 epoch 数据

**三划分 AP（key epochs）**

| run | epoch | Train AP | Valid AP | Test AP |
|---|---:|---:|---:|---:|
| W3_WD10 | 20 | 0.585810 | 0.262253 | 0.255297 |
| W3_WD10 | 25 | 0.634845 | 0.263540 | 0.250374 |
| W3_WD10 | 30 | 0.668712 | 0.262187 | 0.249543 |
| W3_WD10 | 35 | 0.687705 | 0.261206 | 0.250998 |
| W3_WD10 | 40 | 0.694325 | 0.259214 | 0.249636 |
| W3_DO20 | 20 | 0.567099 | 0.261418 | 0.255772 |
| W3_DO20 | 25 | 0.609934 | 0.262313 | 0.253970 |
| W3_DO20 | 30 | 0.640443 | 0.262213 | 0.255676 |
| W3_DO20 | 35 | 0.657856 | 0.262419 | 0.255701 |
| W3_DO20 | 40 | 0.663519 | 0.260978 | 0.255280 |
| W3 historical | 20 / 25 / 30 / 35 / 40 | — | 0.262726 / 0.263670 / 0.262884 / 0.260242 / 0.259004 | — |

- **峰值形态**：两个新 run 都在 epoch 19 取到 Best Valid；WD10 只有 e19 一个 epoch 高于历史 W3 峰值，
  DO20 没有任何 epoch 高于历史峰值。
- **20–40 趋势**：WD10 Valid −8.5e-5/epoch、Test −1.7e-4/epoch；
  DO20 Valid +5.9e-5/epoch、Test −4.7e-5/epoch；两者 Train 仍在上升（+5.3e-3 / +4.8e-3 per epoch）。
- 图：`analysis/figures/{valid_ap,test_ap,train_ap,train_valid_gap}.png`，
  数据：`analysis/curves_epoch.csv`。

**Train–Valid gap 的正确读法**

Train 与 Valid/Test 的 macro AP 不是同分布指标：Train 128 个任务全部可评分，Valid/Test 只有 127 个
（task 45 在 Valid 中没有正例，不进入官方 macro AP）；观测条目占比 Train 60.57%、Valid 62.04%、Test 60.24%，
正例率 Train 1.325%、Valid 1.609%、Test 1.679%，且划分是官方 scaffold split，
scaffold 分布不同。因此只在**双方都可评分的共同任务集合（127 个）**上比较：

| 模型 | epoch 40 Train（共同 127 任务） | epoch 40 Valid（共同 127 任务） | gap |
|---|---:|---:|---:|
| 历史 W3 | 0.705759 | 0.259004 | **0.446755** |
| W3_WD10 | 0.694577 | 0.259214 | **0.435304** |
| W3_DO20 | 0.663974 | 0.260978 | **0.402996** |

（按各自全部可评分任务计算的 gap 与上表相差 <0.001：WD10 0.43511，DO20 0.40254。）

---

## 3. 实验变量与预检（全部通过）

`analysis/audit.json` 汇总，`analysis/{data_audit,model_audit,test_isolation,evaluation_cost_probe,historical_test}.json` 为原始记录。

| 检查 | 结果 |
|---|---|
| 两个模型参数量 = **2,120,844**（trunk 1,809,408 + readout 311,436） | 通过（两者 `parameters_exact`） |
| 两组初始 `state_dict` 逐张量相等、eval 初始 forward 完全相同（max |Δ| = 0） | 通过 |
| 两组只在 `{dropout, weight_decay}` 上不同（`(0.10,0.010)` vs `(0.20,0.001)`） | 通过（`recipe_diff.differing_keys` 精确匹配） |
| `W3_WD10` 的 AdamW 参数组 `weight_decay = 0.010`；另一组 0.001（从 optimizer 读回） | 通过（单参数组，2,120,844 参数） |
| `W3_DO20` 实际 dropout 层 `p = 0.20`，另一组 0.10——在**真实 forward 路径**上验证 | 通过：train 模式 h 的零分数 0.20015±0.00033 / 0.10002±0.00031（12 次采样，632,256 个元素），eval 模式零分数 0 且两次 forward 完全一致 |
| 同一 batch、同一初始参数下干预生效 | 通过：eval logits 完全相同（max Δ = 0），train logits 不同；AdamW 衰减乘子不同（比值 10.0） |
| 训练顺序 row-id 哈希、总 step 数与历史 W3 规则一致 | 通过：`global_plan_sha256 06f43729…b6f6`、40 个 epoch 的 `row_order_sha256` 与历史 `training_order.json` **逐 epoch 相同**、2,738 step/epoch、109,520 总更新、prefix `1a24dfca…` |
| 官方 split 行号互不重叠、覆盖全部 437,929 行 | 通过（350,343 / 43,793 / 43,793，交集为 0，union 唯一值 437,929） |
| 特征 / normalization / 图缓存 / collate 与历史 W3 一致 | 通过（`feature_spec_hash 7270f767…`、`norm_fingerprint 6ce50f98…`、`split_partition c872bd63…`、`label_artifact 5c3b5fbe…`） |
| Test 标签对齐与隔离 | 通过：官方 `raw/graph-label.csv.gz` 分块读入 (437,929×128)，Train/Valid 与冻结 artifact **含 NaN mask 逐值相等**；Test 非缺失值仅 0/1（3,376,672 观测 / 56,698 正例）；Test 标签只进入 metric 函数（`test_isolation.json` 9 项静态+结构检查全通过） |
| task 45 不被错计入 macro AP | 通过：Valid 正例 0 → 官方 evaluator 不计入，Valid/Test 均为 127 个可评分任务 |
| 每个 split 全量、不重不漏 | 通过：每 epoch 三个 split 行数恒为 350,343 / 43,793 / 43,793，官方 AP 与独立 sklearn 逐任务复算 **max |Δ| = 5.6e-16**（全部 120 条记录） |
| eval 不消耗训练 RNG、不调用 optimizer、不改梯度 | 通过：每个 epoch 前后 params/optimizer/grads/RNG 四类指纹全部相同（`eval_state_integrity_all_epochs = true`，40/40） |
| 保存/恢复不丢已完成评价行、不重复训练已完成 epoch | 通过（单元测试：1→2 epoch resume，epoch-1 三划分 AP 完全保持，per-task 行数 3×128/epoch） |
| 真实 smoke（10 个真实 step + 三划分完整评估） | 通过：loss 1.5588→0.5608，三划分行数正确、官方=sklearn、eval 状态保持 |

**GPU 预检与耗时预估**（`analysis/evaluation_cost_probe.json`，`res` GPU1，commit `2267c97b`）：

| 项 | 实测 |
|---|---:|
| step p50 / p90 | 46.98 ms / 97.84 ms |
| 预计 epoch 训练 | 128.6 s |
| 完整 Train / Valid / Test 推理+指标 | 46.2 s / 5.66 s / 5.68 s |
| 预计 epoch 总时长 | 186.1 s（eval 占 30.9%） |
| 预计两 run 合计 | **4.14 GPU-h**（≤ 10 目标，≤ 16 硬上限） |
| smoke 峰值显存 | 374 MB |

---

## 4. 执行、资源账与命令

**运行状态**

| run | rr run id | pool | CUDA_VISIBLE_DEVICES | 窗口（CST） | 状态 |
|---|---|---|---|---|---|
| `W3_WD10` | `rd-w3-wd10-20261110-141226-701a82eb` | `res-gpu1` | 1 | 14:12:29 → 16:23:41 | COMPLETED 40/40 |
| `W3_DO20` | `rd-w3-do20-20261110-163807-dc3e2b04` | `res-gpu1` | 1 | 16:38:07 → 18:36:03 | COMPLETED 40/40 |

GPU0 **未使用**：`analysis/gpu_preflight.json` 与 `analysis/gpu_occupancy_samples.txt` 记录，
启动前 GPU0 在 6/6 次（60 s，10 s 间隔）采样中利用率 100%，驻留外部作业 pid 3841932（16,082 MiB），
该进程在两次正式启动前的复查中仍然存在。
按 `PROTOCOL.md` §8 的共享条件（无持续满载争用），两个 run **串行**跑在 `res-gpu1`。
每次启动前重新采样（`rd-gpucheck2/3`），GPU0 始终 100%。未对其他进程做任何操作。

**资源账（`analysis/gpu_ledger.json`，按 rr 记录的实际墙钟计费）**

| job | pool | wall | 计入 GPU-h |
|---|---|---:|---:|
| `rd-gpucheck` ×3（只读 nvidia-smi） | res-local | 63/63/62 s | 0 |
| `rd-smoke`（GPU smoke + 三划分成本探针） | res-gpu1 | 82 s | 0.0228 |
| `rd-w3-wd10` | res-gpu1 | 7,872 s | 2.1867 |
| `rd-w3-do20` | res-gpu1 | 7,067 s | 1.9631 |
| **合计** | | | **4.1725** |

目标 ≤10，硬上限 ≤16 → **`WITHIN_TARGET`**（占目标 42%，占上限 26%）。

**每个 run 的实测耗时/内存**

| run | epoch 均值 | step p50 均值 | steps/s | 训练 s | 三划分评估 s | 峰值显存 | 峰值 RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| W3_WD10 | 195.4 s | 44.5 ms | 21.58 | 5,092.8 | 2,722.3（Train 2,147 / Valid 277 / Test 267） | 371 MB | 8.9 GB |
| W3_DO20 | 175.3 s | 43.6 ms | 23.29 | 4,707.9 | 2,304.4（Train 1,825 / Valid 227 / Test 225） | 371 MB | 8.9 GB |

两 run 的纯训练 step 吞吐与历史 W3（23.77 steps/s）同量级；显存峰值 371 MB，GPU 仍是剩余资源。
峰值 RSS 8.9 GB 来自指标计算（350k×128 的 float64 转换），非模型或缓存。无 OOM、无非有限值、
无断点重跑；gradient clipping 各触发 10 次，**全部发生在 epoch 1**（grad_norm 最大 13.55/13.58，阈值 5.0）。

**实际执行命令**（全部经 `rr`；`<round>` = `tracks/structural-sufficiency/lcrc_pcba_w3_regularization_diagnostic_v1`）

```bash
uv run --no-sync pytest -q <round>/tests                       # 38 passed
rr doctor res && rr deploy res
rr run res rd-gpucheck  --pool res-local --result <round>/runs/audit -- bash -c "<nvidia-smi 采样 ×6>"
rr run res rd-smoke     --pool res-gpu1  --result <round>/runs/audit -- \
  bash -c "mkdir -p <round>/runs/audit && uv run --no-sync python <round>/code/rd_audit.py --modes smoke --device cuda --out-dir <round>/runs/audit"
rr run res rd-w3-wd10   --pool res-gpu1  --result <round>/runs/W3_WD10 -- \
  bash -c "mkdir -p <round>/runs/W3_WD10 && uv run --no-sync python <round>/code/rd_train.py --run W3_WD10 --seed 0 --device cuda --out-dir <round>/runs/W3_WD10 --soft-seconds 16000 --hard-seconds 18000"
rr run res rd-w3-do20   --pool res-gpu1  --result <round>/runs/W3_DO20 -- \
  bash -c "mkdir -p <round>/runs/W3_DO20 && uv run --no-sync python <round>/code/rd_train.py --run W3_DO20 --seed 0 --device cuda --out-dir <round>/runs/W3_DO20 --soft-seconds 16000 --hard-seconds 18000"
rr pull rd-w3-wd10 && rr pull rd-w3-do20
uv run --no-sync python <round>/code/rd_compare.py
uv run --no-sync python <round>/code/rd_finalize.py
```

**代码版本**：`rd-w3-wd10` 运行于 commit `07c947cd`，`rd-w3-do20` 于 `d3bd7685`
（后者只增加了 analysis 侧的 `rd_finalize.py`）。两 run 的 `run.json.module_sha256` **15 个文件逐个相同**，
`rd_{common,data,model,train}.py` 与继承模块在两 run 之间**字节一致**（逐文件哈希见两 run 的 `run.json`）。

---

## 5. Test 暴露与选择规则

- Test **每个 epoch 都完整评分**（40/40），标记 `EXPOSED_TEST_EXPLORATORY`；两个 run 合计
  **80 次全量 Test 评分**，每次都覆盖 43,793 行 / 127 个可评分任务。
  Test 标签来自官方 `raw/graph-label.csv.gz`，与冻结训练 artifact 严格隔离（Train/Valid 逐值+Nan mask 对齐，
  Test 只进入 metric）。训练 batch 只来自 `task.labels[plan rows]`（Test 行仍为 NaN）。
- 主 checkpoint 选择规则固定为 **Validation AP 最大、平局取更早 epoch**；本轮两 run 的 best 都是 epoch 19。
  没有任何 epoch 以 Test 选点；`comparison.json` 里的 Test 数字全部是**事后描述**。
- 历史 W3 的 Test AP 只用一次只读推理补齐（best-Valid e25 与 e40 两个 checkpoint），未重训。

---

## 6. 事故、偏差与限制

**无事故**：0 次失败启动、0 次取消、0 次重跑、0 次 OOM、0 次非有限值。

**与协议的偏差**：无规则偏差；两条 append-only amendment 都在**第一个正式 step 之前**记录：
1. §12.1 把 §10 的定性规则固定为可复现的算式与优先级（阈值本身未改）；
2. §12.2 记录“GPU0 满载 → 两 run 串行在 GPU1”这一 §8 规则的执行结果（含预测 4.14 GPU-h）。

**必须承认的限制**：
- **单 seed**。末 10 epoch 的 Validation std 为 0.0008–0.0012（历史 W3 0.00088），所以 |Δ| ≲ 0.001–0.002 不可解释为效应；
  DO20 的 +0.00197 恰在边界上。
- **两个旋钮各只测了一个取值**，且 wd=0.010 在真实衰减预算意义上仍然很弱（~15.6% 收缩）。
- Train macro AP（128 任务）与 Valid/Test（127 任务）不同分布，正例率 1.33% vs 1.61% / 1.68%；
  已用共同 127 任务集合重算 gap，但 scaffold 分布差异无法消除。
- 没有做多次运行的方差估计，也没有任何显著性检验；所有标签都是方向性/资源性判断。
- 历史 W3/W2 的 Test 数字是死后比较（historical W3 是本轮补的只读推理），不是盲测估计。

---

## 7. 结论

1. **wd 0.001→0.010**：Best Valid +0.00047、Valid@40 +0.00021，Train@40 −0.011。没有可测的泛化收益。
2. **dropout 0.10→0.20**：Best Valid −0.00017、Valid@40 +0.00197，Train@40 −0.042，
   Test@best +0.00587（探索性）。拟合被压低、后期更平，但**峰值没有提高**。
3. 两个 run 都标 **`NO_CLEAR_GAIN`**：Validation 差异小于预注册门槛，也基本在噪声之内；
   有支撑 assay 没有普遍改善，稀疏 assay 净负。
4. 对研究问题的回答：**没有证据表明“正则化配方未适配规模”是历史 W3 Validation 停滞的主要原因**；
   在这两个单旋钮取值上，正则化的确改不了 Validation 峰值的量级。
   反过来也**不能**断言 LCRC 的配方空间已经用尽（见 §Q1 的衰减预算与 §6 的限制）。
5. 按本轮预注册的资源决策：**不以此为依据继续投入单旋钮正则化配方调优**；
   若未来还要在这个容量区间做配方实验，需要先给出与本次不同的、可测的干预量级（例如真正改变
   衰减预算或同时改变多个旋钮），并且必须换 seed 或增加 seed 数才能谈稳定性。

**本轮交付**：2/2 个 40-epoch run 完整完成，120 条 epoch×split 官方 AP 记录，30,720 条 per-task AP 记录，
6 个 checkpoint（SHA-256 校验全部与 `run.json` 一致），全部预检通过，Test 按协议暴露但未参与选择。
