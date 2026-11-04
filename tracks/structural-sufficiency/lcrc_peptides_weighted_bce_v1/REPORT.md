# LCRC Peptides-func Weighted BCE Diagnostic v1 — 事实报告

- **Round ID**：`lcrc-peptides-weighted-bce-v1`
- **结果标签**：`EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC`
- **唯一干预**：训练 loss 由普通 BCE 改为基于官方 Train 标签频率的 `pos_weight` 加权 BCE。
- **正式代码 commit**：`e8def281d60f4c6aff55bfbc75c54fb2861a0143`（`git_tracked_dirty = False`；`git_dirty = True` 仅因远端未跟踪的 `runs/` 输出目录）
- **正式 run id**：`lcrc-pepweighted-p2-seed0-20261104-102145-361ca7ae`；实验名 `lcrc-pepweighted-p2-seed0`
- **机器 / 执行窗口（UTC）**：`res` / `a100-2`，1× NVIDIA A100-SXM4-40GB（`CUDA_VISIBLE_DEVICES=0`），driver 550.163.01，torch 2.5.1+cu124（CUDA 12.4），Python 3.12.14；2026-11-04T02:21:48Z – 2026-11-04T02:57:45Z
- **报告口径**：只记录可核验事实、差值与实验局限；不给出“架构问题已解决/已证明”的结论。

> **Test 政策**：正式训练与分析入口安装 `sys.addaudithook`，任何打开历史 Test artifact 的行为立即失败。
> 本轮 `test_evaluated = False`，`test_access_attempts = []`，
> 训练进程 `guard.attempts = []`。没有读取 Test 数据、Test 标签或 Test 预测。

---

## 0. 执行摘要

| 项 | 事实 |
|---|---|
| 正式训练 | 成功完成，status = `complete_no_test`，240/240 epoch，无早停、无 OOM、无时间上限阻塞 |
| 唯一干预 | Weighted BCEWithLogitsLoss，10 个 Train-only `pos_weight` |
| 架构 / 参数量 | LCRC-C1 / NO_BOND_MEAN；464,739 参数；init hash 与历史 seed0 完全相同 |
| best Validation macro AP | **56.89%** @ epoch 221 |
| 历史普通 BCE seed0 | 60.62% @ epoch 231 |
| 本轮 − 历史 | **-3.73 pp**（Validation macro AP） |
| Train macro AP @ best | 63.43%（历史 67.94%，-4.52 pp） |
| Train/Validation macro gap | 6.54 pp（历史 7.32 pp，-0.78 pp） |
| GPU / 时间 | 正式 0.596 A100 GPU-hours；2145.4 s（上限 1.25 h / 4200 s） |
| 正确性门禁 | `analysis/gates.json`：48 项全部通过 = True |
| Test 访问 | 0 次；`test_evaluated = false` |

---

## 1. 唯一干预与冻结不变量

`pos_weight` 逐标签定义为

```
pos_weight_j = train_negative_j / train_positive_j
```

只用官方 Train split 计算；没有截断、平滑、开方、归一化、上限或强制 `max(1, w)`。`antimicrobial` 的权重小于 1，`antiparasitic` 的权重约为 49。

**没有修改**：LCRC 架构、S1/S2、原子/键特征、readout、数据划分、特征缓存、normalization、初始化映射、AdamW 参数、peak LR、weight decay、dropout、batch size、gradient clipping、cosine horizon、early-stop 规则、checkpoint 选择指标。除 loss 外，训练配方与 Diagnostic v1 P2 相同（240 / 80 / 40；peak LR 3e-4；wd 1e-3；dropout 0.1；batch 64；clip 5.0；horizon 240）。

结构门禁（`run.json → structural_gate`）：

- 参数量 **464,739**，与历史 Fast v1 / Diagnostic v1 相同；
- `init_state_hash` = `4b7ca35f3f791fc54921cf20b7e470a22ec9f90aff99f353ec60f8e997fec1a8`，与 Fast v1 seed0 和 Diagnostic v1 seed0 均相同；
- feature spec hash、norm fingerprint、train/valid cache SHA-256 与 Diagnostic v1 完全一致；
- batch size 64，未触发 32/16 fallback；
- 因此本 run 是 loss 的单变量对照（单 seed）。

正确性检查在正式训练前完成，`analysis/gates.json` 覆盖：Train-only 权重与计数严格对应、标签顺序/维度/dtype/device、`ones(10)` 与普通 BCE 的数值及梯度一致、极端不平衡手工公式、缺失标签 mask 不广播错误、梯度有限且 AdamW 可更新参数、Test guard 拦截并记录、训练入口源码不引用 Test artifact。

---

## 2. 十个标签的 Train 正负计数与 `pos_weight`

来源：`analysis/pos_weight.json`（Train-only；`valid_used = false`，`test_used = false`）。权重为精确比值；GPU 上的 loss tensor 以 float32 保存，与精确比值的最大差异为 5.9e-7。

| idx | label | Train pos | Train neg | Train 阳性率 | pos_weight |
|---:|---|---:|---:|---:|---:|
| 0 | antifungal | 966 | 9907 | 8.88% | 10.255694 |
| 1 | cell_cell_communication | 385 | 10488 | 3.54% | 27.241558 |
| 2 | anticancer | 698 | 10175 | 6.42% | 14.577364 |
| 3 | drug_delivery_vehicle | 677 | 10196 | 6.23% | 15.060561 |
| 4 | antimicrobial | 6820 | 4053 | 62.72% | 0.594282 |
| 5 | antiviral | 2148 | 8725 | 19.76% | 4.061918 |
| 6 | antihypertensive | 1162 | 9711 | 10.69% | 8.357143 |
| 7 | antibacterial | 2002 | 8871 | 18.41% | 4.431069 |
| 8 | antiparasitic | 217 | 10656 | 2.00% | 49.105991 |
| 9 | toxic | 2825 | 8048 | 25.98% | 2.848850 |

`antiparasitic`：Train 217 正例，`pos_weight = 49.1059907834`。`antimicrobial`：Train 6,820 正例，`pos_weight = 0.5942815249`（小于 1，未截断）。

---

## 3. 正式训练事实

- 运行：`Peptides-func / LCRC-C1 / seed 0`，从 scratch 初始化，max 240 epoch，min 80，patience 40。
- 实际跑满 **240 epoch**；`stopped_early = false`。best Validation macro AP 在 **epoch 221**，距离 240 上限 19 个 epoch；之后到 240 没有新的严格提升。
- best checkpoint 的 Validation macro AP（选择指标）：**56.89%**；重新加载 checkpoint 后复算 56.90%，差值 5.87e-05（< 1e-4）。
- best checkpoint 的 Train macro AP：**63.43%**；逐标签 reloaded Validation AP 的均值为 56.89%（与选择指标相差 0.0004 pp，来自 checkpoint reload / CUDA reduction）。
- 最后 epoch（240）：Train weighted BCE loss = 0.56057，Validation macro AP = 56.62%；比 best 低 0.27 pp。
- 分阶段最佳：epoch 1–120 为 54.34%@107；epoch 121–240 为 56.89%@221。

典型 epoch（weighted BCE loss 的绝对数值不能与普通 BCE 直接比较）：

| epoch | Train weighted BCE | Validation macro AP % | grad norm mean / max | LR | best |
|---:|---:|---:|---:|---:|---|
| 1 | 2.16982 | 30.22 | 115.98 / 375.66 | 3.000e-04 | 是 |
| 60 | 0.76239 | 49.26 | 12.23 / 61.55 | 2.575e-04 | 是 |
| 120 | 0.64927 | 54.06 | 9.94 / 40.59 | 1.550e-04 |  |
| 180 | 0.58357 | 56.29 | 8.60 / 39.40 | 5.247e-05 |  |
| 221 | 0.56503 | 56.89 | 8.11 / 44.92 | 1.446e-05 | 是 |
| 231 | 0.56138 | 56.67 | 8.57 / 32.10 | 1.101e-05 |  |
| 240 | 0.56057 | 56.62 | 8.59 / 34.82 | 1.000e-05 |  |

资源与数值：

- 正式运行时间 **2145.45 s**（0.5960 A100 GPU-hours；上限 1.25 h）。另有两次最小 CUDA 工程 smoke（22.47 s、20.18 s，不用于结果）；GPU 总计约 **0.608 h**，仍低于 1.25 h。
- GPU peak：allocated **341.27 MiB**，reserved **738.00 MiB**。
- 梯度：全训练 40,800 次更新，clip 事件 **39,144**（95.94%）；历史普通 BCE 为 2,918（7.15%）。最大未裁剪梯度 **375.66**（历史 229.68）。没有非有限 loss 或梯度；没有 OOM、batch fallback 或时间上限阻塞。
- 最佳 epoch 的 Train weighted BCE = 0.56503；epoch 梯度 mean/max = 8.108/44.917。

---

## 4. 三路方法对照

| 方法 | Seed | Loss | Best epoch | Train AP | Validation AP |
|---|---:|---|---:|---:|---:|
| 历史 LCRC-C1 | 0 | 普通 BCE | 231 | 67.94% | 60.62% |
| 本轮 LCRC-C1 | 0 | Weighted BCE | 221 | 63.43% | 56.89% |
| ECFP4 + LightGBM | 0 | 独立树模型目标 | — | 不适用 | 72.98% |

- 本轮 − 历史（Validation macro AP）：**-3.73 pp**（60.62% → 56.89%）。
- 本轮 − 历史（Train macro AP）：**-4.52 pp**。
- ECFP4 + LightGBM 为固定背景参照，不重训、不是单变量对照；其与 LCRC 的差值不能解释为 loss 或架构效应。
- **普通 BCE 与加权 BCE 的训练 loss 数值不可直接比较**：两者对正例的权重不同，loss 的绝对尺度不同。

---

## 5. 逐标签 Validation AP：改善与退步

| label | Train pos / n | Train 阳性率 | pos_weight | 历史 Valid AP % | 本轮 Valid AP % | Δ pp |
|---|---:|---:|---:|---:|---:|---:|
| antifungal | 966/10873 | 8.88% | 10.255694 | 36.97 | 35.89 | -1.07 |
| cell_cell_communication | 385/10873 | 3.54% | 27.241558 | 50.63 | 43.33 | -7.30 |
| anticancer | 698/10873 | 6.42% | 14.577364 | 46.97 | 40.89 | -6.08 |
| drug_delivery_vehicle | 677/10873 | 6.23% | 15.060561 | 47.68 | 41.83 | -5.85 |
| antimicrobial | 6820/10873 | 62.72% | 0.594282 | 93.57 | 92.20 | -1.37 |
| antiviral | 2148/10873 | 19.76% | 4.061918 | 74.11 | 66.51 | -7.60 |
| antihypertensive | 1162/10873 | 10.69% | 8.357143 | 90.09 | 90.09 | +0.00 |
| antibacterial | 2002/10873 | 18.41% | 4.431069 | 52.94 | 50.45 | -2.49 |
| antiparasitic | 217/10873 | 2.00% | 49.105991 | 32.09 | 29.34 | -2.75 |
| toxic | 2825/10873 | 25.98% | 2.848850 | 81.23 | 78.35 | -2.87 |

- 改善的标签：antihypertensive (+0.0017 pp)。
- 退步的标签数量：**9/10**；最大退步为 antiviral (-7.60 pp)、cell_cell_communication (-7.30 pp)、anticancer (-6.08 pp)。
- `antihypertensive` 的 +0.0017 pp 在两位小数下为 0.00 pp，应视为基本持平；其余 9 个标签均为负差值。
- macro AP 的 −3.73 pp 不是由单一标签造成：`antiviral`、`cell_cell_communication`、`anticancer` 三个标签的 Δ 合计为 −20.98 pp（按 10 标签均值折合 −2.10 pp，约占 macro 降幅的 56%）。

### 5.1 逐标签 Train AP 与 Train/Validation gap

gap = Train AP − Validation AP（正值表示 Train 更高）。

| label | 历史 Train AP % | 本轮 Train AP % | ΔTrain pp | 历史 gap pp | 本轮 gap pp | Δgap pp |
|---|---:|---:|---:|---:|---:|---:|
| antifungal | 45.72 | 39.71 | -6.01 | 8.75 | 3.82 | -4.94 |
| cell_cell_communication | 58.99 | 54.15 | -4.84 | 8.35 | 10.81 | +2.46 |
| anticancer | 57.44 | 52.99 | -4.45 | 10.47 | 12.10 | +1.63 |
| drug_delivery_vehicle | 56.25 | 49.76 | -6.49 | 8.58 | 7.94 | -0.64 |
| antimicrobial | 95.85 | 93.83 | -2.02 | 2.28 | 1.63 | -0.65 |
| antiviral | 80.53 | 71.80 | -8.74 | 6.42 | 5.29 | -1.13 |
| antihypertensive | 94.13 | 92.40 | -1.73 | 4.03 | 2.30 | -1.73 |
| antibacterial | 62.63 | 57.35 | -5.28 | 9.69 | 6.90 | -2.78 |
| antiparasitic | 45.00 | 43.48 | -1.51 | 12.90 | 14.14 | +1.24 |
| toxic | 82.90 | 78.79 | -4.11 | 1.67 | 0.44 | -1.23 |

- macro Train/Validation gap：7.32 pp → 6.54 pp，变化 **-0.78 pp**。
- 逐标签 gap 下降的标签：7 个；上升的标签：3 个（cell_cell_communication, anticancer, antiparasitic）。
- 该 gap 收窄的同时 Train AP（-4.52 pp）与 Validation AP（-3.73 pp）都下降，因此不能把 gap 变小单独解释为泛化改善。

---

## 6. 对八个验收问题的直接回答

1. **Weighted BCE 是否成功执行？是否严格保持同一架构与训练配置？**
   是。正式 run `complete_no_test`，240/240 epoch；参数量 464,739，init hash、feature spec、norm fingerprint、train/valid cache 与历史 seed0 完全一致；batch 64、无 fallback；除 loss 外全部配方冻结。执行记录注记：远端 `git_dirty=true` 来自未跟踪的 `runs/` 输出，`git_tracked_dirty=false`。
2. **10 个标签的 `pos_weight` 分别是多少？**
   按标签顺序：antifungal=10.255694, cell_cell_communication=27.241558, anticancer=14.577364, drug_delivery_vehicle=15.060561, antimicrobial=0.594282, antiviral=4.061918, antihypertensive=8.357143, antibacterial=4.431069, antiparasitic=49.105991, toxic=2.848850。
3. **Weighted BCE 的最佳 Validation macro AP 是多少？**
   **56.89%**（epoch 221），reload 复算 56.90%。
4. **相比原普通 BCE 的 60.62%，提高还是降低了多少百分点？**
   降低 **3.7337 pp**（60.6229% → 56.89%）。
5. **哪些标签的 AP 改善最多？哪些下降？**
   仅 `antihypertensive` 为正（+0.0017 pp，基本持平）；其余 9 个标签全部下降。下降最多的是 `antiviral`（-7.60 pp）、`cell_cell_communication`（-7.30 pp）、`anticancer`（-6.08 pp）。
6. **最佳 checkpoint 在第几个 epoch？是否再次接近 240 epoch 上限？**
   epoch **221**；跑满 240 epoch，没有早停；best 距上限 19 个 epoch，之后 19 个 epoch 没有新的严格提升。仍属于接近 240 上限才达到最佳。
7. **Train/Validation gap 如何变化？**
   macro gap 7.32 pp → 6.54 pp，**-0.78 pp**。但 Train AP 和 Validation AP 同时下降（-4.52 pp / -3.73 pp），所以这不是“泛化更好”的证据。
8. **是否发生数值不稳定、过大的梯度或资源阻塞？**
   没有非有限 loss/梯度、OOM、batch fallback 或时间/预算阻塞。梯度显著大于历史：clip 事件 39,144/40,800（95.94%）对比历史 2,918（7.15%），最大未裁剪梯度 375.66 对比 229.68；训练靠固定 clip 5.0 完成。正式 GPU 使用 0.596 h（含两次最小 smoke 约 0.608 h），低于 1.25 h；实际耗时 2145.4 s，低于 4200 s。

---

## 7. 产物、哈希与复现

- 代码 commit：`e8def281d60f4c6aff55bfbc75c54fb2861a0143`；正式 run：`lcrc-pepweighted-p2-seed0-20261104-102145-361ca7ae`。
- checkpoint：`runs/weighted_bce_seed0/checkpoint.pt`，SHA-256 `8d9f7f6f78f1aed27c137ab59e3399bc2551f0ba283a62237acbc968b4c7d26b`。
- curve：`runs/weighted_bce_seed0/curve.csv`，SHA-256 `dba53190022864c6f813450056c36e3cb8811c7634805fd188a7c060b32289ae`。
- 配置：`configs/protocol.json`；协议：`PROTOCOL.md`。
- 权重：`analysis/pos_weight.json`；门禁：`analysis/gates.json`（48 项，全部通过 = True）。
- 逐 epoch 训练记录：`runs/weighted_bce_seed0/curve.csv`；选择记录：`runs/weighted_bce_seed0/selection.json`。
- 对照表：`results/validation_comparison.csv`、`results/per_label_comparison.csv`。
- 资源：`analysis/resource_ledger.json`（GPU budget 1.25 h；formal 0.5960 h；peak allocated 341.27 MiB；reserved 738.00 MiB）。
- Test：`test_evaluated = false`，`test_access_attempts = []`，guard attempts = `[]`。

复现命令（仓库根目录）：

```bash
# Train-only weights + gates + tests
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_compute_pos_weight.py
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_check.py
uv run pytest -q tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/tests

# The single formal run (committed revision, res-gpu0, 1 A100)
rr run res lcrc-pepweighted-p2-seed0 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 01:15:00 \
  --result tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 -- \
  uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_p2_train.py \
  --device cuda --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 \
  --max-seconds 4200

# Pull + comparison tables
rr pull lcrc-pepweighted-p2-seed0
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_analyze.py
```

---

## 8. 实验局限

- **单 seed、单配置**：只有一个 seed 0 和一个预先冻结的加权 BCE 配置；没有第二次训练、没有调参、没有根据 Validation 结果选择第二套配置。因此无法估计 seed 方差，也不能把差值分解为 loss 效应与优化噪声。
- **Validation-only**：本轮只报告 Train/Validation；Test 完全未访问，不能据此推断 Test 表现。
- **目标函数已改变**：普通 BCE 与 weighted BCE 的 loss 尺度不同，loss 数值不能用于判断哪个目标“训练得更好”。
- **单变量对照的边界**：ECFP4 + LightGBM 是不同模型/不同目标的固定背景参照，不是单变量对照。
- **指标细节**：`best_valid_macro_ap` 是训练中实时模型的 checkpoint 选择指标；逐标签 AP 来自 reloaded checkpoint，reload 差值为 5.87e-05。
- **gap 解释限制**：Train/Validation gap 收窄伴随两个 AP 同时下降，不能单独解释为泛化改善。
- **执行环境**：远端 `git_dirty=true` 仅由未跟踪的 `runs/` 输出目录触发；实际执行 commit 为 `e8def281d60f4c6aff55bfbc75c54fb2861a0143`，`git_tracked_dirty=false`。
- 本轮到此停止；不提出、不执行下一轮实验。上述事实不足以支持“架构问题已解决”或“架构问题已证明”的结论。
