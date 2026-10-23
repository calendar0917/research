# ZINC 性能搜索轮次报告 — `zinc_performance_search_v1`

**轮次**：2026-10-23 16:06 UTC 起，硬截止 2026-10-24 00:06 UTC（8.0 h 墙钟）
**硬件**：`res` 单机 2×A100-40GB（`res-gpu0` / `res-gpu1`），torch 2.5.1+cu124，PyG 2.6.1
**数据**：官方 PyG `ZINC-subset`（train 10,000 / valid 1,000 / test 1,000，原始 split，原始标签 `y_raw`）
**指标**：raw-y MAE；单模型、≤500k 部署参数、无预测融合/seed 平均/checkpoint soup/递归隐状态 MP
**目标**：单模型 test MAE < 0.060

---

## 0. 结论摘要

| 问题 | 回答 |
|---|---|
| 是否达到 test MAE < 0.060？ | **没有**。本轮最佳单模型 test MAE = **0.066267**（B-R0 / seed 0 / epoch 489），比目标高 **0.006267**（相对 0.060 高 10.4%）。 |
| 哪个家族被冻结并复现？ | **B-R0**（DCC-L 双通道 composer，494,193 参数，R0 配方）。4 个 seed 全部跑完：valid 0.075990 ± 0.002137，**test 0.067319 ± 0.000709，range [0.066267, 0.067785]**。 |
| 同期基线？ | A-R0（P3Null，413,505 参数）4 seeds：valid 0.080761 ± 0.003033，**test 0.070879 ± 0.001830，range [0.069865, 0.073621]**。B 的 **4 个 seed 全部优于 A 的 4 个 seed**。 |
| 训练配置是否改变排序？ | 是，但只有一个方向稳定：对 A 主体 R1（更低 lr / 更高 wd）**确定性变差**（valid +0.0113，test +0.0053）；对 B 几乎中性（valid +0.0007，test +0.0017）。R0L（1200 epoch 长日程）对 A **无收益**（valid 0.079491 vs 0.079250；test 0.071844 vs 0.070879，反而略差）。 |
| 第三主体（C）？ | 指定的 GRIT 路线**已实现并验证，但被实测成本否掉**（distillation 见 §2.3），按预案换成原生 DBT 600 epoch：valid 0.082479（sd 0.0074，两 seed），test 0.071329（sd 0.0034）。**在本轮预算内不具竞争力**，但全 horizon 的历史 DBT（3.69 GPU·h）valid 0.077768 / test 0.069401 说明它输在"被砍掉的 horizon"，不是主体本身。 |
| EMA 策略？ | **整轮未使用**。实现里 EMA→模型的拷贝方向写反，EMA 分支评估的是随机初始化（valid 恒 1.60004）；已修复但无预算重跑，所有 EMA 数字作废。见 §2.4。 |
| 证据等级 | 家族层：4 seeds 的均值/range + 逐 seed test；行层：仅描述性的行 bootstrap（说明样本不确定性，不代表训练噪声）；test 在本轮一次性读取并冻结，之后无任何调参。 |

**一句话**：在 ≤500k 参数、单模型、~0.35 GPU·h/run 的约束下，"靠主体/配方/日程搜索把 ZINC 推到 0.060" 这一假设被证伪——最好的家族只到 0.0663，而所有已试杠杆的收益都 ≤0.005。

---

## 1. 协议、时间线与预算

### 1.1 冻结顺序（与 `analysis/test_freeze.json` 一致）

1. 发现阶段只用 valid（1,000 分子）与 train；训练代码从未构造 test。
2. 发现阶段关闭时按 **seed0/1 的 best-valid 均值**排序（并列看 W20，再看 GPU·h），选出赢家家族。
3. 赢家补 seed2/3；同期跑 A-R0 的 seed2/3 作为匹配基线。
4. 写 `analysis/frozen_shortlist.json`（14 条，含 checkpoint 路径与记录的 valid MAE）并提交部署。
5. `zps_test.py` 先对每条重算 valid 并核对到 1e-6，写 `analysis/test_freeze.json`（含 checkpoint sha256），
   **之后**才第一次读 test 标签；test 只用于描述，不参与任何选择。

### 1.2 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 16:06 | 轮次开始，`res` 体检（2×A100-40GB、torch 2.5.1+cu124、PyG 2.6.1） |
| 16:19 | 首批 R1+EMA 训练上卡（收尾时暴露 EMA 保存 bug） |
| 17:15 | 修复 EMA 保存；重跑 A/B-R1 |
| 17:21 | C 替代臂（DBT 600ep）与 A-R0 seed1 上卡 |
| 18:51 | C-DBT seed0 完成（1.476 GPU·h，valid 0.077260） |
| 19:41 / 20:52 | B-R0 seed2 / seed3 上卡 |
| 21:29 | 发现阶段关闭：冻结 B-R0；短名单 14 条提交 |
| 21:34 | 终端 test 一次性读取完成（14 条，含 valid 复核） |
| 21:40 | 报告与交付物收尾（硬截止 00:06 前 2.4 h；两块卡均已空闲） |

### 1.3 预算

| 项 | 值 |
|---|---|
| 墙钟 | 8.0 h（硬），实际收尾 ~5.9 h |
| GPU·h 上限 | 16.0（硬）/ 15.0（软） |
| 实际 GPU·h（含失败与计时） | **8.39** |
| 其中失败/取消浪费 | 0.97（两次 EMA 保存失败各跑满 600 epoch 0.44 + C seed1 首次启动 5 min 取消 0.09） |
| 有效训练 GPU·h | 7.42 |

逐项成本见 `LEDGER.md` §1–§3。

---

## 2. 候选池与配方

### 2.1 主体

| 臂 | 主体 | 参数 | 来源 |
|---|---|---|---|
| A | `ad2_models.P3NullModel`：P3 局部编码器（SSL 初始化、微调）+ Null composer + reader | 413,505 | `zinc_recipe_trunk_v2` |
| B | `dcc_models` L 臂：P3 编码器 + 双通道（局部/全局）composer | 494,193 | `zinc_dual_channel_composer_v1` |
| C | 指定的 10 层 GRIT（473,473）→ 因成本回退为原生 DBT（452,523） | 452,523 | 见 §2.3 |

A/B 都从 seed 匹配的 **train-only SSL 编码器** 初始化（`pretrain_encoder_s0.pt`，sha256 `006fabc1…`；
仓库只有 s0/s1 两份，seed≥2 复用 s1，见 §6.3）。

### 2.2 配方（启动前声明）

| 名称 | 定义 | 状态 |
|---|---|---|
| R0 | AdamW lr 1e-3→1e-5，wd 1e-5，dropout 0，L1，batch 128，600 ep（47,400 updates），warmup 500，clip 5 | A×4 seeds，B×4 seeds |
| R1 | lr 5e-4→5e-6，wd 1e-4，其余同 R0 | A×1，B×1 |
| R2 | R0 + wd 1e-3 + dropout 0.05 | **未运行**（预算让位给复现） |
| R3 | R1 + SmoothL1 β=0.02 | **未运行**（同上） |
| R0L | R0 数值 + 1200 ep 余弦（94,800 updates） | A×2（本轮唯一针对性额外配置） |

### 2.3 C 臂：GRIT 为什么被排除（实测，不是估计）

我实现并审计了 10 层 / 64 宽 / 8 头（473,473 参数，RRWP 1..20）的**稠密 padded 端口**
`code/zps_grit_dense.py`：参数名/形状与 vendored 上游逐项对应，三项验证全部通过——

- RRWP vs 冻结 cache：max diff 2.4e-7；
- 与稀疏上游 forward 对照（dropout off）：1.7e-6；
- 梯度相对范数误差 9.6e-5，BN buffer 1.2e-7（`analysis/dense_parity.json`）。

然后实测（A100，独立卡，无争用）：

| 路径 | batch | s/epoch | 投影 |
|---|---|---|---|
| 稠密端口 | 32 | 30.97 | 2000 ep = 17.2 h |
| 稠密端口 | 64 | 18.65 | 2000 ep = 10.4 h |
| 稠密端口 | 128 | 14.84 | 2000 ep = 9.5 h |
| 稀疏上游（同实现，8 层，ZSRT 计时） | 128 | ~9.5 | 1500 ep ≥ 4 h |

全部 **超过 2 h/run 上限**。把 GRIT 砍到 ~600 epoch 能塞进 2 h，但离它 625k updates 的原配方差一个数量级，
等于用截断模型冒充"原生配置"。故按预案回退：**C = 原生 DBT，600 epoch 完整余弦（94,200 updates）**，
1.48 GPU·h/run，horizon 偏离声明在 `runs/dbtc_/dbt_s0/search.json`。

> GRIT 不是"做不出来"，而是"做出来也付不起"。parity/RRWP 产物保留，下一轮若给到 ≥5 GPU·h/run 可直接续用。

### 2.4 EMA 策略：为什么整轮弃用

`ad_train.py` 的 EMA 评估里 `zip(ema_model.parameters(), ema_state)` 的拷贝方向写反（把新构建的模型写进 EMA buffer），
于是 EMA 分支每轮评估的都是随机初始化：**valid 恒为 1.60004**，train MAE 1.63（A-R1 与 B-R1 完全一致）。
修正为 `module_param.copy_(ema_slot)` 后用"载入历史 best.pt → 拷贝 → 重算 MAE"的手工检查确认（0.0926 两路径一致）。

但重跑合法 EMA 分支需 2×0.35 GPU·h，且冻结前拿不到 seed0/1 证据，所以本轮 **不把 EMA 作为候选策略**：
A-R1/B-R1 只按 online 分支排名，`*_ema` 曲线/checkpoint 标记作废。这是本轮最明确的未完成项。

---

## 3. 发现阶段（seeds 0/1）

排序单位 = 同一 model+recipe+policy 的 seed 均值；本轮 policy 全部为 online。

| config | seed | params | best-valid | ep | W20 | final | train@best | d(row172) | d(top10) | GPU·h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-R0 | 0 | 413,505 | 0.076224 | 377 | 0.078081 | 0.078179 | 0.014680 | 0.074735 | 0.065096 | 0.301 |
| A-R0 | 1 | 413,505 | 0.082276 | 470 | 0.083121 | 0.083081 | 0.008450 | 0.076223 | 0.066317 | 0.360 |
| A-R1 | 0 | 413,505 | 0.087570 | 433 | 0.088577 | 0.088653 | 0.011625 | 0.079922 | 0.068124 | 0.313 |
| B-R0 | 0 | 494,193 | 0.078537 | 489 | 0.079082 | 0.079076 | 0.005510 | 0.072510 | 0.061802 | 0.345 |
| B-R0 | 1 | 494,193 | 0.073837 | 424 | 0.075159 | 0.075183 | 0.011016 | 0.068487 | 0.060867 | 0.402 |
| B-R1 | 0 | 494,193 | 0.079246 | 437 | 0.080613 | 0.080621 | 0.008968 | 0.073947 | 0.062609 | 0.416 |
| A-R0L | 0 | 413,505 | 0.076699 | 557 | 0.082230 | 0.082315 | 0.023196 | 0.075237 | 0.066815 | 0.717 |
| A-R0L | 1 | 413,505 | 0.082283 | 577 | 0.084112 | 0.084261 | 0.022968 | 0.075739 | 0.067418 | 0.617 |
| C-DBT600 | 0 | 452,523 | 0.077260 | 449 | 0.079183 | 0.079162 | 0.016491 | 0.073091 | 0.063901 | 1.476 |

关键观察（seed0/1 层面）：

1. **B-R0 领先**：0.076187 vs A-R0 0.079250，但分布重叠（B 最差 0.078537 > A 最好 0.076224），
   2 seeds 不足以宣称"B 主体更好"→ 进入 S3 复现。
2. **配方方向对 A 稳定有害**：R1 让 A 变差 +0.0113（≈3.7× seed sd），对 B 几乎中性（+0.0007）。
   说明 A 的瓶颈不是正则不足；加深正则只会更差。
3. **更长日程救不了 A**：R0L 两 seed 均值 0.079491 ≈ R0 的 0.079250，但 final 更差（0.0823/0.0843 vs 0.0782/0.0831）：
   1200 ep 的余弦在 600 ep 处仍处高位，跑到 1200 ep 反而过拟合。**"best checkpoint 在 377/600" 不是日程太短的证据。**
4. **C 单 seed 有竞争力、方差巨大**：0.077260 / 0.087697（sd 0.0074），是全部配置中最不稳定的；
   缩短 horizon 对 DBT 的伤害在 seed1 上暴露得很清楚。

---

## 4. 冻结与复现（seeds 2/3）

冻结规则：seed0/1 均值最小 → **B-R0**（0.076187；次席 A-R0 0.079250；C 只有 1 seed，不参与冻结）。
冻结后 B-R0 补 seed2/3，A-R0 同期补 seed2/3 作为匹配基线
（A-R0 四 seed 合计 1.274 GPU·h：0.301 / 0.360 / 0.305 / 0.308；B-R0 四 seed 合计 1.489 GPU·h）。

| config | valid s0 | s1 | s2 | s3 | valid mean | sd | range | W20 mean | test mean | test range |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| **B-R0** | 0.078537 | 0.073837 | 0.074677 | 0.076911 | **0.075990** | 0.002137 | 0.073837–0.078537 | 0.077075 | **0.067319** | 0.066267–0.067785 |
| **A-R0** | 0.076224 | 0.082276 | 0.081998 | 0.082546 | **0.080761** | 0.003033 | 0.076224–0.082546 | 0.082290 | **0.070879** | 0.069865–0.073621 |
| A-R0L | 0.076699 | 0.082283 | — | — | 0.079491 | 0.003949 | 0.076699–0.082283 | 0.083171 | 0.071844 | 0.070413–0.073275 |
| C-DBT600 | 0.077260 | 0.087697 | — | — | 0.082479 | 0.007380 | 0.077260–0.087697 | 0.083766 | 0.071329 | 0.068944–0.073713 |
| B-R1 | 0.079246 | — | — | — | 0.079246 | — | — | 0.080613 | 0.069022 | — |
| A-R1 | 0.087570 | — | — | — | 0.087570 | — | — | 0.088577 | 0.075147 | — |

- valid 均值差 **0.004771**（B 更好），约为合并 sd（0.0026）的 **1.8×**；test 均值差 **0.003560**，方向一致。
- 更强的一句话：**B-R0 的 4 个 seed 的 test MAE 全部低于 A-R0 的 4 个 seed**（0.067785 < 0.069865），
  valid 上两个家族仍有重叠（A 的 seed0 0.076224 优于 B 的 seed0 0.078537 与 seed3 0.076911）。
- W20 与 final-epoch 的排序与 best-valid 一致 → 不是"挑中了尖峰"。
- **valid 名次不能替代 test 名次**：B-R0 内部 valid 最好的 s1（0.073837）test 是 0.067688，
  而 valid 最差的 s0（0.078537）test 最好（0.066267）；A-R0 内部则相反（s0 最好、s3 最差）。
  family 级排序（B < A）在 valid/test 上一致，seed 级排序不一致 —— 这是 best-valid 选择噪声的直接证据。

---

## 5. 终端 test 读取

冻结文件 `analysis/test_freeze.json`（先写、后读 test）；短名单 `analysis/frozen_shortlist.json`（14 条）。
每条在读取 test 前重算 valid 并核对到 1e-6（14/14 通过）。

**管道自检**：A-R0 s0 重算 test = 0.069865，与历史一致；B-R0 s0 重算 test = 0.066267，与历史一致。
说明数据 cache、checkpoint 与评估路径与历史轮次完全对齐。

| model | seed | epoch | valid | **test** | RMSE | p90 AE | d(row172) | d(top10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B-R0** | 0 | 489 | 0.078537 | **0.066267** | 0.1613 | 0.1340 | 0.066307 | 0.066933 |
| B-R0 | 1 | 424 | 0.073837 | 0.067688 | 0.1389 | 0.1433 | 0.067748 | 0.068369 |
| B-R0 | 2 | 392 | 0.074677 | 0.067538 | 0.1425 | 0.1348 | 0.067573 | 0.068216 |
| B-R0 | 3 | 455 | 0.076911 | 0.067785 | 0.1400 | 0.1393 | 0.067819 | 0.068467 |
| A-R0 | 0 | 377 | 0.076224 | 0.069865 | 0.1480 | 0.1509 | 0.069880 | 0.070568 |
| A-R0 | 1 | 470 | 0.082276 | 0.070084 | 0.1397 | 0.1445 | 0.070143 | 0.070789 |
| A-R0 | 2 | 426 | 0.081998 | 0.069948 | 0.1612 | 0.1414 | 0.069990 | 0.070650 |
| A-R0 | 3 | 432 | 0.082546 | 0.073621 | 0.1461 | 0.1458 | 0.073610 | 0.074362 |
| A-R0L | 0 | 557 | 0.076699 | 0.070413 | 0.1416 | 0.1363 | 0.070471 | 0.071122 |
| A-R0L | 1 | 577 | 0.082283 | 0.073275 | 0.1512 | 0.1478 | 0.073338 | 0.074010 |
| C-DBT600 | 0 | 449 | 0.077260 | 0.073713 | 0.1643 | 0.1493 | 0.073772 | 0.074455 |
| C-DBT600 | 1 | 509 | 0.087697 | 0.068944 | 0.1573 | 0.1421 | 0.069002 | 0.069636 |
| B-R1 | 0 | 437 | 0.079246 | 0.069022 | 0.1488 | 0.1409 | 0.069061 | 0.069717 |
| A-R1 | 0 | 433 | 0.087570 | 0.075147 | 0.1559 | 0.1537 | 0.075178 | 0.075903 |

- test − valid 的偏移：B-R0 平均 **−0.0087**（0.07599→0.06732），A-R0 **−0.0099**，C **−0.0112**；
  与历史轮次同量级（历史 A/B ≈ −0.006…−0.012），说明本轮没有"只调 valid"导致 test 崩坏。
- row172 与 top-10 的敏感性都 <0.0007：本轮结论**不由任何单个分子驱动**（与早期轮次不同）。
- 行 bootstrap（10k，seed 20261023，单 seed 配对）给出的 95% CI 很宽，例如
  B-R0-s0 vs A-R0-s0：Δ=−0.003598，CI [−0.008071, +0.001226]，P(a better)=0.927。
  这描述的是 **1,000 个分子这个样本**的不确定性，不是"训练可重复性"；
  family 差异的稳定性证据来自 4 seeds 的完全分离，而不是行 bootstrap 的显著性。
- 逐分子文件：`analysis/test_pred_<name>_s<seed>.csv`（row, y_raw, pred, signed_error, abs_error），
  valid 逐分子：`analysis/per_molecule_valid.csv` + `analysis/valid_pred_<name>_s<seed>.npz`。

---

## 6. 证据等级与稳健性

### 6.1 站得住的结论

1. 官方 split、R0、≤500k 参数、单模型：**本轮所有主体都到不了 test 0.060**；最好 0.066267，差 0.006267。
2. **B-R0 > A-R0**：valid 均值差 0.0048（1.8× 合并 sd），test 均值差 0.0036，且 4/4 seeds 在 test 上完全分离。
3. **R1 对 A 稳定有害**（valid +0.0113，test +0.0053，1 seed + 3 seeds baseline），对 B 中性。
4. **1200 ep 长日程对 A 无收益**（valid 持平，test 略差，final 更差）。
5. GRIT 在本仓库现有实现下 **≥4 h/run**（含稠密端口实测 30.97 s/epoch @batch32），2 h 内无法给原生配置。

### 6.2 不能声称的

- 不能声称"B 主体本质更优"：4 seeds 的均值差 1.8×sd，且 valid 上仍有重叠；正确表述是"在本协议与本轮预算下，
  B-R0 的 seed 均值更低、test 完全分离"。
- 不能声称"C/DBT 比 A/B 差"：C 是 **600/1500 horizon 的截断版**；历史全 horizon DBT（3.69 GPU·h）
  test 0.069401，与本轮 A-R0 同量级、略优于 A-R0 的均值 0.0709。
- 不能把行 bootstrap 的 CI 当作训练重复性；也不能用 valid 名次预测 test 名次（§4 已给出反例）。

### 6.3 已知口径限制

1. **SSL 编码器只有 s0/s1 两份**：seeds ≥2 复用 s1 初始化，seed 2/3 的离散度**低估**了完全独立初始化的方差。
2. **test 已花**：本轮是"已暴露 test 的描述性复核"，不是新盲测；所有选择都发生在读取之前。
3. **best-valid 选择偏差**：在 600 个 epoch 上取最小，本身带 0.001–0.002 的选择偏差；W20/final 列用于识别这一点。
4. **R2/R3 未跑**：预算让位给赢家的 seed2/3 与 A 的匹配基线。R2 是已证有害方向（更强正则）的加强版；
   R3（SmoothL1 β=0.02，对 |e|≫0.02 与 L1 几乎相同）预期 ≈ R1。

### 6.4 冻结分组（valid）

| run | all_1000 | seen_749 | unseen_251 | ordinary_935 | tail_65 | unseen_ordinary_225 |
|---|---:|---:|---:|---:|---:|---:|
| A-R0 s0 | 0.07622 | 0.05105 | 0.15134 | 0.06828 | 0.19054 | 0.13394 |
| B-R0 s0 | 0.07854 | 0.05427 | 0.15096 | 0.06554 | 0.26554 | 0.13077 |
| B-R0 s1 | 0.07384 | 0.05319 | 0.13544 | 0.06308 | 0.22850 | 0.12182 |
| B-R0 s2 | 0.07468 | 0.05233 | 0.14136 | 0.06484 | 0.21616 | 0.12225 |
| B-R0 s3 | 0.07691 | 0.05645 | 0.13796 | 0.06475 | 0.25177 | 0.12301 |
| C-DBT600 s0 | 0.07726 | 0.05385 | 0.14710 | 0.06661 | 0.23041 | 0.13487 |
| C-DBT600 s1 | 0.08770 | 0.05851 | 0.17478 | 0.07525 | 0.26681 | 0.16406 |

unseen（251）的 MAE 是 seen（749）的 2.5–3 倍，tail（65）又是 unseen 的 1.5–2 倍；
**没有任何配方改变这个结构**（R1 让 tail 更差）。瓶颈在"训练中未见的原子/环境"子群，而非整体拟合——
与历史 `zinc_generalization_audit_v1` 的结论一致。

---

## 7. 未达成目标的原因与开放问题

- **差距的量级**：要 test <0.060，valid 需要 ≈0.066（test−valid ≈ −0.009）。本轮最好家族 valid 0.0760，
  差 0.010 —— 3–5 个 seed sd。已试杠杆（lr、wd、horizon、第三主体、双通道 composer、R1）单点收益都 ≤0.005。
- **收益最大的一步是"换主体"**（A→B：test −0.0036），但只走完了已有两具身体；第三具（GRIT）被成本挡住。
- **开放问题**：
  1. 给到 ≥5 GPU·h/run，10 层 GRIT（RRWP 1..20，原生 625k updates）能否把 valid 推到 ≤0.070？本轮只证明 2 h 不够。
  2. 全 horizon DBT（1500 ep，3.7 GPU·h）两 seed 的均值是多少？单 seed 0.0778 valid 不足以定论。
  3. EMA 修好后是否稳定改善 best-valid？本轮无数据。
  4. seeds ≥2 需要独立 SSL 编码器；否则 seed range 低估方差。
  5. R3（SmoothL1）与 R2（强正则）仍未跑；按现有证据 R2 预期为负，R3 ≈ R1。

---

## 8. 可运行命令（champion = B-R0）

```bash
# 0) 提交 + 部署
git add -A && git commit -m "..." && rr deploy res

# 1) 冠军配置（B 主体 + R0），单 seed，约 0.35 GPU-h
rr run res zps-B_R0_s0 --pool res-gpu0 --gpus 1 --time 1:00:00 \
  --result tracks/structural-sufficiency/zinc_performance_search_v1/runs \
  -- python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_runner.py \
     train --arm B --recipe R0 --seed 0 --device cuda:0 --force

# 2) 拉回轻量产物（train.json / curve.csv / predictions）
rr pull zps-B_R0_s0 --light

# 3) 重算 valid 表 / 分组 / 对照 / 报告表格
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_analyze.py
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_report.py

# 4) 终端 test（只在冻结后执行一次）
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_shortlist.py \
  --spec tracks/structural-sufficiency/zinc_performance_search_v1/analysis/shortlist_spec.json
git add -A && git commit -m "freeze shortlist" && rr deploy res
rr run res zps-test --pool res-gpu0 --gpus 1 --time 0:40:00 \
  --result tracks/structural-sufficiency/zinc_performance_search_v1/analysis \
  -- python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_test.py --device cuda:0
```

C 替代臂：

```bash
rr run res zps-dbtc-s0 --pool res-gpu1 --gpus 1 --time 3:00:00 \
  --result tracks/structural-sufficiency/zinc_performance_search_v1/runs \
  -- python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_dbt.py \
     train --seed 0 --device cuda:0 --epochs 600
```

---

## 9. 交付物

| 文件 | 内容 |
|---|---|
| `PLAN.md` / `DECISIONS.md` | 预注册设计、逐条决策与偏离（D1–D7） |
| `LEDGER.md` | 全部 run（含失败/取消）、配置、成本、状态 |
| `code/zps_runner.py` | A/B 臂 + 配方 R0–R3/R0L 的声明式入口 |
| `code/zps_grit_dense.py` / `code/zps_dense_timing.py` | GRIT 稠密端口与其成本测量 |
| `code/zps_dbt.py` | C 替代臂（DBT，可配 horizon） |
| `code/zps_analyze.py` / `code/zps_report.py` | valid 表、冻结分组、W20、bootstrap、报告表格生成 |
| `code/zps_shortlist.py` / `code/zps_test.py` | 冻结 → 一次性 test 读取 → 逐分子产物 |
| `analysis/main_table.{md,json}` | 全部 run 的 valid 指标 |
| `analysis/report_tables.md` | 本报告所有表格的机器生成版 |
| `analysis/test_table.{md,json}` / `analysis/test_freeze.json` | 终端 test 结果与冻结记录（含 sha256） |
| `analysis/test_pred_*.csv` / `analysis/per_molecule_valid.csv` | 逐分子 valid/test 预测与误差 |
| `analysis/curves_valid.csv` / `analysis/curves_valid.md` | 14 条训练曲线的逐 epoch 数据与每 50 epoch 采样表（`code/zps_curves.py` 生成） |
| `analysis/dense_parity.json` / `analysis/dense_timing.json` | GRIT 端口 parity 与成本证据 |
| `runs/<run>/` | 每个 run 的 train.json、curve.csv、predictions、search.json（checkpoint 在远端） |

---

## 10. 给下一轮的一句话

本轮把"在 ≤500k 参数、单模型、0.35 GPU·h/run 的格子里靠配方/主体搜索把 ZINC valid 从 0.076 推到 0.066"
**证伪**了：所有杠杆都在 ±0.005 内，而目标是 0.010。下一轮要在两条路里选一条——
给 GRIT/DBT 这类显式关系模型 ≥4–5 GPU·h/run 的原生 horizon 预算，或者换问题设定（更大模型/更多数据/不同 readout）；
继续在同一格子里调参的期望收益接近 0。
