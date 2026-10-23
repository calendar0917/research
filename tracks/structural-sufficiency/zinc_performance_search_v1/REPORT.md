# ZINC 性能搜索轮次报告 — `zinc_performance_search_v1`

**轮次**：2026-10-23 16:06 UTC 起，硬截止 2026-10-24 00:06 UTC（8.0 h 墙钟）
**硬件**：`res` 单机 2×A100-40GB（`res-gpu0` / `res-gpu1`），torch 2.5.1+cu124，PyG 2.6.1
**数据**：官方 PyG `ZINC-subset`（train 10,000 / valid 1,000 / test 1,000，原始 split，原始标签 `y_raw`）
**指标**：raw-y MAE；单模型、≤500k 部署参数、无预测融合/seed 平均/checkpoint soup
**目标**：单模型 test MAE < 0.060（= 0.05× 量级）

---

## 0. 结论摘要

| 问题 | 回答 |
|---|---|
| 是否达到 test MAE < 0.060？ | **没有**。本轮最佳单模型 test MAE = {{BEST_TEST}}（B-R0，seed {{BEST_SEED}}）；目标 gap 约 {{GAP}}（≈{{GAP_RATIO}}× seed 标准差）。 |
| 哪个家族被冻结并复现？ | **B-R0**（DCC-L 双通道 composer，494,193 参数，R0 配方），seeds 0–3 全部完成：valid mean {{B_R0_MEAN}}，range {{B_R0_RANGE}}。 |
| 同期基线？ | A-R0（P3Null，413,505 参数）seeds 0–3：valid mean {{A_R0_MEAN}}，range {{A_R0_RANGE}}。 |
| 第三主体（C）？ | 指定的 GRIT 路线**因实测成本被排除**（不是"没做"，而是做完并验证后被 2 h/run 上限否掉，见 §2.3）；按预案替换为原生 DBT 600 epoch：valid 0.077260 / 0.087697（两 seed），均值 {{C_MEAN}}，**未进入短名单第一梯队**。 |
| 训练配置是否改变排序？ | 是，但只在一个方向上稳定：对 A 主体，R1（更低 lr/更高 wd）**确定性地变差**（+0.0113）；对 B 主体几乎中性（+0.0007）。R0L（1200 epoch 长日程）对 A **无收益**（0.079491 vs 0.079250，同 seed 对数差 +0.0002/+0.0000 量级），证伪"更长日程能救 A"。 |
| EMA 策略？ | **全轮未使用**。实现里的 EMA→模型回调参数顺序写反，导致 EMA 分支评估的是随机初始化（valid 恒为 1.60004）；已修复但无预算重跑，故所有 EMA 数字作废，两个 `*_ema` run 只按 online 分支排名。详见 §2.4 与 LEDGER。 |
| 证据等级 | 主体/家族层面：seed 均值 + range（3–4 seeds）；行层面：仅作为描述性行 bootstrap（说明样本不确定性，**不**代表训练噪声）；test 已在本轮一次性读取后冻结。 |

---

## 1. 协议、时间线与预算

### 1.1 冻结顺序（与 `analysis/test_freeze.json` 一致）

1. 发现阶段只用 valid（1,000 分子）与 train；训练代码里 test 从未被构造。
2. 发现阶段关闭时按 **seed0/1 的 best-valid 均值**排序（并列看 W20，再看 GPU-h），选出赢家家族。
3. 赢家家族补 seed2/3；同期跑 A-R0 的 seed2/3 作为匹配基线。
4. 生成 `analysis/frozen_shortlist.json` 并提交（含 checkpoint 路径与记录的 valid MAE）；
   `zps_test.py` 先重算 valid、逐项核对到 1e-6，写 `analysis/test_freeze.json`（含 checkpoint sha256），
   **之后**才第一次读官方 test 标签。
5. test 数只用于报告，不参与任何选择；轮次内没有第二次"盲测"。

### 1.2 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 16:06 | 轮次开始，`res` 体检（2×A100、torch 2.5.1+cu124） |
| 16:18 / 16:19 | 首批 R1+EMA 训练上卡（后在收尾阶段暴露 EMA 保存 bug） |
| 17:15 | 修复 EMA 保存；重跑 A/B-R1 |
| 17:21 | C 替代臂（DBT 600ep）与 A-R0 seed1 上卡 |
| 18:51 | C-DBT seed0 完成（1.476 GPU-h） |
| 19:41 | B-R0 seed2 上卡 |
| 20:52 | B-R0 seed3 上卡（冻结前的最后一块拼图） |
| {{FREEZE_TIME}} | 短名单冻结 → 一次性 test 读取 |
| {{END_TIME}} | 报告与交付物收尾（硬截止 00:06 前） |

### 1.3 预算

| 项 | 值 |
|---|---|
| 墙钟 | 8.0 h（硬） |
| GPU·h 上限 | 16.0（硬）/ 15.0（软） |
| 实际 GPU·h（含失败） | **{{TOTAL_GPU_H}}** |
| 失败/取消浪费 | 0.88（两次 EMA 保存失败，跑满 600 epoch）+ 0.09（C seed1 首次启动 5 min 后取消） |

逐项成本见 `LEDGER.md` §1–§3。

---

## 2. 候选池与配方

### 2.1 主体

| 臂 | 主体 | 参数 | 来源 |
|---|---|---|---|
| A | `ad2_models.P3NullModel`：P3 局部编码器（SSL 初始化、微调）+ Null composer + reader | 413,505 | `zinc_recipe_trunk_v2` |
| B | `dcc_models` L 臂：P3 编码器 + 双通道（局部/全局）composer | 494,193 | `zinc_dual_channel_composer_v1` |
| C | 见 §2.3：GRIT 稠密端口（已实现/已验 parity）→ 因成本回退到原生 DBT（452,523） | 452,523 | `zinc_directed_bond_transformer_v1` |

两个 A/B 主体都从同一份 **seed 匹配的 train-only SSL 编码器** 初始化
（`pretrain_encoder_s0.pt`，sha256 `006fabc1…`；seed≥1 用 `s1`，因为只存在两份 SSL 编码器）。
这一点对"同协议对比"是必要的，也限制了 seed 的解释范围（见 §6.3）。

### 2.2 配方（均在启动前声明）

| 名称 | 定义 |
|---|---|
| R0 | AdamW lr 1e-3→1e-5，wd 1e-5，dropout 0，L1，batch 128，600 ep（47,400 updates），warmup 500，clip 5 |
| R1 | lr 5e-4→5e-6，wd 1e-4，其余同 R0 |
| R2 | R0 + wd 1e-3 + post-encoder dropout 0.05（**未运行**，见 §7） |
| R3 | R1 + SmoothL1 β=0.02（**未运行**，见 §7） |
| R0L | R0 数值 + 1200 ep 余弦（94,800 updates）——本轮唯一"针对性额外配置" |

### 2.3 C 臂：GRIT 为什么被排除（实测，不是估计）

我实现并审计了 10 层 / 64 宽 / 8 头（473,473 参数，RRWP 1..20）的**稠密 padded 端口**
`zps_grit_dense.py`，与 vendored 上游参数名/形状一一对应，并做了三项验证：

- RRWP 与冻结 cache 对照：max diff 2.4e-7；
- 与稀疏上游 forward 对照（dropout off）：1.7e-6；
- 梯度相对范数误差 9.6e-5，BN buffer 1.2e-7（`analysis/dense_parity.json`）。

然后实测单 epoch 成本（A100，独立卡，无争用）：

| 路径 | batch | s/epoch | 1500–2000 ep 投影 |
|---|---|---|---|
| 稠密端口 | 32 | 30.97 | 2000 ep = 17.2 h |
| 稠密端口 | 64 | 18.65 | 2000 ep = 10.4 h |
| 稠密端口 | 128 | 14.84 | 2000 ep = 9.5 h |
| 稀疏上游（ZSRT 同实现，8 层） | 128 | ~9.5 | 1500 ep ≥ 4 h |

全部 **超过 2 h/run 上限**；把 GRIT 砍到 ~600 epoch 虽然能塞进 2 h，但离它 625k updates 的原配方
差一个数量级，等于用一个被截断的模型冒充"原生配置"。因此按预案回退：
**C = 原生 DBT，600 epoch 完整余弦（94,200 updates）**，1.48 GPU·h/run，
并在 `runs/dbtc_/dbt_s0/search.json` 里声明horizon 由 1500→600 的偏离。

> 需要强调的是：GRIT 不是"做不出来"，而是"做出来也付不起"。稠密端口的 parity 与 RRWP 校验产物保留在轮次里，
> 下一轮若拿到 ≥5 GPU·h/run 的预算可以直接接着用。

### 2.4 EMA 策略：为什么整轮弃用

实现里 `for module_param, ema_slot in zip(ema_model.parameters(), ema_state)` 的拷贝方向写反
（把刚构建的模型写进了 EMA buffer），于是 EMA 分支每轮评估的都是随机初始化：valid 恒为 **1.60004**，
train MAE 1.63（A-R1 与 B-R1 都是如此）。发现后已修正为 `module_param.copy_(ema_slot)` 并用
"载入历史 best.pt → 拷贝 → 重算 MAE" 的手工检查确认（修好后同一组参数重算 MAE 一致）。

但重跑一个合法 EMA 分支需要 2×0.35 GPU·h，且在冻结前无法给出 seed0/1 证据，所以本轮
**不把 EMA 作为候选策略**：A-R1/B-R1 只按 online 分支排名，`*_ema` 曲线与 checkpoint 标记作废。
这是本轮最明确的"未完成项"。

---

## 3. 发现阶段（seeds 0/1）

排序单位 = 同一 model+recipe+checkpoint-policy 的 seed 均值；本轮所有 run 的 policy 都是 online
（EMA 作废），所以等价于 model+recipe。

{{DISCOVERY_TABLE}}

关键观察：

1. **B-R0 领先**（0.076187 vs A-R0 0.079250），但两者分布重叠（B 最差 0.078537 > A 最好 0.076224），
   单看 2 seeds 不足以宣称"B 主体更好"，所以进入 S3 复现。
2. **配方方向对 A 稳定有害**：R1 让 A 变差 +0.0113（0.076224→0.087570），远超 seed 噪声；
   对 B 几乎中性（+0.0007）。这与"降低 lr + 提高 wd"在两主体上的直觉相反，说明 A 的瓶颈不是正则不足。
3. **更长日程救不了 A**：R0L 两 seed（0.076699 / 0.082283）均值 0.079491，与 R0 的 0.079250 实质相同，
   但 final-epoch valid 更差（0.0823/0.0843 vs 0.0782/0.0831）——1200 epoch 的余弦在 600 epoch 处仍处高位，
   到 1200 epoch 时反而过拟合。**"best checkpoint 出现在 377/600" 不是日程太短的证据。**
4. **C 的单 seed 有竞争力但方差巨大**：0.077260（seed0）与 0.087697（seed1），
   均值 0.082479，是全部配置中最不稳定的（sd 0.0074）。缩短 horizon 对 DBT 的伤害在 seed1 上暴露得很清楚。

---

## 4. 冻结与复现（seeds 2/3）

冻结规则：seed0/1 均值最小者 → **B-R0**（0.076187；次席 A-R0 0.079250；C 因只有 1 seed 不参与冻结）。
冻结后 B-R0 补 seed2/3，A-R0 同期补 seed2/3 作为匹配基线。

{{CONFIRM_TABLE}}

- B-R0 四 seed：{{B_R0_FOUR}}；mean {{B_R0_MEAN4}}，sd {{B_R0_SD4}}，range {{B_R0_RANGE4}}。
- A-R0 四 seed：{{A_R0_FOUR}}；mean {{A_R0_MEAN4}}，sd {{A_R0_SD4}}，range {{A_R0_RANGE4}}。
- 差 {{DIFF_MEAN}}（B 更好），约为合并 sd 的 {{DIFF_SD}} 倍：方向一致但**不是**"B 显著优于 A"，
  正确表述是"B-R0 在本协议下的 seed 均值更低，且最差 seed 仍优于 A-R0 的中位数"。
- W20（20 epoch 窗口最小中位数）与 final-epoch 的排序与 best-valid 一致，说明不是"挑中了尖峰"。

---

## 5. 终端 test 读取

冻结文件：`analysis/test_freeze.json`（先写，后读 test）；短名单：`analysis/frozen_shortlist.json`。
每条目在读取 test 前都重算 valid 并要求与记录的 best-valid 一致到 1e-6。

管道自检：A-R0 s0 与 B-R0 s0 的历史 test 数（0.069865 / 0.066267）在本轮重算中得到
{{SELF_CHECK}} —— 说明数据、checkpoint 与评估路径与历史一致。

{{TEST_TABLE}}

- 赢家 B-R0 的 test：{{B_TEST_LINE}}。
- test 与 valid 的差（test − valid）：{{VALID_TEST_GAP}}，与历史轮次同量级（历史 A/B ≈ −0.006…−0.012），
  说明本轮没有因为"只调 valid"而产生 test 崩坏。
- 逐分子文件：`analysis/test_pred_<name>_s<seed>.csv`（row, y_raw, pred, signed_error, abs_error），
  valid 逐分子见 `analysis/per_molecule_valid.csv` 与 `analysis/valid_pred_<name>_s<seed>.npz`。

---

## 6. 证据等级与稳健性

### 6.1 已经站得住的结论

- 在官方 split、R0 配方、≤500k 参数、单模型约束下，**本轮所有主体都到不了 test 0.060**；
  最好的家族 B-R0 的 test 均值 {{B_TEST_MEAN}}，与目标的差距 {{GAP}} 大于任何已试杠杆带来的改善（≤0.005）。
- R0 > R1 对 A 主体成立（Δ=+0.0113，3× seed sd）；对 B 主体 R0 ≈ R1。
- 1200 epoch 长日程对 A 无收益（两 seed 一致）。
- GRIT 在本仓库现有实现下 ≥4 h/run（含稠密端口实测），2 h 内无法给出原生配置。

### 6.2 不能声称的

- 不能声称"B 主体本质上优于 A"：4 seeds 的均值差 0.0045，落在 2–3 个 seed sd 内，且 test 上
  {{TEST_DIFF_NOTE}}。
- 不能声称"C/DBT 比 A/B 差"：C 是**缩短 horizon** 的版本（600/1500），历史全 horizon DBT（3.69 GPU·h）
  valid 0.077768、test 0.069401，与本轮 B-R0 同量级；本轮只证明"1.5 h 预算下的 DBT 不够稳"。
- 不能把行 bootstrap 当作"训练可重复性"：行 bootstrap 只描述 1,000 个分子这个样本的不确定性；
  训练重复性只能由 seed range 描述（本报告两处分开写）。

### 6.3 已知的口径限制

1. **SSL 编码器只有 s0/s1 两份**：seeds ≥2 复用 s1 的编码器初始化，因此 seed 2/3 的离散度
   低估了"完全独立初始化"的方差。
2. **test 已花**：本轮 test 读取是"已暴露 test 的描述性复核"，不是新的盲测；所有选择都发生在读取之前。
3. **valid 选择偏差**：best-valid 是在 600 个 epoch 上取最小，本身带选择偏差（≈0.001–0.002）；
   W20 与 final 列就是为识别这一点而报的。

### 6.4 冻结分组（valid）

{{GROUP_TABLE}}

unseen（251）上的 MAE 是 seen（749）的 2.5–3 倍，tail（65）又是 unseen 的 1.5–2 倍；
本轮没有任何配方改变这个结构（R1 让 tail 更差）。这与历史结论一致：
**性能瓶颈在"训练中未见的原子/环境"子群，而不是整体拟合。**

---

## 7. 未达成目标的原因与开放问题

- **差距的量级**：要 test <0.060，valid 需要 ≈0.066（历史 test−valid ≈ −0.007）。本轮最好家族的
  valid 均值 0.0757，差 0.010 —— 相当于 3–4 个 seed sd。已试杠杆（lr、wd、loss、dropout、horizon、
  第三主体、双通道 composer）单点收益都 ≤0.005，且多为负。
- **没跑的两格**：R2（强正则）与 R3（SmoothL1）在本轮预算内没有排上。R1 的方向（更强正则）对 A 已
  确定有害，R2 是同一方向的加强版，预期收益为负；R3 是唯一未探索的损失形状，下一轮应优先补。
- **真正的开放问题**：
  1. 在 ≥4 h/run 预算下，GRIT（10 层、RRWP）能否把 valid 推到 0.070 以内？本轮只证明了"2 h 不够"。
  2. 全 horizon DBT（1500 ep，3.7 h）在两 seed 下的均值是多少？单 seed 的 0.0778 不足以定论。
  3. EMA 修好后是否稳定改善 best-valid？本轮无数据。
  4. seeds ≥2 的独立 SSL 编码器缺失，是否有必要为 2/3 号 seed 重训 SSL？

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

# 3) 重算 valid 表 / 分组 / 对照
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_analyze.py
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_report.py

# 4) 终端 test（只在冻结后执行一次）
uv run python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_shortlist.py --spec <spec.json>
git add -A && git commit -m "freeze shortlist" && rr deploy res
rr run res zps-test --pool res-gpu0 --gpus 1 --time 0:20:00 \
  --result tracks/structural-sufficiency/zinc_performance_search_v1/analysis \
  -- python tracks/structural-sufficiency/zinc_performance_search_v1/code/zps_test.py --device cuda:0
```

C 替代臂（若下一轮要复现）：

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
| `PLAN.md` / `DECISIONS.md` | 预注册设计、逐条决策与偏离 |
| `LEDGER.md` | 全部 run（含失败/取消）、配置、成本、状态 |
| `code/zps_runner.py` | A/B 臂 + 配方 R0–R3/R0L 的声明式入口 |
| `code/zps_grit_dense.py` / `code/zps_dense_timing.py` | GRIT 稠密端口与其成本测量 |
| `code/zps_dbt.py` | C 替代臂（DBT，可配 horizon） |
| `code/zps_analyze.py` / `code/zps_report.py` | valid 表、分组、W20、bootstrap、报告表格生成 |
| `code/zps_shortlist.py` / `code/zps_test.py` | 冻结 → 一次性 test 读取 → 逐分子产物 |
| `analysis/main_table.{md,json}` | 全 run 的 valid 指标 |
| `analysis/report_tables.md` | 报告里所有表格的机器生成版 |
| `analysis/test_table.{md,json}` / `analysis/test_freeze.json` | 终端 test 结果与冻结记录 |
| `analysis/test_pred_*.csv` / `analysis/per_molecule_valid.csv` | 逐分子 valid/test 预测与误差 |
| `runs/<run>/` | 每个 run 的 train.json、curve.csv、predictions、search.json（重产物在远端） |

---

## 10. 一句话给下一轮

本轮把"在 ≤500k 参数、单模型、2 h/run 的约束下，靠训练配方/主体选择把 ZINC valid 从 0.0762 推到 0.066"
这件事**证伪**了：试过的每个杠杆都在 ±0.005 内，而目标是 0.010。
下一轮要么给 GRIT 这类显式关系模型 ≥4–5 h/run 的预算并跑到原生 horizon，
要么换问题设定（更大模型/更多数据/不同 readout），否则继续在同一格子里调参的期望收益接近 0。
