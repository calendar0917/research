# REPORT — LCRC MolPCBA 单种子续训、学习率配方诊断与官方 Test 评估

Round: `lcrc-pcba-continuation-recipe-v1`  
Status: **COMPLETED**  
Seed: **0 only**  
Test label: **`EXPOSED_TEST_DESCRIPTIVE`**（本轮完成后 Test split 已暴露；后续使用必须披露）

---

## 0. 直接回答

### 0.1 A/G/V 在 epoch40、epoch60、epoch80 的 Validation AP

| 模型 | AP40 | AP50（L / R） | AP60（L / R） | AP70（仅 V-R） | AP80（仅 V-R） |
|---|---:|---:|---:|---:|---:|
| A（attention pooling, no-MP） | 0.246358691 | 0.246049618 / 0.246281916 | 0.246085356 / 0.246284525 | — | — |
| G（graph context, no bond MP） | 0.250847581 | 0.251518124 / 0.249587951 | 0.251167855 / 0.250716944 | — | — |
| V（bond MP + graph context） | 0.257776567 | 0.258966905 / 0.257996405 | 0.258767144 / 0.260615376 | 0.259995658 | 0.260868081 |

A、G 按预注册 Stage-2 规则停在 epoch60；**不能把它们的 epoch60 结果当作 epoch80 结果**。  
只有 V-R 晋级并真正训练到 epoch80。

### 0.2 固定低 LR（L）与 LR restart（R）哪个更好？

| 模型 | D_L = AP_L60 − AP40 | D_R = AP_R60 − AP40 | D_RL = AP_R60 − AP_L60 | R 胜出任务数 / 127 | 标签 |
|---|---:|---:|---:|---:|---|
| A | −0.000273335 | −0.000074166 | **+0.000199169** | 67 | `NO_CONTINUATION_GAIN` |
| G | +0.000320274 | −0.000130637 | **−0.000450911** | 63 | `NO_CONTINUATION_GAIN` |
| V | +0.000990577 | **+0.002838809** | **+0.001848232** | 75 | `NO_CLEAR_RECIPE_WINNER` |

- 没有一个模型达到 `RESTART_PROMISING`（要求 `D_RL ≥ 0.003` 且 R 至少赢 70/127 个任务）。
- V-R 是唯一在 40→60 绝对增益上跨过 `+0.002` 门槛的 run（+0.002839），但 R 相对 L 的配对优势只有 +0.001848，低于 0.003；因此配方标签仍是 `NO_CLEAR_RECIPE_WINNER`，不是强 restart 证据。
- G 的 L 略好于 R（−0.000451），A 的 L/R 差异可忽略（+0.000199）。**没有跨模型一致的“提高 LR 更好”趋势。**

### 0.3 目前观察到的最高 Validation AP

**0.260868081**，来自 **V-R epoch80**（`stage2_V_R/checkpoint_epoch80.pt`，SHA256 `e5f0f7d6…d82969`）。  
这是本轮固定选择规则下的 observed best，不是理论上限。

### 0.4 继续训练有没有明确收益？哪个模型更接近平台？

- **A：无收益。** 40→60 的 L/R 都略低于 AP40；Train AP 上升，但 Valid AP 停滞。没有 Stage 2。
- **G：无实质收益。** 最好的 L60 仅比 AP40 高 +0.000320；R60 更低。没有 Stage 2。
- **V：小幅收益，接近平台。** V-R 40→60 +0.002839，60→80 仅 +0.000253；70→80 +0.000872。Stage-2 标签为 `SMALL_LATE_GAIN`。
- 因而：**A/G 是停滞或退化；V 是“小幅收益、接近平台”，不是持续线性上升。** A/G 的 80 epoch 行为因规则提前停止而无法判断。

### 0.5 C/A/G/V 官方 Test AP

| 逻辑模型 | 实际 checkpoint | Validation AP（replay） | Official Test AP | Valid − Test |
|---|---|---:|---:|---:|
| C1 epoch40 | C-40 | 0.231944668 | **0.227246093** | −0.004698576 |
| A epoch40 | A-40 | 0.246356419 | **0.246068304** | −0.000288116 |
| G epoch40 | G-40 | 0.250848332 | **0.241696564** | −0.009151768 |
| V epoch40 | V-40 | 0.257771265 | **0.247694250** | −0.010077014 |
| A selected | A-40（alias） | 0.246356419 | **0.246068304** | −0.000288116 |
| G selected | G-L60 | 0.251168328 | **0.241700788** | −0.009467541 |
| V selected | V-R80 | 0.260874668 | **0.246589726** | −0.014284941 |

续训前后 Test 描述性变化：

- A selected 与 A40 是同一 checkpoint：ΔTest = **0**。
- G selected（G-L60）相对 G40：ΔTest = **+0.000004**，实质不变；逐任务 54 改善 / 73 退化。
- V selected（V-R80）相对 V40：ΔTest = **−0.001105**；逐任务 68 改善 / 59 退化。
- **V 的 Validation 增益没有转移到 Test；G 基本持平；A 没有变化。** 这是单 seed 的描述性外部评估，不是显著性检验。

### 0.6 无键上传播的 G/A 是否缩小了与 V 的差距？

**没有。** 在可比的 epoch40 和 epoch60 终点上，差距反而扩大：

| 比较 | epoch40 | epoch60-L | epoch60-R |
|---|---:|---:|---:|
| V − A | +0.011417876 | +0.012681788 | +0.014330852 |
| V − G | +0.006928986 | +0.007599289 | +0.009898432 |

V 在 epoch60 仍在提高，而 A/G 基本停在 AP40 附近；因此 G 没有接近 V。  
epoch80 只有 V 运行，**没有可比的 A/G epoch80**，不能报告 80 epoch 的架构差距。

### 0.7 这轮能确认什么，什么仍然未知？

**可以确认：**

- 四个源 checkpoint（C/A/G/V）的模型参数、AdamW moments、step、scheduler、RNG 可严格恢复；epoch40 Validation AP 重放误差均 ≤1e-4。
- L/R 从同一 epoch40 状态分叉，共享同一 batch order；首步模型参数和 optimizer moments 在审计容差内一致。
- A/G 在 40→60 没有可利用的 Validation AP 增量。
- V-R 在 40→80 有小幅 Validation 增益（+0.003092），但 60→80 只有 +0.000253，接近平台。
- 官方 Test 在本轮预测冻结之后只读取一次；官方 evaluator 与独立逐任务 AP 一致（最大差 ~1e-16）。

**仍然未知：**

- 模型/数据集的理论性能上限；不能通过拟合最后几个 epoch 外推 ceiling。
- A/G 在 80 epoch 的行为（按规则未运行）。
- 其他学习率峰值、warmup 长度、schedule 形状是否更好；本轮只比较了预注册的 L/R。
- 单 seed 的 run-to-run 波动；本轮没有 seed 1/2，不能把 0.001 量级差异当作稳定效应。
- 完整 80-epoch from-scratch 配方是否更好；本轮 R 只是从 epoch40 checkpoint 的 LR restart，不是 from-scratch 对照。
- Test 上的 −0.0011（V）或 +0.000004（G）是否只是采样噪声；本轮不据此改变任何选择。

### 0.8 实际 GPU-hours 与墙钟

- **总 GPU 消耗：5.144 GPU-hours**（含两次 Stage-0 preflight 0.0208、Stage-1 4.3006、Stage-2 0.7950、Test freeze 0.0274）。
- **目标 7 GPU-h / 硬上限 8 GPU-h：均满足。**
- **正式 GPU 工作墙钟：约 3 h 43 min**（2026-11-06 13:23:30 → 17:06:35 CST，preflight 到 Test scoring）。含本地审计/部署的整轮会话约 4 h 15 min，仍低于 5 h 目标。
- 最多同时 2 张 A100，符合约束。

---

## 1. 执行事实

### 1.1 源 checkpoint 恢复

| 模型 | 源 checkpoint SHA256 | 历史 AP40 | GPU replay AP40 | 绝对差 | 参数 | state tensors | optimizer params | step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | `f110935a…93a437` | 0.246356096 | 0.246358691 | 2.59e−6 | 287,644 | 57 | 51 | 109,520 |
| G | `e363e372…6849bb` | 0.250848114 | 0.250847581 | 5.33e−7 | 286,912 | 53 | 47 | 109,520 |
| V | `2d75e444…8939a0` | 0.257772325 | 0.257776567 | 4.24e−6 | 287,808 | 55 | 49 | 109,520 |
| C | `43ff8395…ea0f4e` | 0.231944774 | 0.231944668 | 1.05e−7 | 245,184 | 37 | 31 | 109,520 |

所有源 checkpoint：`global_step = 109,520 = 40 × 2,738`；AdamW 每参数 step 也是 109,520；scheduler 记录 `lr_now = 1e-5`；保存状态为 `COMPLETED`。

恢复审计（`analysis/paired_initialization_audit.json`、`analysis/source_checkpoint_audit.json`）：

- `state_dict` bitwise 相等。
- AdamW state tensor 完整比对通过；参数 ID、moments、step 全部一致。
- RNG（Python / NumPy / torch CPU / torch CUDA）恢复。
- L/R 首步模型参数与 optimizer moments 在审计容差内相同；V 的最大首步梯度差约 4.7e−10，属于 CUDA 浮点归约噪声，低于 1e−7 门槛。
- A/G 的 `NMPModel` 与 V 的 `SGModel` 没有混用。

### 1.2 固定数据身份

| 项目 | 值 |
|---|---|
| Dataset | `ogbg-molpcba` |
| Train / Valid / Test | 350,343 / 43,793 / 43,793 |
| Feature spec hash | `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee` |
| Norm fingerprint | `6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb` |
| Split partition SHA256 | `c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f` |
| Label artifact SHA256 | `5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01` |
| Cache manifest SHA256 | `caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035` |

- 训练 artifact 的 Test 行全为 NaN；loader 拒绝任何有限 Test 标签。
- 新生成的 epoch41–80 batch plan 只包含 Train 行，每个 epoch 是完整 Train 行的置换。
- Batch plan 全局 SHA256：`67e865291cd03d1bea699936fcbd10bfa230e87460cf161502abef9b74acef6d`。
- Plan 的生成独立于全局 dropout RNG（专用 `torch.Generator(seed=0)`），并用前 40 个 epoch 的 randperm 作为 41–80 的延续。

### 1.3 学习率配方

| 配方 | 定义 | schedule SHA256 |
|---|---|---:|
| L | 从 step 109,520 起恒为 `1e-5` | `290cd33809b49434ecb7addda387406a43bb7cf5b831dc4b21794e5692232619` |
| R | epoch41：`1e-5 → 5e-5` 线性；epoch42–80：per-step cosine `5e-5 → 1e-5`，horizon 为 epoch40→80 | `79b2e9ef8b56d692145a7be68f4e1cd3d74c3987a0c4b315294fe56957475bfd` |

R 只重启 LR schedule，不重置模型参数或 AdamW moments；不能称为 80-epoch from-scratch 配方。

### 1.4 训练配置

Batch 128；AdamW（betas 0.9/0.999，eps 1e-8，weight decay 1e-3）；dropout 0.10；FP32；observed-entry masked BCE；gradient clip 5.0；无 early stopping；seed 0；完整官方 Train。

所有 run 均无 gradient clipping 事件、无非有限 loss/gradient/logit；峰值显存约 0.13 GB allocated / 0.20 GB reserved。

---

## 2. Stage 1：epoch40 → 60

### 2.1 六条 run

| Run | 起始 | 结束 | GPU-h | 首步 loss | 首步 grad norm | 最终 AP60 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| A-L | 41 | 60 | 0.8080 | 0.035442945 | 0.070127077 | 0.246085356 | COMPLETED |
| A-R | 41 | 60 | 0.6746 | 0.035442945 | 0.070127077 | 0.246284525 | COMPLETED |
| G-L | 41 | 60 | 0.7602 | 0.033638556 | 0.082961060 | 0.251167855 | COMPLETED |
| G-R | 41 | 60 | 0.6131 | 0.033638559 | 0.082961061 | 0.250716944 | COMPLETED |
| V-L | 41 | 60 | 0.7933 | 0.033162121 | 0.081046253 | 0.258767144 | COMPLETED |
| V-R | 41 | 60 | 0.6514 | 0.033162117 | 0.081046261 | 0.260615376 | COMPLETED |

同一模型的 L/R 首步 loss 差异在 1e−9 量级、grad norm 差异在 1e−8 量级，符合复现容差。  
Stage-1 总 GPU：**4.3006 GPU-h**。

### 2.2 配方比较

| 模型 | D_L | D_R | D_RL | R 胜 / L 胜 / 平 | 标签 |
|---|---:|---:|---:|---:|---|
| A | −0.000273335 | −0.000074166 | +0.000199169 | 67 / 60 / 0 | `NO_CONTINUATION_GAIN` |
| G | +0.000320274 | −0.000130637 | −0.000450911 | 63 / 64 / 0 | `NO_CONTINUATION_GAIN` |
| V | +0.000990577 | +0.002838809 | +0.001848232 | 75 / 52 / 0 | `NO_CLEAR_RECIPE_WINNER` |

解读：

- **A**：两种配方都没有把 AP 提高；R 略好于 L，但差距只有 +0.000199。
- **G**：L 略好于 R；两者都未达到 +0.002 的 continuation-gain 门槛。
- **V**：R 明显好于 L（+0.001848，75/127 任务胜出），但未达到 `RESTART_PROMISING` 的 0.003 门槛；绝对增益 +0.002839 使 V-R 获得 Stage-2 资格。

不能把 A/G 的 L/R 差异解释为跨架构的 LR 效应；这些是模型内配对比较。

---

## 3. Stage 2：V-R 60 → 80

只有 V-R 通过预注册资格：

- AP60 0.260615376 ≥ AP40 0.257776567 + 0.002。
- AP60 0.260615376 ≥ AP50 0.257996405 − 0.001。
- 所有正确性检查通过。

A/G 没有 Stage-2 run。

| 评价点 | AP |
|---|---:|
| epoch40 | 0.257776567 |
| epoch50（R） | 0.257996405 |
| epoch60（R） | 0.260615376 |
| epoch70（R） | 0.259995658 |
| epoch80（R） | **0.260868081** |

| 增量 | 值 |
|---|---:|
| 40→60 | +0.002838809 |
| 60→80 | **+0.000252705** |
| 40→80 | **+0.003091514** |
| 70→80（最后十个 epoch） | +0.000872423 |

其它诊断：

- Valid BCE：0.045642753（60）→ 0.045733257（80），**变差 +0.000090504**。
- Train AP：0.442683268（60）→ 0.452668758（80）。
- Train−Valid AP gap：0.182067891（60）→ 0.191800677（80），**扩大 +0.009732785**。
- 任务层 80 vs 60：66 改善 / 61 退化。
- 分层：≥20 正例任务 +0.000487（107 个任务，58 好 / 49 差）；5–19 正例 −0.000970（18 个任务，7 好 / 11 差）；<5 正例 −0.001304（仅 2 个可评分任务，1 好 / 1 差）。
- 标签：**`SMALL_LATE_GAIN`**（60→80 增益在 0 到 0.003 之间）。

结论：V-R 在 40→80 有小的绝对增益，但 60→80 已接近平台，且 Train/Valid gap 继续扩大；不能据此声称模型还有持续、可外推的长期收益。

---

## 4. Validation 冻结选择

规则：先取最高 Validation AP；差距 ≤1e−4 时选训练 epoch 更少者；同 epoch 再优先 L；不允许用 Test。

| 模型 | 候选（AP） | 选中 | 理由 |
|---|---|---|---|
| A | epoch40 0.246358691；L60 0.246085356；R60 0.246284525 | **epoch40** | 最高 AP 是 epoch40；R60 差距 7.42e−5 ≤ 1e−4，按规则选更少 epoch 的 epoch40 |
| G | epoch40 0.250847581；L60 0.251167855；R60 0.250716944 | **G-L60** | L60 最高，且比 epoch40 高 3.20e−4 > 1e−4 |
| V | epoch40 0.257776567；L60 0.258767144；R60 0.260615376；R80 0.260868081 | **V-R80** | R80 最高 |

因此：

- A 的最终模型与 A epoch40 相同，Test 评估中标记为 **alias**，不重复计分。
- G 的最终模型是 **G-L60**（`checkpoint_epoch60.pt`，SHA256 `0679e305…f9574a`）。
- V 的最终模型是 **V-R80**（`checkpoint_epoch80.pt`，SHA256 `e5f0f7d6…d82969`）。

固定 epoch60 配方对照与最终 Validation-selected 结果是两个不同层次，不能混为一个结论：例如 A 的配方对照显示 R60 略高于 L60，但最终选择仍是 epoch40；G 的配方对照显示 L60 最高，最终也选 L60；V 的配方对照 R60 已高于 L60，Stage2 后 R80 再略高。

---

## 5. 官方 Test：一次冻结后评分

### 5.1 冻结

- 预测冻结 manifest：`analysis/test_prediction_freeze.json`
- Seal SHA256：`6b205134db4c37cc2d57134581f99026ed3a57f06a6cdfe712eebc8e8d4f0a6f`
- 冻结时 `test_labels_read = false`；7 个逻辑模型，6 个唯一 checkpoint（A_selected 是 A_epoch40 的 alias）。
- Test row IDs SHA256：`136acbfea1915317…`；43,793 行，shape `(43793, 128)`，dtype float32。
- 每个预测文件在评分前都重新校验 SHA256；评分还校验 selection manifest 的 SHA256 没有在冻结后变化。
- 官方 Test 标签只在 freeze seal 验证通过后，由 `cc_test.py score` 打开；此前训练 loader 从未读取 Test 标签。
- 历史 PCBA rounds 未读取 Test；`lcrc_sparse_identity_v1` 的 Test 暴露属于 MolHIV，不是本数据集的 MolPCBA。本轮之后 MolPCBA Test 已暴露，未来使用必须披露。

### 5.2 Test AP 与 Valid/Test 对照

| 逻辑模型 | Valid AP | Official Test AP | 独立逐任务 AP | 官方−独立 | Test BCE | Valid−Test |
|---|---:|---:|---:|---:|---:|---:|
| C1 epoch40 | 0.231944668 | 0.227246093 | 0.227246093 | 2.78e−17 | 0.048784948 | −0.004698576 |
| A epoch40 | 0.246356419 | 0.246068304 | 0.246068304 | 5.55e−17 | 0.047937263 | −0.000288116 |
| G epoch40 | 0.250848332 | 0.241696564 | 0.241696564 | 2.78e−17 | 0.048594959 | −0.009151768 |
| V epoch40 | 0.257771265 | 0.247694250 | 0.247694250 | 2.78e−17 | 0.048068887 | −0.010077014 |
| G selected（G-L60） | 0.251168328 | 0.241700788 | 0.241700788 | 2.78e−17 | 0.048790343 | −0.009467541 |
| V selected（V-R80） | 0.260874668 | 0.246589726 | 0.246589726 | 8.33e−17 | 0.048601323 | −0.014284941 |
| A selected（alias A40） | 0.246356419 | 0.246068304 | 0.246068304 | 5.55e−17 | 0.047937263 | −0.000288116 |

官方 `ogbg.graphproppred.Evaluator("ogbg-molpcba")` 与独立 sklearn/per-task AP 的差在 1e−16 量级，说明评分路径一致。

### 5.3 续训前后的 Test 描述性变化

| 模型 | Test(epoch40) | Test(selected) | ΔTest | 改善任务 / 退化任务 |
|---|---:|---:|---:|---:|
| A | 0.246068304 | 0.246068304（alias） | 0 | 0 / 0（同一 checkpoint） |
| G | 0.241696564 | 0.241700788（G-L60） | **+0.000004224** | 54 / 73 |
| V | 0.247694250 | 0.246589726（V-R80） | **−0.001104524** | 68 / 59 |

- G 的 Test 变化实质为零；V 的 Test 略降；A 无变化。
- 这与 Validation 上的 V 小幅增益方向不一致，说明该增益没有在 Test 上重现。
- Test 仅用于描述性外部评估；没有根据 Test 改变模型、recipe、epoch 或选择。

### 5.4 Test 标签统计

| 项目 | Validation | Test |
|---|---:|---:|
| 行数 | 43,793 | 43,793 |
| 观测标签数 | 3,477,656 | 3,376,672 |
| 正例数 | 55,964 | 56,698 |
| 观测正例率 | 0.016092448 | 0.016791089 |

Valid/Test 正例率差：**+0.000698641**。Test 的有效标签略少、正例率略高。

### 5.5 推理成本

- Test freeze（6 个唯一模型，含 Validation 重放 + Test logits）：**98.78 s**，**0.02744 GPU-h**。
- 单模型 Test 推理约 6.8–7.2 s；Validation 重放约 8.3–10.4 s。
- 逐模型 logits、shape、dtype、SHA256 均在 freeze manifest 中。

---

## 6. Q1–Q5 汇总

### Q1：40 epoch 是否限制了模型表现？

| 模型 | AP40 | AP60（最好配方） | AP80 | 40→60 / 40→80 | 判断 |
|---|---:|---:|---:|---:|---|
| A | 0.246358691 | 0.246284525（R） | 未运行 | −0.000074 / — | 停滞或退化；80 无法判断 |
| G | 0.250847581 | 0.251167855（L） | 未运行 | +0.000320 / — | 小幅收益，接近平台；80 无法判断 |
| V | 0.257776567 | 0.260615376（R） | 0.260868081（R） | +0.002839 / +0.003092 | 小幅收益，接近平台 |

V 的 60→80 只有 +0.000253，最后十个 epoch +0.000872；没有持续上升的证据。A/G 在 60 已无增益，按规则停止。

### Q2：是不是 LR 衰减过早？

**没有跨模型的一致证据。** R 相对 L：

- A：+0.000199（可忽略）。
- G：−0.000451（L 略好）。
- V：+0.001848（R 略好，75/127 任务胜出，但低于 0.003 门槛）。

因此只能说：**在本轮预注册的两种配方下，提高 LR 没有普遍解锁后续性能；V 上有一个小的正向交互，但不足以称为强 restart 效应。**  
不能推广为“R 失败，所以 80-epoch cosine from-scratch 一定没用”；本轮没有 from-scratch 对照。

### Q3：性能上限在哪里？

- **Observed best**：0.260868081（V-R80）。
- **Late-training gain**：V 40→60 +0.002839；60→80 +0.000253；40→80 +0.003092。
- **Empirical plateau evidence**：V 60→80 增长很小，70→80 +0.000872；A/G 在 60 已平。曲线更像接近平台，而不是持续线性上升。
- **Unknown upper bound**：本轮不能识别理论 ceiling；不能通过拟合最后几个 epoch 虚构一个确定上限。若要做下一轮资源规划，只能把“V 再增加 20 epoch 可能只有 ~0.001 量级”当作粗略假设，不是测量到的上限。

### Q4：无 MP 路线是否缩小了与 V 的差距？

**没有。** epoch40 的 V−A = +0.011418、V−G = +0.006929；到 epoch60-R 扩大到 +0.014331、+0.009898。G 没有接近 V。  
因为 A/G 没有运行到 80，不能报告 epoch80 的架构差距；不能把 V 的 80 epoch 结果与 A/G 的 60 epoch 结果直接当作同终点比较。

### Q5：收益是否真实泛化？

**Train AP 上升、Valid AP 停滞/仅小升，是 A/G/V 的共同现象。**

| 模型 / 配方 | Train AP40 | Train AP60 | ΔTrain | Valid AP40 | Valid AP60 | ΔValid | Gap40 | Gap60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-L | 0.419657 | 0.424511 | +0.004854 | 0.246359 | 0.246085 | −0.000273 | 0.173298 | 0.178425 |
| A-R | 0.419657 | 0.433799 | +0.014142 | 0.246359 | 0.246285 | −0.000074 | 0.173298 | 0.187514 |
| G-L | 0.410716 | 0.415985 | +0.005268 | 0.250848 | 0.251168 | +0.000320 | 0.159869 | 0.164817 |
| G-R | 0.410716 | 0.425538 | +0.014821 | 0.250848 | 0.250717 | −0.000131 | 0.159869 | 0.174821 |
| V-L | 0.428159 | 0.433028 | +0.004868 | 0.257777 | 0.258767 | +0.000991 | 0.170383 | 0.174261 |
| V-R | 0.428159 | 0.442683 | +0.014524 | 0.257777 | 0.260615 | +0.002839 | 0.170383 | 0.182068 |

按 **Validation 正例数** 分层（不是 Train 正例数）：

| 模型 / 对比 | <5 正例 | 5–19 正例 | ≥20 正例 |
|---|---:|---:|---:|
| A: L60 − AP40 | +0.008856（2 个可评分任务） | −0.000271（18） | −0.000444（107） |
| A: R60 − AP40 | +0.020701（2） | −0.003697（18） | +0.000147（107） |
| G: L60 − AP40 | −0.002385（2） | +0.000944（18） | +0.000266（107） |
| G: R60 − AP40 | −0.013025（2） | −0.005365（18） | +0.000991（107） |
| V: L60 − AP40 | −0.001726（2） | −0.000429（18） | +0.001280（107） |
| V: R60 − AP40 | −0.003084（2） | **+0.007217（18）** | **+0.002213（107）** |
| V-R: 80 − 60 | −0.001304（2） | −0.000970（18） | +0.000487（107） |

要点：

- V-R 的宏观增益主要来自 107 个 ≥20 正例任务（+0.002213）和 18 个 5–19 正例任务（+0.007217），不是少数低支持度任务。
- <5 正例层只有 2 个可评分任务（第 3 个任务无足够标签），其均值波动很大；**不能用它作为稀疏任务稳健性证明**。
- Train AP 上升幅度（A-R +0.0141、G-R +0.0148、V-R +0.0145）远大于 Valid AP 变化；V-R 的 60→80 Train−Valid gap 还扩大了 +0.009733。这更像继续拟合 Train 而非获得可泛化增益。

---

## 7. 资源与运行记录

### 7.1 正式 run

| Stage | Experiment | rr run_id | Pool / node | Started (CST) | Completed (CST) | GPU-h | Checkpoint SHA256 |
|---|---|---|---|---|---:|---:|---|
| Stage1 | A-L | `cc-s1-A-L-20261106-132727-12e0ffc7` | res-gpu0 / a100-2 | 13:27:30 | 14:16:08 | 0.8080 | `06f1e89c…` |
| Stage1 | A-R | `cc-s1-A-R-20261106-132737-189d7253` | res-gpu1 / a100-2 | 13:27:40 | 14:08:19 | 0.6746 | `ccfe0e77…` |
| Stage1 | G-L | `cc-s1-G-L-20261106-142307-93b6fa6e` | res-gpu0 / a100-2 | 14:23:10 | 15:08:57 | 0.7602 | `0679e305…` |
| Stage1 | G-R | `cc-s1-G-R-20261106-142319-3ea890c1` | res-gpu1 / a100-2 | 14:23:21 | 15:00:18 | 0.6131 | `3046d32c…` |
| Stage1 | V-L | `cc-s1-V-L-20261106-151448-cee2c74b` | res-gpu0 / a100-2 | 15:14:56 | 16:02:41 | 0.7933 | `d4817d81…` |
| Stage1 | V-R | `cc-s1-V-R-20261106-151517-89c1f0f9` | res-gpu1 / a100-2 | 15:15:25 | 15:54:39 | 0.6514 | `dd127911…` |
| Stage2 | V-R | `cc-s2-V-R-20261106-160714-502f0f60` | res-gpu0 / a100-2 | 16:07:16 | 16:55:07 | 0.7950 | `e5f0f7d6…` |

所有正式训练均从 clean commit 通过 `rr` 启动。A 的 training commit 是 `4707e428…`；G/V/Stage2 是 `f9f5457f…` / `4942c889…`。这些 commit 之间 `cc_train.py`、`cc_common.py`、`cc_data.py`、`cc_restore.py` **逐字节相同**，差异只在 `cc_test.py` / `cc_analyze.py`；训练代码没有不一致。

### 7.2 GPU 预算

| 项目 | GPU-h |
|---|---:|
| Stage-0 preflight（最终版；A/G/V replay + paired first step + smoke） | 0.0101 |
| Stage-0 preflight（首版，已被 PinnedStager 版取代；仍计入预算） | 0.0107 |
| Stage-1 A-L / A-R | 0.8080 / 0.6746 |
| Stage-1 G-L / G-R | 0.7602 / 0.6131 |
| Stage-1 V-L / V-R | 0.7933 / 0.6514 |
| Stage-2 V-R | 0.7950 |
| Test freeze（6 个唯一模型） | 0.0274 |
| **Total** | **5.1438** |

目标 7 GPU-h、硬上限 8 GPU-h：**均满足**。  
`gpu_hours_process` 是完整训练进程墙钟（含数据加载、Validation、最终 Train 评估），因此是 GPU kernel 时间的上界；失败尝试、preflight 和 Test 推理均计入。

### 7.3 墙钟

- 正式 GPU 工作窗口：**13:23:30 → 17:06:35 CST = 3 h 43 min 05 s**。
- 目标 5 h / 硬上限 6 h：**均满足**。
- 本地审计、代码准备、结果整理另计；整轮会话约 4 h 15 min。

### 7.4 失败、修复与重试

正式训练无失败、无重跑。执行中的问题均为软件/部署问题，且未改变配方或模型选择：

1. **GPU preflight 首轮梯度审计**：V 的首步梯度 bitwise 不等，最大差 4.66e−10，属于 CUDA 浮点归约噪声；将审计改为 1e−7 容差比较后通过。没有改训练配置。
2. **`cc_analyze` Stage-2 tie-break bug**：V 晋级后触发 `float(dict)` 错误；修正表达式后重新生成分析。发生在正式训练之后、Test 之前，未改变任何 AP。
3. **`cc_test` np.save 临时文件后缀 bug**：`.npy.tmp` 会被 NumPy 追加 `.npy`；改为文件句柄写入后通过测试。发生在 freeze 之前。
4. **远端 dirty checkout**：两次 `rr deploy` 被之前的预检结果文件阻塞；用 `cc-cleanup`（run_id `cc-cleanup-20261106-132054-3be941c4`、`cc-cleanup-20261106-142142-c90e841e`）清除未跟踪/被修改的生成文件后重新部署。未触碰训练 artifact。
5. **Test freeze 首次实现**：修复了 prediction 路径与 selection SHA 校验；正式 freeze 一次通过。

---

## 8. 结论边界

- 本轮是 **单 seed（0）** 的配对续训诊断，不是显著性检验，也不是完整 80-epoch from-scratch 对照。
- A/G 停在 epoch60 是按预注册规则执行的，不能把它们当作 epoch80 结果。
- V-R 的 Validation 增益很小（40→80 +0.003092），且 60→80 接近平台；没有证据支持继续无界延长训练。
- 官方 Test 结果只用于描述性外部评估；V 的 Test 略降、G 基本持平、A 不变，说明 Validation 上的小增益没有在 Test 上重现。
- 本轮之后 MolPCBA Test split 已暴露；未来在同一 Test 上做模型迭代必须披露这一事实，不能称为新的盲测确认。
- 不能声称已经找到 PCBA 的理论性能上限；也不能把本轮 L/R 结果推广为所有学习率配置的最优性证明。

**本轮到此停止，等待研究者决定下一步。**

---

## 9. 关键工件

| 工件 | 路径 |
|---|---|
| Protocol | `PROTOCOL.md` |
| Source audit | `analysis/source_checkpoint_audit.json` |
| Paired initialization audit | `analysis/paired_initialization_audit.json` |
| LR schedule audit | `analysis/lr_schedule_audit.json` |
| Stage-0 final GPU preflight | `analysis/preflight_gpu.json` |
| Stage-0 superseded GPU preflight | `analysis/preflight_gpu_superseded.json` |
| Epoch41–80 plan | `analysis/epoch41_80_plan.json` |
| Stage-1 comparison | `analysis/epoch60_recipe_comparison.json` |
| Stage-2 continuation | `analysis/epoch80_continuation.json` |
| Validation selection | `analysis/validation_selection.json` |
| Test prediction freeze | `analysis/test_prediction_freeze.json` |
| Test results | `analysis/test_results.json` |
| Per-task Valid/Test | `analysis/per_task_valid_test.csv` |
| GPU ledger | `analysis/gpu_ledger.json` |
| Run registry | `analysis/run_registry.json` |
| Training runs | `runs/stage1_*`, `runs/stage2_V_R`（Git-ignored） |
| Frozen Test logits | `runs/test_freeze/test_predictions`（Git-ignored） |
