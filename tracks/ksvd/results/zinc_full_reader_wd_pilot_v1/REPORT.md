# zinc_full_reader_wd_pilot_v1 — reader weight decay 单变量配对 pilot

**决策：`FAIL`。candidate(10× reader wd) 相对 concurrent control 无改善（cal valid gain `−1.84e-5`），
远低于 0.003 gate。关闭「父 soup + 10× reader decay + H=20 warm」这一配置；本轮结束，不扫 3×/30×，
不宣布正则无用或表示已充分。**

| | cal valid MAE | raw valid | train-median bias | wall |
|---|---:|---:|---:|---:|
| 父 Full（背景参照） | 0.115066 | 0.119154 | −0.034288 | — |
| **本轮 control**（Adam, wd 1e-5, H=20, 末5 soup） | **0.115023** | 0.115010 | −0.000272 | 420 s |
| **本轮 candidate**（仅 reader wd 1e-4 = 10×） | **0.115041** | 0.115000 | −0.001582 | 424 s |

主要 gain：`MAE_control − MAE_candidate = −1.84e-5`（candidate 略差）。实际总耗时 **26 分钟**
（`13:48:37Z` 起，`14:14:24Z` 交付；第 26 分钟前停止计算）。

## Gate（预注册，三项全满足才 PASS）

| 项 | 实测 | 判定 |
|---|---:|---|
| 总 calibrated gain ≥ 0.003 | **−1.84e-5** | ✗ |
| gain_without_id172 > 0 | +1.82e-5 | ✓（量级≈0） |
| G0 贡献恶化 ≤ 0.001 | **−6.63e-5**（改善） | ✓ |
| **PASS** | | **false** |

raw gain `+1.04e-5`；bias control `−0.000272` vs candidate `−0.001582`。标定未主导差异（两 arm bias 都近 0），
raw 与 cal 方向一致且都 ≈0。

## 分组（calibrated valid，`signed=pred−y`）

| 组 | n | control MAE / 贡献 | candidate MAE / 贡献 | 贡献 gain |
|---|---:|---:|---:|---:|
| G0 | 965 | 0.088838 / 0.085728 | 0.088769 / 0.085662 | +6.63e-5 |
| G1 | 34 | 0.295972 / 0.010063 | 0.297386 / 0.010111 | −4.81e-5 |
| G172 | 1 | 19.2314 / 0.019231 | 19.2680 / 0.019268 | −3.66e-5 |
| 全体 | 1000 | 0.115023 | 0.115041 | −1.84e-5 |

各组贡献之和 = 全体 MAE（已核对）。收益/损失都落在 1e-5 量级，无组级信号。

## 1. 干预是否真实执行、强度如何？

是。Adam（**coupled L2**，不是 decoupled AdamW）；reader = `reader.net.{0,2,4}.{weight,bias}` 共 6 个张量 /
33,385 参数，进入独立参数组，`wd = 1e-4`（control 组 `1e-5`）；其余 43 个参数组 `wd = 1e-5`。
每参数恰好一次，两 arm 用同一显式分组；`mult=1` 时与原单组 Adam 数学等价。optimizer 全新、step 数
`20×79 = 1580`/arm，两 arm 从同一父 soup 起跑、同 seed/同 shuffle。

**但强度很弱**：单个 batch 上 `|data_grad|` 比 `|wd·p|`(1×) 大 `4.2e3–9.7e5` 倍；10× 后仍大 `4e2–1e5` 倍。
reader soup 范数相对父仅移动 0.3–6%（如 `net.0.weight` 父 `11.773` → control `11.737` / candidate `11.383`）。
→ 该 warm horizon 内 reader decay 对优化轨迹近乎无约束力，这解释了为何两 arm 几乎相同。

## 2. 是否改善总体与 G0？收益落在哪里？

**没有。** 总体 −1.84e-5，G0 +6.6e-5（可忽略），G1/G172 略差。收益不由 bias 主导（bias 都近 0），
也不由 id172 主导（去掉后 +1.8e-5，仍≈0）。train 上 candidate 末轮 `train_mae 0.031006` vs control `0.031066`，
差 6e-5，同样无实质区别。

## 3. 四次前向：支持“其他块差异”还是“对共同拓扑输入响应不同”？

`T0 = parent_topology_encoder(mean_train(topo_model_input))`（不是 `mean(topology8)`），`‖T0‖=0.890`。

| | native | 共同 T0 |
|---|---:|---:|
| train:3776 | p_t = −22.5144 | p_t0 = −17.1775 |
| valid:0172 | p_v = −1.3886 | p_v0 = +0.7206 |

`D_native=21.1258`，`D_ref=17.8981`，`δ_v=−2.1091`，`δ_t=−5.3369`；
恒等式 `D_native = D_ref + (δ_v−δ_t)` 精确成立（残差 0.0）。

**修正上一份审计的强归因**：即使两行喂入**完全相同**的拓扑输入 T0，train/valid 预测仍差 `D_ref=17.90`。
所以预测差**主要来自其他 R 块**（unary/pair/global 不同）及其与 T 的条件交互，而**不是** T 本身。
上一轮“相同 T → 相同拓扑贡献”的表述过强；本轮只做相对 T0 的分解，**不证明因果贡献、可达收益或排除其他机制**。

## 4. 原假设得到多少支持？warm 失败关闭什么？

原假设（“reader 组合约束不足，仅加强 reader decay 可改善泛化”）在**这一配置下未获支持**：
干预执行了但过弱，收益 ≈0。**这不构成“正则无用”的证据**——只关闭「此父 checkpoint、10× reader decay、
H=20 warm」的购买。warm 失败只覆盖这个 warm 干预；**不能**推导从头训练或所有正则方向失败，
也**不能**宣布表示已证充分。父→control 的 `−4.3e-5` 属训练噪声，不计入 candidate gain。

## 5. 下一步（唯一动作）

**关闭本配置并停止**，不再扫 decay 倍数、不换 loss/optimizer/选模重试。若未来要检验正则假设，
前提是先让干预具备强度（更大倍数或更长 horizon）并作为**新的**独立预注册，而非本轮延伸。

## 边界与身份

- 对象 `SCALE-FULL-seed0_soup_state.pt` sha `17f5fcc3…574eb`；split fingerprint `58c69506…f28a`。
- loss = L1 + 33.95873017865987·reconstruction；batch 128；clip 5；lr 1e-4；seed 0；末 5 轮 soup（不是 best-5）。
- **未完成项**：无（两 arm 均完整 20 epoch，身份有效）。仅单配对 seed。
- **边界**：未访问 official test；未新增结构/特征；未拟合新 head；未改字典/输入；未 commit/push；
  official valid 已被反复探索，本轮仅探索性证据。`reader_norms_init` 字段实为末轮状态（已在
  `summary.json::reader_norm_diagnostic` 更正；父范数为真实 init）。

## 复现

```
uv run research run zinc_full_reader_wd_pilot_v1 --set model.stage=timing
uv run research run zinc_full_reader_wd_pilot_v1 --set model.stage=pair --set model.h=20 --set model.reader_wd_mult=10
# pair run: tracks/ksvd/runs/2026/10/02/20261002-215628-c0b671bb (854.5 s)
```
产物：`summary.json`、`valid_paired_predictions.csv`（1000 行）、`four_forwards.json`、
`pilot_{control,candidate}_soup_state.pt`；protocol/config/runner 已注册。