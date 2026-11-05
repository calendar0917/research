# REPORT — LCRC PCBA 冻结表征快速读出探针（`lcrc-pcba-frozen-readout-probe-v1`）

**标签：`WEAK_OR_MIXED`**（`NONLINEAR_EXTRA_SIGNAL = False`）
**消耗：0.087 / 0.4 A100 GPU-hour**（含失败重跑与诊断；正式三段 0.0562 h）
**Test 未读取、未评分**（`test_evaluated: false`，`test_never_finite_or_indexed: true`）
**正式运行 commit：`343bca732c6a`**（阶段 A 与阶段 B 同一 commit）

---

## 1. 结论摘要

在完全冻结的 LCRC-C1 编码器上（245,184 参数，`eval()` + `inference_mode()`，无梯度），
用**完全相同配方**短程训练两个零初始化残差头（8 epochs，AdamW lr=3e-4，batch 1024，seed 0）：

| 臂 | 结构 | 可训练参数 | epoch 8 官方 Valid AP | `D = AP − AP0` | 改善任务 |
|---|---|---:|---:|---:|---:|
| — | 冻结基线（重放） | 0 | 0.2374915929 | `+1.3e-06` | — |
| L | `Linear(171→128)`，全零初始化 | 22,016 | **0.2386741759** | **+0.001184** | 73/127 |
| N | `Linear(171→128)→SiLU→Linear(128→128)`，末层零初始化 | 38,528 | **0.2403392810** | **+0.002849** | 90/127 |

`D_NL = +0.001665`。**两个头都是小幅正向改善，都远低于预注册的 +0.005 门槛**；
改善的方向性在任务层面是广泛的（73/90 of 127），但幅度不足以支持
“冻结表征 + 快速读出即可拿到有意义的增益”这一资源结论。

---

## 2. 交付物

| 路径 | 内容 | SHA256（前 16 位） |
|---|---|---|
| `PROTOCOL.md` | 预注册协议 + 唯一的阈值修订记录（§1 末尾） | — |
| `code/fr_common.py` | 冻结常量、判定规则、I/O 工具 | — |
| `code/fr_metrics.py` | 标准化、masked BCE、逐任务 AP、行计划与隔离审计、头构造 | — |
| `code/fr_model.py` | `RhoInputCapture`（forward pre-hook）、冻结加载、`rho_in` 重建与重放 | — |
| `code/fr_extract.py` | 阶段 A（写缓存 + 两份审计） | — |
| `code/fr_heads.py` | 阶段 B（两头顺序训练、曲线、checkpoint） | — |
| `code/fr_compare.py` | 阶段 C（判定与逐任务表） | — |
| `code/fr_preflight.py` | GPU smoke（6 类门槛） | — |
| `code/fr_replay_diag.py` | 重放非确定性诊断（本次新增，见 §7） | — |
| `tests/test_fr_probe.py` | 12 项 CPU 测试，全部通过 | — |
| `analysis/feature_extraction_audit.json` | 阶段 A 全部门槛与哈希 | `c3ee7ea5b13663a1` |
| `analysis/frozen_baseline_replay.json` | 冻结基线 Train/Valid AP、BCE、逐任务 AP | `2da68feb787133dc` |
| `analysis/readout_comparison.json` | 判定、主比较、分层、gap、GPU 总账 | `0b9b3147de7d54a3` |
| `analysis/per_task_deltas.csv` | 128 行（127 可评分 + 任务 45） | `1cfbb025b9357cbb` |
| `analysis/readout_curve.csv` | 两臂 × 4 个评价点 | `2453e33387021bc0` |
| `runs/heads/{L,N}/` | `train_curve.csv`、`per_task_ap.csv`、`checkpoint_final.pt`、`checkpoint_best.pt`、`arm.json` | 见 `arm.json` |
| `runs/feature_cache/manifest.json` | 缓存清单（形状、dtype、file/content 双哈希） | `a3729d922d228c7f` |
| `runs/preflight/preflight.json`、`runs/replay_diag/replay_diag.json` | smoke 与诊断记录 | — |

大数组（`runs/feature_cache/*.npy`，约 1.1 GB）与 checkpoint 不入 Git，其内容哈希
写入 manifest 与 `analysis/*.json`；两个头的 checkpoint 已随本轮入库（每个 < 170 KB）。

---

## 3. 阶段 A：一次性冻结提取（事实）

运行 `fr-extract-20261105-231218-b0c121a3`，`all_passed=true`，21 个门槛全绿。

| 项 | 值 |
|---|---|
| 行数 | Train 350,343；Valid 43,793（官方 split；Test 行从未被索引） |
| `rho_in` 维度 | **171**，与 `model.rho.lins.0.weight.shape=(64,171)` 一致；分块 `g=42`、`mean_h=64`、`mean_h2=64`、`log1p_n=1` |
| 标准化 | 只用 Train（350,343 行），`ddof=0`，零方差列 `scale=1.0`；实际零方差列 2 个：`g[24]`、`g[41]` |
| Valid 官方 AP | **0.2374915929**，对 `AP0` 偏差 `1.30e-06`（门槛 1e-4） |
| Train macro AP | **0.3883038060**，对历史 0.3883355995 偏差 `3.18e-05`（门槛 1e-4） |
| Valid / Train masked BCE | 0.0460386417 / 0.0318040729 |
| `rho_in` 独立重建 | 最大差 `2.86e-06`（Train）/ `3.34e-06`（Valid），门槛 1e-5 |
| `rho(rho_in)` 重放 | 最大差 **0.0**（同一 batch 内，逐位一致） |
| `contrib + rho == logits` | 最大差 **0.0**（逐位） |
| 独立重放（`predict_logits`，batch 256） | logit 最大差 `1.07e-04`（门槛 1e-3）；AP 差 `2.18e-06`（门槛 1e-5） |
| 冻结参数 | 前后指纹一致 `bd969c6328749735…`，无梯度 |
| 吞吐 | 10,534（Valid）/ 12,138（Train）mol·s⁻¹；0.0177 GPU-hour；峰值显存 0.86 GB |

观测条目数 27,162,842（Train）+ 3,477,656（Valid）= 30,640,498，与历史 data audit 一致
（`observed_entries_match_data_audit` 通过；Train/Valid/Test 行 id 两两不交，Test 标签无有限值）。

---

## 4. 阶段 B：两个残差头（事实）

运行 `fr-heads-20261105-231936-9bd6053f`，`all_passed=true`。
共享行计划 `b9ebd85ccb340f93…`（每 epoch 一个 `randperm(350343)`，seed 0），
隔离审计 6/6 通过（每个 epoch 是 `range(350343)` 的置换；Train/Valid/Test 行 id 两两不交）。
两头 epoch 0 的残差**恰好为 0**，输出 logits 与冻结 logits **逐位相同**（`max|Δ| = 0.0`）。

| epoch | L Valid AP | L Train AP | N Valid AP | N Train AP |
|---:|---:|---:|---:|---:|
| 0 | 0.2374916 | 0.3883038 | 0.2374916 | 0.3883038 |
| 2 | 0.2374217 (−6.9e-05) | 0.3953844 | 0.2373768 (−1.1e-04) | 0.3920719 |
| 4 | 0.2381579 (+6.7e-04) | 0.3988993 | 0.2385275 (+1.0e-03) | 0.3949961 |
| **8** | **0.2386742 (+1.18e-03)** | 0.4038603 | **0.2403393 (+2.85e-03)** | 0.4025703 |

补充事实：

- 两臂 best-Validation epoch 都是 **8**（辅助口径与主口径一致，没有“靠挑 checkpoint 取巧”）。
- 两臂在 epoch 2 都**先低于** `AP0`，之后才转为正向；曲线不是单调的。
- epoch 8 的 Valid masked BCE 基本不动：L `0.046039 → 0.046009`（−3.0e-05）、
  N `0.046039 → 0.046085`（**+4.6e-05**）。即 AP 的改善来自排序，不来自似然拟合
  （N 的 Valid BCE 甚至轻微变差）。
- 步数 8×343 = 2,744（两臂一致）；训练 6.4 s（L）/ 6.9 s（N）每 epoch；
  梯度范数最大 0.0138（L）/ 0.0084（N），无裁剪、无 scheduler、无 early stopping。
- 编码器：训练前后指纹相同、`base_model_gradients_present=false`。

---

## 5. 逐条回答 10 个问题

**1. 冻结基线是否复现？**
是。Valid 官方 AP `0.2374915929`，对 `AP0=0.2374902968831333` 偏差 `1.3e-06`（容差 1e-4）；
Train macro AP `0.3883038`，对历史 `0.3883356` 偏差 `3.2e-05`（容差 1e-4）。
127 个可评分任务（任务 45 在 Valid 只有 41 个分子、0 个正例，不可评分，与历史一致）。

**2. 两头 epoch 0 是否与冻结 logits 逐位相同？**
是，且是严格逐位：`residual_is_exactly_zero=true`，`max|logits_head0 − logits_frozen| = 0.0`，
epoch 0 的 Valid AP 与从缓存重放的冻结 AP 完全相同（`0.2374915929`）。
线性头全零初始化、非线性头末层零初始化都按协议执行，初始指纹记录在 `arm.json`。

**3. 两头 epoch 8 的 Train macro AP？**
L `0.4038603`，N `0.4025703`（冻结基线 `0.3883038`）。两头都把 Train AP 抬了约 +0.015，
线性头略高（+0.0013 于非线性头）。

**4. 两头 epoch 8 的官方 Validation macro AP？**
L `0.2386741759`，N `0.2403392810`（冻结基线 `0.2374915929`）。

**5. 相对 `AP0 = 0.2374902968831333` 的增减？**
`D_L = +0.0011839`，`D_N = +0.0028490`。两者都落在预注册 `+0.005` 门槛的 **24% 和 57%**。

**6. 非线性头是否优于线性头？**
方向上是：`D_NL = +0.0016651`，逐任务 84/127 优于线性头，中位数 `+0.00128`。
但 `D_NL` 只有 `+0.003` 门槛的 56%，且 `NONLINEAR_EXTRA_SIGNAL` 按协议必须以
`READOUT_SIGNAL` 成立为前提——该前提不成立，因此附加标志为 **False**。
换言之：**“非线性多一点”只是一个方向性观察，不构成机制结论**。

**7. 改善任务的覆盖与集中度？**
覆盖：L 73/127、N 90/127 严格改善；恶化（Δ<0）L 54、N 37；无并列。
N 与 L 的符号一致 94/127（N 独有改善 25，L 独有 8）。
Δ ≥ +0.005 的任务：L 20 个、N 31 个；Δ ≤ −0.005 的任务：L 12 个、N 11 个——
增益面比损失面宽，但两侧都不小。
集中度：N 的 top-1 任务占全部正增量之和的 15.3%，top-5 占 45.8%（L 为 20.7% / 77.0%）；
按 Valid 正例数分层，N 的均值增益为 sparse（≤100 正例，54 任务）`+0.00356`、
medium（60 任务）`+0.00236`、ample（>1000，13 任务）`+0.00216`。
**最大的几个单任务增益恰好落在支持度极小的任务上**（任务 120：13 正例 +0.055；
任务 117：9 正例 +0.040；任务 108：**1 个正例**，AP 只有 0.05/−0.017），
单任务的 AP 在这些任务上以离散步长跳变，不适合当作独立证据。
逐任务最大负值：L 任务 108 `−0.0177`；N 任务 105 `−0.0261`。
说明：宏观 AP 是 127 个任务等权的平均，因此若干低支持度任务的大幅波动会直接进入主指标；
这也是主指标只动了 +0.0028 而个别任务动了 +0.05 的原因。

**8. Train–Valid gap 是否扩大？谁更大？**
两臂都扩大，线性头更大：
冻结 `0.15081` → L `0.16519`（+0.01437）→ N `0.16223`（+0.01142）。
机制上不奇怪：编码器冻结、特征固定，头只能拟合训练残差；Train BCE 从 0.03180 降到
0.03142（L）/ 0.03123（N），而 Valid BCE 几乎不动。**这是“头在拟合训练噪声”的证据，
不是关于编码器表征质量或某个具体机制缺陷的证据。**

**9. 是否值得继续投入？**
按**预先注册的判定规则**：`WEAK_OR_MIXED`。
两头都完整跑完、方向一致为正、任务覆盖面过半，但没有任何一头达到 `+0.005`，
因此不能写 `READOUT_SIGNAL`，更不能写 `NONLINEAR_EXTRA_SIGNAL`。
若只看趋势：两臂在 epoch 8 仍在上升（L 最后 4 个 epoch `+1.3e-04/epoch`，
N `+4.5e-04/epoch`）。按该斜率外推，N 大约还需 5 个 epoch 才能碰到 `+0.005`，
L 大约还需 30 个（且斜率在衰减）。**这是单 seed 外推，未运行、也不算结果**；
把它当作“值得再投一轮”的理由是不可靠的——真正的下一步需要新的、独立预注册的轮次
（更长训练或显式 schedule + 多 seed 重复），本轮不启动。

**10. GPU 消耗与可复现性？**
正式三段：preflight `0.0050` + 提取 `0.0177` + 两头 `0.0335` = **0.0562 h**；
加上失败的 smoke（`0.00505`）、失败的提取（`0.01819`，哈希口径 bug）、
重放诊断（`0.00323`）与失败的 heads 尝试（约 16 s ≈ `0.0044`），
**合计约 0.087 h，占 0.4 h 预算的 22%**。
峰值显存：提取 0.86 GB，两头 1.26 GB（预留 1.30 GB）。
可复现性：两臂同一 commit `343bca732c6a`、同一行计划、同一缓存内容哈希
（`train_rho_in`/`train_logits` 的 `content_sha256` 两臂相同）、同一配方；
preflight 里 20 步双跑得到**完全相同的 loss 序列与参数指纹**；
checkpoint / 曲线 / 逐任务 CSV 的 SHA256 均记录并在拉回后本地复核一致。

---

## 6. 判定（规则逐条落地）

| 条件 | 结果 |
|---|---|
| 两臂完整跑完 8 epochs + 4 个评价点 | ✅（`2,744` 步，`[0,2,4,8]`） |
| 所有正确性/数据隔离门槛 | ✅（阶段 A 21/21、阶段 B 全绿） |
| `max(D_L, D_N) = +0.002849 > 0` | ✅ |
| `∃ head: D ≥ +0.005` | ❌（`0.001184` / `0.002849`） |
| `∃ head: ≥ 64/127` 改善 | ✅（73 / 90） |
| `D_NL ≥ +0.003`（前置 `READOUT_SIGNAL`） | ❌（`+0.001665`，且前置不成立） |
| **标签** | **`WEAK_OR_MIXED`** |

---

## 7. 本轮暴露的三处实现问题（全部在正式数字之前修复，配方与阈值未动）

1. **独立重放门槛不可达。** 冻结前向的 `mean_h` / `mean_h2` 用 CUDA `index_add_`
   （atomic）归约，同一输入两次前向的 logits **不可能**逐位相等。专门诊断
   （`runs/replay_diag/replay_diag.json`，8,192 行）测得：同代码同 batch 两次提取
   logits 差 `6.10e-05`、AP 差 `6.40e-06`；`g` 与 `log1p_n` 分块**逐位相同**，
   `mean_h` `8.9e-07`、`mean_h2` `2.4e-06`。因此把“独立重放”改成双条件
   （**AP ≤ 1e-5** 且 logit ≤ 1e-3），未放宽 `rho_in` 重建/重放与
   `contrib+rho==logits` 的门槛。修订在 §1 有完整记录与时间戳。
   **影响解读**：缓存中的 `(rho_in, logits)` 来自**同一次前向**，内部自洽；
   配对比较不受影响，受影响的是“跨进程逐位复现”这一期望本身。
   全量 Train 上的同一效应也可见：两次提取的 Train AP 为 `0.3883038` 与 `0.3883922`
   （差 `8.8e-05`），都仍在 1e-4 容差内。
2. **哈希口径不一致**：阶段 A 把 `.npy` **文件哈希**与 dtype+shape+payload 的
   **内容哈希**对比，导致产物一致性门假失败。已分开记录 `file_sha256` 与
   `content_sha256`，阶段 B 用 `np.load` 后的内容重算校验。
3. **行计划隔离审计的坐标系错误**：计划置换的是 Train 数组内的**位置**
   （`0..350342`），审计却拿全局分子 id 去屏蔽，误报 71,966 个“违规”。
   现改为两条独立检查（每个 epoch 是 `range(n_train)` 的置换；Train/Valid/Test
   行 id 两两不交），并补了反例测试（漏 split、越界置换都必须失败）。

三处修复都不改学习率、宽度、训练长度、损失、评价点、判定阈值或数据划分；
每次修复后重跑同一阶段，最终交付的三个产物都来自修复后的代码。

---

## 8. 这些事实**不能**支持什么

- 不能说明 S1/S2 表征无效：本轮只测了“冻结特征 + 线性/浅非线性读出”这一条通路，
  且 8 epochs 时两臂仍在上升。
- 不能说明原模型存在某个特定机制缺陷：残差头只是在既有 logits 上加了一个小修正。
- 不能声称统计显著：单 seed、单个 Validation split，无重复、无置信区间。
- 不能把本次 Validation 当作独立最终确认：`AP0` 与所有决策都在同一份 Validation 上取得。
- 不能外推到 Test：Test 从未被读取或评分。
- 不能把 `+0.0028` 读成“接近 0.005 门槛”就等同于“接近成功”：`D_N` 只有门槛的 57%，
  且 Valid BCE 未改善、gap 扩大。

---

## 9. 复现

```bash
# 0) 环境与主机
rr doctor res
rr deploy res                      # commit 343bca732c6a

# 1) 阶段 A（冻结核验 + 一次性提取）
rr run res fr-extract --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/analysis \
  --result tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/runs/feature_cache \
  -- timeout 900s uv run python -u tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/code/fr_extract.py \
     --device cuda --out-dir .../runs/feature_cache --analysis-dir .../analysis

# 2) 阶段 B（两头顺序训练）
rr run res fr-heads --pool res-gpu1 --result .../analysis --result .../runs/heads \
  -- timeout 1200s uv run python -u tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/code/fr_heads.py \
     --device cuda --out-root .../runs/heads --assigned-pool res-gpu1

# 3) 阶段 C（判定，CPU）
uv run python tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/code/fr_compare.py \
  --analysis-dir .../analysis
```

单测：`uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_frozen_readout_probe_v1/tests/`（12 passed）。

---

## 10. 停止声明

本轮按协议在第 8 个 epoch、四个评价点、127 个可评分任务上停止；未做完整 LCRC 训练、
未增加 seed 或架构、未读 Test、未自动开启下一轮。上述标签与数字即为本轮的最终产出。
