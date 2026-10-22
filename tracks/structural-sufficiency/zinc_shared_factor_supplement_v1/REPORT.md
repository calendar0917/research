# zinc_shared_factor_supplement_v1 — T vs N（seed 1）补充机制实验报告

预注册：`PROTOCOL.md`（训练前提交 `cf116a0`，per-seed carrier 声明 `6b94bd5`）。
性质：**补充 study**。原 `zinc_shared_factor_v1` 的 `A_FAIL` 判定、门槛、数值与记录均未改动；
本报告不是原协议的阶段 B/C，也不据此自动启动下一步。

## 0. 四个问题的直接回答

1. **本次 T/N 对照是否支持共享机制？——不支持。**
   预注册的三条判定只过两条：`best_valid` T 0.079088 < N 0.079887（过），
   但 20-epoch 滑动中位数 **T 0.082070 > N 0.081238（不过）**。而且
   逐 epoch 上 T 只在 210/600（ep100 之后 33%）的 epoch 比 N 好；在 935 个普通分子
   （y > −3）上 **N 0.067666 优于 T 0.070187**；final-valid 也是 N 更好（0.081271 vs 0.082169）。
   T 唯一领先的 `best_valid` 完全来自 39 个 `seen+tail` 分子（T 0.1725 vs N 0.2570，
   对总体 MAE 贡献 −0.0033），同时 T 在 710 个 `seen+ordinary` 上更差（+0.0033）。
   即：**一次取最小的选择噪声，不是机制收益。**

2. **T 的绝对水平是否在另一个 seed 保持？——不保持。**
   T seed0 best 0.077045（ep384）→ seed1 best 0.079088（ep363），差 +0.00204；
   T1 在 ep100 之后只有 20.6% 的 epoch 比 T0 好。两 seed 都在 0.075 参考线之上，
   因此「T 稳定在 0.077 左右」不成立，更接近「停在 0.077–0.079 这一段」。

3. **未见普通分子的主要误差是否仍然不动？——仍然不动。**
   225 个 `unseen+ordinary` 分子：D 0.135892 / T0 0.133880 / T1 0.132872 / N1 0.132715，
   历史 P3 0.133939597（描述性）。四个 arm 的极差只有 0.0032，T1 与 N1 相差 +0.00016
   （对总体 MAE 贡献 +0.000035）。共享与否、因子化与否，都没有把这一格推离 ~0.133。

4. **据此应停止该具体方案，还是已有足够理由提出下一次有限确认？——停止该具体方案。**
   `shared-atomic-state` 作为 P3/LF 线的改进，本轮给出的机制增量是 0.0008 且在平滑指标上反向；
   再加上第 5 节的读出诊断（整个结构块在冻结 checkpoint 上被置零只改 7e−6），
   继续加 seed、加通道、调温度都只是把同一件事再测一遍更大的噪声。
   **不建议**用「再调几个超参数」作为下一步。若之后还要动这条线，必须是一个**不同的问题**
   （例如「如何让结构块真的被读出使用」），且需要新的预注册与新预算；本报告不启动它。

## 1. 设置（与原协议同条件，只换监督 seed）

| 项 | 值 |
| --- | --- |
| 代码 | 训练路径与 T0 逐字节相同：`sf_models.py` / `sf_core.py` 未改；`sf_runner.py` 唯一改动是 gate 的 per-seed carrier 校验（见 §2），不触碰模型/优化器/顺序/确定性 |
| 配置 | 冻结的 `../zinc_shared_factor_v1/configs/sf_atomic_v1.json`（sha `d1f6b67f…`），含 seed1 的 init/hash |
| 臂 | T seed1（共享非根原子状态）与 N seed1（非根边独立状态副本），各 451,777 参数 |
| 初始化 | 只加载同一个 train-only SSL seed0 checkpoint（sha `006fabc1…`）；T/N seed1 init `db2792ca…`、content `7b023184…`、carrier `750be561…` 三项**逐项相同**（gate 内 `paired_T_init.identical=true`） |
| 监督随机性 | seed 1：不同 init 与不同 shuffle 流（generator seed 102，seed0 为 101）。**只重复监督训练随机性**，不重训 SSL |
| recipe | 600 epochs × 79 steps = 47,400 updates，AdamW 1e-3→1e-5、warmup 500、clip 5、raw-y L1、batch 128、全参数、无 EMA/soup/早停、严格 `<` 选单一 best |
| 确定性 | 未改动（未开 `torch.use_deterministic_algorithms`，未改内核/求和顺序），与旧结果同条件但非逐位可复现 |
| 数据 | ZINC subset train 10,000 / valid 1,000；**official test 从未读取** |
| N 的独立性 | N 从自己的初值独立训练 47,400 updates，**没有**用 T 的权重在推理时切换 |

## 2. 必要检查（`analysis/supplement_checks.json`，6/6 PASS，CPU-only）

| 检查 | 结果 |
| --- | --- |
| `init` | seed1 的 D/T/N 三项 hash 全部等于冻结配置；carrier `750be561…` 在 seed1 内三臂**相同**，且与 seed0 `21963b43…` 不同 |
| `order` | T1 与 N1 的 **600/600 epoch batch order hash 完全一致**（generator seed 102）；seed1 流与 seed0（101）不同 |
| `branch` | N 的 best.pt 重载后 valid 0.079886718（冻结值 0.079886710，差 1e−8）→ 训练/打分走的是 N 分支；同一权重用 T 分支前向只改 7e−8（见 §5） |
| `budget` | 两臂 451,777 参数、79 steps/epoch、47,400 updates、`W_s` 初始化严格为 0；wall 1945.5 s / 2913.8 s，均 < 3600 s |
| `collapse` | 结构块 T−N 平均差 0.0030（= 该块跨分子 std 0.026 的 18%）；`q` 熵 1.3039/1.386；`K` 通道内 std 0.345（范围 0.415–2.426）→ 参数层面**没有**退化 |
| `ablation` | 冻结 checkpoint 上把结构块置零 / 跨分子打乱：valid MAE 变化 ≤ 7e−6（见 §5） |

只做了与本任务直接相关的检查，没有重跑全仓库测试。

## 3. 结果

| arm | seed | best valid | best ep | train@best | final valid | best20 中位数（窗口终点） | tail-100 中位数 | s/epoch | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DensePhi (D) | 0 | 0.081252 | 339 | 0.019250 | 0.082611 | 0.082610 (587) | 0.082645 | 2.37 | 0.411 |
| SharedState (T) | 0 | **0.077045** | 384 | 0.016686 | 0.078846 | 0.078646 (528) | 0.078800 | 3.58 | 0.620 |
| SharedState (T) | 1 | 0.079088 | 363 | 0.019183 | 0.082169 | 0.082070 (477) | 0.082243 | 3.13 | 0.540 |
| IndependentCopies (N) | 1 | 0.079887 | 415 | 0.012027 | **0.081271** | **0.081238 (565)** | **0.081305** | 4.69 | 0.809 |

D seed0 / T seed0 为上一轮的冻结结果，此处仅作上下文；本轮新增只有 T seed1 与 N seed1。

**配对逐分子差异（best checkpoint，1,000 个分子）**

| 对比 | ΔMAE | 95% CI（仅分子抽样） | 改善分子占比 | median ΔAE |
| --- | --- | --- | --- | --- |
| **T1 − N1** | **−0.000799** | [−0.009152, +0.004835] | 0.459 | +0.002766 |
| T1 − T0 | +0.002042 | [−0.003078, +0.007660] | 0.486 | +0.000929 |
| N1 − T0 | +0.002841 | [−0.004854, +0.015050] | 0.533 | −0.001775 |
| T0 − D0 | −0.004207 | [−0.008309, −0.000318] | 0.526 | −0.002131 |
| N1 − D0 | −0.001365 | [−0.008478, +0.008947] | 0.556 | −0.004351 |

T1−N1 的区间包含 0，且**中位分子在 T 下更差**（median ΔAE = +0.0028，改善占比 45.9%）。
区间只表示「固定这两个模型时的分子抽样不确定性」，**不是 seed 方差**。

**普通分子（y > −3，n = 935）**

| arm | MAE |
| --- | --- |
| D seed0 | 0.071447 |
| T seed0 | 0.069207 |
| T seed1 | 0.070187 |
| **N seed1** | **0.067666** |

ΔMAE（配对，分子级 CI）：T1−N1 **+0.002521** [−0.000415, +0.005401]；
N1−T0 −0.001541 [−0.004961, +0.001861]；T1−T0 +0.000980；T0−D0 −0.002240。
**在占 93.5% 的普通分子上，独立副本 N 是四个 arm 里最好的**——与「共享更好」相反。

**epoch 级方向**：T1 < N1 仅 210/600（ep100 后 33.2%）；T0 < D0 在 ep100 后 85.2%；
T1 < T0 在 ep100 后 20.6%。

**四个 radius-2 分组（全部 1,000 个分子，分组来自冻结参考）**

| cell | n | D0 | T0 | T1 | N1 | Δ(T1−N1) | Δ(T1−T0) | P3（描述性） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seen, ordinary | 710 | 0.0510 | 0.0487 | 0.0503 | 0.0471 | +0.0033 | +0.0016 | 0.0475 |
| seen, tail | 39 | 0.1605 | 0.1472 | 0.1725 | 0.2570 | −0.0845 | +0.0253 | 0.1163 |
| unseen, ordinary | 225 | 0.1359 | 0.1339 | 0.1329 | 0.1327 | +0.0002 | −0.0010 | 0.1339 |
| unseen, tail | 26 | 0.3149 | 0.2537 | 0.2591 | 0.2538 | +0.0053 | +0.0054 | 0.3019 |

**净贡献分解**（`n/1000 × cell ΔMAE`，四格之和严格等于总体 ΔMAE，|diff| ≤ 1.4e−17）

| cell | n/1000 | Δ(T1−N1) | 贡献 | Δ(T1−T0) | 贡献 |
| --- | --- | --- | --- | --- | --- |
| seen, ordinary | 0.710 | +0.003270 | **+0.002322** | +0.001609 | +0.001143 |
| seen, tail | 0.039 | −0.084466 | **−0.003294** | +0.025297 | +0.000987 |
| unseen, ordinary | 0.225 | +0.000157 | +0.000035 | −0.001008 | −0.000227 |
| unseen, tail | 0.026 | +0.005306 | +0.000138 | +0.005366 | +0.000140 |
| **合计** | 1.000 | | **−0.000799** | | **+0.002042** |

读法：T 相对 N 的唯一「收益」是 39 个 `seen+tail` 分子贡献的 −0.0033，被 710 个普通分子的
+0.0023 抵消；剩下两格几乎为零。`seen+tail` 这一格在五个 arm 上跨度为 0.116–0.257，
是典型的少数极端分子格（n=39，y ≤ −3），不适合承担一次架构判断。

## 4. 完整学习曲线

`analysis/valid_curves.csv`（600 行 × 4 arm，逐 epoch 全量 valid MAE）与 `analysis/curves.png`。
形状（逐 epoch 直接对读）：两臂从**同一初值**出发但很快分开——前 100 个 epoch 的
|valid(T1) − valid(N1)| 平均 0.028、最大 0.196；ep101–600 收窄到最大 0.056。
低 lr 段两臂都进入平台（最后 100 epoch 的 std 分别为 0.000166 / 0.000156，几乎相同），
平台水平 **T1 0.082257 vs N1 0.081319（N 更低 0.00094）**；N 的 best 出在 ep415
（晚于 T 的 363），20-epoch 中位数窗口终点也更靠后（565 vs 477）。
也就是说：T 在 600 个 checkpoint 里恰好点到一个更低的点，但整段平台是 N 更低——
这正是 best 与 median 反向、且方向不稳的来源。

## 5. 机制诊断：结构块在读出端几乎没有被使用（辅助证据）

同一个冻结 checkpoint 换一种前向就能回答「T 与 N 到底差多少、这个差有没有被读到」：

1. **算子本身没有退化**：`q` 熵 1.3039/1.386，`K` 通道内 std 0.345（0.415–2.426），
   T 与 N 的结构块在同一权重下平均相差 0.0030，且其中 95% 是逐分子变化的
   （不是能被 bias 吸收的常数）；该差值相当于结构块自身跨分子 std（0.026）的 18%。
2. **但这个差对预测没有影响**：把同一份 N 权重改用 T 前向，valid MAE 只变 7e−8
   （逐分子最大 |Δpred| = 3e−5）。
3. **直接消融（冻结 checkpoint，未重训、未重选）**：valid MAE 变化
   ——结构块置零：T0 +7.1e−6、T1 −2.0e−7、N1 +2.3e−6；结构块跨分子打乱：
   T0 +5.7e−6、T1 +6.2e−7、N1 +2.6e−6；作为尺度参照，`q_root` 块置零：
   T0 +4.5e−4、T1 +5.2e−6、N1 −2.0e−5。
4. **读出权重**（`‖W1‖` × 该块跨分子 std 的典型前激活量）：`q_root` 0.21、
   结构块 0.09–0.10、family+计数块（只有 5 维）0.26。

结论：在这条 recipe 与这个预算下，**64 维结构对数配分块基本是「死」特征**；
T 与 N 的区别落在一个模型不读的通道上，所以 T vs N 只能由训练噪声决定，
这与 §3 的观测（0.0008、方向不稳、普通分子上 N 更好）完全一致。
注意这是**读出端**证据：它说明「共享与否在这套读出下没有可用的差别」，
不等于「共享状态这个参数化本身在别处也没用」。

## 6. 预算账目（本补充：每臂 ≤1 GPU-h，合计 ≤2 GPU-h）

| 项 | 秒 | GPU-h |
| --- | --- | --- |
| T seed1（含 4-epoch timing、逐 epoch 全量 valid、best/final 打分） | 1945.5 | 0.540 |
| N seed1（同上） | 2913.8 | 0.809 |
| 检查与重试（CPU gate + 6 项 supplement checks：重算 init hash、14 次 valid 评估） | ≈4 s wall | 0（CPU） |
| **合计** | | **1.350 / 2.0** |

两臂均未触发 3600 s 硬停；600 epochs 未缩短；未访问 official test；未新增 seed/臂/超参。

## 7. 限制（预先声明，逐条仍成立）

1. **单 seed 的 T/N 对照**只说明「在 seed 1 这一次训练里共享没有更好」，不是 seed 级推断；
   两个 seed 不构成 seed 方差估计。
2. 没有 D seed1，因此 T1 的绝对水平无法与「seed 1 更难」分离；T1 与 N1 因为同 seed、
   同 init、同 batch order，才隔离出机制差异。
3. 消融是后验的（在已训练好的读出上置零/打乱），不能回答「如果结构块一开始就被强制使用会怎样」。
4. 上一轮 D 与 LF-D 的 0.0026 差值是一次观测，不是噪声下界（见原报告的勘误）。
5. 本补充不改确定性设置，因此与旧结果同条件，但不是逐位可复现。

## 8. 复现

```bash
# 前置：cpu_checks.json 仍在（对应冻结 config sha），无需重跑
rr deploy res
rr run res sf-supp-T-s1 --pool res-gpu0 -- uv run python \
  tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_runner.py run --arm T --seed 1 --device cuda:0
# N 的 gate 需要配对 T 的 init.json，T 通过 gate 后再启动 N
rr run res sf-supp-N-s1 --pool res-gpu1 -- uv run python \
  tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_runner.py run --arm N --seed 1 --device cuda:0

rr pull sf-supp-T-s1 --path tracks/structural-sufficiency/zinc_shared_factor_v1/runs/SF_T_s1
rr pull sf-supp-N-s1 --path tracks/structural-sufficiency/zinc_shared_factor_v1/runs/SF_N_s1
uv run python tracks/structural-sufficiency/zinc_shared_factor_supplement_v1/code/supplement_analysis.py
uv run python tracks/structural-sufficiency/zinc_shared_factor_supplement_v1/code/supplement_checks.py
```

产物：`analysis/{summary.json, main_table.md, supplement_checks.json, preds_valid_TN_s1.csv,
valid_curves.csv, curves.png}`；逐臂冻结记录在
`../zinc_shared_factor_v1/analysis/SF_{T,N}_s1/{freeze.json,init.json,plan.json,timing.json}`，
checkpoint 与逐 epoch 曲线在远端 `runs/SF_{T,N}_s1/`（`runs/` 被 `.gitignore` 排除，
`freeze.json` 记有 sha256；本地重载 N 的 best.pt 得 0.079886718，与冻结值一致）。
