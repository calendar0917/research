# `zinc_relation_composition_probe_v1` — 片段状态条件化关系组合（D/S 探针）

* 协议：`PLAN.md`、`configs/probe_v1.json`（分支 `probe/relation-composition-v1`）
* 轨道：`structural-sufficiency`；本轮只回答一个问题，不做配方/深度/宽度扫描
* 官方 test：**未加载、未评分**（本轮不读 test，不主张 0.05× 目标）

## 1. 这一轮问什么

在**完全相同的显式结构输入、局部编码、参数量级与训练配方**下，让关系状态 $e$ 的写入
依赖于**当前键 token 的状态**（D 臂），是否比只依赖**固定结构**的关系写入（S 臂）买到性能？
以及这个新主体是否值得相对已确认基线 `B-R0`（DCC-L，494,193 参数，4-seed test 0.067319）继续。

与 `zinc_native_relation_v1` / `zinc_rooted_moment_v1` 的区别：那两轮比较的是**读出端**（多一个
关系/moment 读出通路，本身带来固定代价）；这一轮把差异**只放在关系状态的写入规则**上，
两支的参数量、张量形状、初始化与配方逐位相同，且都使用同一个真实 `xi` 做 token 更新。

## 2. 两支的唯一差异

核心（4 层，每层注意力只在**键 token** 上，键 token 数 = 物理键数，单 token 不复制）：

* 关系状态：$e^{l+1}_{bc} = \mathrm{BN}_e\!\left(e^l_{bc} + O_e \xi_{bc}\right)$
* D 臂：$\xi_{bc} = \mathrm{ReLU}(\mathrm{signed\_sqrt}(E_r x_{bc}))$，其中 $x_{bc}$ 由**当前** h 的
  两端与关系残差拼接（**依赖当前键 token 状态**）
* S 臂：$\xi^{\text{static}}_{bc} = \mathrm{ReLU}(\mathrm{signed\_sqrt}(c_l \odot E_w + E_b))$，
  只由固定结构（局部 composer 输出 $c_l$ 与关系类型偏置）决定，**与 h 无关**
* 两支都用**真实的 $\xi$** 做 token 更新（token 通路完全相同），**没有 `h.detach()`**（那会变成另一个问题）
* 前 3 层写 `e`，第 4 层只更新 h（不读写 $O_e/\mathrm{BN}_e$）
* `signed_sqrt(x)=sign(x)·sqrt(|x|+1e-6)`，两支相同

关系定义（无标签、离线预算、只覆盖 train/valid）：$R_{bc}[k] = (M P^k M^\top)_{bc}$，
$M$ 端点权重 $1/\sqrt2$，$P=\mathrm{diag}(deg)^{-1}A$，$k=0..20$；键上的 lifted RRWP，
显式**不是**线图游走。缓存可重建（`cache/` 已 git-ignore）。

复用的既有件：`ad_train.train` / `ad_data` / `ad_models`（P3 encoders）/ `dcc_models` 的
composer 分组与 B 初始状态；SSL 初始化沿用 seed0→s0、seed1→s1 的冻结 checkpoint
（sha256 `006fabc1…bfa9-` 系列，`analysis/checks.json` 中记录）。第二路 RNG（偏移 90731）只用于
新模块初始化，保证 D/S 初始状态**逐张量相同**。

## 3. 合规检查（`analysis/checks.json`，7 个套件全 PASS）

| 套件 | 关键证据 |
|---|---|
| relation | 与四端点直接枚举的最大差 2.94e-08；缓存 vs 在线 2.94e-08；翻转/重标号不变性 1e-16；$R[0]$ 对角 1、非对角 ∈{0,0.5} |
| identity | 470,321 参数 ≤500k；D/S 逐张量差 0.0；旧 B 读出层（513 维）在模型里不存在；优化器覆盖全部参数；encoder 与 SSL sha 一致 |
| replay | 同 seed 重放逐张量差 0.0 |
| mechanism | 扰动 h 后 D 的 layer0 写入变化 0.1240，S 为 0.0；layer0 的 h 相同（0.0），layer1 的 h 开始分歧（0.1490） |
| masking | padding / 零键 / 块对角（每分子独立）全部通过，零键 token 输出 0 |
| numerical | `signed_sqrt` 在 0 处梯度 0.0 且有限；118 个梯度张量全部非零有限；小批 loss 0.978→0.093 |
| checkpoint | 存取往返差 0.0；D 权重换 S 规则预测变化 5.16e-4 |

参数审计（`analysis/param_audit.json`）：encoders 202,048 / group_composer 62,976 /
local_composer 31,536 / core 115,776 / reader 57,985 = **470,321**；写关系标志位
`[True, True, True, False]`。与任务书 ≈470,513 的差 −192 = 4×d 已解释。

## 4. 代价（`analysis/timing_D_s0.json` + `analysis/gpu_ledger.json`）

`rc_runner.py measure` 在完整 10k/1k 上逐 epoch 计时（A100-40GB，d=48，batch 128，79 步/epoch）：
**6.68 s/epoch** → **1.113 GPU·h / 600-epoch run**。按预先写定的代价规则，
4 个 run（2 seed × D/S）= 2 个并行对 ≈ 2.23 h 训练墙钟，**在剩余预算内放不下**，
而 d=40 也放不下（瓶颈是已消耗的墙钟而不是单 run 成本，d=40 每对只省 ≈9 min）。
故执行规则第 3 级：**只跑 seed 0 的 D/S 对**（`rc-D-s0` / `rc-S-s0`，两张卡并行、同一提交）。
同一决定也已及时写回 `PLAN.md` §5b（在 seed-0 对结束**之前**记录，不是事后合理化）。

## 5. 结果（valid only；官方 test 未加载；单 seed 0）

正式两个 run 均 exit 0，同一提交 `7464c0d30441`，两张卡并行：
`rc-D-s0` 07:57:32→09:04:13、`rc-S-s0` 07:58:07→09:04:51（各 1.111 GPU·h，共 **2.223 GPU·h**）。

**预算诚实记录**：本轮 GPU 侧只用 2.223 GPU·h（含计时/诊断 2.240），远低于 6 GPU·h；
但**墙面超出 3 h 硬上限约 30–40 min**（会话约 05:47–06:00 CST 开始，09:30 CST 收尾，≈3.5–3.7 h）。
多出来的墙面**不是**额外 GPU 工作，而是：(a) 实现+读上下文的开发阶段本身较长；
(b) 远端部署被"远端 checkout 里未跟踪文件与本次提交新增文件重名"阻塞，需要三轮诊断/清理（`rc-hygiene*`，CPU、<2 min）才能部署；
(c) 计时任务暴露两个 runner 缺陷（`_measure_*` 目录未创建、fresh checkout 缺 R 缓存）各修一次。
按事前规则这属于"到点收尾"而不是"加跑更多实验"：**没有**因为超时而补第三、第四个 run，也没有缩短日程。

| arm | params | best-valid | ep | W20 | T100 | final | train@best(eval) | train@final(eval) |
|---|---|---|---|---|---|---|---|---|
| D（h 条件化写入） | 470,321 | **0.084493** | 566 | 0.094786 | 0.100461 | 0.095439 | 0.023578 | 0.033238 |
| S（仅固定结构写入） | 470,321 | **0.081838** | 557 | 0.103041 | 0.110148 | 0.099211 | 0.053499 | 0.108899 |
| B-R0（历史参照，非本轮重训） | 494,193 | 0.078537 | 489 | 0.079082 | n/a | 0.079076 | 0.005510 | n/a |

**D − S**：best-valid **+0.002655**（D 差），W20 **−0.008255**、T100 **−0.009686**、final **−0.003772**（D 好）。
**符号在行间翻转**：best-valid 偏向 S，而 W20/T100/final 偏向 D。原因是 valid 曲线本身极不稳定
（无 per-graph LayerNorm + core attention dropout 0.2 的声明配置下，valid 在 600 epoch 内多次在
0.08–0.30 间跳动），best-valid 是单个"幸运 epoch"，与窗口统计量指向相反。

配对 bootstrap（valid 1000 行重采样、固定 checkpoint，描述性）：

| 对比 | ΔMAE | 95% CI | 前者更好的行占比 |
|---|---|---|---|
| D − S | +0.002655 | [−0.003758, +0.009659] | 0.492 |
| B-R0 − D | −0.005956 | [−0.014483, +0.004588] | 0.569 |
| B-R0 − S | −0.003301 | [−0.014960, +0.012454] | 0.598 |

三个对比的 95% CI **都跨 0**，且"更好的行占比"都接近 0.5。

稳健性：去 row172 后 D 0.084493→0.082009、S 0.081838→0.081548；去自身 top10 后
D 0.073438、S 0.072776。剔除极端分子不改变任何结论，也不制造分离。

冻结分组（n / MAE）：

| arm | all_1000 | seen_749 | unseen_251 | ordinary_935 | tail_65 | unseen_ordinary_225 | seen_ordinary_710 | unseen_tail_26 |
|---|---|---|---|---|---|---|---|---|
| D | 0.0845 | 0.0612 | 0.1540 | 0.0750 | 0.2206 | 0.1449 | 0.0529 | 0.2325 |
| S | 0.0818 | 0.0582 | 0.1523 | 0.0747 | 0.1844 | 0.1407 | 0.0538 | 0.2534 |
| B-R0 | 0.0785 | 0.0543 | 0.1510 | 0.0655 | 0.2655 | 0.1308 | 0.0449 | 0.3256 |

D/S 在组间的方向不一致（seen_ordinary 上 D 略好，tail/unseen 上 S 略好），没有一致结构。
与 B-R0 的分组对照里有一个**方向性**差异：新主体在尾部（tail_65 0.22/0.18 对 0.2655）更好，
但在普通已见分子上更差（seen_ordinary 0.053/0.054 对 0.0449），即两者误差分布的形状不同，
而整体更差。

## 6. 固定权重干预（训练后的 D，规则改成 S，完整重前向）

| 量 | 值 |
|---|---|
| 本地重前向复现报告的 best-valid | 0.084493，差 **4.06e-09**（checkpoint 与 CPU/GPU 一致） |
| 同权重、D 规则 | 0.084493 |
| 同权重、S 规则 | **0.122155** |
| 差值（S−D） | **+0.037661**（变差 45%） |
| 预测变化 | mean 0.084941 / p50 0.054170 / p90 0.216573 / max 1.77703 |
| layer0 关系状态变化 | max 9.1732 / mean 0.3997 |

对照：**未训练**模型上做同样的规则切换，预测变化只有 2.5e-4（见开发期自检）。
即训练后的 D 模型**确实依赖**由当前键 token 状态写入的关系状态：把写入换成纯结构形式后
性能崩塌 +0.0377。这条证据排除了"关系核心是死代码/没被读"的解释。

## 7. 判定

**`NO_STABLE_D_S_DIFFERENCE_AND_WORSE_THAN_B_AT_THIS_HORIZON`（建议停止这条线）**

1. **D 相对 S 没有稳定收益。** 单 seed 下 best-valid 支持 S（+0.00266），
   W20/T100/final 支持 D（−0.0083/−0.0097/−0.0038），符号翻转；固定 checkpoint 的配对
   bootstrap CI 跨 0（[−0.0038, +0.0097]），行级胜负 0.492/0.508。无论怎么选指标，都**不能**
   说"D 买到性能"。工作假设（关系状态条件化于当前 token 状态能帮助组合）在本日程下**未被支持**。
2. **新主体整体不如已确认的 B-R0。** 在稳定量（W20/final）上差 **+0.0157/+0.0164（D）** 与
   **+0.0240/+0.0201（S）**；在 best-valid 上差 +0.0060/+0.0033。B-R0 的 W20≈best（曲线稳定），
   本轮两条曲线都不稳定（W20 比 best 高 0.010–0.021）。按"单模型 ≤500k 参数"的部署判据，
   继续扩大这一主体的理由不足。
3. **但"机制没有接上"这个解释被排除。** 固定权重干预显示训练后的 D 对 h 条件化写入有强依赖
   （+0.0377）；`train@best` D 0.0236 远低于 S 0.0535，说明 D 把更多容量装进了训练集，
   而收益没有转到 valid。这更像**主体在该日程/容量下不划算（under-trained + 不稳定）**，
   而不是"关系未被使用"。
4. **仍无法区分的三件事**（本轮的边界）：
   (a) 是 D 无用，还是 600-epoch/无 per-graph norm 的日程对 D 太短/太抖；
   (b) 是关系组合本身不划算，还是我们选的读出（单 reader、z=322）没有利用它；
   (c) S 的"静态写入"是否只是起了正则作用（其 train@final 0.1089 仍远未拟合训练集））。
   要区分它们需要：≥2 seed、per-graph 归一的稳定化 + 更长日程、以及一个 D/S 都能用的读出对照。

**下一步建议（不在本轮执行）**：不要继续在这一格子里加 seed 或调超参。若重启，唯一有意义的一步是
先修日程/稳定性（per-graph 归一是被本轮明确移除的变量），在**稳定的 valid 曲线**上重做 D/S 对照，
并让两支的 `train@best` 同量级后再比较；否则 D/S 的差异会被 valid 噪声淹没。


## 8. 限制与非主张

* **单 seed**：本轮只有 seed 0 的一对 D/S。任何 D/S 差异都带 `seed 0` 限定，不构成跨 seed 可重复性主张。
* **日程是探索日程**：600 epochs / 47,400 updates，不是 GRIT 原生 2000-epoch 配方；"没赢"不等于"该主体没有用"。
* **B-R0 是历史参照**，不是本轮重训：它来自 `zinc_performance_search_v1`（DCC-L，494,193 参数，4 seeds），
  test 数字是那一轮一次性读取的；本轮只用它的 valid/per-molecule 预测做同 split 参照，**不构成同配方对照**。
* **未读 test**：本轮任何数字都不是 test 主张，0.05× 目标未评估。
* 关系缓存覆盖 train/valid；`cache/` 可从原始图重建（`rc_runner.py prep`）。

## 9. 复现

```bash
# 本地（CPU）：关系缓存 + 检查
uv run python tracks/structural-sufficiency/zinc_relation_composition_probe_v1/code/rc_runner.py prep
uv run python tracks/structural-sufficiency/zinc_relation_composition_probe_v1/code/rc_checks.py --suite all
# 远端（A100）：计时 + 一对 D/S
rr run res rc-D-s0 --pool res-gpu0 -- uv run python .../code/rc_runner.py train --arm D --seed 0 --device cuda:0
rr run res rc-S-s0 --pool res-gpu1 -- uv run python .../code/rc_runner.py train --arm S --seed 0 --device cuda:0
# 本地分析（含固定权重干预）
uv run python tracks/structural-sufficiency/zinc_relation_composition_probe_v1/code/rc_analyze.py
```
