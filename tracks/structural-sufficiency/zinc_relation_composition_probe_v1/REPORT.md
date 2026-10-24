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

<!--RESULTS-->

## 9. 限制与非主张

* **单 seed**：本轮只有 seed 0 的一对 D/S。任何 D/S 差异都带 `seed 0` 限定，不构成跨 seed 可重复性主张。
* **日程是探索日程**：600 epochs / 47,400 updates，不是 GRIT 原生 2000-epoch 配方；"没赢"不等于"该主体没有用"。
* **B-R0 是历史参照**，不是本轮重训：它来自 `zinc_performance_search_v1`（DCC-L，494,193 参数，4 seeds），
  test 数字是那一轮一次性读取的；本轮只用它的 valid/per-molecule 预测做同 split 参照，**不构成同配方对照**。
* **未读 test**：本轮任何数字都不是 test 主张，0.05× 目标未评估。
* 关系缓存覆盖 train/valid；`cache/` 可从原始图重建（`rc_runner.py prep`）。

## 10. 复现

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
