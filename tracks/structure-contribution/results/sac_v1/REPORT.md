# BJOC-SAC v1 — 结构化约束的二阶交互张量（图级组合，非消息传递）

协议 `cscl-bjoc-sac-v1`（冻结于 `protocol_freeze.json`，amendment 1/2 见 `AMENDMENTS.md`）
轨道 `tracks/structure-contribution`，split tag **S**，两个 seed（GPU0=s0，GPU1=s1）。

---

## 1. 结论摘要（先回答四个问题）

**(1) SAC 是否超过 P3？——没有。** Locked 上 SACR − P3 = **−0.001916**（≈ −2.2% 相对），
配对 bootstrap CI [−0.005909, +0.001287] **跨 0**，两个 seed 方向相反
（s0 +0.001206 / s1 −0.005038）；Monitor 上 −0.000296（CI [−0.0034, +0.0031]），不复制。
预注册强阳性门槛是 **≤ −0.004**，未达到。

**(2) 真实结构 vs 通用二阶容量？——只有一条单区、且不复制、且不满足 FLOP/行和匹配的弱证据。**
Locked 上 SACR − SACU = **−0.003088**，CI [−0.005548, −0.000741] 排除 0、两 seed 同向、
ordinary 与 top-10-removed 都更优——这是本轮唯一"指向真实关系赋值"的预注册信号。
但 (a) Monitor 上只有 −0.000493，且 seed 1 反号（+0.000659），**未复制**；
(b) SACU 与 SACR 参数/初始化/预算相同但**不等 FLOP、不等行和**，因此该对比
**不是**"真实连接性"的因果效应；(c) SACU − P3 = +0.001172（null 并不比 P3 好）。
正确表述：在 SACR 训练好的权重里，喂入真实关系张量比喂入内容 null 张量好 ≈0.003；
不能推广为"结构带来因果增益"。

**(3) 张量是否真的被使用？——是，且是强证据（label-free，在读标签之前冻结）。**
把 SACR 的 T 换成 content-null：Locked 预测平均改变 **0.00697**（门槛 0.0025），
最大 0.5747（门槛 0.02）；把 T 置零：Locked MAE 从 0.0849 恶化到 **0.0995**（+0.0147）；
soup 之后 SACR 对 T₁/T₃ 的梯度仍有 3.5e-4…6.6e-3，而 **SACU 对 T₁ 的梯度塌到 2.3e-9**
（null 张量在 R1 上退化为近似常向量方向）。"被使用"成立，但**"被使用"≠"有用"**——
模型确实依赖张量，而依赖并没有转化为对 P3 的净增益。

**(4) 是否值得买 full10k 下一阶段？——不值得，按预注册规则停止。**
主对比既未过 −0.004 门槛、也未在 Monitor 复制；Fit 上 P3/SACR/SACU 的 soup Fit-MAE
几乎相同（0.00093–0.00103），说明三个臂在训练集上同等饱和，差异只出现在泛化端的
±0.002 量级重排。把 8k→10k 不会改变"P3 主干是瓶颈"这一结构事实，且预注册明文规定
无增益即停（不做 full10k、不做官方 valid/test、不做 rescue 变体）。

**最终判定：`CASE_D_NO_GAIN`**（`decision.json`）。6 条预注册强阳性条件中只有 **3 条**满足
（S2 幅度、S5 分区不变差、S6 张量参与）：S1（R−P3 ≤ −0.004）不足，S3/S4 的
**R−P3 半边**不成立（两 seed 反号、CI 跨 0），且唯一满足的 S2 在 Monitor 未复制。

---

## 2. 实验设计（与执行前冻结的版本完全一致）

### 2.1 三个臂

| 臂 | 参数 | 图级组合 | reader 输入 |
|---|---|---|---|
| **P3** | 413,505 | 无（BJOC-P3 原样：atom/bond/junction 编码 + Null composer 逐键 c_e） | P3 原始 148 维 |
| **SAC-R** | 439,689 | `T_s = Σ_{(e,f)∈R_s} q_s(e)·q_s(f) / sqrt(max(1,|R_s|))`，`q_s = tanh(W_s c_e) ∈ R^24` | `concat(P3_reader_input, T_1,T_2,T_3, log1p|R_1..3|)` = 396 维 → 256 → 256 → 1 |
| **SAC-U** | 439,689 | `T_s^U = [m_s/C(n,2)]·Σ_{e<f} q_s(e)q_s(f)/sqrt(max(1,m_s))`，`m_s` = 该分子真实 | 同上 |

- 三个臂共享：BJOC-P3 的 atom/bond/junction 编码器、masked SSL 初始化、Null composer 的
  逐键 `c_e ∈ R^96`、L1 raw-y、R3 配方（360 epoch、cosine 1e-3→1e-5、wd 1e-5、clip 5、
  batch 128、last-5 soup）、batch 顺序、更新数。
- SAC 与 P3 的差异仅两处：① 新增 3 个 96→24 投影（6,984 参数）；② reader 从
  148→256→256→1 换成 396→256→256→1（167,681 参数，**P3 的旧 reader 被删除**，
  不留死参数）。新增 26,184 = 6,984 + 19,200，审计值非估计值（`parameter_audit`）。
- 图级组合**没有** `h_e ← Σ_f …` 的逐键刷新：`T_s` 只作为图级向量进 reader，键表示不更新，
  因此不是消息传递（CPU gate G8 扫描全代码 + 逐键依赖探针）。

### 2.2 关系定义（物理键线图上，互斥、无序、e≠f）

- **R1**：线图距离 1；
- **R2**：线图距离 2；
- **R3**：同一"含环双连通块"内且线图距离 ≥ 3。
- R1/R2/R3 **两两不相交**（gate G2：最大重数 1）；桥键/螺环/稠合/不连通分子的语义
  由 G3 在真实分子上逐例验证（链 m=[5,4,0]、C6 m=[6,6,3]、螺环 2 块且 R3=0、
  稠合 1 块且 R3>0、不连通 R3=0）。
- Fit 上平均每分子 R1 34.70 / R2 44.01 / R3 12.10，合计 90.8 对，占全部无序键对
  C(n,2) 的 **29.0%**（每分子 313.0 对）。R3 在 6.05% 分子上为空（链状/无环块）。

### 2.3 数据与 split

- 仅官方 train 10k（PyG `subset=True`）；剔除 4 个历史 SMILES/PyG 不一致行 → 9,996 行。
- **新 salt `cscl-bjoc-sac-v1` 显式传入**（p3 轮 salt 默认值 bug 的回归测试）：
  save-time salt 可复现、由 metadata salt 重算得到同一 split、salt 探针会改变行集合、
  split E/F 按各自 **effective** salt 逐行复现（F 的 metadata salt 记录错误被确认：
  effective = `cscl-bond-relation-attribution-v1`）。
- 尺寸 7,996 / 1,000 / 1,000，9,993 个 identity group，**0** 个 group 跨区；
  新 zone hash `bb01cf9e… / 517acaab… / 1c08ea5f…` 与历史任何 zone hash **零碰撞**。
- **披露**：split S 是对"已被历史轮次用过的同一个 10k 池"的**新划分**，不是盲测。
  9,996 行全部在某个历史 zone 出现过（新 locked 对应旧 F：fit 786 / locked 106 / monitor 108）。
  轮内三臂对比仍然无标签泄漏（同一 Fit 训练、同一 Zone 评估、标签只读一次）。

### 2.4 评分与判据（全部在执行前冻结）

- 主评估区 **Locked**（1,000），复制区 **Monitor**（1,000）；分子级配对 bootstrap（5,000 次，
  单位=分子，seed 固定）；Δ 定义为 `mean(|err_A| − |err_B|)`，负 = A 更好。
- 强阳性 6 条（全部满足才算"强阳性"）：
  S1 `SACR−P3 ≤ −0.004`；S2 `SACR−SACU ≤ −0.003`；S3 两 seed 同向（R−P3、R−U 各自）；
  S4 配对 CI 排除 0（R−P3、R−U 各自）；S5 ordinary 与 top-10-removed 都不更差；
  S6 关系张量参与（label-free swap-null：mean|Δpred| ≥ 0.0025 且 max ≥ 0.02）。
- 决策词表 A/B/C/E/D 冻结在代码里（`sac.sac_case`，有序互斥）。

---

## 3. 执行记录

| 阶段 | 位置 | 结果 |
|---|---|---|
| split / verify-inputs / objects / relations / cpu-gates | 本地 CPU | split 9,996 行；参数审计 413,505 vs 439,689；**12/12 gates PASS** |
| freeze | 本地 | code `be4b2100…`、config `b03b70f5…`；amendment 1/2 后重冻（见 `AMENDMENTS.md`） |
| stage-inputs | res | 三个 zone 的 pair 总数、complete pair 总数、行序、Fit-only 标准化器全部匹配冻结值 |
| smoke | res-gpu0 | 20-update SSL + 每臂 1 epoch；实测投影 1.12 GPU-h（cap 4.0） |
| ssl ×2 | res-gpu0 / gpu1 | 7,560 updates，s0 1,622 s、s1 1,669 s；encoder hash `a1ede1c7… / 5df561c6…` |
| train ×6 | res-gpu0 / gpu1 | P3 511/627 s，SACR 691/796 s，SACU 734/914 s |
| predict ×2 | res-gpu0 / gpu1 | 每 seed 12 个 own + 4 个 swap + 4 个 zero-T |
| freeze-predictions | res | **28** 个预测文件 + label-free 参与度，combined `137cab44343528c7…` |
| score-holdout | res | **唯一一次** Monitor+Locked 标签读取（`holdout_read.lock`） |
| post-hoc / finalize | res | 梯度、机制、分区、分位；最终证据清单 91 项哈希 |

- GPU 时间 **2.101 h**（cap 4.0 h，未超）；未做任何超参/宽度/初始化/核函数搜索，未加 seed。
- 训练期间 `LabelGuard` 的 forbidden-label 访问尝试 = **0**（SSL 与训练 JSON 内均有记录）；
  `official_valid/test` 从未加载。
- 三个臂在**同一 seed 内**的 `encoder_sha256` 与 `shared_trunk_sha256` 完全一致
  （s0 `a1ede1c7…`/`5002baa9…`，s1 `5df561c6…`/`429de1f2…`），SACR 与 SACU 的
  **全状态哈希也相同**（s0 `a577d075…`，s1 `63849e7f…`）。

---

## 4. 主表

### 4.1 Locked（主评估区，n=1000，主指标 = 两 seed 平均 MAE，越低越好）

| 臂 | seed 0 | seed 1 | **seed 均值** | 相对 P3 |
|---|---|---|---|---|
| P3 | 0.082568 | 0.090999 | **0.086783** | — |
| SAC-R | 0.083774 | 0.085961 | **0.084867** | −2.21% |
| SAC-U | 0.086456 | 0.089455 | **0.087956** | +1.35% |

| 对比 | Δ（seed 均值） | 配对 CI | 判定 | seed 0 | seed 1 | ordinary Δ | top-10-removed Δ |
|---|---|---|---|---|---|---|---|
| SACR − P3 | **−0.001916** | [−0.005909, +0.001287] | NEUTRAL | +0.001206 | −0.005038 | −0.000199 | −0.000498 |
| SACR − SACU | **−0.003088** | [−0.005548, −0.000741] | **GAIN** | −0.002682 | −0.003494 | −0.002001 | −0.002617 |
| SACU − P3 | +0.001172 | [−0.001578, +0.003684] | NEUTRAL | +0.003888 | −0.001544 | +0.001802 | +0.001944 |

（`ordinary` = y > −3，n=933；`top-10-removed` 按两臂最大绝对误差取前 10 个分子剔除后重算。）

### 4.2 Monitor（复制区，n=1000）

| 臂 | seed 0 | seed 1 | **seed 均值** |
|---|---|---|---|
| P3 | 0.079688 | 0.084283 | **0.081985** |
| SAC-R | 0.081354 | 0.082023 | **0.081689** |
| SAC-U | 0.083000 | 0.081364 | **0.082182** |

| 对比 | Δ（seed 均值） | 配对 CI | 判定 | seed 0 | seed 1 | 复制 |
|---|---|---|---|---|---|---|
| SACR − P3 | −0.000296 | [−0.003402, +0.003067] | NEUTRAL | +0.001667 | −0.002259 | **否**（符号相反） |
| SACR − SACU | −0.000493 | [−0.003933, +0.003050] | NEUTRAL | −0.001645 | +0.000659 | **否**（符号相反） |
| SACU − P3 | +0.000197 | [−0.002019, +0.002440] | NEUTRAL | +0.003312 | −0.002919 | 否 |

**结论：Locked 上唯一的 GAIN 标签（SACR−SACU）在 Monitor 上既不过门槛、也不复制。**

### 4.3 误差结构（Locked，post-hoc，仅描述）

| | P3 | SAC-R | SAC-U |
|---|---|---|---|
| ordinary MAE（y>−3） | 0.079400 | 0.079201 | 0.081202 |
| tail MAE（y≤−3, n=67） | 0.189602 | 0.163767 | 0.181998 |
| top-decile-error MAE | 0.370800 | 0.359950 | 0.368190 |
| top-10 SSE 占比 | 0.603 | 0.456 | 0.517 |

Locked 尾部（n=67）SACR−P3 = −0.0258（post-hoc bootstrap CI [−0.0727, +0.0022]），
但 Monitor 尾部为 **+0.0091**（方向相反）→ **不复制，不构成结论**。
Δ 分布（Locked，SACR−P3）：median −0.00137，q10 −0.0398，q90 +0.0372，52.2% 分子更好——
是"宽分布上的轻微不对称"，不是"普遍改善"。

---

## 5. 机制证据（label-free，在读标签之前与预测一起冻结）

### 5.1 参与度（swap-null / zero-T）

| 区 | 诊断 | mean\|Δpred\| | max\|Δpred\| | 门槛 | 通过 |
|---|---|---|---|---|---|
| Locked | SACR 换成 content-null T | **0.006970** | **0.574720** | 0.0025 / 0.02 | ✅ |
| Locked | SACU 换成 real T | 0.003450 | 0.144300 | — | — |
| Monitor | SACR 换成 content-null T | 0.008530 | 0.768210 | — | ✅ |

| Locked 机制对照 | SACR | SACU |
|---|---|---|
| 自身 MAE | 0.084867 | 0.087956 |
| 换成对方张量后的 MAE | 0.086340（U） | 0.087800（R） |
| 张量置零后的 MAE | 0.099540 | 0.094010 |

即：置零 T 让 SACR 变差 **+0.0147**、SACU 变差 +0.0061（两者都真在用张量）；
把 SACR 的张量换成 null 让它变差 +0.0015，把 SACU 的张量换成真实关系仅变好 −0.0002。

### 5.2 梯度（Fit 数据，soup 之后）

| 臂 | \|∂L/∂T₁\| | \|∂L/∂T₃\| |
|---|---|---|
| SACR s0 / s1 | 3.5e-4 / 6.8e-4 | 6.6e-3 / 5.1e-3 |
| SACU s0 / s1 | **2.3e-9** / 2.7e-5 | 5.1e-3 / 5.2e-3 |

SACU 的 R1 张量在训练后**功能性死亡**（null 在 R1 上只剩一个近乎常数的方向，
`q_s(e)q_s(f)` 的行和结构被 `m_s/C(n,2)` 抹平），而 SACR 的 R1/R3 通道始终有有限梯度。

### 5.3 训练饱和度

soup Fit-MAE：P3 0.001011/0.000932，SACR 0.001007/0.000946，SACU 0.001027/0.000962
（单位同原协议）——三臂在 Fit 上同等饱和，没有"欠拟合"或"过拟合差异"可解释主表差异。

---

## 6. 预注册强阳性清单

| 条件 | 判定 | 数值 |
|---|---|---|
| S1 SACR−P3 ≤ −0.004 | ❌ | −0.001916 |
| S2 SACR−SACU ≤ −0.003 | ⚠️✅（余量仅 0.000088） | −0.003088 |
| S3a R−P3 两 seed 同向 | ❌ | +0.001206 / −0.005038 |
| S3b R−U 两 seed 同向 | ✅ | −0.002682 / −0.003494 |
| S4a R−P3 CI 排除 0 | ❌ | [−0.00591, +0.00129] |
| S4b R−U CI 排除 0 | ✅ | [−0.00555, −0.00074] |
| S5 ordinary 与 top-10 不更差 | ✅ | −0.00020 / −0.00050 |
| S6 张量参与达标 | ✅ | 0.00697 / 0.5747 |
| **Monitor 复制（R−P3、R−U）** | ❌❌ | −0.00030 / −0.00049，均符号相反 |

按 6 条主条件计：S1 ❌ / S2 ✅ / S3 ❌（R−P3 半边失败）/ S4 ❌（R−P3 半边失败）/
S5 ✅ / S6 ✅ → **3/6，不是强阳性**。S2 的余量只有 8.8e-5，属于"刚好越线"，
不应作为可靠结论引用。

---

## 7. 决策与停规则

- 冻结词表判定：**`CASE_D_NO_GAIN`**（`sac.sac_case`：R−P3 不是 GAIN → D）。
- 注意：CASE_B（"关系被用但无头条增益"）要求 R−U **在 Monitor 复制**；本轮 Monitor R−U
  为 −0.0005 且两 seed 反号，因此**不能**升格为 B。
- 停规则执行：**不做** full10k、**不加载**官方 valid/test、**不**增加 seed、
  **不**试 rank/核函数/宽度/初始化 rescue。SAC 路线到此为止。

---

## 8. 科学判断（含必须声明的限制）

1. **这一轮否证的是"加一个图级二阶交互张量就能把 P3 推过 0.004"这个假设**，
   而不是"结构关系在 ZINC 上无用"。二阶张量的信息瓶颈很硬：它对 reader 的贡献是
   24×3 维的**全局池化**向量，任何键级差异在求和后都会被平均掉；而 P3 的逐键表示
   本来就无损地进入了同一 reader。这与历史证据一致：`bond_relation_attribution_v1`
   （逐键 R−D −0.0175）和 `p3_bond_relation_bridge_v1`（零初始化门坍缩，swap Δpred=0）
   都表明"键级关系"比"图级关系摘要"更有信息量，但都需要一条**不被零初始化门压死**的通道。
2. **本设计的检出力有限，不能宣称"证明了无增益"。** Locked（1,000 分子）上 P3 自身的
   seed 间差就有 0.0084，而目标是 0.004；配对 CI 半宽约 0.0035，所以
   "真效应 = −0.004"与观测并不矛盾（CI 下界 −0.0059）。预注册要求"两 seed 同向 + CI 排除 0"
   正是为了防这种事，但它换来的是**低检出力**；正确读法是"未证明有增益"，
   而不是"有增益被排除"。
3. **SACR−SACU 的 −0.0031 不能读成因果。** SACU 与 SACR 的**计算图逐节点相同、
   参数与初始化逐位相同**（全状态哈希相等，gate G5），但 null 的每对贡献被
   `m_s/C(n,2)/sqrt(m_s)` 缩放，因此两者**不等 FLOP、不等行和、不等数值尺度**；
   Δ 里混着"关系内容"和"数值尺度/优化条件"两件事。这是本轮的已知设计缺陷
   （PROTOCOL.md §10 已披露），任何"结构带来 0.003"的说法都超出证据。
4. **"张量被使用"是本轮最硬的结论**：swap 与 zero-T 两个独立诊断、在读标签之前冻结、
   幅度是门槛的 2.8×/28×，且 SACU 的 R1 梯度塌到 2e-9 作为反证。模型确实把
   R1/R2/R3 的信息编进了图级表示，只是这份信息没有跑赢"P3 逐键内容 + 更大 reader"。
5. **是否买 full10k**：不买。三个理由：(a) 主对比未过门槛且不复制；(b) Fit 上三臂同等饱和，
   扩数据只会同比例放大三臂的拟合能力，不改变相对排序的机制来源；(c) 预注册停规则。
   如果未来还要碰这条线，应该先解决"图级池化丢信息"这个瓶颈（例如把关系张量做成
   逐键条件项但**不用零初始化门**），而不是先扩数据。
6. **与历史数字的可比性（重要）**：本轮 Locked 的绝对值（P3 0.0868）**不能**与广为引用的
   官方 test MAE 0.0698 相比——后者是官方 test 划分（更简单、且本轮从未加载），
   本轮是同一个 10k 池上的新 Locked 划分（67 个 y≤−3 的困难分子主导误差）。
   与同样在 Locked 上的上一轮 `p3_bond_relation_bridge_v1`（P3 0.0833、P3R−P3
   −0.000099、P3U−P3 +0.0012）对照，量级一致、方向一致（null 略微有害），
   这支持"本轮 P3 基线与历史同构、没有实现退化"的判断；但两者 split 不同，
   **不构成复现**。

---

## 9. 复现

```bash
# 本地（CPU）：数据/关系/门/冻结
uv run python tracks/structure-contribution/code/run_sac_v1.py split
uv run python tracks/structure-contribution/code/run_sac_v1.py verify-inputs
uv run python tracks/structure-contribution/code/run_sac_v1.py objects
uv run python tracks/structure-contribution/code/run_sac_v1.py relations
uv run python tracks/structure-contribution/code/run_sac_v1.py cpu-gates
uv run pytest -q tracks/structure-contribution/tests/test_sac_v1.py

# 远端（res）：每个 seed 一条 GPU
rr deploy res --pool res-gpu0
rr run res sac-ssl  --pool res-gpu0 --result tracks/structure-contribution/results/sac_v1 -- \
  python tracks/structure-contribution/code/run_sac_v1.py ssl     --seed 0 --device cuda:0
rr run res sac-train --pool res-gpu0 --result tracks/structure-contribution/results/sac_v1 -- \
  python tracks/structure-contribution/code/run_sac_v1.py train   --seed 0 --device cuda:0
rr run res sac-pred --pool res-gpu0 --result tracks/structure-contribution/results/sac_v1 -- \
  python tracks/structure-contribution/code/run_sac_v1.py predict --seed 0 --device cuda:0
# ... seed 1 用 res-gpu1；然后（顺序不可换）
#   freeze-predictions -> score-holdout（唯一一次标签读取）-> post-hoc -> finalize
```

关键哈希：
- code `be4b210076fe…`（`sac_v1.py`）、config `b03b70f5d518…`
- split S zone hash `bb01cf9e… / 517acaab… / 1c08ea5f…`
- SSL encoder `a1ede1c7…`（s0）、`5df561c6…`（s1）
- soup：P3 `b44ad052…`/`a5069c7b…`，SACR `c0af629c…`/`9f0659d3…`，SACU `ccf9d06e…`/`6f7bf498…`
- 28 个预测 combined `137cab44343528c7…`
- 完整清单：`evidence_manifest.json`（91 项 artifact，含 REPORT/DEVIATIONS + 8 项 GPU ledger + checkpoints）

---

## 10. 产物清单

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 预注册协议（含 §10 已知限制） |
| `AMENDMENTS.md` | amendment 1/2（均为正式训练前的报告/探针 bug，语义未变） |
| `DEVIATIONS.md` | 与协议的偏离及披露 |
| `protocol_freeze.json` | 冻结的 code/config/inputs 哈希 |
| `cpu_gates.json` | 12/12 gate 明细 |
| `split_S.json`, `split_compare_S.json` | split S 与历史 split 的对照、重叠披露 |
| `relations_manifest_S.json`, `relation_audit.json` | 关系对统计与逐分子审计 |
| `prediction_manifest_S.json` | 28 个预测 + label-free 参与度（读标签前冻结） |
| `holdout_read.lock` | 唯一一次标签读取的凭证 |
| `holdout_score_S.json`, `decision.json` | 主表、对比、决策 |
| `post_hoc_S.json` | 梯度、机制、分区、Δ 分位 |
| `per_molecule_errors_{locked,monitor}.csv` | 逐分子误差 |
| `gpu_budget_ledger.json` | 2.101 h ≤ 4.0 h |
| `evidence_manifest.json` | 91 项 artifact 的 sha256 |
