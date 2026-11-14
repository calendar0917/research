# REPORT — lcrc-lipo-context-small-s0-v1

`round_id = protocol_id = lcrc-lipo-context-small-s0-v1`，单 seed（0），官方
`ogbg-mollipo` scaffold 划分，六条训练各一次（五条 C1 家族 + 一条现成 D-MPNN
参照）。所有数字来自 `analysis/results/<arm>/metrics.json` 与
`analysis/results/<arm>/pred_*_{best,last}.npz`，由 `code/lp_analysis.py`
机械汇总，人工未改任何数值。

## 1. 判定（先给结论）

**臂级标签：`NO_SEPARATED_NEIGHBOR_GAIN_ON_LIPO`**

按预注册规则机械套用：

| 比较 | 标签 | 关键数字（Test，best-Val 点） |
|---|---|---|
| D2 vs S | `NO_CLEAR_INCREMENT_THIS_TASK` | −1.04%，MAE −0.28% |
| D2 vs R2 | `SELECTION_OR_SPLIT_SENSITIVE` | −0.18%（Val +0.79%，末窗 +0.18%，方向不一致） |
| D2 vs B | `SELECTION_OR_SPLIT_SENSITIVE` | −3.41%（Val +0.14%，末窗 −3.48%） |
| D2 vs R1 | `SELECTION_OR_SPLIT_SENSITIVE` | +1.54%（Val −1.24%，方向相反） |
| R2 vs B | `FIT_ONLY` | −3.22%（Train +2% 以上、Val/Test 均 ≤0） |
| G vs B | `PROMISING_THIS_TASK` | +2.61%（Val +18.22%） |
| G vs D2 | `PROMISING_THIS_TASK` | +5.82% |

一句话结论：**在这份很小的 scaffold 划分上，把"局部编码后的邻域状态交换"
加进冻结 C1 主干，没有带来可分离的预测增量，也没有修好可定位的结构性失败；
现成 D-MPNN 参照在 Test 上仍好 2.6%（验证集上差距 18%），说明缺的是容量/
归纳偏置层面的东西，而不是"多交换一次邻域状态"。**

## 2. 数据与划分核验（先于一切训练）

* 官方 `lipophilicity.zip` sha256 `c58d285a…25d0`，`mol.csv.gz` 4200 行。
* `GraphPropPredDataset('ogbg-mollipo').get_idx_split()` 原样使用：
  Train 3360 / Valid 420 / Test 420，与官方 `split/scaffold/*.csv.gz`
  逐一相等、两两不相交、并集 4200（`data/split_v1.json`；`code/lp_source.py`
  的 19 项身份核验全部通过 —— 16 条布尔断言全为真（划分相等/不相交/覆盖、
  标签一致、原子与有向边计数、OGB reader 一致、未用 random_split 等），
  另 3 条记录 reader 元数据与官方划分来源，结果写到 `data/manifest.json`）。
* 标签与结构分离：`data/labels_v1.npz`（4200 标签）与
  `data/raw_attributes_v1.json.gz`（无标签结构属性）；特征构建函数拒绝
  带标签行。目标统计只用 Train：`mean=2.16216`、`scale=1.21154`。
* `Lipop.random122.ind.pkl` 是 ChemBench 随机划分，**不是**官方 scaffold，
  本 round 未使用；旧 random122 checkpoint 仅作背景。

## 3. 训练、选择与评估

* 五条 C1 家族臂共用同一冻结主干（C1/`NO_BOND_MEAN`）与同一份私有初始化
  （`seed=17`，`fc2` 清零），epoch-0 预测与 B 逐位相等（max |Δ| = 0）。
* 参数量实测：B 228674；S/R1/R2/D2 各 262978（+34304）；G 318301。
* 每条臂 360 epoch（G 100 epoch），每个 epoch 全量评估 Train/Val/Test，
  评估用独立迭代器 + RNG 快照恢复，不扰动训练流（preflight d/e 逐位验证）。
* 选择只看 Valid RMSE（严格改进，平局取较早 epoch）。Test 仅用于本轮
  开发评估，读取记录在 `analysis/test_access_log.json`。
* 六条 run 合计 **1.21 A100·h**（soft 1.5 / hard 2.5 以内），峰值显存 0.19 GB
  （G 0.95 GB）。
* 远端 GPU preflight a–f 全通过（`analysis/preflight.json`），含
  "来源通道确实被使用"（`c_channel_used: true`）。

## 4. 主结果

见 `analysis/tables/TABLES.md` 主表。摘录（原始 logD，best-Val 点）：

| 臂 | Val RMSE | Test RMSE | Test MAE | best ep |
|---|---:|---:|---:|---:|
| B | 0.7371 | **0.7211** | 0.5301 | 180 |
| S | 0.7261 | 0.7380 | 0.5409 | 202 |
| R1 | 0.7270 | 0.7573 | 0.5567 | 175 |
| R2 | 0.7419 | 0.7443 | 0.5401 | 203 |
| D2 | 0.7360 | 0.7457 | 0.5424 | 175 |
| G | **0.6028** | 0.7023 | 0.5309 | 42 |

三条与"改进"有关的事实：

1. **基线本身就是最强的 C1 家族成员**：B 的 Test 0.7211 低于全部四条上下文臂；
   R2（固定扩展来源）与 D2（学习到的邻域状态）都退到 0.744–0.746。
2. **Val 与 Test 方向不一致**：S/R1/R2 在 Val 上小幅优于 B（+1.4%~+1.5%），
   Test 上反而更差 2.3%~5.0%。选择规则只看 Val，于是 best-Val 点选到了
   "Val 乐观"的位置，这是 `SELECTION_OR_SPLIT_SENSITIVE` 标签的来源。
3. **末窗与 last epoch 不救场**：末窗 Test（B 0.7186、S 0.7349、D2 0.7435）
   与 best 点同序，说明这不是"最佳点选择噪声"，而是持续的方向。

配对 scaffold 簇扰动（500 次，Exp(1)，共享权重）显示所有上下文对比的
Test ΔRMSE 区间都跨 0（例如 D2 vs S：−0.0077 [−0.0395, 0.0206]；
D2 vs B：−0.0246 [−0.0549, 0.0006]）。所以既不能说"更好"，也不能说"显著更差"，
只能说**没有可分离的增量**。

## 5. 结构群体诊断

群体在预测之前定义，在 **B 的 best-Val 点**选组，Test 冻结后才读：

| 群体 | 定义 | Valid/B gap | Test/B gap | 可估计 |
|---|---|---:|---:|---|
| H1 | radius-2 环境在 Train 中出现 0 次 | – | – | 否（Valid 非群体仅 17 个分子 < 40） |
| H2 | radius-2 支持 ≥ 5 但 radius-3 支持全 0 | **0.0913** | 0.1159 | 是 |
| H3 | SMARTS `[NX3]-[CX3](=[OX1])` | −0.0819 | −0.0657 | 是 |

* 选组结果 `FAILURE_GROUP_FROZEN`：**H2**（Val gap 0.0913 ≥ 0.05 阈值）。
  Test 上 B 在 H2 内的 MAE 比组外高 0.1170（[0.0171, 0.2115]），失败位置真实存在。
* 但上下文臂**没有缩小**这个 gap，反而更大：Test H2 gap
  S 0.1714、R1 0.1654、R2 0.1540、D2 0.2037，只有 G 是 0.0913。
  D2 vs B 的 Δgap = −0.0867（[−0.1386, −0.0372]），即 D2 让 H2 的相对劣势
  显著变大；H1 上同样为负（−0.0700）；只有 H3 接近 0 且区间跨 0（+0.0113）。
* 也就是说：**"这个位置没见过"这类失败是可定位的（H2），但不是靠再交换一次
  邻域状态能修的**；在给定预算下，邻域交换把误差更多地推给这些位置。

H1 在本划分上不可估计（非群体分子太少），按规则如实记为不可估计，没有
为了凑数放宽阈值。

## 6. 冻结来源干预（D2）

对 D2 的 `best`（epoch 175）与 `last`（epoch 360）冻结权重，在完整 Valid/Test
上替换第 2 层的消息来源：

| 变体 | Valid RMSE | Test RMSE |
|---|---:|---:|
| NORMAL | 0.7360 | 0.7457 |
| REUSE_ALL（来源 = `h0`，R1 语义） | 0.7472 | 0.7484 |
| FIXED_ALL（来源 = 固定 `c_j`，R2 语义） | 0.7613 | 0.7668 |
| REUSE_TARGET（只对 H2 根换源） | 0.7367 | 0.7450 |
| REUSE_CONTROL（同分子 degree 匹配对照根） | 0.7365 | 0.7464 |

读法（诚实版）：

* 学习到的来源**确实被用到**（去掉它 Test 变差 0.0027；换成固定 `c_j` 变差
  0.021），但幅度只有 0.4%~2.8%，与第 4 节的"无分离增益"一致。
* 只对 H2 根换源（0.7450）与对 degree 匹配对照根换源（0.7464）差 0.0014，
  远小于扰动区间；**没有"这些根的邻域状态特别关键"的证据**。
* 恒等式 `Δpred = Δcontrib + Δρ` 残差 ≈ 2.9e-7（float32 累加量级），
  说明干预确实只改了预测函数、没有别的路径。
* 这是对冻结预测函数的内部编辑，**不是可训练收益**，不能当作"再训一次会更好"。

## 7. 与 D-MPNN 参照的关系

G 在 Valid 上大幅领先（0.6028 vs 0.7371，+18.2%），Test 上是 0.7023 vs 0.7211
（+2.61%，区间 [−0.0257, 0.0643] 跨 0，删掉 control 最差 5% 后优势转为 −8.4%）。
因此本轮的诚实表述是：

* D-MPNN 参照在 Test 上仍有小幅优势，但**这份小 Test 无法把它分辨成显著**；
  Val→Test 的巨大收缩本身说明这个划分上的 Valid 不能当作能力的稳定量尺。
* 无论 G 还是 B，C1 家族都没有从"邻域状态交换"里拿到 Test 增量；
  G 的优势不是通过该通道实现的（它没有这种显式残差通道）。

## 8. 限制

1. **单 seed**。所有对比都是同 seed 下的一次训练，扰动区间只描述固定模型的
   样本不确定性，**不是**训练方差；因此"没有增量"只能说没有可分离增量。
2. **Test 是开发评估**。本轮被授权直接读 Test 并据其判断路线；这不等于
   Test 成为选择集——选择始终在 Valid 上。但重复使用会污染后续 fresh 评估。
3. **群体定义是代理**。"环境支持度"用 Morgan count-fingerprint（radius 3，
   redundant environments）的 Train 文档频率近似，不是精确同构计数；H1/H2
   的分子覆盖率很高（Train 中 2837/2606 个分子至少含一个这样的根），
   因此它们是"含这类位置的分子"，不是纯子集。
4. **小数据**。4200 分子、Test 420 个；任何 1–2% 的差异都在噪声量级。
5. H3 覆盖 1715/3360 个 Train 分子，"酰胺类片段"这个定义本身不够锐利。

## 9. 边界声明

本 round 只做了：单 seed、官方 scaffold、六个臂、`ogbg-mollipo`。**没有**启动
PCBA、ZINC、多 seed、多 split、预训练或任何其他数据集；没有改全局守卫，
没有伪造历史 freeze；`tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/`
等旧文件保持只读（哈希记录在 `lp_common.REUSED_MODULES`）。本轮的阴性结果
**不否定** PCBA 上的动态正信号，本轮的阳性（若出现）也**不能替代** PCBA 验证。

## 10. 产物与复现

| 产物 | 路径 |
|---|---|
| 规则（先冻结） | `PROTOCOL.md`、`protocols/*.yaml`、`configs/*.yaml`、`studies/*.yaml` |
| 数据身份 | `data/{manifest.json,split_v1.json,labels_v1.npz,raw_attributes_v1.json.gz,dmpnn_features_v1.npz}` |
| preflight（远端 GPU，a–f 全过） | `analysis/preflight.json` |
| 主表 / 比较 / 判定 | `analysis/tables/{TABLES.md,main.json,comparisons.json,judgements.json}` |
| 扰动区间 | `analysis/tables/bootstrap.json` |
| 群体 | `analysis/groups/groups.json`、`analysis/figures/groups.png` |
| 干预 | `analysis/interventions/interventions.json`、`analysis/figures/interventions.png` |
| 曲线与敏感性图 | `analysis/figures/{learning_curves,rmse_mae_tail}.png` |
| Test 读取记录 | `analysis/test_access_log.json` |
| 重复性核验 | `analysis/repro_check.json` |
| 逐臂原始输出（不入库） | `analysis/results/<arm>/`（`metrics.json`、`curve.csv`、`pred_*`、`checkpoint_*`） |

### 重复性

正式 sweep 里的 B 用 `rr run` + `lp_train.py` 直接启动；收尾又用控制平面
`research run lcrc_lipo_context_small_v1 --set stage=train --set arm=B` 独立重放一次
（`analysis/repro_check.json`）：best epoch 同为 180，六个 RMSE/MSE 数字零差，
360 个 epoch 的 `state_hash`、`lr`、`steps`、全部指标列逐位相同，六个预测文件
`max|Δ| = 0`；只有 `train_seconds`/`eval_seconds` 因 GPU 争用不同。也就是说
新 run 已接入 `research run`，并且协议在同一 commit 系列下可逐位重放。

总 GPU 计费 1.21 + 0.17 = **1.38 A100·h**（soft 1.5 / hard 2.5 以内）。

复现：`uv run --group lipo python -m pytest tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/tests`
（10 项通过），再按 `PROTOCOL.md` §9 走 `lp_preflight.py` → `lp_train.py` →
`lp_analysis.run_all()`。正式 run 的 commit 为 `678f7f602b18`，分析代码 commit
见 `git log`（分析只读取已拉回的 run 输出，不重训）。
