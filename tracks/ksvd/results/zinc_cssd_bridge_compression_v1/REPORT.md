# zinc_cssd_bridge_compression_v1 研究报告（2026-10-08）

**Protocol**: `zinc-cssd-bridge-compression-v1`（冻结提交 `021c07f`；实现与协议在同一提交内
冻结，先于本轮任何 dev 评分）。冻结对象 = `zinc_cssd_consumer_replacement_v1` 的 DICT
消费者：297,539 参数 M_COMP 骨架 + 冻结 CSSD 基底（U/common_rms/Dbar，non-persistent
buffers，非 Q/基底重拟合）+ 共享冻结 Q + 历史 fit 8001 / dev 1999（sorted
union(select, confirm)）+ 原训练配方（Adam 1e-3 / coupled wd 1e-5 / batch128 / clip5 /
FP32 / COMP 0.5）。两臂均为 DICT 消费者，各从头训练 240 轮。official valid/test 从未实例化。
study `zinc-context-gap`；全部正式执行经 `rr` 于 res-2（res2-cpu + res2-cu124，A100
c05，torch 2.5.1+cu124，driver 525.85.12）。

---

## 0. 一句话决定

**分支 D（inconclusive）：冻结基底与接入不动的前提下，容量更小的 B72 完整配方
（bridge 144→72→144，bridge 参数 20,736、总参数 235,331，−62,208 ≈ −20.9%）的完整
dev y_raw 效应在两个 seed 上方向翻转（s0 −0.00050 改善、s1 +0.00221 变差，均值
+0.00086），且分子抽样轴的组配对 CI95 [−0.00189, +0.00345] 同时跨入 A 的改善区与
超出 B 容忍区——不确定性本身横跨整个判读空间；字典读出完好（B72 alpha 均值干预
p95 2.84/2.64 ≫ marker 1e-4，响应比例 1.000，破坏性 ΔMAE +1.42/+1.37）。因此
**既不保留 B72 为性能/更省参数候选，也不关闭容量约束方向**；不加 seed3、不加其它
宽度、不改门槛。同期 CTRL288 复现历史 DICT soup 在 +0.00053/+0.00143（GPU 训练
非确定性量级，同节点同 regime），历史强模型与本轮 CTRL288 保留。**

## 1. 四个交付问题的直接回答

**Q1：B72 改善性能了吗？**
**没有可判定的改善。** 每 seed 配对 Delta_y = MAE(B72) − MAE(CTRL288)（完整 dev
y_raw，负=改善）：s0 **−0.00050**、s1 **+0.00221**，均值 **+0.00086**（≈ +0.71%）；
Delta_g：s0 **+0.0000003**、s1 **+0.00180**，均值 +0.00090。方向翻转且幅度落在
本配方噪声/种子涨落尺度内。
- 分支 A（保留性能候选）未命中：s1 变差、Δg 超 +1e-4 带、均值 Δy > −1e-4、CI 上界 > 0。
- 分支 B（更省参数、性能在工程容忍内）未命中：s1 Δy = +0.00221 > +0.002，且均值
  Δy CI 上界 +0.00345 > +0.002（Δg 同理 CI 上界 +0.00354）。
- 分支 C（读出不受支持）不成立：B72 两 seed 干预 p95 2.84/2.64 ≫ marker 1e-4。
- 分支 D 命中：两 seed 方向翻转 **且** 均值 Δy CI 同时满足 lower < −1e-4 与
  upper > +0.002。
因此**不能声称 B72 改善泛化，也不能声称压缩无效**；2 个 seed 无法分辨 ~±0.002 的
效应（这是本轮最重要的限制，见 §9/§10）。

**Q2：字典读出保持了吗？**
**保持，且未被钝化。** 唯一 alpha 均值干预对四个 soup 全部施行（fit 根均值 alpha 替换
全部根稀疏码，c/化学/J/Sem108/关系/Q 不动，不重训不重校准；被替换稀疏码使 phi_hat 输入绝对变化中位 0.0315）：

| soup | \|Δpred\| p95 | 响应比例(>marker) | ΔMAE(y_raw) | ΔMAE(g) |
|---|---|---|---|---|
| CTRL288_s0 | 2.932 | 1.000 | +1.423 | +1.430 |
| CTRL288_s1 | 2.099 | 1.000 | +0.891 | +0.895 |
| B72_s0 | 2.842 | 1.000 | +1.422 | +1.427 |
| B72_s1 | 2.637 | 1.000 | +1.371 | +1.380 |

B72 与 CTRL288 同量级的“有害可换”读出：更窄的 bridge 仍在真实读取稀疏码。噪声界
η = 1.67e-6（重复前向 + 同权重保存重载重放，本轮**不引入** raw/decoded identity
hook），marker = 1e-4。

**Q3：fit/dev 如何变化？**
**每 seed 内 fit 与 dev 同向移动**（fit Δy：s0 **−0.01117**、s1 **+0.00556**；
fit Δg：s0 −0.01105、s1 +0.00525）——方向翻转是整条轨迹/优化盆地层面的效应，不是
“容量约束改善泛化”的机制证据。gap = MAE_dev(g) − MAE_fit(g) 单列（§2）：s0 由
0.05413 扩到 0.06518（+0.01105），s1 由 0.06486 缩到 0.06141（−0.00345）——**没有
系统性 gap 收缩**。fit 与 dev 一起变（每 seed 内部）也说明：s0 的“更好”不是纯粹
容量正则化故事，s1 的“更差”也不是干净的过拟合故事。

**Q4：保留哪一个，或明确关闭什么？**
- **不保留任何候选**（A/B 均未命中）。
- **不关闭容量约束方向**（D，而非 E）：本配方效应未超种子涨落；B72 既非保留候选，
  也非已关闭配方。
- 保留：本轮 CTRL288（同期配对对照）与历史强 DICT 模型（replacement DICT_s0/s1
  dev y_raw 0.121518/0.118351）。
- 不购买：seed3、其它 bridge 宽度、初始化搜索、窗口/聚合/COMP/WD/LR 调整、
  EARLY120/LONG360（上一轮未运行，本轮亦未运行，不得写成已有负结果）。

## 2. 主表（四次 run，估计器 = mean(236..240) 全状态等权 FP32）

| 估计器×seed | 参数 | dev y_raw | dev y_cal | dev g | dev ell | dev s | fit y_raw | fit g | gap(g) | 步数 | 训练秒 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTRL288 s0 | 297,539 | 0.122044 | 0.118694 | 0.094245 | 0.049474 | 0.082530 | 0.048464 | 0.040114 | 0.054131 | 15120 | 1122 | 378 MB |
| CTRL288 s1 | 297,539 | 0.119777 | 0.119080 | 0.093126 | 0.053049 | 0.079801 | 0.036385 | 0.028266 | 0.064860 | 15120 | 1019 | 378 MB |
| B72 s0 | 235,331 | 0.121546 | 0.121546 | 0.094245 | 0.053675 | 0.080079 | 0.037297 | 0.029064 | 0.065181 | 15120 | 1080 | 372 MB |
| B72 s1 | 235,331 | 0.121986 | 0.121193 | 0.094922 | 0.050989 | 0.083277 | 0.041949 | 0.033513 | 0.061409 | 15120 | 1106 | 372 MB |

- 分量 MAE 不相加为总误差预算；ell/s 抵消诊断（辅助描述，不改 COMP 权重）：dev
  opposite-sign 比例 0.545/0.553/0.560/0.561，triangle gap 0.0378/0.0397/0.0395/0.0393，
  与历史一致，未随 bridge 宽度变化。
- k 组（既有定义，不扩展）：k=0 占 dev y 误差质量的 73.8–74.6%（n=1930）；k=0 内部
  两预注册轴（n_nodes ≤23 / 24..27 / ≥28 与 |g|≤p90 / >p90）仅作误差分析，
  B72 与 CTRL 无一致方向的组结构变化，不据此开新臂。
- b_y 只由各自 fit 残差中位数得到（s0：CTRL −0.02638 / B72 −0.00444；s1：
  CTRL +0.01045 / B72 −0.02007）；y_cal 可报告，校准变好不计正信号（本轮 B72
  的 y_cal 在 s0 与 y_raw 相同、s1 略降，无额外信号）。

## 3. 配对 delta 与组配对 bootstrap

Delta = MAE(B72) − MAE(CTRL288)，同 seed 配对（负=改善）：

| 指标 | s0 | s1 | 均值 |
|---|---|---|---|
| dev y_raw | −0.000497 | +0.002210 | +0.000856 |
| dev g | +0.0000003 | +0.001796 | +0.000898 |
| fit y_raw | −0.011166 | +0.005564 | −0.002801 |
| fit g | −0.011050 | +0.005247 | −0.002902 |

canonical-SMILES 组配对 bootstrap（2000 次，固定 seed 20261007，四 soup 与两指标
共享同一组抽样；每次先算每 seed 配对差，再取两 seed 平均，再取 CI）：

| 指标 | s0 CI95 | s1 CI95 | 平均 Delta CI95 | 平均点估计 |
|---|---|---|---|---|
| Delta_y | [−0.00408, +0.00311] | [−0.00179, +0.00620] | **[−0.00189, +0.00345]** | +0.00084 |
| Delta_g | [−0.00390, +0.00373] | [−0.00215, +0.00577] | [−0.00183, +0.00354] | +0.00088 |

平均 CI 上界 +0.00345 已超 B 容忍 +0.002，下界 −0.00189 进入 A 改善区；即分子抽样
不确定性本身横跨整个判读空间。**CI 只含分子抽样轴，不含训练 seed/基底不确定性，
也未校正候选选择；2 seed 不构成方法稳定性证明；dev 是反复使用的历史开发集，不是
新的 confirm/test。**

## 4. 噪声界与字典干预

- η = 1.67e-6（四 soup 的重复前向与同权重保存重载重放的最大 |Δpred|，同一 A100
  regime）；marker = max(1e-4, 10η) = 1e-4。本轮按授权**不使用** raw/decoded
  identity hook，停用 CSSD 解码的真实输入变化不被计为数值噪声。
- 干预（§1 Q2 表）：B72 两 seed p95 2.84/2.64 ≫ marker → 明确响应；破坏性干预使
  y_raw MAE 恶化 +1.42/+1.37，与 CTRL288（+1.42/+0.89）同量级。该干预只检验
  “消费者依赖稀疏码”，**不证明字典增量收益，也不优于 RAW**（本轮无 RAW 臂）。

## 5. 历史锚点与成本

- **同期 CTRL288 vs 历史 DICT soup**（replacement dev y_raw 0.121518/0.118351）：
  y_raw +0.00053（s0）/+0.00143（s1），g +0.00034/+0.00188。可定位项全部核验一致
  （init 哈希逐位复现历史 DICT、240 轮 schedule 逐位一致、参数审计 297,539、基底/Q
  哈希一致、同节点 c05 同 torch/driver），残余属 GPU 训练非确定性（15,120 步 cuBLAS
  级漂移），如实报告；同期配对结果不受影响。
- **B72 vs 历史（仅描述性）**：s0 +0.00003（几乎相同）、s1 +0.00364。**本轮配方效应
  以同期配对 CTRL288 为准**，不把弱化的新对照包装成对旧强模型的大幅提升。
- **成本**：参数 235,331 vs 297,539（−62,208，−20.9%）；实测训练时长
  s0 1080s vs 1122s（−42s）、s1 1106s vs 1019s（+87s）——**无一致的时长节省**；
  峰值显存 372 vs 378 MB（无实质差异）。四次训练 wall 合计 4348s，全部落在 c05
  （同 seed 两臂同节点，最理想情况）。

## 6. 冻结判读

| 条件 | 结果 |
|---|---|
| A 两 seed Δy < 0 | ✗（s1 +0.00221） |
| A 两 seed Δg ≤ +1e-4 且均值 < 0 | ✗ |
| A 均值 Δy ≤ −1e-4 且 CI 上界 < 0 | ✗ |
| A B72 两 seed 干预 p95 > marker | ✓（2.84/2.64） |
| B 每 seed Δ ≤ +0.002、均值 ≤ +0.001、CI 上界 ≤ +0.002（y 与 g） | ✗（s1 Δy +0.00221；CI 上界 +0.00345/+0.00354） |
| C 任一 seed 干预 p95 ≤ marker | ✗（响应明确） |
| D 方向翻转 / CI 横跨判读空间 | ✓（两者均命中） |
| **分支** | **D inconclusive** |

判读：**本固定 B72 配方（窄 bridge + 其必需的教师无关新初始化）没有可判定的性能
效应**；不能升级为主线，也不能据此关闭容量约束方向或宣告数据底噪。执行与读出接口
全部有效（roster 11 项检查全过，见 §7）。

## 7. 工程披露

1. **本轮无任何远端重试或定位修复**：四次正式训练、smoke、source/pretrain checks、
   终评均为首次提交运行成功（clean committed revision `021c07f`，未用
   allow-dirty/allow-stale）。提交前本地发现并修复 3 个纯本地问题（均在提交前、
   任何远端运行前）：(a) pretrain-checks 的 “无 bias” 断言布尔取反；
   (b) 一个变量名笔误（`wloc`→`wloc_g`）；(c) 两个测试用例自身数值/容差缺陷
   （CI 上界用例写成负数；bootstrap 平移用例改用精确 2× 缩放结构）。三者都不改变
   训练/评分语义，也未产生任何正式数字。
2. **`research-judgment` skill 未安装**于本环境（`~/.pi/agent/skills/` 仅
   remote-research-runner；定点确认一次，如实记录），按“命题—证据—竞争解释—决定”
   原则执行。
3. `research verify`：11 ok / 0 fail。其中 `records.load` 对历史 claim
   `claim-zinc-local-dictionary-component-supervision-seed0-v1-directional-not-confirmed-20261005.yaml`
   报 YAML ScannerError——该文件来自历史提交 `9f0a595`，**与本轮无关**，未修改。
4. **提交顺序**：协议/runner/config/实现/测试在同一提交 `021c07f` 中冻结，先于
   本轮任何 dev 评分；终评 `terminal_eval.json` 为 one-shot，未覆盖。
5. 预算：GPU 侧 smoke 48s + 四次训练 4348s + 终评 59s ≈ 74 min ≪ 3h；CPU 侧
   source 11s + pretrain 14s。同 seed 两臂同节点（c05），无节点/驱动混杂。
6. **来源/初始化相同不等于已证明所有漂移都来自 GPU 非确定性**：本报告只称残余
   CTRL-vs-历史差为“GPU 训练非确定性量级”，未购买独立确定性审计。

## 8. 来源、执行与复现

- **来源（只读）**：`results/zinc_cssd_nonlinear_binding_v1`（fold/targets/payload/
  kappa/prep/basis/Q，哈希核验）；`results/zinc_cssd_consumer_replacement_v1`
  （DICT_s0/s1 run 产物 + terminal_eval 锚点）。
- **代码**：experiments `luyin16/zinc_cssd_bridge_compression_v1.py`、runner
  `src/ksvd_research/runners/zinc_cssd_bridge_compression_v1.py`、config/protocol
  （`zinc-cssd-bridge-compression-v1`）、registry 已注册；测试
  `tests/test_zinc_cssd_bridge_compression_v1.py`（17 项全过）。
- **rr/Slurm**（res-2，均 clean committed revision `021c07f`）：source 56265（c02，
  res2-cpu）、pretrain 56266（c02）、smoke 56267（c05）、train CTRL288_s0 56268 /
  B72_s0 56269 / CTRL288_s1 56270 / B72_s1 56271（均 c05，res2-cu124）、terminal
  56299（c05）。8 条成功 run 已 promote（`records/runs/20261008-*`）。
- **产物**：`results/zinc_cssd_bridge_compression_v1/`：source_manifest.json、
  pretrain_checks.json、smoke.json、terminal_eval.json、runs/{CTRL288,B72}_s{0,1}/
  （manifest/curve/probes/schedule/init/last/epoch120/epoch240/members 236..240/
  soup/resume_state/fit_predictions + terminal 导出的 fit/dev 逐行预测 npz）。
- **复现**：
  `uv run research run zinc_cssd_bridge_compression_v1 --set model.stage=<stage> [--set model.arm=<CTRL288|B72>] [--set model.seed=<s>] [--set runtime.device=cuda:0]`
  （或 `python -m tracks.ksvd.experiments.luyin16.zinc_cssd_bridge_compression_v1`）；
  stages：source-checks → pretrain-checks → smoke → train（四 run）→ terminal-eval
  （one-shot，重放需新目录）。

## 9. 范围、限制与不外推

- dev 是反复使用的历史开发比较集，**不是独立 confirm**；bootstrap CI 只覆盖分子
  抽样轴；2 seed 不是方法稳定性声明。
- 分量 MAE 不相加；COMP 0.5 权重未动；不删行、不 clipping、无 oracle Q、校准不计
  正信号；不扩展子群/特征搜索。
- B72 是新宽度必然带来的新初始化，故本轮比较的是**B72 完整配方**；正/负结果都
  **不能单独因果归因为参数数量减少**，也不声称 step-0 函数相同。
- D 的结论限定于“本冻结接口/配方下、2 seed、该固定估计器”的 B72 宽度；不扩为
  字典无效（里程碑保持）、所有神经消费者无效、容量约束方向无效、或已达数据底噪。
- 不重开：XGBoost/CatBoost/FM、head-only、N/E joint、旧可训练 IHT、prototype、
  K/T/原子数搜索、消息传递/Transformer、WD/LR/EMA/SWA/窗口/HPO、COMP 权重、
  语义绑定/关系架构。

## 10. 未排除的解释与下一步

**仍未排除**：(a) 种子/优化盆地涨落在该尺度上主导宽度效应（2 seed 无法分辨
~±0.002）；(b) 更窄 bridge 存在真实但 seed 依赖的小效应；(c) 新初始化（宽度变化
不可避免）的贡献；(d) “容量约束改善 g 泛化”未被支持——gap 无系统收缩、fit/dev 每
seed 同向移动。

**下一步是否已有具体购买理由？没有。** 本轮唯一候选因两 seed 方向翻转落入 D，
没有可支撑新购买的效应方向或精度目标；在授权范围内不追加 seed/宽度/窗口，也不为
形成下一轮而另编候选。若研究者要推进容量约束方向，需要**新的预注册**明确精度目标
与配对控制；当前证据不构成该购买。
