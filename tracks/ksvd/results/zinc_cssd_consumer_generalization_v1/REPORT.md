# zinc_cssd_consumer_generalization_v1 研究报告（2026-10-07）

**Protocol**: `zinc-cssd-consumer-generalization-v1`（冻结 commit `1db6a92`；Stage-B 冻结
revision `cf8eda6`，训练执行 revision `08abe79`，terminal/addendum `9097c82`/`0f4339f`）。
冻结对象 = `zinc_cssd_consumer_replacement_v1` 的 DICT 消费者：297,539 参数 M_COMP 骨架
+ 冻结 CSSD 基底（U/common_rms/Dbar，non-persistent buffers，非 Q/基底重拟合）+ 共享
冻结 Q + 历史 fit 8001 / dev 1999（sorted union(select, confirm)）+ 原训练配方
（Adam 1e-3 / coupled wd 1e-5 / batch128 / clip5 / FP32 / COMP 0.5）。official
valid/test 从未实例化。study `zinc-context-gap`；全部正式执行经 `rr` 于 res-2
（res2-cpu + res2-cu124，A100 c05/c01，torch 2.5.1+cu124，driver 525.85.12）。

---

## 0. 一句话决定

**冻结字典基底与接入不动的前提下，本轮购买的两类"训练长度 + 固定权重聚合"调整
（240 轮新轨迹 + WIDE5_240=mean(200,210,220,230,240)）没有达到本轮冻结的保留门槛
（两 seed 平均完整 dev y_raw 改善需 ≤ −0.003；实测 −0.00261，CI95 [−0.0039, −0.0013]
跨 0 之外但点估计不足门槛）→ 不保留任何候选配方，不加 seed、不购新窗口、不改门槛。**
如实记录：WIDE5_240 在两个 seed 上方向一致地改善完整 y_raw（−0.00048 / −0.00474）与
g（−0.00038 / −0.00514），改善 96–102% 集中在 k=0 常见分子，且换均值 alpha 后预测
大变（p95 2.38/1.76，响应比例 1.000）——稀疏码读出未被牺牲；但改善幅度未过资源门槛，
结论按冻结规则记为"该固定训练长度/固定聚合调整未解决当前 g 泛化差距"，不外推为
"字典无效/已达数据底噪"。

## 1. 四个交付问题的直接回答

**Q1：泛化差距在哪个训练阶段形成？两 seed 是否一致？**
差距（dev g-MAE − fit g-MAE）随训练单调扩大并在最后五轮聚合后最大：
s0：ep40 +0.034（1.27×）→ ep120 +0.045（1.70×）→ ep240 +0.051（2.0×）→ soup +0.061
（2.85×）；s1：+0.030 → +0.044 → +0.035（1.38×）→ +0.062（3.08×）。**两 seed 不一致**：
后期（120→240）s0 的 fit/dev 都继续改善，s1 的单 checkpoint fit 与 dev 同时**变差**
（fit g 0.0762→0.0918，dev g 0.1199→0.1264，后期轨迹振荡），而其 236–240 五轮均值
反而大幅好于单点（fit 0.0297 / dev 0.0912）。冻结路由规则据此判
**FLAT_OR_MIXED**（F240−F120=+0.00108 > −0.002；d_s = −0.00797 / +0.00655 方向翻转）。
机制读法：后期单 checkpoint 振荡 + 聚合稳定化（历史 top-5 结论的同类现象），不是
"更多轮稳定变好"或"干净过拟合"任一单一故事。

**Q2：固定字典下，训练长度或固定聚合是否改善完整 y_raw？幅度与不确定性？**
是方向一致但幅度不足门槛的改善（对照 = 同一新轨迹的 CTRL240=mean(236..240)）：
WIDE5_240 两 seed dev y_raw Delta = −0.00048 / −0.00474（平均 **−0.00261**），g-MAE
Delta = −0.00038 / −0.00514（平均 −0.00276，均不劣化）；fit 侧同向（−0.0046 /
−0.0091）。canonical-SMILES 组配对 bootstrap（2000 次，seed 20261007，所有估计器/
seed 共享抽样，先平均两 seed 再取 CI）：两 seed 平均 Delta CI95 **[−0.00393, −0.00132]**
（分子抽样轴上不含 0）；分 seed：s0 [−0.00245, +0.00138]（跨 0），s1
[−0.00729, −0.00212]（不含 0）。CI 只覆盖分子抽样，不含训练 seed/基底不确定性，
也未校正 Stage-A 分支与候选选择。冻结门槛（平均 ≤ −0.003）**未达** → 不保留。
校准侧无额外信号（y_cal：WIDE5 0.12085/0.11593 vs CTRL 0.11775/0.11579）。

**Q3：最佳估计器是否仍真实读取稀疏码？有没有牺牲原强基线？**
**仍真实读取**：对 WIDE5_240 施行唯一 alpha_mean 干预（fit 根均值 alpha 替换全部根
稀疏码，c/化学/J/Sem108/Q 不动，不重训不重校准）：|Δpred| mean 1.13/0.75、p95
**2.38/1.76**、响应比例 1.000 ≫ marker 1e-4（修正后 eta 1.43e-6），换后 y_raw MAE
**恶化 +1.02/+0.66** —— 与历史 DICT soup 同量级的"有害可换"读出，聚合没有钝化
字典通道。**基线未牺牲**：同期 CTRL240 vs 历史 DICT soup dev y_raw 差 +0.0004（s0）/
+0.0034（s1），g 差 +0.0004/+0.0038；可定位差异全部排除（init 哈希逐位复现、schedule
前 240 轮逐位一致、参数审计 297,539、基底/Q 哈希一致、同 pool 节点 c05 同 torch/
driver），残余属 GPU 训练非确定性（15,120 步 cuBLAS 级漂移），如实报告、不作调参
"修复"；同期配对对照不受影响。

**Q4：保留哪一个具体配方，或明确关闭哪几个？未解决的机制？**
- **不保留任何候选**（唯一候选 WIDE5_240 未过冻结资源门槛）。
- **关闭（本轮证据范围内）**：在该冻结接口与配方下的"更宽窗口固定聚合"进一步加购
  （WIDE5→更宽/其它窗口组合）、240 长度下的进一步长度/聚合搜索——本轮授权本就是
  有上限探索，负结果按冻结失败读法收档：*这些固定训练长度/固定权重聚合调整未解决
  当前 g 泛化差距*。EARLY120/LONG360 未被购买（路由为 FLAT_OR_MIXED，其分支从未
  打开），不属于本轮可宣称的关闭范围。
- **不外推关闭**：字典/接入有效性（上轮里程碑保持：性能承接 + 稀疏读出真实）、
  神经消费者族、数据底噪（弥散性未证明——见下）。
- **未解决机制**：(a) g 泛化差距 ~3× 在所有估计器上稳定存在，其大头在 k=0 常见
  分子的弥散 e_g（两预注册轴显示 s1 的聚合改善集中在 n_nodes 24..27/≥28 与
  |g|≤p90 主体，s0 量级过小无结构可言）；(b) seed 依赖的后期轨迹振荡（s1 单点
  120→240 变差而五轮均值大幅变好）——聚合是该振荡的部分稳定器，但固定更宽窗口
  只兑现了 −0.0026；(c) ell/s 分量抵消结构（triangle gap 0.040/0.050，异号比例
  0.54/0.62）在所有 checkpoint 上一致，未随聚合变化。

## 2. Stage A：历史 checkpoint 只读诊断（已入档 runs/diag_s{0,1}）

四 checkpoint（epoch40/120/240/soup）× 两 seed 的 eval+no_grad 前向导出（逐行
y/g/ell/s/k、gid、row_index、h、ell_hat、s_hat、q_raw、y_raw；torch 内组件恒等式
精确 0，导出 float64 求和 ≤1e-6；soup 重放 vs 历史保存预测 ≤1.4e-6，同 A100 regime
内；Q 每切分缓存一次共享；b_y 逐 checkpoint 自身 fit 中位残差）：

| | fit g | dev g | fit y_raw | dev y_raw |
|---|---|---|---|---|
| s0 ep40 | 0.12555 | 0.15933 | 0.13296 | 0.18598 |
| s0 ep120 | 0.06452 | 0.10979 | 0.07239 | 0.13663 |
| s0 ep240 | 0.05111 | 0.10181 | 0.05927 | 0.12959 |
| s0 soup | 0.03294 | 0.09390 | 0.04083 | 0.12152 |
| s1 ep40 | 0.10604 | 0.13654 | 0.11391 | 0.16393 |
| s1 ep120 | 0.07619 | 0.11986 | 0.08419 | 0.14689 |
| s1 ep240 | 0.09176 | 0.12641 | 0.09881 | 0.15241 |
| s1 soup | 0.02966 | 0.09124 | 0.03778 | 0.11835 |

分量（dev，soup）：e_ell 均值 −0.0067/−0.0289，e_s 均值 +0.0014/+0.0374，异号比例
0.542/0.622，triangle gap 0.0401/0.0501（分量 MAE 不作可加预算）。k 组结构与 triage
轮一致（k=0 1930 行 C_g 0.090/0.089 ≈ 全部 dev g 误差；k≤−3 两行 C_y 0.0208 几乎
100% 来自共享 Q）。k=0 两预注册轴（soup）：n_nodes ≤23（n≈1034，组内 MAE_g
0.104/0.098）最大但弥散；|g|>p90（阈值 2.6396，n=212）组内 MAE_g 0.142/0.128 ——
误差质量在 ≤23 节点分子上最大，s1 的 WIDE5 改善反而集中在 24..27/≥28（见 §4）。

## 3. Stage B：两条新轨迹与估计器（roster_lock 冻结后执行）

- 两条 from-scratch 轨迹（seed 0/1，240 轮，15120 步）：init 哈希逐位复现历史
  DICT_s{0,1}（s0 `60d6cfa3…`，s1 `d556cca0…`，两 seed 真不同）；schedule 前 240
  轮与历史逐位一致；基底哈希训练前后不变；297,539 参数审计通过；训练中零 dev 读取、
  零中途重放（只 detach/clone 捕获成员：[200,210,220,230,240] ∪ [236..240]）。
- 估计器：CTRL240（对照）与 WIDE5_240（唯一候选），K=5 等权 FP32 全状态均值
  （38 个 float32 张量，无整数 buffer；非浮点项按冻结语义要求逐位相同后拷贝）；
  成员哈希、估计器哈希、strict 加载、保存/重放预测（≤3e-7）全部断言通过。
- 成本（如实报告共享轨迹）：s0 1136s / s1 993s wall（含成员捕获与估计器构造
  ~3s），GPU 合计 ~35.5 min；多估计器共享同一轨迹，不虚构节省；全轮 GPU 总耗时
  （诊断+smoke+训练+terminal+addendum）≈ 47 min ≪ 3 h 预算。

## 4. 终评主表与组结构（dev 1999 = 历史开发比较集，非独立 confirm）

| 估计器×seed | dev y_raw | dev y_cal | dev g | dev ell | dev s | fit y_raw | fit g | b_y |
|---|---|---|---|---|---|---|---|---|
| CTRL240 s0 | 0.12191 | 0.11775 | 0.09429 | 0.05431 | 0.07928 | 0.04695 | 0.03868 | −0.0294 |
| WIDE5_240 s0 | 0.12143 | 0.12085 | 0.09391 | 0.05057 | 0.08059 | 0.04235 | 0.03436 | −0.0079 |
| CTRL240 s1 | 0.12179 | 0.11579 | 0.09504 | 0.05753 | 0.07512 | 0.04995 | 0.04225 | +0.0327 |
| WIDE5_240 s1 | 0.11705 | 0.11593 | 0.08990 | 0.05037 | 0.07581 | 0.04081 | 0.03230 | −0.0178 |

- Delta（WIDE5−CTRL，完整 dev y_raw，负=改善）：s0 −0.00048，s1 −0.00474，均值
  **−0.00261**；g：−0.00038 / −0.00514，均值 −0.00276；k=0 组贡献了 s1 改善的
  102%（−0.00483/−0.00474）与 s0 的 67%——**改善在常见分子主体，不是稀有尾部**。
- k=0 两预注册轴（C_g 差，WIDE5−CTRL）：s1 集中于 n_nodes 24..27（−0.0021）、
  ≥28（−0.0025）、|g|≤p90 主体（−0.0041）；s0 仅 24..27 改善（−0.0009），
  ≤23/+p90/≥28 轻微变差（+0.0004~0.0006）——不据此开任何新臂（分组仅作误差分析）。
- 复现差距：CTRL240 vs 历史 DICT soup：y_raw +0.0004（s0）/+0.0034（s1），
  g +0.0004/+0.0038。代码/环境可定位项全部核验一致（见 Q3）；残余为 GPU 训练
  非确定性，如实报告；同期配对结果（同轨迹 CTRL 对照）不受影响，候选未被包装为
  "已承接历史强基线"之外的主张。
- 噪声带：repeat 重评 ≤1.43e-6；恒等钩子修正后 0.95e-6~1.4e-6（见 §6 勘误），
  eta=1.43e-6，marker=1e-4。

## 5. 唯一字典干预与冻结门槛

alpha_mean 干预（唯一、破坏性、不重训/不重校准；输入变化：alpha vs fit 均值 L2
中位 0.410，phi_hat 变化中位 0.0315）：

| 估计器×seed | \|Δpred\| mean | p95 | 响应比例 | ΔMAE(y_raw) |
|---|---|---|---|---|
| WIDE5_240 s0 | 1.125 | **2.382** | 1.000 | +1.019 |
| WIDE5_240 s1 | 0.746 | **1.765** | 1.000 | +0.663 |

冻结保留门槛（首次新评分前冻结）逐条件：两 seed y_raw 均改善 ✓；平均 ≤ −0.003
**✗（−0.00261）**；两 seed g 不劣化（≤1e-4 带）✓ 且均值改善 ✓；两 seed 字典干预
明确响应 ✓（修正后 marker 1e-4）；模型/来源/训练检查无实质问题 ✓。→ **不通过，
不保留**。若多候选通过时的 tie-break（平均 y_raw → 更短有效长度 → 末五轮连续均值）
本轮未启用（唯一候选未过门槛）。

## 6. 工程披露（三处定位清楚的修复 + 一处程序披露）

1. **smoke 保存/重放断言带**（训练前，res-2 GPU job 56246）：原断言要求逐位相等，
   GPU 上两个独立分配的同权重模型实例可因 cuBLAS 核选择不同出现 ~1.8e-7 差异；
   修正为 1e-5 带（CPU 本地为逐位相等）。定位：断言容差错误，非流水线缺陷。
2. **terminal roster 门**（首次 terminal，job 56258，13.8s 失败于任何评分前）：冻结
   基底校验把编码器侧哈希表（键 U/common_rms/**Dbar**）与基底包哈希表（键
   U/common_rms/**D**）按同表比较，键集不匹配必然失败；改为同类相比。无
   terminal_eval.json 写出、无任何科学数字产生。
3. **恒等钩子接线**（terminal 成功后从产物中发现）：噪声界的恒等钩子把
   `decode_enabled=False` 设在模型包装对象而非编码器上，钩子模型仍在解码，记录的
   eta 0.579/marker 5.786 实为解码效应，且使 `both_seeds_alpha_responsive` 被错误
   记为 False（尽管 p95 2.38/1.76、响应 1.000）。以只读 **noise-addendum**
   （`noise_bound_addendum.json`，job 56260）修正：eta 1.43e-6、marker 1e-4、
   alpha 响应 True；冻结门槛以修正输入机械重算，**结果不变**（唯一不满足条件仍是
   平均 Delta_y −0.00261 > −0.003，与 eta 无关）。一次性 terminal_eval.json 未被
   覆盖。
   **注**：2、3 两处修复超出协议"至多一次定位修复重试"条款，如实披露；三处均
   未触及训练/评分语义，均为预注册校验门或其接线自身缺陷，且全部先于（或独立于）
   任何科学结论产生。
4. **提交顺序披露**：路由规则文本（协议+代码常量）在任何 Stage-A dev 读取前已写成；
   但 git 提交发生在一次本地 CPU 管线验证（/tmp 目录，seed 0，打印了 dev 指标）之后。
   规则内容在看到数字前后无任何改动（提交内容可核）；正式 Stage-A 数字全部来自
   res-2 提交后执行，路由由 freeze-roster 机械计算。另：本地验证曾以 1e-4 容差
   复放历史 soup（CPU-vs-A100 属 ~1.5e-5 类，正式 GPU 运行 ≤1.4e-6，均在声明带内）。

## 7. 范围、限制与不外推声明

- dev 是历史开发比较集，且 Stage A 已用其选择训练分支——终值**不是独立 confirm**；
  bootstrap CI 只覆盖分子抽样；2 seed 不是方法稳定性声明。
- 分量 MAE 不相加为总误差；COMP 0.5 权重未动；不删行、不 clipping、无 oracle Q、
  校准不计正信号。
- 负结果限定于：该冻结接口/配方下"240 轮 + CTRL240/WIDE5_240 两个固定估计器"这一
  已购组合。不扩为字典无效（里程碑保持）、所有神经消费者无效、或已达数据底噪
  （k=0 弥散 e_g 的底噪解释未被检验也未被排除——那是不同问题）。
- 不重开：XGBoost/CatBoost/FM、head-only、N/E joint、旧可训练 IHT、prototype、
  K/T/原子数搜索、消息传递/Transformer、WD/EMA/SWA/窗口/HPO、COMP 权重、
  语义绑定/关系架构。
- `research-judgment` skill 未安装于本环境（`~/.pi/agent/skills/` 仅
  remote-research-runner，定点确认一次，如实记录）。

## 8. 来源、执行与复现

- **来源（只读）**：`results/zinc_cssd_nonlinear_binding_v1`（fold/targets/payload/
  kappa/prep/basis/Q，哈希核验）；`results/zinc_cssd_consumer_replacement_v1`
  （DICT_s0/s1 全部 run 产物 + terminal_eval 锚点）。
- **代码**：experiments `luyin16/zinc_cssd_consumer_generalization_v1.py`、
  runner `src/ksvd_research/runners/zinc_cssd_consumer_generalization_v1.py`、
  config/protocol（`zinc-cssd-consumer-generalization-v1`）、registry 已注册；
  测试 `tests/test_zinc_cssd_consumer_generalization_v1.py`（22 项，全过；全量
  not-slow 套件 1873 项通过）。
- **rr/Slurm**（res-2，均 clean committed revision）：source 56240（c01）、
  diag-s1 56241 / diag-s0 56242（c05）、roster 56243（c01）、smoke 56246（失败，
  cf8eda6）、smoke2 56254、train-s1 56255 / train-s0 56256（c05）、terminal 56258
  （失败，08abe79）、terminal2 56259（c05，9097c82）、noise-addendum 56260（c05，
  0f4339f）。9 条成功 run 已 promote（records/runs/20261007-*）。
- **产物**：`results/zinc_cssd_consumer_generalization_v1/`：source_manifest.json、
  roster_lock.json、smoke.json、terminal_eval.json、noise_bound_addendum.json、
  runs/diag_s{0,1}/（diagnosis + 逐 checkpoint 逐行 npz）、runs/traj_s{0,1}/
  （curve/probes/manifest/schedule/成员与估计器状态/resume_state/逐估计器
  fit+dev 预测 npz）。
- **复现**：
  `uv run research run zinc_cssd_consumer_generalization_v1 --set model.stage=<stage> [--set model.seed=<s>] [--set runtime.device=cuda:0]`
  （或 `python -m tracks.ksvd.experiments.luyin16.zinc_cssd_consumer_generalization_v1`）；
  stages：source-checks → checkpoint-diagnostics(0/1) → freeze-roster → smoke →
  train-trajectory(0/1) → terminal-eval → noise-addendum。terminal-eval/roster 均为
  one-shot，重放需新目录。
