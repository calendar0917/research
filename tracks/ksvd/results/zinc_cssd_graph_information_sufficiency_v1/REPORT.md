# REPORT — zinc_cssd_graph_information_sufficiency_v1

**Protocol**: `zinc-cssd-graph-information-sufficiency-v1`（track ksvd，study zinc-context-gap）
**预注册**: notes/protocols 于 commit `031441b` 提交，**先于任何带标签评分**（带标签阶段只发生在其后的 heads run）。
**执行**: 全部本地 CPU（`uv run research run ... --mode scratch`，runner `zinc_cssd_graph_information_sufficiency_v1`，test_access=blocked），**0 GPU**，无 backbone 训练，official valid/test 从未加载。

## 1. 一句话决定

**分支 B（没有买到增量信息）**：在冻结的 CSSD 字典与两个冻结 DICT 消费者（s0/s1）上，
"字典原子支持 × 真实 patch 共覆盖"见证 `W`（32D，无标签、机制干预验证有效）在容量匹配
的 R-only 残差头对照和同架构的 shuffled 伪见证对照面前，**没有表现出任何真实对应关系
特异的增量预测信号**——OOF 上 `Δg_R = −0.00248`（CI95 [−0.00302, −0.00191]，真实见证
反而更差），dev 上 `Δg_R = +0.00005`（CI 跨零）；真实 vs 伪见证在两套评估中均无分离
（OOF `Δg_sham = +0.00022` CI 跨零；dev `+0.00062` CI 跨零、且低于预注册 GO 门）。
**本轮没有发现当前 CSSD 消费者的图级统计（814D reader 输入 `R_G`）遗漏该特定
字典—patch 组合统计的有效证据。** 这不授权"所有高阶关系/组合模型无用"的一般结论。

## 2. 被测对象与冻结定义（详见预注册）

- **见证**：`b[v,k]=1[alpha[v,k]≠0]`（部署算子精确数学：`tied_iht_codes(Dbar, phi−(phi@U)@Uᵀ, s=8, steps=10)`，冻结基底，恰 8-sparse）；`C(x)`=env 缓存中覆盖真实原子 x 的 root 集合；`n[x,k]=Σ_{v∈C(x)}b[v,k]`；`W_k=Σ_x C(n[x,k],3)/(Σ_x C(|C(x)|,3)+1e-9)`。
- **伪见证（sham）**：分子内固定种子（20261021+gid）置换支持行，真实 incidence 不动——保留每分子每原子使用次数、|C(x)|、分母，只破坏"支持—原子覆盖"绑定。
- **机制门（无标签）**：mean ‖W_real−W_shuf‖₁ = **0.781**（中位 0.750，p95 1.254），分母>0 的分子 100% 距离非零，10k 行中 0 个零分母 → **PASS**。干预真实有效，"真实≈伪"不是机制失效的产物。
- alpha 恒 8-sparse（nnz=8.0），每分子平均 10.9 个不同支持行（支持在分子内确有异质性）。

## 3. 冻结恢复与导出检查（全部通过；`restore_checks.json`）

- 基底 U/common_rms/D SHA 与冻结轮 manifest 一致；Q soup state_hash = `14d175e3…` 与 Q_meta 一致。
- 两个冻结 DICT soup 重放 fit/dev 预测 vs 已存 npz：max |Δh| ≤ 1.53e-5（fp32 保存精度，<1e-4 容差）。
- alpha 算子等价：冻结数学 vs 部署 `cssd_decode` max |Δphi_hat| = 2.4e-7。
- **真 `R_G`**：reader forward-pre-hook 捕获，实测 **814D**（unary 289 + pair 485 + global 32 + topology 8；历史 clarity audit 的 302D 属于旧 H1 骨架）；`aux["coord"]` 确认为 `DeployFull.code()` 的零占位，从未使用。
- 见证 batch 拼接/顺序不变性、真实分子节点重标号不变性：精确 0.0。
- 见证/R 导出阶段结构上不读 targets（字段清单核验：仅 W_real/W_shuf/gid/row_id）。

## 4. 三臂对照（冻结协议：头 846/814→13→13→1，L1，Adam 1e-3，full-batch，1500 epochs，分组 selection 早停；5 折 canonical-SMILES 分组 OOF；两种子分开）

**主指标：未校准 g MAE（OOF，8001 fit 行）**

| 臂 | 参数 | DICT_s0 | DICT_s1 | 两 seed 平均 |
|---|---|---|---|---|
| R-only | 10,791 | 0.03425 | 0.03315 | 0.03370 |
| R+W_shuffled | 11,207 | 0.03907 | 0.03375 | 0.03641 |
| R+W_real | 11,207 | 0.03891 | 0.03347 | 0.03619 |

| 对照 | 两 seed 平均 | CI95（组配对 bootstrap，2000 次，共享组抽样） | P(>0) |
|---|---|---|---|
| Δg_R = MAE(R-only) − MAE(R+W_real) | **−0.00248** | [−0.00302, −0.00191] | 0.000 |
| Δg_sham = MAE(R+W_shuf) − MAE(R+W_real) | **+0.00022** | [−0.00016, +0.00060] | 0.878 |

- Δg_R 为负（两 seed 同向：−0.00466 / −0.00031）：在 backbone 已在 fit 标签上训练的前提下，真实见证对 fit 行残差没有任何 OOF 可泛化的增量贡献，反而因多 32 维输入略增过拟合。
- Δg_sham 跨零：**真实绑定相对伪绑定没有可分辨的优势**——这是本轮机制层面的决定性阴性。
- top-20 分子贡献份额：不可定义（总增益为负，−19.88 over 2×8001 行）；分层（k=0 / 尺寸三分位）无任何子群出现方向一致的有利信号（OOF 分层带符号均值全部 ≥ −0.0013）。

**次指标（探索性；dev 1999 行 = 历史开发比较集，全 fit 训练的头）**

| 臂 | g MAE（两 seed 平均） | y_raw MAE（两 seed 平均） |
|---|---|---|
| R-only | 0.09327 | 0.12053 |
| R+W_shuffled | 0.09384 | 0.12085 |
| R+W_real | 0.09323 | 0.12030 |

| 对照 | 两 seed 平均 | CI95 | P(>0) |
|---|---|---|---|
| Δg_R（dev） | +0.00005 | [−0.00127, +0.00133] | 0.52 |
| Δg_sham（dev） | +0.00062 | [−0.00008, +0.00131] | 0.96 |

- dev 上 R+W_real 的 g/y 点估计略优于 R-only（−0.00004 g / −0.00023 y），但远小于预注册最小可分辨效应（+0.001），CI 全部跨零。
- dev 上 Δg_sham（+0.00062）> Δg_R（+0.00005）：若真实绑定携带可利用信息，应出现相反排序；实际模式与"任何 W 块都是额外噪声输入、真实内容不比打乱内容更有用"完全一致。
- 参照：冻结 DICT 自身 dev g 两 seed 平均 0.09257 / y 0.11993——所有头都未超过冻结模型本身。

## 5. 冻结决策规则的执行（未移动任何门槛）

GO(A) 要求 Δg_R ≥ +0.0010 且 Δg_sham ≥ +0.0005 且 CI 下界>0 且两种子同向且 top-20 份额<50%。实际：Δg_R = −0.0025（FAIL）、Δg_sham CI 跨零（FAIL）→ **分支 B**。机制门通过，恢复/一致性检查全部通过，非单 seed/单分子驱动。诚实记录执行中的工程故障（均不触碰科学语义）：

1. witness-export 首跑失败×2（`local_mol_id` 未挂到原始 train-only 行——改为"全局 train 行号=列表位置"恒等式并加校验；结果目录未创建）；成功 run `20261008-233715-de46e411`。
2. restore-checks 失败×3（provenance manifest 键名、dev npz 无逐行分量预测、`remap_molecule` 的置换方向不一致——后者由合成测试捕获后修复，重标号不变性修复后为精确 0.0）；成功 run `20261008-234214` / `234419`。
3. rg-export 失败×1（把 per-graph 的 R 误断言为 per-node；捕获本身正确）；成功 run `20261008-234557`。
4. **heads 完成×3**：`20261008-234818`（**作废**：OOF 评估误喂未标准化输入——数字离谱（OOF MAE 0.23–0.82 vs 基线 0.031）在读取任何对照前即判定为 wiring bug）、`20261008-235139`（修复后；分层指标呈现方式有误）、`20261008-235547`（**最终**：带符号分层对比）。三次修复均为机械性接线/呈现修复，预注册设计、阈值、种子、对照定义零改动；作废 run 的数字未用于任何判断。

## 6. 判读边界（如实声明）

- **这不等价于历史实验的重复**：旧 readout/pair/centre/triad 审计全部针对旧 compact-v4 表示（学习型 16D pair 态/48D patch 态），本轮是 CSSD 字典支持与真实物理原子共覆盖的绑定，且在当前部署消费者上执行。但**结论方向与历史一致**：又一类"数学上不在 `R_G` 里的高阶统计"在任务上没有买到增量。
- **不被本结果证伪的**：更大容量/不同读出形状的 W 利用、其他字典统计量（值而非支持、跨原子 k 的联合分布等）、phi65 输入表示、字典机制本身、Reader 泛化。本轮只关闭**这一个**预注册统计量。
- OOF 头 ≠ 整模型独立 OOF（backbone 见过 fit 标签）；dev 是历史开发比较集；头为单 init seed（训练 seed 不确定性未伪装成分子 bootstrap CI）。
- CSSD 基底历史 refit 含任务监督（非纯无监督）；W 的任何"信号"即便存在也不能归因字典独有价值——本轮无正信号，此点仅作范围备注。

## 7. 成本与产物

- 计算：全部本地 CPU；heads ≈ 105–127 s/run；总计 < 15 min CPU；**0 GPU 秒**。
- 产物：`witness_export.{npz,json}`、`restore_checks.json`、`rg_export.{npz×2,json}`、`heads_residuals.npz`、`heads_results.json`、`report_manifest.json`。
- Promoted runs：`20261008-233715`（witness-export）、`20261008-234214`/`234419`（restore-checks）、`20261008-234557`（rg-export）、`20261008-235547`（heads 最终）；作废/中间 heads runs（`234818`/`235139`）与本 REPORT 的故障记录对应，不作为证据。
- 代码：`tracks/ksvd/experiments/luyin16/zinc_cssd_graph_information_sufficiency_v1.py`；测试 18 passed（`test_zinc_cssd_graph_information_sufficiency_v1.py`，纯合成、无数据依赖）；快测子集 1935 passed。
