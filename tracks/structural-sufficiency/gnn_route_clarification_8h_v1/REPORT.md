# GNN 路线澄清 8h v1 — 报告

- 轮次：`gnn-route-clarification-8h-v1`；设计基线 `7de32adf4c89`
- 运行设备：res 双 A100-SXM4-40GB（res-gpu0 / res-gpu1），单 seed=0，官方 split，每 epoch 全量三划分 eval
- 训练/校准单元 GPU 时间：2.68 卡·小时（按单元 `seconds_total` 求和；两卡并行，墙钟 17:06→20:01Z，含屏障等待与两次崩溃重启）
- Test 每 epoch 被观察；下文 Test 数字只作固定状态描述，不参与任何选择。

## Q1. LCRC-C1 与 GINE 的差距依赖什么？

| 数据集 | 模型 | 选中 recipe | Val RMSE | Test RMSE | Val epoch | 参数 |
|---|---|---|---|---|---|---|
| ogbg-molesol | L | Rf | 0.7958 | 0.9534 | 260 | 228674 |
| ogbg-molesol | G0 | Rf | 0.8587 | 0.8902 | 136 | 377094 |
| ogbg-molesol | Gx | Rf | 0.9198 | 1.8997 | 177 | 379014 |
| ogbg-mollipo | L | Rf | 0.7069 | 0.7293 | 124 | 228674 |
| ogbg-mollipo | G0 | Rf | 0.6161 | 0.6800 | 183 | 377350 |
| ogbg-mollipo | Gx | Rf | 0.6307 | 0.6749 | 128 | 379014 |

**配对分子 bootstrap（2000 次，rng 1703，行共享重采样，负值=前者更好）**

| 数据集 | split | 对比 | ΔRMSE | 95% CI | 比值 | P(前者更好) |
|---|---|---|---|---|---|---|
| ogbg-molesol | valid | L − G0 | -0.0638 | [-0.1734, +0.0459] | 0.9281 | 0.867 |
| ogbg-molesol | test | L − G0 | +0.0578 | [-0.0751, +0.1873] | 1.0664 | 0.203 |
| ogbg-molesol | valid | L − Gx | -0.1245 | [-0.2510, +0.0039] | 0.8647 | 0.971 |
| ogbg-molesol | test | L − Gx | -0.7795 | [-2.0565, +0.0981] | 0.6548 | 0.845 |
| ogbg-mollipo | valid | L − G0 | +0.0886 | [+0.0177, +0.1653] | 1.1438 | 0.007 |
| ogbg-mollipo | test | L − G0 | +0.0495 | [-0.0020, +0.1029] | 1.0732 | 0.033 |
| ogbg-mollipo | valid | L − Gx | +0.0742 | [+0.0032, +0.1424] | 1.1174 | 0.020 |
| ogbg-mollipo | test | L − Gx | +0.0544 | [+0.0013, +0.1077] | 1.0808 | 0.024 |

**配方敏感性（同一 recipe 下 L−GINE 的 Val 差距）与 Gx−G0 输入桥**

| 数据集 | 模型 | R3 gap | Rf gap | 方向翻转? |
|---|---|---|---|---|
| ogbg-molesol | G0 | -0.0473 | -0.0630 | no |
| ogbg-molesol | Gx | -0.0638 | -0.1241 | no |
| ogbg-mollipo | G0 | 0.0848 | 0.0908 | no |
| ogbg-mollipo | Gx | 0.0505 | 0.0762 | no |

- Gx − G0，ogbg-molesol/valid：ΔRMSE=+0.0607 [-0.0643, +0.1866]，比值 1.0773
- Gx − G0，ogbg-molesol/test：ΔRMSE=+0.8373 [-0.0388, +2.1730]，比值 1.9457
- Gx − G0，ogbg-mollipo/valid：ΔRMSE=+0.0144 [-0.0287, +0.0583]，比值 1.0240
- Gx − G0，ogbg-mollipo/test：ΔRMSE=-0.0049 [-0.0514, +0.0442]，比值 0.9937

判读：**RECIPE_SENSITIVE = NO**（定义：两配方下 L−GINE 差距翻转，或 Rf 下差距减半；精确数值见上表，不据此编造优化理论）。

L/T 的 Val 比值（>1 表示 L 更差）：ogbg-molesol=0.9267, ogbg-mollipo=1.1474
判读：**C1_BEHIND_THIS_BUDGET = YES**，具体任务：ogbg-mollipo（门槛 5%，只适用本轮预算与这两套 recipe）。 注意方向在任务间相反：ogbg-molesol 上 L 的 Val 不差于 T，因此差距是任务依赖的，不是普遍落后。

## Q2. 强消息传递骨干 + 无可加全图残差的可加读出，能否保住预测能力？

| 数据集 | trunk | A recipe | A Val | T Val | A/T | A Test | T Test | A−T Test | 95% CI (Test) | A_VAL_GATE(1.05) |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | G0 | Rf | 1.0229 | 0.8587 | 1.1912 | 0.9314 | 0.8902 | +0.0380 | [-0.0669, +0.1363] | FAIL |
| ogbg-mollipo | G0 | Rf | 0.7474 | 0.6161 | 1.2131 | 0.6990 | 0.6800 | +0.0195 | [-0.0369, +0.0820] | FAIL |

判读：**不能**。Val 的 A/T 比值为 1.19/1.21，两个任务都超过 1.05 门槛；Test 方向一致但 95% CI 都含 0，因此正确的说法是“没有保住 Val 门槛，且 Test 也没有显著更差”，不是“显著退步”。A 的读出预算：H=253、实际读出参数 33,153（解析 nominal 33,192，Δ=39，`analysis/tables/stage23.json`）。

A 的训练/验证差显示它在拟合训练分子：ogbg-molesol Train RMSE 0.411 vs Val 1.023，ogbg-mollipo Train RMSE 0.232 vs Val 0.747（`analysis/tables/stage23.json`）——可加读出不是单纯“没训够”。

## Q3. 端到端可加模型失败时，是读出表达力还是联合训练优化问题？

| 数据集 | D Val fidelity (std) | D Val RMSE | T Val RMSE | D/T Val | D Test / T Test | D−T Test | 95% CI |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | 0.2244 | 1.0568 | 0.8587 | 1.2331 | 0.9778 / 0.8902 | +0.0858 | [-0.0017, +0.1779] |
| ogbg-mollipo | 0.2438 | 0.7059 | 0.6161 | 1.1423 | 0.6948 / 0.6800 | +0.0150 | [-0.0170, +0.0425] |

判读：**REPRESENTATION_VS_READOUT = 读出限制嫌疑**——A 与 D 都没有保住；D 在冻结表示上仍未达到 0.02σ 保真（0.224/0.244），但它的 Train fidelity 只有 0.119/0.129 而 Val fidelity 0.224/0.244（`analysis/tables/stage23.json` 的 curves），说明失败里同时有优化/过拟合成分；0.02σ 是冻结的严格门槛，本轮不能把“未达到”升级为“形式不可表达”。

## Q4. 原生贡献账目相对归因的增量价值

**oracle 预测精度与 Δ 预测（全部 823 对；σ = oracle Train 目标标准差）**

| property | model | Val RMSE/σ | Test RMSE/σ | Δf MAE/σ | 分量等权 Δ/σ | Δ 符号准确率 |
|---|---|---|---|---|---|---|
| MOLWT | A | 0.0687 | 0.0652 | 0.0076 | 0.0070 | 0.9975 |
| MOLWT | L | 0.0420 | 0.0589 | 0.0057 | 0.0054 | 1.0000 |
| MOLWT | T | 0.0311 | 0.0140 | 0.0050 | 0.0045 | 1.0000 |
| WC_LOGP | A | 0.0843 | 0.1060 | 0.0134 | 0.0121 | 1.0000 |
| WC_LOGP | L | 0.0887 | 0.0724 | 0.0109 | 0.0101 | 1.0000 |
| WC_LOGP | T | 0.0801 | 0.0912 | 0.0131 | 0.0135 | 1.0000 |

预测精度门槛（8.6）：A、T 的 Val/Test RMSE 均需 ≤0.10σ；可比性 |RMSE_A−RMSE_T|≤0.02σ。逐性质判读见下。

- MOLWT: CALIBRATION_OK；NOT_COMPARABLE；H 独立分量 0（<30，不能声明 A> T 的工作信号）。
- WC_LOGP: CALIBRATION_INCONCLUSIVE（A/T 未同时达到 0.10σ）；COMPARABLE；H 独立分量 0（<30，不能声明 A> T 的工作信号）。

**已知生成贡献校准**（truth = Crippen WC_LOGP / MolWt 的指定生成分解；E 按 8.5 定义，先分量内均值再分量均值）

| property | model | source | E (pair) | E (component) | var MAE | core MAE | size | global |
|---|---|---|---|---|---|---|---|---|
| MOLWT | A | ig | 0.3540 | 0.3688 | 18.5351 | 0.1590 | 18.4029 | 0.0000 |
| MOLWT | A | native | 0.3467 | 0.3607 | 18.2217 | 2.5274 | 15.5841 | 0.0000 |
| MOLWT | L | native | 0.1812 | 0.1967 | 8.9607 | 1.1524 | 0.0000 | 8.8722 |
| MOLWT | T | ig | 0.3242 | 0.3538 | 16.9448 | 0.1965 | 16.8352 | 0.0000 |
| MOLWT | zero | — | 0.1886 | — | — | — | — | — |
| WC_LOGP | A | ig | 0.3410 | 0.3865 | 0.1989 | 0.0869 | 0.1538 | 0.0000 |
| WC_LOGP | A | native | 0.3335 | 0.3770 | 0.2008 | 0.0999 | 0.1294 | 0.0000 |
| WC_LOGP | L | native | 0.3114 | 0.3225 | 0.1659 | 0.1027 | 0.0000 | 0.1329 |
| WC_LOGP | T | ig | 0.2781 | 0.3302 | 0.1582 | 0.0902 | 0.1102 | 0.0000 |
| WC_LOGP | zero | — | 0.4253 | — | — | — | — | — |

注：E 的 raw 单位是目标 σ；`zero` 为每对预测零贡献的对照。

- MOLWT: E_A(native)/E_T(IG) = 1.069，95% CI 0.982–1.161，P(A 更低)=0.057，分量 69（OK）。
- WC_LOGP: E_A(native)/E_T(IG) = 1.199，95% CI 1.108–1.293，P(A 更低)=0.000，分量 69（OK）。

由于主集合 H 的独立分量 <30，以上 A vs T 只作为固定状态描述，不构成“A 解释优于 T”的工作信号。

**读法（按 8.5 的冻结分组约定）**

- WC_LOGP：T-IG（E≈0.278）优于所有 native 账目（A≈0.334/0.341，L≈0.311），zero=0.425；A-native/T-IG=1.199，95% CI 1.108–1.293，即 **A 的原生账目显著更差**，不是“增量”。
- MOLWT：A-native 与 T-IG 基本持平（1.069，CI 含 1）；L-native 的 E≈0.181 看似最低，但其误差 8.87 全部落在 `global` 通道——L 的原生读出把整图质量项放进独立 rho（任务书允许的全图项），按冻结分组该部分记作误差。zero=0.189 已接近 L，因此不能把 L 的低 E 读成“更准的原子贡献”；这是分组/rho 约定的结果，不是 native 账目的解释优势。
- 两个性质都只有 train-train 分子上的 E（H 为空），所以即使方向有利也不构成未见配对上的工作信号。
- A 的 MOLWT E 由 size 通道主导（15.6–18.4，占 E 的绝大部分）：它把质量效应放进 N 通道，按 7.7 的分组必须记作大小通道成本，而不是变量贡献；这与 L 把质量放进 rho 是同一类“分组约定决定 E”的现象。

**Q4 判读：增量价值 = 未显示（NO_INCREMENT_THIS_BUDGET）**。WC_LOGP 是信息量较高的一栏：native 账目（A/L）显著差于 T-IG（A/T=1.199，CI 1.108–1.293）。MOLWT 一栏被分组约定主导：A 的 E（0.347）甚至高于 zero 对照（0.189），因为 A 把质量放进 size 通道、truth 把质量放在原子变量；L 的 E（0.181）低是因为它把质量放进 rho。两者都不是“更接近真值”，而是同一分配在不同通道上的记账差异——所以 MOLWT 不能用来判 native 与 IG 的优劣。综合两栏，本轮没有可复现的 native 增量。

- MOLWT/A: scope=FULL_MP, quadrature unresolved 0/96 
- MOLWT/T: scope=FULL_MP, quadrature unresolved 0/96 
- WC_LOGP/A: scope=FULL_MP, quadrature unresolved 6/96 (>5% → IG 部分标 NUMERICALLY_UNRESOLVED)
- WC_LOGP/T: scope=FULL_MP, quadrature unresolved 6/96 (>5% → IG 部分标 NUMERICALLY_UNRESOLVED)

**真实 MMP 预测对照（全部 823 对，不按标签挑核心）**

| model | stratum | n | 分量等权 |Δf−Δy| | pair 等权 |Δf−Δy| | 符号准确率 (|Δy|≥0.1) | 分量数 | 支持 |
|---|---|---|---|---|---|---|---|
| A | mmp_main_H | 0 | — | — | — | None | None |
| A | train-test | 27 | 0.3590 | 0.4491 | 0.9365 | 7 | LOW_SUPPORT |
| A | train-valid | 5 | 0.3459 | 0.3943 | 1.0000 | 4 | LOW_SUPPORT |
| A | train-train | 791 | 0.2735 | 0.2584 | 0.9096 | 228 | OK |
| A | ALL | 823 | 0.2749 | 0.2654 | 0.9113 | 231 | OK |
| D | mmp_main_H | 0 | — | — | — | None | None |
| D | train-test | 27 | 0.3683 | 0.4880 | 0.9008 | 7 | LOW_SUPPORT |
| D | train-valid | 5 | 0.3550 | 0.3552 | 1.0000 | 4 | LOW_SUPPORT |
| D | train-train | 791 | 0.3325 | 0.3168 | 0.8694 | 228 | OK |
| D | ALL | 823 | 0.3332 | 0.3227 | 0.8745 | 231 | OK |
| G0 | mmp_main_H | 0 | — | — | — | None | None |
| G0 | train-test | 27 | 0.3928 | 0.5037 | 0.9365 | 7 | LOW_SUPPORT |
| G0 | train-valid | 5 | 0.3608 | 0.3521 | 1.0000 | 4 | LOW_SUPPORT |
| G0 | train-train | 791 | 0.2796 | 0.2788 | 0.8906 | 228 | OK |
| G0 | ALL | 823 | 0.2827 | 0.2866 | 0.8926 | 231 | OK |
| Gx | mmp_main_H | 0 | — | — | — | None | None |
| Gx | train-test | 27 | 0.5747 | 0.5247 | 0.8571 | 7 | LOW_SUPPORT |
| Gx | train-valid | 5 | 0.4466 | 0.4317 | 1.0000 | 4 | LOW_SUPPORT |
| Gx | train-train | 791 | 0.2813 | 0.2840 | 0.8855 | 228 | OK |
| Gx | ALL | 823 | 0.2885 | 0.2928 | 0.8882 | 231 | OK |
| L | mmp_main_H | 0 | — | — | — | None | None |
| L | train-test | 27 | 0.4023 | 0.4513 | 0.9127 | 7 | LOW_SUPPORT |
| L | train-valid | 5 | 0.6106 | 0.6315 | 1.0000 | 4 | LOW_SUPPORT |
| L | train-train | 791 | 0.2655 | 0.2494 | 0.8977 | 228 | OK |
| L | ALL | 823 | 0.2702 | 0.2584 | 0.9031 | 231 | OK |
| T | mmp_main_H | 0 | — | — | — | None | None |
| T | train-test | 27 | 0.3928 | 0.5037 | 0.9365 | 7 | LOW_SUPPORT |
| T | train-valid | 5 | 0.3608 | 0.3521 | 1.0000 | 4 | LOW_SUPPORT |
| T | train-train | 791 | 0.2796 | 0.2788 | 0.8906 | 228 | OK |
| T | ALL | 823 | 0.2827 | 0.2866 | 0.8926 | 231 | OK |
| zero | ALL | 823 | 0.6457 | 0.6323 | — | 231 | OK |

注意：H（两端均非 Train）为空集，主集合指标只能报 `LOW_SUPPORT/NO_H_SUPPORT`；上表的非 train-train 结论仅基于 5 个 train-valid + 27 个 train-test 对。`T` 与 `G0` 是同一 checkpoint（本轮 T=G0），两行相同不是重复表错误。

**解释样本上的局部效应一致性（只把 variable 通道与 Δy 对比；这是代理指标，不是原子真值）**

| model/source | n | MAE | 分量等权 MAE | 符号准确率 | Pearson r | 支持 |
|---|---|---|---|---|---|---|
| A/ig | 96 | 0.3412 | 0.2992 | 0.8554 | 0.7900 | OK |
| A/native | 96 | 0.3595 | 0.3192 | 0.8554 | 0.7876 | OK |
| D/native | 96 | 0.3625 | 0.3320 | 0.8795 | 0.8012 | OK |
| L/native | 96 | 0.4014 | 0.3610 | 0.8554 | 0.7323 | OK |
| T/ig | 96 | 0.3452 | 0.2907 | 0.8675 | 0.7608 | OK |
（对照：该样本 |Δy| 均值 0.5583；样本 96 对，其中非 train-train 32 对/11 分量。）

## Q5. 最终路线建议

**(d) 暂停“更可解释因此值得牺牲性能”的方法主张**：A/D 没有性能-解释联合优势；IG 在冻结的 256 点预算下并非普遍可用（真实 Lipo T 账目 22/96 pair 未闭合，oracle WC_LOGP 6/96），只有 MOLWT 全部闭合；oracle 精度虽有部分达标，但 A/T 在 MOLWT 上超出 0.02σ 可比性、WC_LOGP 上 A 的 Test 0.106σ 未达 0.10σ。解释证据只达到“能重建/能描述”的层级；保留强预测参照与工具，暂停方法学主张。暂停是资源判断，不是证明不可能。

**现在相信 / 反对证据 / 关键未知 / 最值得研究 / 应停止**

- 本轮能复现的最强对象是强 GINE 预测器；原生可加账目与 IG 都能在端点重建预测，但都没有转化为可验证的解释优势。
- A 与 D 都没保住预测；IG 在 256 点内未满足完整性门槛的 pair 超过 5%；H 为空使 oracle 的工作信号无法成立。
- 这是优化/正则问题、读出形式问题，还是评价门槛（0.02σ 保真、256 点积分）本身过严；本轮设计无法分离。
- 先把“可加读出为什么在冻结表示上也拟合不到 T”的原因查清（优化/容量/正则），再用同一 oracle 判据检验任何新读出。
- 暂停新增架构与“可解释性换性能”的叙事；停止用完整性/守恒本身当作解释优势。

## 数值核验

| 项目 | 结果 |
|---|---|
| A scope | FULL_MP |
| A endpoint_max_abs_std | 0.00000002 |
| A ig_reconstruction_max_abs_std | 0.00225041 |
| A native_reconstruction_max_abs_std | 0.00000000 |
| A state_unchanged | True |
| D scope | NATIVE |
| D native_reconstruction_max_abs | 0.00000000 |
| D state_unchanged | True |
| L scope | NATIVE |
| L native_reconstruction_max_abs | 0.00000000 |
| L state_unchanged | True |
| T scope | FULL_MP |
| T endpoint_max_abs_std | 0.00000000 |
| T ig_reconstruction_max_abs_std | 0.00144051 |
| T native_reconstruction_max_abs_std | 0.00000000 |
| T state_unchanged | True |
| batch_single T | pass=True worst=0.00000055 |
| antisymmetry T | pass=True worst=0.00000007 |
| renumbering T | pass=True worst=0.00000038 |
| batch_single A | pass=True worst=0.00000002 |
| antisymmetry A | pass=True worst=0.00000002 |
| renumbering A | pass=True worst=0.00000013 |
| reallocation A | λ=0.0981 max|Δsum|=0.00000000 max rank change=33 sign-flip frac=0.6771 |
| reallocation D | λ=0.0650 max|Δsum|=0.00000000 max rank change=34 sign-flip frac=0.7135 |
| multi-mapping | NOT_RUN |

## 限制与不声称

- 两套 recipe 的预测差距不证明形式表达能力强弱；A/D 的对照只把瓶颈定位到“联合训练 vs 读出”，不构成不可表达性证明。
- 求和正确不等于化学解释正确；oracle E 只在该指定生成规则下校准，不冒充实验性质真值。
- H 集为空、非 train-train 分量少，MMP 主集合标 LOW_SUPPORT；不把 train-train 指标称为完全未见配对泛化。
- Test 已每 epoch 观察，不能称为盲测；bootstrap 只描述固定 checkpoint 的有限样本不确定性，不含 seed/配方选择偏差。
- IG 只对声明的连续输入表示与路径成立，中间点不是合法分子，不是物理干预。
- MolHIV：**BUDGET_NOT_RUN**。前置门（A/D 联合优势或 oracle 校准支持）未成立，按任务书不把它当作可解释性收益的证据；本轮不报告 HIV 数字。
- 多映射检查：**NOT_RUN**（冻结 catalog 每对只有单一已验证恒等核心映射；不制造替代映射以避免“最有利映射”选择）。
- 512/2048 点积分只作为诊断，不是冻结结果；256 点未过的 pair 仍按 NUMERICALLY_UNRESOLVED 处理，没有用通过子集宣布优胜。
- H 为空来自冻结 pair catalog 的构成（823 对中 791 对为 train-train），不是模型选择的结果；这也是路线 (b) 无法触发的直接原因。

