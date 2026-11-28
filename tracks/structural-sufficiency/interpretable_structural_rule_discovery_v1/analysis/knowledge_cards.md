# 修正后的知识卡片（ISRD-v1）

全部数字由 `write_report.py` 从 `phaseBC.json` / `phaseA_rules.json` / 冻结数据生成。
只描述三类事实：数据关联、模型内计算作用、经验等价表述。
**不声称化学因果机制**。

## A_sulfur_full2078 — S002（R128 第 3 位）

* 条件：`x_F[2078] < 0.13130644` → x < t（该结构“不满足”）
* 阈值：`0.13130644`；Train 上**不**等价于任何单一原始计数阈值
* 结构：size-4，4 节点 / 3 键；node=`b:C.ar.R.sp2.H1.deg3x1 a:C.ar.R.sp2.deg3x2 c:S.sp3.deg4x1`；bond=`a-a:AROMATIC/cjx1 a-b:AROMATIC/cjx1 a-c:SINGLEx1`；element=`CCCS`；nH=`0,0,1,0`；aromatic=`1110`
* 来源：X-F，26 棵来源树；规则身份 `F[2078] < 0.13130644`
* 补方向：`S003`，ge 方向，Train 上经验等价于 `raw_count ≥ 1`
* R128 系数：standardised `-0.2007`，raw-unit `-1.5956`（正系数 = 条件成立时 logit 升高）
* R128 计算作用：条件成立时该项贡献 `-0.0257`（standardised logit），不成立时 `+1.5699`；Train 激活比例 `0.9839`

| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |
|---|---:|---:|---:|---:|---:|
| train | 32372 | 0.0320 | 0.3724 | 10712 | 0.9839 |
| valid | 4081 | 0.0174 | 0.3125 | 4080 | 0.9922 |
| **DEVELOPMENT_TEST** | 4061 | 0.0251 | 0.5385 | 4060 | 0.9874 |

实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）：
* 条件成立 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`
* 条件成立 / 反例：graph 0（test），label=0，n_atoms=19，raw_count=0，scaled=0，`CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2`
* 结构存在 / 正例：graph 271（train），label=1，n_atoms=48，raw_count=2，scaled=0.288675，`Cc1cc(-c2ccc(N=Nc3cc(S(=O)(=O)O)c4ccccc4c3N)c(C)c2)ccc1N=Nc1cc(S(=O)(=O)O)c2ccccc2c1N`
* 结构存在 / 反例：graph 3（train），label=0，n_atoms=24，raw_count=2，scaled=0.408248，`Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1`
* 结构不存在 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`

* 说明：C–S 家族的最强规则来源；在 R128 中以“F[2078] 缺失”形式出现
* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。

## B_strongest — S381（R128 第 41 位）

* 条件：`x_F[2057] >= 0.120385855` → x ≥ t
* 阈值：`0.120385855`；Train 上经验等价于 `raw_count ≥ 1`（仅观察数据）
* 结构：size-4，4 节点 / 3 键；node=`b:C.ar.R.sp2.H1.deg3x2 a:C.ar.R.sp2.deg3x1 c:C.sp3.deg4x1`；bond=`a-b:AROMATIC/cjx1 a-c:SINGLEx1 b-b:AROMATIC/cjx1`；element=`CCCC`；nH=`0,1,1,0`；aromatic=`1110`
* 来源：X-F，4 棵来源树；规则身份 `F[2057] >= 0.120385855`
* 补方向：`S380`，lt 方向
* R128 系数：standardised `-0.8231`，raw-unit `-6.3412`（正系数 = 条件成立时 logit 升高）
* R128 计算作用：条件成立时该项贡献 `-6.2325`（standardised logit），不成立时 `+0.1087`；Train 激活比例 `0.0171`

| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |
|---|---:|---:|---:|---:|---:|
| train | 564 | 0.0000 | 0.0381 | 308 | 0.0171 |
| valid | 41 | 0.0000 | 0.0199 | 41 | 0.0100 |
| **DEVELOPMENT_TEST** | 57 | 0.0000 | 0.0321 | 57 | 0.0139 |

实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）：
* 条件成立 / 反例：graph 145（train），label=0，n_atoms=26，raw_count=1，scaled=0.196116，`CC(=O)OC(C)(C)c1ccc2c(c1)C(=O)CC1C(C)(C#N)CCCC21C`
* 结构存在 / 反例：graph 145（train），label=0，n_atoms=26，raw_count=1，scaled=0.196116，`CC(=O)OC(C)(C)c1ccc2c(c1)C(=O)CC1C(C)(C#N)CCCC21C`
* 结构不存在 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`

* 说明：R128 中标准化系数绝对值最大的规则
* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。

## C_readable_count — S039（R128 第 62 位）

* 条件：`x_C[483] >= 0.0966736525` → x ≥ t
* 阈值：`0.0966736525`；Train 上经验等价于 `raw_count ≥ 1`（仅观察数据）
* 结构：size-4，4 节点 / 3 键；node=`a:Cx2 b:Carx1 c:Nx1`；bond=`a-a:SINGLEx1 a-b:SINGLEx1 a-c:SINGLEx1`；element=`CCCN`；nH=`None,None,None,None`；aromatic=`10`
* 来源：X-C，11 棵来源树；规则身份 `C[483] >= 0.0966736525`
* 补方向：`S038`，lt 方向
* R128 系数：standardised `-0.1931`，raw-unit `-0.9821`（正系数 = 条件成立时 logit 升高）
* R128 计算作用：条件成立时该项贡献 `-0.9426`（standardised logit），不成立时 `+0.0396`；Train 激活比例 `0.0403`

| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |
|---|---:|---:|---:|---:|---:|
| train | 1325 | 0.0068 | 0.0387 | 591 | 0.0403 |
| valid | 216 | 0.0046 | 0.0205 | 216 | 0.0525 |
| **DEVELOPMENT_TEST** | 132 | 0.0076 | 0.0324 | 132 | 0.0321 |

实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）：
* 条件成立 / 正例：graph 4031（test），label=1，n_atoms=19，raw_count=2，scaled=0.458831，`CN1C(C)(c2ccccc2)c2ccccc2S1(=O)=O`
* 条件成立 / 反例：graph 38（train），label=0，n_atoms=29，raw_count=1，scaled=0.185695，`COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1C(NC(C)=O)CC2`
* 结构存在 / 正例：graph 4031（test），label=1，n_atoms=19，raw_count=2，scaled=0.458831，`CN1C(C)(c2ccccc2)c2ccccc2S1(=O)=O`
* 结构存在 / 反例：graph 38（train），label=0，n_atoms=29，raw_count=1，scaled=0.185695，`COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1C(NC(C)=O)CC2`
* 结构不存在 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`

* 说明：Train 上经验等价于原始计数 ≥k 且系数最大的未用规则
* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。

## D_sulfur_parent — S320（R128 第 1 位）

* 条件：`x_C[419] < 0.262612879` → x < t（该结构“不满足”）
* 阈值：`0.262612879`；Train 上**不**等价于任何单一原始计数阈值
* 结构：size-4，4 节点 / 3 键；node=`a:Carx3 b:Sx1`；bond=`a-a:AROMATICx2 a-b:SINGLEx1`；element=`CCCS`；nH=`None,None,None,None`；aromatic=`1110`
* 来源：X-C，4 棵来源树；规则身份 `C[419] < 0.262612879`
* 补方向：`S321`，ge 方向
* R128 系数：standardised `+0.0444`，raw-unit `+0.2313`（正系数 = 条件成立时 logit 升高）
* R128 计算作用：条件成立时该项贡献 `+0.0089`（standardised logit），不成立时 `-0.2225`；Train 激活比例 `0.9617`

| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |
|---|---:|---:|---:|---:|---:|
| train | 31642 | 0.0309 | 0.2010 | 10431 | 0.9617 |
| valid | 3968 | 0.0171 | 0.0897 | 3967 | 0.9647 |
| **DEVELOPMENT_TEST** | 3945 | 0.0246 | 0.1964 | 3944 | 0.9592 |

实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）：
* 条件成立 / 正例：graph 16（train），label=1，n_atoms=12，raw_count=0，scaled=0，`NNP(=S)(NN)c1ccccc1`
* 条件成立 / 反例：graph 0（test），label=0，n_atoms=19，raw_count=0，scaled=0，`CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2`
* 结构存在 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=2，scaled=0.426401，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`
* 结构存在 / 反例：graph 2（test），label=0，n_atoms=21，raw_count=2，scaled=0.436436，`CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21`
* 结构不存在 / 正例：graph 16（train），label=1，n_atoms=12，raw_count=0，scaled=0，`NNP(=S)(NN)c1ccccc1`

* 说明：C–S 家族的 COARSE 父结构；系数接近零
* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。

## E_weakest — S300（R128 第 45 位）

* 条件：`x_C[388] < 0.696310639` → x < t（该结构“不满足”）
* 阈值：`0.696310639`；Train 上**不**等价于任何单一原始计数阈值
* 结构：size-4，4 节点 / 3 键；node=`a:Cx2 b:Nx1 c:Ox1`；bond=`a-b:SINGLEx2 a-c:DOUBLEx1`；element=`CCNO`；nH=`None,None,None,None`；aromatic=`0`
* 来源：X-C，4 棵来源树；规则身份 `C[388] < 0.696310639`
* 补方向：`S301`，ge 方向
* R128 系数：standardised `-0.0030`，raw-unit `-0.0179`（正系数 = 条件成立时 logit 升高）
* R128 计算作用：条件成立时该项贡献 `-0.0005`（standardised logit），不成立时 `+0.0174`；Train 激活比例 `0.9705`

| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |
|---|---:|---:|---:|---:|---:|
| train | 31931 | 0.0370 | 0.0536 | 10509 | 0.9705 |
| valid | 3975 | 0.0199 | 0.0145 | 3974 | 0.9664 |
| **DEVELOPMENT_TEST** | 3951 | 0.0316 | 0.0309 | 3950 | 0.9606 |

实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）：
* 条件成立 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`
* 条件成立 / 反例：graph 0（test），label=0，n_atoms=19，raw_count=0，scaled=0，`CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2`
* 结构存在 / 正例：graph 1825（train），label=1，n_atoms=12，raw_count=4，scaled=1.1547，`CCC1(Br)C(=O)NC(=O)NC1=O`
* 结构存在 / 反例：graph 32（train），label=0，n_atoms=33，raw_count=1，scaled=0.174078，`CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1`
* 结构不存在 / 正例：graph 11（train），label=1，n_atoms=22，raw_count=0，scaled=0，`O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`

* 说明：R128 中标准化系数绝对值最小的规则（含反例）
* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。

