# MolHIV：patch 化学信息与关系读出初步审计

日期：2026-08-17

## 问题

MolHIV 的节点有 9 维离散原子属性，边有 3 维离散键属性。现有
`node_struct.py` 的 `vectorize_patch` 只使用补齐后的邻接矩阵，因此属于
拓扑 patch；图级 `wl_chem_ring` 分支才把原子/键属性加入了置换不变的
patch 描述。

本审计固定同一数据子集、同一覆盖采样、同一 KSVD 字典和同一逻辑回归，
比较结构 patch 与带化学属性 patch，并测试把 patch 稀疏码的原子共现关系
作为图级特征。

## 结果（3000 个图，2400/300/300）

### 只比较 patch 内容

| patch | 验证集 ROC-AUC | 测试集 ROC-AUC |
|---|---:|---:|
| 拓扑 `wl` | 0.4991 | 0.4082 |
| 化学 `wl_chem_ring` | **0.6927** | **0.6021** |

这不是完整 MolHIV 结论，但说明在当前设置下，是否保留原子和键属性会
显著改变结果。纯拓扑 patch 不能与直接使用 MolHIV 节点/边属性的 CIN、
GINE 做公平比较。

### KSVD 原子共现关系

对相邻或重叠 patch 对 `(p,q)`，用稀疏码的绝对值累加
`|x_p| |x_q|^T`，得到字典原子之间的关系矩阵。结果如下：

| 输入 | 验证集 ROC-AUC | 测试集 ROC-AUC |
|---|---:|---:|
| 化学 patch 袋 `bag` | 0.6440 | 0.5445 |
| 关系矩阵 `relation` | 0.6463 | 0.4437 |
| 袋＋正确关系 | **0.7857** | 0.5338 |
| 袋＋打乱关系 | 0.7052 | 0.4570 |

拓扑 patch 的对应结果为：

| 输入 | 验证集 ROC-AUC | 测试集 ROC-AUC |
|---|---:|---:|
| 拓扑 patch 袋 | 0.5782 | 0.4857 |
| 关系矩阵 | 0.7211 | 0.3639 |
| 袋＋正确关系 | 0.5663 | 0.4273 |
| 袋＋打乱关系 | 0.5550 | 0.3963 |

### 三个抽样种子的补充检查（化学 patch）

三个 3000 图抽样的平均结果（括号内为标准差）为：

| 输入 | 验证集 ROC-AUC | 测试集 ROC-AUC |
|---|---:|---:|
| patch 袋 | 0.6699 (0.0561) | **0.6261 (0.0762)** |
| 关系矩阵 | 0.5962 (0.0761) | 0.5752 (0.0939) |
| 袋＋正确关系 | 0.6893 (0.1024) | 0.6200 (0.1086) |
| 袋＋打乱关系 | 0.6833 (0.0156) | 0.6100 (0.1194) |

因此当前这个最初的关系读出还不能声称提升 MolHIV 分类。它在个别抽样
上有增益，但平均值略低于只用化学 patch 袋，而且波动较大。

## 当前解释

1. 化学信息不是小修补，而是 MolHIV patch 的必要组成部分。
2. 原子关系矩阵在个别验证集上显示出增量，但三个抽样的测试集没有稳定
   增量；3000 子集的测试集很小，不能据此声称已经提升下游分类。
3. 关系矩阵本身仍有较高训练拟合，说明当前读出维度偏大，且关系定义过于
   粗糙。需要降维、正则化和多折验证。
4. 目前最合理的主线是“结构通道＋化学通道＋关系读出”，而不是继续堆叠
   Transformer。

## 官方训练集内部三折

为了排除 3000 子集划分的偶然性，进一步使用官方训练集内部固定三折，
每一折只用训练部分拟合 KSVD 字典，并在留出部分比较。这里使用的是
化学 patch 和压缩后的对称关系读出：关系矩阵取对角、行和与特征值，
不直接展开全部矩阵元素。

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| patch 袋 | 0.6065 | 0.5763 | 0.6129 | 0.5986 |
| 关系摘要 | 0.5923 | 0.5531 | 0.5869 | 0.5775 |
| 袋＋正确关系 | **0.6246** | **0.5976** | **0.6381** | **0.6201** |
| 袋＋打乱关系 | 0.6204 | 0.5817 | 0.6112 | 0.6044 |
| 图大小基线 | 0.6583 | 0.6218 | 0.6007 | 0.6269 |

相对于只用 patch 袋，加入正确关系平均增加 `+0.0215`；打乱关系只增加
`+0.0058`。因此，在控制字典、采样方式和分类器后，patch 关系已经显示出
可重复但幅度有限的下游信号。它还没有超过简单的图大小基线，不能宣称
已经解决 MolHIV，但“关系不是纯噪声”这一点得到了更强证据。

## 下一轮要求

- 使用完整官方训练集的固定内部折，而不是只看 3000 子集；
- 保持化学 patch 的中心标记和键属性；
- 对关系矩阵做低秩/对称压缩，避免直接展开全部 `K×K` 元素；
- 比较正确关系、打乱 patch 关系、完全不使用关系；
- 与原子/键属性的简单图级统计、CIN/GINE 基线同时报告。

## 化学类型关系的后续实验

随后把关系拆成多个通道：

- patch 的几何重叠/相邻关系；
- 共享原子的类型；
- 跨 patch 连接边的类型。

每个通道分别形成原子关系矩阵，再用对角、行和与特征值做低维读出。官方
训练集内部三折结果为：

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 化学 patch 袋 | 0.6065 | 0.5763 | 0.6129 | 0.5986 |
| 袋＋化学类型关系 | **0.6301** | **0.6309** | **0.6563** | **0.6391** |
| 袋＋打乱化学类型关系 | 0.6288 | 0.6102 | 0.6373 | 0.6254 |
| 图大小基线 | 0.6583 | 0.6218 | 0.6007 | 0.6269 |

相对 patch 袋，正确化学类型关系平均增加 `+0.0405`；打乱关系增加
`+0.0268`；正确关系相对打乱关系仍有 `+0.0137` 的净增量。它在平均值上
超过了图大小基线，说明“关系＋化学”已经是值得继续发展的方向，但打乱
关系也有较大增益，仍需进一步降低关系读出的冗余和模型容量。

## 全局低秩压缩的反例

为控制关系特征维度，又在每个训练折内单独拟合标准化和 PCA，只保留 16
个主成分。结果为：

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 化学 patch 袋 | 0.6065 | 0.5763 | 0.6129 | 0.5986 |
| 袋＋正确关系的 PCA | 0.6148 | 0.5894 | 0.6450 | 0.6164 |
| 袋＋打乱关系的 PCA | 0.6341 | 0.5880 | 0.6230 | 0.6151 |

正确关系相对打乱关系的净增量只有 `+0.0013`。因此不能把全部关系特征
直接混合后用普通 PCA 压缩；这种方法破坏了关系类型和字典原子对应关系。
下一步应使用保留通道结构的压缩方式，或者用受限的关系传播，而不是无差别
地做全局降维。

## 保留通道的低维关系

最后只保留每个关系通道对应到每个字典原子的行和，不在通道之间做混合。
关系维度从 600 降到 200，但仍保留“几何关系、共享原子类型、跨 patch
键类型”这三类关系的边界。

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 化学 patch 袋 | 0.6065 | 0.5763 | 0.6129 | 0.5986 |
| 袋＋通道保持关系 | **0.6477** | **0.6115** | **0.6669** | **0.6420** |
| 袋＋打乱通道关系 | 0.6377 | 0.6032 | 0.6477 | 0.6296 |
| 图大小基线 | 0.6583 | 0.6218 | 0.6007 | 0.6269 |

通道保持关系相对 patch 袋平均增加 `+0.0434`，相对打乱关系仍有
`+0.0124` 的净增量。这个结果比全局 PCA 更稳定，支持“保留关系类型、
压缩每个关系通道”的设计。

## GINE 公平侧路检查

为了确认关系是否能帮助一个正常使用原子/键属性的图模型，又在 8000 图开发
集的固定内部三折上训练相同的三层 GINE。关系作为图级旁路输入，与 GINE
图表示拼接；所有旁路特征都只用训练折标准化。

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 原始 GINE | 0.7301 | 0.6833 | 0.6486 | 0.6873 |
| GINE＋patch 袋 | 0.7065 | 0.7161 | 0.6481 | 0.6902 |
| GINE＋正确关系 | 0.6902 | 0.7329 | 0.6466 | 0.6899 |
| GINE＋打乱关系 | 0.7065 | 0.7329 | 0.6248 | 0.6881 |

直接拼接关系没有带来稳定提升。说明关系对简单线性读出有用，但 GINE
已经自己传播原子和键信息，额外的图级关系摘要容易与它的表示重复或互相
干扰。后续不能继续做简单拼接，应改为关系条件下的节点/边传播，或者使用
冻结 GINE 后的受控残差融合。

### 受控残差补充

又测试了一个零初始化的小残差分支：GINE 保持主分类头，KSVD 关系只通过
一个可学习的小门控残差进入输出。三折结果为：

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 原始 GINE | 0.7301 | 0.6833 | 0.6486 | 0.6873 |
| GINE＋正确关系残差 | 0.7245 | 0.6968 | 0.6394 | 0.6869 |
| GINE＋打乱关系残差 | 0.7272 | 0.7195 | 0.6575 | 0.7014 |

受控残差也没有证明关系能独立帮助 GINE，打乱关系甚至更好。因此当前
“图级关系旁路”应当停止。关系需要在 patch/节点内部参与传播，而不是在
GINE 的最终图表示后面再拼一个向量。

## patch 层一层传播的检查

最后实现了一个只在 patch 图上做一层传播的轻量模型。每个 patch 是一个
节点，关系分为几何、共享原子和跨 patch 键三类；图级分类只对传播后的
patch 表示做平均汇总。

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 不传播 | 0.5919 | 0.6308 | 0.5895 | 0.6041 |
| 正确关系传播 | 0.5945 | 0.6284 | 0.5845 | 0.6025 |
| 打乱关系传播 | 0.5844 | 0.6269 | 0.5753 | 0.5955 |

打乱关系会稳定变差，说明模型确实在读取关系；但正确关系暂时没有超过
不传播版本。这说明关系传播方向比图级拼接更合理，但当前 patch 节点只用
KSVD 稀疏码，信息仍然不足，传播形式也过于简单。下一步应让 patch 节点
同时保留局部化学描述，并使用关系条件的门控传播，而不是直接相加。

## 补回原始化学摘要后的门控传播

将每个 patch 的 KSVD 稀疏码与原子/键类型直方图拼接，再使用按关系类型
计算门值的传播层。三折结果为：

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 不传播 | 0.6434 | 0.7104 | 0.6119 | 0.6552 |
| 正确关系门控传播 | 0.6721 | 0.6955 | 0.6287 | 0.6654 |
| 打乱关系门控传播 | 0.6780 | 0.6952 | 0.6397 | 0.6710 |

正确关系虽然比不传播高，但仍然低于打乱关系。当前传播层主要读取了关系
的数量和分布，还没有真正利用 patch 对应关系。不能继续靠增加传播层数或
模型容量补救；下一步需要改关系学习目标，例如加入“正确关系和打乱关系
必须可区分”的辅助任务，或直接使用关系重构作为预训练约束。

## 关系辨别辅助任务

最后加入了关系辨别损失：除了图级分类，模型还要根据 patch 表示判断每类
patch 连接是真实的还是打乱的。三折结果为：

| 输入 | fold 0 | fold 1 | fold 2 | 平均 |
|---|---:|---:|---:|---:|
| 不加关系辅助 | 0.6500 | 0.7152 | 0.6098 | 0.6583 |
| 正确关系＋辅助任务 | 0.6589 | 0.6929 | 0.6063 | 0.6527 |
| 打乱关系＋辅助任务 | 0.6500 | 0.7032 | 0.6272 | 0.6602 |

第 0 折出现了正确关系优于打乱关系的现象，但另外两折相反，平均结果也
没有提升。因此当前辅助任务还不能稳定地把“具体 patch 对应关系”转成
MolHIV 分类收益。关系路线的机制证据已经存在，但其下游标签相关性仍然
不足或不稳定。

当前实验代码：

- `tracks/ksvd/code/run_molhiv_chemistry_relation_probe.py`
- `tracks/ksvd/code/run_molhiv_chemistry_relation_folds.py`
- `tracks/ksvd/code/run_molhiv_typed_relation_folds.py`
- `tracks/ksvd/code/run_molhiv_typed_relation_pca_folds.py`
- `tracks/ksvd/code/run_molhiv_channelwise_relation_folds.py`
- `tracks/ksvd/code/run_molhiv_gine_ksvd_sidecar_fold.py`
- `tracks/ksvd/code/run_molhiv_gine_residual_relation_fold.py`
- `tracks/ksvd/code/run_molhiv_patch_relation_message_fold.py`
- `tracks/ksvd/code/run_molhiv_enriched_patch_gated_relation_fold.py`
- `tracks/ksvd/code/run_molhiv_patch_relation_aux_fold.py`
- `tracks/ksvd/results/molhiv/feature_audit_topology_n3000.json`
- `tracks/ksvd/results/molhiv/feature_audit_chemring_n3000.json`
- `tracks/ksvd/results/molhiv/chemistry_relation_wl_chem_ring_n3000.json`
- `tracks/ksvd/results/molhiv/chemistry_relation_wl_n3000.json`
- `tracks/ksvd/results/molhiv/chemistry_relation_official_train_fold0.json`
- `tracks/ksvd/results/molhiv/chemistry_relation_official_train_fold1.json`
- `tracks/ksvd/results/molhiv/chemistry_relation_official_train_fold2.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold0.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold1.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold2.json`
- `tracks/ksvd/results/molhiv/gine_residual_relation_fold0.json`
- `tracks/ksvd/results/molhiv/gine_residual_relation_fold1.json`
- `tracks/ksvd/results/molhiv/gine_residual_relation_fold2.json`
- `tracks/ksvd/results/molhiv/patch_relation_message_fold0.json`
- `tracks/ksvd/results/molhiv/patch_relation_message_fold1.json`
- `tracks/ksvd/results/molhiv/patch_relation_message_fold2.json`
- `tracks/ksvd/results/molhiv/enriched_patch_gated_fold0.json`
- `tracks/ksvd/results/molhiv/enriched_patch_gated_fold1.json`
- `tracks/ksvd/results/molhiv/enriched_patch_gated_fold2.json`
- `tracks/ksvd/results/molhiv/patch_relation_aux_fold0.json`
- `tracks/ksvd/results/molhiv/patch_relation_aux_fold1.json`
- `tracks/ksvd/results/molhiv/patch_relation_aux_fold2.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_official_train_fold0.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_official_train_fold1.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_official_train_fold2.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_pca_official_train_fold0.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_pca_official_train_fold1.json`
- `tracks/ksvd/results/molhiv/chemistry_typed_relation_pca_official_train_fold2.json`
- `tracks/ksvd/results/molhiv/channelwise_relation_row_folds.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold0.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold1.json`
- `tracks/ksvd/results/molhiv/gine_ksvd_sidecar_fold2.json`
