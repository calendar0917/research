# ZINC 任务字典统一扩容的执行任务

请从 `calendar0917/research` 当前 main 继续，完成一个统一 Small 与 Full 的任务字典模型，并在本地 CPU 上只做一次 Full 的完整绝对性能筛查。请先读取本文件和同包方案、NumPy 参考、审计 JSON，减少重新探索。

## 目标与执行范围

本轮检验的是：在同一局部对象、同一字典求码规则、同一静态关系形式下，放宽有效环境与关系容量，是否能把当前约 0.121 的 valid MAE 降到值得进一步投入的区间。不要先做 dense/random/粗组对照或 seed 1，不重跑已有实验，不购买多个宽度，不使用 ensemble。

约束是单一模型、共享字典为主局部表示、无消息传递、无 Transformer 或 attention、无节点/边隐状态递归更新、无 pair→node 写回。允许从原始图构建现有静态局部对象，以及在单个对象内部迭代求码。全程 CPU，8 线程；不调用服务器，不使用任何 GPU。若可用，阅读 `remote-research-runner` skill，采用它的本地环境、数据与 provenance 检查部分；本任务的 CPU 指令覆盖远端运行默认流程。

交给你执行本文件即授权这一项新筛查和必要的实现、验收、记录工作。旧轮次已经关闭，保留旧结论；不要把旧轮次的停止状态当成新方案需要重复询问的理由。按仓库已有规则处理提交与发布，未获发布授权时交付本地提交及可导入包。

## 只读这些入口再开始

已核对 revision 为 `00c203624682a9ca46d54a225721e25015d8b28c`。先检查当前 HEAD、工作区和 AGENTS.md；若 HEAD 有新变化，仅查看与本任务相关的 diff，不回滚用户工作。

| 路径前缀 tracks/ksvd | 需要读取的内容 |
|---|---|
| experiments/luyin16/e2e_dictenv_latent_bridge_v1.py | LatentDictionaryBridge、LatentBridgeSEM108、builder、参数审计 |
| experiments/luyin16/zinc_e2e_dictenv_latent_bridge_v1.py | 数据/缓存准备、优化器、train/soup、端点 train eval、控制面 stage 映射 |
| notes/zinc_e2e_dictenv_latent_bridge_v1_analysis.md | 最新结果与已知边界，避免重跑 |
| experiments/luyin16/e2e_dictenv_sem108_v1.py | SEM108Model、semantic_interface、_environment_from_parts、environments_masked |
| experiments/luyin16/e2e_dictenv_h1_clarity_audit.py | AuditModel.forward、动态池化、静态 readout block 索引 |
| experiments/luyin16/e2e_dictenv_p2_abs.py | _mlp、decoder_layers、继承来的 backend 构造 |
| experiments/luyin16/zinc_patch_path_pooling.py | 只定位 _MLPBlock；不要通读此大文件 |
| experiments/luyin16/zinc_graph_head_function_family.py | 只定位 GenericReader |
| runners/zinc_e2e_dictenv_latent_bridge_v1.py 与 configs/luyin16/zinc_e2e_dictenv_latent_bridge_v1.yaml | 复制最小控制面与配置形式，检查 scratch/screen stage 映射 |

其他依赖只在必要时用 rg 定位：common_subspace_dictionary 的 33 维结构坐标、clean_mechanism 的 C6_MASK、v0 的 pool_moments/pool_pair_moments，以及 registry 的现有注册入口。不要重新进行文献搜索、历史全仓审计、字典容量扫描或 K-SVD 拟合。

背景数字：现有 bridge 为 106925 参数，valid soup 0.121058，eval-mode soup train 0.055133。它的参数是 body 97709 + D/V 9216；码平均 89.1/96 个非零，不能声称稀疏。Sem108 0.123705 是无匹配背景，不能充当本轮因果对照。其他 Agent 的历史 RPD 代码不存在，禁止尝试重建整个旧实验分支。

## 一个宽度规格驱动同一模型

建议新建 `e2e_dictenv_scale_v1.py` 核心与 `zinc_e2e_dictenv_scale_v1.py` stage 模块，以及对应控制面、配置、测试、预注册文件。命名可适配仓库惯例，最终只保留一个清晰入口。不要复制出两份独立模型算法。

定义不可变 `ScaleSpec`。公开 presets 只有 Small 的 m=1 和 Full 的 m=3；m=2 仅允许零训练的维度审计，不加入训练候选。

```text
d = 48*m                 # fusion 输出与字典解码输出
K = 2*d                  # D[d,K]、V[K,d]
p = 16*m                 # 静态关系端点与 pair 输出
fusion = 446 -> 114*m -> d，SiLU，保持原层数
pair_projection = Linear(d,p,bias=False)
relation_encoder = _MLPBlock(15,32*m,p,dropout=0.05)
distance_gate = Embedding(5,p)
pair_encoder = _MLPBlock(4*p,64*m,p,dropout=0.05)
reader_input = (2*d+1) + 5*(2*p+1) + 32 + 8
reader = GenericReader(reader_input,(13*m,13*m))
```

以下接口和模块保持现有规格，参数继续接受原训练协议的梯度：Sem108+size2 的 110 维原始语义接口；CSSD-q1 的 33 维结构坐标与 K32/s8/IHT10；node binding 宽度 96；edge binding 宽度 48；node slot encoder 96→64→48；edge slot encoder 48→48→32；3 个 node shell 与 6 个 edge shell-pair。因此 fusion 输入始终是 110+3×48+6×32=446。

global_encoder 始终 62→32→32，topology_encoder 始终 25→16→8。不要在 Full 中套用 `max(d//2,32)` 改大 global 分支。上述 `_MLPBlock` 有 LayerNorm 的 gamma/beta 和最终 ReLU，不能简化成两层裸 Linear；参数账包含它们。

预期参数量 Small=106925、Full=408651，其中 Full 的 D/V=82944、其余 body=325707；reader 输入分别 302/814。字典、node/edge binding 与 slot 编码器在新旧模型中的固定规格分别为 2080、5856、4944、9328、3920 参数。以实际 numel 为最终验收，逐模块列出总量与 trainable 数量，任何差异先定位，不能修改预期数来掩盖错误。

扩容覆盖主任务路径，既不增加新的图对象，也不增加图内传播步数。所有 unary/pair 路径使用同一个 E，没有新 h 残差或额外 raw 输入旁路。保持既有图级辅助通道及 C6 mask。

## 实现时最容易踩的地方

优先用最小继承实现。m=1 直接走现有构造、同名 state_dict 和 RNG 顺序，不能先随机构造一套新模块再替换回去。m=3 可先构造现有 Small 骨架，再在单独、已记录的初始化随机流中替换 fusion、bridge、pair projection、relation encoder、distance gate、pair encoder 和 reader；未替换的 stem 与图辅助模块初态保持一致。Full 从头训练，不加载 Small 的训练检查点。

当前 `LatentDictionaryBridge` 本身支持任意 d 且 K=2d，但 `LatentBridgeSEM108.__init__` 把 d 限为 48。不要猴子补丁 `p2.ENV_DIM` 或 `PAIR_HIDDEN` 等模块级全局变量。新的类可以按父构造先建 48 维 bridge，再在 m>1 的分支替换为 d 维模块，所有尺寸由实例规格导出。

桥的唯一插入位置是 `_environment_from_parts`。plain 的 `environments` 与训练用的 `environments_masked` 都经此处：仅覆写 environments 会漏掉正式 C6 训练路径。两个路径均应只调用一次 bridge，unary 与 pair 都读同一个结果。

池化是 sum 与 sum-of-squares，不是 mean/std。C6 会将 unary/pair 的 count 置零，但仍保留相应占位维度。父 `pool_moments_masked` 和 `pool_pair_moments_masked` 主要按实际张量宽度计算；需要特别处理 `UNARY_BLOCKS`、`PAIR_BLOCKS`、`PAIR_BLOCK_DIM`、`apply_readout_permutation` 和某些 fill-policy reshape 中的 48/16 静态索引。新模型的诊断必须使用 d/p 推导的块布局。正式 forward 的 mask、距离桶、全图字段分组与原版一致。

旧 `bridge_parameter_audit`、Sem108 的参数匹配规则和 P2 的 130k 上限均针对旧轮次，不适用于 Full。新审计使用 ScaleSpec 推导的预期尺寸、固定模块一致性与 Full≤420k 的预算，不能绕过正确性门或继续输出“Sem108 body unchanged”。

## 保持字典求码与损失不变

直接复用现有 16 步 unrolled ISTA：row RMS 归一化，列归一化 D，保守 step `1/(1.05*eigmax(Dbar Dbar.T)+lambda2)`，step 估计 detach；lambda1=0.05、lambda2=0.01；alpha 从零开始；E=rho*(alpha@V)。D 与 V 都接受性质损失梯度，初始 D=[I_d,Q_d]、V=D.T，Q 用私有 NumPy RNG，seed 0。不要把结构字典 D 与任务字典 D_L/V_L 混用。

保留现有 MAE + 原结构重建项及其既定权重。本轮不增加低维重建损失、硬 top-k、温度、字典正交罚、dropout/weight-decay 搜索，也不更换学习率调度。Adam、lr、weight decay、gradient clip、shuffle offsets 从现有 bridge/p2run 实际引用，正式预注册将展开后的数值写清，不依赖默认值猜测。

## 开训前的真实数据验收

先读取同包 `dictionary_scale_audit.json`；它是合成参考证据，不能替代以下 PyTorch 检查。参考脚本 `embed_small` 给出 eval 模式下的函数包含构造，可移植为测试工具；正式训练禁止使用该复制初始化。

1. 同 seed 的新 Small 与原 LatentBridgeSEM108：参数名称、形状、初值相同；在真实 train batch 的 plain 与 C6 masked 前向最大差≤1e-5；参数总数恰好106925。此项不重训 Small。
2. Full 实际参数总数408651；中间 h/E/alpha/u/pair/reader 输入宽度为144/144/288/48/48/814；所有固定模块的初值和缓存字段与 Small 相同。
3. 将 Small 参数嵌入 Full，在 eval 模式下两条前向的预测最大差≤1e-4；只作为函数包含验收。D/V 用 block diagonal，fusion、pair、LayerNorm 和 reader 的宽度复制按参考输入映射处理，保留每个 count 的单独位置。若移植此工具耗时，先定位误差，不能将未通过写成已通过。
4. Full 正式独立初始化时，lambda1=lambda2=0 的 bridge E≈h，相对 L2≤1e-6；正式 lambda 下相对平方误差<0.05。重复求码应有限、无 NaN。
5. 真实 train batch 的 MAE-only backward：D_L/V_L、fusion 两层、pair projection 和 pair encoder 有有限且非零梯度；一个合理的小更新能移动参数。旧 node binding 已有死亡记录，不能用其终点活性作为新 Full 开训的必要条件。
6. plain 与 masked 两条环境路径都恰好一次 bridge；无原 h 旁路；zero-code 时 E 为零；端点交换不变、图内重标号、独立图拼接/索引偏移通过。沿用 float32 重标号容差1e-4，不临场放宽。
7. 禁止读取 test，缓存/数据准备只能显式 train/valid，不调用会自动处理所有 split 的流程。测试 test blocker，记录 split 与缓存 fingerprint。

只写对这些风险有用的针对性测试。不扩大到历史模块全仓回归，除仓库规定的必要检查。先解决真实错误，再冻结代码与预注册；不要带着 gate 失败开始正式训练。

## CPU 计时与预算

工程、真实 batch 验收和 smoke 合计目标≤60分钟；正式320轮训练与例行评估上限4小时。无需重做上一轮40轮双模型短训练。

以固定 seed 的 official train 子集做至多一个短 smoke：3 epoch、1024 个分子、不看 valid，用于前反向正确性及计时。额外采样固定索引的大分子 batch 做 RSS 检查，不用标签或 valid 选模型。记录冷启动与稳态时间、forward/backward/eigen 分项，确认确实 CPU/8线程且没有其他任务争用。

真实预算预测须覆盖完整 epoch 的 batch 数和按原频率做的 valid 评估；给出计时方法与保守余量，不能用参数量乘法推断时间。宽度3的 ISTA 主要矩阵计算约按宽度平方增长，运行时间不是必然3倍。若预计超过4小时，交付完成的实现与计时结果，不启动正式运行，不自动改成m=2、减少求码步数或截断epoch。资源未满足时明确写“未进行性能筛查”，不是算法失败。

## 一次正式筛查

仅 Full，seed0，official train全部10k/valid1k，320 epoch，从头初始化，沿用现有选模频率与 Top-5 参数 soup。smoke 不与正式训练共享已更新权重。先提交/记录冻结 revision、配置、输入/缓存 fingerprint 和新预注册，再启动。

保留每轮训练曲线、valid曲线、lr、时间、字典基本统计；诊断频率限制在1/80/160/240/320，避免仪表盘拖慢CPU。不得因中间 MAE 不好而换超参数或停掉固定 horizon；仅数值错误和实际资源异常可终止，终止记录不能当完整负结果。

端点用同一 eval mode、完整数据、不 shuffle，重算 soup train 与 valid，报告 gap、best、Top-5成员、参数量、RSS、总耗时。参数 soup 的 buffer 处理沿用已修复规则；不要平均不兼容状态。

在冻结 soup 上测 D/V 的位移与 MAE-only 梯度、码非零率/有效原子、zero-code 和一次固定 seed 的分子内环境置换所产生的预测 RMS/MAE 变化。按读出块推导宽度。机制结果仅判断学习与使用，不以大消融代价声称性能优势，也不从高非零率主张稀疏性。

建议资源决策分档，正式开训前冻结：

| Full soup valid MAE | 解释与后续 |
|---|---|
| ≤0.110 | 较强单seed绝对性能信号，记录为候选，不自动加跑 |
| 0.110–0.115 | 有希望，值得下一轮配对种子与训练配方验证 |
| 0.115–0.120 | 有限信号，本轮关闭，不继续扩宽 |
| >0.120 | 没有进入明显更好的区间，关闭本次容量路线 |

分档是预算门，不是显著性检验。已有 Small 可作绝对数值背景，不能从单seed两个不同模型推出因果结论。不要复活旧失败实验，也不把本轮未达到门限解释成所有静态字典必然无效。

## 收口交付

交付统一模型、针对性测试、新控制面/配置、预注册、参数分项、验收与 smoke/预算记录、正式曲线与checkpoint引用、分析与run record。正式运行完成后按仓库惯例 promote、verify、claim/decision/STATE 与复现命令收口；不要用旧 protocol id 覆盖历史结果。

仓库已忽略的大checkpoint和cache不强行入git，写明本地保存位置与哈希；若运行环境支持用户持久化产物，保存可重放的 soup 和配置，不声称未来 Agent 能从git取到忽略文件。只有进入预算门并跑满320轮才汇报完整性能结果。

最终中文报告优先回答：Full是否进入≤0.115区间；训练和验证是否同时改善；有效容量有没有实际增加且被使用；CPU代价多少。执行范围到这一个候选为止，下一轮由用户根据结果决定。
