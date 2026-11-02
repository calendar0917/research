# PROTOCOL — LCRC-A / Lipophilicity 模型理解与简化（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/`
Round ID / protocol ID: `lcrc-lipo-model-dissection-v1`
Frozen: 2026-11-02 (Asia/Shanghai)

本文件是本轮**冻结执行协议**：第 0 节为执行映射（文件入口、机器可读配置、
验收检查），其后为研究者下发的原始 Prompt 全文（逐字节保留）。任何科学定义
（数据、划分、模型变体、训练配方、选择/购买规则、预算、报告规则）以原始
Prompt 为准；实现不得自行替换。

## 0. 执行映射（实现索引）

| 协议要素 | 机器可读/代码入口 | 冻结记录 |
|---|---|---|
| 变体定义（18 个） | `configs/variants.json`、`code/dissect_common.py::VARIANT_DEFS` | `analysis/protocol_freeze.json` |
| run matrix（54 + ≤18） | `configs/run_matrix.json` | 同上 |
| 数据源与 3 个 repeat 的字节哈希 | `code/dissect_source.py`、`code/dissect_common.py` | `analysis/protocol_freeze.json` |
| 结构 cache（label-free，可重建） | `code/dissect_data.py`、`code/dissect_prepare.py` | `cache/manifest.json`（gitignored；指纹亦写入每个 run 的 `train.json`） |
| FULL 行为等价重构 + 变体裁剪 | `code/dissect_model.py` | `analysis/audit_cpu.json` |
| CPU 高风险核验 | `code/dissect_audit.py` | `analysis/audit_cpu.json` |
| 单run训练（R3, 9720 updates） | `code/dissect_train.py` | `runs/<repeat>_<variant>_s<seed>/train.json` |
| stage 冻结（仅 Valid） | `code/dissect_freeze.py` | `analysis/freeze/<stage>_r<repeat>.json` |
| Test 一次性评分（EXPOSED） | `code/dissect_score.py` | `runs/.../test_eval.json`、`analysis/test_access_log.json` |
| COMPACT 选择（仅 Valid） | `code/dissect_compact.py` | `analysis/compact_selection.json` |
| 主表/对照/bootstrap/跨repeat | `code/dissect_tables.py` | `analysis/tables/*.csv`、`analysis/bootstrap.json` |
| luyin21 回溯（原子项 + 路径干预） | `code/dissect_attrib.py` | `analysis/attribution/*` |

固定命名：run = `r<repeat>_<variant>_s<seed>`（例：`r122_A0_s0`）。
执行顺序：Stage A (A0..A5) → 冻结+Test → Stage B (B0..B6) → 冻结+Test →
Stage C (C0..C4) → 冻结+Test → COMPACT（仅 Valid）→ 123/124 的 FULL/COMPACT/B5 →
CPU 回溯与报告。最多 2 个 A100 并发，单 run 一张 A100。

关键冻结常量（与代码一致）：

* 数据：ChemBench `Lipop`，commit `0b10d4c1ac72b6929503694a5f60446824736391`，
  CSV SHA256 `34280f0a…`；repeat 122/123/124 的 split 文件与 int64 行索引字节哈希见
  `analysis/protocol_freeze.json`（与 Prompt 第五节逐字节一致）。
* 输入维度：atom_feat=51、bond_feat=13、atom_scalar=9、global=42、S1 code 80、
  path code 400；LocalMLP 529 列布局与 rho 171 列布局见 `dissect_common.py`。
* FULL 可训练参数 = 247106；18 个变体的期望参数量见 `configs/variants.json`。
* 训练：AdamW(lr 3e-4→1e-5 per-step cosine, betas .9/.999, eps 1e-8, wd 1e-3)、
  batch 128、360 epochs × 27 batch = 9720 updates、fp32、clip 5.0、MSE on
  Train-standardised target、best-Validation-RMSE（严格更小，平局取更早）。
* 选择容差（实用购买规则，非统计等价）：mean_s dR ≤ 0.010、max_s dR ≤ 0.020、
  mean_s dA ≤ 0.010（logD）。
* 预算：软 6 A100 GPU-hours，硬 8 A100 GPU-hours，正式 run 上限 72。

## 1. 研究者下发的原始 Prompt（逐字节）

```text
LCRC-A / Lipophilicity 模型理解与简化：冻结执行 Prompt
设计日期：2026-11-02（Asia/Shanghai）
研究仓库：https://github.com/calendar0917/research
设计依据快照：e673a8bd375c64ff281eb813e36e1867a9ecb792

请完整执行本 Prompt。你的角色是实现、验证、运行实验和按规则整理结果。研究问题、数据划分、模型变体、训练配方、对照解释范围和购买规则已经在下文确定。

你不负责文献检索、重新设计实验、提出新架构、选择新数据集或根据 Test 结果决定下一轮。允许在仓库内用 rg 等工具定位实现、读取给定来源及解决工程问题；这不等于授权你开展新的研究检索。遇到需要改变科学定义的歧义，记录阻塞点，不自行替换实验问题。给定范围与预算内的常规实现、训练、评分不需要逐阶段再次询问授权。

一、目标与边界

本轮目标：把当前 LCRC-A 的信息来源、计算路径、各分支的用途和冗余关系做清楚，获得一个经过复核的简化候选。保留性能作为判断工具，本轮不追求排行榜名次，也不复现 PNA/ESA/ChemFM。

具体回答：
Q1 原始全图统计 g 与学习式局部表示 h，分别贡献多少预测价值？
Q2 原子贡献求和头与池化后的非线性读出，是否互补，是否有可替代部分？
Q3 S2 类型联合项与门控末端内容各自起什么作用？收益是否超出各侧边际统计？
Q4 S1 的重复汇总、额外原子标量、二阶池化和闭环通道，哪些值得保留？
Q5 能否准确追踪一个局部统计变化如何经局部项和 pooled readout 改变最终预测？

区分三类活动：
1. 行为等价重构：实现组织变化，模型函数与训练语义不变。
2. 简化实验：允许参数减少，检验删减后重新训练是否保持开发性能。
3. 机制对照：尽量固定参数和计算结构，只改变指定联合信息的组织。
不要用简单删减实验直接宣称信息论必要性，也不要为了参数匹配给每个删减再添加复杂替代网络。

禁止新增注意力、递归消息传递、SSL、其他预训练、辅助损失、蒸馏、ensemble、EMA、SWA、soup、新描述符或超参数搜索。旧实验与冻结来源只读，禁止覆盖或改写其数字。

二、仓库入口与执行环境

遵守仓库当前 AGENT.md / 适用 AGENTS.md。先查看 git status，保留用户修改。按仓库规定使用 uv 和远端执行机制；远端任务必须走 skills/remote-research-runner/SKILL.md 与 rr，不手写 SSH/nohup/sbatch 绕过。最多两个训练任务并发，每任务一张 A100；采用已配置的可用 A100 pool，资源调度属于工程选择。

本轮新目录固定：
tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/
建议结构：code/、configs/、tests/、analysis/、runs/、MODEL.md、PROTOCOL.md、REPORT.md。
大 checkpoint 和可再生 cache 按仓库 gitignore 约定保存；源代码、冻结配置、紧凑指标和说明保留为可复现记录。不要把未授权的 checkpoint 大文件强行提交到 Git。仅在仓库约定要求时更新导航指针，不重写旧报告。

依据文件：
1. docs/luyin/luyin21.txt
2. tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_model.py
3. 同目录 feature_spec.py、lcrc_data.py、molecular_graph.py、adapters.py
4. tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_common.py、mb_source.py、mb_data.py、mb_model.py、mb_train.py
5. tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_model.py、mb2_train.py
6. tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench_source_manifest.json
7. 同轮 analysis/split_manifest_Lipophilicity.json
8. tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2/REPORT.md 与 DECISION.md
9. tracks/structural-sufficiency/lcrc_lipo_best_epoch_distance_fix_v1/REPORT.md
10. tracks/structural-sufficiency/zinc_structural_binding_feasibility_v1/REPORT.md

当前科学参照为 lipo_v1、scratch LCRC-A、R3、360 epochs、best-Validation-epoch。不是带上下文新增模块的 ROLEBLIND/G2L/DIST 版本，也不是带 ZINC SSL 的版本。容量匹配 E 的特殊替代网络不是 FULL 的组成，不引入本轮。

若当前工作树晚于依据快照，只比较上述评分关键文件的变化；记录差异。工程修改可以继续使用，科学定义以此 Prompt 为准。不得重置用户工作树以迁就旧快照。

三、公开固定数据划分：不重新划分

数据：ChemBench Lipop，4200 分子，目标 exp，原始 logD 单位；每个公开 repeat 为 Train3360 / Valid420 / Test420。
来源：https://github.com/shenwanxiang/ChemBench
固定 commit：0b10d4c1ac72b6929503694a5f60446824736391

使用仓库已 vendor 的原文件。当前实际目录：
tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/Lipop/
历史 manifest 内有 data/chembench 的旧路径字符串；以实际文件内容和下列哈希为准，不据旧字符串另选数据。

Lipop.csv.gz SHA256：
34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758

Lipop.random122.ind.pkl SHA256：
bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d
Train / Valid / Test 索引 int64 原顺序字节的 SHA256：
4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96
31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365
a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93

Lipop.random123.ind.pkl SHA256：
afaed4c6d95b64ac64e8fb760b6de516802008965befe1601b3664303119a0e8
Train / Valid / Test 索引 SHA256：
a3af8ce779db41a288e4e82a7d7ce3d07a556ecf8f2e54ef60f43ad480626db4
a79e192e922d95bbfa9ff4be1564405d80a0d5ca4c1421fdb5abef5565170982
6c335c844cb1a53273fb784de389d7df25281df86a057bbe22882f130aa06df2

Lipop.random124.ind.pkl SHA256：
2b48a41ccc39b622bbb74f18331ca9d9513190eec2de29144ce2bc7d47ef7ef3
Train / Valid / Test 索引 SHA256：
dacdee79f1e34d258c55e3bdc6b7b87ecf9bd63039516b64df9f2b2a68cbd887
e2100b1b236d236f7a3542b057f9026ff77f7bb920aeb662f64f924c3991ef94
c4a5b71d8245ab426cffe475cd8862e11b16106d303b3ddb5dc409611fd0427d

122 是所有预定消融的主开发划分；123/124 只用于第九节预定的跨公开划分复核。不能重新 random_split、重新 scaffold split、调换 Valid/Test、重新排序 CSV 后沿用旧行号，或者根据成绩挑 repeat。split 文件名而不是 glob 返回顺序是 repeat 的身份。

来源文件缺失时，只允许从以下固定目录取回相同字节：
https://raw.githubusercontent.com/shenwanxiang/ChemBench/0b10d4c1ac72b6929503694a5f60446824736391/src/chembench/data_and_index/Lipop/
若 hash、行数、行对应无法复原，停止正式训练并报告，不能自行生成替代划分。

保持全部分子与源顺序；不去盐、不去重、不删除困难分子。复用冻结 lipo_v1 原始属性适配与结构缓存语义，不引入 Chem.AddHs 版本或 Chemprop 特征替代。输入维度：atom_feat=51、bond_feat=13、atom_scalar=9、global=42；S1 code 80 类、S2 path code 400 类。模型词表固定，不从 Valid/Test 拟合。

每个 repeat 独立使用该 repeat 的 Train 结构拟合 feature normalisation，使用该 repeat 的 Train 标签拟合 target mean/std（ddof=0）。各变体共用该 repeat 的同一组归一化参数。禁止跨 repeat 混用 norm。

训练接口只暴露 Train/Valid 标签；不把老 loader 返回的含全部 exp 列 dataframe 传入训练器。结构构建可读取全体无标签结构，模型选择不能接触 Test 目标。标签读取器只将许可行的目标转为数值，不能将整列 Test 目标留在训练对象中。

四、公开报告的可比范围：事实已核对，无需 Agent 检索

公开划分的来源有以下依据：
A. MolMapNet 2021 论文的数据可用性直接指向 ChemBench/v0。
https://www.nature.com/articles/s42256-021-00301-6
B. ChemFM 官方代码使用 chembench.load_data(dataset)，依次使用 induces[seed]，lipo 配置 num_data_seeds=3。
核对 commit：f6b508955032a9beaca2036c22f82113102db552
https://github.com/TheLuoFengLab/ChemFM/blob/f6b508955032a9beaca2036c22f82113102db552/finetuning/property_prediction/utils.py
https://github.com/TheLuoFengLab/ChemFM/blob/f6b508955032a9beaca2036c22f82113102db552/finetuning/property_prediction/configs/moleculenet/lipo.yml
C. 该官方 README 的 MoleculeNet results / Lipophilicity / RMSE 行列示：
GraphConv 0.655；Chemprop 0.555；MMNB 0.625；ChemFM-3B 0.545。
https://github.com/TheLuoFengLab/ChemFM/blob/f6b508955032a9beaca2036c22f82113102db552/finetuning/property_prediction/readme.md
D. ChemFM 论文为 https://doi.org/10.1038/s42004-025-01793-8 ，使用外部大规模预训练；不是本轮 scratch LCRC 的同成本对照。

在报告中将 C 明确标为“ChemFM 官方 README 列示的文献参照，未由本轮复现”，不虚构标准差，不称所有行都经过我们独立逐字节验证。ChemFM 未在该安装说明中固定 ChemBench commit，且 load_data 的 repeat 顺序依赖库实现，因此目前证据支持公开划分系列与评价范式相连，不能宣称其历史训练与我们的固定版本逐字节一致。

122 单个 repeat 的数值不能直接当作上述多 repeat 报告的同口径结果。跨 122/123/124 汇总表才具备更接近的比较口径；仍注明 feature、预训练、规模、训练随机性与版本差异，不据这些参考值宣布 SOTA。旧 random42 的 PNA/ESA 成绩只能放背景附表，不能混入本轮主排名。

不下载或训练 ChemFM，不扩展参考论文，不由 Agent 补写一张新的文献排行榜。

五、固定基座、行为等价重构与初始化

FULL 计算：
a_v = AtomEncoder(atom_feat_v)
b_e = BondEncoder([a_u+a_v, abs(a_u-a_v), a_u*a_v, bond_feat_e])
h_v = LayerNorm(LocalMLP([a_v, b_sum_v, b_mean_v, S1joint_v, S2_v, scalar_v]))
训练时 h_v 在上述 LayerNorm 后做一次 p=0.10 dropout，两个读出共享同一个 dropout 后的 h。
L = sum_v f(h_v)
P_g = rho([g, mean(h), mean(h*h), log1p(n)])
y_standardised = L + P_g
y_raw = target_mean + target_std * y_standardised

默认宽度固定：atom_hidden=128、atom_out=96、bond_hidden=160、bond_out=96、stat_rank=32、local_hidden=192、width=64、rho_hidden=64。f 为 64→64→1，rho 为 171→64→1，MLP 非末层 SiLU；LocalContent 末尾为现有 LayerNorm。FULL trainable parameters 必须为 247106。

把分支和 dropout 显式实现于新目录，旧模块只读。保留 checkpoint 的可追踪加载映射；不用反射式 hook 隐藏研究对象的核心行为。新旧数值核对：
1. 同一 state、同一 norm，CPU eval 预测和 aux 最大绝对差 <=1e-6。
2. CPU train 模式恢复相同 RNG 状态后，dropout 后的 h、loss、梯度与一次 optimizer 更新相符（浮点容差 <=1e-6，报告最大差）。
3. 原子重编号、batch 拆分/合并、孤立或空 S2 输入、checkpoint save/reload 通过针对性核对。
4. 复播已有 public122 FULL seed0 checkpoint 的 Valid 指标，误差 <=1e-5；已有记录 Valid RMSE 约0.710880，Test RMSE约0.572747。这里只复播 Valid，Test 记录不用于改动。
5. CUDA eval 的非确定性另外测量和记录；不要求训练轨迹逐位一致，不将 CUDA 重训差异误当重构错误。CPU 等价失败必须先修复，不进入正式训练。

为每个 seed 先构造一份完整 FULL 初始 state。每个变体的共享参数复制同一 state，不允许构造不同模块时消耗 RNG 导致共享初值漂移。裁剪输入维度的 Linear 第一层按固定列映射复制 FULL 权重和 bias，后层复制；删除不用的学习模块与列，报告实际 trainable 参数。新增标量 b0 初始化为0。不用不可达参数凑预算。

每个 run 的数据 shuffle generator 固定为 seed+917000，同一 seed、同一 repeat 的批次索引顺序一致。分支减少使 dropout 后续 RNG 消耗不同属于计算差异；记录而非声称所有随机操作路径完全相同。

LocalMLP 原始529列布局，零基、左闭右开：
a 0:96；b_sum 96:192；b_mean 192:288；S1joint 288:320；
S2_path_type 320:352；S2_end_type 352:384；S2_joint 384:416；
S2_closure 416:424；S2_content 424:520；atom_scalar 520:529。
rho 171列布局：
g 0:42；mean_h 42:106；mean_h2 106:170；log1p(n) 170:171。

六、冻结训练与 Test 报告规则

所有正式 run：
训练 seeds=[0,1,2]，与 split suffix122/123/124分开命名。
AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-3。
peak lr=3e-4，min lr=1e-5，复用 mb_train/mb2_train 的 R3 per-step cosine 语义；不新增 warmup。
batch=128，360 epochs，每 epoch 覆盖 Train3360 一次（27个batch），共9720个 optimizer updates。
fp32，clip_grad_norm=5.0，MSE on Train-standardised target。
每 epoch 完整评估 Valid420 原始单位 RMSE。严格更小才保存 best；相同值保留更早 epoch。
不 early stop、不用 Test 选 epoch、不用 soup/EMA/SWA、不挑 seed、不做预测 ensemble。
G_ONLY 没有 h，因此不额外加入 dropout；其余保留 h 的变体仍在同一位置使用 p=.10。所有 head 内不新增 dropout。

每个完整阶段的所有预定 run 训练结束后：
1. 冻结变体定义、配置、代码 hash、checkpoint hash、best epoch、Valid值及行索引。
2. 从 best checkpoint 重新推理 Valid，核对 curve argmin 与 reload 的指标 <=1e-5。
3. 再评分该阶段所有 frozen checkpoint 的 Test420，一模型/seed/repeat仅评分一次，并报告 Train/Valid/Test。
4. 评分失败可用已保存预测修复报告，不通过重读 Test 或重新选 epoch“修复成绩”。记录真实访问情况，不为追求“只读一次”隐藏实际重读。

用户明确要求每次 valid选ep、报告test，本轮按上述规则执行。所有 main122 消融与阶段均预先固定，后续不能根据已看到的 Test 分数删增模型、改配方或选简化候选。
三个公开 Test 已在历史工作中暴露，本轮全部 Test 结果标为 EXPOSED_TEST_EXPLORATORY。重复公开划分也不制造新盲测。可以报告，但不得写成独立确认或无选择偏差的最终泛化估计。
不要重新引入旧轮“gate未过不读Test”的限制覆盖本 Prompt；这里已明确授权为每个预定模型作描述性 Test 报告。

七、正式实验矩阵：顺序固定，全部先在 public122

Stage A：大分支与读出，6个模型×3 seeds=18 runs。

A0 FULL：上述完整模型。
A1 G_ONLY：仅标准化后的42维g→64(SiLU)→1，不实例化局部编码与f；其head对应复制 FULL rho 第一层的g列与完整bias、第二层。预期2817参数。
A2 NO_G：L + rho_h([mean_h,mean_h2,log1p(n)])，rho_h=129→64→1；局部编码、f保留。预期244418参数。
A3 NO_ADD：rho([g,mean_h,mean_h2,log1p(n)])，保留完整局部编码但去掉f。预期242881参数。
A4 POOL_ONLY：rho_h([mean_h,mean_h2,log1p(n)])，无原始g、无f。预期240193参数。
A5 ADD_ONLY：L+b0，保留完整局部编码与f，去掉rho；b0为一个可学习全图常数偏置、初值0。预期236034参数。

A0/A2/A3/A4组成“有无原始g × 有无原子求和头”的四格；不能只比较同时去掉g和f的两个端点，就把效果归因给f。

Stage B：S2内部，7个新模型×3 seeds=21 runs。全部以 FULL 读出和其余结构为载体，不能换成Stage A赢家再做，以免更换机制背景。

B0 TYPED_S2_OFF：
删除整个200维typed-S2块及其学习模块，LocalMLP从529输入裁到329；保留9个atom_scalar及原始g。预期176514参数。
注意9个scalar仍含两跳计数和closure_ratio，故名字只能叫typed-S2-off，不能说“完全不含两跳信息”。

B1 NO_JOINT：
仅删除32维S2_joint及path_u/path_v/joint_proj；其余S2保留，尤其保留path_gate*a_end。预期224546参数。

B2 NO_CONTENT：
仅删除96维S2_content及path_gate；S2_joint与其余S2保留。预期228274参数。

B3 NO_JOINT_CONTENT：
同时删除B1与B2指定分支，其余S2保留。预期205714参数。

B4 MARGINAL_ALL：
保留FULL所有学习模块和参数维度，只替换两个联合聚合；在每个根原子v内部计算。
记P_v为原有非回溯两跳路径occurrence多重集，n=|P_v|；
u_p=path_u(path_cat_p)，w_p=path_v(path_end63_p)，t_p=path_gate(path_cat_p)，z_p=a[path_end_p]。
FULL的联合和内容聚合分别为：
J_v=sum_p joint_proj(u_p*w_p)
C_v=sum_p t_p*z_p。
替换为：
J_v=joint_proj.weight @ (n*mean_p(u_p)*mean_p(w_p)) + n*joint_proj.bias
C_v=n*mean_p(t_p)*mean_p(z_p)。
乘积逐通道，原path_type、end_type、closure的聚合完全不变。n=0时J=C精确为零；n=1时与FULL相同。不能把joint_proj的bias只加一次。不能跨根、跨分子计算均值。

B5 MARGINAL_BT2：
同B4，但先按“根原子v，第二条键类型bt2”分组，在每组内部应用上述边际乘积，再对组求和。
bt2=path_cat % 5，并核对bt2==path_end63 % 5；这是lipo_v1实际零基编码。
此控制保留第二条键型的相容关系，进一步隔离其内部的类型/末端内容配对。
B4/B5均预期247106参数；没有新的投影或学习参数。
它们是代数联合统计对照，不是生成新的合法分子，也不自动排除全部价态、度数、环等简单相容性解释。禁止称其为“化学因果反事实”或据其差距宣称发现非平凡化学规律。
记录B5可实际改变的根/分子比例、组大小与summary差异。若有效变化很少，结果必须连同干预覆盖率报告，不把null当作绑定无效。

B6 NO_CLOSURE_CHANNELS：
删除8维close_emb块及atom_scalar里的closure_ratio（原scalar第2列）；其他atom/ring/g字段保留。预期245362参数。
这是显式closure通道的条件性消融，不能称去掉了图中全部环/三角形信息。

Stage C：其他可疑冗余，5个新模型×3 seeds=15 runs。仍以FULL其余结构为载体。

C0 NO_S1JOINT：删除32维S1joint及nbr_emb，不读取nbr_code/nbr_atom。保留b_sum和b_mean。预期238402参数。
C1 NO_BOND_MEAN：删除96维b_mean块，保留b_sum和其余输入。预期228674参数。degree仍可由原始属性和scalar获知；此对照涉及参数化/学习偏置，不声称移除一种独立信息。
C2 NO_SCALARS：删除全部9维atom_scalar输入，其他原始atom_feat与g保留。预期245378参数。不能称“没有环/度数/两跳信息”。
C3 NO_M2：仅从rho输入删除64维mean_h2，保留g、mean_h、log1p(n)、f与局部编码。预期243010参数。
C4 MEAN_ADD：FULL中的L由sum_v f(h_v)改成mean_v f(h_v)，rho及其所有输入不变。预期247106参数。检验原子求和的读出偏置，不先验声称logD应服从某种可加化学定律。

按A0..A5、B0..B6、C0..C4的顺序执行，每个stage内部按seed0、1、2循环。上述18种配置是本轮完整定义，不因Test成绩改变。全量public122为54个正式runs。A0也正式重训，旧checkpoint仅用于等价复播，不代替本轮共用初始化/批次约定的参照。

八、实现前的必要核验，不制造庞大审计框架

把本Prompt原样转成冻结PROTOCOL与机器可读run matrix，不重新写研究方案。针对以下高风险点做最小有意义验证：
1. 数据源/三个split hash、索引集合大小和互斥、cache位置/CSV row/标签一一对应、norm只用Train。
2. FULL行为等价、参数数目、dropout确实调用一次且作用于两个读出的共同h。
3. 删除模块的输入列映射正确，被删通道不能经原始同名tensor意外读回；共享参数初值与shuffle索引一致。
4. B4/B5的独立Python/float64小例子，n=0、n=1、重复occurrence、不同bt2、batch offset、bias计数；保留边际与计数，FULL与边际control在非退化样例确实不同。
5. 各保留学习分支有梯度；B4/B5不通过detach或缓存固定embedding偷换端到端训练。
6. best epoch保存/重载/曲线argmin正确、原始单位反标准化正确、训练入口不返回Test目标。

先做CPU核验，再至多2 epoch的CUDA工程smoke与时间估计。smoke不参与科研结论。若发现实现偏离上述定义，修工程bug；修复评分关键代码后，不混用修复前的受影响结果。与定义冲突不能由Agent发明新臂解决。

九、简化候选与跨公开划分复核：规则已固定

本轮不进行自由组合剪枝，也不把若干单独可删分支直接同时删掉。单删可替代性不保证联删保持性能。

在54个public122 run完成后，只从Valid表选择一个COMPACT；选择脚本的输入不能包含Test列。

对每个variant及同seed FULL，计算原始单位：
dR_s=ValidRMSE(variant,s)-ValidRMSE(FULL,s)
dA_s=ValidMAE(variant,s)-ValidMAE(FULL,s)。

可进入COMPACT候选的固定条件：
a. 实际trainable参数严格少于FULL；
b. mean_s(dR_s)<=0.010；
c. max_s(dR_s)<=0.020；
d. mean_s(dA_s)<=0.010。
这些0.01/0.02 logD只是本轮允许用来购买简化复核的实用容差，不是统计等价证据、普适阈值或投稿标准；报告中必须如此表述。
合格候选先按参数数目由少到多，再按平均Valid RMSE由小到大，再按本Prompt的variant顺序选一个。若无合格候选，COMPACT=FULL，不设计救援模型。

然后固定在公开123、124上运行：
FULL、已选COMPACT、B5 MARGINAL_BT2，各seeds0/1/2。
相同variant去重，不重复训练。最多新增18个runs，全轮最多72个正式训练。
每个repeat重新拟合Train norm，从头初始化训练；不可将122训练权重迁移过去。仍各用自身Valid选epoch，冻结后报告自身Test。
每个跨划分阶段的模型清单在评分其Test前冻结。完成后不重新挑COMPACT。

简化复核标记：
若COMPACT!=FULL，且123、124各自都满足上面b/c/d（各与自身FULL配对），标记VALIDATION_TOLERANCE_REPLICATED。
否则标记SPLIT_DEPENDENT_OR_UNCONFIRMED_SIMPLIFICATION；保留全部结果，不用Test重新选择第二候选。
如果COMPACT=FULL，标记NO_SIMPLER_CANDIDATE_SELECTED。
Test有方向翻转必须报告；它不触发改模型、换候选或增加seed。

B5跨repeat结果用于检查一个预先指定的联合统计对照是否稳定，不根据其122方向决定是否执行。

十、分析：按定义算结果，不代替研究者作结论

主表每行至少：
repeat、variant、seed、nominal/trainable参数、9720 updates、best epoch、Train@best/Valid/Test RMSE、MAE、R2、训练/推理时间、GPU时长、checkpoint与预测路径。
均值表必须是每seed指标的平均，不将seed预测平均后冒充单模型。报告ddof=1的训练seed SD、每seed配对差、min/max。样本抽样不确定性与训练随机性分开。

预定义对比：
A四格：NO_G-FULL、POOL_ONLY-NO_ADD、NO_ADD-FULL、POOL_ONLY-NO_G，以及这两组差值之差；这是读出/输入设计的条件性差异，不是唯一物理因果贡献。
B四格：NO_JOINT-FULL、NO_JOINT_CONTENT-NO_CONTENT、NO_CONTENT-FULL、NO_JOINT_CONTENT-NO_JOINT，以及交互差。
B4/B5-FULL、TYPED_S2_OFF-FULL、NO_CLOSURE_CHANNELS-FULL。
所有C变体-FULL。统一“候选减参照”，负值表示候选误差更低。

评估样本区间：
按固定Bemis–Murcko scaffold分组做2000次paired bootstrap（seed=20261102）；空scaffold使用canonical SMILES作为组key。每次重采样组，同一次抽样用于所有variant和seed，保留分子多重性，计算配对RMSE/MAE差。报告的是固定训练seed集合下的评估样本区间，不能把它说成训练随机性CI。各seed差值与SD另列。分组不改变公开train/valid/test划分。
不做几十个p值后挑显著故事，不把区间跨零当作证明等价。

跨3个公开repeat的表：
逐repeat先列3个训练seed的均值/SD；另外列每个seed跨3repeat的平均，以及总平均。公开repeat共享原始分子，不能把9个cell当9个独立数据集来算置信度。
文献参照表与本轮复现表分开；不输出“击败ChemFM”的结论。

模型依赖检查（CPU eval为主）：
在122 Valid固定前64个row上，对FULL三个seed进行固定权重干预，记录mean/max abs(delta prediction)、RMSE变化、干预coverage和输入/activation norm变化：
g标准化后设零；
sum贡献项设零；
mean_h2输入rho设零；
typed-S2设零（raw scalar和g固定）；
FULL的联合/内容聚合切换为B5定义。
这些是已训练函数依赖诊断，可能存在分布变化，不能独立证明泛化收益或化学机制。
仅记录contrib/rho的均值、标准差与相关；禁止凭两项绝对幅度大小判断重要性，因为加性分解可重新分配偏置。

与尺寸有关的诊断固定为：用122 Train原子数的25/50/75分位点分组，报告各组完整误差和配对差；少样本组标注样本数，不据其改主指标或选择候选。不要扩展化学描述符搜索或自动挖掘新的困难子集。

十一、luyin21所需的预测回溯

用122 FULL seed0的Valid绝对误差排序（同误差按CSV row id升序），固定取排名：
0、1、209、210、416、417、418、419，共8个例子。
这只是开发集上的固定展示规则，不据Test挑好看的分子。

对FULL三个seed和COMPACT（若含局部项），保存：
row id、SMILES、真实exp、预测、原子索引与元素、每原子f(h)、L、rho、target_mean。
必须换算回原始单位：
c_v_raw=target_std*f(h_v)，r_raw=target_std*rho，单独列target_mean；
核对target_mean+sum(c_v_raw)+r_raw==prediction_raw。
ADD_ONLY的b0属于全图常数；无局部求和头的模型contrib明确记0，不伪造原子贡献图。

在FULL seed0上，针对这8个例子的每个原有S2路径occurrence，进行精确leave-one-occurrence-out：
只从该根原子的typed-S2输入中移去该条record；原图、S1、raw scalars、g保持不变，模型eval。
这是S2输入通道干预，不是删除原图上的化学键，也不是合法分子编辑；不得用真实分子编辑的语言描述。
记录该路径的root/middle/end原子与两条键型、删除前后根h、根f、其他原子f、rho与最终预测差。
检查这次干预只改变对应根的局部项，其他原子局部项保持数值容差内不变；全图rho可以随mean_h/mean_h2改变。
验证delta prediction = delta local-sum + delta rho（原始单位）。
给出按abs(delta prediction)排序的路径表及每个例子的图。路径间有非线性交互，leave-one-out差不能相加后称“唯一的路径贡献分解”。

图明确标“模型内部预测回溯”。真实标签只有分子级，不能宣称已把真实误差唯一分配给某个原子/路径，或已经发现化学因果规律。若COMPACT去掉原子求和头，如实说明本轮未支持保留该解释接口。

十二、解释规则与研究者判断权

生成一张问题—干预—观测—限制表，而不是让Agent编造机制故事：
固定权重敏感：报告当前函数依赖该输入/分支。
删减后重训保持表现：报告在此协议下存在替代方案/简化候选，不报告原分支从未被用过。
删减后变差：报告该删减实现的条件性收益，注明容量、训练拟合与优化路径是否变化。
FULL稳定优于B5：报告在该对照下联合统计有增量预测价值；不能扩大成全图信息充分、递归MP无用或非平凡化学机制已成立。
B5几乎不改变输入或样本覆盖不足：报告该干预对该数据的检验强度不足。
两分支单删不差、合删变差：报告可替代/交互线索，不能据单删结果同时删除。
控制差异落在seed或样本不确定性内：报告UNRESOLVED，不自动买更多seed。

可以机械标记“3/3 seed同方向”“paired区间是否跨0”“是否通过简化购买规则”，不能据此替用户决定论文主张、新颖性、课题继续/放弃，或提出下一轮模型。

luyin21的本轮对应：
显式统计：在MODEL.md逐项给出输入、公式、统计范围、学习参数、输出与去向；说明环统计来自图分析，不能把所有“原子归属特征”都说成严格radius-2信息。
无递归传播：根据真实forward说明一次编码及固定结构汇总，不能仅凭模块名字证明。
回溯：完成第十一节的局部项与rho响应分解。
注意力：导师允许考虑，不是本轮必须添加的模块。
强方法阅读与成本比较：本轮只使用第四节已给定事实和本轮自身成本；不声称本轮消融已完成全面外部方法验证。

十三、预算、故障和停止

软预算6 A100 GPU-hours，硬上限8 A100 GPU-hours。GPU过程的完整wall时间计费，包含smoke、失败、重试、GPU推理；双卡时分别累加。最多两个并发run。CPU-only记CPU时间，不冒充GPU。
执行优先级：等价核验→Stage A→Stage B→Stage C→COMPACT选择→123/124的FULL/COMPACT→123/124的B5→CPU回溯与报告。
全部正式定义最多72个训练。先用smoke估计每run成本并做排程。每启动新run前核对累计成本和预计成本；预计超过硬上限就停止新训练，输出完整已完成结果和未执行清单，不缩短epoch、不删seed冒充完整矩阵、不自动扩预算。
软预算不是提前停止已允许固定实验的科学gate；到6h报告剩余计划，在硬上限内继续。

数据hash/行对齐、CPU等价、finite loss/gradient、checkpoint选择核验失败时，暂停受影响实验，修实现，再继续预定定义；修复影响哪些run就重做哪些，并计入预算。单个训练故障至多一次同配置重试；再次失败记录为缺失，不擅自用另一配置补位。
既有结果矛盾时写出具体矛盾，不偷偷改阈值、旧报告或实验定义。

十四、交付与结束条件

只保留必要文件，避免每个小检查生成一套冗长文档：
1. MODEL.md：实际计算图、每个模块/字段的含义与来源、学习/预计算边界、训练技巧、参数量、当前能支持的回溯范围。
2. PROTOCOL.md及机器可读配置：本Prompt、source/split hashes、变体定义、训练与选择规则、预算，正式run前冻结。
3. 干净的模型/训练/评分实现及针对高风险点的测试。
4. 每run config、代码版本、curve、best checkpoint、Train/Valid/Test预测、指标、时间与异常记录。
5. 122完整表、A/B四格对照表、C消融表、bootstrap区间与seed差、跨公开repeat表、COMPACT选择/复核JSON、GPU账本。
6. 8个Valid案例的原子项、rho、路径干预表与清晰图。
7. REPORT.md：先回答Q1-Q5，分开观测与解释范围，明确未完成项和Test已曝光状态；展示所有预定模型，不能只展示赢的模型。以中文解释结果，代码/数字可用英文标识。
8. 可复现运行命令及精确入口；若预算停止，清楚说明下次从哪个未完成run继续，不重跑健康结果。

本轮结束时不给新的架构/数据集/损失建议，不开展文献检索，不自动发起下一轮。把实现和实验结果交还研究者，由研究者讨论哪些模块保留、如何解释、是否值得回到外部性能比较。
```

## 2. 实现补充说明（不改变科学定义）

1. `DissectLCRC` 以显式分支重建冻结计算图；FULL（A0）在 CPU 上与
   `lcrc_model.LCRC` + 冻结 dropout 位置**逐位一致**（初值、预测、aux、dropout
   后的 h、loss、梯度、一次 AdamW 更新均为 0 差异），并复播 public122 FULL
   seed0 checkpoint 的 Valid RMSE。
2. 每个 seed 先构造一份完整 FULL 初始 state（CPU RNG，`torch.manual_seed(seed)`），
   所有变体从同一 state 按固定列映射复制共享参数；新标量 `b0` 初始化为 0；
   无不可达参数。共享参数写入后在每个 run 记录中逐项核对。
3. 训练入口只构造 Train/Valid 视图；Test 结构 cache 只在 stage 冻结之后的
   `dissect_score.py` 中构建，Test 标签只在该脚本中解码，并记录访问日志。
4. Test 评分在 CPU 上从冻结 checkpoint 确定性执行；CUDA eval 的非确定性只在
   smoke 中测量并记录（不参与科学结论）。
5. 所有正式结果标注 `EXPOSED_TEST_EXPLORATORY`；不据此改模型、换候选或加 seed。
