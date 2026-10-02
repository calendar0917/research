# 执行任务：先查性质信号，再跑一个共享字典的静态带类型环模型

你正在继续 `calendar0917/research` 的 ZINC 字典路线。请执行本任务，先做廉价的 train-only 对象检查；只有购买门通过，才实现并完整训练下面唯一一个模型。不要扩大搜索，不要重跑已有实验。用户负责提供可运行的本地仓库和数据；本任务只授权本地 CPU，不使用 SSH、GPU 或远端服务器。

## 0. 目标、已有事实、边界

审计起点是 `7fa46ab300b43c6457499e45408296c5a4bd4c5c`。如本地有更新，只核对相关文件是否改变，记录真实 revision；不覆盖用户修改，不自动 reset 或清理现有未跟踪结果。

历史背景：Small 106,925 参数，soup valid 0.121058 / train 0.055133；Full 408,651 参数，valid 0.119154 / train 0.045397。Full 已关闭容量路线，不再加宽。不同模型的历史数值不是本轮匹配对照。

本轮问题：现有 shell 化学统计之外，**同一个闭合路径内原子与键的组成、排列**是否提供任务相关信息？若是，把这类静态对象与节点环境放入同一个任务字典、一个图级读出，是否能进入更好的绝对性能区间？

约束：保留字典为核心；无消息传递、Transformer、attention、节点/边隐状态写回、双模型 ensemble。允许从原始图提取固定对象；允许对单个对象求解现有 16 步字典编码。官方 test 始终关闭。没有 seed 1、dense/random 控制臂、字典宽度/稀疏度/学习率/epoch/环长扫描或性能不佳后的 rescue。

本包 NumPy 检查已经验证一种表示盲区；没有验证真实 ZINC MAE。合成立方体的不同键分配可能具有相同性质标签，不得把它写成当前平台的已证原因。

## 1. 降低探索成本：只读这些入口

先读仓库根 `AGENTS.md`，以及 `tracks/ksvd/AGENTS.md`（若存在）。后续遵循仓库控制面。默认在仓库根执行 `uv run research ...`；不要另写绕开 run record 的正式训练脚本。

优先按以下顺序定位；无需遍历全部研究史：

| 入口（仓库相对路径） | 用途 |
|---|---|
| `tracks/ksvd/notes/zinc_e2e_dictenv_scale_v1_analysis.md` | 最新宽度负结果与运行口径；如命名不同，用 `rg --files tracks/ksvd/notes` 定位一次 |
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_latent_bridge_v1.py` | `LatentDictionaryBridge`、`LatentBridgeSEM108`、`_environment_from_parts`、`local_dictionary_bridge` |
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py` | Small 与父模型的逐位包含、参数盘点；复用证明，不用 Full 做候选 |
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_sem108_v1.py` | `sem108_block`、`sem110_interface`、plain/masked 环境路径 |
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_p1.py` | `env_collate`、`make_env_loader`、发生位置和跨图 offset |
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_h1_clarity_audit.py` | `AuditModel.forward` 的 C6 路径和现有静态 backend；继承链中实际 forward 以源码为准 |
| `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_latent_bridge_v1.py` | `_real_batch`、acceptance、`stage_train`、`_evaluate_detailed`、interventions、soup 与报告 |
| `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_scale_v1.py` | 最近运行的统一阶段驱动、缓存 reuse、冻结 train 状态后的分析 |
| `tracks/ksvd/src/ksvd_research/runners/zinc_e2e_dictenv_scale_v1.py` | 控制面 runner/register 的最小接法 |
| `tracks/ksvd/configs/luyin16/zinc_e2e_dictenv_latent_bridge_v1.yaml` | CPU8threads、320epoch、训练与选模冻结值 |
| `tracks/ksvd/experiments/luyin16/zinc_patch_path_pooling.py` | 原始 `patch_cont`、`global_context` 和原子/键类别映射 |
| `tracks/ksvd/experiments/luyin16/fsar_r2_ar0.py`、`zinc_topology_features.py` | 原始 φ65 和全图 topology25；已经有无类型环信息 |

历史环 token 条件化不是本候选：过去完整环 token 查表/条件化失败。本轮禁止完整环 ID embedding、哈希整个环当词条、按标签筛环、增加一套 K-SVD 训练。不要寻找那个无法取得的其他对话或 RPD checkpoint。

将附件里的小参考包放到本轮独立目录，例如 `tracks/ksvd/experiments/luyin16/typed_cycle_reference/v1/`，保留来源与 SHA256。不把它的 NumPy spine 当成正式模型替换源码。

## 2. 时间预算与阶段顺序

总墙钟目标 ≤4小时，按实测累计。CPU 固定8线程；设置 Torch/BLAS 线程，禁止并行争用 CPU 的训练。两臂 ridge 顺序跑，不构建三臂深度训练。

- A：对象提取、缓存和三折廉价检查，≤45分钟。先测一小批再估算全量枚举；达不到预算就资源关闭，交付记录，不缩对象范围并冒充原方案。
- B：仅 A 通过后，正式 PyTorch 实现、验收、计时与 smoke，目标≤45分钟。
- C：一个 seed-0 / 320epoch 正式训练、尾态干预与封档。正式开始前用真实完整 epoch 计时，按 `320×实测稳态epoch×1.15 + 15分钟分析` 估算，必须能放入剩余总预算。计时/smoke 的状态丢弃，不进入正式模型；正式开始后按锁定协议完成，不因中间 valid 不好提前宣布失败。

如预估超过预算，停在开始正式训练之前，报告预计耗时和已完成产物。本任务不授权改 horizon、上 GPU、换 max_cycle_length 或追加 rescue。

阶段 A 前提交短预注册，冻结特征、折、ridge 网格和购买门；阶段 C 前再提交模型/训练预注册与干净实现 revision。阶段 A 通过即已授权 B/C，无需再次问用户常规实现选择。A 未过则收口，不留下已启动的半轮训练。

## 3. 阶段 A：train-only 的廉价性质信号检查

### 3.1 对象枚举

读取已有正确的 official train10k 缓存，只加载 train；沿用仓库避免 PyG 自动处理所有 split 的脚本。记录 split fingerprint、原始类别映射、缓存来源。禁止读 valid/test 来定对象或门限。

把有向重复边去重成无向边，核验两向键类型一致且没有自环；原子类别0..27，键类别0..3，使用仓库映射，不重新解释为原子序数。

枚举长度3..10的**全部无弦简单环**，不是 cycle basis。环内非连续顶点间存在额外边则排除该环。使用本包 `chordless_cycles` 的早期弦剪枝/去重思路或一个验证等价的实现。节点 ID 仅用于枚举去重，不进入模型特征。不要使用会随节点顺序变化的环基选择。

记录每图对象数、各长度分布、总视图数、无环比例、p95/p99/max 及实际时间。长度>10不加到这条新分支，旧 topology25 保留；不得按最大前N个环截断或采样，因为会引入顺序和密度偏差。

### 3.2 两组原始特征

调用 `cheap_probe_features.py`，两个 ridge 臂只差一块，均为 label-free graph functions：

`X_base` 预期573维：
1. 每个根的**原始** Sem110，图级 sum 与 sum-of-squares：220维；
2. 每个根的原始 φ65，图级 sum 与 sum-of-squares：130维；
3. 原始 global62 + topology25：87维；
4. 无类型无弦环长度3..10计数：8维；
5. `edge_type_counts` 的固定128维原子对×键类型计数。

`X_typed` 1040维，由 `ring_features` 提取：按环长的原子/键组成及二阶统计、环内键类型对的循环间距、带环长的原子对×键型部分模式。固定 hash 只用于**部分模式**，不是完整环 token。没有 label-based 词条选择。

这些廉价描述仅用于判断是否购买模型；正式模型读取环序列，不直接读取这1040维。

禁止用已经在全10k标签上训练过的 node E、task code 或预测残差作为 cross-validation 特征；否则外层留出会泄漏。优先复用 label-free 原始缓存，不重跑数小时 K-SVD。若缓存只有标准化值，用已核验的 train scaler还原原始值；每个 CV fit 部分重新拟合本包的变换。不能凭一行 `label_free_features=true` 代替数据 provenance。

按已有验证过的 canonical molecule key 分组重复分子（若可用）；不为此新建一套图同构工程。若无该键，使用固定官方行 ID，并明确记录未分组重复的限制。NPZ 不得包含 object dtype，行序与 y 对齐。

### 3.3 固定检验与门

`cheap_property_probe.py`：固定seed20261002，三外层折，每折固定内部dev，分别为两臂选 `lambda={1e-3,1e-2,1e-1}`。损失为 mean MSE + λ||w||²，用内部 MAE 选λ，再在该外层全部训练行拟合；只把外层留出预测合成OOF。`asinh` 是固定变换，均值/尺度/常数和稀有列去除只基于当前 fit 行。

输出两臂OOF MAE、逐折差、λ和列数；购买门固定为 **OOF绝对MAE改善≥0.003，且至少2/3折改善**。不做额外λ、重复划分、表示宽度或特征子集搜索。

未过：`TYPED_CYCLE_OBJECT_SCREEN_STOP`，保存负结果，结束本任务；不继续 B/C。此结论只决定本候选的计算预算，不证明环对象理论无用。

通过：`BUY_ONE_TYPED_CYCLE_SCREEN`，进入 B/C。此结果不证明相对现有 Small 的增益，也不证明神经模型能得到相同改善。

## 4. 阶段 B：唯一正式模型 Small + typed cycles

### 4.1 精确原型

保留原 Small 的全部参数布局、既有结构字典及其训练方式、K32/s8、C6 masks、节点环境、静态 pair、全图通道和训练损失。冻结的是本轮协议与既有布局，不要把原本可学习的结构字典误改成冻结参数。

每个长度ℓ的环生成全部2ℓ个二面体视图。每个视图338维：按位置填10×28原子one-hot、10×4出边键型one-hot、10个有效位置mask、8维环长one-hot；尾部补零。位置t的键必须是 `(vertex_t, vertex_(t+1 mod ℓ))`，反向视图从原图键表重新取值，不能只把bond数组简单翻转。

共享 `ring_encoder = Linear(338,64) → SiLU → Linear(64,48)`，带bias，无额外 LayerNorm/dropout。**先对每个视图做非线性编码，再对同环2ℓ视图取平均**：

`h_cycle = mean_g ring_encoder(view_g)`。

之后调用同一个 `local_dictionary_bridge(h_cycle)`：D_L48×96、V_L96×48，λ1=.05、λ2=.01、ISTA16与现有 detached 步长、RMS缩放逻辑不变。没有第二套 D/V，没有新增重构损失。现有结构重构项保持父配置。

得到 E_cycle48，图级汇总为 `[sum(E_cycle), sum(E_cycle²), zero_count_slot]`，97维。无环图为97个精确零；C6 count 槽保持0，不能偷偷打开count捷径。

旧 unified302维 + 新 ring_pool97维 → **同一个** reader399→13→13→1，后续层和激活完整沿用原 GenericReader。参数精确预算：106925+24816+1261=**133002**，其中新任务字典参数为0。

实现可继承 `LatentBridgeSEM108`，基于实际 `AuditModel.forward` 最小追加一处 ring pool；保留 mask kwargs 与 `return_aux`，不要删 C6 或切换到另一条 forward。plain/masked 两条环境路径仍各经过现有 bridge 一次；环对象独立调用同一模块是预期行为。不要用会跨 batch 残留的隐藏缓存捕获 reader 输入。

先按原seed构造Small，新增层用隔离RNG初始化。替换reader首层时复制旧302列/bias和后续层，新增97列初始化为零。父参数逐项相同；只增加新函数自由度。正式训练从头初始化，不读取旧训练 checkpoint 热启动。

### 4.2 数据与 collate

缓存静态环视图；不每个epoch重新枚举。建议字段：

- `ring_view_atom[V,10]`、`ring_view_bond[V,10]`、`ring_view_mask[V,10]`、`ring_view_length[V]`：无offset。
- `ring_view_cycle[V]`：当前图的环序号，拼批时加**累计环数**。
- `ring_cycle_anchor[C]`：当前图的任意一个环节点用于路由，拼批时加**节点ptr**；graph_id=`data.batch[ring_cycle_anchor]`。
- `ring_cycle_length[C]`：无offset。

用独立wrapper复用旧 `env_collate`，不要修改旧全局 offset 白名单。PyG 对包含 `index` 的字段可能自动递增；避免自动+手动双offset。环锚点只用于路由，不能作为编码器输入。空图环字段形状也必须可拼批。

### 4.3 真实实现验收，未通过不得正式训练

1. 精确参数133002；旧参数与同seedSmall对应相等。新增reader列为零时，真实train batch32的plain/masked预测与Small最大差≤1e-5；若浮点归约误差需事前解释，不能靠放宽掩盖路径不同。
2. 实际原始图重标号、环旋转/反向、端点交换、图顺序、跨图拼批、无环/有环混合：预测不变；实际容差按现有基准记录，ring对象本身的整数视图集合严格相等。
3. 两个完整图反例经**真实仓库输入 builder 和 PyTorch**验证旧输入碰撞、新环汇总不碰撞；若旧输入没有碰撞，记录差异并撤回“现有模型无法区分此对”的主张，不能只展示NumPy参考替代。新表示应至少能区分本包组成相同/排列不同的六环例子。
4. ring-only路径确实经过同一个D/V；没有 raw ring→reader 或raw ring→scalar旁路；没有节点/边状态写回。改变ring对象不改变同状态下的节点E。
5. **两步冷启动梯度检查**：新增reader列零初始化，step0环MLP梯度为0是正常的；此时新增reader列必须有任务梯度。一次Adam更新后，在下一批/第二次backward检查环MLP和共享D/V有非零MAE-only梯度。不要把正确零初始化误判为死分支；不要提前加入非零旁路解决它。
6. 梯度有限，ring输出/ISTA有限；24步短smoke通过。mini梯度检查或计时不使用officialvalid挑方案。
7. `zero_ring`只能在bridge解码后归零环E/pool；不能设共享bridge的全局zero_code，否则节点也被删。确认zero_ring下节点E完全相同。
8. 测试阻断实际成立；不允许dataset构造自动下载/处理test。

测试集中覆盖上述会改变结论的性质，不堆镜像实现的冗余单元测试。记录 acceptance、timing、模型inventory、cache/provenance哈希。

## 5. 阶段 C：只购买一个完整 CPU 筛查

新实验名建议 `zinc_e2e_dictenv_typed_cycle_v1`；独立core、driver、runner、config和聚焦tests。复用最近scale/latent-bridge runner的 stage 与防重跑接口；不要复制整套历史审计到新代码。

沿用父配置：seed0，official train10k/valid1k，CPU8threads，batch128，Adam lr0.001、weight_decay1e-5、clip5，320epoch，原结构重构项，固定C6，Top-5参数soup。源码配置如与上述值不符，训练前报告并锁定父协议真实值；不可训练后再改。沿用既有soup buffer处理，不平均整型buffer，不从多个模型平均预测。

正式训练前提交预注册与代码，run record记录真实冻结revision、dirty/diff、所有拟合缓存的train-only来源、split fingerprint、设备和预算。正式期间tracked code干净；报告代码的dtype修复不得改变模型、训练或选模，若复用训练状态，记录checkpoint哈希和原train run。

固定分档：
- soup valid≤0.115：`TYPED_CYCLE_PROMISING_ABSOLUTE_SIGNAL`，值得后续确认；本任务仍不追加seed/对照。
- 0.115<MAE≤0.120：`TYPED_CYCLE_LIMITED_SIGNAL`，封档，不加宽/救援。
- MAE>0.120：`TYPED_CYCLE_STOP`，封档，不买下一轮同族变体。

分档首先由绝对性能决定。训练误差低、通道活、消融损失大不能改判。

终点仅做便宜推理诊断：
- 同口径soup train/valid MAE与gap，best、soup成员、曲线、wall/RSS。
- `zero_ring` 的ΔMAE与预测RMS。
- 固定seed11/22，**跨分子、相同环长的环对象置换**：保留每图环数量/长度分布及全局对象多重集，再编码、汇总。报告换到不同分子对象的比例。图内环行置换是sum池化的不变性，应保持预测，不是liveness探针。
- 环码与节点码分别报告非零率、有效原子、MAE-only梯度；近稠密就称软阈值任务字典，不能称稀疏字典优势。

无需重跑Small/Full，也不对历史已完成模型做新对照。旧成绩列为unmatched背景；推理干预只能说明依赖，不能证明训练增益。不要用officialvalid挑新的环长、hash、正则、threshold或调ensemble权重。

## 6. 收口与交付

无论 A 停止、资源停止或 C 完成，都交付：结论优先的中文报告、实际阶段/时间、source revision、命令、fold或run ID、可复现的输入/缓存哈希、limitations和后续决策。

预注册、分析、claim/decision、STATE指针、REPORT/DECISION/run_chain及promoted run按仓库规范记录，`uv run research verify` 通过。只把代码和小证据入git，大cache/checkpoint留gitignore；不要清理既有未跟踪产物。可完成本地提交；本任务不要求远端推送。

报告必须清楚区分：本包已做的合成审计、你新跑的真实train-only性质检查、真实PyTorch验收、正式valid结果。若没有进入正式训练，写“未训练”，不要给估计MAE。
