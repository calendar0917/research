# 执行任务：已有 Full checkpoint 上的图级原型字典读出筛查

请从 `calendar0917/research` 当前状态继续，执行一个新的、**不重训骨干**的 ZINC 筛查。需要完成代码接入、针对性验证、本地CPU拟合、分析和科研记录。不要启动320epoch，不要把本任务扩展为架构/HPO搜索。

本任务包已完成NumPy参考验收；真实ZINC新性能尚未验证。你首先要读取 `README.md` 和 `graph_dictionary_acceptance.json`，再使用本包API。PyTorch脚手架未在提供方环境实际运行，须由你重新验收。

## 1. 问题与约束

上一轮typed-cycle正式MAE0.120779；其ridge OOF0.362618→0.331320并未预测强主模型收益。我们不再用弱代理的增益购买大训练。

本轮只问：**已有Full的814维图级表示，在一个固定原型字典和带正则的MAE目标下，能否直接给出明显优于同一Full原读出的预测？**

新模型：现有局部任务字典/静态组合骨干冻结，旧MLP读出由图级原型字典替换。一个骨干、一个标量输出；没有旧预测相加、模型集成、消息传递、Transformer、注意力、跨节点状态更新。图级原型对照只是固定字典匹配，不是self-attention。

只用本地CPU、8线程。不得SSH/CUDA/GPU。官方test不加载、处理或评估。旧容量/typed-cycle负结果不改写；复用已训练状态不等于重开加宽路线。

## 2. 预算与限定阅读

总任务预算60分钟；计算部分目标15–30分钟，预留10分钟封档。若必要缓存/checkpoint缺失，10分钟内交付具体阻塞，不重训。一次拟合收敛失败，记录INCOMPLETE，不反复修改算法/正则救援。

先运行 `git status --short`、`git log -3 --oneline`、`uv run research context`，读取根 `AGENT.md`。提供方审计HEAD `339dbea63ade4fa1ab2e51c32bd8f04fd52964d7`；以真实本地HEAD为准，不reset，不清理已有结果。探索最多10分钟。

只优先读：

1. `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_scale_v1.py`：`_soup_model`、`_soup_state`、`_model_factory`、`_evaluate_detailed`、`CHECKPOINT_DIR`、`TAG`。
2. `tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py`：`FULL`、`readout_layout`、`ScaleSpec`、`LatentScaleSEM108`。
3. `tracks/ksvd/experiments/luyin16/e2e_dictenv_p1.py`：`make_env_loader`。
4. `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_p1.py`：`load_split`。
5. `tracks/ksvd/experiments/luyin16/e2e_dictenv_clean_mechanism_v1.py`：既有`C6_MASK`。
6. `tracks/ksvd/src/ksvd_research/runners/zinc_e2e_dictenv_scale_v1.py`：最近runner接法。
7. 最近typed-cycle和scale的analysis，仅核对结果与协议，不展开历史档案。

早期compact-v4的richer-readout、FM、hinge和capacity审计已有负/不稳定结果。不要重跑它们，不套用不同骨干的旧OOF缓存充当本轮证据。

可应用 `remote-research-runner` skill的本地uv、数据与provenance部分；CPU约束覆盖远端步骤，不为使用skill连接服务器。

## 3. 复用特征，禁止重训

唯一骨干：Full m=3、seed0的**已完成参数soup**。默认精确路径：

`tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`。

核对真实run/checkpoint哈希、split fingerprint、source constructor、408651参数、C6以及冻结状态。缺失就停止并报告；不要替换为Small、typed-cycle、best单epoch、其他seed或重新拟合父字典。

用 `_soup_model()` 加载；`eval()`，所有骨干参数 `requires_grad_(False)`。只加载现有train/valid缓存，禁止调用旧整套prepare/train。

在 `model.reader.register_forward_pre_hook` 捕获**实际到达reader**的R，不手算替代、不抓第一隐藏层。Full输入是814维：unary289、5bucket静态pair485、global32、topology8。Hook每个batch恰一次；保留原reader完成H0重放，捕获结束立即remove。

本包 `extract_full_features.export_split` 提供最小脚手架，补上控制面provenance和验收即可。固定官方行序、无shuffle，batch128。train保存10000×814，valid保存1000×814；NPZ字段R/y/p_base/ids、split、checkpoint_sha、frozen_backbone、official_test_loaded。CPU状态和源代码不能在拟合中变化。

能复用上一轮验证过的canonical group_ids时，将其按官方train行序附到train NPZ；必须核对相同split/顺序。不要为此重新购买图同构工具。没有该缓存时，用官方行ID，记录重复分组限制。

原模型同checkpoint重放valid应约0.119154，需与该run精确指标核对，误差≤2e-6。原reader对捕获R重放也须匹配逐分子预测≤2e-6；跨batch尺寸与排列对齐，保证特征、label、p_base及ID都是同一行。

valid仅用于基线重放和最终筛查，不拟合scaler、prototype、带宽或lambda。它曾用于历史soup选择，结果只能是探索性筛查，不能称独立确认。

## 4. 图级原型字典与固定拟合

复用 `prototype_dictionary.py`，不另写新核族：

- R先逐列asinh；mean/std只在本次head fit行计算。固定零方差mask；std floor是同块活跃列std中位数的0.05倍。
- 四块按活跃维数及非空块数平衡，各块平均坐标能量量级相近。C6零槽不能偷偷打开。
- K固定256，seed20261002，从fit行无放回抽取代表作为C；不按label/残差挑原型，不把ID加入特征。
- 从fit行4096对固定seed随机图的正距离拟合唯一带宽：σ²=median(||za−zb||²)。不扫描带宽。
- k(z,C)=exp(−||z−C||²/(2σ²))，W=k(C,C)，ψ=k(z,C)W^(−1/2)。采用代码的固定谱截断处理重复原型。
- 直接预测 `median(y_fit) + [1,ψ]w`。常数atom同样受L2正则；目标 `mean|Aw−(y−median)| + λ/2||w||²`。

正则只允许 `λ={1e-5,1e-4,1e-3}` 三档。按固定seed将train groups每10组取1组作为head dev，其余head fit；分别拟合三档，用head dev MAE选λ（同分优先较大λ）。然后只用选定λ，在完整train10k上重新拟合scaler/原型/带宽和head一次。

这不是端到端OOF：Full骨干已经看过head dev的train标签。内部dev只用于head正则选择，不可把其增益当泛化购买门。真正的下一轮判定来自冻结后的同一1000个official valid。

用代码ADMM和dual bound，所有lambda及最终fit都必须 `status=CONVERGED`、primal-dual gap≤1e-6；单fit上限180秒，算法设定和最大迭代沿用本包。任何一次未达标则 `INCOMPLETE_SOLVER`，不得跳过该λ择优、不得拿未收敛结果当路线失败。不换MSE/Huber/LR/K/核，也不延长预算救援。

`run_cached_head.fit` 不接受valid路径。它先写 `model.npz` / `FIT.json`及SHA；模型冻结之后，独立 `evaluate` 才读取valid。保证无法用valid逐次改λ、prototype或算法。

字典保存是固定代表坐标，不是新增端到端学习的D；代码近稠密，不作稀疏性主张。分别报告冻结backbone参数、原型/normalizer/scaler的buffers、257个拟合读出系数及总存储，不能把buffer数量当训练容量增益。

## 5. 只有一次实际配对筛查

对固定模型和同一valid1000，输出原Full H0 MAE、新模型MAE、逐分子误差差、改善比例、均值/中位数。五个诊断组按固定ID排序后stride5划分，不能按看到的误差重分组。

继续门全部满足：

1. 新模型相对同checkpoint H0的绝对MAE改善≥0.006；
2. 新模型MAE≤0.113；
3. 五组至少四组改善。

满足：`GRAPH_DICTIONARY_READOUT_PROMISING`；本轮已经得到一个完整阶段式单模型，不需要立刻再买320epoch。只做下节的真实部署等价验收并封档。后续独立seed/对照确认需要新任务。

未满足：`GRAPH_DICTIONARY_READOUT_STOP`，封档，不加K/改带宽/换norm/换对象，不启动骨干训练。它只排除本配置，不证明现有表示充分或不足、不证明字典路线的理论上限。

不对照旧0.36ridge，不把0.12模型的失活消融当增益，不使用未匹配历史差声称统计显著。该valid被历史多次使用，不输出“独立显著性确认”的叙事。

## 6. 正信号才装配单体部署

构造一个独立Full实例，先严格load原state，再把 `model.reader` 替换为本包 `GraphPrototypeReadout`。保留一个骨干forward，旧MLP彻底退出预测；没有旧预测相加、加权平均、另一套全图特征或raw输入旁路。

Nyström normalizer折叠成prototype values：`values=W^(−1/2)w[1:]`，`offset=median_y+w[0]`。在线只算kernel(z,C)及一次加权和。需要保存审计态normalizer用于复现，但部署不必保留它。

该Torch脚手架提供方只语法检查过。你需要验证：

- 真实valid逐分子预测与缓存NumPy head最大差≤1e-5，MAE差≤2e-6；
- train/valid下C6完整保留；参数/状态在导出、拟合、评估中未变；
- batch大小/图顺序、已有重标号helper的一小批预测不变；不重做全仓对象审计；
- 原型/scaler均为buffers，在线不fit，不按当前batch重算统计；
- 字典读出直接进入唯一输出；同一forward只调用一次骨干路径；
- test blocker实际阻断，训练/部署脚手架不调用test；
- float64核计算后按旧输出dtype返回，避免方差小或谱截断方向的数值放大；
- 保存并重新加载state_dict可重放预测。不要pickle完整动态模型实例。

失败是实现/数值问题，限定10分钟定位；不以修改核/正则改善结果。输出清楚标注“缓存筛查通过、部署验收未完成”或具体阻塞，不宣称完整可部署。

## 7. 控制面与交付

新runner建议 `zinc_graph_dictionary_readout_v1`，独立protocol。stages可为export/fit/evaluate/deploy/analysis；关键scratch/screen都走 `uv run research run`，driver调用包内API。不要调用旧scale `all` 或train stages。

先提交短预注册和实现，固定parent checkpoint SHA、特征块、proto规则、lambda三档、求解精度/预算和购买门。真实代码revision、split fingerprint、cache hash、fit状态SHA、设备、solvercertificate、CPU/RSS/时间进入run record。检查现有未跟踪数据，保留原样。

默认流程示例（按你的runner schema实现，不要仅复制空接口）：

```bash
uv run research run zinc_graph_dictionary_readout_v1 --study zinc-context-gap --mode scratch --purpose "Frozen Full feature export and train-only graph dictionary fit, CPU" --set runtime.device=cpu --set model.stage=fit
uv run research run zinc_graph_dictionary_readout_v1 --study zinc-context-gap --mode screen --purpose "Locked direct graph dictionary readout versus replayed Full, CPU" --set runtime.device=cpu --set model.stage=evaluate
```

只跑本轮相关测试和真实mini-batch门，不跑全仓训练套件。达到停止或完成条件就收口：REPORT/DECISION、analysis、claim、STATE指针、promoted小记录、命令、cache和state hashes、限制、时间。`uv run research verify`通过。

可提交本轮代码和小证据；本任务不要求远端push，大checkpoint和R缓存按gitignore保存。科研记录清楚区分：提供方合成参考验收、你完成的真实Full重放、真实缓存head拟合、officialvalid配对筛查、部署验收。

本任务授权上述本地CPU低成本筛查与必要装配，不授权320轮、seed1、控制臂深度训练、超预算或test访问。常规实现和完整fit/evaluate已授权，无需再问是否开始。若负结果，最终明确写“本次不购买新的完整神经训练”，并停止。
