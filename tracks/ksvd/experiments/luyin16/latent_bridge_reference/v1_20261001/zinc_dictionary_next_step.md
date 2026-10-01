# ZINC 字典下一步：保留有效局部编码和静态关系，加入低维任务字典

审计版本：`65d4b8fb82261e9ec0a94bbf02d7ccc36e14219f`。

状态：方案与 NumPy 数学参考已完成；结构/数值检查通过。未加载 ZINC 数据、未执行 PyTorch backward、未做真实小样本或全量训练。没有新 MAE 结论，也没有修改或推送研究仓库。本文件是后继实验的设计与交接说明，不是已完成实验的 claim。

## 1. 这次负结果说明什么

HierRel 的官方 valid 0.182343、eval soup train 0.091181，关系通道活跃，足以停止这个具体候选。它没有证明字典或无消息传递方法无效。

上次方案同时改变了太多东西：

| 原 Sem108 路径 | HierRel 路径 |
|---|---|
| 化学 shell、结构字典绑定经局部非线性编码与 fusion 得到48维 E | 712维输入仅经过一个线性128维节点字典 |
| 一次计算静态 patch pairs，按5个距离桶分别汇总 | 只编码物理相邻键，统一汇总 |
| 非线性 pair encoder，保留 global_context 与 topology 辅助 | IHT关系码矩统计，只有 topology 辅助 |
| 小图级 reader | 新的大图级 head |

因此不能由 train/valid gap 唯一推断“主因是正则不够”。泛化差距存在，但表示、关系范围和读出归纳偏置也一起变了。上一版设计没有保护已经有效的模块，这是需要纠正的研究路径。

历史证据应分开使用：
- 冻结 Joint709：坐标惰性、编码误差较大；不能继续往原绑定中塞大输入。
- Increment：通道在用，但未有可观增量；信息被使用不等于性能更好。
- HierRel：通道在用，但完整重写明显退步；不能把“字典被用”当成功。
- 其他 Agent 的 RPD：约0.119的日志值得作为线索，但源码缺失、细/粗组差异没有形成一致收益；不能据此确定粗组均值分解是关键。
- 双模型0.10972说明误差可能互补，不证明如何做成统一模型；不继续预测集成。
- 本轮优先回到有代码、有清楚静态契约的 Sem108（历史 soup 0.123705）作为结构起点，而非重写第三个从零系统。

## 2. 已完成的轻量验证与迭代

运行 `python validate_dictionary_bridge.py`，只需要 NumPy。

### 2.1 线性矩统计的明确限制

两个标量对象集合 `[0,0,3,3]` 与 `[0,1,1,4]` 的 sum、mean、std 完全一致。任意共享线性向量映射之后，这三类统计仍一致。实际128维数值检查最大差约 `4.44e-16`。

先逐对象做 `ReLU(x-1)` 再求和，两者分别为4和3。这说明非线性局部编码不可由一个更大的图级 head普遍替代。它是节点分支的反例，不是整个 HierRel 模型或真实 ZINC 分子的碰撞证明。

### 2.2 对硬截断和图内组均值的检查

在 top-1 排名边界上，输入变化约 `2.83e-7`，硬截断的输出变化约1.414；软阈值输出变化仍为 `2.83e-7`。这只是一个求解边界例子，不证明已有负结果由 IHT 引起。

固定键端点的局部码，给图中加入同原子类型的其他节点，会改变旧 μ/δ 关系对象。它证明该对象依赖图内组构成，而非纯局部环境配对；不证明这种上下文本身有害。

基于这两项检查，候选不再采用新的 μ/δ 分解和随机128→32投影，也不再强制节点保留固定 top16。

### 2.3 低维字典桥接数值结果

候选的48→96→48桥接：
- 单位范数字典列、tight-frame 初始化误差约 `1e-15`；
- 检查模式 λ1=λ2=0 时，float32 相对 L2 恒等误差约 `1.22e-7`；
- 正式初态 λ1=0.05、λ2=0.01、16步 ISTA：合成样本相对平方误差 `0.0023695`，单行误差95分位 `0.0024876`；
- 初始平均非零码约76.7/96：它允许可变密度，不能称为强稀疏；
- 重标号、batch组合、端点交换检查通过；
- 显著扰动字典后，内层凸目标逐步下降；
- 合成 MAE 的有限差分对字典和词条值表均非零，小步更新使合成 MAE下降。

这些检查只支持数学实现可行和初始接口受到保护，不支持降低真实 MAE，也不替代 PyTorch 梯度、实际缓存或分子测试。原始结果在 `dictionary_bridge_audit.json`。

## 3. 唯一推荐候选：SEM108 + 局部任务字典桥接

不要购买同一 HierRel 的 train-size/dropout/weight-decay 矩阵；不要再加一个关系变体。

保留 Sem108 的原始输入、结构字典与 common 坐标、局部 nonlinear encoder/fusion、静态配对范围、距离分桶、global/topology辅助、reader，以及原训练损失与协议。

在局部 fusion 的输出 `h_i ∈ R^48` 上加入：

```text
已有静态局部对象
    → 原 Sem108 非线性局部编码 h_i[48]
    → 一套共享任务字典，软阈值编码 α_i[96]
    → 词条值表解码 E_i[48]
    → 原有 unary 与一次静态 pair 计算
    → 原来的一个图级读出
```

节点/局部环境之间不进行状态写回。稀疏求解中的16次更新只在每个对象内部进行。新增层不接受 edge_index、graph_id 或其他节点隐藏状态。

所有局部与 pair 路径均读取新 E，不加入 `h + correction` 的残差旁路。不增加新的原始特征旁路；已有低维全图辅助保留。

### 3.1 数学定义

```text
ρ_i = sqrt(mean(h_i²) + 1e-12)
x_i = h_i / ρ_i
D_L: [48,96]，按列单位范数，接受任务梯度
V_L: [96,48]，接受任务梯度

α⁰_i = 0
L = 1.05 * eigmax(D̄_L D̄_L.T) + λ2
η = 1 / L
α^{t+1}_i = soft_threshold(
    α^t_i + η*((x_i - α^t_i D̄_L.T) D̄_L - λ2 α^t_i),
    η λ1
)

λ1=0.05，λ2=0.01，t=0..15
E_i = ρ_i * (α_i V_L)
```

固定16步，称作展开的 ISTA 编码器，不能宣称等于精确 elastic-net 最优解。L 为安全步长估计，在 PyTorch 中 detach；编码本身必须保留梯度。

`ρ_i` 只由当前对象计算，不拟合 train/valid 的新 moving scaler，并保留梯度。全零 h 返回全零 E，无NaN。稀疏度由软阈值决定，不做固定 l0 预算，不另加 softmax/attention。

初始化：
```text
Q = seed0 确定性 QR 得到的48×48正交矩阵
D_L = [I48, Q]
V_L = D_L.T
```

两个正交基是同一本96词条字典的初始化，不是两个模型。两者均可训练。私有 generator/NumPy RNG 不得改变父模型的初始化、dropout 或数据顺序。

这是任务驱动的潜在局部字典，词条是可复用的潜在环境方向，不应包装成已可视化的精确化学片段。D 编码，V 是同一词条的预测值表。所有新分子使用同一 D、V。

新增参数9216；原 Sem108 文档总参数97709，候选预计106925，实际计数验收。不采用高维K-SVD，不使用或重新拟合Joint709缓存。

### 3.2 损失

沿用父模型 `MAE + 原结构字典重构项`，不改原 lambda、mask 或结构码定义。新增字典接受 MAE 梯度；不新增高维重构辅助损失，也不继承 HierRel 的0.05/0.02双重构损失。

低维 `||x - α D̄_L.T||²` 作为诊断报告，不加到外层损失。软阈值的 λ1/λ2 是内层编码定义，与外层损失权重不同。

它借鉴任务驱动字典的思想，并非复现 Mairal 等人的完整优化算法。参考：
- https://arxiv.org/abs/1009.5358
- https://papers.nips.cc/paper/6931-deep-sets

## 4. 给实施 Agent 的最短代码路线

先 `git status --short`、`uv run research context`、读 `AGENT.md`，保留已有修改。定向阅读最多15分钟。

关键文件均在 `tracks/ksvd/experiments/luyin16/`：

1. `e2e_dictenv_sem108_v1.py`
   - `SEM108Model`；`build_sem108_model`；
   - **公共插入点 `_environment_from_parts(...)`** 返回 `self.fusion(fused)`，48维；
   - 普通 `environments` 与训练所用 `environments_masked` 都经过这个公共点。

2. `e2e_dictenv_p1.py`
   - `P1Model.forward`：E 同时用于 unary 和 pair_projection；
   - 静态pair计算、5距离桶池化、global/topology辅助、reader；
   - `make_env_loader`、`env_collate`。

3. `e2e_dictenv_common_subspace_dictionary_v1.py`
   - `train_cssd(..., model_factory=...)` 可以复用；
   - factory签名 `(dictionary, seed, subspace)`；
   - 训练调用 `model(batch, mask=mask, return_aux=True)`，不能只改普通 environments；
   - `reconstruction_loss(aux['phi'], aux['coord'])` 是父结构损失，保持不变。

4. `zinc_e2e_dictenv_sem108_v1.py`
   - 模型构建、公共子空间、数据/阶段与记录方式。

5. 新控制平面 runner/config 可参考最近 HierRel 的外壳，只参考控制平面，不复制其独立数据类和新损失。

推荐实现：
```python
class LatentBridgeSEM108(sem.SEM108Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_dictionary_bridge = LatentDictionaryBridge(...)

    def _environment_from_parts(self, *args, **kwargs):
        h = super()._environment_from_parts(*args, **kwargs)
        return self.local_dictionary_bridge(h)
```

这样保留旧 state_dict 的所有共有参数名，且 masked/unmasked 两条路径都经过桥接。新参数不要命名成父类的 `self.D`，它仍指结构字典。

旧 checkpoint 只用于接口一致性检查。加载时只允许缺失桥接的两个新参数；不能泛用 strict=False 忽略其他错误。

不要热启动全量候选：正式从相同的父模型随机初态加新字典初态开始，完整320轮。已训练父模型不得用于内部留出集上的“泛化验证”，它已经看过官方train。

## 5. 先小验证，再让用户购买全量训练

### A. 实际实现验收，预算10分钟

- 相同共有初态、eval模式、同一真实train batch；λ1=λ2=0且V初态时，新E和父E相对L2误差≤1e-6、最终预测max差≤1e-5。
- 该恒等检查模式不进入正式训练，也不作为可选择候选。
- 正式初态与父E比较：报告train-only样本相对误差和逐行尾部误差。若出现大于5%的总体相对平方误差，先检查归一化、V转置、步长和wiring，不能直接全量。
- 新 D_L/V_L 的 MAE-only 梯度有限、实际方向/值更新；真实train batch forward/backward。
- 普通和masked路径都会调用桥接，调用计数一次，E同时进入unary和pair。
- 重标号、batch offset、端点交换、纯净zero-code干预；关系变化不能写回局部码。
- test blocker。

### B. 短程可训练性检查，预算10分钟

只使用official train中的固定1024分子，父模型与候选各从头运行40轮，共享初始化和数据顺序。两边均用父损失和相同CPU线程。

这不是重跑父模型的320轮结果，也不评价最终MAE。没有official valid或test访问，报告train task MAE下降、新字典任务梯度、码密度、初态→终态误差和实际吞吐。

若候选无法学习或出现严重拟合退步，最多10分钟审计真实实现与求解，不调K、λ、LR来救候选。短程没有优于父模型不能当作科学停止结论；只有数值/接线错误或明显不满足可训练性才阻断正式运行。

A/B的配对小检查本轮必须保持为scratch/dev证据，不能称为全量匹配增益。不要因选短程更好的配置而扩成多候选搜索。

### C. 用户执行的正式单候选实验

新runner建议 `zinc_e2e_dictenv_latent_bridge_v1`，全量前提交实现、协议和配置：

```bash
uv run research run zinc_e2e_dictenv_latent_bridge_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "SEM108 local task-dictionary bridge absolute-performance CPU screen" \
  --set runtime.device=cpu
```

- official train10000、valid1000；test从未实例化；
- CPU8线程、seed0、batch128、Adam1e-3/WD1e-5、clip5；
- 320轮、父固定LR、父Top-5 soup；
- 完整模型参数平均，包含结构字典、新D_L/V_L；固定buffers一致；
- 不重跑旧320轮，不额外dense/PCA对照，不加seed，不延长horizon；
- 预注册与formal revision clean；研究运行走控制平面；
- prepare只复用所需Sem108/sdb/common对象，严禁转调旧Joint709整套prepare；
- formal前预估预算，并预留收口时间；中断报告INCOMPLETE，不当失败。

主要分档：soup≤0.115为强信号；≤0.120为有前景；0.120–0.1233为边界；>0.1233停止当前候选。只有一个seed，不宣称小差异显著。

终点报告：
- best/soup、train/valid同一eval口径、末段曲线、参数与时间；
- 码分布与逐对象l0，不能把变量密度误写成固定s；
- 新字典MAE-only梯度；zero整个桥接码、同图内码置换；
- 保持已训练局部编码器与head，只把新字典/词条值重置为初态的推理诊断；
- 零化只能证明该通道承载信息，不能证明字典学习比匹配训练对照更有价值；
- 旧Sem1080.123705仅作背景，因新增参数/协议不同不能做因果增益claim。

当前 HierRel 的 `delta_pred_rms` 实际是 RMS(干预预测)-RMS(原预测)，可能为负；另行报告真正的 `sqrt(mean((干预预测-原预测)^2))`。原活性布尔门使用了后者，负数字段不改变活性结论，但下一轮应清楚区分。

## 6. 成功和失败以后各做什么

通过≤0.120且字典确实学习后，才考虑确认重复种子与匹配对照。先确认绝对性能稳定，再讨论稀疏特有价值。

若保留有效结构、避免高维强压缩的候选仍没有前景，就停止在现有表示上反复调整字典的K/s/λ。届时应重新评估局部对象或学习目标，而不是声称加大head/正则一定能解决。

本轮没有证据能保证达到0.06–0.07。推荐这个候选的理由是：它保留已验证的计算结构、让字典进入所有局部关系主路径，并且先用数值门保护已有表示，降低下一次完整试错的成本。
