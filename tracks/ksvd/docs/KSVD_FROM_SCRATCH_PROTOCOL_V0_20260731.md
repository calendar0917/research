# KSVD 从零实验协议 v0：先验证结构基恢复

> 日期：2026-07-31  
> 状态：**冻结 v0**  
> 范围：只做 E0 数值恢复与 E1 固定槽位图结构基恢复；不进入采样、置换、图分类或真实数据。

## 1. 本轮唯一研究命题

> 在 patch 信号确实由少量共享结构基稀疏组合生成时，普通 KSVD 能否从训练数据中恢复这些结构基及其组合系数，并在未参与字典学习的测试 patch 上保持可重构性？

本轮把 atom 定义为**可组合的结构基元**，不要求每个学习 atom 自身是一张完整、合法的原型子图。后者属于更强的“合法 motif prototype”命题，暂不混入。

## 2. 为什么先不做其他内容

完整路线包含四个可独立失败的模块：

1. patch 构造 / 采样；
2. patch 的置换不变向量化；
3. KSVD 字典与稀疏编码；
4. graph readout、occurrence/incidence 与下游任务。

本轮使用 oracle synthetic signal 和固定节点槽位，主动拿掉 1、2、4，只检验 3。因而暂时禁止：

- CIN、MolHIV 和任何真实数据主表；
- B0/R2/RW/PPR/CoverageRW 比较；
- WL、canonical、rooted canonical 多表示比较；
- relation graph、true/shuffled binding；
- GNN 融合、复杂分类器和嵌套交叉验证；
- projected-KSVD 或“atom 必须是真实 patch”的约束。

## 3. 两级正控制

每个实验都先做 oracle-dictionary control，再做 learned-dictionary test。

### 3.1 Oracle-dictionary control

直接把真实字典 \(D^*\) 给 OMP，只检验：

- 数据生成是否正确；
- 稀疏编码器是否能恢复 \(X^*\)；
- matching、support F1 和图重构 evaluator 是否正确。

Oracle control 失败时，不允许解释 KSVD learner 的结果。

### 3.2 Learned-dictionary test

仅用训练 patch 学习 \(D\)，测试 patch 只用于冻结评估。改变 KSVD 初始化 seed，数据集本身保持不变，以隔离优化稳定性。

## 4. E0：纯数值稀疏字典恢复

### 4.1 数据

- 特征维度：\(d=15\)；
- 真实原子数：\(K=4\)；
- 真实字典：随机生成后 QR 正交化；
- 两个严格分开的条件：T1 只含单原子样本；T2 含 1 或 2 原子组合；
- singleton probability：0.5，确保每个原子有直接观测；
- 非零系数：随机正负号，幅值位于 \([0.5,1.5]\)；
- 第一版无噪声：\(Y=D^*X^*\)；
- 训练集 1000，测试集 300；
- KSVD：4 atoms；T1 使用 \(T=1\)，T2 使用 \(T=2\)；25 iterations，`T_min=1`；
- 优化初始化 seeds：0–9。

### 4.2 指标

1. test relative reconstruction error；
2. test NMSE；
3. Hungarian/exhaustive matched absolute atom cosine；
4. matching 后 coefficient relative error；
5. sparse support precision、recall、F1；
6. 不同 learner seeds 间的 pairwise matched atom cosine。

原子顺序、符号和尺度不具有语义；评估前必须进行 permutation/sign alignment。

### 4.3 E0 晋级门槛

Oracle control：

- reconstruction relative error \(\le 10^{-8}\)；
- support F1 \(\ge 0.999\)。

Learned dictionary，T1、T2 分别以 10 seeds 汇总：

- mean atom cosine \(\ge 0.95\)，且 worst-seed \(\ge 0.90\)；
- mean support F1 \(\ge 0.95\)；
- mean test relative reconstruction error \(\le 0.05\)；
- pairwise seed stability \(\ge 0.90\)。

失败解释优先级：实现/初始化 → 样本覆盖 → 字典可辨识性。E0 未通过，不进入图实验结论。

## 5. E1：固定槽位图结构基恢复

### 5.1 表示

- patch 固定为 6 个有语义的节点槽位；
- 使用无向邻接矩阵上三角，共 15 维；
- 本轮不随机置换节点，不做 canonicalization。

### 5.2 真实结构基

预定义 4 个 edge masks。第一版允许共享节点，但**边支持互不重叠**：

- A：槽位 0、1、2 上的三角；
- B：边 (2,3)、(3,4)；
- C：边 (5,0)、(5,3)；
- D：边 (1,4)、(2,5)。

每个标准化 atom 乘以对应边数平方根作为真实系数，因此组合后得到严格二值邻接向量：

\[
Y=D^*X^*,\qquad Y\in\{0,1\}^{15\times N}.
\]

分开运行 T1（每个 patch 恰有 1 个结构基）与 T2（每个 patch 含 1 或 2 个结构基）；训练/测试规模与 E0 相同。

### 5.3 附加图指标

除 E0 指标外，增加：

1. reconstructed edge precision、recall、F1；
2. exact patch adjacency recovery rate；
3. 每个匹配 atom 的 top-|true support| edge-support F1。

邻接重构以 0.5 为阈值。第一版 ground truth 为二值且严格线性，因此该阈值无歧义。

### 5.4 E1 晋级门槛

Oracle control：

- edge F1 = 1；
- exact patch recovery = 1；
- support F1 \(\ge 0.999\)。

Learned dictionary，T1、T2 分别判定：

- mean atom cosine \(\ge 0.90\)；
- mean atom edge-support F1 \(\ge 0.90\)；
- mean code support F1 \(\ge 0.90\)；
- mean reconstructed edge F1 \(\ge 0.95\)；
- mean exact patch recovery \(\ge 0.90\)；
- pairwise seed stability \(\ge 0.90\)。

E0 通过而 E1 失败，说明问题不是 KSVD 一般实现，而是邻接结构信号与当前稀疏线性模型的对应方式。

## 6. 本轮不用于晋级的内容

- 分类准确率；
- PCA/NMF/真实 patch baseline；
- 重构误差与分类性能相关性；
- 原子是否合法子图；
- IID/OOD 图分类；
- 采样覆盖率与边重复率。

这些内容分别属于后续 E2–E6，不应提前改变本轮结论。

## 7. 后续阶梯，但本轮不执行

只有 E0/E1 通过后才依次进入：

1. **E1B**：共享边、binary union、edge-flip noise、原子长尾频率；
2. **E2**：随机节点置换，raw flatten vs exact canonical；
3. **E3**：oracle patch vs B0/R2/path/RW 的结构捕获率；
4. **E4**：由 atom presence/count 决定标签的图级任务；
5. **E5**：motif multiset 相同、只改变 incidence 的组合任务；
6. **E6**：真实数据。

## 8. 复现入口

从仓库根目录运行：

```bash
uv run --with numpy python -m tracks.ksvd.code.test_from_scratch_recovery
uv run --with numpy python -m tracks.ksvd.code.run_from_scratch_e0_e1
```

默认输出：

- `tracks/ksvd/results/from_scratch/e0_e1_recovery_20260731.json`
- `tracks/ksvd/results/from_scratch/E0_E1_RECOVERY_20260731.md`

JSON 是数字唯一源；Markdown 只做可读汇总。
