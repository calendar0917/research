# E2E-DictEnv-Joint709-Absolute-v1 — pre-registration (frozen)

Round: **E2E-DictEnv-Joint709-Absolute-v1** (`e2e_dictenv_joint709_absolute_v1`).
Protocol: `zinc-context-gap` (study `zinc-context-gap`), track `tracks/ksvd`.
Runner: `zinc_e2e_dictenv_joint709_absolute_v1`.
Write-then-follow: nothing below may change after the first formal (`screen`) run.

任务来源（中文摘要）：路线二（联合 709 维环境字典）在
`e2e_dictenv_rolecorr_increment_v2` 中只完成了定义与准备阶段（该轮路线一
未过门，路线二训练阶段被有意留空）。用户在本轮明确授权：不重跑 A/B/C、
不跑 PCA48、不做随机字典/shuffle 训练/无字典对照，**只跑一个新的正式候选**
——冻结的 K48/s12 联合环境字典，从零训练，seed 0、320 epochs、固定 Top-5 soup，
先筛该字典路线的**绝对性能**。

新假设（唯一检验对象）：

> 结构残差、化学属性（Sem108）与对应关系对象（RoleCorr 536）需要在**字典编码
> 阶段联合组织**，而不是分别编码后追加到结构坐标：即候选的新假设是
> “联合对象 + 单个共享稀疏字典”这一编码对象的绝对性能前景，而不是
> “联合对象相对某控制的增量”。

本轮**不做对照实验**（不训练 PCA48、不训练无字典/随机字典/shuffle 臂），
因此本轮不产生、也不允许声明任何“优于/劣于某控制”的因果结论。

---

## 0. Execution regime

* 全流程本地 CPU：`runtime.device=cpu`（显式），`runtime.torch_threads=8`
  （沿用前轮设置，实际值在每个 artifact 中记录）。
* 不启动 GPU / 远端训练；不使用消息传递、GNN、Transformer、注意力、递归或
  多轮环境状态传播。模型仍是“一次局部编码 → 一次静态组合 → 图级读出”：
  节点级坐标 `z_v` 由一次局部 patch 编码得到，字典编码内部的固定步数 IHT
  迭代属于同一次局部编码，不传播跨节点隐藏状态。
* 官方 ZINC **test 永不加载**：control plane `test_policy: terminal` 下
  `screen` 模式 `test_access: blocked`；runner 另外显式拒绝 granted test access。
* 固定内部划分：official train 10000 / official valid 1000；不根据历史 test
  选择模型。split fingerprint 由 control plane 记录。

---

## 1. 唯一候选（JOINT-SPARSE）

### 1.1 坐标（宽度 49，冻结）

```
z_v = [ c~_v (1) ; alpha_v (48) ]                       in R^49
c~_v   = (phi_v · U) / common_rms                        (frozen CSSD-q1 U in R^{65x1})
alpha_v = IHT_10( colnorm(D_joint48), x_v )              (exact top-12, tied IHT)
```

* `IHT_10` 是仓库已验证的 `tied_iht_codes`（固定步长 `eta = 1/(sigma^2+eps)`，
  `sigma` 由确定性 power-iteration 给出）。**迭代次数是 10**；
  `s = 12` 是每行非零个数上限，两者不可混写。
* 精确稀疏约束：`l0(alpha_v) <= 12`。

### 1.2 联合输入 `x_v`（709 维，冻结 scaler）

```
x_v = [ w_struct * r_v / scale_struct * mask_struct ;
        w_sem    * s_v / scale_sem    * mask_sem    ;
        w_corr   * c_v / scale_corr   * mask_corr ]
r_v = (I - U U^T) phi_v            (65, structural residual)
s_v = Sem108(patch)                (108, frozen Sem108 interface)
c_v = RoleCorr correspondence       (536 = node 392 + edge 144, raw object)
```

* scaler 只在 **official train** 上拟合（train-RMS 缩放 + 零 RMS 掩码 +
  三块平均能量平衡 `w_b = 1/sqrt(E_train[||block_b||^2]+eps)`），拟合后**冻结**，
  valid 只做变换。定义复用 `e2e_dictenv_rolecorr_increment_v2` 的
  `fit_joint_scaler` / `apply_joint_scaler`（路线二冻结定义，不改）。
* 三块顺序与行对齐：`[struct(0:65) ; sem(65:173) ; corr(173:709)]`，行序为
  patch 序（逐分子、逐 root 0..n-1），与 official valid 评估的分子序一致。

### 1.3 联合字典 `D_joint`（K48/s12，冻结）

* 无标签 K-SVD（`sdb.fit_ksvd`，detached），拟合输入是上述 709 维平衡后的
  **全部 official train 行**；`atoms=48`、`s=12`、`epochs=10`、
  `seed=20260924`（沿用已冻结的路线二定义）。
* 拟合后冻结：`requires_grad=False`；训练中字典不更新（不端到端微调）。
* 真实重构诊断（不改变任务梯度，也不参与选择）：用模型实际使用的
  tied-IHT(top-12) 计算 `E_joint = mean||x - Dbar alpha||² / mean||x||²`，
  同时报告三块分块误差与 OMP(top-12) 参考（固定子集）。
  旧 route-2 `reconstruct` 的零占位**不得**当作重构质量。

### 1.4 其余全部冻结（沿用 Sem108 父模型与 increment-v2 路线二）

* Sem108 接口、C6 mask、融合、关系特征、pooling、后端、reader、优化器与
  初始化规则、batch、lr、weight decay、clip、epochs、Top-5 soup 规则全部沿用。
* binding 扩展：历史 `W_A_S[33,48]` / `W_E_S[99,48]` 复制为
  `[49,48]` / `[147,48]`，新增行零初始化（确定性；本轮只有单臂，无跨臂共享问题）。
* 从头训练，不热启动旧 soup；本候选不同时改后端、不放开字典更新、
  不增加层、不扫 K/s/LR、不延长 horizon。

---

## 2. 冻结复用对象（运行前核对 path / shape / train-only 来源 / SHA）

| object | identity check |
|---|---|
| common subspace (CSSD-q1) | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` file SHA-256 `36636ce9…8f6c24` |
| frozen SDB K32/s8 structural dictionary | `p2run.load_dictionary(H1_CONFIG.dict_kind)`（模型内的诊断对象，route-2 坐标不使用它） |
| raw RoleCorr 536-D caches (train/valid) | node/edge SHA-256 与 `incrun.EXPECTED` 中冻结值一致；n_patches 231664 / 23083（实际值记录在 artifact） |
| Sem108 interface | 代码级复用（`sem.SEM108Model` / `resolve_sem108_geometry`） |
| env caches / split | 复用 `zinc_e2e_dictenv_p1` 的冻结 env cache；split fingerprint 由 control plane 记录 |

**缺失对象只补建缺失部分**：本机 `results/e2e_dictenv_rolecorr_increment_v2/`
下不存在 `joint_standardizers.json` / 联合缓存 / `dictionary_joint.pt`，因此
本轮补建这三类对象（train-only 拟合），不重跑旧模型、不重新拟合任何已存在的
冻结对象。**不运行 PCA48**（本轮无对照）。

---

## 3. 阶段与启动命令（正式训练必须从干净、已提交版本启动）

预备（scratch，train-only，无 test）：

```bash
uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
  --study zinc-context-gap --mode scratch \
  --purpose "Joint709 prepare (train-only scaler/cache/frozen K48/s12 dictionary) CPU" \
  --set runtime.device=cpu --set model.stage=prepare
```

正式（screen，唯一科学运行）：

```bash
uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "Joint709 dictionary absolute-performance CPU screen" \
  --set runtime.device=cpu
```

训练协议（冻结）：seed 0；epochs 320；batch 128；Adam `lr=1e-3`,
`weight_decay=1e-5`, grad clip 5.0；loss = task L1 + `lambda_rec * rec_diag`
（冻结字典下 rec 项梯度惰性，值与父协议一致）；无 early stop；
Top-5 soup 按 valid MAE；数据顺序沿用冻结 loader seed
（`seed + TRAIN_SHUFFLE_OFFSET` / `seed + EVAL_SHUFFLE_OFFSET`）；
每 10 epochs 保存可恢复 checkpoint（model+optimizer+RNG+curve+soup bookkeeping）。
截断的运行记 `completed: false`，永不写成科学负结果。

---

## 4. 训练前正确性门（任一 FAIL ⇒ STOP）

* **G0 geometry** — 联合输入 709；坐标 49（`common [0,1)`，`joint [1,49)`）；
  `W_A_S [49,·]`，`W_E_S [3*49,·]`。
* **G1 split provenance** — scaler/cache/dictionary 元数据齐全且记录
  `fit_split="official train"`、`official_test_loaded=false`；cache SHA 与
  元数据一致；`n_patches`（train/valid）与冻结 corr cache 一致。
* **G2 row alignment** — 用 `load_split` + 原始 corr cache 重新计算若干
  （分子, 节点）位置的 709 维 scaled 向量，与缓存行逐位（或 1e-6 内）一致；
  再验证 `_attach_arm_block` 的 `joint_vec` 切片与 `node_sizes` 偏移一致
  （每分子、含 batch padding 前缀子集）。
* **G3 scaler semantics** — train-only 重拟合得到相同的 scale/mask/weight；
  三块 scaled 平均能量 ≈ 1；零 RMS 掩码坐标数记录。
* **G4 freezing** — `D`、`D_block`（=D_joint）`requires_grad=False`；
  `D_joint` shape `(709, 48)`、列归一化；`D_joint` SHA 与元数据一致。
* **G5 sparsity** — 真实 valid 批次上 `l0(alpha) <= 12`（并记录均值/最大值）；
  `IHT_STEPS == 10`（不是 12）。
* **G6 channel in prediction path** — 扰动 `joint_vec`（或 zero）时坐标与
  预测都改变；zero 只影响 `JOINT_SLICE`（其他列逐位不变）。
* **G7 gradients / updates** — 小批量 forward/backward 数值有限；readout /
  binding（含 joint 行）/ fusion 梯度非零；`D`、`D_block` 无梯度；
  一次 optimizer step 后目标参数确实被更新（快照比较）。
* **G8 relabel / batch-offset** — 同一分子节点重标号后每节点坐标按同一置换
  不变；同一批样本在 batch offset 平移（或前后拼接不同分子）后每节点坐标
  逐位不变。
* **G9 official-test blocker** — 每个 payload 断言
  `official_test_loaded = false`；test split 从未实例化（loader 拒绝）。

smoke（2–4 epochs，小数据子集）只用于 correctness / 梯度 / 数值稳定性 /
耗时估计，不作为科学筛选结果。

---

## 5. 绝对性能门（资源分配阈值，写入预注册；不是统计显著性门）

使用完全相同的评估口径（official valid 1000 分子，Top-5 soup，MAE）:

| soup valid MAE | 区间 | 本轮建议动作 |
|---|---|---|
| `<= 0.115` | `strong_prospect` | 强前景，建议进入后续确认（另开预注册） |
| `0.115 < mae <= 0.120` | `promising` | 有前景，值得下一轮验证 |
| `0.120 < mae <= 0.1233` | `borderline` | 边界结果，暂不追加实验 |
| `> 0.1233` | `stop` | 本候选未进入更好区间，停止 |

历史数字仅作背景（EXTRA-STRUCT soup 0.12639307；CORR-ADD soup 0.12566183；
CORR-PCA-ADD soup 0.12329247；Sem108 历史 soup ≈0.123705，CPU/不同 regime），
**不能**作为本轮的匹配对照，也不能据此得出“优于/劣于”任何方案的因果结论。

停止纪律：只跑这一个候选、一个 seed。不扫 K/s/LR、不延长 horizon、不增加层、
不端到端微调字典、不因早期（如 40 epochs）表现一般而淘汰完整候选（早期只检查
错误与数值稳定性）。若通道失活，最多做一次 10 分钟定位；实现错误可修复，
优化/结构救援属于另一候选，本轮不反复尝试。结果不错也不自动购买 seed 1。

---

## 6. 报告与记录（交付要求）

* best valid MAE 及 epoch、Top-5 soup、末段 train/valid 趋势；
* 用 soup 模型在 **official train** 上做一次同口径评估（判拟合/泛化，不作选择）；
* 字典活跃原子/有效原子数/l0/真实联合重构诊断（含分块误差）；
* soup 状态下联合编码 zero 与少量分子内行 shuffle（两个固定种子）——
  仅作通道使用诊断，不训练新模型、不构成因果结论；
* trainable/total 参数、CPU 环境、线程、时间、内存、代码 revision、
  split fingerprint、artifact SHA；
* 决策先行（属于哪个绝对区间、是否值得继续），再解释训练是否正常、
  是否有泛化瓶颈、证据限制；
* promote 有效 run，更新 analysis/claim/decision/STATE 指针，提交并 push
  代码与小型记录；大 checkpoint 不入 Git。

本轮可以主张“该联合字典候选达到某个 official-valid MAE / 落在某个绝对区间”，
**不能**主张“联合对象带来增量”“稀疏字典优于 PCA”“zero/shuffle 损害证明
因果必要性”（只说明通道被使用）。

## 7. 预算

默认总墙钟 3 小时：探索/核验/预注册 15 min；实现与针对性验证 50 min；
smoke + 完整训练 90 min；分析/记录/交付 25 min。
实测 K-SVD 预备（K48/s12、709 维、10 epochs、全部 231,664 train 行）约
70–75 min（单 chunk 基准外推），因此实际分配为：
预备（scaler/cache/K-SVD/dictionary/diagnostic）~85 min，
实现+测试 ~45 min，correctness+smoke+训练 ~60 min，分析/记录/交付 ~25 min。
若预计实质性超预算，先报告剩余工作、预计耗时与已保存状态；预算截断只记
“未完成”，不记科学负结果。数据集准备可适度并行，但只有一个训练进程，
避免 CPU 线程争抢。
