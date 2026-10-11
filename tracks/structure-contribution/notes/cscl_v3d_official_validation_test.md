# CSCL-v3d 冻结模型的官方 validation/test 终点评估

- 协议：`configs/cscl_v3d_frozen_official_eval.yaml`（commit 55e088b 冻结；amendment 1 在读取任何官方数值前于 a3d0842 提交）
- 评估代码：`code/eval_cscl_v3d_official.py`（最终代码 0fdb506；评分程序哈希 a5dd59d0d… 在 access record 中钉住）
- 产物目录：`results/cscl_v3d_official_eval/`（git-ignored，按本仓库 results 惯例）
- 训练次数：**0**。优化器不存在于评估模块；三个模型 `eval()` + `torch.no_grad()`；checkpoint SHA256 在评估前后逐位相同。

## 0. 先回答：实际数字

| 模型 | 内部 dev（历史，记录） | Official validation（1000） | Official test（1000） |
|---|---:|---:|---:|
| F（Full-Y soup） | 0.11641 | **0.13295** | **0.11410** |
| O-bucket（best-monitor） | 0.11610 | **0.13789** | **0.11349** |
| S-bucket（best-monitor + fit-only gauge） | 0.12694 | **0.15048** | **0.12511** |

配对差（有符号逐分子 `d_i=|e_A_i|-|e_B_i|`，5000 resamples，seed 0，percentile CI95）：

| 配对 | Official validation | Official test |
|---|---|---|
| S − O | +0.01260 [+0.00554, +0.02002] | +0.01162 [+0.00403, +0.01897] |
| S − F | +0.01753 [+0.00698, +0.03062] | +0.01101 [+0.00243, +0.01947] |
| O − F | +0.00494 [−0.00350, +0.01465] | −0.00061 [−0.00785, +0.00711] |

逐分子胜负数（strictly better / worse）：test 上 S 优于 O 459/541，S 优于 F 473/527，O 优于 F 535/465；valid 上分别为 470/530、465/535、499/501。

**预注册判断**：`judgment.json` → band = **strong_support**（test S−O CI 上界 0.01897 ≤ 0.02 且 S−F 上界 0.01947 ≤ 0.02，方向一致；全部工程门通过）。O−F 在 test 上 CI [−0.00785, +0.00711] 完整落入预声明等价区间 [−0.01,+0.01]，允许“统计等价”表述；valid 上 O−F +0.00494 CI 跨零，未达等价标准，只报告观察差。

dev → official 变化：F +0.0165 / −0.0023，O +0.0218 / −0.0026，S +0.0235 / −0.0018（valid/test）。S 相对 O 的 MAE 相对增幅：valid +9.1%，test +10.2%。

**valid MAE 偏高的原因（描述性，不修改主指标）**：official valid 的 y 尾部含极端离群（min = −20.34，test min = −9.68，train min = −42.04）。F/O/S 在 valid 上最大绝对误差 18.3/20.6/20.4，均由该尾部驱动；test 最大误差 1.8/3.6/3.7。逐分子配对统计不受影响（同一批分子配对比较）。不删重复、不删离群，主指标始终为官方全量 1000。

## 1. 冻结与工程门槛（先于评分，全部通过）

### 1.1 Phase A 冻结（commit 55e088b；amendment 1 于 a3d0842）

- 三模型 checkpoint/prep/词表来源哈希全部与协议一致（F soup 75d1737c…、O f9135fc4…、S 915f63e5…、gauge 60b22be3…、prep_fit7200 6b797e27…；参数量 F 408,651 / O 200,513 / S 193,128）。
- 严格加载：三份 state_dict `strict=True` 全部通过；gauge offsets `(761,)` 原样加载，fit-only 统计交叉核对复现 `gauge_meta_s0.json` 的 fit_unit_type_hash / fit_rows_hash / counts_hash（39,689 fit 单元，761 类型，746 known）。
- 历史 test 接触分层披露（见 §5）。
- 结构重复/重叠审计（标签无关；WL 哈希 + 精确图同构，先在与 RDKit canonical SMILES 表的对照上通过：2000/2000 纯度 1.0）：
  - valid 内部重复 0；test 内部 1 对（行 79/794）；
  - valid↔test 0；valid↔train 1 对（376/337）；test↔train 1 对（556/475）。
  - 影响可忽略（≤0.2% 分子），主指标不做任何删除。

### 1.2 Phase C 训练数据回放（`input_replay_audit.json`）

| Gate | 结果 | 容差 |
|---|---|---|
| F 对 monitor 重放 vs 已存导出 | max\|Δpred\| 1.67e-6 | 5e-5 |
| F 对 dev 重放 vs 已存导出 | max\|Δpred\| 1.91e-6 | 5e-5 |
| 新推理入口对 train 前 64 分子重建 vs 已验证缓存 | 结构字段完全一致；float max\|Δ\| 3.81e-6（缓存为 res GPU1 构建，CPU 回放的 float32 设备舍入；非 bit-exact 已记录） | 1e-5 |
| S dev 重放 vs 已存 canonical 预测 | max\|Δpred\| 3.81e-6 | 1e-5 |
| S dev MAE | 0.12694209793533082 | vs 0.12694 ±1e-4 |
| O dev MAE | 0.11610291503623012 | vs 0.11610 ±1e-4 |
| S raw vs canonical（dev） | 9.54e-7 | 1e-5 |

### 1.3 Phase B 官方输入与守恒（`valid_extract_audit.json` / `test_extract_audit.json`）

- 行对齐（标签无关）：两个 split 全 1000 分子，原子数不匹配 0、`dict_atom == x` 不匹配 0。
- 全量守恒审计（`extract_batch_bucket` 内建门 + 存储块独立重建）：unary/pair 一/二阶矩重建 ≤7.0e-7；count 重建 2.4e-7；masked count blocks == 0；pair 总数 = n(n−1)/2 缺口 0；桶边际包含旧边际 ≤1.4e-13（valid）/5.7e-14（test）。
- 批组成不变性：64 分子拆 4 批重提取 max float 偏差 4.8e-7（valid）/9.5e-7（test）（batch-shape float32 舍入；结构性字段完全一致）。
- C6_MASK 语义：masked count blocks 全 0，未变。
- y 隔离：提取接口传 `ys=zeros`，断言提取记录 `.y == 0.0`；预测文件明确 `y_in_file=False`；官方 y 仅在 score 阶段读取一次/ split。
- checkpoint SHA256 前后相同（determinism）。

### 1.4 S-bucket 加和与 gauge（官方 split 全分子，`_verify_contrib_file` 从保存的 npz 独立重建）

| 检查 | valid | test | 容差 |
|---|---:|---:|---|
| `pred = bias + Σ(α+δ) + Σγ + g_ctx + g_topo` 残差（canonical） | 1.65e-6 | 9.25e-7 | 1e-5 |
| 同式（raw） | 1.63e-6 | 8.99e-7 | 1e-5 |
| `α − α_raw == offset[type]` | 1.04e-7 | 1.16e-7 | 1e-5 |
| `δ_raw − δ == offset[type]` | 1.19e-7 | 1.19e-7 | 1e-5 |
| 逐分子 canonical/raw unit-sum 平移抵消 | 1.79e-7 | 2.67e-7 | 1e-5 |
| 同类型 α 组内最大散布（α 为共享逐类型标量） | 0.0 | 0.0 | 1e-5 |
| γ / g_ctx / g_topo / bias 在 raw vs canonical 下逐位相同 | 是 | 是 | exact |
| GLOBAL_UNK 使用单元数 | 0 | 0 | — |

## 2. 真实结构贡献的外部泛化审计（`attribution_integrity.json` / `subgroup_analysis.json`）

### 2.1 跨分子共享与类型覆盖

- 同类型 α 完全一致（官方两 split 组内散布 = 0）；fit-known 类型出现 417/746（valid）、437/746（test）。
- UNK 路由：valid 484/5520 单元（446/1000 分子）、test 446/5539（417/1000 分子）；全部落入 fit-only UNK 桶，无词表扩展，无 GLOBAL_UNK。
- UNK 分子明显更难：valid F/O/S = 0.191/0.202/0.224（无 UNK 0.086/0.086/0.092）；test 0.146/0.157/0.171（无 UNK 0.091/0.082/0.092）。S−O 差 UNK 组 +0.021（valid）/+0.014（test）略大于非 UNK 组 +0.006/+0.010——与 v3d dev 的同质性结论一致（不扩大）。
- 关系贡献：γ 逐分子和 valid mean 0.042 / std 0.236 / |mean| 0.166，test 0.029/0.208/0.154；直接键连单元对（bucket-0）关系 valid 4520、test 4539。γ 与单元通道存在统计相关（v3d/v3b 已披露），不称独立解释率，不作化学因果解释。

### 2.2 子群（同一批分子配对比较；n 全部给出）

| 子群 | n (valid/test) | F valid/test | O | S | S−O valid | S−O test |
|---|---:|---:|---:|---:|---|---|
| has_UNK | 446/417 | 0.191/0.146 | 0.202/0.157 | 0.224/0.171 | +0.021 [+0.007,+0.037] | +0.014 [+0.001,+0.027] |
| no_UNK | 554/583 | 0.086/0.091 | 0.086/0.082 | 0.092/0.092 | +0.006 [−0.000,+0.012] | +0.010 [+0.001,+0.019] |
| y ≤ −3 | 65/77 | 0.523/0.203 | 0.613/0.239 | 0.689/0.297 | +0.076 [+0.005,+0.163] | +0.058 [+0.012,+0.105] |
| y > −3 | 935/923 | 0.106/0.107 | 0.105/0.103 | 0.113/0.111 | +0.008 | +0.008 |
| ring_units ≥ 2 | 883/887 | 0.106/0.106 | 0.110/0.105 | 0.122/0.116 | +0.012 [+0.005,+0.020] | +0.011 [+0.004,+0.018] |
| ring_units == 1 | 116/103 | 0.334/0.167 | 0.343/0.172 | 0.358/0.178 | +0.015 [−0.008,+0.039] | +0.006 [−0.020,+0.032] |
| cyclomatic ≤ 3 | 787/792 | 0.131/0.110 | 0.134/0.106 | 0.149/0.119 | +0.015 | +0.012 |
| cyclomatic ≥ 4 | 213/208 | 0.138/0.128 | 0.152/0.140 | 0.157/0.149 | +0.005 [−0.013,+0.028] | +0.009 [−0.009,+0.030] |
| atoms 11–20 | 265/281 | 0.147/0.146 | 0.149/0.140 | 0.166/0.156 | +0.017 [+0.004,+0.029] | +0.015 [−0.002,+0.032] |
| atoms > 20 | 734/719 | 0.128/0.102 | 0.134/0.103 | 0.145/0.113 | +0.011 | +0.010 |

- 尾部（y≤−3）仍是最大缺口；S 在 test 尾部 0.297 首次明显优于 F 的 0.203 之下……准确说：尾部所有模型都差，S 相对差 +0.058（test）。valid 尾部数字被 y≈−20 离群放大。
- 复杂环（cyclomatic ≥ 4）上 S−O 的 CI 跨零（valid/test），与 v3c 记录的 0-1 环小分子弱项不同——本轮 ring_units==0 的 n 只有 1（valid）/10（test），0-1 环小分子已不再是主要缺口。
- 不根据子群调参；以上只作预先声明的辅助分析。

## 3. 新增 per-bucket moments 的真实使用（冻结敏感性，零训练、标签无关）

对 fit_inner 固定 1000 分子（训练域）与 official valid 1000（结构外推域）置零新增列并观察预测变化（`moment_sensitivity.json`）：

| 扰动（S-bucket） | mean\|Δpred\| fit/valid | p90 | frac>1e-4 |
|---|---:|---:|---:|
| unit_bucket moments 置零（count 保留） | 0.365 / 0.350 | 0.63/0.65 | 1.000 |
| unit_bucket 全置零（含 count） | 0.791 / 0.746 | 1.51/1.47 | 1.000 |
| rel_bucket moments 置零（count 保留） | 0.121 / 0.121 | 0.27/0.26 | 0.995/0.999 |
| rel_bucket 全置零 | 0.147/0.149 | 0.30/0.31 | 0.996/0.999 |

- 逐桶敏感性稳定（unit b0 0.22、b1 0.19、b2 0.08、b3 0.05/0.03、b4 0.01；rel b0–b4 0.03–0.07），train 域与 valid 域同量级 → 新增 moment 数值列被真实消费（若为 no-op 特征 Δ≡0）。O 臂同向（unit 0.33/0.32，rel 0.14/0.14）。
- 关键修正：v3d 的合成任务 B 真值只含 bucket-count 调制（`gamma* = 0.9·bucket0/total`），它验证的是“关系算子恢复已知距离组成调制”，**不能单独证明新增 moment 列被使用**；本敏感性诊断（count 保留时置零 moment 即改变预测）补上了“被使用”这一环，但仍是零训练的诊断，不是对 moment 形式的机制辨识（形式辨识仍只有合成 A/B）。
- 不据此调参、不重训、不用 test 标签设计扰动。

## 4. 有效性与局限

### A. 实现与守恒事实（本次直接验证）
行对齐、逐分子守恒、桶矩重建、批组成不变性、C6_MASK、S 加和/gauge、UNK 路由、checkpoint 前后哈希、y 隔离——全部通过（§1、§2.1）。

### B. 官方 validation/test 冻结预测结果（本次一次性评分）
§0 表与配对 CI。两 split 各正式评分一次；access record 钉住预测文件哈希、官方 split 文件哈希、评分程序哈希与时间；无重跑。

### C. 严格加和的模型归因事实
`pred = bias + Σ(α+δ) + Σγ + g_ctx + g_topo` 在官方两 split 全分子成立（≤1.65e-6）；α 为跨分子共享的逐类型标量（组内散布 0）；α/δ 的 fit-only gauge 约定在 valid/test 上原样保持（未重新中心化）；贡献导出可独立重建。

### D. 合成机制 + 冻结敏感性证据
合成任务 A（attachment）与 B（预冻结桶调制真值）来自 v3d 轮；本轮新增 moment 使用敏感性（§3）。合成证据支持“恢复已知距离组成调制”，敏感性支持“新增列被真实消费”。

### E. 尚未解决的问题
- **单 seed 0**：所有区间反映固定模型在分子样本上的不确定性，不反映训练种子随机性；跨 seed 稳定性未验证（H2）。
- **容量混杂**：S/O 相对历史各 +61,760 参数（两臂对称），无同容量反事实控制；O−F 的“距离桶压缩解释”有对照但无容量匹配消融。
- **表达边界而非信息边界**：S−O ≈ +0.011–0.013 在 dev/valid/test 三个独立分子集上几乎不变，且不随信息恢复变化（v3d）——指向加性读出的表达/优化边界；该结论仍待独立机制实验。
- **化学因果**：γ 的距离桶敏感性 ≠ 化学因果；通道相关不称独立解释率。
- **历史接触**：
  1. KSVD 路线曾读取 official test（`zinc_e2e_dictenv_p1.py::unlock_test_stage`，`official_test_unlock.json`：`project_wide_pristine=false`，"historical unrelated ZINC official-test reads already exist"）。
  2. KSVD 架构族的历史开发以 official valid 为主指标（`tracks/ksvd/STATE.yaml` 的 `*_valid_mae`），因此 Full-Y 架构/超参继承了历史 valid 反馈。
  3. CSCL-v3d 本身的训练/分析从未加载 official valid/test（`train_{S,O}_bucket_s0.json` 的 `official_valid_loaded=false` / `official_test_loaded=false`；encoded_valid 缓存 audit 同）；本轮全部拟合对象（词表/标准化/τ/κ/gauge）来自 fit_inner 7200。
  → 准确表述：**“CSCL-v3d 的方法/模型/checkpoint 在本轮读取 official validation/test 之前已冻结；本次是冻结版本的终点评估”，而非“全项目首次官方盲测”。**

### 训练预算披露（避免跨论文误比）
- backbone（Full-Y）与 reader heads 的标签监督均只用 fit_inner 7200（raw y）；official train 10000 未全量使用。
- checkpoint 规则：F = monitor top-5 soup（历史规则）；O/S = best-monitor 单检查点（800 内部 monitor，训练期不读 dev）。两者规则不同，比较时已披露。
- 与论文 benchmark 比较只在相同 split、相同训练预算、相同评估指标下才有效；本报告数字仅适用于该冻结版本。

## 5. 结论（对应任务的最终问题）

> CSCL-v3d 冻结模型在官方 ZINC 上的表现：F 0.13295 / 0.11410（valid/test），O 0.13789 / 0.11349，S 0.15048 / 0.12511。S 相对 O/F 的预测代价在 test 上为 +0.0116 / +0.0110（相对 ~+10%），与内部 dev 的 +0.0108/+0.0105 几乎不变；预注册 strong_support 带成立，O−F 在 test 上落入预声明等价区间。全部守恒、加和、gauge 与共享类型不变量在未见结构上成立；新增距离桶矩列被模型真实消费。因此，**以较小且稳定的预测代价实现模型内生、跨分子共享的结构单元与关系贡献**在本冻结版本上获得官方 validation/test 的独立支持；跨 seed 稳定性、容量匹配归因与化学因果仍是未决问题，且本次结果不得再用于任何下一版配置的“test 独立”声明。

## 复现

```bash
cd tracks/structure-contribution/code
uv run python eval_cscl_v3d_official.py freeze     # 哈希 + fit-only 交叉核对 + 重叠审计
uv run python eval_cscl_v3d_official.py replay     # 训练数据回放（特征入口 + F/S/O）
uv run python eval_cscl_v3d_official.py extract --split valid   # 预测落盘 + 全部工程门（y 不读）
uv run python eval_cscl_v3d_official.py extract --split test
uv run python eval_cscl_v3d_official.py score --split valid     # 一次性评分
uv run python eval_cscl_v3d_official.py score --split test
uv run python eval_cscl_v3d_official.py analyze    # 归因完整性 + 子群 + 矩敏感性
uv run python eval_cscl_v3d_official.py judgment   # 预冻结判断带
```

（`--limit N` 为工程冒烟，只写 `*_smoke` 文件，永久不会覆盖正式产物；本轮冒烟在正式提取前对两个 split 各执行过一次，未评分。）
