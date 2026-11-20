# PROTOCOL — 结构词 / 局部句 / 整图路线判定（`word_sentence_port_route_v1`）

Status: **frozen before any formal training run.** 代码提交、机器配置、结构输入 hash
与 Stage-A 检查结果都记录在 `run_manifest.json` 和每个 run 的 `run.json` 中。
本文件以外的每个分支规则都来自任务书 `word_sentence_port_research_agent_prompt.md`
（v2026-11-20/v1），不做结果驱动的修改。

## 0. 研究问题

1. 词内容用 ID、CODE（显式小图）、WGINE 哪种更合适？
2. 真实连接位点布局（`R(D;p,q)`）是否比只有词与邻居内容更有用？
3. 真实结构词是否比原子字符（CHAR）或词袋（BAG）提供价值？
4. 这条路线是否具备继续投入的预测水平？
5. 如有正信号，句之间的关系（PARA_REL）是否优于相同大小的逐句容量（PARA_CAP）？
6. 哪些事实支持“模型使用了结构词”，哪些仍不能支持“化学贡献规律”？

## 1. 数据、身份与评价

* 主任务 `ogbg-mollipo` official Train/Valid/Test = 3360/420/420；条件外推任务
  `ogbg-molesol` = 902/113/113。没有第三个数据集。
* 原始图身份：`tracks/structural-query-learning/data/ogbg_{mollipo,molesol}_sq-pilot-cache-v1.npz`，
  经 `sq_common.EXPECTED_CACHE_SHA256` 逐字节校验（`rd_common.dataset_cache`）。
  只用 official row id 与 official split；9 列原子、3 列键原始属性全部保留，
  不改盐/多分量/立体化学，不删难例。
* 目标按 Train mean/std 标准化，报告恢复原单位 RMSE（主）与 MAE（辅）。
  Lipo: mean 2.16216, std 1.21154；ESOL: mean −2.86688, std 2.06672。
  ⇒ Lipo `delta_task=0.01212`、`delta_port=0.00606`、`band=0.04846`；
  ESOL `delta_task=0.02067`、`delta_port=0.01033`、`band=0.08267`。
* 字典、合并、标准化只用各任务 Train。每 epoch 评估 Train/Valid；
  best-Valid RMSE 选 checkpoint（严格改善，平局取更早 epoch），同时保存 final
  与 last-20 摘要。
* **Test policy = `TERMINAL_DEVELOPMENT_TEST`**：训练对象中没有 Test loader，
  不逐 epoch 评分 Test；只在全部分支结束后对已冻结的 finalist 一次性评分。
  它仍是长期已暴露的开发 Test，不称 blind/fresh confirmation。
* 参考量 `sigma = Train target std`；`delta_task=0.01σ`、`delta_port=0.005σ`、
  `band=0.04σ` 只是本轮预算分配参考，不是显著性/等价性/化学意义阈值。

## 2. 词型与实例

* 词型 = 1–6 原子连通带类型子图模板。原子标签只用官方 OGB 列 `[0,3,7]`
  （`atomic_num, formal_charge, is_aromatic`，运行时按列名核对）；内部键标签只用
  列 `[0]`（`bond_type`）。其余 6 列原子属性和 2 列键属性不丢弃，保存在实例记录中。
* 词型同一性 = 彩色图的 **exact canonical serialization**：在全部 ≤6! 个原子置换上取
  带标签序列化的字典序最小者（原子标签 18 位 + 键矩阵上三角 21 位）。SHA-256 只作索引，
  同一性以完整 canonical 串比较为准，不用 WL hash。
* 词实例保存：`word_id`、模板位置→原始 atom id、所属原子的全部 9 列、内部键全部 3 列、
  所有跨词物理键（两端位置 `p/q`、两端原始 atom、全部 3 列键属性）。同一原子可连多条跨词键。

### 2.1 Train-only 合并（唯一算法）

从每个 Train 分子的单原子 partition 起，逐轮：

1. 枚举当前相邻两个 piece 的并集（总原子数 ≤6，保留原图中这两块原子之间的**全部**内部键，
   允许环边，无 bridge/共轭限制）；
2. 对并集模板做 exact canonicalization，按**不同 Train 分子的 document frequency** 计数；
3. 取 df 最大的候选，要求 `df ≥ max(3, ceil(0.002·N_train))`；平局按较小原子数、
   再按 canonical 串排序。候选模板已在词表时允许继续合并但不增词数；若某候选不改变
   partition 则跳过它取下一个；
4. 在每个分子中合并该模板的**互不重叠**实例；冲突消解顺序由**完整原始属性图**
   （9/3 列 + pynauty `canon_label`）的规范原子序决定，禁止按原始 atom id 决定；
5. 记录模板、合并顺序、df 与实例数。到 128 个不同的多原子词，或无合格候选，
   或 512 轮上限时停止。

用于 N1/ATTRPAIR 与 §12.2 的布局定义都需要「实例的模板位点集」，位点集由词实例的
真实跨词键端点决定。

### 2.2 结构扩容（唯一一次，接入任何任务标签/Valid 误差之前）

若 Train 多原子词覆盖原子比例 <30%，或「有两个不同外部连接位点、且该词型在 Train
实际出现过 ≥2 种 exact 双位点布局」的实例 <200，则继续同一过程至 256 个多原子词；
否则冻结在 128。判定只看结构，不看 Valid。

### 2.3 Valid/Test 重放

Valid/Test 只按冻结合并历史执行同一合并；未知结构保留为已知小 piece，最终为单原子，
不增补测试词。单原子基础词的标签走官方类别的因子化 embedding（每字段 4 维），
不把未见组合全部映射成同一个 UNK；若某单字段类别根本没在 Train 出现，该 embedding
未训练的事实单独报告。

## 3. 三个词内容编码器（都为 64 维）

三者只编码词型本身；实例属性通道与接口通道完全相同；编码器 embedding 与共同属性通道的
`RawAtom`/`RawBond` embedding 分开命名、独立参数。

* **ID**：每个训练多原子词一个 64 维向量 → `64→64→64` SiLU MLP + LayerNorm；
  单原子基础词用三字段各 4 维 embedding 拼接 `12→64→64`。
* **CODE**：模板 → 固定 tensor：三字段 atom embedding 各 4 维（6×12）、内部键类型含
  no-edge 的 4 维 embedding（6×6×4，no-edge 固定为零）、6 位 atom mask、
  7 位原子数 one-hot ⇒ `229→64→64`，SiLU + LayerNorm。
* **WGINE**：模板上两层 GINE，width 32；三字段 atom embedding 求和、bond type
  embedding 32；每层 `32→64→32` MLP + SiLU + residual + LayerNorm + dropout 0.1；
  pool `[sum32, mean32, log1p(n)]` ⇒ `65→64→64`。

新模型总参数上限 500,000。若超限，只按统一下调句层宽度 64→48 的后备执行一次。
编码器比较属于这三个具体方案的比较，不宣称隔离了容量。

## 4. 实例属性、接口关系与句计算

* 共同实例通道：9 列原子各 embedding 4→concat 36，sum/mean；内部键 3 列各 4→concat 12，
  sum/mean；再拼 `log1p(n_i)`、`log1p(boundary_degree_i)` ⇒ `98→64→64` SiLU/LN 得 `a_i`。
  无内部键时键池化为零。**已知限制**：该通道池化了非端口内部属性的位置。
* 位点关系 `R(D;p,q)`（15 维，对称）：`p=q` 标志、最短路 one-hot(0..5)、
  `(A^r)[p,q]` r=1..5 用 `log1p(v)/log1p(6^r)` 缩放、两端模板内部度（无序）、
  两端标签相同标志。另有 exact 双位点 canonical key 作结构审计身份，不作连续特征。
* 每条跨词物理键给词 i 一条有向 incidence α（自身位点、邻词、邻词位点、键 3 列、
  两端原子 9 列）。同一邻词多条键保留多条 incidence。
* 对 incidence pair α<β 记录：是否同一邻词、两个邻词之间是否存在原始跨词键与 5 类
  bond type 计数（7 维 pair 字段）。
* `m_α = MLP_msg([e_j,a_j,own_atom36,neighbor_atom36,bond12]) → 64`
* `u_i = MLP_self([e_i,a_i,g])`；`v_i = mean_α MLP_neighbor([e_i,a_i,m_α,g])`；
  `w_i = mean_{α<β} MLP_pair([e_i,a_i,m_α+m_β,m_α*m_β,R,sentence_pair_fields,g])`；
  无 incidence / 少于 2 条 incidence 时相应项为 0。`g=[log1p(N_atoms),log1p(N_physical_bonds)]`。
  所有普通 MLP 为两层 `input→64→output` + SiLU，隐藏 dropout 0.1；
  `MLP_neighbor`、`MLP_pair` 的最终线性层零初始化。
* `F_std(G) = b + Σ_i (n_i/N_atoms)·(u_i+v_i+w_i)`；质量权重和为 1，每个中心词只读出一次。
  原始单位贡献：`intercept=mu+sigma*b`，各项乘 sigma，不给每个词重复加 mu。

## 5. 对照

* **ERASED**：只把 `R(D;pα,pβ)` 换成该模板中与两端 intrinsic atom labels 相容的所有
  有序位点 pair 的 `R` 均值（允许相同位点，候选规则固定，不参考实例实际位点）。
  其余张量、参数、初始化、消息、属性、邻词类型、一阶词邻接一律不变。
* **BAG**：用 CODE 的 `e_i` 与同样 `a_i`，`z_i=MLP_bag([e_i,a_i])`，
  `[Σ mass_i z_i, Σ z_i]` 拼 `g` ⇒ `130→64→1`；不读 incidence / R / pair。
* **CHAR_FULL**：每个原子一个词实例，复用 FULL_CODE 的全部句公式（模板 1 原子，R 恒定）。
* **GINE_REF**：`sq_gine.py` 的 5 层 hidden128 BN/residual 强整图参考，degree 0..15
  (`≥16→16`)、min-ring 0/3..12 (`≥13→13`)，9/3 列用官方 OGB 常量，统一 train/valid-only 循环。

## 6. 训练配方 R0

seed 决定分子顺序；batch 128 原始分子；MSE on full `F_std`；AdamW peak 1e-3、wd 1e-5；
5 epoch 线性 warmup 1e-4→peak，随后每 optimizer-step cosine 到 1e-5；clip 5；FP32；
TF32 off；无 AMP/EMA/SWA/预训练。

Horizon：Stage-A 短计时（2 warm + 5 计时 epoch，full Train，不评分 Valid/Test）取最慢
实测每 epoch 时间，投影 `epoch_seconds·(1+eval_allowance)·E·1.25`；`E=360`，若 >24 分钟
则全部臂共同改 240，再 >24 分钟则共同 160。E 冻结后不按某臂结果改动。
若 160 epoch 仍 >30 分钟：标 `COMPUTE_INFEASIBLE_AT_THIS_IMPLEMENTATION`。

参考 LR 校准（唯一例外）：Lipo 的 GINE_REF seed0 Valid 若差于 0.6091 的 1.10 倍，
允许补跑 peak 3e-4 的原参考 LR（其他不变），以 Valid 选参考 LR 并记录；ESOL 触发值 0.8587。
最多两个额外参考 run。

OOM 只按 128→64→32 减 batch 并用累积梯度保持有效 128；无法保持时记录 batch 改变。

## 7. 阶段与分支（全部事前定义）

* **A**：数据保真、内容探针、接口探针、最小训练健康性（结果见 §10）。
* **B**：Lipo seed0 七臂：FULL_CODE/ERASED_CODE → GINE_REF/BAG → FULL_ID/FULL_WGINE → CHAR_FULL。
  令 L = 三种 FULL 中 Valid RMSE 最低者；差在 `0.002σ` 内时优先 CODE，其次 ID，最后 WGINE。
  若 L 不是 CODE，再补一个同编码器的 ERASED seed0。
  * **P**：L 不差于 `GINE_REF+band`，且（`gain(L,BAG) ≥ delta_task`）或
    （`gain(L,ERASED_L) ≥ delta_port` 且 L 不差于 `BAG+delta_port`）；去掉对该差值贡献最有利的
    一个 Valid 分子后方向仍需为正。
  * **B**：L 不差于 `GINE_REF+2·band` 但非 P → 一次 seed1 匹配确认，之后 P 或 N。
  * **N**：其余 → 只做一个修复族 N1/N2/N3 或 N4。
* **C**：P 分支 seed1（FULL_CODE/FULL_ID/FULL_WGINE/ERASED_L/BAG/GINE_REF）+ seed2
  （FULL_L/ERASED_L/BAG/GINE_REF）。核心确认：L 相对 BAG 或 ERASED 三 seed 平均增益
  ≥ ½ 对应 delta、≥2/3 seed 同向，且 L 平均不差于 `BAG+delta_port`、`GINE_REF+band`。
* **N1**（若 CHAR_FULL 优于 L ≥ `delta_task` 且多原子词覆盖 ≥30%）：ATTRPAIR，
  `J_i=mean_{a<b∈word} MLP([z_a+z_b,z_a*z_b,R(D;a,b)])→64`，`a_i←a_i+Linear64(J_i)`（末层零初始化），
  ERASED 版同步平均 R；seeds 0/1。
* **N2**（若有具体优化不足证据）：唯一 R1 配方 peak 3e-4、wd 1e-3，其余与 E 不变；seeds 0/1。
* **N3**（若 CODE 在 Train 与 Valid 都差于 ID 或 WGINE ≥ `delta_task`）：CODE_ROOTMEAN，
  每个模板原子加 root 标记做 rooted exact canonicalization，共享 CODE MLP 后对根取 mean，
  229 维上再加 6 位 root mask；seeds 0/1。
* **N4**：无明确可修复证据 → 不发新训练，给有边界的阴性结论。
  N1/N2/N3 两 seed 都比原 L 改善、相对匹配 ERASED 保持正向、距 GINE_REF 均值 ≤ `band`
  → `RESCUED_CANDIDATE`，余量只把该族核心对照补到 seed2。
* **D1**（P 核心确认后才做）：ESOL 重学自己的词表，内容编码器固定为 Lipo 选定的 L，
  跑 FULL_L/ERASED_L/BAG/GINE_REF seeds 0/1。
* **D2**（P 核心确认后）：`h_i=MLP([e_i,a_i,mean m_α,mean pair hidden])`，
  两个新臂共用 16 维调制 `r_i=Σ_j w_para[j]·tanh((U h_i+V g_i)_j/√2)/√16`，`w_para` 全零初始化。
  `PARA_REL`：`g_i` = 词关系图距离 1 或 2 的**不同中心词**的 `h` 均值（唯一二跳索引，不按路径重复计数）；
  `PARA_CAP`：`g_i=h_i`。两臂 seeds 0/1，另加局部 L 作参照。

## 8. 评价与解释

* 逐 seed 报告 best-Valid 对应 Train/Valid/Test（Test 仅终局）、final、last20；
  有效参数、每 epoch 时间、CPU 准备时间、峰值显存/RSS、每分子推理时间。
* 核心差值做 1000 次按 scaffold 成组的配对 bootstrap（seed 1703）；列出独立 scaffold 数；
  seed 方差另列。CI 跨 0 不等于等效，不为 CI 追加 seed。
* §12.2 两个结构集合（已知词；已知内容的新位点布局）只按冻结结构定义，在 Valid 与终局 Test
  报告分子/scaffold/词型/布局数与整分子误差；该子集不用于重选模型。
* 从中间项恢复完整 `F`（raw 误差 ≤1e-5）；在 8 个固定 Valid 案例（4 个结构支持案例 +
  4 个最大误差案例，按 row id 选）展示词、映射、端口、u/v/w（可选 r）与 y；
  对冻结 L 做 R→均值干预，报告预测与误差变化，与 ERASED 重训收益分开。
* 附录：复用旧固定真实编辑目录的 `E_T`/`E_HH_T`，对终局完整 `F` 做 B−A 差值误差，
  与 GINE_REF 比较；`E_STRICT` 两对只列案例。此项不选择分支。

## 9. 预算与调度

T0 = 2026-11-20T16:09:55Z（单调时钟记账）。硬截止 T0+8h；GPU 硬上限 16、软上限 14；
T0+6h40 停止启动训练，T0+7h15 前结束/取消所有训练；≤32 个正式训练单元；两卡并行时
按 2 GPU·h/h 计。每个 worker 接收绝对截止时间并自保存。阶段参考：A ≤1.5h/0.4 GPU·h、
B ≤3.2 GPU·h、C ≤4.8、D ≤4.2、E 最后 ≥45min ≤0.3。

## 10. Stage-A 检查（本轮实际结果，全部记录在 `results/probes_mollipo.json`）

| 检查 | 结果 |
|---|---|
| 词/实例/内部键/跨词键还原原始 9/3 列图（Lipo Train+Valid） | 通过：原子恰好一次、内部键无重复、内部键 + 唯一跨词键 = 全部物理键 |
| 32 分子 + 特殊样本重编号 / 拆分 batch / 单分子 | 通过：max abs pred diff ≤ 4e-8（tol 1e-5），词型重编号不变 |
| 内容探针（三棱柱 vs K3,3） | exact 词身份不同；CODE/ID 可拟合两标签；WGINE sum/mean 表示完全相同（预期表达边界） |
| 接口探针（同 6 环 + A/B，环内距离 1/2/3） | FULL train/dev RMSE 3.1e-4（未见 A/B 组合）；ERASED 等价输入组预测完全相同（spread 0.0），ERASED dev RMSE 0.276 |
| 训练健康性（真实 Train 32 分子，60 步） | loss 1.552→0.049，梯度全部有限，`pair_R` 梯度非零，pair head 梯度非零 |

## 11. 冻结身份

* 配置：`configs/ws_v1.json`（sha256 记录在 freeze 与 run manifest）。
* 结构输入：`data/<dataset>_words_ws-word-route-v1.npz` 的 **content hash**
  （与 zip 时间戳无关）写入 `<dataset>_..._freeze.json`；`load_prepared` 在每次训练前校验。
  Lipo `42aa4fc9…`，ESOL `0ca99990…`（完整值见 freeze 文件）。
* 模板表：Lipo 148 个模板（128 多原子词 + 20 单原子基础词），ESOL 144（128+16）；
  合并历史、df、provenance 与 exact canonical 串全部写入 freeze。
* 结构扩容判定：两个数据集都在 128 后满足覆盖与双位点支持阈值，`triggered=false`，
  `LAYOUT_SUPPORT_OK`（Lipo 覆盖 0.931、双位点支持实例 4888；ESOL 覆盖 0.898、380）。

---

## 附录 A：执行记录（训练全部结束后追加；不修改 §0–§11 的任何规则）

**A1 实际执行顺序与时刻（UTC）**

| 时刻 | 事件 | 产物 |
|---|---|---|
| 16:09:55 | T0（预算起点） | `configs/ws_v1.json` `budget.t0_monotonic_wall` |
| 16:35–16:40 | `prepare` 两数据集（Train-only 词表 + 冻结合并历史） | `analysis/structure_freeze_*.json` |
| 16:41–16:45 | `timing`（FULL_CODE/FULL_WGINE/GINE_REF，2 warm + 5 timed） | `results/timing_mollipo.json` → **E = 160 冻结** |
| 16:46–16:48 | Stage-A 探针（保真/重编号/内容/接口/健康性） | `results/probes_mollipo.json` |
| 16:52–17:20 | Stage-B 八臂（Lipo seed0） | 8 个 run |
| 17:17–17:40 | N2 修复族（R1 配方，FULL/ERASED × seed0/1） | 4 个 run |
| 17:56–18:01 | 分析 + 终局 Test（finalist + GINE_REF 各一次） | `analysis/*`、`results/terminal/*` |

**A2 与 §7 的偏差与补充（全部为“补齐/修复”，未新增科学假设）**

1. **`ERASED_WGINE`（R0）在 N 判定之后补跑**。§7 要求“若 L 不是 CODE，再补一个同编码器的 ERASED
   seed0”。L 只有在八臂全部完成后才能确定（= WGINE），因此该 run 晚于分支算术执行；
   它只用于 §12.1 的 `L/ERASED` 报告与端口结论，**不参与任何分支判定**（P/B/N 的算术不依赖它，
   见 `finalist_manifest.json` 的 `repair_checks`）。
2. **Stage-A 探针重跑一次**：01:31 巡检发现 `results/probes_mollipo.json` 缺失（本地清理误删），
   用同一冻结代码与配置重跑（`ws-probes-rerun-01`），数字与首轮一致（§10）。属工程检查，不计正式单元。
3. **终局 Test 前两次脚本崩溃**：`ws-terminal-finalist`（run record 缺 `run_dir` 默认值）与
   `ws-terminal-finalist-2`/`ws-terminal-diag`（Test 侧 support-subset 需要现场构建的数组）。
   第二次崩溃发生在已算完 Test 预测、尚未写盘的位置；数值未写出、未被读取、未用于任何选择，
   且同 checkpoint 的 eval 确定性（无 dropout，BN 用 running stats），第三次运行数字与之一致。
   披露记录见 `REPORT.md` §9 与 `run_manifest.json` 的 failed jobs。

**A3 §12.2 子集定义**：本 PROTOCOL 早期版本未收录该节，本轮按任务书 §12.2 原文实现
（已知词 / 已知内容的新位点布局），只在 Valid 与终局 Test 做描述性报告，不用于重选。

**A4 horizon**：360 epoch → 40.3 min、240 → 26.9 min 均超 24 min 上限，按 §6 规则全体冻结 **E = 160**
（17.9 min 预估）。实测 FULL_CODE 543 s、FULL_WGINE 170 s、GINE_REF 157 s（含每 epoch Train+Valid 评估，
两卡并行时受同机 CPU 竞争影响）。

**A5 预算记账**：正式单元 11 + 参考 1；GPU 1.17 GPU·h（软 14 / 硬 16）；墙钟 ~1.9 h（硬 8 h）。

**A6 判定结果**：L = `FULL_WGINE`（Valid 0.76795）→ 不满足 P/B → **N**；N1 不满足、**N2 满足并执行**
（Train 标准化 MSE ratio 4.57）→ `RESCUE_FAILED`（两 seed 均未改善，+0.0147/+0.0626）；
finalist = `lipo_full_wgine_s0`，Test RMSE 0.83606（`GINE_REF` 0.66312）。
总体标签 `NO_TASK_ADVANTAGE_IN_THIS_FAMILY`，分层证据见 `REPORT.md`。
