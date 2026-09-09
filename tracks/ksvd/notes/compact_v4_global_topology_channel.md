# Compact-v4: Local–Global Structural Representation (global topology channel)

> 实验对象：`luyin16-zinc-hierarchical-patch-relation-context-compact-v4-topology-*`
> 基线：Compact-Hybrid-v2（selection 98,549 params / refit 99,613 params；
> valid MAE 0.18415821571176638 @ epoch 56；test MAE 0.1353615）
> 协议：`zinc-context-gap`（seed 0；PyG ZINC subset 官方 split；train-only fitting scope）
> 代码：`experiments/luyin16/zinc_topology_features.py`（feature + cache）、
> `zinc_patch_path_pooling.py`（GlobalTopologyEncoder + unified fusion）、
> `zinc_topology_analysis.py`（分析）；configs：`configs/luyin16/zinc_compact_v4_topology_*.yaml`
> 测试：`tests/test_zinc_path_patch_topology_v4.py`（11 tests，含 permutation invariance）
> 本实验是 **Compact-v2 的 graph-level 补充通道研究**，不是新 patch 表示。

---

## 1. Motivation：fixed-radius 的 global closure blind spot

Compact-v2 以 radius-2 ego-graph 为中心对象：

```
p_v = radius-2 ego graph of v
```

它对每个中心节点 v 表达：局部原子/键结构、局部拓扑、局部化学环境、patch
identity、以及（经 pair 路径）中心间的局部关系。它擅长回答：

```
“我附近长什么样？”
```

但它天然无法显式回答：

```
“沿着整个 graph 走多远才闭合？”
```

**同一个局部 patch 可以同时出现在 6-cycle 和 12-cycle 中**——局部邻域
完全相同，graph-global closure scale 完全不同。这就是本实验要补的缺口：

```
local neighborhood information  !=  graph-global closure information
```

信息缺口审计与 long-cycle 审计（§2）已经给出证据：compact-v2 的残差不是
「缺失的局部信息」（13 个 probe 全 NO-GO，包括 2048D 精确 patch-pair hash），
而是集中在极少数长环分子上的、benchmark target 定义中的 graph-global
cycle 项。

## 2. Evidence（long-cycle audit 的 invariant oracle，已核实）

| fact | 值 | 出处 |
|---|---|---|
| valid 基线 | 0.184158（selection, seed 0） | v2 note / info-gap audit 逐 epoch 复现 |
| 极端负 target 的机制 | target 定义中的 long-cycle penalty（GVAE），非 generic heavy tail | zinc_long_cycle_audit.md |
| residual ↔ 缺失 cycle 项 | valid Pearson 0.755 | audit Table D |
| 3.5% 长环分子承担的 valid MAE 质量 | 27.1%（valid:0172 单分子 11.3%） | audit Table B |
| 最佳 invariant oracle | isotonic on mean max-basis over random orders：**ΔMAE +0.0117**（0.1842→0.1725） | audit Table E |
| target-exact（order-dependent）oracle | +0.0247（其中 53% 是 benchmark ordering artifact，不变量模型不可达） | audit Table E |
| 线性探针 | 全部 NO-GO（linear residual head 反而变差 −0.0175）→ 正确形状是**单调非线性** | audit Table E |
| 局部环条件化（v3） | coarse/typed ring 全 NO-GO（局部环尺度 ≠ 全局环统计量） | v3 note |

因此本阶段只回答一个问题：

> 一个非常轻量、permutation-invariant 的 global topology representation，
> 能否回收 long-cycle audit 中发现的 invariant residual signal？

判据（预注册，同任务书 §24）：

| 判定 | valid ΔMAE vs v2 | valid MAE（基线 0.184158） |
|---|---|---|
| NO-GO | < 0.003 | > ~0.181 |
| Mild | 0.003 – 0.008 | 0.1762–0.181 |
| **GO** | >= 0.008 | <= ~0.1762 |
| Very strong（接近 oracle） | >= 0.010 | ≈ 0.1725 |

## 3. Difference from CIN

CIN 的解法：把高阶对象（ring/cell）变成 **computation graph 里的新节点**，
做 ring/cell message passing / 2-cell propagation。v4 **不做**任何这些：

- 没有 ring node / cell node / atom-ring incidence graph / patch-ring bipartite；
- 没有 ring message passing / cellular message passing / 2-cell 传播；
- 没有高阶 computation graph / ring Transformer；
- patch 编码器、pair 编码器、pooling、center-context、graph_head 全部不动。

v4 的解法：

```
local representation (unchanged compact-v2)
        +
global invariant topology summary (a graph-level vector)
        ↓
unified graph representation
        ↓
single regression head
```

global topology 只是 graph-level representation 阶段的一个**附加向量**，
不进 patch 计算、不产生新计算节点。

## 4. Difference from residual ensemble

主模型**不是**：

```
frozen compact-v2 + standalone cycle predictor → y_v2 + delta_topology
```

主模型是联合训练的单一表示：

```
y_hat = H([z_local ; z_topology])          H = 唯一 regression head
```

- `z_local`：compact-v2 原 graph-level 表示（292D 局部 + 32D global，见 §5）
- `z_topology`：`GlobalTopologyEncoder(topology_raw)` ∈ R^8
- 一个 head、一个 L1 loss、端到端梯度。

frozen+residual 形式仅作为 **diagnostic ablation** 候选（本轮未启用；若
unified fusion NO-GO 而 audit oracle 存在，才允许跑一次 residual head
诊断——见 §Decision）。

## 5. Q1：Compact-v2 当前 graph-level representation 的真实数据流

从 `zinc_patch_path_pooling.py` 逐行核实（selection phase，train-only
vocab 6785）：

```
patch-level processing (per centre node):
  typed_token   : radius-2 canonical typed certificate (pynauty)
                  -> _HybridEmbedding 768x16 full + rank-4 factorized (36,420)
  parent_token  : radius-1 certificate -> parent embedding 32x8 (256)
  patch_cont    : 146D radius-2 shell descriptor (standardized, train-only fit)
  patch_context : 0D (context_radius = 0 in v2/v4)
  -> patch_encoder: Linear(146+16+8=170 -> 64) LN ReLU Dropout Linear(64 -> 48)   (14,192)
       e_patch ∈ R^48 per centre
unary pooling:
  moments(mean, std, log-count) over patch states            -> 2*48+1 = 97D     (per graph)

pair-level processing (every unordered centre pair):
  pair_projection: Linear(48 -> 16, bias=False)                                  (768)
  relation        : 23D shortest-path relation descriptor
                    -> relation_encoder Linear(23->32) LN ReLU Dropout Linear(32->16) (1,360)
  distance_gate   : Embedding(5,16); gate = 1+tanh(bucket)                       (80)
  pair_input      : [p_i+p_j, |p_i-p_j|, (p_i*p_j)*gate, r_ij] (64D)
                    -> pair_encoder Linear(64->64) LN ReLU Dropout Linear(64->16) (5,328)
  center-context  : per-centre per-bucket mean/std/mass of incident pairs (165D)
                    -> center_update Linear(48+165 -> 60) LN ReLU Dropout
                       Linear(60 -> 48) (zero-init residual; patch += update)    (15,888)
  unary recomputed AFTER center-context update
pair pooling:
  moments per distance bucket (5 buckets)                    -> 5*33 = 165D      (per graph)

graph-level:
  global_context  : 62D label-free global feature views
                    -> global_encoder Linear(62->32) LN ReLU Dropout Linear(32->32) (3,136)
  local pooled    = concat(unary 97D, pair 165D) = 262D
  existing global = 32D
  v2 graph rep    = concat(local 262D, global 32D) = 294D
  v2 graph_head   : Linear(294 -> 64) LN ReLU Dropout Linear(64 -> 32) ReLU Linear(32 -> 1)
                    (21,121; total 98,549 selection / 99,613 refit)
```

Q1 答案：local pooled = **262D**（unary 97D + pair-by-bucket 165D），
existing global encoder = **32D**，v2 concat = **294D** → head 294→64→32→1。

## 6. Q2/Q3：topology branch 插在哪，为什么 graph-level？

```
z_local = compact-v2 graph representation (262D local + 32D existing-global = 294D)
z_topology = GlobalTopologyEncoder(standardized topology_raw)     # d -> 16 -> 8
z_graph = concat(z_local, z_topology)                             # 302D
y_hat = graph_head(z_graph)                                        # single head
```

插在 **graph-level representation stage**（pooling 之后、head 之前），而不是
patch-level conditioning（v3 路线）：

1. 待补信号是**每图一个的 scalar 全局量**（closure scale / longest cycle），
   不属于任何单个 patch；把它灌给每个 patch 会制造「哪个 patch 拥有全局量」的
   归属歧义，且 v3 已证明 patch-level 局部环条件化在 valid 上无效；
2. fixed-radius patch encoder 是 v2 的核心资产，本轮 Single-variable 原则要求
   **不动 patch 计算图**，只动 graph-level 融合；
3. audit oracle（isotonic on invariant statistic）本质是 graph-level 单调函数，
   说明信号在 head 输入层级即可表达。

## 7. GlobalTopologyEncoder

```python
topology_encoder = nn.Sequential(
    nn.Linear(d, topology_hidden_dim),   # d = raw width (config)
    nn.ReLU(),                            # nonlinearity 是必须的（audit: linear NO-GO）
    nn.Linear(topology_hidden_dim, topology_out_dim),
)
# config: topology_hidden_dim = 16, topology_out_dim = 8
# activation 选 ReLU（仓库当前风格）；不搜索 activation
```

- 输入只允许 graph topology / permutation-invariant cycle statistics；
- **第一版不含**：atom counts、edge counts、RDKit descriptors、target formula、
  normalized cycle penalty、benchmark-specific label component；
- **禁止输入** target 的 `max(0, L-6)` 归一化 cycle penalty 或其等价量
  （那等于 reverse-engineer target；见 §9/§13）。record 字段
  `topology.target_formula_input: False`。

## 8. Feature definitions（精确、全部 permutation-invariant）

所有值在 **dataset preprocessing** 阶段按 molecule 预计算并缓存
（`results/zinc_topology_cache/`，见 §10）；训练时只做 train-only
standardize + 直接读取 tensor。

| 组 | feature | 定义 |
|---|---|---|
| A | `longest_simple_cycle` L | 最长简单环长度（exact）。**复用 long-cycle audit 已验证的 exact-longest 实现/缓存**（`exact_longest_all.csv`，12,000 分子 status=0 全部 exact）；cache build 用独立的完整 simple-cycle 枚举交叉验证：12,000/12,000 一致 |
| B | `n_cycle_len_3..10`, `n_cycle_len_gt10` | 每长度的 **simple cycle 精确计数**（长度 3..10 各一维 + >10 一维）。枚举 = structural_context 中已测试的 min-node DFS（无上限版本）；ZINC-subset 实测每分子 ≤ 38 个 simple cycles（max 长度 26），无指数爆炸，计数精确、非 basis 近似。cycle 集合是 graph invariant → 计数 permutation-invariant |
| C | `mcb_count`, `mcb_max_length`, `mcb_mean_length`, `mcb_total_length` | `nx.minimum_cycle_basis` 的 summary stats。audit 已证 MCB stats 在 random orders 下稳定（20×20 perms 0 unstable）；cache build 对全部 12,000 分子做 3-perm 鲁棒性检查：**0 分子不一致** |
| D | `cycle_rank` | E − V + C（disconnected-aware） |

raw dims：

| mode | 内容 | dim |
|---|---|---|
| `none` | —（退化 = compact-v2） | 0 |
| `longest` | [L, cycle_rank] | 2 |
| `spectrum` | [L, n3..n10, n_gt10, mcb_4, cycle_rank] | **15** |
| `hinge` | spectrum + [L, L², ReLU(L−3)..ReLU(L−10)] | 25 |
| `capacity_control` | constant zeros（宽度 = 15，与 spectrum 等容量） | 15 |

hinge 的 threshold ladder（3..10）是**通用整数阶梯**，不是 target 的单一
`max(0, L−6)`；不把 threshold 6 单独做成 feature。L 是 invariant longest
（≠ target 用的 order-dependent basis max），hinge 是对「closure scale 到
一定程度后才重要」的通用单调饱和先验的 feature-side 版本。

标准化：`Standardizer.fit` 只 fit 在 official train（selection phase）；
terminal phase 按协议 refit 在 train+valid。capacity_control 全 0 → 标准化后
仍全 0。

## 9. 什么被明确排除（Q5：答案 NO）

主模型输入**没有**：

- `max(0, L − 6)` 或任何单阈值 target 形式（hinge 是通用 ladder，且 L 是
  invariant longest，不是 target 的 basis max）；
- 任何 normalized cycle component / `label_effective_cycle` 值 / target
  formula 的归一化版本；
- RDKit 环统计、原子/键计数、target 派生量。

这些只允许出现在 diagnostic 中，不进入主模型。

## 10. Preprocessing / cache

| 量 | 值 |
|---|---|
| 覆盖 | 12,000 分子（train 10,000 / valid 1,000 / test 1,000） |
| 运行时 | ~21.5 ms/graph（首次 build train ≈ 215s / valid ≈ 23s / test ≈ 21s）；之后直接读缓存 |
| cache 大小 | 3 × CSV ≈ 1.3 MB（`results/zinc_topology_cache/*_topology_features.csv` + `meta.json`） |
| 复用 audit 缓存 | `exact_longest_all.csv`（longest 交叉验证 100% 一致） |
| MCB 鲁棒性 | 3 perms/分子：0/12,000 不一致 |
| 训练期成本 | 0（tensor 直接读取；standardizer 已在 preprocessing fit） |

## 11. Parameter audit

v4 只新增：`GlobalTopologyEncoder`（Linear(d,16)+Linear(16,8)）+
graph_head 输入 +8（294→302 的 input 权重增量）。**没有** head hidden
redesign（Δ 太小，不需要 compensation）。

| mode | enc params | head +8Δ | selection total | refit total* |
|---:|---:|---:|---:|---:|
| v2 / none | 0 | 0 | 98,549 | 99,613 |
| longest (2D) | 184 | 512 | 99,245 | 100,309 |
| spectrum (15D) | 392 | 512 | 99,453 | 100,517 |
| hinge (25D) | 552 | 512 | 99,613 | 100,677 |
| capacity_control (15D zeros) | 392 | 512 | 99,453 | 100,517 |

*refit = train+valid vocab 7051。全部 <= 105k（预算上限 110k 未触及）。
预算 `expected_max_trainable_params: 105000`，parameter_audit 在 run 内强制
PASS/FAIL。

## 12. Training protocol（与 compact-v2 严格一致）

| 变量 | 值 |
|---|---|
| split / seed | PyG official train/val/test; seed 0 |
| optimizer / LR / wd | Adam / 1e-3 / 1e-5 |
| batch / epochs / patience | 128 / 60 / 12 |
| loss / scheduler | L1（MAE）/ none |
| selection | min valid MAE；early stop patience 12 |
| checkpoint / refit | best-state in memory；terminal 时 train+valid refit selected epoch 后单次 test |
| vocab / standardizer scope | selection = train only；refit = train+valid（协议） |
| 变更清单 | **只加** global topology channel（config `topology_mode`）；其余零改动 |

Single-variable 原则：同 loss / epoch / optimizer / scheduler / local encoder /
path pooling。v4-none 配置逐 epoch 复现 v2（见 Results）。

> 执行纪律：全部数值 run 在**空闲机器单进程**下运行（曾观察到 ≥3 进程并发
> 时 CPU 计算出现 run-to-run 非逐位确定的分叉；2 进程并发实测仍与 canonical
> trace 逐位一致——v4-none 2-way run valid=0.18415821571176638 @epoch 56 完全
> 复现 v2）。为严谨起见本表全部结果采用单进程复跑。

## 13. Configs

| config | topology_mode | 内容 |
|---|---|---|
| `zinc_compact_v4_topology_none.yaml` | none | 退化 guard（= v2） |
| `zinc_compact_v4_topology_longest.yaml` | longest | 最小 mechanism test（2D） |
| `zinc_compact_v4_topology_spectrum.yaml` | spectrum | 主 v4（15D） |
| `zinc_compact_v4_topology_hinge.yaml` | hinge | 主 v4 + generic hinge（25D） |
| `zinc_compact_v4_topology_capacity_control.yaml` | capacity_control | 等容量无信息控件 |


## 14. Results（selection，seed 0；全部数值为空闲机单进程确定性运行，见 §12 执行纪律）

| model | params | valid MAE | Δ vs v2 | shuffled MAE | test MAE* |
| ----- | -----: | --------: | ------: | -----------: | --------: |
| v2 (canonical) | 98,549 | 0.18415821571176638 @56 | — | — | 0.1353615 |
| v4-none guard | 98,549 | 0.18415821571176638 @56 | 0.000000 | — | 0.1353615188403055 |
| v4-longest | 99,245 | 0.18568143215967575 @60 | −0.001523 | 0.189623 | — |
| v4-spectrum | 99,453 | 0.18767391041567316 @39 | −0.003516 | 0.412478 | — |
| **v4-hinge** | 99,613 | **0.17006561887910357 @53** | **+0.014093** | 0.408382 | **0.13944621286727488** |
| capacity_control | 99,453 | 0.18425278290727876 @60 | −0.000095 | 0.184253 | — |

*terminal refit/test（refit epochs = 各自 valid-selected epoch：v2 56 / hinge 53；
v2 test 为 09-07 canonical 值）。

确定性记录：v4-none 在 screen（2-way）、terminal ×2（solo）共三次逐位复现
valid=0.18415821571176638 @56 与 test=0.1353615188403055；v4-hinge 的 terminal run
逐位复现 solo screen 的 selection trace（valid=0.17006561887910357 @53）；
capacity_control 的 2-way 与 solo run 完全一致（0.18425278290727876 @60）。
所有跨 run 比较均在同一（空闲、单进程）机器状态语义下成立。

### 14.1 Valid 判定（预注册 §24）

- **v4-hinge：Δ = +0.014093 ≥ 0.010 → “Very strong” 区**（预期 ≈ oracle 0.1725；
  实测 0.170066 甚至略优于 OOF isotonic oracle）。方法论注意：oracle 是 OOF 残差上对
  单一统计量的 isotonic 拟合；v4-hinge 是端到端联合训练（10k 图 + 完整不变量特征集 +
  与主模型共享单一 head），信息量严格更大——超出 oracle 不矛盾，也不构成
  “reverse-engineered target”证据（§9 排除项仍全部满足）。
- v4-longest / v4-spectrum：NO-GO（2D 机制太薄；无 hinge 的 15D raw 计数反而有害）。
- capacity_control：Δ ≈ 0（容量不是 driver；对极端 tail 反而退化，见 §15）。

### 14.2 Official terminal test（frozen winner 的单次 refit/test）

v4-hinge refit/test：valid 0.170066（逐位复现），**test = 0.139446 vs v2 0.1353615
（−0.0041，官方口径下退化）**。

归因诊断（post-hoc：对 frozen winner 的 selection/refit state 在 test 上单次 forward，
非 official、不用于重新选择，只用于定位退化发生在哪个 phase）：

| 模型 | test MAE（test 完全未见/未参与选择） |
| --- | ---: |
| v2 selection model（train-only，valid-selected） | 0.154284 |
| v4-hinge selection model（train-only，valid-selected） | **0.133901**（−0.0204 vs v2-sel；甚至优于 v2 refit 0.135362） |
| v2 refit model（train+valid，official） | 0.135362 |
| v4-hinge refit model（train+valid，official） | 0.139446 |

→ hinge **representation** 在 selection 层面泛化优秀（对完全未见的 test 全分组改善，
§15.1）；官方数字的退化发生在 **refit phase**（train+valid 11k 重训，valid 的极端
tail —— 尤其唯一 excess-6、y=−20.34 样本 —— 进入训练集并主导 L1 梯度）。refit 模型
对 valid 的 in-sample 分组误差呈现同一现象（valid-A：v2 0.0828 vs hinge 0.0922；
C：v2 5.851 vs hinge 4.694），即 refit 的 hinge 模型把容量投向 valid tail 的
in-train 拟合，牺牲 bulk。解释见 §18。

## 15. Subgroup results（valid，label-excess：A=0 n=965 / B=1 n=30 / C=≥2 n=5）

Selection models（per-molecule predictions 来自 run JSON，组内分子平均）：

| group | n | v2 MAE | hinge MAE | Δ(hinge) | longest Δ | spectrum Δ | capacity Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A（ordinary）| 965 | 0.139130 | 0.137762 | **+0.001368 ✓** | −0.005835 ✗ | −0.012768 ✗ | +0.002668 ✓ |
| B（mild）| 30 | 0.466200 | 0.226189 | **+0.240011** | +0.138795 | +0.245945 | +0.096484 |
| C（extreme）| 5 | 7.182416 | 6.067976 | **+1.114440** | −0.011163 | +0.285362 | −1.112734 |

- bulk-degradation 预注册检查（Group A Δ ≥ −0.002）：**hinge +0.0014 PASS**；
  longest/spectrum FAIL（这正是它们 valid 负增益的来源）；capacity_control 的 A PASS
  但 C 显著退化 → 纯容量对极端 tail 无帮助。
- tail（B+C，n=35）承担的 valid MAE 质量：v2 49.9/184.2 = 27.1% → hinge 21.9/170.1 =
  12.9%（B：14.0→6.8；C：35.9→30.3）。与 audit Table B 的 27.1% 完全一致。

### 15.1 Test 分组（label-excess 定义同 valid；test：A=948 / B=44 / C=8，excess max 2）

| 模型对 | A MAE | B MAE | C MAE | overall |
| --- | ---: | ---: | ---: | ---: |
| v2 selection | 0.1366 | 0.3545 | 1.1431 | 0.154284 |
| hinge selection | 0.1247（+0.0119）| 0.2038（+0.1507）| 0.8379（+0.3052）| 0.133901 |
| v2 refit（official）| 0.1279 | 0.2535 | 0.3645 | 0.135362 |
| hinge refit（official）| 0.1341（−0.0061）| 0.2295（+0.0240）| 0.2798（+0.0847）| 0.139446 |

selection 模型：全分组改善（A 也 +0.0119）。refit 模型：tail 增益缩水（test tail 温和，
v2-refit 已把 mild tail 学得不错），而 A 组退化 −0.0061 × 948 = −5.8 mass 超过
B+C 总增益 1.7 mass → 官方净 −4.1 mass。**“Valid GO” 的收益几乎全部来自 valid 独有
的极端 tail；test 没有那种 tail（audit Q12 预言成立），refit 模型在 tail 上的过度拟合
就只剩下代价。**

## 16. Shuffle test（derangement, seed 0, selected best state）

| model | clean valid MAE | shuffled MAE | 结论 |
| --- | ---: | ---: | --- |
| v4-hinge | 0.170066 | 0.408382 | 增益完全依赖真实 topology 对应关系 |
| v4-spectrum | 0.187674 | 0.412478 | 同样依赖（但其 clean 差 → raw 表示有害）|
| v4-longest | 0.185681 | 0.189623 | 几乎不依赖 → 2D 机制未被充分利用 |
| capacity_control | 0.184253 | 0.184253 | 零特征 → 无变化（sanity ✓）|

## 17. Hinge 第一层权重分析（topology_encoder.0.weight，per-feature |w| L1 norm）

前列：mcb_max_length 7.16、mcb_mean_length 3.16、n_len7 2.85、n_len6 2.73、n_len8
2.50、n_len5 2.19、mcb_total 1.97、n_len4 1.95、hinge_10 1.92、hinge_8 1.88。
hinge ladder（hinge_3..hinge_10）权重分布 1.25–1.92，**无单一 threshold-6 尖峰**；
longest / L² 中等（1.45–1.73）。解释克制：模型对「closure scale ≥ ~6–8 + MCB 尺度」
敏感，与 audit 的机制一致；**不**声称模型发现了 target formula（threshold 6 只是通用
ladder 的一档，且 L 是 invariant longest，≠ target 的 order-dependent basis max）。

## 18. Interpretation（设计问题 Q1–Q6 + audit 结转）

1. **Global topology signal 可学吗？** 可学，且需要 hinge 形状（§14：spectrum raw
   −0.0035 vs hinge +0.0141；capacity ≈ 0；shuffle 全消）。
2. **Nonlinear topology encoder 必要？** 是（audit：linear residual head −0.0175，
   isotonic oracle GO；本模型 encoder 含 ReLU）。未单独消融 encoder 非线性（本轮以
   整通道为单变量）。
3. **Spectrum 优于 longest？** 否。15D raw 有害（A 退化 −0.0128）；2D longest 不足
   （B 改善但 A 退化）；25D hinge（longest + counts + 通用阶梯）是唯一显著赢家。
   未隔离“阶梯 vs 维度数”（本轮 single-variable，若进入下一轮可做 hinge-minus-spectrum）。
4. **只是容量？** 否（capacity_control Δ ≈ 0 且 C 退化）。
5. **Ordinary molecules 受损？** selection 层面不损：valid-A +0.0014 ✓，test-A
   （selection model）+0.0119 ✓。refit 模型受损：valid-A in-sample −0.0093 ✗，
   test-A −0.0061 ✗（§14.2/§15.1）。→ 损伤是 refit-phase 现象。
6. **接近 invariant oracle？** valid 0.170066 越过 oracle 0.1725；但 official test
   0.139446 劣于 v2 的 0.1353615。audit Q12 预言（test tail 温和：excess max 2、
   无 excess-6）成立，这里量化了其后果：valid 上可回收的 tail 质量（27.1%）在 test
   上不存在同等规模；refit 为拟合 valid 极端样本而牺牲的 bulk 误差无处对冲。

**Q6 结转（Has the model learned the penalty at all?）**：selection hinge 把 B 组 MAE
从 0.466 → 0.226（减半）、C 组从 7.18 → 6.07（含对 −20.34 样本从 λ≈0 的 +0.45 预测
到可追尾）；refit 后 C 组 in-train 至 4.69。excess ≥ 4 的机制性不可见基本修复。

## 19. GO/NO-GO decision

**Valid（预注册判据 §24）：GO — Very strong。** v4-hinge Δ = +0.014093 ≥ 0.010
（valid 0.170066 ≤ 0.1762，达到 oracle 区 0.1725），bulk 检查 PASS（A +0.0014），
shuffle 全消（0.1701 → 0.4084），capacity 控件排除容量解释，参数 99,613 ≤ 预算。

**Official terminal test：不确认。** 0.139446 vs v2 0.1353615（+0.0041）——frozen
winner 的单次官方 refit/test 为负；post-hoc selection-state 诊断显示表示本身泛化优秀
（test 0.133901，全分组改善，A 也 +0.0119），退化定位于 refit phase（train+valid
的极端 tail 主导 L1 优化，§14.2/§18.5）。

**Round 总判定：valid-GO / official-test-negative（分裂）→ 不宣布干净 GO，也不关闭
topology 主线。** hinge 表示的真实性由 selection 层面三重支持（valid 全分组 + 未见过
的 test 全分组、shuffle、容量对照）；阻塞点是 refit-phase 行为，属协议/优化问题而非
表示失效。

## 20. Next step（唯一推荐）

**Post-v4 residual audit 之前，先以 no-test 模式解决 refit-phase 退化**（下一 stage
的 pre-requisite；全部可在 selection-only 协议内评估，不触碰 test）：

1. **稳定性确认**：hinge refit 的 A-group 退化是否 seed-鲁棒（seed ×2–3，solo）；
2. **机制候选（no-test，valid 为诊断）**：(a) tail-aware objective（对极端 |y| 的
   per-molecule loss 封顶/重加权 —— 预注册 pivot 方向）；(b) refit 集 tail 的
   downsampling/截断；(c) refit epochs 的二次选择。gate：refit 模型 in-sample
   valid-A ≥ v2 水平 且 valid tail 增益保留；
3. Gate 通过后才做 terminal 确认；随后才进入 **post-v4 residual audit**（frozen
   hinge 之上还剩什么 —— 本轮不新增任何 cycle feature：evidence 表明 hinge 已达/越过
   不变量 oracle 区，更多 cycle 特征边际为负）。

若 refit 退化被证明不可修复 → 关闭 topology 主线，按预注册转向 tail-aware objective
（在 v2 或 hinge 表示上），并记录“valid-GO 但不转移”的教训：**valid 上被单一极端样本
主导的增益必须先在 no-test 协议里做 refit 稳定性检查，再消耗 terminal 配额**。

## 21. 产物清单

| 文件 | 内容 |
|---|---|
| `results/zinc_topology_cache/` | 12,000 分子 topology 预计算缓存 + meta.json（3 CSV ≈ 1.3 MB）|
| `runs/2026/09/09/*` | 各 variant selection/solo/terminal run（manifest/config/trace/state_dict/per-molecule valid preds）|
| `notes/compact_v4_global_topology_channel.md` | 本文档 |
| `experiments/luyin16/zinc_topology_analysis.py` | 汇总/分组/shuffle/hinge 权重分析 |
| 测试 | `tests/test_zinc_path_patch_topology_v4.py`（11 tests）|

关键 run ids：hinge official terminal `20260909-171303-95a4f542`（valid 0.17006561887910357
@53，test 0.13944621286727488）；hinge solo selection `20260909-165340-96be4349`；
none guard terminal `20260909-172752-4881725e` 与 statedict 复现 `20260909-174717-2b498e06`
（valid/test 均逐位 = v2 canonical）；longest `20260909-164801-30e49a0d`；spectrum
`20260909-164306-c0a568c9`；capacity_control `20260909-165917-b544452b`（与 2-way run
`20260909-161302-97c62f91` 完全一致）；v2 canonical screen `20260907-193612-46c1a12f`、
terminal `20260907-194818-604fa0f5`。

---

## Addendum (2026-09-09): protocol naming fix + multi-seed confirmation (GO)

**Wording correction (do not delete the history above).** 本文档 §14.2/§19/§20 中
的 "Official terminal test" / "official refit/test" 指 **train+valid refit → test**。
Benchmark audit（Dwivedi et al. arXiv:2003.00982v4 + benchmarking-gnns 代码 + CIN
arXiv:2106.12575v2，见 `compact_v4_multiseed_protocol_confirmation.md` §2）确认文献
ZINC-12k 主协议是 **train only → validation selection → frozen checkpoint → single
test**，**没有** train+valid refit。因此：

- 本文档的 "Official terminal test 0.139446" 应改读为
  **secondary refit-robustness test**（内部 robustness 协议）；
- 本文档 §14.2 的 selection-checkpoint test 数字（v2 0.154284 / hinge 0.133901）才是
  **benchmark-comparable** 数字；§15.1 的 "selection 模型" 分组表即 benchmark 口径。

**Multi-seed confirmation (seeds 0–3, serial, bit-verified; 详见
`compact_v4_multiseed_protocol_confirmation.md`).** 在 benchmark 口径下
（frozen best-valid checkpoint 单次 test，train-only fits）：

- v2 test mean **0.146069 ± 0.006177**；hinge test mean **0.136885 ± 0.005552**；
- paired mean improvement **+0.009184**（median +0.007747），**4/4 seeds** 为正；
  valid 方向 4/4 一致（+0.013393，p=0.049）；
- test subgroups（4-seed 汇总）：A +0.0031±0.0103（2/4 wins，sign flips），
  B +0.0722±0.0558（4/4），C +0.3882±0.3017（4/4）；
- seed-0 guard：valid traces 与 refit test 逐位复现本文档 §14 数字；in-run
  selection-checkpoint test 逐位复现 §14.2 的 post-hoc 值
  （0.15428431200794876 / 0.13390139845572413）。

**修正后的 round 判定：GO（benchmark 口径，非 Strong GO）**，取代 §19 的
"valid-GO / official-test-negative（分裂）"表述 —— 分裂来自把 secondary refit
协议当 primary；benchmark 口径下 valid 与 test 同向。seed-0 refit 退化 (+0.0041)
保留为 secondary 现象记录（train+valid refit 稳定性问题，另行研究）。下一步：
**post-v4 residual audit**（以 v4-hinge 为冻结 baseline）。工具：terminal run 现可
in-run 测 `test_with_selection_checkpoint_mae`（flag
`evaluation.selection_checkpoint_test`，commit 0a61b23，additive，默认 off）。
