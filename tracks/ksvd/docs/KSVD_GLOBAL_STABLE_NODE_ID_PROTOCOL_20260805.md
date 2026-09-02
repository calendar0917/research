# Graph-global stable node ID 与 Beam8 记录稳定性协议

> 日期：2026-08-05
> 状态：结果前冻结
> 来源：`docs/luyin/luyin13.txt` 中“尽可能把节点编号固定；相邻 patch 有相对排序”的要求。
> 范围：只研究同一抽象图在 numeric relabel 下的 graph-global identity、patch 记录和 Beam8 重采样稳定性；不改变 Beam8 coverage objective，不使用 graph labels。

## 1. 问题拆分

本协议区分两套坐标：

1. **graph-global stable ID**：回答“这个真实节点是谁”，用于 patch membership、共享节点、transition、stitching 和 residual edge endpoints；
2. **rooted-canonical local slot**：回答“这个节点在当前 patch 中处于哪个局部拓扑位置”，用于 45D adjacency 和 KSVD。

Global ID 不直接进入 45D KSVD vector；local slot 不承担跨 patch 的永久身份。相邻 patch 通过：

```text
local slot_t -> global stable ID -> local slot_(t+1)
```

建立对应。

## 2. 数学边界

若两个节点属于同一个 graph automorphism orbit，纯结构不可能给它们分配可证明重编号不变的不同 ID。本协议不使用 numeric ID 伪造唯一性，而输出：

- stable structural class；
- class 是否 singleton；
- singleton 节点的 unique stable ID；
- non-singleton class 内仍需 opaque/raw handle 才能恢复 labeled graph。

因此必须分别报告 class stability 和 unique-ID coverage，不能只报告一个“编号成功率”。

## 3. Frozen graph bank

正式设置：

```text
graph_bank_seed = 810001
families = regular / small_world / block
n = 50
degrees = 15 / 20 / 25
8 graphs per cell = 72 graphs
permutation seeds = 960101 / 960102 / 960103
cover seed = 970101
Beam8/R1, s=10, o=3, m=1.5
```

先允许 18 图 smoke；正式结论使用 72 图 × 3 permutations。每个 permutation 采用 `new_index -> old_index` 语义并显式映回抽象节点身份。

## 4. Stable-ID branches

### 4.1 INPUT_ID control

直接使用输入数组下标。它能在一次运行内部完成 bookkeeping，但 relabel 后不应稳定。

### 4.2 GLOBAL_WL

初始离散 node signature：

```text
degree
triangle count
k-core number
distance histogram
closed-walk diagonal counts A^2..A^5
```

随后在整图上做确定性 1-WL/color refinement，所有 color vocabulary 均由结构 tuple 排序生成，禁止 numeric ID 进入 signature 或 color rank。

### 4.3 ROOTED_WL

在 GLOBAL_WL 的基础上，对每个候选 root 单独赋 singleton root color，再执行确定性 color refinement。root fingerprint 包含：

```text
initial structural signature
refinement color-count trajectory
final root color
final color class sizes
color-quotient directed edge-count matrix
```

最终 stable class key 为：

```text
(global final color, rooted fingerprint)
```

若某个 GLOBAL_WL color 已是 singleton，则 ROOTED_WL 对该节点使用确定性 shortcut，不再重复计算 rooted fingerprint；因为第一项已唯一，省略第二项不会改变 class 或 rank。只有 GLOBAL_WL tied classes 才运行 per-root refinement。

按 key 的结构字典序生成 class rank。singleton class 的 class rank 即 unique stable ID；non-singleton class 只获得 stable class ID。

本轮不引入浮点 PageRank/centrality 作为硬 tie-break：对无向正则图 PageRank 本身不能区分节点，浮点 eigensolver 还可能引入平台相关 tie。导师建议的 centrality/PageRank 作为解释性 baseline 讨论，不进入冻结主分支。

## 5. Stable-ID relabel audit

对原图和每个 relabeled graph 独立计算 ID，映回抽象节点后报告：

- stable-class exact match rate（所有节点）；
- singleton unique-ID exact match rate；
- singleton-node fraction；
- fully-singleton graph fraction；
- largest ambiguous class；
- ambiguous class count；
- reordered canonical-adjacency exact match。

硬 invariant：GLOBAL_WL 与 ROOTED_WL 的 class exact match 必须为 1.0；否则实现失败。

采用 ROOTED_WL 作为 practical stable ID 需要：

```text
singleton-node fraction >= 0.90
singleton unique-ID match >= 0.999
reordered canonical-adjacency match >= 0.99
```

若 class 稳定但 singleton fraction 未过门槛，结论限定为 stable equivalence classes，不宣称 practical unique ID。

## 6. Frozen-cover record audit

先在原图生成一条旧 Beam8 cover，再把同一抽象 cover 随 permutation 搬运到 relabeled graph。分别用 INPUT_ID、GLOBAL_WL、ROOTED_WL 序列化：

- ordered patch membership IDs；
- unordered patch membership class multisets；
- transition shared-node identity records；
- residual uncovered-edge endpoint records。

报告 exact match rate。对 non-singleton classes，另报 quotient-aware class-multiset match；它不等同于恢复具体 labeled node identity。

预期：ROOTED_WL 的 singleton-covered记录应稳定；若 patch 含 ambiguous class 成员，具体 record 允许失败但 class-multiset必须稳定。

## 7. Beam8 stable-preorder audit

保持 Beam8 的边覆盖 objective、budget、beam 和 restart 不变，只比较：

1. `RAW_BEAM8`：在输入 numeric order 上直接运行旧 sampler；
2. `STABLE_ID_PREORDER_BEAM8`：预注册时先按 ROOTED_WL stable class/ID 重排整图，再运行完全相同 sampler。若正式审计证明 GLOBAL_WL 与 ROOTED_WL 在全部 base/relabel 图上 concrete order 严格相同，则结果报告可选择更简单的 GLOBAL_WL 实现；这属于 matched simplification，不改变实际 sampler 输入。

对 non-singleton class，类内仍按输入 opaque handle 排列，并显式记录该图存在 unresolved ambiguity。不得把该 fallback 称为结构稳定 ID。

原图和 relabeled graph 使用相同 scalar cover seed，并把 cover 映回同一抽象节点身份空间后比较：

- abstract-node exact ordered chain match；
- abstract-node position-wise patch-set Jaccard；
- rooted-canonical 45D vector row match；
- transition-slot-map match；
- edge/pair coverage absolute delta；
- RAW zero-fill RMSE absolute delta；
- connected/single-chain/exact-overlap invariants；
- wall-clock。

另报告 canonical-coordinate replay 与 equivalence-class replay。前者只是 preprocessing + deterministic sampler 的 consistency check；当两边 canonical adjacency 相同时，它不能单独作为 sampler gain 证据。后者用于区分真正漂移与不可辨识节点互换。

`STABLE_ID_IMPROVES_BEAM8_REPLAY` gate：

```text
all invariants pass
mean edge coverage not worse by > 0.01
mean pair coverage not worse by > 0.01
exact-chain match improves by >= 0.50 absolute
rooted-vector row match improves by >= 0.30 absolute
```

若 fully-singleton graphs 明显优于 ambiguous graphs，必须分层报告，不能把对称图失败归因于实现缺陷。

## 8. 总分类

- ID gate 与 sampler gate 均通过：`ADOPT_GLOBAL_STABLE_ID_PREORDER`；
- ID gate 通过、sampler gate 不通过：`ADOPT_STABLE_IDS_FOR_RECORDS_ONLY`；
- 只有 class invariant 通过：`STABLE_EQUIVALENCE_CLASSES_ONLY`；
- class invariant 失败：`FAIL_GLOBAL_STABLE_ID_INVARIANTS`。

无论分类如何，rooted-canonical local slots 保持当前冻结地位；本协议不会用 global ID sort 替换 KSVD local ordering。

## 9. 边界

- 本轮只检验 numeric relabel，不检验加减边后的 perturbation stability；后者是不同问题。
- stable ID 是每张图内部的 canonical identity，不宣称图 A 的 ID 7 与图 B 的 ID 7 有共同语义。
- quotient-aware record 稳定不等于 labeled graph identity 已完全编码；ambiguous class 仍需 opaque handle/sidecar。
- 若 ROOTED_WL 仍留下非 automorphic ties，本轮只把它们保守标为 unresolved；exact whole-graph canonical search 作为结果后的独立 follow-up，不静默加入主分支。

## 10. 实现审查勘误

初版 runner 曾直接用 canonical-coordinate cover 计算 sampler exact-chain gain。严格审查指出：canonical adjacency 已相同时，同 seed 重跑得到 1.0 是合理 consistency 结果，但比较空间不符合上述主 gate，且会隐藏 ambiguous twin 的抽象身份互换。正式结果必须改用 abstract-node 指标计算 gain，并将 fully-singleton / ambiguous 分层；本勘误不改变方法、阈值或不利结果。
