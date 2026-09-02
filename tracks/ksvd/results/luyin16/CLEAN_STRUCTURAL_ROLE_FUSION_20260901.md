# MolHIV clean structural-role fusion：2026-09-01 阶段判定

本轮针对上一版 `shell × degree × cycle` role 的三个混淆做了重新建模：

1. atom attribute 去掉 OGB `degree` 与 `is_in_ring`，避免与结构角色直接重复；
2. 用 topology-only rooted WL 从完整 radius-2 induced ego 自动产生角色；
3. 将纯 `T`、纯 `A`、`T+A`、raw joint、centered binding 和 matched shuffle
   分开评分，导师旧 `S` 只作为最后的性能 overlay。

固定 all-center radius-2、同一 XGBoost、同一 scaffold folds；没有 K-SVD、attention、
Optuna 或 official test。

## 表示

```text
substrate: full all-center radius-2 induced ego
T coarse: shell × induced degree × cycle
T rooted-WL: root/shell/degree initialization + 2 rounds topology-only refinement
A strict: OGB chemistry minus degree and is_in_ring
F factorized raw: T+A + node-role×atom + edge-role×bond
F centered: T+A + [P(role,attr)-P(role)P(attr)]
null: independent within-patch attribute-row shuffle, preserving both patch marginals
```

额外测试了 hashed `(edge role, endpoint element pair, bond tuple)` typed-edge；它只作
高阶融合 ablation。

## 无标签对象审计

在 official-valid 冻结协议抽取的 64/64 个 train/valid 图上：

- exact rooted topology valid patch mass coverage：`0.9483`；
- exact topology valid type coverage：`0.4947`；
- coarse signature 中 `0.97%` 的 key 对应多个 exact topology；
- rooted-WL signature 在该样本中无 exact-topology collision；
- coarse/rooted-WL 的最终图特征均通过 32 图 × 2 次重标号审计。

这说明 exact topology 的高频质量覆盖较高，但长尾类型跨 scaffold 覆盖只有约一半；
直接采用 exact topology/orbit vocabulary 会面临明显 OOV 稀疏性，rooted-WL backoff
比继续追 exact slot 更合理。

## official-train 内部三折

| view | coarse AUC | rooted-WL AUC | rooted-WL − coarse |
|---|---:|---:|---:|
| `T` | 0.6608 | 0.6941 | +0.0333 |
| `T+A` | 0.6759 | 0.6959 | +0.0201 |
| factorized raw | 0.6781 | **0.7327** | **+0.0547** |
| centered | 0.7050 | 0.7257 | +0.0206 |

rooted-WL 内部 gate：

- factorized raw − `T+A`：`+0.0368`，3/3 folds；
- factorized raw − shuffle：`+0.0240`，2/3 folds；
- centered − `T+A`：`+0.0297`，3/3 folds；
- centered − shuffle：`+0.0673`，3/3 folds；
- rooted-WL factorized raw − coarse：`+0.0547`，3/3 folds。

因此 richer structural coordinate 在 official-train 内部有明确价值；上一版 coarse role
确实限制了融合表示。

高阶 typed-edge 为稳定负值：相对 factorized raw `-0.0164`，0/3 folds。它没有
带来更强融合，按快速停止原则关闭。

## 冻结 official-valid（XGBoost seeds 0/1/2）

| view | ROC-AUC |
|---|---:|
| strict attribute `A` | 0.7738 |
| rooted-WL topology `T` | 0.7583 |
| `T+A` marginals | **0.7865** |
| factorized raw | 0.7731 |
| centered | **0.7872** |
| old mixed `S` | 0.7775 |
| `S` + factorized raw | 0.7855 |
| `S` + centered | 0.7864 |

关键 gate：

- factorized raw − `T+A`：`-0.0134`，FAIL；
- factorized raw − shuffle：`+0.0028`，低于 `+0.003`，FAIL；
- centered − `T+A`：`+0.0007`，FAIL；
- centered − shuffle：`+0.0156`，PASS；
- typed-edge − factorized raw：`-0.0048`，FAIL；
- `S+factorized−S=+0.0081`、`S+centered−S=+0.0089`，但二者均未超过
  单独的 `T+A`/centered。

centered 的三个 model-seed AUC 为 `0.7993/0.7864/0.7759`，说明单 seed 的约
`0.80` 不能当成稳定结论；三 seed 均值才是本轮冻结数字。

## 阶段结论

1. **结构编码问题得到澄清。** rooted-WL 明显优于 coarse role，说明完整邻接产生的
   自动角色比 shell/degree/cycle 更合适。
2. **结构—属性依赖真实存在。** 在去掉 degree/ring 重复后，centered true 仍稳定优于
   shuffle；此前信号并非完全由重复字段制造。
3. **依赖不是稳定标签增量。** official-valid 上 centered 没有超过 `T+A`，raw joint
   反而下降；内部三折增益具有 scaffold-specific 成分。
4. **目前最稳的表示仍是 marginals。** `T+A=0.7865`；复杂融合未稳定提高它。
5. **高阶 typed edge 关闭。** 不继续加 path/cycle/attention 来修复。
6. **exact orbit 暂不进入。** rare exact topology 的跨 scaffold type coverage 只有约
   `49.5%`，更高容量会加剧稀疏/OOV，而不是解决已观察到的迁移问题。
7. **K-SVD 仍不返回主线。** 当前瓶颈是融合与标签的跨 scaffold 对齐，不是压缩容量。

准确的一句话结论是：

> topology-only rooted-WL 成功修复了结构坐标过粗的问题，也证明了干净的
> role–attribute dependence；但该 dependence 没有在 official-valid 上形成超越
> `T+A` marginals 的稳定任务增量。

## 停止项与后续

停止扫描：WL rounds/bins、exact orbit、typed path/edge、centered 维度、K/T、attention。

若继续机制研究，新的问题必须改成“哪些 interaction 在 scaffold 间稳定”，并使用
cross-fitted residual 或明确的 task-aligned sparse interaction selection；不能再把全部
joint tensor直接拼接后归因于融合。该方向应另立协议，不在本轮 official-valid 上继续调参。

原始结果：

- [`structural_role_fusion_screen/summary.json`](structural_role_fusion_screen/summary.json)
- [`structural_role_fusion_screen_official_valid/summary.json`](structural_role_fusion_screen_official_valid/summary.json)

