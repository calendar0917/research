# MolHIV task-aligned interaction 与超参数审计：2026-09-01

本轮只回答两个问题：

1. dense structure--attribute interaction 在 fixed XGBoost 下的结论，是否只是没有
   做 Optuna 搜索造成的；
2. 相比直接拼接全部 interaction，cross-fitted、task-aligned 的稀疏 residual 是否能
   跨 scaffold 稳定提高 `T+A`。

协议全程只使用 official-train：三组 scaffold 依次作 outer holdout；每个 outer train
由另外两组 scaffold 构成，并作双向 inner holdout。official-valid/test 均未编码、未评估。

固定 full all-center radius-2 induced ego、topology-only rooted-WL、strict chemistry
和同一图读出。每个 dense view 在每个 outer fold 获得相同的 6-trial Optuna 预算；旧
fixed probe 被显式加入候选。outer fold 使用 3000 train / 1500 valid，最终数字为
XGBoost seeds 0/1/2 的均值。

## 等预算 XGBoost 搜索

| view | fixed outer AUC | tuned outer AUC | tuning gain |
|---|---:|---:|---:|
| `T+A` | 0.6977 | **0.7129** | **+0.0152** |
| factorized raw | 0.7235 | 0.7084 | -0.0152 |
| centered | 0.7306 | **0.7329** | +0.0023 |

相对 `T+A` 的机制 gate：

| comparison | mean delta | fold wins |
|---|---:|---:|
| fixed raw − fixed `T+A` | +0.0258 | 3/3 |
| tuned raw − tuned `T+A` | -0.0045 | 1/3 |
| raw relative-gap change after tuning | **-0.0303** | 0/3 |
| fixed centered − fixed `T+A` | +0.0329 | 3/3 |
| tuned centered − tuned `T+A` | +0.0201 | 3/3 |
| centered relative-gap change after tuning | **-0.0128** | 0/3 |

因此，超参数确实影响绝对 AUC，而且不同表示的最优正则化不同；但它没有“救回”
interaction。相反，旧 fixed probe 对 `T+A` 更不利：调参主要提高了低维 marginals，
raw joint 被明显削弱，centered 相对优势也缩小。

准确结论不是“超参数不重要”，而是：

> 缺少 Optuna 不是此前 interaction 迁移失败的解释；公平调参后，interaction 相对
> `T+A` 的优势没有扩大，在 3/3 outer folds 上反而缩小。

## Cross-fitted sparse residual

对每个 outer train：

1. 用 inner scaffold folds 产生 `T+A` cross-fitted probabilities；
2. 用 `y-p(T+A)` 作为 residual，对 role×attribute interaction 排序；
3. 只保留在两个 train scaffold groups 中相关方向一致、支持度足够的 top-k；
4. 用 L1 logistic correction 比较 true interaction 与 matched shuffle；
5. top-k 与 C 只通过 inner directional scaffold transfer 选择。

| source | outer AUC | delta vs tuned `T+A` |
|---|---:|---:|
| tuned `T+A` | **0.7129** | +0.0000 |
| sparse raw | 0.6986 | -0.0143 |
| sparse centered | 0.7072 | -0.0057 |
| sparse raw shuffle | 0.7007 | -0.0122 |
| sparse centered shuffle | 0.6779 | -0.0350 |

centered true 相对 shuffle 为 `+0.0293`、2/3 wins，说明干净 binding 信号再次被
检测到；但 centered 相对 tuned `T+A` 为 `-0.0057`、仅 1/3 wins。raw 同时未超过
baseline 或 shuffle。

这再次区分了两种命题：

- role--attribute dependence 存在；
- dependence 能提供跨 scaffold 的额外标签信息。

本轮支持第一条，不支持第二条。

## 阶段决定

1. 不扩大 Optuna 预算：interaction 的 relative-gap rescue gate 为负且 0/3 wins；
2. 不继续 top-k/C、interaction selector 或 residual learner 搜索；
3. 不在本轮重新打开 official-valid；否则会把已经看过的 split 继续用于路线迭代；
4. 当前最稳主线仍是 rooted-WL `T+A` marginals；centered 保留为 dependence
   diagnostic，而不是已经成立的 predictive fusion；
5. K-SVD 仍不进入，因为当前问题不是压缩器容量。

原始结果与可复用特征缓存：

- [`task_aligned_interaction_screen/summary.json`](task_aligned_interaction_screen/summary.json)
- [`task_aligned_interaction_screen/summary.md`](task_aligned_interaction_screen/summary.md)
- `task_aligned_interaction_screen/features_rooted_wl.npz`（44 MB，official-train-only）
