# 导师 50-node 真实子图 Beam8 pilot：人工结论与下一步

> 日期：2026-08-06  
> 自动报告：`tracks/ksvd/results/mentor_subgraphs/BEAM8_PILOT_20260806.md`  
> 下一轮冻结协议：`tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_GROUPED_RECONSTRUCTION_PROTOCOL_20260806.md`

## 1. 结论

本轮判定为：

```text
READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP
```

这意味着真实数据已经通过数据契约、stable preprocessing、Beam8 sampler 和三种 geometry 的覆盖可行性门槛。它不意味着 KSVD 已经学到语义 motif，也不意味着可以直接训练 Transformer。

## 2. 当前最可靠的发现

### 2.1 Stable preorder 的价值是可重放，不是覆盖增益

numeric relabel replay 中，stable class、singleton ID、canonical adjacency、rooted vector 和 transition 都达到 1.0，edge/pair coverage delta 为 0。与此同时，stable order 与 source input order 的 BASE coverage/RMSE 几乎相同，而具体 patch chain Jaccard 只有 `0.47–0.67`。

因此应采用以下解释：

- `GLOBAL_WL preorder` 成功提供了结构坐标与重放稳定性；
- 它没有证明能直接提高 coverage；
- exact ordered chain 只有 0.45 不构成失败，因为 automorphism class 内具体 node identity 不可识别，真正通过的是 canonical vector/transition 与 coverage invariants。

### 2.2 三种 geometry 都可行，但没有 pooled-mean 单一赢家

BASE 上：

- `s8/o2`：patch 较多、edge coverage最高、residual最少、RAW RMSE最低；
- `s12/o4`：patch/code最少、pair coverage最高，但在中高密度图上 residual较大；
- `s10/o3`：处于二者之间，是重要的中间 control。

FAIR95 后三者 edge coverage 都约 `0.98`，node p10 都约 `0.95`，但成本分别约 `23.34 / 16.93 / 12.94` patches。应将三者视为 rate–distortion frontier，而不是凭 pooled mean冻结单一 geometry。

### 2.3 Density dependence 是主效应，不能被总体均值掩盖

BASE 的低尾 node recall 在 `d5_10` 和 `d10_15` 最差。`s12/o4` 在极稀疏 `lt5` 上很好，但在 `d15_25/ge25` 的 residual明显升高；`s8/o2` 则随密度升高表现更稳。

因此下一轮必须：

- 保留五个 density strata 的条件报告；
- 不能把 density 当 label 或提供给 dictionary；
- geometry选择必须检查各 strata，而不是只看 500 图 pooled mean。

### 2.4 FAIR95 证明额外覆盖可达，但 residual sidecar 仍是关键系统选择

三种 geometry 的 FAIR95/EDGE100 均可达，说明真实图上不存在明显 sampler ceiling。可是前一轮 synthetic KSVD follow-up 已显示：在 exact residual作为第二通道时，把更多 residual edges搬入固定 K24/T3 patch channel可能恶化 compression-only reconstruction。

所以真实 grouped reconstruction 必须同时比较：

- uncorrected full reconstruction；
- exact-residual-corrected reconstruction。

不能只用 coverage 更高就宣布 FAIR95 必然优于 BASE。

## 3. Split 现实与结论边界

在冻结 500 图中：

- 共享至少一个 source node 的 graph-overlap graph 是单一 connected component；
- 共享至少一条 source edge 时也仍是单一 connected component；
- 因而严格 source-node-disjoint 或 source-edge-disjoint 3-fold 会塌缩，无法形成有意义评估。

下一轮采用两个视图：

1. density-balanced random-reference；
2. root-candidate-grouped 主视图，同一 root 不跨折。

同时报告 source node/edge exposure，而不是伪称完全 source-disjoint。由于 first insertion node 的 root 语义尚未确认，报告必须一直使用 `root_candidate`。

## 4. 下一步执行顺序

1. 实现并测试 deterministic random/root-grouped folds 与 exposure audit；
2. 为同一 500 图重建三 geometry 的 BASE/FAIR95 rooted-canonical examples；
3. 每个 fold 仅使用 train patches 完成 mean、PCA3、RANDOM、INIT、FINAL；
4. 输出 uncorrected/residual-corrected stitching 与 dictionary health；
5. 只有 root-grouped 下 FINAL 稳定优于 INIT，并相对 RANDOM/PCA形成可解释 Pareto，才进入 relation-token downstream ablation。

## 5. 当前建议

- **保留全部三种 geometry 跑冻结 reconstruction**。6 branches × 2 views × 3 folds 的字典训练规模仍可控，且删掉中间 geometry 会削弱 Pareto解释。
- **不增加 cover seeds**。pilot 已完成 sampler可行性；下一轮目标是 dictionary attribution，不是重新估计 sampler variance。
- **不扫描 K/T/updates**。固定 `K24/T3/u25` 才能判断真实迁移，而不是用调参修复结果。
- **不启动 Transformer**。先完成 reconstruction gate；失败时应保留 RAW/INIT 或 residual channel，不应继续扩大模型。
