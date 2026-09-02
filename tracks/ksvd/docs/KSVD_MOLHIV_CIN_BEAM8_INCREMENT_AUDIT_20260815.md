# MolHIV CIN + Beam8 增量审计（2026-08-15）

## 结论

在当前证据下，应关闭 **CIN + Beam8 structure-endpoint 直接 residual** 路线。

降低融合强度并增加 branch dropout 后，aligned Beam8 的三折均值从强融合时的
`.6850/.1189` 改善到 `.6960/.1398`，表面上高于 CIN 的
`.6860/.1239`。但 matched controls 表明这不是稳定、不可替代的 Beam8 patch
增量：

- aligned 只在 fold 0 同时明显胜过 CIN、shuffled 和 no-patch；
- fold 1 虽然 AUC 略高于 CIN/shuffled，但低于 no-patch，且 AP 大幅低于
  shuffled；
- fold 2 低于 CIN、shuffled 和 no-patch；
- aligned 相对 no-patch 只赢 1/3 folds；
- aligned 与 no-patch 的平均 AUC 几乎相同（`.6960` vs `.6953`），而 shuffled
  的平均 AP 反而更高（`.1463` vs `.1398`）。

正则化修复了部分过强融合造成的负迁移，但没有把正确 patch context 变成稳定
增量。当前不能据三折均值声称 Beam8 改善了 CIN。

## 这不是论文约 0.80 CIN 的复现

本实验只回答“在同一内部划分上，Beam8 是否给 CIN-compatible 主干提供条件增量”，
不回答“本实现能否达到论文 CIN 的 official-test 结果”。

- 数据使用 `ogbg-molhiv` 按 official split 内分层抽样得到的 8000 图子集，其中
  official-train 为 6400 图；
- 评估使用 official-train 内部 3-fold scaffold split；
- 没有编码或评估 official-valid / official-test；
- 实现是 PyG-2、dependency-light 的 Sparse CIN-compatible operator，不是旧
  PyG-1.6 官方代码的 bit-for-bit 复现；
- 官方 CWN 参考 commit 为
  `c4ddd24e251929f934f8a2467da98fdba376864d`。

因此这里 `.62--.74` 的内部 held-out AUC 不能与论文使用完整数据、官方 split、
不同软件栈得到的约 `.80` 直接横比。绝对分数不是本审计的判据；matched delta
才是判据。

## 实现

### CIN-compatible 主干

- 0-cell：atom；
- 1-cell：undirected typed bond；
- 2-cell：3--6 元 chordless cycle；
- boundary aggregation；
- 使用 shared coboundary state 的同维 upper adjacency；
- 2 layers、hidden 48、dropout 0.5、batch size 128、150 epochs、Adam
  `lr=1e-4`；
- 每个 cell dimension 做 mean readout，经投影后求和。

6400 图上的 ring coverage：

- `94.66%` 分子至少含一个 3--6 元 chordless cycle；
- mean ring cells：`2.835`；
- mean bond cells：`27.112`。

### Beam8 structure-endpoint branch

该分支不是 graph-level patch mean readout。它在第一层 CIN 后形成节点级 residual：

1. 使用 patch internal typed bonds、canonical bond pair、position 和
   base/completion role 构造 patch context；
2. 每条 patch bond 向两个具体 endpoint atom 发送消息；
3. 消息保留 endpoint canonical slot、bond type 和 pair identity；
4. 聚合回具体 atom 后再进入第二层 CIN。

EDGE100 的 6400 图覆盖为：edge coverage `1.0`、node coverage `1.0`，因此本轮不再
存在原始约 74% coverage 的混淆。

Matched variants：

- `cin_beam8`：正确 patch context 与 endpoint 绑定；
- `cin_beam8_shuffled`：保留边际统计，但把 patch context 错配；
- `cin_beam8_no_patch`：保留 endpoint bond/pair/slot/role 与相同参数量，不使用
  patch context。

所有 variants 在构造前重置同一 seed，且所有模型都实例化相同模块，使 CIN core
初始化一致。

## 固定强融合：失败

配置：`fusion_scale=1.0`、`branch_dropout=0.0`。数值为 held-out
`ROC-AUC / AP`。

| Variant | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| CIN | .6989 / .1303 | .6242 / .0513 | .7348 / .1900 | .6860 / .1239 |
| CIN + Beam8 | .7271 / .1624 | .6352 / .0657 | .6927 / .1287 | .6850 / .1189 |
| Beam shuffled | .6825 / .1429 | .6123 / .0830 | .7134 / .1748 | .6694 / .1336 |
| Beam no-patch | .7142 / .1284 | .6360 / .0593 | .7304 / .1742 | .6936 / .1206 |

fold 0 有强 aligned 信号，fold 1 较弱，fold 2 出现大幅负迁移。Beam variants 的
fit AUC 接近 1；强 residual 主要放大了过拟合与 scaffold-specific interference。

## 正则化 residual：改善但仍未通过归因门槛

配置：`fusion_scale=0.25`、`branch_dropout=0.5`，其余完全相同。

| Variant | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| CIN（固定基线） | .6989 / .1303 | .6242 / .0513 | .7348 / .1900 | .6860 / .1239 |
| CIN + Beam8 | .7362 / .1764 | .6363 / .0591 | .7154 / .1837 | .6960 / .1398 |
| Beam shuffled | .7026 / .1342 | .6294 / .1217 | .7219 / .1831 | .6847 / .1463 |
| Beam no-patch | .6987 / .1362 | .6393 / .0637 | .7479 / .1891 | .6953 / .1297 |

aligned 的逐折差值：

| Fold | vs CIN | vs shuffled | vs no-patch |
|---|---:|---:|---:|
| 0 | +.0373 / +.0461 | +.0336 / +.0422 | +.0375 / +.0403 |
| 1 | +.0121 / +.0078 | +.0069 / -.0626 | -.0029 / -.0046 |
| 2 | -.0194 / -.0063 | -.0065 / +.0007 | -.0325 / -.0053 |

这里不能用 mean 掩盖 fold 2，也不能把 fold 1 的小幅 AUC 改善与大幅 AP 退化分开
包装。按预注册式判据，aligned 必须在多数 folds 同时胜 CIN、shuffle 与 no-patch；
本结果没有通过。

正则化后分支仍然是活跃的：aligned 的 fusion weight norm 在三折分别为
`1.208 / 1.577 / 1.570`。因此失败不是零初始化分支没有学起来。

## CIN ring operator sanity check

fold 0 的 CIN-only ring attribution：

| Variant | Held-out AUC / AP |
|---|---:|
| CIN | .6989 / .1303 |
| CIN no-rings | .6958 / .1399 |
| CIN ring-shuffled | .6781 / .1116 |

正确 ring incidence 相对 ring shuffle 为 `+.0208 AUC / +.0187 AP`。这说明
CIN-compatible operator 确实使用了正确 cycle--bond 对齐；Beam8 失败不能简单归因
为主干完全忽略高阶结构。

## 路线判定

### 关闭

- 在第一层 CIN 后直接把 Beam8 structure endpoint context 加到 atom state；
- 继续只调 `fusion_scale`、branch dropout 或 residual 宽度；
- 用三折均值掩盖 fold-level 符号翻转；
- 在当前证据下进入 official-valid / official-test。

### 尚未被本实验否定，但必须换机制

如果继续探索 Beam8，应避免让分支端到端替换 CIN 表征，并提出新的可证伪机制：

1. **冻结 CIN 的 OOF late residual**：先在 fit 内生成严格 OOF CIN logits/embeddings，
   再训练低容量 Beam8 sidecar，只允许预测 CIN 的剩余误差；与 shuffled/no-patch
   sidecar 同桌。这样可直接测试条件信息，而不是联合训练干扰。
2. **稀疏样本级 gate**：gate 只用 train-derived、label-free 的结构量（例如 Beam
   重叠冲突、ring/patch boundary interaction）决定是否启用 residual，并对 gate 做
   shuffle attribution。不能根据 held-out scaffold 事后挑子群。
3. **bond/ring-cell 条件消息**：当前 Beam8 回写 atom，而 CIN 已经显式建模 bond 和
   ring。若有新机制，应直接表达“Beam patch boundary 如何连接多个 CIN cells”，并
   与只保留局部 bond role 的 no-patch control 比较。
4. **官方 CIN 栈上的最终复核**：只有新机制在内部 folds 稳定胜全部 controls，才值得
   使用官方 CWN 依赖栈或冻结 checkpoint 做一次更强主干复核。不要先花算力把当前
   compatible baseline 调到约 `.80` 再重复同一 residual。

这些是新假设，不是当前结果支持的增益主张。

## 可复现文件

- CIN/Beam8 lift：`tracks/ksvd/code/molhiv_cin_beam8.py`
- runner：`tracks/ksvd/code/run_molhiv_cin_beam8.py`
- tests：`tracks/ksvd/code/test_molhiv_cin_beam8.py`
- strong results：`tracks/ksvd/results/molhiv/cin_beam8_full_fold{0,1,2}_seed0_20260815.json`
- regularized results：
  `tracks/ksvd/results/molhiv/cin_beam8_regularized_full_fold{0,1,2}_seed0_20260815.json`

验证：相关 Python 文件 `py_compile` 通过；Beam8 incidence + CIN lift 共 9 项
`unittest` 全部通过。

本审计的 `official_valid_evaluations=0`、`official_test_evaluations=0`。
