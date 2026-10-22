# zinc_local_factorization_v1 — 局部内容函数：稠密 MLP(D) vs 因子化低阶统计(F)

一轮、两臂、单模型、单 seed、同一 P3 载体与同一接入口。两臂**只差局部内容函数 phi**（独有参数 10,512 vs 10,368，差 144 / 0.032%）。
判定规则在训练前写进 `configs/lf_content_v1.json`，并在 CPU 上过 gate（10/10，`analysis/cpu_checks.json`）。

## 0. 结论（先给答案）

| 问题 | 结果 |
|---|---|
| F 是否比 D 好 | **没有**。best-valid F−D = **−2.0e-5**（F 略低，499 分之差的可忽略量级）；test F−D = **+0.00164**（F 更差，方向与 valid 相反） |
| 是否"多个 epoch 维持清楚优势" | 不是"清楚"。逐 epoch valid：F 更好 326/600，差值中位数 **−1.2e-4**；末段 100 个 epoch F 更好 78/100、中位 **−1.5e-4**。这是 1e-4 级、随 epoch 反复易号的小偏移 |
| 是否 test 同向改善 | 否，反向（+0.0016） |
| 是否"train 更差但 held-out 更好"（有益约束） | 否。F 在各自 best 点 eval-train 0.0232 vs D 0.0074（F 拟合更浅），held-out 没有更好 |
| 接入口是否被忽略 | 否。W_s 范数 0.23→6.41(D)/5.27(F)，port_rms 0.0085→0.061 / 0.0067→0.059，s_rms 0.100，phi 末段每 epoch 仍在变（Δ 0.006） |
| 到 0.05x 的距离 | 没靠近。最好 test = D 0.070579 = 0.050 的 **1.41×**；比历史最好（L_s0 0.066267）差 +0.0043 |

**判定：本轮落在预注册的"F≈D 且 test 略负"一侧 → 关闭这个 "rank 32 / 最高三阶 / 198→128 固定 W_s 接入口" 的具体组合。** 这不等价于"组合/因子化没用"：本轮只检验了一个 rank、一个阶数上限、一个接入口。

## 1. 固定不变项

- official PyG ZINC subset，train 10,000 / valid 1,000 / test 1,000，raw-y MAE，单模型。
- 共享 P3NullModel 载体 **413,505** 参数，S3/S4/F4 三个 bf_data 对象（已审计的 padded 张量、完整类型化诱导邻接、多重性、occurrence 权重）。
- 内部 root-preserving 置换平均：4 节点 6 个 / 5 节点 24 个排列，先算 phi 再平均成 32 维。
- 每 root × family 聚合（occurrence 加权 mean / mean_sq / count / existence）→ 198 维 → 198→128 **无 bias 零初始化** W_s → 接到 encoder 第一层残差口。
- 内容布局：5 个 atom slot（词表 28）、10 个无序 pair（0=无边，1..3=键型）、node_mask 5、pair_mask 10、family one-hot 3。pattern id 只是静态缓存地址，**没有任何 whole-pattern 可学习 embedding**。
- 初始化：seed0 原构造顺序生成 P3 核心 → 只加载 train-only masked-SSL encoder（`pretrain_encoder_s0.pt`，SHA256 `006fabc1…bfa97`）→ 两臂复制同一 public state（W_s=0）→ 共享内容 embedding（E_atom 28×8、E_pair 4×4）用 seed101 生成一份、两臂复制 → D/F 的 phi 分别 seed202/303。composer/reader 由 seed0 随机初始化，不加载任何旧 best.pt。
- 配方：AdamW 单组、peak 1e-3 / min 1e-5 / wd 1e-5 / betas(0.9,0.999) / eps 1e-8、warmup 500、per-step cosine、raw-y L1、clip 5、dropout 0、fp32、batch 128、79 updates/epoch、600 epoch（47,400 updates）、每 epoch 全量 valid、strict `<` 选 best（相等留早）。无 SAM / 辅助任务 / EMA / soup。

两臂唯一差别：

| | D LF-Dense | F LF-Factorized |
|---|---|---|
| phi | `98→80(SiLU)→32`，两层 bias | 逐通道 z1/z2/z3（真实字段数作分母）→114→`64(SiLU)→32` |
| 独有参数 | **10,512** | **10,368** |
| 总参数 | **449,601** | **449,457** |

F 的实现细节：A(8→32)、B(4→32)、R_atom[5,32]/R_pair[10,32]~N(0,0.1)；字段 `f = mask*tanh(proj+role)`；`m` 为该通道实际字段数（10 或 15）；mask 先乘进 p1/p2/p3；`m<k` 时 z_k=0；`u=concat(z1,z2,z3,node_mask,pair_mask,family)`（114）。

## 2. 核验（训练前，10/10）

`uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_checks.py` → `analysis/cpu_checks.json`（config sha256 `44753f28…f8cd`）。

1. **缓存/内容一致**：train/valid 内容数组 digest 与此前审计一致（`ce643fda…` / `91e0e81b…`），n_entries 312,110 / 31,038，unique 1,280 / 644，occurrences S3/S4/F4 = 82,568/3,104/247,704 与 8,225/314/24,675 全对；8 个 valid 分子重枚举无 mismatch；split-local id 语义（id=排序内容 key 的位置）已确认为约定而非可依赖性质。
2. **等变性**：原子重编号+反向边 max|Δ|=**0.0**；batch 重组/padding max|Δ|=9.2e-7（两臂）。
3. **置换平均**：与独立 float64 显式枚举 rel_l2 ≤1.3e-7；重标号不变性 ≤3.0e-8；重复行合并权重精确（≤3e-8）。
4. **z1/z2/z3**：m=0/1/2/10/15 与显式成对/三元枚举误差 ≤1.1e-15；分母是真实字段数；padding 的大值被 mask 掉。
5. **聚合语义**：mean = phi_bar、mean_sq = phi_bar²、count/existence、空 family 全 0、两次 occurrence 与 count=2 正确；**mean_sq 在置换平均之后**（置换方差残差 D 0.0136 vs F 6.5e-7）。
6. **类型化邻接**：同原子/键型 multiset、chord 1–2 vs 1–3 的单布局 phi_bar 相对差 D 0.056 / **F 3.4e-5**（F 的通道矩几乎不受"哪一对是 chord"影响 —— 这正是因子化的性质）。
7. **寻址**：同内容不同地址 phi_bar 一致；embedding 只有 (28,8)/(4,4)；branch 无 pattern 级参数、无缓存隐状态。
8. **公共初值**：两臂 public state hash 相同 `21963b43…`（= P3 载体），共享 embedding hash 相同 `1d454a3e…`，W_s 精确 0；初始前向输出与 SSL-start P3、与上一轮 A_joint_BF 载体 **max|Δ|=0.0**；D vs F 初始输出 → **0.0**。
9. **接入口梯度**：init 时 phi 梯度为 0（W_s=0 的数学后果，预期），W_s 梯度 20.4/17.5；打开后 branch 参数梯度全非零。
10. **预算/调度**：79 steps×600=47,400；warmup 端点=peak、末端=min、warmup 后单调；optimizer 覆盖全部参数；cap ok；checkpoint 重载 max|Δ|=0；两臂逐 epoch batch order hash **完全相同**（首 `8eca27da…`、末 `db5b8e71…`，与本轮之前的 P3 轮同一顺序流）。

## 3. 主表

见 `analysis/main_table.md`（同内容）。参数、init 来源、E/updates、best-valid、best epoch、eval-train@best、final-valid、test MAE/RMSE/p95/maxAE、GPU-h、墙钟。

| model | 参数 | best valid | ep | train@best | final valid | test MAE | test RMSE | GPU-h | 墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D LF-Dense** | 449,601 | 0.078613 | 455 | 0.00741 | 0.079964 | **0.070579** | 0.141524 | 0.415 | 1495 s |
| **F LF-Factorized** | 449,457 | 0.078593 | 329 | 0.02324 | 0.079792 | **0.072216** | 0.141749 | 0.581 | 2090 s |
| ref A_joint_BF_s0（上轮，复用） | 447,505 | 0.076731 | 375 | 0.01567 | 0.078218 | 0.070067 | – | 0.364 | 1310 s |
| ref P3B_s0（复用） | 413,505 | 0.076224 | 377 | – | – | 0.069865 | – | 0.301 | 1085 s |
| ref L_s0（复用） | 494,193 | 0.078537 | – | – | – | **0.066267** | – | – | – |
| ref DBT_s0（复用） | 452,523 | 0.077768 | – | – | – | 0.069401 | – | – | – |
| ref P3-Control（上轮，复用） | 413,505 | 0.079588 | 431 | – | 0.081668 | 0.072689 | – | 0.325 | 1169 s |

**主对比 F−D：valid −2.0e-5，test +0.00164（MAE），test RMSE +0.00023。**

## 4. 曲线（`analysis/curves.png`，逐 epoch）

- 两臂同步：1 → 50 → 100 → 150 → 200 → 250 → 300 → 350 → 400 → 450 → 500 → 600 的 valid：
  D 0.6635 / 0.1535 / 0.1217 / 0.1120 / 0.0982 / 0.0927 / 0.0897 / 0.0817 / 0.0810 / 0.0803 / 0.0795 / 0.0800
  F 0.6721 / 0.1456 / 0.1161 / 0.1014 / 0.1000 / 0.0867 / 0.0917 / 0.0828 / 0.0817 / 0.0791 / 0.0802 / 0.0798
  （F 在中段若干处更低，但差距 ~1e-3 且来回；两者末段都贴在 0.0798–0.0800。）
- 末 100 epoch valid 中位数：D 0.079967 / F 0.079825 → 差 −1.4e-4。
- best 点是孤立低谷：D 0.078613@455、F 0.078593@329；用尾段中位数比较更稳，结论相同（不可分辨）。
- 训练集：online train 最后 D 0.00072 / F 0.00067（接近），但各自 best 点 eval-mode train 差 3 倍（0.0074 vs 0.0232），final eval-train 0.00076 vs 0.00120 —— F 拟合更浅，没有换来 held-out 收益。
- 单侧配对 bootstrap（只表示分子抽样不确定性，不代表跨 seed 可重复性）：
  valid ΔAE −2.0e-5，95% CI [−0.0043, +0.0037]；test ΔAE +0.00164，CI [−0.0013, +0.0047]。两者都跨 0。
- 历史同配置复跑漂移约 0.003（valid/test 量级）→ 本轮 1e-3 以下的 F−D 无法用单 seed 分辨。D 相对上轮 A_joint_BF（同载体、无置换平均）：valid +0.0019 / test +0.0005，也在漂移尺度内。

## 5. 分组与极端误差（`analysis/groups_valid.json`、`groups_test.json`）

test（n=1000）：

| group | n | D | F | F−D |
|---|---:|---:|---:|---:|
| covered / ordinary | 705 | 0.05046 | 0.05240 | **+0.00194** |
| covered / tail | 51 | 0.09529 | 0.10570 | **+0.01041** |
| unseen / ordinary | 218 | 0.11727 | 0.11642 | −0.00085 |
| unseen / tail | 26 | 0.17629 | 0.17326 | −0.00303 |

valid：covered_ordinary +0.0011、covered_tail −0.0283(n=39)、unseen_ordinary +0.0015、unseen_tail −0.0020。

- 差异在 test 上**集中在被覆盖（covered）的分子**，尤其是 covered_tail（n=51，噪声大）；unseen 两组反而略偏 F。分组样本量小，只作描述，不作机制结论。
- 极端误差**没有退化**：AE>0.5 的行 13(D) vs 15(F)；AE>1.0 4 vs 4；AE>1.4 2 vs 3。最大误差 D 1.572（row 342）/ F 1.462（row 402）。row172（y=−0.497）两臂都好（AE 0.036 / 0.020）。
- 位移是**弥散**的：逐行 |F−D| 均值 0.038（最大 0.556 @row 298），中位 AE 0.0391(D) vs 0.0428(F)；F 逐行变差 >0.05 的 105 行、变好 >0.05 的 91 行。

## 6. 时间

- 计时（4 个真实 epoch，含对称化、统计、backward、全量 valid）：t_slow D 2.409 s / F 3.566 s → `1.15·t·600+120` = 1782 s / 2580 s ≤ 3600 → 两臂同取 **E=600**。
- 正式：D 1495 s（0.415 GPU-h，24.9 min），F 2090 s（0.581 GPU-h，34.8 min）；两卡并行总墙钟 **15:29:10 → 16:04:07 = 34.9 min**。
- F 每 epoch 慢 1.48×：15 个字段的 z1/z2/z3 + 114→64 比 98→80 的 MLP 更贵，尽管参数更少。**因子化的成本是算力，不是参数**。

## 7. 解释边界

- F 的 factorized 统计量在"哪一对承载 chord"这件事上近乎不变（rel 3.4e-5，D 是 5.6%），所以 F 的判别必须靠 R_atom 槽位与 114→32 readout 补回；本轮它没有补回，代价约 +1.6e-3 test MAE。
- 两臂共享的对称化与 P3 载体不属于 F−D 的归因范围；本轮不重做覆盖审计（旧"root 覆盖 2.18%"的 erratum 已披露为 56.62%）。
- official test 历史上已暴露（含此前的多次读取）；本轮仍按预注册只读一次、冻结后读、不参与选择，**不声称盲测**。
- 单 seed、单 rank、单接入口：F−D 的 CI 跨 0，不能得出"组合或因子化没用"的一般结论。
- 接入口**不是**"没干预"：W_s 与 phi 都在动、port 量级可观，所以"内容无用"不是本轮的解释；能被否的只是"在这个 rank/阶数/接入口下，F 的约束没有换来 held-out 收益"。

## 8. 复现

```
config          tracks/structural-sufficiency/zinc_local_factorization_v1/configs/lf_content_v1.json  (sha256 44753f28…f8cd)
run commit      70275b31e190f90b8ff3e45000fb41f268b4f0d9   (code only; analysis is this commit's follow-up)
gate            uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_checks.py
unit tests      uv run pytest -q tracks/structural-sufficiency/zinc_local_factorization_v1/code/tests   (17 passed)
arm D           rr run res lf-D-s0 --pool res-gpu0 -- uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_runner.py run --arm D --device cuda:0
arm F           rr run res lf-F-s0 --pool res-gpu1 -- uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_runner.py run --arm F --device cuda:0
report          uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_report.py
best  D         41f0abca14ee1163574ccf706b5cba4fd7b82d04bd250b17f7358b1b1ab5620f   (ep 455)
best  F         5f6492cb691d9c9cb37e4f5638d1dbc2aba0e3d934484270e82f11e9a8649f4f   (ep 329)
final D         50016bd7c537b051cd091a0a3ebbd92f5e44238e433ad3801702182ecef703e1   (ep 600)
final F         919218d03f43c663d326c6af65fd5381993982e0f3525c3f5d2eb6203c0ba33e   (ep 600)
test labels     sha256 66fd23ecdb1b59c76cee597efc81272e0d2744eeac80744f4159d436e5388768 (两臂相同)
per-row preds  analysis/preds_test.csv, analysis/LF_{D,F}/preds_{valid,test}_best.csv
```

## 9. 下一步建议（只提建议，本轮两臂到此收口）

1. **不要在这个接入口上做 rank/阶数扫描**。1e-4~1e-3 的效应被单 seed 噪声与 0.003 的历史漂移完全覆盖，扫参只会得到不可分辨的曲线。
2. 若要继续检验"组合共享是否提供约束"，先改**可分辨性**而不是改 F：同一数据顺序下跑 3 个 seed 的 D 基线，量出真正的 seed 方差，再决定最小可检测效应；否则任何 F 变体都无法判定。
3. 若仍想用因子化，值得试的是**让 F 保留低阶统计之外的判别信息**（例如 pair 槽位上的键型直方图/角色交叉项），而不是提高 rank —— 本轮证据指向"z1/z2/z3 丢失了 chord 归属的信息，readout 补不回来"，不是"rank 不够"。
4. 载体层面（覆盖 56.62%、unseen 组 0.117 的瓶颈）比局部内容函数更接近 0.05x 的主要损失来源：unseen_ordinary（n=218）贡献了 test MAE 的大头，而 covered 组已在 0.050–0.052。
