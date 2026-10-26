# ZINC 冻结键表示审计 v1 — 报告

**Track:** `tracks/structural-sufficiency/zinc_frozen_bond_representation_audit_v1/`
**起点 commit:** `072eeaa`（工作树未改动旧 track）
**主判定:** **`MIXED_OR_INCONCLUSIVE`**
**边界:** GPU 0 h；0 次梯度更新/optimizer step；0 次训练；只读冻结 checkpoint；只读 official-train 10,000 个结构（历史冻结结构缓存 `ad_train.pt`，未索引 `labels`）；未打开 official valid/test cache；未计算 target MAE。

---

## 1. 对象与 checkpoint 核对

从真实 forward 取对象（forward hook，非按报告复现）：

| 状态 | 权重 | 对象 |
|---|---|---|
| D0 | `dcc_models.build_initial_state(norm, 0)` | P3 bond encoder `b`，group composer `c_glob`，local composer `c_loc` |
| D1 | `runs/L_s0/best.pt`，epoch 489 | 同上；`organized_local=True` |
| R0 | `rcc_models.build_initial_state(norm, 0)` | 统一 composer `h_RCC`，rank-16 投影 `Z_RCC=z_proj(h_RCC)` |

核对结果：

- D0 state hash `f5c34c28…66e33` 与历史 `artifacts.json` 的 `initial_state_sha256` 完全一致；P3 reference hash `21963b43…16a30` 也一致；SSL sha `006fabc1…bfa97`。
- D1 `best.pt` sha `fab04ec0…26cd5` 与 `artifact_hashes.json` 一致；epoch 489、stored valid 0.0785371496 与 `artifacts.json` 的 best 一致；candidate `DCC_L`，seed 0，494,193 参数；与 D0 同 state keys/形状；相对 L2 距离 **1.1987**；`best.pt` 与 `final.pt`（sha `bc0ed5eb…a7c21`）不同。`final.pt` 未加载。
- D1-s1 **不存在**；未补训。RCC **没有监督 checkpoint**（`runs/` 为空），所以本轮没有 RCC 训练后状态。
- R0 当前 state hash `230f44f8…5f4a`；`prep.json` 记录的是 `5f2afdee…53a80`。差异只来自 `t_scale` buffer：把 `t_scale` 置 1 后 hash 精确等于 `5f2afdee…`（仅用于核对，随后恢复；`h_RCC`/`Z_RCC` 不依赖该 buffer）。RCC 旧 PR 的精确复现也确认对象与预处理一致。
- 物理键：cache `split='train'`、`n_molecules=10000`；`bond_ptr[-1]=249,279`，与 `bond_index` 行数一致；每条物理键只算一次，未展开 PyG 双向 edge storage。D0/D1/R0 使用同一 10k 行、同一 batch 顺序、同一物理键顺序。
- 重排检查（5 个分子）：bond permutation 最大绝对差 DCC `≤2.98e-8`、RCC `0.0`；atom permutation 全部 `0.0`；通过 1e-5 阈值。

---

## 2. 定义与主谱

对每个分子 `G`：`H_G ∈ R^(n_G×d)`，`mean_G = mean_e H_G[e]`，`H_within_G = H_G − mean_G`。

- `E_total = Σ||H_G||²`；`E_common = Σ n_G||mean_G||²`；`E_within = Σ||H_within_G||²`；实测加和残差为 0（float64 舍入内）。这些是**原点依赖的能量**，不是平移不变的 PCA 方差占比。
- `C_all`：全部键按全局均值中心化；`C_within`：先分子内中心化再 pool；`C_mw`：每分子先算 `C_within_G`，再分子等权平均。`PR(C)=tr(C)²/tr(C²)`。
- 主结果按全部真实键加权；`C_mw` 是预固定的分子等权稳健性口径。`pairwise RMS = sqrt(2W_G/(n_G−1))` 是分子内不同键对的 RMS 距离，除以 bond RMS 后尺度无关。

| 状态/对象 | E_common/E_total | E_within/E_total | PR_within (bond) | PR_within (mol) | PR_all | top1_within | dims90_within | 每分子 PR 中位 [p10,p90] | pairwise RMS / bond RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 `b` | 0.171 | 0.829 | 11.269 | 11.328 | 11.418 | 0.187 | 18 | 6.27 [4.82, 7.53] | 1.319 |
| D0 `c_loc` | 0.370 | 0.630 | 2.605 | 2.576 | 2.565 | 0.600 | 8 | 2.58 [1.49, 3.86] | 1.036 |
| D0 `c_glob` | 0.893 | 0.107 | 2.685 | 2.719 | 2.943 | 0.600 | 12 | 1.97 [1.46, 2.72] | 0.496 |
| D1 `b` | 0.062 | 0.938 | 18.607 | 18.733 | 19.105 | 0.144 | 31 | 8.22 [6.13, 10.04] | 1.401 |
| D1 `c_loc` | 0.117 | 0.883 | 3.161 | 3.115 | 2.989 | 0.549 | 15 | 2.99 [1.67, 5.68] | 1.384 |
| D1 `c_glob` | 0.299 | 0.701 | 5.588 | 5.465 | 5.609 | 0.386 | 17 | 3.83 [2.24, 5.79] | 1.341 |
| R0 `h_RCC` | 0.472 | 0.528 | 2.478 | 2.444 | 2.528 | 0.621 | 8 | 3.10 [1.63, 4.41] | 0.940 |
| R0 `Z_RCC` | 0.475 | 0.525 | 1.594 | 1.585 | 1.636 | 0.785 | 3 | 1.99 [1.28, 2.98] | 0.912 |

| 状态/对象 | `C_within` top1/2/4 | `C_all` top1/2/4 | `C_within` dims90/95 | `C_all` dims90/95 |
|---|---:|---:|---:|---:|
| D0 `b` | 0.187/0.329/0.519 | 0.186/0.322/0.514 | 18/26 | 18/26 |
| D0 `c_loc` | 0.600/0.705/0.836 | 0.606/0.713/0.839 | 8/14 | 7/14 |
| D0 `c_glob` | 0.600/0.673/0.763 | 0.561/0.694/0.795 | 12/21 | 12/20 |
| D1 `b` | 0.144/0.226/0.369 | 0.141/0.221/0.361 | 31/43 | 32/44 |
| D1 `c_loc` | 0.549/0.629/0.724 | 0.566/0.645/0.738 | 15/23 | 14/22 |
| D1 `c_glob` | 0.386/0.497/0.638 | 0.346/0.552/0.701 | 17/26 | 16/24 |
| R0 `h_RCC` | 0.621/0.709/0.840 | 0.616/0.696/0.818 | 8/17 | 9/18 |
| R0 `Z_RCC` | 0.785/0.875/0.940 | 0.775/0.859/0.932 | 3/5 | 4/5 |

| 状态/对象 | 每分子 common/total 中位 [p10,p90] | 每分子 within/total 中位 [p10,p90] |
|---|---:|---:|
| D0 `b` | 0.166 [0.109, 0.253] | 0.834 [0.747, 0.891] |
| D0 `c_loc` | 0.487 [0.351, 0.576] | 0.513 [0.424, 0.649] |
| D0 `c_glob` | 0.882 [0.847, 0.909] | 0.118 [0.091, 0.153] |
| D1 `b` | 0.058 [0.036, 0.099] | 0.942 [0.901, 0.964] |
| D1 `c_loc` | 0.083 [0.031, 0.231] | 0.917 [0.769, 0.969] |
| D1 `c_glob` | 0.138 [0.047, 0.386] | 0.862 [0.614, 0.953] |
| R0 `h_RCC` | 0.577 [0.479, 0.651] | 0.423 [0.349, 0.521] |
| R0 `Z_RCC` | 0.602 [0.495, 0.680] | 0.398 [0.320, 0.505] |

上表已给 `C_within`/`C_all` 的 top1/2/4 与 dims90/95；`C_mw`/`C_common` 的对应谱、所有 trace/Frobenius、负特征值、零/近零特征值与 1e-12/1e-10/1e-8/1e-6 相对阈值的数值秩，以及 D0→D1 主子空间重叠，均在 JSON。

**共同项的影响。** 能量上共同项很大（D0 `c_glob` 89.3%，R0 `h_RCC` 47.2%，D0 `c_loc` 37.0%）；DCC-L 从 D0 到 D1，`c_glob`/`c_loc`/`b` 的共同能量占比分别降到 29.9%/11.7%/6.2%。在共同项占能量很大份额的对象上，它本身是低维的（`C_common` 的 PR：D0 c_glob 2.57、D0 c_loc 2.27、R0 h 1.82），所以 `C_all` 与 `C_within` 的 PR 差通常只有 −0.17 到 +0.50。**共同项主要污染“原始范数/能量”的解释，而不是把 within-PR 抬高。** 注意 common share 的 bond 加权与分子等权可以差很多（D1 `c_glob` 0.299 vs 0.138；D0 `c_loc` 0.370 vs 0.487），因为共同项随分子键数/尺度变化；这是必须报告 per-molecule 分布的原因。每分子 PR 的中位数约为 2–8.2，远低于 `n_G−1` 的秩上限（中位 24），所以低 PR 不是键数上限造成的。

`c_loc` 的数值秩上限是 **80**（local composer hidden width 80；96 维输出最多 80 个非零特征值，其余 16 个方向为零/近零）；其 within-PR 只有 2.6–3.2，远低于该数值秩边界，因此不是数值秩伪影。`Z_RCC` 的上限是 16，PR 1.59。

---

## 3. DCC-L：D0 → D1 到底改变了什么

| 对象 | ΔPR_within | Δtop1_within | Δ(E_common/E_total) | Δ(trace C_within) | 每分子 Δ(pairwise RMS / bond RMS) 中位 |
|---|---:|---:|---:|---:|---:|
| `b` | **+7.338** (11.269→18.607) | −0.043 | −0.109 | −5.59 | +0.082 |
| `c_glob` | **+2.903** (2.685→5.588) | −0.214 | −0.593 | +1.69 | **+0.830** |
| `c_loc` | **+0.556** (2.605→3.161) | −0.051 | −0.253 | **−46.56** | +0.330 |

主方向是否只是被放大？把 D0 与 D1 的 `C_within` 主特征子空间对齐：

| 对象 | top1 cos²(D0,D1) | D1 trace 落在 D0 top4 | D0 trace 落在 D1 top4 | D1 trace 落在 D0 top16 |
|---|---:|---:|---:|---:|
| `b` | 0.062 | 0.242 | 0.312 | 0.575 |
| `c_loc` | 0.000165 | 0.076 | 0.052 | 0.247 |
| `c_glob` | 0.000003 | 0.070 | 0.121 | 0.174 |

D1 的主方向与 D0 几乎正交，`c_loc`/`c_glob` 尤其明显；因此 `c_loc` 的小幅 PR 增加**不是**把旧的少数方向放大，而是换成了一批新的、但仍然低维的方向。`b` 保留的重叠更多（D1 trace 的 24% 在 D0 top4），但仍有大量新方向。`c_loc`/`c_glob` 的 top1 份额分别为 0.55/0.39，top1 与后续特征值分离，所以 top1 cos²≈0 不能只用“简并子空间内旋转”解释；top4 子空间重叠也只有 7.6%/7.0%。分子等权口径同结论：top1 cos² = 0.065 / 1.03e-4 / 7.97e-5，D1 trace 落在 D0 top4 = 0.242 / 0.079 / 0.069（`b`/`c_loc`/`c_glob`）。

- `b` 和 `c_glob`：within-PR 明确上升，top1 明显下降，共同能量占比大幅下降；`c_glob` 的 pairwise 绝对与归一化尺度都上升。这是“增加了独立键内方向”的明确证据，不只是 embedding norm 变大。
- `c_loc`：PR 只从 2.605 升到 3.161（+21%），top1 从 0.600 降到 0.549；**整体尺度反而缩小**（trace 49.56→3.00，绝对 pairwise RMS 6.48→2.14），但归一化 pairwise 1.036→1.384。所以不是单纯放大少数方向，也不是单纯数值归一化；但绝对有效维度仍然低，top1 仍占 0.55。`dims90` 从 8 增到 15，但 top1 仍 0.55、PR 仅 3.16；`dims90` 受长尾影响，不能单独当作“主维度增加”的证据。
- 分子等权口径与 bond 等权一致：`c_loc` `C_mw` PR 2.576→3.115；`c_glob` 2.719→5.465；`b` 11.328→18.733。没有“换加权方式结论反转”。
- `c_loc` 按键数分层（固定 8–16 / 17–23 / 24–29 / 30–41）：pooled PR Δ = **+0.057 / +0.410 / +0.853 / +0.292**；每分子 PR 中位 Δ = **−0.168 / +0.116 / +0.563 / +0.820**。最小分子组没有改善，其余组有，但幅度差异大。因此 `c_loc` 的提升不是全分布一致的。

**解释边界：** 可以说 DCC-L 监督训练**在这一个 seed 上**改变了 within 表示；不能说“训练后 DCC-L 的每个 96D 键通道都变高维”。`c_loc` 仍是一个 ~3 维、top1≈0.55 的表示。

---

## 4. 结构算子响应（冻结 `B=K_R−K_D`）

`B` 使用 RCC 的真实 `rcc_kernels.py` 定义，float64 构建；与表示用同一分子、同一物理键顺序。检查：

- `B@1`：max abs **7.36e-16**，相对 `||B||_F` **1.00e-15**。
- `B@H` 与 `B@H_within`：所有状态/对象的最大相对差 **≤3.1e-15**。
- 响应均值能量占比 ~`3e-33`（即 `B` 只作用于 within 部分）。

| 状态/对象 | response PR | response top1 | 每分子 top1 中位 | top4 中位 | amp = ‖B H_within‖/‖H_within‖ 中位 |
|---|---:|---:|---:|---:|---:|
| D0 `b` | 9.733 | 0.224 | 0.397 | 0.838 | 0.678 |
| D0 `c_loc` | 2.281 | 0.647 | 0.621 | 0.945 | 0.755 |
| D0 `c_glob` | 2.989 | 0.564 | 0.677 | 0.935 | 0.630 |
| D1 `b` | **20.320** | 0.122 | 0.301 | 0.740 | 0.605 |
| D1 `c_loc` | **2.441** | **0.631** | 0.635 | 0.901 | 0.678 |
| D1 `c_glob` | **5.609** | 0.380 | 0.481 | 0.862 | 0.575 |
| R0 `h_RCC` | 2.119 | 0.676 | 0.571 | 0.930 | 0.750 |
| R0 `Z_RCC` | 1.492 | 0.813 | 0.727 | 0.972 | 0.755 |

D0→D1 的响应变化与 within 维度同向：`b` response PR +10.59、top1 −0.102；`c_glob` +2.62、−0.184；`c_loc` +0.160、−0.016，amp 中位 −0.076。结论只能是：**冻结关系算子在 D1 表示上有非零、非单方向的代数响应；`c_loc` 的响应仍被少数方向主导（top1≈0.63），`b`/`c_glob` 的响应谱更分散。** 响应幅度/PR/top1 的 p10/p90、逐分子分布与分层统计在 `structure_response.json`。这不是标签效用或泛化收益的证据。

---

## 5. 判读与决策

### 主结论：`MIXED_OR_INCONCLUSIVE`

- **初始化低秩成立（对 RCC）**：`h_RCC` within-PR 2.478（top1 0.621），`Z_RCC` 1.594（top1 0.785）；在 RCC 旧的 2,560 行/63,921 键子集上精确复现 PR **2.570697**、top1 61.0%、top5 85.0%、top10 91.4%。
- **DCC-L 不是单一结论**：`b` 和 `c_glob` 从 D0 到 D1 的 within 有效维度明确增长，符合 `INIT_ONLY_DEGENERACY_INDICATED` 的子情形；`c_loc` 只从 2.61 升到 3.16，top1 仍 0.55，且最小键数分层无改善，更接近 `PERSISTENT_LOW_DIMENSION` 的子情形。`c_loc` 的主方向几乎完全换掉（top1 cos²=1.65e-4），所以它不是“放大旧方向”，而是“重组成新的低维方向”。
- **不能把 `c_loc` 的 D0→D1 变化当成 RCC 的反事实**：`h_RCC` 是“同时消费 ownership 与 global moments 的统一 composer”的输出，`c_loc` 是另一个函数；RCC 没有监督 checkpoint。注意 R0 `h_RCC` (2.478) 与 D0 `c_loc` (2.605) 在初始化时数值接近，但对象不同；不能把 `D1 c_loc (3.161) − R0 h_RCC (2.478)` 当作“RCC 训练增量”。
- **没有 seed1**：所有 D0→D1 差异都是单 seed 的描述性变化，不是跨 seed 确认。
- 因此不能选 `INIT_ONLY_DEGENERACY_INDICATED`（`c_loc` 仍低维、最小键数层无改善），也不能选 `PERSISTENT_LOW_DIMENSION`（`b`/`c_glob` 明确增长）；选 `MIXED_OR_INCONCLUSIVE`。
- 禁止把本轮的高 PR 读成“有效预测信息增加”，把高结构响应读成“泛化收益”，把 `R²=0.994` 读成“监督信息覆盖率”，或把 DCC-L 训练后的表示当成 RCC 训练后的反事实。

### 三个命题的边界

1. **RCC 初始化低秩**：本轮直接成立（h 2.48，Z 1.59；旧统计复现）。
2. **DCC-L 监督训练是否提高自己的键内有效秩**：组件依赖。`b`/`c_glob` 明确提高；`c_loc` 小幅提高但仍低维。只覆盖 DCC-L seed0 L 臂。
3. **RCC 若接受监督训练是否也会提高并带来泛化收益**：**无法回答**；没有 RCC 监督 checkpoint，本轮未训练。

### 对下一代确实被数据支持的约束

1. 分子共同项可以占键表示能量的 89%，所以任何“键表示维度/范数”结论必须先做分子内中心化；否则读到的多半是分子级共同项，而不是键内变化。
2. 监督训练不会均匀地修复低维：同一 DCC-L 内，`b`/`c_glob` 明显变高维，`c_loc` 仍约 3 维且 top1≈0.55；因此不能把“训练后会变高维”当作默认假设。
3. 在内容通道的 within 有效维度仍只有 ~3 时，追加图级二阶通道没有证据支持；本轮只说明冻结 `B` 对 D1 表示有非零代数响应，不说明该响应可预测或可泛化。
4. 没有 RCC 监督 checkpoint，不能把 DCC-L 的 D0→D1 结果外推成 RCC 的训练效果；下一代若重启 RCC 类设计，必须先有一个可测的“within 表示形成”阶段，而不是先加二次核。

---

## 6. 复现与哈希

```bash
uv run python tracks/structural-sufficiency/zinc_frozen_bond_representation_audit_v1/code/frozen_representation_audit.py
```

只写 `analysis/representation_spectrum.json` 与 `analysis/structure_response.json`；不落 10k embedding bank。

| 文件 | sha256 |
|---|---|
| `code/frozen_representation_audit.py` | `5b6cf582a4223682cc5a94f5f89c39fb59a9e5d88c1408ff6785ad2e0874a8e2` |
| `dcc_models.py` | `ac6c11edbcbd558ceabe9beea140d784ab79234ed4052bb2fd7484bebefa1974` |
| `rcc_models.py` | `e2cf981cabe5452e171e569060e36206f634de7e8d5ed5cfe3b4e2a293734d84` |
| `rcc_kernels.py` | `eed4494539b99f0665b38d6321355de204faeff84db7d2d7c5da93d5d2bb0476` |
| `ad_train.pt` | `d85d9875878a3cc0df27aeb8ad80e6cddf809e9d8265d77ef423a91bef4e19e3` |
| `kernel_bank.npz` | `19cfcecd99f76a425c71eddd949df07f602fc49a72c0069fc955a9f596a833a3` |
| `L_s0/best.pt` | `fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5` |
| D0 state | `f5c34c285c98b4d8a88361cb7dc1043b5abf07799f4f6121b25a59c904166e33` |
| D1 state | `c947bb7217a5358685bee3be762c9dc0b4ac8a26dfe94b8aa15f9395d02a1642` |
| R0 state（当前 t_scale） | `230f44f8b2519dc816e262b7e4cf051b9e43c61b54f82cb6482c873d34105f4a` |
| R0 state（t_scale=1，旧 prep 记录） | `5f2afdee649e04a8acdbd3e80d8d1fa21e6af7569914d64b9ec2d5c7f9f53a80` |

完整 provenance、逐项谱、分层、重排检查与响应统计见两个 JSON。

---

## 7. 五句话回答

1. **RCC 的初始化低秩是否成立？** 成立：`h_RCC` within-PR 2.478、top1 0.621，`Z_RCC` PR 1.594，并在 RCC 旧子集上精确复现 2.570697 的旧统计。
2. **DCC-L 监督训练后键内有效维度如何变化？** `b` 11.27→18.61、`c_glob` 2.69→5.59，明确变高维；`c_loc` 仅 2.61→3.16，top1 0.60→0.55，且主方向几乎完全换掉（top1 cos²≈1.7e-4），但仍低维。
3. **共同分子信息影响多大？** 它占能量 6–89%，但本身低维，所以 `C_all` 与 `C_within` 的 PR 差通常只有 −0.17 到 +0.50；它主要破坏基于原始范数/能量的判断，而不是 within-PR 本身。
4. **冻结关系算子在 D1 表示上还有可用响应吗？** 有非零代数响应（`B@1≈0`、`B@H=B@H_within` 到 1e-15）；D1 `c_loc` 仍 top1≈0.63/PR 2.44，`b`/`c_glob` 更分散（PR 20.3/5.61），但这只证明算子可作用，不证明表示已学到可用的结构关系（无标签/泛化证据）。
5. **对下一代形成了什么确实被数据支持的约束？** 必须把共同项从能量/范数解释中剥离；不能假设监督训练会自动把 `c_loc` 类内容变成高维；在 within 有效维度仍 ~3 时追加图级二阶通道没有证据支持；且没有 RCC 监督 checkpoint，不能把 DCC-L 的结果当作 RCC 的反事实。
