# lcrc_scaffold_error_pilot_v1 — REPORT

**Scope.** P0 只复用旧 scaffold split 的 9 个冻结 checkpoint 做逐分子误差归因，不训练。P1 在同样的 4,200 个 Lipophilicity 分子上生成**一个新的** Bemis–Murcko scaffold 划分（`split_seed=20261102`，3360/420/420），训练 BASE 与 G2L-ALL × seeds 0/1/2 共 6 个 run。P1 是 development-style 的新划分，不是 blind external validation；结论只针对这一个 split。

## 1. 结论摘要

1. **P0 复现通过。** 9 个旧 checkpoint 重推理后 valid/test RMSE、MAE 与旧 `metrics.json` 全部在 `1e-5` 内一致；未重训、未改旧文件。
2. **旧 scaffold Test 上，G2L-ALL 的 MAE 优势是“广泛小幅改善 + 少数大幅损伤”。** 三个 seed 的 MAE 都更好（+0.0197/+0.0181/+0.0068），但 RMSE 在 seed 1/2 更差（+0.0184/+0.0073）。SSE 损伤高度集中：top 5% 损伤分子占 gross damage 的 52.6/60.9/56.3%，而 seed 1/2 的净 SSE 恶化只有 +11.0/+4.3。
3. **新 scaffold split 上，inner Validation 与 outer Evaluation 方向相反。** Inner Validation 上 G2L-ALL RMSE 三个 seed 都略差（G2L−BASE = +0.0042/+0.0028/+0.0038）；outer 上 G2L-ALL RMSE 三个 seed 都更好（−0.0049/−0.0432/−0.0081，均值 −0.0187）。Outer MAE 只在 seed 1 更好，seed 0/2 更差；均值 −0.0048 由 seed 1 主导。
4. **Outer RMSE 优势也不是“每个分子都更好”。** 三个 seed 的 `frac ΔAE>0` 只有 0.510/0.543/0.486；outer RMSE 净增益是大幅 recovery 与大幅 damage 相抵后的小残差，top 5% recovery 是净增益的 8.4/0.84/6.3 倍（seed 1 较分散，seed 0/2 高度集中）。
5. **当前证据不足以把 G2L-ALL 视为稳定的结构泛化收益。** 旧 split 的 MAE 优势没有在新 split 复现；新 split 的 RMSE 优势主要来自 seed 1 和少数分子；inner 选择也看不到这个优势。需要独立 split 复核后才能决定是否继续投入。

## 2. P0：旧 scaffold checkpoint 的逐分子归因

旧 scaffold Test（420 分子，3 seeds；`Δ = BASE − G2L`，正数 = G2L 更好）：

| Seed | BASE RMSE | G2L RMSE | ΔRMSE | BASE MAE | G2L MAE | ΔMAE | frac ΔAE>0 | net SSE (G2L−BASE) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.72330 | 0.71278 | +0.01053 | 0.54739 | 0.52766 | +0.01973 | 0.536 | −6.349 |
| 1 | 0.70238 | 0.72076 | −0.01838 | 0.54318 | 0.52508 | +0.01810 | 0.564 | +10.987 |
| 2 | 0.70325 | 0.71052 | −0.00727 | 0.52877 | 0.52199 | +0.00678 | 0.507 | +4.316 |

旧 scaffold Validation 上三个 seed 的 ΔRMSE 全为正（+0.02456/+0.05085/+0.03178），ΔMAE 也全为正（+0.01290/+0.02472/+0.01207）。因此旧 split 的结论是：**Validation 上 G2L 全面更好；Test 上 MAE 更好但 RMSE 在 2/3 seeds 更差。**

损伤集中度（Test，G2L）：

- top 5 个损失分子贡献 gross damage 的 23.9/29.8/27.0%；top 10 个为 36.6/45.0/40.8%；top 20 个为 51.5/59.9/55.2%。
- 按分子比例，top 5%（21 个）为 52.6/60.9/56.3%，top 10%（42 个）为 71.1/77.9/71.4%。
- seed 1/2 的净 SSE 恶化仅 +11.0/+4.3，而 top 5% 损伤已达 37.1/30.7（3.4/7.1 倍）。剔除 top 5% 损伤分子后 G2L 在三个 seed 都反超 BASE；改按 top 5% BASE 误差剔除时只有 seed 0 反超。该剔除只用于诊断来源，不能替代完整 Test 指标。
- 损伤分子跨 seed 不稳定：top 5% 三 seed 交集仅 2/21，top 10% 为 8/42，损失秩相关仅 0.22–0.31。两个分子（row 231、688）三 seed 都进入 top 5%，其中 row 231 平均损失最大（+3.93）；但不存在一个稳定的“困难分子集合”。

化学特征（Test，G2L）：

- 没有强单调关系：true logD、重原子、芳香环、杂原子的 `|Spearman ρ|` 多 <0.13。
- 唯一较明显的是 `max_train_tanimoto`：`ρ(ΔSE)=+0.150`；`[0.2,0.3)` 组 n=50，平均 ΔAE=−0.075、ΔSE=−0.215；`[0.7,1.0]` 组为正。但 Validation 的同一 `[0.2,0.3)` 组方向相反（ΔSE=+0.272），所以只能算 test-specific 线索。
- 每个 seed 的 top 20 损失分子覆盖 20 个不同 Bemis–Murcko scaffold，没有共同骨架。

## 3. P1：新 scaffold split 与 6 个正式 run

划分：4,200 分子全部保留，3360/420/420；2,408 个 Bemis–Murcko scaffold group（1,857 个 singleton）；scaffold group 不跨 split；split audit 通过；Train-only target normalization（mean 2.1711011905，scale 1.2179624532）。完整审计见 `analysis/split/split_audit.json`。

训练：冻结旧引擎的 BASE / G2L-ALL 结构、AdamW 配方（peak_lr 3e-4、min_lr 1e-5、weight_decay 1e-3、batch 128、360 epoch cap、dropout 0.10）；同一 seed 的 shared init 逐位一致（`shared_init_all_shared_bitwise_identical=true`），shuffle seed 固定为 917000。Best epoch 由 Inner Validation RMSE 选择；6 个 checkpoint 先冻结 manifest，再读取 outer 标签。

| Model | Seed | Best epoch | Inner Valid RMSE | Outer RMSE | Outer MAE |
|---|---:|---:|---:|---:|---:|
| BASE | 0 | 151 | 0.60269 | 0.66422 | 0.48301 |
| BASE | 1 | 180 | 0.59399 | 0.70757 | 0.50098 |
| BASE | 2 | 195 | 0.57895 | 0.64299 | 0.44298 |
| G2L-ALL | 0 | 86 | 0.60687 | 0.65928 | 0.48999 |
| G2L-ALL | 1 | 146 | 0.59679 | 0.66439 | 0.46685 |
| G2L-ALL | 2 | 115 | 0.58274 | 0.63492 | 0.45586 |

3-seed mean ± sd：

| Model | Inner RMSE | Inner MAE | Outer RMSE | Outer MAE |
|---|---:|---:|---:|---:|
| BASE | 0.59187 ± 0.01201 | 0.42685 ± 0.00775 | 0.67159 ± 0.03292 | 0.47566 ± 0.02970 |
| G2L-ALL | 0.59547 ± 0.01212 | 0.42758 ± 0.00974 | 0.65286 ± 0.01574 | 0.47090 ± 0.01742 |

同 seed 配对差（G2L−BASE；负数 = G2L 更好）：

| Seed | Inner ΔRMSE | Inner ΔMAE | Outer ΔRMSE | Outer ΔMAE |
|---:|---:|---:|---:|---:|
| 0 | +0.00418 | +0.00320 | −0.00493 | +0.00697 |
| 1 | +0.00281 | −0.00032 | −0.04319 | −0.03413 |
| 2 | +0.00380 | −0.00070 | −0.00806 | +0.01288 |
| mean ± sd | +0.00359 ± 0.00071 | +0.00073 ± 0.00215 | −0.01873 ± 0.02124 | −0.00476 ± 0.02561 |

Win counts（G2L 更好）：Inner RMSE **0/3**，Inner MAE 2/3；Outer RMSE **3/3**，Outer MAE **1/3**。

**Inner vs outer 必须分开看。** Inner Validation 没有给出 G2L 的 RMSE 优势，反而三个 seed 都略差；outer RMSE 的优势是另一个方向，且 seed 1 的 −0.0432 远大于 seed 0/2 的 −0.0049/−0.0081。Outer MAE 均值改善 −0.0048，但只有 seed 1 改善，seed 0/2 分别变差 +0.0070/+0.0129。因此不能把 outer RMSE 的结果说成“inner 选择出的稳定增益”。

**Outer 逐分子诊断（只看已保存预测，不参与选择）：**

- `frac ΔAE>0` = 0.510/0.543/0.486；三个 seed 都只是“约一半分子更好”。
- net SSE gain（BASE²−G2L²，正 = G2L 更好）= +2.74/+24.88/+4.33。
- top 5% recovery 是净增益的 8.4/0.84/6.3 倍；top 5% recovery 占 gross recovery 的 54.0/45.5/55.8%；top 5% damage 占 gross damage 的 43.0/43.9/45.9%。
- 也就是说：**RMSE 净优势是大幅 recovery 与大幅 damage 相抵后的残差**；seed 0/2 尤其集中，seed 1 的增益更分散。该诊断不能反过来用于挑分子或重排名。

## 4. 高风险步骤检查

| 检查 | 结果 |
|---|---|
| P0 checkpoint 重推理一致性 | 9/9 复现，valid/test RMSE 差 <1e-5 |
| P1 row ID / SMILES / label 对齐 | 420 valid + 420 outer 唯一；valid/outer 行交集 0；行哈希写入 `metrics.json` |
| Scaffold leakage | audit passed；scaffold group 不跨 split；3360/420/420 |
| Train-only normalization | `fit_on=pilot Train3360 only`；`valid_or_test_used=false` |
| Best epoch 保存 | 6/6 `curve argmin == checkpoint best_epoch`；重载 inner RMSE 差 <4.4e-8 |
| Shared init / shuffle | 同 seed shared init 逐位一致；shuffle seed 均为 917000 |
| Outer 只读一次 | 6 个 checkpoint 先写入 `manifest.json` + `MANIFEST_FROZEN`；之后才运行 `eval-outer`；`outer_used_for_selection=false` |
| Outer 标签未用于训练 | 6 个 `train.json` 均记录 `outer_labels_read=false`、`test_row_values_used=false`；`label_scope=train3360_and_valid420_only` |

执行层面有两次**未进入正式训练**的失败：最初两条 launch 命令多传了 `--json`，argparse 退出；修正后又发现 GPU 主机 venv 没有 RDKit，训练视图在构造时依赖 RDKit。已移除训练路径的 RDKit 依赖、重新 freeze、重新 deploy；6 个正式 run 都是第一次成功训练，没有重跑/重试正式 run。首次 `rr deploy` 还因远端旧 `smoke_cuda` 未跟踪文件与本地新 commit 冲突而失败，移走旧文件后 deploy 成功。

## 5. 解释与不确定性

**可复现的观察：**

- 旧 scaffold Test：G2L MAE 三个 seed 都更好，RMSE 两个 seed 更差；SSE 损伤集中在少数分子，跨 seed 复发弱。
- 新 scaffold split：inner G2L RMSE 三个 seed 都略差；outer G2L RMSE 三个 seed 都更好，但 outer MAE 只 1/3 更好；outer 增益是少数大 recovery 抵消大量小 damage 的净结果。
- 两个 split 的指标方向不一致：旧 split 是“MAE 好、RMSE 混合”，新 split 是“RMSE 好、MAE 混合”。这不是 replication，而是两个不同 partition 的样本差异。

**可能但未验证的解释：**

- G2L 的 correction 可能只在少数“上下文匹配”的分子上产生大修正；这些分子在某个 split 的 outer 集里恰好偏向 recovery，在另一个 split 里偏向 damage。P0 的 `max_train_tanimoto` 弱关联和 P1 的少数大 recovery 与此相容，但都没有证明机制。
- Inner/outer 方向相反可能来自 420 行 inner/outer 的组成差异、best epoch 的早停位置（G2L 的 best epoch 明显更早：86/146/115 vs 151/180/195），或 seed 1 的偶然大增益；目前无法分离。
- MAE 与 RMSE 的分歧说明误差分布形状在变：G2L 可能在少数分子上大幅改变误差，而多数分子只有小幅变化；这仍是描述，不是机制。

**相反证据 / 限制：**

- 旧 scaffold Test 的 MAE 优势没有在新 split 复现（新 outer MAE 仅 1/3 更好）。
- P1 outer RMSE 均值优势的 sd 很大（−0.0187 ± 0.0212），且 seed 1 一项就贡献了大部分；n=3、一个 split，不能估计 split-level 方差。
- Inner Validation 完全没有显示 G2L 的 RMSE 优势；如果只用 inner 做模型选择，G2L 不会被选中。
- P0 的严重损失分子跨 seed 不稳定（top 5% 交集 2/21，秩相关 0.22–0.31），“固定的少数困难分子”假设不成立。
- 没有发现强化学特征解释；`max_train_tanimoto` 的 test 关联在 valid 上方向相反，不能作为可靠规律。
- P1 是同一 4,200 分子的新 scaffold partition，outer 不是外部数据集；不能外推到新的化学空间或“真实部署”。

## 6. 下一步选项（不在本轮执行）

**选项 A（推荐，若继续 G2L 线）：再用一个独立 Bemis–Murcko split 做同样的 6-run 复核。** 假设：P1 的 outer RMSE 优势不是该 split 的偶然组成。最小实验：新 `split_seed`，同样 3360/420/420、BASE/G2L-ALL × 3 seeds、同一配方、先 freeze 后 outer 一次；约 1 GPU·h。判断规则：若 G2L 在 ≥2/3 seeds 的 outer RMSE 更好且 3-seed mean MAE 不差于 BASE，则继续；若 RMSE 方向翻转或 MAE 继续 2/3 更差，则停止 G2L-ALL 线。

**选项 B（停止/降级）：保留 BASE，停止 G2L-ALL 的新训练。** 假设：现有证据不足以证明 G2L 的增益超过 split/seed 噪声。最小验证：不新增训练，整理 P0+P1 与旧轮结果，明确“未复现旧 split 的 MAE 优势”；只有未来出现独立 split 上 RMSE 与 MAE 同向的 ≥2/3 seed 证据时才重启。该选项与 A 互斥。

## 7. 产物与复现

- P0：`P0_FINDINGS.md`、`analysis/p0/per_molecule_scaffold.csv`、`analysis/p0/error_attribution.csv`、`analysis/p0/p0_summary.json`、`analysis/p0/figures/fig1–3`。
- Split：`analysis/split/split_pilot_v1.json`、`analysis/split/split_audit.json`。
- Freeze：`analysis/protocol_freeze.json`（split/config/code hashes）。
- Runs：`analysis/runs/pilot_{base,g2l_all}_s{0,1,2}/`（`train.json`、`curve.csv`、`checkpoint.pt`、`diagnostics.json`、predictions）。
- Manifest：`analysis/manifest.json`、`analysis/MANIFEST_FROZEN`。
- Outer eval：`analysis/eval/table_main.csv`、`table_summary.csv`、`table_paired.csv`、`table_attribution.csv`、`per_molecule_outer.csv`、`metrics.json`。
- Provenance：`analysis/gpu_ledger.json`、`analysis/rr_status/`。
- 代码：`code/pilot_common.py`、`pilot_data.py`、`pilot_p0.py`、`pilot_train.py`、`run_pilot.py`；配置：`configs/pilot.yaml`。
- GPU：6 个正式 run，A100-SXM4-40GB，各 1 GPU；按训练进程墙钟合计 **1.024 A100 GPU·h**（cap 2.0）。`gpu_ledger.json` 的 `rr_elapsed_seconds` 因 rr status 字段差异为 null，但 `process_wall_seconds`/`train_wall_seconds` 已完整记录，GPU 估算用前者。
