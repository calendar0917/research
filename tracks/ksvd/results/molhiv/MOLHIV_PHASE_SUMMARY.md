# molhiv 阶段首轮结果（2026-07-25）

> 协议：`molhiv-struct-probe-v0` / 烟测 `ogb-molhiv-v0`（**非**全量 10-seed 正式表）

## 环境

- `gsn-official`：torch 2.7.1 · pyg 2.6.1 · ogb 1.3.6
- 数据：`data/ogb/ogbg-molhiv`（41127 图）

## P1 结构-only（LR on \(s_G\)，共享 D）

### n=8000 子采样 scaffold（可信度中）

| 方法 | valid AUC | test AUC |
|------|-----------|----------|
| degree hist | — | **0.483** |
| pool=mean | 0.549 | 0.526 |
| pool=max | **0.571** | **0.611** |
| pool=attn | 0.490 | 0.629 |

- **结论**：结构通道 **弱但非噪声**（> degree）；`max` 在 val 上最稳。
- **attn** test 略高但 val 差 → 本轮 **默认 pool 改用 max** 做融合。
- n=2000 的 test AUC 0.87+ **作废**（test 仅 2 个正例）。

## P2 双通道烟测（n=3000，25 epoch，CPU）

| fusion | best val AUC | test@best_val |
|--------|--------------|---------------|
| gine_only | 0.904 | **0.781** |
| concat + s_G(max) | 0.874 | **0.698** |

- **结论（烟测）**：concat **未超过** gine_only；与 MUTAG 融合弱一致。
- 子集 + 短训 **不能** 当正式 molhiv 数字；仅说明管道通、结构通道暂无增益。

## 合成池化（对照）

C4：mean/max/attn ≈ 92%；rich+mean 94%（任务已饱和）。

## 下一步

1. 全量 scaffold 结构-only（`max_graphs=None`，pool=max/mean）— 过夜级  
2. 全量 gine_only vs concat/gate，≥3 seeds  
3. 结构通道升级：环偏置采样 / 边级 \(s\) / 可学习 attention（监督）  
4. 暂缓 FDDL；先证明融合不降分

## 复现

```bash
export PY=/home/calendar/.conda/envs/gsn-official/bin/python
cd tracks/ksvd
$PY -m code.run_molhiv_probe --max-graphs 8000 --pools mean,max,attn
$PY -m code.run_molhiv_dual --max-graphs 3000 --fusion gine_only --pool max --epochs 25
$PY -m code.run_molhiv_dual --max-graphs 3000 --fusion concat --pool max --epochs 25
```
