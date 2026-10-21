# PROTOCOL — P3 dual-channel local/global composer v1

**Protocol ID:** `zinc-dual-channel-composer-v1`
**起点:** `074c13a5937c988c8bb03734fa6fb7b9c0b11f44`
**数据:** official PyG ZINC subset=True，train 10,000 / valid 1,000，raw-y MAE。
valid 是反复使用过的开发集，不是新盲测。official test 全程不读取。
**唯一正式 run:** L seed 0 与 C seed 0，单 seed；不跑 seed 1，不加第三臂。

## 1. 问题

审计发现 0.076 平台是一个共享重尾难例核心；25.1% 的 valid 分子带有
train 未见的 radius-2 局部环境并承担 49.8% 绝对误差。审计提出的待检验接口
假设是：P3 Null 分组把局部 junction 身份稀释为 `n_b × graph_mean(w)`。
本协议直接检验：在保留 P3 全局通路的同时，并联一个 organized 局部 composer，
是否能改善 valid MAE；同参数 Null-local 对照 `C` 用于把局部 ownership 与
额外容量分开。

审计的 H2（样本/结构支持不足）**未被排除**；本轮不把它当作已拒绝。H3 也
不是已确认主因，只是本轮待检验的接口假设。单 seed 结果只作 pilot 信号。

## 2. 模型

Base（完全保留 P3）：

```
a, b, w = P3 encoders(batch)
Wg_b = n_b * graph_mean(w)
Vg_b = n_b * graph_mean(w^2)
c_glob_b = group_composer([b_b, Wg_b, Vg_b, log1p(n_b), bond_feat_b])
```

局部通路：

```
L: Wl_b = sum_{j touches b} w_j
   Vl_b = sum_{j touches b} w_j^2
C: Wl_b = n_b * graph_mean(w)
   Vl_b = n_b * graph_mean(w^2)

c_loc_b = LocalComposer([b_b, Wl_b, Vl_b, log1p(n_b), bond_feat_b])
```

`LocalComposer = MLP(296 -> 80 -> 96)`，SiLU，dropout 0；只做一次
per-bond 静态 incidence 聚合，不做递归局部邻居 hidden-state 更新，不做 GNN /
message passing / attention。

Reader：

```
reader_in = [sum_b c_glob_b, sum_v a_v, sum_v a_v^2, standardized_global33,
             sum_b c_loc_b, sum_b c_loc_b^2]   # 513 dims
reader = MLP(513 -> 256 -> 256 -> 1)
```

**参数:** 494,193（P3 413,505 + local composer 31,536 + reader delta 49,152），
cap 500,000，headroom 5,807。

## 3. 初始化

1. 先按原 P3N builder 的完全相同路径构建 seed 0 P3（`torch.manual_seed(0)`），
   加载同一 train-only masked SSL encoder（sha256 `006fabc1...`）。
2. 再以固定第二 seed 构建 dual skeleton，复制 P3 全部非 reader 权重，reader
   第一层前 321 列复制 P3、后 192 列置零，bias 和后两层复制。
3. L/C 从**同一份完整 dual 初始 state** deep-copy；不分别构造导致 RNG 消费
   不同。CPU gate 比较两臂 state 逐 tensor 相同并记录 state hash。
4. 旧/新参数全部 fine-tune；不加零输出门控、不冻结 encoder、不缩小 encoder LR。
5. 不加载 P3B `best.pt` 作为初始化；参照 best 只用于比较。

## 4. 训练协议

复用 `zinc_recipe_trunk_v2` 的真实实现 `ad_train.train` 与 recipe B：

- AdamW；batch 128；600 个完整 train epochs；79 updates/epoch；47,400 updates。
- 500-step warmup；per-update cosine 1e-3 → 1e-5；wd 1e-5；clip 5；raw-y L1；
  dropout 0（保持 P3 配方）。
- 每 epoch 完整 valid-1000；strict `<` 保存 best，同值取更早 epoch。
- 保留 best.pt 与 final.pt；不重置 schedule、不改变 horizon、不按子群挑 checkpoint。
- L/C 同 seed、同 SSL、同 batch order（`seed + 101`）、同优化器/选择。

## 5. 预算与计时

- 硬 cap：`0.8594111111111111` A100-hours（上一轮 ledger `remaining_after`；
  若发现更低余额则以更低值为准）。两卡实际占用相加，不把并行误记为单卡墙钟。
- timing 必须是实际 A100 上 L/C 各 2–3 个**完整 10k train epochs**，每个都包含
  完整 valid-1000，确认 79 steps/epoch；1024-row smoke 只可查可运行，禁止用其
  “每 epoch”时间乘 600。
- 投影：`600 × mean_epoch_seconds × 1.15 + startup + final best/final eval +
  30s process reserve`；两臂 + 已花 timing <= 0.80h 软目标且 <= 硬 cap 才开正式。
- 若容纳不了两条完整轨迹：记录真实成本并 STOP；不能只跑 L、缩短 epochs 或只改
  一臂配方。
- 正式 job 的 walltime 上限之和 + 已花 <= 总 cap；接近 cap 时保存并停止剩余
  任务；未完成 600 epochs 标记 INCOMPLETE。

## 6. 预注册判据

主表：P3B / L / C 的 best-valid、best epoch、final-valid、best/final 固定
checkpoint full-train eval MAE、参数、GPU-hours。在线 train MAE 与 eval-mode
train MAE 分开报告。

性能信号（分别评价 L、C）：

- 某臂 overall MAE 比 P3B 低至少 0.003：值得下一轮确认的开发信号，不是统计
  显著性或 SOTA 阈值。
- 小于 0.003 的正向改善如实报告为“弱信号”。
- 若 C 赢 P3B、L 不比 C 强：保留 C 的性能线索，不因不支持 ownership 关闭。

归属信号：

- `L − C <= −0.002`：这个训练设置下真实分组有用的 pilot 信号。
- 只有 L 同时赢 P3B，才同时具备新性能和归属收益；L 只赢 C 但输 P3B 不算
  性能进展。
- L/C 相近且都赢：只能说并联通路/新增统计/训练路径有用，不能断言“只有容量
  起作用”或“Null pooling 是全部病因”。

安全性（报告 tradeoff，不用分组门槛替代总体 MAE）：

- ordinary `y > -3` 相对 P3B 退化不得超过 0.002；
- 固定 P3B worst-100 的绝对 MAE 不得恶化；
- top-100 占总误差的份额只作描述，不作硬 gate。

固定子群：signature-unseen 251 vs covered 749；ordinary `y>-3` vs `y<=-3`；
`n_rings>=4`；`n_rings>=4 且 unseen`；固定 P3B worst-100。每组报告 n 与
P3B/L/C MAE、差值。另报告 `L−P3B`、`L−C` 在 unseen 与 covered 的平均差异之差。
不按真实 y 或 unseen 分组做模型路由，不按子群选 epoch。

结果分类：

- A：L 赢 P3B 且赢 C → 局部分组有可购买信号，建议下一轮确认。
- B：C 或 L/C 均赢 P3B 但 L−C 不支持 → 保留性能候选，修正机制解释。
- C：L 赢 C 但未赢 P3B → 无新性能水平，结束该配置。
- D：均无收益 → 结束该配置；结论不外推为“未见环境不可泛化/全部局部组合无用”。

单 seed 在 exposed valid 上的 paired bootstrap 只是固定 checkpoint 的描述性
区间，不是跨 seed 确认。

## 7. 交付

- `code/`：`dcc_models.py`、`dcc_runner.py`、`dcc_cpu_checks.py`。
- `configs/zinc_dual_channel_composer_v1.yaml`。
- `runs/L_s0/`、`runs/C_s0/`：best.pt、final.pt、curve.csv、train.json、
  predictions/valid_best.npz、artifacts.json（大 artifact git-ignore，hash 入库）。
- `analysis/`：cpu_checks、timing、model_metrics、paired_contrasts、subgroups、
  per_molecule_dual.csv、decision、budget_ledger、artifact_hashes、curves.png。
- `REPORT.md`：Observed / Interpretation / Decision，明确 valid 暴露、test 未读、
  单 seed 局限。

## 8. 复现

```bash
# CPU gates
uv run python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_cpu_checks.py

# remote timing (correct rr form: host before experiment)
rr run res dcc-timing-L --pool res-gpu0 --gpus 1 --time 0:30:00 --result tracks/structural-sufficiency/zinc_dual_channel_composer_v1/runs -- \
  python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_runner.py timing --arm L --epochs 3 --device cuda:0

# formal L / C
rr run res dcc-L-s0 --pool res-gpu0 --gpus 1 --time <WALLTIME> --result tracks/structural-sufficiency/zinc_dual_channel_composer_v1/runs -- \
  python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_runner.py train --arm L --seed 0 --recipe B --device cuda:0
rr run res dcc-C-s0 --pool res-gpu1 --gpus 1 --time <WALLTIME> --result tracks/structural-sufficiency/zinc_dual_channel_composer_v1/runs -- \
  python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_runner.py train --arm C --seed 0 --recipe B --device cuda:0
```
