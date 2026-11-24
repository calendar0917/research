# GSRO v0 — 冻结实验定义

- Round：`graph-structural-response-operator-v0`；协议 id `gsro-v0-lipo-s0`。
- 数据：`ogbg-mollipo` 官方 scaffold split（Train/Valid/Test = 3360/420/420），seed 0。
  原始图、物理键、split、row id 与标签复用 `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`
  （SHA-256 见 `configs/gsro_v0.json`；`gsro_data.py` 直接调用该 track 的 loader）。
- Test 身份：`DEVELOPMENT_TEST`。checkpoint、horizon 与任何选择只由 Train/Valid 决定。
- 两臂：`FULL`（直接求解 `K x = s`）与 `LOCAL2`（归一化逆算子的 0/1/2 阶显式截断），
  共享同一参数生成器、初始化、数据与优化配置；固定 `K=16` 个响应通道。
- 输入：9 个原始 atom categorical（每列 embedding 16）+ 一跳 `atomic number × bond type`
  联合计数（log1p，Train-only 缩放）；3 个原始 bond categorical（每列 embedding 16）。
- 读出：`y_hat_std = b + Σ_k (α_k/n̄ + β_k/n) Q_k`，目标按 Train 均值/标准差标准化，
  报告恢复到原始单位。
- 训练：AdamW，peak LR 1e-3，min LR 1e-5，weight decay 1e-4，batch 128，
  warmup 5 epochs + cosine，MSE（标准化目标），grad clip 5，FP32。
  每 epoch 全量评估 Train/Valid/Test（eval mode），Valid RMSE 选择 best checkpoint。
- 训练长度：正式训练前做简短吞吐量预检（只看速度，不看指标）。若预计 120 epochs
  超出 10–15 分钟目标，则在 90 / 60 中选择可行的共同长度；60 epochs 仍超出 25 分钟
  硬上限时报告计算瓶颈并停止。预检后重新初始化模型与随机状态。
- 正式训练：`res` 上 FULL 与 LOCAL2 分别使用 `res-gpu0` / `res-gpu1` 同时运行。
- 检查：正式训练前运行 `checks.json` 的 8 项（重编号/边序、batch vs 单图、K 对称正定
  与残差、FULL vs 独立 FP64、LOCAL2 vs 矩阵多项式、w=0 一致、梯度有限、分解复算）
  以及参数量/共享初值一致性。
- 解释性（仅 best-Valid FULL checkpoint）：64 个固定 hash Valid 分子的精确能量分解；
  每个分子至多 3 条边的解析耦合敏感度（FP64 autograd + 有限差分 spot check）与
  全通道 w=0 直接求解干预；完整 Valid 上的冻结替换（LOCAL2 / w=0）。
- 明确限制：耦合干预不是化学删键；能量贡献不是删边后的预测变化；不宣称真实化学因果。

## 运行入口

```bash
# 预检（checks + throughput pilot），在 res-gpu0
python tracks/structural-sufficiency/graph_structural_response_operator_v0/code/gsro_preflight.py \
  --mode both --out-dir tracks/structural-sufficiency/graph_structural_response_operator_v0/runs/preflight

# 两臂（不同 GPU）
python .../code/gsro_train.py --arm full   --epochs <horizon>
python .../code/gsro_train.py --arm local2 --epochs <horizon>

# best-Valid FULL checkpoint 的解释性验证
python .../code/gsro_explain.py --checkpoint .../runs/full/best.pt --out .../runs/full

# 本地汇总（main_table / curve / paired bootstrap）
python .../code/gsro_analyze.py
```

## 产物

`results/checks.json`、`results/pilot.json`、`results/curve_{full,local2}.csv`、
`results/main_table.csv`、`results/predictions_{full,local2}.npz`、
`results/{full,local2}_best.pt`、`results/explanation.json`、实验根目录的 `REPORT.md`。
