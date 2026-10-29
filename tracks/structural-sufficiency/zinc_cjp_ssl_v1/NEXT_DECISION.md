# NEXT DECISION — CJP-SSL v1

**日期**：2026-10-29
**依据**：`REPORT.md`、`analysis/{pretext,supervised}_summary.json`、`analysis/paired_comparisons.json`、`analysis/gpu_ledger.json`
**判决**：**D `NO_GAIN`**（+ 次级标签 `JOINT_PRETEXT_LEARNED_NO_DOWNSTREAM_GAIN`）

---

## 1. 决定

1. **不把 conditional joint-path SSL 设为默认预训练方式。** 在本任务书冻结的路径、目标、初始化与监督配方下，MARG-SSL / JOINT-SSL 相对同轮 A-REPLAY 的 seed-mean full valid MAE 为 `+0.00752` / `+0.00572`（正 = 更差），两 seed 同向；ordinary、去 row172、r2-unseen ordinary 均无改善。
2. **保留现有默认**：atom/bond SSL encoder checkpoint + LocalContent 标准初始化（即 A-REPLAY 配置）。本轮把它与 SCRATCH 的贡献重新测为 `−0.00812 [−0.01409, −0.00274]`（≈ −9.3% 相对），两 seed、所有子群一致。
3. **停止**继续在本协议上追加 seed、加长预训练、调掩码率/损失权重/头宽/epoch/LR。这些都属于本任务书明确禁止的调参，且已有两个 seed 的同向负结果。
4. **不**把本轮的负结果外推为「LocalContent 不能受益于自监督」的一般结论；它只否定本任务书固定下来的这一套协议（见 §4）。

---

## 2. 直接后果（工程侧）

| 项 | 决定 |
|---|---|
| 下游默认初始化 | 不变：旧 SSL encoders + 标准 LocalContent |
| `cjp_*` 代码 | 保留在 `tracks/structural-sufficiency/zinc_cjp_ssl_v1/`，作为可复现的负结果与门禁/预训练/分析工具；不接入默认训练管线 |
| 预训练产物 | `runs/pretrain_{marg,joint}_s{0,1}/local_state.pt`、`pretext_head_state.pt` 保留（带 SHA），不进入任何部署路径 |
| 部署参数 | 仍为 237,519；本轮未改变 LCRC-A 推理图 |
| official test | 仍未加载；未来若要用 test，必须另开任务书并先冻结选择规则 |

---

## 3. 若未来要重开这个方向，最低门槛

不是现在要做的重设计，只是把「什么证据才值得再投入」写清楚，避免下次用同一套协议重复烧 GPU：

1. **先解决评估分辨率**：valid 只有 1,000 行且被 `valid:0172`（`y=−20.34`）单行主导（同 pipeline 该行误差 6.0–12.7）。任何新方案在开 GPU 前应先冻结一个**对单行不敏感的主指标**（例如同时预注册 full 与 −row172，或改用按 y 分层的稳健汇总），否则 ±0.006 的差异无法解释。
2. **预注册 seed 数与判定阈值**：本轮 2 seed 的最小可检测效应约 0.005–0.008；若目标效应是 0.003，需要更多 seed 或配对设计（同 init、同 batch order、仅换目标）并预注册。
3. **把「初始化」与「目标」解耦**：本轮 MARG/JOINT 同时换了 LocalContent 的初始化来源；若想单独回答「joint vs marginal 目标」，需要一个两者共享**同一** LocalContent 初始化、只改 pretext 目标的对照。本轮已确认 MARG/JOINT 共享 init（`a3e8d8ad…`），但相对 A-REPLAY 仍混合了「SSL vs 标准 init」。
4. **先做小规模迁移探针**：例如固定 50–100 epoch 的监督预算、多 seed，测「预训练 LocalContent 是否在早期就落后」；若早期即落后，再加长训练没有意义。
5. **明确预算**：本轮 12 个正式 run 共 5.177 GPU·h / 3 h16m wall；任何重开应给出同量级的硬上限，并沿用 `cjp_ledger.py` 记账。

---

## 4. 本轮结论的边界（写进下次任务书，避免误读）

- 只否定：**7,560 updates、LR 1e-3 常数、WD 1e-5、batch 128、dropout 0、63 类目标、≤3 条顶点不相交二跳路径、LocalContent-only 可训练、冻结 atom/bond encoders、`f`/`rho` 不参与**这一组合。
- 只测：LCRC-A（`atom_encoder`/`bond_encoder`/`f`/`rho`/`LocalContent`）在 ZINC 10k → official valid 1,000 上的 raw-y MAE。
- 只测：冻结监督配方（600 epoch / 47,400 updates / AdamW 1e-3→1e-5 / warmup 500 / per-step cosine / WD 1e-5 / clip 5 / best-MAE checkpoint）。
- 预训练目标本身是**可学的**（held-out CE 0.26–0.32，远优于 conditional prior 1.42；JOINT 交互 RMS 3.0–3.2），负结果发生在**迁移**，不是优化失败。

---

## 5. 状态

- 交付物齐全并已提交（见 `REPORT.md` §11）；`analysis/cpu_checks.json` 全绿；8/8 监督 run、4/4 预训练 run 完成；`analysis/gpu_ledger.json` 记录 5.177 GPU·h，未超软/硬上限。
- 本轮**没有**未解决的 correctness blocker；所有冻结项均按任务书实现（G0–G12 + 转移校验）。
- 结论：**关闭本任务书的 Conditional Path SSL 方向；默认预训练不变。**
