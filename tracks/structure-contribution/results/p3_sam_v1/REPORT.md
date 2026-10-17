# REPORT — cscl-p3-sam-v1 (P3 vs P3-SAM)

**结论：`SAM_NO_USEFUL_GAIN`。** 固定 `rho=0.05` 的标准两步 SAM 没有改善 P3 的 held-out MAE；
在 Monitor 上两 seed 都明显变差，因此按预注册停止规则**没有读取 Locked 标签**，也没有启动
full10k、test 或任何 rho/seed/优化器变体搜索。

本轮只回答“改变监督优化的平坦性偏置能否缩小 P3 泛化缺口”，不修改任何历史 SAC/P3 判定。

---

## 1. 设计与纪律

| 项目 | 内容 |
|---|---|
| A 臂 | 冻结 P3-R3：413,505 参数 BJOC-Null、Fit-only masked SSL、Adam、cosine、raw-y L1、360 epoch、last-5 soup |
| B 臂 | 与 A 完全相同的模型/初始化/SSL/样本/update 数；唯一改动是 Adam 外层标准两步 SAM，`rho=0.05` |
| SAM | 第一次 backward 求全局 L2 梯度方向并扰动；同一 minibatch 第二次 backward；参数精确恢复；第二次 backward 后按冻结规则 clip 5.0，再 Adam step |
| 数据 | 官方 ZINC-12K train 10k，排除 4 行输入核对异常后 9,996 行 |
| split T | 显式 salt `cscl-p3-sam-v1`，7,996 / 1,000 / 1,000；实际 split hash `aceed4408911304e…` |
| SSL | 仅 Fit 结构，7,560 updates；每 seed 一个 checkpoint，两臂共享 |
| 监督 | 仅 Fit raw y；22,680 updates；同 seed 两臂 batch order 哈希相同 |
| 读取 | 先冻结 8 个 Monitor/Locked 预测与 4 个 soup/2 个 SSL checkpoint 哈希；只读 Monitor 标签 |
| 官方 valid/test | 未读取 |

同 seed 两臂的 pre-SSL full/group/encoder 哈希、SSL encoder 哈希、batch order 哈希和 update
budget 均在训练脚本中逐项断言相同。

CPU 门禁：`tests/test_p3_sam_v1.py` 9/9 通过；driver `cpu_gates_T.json` 7/7 通过，覆盖参数扰动
精确恢复、扰动范数 = rho、optimizer/scheduler step 数、第二次 backward 后裁剪、last-5 soup、
`rho=0` 与冻结 P3 trainer 的 soup 逐位等价、实际 Fit 规模 update budget。

## 2. Monitor 主结果

| 指标（Monitor, n=1000, seed 均值） | P3 | P3SAM | P3SAM − P3 |
|---|---:|---:|---:|
| MAE | **0.087028** | **0.131579** | **+0.044551** |
| per-seed MAE | 0.083518 / 0.090538 | 0.137359 / 0.125799 | +0.053841 / +0.035261 |
| RMSE | 0.183349 | 0.363610 | +0.180261 |
| ordinary MAE (`y > -3`) | 0.078878 | 0.116336 | +0.037459 |
| tail MAE (`y <= -3`, n=77) | 0.184726 | 0.314292 | +0.129567 |
| top-10-removed Δ | — | — | +0.043406 |
| 分子级 paired 95% CI | — | — | **[+0.029202, +0.063349]** |
| seed 间 MAE 极差 | 0.007020 | 0.011560 | +0.004540 |
| seed 间 MAE 标准差 | 0.004964 | 0.008174 | +0.003210 |

所有 7 个冻结 gate 条件均为 false：方向、两 seed 同向、CI 上界、ordinary、top-10-removed、
RMSE 容差、tail 容差全部失败。两个 seed 的 Δ 都是**正**的（SAM 更差），不是“平均被一个 seed 拉坏”。

## 3. 逐问回答

### 3.1 SAM 是否稳定改善 P3 的 held-out MAE？

**否。** Monitor seed 均值从 0.087028 退化到 0.131579（+0.044551，相对 +51.2%）。两个 seed
分别 +0.053841 / +0.035261，方向一致地变差；分子级 paired CI 整段在零以上。ordinary 和
top-10-removed 也同向变差，RMSE 接近翻倍。由于 Monitor 已失败，按协议 Locked 标签未读取，
因此本轮没有 Locked 评价，也不存在“Locked 翻盘”的购买。

### 3.2 是否减少了 seed 间波动？

**否，反而增大。** Monitor MAE 的 seed 极差从 0.007020 增至 0.011560，标准差从 0.004964
增至 0.008174。SAM 没有把两个 seed 拉得更一致。

### 3.3 训练拟合、泛化误差与计算成本如何变化？

| 项目 | P3 | P3SAM |
|---|---:|---:|
| soup Fit MAE (s0 / s1) | 0.001384 / 0.000993 | 0.008593 / 0.008615 |
| 平均 Fit MAE | 0.001189 | 0.008604（约 7.2×） |
| Monitor MAE | 0.087028 | 0.131579 |
| 平均泛化 gap（Monitor − Fit） | 0.085839 | 0.122975 |
| 训练 wall (s0 / s1) | 522.8 s / 623.9 s | 970.0 s / 1220.6 s |
| 训练 wall 合计 | 1,146.7 s | 2,190.6 s（约 1.91×） |
| 每 update 前/反向次数 | 1 / 1 | 2 / 2 |

SAM 的 clean-point Fit MAE 仍然很低（~0.0086，远低于 Monitor 误差），所以这不是“模型没拟合
训练集”的简单欠拟合；但它确实把优化终点推向了更差的 held-out 解，并放大了 seed 间差异。
计算上 SAM 约 1.91× 训练 wall（2 次 forward + 2 次 backward），不能宣称与 P3 等计算量。

总 GPU ledger：**1.8480 / 3.0 GPU-hours**（含 smoke、SSL、两臂训练、预测，以及失败 smoke
与预测的补记账）。SSL 两 seed 合计 0.9095 h，监督训练合计 0.9270 h。

### 3.4 这份收益是否足以支持下一阶段 full10k？

**不。收益为负，不支持 full10k。** 冻结 decision 为 `SAM_NO_USEFUL_GAIN`；不追加 rho/seed/
优化器/ASAM/正则/dropout/loss 变体，不读 Locked，不读 official valid/test，不启动 full10k。

## 4. 证据冻结

| 对象 | sha256 |
|---|---|
| split T | `aceed4408911304e083c859cc050d2a993cb0df3daea9ae2e488ed4d8bf08538` |
| 预测合并哈希（8 文件） | `eea2790c36e81489c4392fcdc8005e2bc951bc38c3262e170fa4ec86c8bdc8e5` |
| P3 soup s0 / s1 | `3bc84f8d…206a` / `bac9afbe…dcc5` |
| P3SAM soup s0 / s1 | `45cc50a0…1ba3` / `e8ea9aa8…83bf` |
| SSL s0 / s1 | `66ba3c97…7c28` / `90986785…5ab4` |

完整的预测文件哈希、checkpoint 哈希、训练/评分记录和 ledger 见
`prediction_manifest_T.json`、`train_*_T_s*.json`、`monitor_score_T.json`、`decision.json`、
`gpu_budget_ledger.json` 与 `evidence_manifest.json`。

## 5. 解释边界与限制

1. 本轮只说明：**这个固定 `rho=0.05` 的标准两步 SAM 配置**在 P3 上没有泛化收益，且明显有害。
   它**不能**证明平坦极小值不是原模型的瓶颈，也不能否定其他 SAM 变体/其他 rho；但按预注册，
   本轮不购买这些变体。
2. SAM 不增加任何输入信息、参数或监督样本；本轮结果不能被解释为“新结构信息”的证据。
3. split T 来自同一个已被历史多轮使用的 official-train 10k 总体；与历史 E/F/S 的行重叠
   覆盖率为 1.0。它只提供一次新的分组重采样，**不是独立盲测**。
4. 只有两个 seed；对很小的效应分辨力有限。本轮观测到的退化远大于冻结阈值，方向也在两个
   seed 上一致，因此停止结论不依赖微小效应。
5. 官方 valid/test 未读取；Locked 标签未读取。
6. 失败 smoke 与预测的 GPU 时间已补记入 ledger；详见 `DEVIATIONS.md`。
