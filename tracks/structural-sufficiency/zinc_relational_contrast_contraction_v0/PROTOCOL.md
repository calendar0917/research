# PROTOCOL — `rcc-v0`（冻结于任何 GPU 轨迹之前）

配置：`configs/rcc_v0.json`（sha256 `09cea39f309a030728bd7797d86990caab0c73e5c50d8102ed9f091abd99614d`）。
协议在任何核/门禁/训练数字产生之前写完；阈值在 `configs/rcc_v0.json.gates` 中冻结，
**看到数字后未改动任何一条**。

## 0. 边界（硬约束）

- 数据：official PyG ZINC subset，只使用 official **train** 的 10,000 行；内部划分 split E
  （fit 7,996 / monitor 1,000 / locked 1,000，`split_E.json` sha256 `5f320cf9…`）。
- official **valid 与 test 从不加载**。门禁里做了双重审计：源码 token 扫描 + 进程内
  `torch.load` 路径记录（只允许 `ad_train.pt` 与 SSL 权重）。
- 标签纪律：**本轮零标签读取**。梯度活性探针用零目标；重建度探针只用无标签特征。
- 无 recursive message passing（原图/线图邻域隐状态都没有），无 Attention，无 ensemble，
  无目标派生特征，无 distance/scale 扫描，无 rank/reader 宽度搜索，无 seed 1，无第三个对比臂。

## 1. 架构（见 `DESIGN.md` §1）

共享单 composer（488→128→96）+ `Z=Linear(96,16)` + `T=triu(ZᵀM Z)`（136）+ reader
（558→256→256→1）。`M` 为 `B=K_R−K_D`（arm `real`）或 `K_D`（arm `null`）。

## 2. 冻结常量

| 项 | 值 |
|---|---|
| 核 | `K_R = sym(P)`，`P=(I+D⁻¹A_L)/2`，零度行 one-hot |
| Null | `d=K_R1`，`K_D=ddᵀ/Σd`，`Σd = n_bonds`（精确恒等式，门禁 1e-12） |
| rank / T 维 | 16 / 136（对称上三角含对角） |
| T 尺度 | 单标量 = 两臂 T 在 fit 行上的 RMS 合并值（12.9060），两臂共用 |
| batch / epochs / updates | 128 / 752 / 47,376（对齐历史 DCC-L 10k 的 47,400 updates，差 −0.05%） |
| 优化器 | AdamW，lr 1e-3 → 1e-5 per-step cosine，warmup 500，wd 1e-5，clip 5 |
| 损失 | raw-y L1 |
| 选择 | Monitor MAE 严格 `<`，等值留早 |
| batch order | `torch.Generator().manual_seed(seed+101)` + 每 epoch `randperm(len(fit))` |
| 预算 | 2.0 A100·h（双卡占用求和，含 probe/预测/失败） |

## 3. 启动门禁（24 条，`code/rcc_cpu_checks.py`）

必须在 GPU 之前全部通过。实现类：参数上限、两臂初始同一、核对称、`Σd=n_bonds`、`B·1=0`、
`K_D` 行和匹配与 rank-1、核非负、torch vs float64 二重求和（1e-10）、平移不变、规则图解析见证
（C3=0.5、C4=0.5、K4=2/3）、键重排/批次/跨分子、空键与单键与断连边界、梯度活性、无 valid/test
路径、确定性、reader 各块数值活跃。

信息量类（**决定是否购买 GPU**）：

| 门禁 | 阈值 | 含义 |
|---|---|---|
| `t_distinct_min_rel` | ≥ 0.05 | `T_real` 不能等同于度数 Null |
| `t_top1_share_max` | ≤ 0.90 | `T_real` 不能退化成 rank-1 |
| `t_redundancy_r2_max` | ≤ 0.90 | `T` 不能被已有 reader 块重建 |
| `t_redundancy_r2_max` | ≤ 0.90 | **`T_real − T_null`**（关系核真正多加的东西）也不能被重建 |

## 4. 决策规则（冻结）

- 任一硬门禁失败 → **STOP_BEFORE_GPU**。
- 参数 > 500,000 → STOP。
- `T` 在真实分子上退化（与 Null 不可区分 / rank-1 主导）→ STOP。
- `T` 主要由已有局部矩重复（重建 R² > 0.90）→ STOP（等价重参数化，符合用户给出的停止条件）。

若通过：

- GPU0 = `real`，GPU1 = `null`，两臂共享初始化/形状/数据/更新数。
- 附加参照（报告用，不作为"第二对比臂"）：`dccl` = 冻结 DCC-L 双通道模型 `organized_local=True`，
  同初始化 seed、同一 SSL、同一 recipe、同一 fit 行。
- 先做 timing probe（2 epoch，全 fit），投影 × 1.15 必须 ≤ 2.0 h；否则 STOP，**不缩短 horizon**。
- 判定：
  1. `real` 明显优于 `null`？→ 预注册为 Locked 上配对 bootstrap CI95 上界 < 0 且 `Δ ≤ −0.003`。
  2. `real` 是否达到/超过匹配 DCC-L Fit8k？→ `real − dccl ≤ 0` 且 CI95 上界 < 0。
  3. 仅 (1) 成立但绝对水平落后 → 关闭，记为"结构信号证据"，不晋级。
  4. `real ≈ null` → 关系对比假设不被支持，**即使绝对 MAE 更好也不得归因于关系核**。
  5. 只有 (1) 与 (2) 同时成立且持续指标（best20 中位数、tail-100 中位数）同向，才建议下一轮
     第二 seed 与 4k/8k 学习曲线。

## 5. 明令禁止

完整 10k 训练；读 official test；读 official valid；rank / reader / 核尺度 / 归一化搜索；
事后修复（post-hoc fix）；改阈值；把 `real−null` 的差异单独解释成"对应身份"的因果效应
（稀疏性/支撑集混杂仍在，见 `bond_relation_attribution_v1`）；用 full-10k SSL 而不披露。

## 6. 本轮执行状态

- CPU 门禁：**22/24 通过**，失败 `G11` 与 `G11b`（`analysis/cpu_gates.json`）。
- 判定：**`STOP_BEFORE_GPU_REDUNDANT_SECOND_ORDER_CHANNEL`**。
- GPU：**未部署、未运行**，开销 0.0 A100·h。因此没有"shortened horizon"问题，也没有任何
  Locked/Monitor 数值。
