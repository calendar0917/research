# HIS-v0 预注册协议（在任何 GPU 结果之前冻结）

**Round:** `zinc-his-v0` · **Track:** `tracks/structural-sufficiency/zinc_his_v0/`
**冻结配置:** `configs/his_v0.json` · **冻结代码:** `code/his_{common,anova,train,cpu_checks,phase_a}.py`
**official test:** 不构造、不加载、不评分。
**seed:** 0 only。**绝不运行 seed 1。**

> **状态（执行后追加，协议正文未改）：全部阶段已执行完毕。** Phase A 通过（`‖H₃‖/‖H‖ = 0.655055`，预测位移 0.087689）；λ 一次校准为 `228.6982356901754`；2-epoch smoke 通过；两臂 600 epoch 并行完成（合计 0.843 A100·h）；Phase C 判定 **`RED`**（full `+0.007372`，ordinary `+0.003096`，r2-unseen ordinary `+0.006829`，正则器确实把 `‖H₃‖/‖H‖` 压到 0.00096）。结论与解释见 `REPORT.md` / `DECISION.md`。
> 执行期唯一未按字面执行的项：任务书正文印出的 ANOVA 公式不是残差（不满足三轴归零），实现改用标准的 `H_≤2 = AB+AC+BC−(A+B+C)+μ`；该修正发生在 GPU 之前并由 `tests/test_his_v0.py` 固定。

---

## 0. 一句话

在冻结的 LCRC-A 上，只增加一项训练期正则 `lambda * mean(H_3**2)`（`H_3` = 二跳路径通道有效响应的纯三阶 ANOVA 残差），
用**同轮同步训练的 REPLAY_s0 vs HIS_s0** 做单 seed 对照，按事前判据给出 GREEN / AMBER / RED / CPU_STOP。

## 1. 执行顺序（严格串行）

| 阶段 | 内容 | 产物 | 门禁 |
|---|---|---|---|
| A1 | CPU 数学/梯度/不变性/类别重编号/存取往返 gates | `analysis/cpu_checks.json` | 10/10 pass，否则 `CPU_STOP_CORRECTNESS` |
| A1' | 纯数学 pytest（fresh-clone safe） | 21 passed | 必须全过 |
| A2 | 冻结 LCRC-A s0 上的三阶能量 + 512 固定 train 分子干预 + 类别 DF + r2 支撑关系 | `analysis/phase_a.json` | 见 §2 |
| A3 | 门禁判定 | `analysis/phase_a.json → decision` | 不通过即停，不救 |
| B0 | commit + `rr deploy res` + GPU 校准 lambda（一次） | `analysis/lambda_calibration.json` | lambda 有限且在护栏内 |
| B1 | 冻结 lambda 进 config，commit | `configs/his_v0.json` | — |
| B2 | 2 epoch GPU smoke：冻结 trainer（REPLAY 参考）、新 runner REPLAY、HIS | `analysis/smoke/*` | REPLAY 与冻结 trainer 一致到执行容差；HIS 正则非零且无 NaN/Inf |
| B3 | 600 epoch 双卡并行正式 run | `runs/REPLAY_s0`, `runs/HIS_s0` | 见 §4 |
| C | 一次性 valid 分析 | `analysis/*` | 见 §5 判决 |

预计 ≤ 2.0 A100·h 硬预算；若 2 epoch smoke 的实测外推使两臂合计 > 2.0 A100·h，**停止并报告成本**，不改跑短训练。

## 2. Phase A 停止条件（事前固定）

任一不满足即停止，写 `NO_MATERIAL_THIRD_ORDER_CHANNEL` / `CPU_STOP_CORRECTNESS`：

1. **正确性**：`analysis/cpu_checks.json → all_pass == true`，且 pytest 21 项全过。
   （覆盖：ANOVA 精确重构、三轴边际为零、≤2 阶张量残差为零、纯三阶残差非零、类别重编号只置换残差、
   save/reload 逐位一致、batch 组合/原子重编号/键方向不变、`path_emb` 与 `W_path` 上梯度非零有限且
   与有限差分一致（相对误差 <5e-3）、penalty 对尺度重参数化不变、切片由真实前向证明。）
2. **`‖H_3‖ / ‖H‖ ≥ 0.10`**（冻结 LCRC-A s0 best checkpoint）。
3. **512 个固定 train 分子上的 `mean_abs_prediction_shift ≥ 0.010`**（把 `path_emb` 替换为其二阶 ANOVA 投影）。

诊断只读 official train；不读 valid/test 标签。

## 3. 正则强度校准（只做一次）

- 冻结首 batch：`randperm(10000, Generator(101))[:128]`。
- 冻结初始状态：`11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e`。
- `lambda = 0.10 * ||g_L1|| / ||g_reg||`，范数只在受影响参数（`path_emb.weight` + `W_path` 32 列）上取。
- 不读 valid/test、不产生优化步、RNG 隔离。
- 护栏：`lambda` 必须有限且 `1e-4 ≤ lambda ≤ 1e4`；否则停止报告（不换校准方法）。
- 写入 config 后**整个训练过程固定**；训练进程重算并与冻结值比对，断言相对差 ≤1e-4
  （护栏，不是科学阈值：CUDA `index_add_` 归约不是逐位确定的，所以要求 4 位有效数字可复现即可；
  校准作业本身会重复同一计算 3 次并记录实际离散度 `lambda_repeat_spread`）。
- **禁止用 valid 成绩重选 lambda。**

## 4. Phase B 执行门禁（两臂对照公平性）

| 门禁 | 判定 |
|---|---|
| 两臂初始参数逐张量相同 | `initial_state_sha256` 均等于 `11c698ae…` |
| 相同 SSL checkpoint | `006fabc1…` |
| 相同 train/valid feature cache | 文件与内容 SHA 均与冻结值一致 |
| 相同 train-only normalization | fingerprint `8c3e8813…` |
| 相同 first-batch rows / 每 epoch 数据顺序 / 更新次数 | `first_batch_rows` 相等；`47400` updates；`79` steps/epoch |
| 相同 LR 轨迹 | 每条 curve 的 `lr` 序列逐 epoch 相等 |
| REPLAY 与冻结 trainer 一致 | 2 epoch：epoch-1/2 full-valid MAE 与 online MAE 差 <1e-5，参数漂移 <1e-3（执行容差，不是逐位相等；CPU 上已预演为逐位相等 0.0，CUDA 上 `index_add_` 归约非逐位确定，实测 4.4e-7/9.0e-7 与 4.3e-5 参数漂移） |
| HIS 与 REPLAY 推理结构相同 | 同一 `build_model("A", …)`，参数数均 237,519 |
| 正则训练中真实非零、无梯度断开、无 NaN/Inf | 每步 penalty 有限；`penalty_first_step > penalty_last_step` 或至少记录；受影响参数梯度非零 |
| official test 未被触碰 | 所有 payload `official_test_loaded: false` |

## 5. Phase C 一次性 valid 分析（冻结清单）

按原规则选 full valid MAE 最小的单 checkpoint，并同时输出 epoch-600 checkpoint。全部 `delta = HIS − REPLAY`，**负 = HIS 更好**：

1. official valid full 1000 MAE
2. ordinary（`y > -3`，n=935）
3. radius-2-unseen ordinary（n=225；掩码来自冻结 anchor 列 `r2_unseen_atom_frac`）
4. seen ordinary
5. tail（`y ≤ -3`，n=65）
6. 去掉历史固定 `row172`
7. 去掉历史 A_s0 固定 worst-10（`[172, 917, 753, 759, 224, 447, 678, 659, 267, 260]`，来自 `zinc_cjp_ssl_v1/configs/cjp_ssl_v1.json`，该列表在该轮 GPU 结果之前冻结）
8. W20、T100、final valid MAE
9. eval-mode train MAE（best 与 final）
10. 逐分子 paired delta、bootstrap CI（10,000 次按分子重采样，seed 固定）、win fraction、正负贡献最大的分子

**判决（事前固定）**

**GREEN — 有继续研究价值**

- full MAE `≤ −0.005`
- ordinary `≤ −0.002`
- r2-unseen ordinary `≤ −0.008`
- 去 row172、去固定 worst10 **仍为改善**（delta < 0）
- W20 与 T100 的 delta **不超过 +0.002**

**AMBER — 局部线索，不能晋级性能模型**

- 不满足 GREEN
- 且 r2-unseen ordinary `≤ −0.008` **且** ordinary `≤ −0.002`
- 且 full 恶化不超过 `+0.002`
- 且改善不能只由一个极端分子驱动（去掉单分子最大负贡献后，该组 delta 仍 < 0）

**RED — 停止**

- 上述条件均不满足；或
- **真实有效正则作用没有建立**（`H_3` ratio 与 penalty 在 HIS 中相对 REPLAY 未出现实质下降，或 penalty 无有限非零梯度）；或
- official test 被意外访问（该 run 直接作废）。

**CPU_STOP** — Phase A 任一门槛/正确性失败。

## 6. 禁止事项（事前锁定）

无论结果如何，本轮**不**：

1. 启动第二个 seed；
2. 读取 official test（任何理由）；
3. 扫描 lambda / 用 valid 重选 lambda / 改校准规则；
4. 换损失、加权重、改模型宽度、加新通道；
5. 因失败而追加"补救"实验（包括去掉 weight decay、放大 penalty、给 `path_u/path_v` 也加约束）；
6. 修改旧 track 的任何既有实验结果。

失败本身就是结果。

## 7. 预算与记录

- 硬上限 **2.0 A100·h**（含校准、smoke、失败作业）。两卡并行（`res-gpu0` / `res-gpu1`）。
- 所有作业通过 `rr run res`；commit 固定；`analysis/rr_jobs.json` 记录作业 provenance。
- 训练产物：`runs/{REPLAY,HIS}_s0/{best.pt,final.pt,train.json,curve.csv,predictions/}`（git-ignored）。
- 分析产物：`analysis/{cpu_checks,phase_a,lambda_calibration,smoke,main_table,groups,contrasts,mechanism,summary}.json`（Git 追踪）。
