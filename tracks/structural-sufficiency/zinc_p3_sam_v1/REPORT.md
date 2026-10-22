# REPORT — zinc_p3_sam_v1

Track `structural-sufficiency`。两臂单模型、seed 0：**C = P3-Control（原 recipe B AdamW）**
与 **S = P3-SAM（同一模型/初值/数据/日程 + vanilla global-L2 SAM, rho=0.05）**。
预注册见 `DESIGN.md` 与 `configs/p3_sam_v1.json`。官方 test 在历史轮次中已多次暴露，
本轮两次读取只作描述，不声称盲测，也不参与任何选择。

## 0. 结论

1. **固定 rho=0.05 的 vanilla SAM 在这个 recipe / 600 epoch / 413k 参数设置下明显变差。**
   C：best-valid `0.079588262` @431，eval-train@best `0.010286`，final-valid `0.081668`，
   test `0.072689`。S：best-valid `0.118310817` @299，eval-train@best `0.041025`，
   final-valid `0.140518`，test `0.100376`。
   **S−C = +0.0387 valid（95% CI +0.0231…+0.0626）、+0.0277 test（CI +0.0188…+0.0367）**，
   方向一致、幅度远大于本轮测到的同配置波动（~0.003），不是边际差异。
2. **SAM 同时恶化了正常拟合与泛化**：整条 valid 曲线在 epoch ≈150 之后一直高于 C
   （尾段 100 epoch 中位数 C `0.081656` / S `0.140731`），normal online train MAE 在
   epoch 600 为 C `0.00061` / S `0.00487`；而 SAM 第二遍在 θ+ε 的损失是 S 正常损失的
   ~10.8×（`0.0527` vs `0.0049`）。按预注册的判定，这是 **情形 C**：关闭这个固定
   rho + recipe + 600 epoch 的干预；它不证明“所有 SAM / 正则化无用”。
3. **Control 自身相对历史 P3B 有偏移**（valid `+0.00336`、test `+0.00282`，两者的
   分子级配对 CI 都跨 0）。这是**同配置重跑的数值漂移**，不是配置/hash/步数错误：
   两臂的 epoch-1 online train MAE 与历史 P3B 相差 `9.3e-9`（valid 相差 `4e-8`），之后
   由 CUDA 非确定性（`index_add_` atomics）逐 epoch 放大；C 的尾段 100 epoch 中位数比
   历史 P3B 高 `+0.0035`。因此**不能把历史偏移算作 SAM 的收益/损失**，也不能把
   `+0.003` 量级的单 seed 差异当成稳定信号；本轮 S−C 的 `+0.028` test 差比这个漂移大约
   10 倍，负结论不受影响。
4. **0.05x 目标仍然没有接近**：本轮最好 test 是 C 的 `0.072689`，为 0.05 的 **1.45×**
   （历史最好 L `0.066267` 为 1.33×）。SAM 没有产生新的性能水平。

正式训练**两卡并行**：C 在 res-gpu0、S 在 res-gpu1，并行墙钟
`13:39:46 → 14:22:06 = 42.3 min`；单臂 wall `1169 s` / `2516 s`，
GPU `0.325 h` / `0.699 h`（合计 `1.024 A100-h`，含 4 个计时 epoch、best/final 打分、
冻结与 test）。S 用 94,800 次梯度计算换 47,400 次外层次更新；这是**同 outer updates**，
不是同算力比较。

## 1. 本轮范围（与预注册一致）

- official PyG ZINC subset：train 10,000 / valid 1,000 / test 1,000；原始 raw-y L1；
  无 target clipping / 变换 / 重加权 / 额外监督。
- seed 0；同一 SSL 起点、同一 `state_dict`（两臂 init hash 相同）、同一份 train-only
  标准化、同一逐 epoch 行顺序（600 个 epoch 的 order hash 两臂逐一相同）。
- batch 128（`drop_last=False`，79 updates/epoch）× 600 epochs = **47,400** outer updates；
  AdamW peak `1e-3` / min `1e-5` / wd `1e-5`，betas `(0.9,0.999)`、eps `1e-8`（从代码核对）；
  500 步 warmup + per-step cosine，horizon 启动前固定；clip 5；dropout 0、无 BN。
- 每 epoch 全量 valid，strict `<` 选 best（相同取更早）；无 last-k soup / EMA / SWA /
  平均 / test 选点 / 早停 / seed 1 / rho 网格；没有第三臂。
- 两臂所有 encoder、composer、reader 全参训练，无冻结模块。

## 2. 实现与核验（训练前通过）

`analysis/cpu_checks.json`：**10/10 PASS**（`code/sam_checks.py`）；`code/tests/`
另有无数据依赖的 12 个快速单测通过。关键证据：

| 检查 | 结果 |
|---|---|
| 参数量 / optimizer 覆盖 | `413,505` = `413,505`，22/22 tensor 全部 require_grad；encoder 202,048 + Null composer 62,976 + reader 148,481 |
| 初始 state_dict | 两臂 `21963b43…16a30`，与 config 记录一致；同一 SSL `006fabc1…bfa97`（文件 sha 与 key 均核对） |
| 与原 P3 前向一致 | 历史 `xBssl_P3N_s0/best.pt` 重新载入后 full-valid = `0.076223738`，与记录 `0.076223751` 差 `1.3e-8`；两次独立构造前向逐值相同 |
| 无 BN / dropout | 0 个 BatchNorm，dropout rate 集合为空；train/eval 切换不改变前向 |
| rho=0 = AdamW | 5 步后 model、AdamW `exp_avg`/`exp_avg_sq`/`step` 逐值相同（diff = 0.0），LR 轨迹相同 |
| rho>0 手工两遍核对 | g1/g2/eps 范数、更新后参数、AdamW 状态与独立手写两遍计算逐值相同（diff = 0.0）；`step` 每批只加 1 |
| 第二遍确实在 θ+ε | 记录 g2 与 θ+ε 处手算 g2 相同，与 θ 处 g2 相对差 0.66；g1 不累加进第二遍（`grads_cleared_before_second_pass`、`accumulated_g1_plus_g2_rejected`） |
| ε 范数 / 第一遍不 clip | 合成大梯度 `‖g1‖=689.3 > 5` 时 `‖ε‖=0.0500000016`（相对偏差 3.2e-8），参数在 step 前已恢复、扰动逐值等于 `backup+ε` |
| 异常路径 | 注入第二遍 forward 异常后参数逐位恢复，无 checkpoint / 评估可写入 |
| 真实 P3 两遍 | 128 分子一批 loss/loss′ 有限，encoder/composer/reader 三部分均更新，扰动/恢复逐值精确 |
| 行顺序 RNG | 额外 79 次 SAM 两遍前向不改变 shuffle；两臂 600 个 epoch 的 `order_sha256` 完全一致（首 `8eca27da…`，末 `db5b8e71…`） |
| 日程计数 | 79 steps/epoch、47,400 updates；warmup 末 `1e-3`、step 501 开始下降、step 47,400 = `1e-5`，之后单调 |

训练期抽样（`analysis/*/sam_diagnostics.json`）：S 的 ε 全局范数均值 `0.0500000000`
（最大相对偏差 `6.7e-8`），26/26 抽样更新“θ+ε 逐值成立”、26/26“step 前逐位恢复”。
S 的第一遍 `g1` 均值范数 0.92、最大 11.97，**34/47,400 次超过 5 但从未被 clip**；
第二遍 `g2` 均值 1.60、最大 20.53，仅 0.61% 的更新触发 clip 5。C 的单遍梯度均值 5.47、
最大 43.06，52.4% 的更新被 clip 5（C 的 clip 语义就是原 recipe）。

## 3. 主表

| arm | params | best valid | ep | eval-train@best | final valid | test MAE | test RMSE | p95 AE | max AE | updates | grads | wall s | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C P3-Control** | 413,505 | 0.079588262 | 431 | 0.010286 | 0.081668123 | **0.072688588** | 0.165587 | 0.209016 | 2.504708 (row 402) | 47,400 | 47,400 | 1169 | 0.325 |
| **S P3-SAM** | 413,505 | 0.118310817 | 299 | 0.041025 | 0.140518076 | **0.100376384** | 0.176018 | 0.337065 | 2.035403 (row 988) | 47,400 | 94,800 | 2516 | 0.699 |

配对差（1000 个 test / valid 分子逐分子，10,000 次 bootstrap 重采样；**不反映跨 seed
训练不确定性**）：

- valid S−C = `+0.038723`，95% CI `[+0.023093, +0.062618]`
- test S−C = `+0.027688`，95% CI `[+0.018752, +0.036656]`；AE(S) < AE(C) 的分子比例 0.339
- 同轮 C−P3B：valid `+0.003365`（CI `[−0.001324, +0.009239]`）、
  test `+0.002824`（CI `[−0.000612, +0.006298]`）
- 历史参照（复用预测，未重训）：P3B `0.076224 / 0.069865`、L `0.078537 / 0.066267`、
  DBT `0.077768 / 0.069401`、A_joint_BF `0.076731 / 0.070067`（valid / test）。

test 的 p95 AE：C `0.209`、S `0.337`；max AE 反而是 S 略小（`2.04` vs `2.50`）。
即 S 不是“少数极端分子爆炸”，而是**误差主体整体上移**（中位/高分段全面变差）。

## 4. 真实曲线与尾段

| epoch | C valid | C online | S valid | S online | S perturbed (θ+ε) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6444 | 1.0420 | 0.8777 | 1.1918 | 1.4823 |
| 50 | 0.1944 | 0.0998 | 0.2342 | 0.1364 | 0.2213 |
| 100 | 0.1214 | 0.0671 | 0.1497 | 0.1036 | 0.1726 |
| 150 | 0.1040 | 0.0569 | 0.1468 | 0.0772 | 0.1355 |
| 200 | 0.0980 | 0.0410 | 0.1310 | 0.0672 | 0.1246 |
| 299 | 0.0896 | 0.0320 | **0.1183** | 0.0492 | 0.1046 |
| 350 | 0.0886 | 0.0241 | 0.1278 | 0.0353 | 0.0893 |
| 400 | 0.0833 | 0.0138 | 0.1246 | 0.0280 | 0.0789 |
| 431 | **0.0796** | 0.0119 | 0.1212 | 0.0245 | 0.0732 |
| 500 | 0.0817 | 0.0043 | 0.1399 | 0.0130 | 0.0636 |
| 600 | 0.0817 | 0.0006 | 0.1405 | 0.0049 | 0.0527 |

- last-100 valid 中位数：C `0.081656`（min `0.081331`）、S `0.140731`（min `0.137045`）；
  两臂的 best epoch（431 / 299）都**不在**尾段 100 epoch 内，best 是曲线中的局部低点。
  best ±20 epoch 窗口均值：C `0.082525`、S `0.127474`；即 S 的 best 相对其平台更“幸运”，
  C 的 best 也略低于平台。这不改变选点规则（仍只用 strict best-valid）。
- S 的 SAM 列（perturbed）只作训练日志：epoch 600 为 `0.0527`，是正常 online 损失的
  ~10.8×，说明 rho=0.05 的绝对半径相对该模型晚期曲率/损失尺度偏大；这**不是**
  “真实 sharpness”，也不作为泛化机制证据。
- C 早期（epoch 1–3）与历史 P3B 几乎逐位相同（online 差 `9.3e-9`，valid 差 `4e-8`），
  之后轨迹分离；C 的尾段中位数比历史 P3B（`0.078135`）高 `+0.0035`，best 高 `+0.0034`。

## 5. 固定四组（仅描述，不参与选点/不新增阈值）

`signature_unseen` × `y>−3` / `y≤−3`，复用既有审计行；row 172 与全部极端标签都在总体数字中。

| split | group | n | P3B | L | DBT | C | S | S−C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | covered_ordinary | 705 | 0.047252 | 0.043744 | 0.048998 | 0.047774 | 0.070071 | +0.022297 |
| test | covered_tail | 51 | 0.128495 | 0.076441 | 0.086733 | 0.116033 | 0.243186 | +0.127153 |
| test | unseen_ordinary | 218 | 0.114921 | 0.114230 | 0.118929 | 0.127107 | 0.150350 | +0.023243 |
| test | unseen_tail | 26 | 0.190233 | 0.254859 | 0.173397 | 0.206961 | 0.222995 | +0.016034 |
| valid | covered_ordinary | 710 | 0.047468 | 0.044864 | 0.042996 | 0.048112 | 0.075266 | +0.027155 |
| valid | covered_tail | 39 | 0.116284 | 0.225481 | 0.258636 | 0.206143 | 0.547175 | +0.341032 |
| valid | unseen_ordinary | 225 | 0.133940 | 0.130773 | 0.141320 | 0.134300 | 0.162268 | +0.027968 |
| valid | unseen_tail | 26 | 0.301922 | 0.325623 | 0.206050 | 0.275846 | 0.270067 | −0.005779 |

S 在 test 的四组全部更差；唯一“更好”的是 valid unseen_tail（n=26，−0.006），不可作为
机制证据。C 与历史 P3B 的组间差异同样在这一轮的单次波动范围内，不重述为结构结论。

## 6. 时间与算力（如实计入两次反传）

| | C | S |
|---|---:|---:|
| 计时阶段（4 个完整 epoch，含全量 valid） | 1.908 s/epoch | 4.099 s/epoch |
| 正式 600 epoch 平均 | 1.86 s | 4.06 s |
| 外层次更新 / 梯度计算 | 47,400 / 47,400 | 47,400 / 94,800 |
| runner wall（含计时、打分、冻结、test） | 1,169 s（19.5 min） | 2,516 s（41.9 min） |
| job wall | 1,176 s | 2,523 s |
| GPU-h | 0.325 | 0.699 |
| 精度 / 设备 | fp32，无 AMP/compile；A100-40GB | 同 |

两卡并行墙钟 **42.3 min**，在预注册的 40–55 min 目标内；两臂都跑满 600 epoch，
没有因前几十 epoch 更差而截断 SAM，也没有缩短 horizon。合计本轮正式训练
`1.024 A100-h`（另有 1 个 1-epoch 冒烟 job，约 20 s）。

## 7. 观察、解释与下一步

**事实层面**：同轮 C 已能正常拟合（train@best 0.010、final train 0.00069）并泛化到
valid 0.0796 / test 0.0727；S 在相同 outer updates 下拟合更慢（train@best 0.041）、
尾段 valid 高 0.059、test 高 0.028，且四个 test 组无一改善。SAM 的两次梯度实现通过
全部核验，因此这不是“第二遍算错/参数漂移”的实现失败。

**解释（限定）**：在 413k 参数、raw-y L1、batch 128、固定绝对 `rho=0.05` 的设定下，
θ+ε 处的损失约为正常损失的 10×，这个上升步把优化目标推得离数据拟合太远；SAM 的
“平坦化”收益没有出现，反而丢掉了正常拟合。该结论**只约束**本 rho/recipe/horizon/
模型规模/单 seed；不排除更小的 rho、按参数范数缩放的 rho、ASAM、更长日程或别的
正则化。也不能从本轮推断“sharp minima 是原瓶颈”或“所有 SAM 无用”。

**同轮波动**：本轮直接测到同配置重跑（C vs 历史 P3B，epoch-1 差 1e-8 级）在尾段
valid 上的漂移约 `0.0035`；因此任何 ~0.001–0.003 的单 seed “改善”都不足以支撑
方向性判断，而本轮 S−C 的差（valid +0.039、test +0.028）比该漂移大约一个数量级。

**下一步**：按预注册停止，**不追加 seed 1、rho 网格或第三臂**。如果这条线还要继续，
最小信息量的新实验是“绝对 rho 与模型/损失尺度匹配”的单臂扫描（例如按 ‖θ‖ 或梯度
范数缩放的 rho，或 ASAM），并需要新的预注册与至少 2–3 个 seed 才能分辨 0.003 量级；
这已超出本轮授权，本轮不启动。

## 8. 复现、provenance 与产物

commit（两臂完全相同）：`c6333a0db10097819d984dfd4f354406886caed7`
（`zinc-p3-sam-v1: pre-register P3-Control vs P3-SAM …`；运行期间未再改动代码）。
host / regime：`res`（a100-2，driver 550.163.01，torch 2.5.1+cu124，CUDA 12.4，
A100-40GB ×2，process backend；C=res-gpu0，S=res-gpu1）。

- init state hash（两臂相同）：`21963b431329532f7b2411845c06e00eece54d0ddbb64774a63719b791716a30`
- SSL：`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
  sha256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`
- config sha256：`afe3244e8098ab2911c9f36a1933291158b2ea45f4f2c078d1a3d49ca571bf8f`
- best checkpoint sha256：C `1cb9cccbe79b3bc31b499d1090fab7e36b446740da381fd80053602d67acc9dd`；
  S `6976c7bdc8ed26b1fb4d468c998c5aa9e68511e082a3d2ccffda97fc913bebaf`
- final checkpoint sha256：C `f9e0c17a8234e637a0ee847d2ee0d5835d2171c3bf043a111902584c8ba909cb`；
  S `9015c4c0c673cd90eaf92a6c5f6c6849a1a7efaf800dfbc5980170a9ec80e132`
- 行顺序 hash（600 个 epoch 两臂相同；seed+101）：首
  `8eca27da1dfa689d7f85578d666bdcf6c882770819df3398648f00506e675e60`，末
  `db5b8e713fed7b188a73ee01ad8d9166582329d1d0760436538051818a864c54`
- test 标签 sha256：`66fd23ecdb1b59c76cee597efc81272e0d2744eeac80744f4159d436e5388768`；
  参考行 `reference_preds_test.csv` 的 row = 0..999，与 official labels 逐值一致，
  P3B/L/DBT 的 MAE 在 test 上重新计算后与历史记录一致（1e-6 内）。

```bash
# 本地：CPU 门控（10 项）与无数据单测
uv run python tracks/structural-sufficiency/zinc_p3_sam_v1/code/sam_checks.py
uv run pytest -q tracks/structural-sufficiency/zinc_p3_sam_v1/code/tests

# 远端：同一 commit 部署后，两臂各一个 job 并行
rr deploy res
rr run res ps-C-control --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_p3_sam_v1/runs/C_control_s0 \
  --result tracks/structural-sufficiency/zinc_p3_sam_v1/analysis/P3_Control \
  -- .venv/bin/python tracks/structural-sufficiency/zinc_p3_sam_v1/code/sam_runner.py run --arm C --device cuda:0
rr run res ps-S-sam --pool res-gpu1 \
  --result tracks/structural-sufficiency/zinc_p3_sam_v1/runs/S_sam_s0 \
  --result tracks/structural-sufficiency/zinc_p3_sam_v1/analysis/P3_SAM \
  -- .venv/bin/python tracks/structural-sufficiency/zinc_p3_sam_v1/code/sam_runner.py run --arm S --device cuda:0
rr pull ps-C-control && rr pull ps-S-sam   # 复制 .rr/pulled/*/result/... 回仓库

# 本地汇总（主表、配对 bootstrap、四组、曲线）
uv run python tracks/structural-sufficiency/zinc_p3_sam_v1/code/sam_report.py
```

`sam_runner.py run` 内含：CPU 门控 → 4 个计时 epoch（单独落盘、随后丢弃）→ 从同一初值
与 RNG 重建 → 600 epoch 正式训练（每 epoch 全量 valid、strict `<`、best/last）→
best/final 在正常权重上的 full-valid/full-train 打分 → `freeze.json` → 唯一一次 test。

产物：

- 每臂 `runs/<arm>_s0/{best,final,last}.pt`（gitignored）、`curve.csv`、
  `order_hashes.json`、`train.json`、`predictions/{valid_best,valid_final,test_best}.npz`；
- 每臂 `analysis/<arm>/{timing,plan,freeze,sam_diagnostics,test_report,status}.json`、
  `curve.csv`、`preds_valid.csv`、`preds_test.csv`；
- 共享 `analysis/{cpu_checks.json,summary.json,main_table.md,groups.md,curves.png}`；
- 代码 `code/sam_{core,checks,runner,report}.py`、`code/tests/test_sam_cpu.py`、
  配置 `configs/p3_sam_v1.json`、预注册 `DESIGN.md`。

## 9. 限制

- 单 seed；同轮 C 与 S 的差是同一硬件、同一初值、同一数据顺序下的**一次**训练对比，
  不是重复实验；本轮的漂移证据（C vs 历史 P3B）只能给出“~0.003”这一个同配置样本，
  不能当作精确的噪声方差。
- SAM 的两遍前向/反传是额外算力：本轮只比较**相同 outer updates**，不是相同 FLOPs 或
  相同墙钟；S 的 GPU 时间是 C 的 2.15×。不能据此说 SAM “同成本无效”。
- 只测了一个固定 rho；负结果不推广到其它 rho、ASAM、缩放规则或更长日程。
- test 已被历史轮次暴露；本轮两臂各读一次是描述性报告，不是新的独立验证。
- 四组 n 很小（tail 26–51），只作描述；未做新的分组审计。
