# 读出诊断收尾 — 报告（中文）

- round: `lcrc-readout-diagnosis-closeout-public122-s0-v1`  ·  status: **DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP**  ·  source commit: `c9474a2f763e`
- 来源：`LIP_C1_R1_F100` 的 E1 R_SEL（update=8165）；单 seed（0）；官方既有划分；每 fit epoch 与公共 grid 全量评估 Train/Valid/Test。Test 参与开发诊断。
- 本报告在研究者看过上一轮全部结果之后设计，属 **DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP**，不称独立确认。

## 0. 四句回答

1. **来源与缓存一致：best_R_SEL.pt 内部 update=8165，eval 重放 Valid/Test RMSE=0.722616/0.592067，逐分子最大差 0.000001（≤1e-5），h 缓存与 return_aux 交叉核对最大差 0.000004。**
2. **计算路径是通的（首步只动 delta 末层，之后 psi/delta 首层梯度非零），但额外通道的输出贡献很弱：1800 步时 ‖∇psi‖≈1.2e-05（f 约 1.8），完整 delta 输出 RMS≈0.0005（标准化，目标尺度≈1.204），通道响应原始单位 RMS≈0.000015，只占 delta 自身输出≈2.3%。**
3. **没有越过采购门槛：三臂的 R_SEL（按 Valid）全部回到共同 step0（update 0），即读出头训练 1800 步没有给出更好的 Valid 状态；Test 终点仅小幅改善且无法由 Valid 选到，事后按 Test 选的 oracle 虽超 1%，但含 Test 选择乐观偏差、不是可选规则。**
4. **值得保留的方向：暂停给这类读出再加模块（本轮证据是额外通道在本配置下没被用起来，不是“读出容量不够”）；仍未识别：池化位置问题（PRE vs POST 在本配置下退化）与表示是否缺任务信息（H_REP 本轮不检验）。**

## 1. 来源恢复与缓存审计

- 上游把三条探针落在 `u0_snapshot.pt`（update 9720），理由是“E1 的 R_SEL 权重未单独持久化”。但 manifest 记录 `runs/LIP_C1_R1_F100/best_R_SEL.pt`，`retained.update=8165`，且 E1 与 E1+E2 的 R_SEL 是同一状态。
- 本次从该权重恢复成功：内部 update=`8165`，state hash=`c9474a2f763edc5a…`；与 u0_snapshot（`9d3ca96c59668eb0…`）确为不同状态。
- eval 重放：Valid RMSE=0.722616（参考 0.722616），Test RMSE=0.592067（参考 0.592067），误差 0.00000001/0.00000003。
- 逐分子对齐上游 update8165 预测（反标准化一次）：train 0.000001、valid 0.000001、test 0.000001；均 ≤1e-5。status=`OK`。
- 新建 **身份化** clean-h 缓存：identity=`20df666213be6852…`（含 source_state_hash / feature_spec / struct_norm / target_norm / split row hashes），不再复用仅按 dataset 命名的旧 `hcache_Lipophilicity`。
- 缓存 vs `return_aux` 固定批次交叉核对：最大差 0.00000370（阈值 1e-5），通过=是。
- 上游 protocol 的来源替换事实原样保留（旧结果仍以 `source: u0_snapshot (9720)` 标注）；本报告只说明：“目标 E1 权重不可得”对该轨迹与 manifest 不一致，未静默改旧协议或旧预测。

## 2. 三条新探针（source = update 8165）

共同 step0（三臂一致）：Valid RMSE=0.722616，Test RMSE=0.592067，Train_fit RMSE=0.133741。

R_SEL（主规则，grid 含 step0，min Valid MSE）。逐臂同时给 step0 与 R_SEL 的 update 与三划分 RMSE/MAE：

| probe | new_params | 状态 | update | Train_fit RMSE | Valid RMSE | Test RMSE | Valid MAE | Test MAE |
|---|---|---|---|---|---|---|---|---|
| P_WARM | 0 | step0 | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |
| P_WARM | 0 | R_SEL | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |
| P_POST | 14705 | step0 | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |
| P_POST | 14705 | R_SEL | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |
| P_PRE | 14705 | step0 | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |
| P_PRE | 14705 | R_SEL | 0 | 0.1337 | 0.7226 | 0.5921 | 0.4112 | 0.4276 |

### 2.1 相对 P_WARM 与共同 step0 的增益（R_SEL）

正号=更好。相对 step0 的比较在同一起点（三臂共享 step0）上给出，避免把 WARM 自身的训练当成基线漂移。

| probe | R_SEL upd | Δ vs WARM Valid | Δ vs WARM Test | 相对WARM Valid | 相对WARM Test | Δ vs step0 Valid | Δ vs step0 Test |
|---|---|---|---|---|---|---|---|
| P_POST | 0 | 0.000000 | 0.000000 | -0.00% | -0.00% | 0.000000 | 0.000000 |
| P_PRE | 0 | -0.000000 | 0.000000 | +0.00% | -0.00% | -0.000000 | 0.000000 |
| P_WARM | 0 | 0.000000 | 0.000000 | +0.00% | +0.00% | 0.000000 | 0.000000 |

### 2.2 采购门槛（>=1% Valid AND Test RMSE gain vs P_WARM and common step0 (R_SEL)）

- **P_POST**：passes=否；相对增益 WARM(Valid/Test)=-0.00%/-0.00%，step0(Valid/Test)=-0.00%/-0.00%
- **P_PRE**：passes=否；相对增益 WARM(Valid/Test)=+0.00%/-0.00%，step0(Valid/Test)=+0.00%/-0.00%

门槛判定：**未达到**。这是研究投入门槛，不是显著性界限。

### 2.3 paired row bootstrap（2000 次，固定预测）

解释为条件于这些权重/单 seed 的样本不确定度，不称跨训练稳定性。差值 = RMSE(候选) − RMSE(WARM/step0)，负号=候选更好。

| probe | 比较 | 点差 | 均值 | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|---|---|
| P_POST | vs_warm_valid | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.694 |
| P_POST | vs_warm_test | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.805 |
| P_POST | vs_step0_valid | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.694 |
| P_POST | vs_step0_test | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.805 |
| P_PRE | vs_warm_valid | -0.000000 | -0.000000 | -0.000000 | 0.000000 | 0.062 |
| P_PRE | vs_warm_test | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.841 |
| P_PRE | vs_step0_valid | -0.000000 | -0.000000 | -0.000000 | 0.000000 | 0.062 |
| P_PRE | vs_step0_test | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.841 |

### 2.4 事后 oracle（含 Test 选择乐观偏差，不是可选规则）

| probe | Test 曲线最小 | 位置 | 相对 step0 的 oracle 相对增益 | Test 终点 | Valid 曲线最小 |
|---|---|---|---|---|---|
| P_POST | 0.5819 | 180 | +1.72% | 0.5883 | 0.7226 |
| P_PRE | 0.5819 | 180 | +1.72% | 0.5883 | 0.7226 |
| P_WARM | 0.5819 | 180 | +1.71% | 0.5883 | 0.7226 |

读法：训练使 Train_fit 下降、Valid 变差，因此预注册的 R_SEL（按 Valid）必然回到 step0。事后按 Test 选的 oracle 相对 step0 可达约 1.7%（>1% 投入尺），但它含 Test 选择乐观偏差、不是预注册可选规则，不能当作采购依据；本轮的采购判定仍只看 R_SEL（见 §2.2）。Test 终点（update 1800）只有约 0.6% 改善。

## 3. 分支是否学到（直接测量，不只看 PRE/POST 接近）

“末层零初始化”只使前层在首步没有梯度；不能单凭 PRE≈POST 说明分支没训练。以下为 1800 步的实测（详见 `analysis/learning_path.json`）：

| probe | update | ‖∇psi‖ | ‖∇delta_first‖ | ‖∇delta_last‖ | ‖∇f‖ | ‖∇rho‖ |
|---|---|---|---|---|---|---|
| P_POST | 1800 | 0.0000 | 0.0006 | 0.0429 | 1.8432 | 0.1671 |
| P_PRE | 1800 | 0.0000 | 0.0006 | 0.0429 | 1.8433 | 0.1671 |

固定 32 个 Train 分子（仅按公开 row 排序）的通道诊断（1800 步）：

| probe | meanψ(h) vs ψ(mean h) 相对量级 | extra 特征 RMS | 完整 delta 输出 RMS（标准化） | 通道响应相对完整 delta | 通道响应原始单位 RMS |
|---|---|---|---|---|---|
| P_POST | 0.936 | 0.1448 | 0.00054 | 0.0195 | 0.000013 |
| P_PRE | 0.872 | 0.1487 | 0.00054 | 0.0233 | 0.000015 |

目标尺度（target norm scale）≈1.204；完整 delta 输出 RMS 与通道响应 RMS 都远小于它，可直接读出额外通道对预测的实际贡献量级。
（`delta([z_pool, extra]) − delta([z_pool, 0])` 只是内部通道使用诊断，不称合法分子干预或独立贡献。注意 `delta` 末层初值为零，其“相对初态位移”无定义，故只报梯度与输出量级。）

- P_POST 通道响应时间线（update：相对完整 delta / 原始单位 RMS）：1:0.0026/0.000013，2:0.0054/0.000010，10:0.0198/0.000028，100:0.0257/0.000017，900:0.0346/0.000013，1800:0.0195/0.000013
- P_PRE 通道响应时间线（update：相对完整 delta / 原始单位 RMS）：1:0.0044/0.000021，2:0.0055/0.000010，10:0.0214/0.000030，100:0.0303/0.000020，900:0.0321/0.000012，1800:0.0233/0.000015
- PRE 与 POST 在 R_SEL 下的预测差：Valid 最大 0.000000，Test 最大 0.000000。
- 各臂相对 WARM 的最大预测差（全 grid）：POST=0.017909，PRE=0.017941（Test）。

**解释分层：`EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG`**，弱通道臂=['P_POST', 'P_PRE']，响应臂=[]。
判读：PRE 与 POST 接近本身不能证明分支没训练；是否训练由上面的梯度/位移/通道响应测量决定。若通道响应很弱，只写 `EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG`，不写“没有信息”或“池化位置无影响”。

**建议（投入决定）**：暂停现有读出扩展（给该读出再加模块）、长训与选点搜索；不为弥补阴性结果自行启动新实验。本轮只排除 这三个探针 / 该来源 / 该预算，不证明表示没有信息或表达力不足；池化位置与 H_REP 仍未识别。

## 4. 旧（update 9720）与新（update 8165）探针分开展示

两组来源不同，不混成一个总体；新结果不称旧协议的事前结果。

| probe | 旧 source=9720 R_SEL Valid | 旧 R_SEL Test | 新 source=8165 R_SEL Valid | 新 R_SEL Test |
|---|---|---|---|---|
| P_WARM | 0.7254 | 0.5897 | 0.7226 | 0.5921 |
| P_POST | 0.7254 | 0.5897 | 0.7226 | 0.5921 |
| P_PRE | 0.7254 | 0.5897 | 0.7226 | 0.5921 |

旧探针 meta：branch displacement vs WARM max_abs=0.018540，PRE vs POST max_abs=0.000524。
新来源 step0 的 Valid 更好（0.7226 vs 旧 ≈0.7254）但 Test 略差（0.5921 vs 旧 ≈0.5897）：换成“原计划 E1 来源”并不等于得到一个更好的探针基线；本轮所有对比都是**同一来源内**的，不混用旧来源数字。

## 5. 已有上游预测的同口径对照（不重算）

Lipo E1 R_SEL Test RMSE 的结构臂与 D-MPNN，以及 BACE fullN（分类：ROC-AUC）：

| arm | N | R_SEL Test | last10 median |
|---|---|---|---|
| C1_R1@Lipophilicity (F100) | 3360 | 0.5921 | 0.5890 |
| C1_R3@Lipophilicity (F100) | 3360 | 0.5730 | 0.5864 |
| CHEM8_R1@Lipophilicity (F100) | 3360 | 0.5917 | 0.5904 |
| DMPNN_D2F@Lipophilicity (F100) | 3360 | 0.5658 | 0.5678 |
| K_R1@Lipophilicity (F100) | 3360 | 0.5859 | 0.5852 |
| C1_R1@BACE (F100) | 1210 | 0.8176 | 0.8150 |
| DMPNN_D2F@BACE (F100) | 1210 | 0.7960 | 0.7868 |

新探针对 R1 的改善与其是否超过更便宜的 R3 是两个问题；单个额外通道偏好不升级为池化瓶颈或化学机制。

## 6. 结论边界（明确不做的推断）

- 更多 N 的旧结果支持固定配方对样本量敏感；不证明“数据量是唯一主瓶颈”，不证明噪声下限，不推导 PCQM 或大规模预训练的必要性。
- 延长/重启的旧结果只排除这些优化方案的实用收益。
- 训练误差更低而留出更差，写“拟合优势未转化为泛化”，不断言“记忆结构身份”。
- 阴性读出结果不能推出表示没有信息或表达力不足；它足以支持暂缓继续给此类读出加模块的投入判断。
- 结构追溯仍是定位计算响应的工具；没有通过预测收益验证的机制观察不作为性能创新。

## 7. 预算与执行

- 正式科学轨迹：3 条（P_WARM / P_POST / P_PRE，各 1800 updates）。
- GPU·h（各进程占卡墙计）：合计 **0.0127**，软上限 0.1，硬上限 0.25；在软上限内=是。
- 来源审计/缓存：eval-only，device=cuda，wall=3.6s。
- regime：`res`（a100-2，A100-SXM4-40GB ×2，driver 550.163.01，torch 2.5.1+cu124），`res-gpu0`/`res-gpu1` 每 pool 一条进程；逐臂 commit 见各自 train.json。
- launch ledger（fail-closed；每个作业独立 out_dir，无重复投放）：
  - rcppost: state=COMPLETE exit=0 pool=res-gpu1
  - rcppre: state=COMPLETE exit=0 pool=res-gpu0
  - rcpwarm: state=COMPLETE exit=0 pool=res-gpu0
  - rcsrc: state=COMPLETE exit=0 pool=res-gpu0

## 8. 产物与复现

```bash
uv run python code/run.py source-audit --device cpu   # 来源恢复 + 身份化 h 缓存（eval，不训练）
uv run python code/run.py smoke --device cpu           # 4-update 冒烟（不计结果）
uv run python code/run.py probe --rule P_WARM --device cuda --out-dir runs/P_WARM
uv run python code/run.py analyze && uv run python code/run.py report
uv run python -m pytest tests -q                       # 不变量 + 合成 h 夹具
```
- 机器量：`analysis/{source_audit,results,learning_path,comparisons,budget}.json`、`analysis/launch_ledger.json`；曲线图 `analysis/figures/probe_curves.png`。
- 每条轨迹：`train.json`（source/cache/参数/norm hashes、50 点 grid + step0、三规则选中权重）、`curve.csv`、`grid_predictions.npz`（逐样本标准化预测）、`learning_path.json`（分支诊断）、`best_{R,A,W}_SEL.pt`、`last.pt`。
- 权重副本：`runs/` 本地保留；远端 `res` 同路径可恢复。h 缓存按身份命名、确定性可重建。
