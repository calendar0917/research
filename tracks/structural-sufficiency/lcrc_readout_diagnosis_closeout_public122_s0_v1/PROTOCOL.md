# PROTOCOL — 读出诊断收尾（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_readout_diagnosis_closeout_public122_s0_v1/`
Round / protocol ID: `lcrc-readout-diagnosis-closeout-public122-s0-v1`
状态标签：**DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP**（研究者已看过上一轮全部结果之后设计；
本文件在新的正式读出预测之前写出；结果**不称独立确认**）。

上游：`tracks/structural-sufficiency/lcrc_bottleneck_diagnosis_public122_s0_v1/`，交付 commit `7e668a9a`。
上游**只读**；保留用户未提交的 `lcrc_lipo_route_clarification_v1/code/lr_dispatch.py`。本目录适配上游实现，
并记录导入模块与源码 hash（见每次 run 的 `provenance.upstream_hashes`）。

科学常量以本文件 + `configs/closeout.yaml` + `code/rc_common.py` 为准；正式预测开始后三者不再改动。

---

## 0. 本轮要回答的两件事

1. **恢复原计划的冻结表示来源**：上游把三条探针落在 `u0_snapshot.pt`（update 9720），理由写的是
   “E1 作用域的 R_SEL 权重未单独持久化”。但上游 manifest 记录
   `runs/LIP_C1_R1_F100/best_R_SEL.pt`，`retained.update=8165`，且 E1 与 E1+E2 的 R_SEL 都是该状态。
   本轮先验证这条权重确实可恢复、能精确重放，再以它为来源重跑三条探针。
2. **区分两种阴性**：到底是「新增读出没有可兑现收益」，还是「PRE/POST 干预没有形成有效差异」。
   这两者需要不同的证据：前者看留出集是否改善，后者看额外通道是否真的被训练、是否形成可测响应。

本轮不重跑主实验、不搜索新架构、不加主干联合训练、不采购 BACE 验证。阴性结果是完整交付。

## 1. 资源与停止规则

- 目标墙钟 45–60 min，硬上限 90 min；GPU 软上限 0.10、硬上限 0.25 GPU·h（按实际占卡墙钟，含占卡
  准备与失败）。不以用完预算为目标。
- **正式科学轨迹恰好三条**：`P_WARM`、`P_POST`、`P_PRE`，各 1800 updates。无主干重训、额外 seed、
  BACE、学习率搜索或替代初始化。
- `res` 两张 GPU 可用，每卡最多一个本轮进程；任务很短，串行也可。经既有 `rr`，pool 缩窄可见设备后
  进程内用逻辑 `cuda:0`。
- 沿用上一轮已修复的 launch ledger / fail-closed：空查询不表示无任务；没有明确终态不重复投放；
  每个作业独立 `out_dir`，绝不并发写同一目录。不取消其它人的任务。
- 允许实现 smoke 与同规格技术重试；每次如实记成本和原因，不因科学结果不好重试。
- **出现来源错配或无法恢复指定权重时停止正式运行，交付阻塞证据**；不自行用其它状态替代。

## 2. 来源权重恢复（CPU 优先；先验证后训练）

### 2.1 目标

`LIP_C1_R1_F100` 的 **E1 R_SEL 状态，update=8165**（与 E1+E2 R_SEL 同一权重）。

### 2.2 读取与检查（`code/rc_source_audit.py`，`analysis/source_audit.json`）

- 读取 `runs/LIP_C1_R1_F100/best_R_SEL.pt`，检查内部 `update==8165`、`criterion`、`rule`，
  计算**实际 state hash** 与文件 sha256。
- 复用上一轮**精确结构 norm、target norm、feature spec 与 row 映射**，不重新拟合任何 norm
  （直接从上游 `cache/struct_norm_*.npz` / `target_norm_*.npz` 读取并记录 hash）。
- 对恢复的完整模型做**一次 eval-mode 重放**，原始单位参考：
  `Valid RMSE=0.7226158853735515`、`Test RMSE=0.5920667914129428`。
- 逐分子对齐上游 `grid_predictions.npz` 中 `update=8165` 的预测；上游存的是标准化空间，**恰好反标准化一次**。
  最大逐分子差允许 `1e-5`；超出时先定位实现/身份错配，**不通过更换权重或放宽容差解决**。

### 2.3 独立 clean-h 缓存（禁止旧缓存复用）

- 新建**身份化**缓存：身份必须包含 `source_state_hash`、`feature/norm hash`、`split row hash`，
  而**不是**仅按 dataset 命名的 `hcache_{dataset}`。
- 逻辑理由：旧 `hcache_path(dataset)` 只要存在就复用，新头权重会静默搭配旧 update9720 的 `h`。
- 固定若干随机批次，将缓存 `h`、`mean_h`、`mean_h2`、`g` 与该完整模型 `return_aux` 直接交叉核对
  （阈值 `1e-5`）。
- 任一步失败 → `SOURCE_BLOCKED`，正式探针拒绝启动。

### 2.4 来源替换事实

上游 protocol 的来源替换（附录 A1）作为历史事实保留；本报告解释其“目标 E1 权重不可得”的理由
与这条轨迹及 manifest 不一致。**不静默改旧协议或旧预测**；旧结果以 `source: u0_snapshot (update 9720)`
原样展示。

## 3. 三条探针规格（科学规格不变）

- 共同冻结来源：第 2 步 verified update=8165。冻结 `atom_encoder / bond_encoder / local`；不更新 encoder，
  不启用 dropout/增广。
- 原 `f/rho` 从该 checkpoint warm start 且可训练。
- `z_pool = [g42, mean_h64, mean_h2_64, log1p(n)]`，171 维。
- `P_WARM`：`p_std = Σ_v f(h_v) + rho(z_pool)`。
- `P_POST`：`p_std = P_WARM + delta([z_pool, psi(mean_v h_v)])`。
- `P_PRE` ：`p_std = P_WARM + delta([z_pool, mean_v psi(h_v)])`。
- `psi = 64→32→16`；`delta = 187→64→1`；隐藏激活 `SiLU`；POST/PRE 新参数均为 **14705**。
- `psi` 与 `delta` 第一层保持原随机初始化，private seed `1101`；仅 `delta` 最后一层 weight/bias 全零。
- POST/PRE 共享初始 `psi/delta`；三臂共享 `f/rho` 初始状态与 minibatch 序列。
- 优化：AdamW，lr `3e-4`、weight_decay `1e-3`、batch `128`、`1800` updates、无 scheduler、seed `0`；与上游一致。
- 只用**全部官方 Train** 的标签训练；官方 Valid/Test 原样完整评估。
- 每个 fit epoch 对 `train_fit/Valid/Test` 完整评估；同时保存 `step0` 与 `round(j*1800/50), j=1..50` 的公共 grid。
- `R_SEL` 主结果：grid **含 step0**，取 Valid MSE 最小、同分取较早。`A_SEL/W_SEL` 沿用上游定义，
  **仅作敏感性表**，不据结果改主规则。
- 保存三规则选中权重、最终权重、逐 grid 预测、每 epoch 曲线、source/cache/parameter/norm hashes。
  本地与 `res` 各有可恢复副本。

## 4. 分支是否学到：必须测量（Part 3）

“末层零初始化”通常只使前层在首步没有梯度，**单凭这一点不能解释 1800 步后的弱响应**。
本节**不**为了让对照分开而改 lr、初始化或加损失；测量只用于判断原干预强度。

### 4.1 实现前微型合成 h 夹具（`tests/test_invariants.py`）

- 三臂 step0 预测一致；POST/PRE 参数与初态一致。
- 对非退化 `h`，`mean psi(h)` 与 `psi(mean h)` 可以不同；无 flatten/广播错误。
- 最后一层零时首步只更新该层；完成一次有效更新后，梯度可进入 `delta` 第一层及 `psi`。
- 夹具只验证计算路径，**不当科研结果**。

### 4.2 正式训练记录（updates 1 / 2 / 10 / 100 / 900 / 1800）

- `f`、`rho`、`delta` 最后层、`delta` 第一层、`psi` 的**实际梯度范数（clip 之前）**、相对初态参数位移；
  注明实际优化的目标尺度（标准化 z 空间 MSE）。
- 固定 **32 个 Train 分子**（仅按公开 row 排序选，不用标签筛选）的 `mean psi(h)` 与 `psi(mean h)`
  差值分布，并报相对额外特征尺度的量级。
- 同一固定批次上，完整 `delta` 输出，以及 `delta([z_pool, extra]) - delta([z_pool, 0])` 的原始单位响应；
  后者只是**内部通道使用诊断**，不称合法分子干预或独立贡献。
- 在公共 grid 上 `PRE/POST` 各自相对 `WARM` 的预测差及 Train/Valid/Test 误差轨迹。

诊断 eval 保存/恢复模型模式与 RNG；不额外改变正式优化步数或梯度。

### 4.3 解释分层

- **A. IMPLEMENTATION_INVALID**：代码路径不通 / 源缓存错配。修复身份或实现后才可开始正式科学轨迹。
- **B. EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG**：路径正常，但额外特征通道响应很弱。
  不写“没有信息”或“池化位置无影响”。
- **C. NO_READOUT_GAIN_THIS_CONFIG**：通道形成可测响应但留出没有改善。仍只排除这三个探针、该来源和预算。
- **D.**：PRE 与 POST 接近本身不能证明分支没训练；也可能是当前 `h`、非线性区域或额外通道利用造成。
  没有直接测量时不指定原因。

## 5. 结果表与判读

1. 第一张表逐臂列 `step0` 与 `R_SEL` 的 update、Train/Valid/Test RMSE/MAE；直接比较新增探针对 WARM、
   以及对共同 step0 的增益。
2. 旧 update9720 探针与新 update8165 探针**分别展示**，不混成一个总体，也不把新结果称旧协议的事前结果。
3. **主采购门槛**：某新增探针在 `R_SEL` 下相对 `P_WARM` **和**共同 `step0`，Valid/Test RMSE 均改善 ≥1%。
   这是研究投入门槛，不是显著性界限。用固定预测的 paired row bootstrap 2000 次给差值区间；
   解释为条件于这些权重/单 seed 的样本不确定度，不称跨训练稳定性。未过门槛时同时报数值和区间，
   **不写“小于 0.01 所以必为噪声”**。
4. 若 PRE 优于 POST，须同时报：相对 WARM 与原模型是否实际改善、分支响应是否形成、区间与个例影响。
   单个额外通道偏好**不升级**为池化瓶颈或化学机制。
5. 仅用已有上游预测，附一张同轮同口径对照表：Lipo E1 R_SEL 的 R1/R3/CHEM8/K/D-MPNN，及 BACE fullN
   C1/D-MPNN。新探针对 R1 的改善与其是否超过更便宜的 R3 是**两个问题**。
6. 不在本轮自动采购 BACE 验证、主干联合训练、支持度正则、更多结构或任何其它后续实验。

## 6. 交付与收束

交付：`PROTOCOL.md`、`configs/closeout.yaml`、`code/` + `tests/`、
`analysis/{source_audit,learning_path,results,comparisons,budget}.json` 与曲线图、`REPORT.md`（中文）。

- 报告开头回答四句：① 来源与缓存是否一致？② 新增通道是否真正形成响应？③ 是否获得超过 WARM 和原模型的收益？
  ④ 因此值得保留哪个方向，哪些仍未识别？
- 结论边界固定：
  - 更多 N 的旧结果支持固定配方对样本量敏感；不证明“数据量是唯一主瓶颈”，不证明噪声下限，不推导
    PCQM 或大规模预训练的必要性。
  - 延长/重启的旧结果只排除这些优化方案的实用收益。
  - 训练误差更低而留出更差，应写“拟合优势未转化为泛化”，不能据此断言“记忆结构身份”。
  - 阴性读出结果**不能**推出表示没有信息或表达力不足；它足以支持暂缓继续给此类读出加模块的投入判断。
  - 结构追溯仍是定位计算响应的工具；没有通过预测收益验证的机制观察不作为性能创新。
- 若达到采购门槛：交付候选及证据，停止，等待研究者设计下一步。
- 若未达到：明确建议暂停现有读出扩展、长训与选点搜索；不为弥补阴性结果自行启动新实验。
- 若仍是弱通道响应：如实保留池化问题未解决，同时停止本轮；不无限修探针直到阳性。
- 完成后提交新目录（不创建 PR），用实际成功/失败状态汇报。禁止将基础设施成功或门禁全绿写成科学假设成立。

## 7. 执行 regime 与产物

全部正式臂经 `rr` 在 `res`（a100-2，A100-SXM4-40GB ×2，driver 550.163.01，torch 2.5.1+cu124）
以 `res-gpu0`/`res-gpu1` 两个 pool、每 pool 一条进程执行；`cuda:0` 进程内可见。逐臂 commit 见各自 `train.json`。
来源审计与缓存构建可先在本地 CPU 完成（eval 重放，非训练）。
