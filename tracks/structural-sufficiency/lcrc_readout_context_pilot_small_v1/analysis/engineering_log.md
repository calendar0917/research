# 工程日志 — lcrc-readout-context-pilot-small-v1

记录本轮**实现层**的修复。规则：只允许修实现（索引、dtype、路径、账目字段、报告模板），
不允许为了让结果好看而改科学定义（容差、阈值、判读规则、配方、划分）。

每条给：发现方式 → 症状 → 修复 → **是否可能影响正式结果**。

正式训练在 `ee72310d` 上执行；下表所有条目都在**正式训练开始之前**发现并修复，
并用 `analysis/checks.json`（13 项 × 2 任务）与 14 个单测重新验证。

| # | 发现方式 | 症状 | 修复 | 可能影响正式结果？ |
|---|---|---|---|---|
| 1 | `import cr_common` | `NameError: CONTEXT_FIELDS`（`cr_models` 已引用） | 在 `cr_common` 定义 `CONTEXT_FIELDS=(log1p_n_atoms, ring_atom_fraction)` 与 `CONTEXT_QUANTILES=(25,75)` | 否（导入失败，无法运行） |
| 2 | `single_variable_functions` 门禁 | `q(0)≠0`：`SingleVariableFunctions` 用了 `nn.init.kaiming_uniform_`，其 `a=√5` 假设不成立于「无输出 bias 的差值形式」 | 改为逐 `Linear` 的 `uniform_(-1/√fan_in, 1/√fan_in)`，并断言 `q(0)` 在 float32 ULP 内为 0 | **是**（若未修，B/C 的 `u_j`、`c_*`、`a_*` 在 `z=0` 处有偏置，账目定义与「零输出→Train 均值」往返都错）。修后被 `shared_init`/`single_variable_functions`/`roundtrip_and_zero_readout` 三项门禁覆盖 |
| 3 | 首次 `prepare` | `_counts` 传 `Long` 被当 `Float` 参与标准化 | 显式 `torch.long` 的计数张量，标准化只作用于浮点列 | 否（prepare 期即报错/审计失败） |
| 4 | 首次 `dataset_worker` | `_ledger_block` 把 `np_` helper 当数组用（`np_` 是 numpy→list 的函数） | 拆成 `parts` dict，块内直接用张量，写盘时才 `np_()` | 否（运行时异常） |
| 5 | 首次账目构建 | `lipo_raw_row` 需要原始 payload，`Bundle` 里只有解析后的 `lipo` | `Bundle` 增加 `lipo_raw`（raw npz payload）；轨迹签名改用 `bundle.lipo_raw["cache_sha256"]` | 否（异常） |
| 6 | 首次 `prepare` | `fingerprints()` 遍历 payload 时把 `round_id` 字符串也当数据集，生成空指纹 | 只对 `DATASETS` 迭代 | 否 |
| 7 | `verify_frozen_files` 调用方 | 返回 `all_match`，调用方读 `pass` → KeyError | 统一在 `audit_dataset` 输出里补 `pass` 键（并保留 `all_match`） | 否 |
| 8 | 首轮 `aggregate` | GINE 参照按 split 局部顺序存储，与官方全局 `row_id` 错位（对齐后 RMSE 明显不对） | 新增 `align_to()`：按 `row_id` 重排，缺失即报错；`check_gine_reference_reuse` 逐样本重算 RMSE/MAE 复核 | **是**（若未修，第 7 节 D 轴与所有 G 对照全错）。修后 G 的 Val/Test 与 `method_G.json` 完全一致 |
| 9 | 首次 checkpoint 重放 | 从 `state_dict` 建模型时缺 MLP 键 → `load_state_dict` 报 missing keys | 重载时先用 `canonical_c1_state(...)` 建结构，再加载选中 `state_dict` | 否（异常；修后重放误差 ~4e-6） |
| 10 | 首次画上下文函数图 | `a_k` 曲线 reshape 用错轴（`(points, M)` vs `(M, points)`） | 显式按字段名 reshape，并从 `Train` 分位裁剪到 Q1–Q99 导出 CSV | 否（图） |
| 11 | `interaction_diagnostics` | 逐 block 的 `atom_batch` 索引用成全局 batch 偏移，`c̄_k` 跨 block 串位 | 用 block 内 `atom_batch` + `path_ptr` 重新划分，并断言三划分分子数与 split 尺寸一致 | **是**（若未修，第 6 节的常数根项/结构偏离项会被错误聚合）。修后 `closure` 与 `root_std_over_rms` 在 Val/Test 上自洽 |
| 12 | `arm A` 等价门禁 | `rho` 输入由 `stack` 变成 `concatenate` 导致维度不符 | 按 C1 原实现的 `cat` 语义拼 `[z_g, mean h, mean h², log1p n]` | 否（形状错误） |
| 13 | `local_input` 账目 | 原计划把每个 root 的冻结输入窗口整块存进账目 → JSON 体积爆炸（数百 MB）且无必要 | 改成 `root_input_digest()`：存窗口的 sha256 定位键 + 宽度，需要的读者可按 (row_id, root index) 从冻结缓存重建 | 否（体积/可读性；账目闭合不受影响） |
| 14 | 并行 worker 与 `checks.json` | 两个 worker 同时写同一 `analysis/checks.json`（read-modify-write 竞态） | 每个 worker 只写自己的 `checks_<dataset>.json`，`aggregate` 合并出 `checks.json`；`aggregate` 以 `all_pass` 作为 `pretrain_ok` 门 | **是**（若未修，并行时会丢一侧门禁记录，`pretrain_ok` 可能被误判为通过） |
| 15 | 预测精度 | `best_predictions.npz` 以 float32 存，重放容差被压缩 | 改存 float64 | 否（只是精度余量；重放本就通过） |
| 16 | 报告模板 | 表格分隔行少一个 `|`；`_reproduction()` 重复定义 | 修正模板与去重；报告改由 `cr_report.py` 从产物生成，禁止手打数字 | 否（文档） |
| 17 | `trace_tolerance` 单测 | 测试用了 `tol(0)=1e-5` 的旧写法 | 按冻结公式 `1e-5 + 1e-6·max(1,|x|)` 断言 `tol(0)=1.1e-5` | 否（测试） |

## 已知且**有意保留**的执行偏离

* **batch 顺序**：本轮按 prompt 冻结用 `np.random.default_rng(0)` 逐 epoch `permutation`；
  `sq-pilot-small-v1` 的历史跑用 `shuffle_offset=917000`。因此 arm A 与冻结 `method_L.json`
  不是逐位相同，而是**同一 best epoch + 1% 量级差异**（见 REPORT §8）。这是设计差异，不是 bug；
  本轮三臂共用同一 batch-order hash，所以三臂之间仍然严格可比。
* **`checkpoint_final.pt` 与逐 epoch 预测矩阵不入库**：体积大且可由同一 commit 重跑复现
  （轨迹签名一致时 trainer 直接跳过）。`best_predictions.npz`、`checkpoint_best_val.pt`、
  全部分析 JSON 与账目均已入库。
* **账目压缩**：分子级/root 级 CSV 以 `.csv.gz` 入库（约 9 MB）；读取时 `cr_common.read_csv`
  透明解压。

## 没有发生的偏离

* 没有为结果好看而调过：容差、`RETENTION_TOL`、`CONTEXT_GAIN_MIN`、`RESOLVE_RMS_MIN`、
  `ROOT_SPECIFICITY_MAX`、bootstrap 的 `RNG=1703`/2000 次、配方、划分、选择规则。
* 没有作废重跑的正式轨迹（`analysis/resource_ledger.json` 的 `wasted_reruns` 为空）。
* 没有删除失败或不理想的数字：molesol 上 B 的 Test 1.3853、C 的损失、G 的领先都照实留在主表。
