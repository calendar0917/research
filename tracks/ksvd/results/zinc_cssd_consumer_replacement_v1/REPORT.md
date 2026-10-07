# zinc_cssd_consumer_replacement_v1 研究报告（2026-10-07）

**Protocol**: `zinc-cssd-consumer-replacement-v1`（track ksvd，study zinc-context-gap）
**协议冻结 commit**: 825f30a8bb24（split/预算/判读全部先于任何训练提交）
**正式四 run 执行 commit**: a308e3490bfe（两次已披露的可定位实现修复后；见 §7）
**远端执行**: 全部通过 `rr`/Slurm（res-2；res2-cpu 准备/checks/probe 作业 + res2-cu124 A100 GPU 作业），无 allow-dirty/allow-stale。

---

## 1. 一句话决定

**冻结判读分支 A（keep-dict-candidate）**：在既定的 8001 行 fit 划分、COMP 辅助监督、共享冻结 Q 与本对冻结 CSSD 基底包的范围内，把强 M_COMP 消费者的局部 phi65 输入替换为冻结基底的完整重构，**完整 y 的 dev 预测性能得到承接（两 seed Δ 均为负、平均 −0.00102，远在 +0.003/+0.005 工程容忍内）**，且**稀疏残差有真实、有害可换的读出**（唯一字典干预：|Δpred| p95 3.38 / 1.74 ≫ 噪声标记 1e-4，响应比例 1.000，ΔMAE(y_raw) +1.76 / +0.58）——该有效局部结构接口可由基底供给，DICT 保留为下一阶段候选。bootstrap CI95 [−0.00359, +0.00140] 跨零：**"DICT 更好"的点估计不显著；本轮只声称性能承接，不声称字典带来增量收益、更不声称无损已证明**。下一步研究由用户决定，本轮不自动启动。

## 2. 主表与配对差（dev = 1999 行 = sorted union(旧 select, confirm)，一次性开箱）

| run | y_raw MAE | y_cal MAE | g_raw MAE | ell MAE | s MAE | b_y | fit y_raw | 训练秒 |
|---|---|---|---|---|---|---|---|---|
| RAW_s0 | 0.12248 | 0.12107 | 0.09499 | 0.05182 | 0.08092 | +0.01355 | 0.03962 | 935 |
| DICT_s0 | 0.12152 | 0.12078 | 0.09390 | 0.05364 | 0.08039 | +0.01404 | 0.04083 | 1098 |
| RAW_s1 | 0.11941 | 0.11942 | 0.09219 | 0.05531 | 0.08121 | −0.00098 | 0.03514 | 935 |
| DICT_s1 | 0.11835 | 0.11837 | 0.09124 | 0.05903 | 0.08236 | −0.00257 | 0.03778 | 1098 |

配对差（Δ_s = MAE(DICT_s) − MAE(RAW_s)，正 = DICT 更差）：

| seed | M_DICT | M_RAW | Δ_s |
|---|---|---|---|
| 0 | 0.12152 | 0.12248 | **−0.00097** |
| 1 | 0.11835 | 0.11941 | **−0.00106** |
| 两 seed 平均 | 0.11993 | 0.12095 | **−0.00102** |

- 工程容忍（两 seed 平均 ≤ +0.003 且任一 seed ≤ +0.005）：**满足**（预算规则，非普遍等价阈值）。
- canonical-SMILES 组配对 bootstrap（2000 次，seed 20261012，两臂两 seed **共用**组重采样，每次重采样先平均两 seed 的 MAE 差）：**CI95 [−0.00359, +0.00140]**——跨零，CI 上界 +0.00140 < +0.003（不利方向被统计地约束在预算内；有利方向不显著）。CI 只覆盖分子抽样，不含训练 seed/基底不确定性；2 个 body seed 不是方法稳定性证明。
- 训练成本（res-2 实测，c05 A100）：RAW 935 s / DICT 1098 s 每 run（DICT 的前向内 IHT-10 解码开销 ~17%）；四 run 总 GPU ~68 min（两并行两排队），预算 3 h 内。fit 侧 DICT 略差（0.0408 vs 0.0396；0.0378 vs 0.0351）——fit/dev 方向一致，无"训练过拟合掩盖表示损失"迹象。

## 3. 字典实际参与证据（响应与收益分开判读）

唯一字典干预（每个 DICT soup 的全部局部结构输入：alpha_v := 同一 fit 根均值 alpha，c 保持，重新解码+前向，不重训不重校准；化学 one-hot、J incidence、Sem108、reader、Q 不动）：

| run | ‖alpha−mean‖ L2 med/p95/max | \|Δphi_hat\| med/max | \|Δpred\| mean/p95/max | 响应比例(>1e-4) | ΔMAE(y_raw) | ΔMAE(g) |
|---|---|---|---|---|---|---|
| DICT_s0 | 0.410 / 0.861 / 1.613 | 0.0315 / 1.113 | 1.850 / 3.379 / 4.751 | 1.000 | **+1.758** | +1.769 |
| DICT_s1 | 0.410 / 0.861 / 1.613 | 0.0315 / 1.113 | 0.661 / 1.743 / 3.895 | 1.000 | **+0.580** | +0.582 |

- FP32 噪声上界 eta = 9.5e-7（重复 eval 全 0 + identity hook 全 0，四 run 取 max）；工程标记 max(1e-4, 10·eta)=1e-4；p95 = 3.38/1.74 ≫ 标记 → **响应远超数值噪声**。
- 换均值 alpha 使 y_raw MAE 大幅恶化（+1.76 / +0.58，即 14×/5× 于基线 MAE）→ **训练后的 DICT 消费者真实依赖逐根稀疏码的分子差异**——与 reuse 轮 z-绑定塌缩（响应 0.000）相反，本轮消费骨架（W_loc 直注 fusion 第一层）确实读出稀疏残差。
- 读法：响应=依赖证据，**不是增量收益**；ΔMAE 为正只说明换掉有害，不说明字典优于普通编码（本轮不设该对照主张）。

训练内轻量局部通道诊断（epoch 1/40/120/240，`probes.json`）：DICT 的 ‖W_loc‖ 0 → 3.70 → 5.66 → 7.38，A_raw 梯度自 ep40 起持续非零（0.026/0.027/0.025），解码相对误差中位数稳定在 0.0036（fit 域内）；ep1 的 A 梯度=0 是 W_loc=0 已知零通道现象，非失活。fit 根码：alpha 恰好 8-sparse（nnz=8.0），fit 重构相对误差中位 0.0036 / p95 0.0140。

训练前接口探针（`interface_probe.json`，fit 256 分子按 gid 升序、标签盲）：冻结基底完整重构 rel-err 中位 0.0036/p95 0.0137；**旧 M_COMP soup（该轮自己的 payload/prep）对 phi→phi_hat 有真实响应**（|Δpred| mean 0.042 / p95 0.146 / 响应比例 0.996，Δg MAE +0.015），对均值 alpha 响应 |Δpred| mean 1.13；identity hook 0。仅作接口诊断，不作泛化证据、不否决从零训练。

## 4. 冻结对象复用清单（来源只读，未重拟合任何对象）

| 对象 | 来源（`results/zinc_cssd_nonlinear_binding_v1/`，sha256 于 `source_manifest.json`） |
|---|---|
| fit 划分 | fold.npz / fold_manifest.json（fit 8001 行，实际保存数组核对，非硬编码） |
| dev 划分 | sorted union(select_idx 999, confirm_idx 1000) = 1999 行；SMILES 组不跨 fit/dev（committed 表核验，0 组跨界） |
| targets/常数 | targets.npz（g/ell/s/k/gid，8001 fit 上拟合） |
| phi scaler / kappa_M / tuple payload / prep | tuple_payload.npz、kappa_M.json、prep.npz（prep 指纹重算逐项一致；RAW/DICT 共用，DICT 无独立 scaler/幅度调整） |
| CSSD 基底 | cssd_basis.npz（U/common_rms/D vs cssd_refit.json；radius-2 untyped phi65、q=1、32 atoms、top-8、10-step tied IHT；**其历史 refit 含训练任务监督头——如实记录，非纯无监督基底**；消费者侧完全冻结 buffer，训练前后 sha 不变） |
| 共享 Q | Q_soup_state.pt（vs Q_meta.json；topology25→64→32→1，8001 fit-only，未重训） |
| 旧 M_COMP soup | `results/zinc_local_dictionary_component_supervision_seed0_v1/M_COMP_raw_soup_state.pt`（sha 核对；仅只读探针） |

缺失对象：无（全部在位，未触发补建）。official valid/test 从未加载（所有 manifest `official_valid_loaded/official_test_loaded=false`）。

## 5. 监督 / 划分边界（如实披露）

- **COMP 是明确的辅助监督条件**：L = MAE(ell_hat+s_hat, g) + 0.5·[MAE(ell_hat,ell)+MAE(s_hat,s)]，y_raw = ell_hat+s_hat+Q_raw，y_cal 仅加一次 b_y=median(y_fit−y_raw_fit)。
- **dev 是开发比较**：1999 行为旧 select∪confirm（历史开发已用），不是新的独立 confirm，更非 official test；一次性开箱（terminal_eval.json 存在即拒绝重跑），四 run 完成后统一评分，训练中不看 dev。
- **范围**：本轮只声称"该有效局部结构接口（tuple 特征中的 phi65 块）由基底供给"；化学 one-hot28/28/4、J incidence、Sem108+size2→fusion（W_loc 注入点）、静态关系、后验 MLP bridge、reader、Q 等其余结构输入照旧。不声称 DICT 胜过 RAW（不要求）、不证明字典必须使用、无 y-only/SOTA 主张、CI 跨零不得称"已证明无损"。

## 6. 检查与冒烟（`checks.json` / `smoke.json`，全部通过）

RAW 工厂与历史 M_COMP 逐 tensor/前向**位级一致**；identity hook 不改预测；297,539 可训练参数两臂一致（基底 buffer 不入 state_dict/optimizer）；同 seed RAW/DICT 所有可训练 tensor 逐项相同、seed0/1 的 10 个随机体 tensor 真实不同而 FRAME_SEED A 帧/W_loc=0 跨 seed 一致；CSSD 编解码与 reuse 参考算子一致（max |Δphi_hat| 4.8e-7，alpha 恰 8-sparse，均值 alpha 干预使 phi_hat 大变而 common 项不动 4.3e-7）；batch 合并/顺序不变（≤7.7e-7）、标签打乱不改预测、root codes 与参考算子 0.0、root/global 分子 ID 对应正确；W_loc 首步即有任务梯度、A_raw 第 3 步起非零（两臂）；冻结基底训练前后不变；同 seed 两臂 batch 顺序计划与训练 RNG 哈希一致、确定性解码不消耗训练 RNG（smoke 逐步 RNG 哈希两臂相同）。

## 7. 执行、故障披露与复现

- 阶段（runner `zinc_cssd_consumer_replacement_v1`，`--set model.stage=...`）：source-objects → checks → probe → smoke → train ×4 → terminal-eval。
- **已披露的可定位实现故障（各一次，均已修复并重跑，非科学救场）**：
  1. commit 825f30a 的首批两个 train 作业在**训练完成后的 manifest 阶段**因 `parent.allocation_probe` 向无参 `zldc.allocation_probe` 转发 device 参数而失败（Slurm 56178/56179，训练 951 s 后失败；另两个作业主动取消 56180/56181）；修复 d062957。
  2. 本地端到端 scratch 验证发现 terminal-eval 的 identity-hook 噪声对比对象错误（decode-off vs 解码基线量到的是 decode 效应而非噪声）；修复 a308e34。正式四 run 全部在 a308e34 clean committed revision 上执行。
- res-2 作业（commit a308e3490bfe，除注明外）：

| 作业 | run_id | Slurm | 节点 |
|---|---|---|---|
| source-objects | cssd-cr-v1-source-20261007-123959-7760cd39 | 56174 | c03（825f30a） |
| checks | cssd-cr-v1-checks-20261007-124201-f3eee4a4 | 56175 | c03 |
| probe | cssd-cr-v1-probe-20261007-124221-c253de92 | 56176 | c03 |
| smoke | cssd-cr-v1-smoke-20261007-124435-49f43a23 | 56177 | c05 A100 |
| train RAW_s0 | cssd-cr-v1-train-RAW-s0-20261007-132014-6d9de9b1 | 56184 | c05 A100 |
| train DICT_s0 | cssd-cr-v1-train-DICT-s0-20261007-132027-00d4c995 | 56185 | c05 A100 |
| train RAW_s1 | cssd-cr-v1-train-RAW-s1-20261007-132040-94de933f | 56186 | c05 A100 |
| train DICT_s1 | cssd-cr-v1-train-DICT-s1-20261007-132052-53faa6f5 | 56187 | c05 A100 |
| terminal-eval（一次性） | cssd-cr-v1-terminal-eval-20261007-140044-d2a09323 | 56194 | c05 A100 |

（失败/取消的首次尝试：56178/56179 failed、56180/56181 cancelled、56177 前的 smoke 在 825f30a——smoke 无 bug 未重跑。）

- 复现：`rr deploy res-2 && rr run res-2 <exp> --pool res2-cpu|res2-cu124 ... -- uv run research run zinc_cssd_consumer_replacement_v1 --set model.stage=<stage> [--set model.arm=<RAW|DICT> --set model.seed=<0|1> --set runtime.device=cuda:0]`；本地 `python -m tracks.ksvd.experiments.luyin16.zinc_cssd_consumer_replacement_v1 --stage ...`。
- 产物：`source_manifest.json`、`checks.json`、`interface_probe.json`、`smoke.json`、四 run 的 `init/last/soup/epoch{1,40,120,240}_state.pt`、`curve.json`、`probes.json`、`schedule.npz`（batch 顺序计划显式保存）、`manifest.json`（训练 RNG 哈希、冻结基底/Q 哈希）、`fit_predictions.npz`、`dev_predictions.npz`（含 gid）、`interventions.json`、`terminal_eval.json`（one-shot guard）。本地验证：`research verify` 11 ok / 0 fail；`pytest -m "not slow"` 1851 passed。
- 判读分支机（A/B/C/D）与容忍规则冻结于协议 `protocols/zinc-cssd-consumer-replacement-v1.yaml`（commit 825f30a），先于任何训练。

## 8. 下一步建议（不自动执行）

- 候选方向（供用户决定）：a) 在不动配置的前提下把这对接入下一个消费任务/更大分子的目标域，检验承接的域外性；b) 检验稀疏码的化学可解释读出（alpha 的分子差异 vs 局部结构变化）；c) 用 RAW/DICT 的逐分子配对误差做分组分析定位 DICT 略优/略差的分子类型（ell 分量 DICT 略差、s/g 略好——见主表）。
- 不建议：加 seed3、改容忍阈值、重开 N/E joint、prototype 救场、关系架构搜索（本轮证据未购买任何一项；CI 跨零，两 seed 一致仅 −0.001，不构成"字典更优"证据）。
