# zinc_cssd_basis_reuse_v1 研究报告（2026-10-07）

**Protocol**: `zinc-cssd-basis-reuse-v1`（track ksvd，study zinc-context-gap）
**设计依据 commit**: 97b414bddaf29f3e2f2691ea9edd820fcca43356（`zinc_cssd_nonlinear_binding_v1`）
**协议冻结 commit**: 30c4ac727741（split/预算/判读全部先于任何训练提交）
**远端执行**: 全部在 30c4ac727741 的 clean committed revision 上通过 `rr`/Slurm（res-2，pool res2-cpu 准备作业 + res2-cu124 GPU 作业），无 allow-dirty/allow-stale。

---

## 1. 主结论（严格限定范围）

在本轮这一对 CSSD 基底包（SOURCE = S_fit 小分子域拟合、TARGET = T_fit 目标域同规格重拟合）、这一 n_nodes≤23→24..37 的大小域变化、2 个 body seeds、同一 Full446/M_COMP+Q 消费骨架与同一共享 Q 的范围内：

- **性能层面**：冻结 SOURCE 基底包被目标域消费者直接复用，完整 y_raw MAE 与目标域重拟合对照统计上不可区分（每 seed R = 1.0098 / 0.9796，两 seed 平均 R = 0.9946，SMILES 组配对 bootstrap 比率 CI95 [0.9610, 1.0280]，上界 ≤ 1.05 工程容忍值；覆盖无严重失效，T_eval 完整相对重构误差中位数 0.0049 vs 0.0046）。
- **使用层面（决定性）**：预注册的唯一字典干预（全部 CSSD 消费入口的 alpha_v := 该臂 T_fit 平均根码）对四个训练完成的 soup 的预测变化全部在浮点噪声量级（|Δpred| max ≤ 9e-7，响应比例 0.000，ΔMAE ~ ±2e-9）。只读定位诊断显示这不是干预失效：替换本身是大幅输入变化（每根 ||alpha − mean|| 中位数 0.357），epoch-1 检查点对同一干预有真实响应（|Δpred| max 2.4e-4 / 3.1e-4），而 A-slot 乘积绑定的 z 投影 **W_A_S 在 240 epoch 训练中单调塌缩**（四个 run 全部：||W_A_S|| ≈ 2.29 @ep1 → ~0 @soup；edge 绑定 alpha 行 4.5 → ~1e-4）。
- **冻结判读分支 D**：两臂性能接近但稀疏支路不发挥（有利）作用 → **本轮不能以性能维持支持"稀疏基底可用性"主张**；预测由旁路（化学 one-hot 乘积项、raw/root-tuple MLP、Sem108 接口、共享 Q）承担，这一竞争解释被保留且已被定位。最终建议（三选一冻结口径）：**当前复用未支持 / 停止**（reuse-not-supported-stop）。不升级 N/E、不重启 prototype 类救场（上轮停止决定不变）。

范围限定：以上不是"字典优于普通编码"的证据（本轮无普通编码对照臂），不是普遍近似无损的证明，不是 SOTA 主张；SOURCE 包省下的是目标域基底拟合成本（本轮 1251 s CPU），总训练成本不为零。

## 2. 上轮解释补充与两 Q 样本证据链（不重跑）

- **解释补充**（`notes/zinc_cssd_nonlinear_binding_v1_followup.md`，引用旧 commit 97b414b 与原始预测文件）：G_N/G_E = sep MAE − joint MAE，正值 = joint 更好；上轮负值表示 joint 更差（原报告相反文字已修正，数值未动）。joint 臂只有 select seed0；A/C00 在 confirm 的 seeds0/1 方向翻转 → 关闭的是该候选族预算，不判"结构—语义融合问题被证伪"，C00 也不是已确认的安全替换。干预后 MAE 变化 ≠ 预测变化量：按保存的 base/shuffle 预测核对 mean/median/p95/max |Δpred| 与 ΔMAE 分开报告。梯度可达/参数接近只排除部分解释。
- **Q 定点核对**（`notes/zinc_cssd_nonlinear_binding_v1_q_spotcheck.md` + `q_spotcheck/spotcheck.json`，全部只读复现）：gid=3775（position 3776）与 gid=1424（position 1424）链条逐项一致（subset_index/gid/smi_line/canonical SMILES/y=−20.2606/−20.7828、k=−6、c=−20.7898），无 ID/接线 bug。gid=1424：T25 输入冲突（旧 8001 行拟合池内 position 1270 与其 T25 完全相同但 k=0——25 维拓扑摘要不可分，任何 T25 的确定性函数不可能同时输出 0 与 −20.79）；gid=3775：T25 单例类未覆盖（其类被划入旧 select；原始 L1 最近邻 8325/1270/2232 均为 k≈0/−2 类）。**已由历史审计回答的部分**：长环尾部严重度（zinc_long_cycle_audit Q1/Q5/Q6）与 cycle_basis 顺序依赖（Q9，train:3776 是 19 个顺序敏感例外行之一）。
- **勘误记录**：spotcheck note 初稿 §3 的 T25 特征数值（4.66/6.57、…、feat16=5196.8）未随任何保存产物产生、与冻结 `topology_features`/`T25_cache` 逐位不符；已按可复现输出修正（差异特征为 4/10/13/14——feat10≡feat14 与 feat0≡feat15 是 T25 冻结契约中的逐位重复列，信息性记录，本轮不据此改 Q 输入）。q 重放值（+0.0083/+0.0069）与其余各节数值不受影响。Q 配方保持原样（本轮 T_fit-only 重训一次），未重启 prototype/稀有类加权/单调头/新 cycle 输入。

## 3. 固定合同与拟合范围

| 项 | 冻结值 |
|---|---|
| 宇宙 | 设计基线 8001 行 fit fold（旧 select/confirm 与 official valid/test 从不进入本轮） |
| 域阈值 t | 23（argmin |#S−#T|，平局取小；source_pool 4126 / target_pool 3875，gap 251） |
| 分组 | canonical SMILES（committed 派生表 `train_canonical_smiles.npz`，sha b9ea41d2…；同一分子永不跨组） |
| T_eval | 969 行 = target_pool 组按 sha256("cssd-reuse-v1-20261006\|"+SMILES) 升序前缀，最接近 25%（969/3875 = 25.01%）；一次性开箱 |
| S_fit / T_fit | 各 2906 行整组（N = min(4126, 2906) = 2906，两 refit 分子数相等）；剩余 unused 1220 / 0，**永不作第二评估集** |
| 消费骨架 | 仅 arm A（Full446 / M_COMP+Q，436,499 参数 = 设计基线 arm A，临时 CSSD 任务头确认未进入消费者） |
| 共享对象 | Q(topology25→64→32→1, 3,777 参数, T_fit-only 300 ep, 296–300 五轮平均, soup sha 38840d78…)；targets/payload/kappa/prep 全部 T_fit 重拟合，两臂逐项相同 |
| 训练配方 | 240 ep、batch 128、Adam 1e-3/wd 1e-5、clip 5、锁定 schedule、236–240 五轮 soup、COMP loss g=ell+s、y_raw=ell_hat+s_hat+Q_raw、单次 fit-median b_y |
| 判读 | R ≤ 1.05 工程容忍；SMILES 组配对 bootstrap 2000 次（seed 20261011）；分支 F→C→A→E→D→B |

描述分布（n_nodes）：

| 集合 | n | min | 中位 | max | phi 行数 |
|---|---|---|---|---|---|
| S_fit（SOURCE 基底拟合域） | 2906 | 9 | 20 | 23 | 57,181 |
| T_fit（目标域训练集，两臂消费者共用） | 2906 | 24 | 26 | 37 | 78,084 |
| T_eval（一次性比较集） | 969 | 24 | 26 | 37 | 25,995 |
| unused_source / unused_target | 1220 / 0 | — | — | — | — |

CSSD refit（同规格、CPU、train seed 0）：SOURCE 2906 分子/57,181 phi 行，952 s，D sha 1347329d…；TARGET 2906 分子/78,084 phi 行，1251 s，D sha 09c41ded…（两基底互异）。

## 4. 两臂 × 两 seed 主表（T_eval = 969 行，一次性）

| run | y_raw MAE | y_cal MAE | g_raw MAE | ell MAE | s MAE | b_y | 步数 | 训练秒 |
|---|---|---|---|---|---|---|---|---|
| SOURCE_s0 | 0.13822 | 0.12997 | 0.12945 | 0.08390 | 0.09617 | +0.04521 | 5520 | 457 |
| TARGET_s0 | 0.13687 | 0.13275 | 0.12886 | 0.08253 | 0.10145 | +0.03446 | 5520 | 454 |
| SOURCE_s1 | 0.13655 | 0.13330 | 0.13084 | 0.08092 | 0.10107 | −0.03583 | 5520 | 427 |
| TARGET_s1 | 0.13940 | 0.13864 | 0.13116 | 0.08454 | 0.10045 | +0.01614 | 5520 | 457 |

配对对比（Δ = M_S − M_T，正 = 复用更差；组配对 bootstrap 2000 次）：

| seed | M_S | M_T | Δ | R | Δ CI95 | R CI95 |
|---|---|---|---|---|---|---|
| 0 | 0.13822 | 0.13687 | +0.00135 | 1.0098 | [−0.00466, +0.00715] | [0.9669, 1.0532] |
| 1 | 0.13655 | 0.13940 | −0.00285 | 0.9796 | [−0.01043, +0.00391] | [0.9250, 1.0291] |
| 两 seed 平均 | 0.13738 | 0.13813 | −0.00075 | 0.9946 | — | [0.9610, 1.0280] |

成本（res-2 实测）：2×CSSD refit 952+1251 s CPU、1×Q（T_fit 2906 行）、4×body 427–457 s A100（c05）、terminal-eval 82 s（1 GPU）、checks 9.5 s + smoke 13 s。分子 bootstrap 不含基底/训练 seed 不确定性；2 个 body seeds 不是方法稳定性证明。

## 5. 覆盖与唯一字典干预（预测响应量与 MAE 分开）

覆盖（完整相对重构 phi_hat = U·(common·RMS) + Dbar·alpha，先按分子均各根再宏平均；无成功阈值套在 N_eff 上）：

| 臂 | 集合 | rel-err 中位 | p95 | 未用 atom | N_eff | 有限 |
|---|---|---|---|---|---|---|
| SOURCE | T_fit / T_eval | 0.0050 / 0.0049 | 0.0080 | 1 / 1 | 14.78 / 14.77 | ✓ |
| TARGET | T_fit / T_eval | 0.0047 / 0.0046 | 0.0081 | 0 / 0 | 17.28 / 17.25 | ✓ |

→ SOURCE 基底对目标域 phi65 的**物理覆盖没有失效**（T_fit→T_eval 无恶化，未触发 severe 规则：无非有限、SOURCE T_eval 中位 0.0049 ≪ 1.0）。

唯一字典干预（alpha_v := 该臂 T_fit 平均根码，补丁位于 model.code 单一编码入口——所有消费 alpha 的支路从替换后 alpha 重新生成；c_v、原图、raw/root-MLP、Sem108、Q 输入不动，不重训不重校准）：

| run | ΔMAE(y_raw) | 响应比例 (>1e-4) | \|Δpred\| max | ΔMAE(g, Q 不变) |
|---|---|---|---|---|
| SOURCE_s0 | +2.1e-9 | 0.000 | 8.9e-7 | +2.8e-9 |
| TARGET_s0 | −1.2e-9 | 0.000 | 6.0e-7 | −2.8e-9 |
| SOURCE_s1 | +1.7e-10 | 0.000 | 6.0e-7 | +3.2e-9 |
| TARGET_s1 | +2.3e-9 | 0.000 | 8.3e-7 | +2.0e-9 |

→ **替换是大幅输入变化**（每根 ||alpha − mean|| 中位 0.357、p95 0.775、max 1.389）而预测不变：敏感度不是"输入没变"，而是"消费者不用它"。机制定位（只读诊断 `binding_collapse_diagnostic.json`）：

| run | ‖W_A_S‖ ep1 → ep40 → ep120 → soup | ‖W_E_S alpha 行‖ ep1 → soup |
|---|---|---|
| SOURCE_s0 | 2.288 → 1.1e-2 → 4.1e-4 → 3.3e-11 | 4.54 → 4.7e-4 |
| SOURCE_s1 | 2.293 → 3.6e-3 → 5.8e-7 → 0 | 4.5 → 1.3e-4 |
| TARGET_s0 | 2.276 → 5.5e-3 → 1.3e-5 → 8.1e-19 | 4.53 → 3.6e-4 |
| TARGET_s1 | 2.285 → 3.1e-3 → 3.9e-9 → 0 | 4.53 → 7.2e-5 |

epoch-1（塌缩前）检查点对同一干预有真实响应（|Δpred| max 2.4e-4 / 3.1e-4，mean 4.5e-5 / 7.7e-5）——干预路径接线正确。**A-slot 乘积绑定的整个 z 通道（common c_v 与 alpha 一并）在训练中塌缩到零**，与设计基线的历史证据边界一致（"旧节点乘积绑定不活跃"）；预测由化学 one-hot 乘积项、raw/root-tuple MLP、Sem108 接口与共享 Q 承担。

读法（按冻结口径分开）：敏感不证明增量收益；没有敏感也不能证明信息不存在——本轮的"无响应"已定位为消费者侧 z-绑定塌缩（旁路承担），**不计数为成功复用**，也不据此宣称"稀疏码无信息"。

## 6. 决定与仍存在的竞争解释

**冻结判读分支 D**（顺序 F→C→A→E→D→B 逐项判定；F 否——两臂两 seed 平均 y_raw 0.137/0.138 ≤ 1.0；C 否——非两 seed R>1.05；A 否——sparse_favorable=false（响应 0 < 0.5）；E 否——非两 seed R<1；D 命中）：性能接近但稀疏支路无有利作用。

仍存在且**未被本轮排除**的竞争解释：
1. 该消费骨架的 A-slot z-绑定在 240 epoch 训练中结构性塌缩（本轮四个 run + 设计基线历史观察一致）——因此"冻结 SOURCE 基底可用性"**未被这个消费者检验到**，而非被证伪；
2. raw/root-tuple MLP、化学 one-hot 乘积项、Sem108 接口与共享 Q 的旁路容量足以承担 T_fit→T_eval 的预测，z 通道冗余；
3. 性能 R 的 CI 宽（±3–4%），两 seed 方向翻转（1.0098/0.9796），无法区分"等价"与"小差异"——按冻结规则记不确定，不购 seed 3；
4. 覆盖健康（rel-err ~0.005）只说明基底能重构 phi65，不说明任务读出用到了它。

未来候选（如有）必须由本轮哪个未解释观察购买：目前**没有**证据直接购买新关系架构或 N/E 重开（上轮停止决定不变）；若要继续字典复用线，需要的是"z-绑定不塌缩的消费者"这一未解释观察（W_A_S 塌缩机制本身），本轮不自动提交下一轮训练。

## 7. 复现入口与执行记录

- 协议/runner/config：`protocols/zinc-cssd-basis-reuse-v1.yaml`、`src/ksvd_research/runners/zinc_cssd_basis_reuse_v1.py`（注册于 `runners/__init__.py`）、`configs/luyin16/zinc_cssd_basis_reuse_v1.yaml`
- 模块：`experiments/luyin16/zinc_cssd_basis_reuse_v1.py`（阶段 `--write-smiles-table`（本地一次）→ `--build-objects` → `--cssd-refit --arm SOURCE|TARGET` → `--train-q` → `--checks` → `--smoke` → `--train --arm X --seed {0,1}` → `--terminal-eval`）；远端统一 `uv run research run zinc_cssd_basis_reuse_v1 --set model.stage=<...>`
- 只读附加：`zinc_cssd_basis_reuse_v1_q_spotcheck.py`（输出 `q_spotcheck/spotcheck.json`）、`zinc_cssd_basis_reuse_v1_binding_diag.py`（输出 `binding_collapse_diagnostic.json`）
- 本地验证：`research verify` 11 ok / 0 fail；`pytest -m "not slow"` 1850 passed（跳过已知顺序敏感用例）；checks 全电池（重推导 split 一致、两臂同 seed 初始逐 tensor 相同且仅 {U, common_rms, D} 不同、seed1 真不同（20 键）而冻结 buffer 跨 seed 相同、y=g+c / g=ell+s 精确、两臂 code 化学-重标号/行-分块不变、bond 去重、occurrence 行序不变（0.0）、端点交换 0.0（报告项）、pair_relation 静态契约 0.0、batch/倒序/节点重标号预测不变（≤1.8e-7）、任务头 436,499=arm A 预期）；scratch 端到端（fabricated bases + 短训 + terminal_eval 全链）
- res-2 作业（全部 completed，commit 30c4ac727741，Slurm）：

| 作业 | run_id | Slurm | 节点/资源 |
|---|---|---|---|
| refit-SOURCE | cssd-reuse-v1-refit-SOURCE-20261007-092922-94def19f | 56163 | c03, 8 CPU |
| refit-TARGET | cssd-reuse-v1-refit-TARGET-20261007-092931-b86bcc09 | 56164 | c03, 8 CPU |
| train-q | cssd-reuse-v1-train-q-20261007-092942-a7888688 | 56165 | c03, 8 CPU |
| checks | cssd-reuse-v1-checks-20261007-095635-6267e946 | 56166 | c03, 8 CPU |
| smoke | cssd-reuse-v1-smoke-20261007-095645-a62bf59f | 56167 | c05, A100 |
| train SOURCE_s0 | cssd-reuse-v1-train-SOURCE-s0-20261007-100218-ffa69464 | 56168 | c05, A100 |
| train TARGET_s0 | cssd-reuse-v1-train-TARGET-s0-20261007-100405-0c3d942a | 56169 | c05, A100 |
| train TARGET_s1 | cssd-reuse-v1-train-TARGET-s1-20261007-100417-9af3aed4 | 56170 | c05, A100 |
| train SOURCE_s1 | cssd-reuse-v1-train-SOURCE-s1-20261007-100511-a5022ba9 | 56171 | c05, A100 |
| terminal-eval（一次性） | cssd-reuse-v1-terminal-eval-20261007-102457-99bd8547 | 56172 | c05, A100 |

- 取回：`rr pull`（`.rr/pulled/cssd-reuse-v1-*/result/tracks/ksvd/results/zinc_cssd_basis_reuse_v1/`），已合并入库于 `results/zinc_cssd_basis_reuse_v1/`（fold/基底/Q/四 run soup/terminal_eval.json/coverage.json/checks.json/smoke.json/两个只读诊断）。
- 数值入口：本报告所有表可由 `terminal_eval.json`、`coverage.json`、`binding_collapse_diagnostic.json`、`q_spotcheck/spotcheck.json`、各 run `manifest.json` 逐项复现；official valid/test 从未读取（所有 manifest `official_valid_loaded/official_test_loaded = false`）；T_eval 仅在 terminal-eval 开箱一次（一次性 guard：`terminal_eval.json` 已存在则拒绝重跑）。
