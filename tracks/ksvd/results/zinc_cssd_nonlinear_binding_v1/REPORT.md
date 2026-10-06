# zinc_cssd_nonlinear_binding_v1 — 轮次报告（zinc-context-gap / ksvd）

日期：2026-10-06；执行：本地开发 + res-2（rr CLI，pool=res2-cu124，node c05，A100-PCIE-40GB，driver 525.85.12，torch 2.5.1+cu124）。
所有远端作业同一条 committed revision 链（30916cb → d7a6095 → 80b2dbf → 4fe75f97 → f49f6fb8），全部声明 `--result tracks/ksvd/results/zinc_cssd_nonlinear_binding_v1`，official valid/test 从未加载（runner `test_access=blocked`，各 manifest 均记录 `official_valid_loaded=false / official_test_loaded=false`）。

## 1. 研究问题与设计

问题：在冻结的 CSSD 结构基底（U/rms/D，K-SVD 320 epochs + 末 5 轮 D soup，fit 监控 MAE 0.09179，D sha 76b667acc623…）之上，**池化前的节点级（N）与真实键级（E）非线性结构—语义对应**（joint：配对送入同一 MLP；sep：分头独立映射后相加）能否改善完整 y 的样本外预测？

五臂共享：冻结 CSSD 基底（buffer，不可训练）、共享 Q 头（3777 参数，soup sha 14d175e3…）、统一训练配方（LR 1e-3、WD 1e-5、clip 5.0、batch 128、240 epochs、末 5 轮 soup、COMP 损失、fit-median 校准 b_y）：

| 臂 | N | E | 参数量 | 说明 |
|---|---|---|---|---|
| A | 原乘积绑定 | 原乘积绑定 | 436,499 | 恢复原槽接口的强度参照（含 24,048 原槽路径） |
| C00 | sep | sep | 429,783 | joint-vs-sep 2×2 基线 |
| C10 | **joint** | sep | 429,779 | |
| C01 | sep | **joint** | 429,751 | |
| C11 | **joint** | **joint** | 429,747 | |

C↔C 参数差仅 36（joint/sep 之差在聚合方式，不在容量）；五臂共享模块 state 哈希完全一致，C 臂 init/schedule/branch seed 逐臂确实不同。fold：fit 8001 / select 999 / confirm 1000（SMILES 分组，seed 20261006，全部 stage-0 对象重拟合）。

## 2. Stage A：seed 0 训练 + select 评估

训练全部健康（每臂 15120 步，~17–24 分钟/臂 GPU；分支/槽位活跃、梯度可达、无非有限参数、冻结基底不变——见 select_eval.json training_health，排除"失活伪影"竞争解释）。

select（999 分子，主读出 = **y_raw MAE**，"仅校准变好不算正信号"）：

| 臂 | y_raw | y_cal |
|---|---|---|
| A | 0.14987 | 0.14588 |
| **C00** | **0.14405** | **0.14401** |
| C10 | 0.14770 | 0.14773 |
| C01 | 0.15218 | 0.14872 |
| C11 | 0.15313 | 0.15173 |

因子对照（raw，正 = joint 更差）与配对 bootstrap CI95（2000 次重采样，分子重采样，描述性）：

| 对照 | 值 | CI95 |
|---|---|---|
| G_N（N joint 主效应） | −0.00230 | [−0.00616, +0.00147] |
| **G_E（E joint 主效应）** | **−0.00678** | **[−0.01035, −0.00320]（不含 0）** |
| I（交互） | +0.00269 | [−0.00474, +0.01062] |
| A−C00（容量/接口） | +0.00582 | [−0.00036, +0.01177] |

冻结决策规则命中"**只有 C00 改善**"分支：C00 优于 A，但没有任何 joint 臂优于 C00 → 归因为共享局部接口/容量而非对应；按规则 seed 1 只复核 A + C00（预算 5+2 ≤ 10 body runs）。

## 3. 机制干预（训练后的 soup，select，y_raw）

对 N / E 配对做保容量池化前 shuffle（同型组内置换）：

| 臂 | n-shuffle ΔMAE | e-shuffle ΔMAE | 配对实际改变比例 |
|---|---|---|---|
| C00 | −2.9e-9 | +4.6e-9 | 24.0% / 28.1% |
| C10 (N joint) | +2.2e-5 | ~1.3e-9 | 23.9% / 28.3% |
| C01 (E joint) | ~−4.6e-9 | −3.7e-5 | 24.3% / 28.0% |
| C11 | −8.4e-5 | −8.3e-5 | 24.2% / 28.4% |

C00 在**训练后**仍精确不变（sep 池化和 = Σf_S(z)+Σf_C(q)，与配对无关，机制闭环验证）。joint 臂活着、梯度可达，但训练后对配对破坏的响应只有 1e-5–1e-4 量级（C01/C11 打乱后 MAE 甚至略降）——**joint 分支实际编码的配对特异性信息接近零**，与 select 的负因子效应一致。注意：shuffle deltas 度量的是"依赖"，不是"增量收益"。

## 4. Stage B：confirm（一次性，冻结 roster：A+C00 × seeds 0/1）

confirm（1000 分子，y_raw 主读出）：

| run | y_raw | y_cal | g_raw | ell | s |
|---|---|---|---|---|---|
| A s0 | 0.09182 | 0.08987 | 0.08935 | 0.04806 | 0.07740 |
| A s1 | **0.08552** | 0.08221 | 0.08281 | 0.04451 | 0.07564 |
| C00 s0 | **0.08599** | 0.08558 | 0.08361 | 0.04989 | 0.07113 |
| C00 s1 | 0.08951 | 0.08964 | 0.08707 | 0.04462 | 0.07767 |

**种子方向翻转**：seed0 上 C00 优于 A（−0.0058），seed1 上 A 反超（+0.0040）；两 seed 合并 C00 仅领先 0.0009，远小于单 seed 噪声（±0.005）。select 上 C00−A = +0.0058（CI 恰好跨 0）的接口/容量信号**未在 confirm 复现**。2 个独立 seed 是重复性证据，但不是稳定性定理。

## 5. 结论与决定

1. **主问题（N/E joint 对应）：负结果，关闭。** select 上无 joint 臂优于 C00；G_E 可靠为负（CI 不含 0），G_N 方向一致但不确定；机制干预显示 joint 配对贡献近置换不变。池化前非线性 joint 绑定在该结构基底/数据规模下没有可用信号。
2. **次问题（共享局部接口/容量 C00 vs A）：未见可重复信号。** select 单 seed 领先在 confirm 翻转，合并效应 ~0.001 在 seed 噪声内。不升级、不购买更多 seed。
3. 决定：**关闭本候选族（池化前非线性 joint 绑定，N 与 E）**；不再加 seed、不加臂、不调超参。C00（sep）与 A 实质打平——若未来需要容量参照，sep 局部分支接口是安全的（机制上精确打乱不变），但不构成推广理由。
4. 竞争解释的排除：训练健康（分支活跃/梯度可达/无失活/无漂移）排除训练病理；C↔C 参数差 36 排除容量差；C00 精确不变性排除实现泄漏；五臂共享基底/Q/配方排除混杂。剩余解释：该任务在 8k 训练分子规模下，节点/键的字典坐标在池化前的配对语义确实不携带超出单点统计的信息。

## 6. 审计与偏差披露

- CSSD 重拟合保留 CPU（K-SVD→320ep→末 5 轮 soup，忠实历史 train_cssd 语义，无数值移植漂移）；监控用固定 1024 行 fit 子集（manifest 已记录；监控仅信息性，绝不用 holdout 选择）。
- 与历史 train_cssd 的偏离（均已在 objects_manifest 记录）：K-SVD 仅用 fit phi 行；soup 固定末 5 轮；监控仅 fit 行。
- 预期失败并如实记录：select-s0（job 56104，`--set model.seeds=0` 标量 vs 列表 TypeError）、select-s0b（job 56105，`_as_seed_list` 插入位置错误 IndentationError）——均为评估层接线 bug，修复于 4fe75f97；模型训练与结果不受影响。
- 节点重标号不变性曾失败（0.0217）：根因是 `_relabel_nodes` 缺失按节点寻址字段（dict_phi/dict_atom/anchor/patch_cont/env_occ_root/env_bond_root/pair_index）的重映射，修复后五臂 ≤2.4e-7（BLAS 浮点噪声级）。
- 本机未安装 research-judgment skill，按附件原则在本地完成判断，未调用外部审查工具。

## 7. 可复现入口

- 模块：`tracks/ksvd/experiments/luyin16/zinc_cssd_nonlinear_binding_v1.py`；runner：`zinc_cssd_nonlinear_binding_v1`（9 stages，拒绝 test access）；config：`tracks/ksvd/configs/luyin16/zinc_cssd_nonlinear_binding_v1.yaml`；预注册：`tracks/ksvd/notes/zinc_cssd_nonlinear_binding_v1_preregistration.md`。
- 测试：`uv run pytest -q tracks/ksvd/tests/test_zinc_cssd_nonlinear_binding_v1.py`（10 passed）。
- CLI：`uv run research run zinc_cssd_nonlinear_binding_v1 --set model.stage=<checks|smoke|train|select-eval|interventions|confirm-eval> --set runtime.device=cuda:0`。
- 关键 Slurm 作业（res-2）：smoke 56091；trains s0：A 56095 / C00 56096 / C10 56097 / C01 56098 / C11 56099；select（成功）56106；interventions 56107；trains s1：A 56108 / C00 56109；confirm 56122。
- 产物（本目录）：checks.json、smoke.json、cssd_basis.npz、cssd_refit(.curve).json、Q_*、fold/targets/payload/kappa/prep、select_eval.json、interventions.json、confirm_roster.json、confirm_eval.json、runs/<arm>_s<seed>/（init/epoch40/120/240/soup 状态、fit/select/confirm 预测、curve、manifest、probes）。
