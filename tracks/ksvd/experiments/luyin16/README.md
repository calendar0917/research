# luyin16 实验工作区

这里是 luyin16 阶段唯一新增实验入口。当前第一个协议是导师思路的独立概念复现：明确构造 `S_v1`、匹配初始化的 `T_init_v1/T_final_v1`，再用固定参数 XGBoost 检验结构表征的独立价值。

它不声称复现导师未知的 69/624 维特征，也不默认查看 official test。

## radius-2 原子中心候选

`mentor_concept_replication` 支持 `sampler.mode: radius2_atom`：每个原子取
距离不超过 2 的诱导 ego 子图，使用中心优先的确定性 BFS 生成 52D
`28D adjacency + 16D atom histogram + 8D bond histogram`。对它逐坐标计算
12 组固定分布统计，得到与导师脚本形状一致的 624D proxy。将
`max_patches_per_graph` 设为 `null` 会保留该图的全部中心 patch；这是图级
局部结构分布的正式候选，不能与只保留 8 个中心的 smoke 数字混用。

当前结果见
[`RADIUS2_ATOM_CHEM_20260830.md`](../../results/luyin16/RADIUS2_ATOM_CHEM_20260830.md)。
冻结特征后先做 official-train-only scaffold screen，再运行 12-trial
Optuna；不要在同一 validation 上继续盲扫 K/T。

只有 view 和 Optuna 参数冻结后，才运行 test 检查：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.evaluate_frozen_test \
  --full-features tracks/ksvd/results/luyin16/mentor_r2_atom_chem_k64_s8_all_official_valid/features_and_predictions.npz \
  --search-result tracks/ksvd/results/luyin16/mentor_r2_atom_chem_k64_s8_all_official_valid/xgb_optuna_promoted_12.json \
  --result tracks/ksvd/results/luyin16/mentor_r2_atom_chem_k64_s8_all_official_valid/frozen_test_evaluation.json
```

该命令同时输出只用 official train 的 test holdout，以及冻结后用
train+valid 重训的 test 结果；test 不参与 view、超参数或停止条件选择。

## 开始方式

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run \
  --config tracks/ksvd/configs/luyin16/mentor_concept_v1_smoke.yaml \
  --dry-run
```

小规模 smoke：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run \
  --config tracks/ksvd/configs/luyin16/mentor_concept_v1_smoke.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_concept_v1_smoke
```

烟测通过后，先运行中等规模 development 配置；它拥有更多 validation 正样本，同时控制旧版 Python K-SVD/OMP 的计算成本：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run \
  --config tracks/ksvd/configs/luyin16/mentor_concept_v1_dev.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_concept_v1_dev
```

development 结果只能用于检查信号、运行成本和冻结候选设置，不能替代完整 official split 实验。

若 development 管线和成本正常，冻结同一 K32 设置后进行完整 official validation 复核：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run \
  --config tracks/ksvd/configs/luyin16/mentor_concept_v1_k32_official_valid.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_concept_v1_k32_official_valid
```

该运行使用完整 official train/valid，但仍不预测 test；不得根据结果回头修改同名配置。

完整固定特征生成后，运行预定义的拓扑/化学组成/K-SVD 特征块归因：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.fixed_feature_block_ablation
```

该归因直接读取冻结的 NPZ，不重新学习特征，只使用 official validation。

导师的未知 `recon_typed[624]` 只能做概念代理。当前 `R_v1` 对 52 维 typed patch 或其重建逐维计算 12 组固定分布统计，因此得到 `12 × 52 = 624` 维；维度相同不代表实现相同：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run \
  --config tracks/ksvd/configs/luyin16/mentor_typed_reconstruction_proxy_v1.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_typed_reconstruction_proxy_v1
```

主要对照是 `R_final−R_init`、`R_final−R_raw` 和 `S+R_final−S`，仍不查看 test。

若把 `624` 理解为一个保留槽位对齐的 typed object，而非 12 组全局统计，运行：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_typed_slot_proxy_v2.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_typed_slot_proxy_v2
```

该 proxy 使用 `8×64` atom-slot one-hot 与 `28×4` bond-slot one-hot，维度自然为 624；仍不代表导师真实 schema。

如果导师的 `recon_typed` 保留 patch occurrence mass，可用 development 配置测试 sum readout：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_typed_slot_proxy_v2_sum_dev.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_typed_slot_proxy_v2_sum_dev
```

若要验证“624 维对象是否被 K32 容量限制”，使用 K64/T8 development：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_typed_slot_proxy_v2_k64_dev.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_typed_slot_proxy_v2_k64_dev
```

K64/T8 development 只有 16 个 validation 正样本；若需要容量的最终判断，运行完整 official-valid 配置：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_typed_slot_proxy_v2_k64_official_valid.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_typed_slot_proxy_v2_k64_official_valid
```

完整开发阶段仍只评估 train/valid。只有配置和 view 已根据 valid 冻结后，才显式增加 `--evaluate-test`。

导师的 AUC 脚本还明确给出了比 `S+R` 更完整的输入结构：
`composition[69] + recon_typed[624] + context mass[5K+5]`，默认 payload
目录名为 `K64_s8`。对应的 shape-aligned ring-context proxy 使用同样的
`s/t/a/st/sa/ta/sta` 消融别名：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.mentor_ring_context_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_ring_context_proxy_k64_s8_smoke.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_ring_context_proxy_k64_s8_smoke
```

烟测通过后再换用 `mentor_ring_context_proxy_k64_s8_official_valid.yaml`。
该实验的 69D composition、624D typed reconstruction 和五类 ring context
仍是本仓 proxy；它复现的是导师脚本的块结构和归因问题，不冒充未知上游实现。

如果要专门检查 T 的信息损失，可运行 K64/T8 的 T-readout 消融：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.mentor_ring_context_proxy \
  --config tracks/ksvd/configs/luyin16/mentor_t_readout_ablation_k64_s8_official_valid.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_t_readout_ablation_k64_s8_official_valid
```

该配置同时导出 `mean(Y)`、`mean(DX)`、`mean(D|X|)`、`mean(X)`、
`mean(|X|)`、rich sparse-code summaries 和 `mean(|Y-DX|)`，并生成各自的
`S+T` 视图。所有分类器选择仍只使用 official-train 内部数据，官方验证集
只在最终冻结后评估。

探索期不再为每个候选完整训练。优先使用 official-train-only scaffold
快筛，三次 trial 后自动标记是否晋级：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.tune_xgb_fused_proxy \
  --features <frozen_train_valid.npz> \
  --result <screen.json> \
  --trials 3 \
  --folds-file tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --screen-only \
  --baseline-view s \
  --promotion-delta=-0.005 \
  --minimum-fold-wins 2 \
  --views s,<candidate-fusions>
```

只有达到 delta 门槛且至少赢 2/3 folds 的融合视图才进入 12-trial
Optuna 和 official-valid 多 seed。当前 T-readout 判定与完整数字见
`../../results/luyin16/T_READOUT_ABLATION_20260830.md`。

协议锁定后，在本目录增加一个有明确职责的 runner，并同时维护：

- `configs/luyin16/`：数据集、split、seed、特征、字典训练范围、`protocol_id`；
- `results/luyin16/`：原始输出、manifest 和小型摘要；
- `notes/luyin16_plan.md`：问题、停止条件和阶段结论。

runner 从 `ksvd_research` 导入可复用逻辑，例如：

```python
from ksvd_research.core import ksvd
from ksvd_research.data import load_molhiv
from ksvd_research.evaluation import GraphLevelConfig
```

`tracks/ksvd/code/` 是历史复现区，不在其中继续平铺新的 luyin16 脚本；旧入口仍保留以便复现既有数字。

## Role × attribute binding（不依赖 K-SVD）

为单独检验“结构角色和属性的对应关系”而不是字典更新，运行：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.role_attribute_binding_screen \
  --config tracks/ksvd/configs/luyin16/role_attribute_binding_screen.yaml
```

该协议把完整 radius-2 ego 中的 shell/诱导度数/局部环角色与 compact OGB
atom/bond semantics 分开编码，比较 marginal、未中心化 joint、centered
binding 和图内属性 shuffle。默认只用 official-train scaffold folds；冻结后的
official-valid 复核使用：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.role_attribute_binding_screen \
  --config tracks/ksvd/configs/luyin16/role_attribute_binding_screen_official_valid.yaml
```

它不学习字典、不扫描 K/T，也不编码 official test。结果和机制 gate 见
`../../results/luyin16/role_attribute_binding_screen*/`。

## Clean structural-role fusion

为排除 coarse role 与 OGB degree/ring 的重复，并测试更完整的结构坐标，运行：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.structural_role_fusion_screen \
  --config tracks/ksvd/configs/luyin16/structural_role_fusion_screen.yaml
```

该协议固定 full all-center radius-2 substrate，将 compact atom attributes 中的
`degree`/`is_in_ring` 去掉，并比较 coarse role 与 topology-only rooted-WL role；
融合视图为 `T`、`A`、`T+A`、factorized raw joint、centered binding 和 matched
within-patch shuffle。exact rooted topology 只用于 label-free coverage/collision audit。

冻结 official-valid 使用：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.structural_role_fusion_screen \
  --config tracks/ksvd/configs/luyin16/structural_role_fusion_screen_official_valid.yaml
```

结果表明 rooted-WL 在内部三折稳定优于 coarse，但 official-valid 上 joint/binding
未稳定超过 `T+A` marginals；高阶 typed-edge 为负。阶段判定见
`../../results/luyin16/CLEAN_STRUCTURAL_ROLE_FUSION_20260901.md`。

## Task-aligned interaction 与超参数审计

为排除 fixed XGBoost 对高维 interaction 不公平，并测试 cross-fitted sparse
interaction correction，运行：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen \
  --config tracks/ksvd/configs/luyin16/task_aligned_interaction_screen.yaml
```

该协议只使用 official-train：三组 scaffold 作 outer holdout，outer train 内剩余两组
作双向 inner search；`T+A`、factorized raw 和 centered 获得相同 Optuna 预算。
结果表明调参主要提高 `T+A`，raw/centered 相对 `T+A` 的差距分别缩小 `3.03pt` 和
`1.28pt`；稀疏 residual 也未超过 tuned `T+A`。判定见
`../../results/luyin16/TASK_ALIGNED_INTERACTION_TUNING_20260901.md`。

## 不变谱交互与条件 binding 摘要

为直接检查 PCA 坐标漂移是否是瓶颈，运行：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.invariant_conditional_interaction_screen \
  --config tracks/ksvd/configs/luyin16/invariant_conditional_interaction_screen.yaml
```

该协议从已缓存的 centre-level covariance/binding 对象构造不依赖 PCA 方向的奇异值、
能量、熵和 role-conditioned concentration 摘要；只在 official-train scaffold folds
内做 12-trial XGBoost 搜索，official-valid/test 不编码。当前结果见
`../../results/luyin16/INVARIANT_CONDITIONAL_INTERACTION_20260902.md`：
`S+marginal` 仍胜过三个新摘要，因此新摘要保留为解释性/稳定性候选，不晋级性能主线。

## Rooted-WL 冻结终端测试

最终候选先在非 test 数据上冻结：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.structural_role_terminal_audit \
  --config tracks/ksvd/configs/luyin16/structural_role_terminal_audit.yaml \
  --stage tune
```

确认 `frozen_manifest.json` 后，才运行一次 terminal test：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.structural_role_terminal_audit \
  --config tracks/ksvd/configs/luyin16/structural_role_terminal_audit.yaml \
  --stage test
```

入口会校验 config/implementation/encoder 哈希，并在 `terminal_test.json` 已存在时
拒绝重复评估。本轮 train+valid refit 的 clean `T+A/centered` 五 seed mean 为
`0.7610/0.7716`，预注册固定 50/50 late-fusion paired mean 为 `0.7825`，十模型
ensemble 为 `0.7839`。完整披露与历史对照见
`../../results/luyin16/STRUCTURAL_ROLE_TERMINAL_TEST_20260901.md`。

## 中心级结构–属性交互冻结评估

中心级路线先用 official-train scaffold folds 搜索五个预注册视图的 XGBoost，
再在 official-valid 做一次冻结报告：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal \
  --config tracks/ksvd/configs/luyin16/cross_center_interaction_terminal.yaml
```

在用户明确授权后，使用独立的 test 入口进行一次 controlled terminal evaluation；
它会校验 valid 搜索结果、代码和交互编码器 hash，拒绝重复写入：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal_test \
  --config tracks/ksvd/configs/luyin16/cross_center_interaction_terminal_test.yaml
```

该入口同时报告严格 train-only 和 train+valid refit；另有固定 train-only PCA、
仅增加 train+valid 标签的 scope control：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.cross_center_interaction_test_pca_control \
  --config tracks/ksvd/configs/luyin16/cross_center_interaction_terminal_test.yaml
```

五个视图的冻结 valid/test 数字见
`../../results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md`、
`../../results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md` 和
`../../results/luyin16/CROSS_CENTER_INTERACTION_TEST_PCA_CONTROL_20260902.md`。

PCA rank=16 的 train-only 快速敏感性检查（复用同一 5121 图、同一 raw cache，
不编码 official-valid/test）使用：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.cross_center_interaction_screen \
  --config tracks/ksvd/configs/luyin16/cross_center_interaction_screen_pca16.yaml
```

该检查只把交互块从 PCA-8 改为 PCA-16；结果写入
`../../results/luyin16/cross_center_interaction_screen_pca16/`。它是 development
sensitivity，不得替代已冻结的 official-valid/test 结果。

随后完成的 PCA-16 `S+cross_cov` 完整调参及冻结 valid/test 整理见
[`CROSS_CENTER_INTERACTION_PCA16_CROSSCOV_SUMMARY_20260903.md`](../../results/luyin16/CROSS_CENTER_INTERACTION_PCA16_CROSSCOV_SUMMARY_20260903.md)。
该结果已存在，不需要重复运行；其中 PCA-16 `cross_cov` 的完整调参使用
official-train scaffold CV，test 只作冻结后的 controlled check。

## 中心级条件融合与关系传播（ZINC）

为检验“图级 readout 过早丢掉中心对应关系”这一假设，新增了一个不依赖 GPS 的
中心级原型：每个原子保留 topology-only rooted-WL 结构表示和同中心化学表示，先做
`[s_v, a_v, s_v*a_v]` 条件融合，再可选地沿显式中心对（最短路、patch overlap、相邻
键类型）传播。模型只使用 sum/mean/std readout，不使用全局 attention。

烟测：

```bash
./.venv/bin/python -m tracks.ksvd.experiments.luyin16.center_relation_network \
  --config tracks/ksvd/configs/luyin16/center_relation_network_smoke.yaml
```

完整官方切分的单 seed、CPU、60 epoch fast 验证：

```bash
./.venv/bin/python -m tracks.ksvd.experiments.luyin16.center_relation_network \
  --config tracks/ksvd/configs/luyin16/center_relation_network_official_fast.yaml
```

结果见 `../../results/luyin16/ZINC_CENTER_RELATION_NETWORK_OFFICIAL_FAST_20260903.md`。
`center_concat` 的 test MAE 为 `0.2722`，同中心乘性交互的
`conditional_fusion` 为 `0.2100`，均明显优于当前 `S+WL-count` 的 `0.3765`；但
`conditional_relation` 的 valid/test 为 `0.2193/0.2382`，相对 fusion 尚未稳定，
因此当前证据支持“中心对齐 + 条件融合”是突破点，不支持继续堆关系传播层。结果是
单 seed fast 诊断，不替代多 seed/调参后的终端比较。

## ZINC typed match 候选

为直接检验“typed match + 统计”是否能改善当前 XGBoost 路线，新增
`zinc_typed_match.py`。它从已有 token cache 取每个中心的 typed-WL
`0/1/2/3` 四层签名，在训练部分按频次建立完整多层 prototype bank；匹配时只允许
同一中心的前缀逐层一致，再对每个分子做 prototype response 的 mean/max 与分布统计。
最终仍只有一个图级 XGBoost，不使用 K-SVD。该实现是可审计候选，不声称等同于导师未提供
的内部 typed-match schema。

```bash
./.venv/bin/python -m tracks.ksvd.experiments.luyin16.zinc_typed_match \
  --config tracks/ksvd/configs/luyin16/zinc_typed_match.yaml
```

单 seed Optuna 结果见
[`ZINC_TYPED_MATCH_20260904.md`](../../results/luyin16/ZINC_TYPED_MATCH_20260904.md)：
official valid/test 为 `0.4100/0.3951`，优于 `S+marginal` 的约
`0.5444/0.5519`，但仍弱于 hierarchical typed-WL backoff 的
`0.3546/0.3455`。这说明“同中心 typed match”确实比独立 marginal 更有用，但当前
prototype 选择和 response readout 还没有超过显式 exact/backoff count。

## ZINC 长程 suite

ZINC 长程、统计对象、K-SVD INIT/FINAL 和属性—结构 factorial 已整合为一个
可断点续跑入口：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.run_suite \
  --config tracks/ksvd/configs/luyin16/suite_long.yaml \
  --suite-dir tracks/ksvd/results/luyin16/suite_zinc_long_range_factorial_20260830
```

重复执行同一命令会跳过 fingerprint 匹配的成功任务。每个 task 的
`status.json/stdout.log/stderr.log` 位于 suite 的 `tasks/<task_id>/`，最终摘要为
`SUITE_SUMMARY.md`。数据只使用 PyG ZINC 官方 URL；preflight 会验证 split size
并记录 raw SHA-256。冻结结果和研究判定见
`../../results/luyin16/ZINC_LONG_RANGE_FACTORIAL_20260830.md`。

## ZINC 机制快筛

`mechanism_screen.py` 在不加载 test 的前提下，对 frozen radius patch 同时比较
typed marginal、topology--attribute joint、图内 attribute-shuffle 和粗粒度
patch-relation：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.mechanism_screen \
  --data-root data/ZINC \
  --radii 2 \
  --max-train-graphs 5000 \
  --max-valid-graphs 500 \
  --train-offset 5000 \
  --valid-offset 500 \
  --shuffle-repeats 3 \
  --model-seeds 0 1 2 \
  --n-jobs 2 \
  --result tracks/ksvd/results/luyin16/ZINC_MECHANISM_SCREEN_RADIUS2_OFFSET5000_20260830.json
```

两个非重叠切片已完成：joint-shuffle 信号复现，但现有 joint 未超过 typed raw，
relation 增益未跨切片达到 `0.01 MAE`，因此不晋级 full。判定与下一轮唯一候选见
`../../results/luyin16/ZINC_MECHANISM_SCREEN_20260830.md`。

`conditional_joint_screen.py` 将 radius patch 转为 train-only rooted structural
signature vocabulary，并检查 signature 条件下的 atom/bond 分布：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.conditional_joint_screen \
  --data-root data/ZINC \
  --radius 2 \
  --max-train-graphs 2000 \
  --max-valid-graphs 200 \
  --max-signatures 64 \
  --shuffle-repeats 3 \
  --model-seeds 0 1 2 \
  --result tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_JOINT_V2_SLICE_A_20260830.json
```

两个小切片和两个大切片均已完成。binding 相对 shuffle 可复现，但 conditional
未稳定超过 typed raw，故不晋级 full；见
`../../results/luyin16/ZINC_CONDITIONAL_JOINT_V2_20260830.md`。

`conditional_residual_screen.py` 将条件联合分布中心化为
`P(s,a)-P(s)P(a)`，再使用 train-only SVD16 压缩，并给每个 shuffle control
拟合同预算投影：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.conditional_residual_screen \
  --data-root data/ZINC \
  --radius 2 \
  --max-train-graphs 2000 \
  --max-valid-graphs 200 \
  --max-signatures 64 \
  --svd-components 16 \
  --shuffle-repeats 3 \
  --model-seeds 0 1 2 \
  --result tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_RESIDUAL_SVD16_SLICE_A_20260830.json
```

两个小切片的 binding control 均通过，但 residual 未稳定超过 typed raw，因此按
协议停止，未运行大切片。见
`../../results/luyin16/ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md`。

`long_range_object_relation_screen.py` 统计距离≥3 的局部对象对，并使用图内
position-shuffle 控制。schema 先按 vocabulary coverage 筛选，正式实验使用
覆盖率最高的 centre-atom relation：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.long_range_object_relation_screen \
  --data-root data/ZINC \
  --radius 2 \
  --max-train-graphs 2000 \
  --max-valid-graphs 200 \
  --min-distance 3 \
  --max-distance-bin 7 \
  --object-schema atom \
  --max-relations 512 \
  --shuffle-repeats 3 \
  --model-seeds 0 1 2 \
  --result tracks/ksvd/results/luyin16/ZINC_LONG_RANGE_ATOM_RELATION_SLICE_A_20260830.json
```

两个小切片未通过预测或 relation gate，未扩大。结果见
`../../results/luyin16/ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md`；阶段综合判断见
`../../results/luyin16/ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md`。

`multiscale_radius_screen.py` 固定比较 r2、r3 与 r2+r3 typed raw：

```bash
uv run python -m tracks.ksvd.experiments.luyin16.multiscale_radius_screen \
  --data-root data/ZINC \
  --max-train-graphs 2000 \
  --max-valid-graphs 200 \
  --r2-max-nodes 12 \
  --r3-max-nodes 20 \
  --model-seeds 0 1 2 \
  --result tracks/ksvd/results/luyin16/ZINC_MULTISCALE_RADIUS_SLICE_A_20260830.json
```

四个切片中 multiscale 均为最优，但增益未稳定达到晋级门槛，因此未运行 full。
结果见 `../../results/luyin16/ZINC_MULTISCALE_RADIUS_20260830.md`。
