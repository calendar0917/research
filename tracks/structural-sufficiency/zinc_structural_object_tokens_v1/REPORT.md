# REPORT — zinc_structural_object_tokens_v1

Track `structural-sufficiency`。两臂单模型（无 ensemble / seed 1 / 网格 / 第三臂），
seed 0，预注册见 `DESIGN.md` 与 `configs/ot_v1.json`。旧 BF 轮的统计 erratum 见
`../zinc_branching_fragment_v1/REPORT.md` §9（旧 2.18% / 5,048 不再引用）。

## 0. 结论

1. **A_joint_BF（447,505 参数，SSL 起点、全参数联合训练 600 epochs / 47,400 updates）**
   best-valid `0.076731` @375，final-valid `0.078218`，**test `0.070067`**。
   与 P3B_s0（`0.069865`）差 `+0.000203`，paired bootstrap 95% CI `[-0.00325, +0.00353]`，
   候选更好的重采样比例 0.467：**与 P3B 无法区分，没有改善**。
   旧 BF 的“监督 P3B 起点 + 100 epochs 早停”不是被早停掩盖的增益——同样的 S3/S4/F4
   对象从 SSL 起点完整训练 600 epochs 也回不到 P3B 以下的错误率。
2. **B_object_tokens（470,469 参数，atom + rooted branch + chordless cycle token，4 层全图
   attention，490 epochs / 38,710 updates）** best-valid `0.135830` @230，final-valid
   `0.149754`，**test `0.108306`**。与 P3B 差 `+0.0384`（CI `[+0.031, +0.046]`，候选更好
   比例 0.31），四个固定组也全部更差。这是一个**明确的负结果**：在本预算下，把容量摊到
   每分子约 59 个对象 token、用单一 LR/单一 reader 训练，明显劣于 P3 的 413k 参数基线。
   best 点的 eval-mode train MAE 为 `0.0512`（A 为 `0.0157`）→ 主要瓶颈是**欠拟合/优化**，
   不是对象覆盖不足。
3. **0.05x 目标没有接近**：本轮最好的 test 是 A 的 `0.0701`，仍是 0.05 的 **1.40x**
   （P3B 1.40x、L 1.33x、DBT 1.39x）。两臂都未产生新的性能水平；边界内没有证据表明
   0.05 可达。

正式训练 GPU 合计 **0.88 A100-h**（A 0.364 h + B 0.518 h）；另有一次因实现瓶颈中止的
B 试跑（4 个 120 s 的计时 epoch）与剖析 job 约 0.17 h，**总计约 1.05 A100-h**。
两卡并行，正式墙钟约 50 分钟。

## 1. 边界（与预注册一致）

- 数据：official PyG ZINC subset，train 10,000 / valid 1,000；每 epoch 全量 valid；
  strict `<` 选 best（相同取早）；**冻结后**两臂各读一次 test（本仓库 test 已历史暴露，
  不声称盲测）。test 不参与任何选择。
- 无递归局部邻居 hidden-state 更新；无真实 SA/logP/cycle 分量、无 y 反推、无推理 gate、
  无 whole-pattern 查表、无 VQ/codebook、无 teacher、无 ensemble/平均预测、无第三候选、
  无 seed 1、无网格。
- 静态 cache 只含内容 / 成员 / 静态关系；不含 live hidden state 或 φ 输出。
- 复用：P3 的 `ad_{train,valid}.pt`、BF 的 `bf_{train,valid}.pt`、seed-0 train-only
  masked-SSL 文件（sha256 `006fabc1…fa97`，两侧构建时都核对）。

## 2. A_joint_BF

结构与静态内容与旧 BF 完全相同（P3 原模块 + φ(98→64→32) 聚合的 198 维 `s_v` 经零初始化
`W_s` 进 atom encoder 第一层；S3/S4/F4 的 rooted typed induced adjacency、root 在 slot 0、
不同节点集合保留 multiplicity、空集为精确空集）。**唯一变量是起点与训练方式**：

- 初始化顺序：`manual_seed(0)` → 按历史顺序构造 P3 核心 → 只加载 SSL 的
  atom/bond/junction encoders → 追加 `w_s`（零）与 BF 模块（正常随机）。
  `ot_checks` 验证 22 个 P3 参数张量 + 2 个 buffer 与历史 `BJOC(seed0)+load_encoder`
  逐 tensor 位级相同；与监督 P3B 的 composer/reader 明显不同；`W_s` 精确为零；
  epoch-0 前向与同初值 P3 完全相同；φ 首步梯度 0、第二步打开。
- recipe B：AdamW 单组，peak `1e-3` / min `1e-5` / wd `1e-5`，batch 128
  （`drop_last=False`，79 updates/epoch），warmup 500，per-step cosine，
  固定 **600 epochs = 47,400 updates**，clip 5，dropout 0，无早停。
- 结果：steady `2.29 s/epoch`，wall `1310 s`（含 4 个计时 epoch 与 test），
  best-valid `0.076731364` @375，eval-mode train@best `0.015666`，
  final-valid `0.078218`（train@final `0.000665`），test `0.070067461`。
- 与参照（同一冻结 test 行）：P3B `+0.000203`、L `+0.003801`、DBT `+0.000666`。
  A 在 covered_tail（n=51）比 P3B 低 `0.0387`，但在 unseen_tail（n=26）高 `0.0174`；
  两个子组都很小，不作为机制证据。

## 3. B_object_tokens

### 3.1 结构（实现即预注册）

- token：每个原子 1 个 A；BF 的真实 rooted S3/S4/F4（实际成员集合、root、typed induced
  adjacency、family，按 `(root, family, node set)` 去重、同内容不同节点集合保留）；
  每条长度 3–8 的 **chordless** 简单环 1 个 C（穷举，无 SSSR/无 sampling；canonical 内容
  取 2L 个旋转/反向的最小字典序，bond 数组与 atom 数组一起旋转/反向，无 root）。
  5 个类型 id，无 atom-index 身份 embedding。
- train 592,848 tokens（231,664 A + 333,376 branch + 27,808 C；branch 家族
  S3 82,568 / S4 3,104 / F4 247,704 与冻结 BF cache 逐 `(molecule, family, root)` 一致）；
  valid 59,083；test（冻结后才构建）59,582。
- 编码：atom 28×8、pair 4×4 共享表；branch φ 98→64→32；ring φ 118→64→32；
  `a_v` 只复用 P3 的 SSL **atom** encoder（26,336 参数，fine-tune），不创建 P3 的
  bond/junction/composer/reader。
- 主干：4 × pre-LN block（width 96 / heads 4 / FFN 192 / SiLU / dropout 0.05），
  SDPA 全图 attention，只屏蔽 padding 与其它分子；5 张共享的加性 per-head 关系表
  （25 type + 9 overlap + 11 distance + 5 bond + 3 root）×4 heads = 212 参数，
  无 `[B,H,T,T,hd]` relation value，无局部/共享成员 hard mask。
- 读出：atom/branch/cycle 三组 `sum / 固定 train 均值`（下限 1）与 `mean`（空集=0）
  6×96 + 标准化 global33 + 3×log1p = 612 → 96 → 48 → 1，dropout 0.05。
- 参数 **470,469**（atom encoder 26,336 / raw emb 240 / branch φ 8,416 / ring φ 9,696 /
  token 投影 62,400 / type emb 480 / 4 blocks 299,136 / 关系表 212 / reader 63,553）。

### 3.2 一个实现瓶颈（训练前修正，不改数学）

首次 B 试跑测得 `119–120 s/epoch`。剖析显示一个 step 的 1.34 s CUDA 时间里
**1.32 s 在 `indexing_backward_kernel`**：五张关系表用整数 advanced indexing 读取
~3.4M 个 token 对，backward 把上百万重复索引 scatter 进 25/9/11/5/3 行、在 atomics 上
串行。改为按 pair 维分块的 `onehot(cat) @ table`（one-hot 无梯度）后，取值逐元素相同
（`ot_checks` 保留与显式查表的对比），表梯度变成快速 reduction；稳态单 batch 约
15 ms forward + 25 ms backward（另有 ~8 ms collate / ~1.5 ms H2D），实测 epoch
`3.7–3.9 s`，比修正前快 ~31x。这只是实现修正，模型/对象/recipe/
预算公式都没有改动；修正后 E_B 才进入可行范围。

### 3.3 日程与结果

- 计时 4 个完整 epoch：steady `3.946 s/epoch`；`E_B = min(600, floor((2400−22−120)/(1.15·3.946)))`
  = `497 → 490`（向下取 10 的倍数），`≥300` 满足；79×490 = 38,710 updates。
- recipe：AdamW peak `1e-3` / min `1e-5` / wd `1e-4`，warmup 500，per-step cosine，
  clip 5；encoder/内容 MLP dropout 0，trunk/reader 0.05；无早停、无中途重置。
- best-valid `0.135830` @230（eval-mode train@best `0.051184`），
  final-valid `0.149754`（train@final `0.063911`），**test `0.108306`**；wall `1866 s`。
- 与参照差：P3B `+0.03844`（CI `[0.0313, 0.0459]`，更好比例 0.31）、L `+0.04204`、
  DBT `+0.03890`；valid 上四个组同样全部更差（covered_tail n=39 为 `0.758`）。

### 3.4 这个负结果说明什么、不说明什么

- 说明：在这套预算（~0.47M 参数、490 epochs、单一 LR、单一 reader）下，
  “每分子数十个对象 token + 全图 attention + 加性静态关系”没有表现出优于 P3 的
  归纳偏置；train@best `0.051` 远高于 A 的 `0.0157`，瓶颈首先在**优化/拟合**，
  而不是“对象家族不够全”。
- 不说明：显式结构一律无用，或 0.07 是理论上限；也不把 B 的差距单独归因于环、关系
  bias、池化或 attention（没有做消融，且预注册禁止本轮做）。

## 4. 主表（含参照，全部参数与训练量）

见 `analysis/main_table.md` / `.json`：

| model | params | init | best valid | ep | eval-train@best | final valid | test MAE | epochs | updates | GPU-h |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0 | 413,505 | train-only masked SSL | 0.076223754 | 377 | — | — | 0.069864841 | 600 | 47,400 | 0.30 |
| L_s0 | 494,193 | train-only masked SSL | 0.078537152 | 489 | 0.005510 | — | 0.066266555 | 600 | 47,400 | 0.35 |
| DBT_s0 | 452,523 | train-only masked SSL | 0.077768402 | 949 | 0.015964 | 0.081314 | 0.069401494 | 1500 | 235,500 | 3.69 |
| old P3-BF_s0 | 447,505 | supervised P3B best.pt | 0.076223756 | 0 | 0.014680 | 0.078363 | 0.069864822 | 100 | 7,900 | 0.065 |
| **A_joint_BF_s0** | 447,505 | SSL atom/bond/junction only | 0.076731364 | 375 | 0.015666 | 0.078218 | **0.070067461** | 600 | 47,400 | 0.364 |
| **B_object_tokens_s0** | 470,469 | SSL atom encoder only | 0.135830103 | 230 | 0.051184 | 0.149754 | **0.108306215** | 490 | 38,710 | 0.518 |

参照行是复用数字、未重训；A/B 的 test 是各自冻结后的唯一一次描述性读取。

## 5. 固定四组（仅分析）

`analysis/groups.md`（`signature_unseen` × `y>−3`，row 172 保留，分组不进模型）：

| split | group | n | P3B | L | DBT | A | B |
|---|---|---|---:|---:|---:|---:|---:|
| test | covered_ordinary | 705 | 0.047252 | 0.043744 | 0.048998 | 0.048276 | 0.086076 |
| test | covered_tail | 51 | 0.128495 | 0.076441 | 0.086733 | 0.089783 | 0.140032 |
| test | unseen_ordinary | 218 | 0.114921 | 0.114230 | 0.118929 | 0.119519 | 0.156073 |
| test | unseen_tail | 26 | 0.190233 | 0.254859 | 0.173397 | 0.207653 | 0.248343 |

valid 的同一张表在 `analysis/groups.md`。子组 n 很小，只描述、不做显著性主张。

## 6. 训练与时间

| | A | B |
|---|---|---|
| steady epoch | 2.29 s | 3.95 s |
| epochs / updates | 600 / 47,400 | 490 / 38,710 |
| wall（含计时、best/final 打分、test） | 1310 s | 1866 s |
| GPU-h | 0.364 | 0.518 |
| 数值精度 | fp32（未用 amp） | fp32（未用 amp） |
| 失败/中止 | 无 | 首次试跑因 120 s/epoch 触发预算规则，按预注册标 INCOMPLETE；实现修正后重跑一次 |

两臂在同一台 `res`（a100-2，driver 550.163.01，torch 2.5.1+cu124）的两张 A100-40GB 上
并行，各自独立 job；B 的两张卡互不干扰。

## 7. 证据与检查

- `analysis/cpu_checks.json`：**11/11 通过**（A 初值/零门/梯度；B 原子覆盖、与 BF 的
  occurrence 与 `(molecule, family, root)` 计数精确一致、`(family, root, node set)` 唯一、
  手工 star/fork/path/6-ring/带 chord 6-ring/K4 与独立穷举一致、环 canonical 在旋转/反向
  下不变且 bond 对齐、原子重编号不变、batch 重组与跨图隔离、空组、共享 embedding 与参数
  计数、五类关系语义逐对精确、bias 与显式查表逐值相同、LR/updates/重载语义）。
- `code/tests/test_ot_cpu.py`：6 个无数据依赖的快速单测通过。
- `analysis/cache_rebuild_parity.json`：test 没有冻结 object cache，是按同一构造器从
  official PyG 重建；在 valid 上逐 array 对比重建路径与冻结 cache：**33/33 array 完全
  相同、0 mismatch**，所有计数一致 → test 结构可信。
- 冻结：`analysis/<arm>/freeze.json`（best sha256、epoch、valid、初值来源与 sha、
  recipe、cache/label hash）。A `237db6cb…`，B `325e2b45…`。
- test 只读一次：`preds_test.csv` 与 `runs/*/predictions/test_best.npz` 的 MAE 与
  `test_report.json` 一致（A `0.0700674609`，B `0.1083062148`）。

## 8. BF 统计 erratum（上一轮，独立提交）

`bf_checks.py` 的 `representation_refinement` 把 `{group: (roots, signatures)}` 的
`len(v)` 当成 root 数（= 2×组数），因此上一轮报告的分母是错的。修正后 root 覆盖率为
**56.62%**（roots 131,168）；`2,524` 组、`99.89%` 有分子覆盖、`38.7%` 多组比例**不变**。
旧值不再引用，完整对照与 identity 检查见 BF REPORT §9。
`descriptor_summary` 的 `ring_bond_count` 改为按权威布局读取（不再是两个连续标量的
argmax），模型输入未变。旧训练/checkpoint/prediction 未触碰；证据在
`../zinc_branching_fragment_v1/analysis/erratum_representation_fix.json` 与 REPORT §9。

## 9. 下一步决策

**本轮到此停止，不再开第三臂/seed 1/环长或宽度网格。** 理由：

1. A 回答了它的问题：把 S3/S4/F4 放进 P3 并从 SSL 起点**联合**训练 600 epochs，
   仍然只是 P3B 水平（`+0.0002`，CI 跨 0）。旧 BF 的 100-epoch 早停不是“被掩盖的增益”，
   也不是初始化方式的错——对象层面的信息增量在 0.47M 参数级别不可见。
2. B 在覆盖更全（原子 + 全部 rooted 分支 + 全部 chordless 环 3–8、212 维静态关系、
   全图 attention）的情况下显著更差，且**训练误差**就远高于 A：说明当前瓶颈是
   优化/容量分配/读出，而不是“还缺一种对象”。在这个诊断下继续加对象家族、
   加环长、加 attention 或换 seed 1 都是低信息量的。
3. 如果要继续这条线，正确的问题是“如何让 P3 级别的拟合在对象 token 上成立”
   （例如分组学习率/更长日程/更平的读出），而不是再加结构；这已经超出本轮预注册，
   需要新的、明确的预算与假设，而不是在本轮追加。
4. 0.05 仍然没有接近的证据：现有最好 `0.0663`（L），本轮 A `0.0701`。
   不改写“显式结构无用”或“0.07 是上限”，只能说：**在本预算与这套配方下没有收益**。

## 10. 复现

```bash
# 本地：CPU 门控（11 项，约 3 s）与无数据单测
uv run python tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/ot_checks.py
uv run pytest -q tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/tests

# 远端（commit 1657b2c / 06b7fe5 之后）：两臂各一个 job，互不共享写路径
rr deploy res
rr run res ot-A-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_structural_object_tokens_v1/runs/A_joint_BF_s0 \
  --result tracks/structural-sufficiency/zinc_structural_object_tokens_v1/analysis/A_joint_BF \
  -- .venv/bin/python tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/ot_runner.py run --arm A --device cuda:0
rr run res ot-B-s0 --pool res-gpu1 \
  --result tracks/structural-sufficiency/zinc_structural_object_tokens_v1/runs/B_object_tokens_s0 \
  --result tracks/structural-sufficiency/zinc_structural_object_tokens_v1/analysis/B_object_tokens \
  -- .venv/bin/python tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/ot_runner.py run --arm B --device cuda:0
rr pull <run-id>

# 本地汇总（主表、四组、曲线）
uv run python tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/ot_report.py
```

`ot_runner.py run` 内含：CPU 门控检查 → (B) 结构 cache 构建 → 4 个计时 epoch →
重建同初值 → 固定日程训练（每 epoch 全量 valid、strict `<`）→ best/final/last →
`freeze.json` → 唯一一次 test → `test_report.json` / groups / preds。
`--epochs` 只用于恢复，会写进 plan；本轮两臂都没有用它。

## 11. 产物与 revision

代码与配置：`code/ot_{data,models,joint,runner,checks,timing,report}.py`、
`code/tests/test_ot_cpu.py`、`configs/ot_v1.json`、`DESIGN.md`。

每臂：`runs/<arm>_s0/{best,final,last}.pt`（A 另有 `epoch0.pt`）、`curve.csv`、
`predictions/{valid,test}_best.npz`、`train.json`；`analysis/<arm>/` 下
`freeze.json`、`plan.json`、`timing.json`、`curve.csv`、`preds_{valid,test}.csv`、
`test_report.json`、`groups_{valid,test}.json`、`groups_rows.json`、`status.json`。

共享：`analysis/{main_table.md,main_table.json,summary.json,groups.md,curves.png,
cpu_checks.json,cache_rebuild_parity.json}`。

结构 cache（gitignored，可重建）：`cache/ot_{train,valid,test}.pt`；
A 使用 `../zinc_branching_fragment_v1/cache/bf_{train,valid,test}.pt`。

revision：核心代码与预注册 `1657b2c`；剖析工具 `46e39f3`/`75fd470`；关系表实现修正
`06b7fe5`（正式 B 训练前）。

provenance 细节：**B 从 `06b7fe5` 启动**；**A 从 `1657b2c` 启动**（11:23），运行期间
checkout 因 B 的剖析/修正前进到 `06b7fe5`，而 `train.json` 记录的是**写文件时**的 HEAD，
所以 A 的 `git_commit` 显示 `06b7fe5`。两者之间变化的文件只有
`ot_models.py`、`ot_checks.py`、`ot_timing.py`、`analysis/cpu_checks.json`，都不被 A 导入
（A 执行的模块在 `1657b2c` 与 `06b7fe5` 完全相同），详见
`analysis/A_joint_BF/provenance_note.json`。

## 12. 限制

- 单 seed；A/B 与参照的差别都在单次运行内，不能写成机制确认。
- B 的日程由预算公式决定（490 epochs），比 A/P3B 短；B 的负结果只约束“本预算 + 本配方”，
  不是对结构的普遍否定。
- test 在本仓库已被多轮读取，此处只是描述性基准；不因 test 改变任何选择。
- 分组结论的 n 很小（tail 26–51），只用于描述误差分布。
