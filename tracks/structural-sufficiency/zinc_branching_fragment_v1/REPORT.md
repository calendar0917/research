# ZINC P3-BF：typed rooted branching fragments（单模型，seed 0）

日期：2026-10-22 ・ 运行 commit：`784d0ada9d50ca6c9e8fe26be7e88afc82486d5f` ・ 协议：`configs/bf_v1.json`
状态：**完成（按预声明规则在 epoch 100 早停）；无 0.05x 证据；不升级、不再开第二 arm。**

一句话结论：在 P3 主体不变、只加 198 维 rooted branching-fragment 统计（S3/S4/F4，W_s 零初始化）的条件下，
新训练没有产生任何优于 epoch 0（=P3B 函数）的 checkpoint；best/test 数值上就是 P3B，
因此本轮不能支持“显式联合分叉类型信息改善单模型泛化”的假设，但也不能把失败单独归因于“信息无用”。

| | P3-BF_s0 |
|---|---:|
| 参数 | 447,505（继承 413,505 + 新 34,000；<500k） |
| 初始化 | P3B best.pt @377，sha256 `4c379c93…dc34f` |
| best valid（epoch 0） | 0.076223756（stored 0.076223753） |
| best 来源 | **epoch 0，继承的 P3B，不是新训练** |
| test raw-y MAE | **0.069864822**（P3B 0.069864841；Δ = −1.9e−8） |
| eval-mode train MAE @best | 0.014679794 |
| 计划 / 实跑 | 600 epoch / **100**（预声明早停），47,400 / **7,900** updates |
| GPU 时间 | formal 217.4 s；job wall 232.2 s（0.0645 GPU-h） |

---

## 1. 设计与协议（冻结内容）

- 数据：官方 PyG ZINC subset，train 10,000 / valid 1,000 / test 1,000；train 标签训练、valid 选点、冻结后 test 只评一次。
- 结构：P3 null model 的 encoder / composer / reader 原样；第一层改为
  `SiLU(W_x · raw108 + b + W_s · s_v)`，`W_s ∈ R^{128×198}` **零初始化**；`s_v` 由每 (root, family) 的
  `[mu, mean-square, log1p(count), exists]` 组成（每族 66 维 ×3 = 198）。
- 对象：S3（root+3 邻居，4 节点）、S4（root+4 邻居，5 节点）、F4（root v 的邻居 u + N(u)\{v} 中两个不同节点，4 节点）；
  完整 typed induced adjacency（含额外边），root 固定在 canonical position 0，非 root 节点最多 24 个排列取字典序最小；
  key 只含 atom category + 10 个 pair type；同一 (root, family, 节点集合) 只计一次，不同节点集合相同内容保留 multiplicity；
  family 独立；空集为精确空集。只从原始 atom/bond 结构提取，**不使用标签**。
- 训练：seed 0、batch 128、drop_last=False（79 steps/epoch）、raw-y L1、clip 5、dropout 0；
  继承参数 lr 峰值 1e-4 / 最小 1e-6，新模块 1e-3 / 1e-5，AdamW wd 1e-5，warmup 200 updates，
  共享相对 per-step cosine；早停规则：≥100 epochs，相对 reference（初值 epoch0 valid）累计改善 ≥0.0005
  才更新 reference 并清零 patience，否则连续 100 epochs 停止；单个更小改善只按严格 < 更新 best，不丢。

## 2. 新编码到底区分什么、覆盖多少

**对象规模（train，structure-only cache）**

| family | occurrences | 非空 root 比例 | unique patterns | max per root | max per graph |
|---|---:|---:|---:|---:|---:|
| S3 | 82,568（24.8%） | 31.6% | 341 | 4 | 29 |
| S4 | 3,104（0.93%） | 1.34% | 73 | 1 | 5 |
| F4 | 247,704（74.3%） | 78.1% | 866 | 7 | 87 |
| 合计 | 333,376 | — | 1,280 | — | — |

**相对 raw108 的新增区分**：231,664 个 train roots 只落在 10,699 个 raw108 组里（98.20% 的 roots 位于多成员组）。
新签名把 2,524 / 6,522 个多成员组细分（38.7%），涉及 5,048 roots（**2.18%**）和 9,989 / 10,000 个分子（**99.89%**）。
> **Erratum（2026-10-23，见 §9）**：`5,048 roots / 2.18%` 是统计 bug（`sum(len(v))` 对 `(roots, signatures)` 元组计成 2×组数）。
> 正确值是 **131,168 roots / 56.62%**；`2,524` 组、`38.7%` 组比例和 `99.89%` 分子比例不受影响。旧训练/checkpoint/预测未改动。
其中纯内容细分 1,735 组、纯 multiplicity 细分 4 组，其余 785 组两者都有（内容 2,520、multiplicity 789）。
因此新增列主要提供的是**联合内容**（额外边 / 两条候选边如何配对），而不是“同一内容数了几次”。

具体地，真实例子里同一 raw108 描述符下 F4 内容不同：
`mol 4 root 5` vs `mol 2383 root 13`（同 root descriptor，F4 的 pair 结构不同）；也出现同一内容 count 1 vs 3 的纯 multiplicity 组。
S3 在 F4 之外补充 root 的三个邻居联合类型，S4 覆盖四邻居（稀疏，仅 1.34% roots 非空）。

**valid 新内容**：full 198 维签名在 train 中未出现的 valid roots 为 276 / 23,083（1.20%）；
但至少有一个 family 的 pattern 未见过的 roots 只有 29 个（**0.126%**）；按 occurrence 加权，S3/F4 的未见比例均为 0.097%，
S4 为 1.91%。所以新编码在 valid 上几乎完全被 train 覆盖——它不是 novelty 机制，而是同一局部构型的更细统计。

**A/B 碰撞例（必须打折的论据）**：合成的 5 节点 A/B 对，root raw108 完全相同、F4 计数 A=1 / B=0；
但两整图的 bond 特征多重集相同、junction 特征多重集不同（4 vs 3 junctions）。
所以这只是一个 **root descriptor 碰撞**，不能证明整个 P3（bond/junction 通道）看不见该差异；本轮也没有做 P3-blindness 的对照。

## 3. CPU 门控（13/13 通过，本地与远端一致）

覆盖：P3B checkpoint 复算 valid 对齐；参数/optimizer groups；零新列输出与 P3B 一致（max|Δ| = 0.0）；
手工图精确计数 + 独立穷举 + 24 排列检查；图重编号/ batch 重组预测不变；multiplicity 与空集；
参数梯度与 LR 调度（phi 首步梯度为 0、第二步非零 0.0033）；cache 完整性；raw108 细分；A/B 碰撞；
train→valid 新内容；以及全部 447,505 参数都在 optimizer 中（新增 34,000，两组 LR 正确）。

13 项检查与表示统计分别在 `analysis/cpu_checks.json`、`analysis/representation_check.json`、`analysis/cache_stats.json`。

## 4. 训练、早停与真实增量时间

- 计时 4 epochs：steady 2.419 s/epoch；按公式 `E = min(600, floor((3600 − used − 180)/(1.15·t_epoch)))` →
  raw 估计 1,224，取 **E = 600**（79 steps/epoch，47,400 updates，feasible）。
- 正式训练从 epoch 0 开始；epoch 0 valid = 0.076223753（继承 P3B）。
- 训练动态：epoch 1 valid 即升至 0.076957，之后在 0.0777–0.0804 间波动；epoch 100 为 0.078363。
  **epoch 1–100 没有任何一个 epoch 严格优于 epoch 0**；online train 从 0.0119 降到 0.0067，
  eval-mode train 从 0.014680 降到 0.005626——模型在拟合 train，而不是改善 valid。
- 早停：epoch 100 时 patience 达到 100（reference 仍是 epoch 0，累计改善未达 0.0005），按预声明规则停止。
  **这是规则停止，不是预算或崩溃**；剩余 500 epochs 的 cosine 未跑，因此不能声称“完整 600 epoch 也不会恢复”。
- 时间：formal 217.4 s（100 epochs，平均 2.15 s/epoch）；整个 GPU job wall 232.2 s（含 4 epoch 计时、
  epoch 0 评测、best/final 评分、test cache 重建与 test）→ **0.0645 GPU-h**；另有 CPU 门控 job 13 s
  （此前一次格式被覆盖的同类 job 23 s）。本轮 GPU 占用合计约 **4.5–5 分钟**，远低于 60 分钟，未触发预算约束。
- test cache 是冻结后在远端从官方 PyG 重建的（`ad_test.pt` 当时不存在；10.2 s），只含结构，test 标签在冻结后才接触。

## 5. 结果

**总体（raw-y MAE）**

| model | params | best valid | ep | eval-train@best | test MAE | updates | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0（参照） | 413,505 | 0.076223754 | 377 | — | 0.069864841 | — | — |
| L_s0（参照） | 494,193 | 0.078537152 | — | — | 0.066266555 | — | — |
| DBT_s0（参照） | 452,523 | 0.077768402 | 949 | — | 0.069401494 | 235,500 | 3.6918 |
| DBT-PA_s0（停止） | 457,443 | 0.140224125 | 287 | — | 0.107658585 | 48,670 | 0.7257 |
| **P3-BF_s0** | **447,505** | **0.076223756** | **0** | **0.014679794** | **0.069864822** | **7,900** | **0.0645** |

P3-BF 的 best valid 是重新加载 `best.pt` 的评测值；checkpoint 内记录为 0.076223753，epoch 0 首次评测为 0.0762237507，
差异均在 5e−9 内（批内浮点归约）。

Paired bootstrap（10,000 reps，按 row 配对，Δ = AE(P3-BF) − AE(ref)，负值更好）：

| 对照 | mean ΔAE | 95% CI | P3-BF 更好的行比例 |
|---|---:|---|---:|
| P3B_s0 | −1.90e−8 | [−4.19e−8, +3.05e−9] | 0.521 |
| L_s0 | +3.598e−3 | [−1.026e−3, +8.046e−3] | 0.474 |
| DBT_s0 | +4.633e−4 | [−4.362e−3, +5.308e−3] | 0.497 |

P3-BF 与 P3B 的 test 差异是纯数值噪声（同一函数，预测最大绝对差 4.29e−6）；
相对 L 与 DBT 的差异也不显著（CI 跨 0），且方向与 P3B 相同：test 上 L 更好、DBT 略差。
这个 round **没有产生新的 test 证据**，只是把 P3B 的 test 结果按协议重报了一次。

**固定四组（analysis-only 的 covered/unseen × ordinary/tail；保留 row172）**

| split | group | n | P3-BF | P3B_s0 | L_s0 | DBT_s0 | Δ vs P3B |
|---|---|---:|---:|---:|---:|---:|---:|
| valid | covered_ordinary | 710 | 0.047467987 | 0.047467989 | 0.044863620 | 0.042996182 | −1.3e−9 |
| valid | covered_tail | 39 | 0.116284397 | 0.116284324 | 0.225481206 | 0.258636023 | +7.3e−8 |
| valid | unseen_ordinary | 225 | 0.133939598 | 0.133939597 | 0.130773440 | 0.141319985 | +1.2e−9 |
| valid | unseen_tail | 26 | 0.301922459 | 0.301922459 | 0.325622742 | 0.206050414 | −2.0e−10 |
| test | covered_ordinary | 705 | 0.047252017 | 0.047252013 | 0.043744029 | 0.048997656 | +4.2e−9 |
| test | covered_tail | 51 | 0.128494090 | 0.128494506 | 0.076440863 | 0.086732626 | −4.2e−7 |
| test | unseen_ordinary | 218 | 0.114921470 | 0.114921486 | 0.114230265 | 0.118928690 | −1.5e−8 |
| test | unseen_tail | 26 | 0.190233487 | 0.190233387 | 0.254858953 | 0.173397248 | +1.0e−7 |

四组 n 在 valid/test 分别合计 1,000/1,000，row172 未被删除或特殊处理；分组只用 y 与结构标记，不参与选点。

## 6. best 点来自哪里

**epoch 0，继承的 P3B，不是新训练。** 证据：

- `freeze.json`：`best_epoch = 0`，`best_is_epoch0 = true`，stored valid 0.076223753。
- `best.pt` 与 `epoch0.pt` 的 state_dict 逐 tensor 相同（同一个 state hash `e63cfe1e525d3735`；文件字节不同仅因 epoch0 多存了 `source` 元数据）。
- 继承的 24 个 P3 参数与 P3B best.pt **逐 tensor 完全相等**；新增 7 个参数中 `w_s.weight` 全零，
  branch 的随机初始化被零门完全屏蔽。
- 因此 best 的函数就是 P3B；test 预测最大差 4.29e−6（独立评测的浮点归约噪声），不是新结果。

新训练最好的 valid 是 epoch 1 的 0.076957（比 epoch 0 差 7.33e−4），之后没有回到 0.0762 附近；
最终 epoch 100 为 0.078363。**没有任何“新训练产生的更好点”可以被报告。**

## 7. 解释边界：哪些说法不能区分；下一笔训练是否值得

本轮能确定的：在冻结的这套 P3-BF 接口与两组 LR/100-epoch 早停规则下，**没有观察到泛化改善**，
且 best 仍是 P3B；新增编码确实在 root 层区分了一批 raw108 无法区分的联合构型（原文 2.18% roots 为统计 bug，正确值 56.62% roots，覆盖 99.89% 分子，见 §9），
但其中绝大多数在 valid 上已被 train 覆盖，且本轮没有证明这些信息对 **full P3** 是新的。

不能区分的至少四种解释：

1. **信息无用**：联合分叉类型对 ZINC 标签没有可迁移的额外信号。
2. **优化破坏**：继承 P3B 已收敛解后，以 1e-4/1e-3 的两组 LR 继续训练，先破坏了原解；
   valid 在 epoch 1 就变差、train 误差持续下降，和这一解释相容。
3. **早停截断**：预声明规则在 100/600 就停止；cosine 还在高位（epoch 100 时 inherited lr 9.36e−5、new 9.36e−4），
   完整 600 epoch 是否会在低 LR 阶段恢复，本轮没有数据。
4. **接口太弱**：198 维统计经一个零初始化线性门注入，可能不足以让 reader 利用这些联合信息；
   或者 phi 的 98→64→32 表示在 7,900 updates 内没有学到有用内容。

本轮**没有** matched P3 continuation control（同 schedule、无新分支），所以 (2) 与 (1) 无法分离；
也没有更长 patience 或更低 new-LR 的对照，所以 (3)/(4) 也无法分离。
A/B 例还提醒我们：raw108 的 root 碰撞不等于 P3 整体不可见，因此“P3 一定缺这个信息”的前提本身未被建立。

**是否值得再买一笔训练？** 以本轮证据，**不值得**在原方向上再加 seed/arm/epochs：
它只会重复同一个不可识别的混合。若一定要买，唯一有信息量的是**匹配的 P3 continuation 对照**
（相同数据、相同两组 LR、相同 100/600 早停、去掉 branch 与 s_v），用来把“LR/优化破坏”从“信息无用”里分出来；
这属于新的预注册，不应在本轮内偷偷补跑。另一个更根本的修正是协议层面：对“先降后恢复”的微调阶段，
100-epoch patience 太短，应该在第一轮就声明更长的观察窗，而不是在结果出来后放宽。

## 8. 交付物、资产与复现

```
tracks/structural-sufficiency/zinc_branching_fragment_v1/
  DESIGN.md                    设计记录
  configs/bf_v1.json           协议冻结
  code/bf_data.py              S3/S4/F4 枚举、canonical key、cache、collate
  code/bf_models.py            BranchingFragmentBranch + P3-BF
  code/bf_checks.py            13 项 CPU 门控（写 analysis/）
  code/bf_experiment.py        gate → timing → formal → score → freeze → test
  code/bf_report.py            本地汇总（main_table / groups / curves / summary）
  code/tests/test_bf_cpu.py    8 个快速单测
  analysis/                    cpu_checks, representation_check, cache_stats,
                               timing, plan, freeze, test_report, groups_*,
                               summary, main_table, groups.md, curves.png, curve_enriched.csv
  runs/bf_s0/                  best.pt, final.pt, last.pt, epoch0.pt,
                               train.json, curve.csv, predictions/{valid,test}_best.npz
```

**Checkpoint / hash**

| 文件 | sha256 | 说明 |
|---|---|---|
| 初始化 P3B best.pt | `4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f` | epoch 377；stored valid 0.0762237511 |
| `runs/bf_s0/epoch0.pt` | `6fa12f0ad2b69a16b6806dc8d15ea721de88438711671d040dbff8cc8e6821a1` | 含 `source` 元数据；state 与 best 相同 |
| `runs/bf_s0/best.pt` | `298246c65ecb3020271c0cc83883123f977b8ee0aa601ffd2568205d189c0b47` | epoch 0；本轮 best |
| `runs/bf_s0/final.pt` | `33a4099b7d75206cd94d64c7fd55ef12bd742203b19e800f408ad3c0dba2eca7` | epoch 100，含 optimizer |
| `runs/bf_s0/last.pt` | `4c9c46650fec5b5980b42e24c38b219c1bdad4bf4b3a3aff0bed66881b069cb8` | epoch 100 state |
| `analysis/freeze.json` | — | 冻结时记录 `best_sha256`、init sha、caches sha |
| `analysis/test_report.json` | — | test 预测 sha `d2353ef1…a97b` |

**cache 语义**：train/valid 由冻结的 P3 object cache 构建（`objects_train10k.pt` /
`objects_valid.pt`，见 `cache_stats.json` 的 source/sha），只读结构、`labels_used=false`、`test_used=false`；
test cache 在冻结后从官方 PyG 重建，只读结构、`test_used=true`（仅表示结构来自 test split，不代表标签参与选择）。
train/valid cache 的 sha256 记录在 `freeze.json`（`287a26a9…` / `8de445d1…`），bf_test 记录在 `test_report.json`
（`0b7bc11b…`，1.04 MB）。cache 不进 git（`.gitignore`），但可由代码从原始对象重建。

**参数拆分**：atom_encoder 26,336；bond_encoder 62,816；junction_encoder 112,896；group_composer 62,976；
reader 148,481；w_s 25,344；branch 8,656；合计 **447,505**，全部 `requires_grad` 且在 optimizer 中；
new 组 34,000，inherited 413,505。

**复现命令**（远端 `res`，pool `res-gpu0`；rr 只运行已提交 revision）：

```bash
# 0) 本地：CPU 检查（也可远端跑）
python tracks/structural-sufficiency/zinc_branching_fragment_v1/code/bf_checks.py

# 1) 部署并跑 CPU 门控（远端）
rr deploy res --pool res-gpu0
rr run res bf-cpu --pool res-gpu0 --gpus 1 --time 0:30:00 \
  --result tracks/structural-sufficiency/zinc_branching_fragment_v1/analysis \
  -- python tracks/structural-sufficiency/zinc_branching_fragment_v1/code/bf_checks.py

# 2) 正式训练：gate → 4 epoch 计时 → E → 训练/早停 → score → freeze → test
rr run res bf-run-s0 --pool res-gpu0 --gpus 1 --time 2:00:00 \
  --result tracks/structural-sufficiency/zinc_branching_fragment_v1/analysis \
  --result tracks/structural-sufficiency/zinc_branching_fragment_v1/runs \
  -- python tracks/structural-sufficiency/zinc_branching_fragment_v1/code/bf_experiment.py run --device cuda:0

# 3) 拉回并汇总
rr pull bf-run-s0
cp -r .rr/pulled/bf-run-s0-*/result/tracks/structural-sufficiency/zinc_branching_fragment_v1/analysis/. \
      tracks/structural-sufficiency/zinc_branching_fragment_v1/analysis/
python tracks/structural-sufficiency/zinc_branching_fragment_v1/code/bf_report.py
```

本轮运行的 commit 是 `784d0ad`（`freeze.json.git_commit`）；报告与汇总文件在此后提交，故报告 commit 会晚于运行 commit。
环境：torch 2.5.1+cu124，A100-40GB，`res-gpu0`。

---

## 中文摘要（回答 5 个问题）

**1. 新编码具体区分什么、覆盖率多少？**
它区分的是 **root 的联合分叉内容**：S3 的三邻居、S4 的四邻居、F4 的“一个邻居 + 该邻居另外两个邻居”的完整 typed induced adjacency（含额外边），
按 15 字节 canonical key 计数；同一节点集合只计一次，相同内容的不同节点集合保留 multiplicity。train 上 31.6% 的 roots 有 S3、
1.34% 有 S4、78.1% 有 F4，共 333,376 个 occurrences / 1,280 个 unique patterns。
相对 P3 的 raw108 root 描述符：98.20% 的 roots 位于多成员 raw108 组；新签名细分了 2,524 / 6,522 个多成员组，
涉及 5,048 roots（**2.18%**，统计 bug，正确值见 §9）和 9,989 / 10,000 个分子（**99.89%**）；其中纯内容细分 1,735 组、纯 multiplicity 仅 4 组。
valid 上 full 签名未见过的 roots 占 1.20%，但“至少一个 pattern 未见过”的 roots 只有 0.126%，按 occurrence 计 S3/F4 约 0.097%、S4 1.91%。
A/B 合成例确实做到 root raw108 相同而 F4 计数 1 vs 0，但两整图的 junction 特征多重集不同，所以这只是 root descriptor 碰撞，
**不能证明 full P3 看不见该差异**；这一点必须保留。

**2. 训练/早停是否完成？真实增量时间多少？**
按预声明协议完成：E=600、47,400 updates 的计划已冻结，但 epoch 100 时触发早停（100 epochs 无 ≥0.0005 的累计改善），
实跑 7,900 updates / 100 epochs；剩余 500 epochs 的 cosine 未跑。不是预算、崩溃或人工中断。
formal 训练 217.4 s，整个 GPU job wall 232.2 s（0.0645 GPU-h），加 CPU 门控 job 13 s，
本轮 GPU 占用约 **4.5–5 分钟**，远低于 60 分钟目标。

**3. best-valid / test / eval-train 相对 P3B、L、DBT 如何？**
best valid = **0.076223756 @ epoch 0**（stored 0.076223753），与 P3B 的 0.076223754 数值等价；
eval-mode train @best = 0.014679794；test = **0.069864822**，与 P3B 0.069864841 的 paired ΔAE = −1.9e−8
（95% CI [−4.2e−8, +3.1e−9]），即同一函数。
相对 L（test 0.066266555）差 +3.60e−3（CI 跨 0），相对 DBT（0.069401494）差 +4.63e−4（CI 跨 0）；
方向与 P3B 相同，没有新的显著性结论。四组 valid/test 表与 P3B 的差异都在 1e−7 以内，row172 保留。

**4. 最好的点来自新训练还是 epoch 0？**
**epoch 0（继承的 P3B），不是新训练。** `best.pt` 与 `epoch0.pt` 的 state_dict 逐 tensor 相同，
继承的 24 个 P3 参数与 P3B best.pt 完全相等，`w_s.weight` 全零、branch 被零门屏蔽；
test 预测与 P3B 最大差 4.29e−6。新训练的最好 valid 是 epoch 1 的 0.076957（比 epoch 0 差 7.3e−4），
之后未回到 0.0762；final epoch 100 为 0.078363。**新增训练没有产生更好的 checkpoint。**

**5. 哪些解释不能区分？下一笔训练是否值得买？**
不能区分：(a) 联合分叉信息对标签无用；(b) 1e−4/1e−3 两组 LR 在已收敛 P3B 上先把解破坏；
(c) 100/600 早停截断了低 LR 恢复窗口；(d) 198 维统计 + 零初始化线性门的接口太弱。
valid 在 epoch 1 就变差、train 误差持续下降，与 (b)/(c) 相容；本轮没有 matched P3 continuation control，
无法把 (b) 从 (a) 中分离；也没有更长 patience / 更低 new-LR 的对照。
以本轮证据，**不值得**在原方向继续买 seed/arm/epochs；若必须再买一笔，唯一有信息量的是
“相同两组 LR、相同早停、去掉 branch/s_v 的 P3 continuation”对照，用来检验 (b)；
这需要新的预注册，不应在本轮补跑。当前没有 0.05x 证据，停止本轮，不升级、不开第二 arm。


---

## 9. Erratum：BF 表示统计的 root 覆盖率（2026-10-23）

**性质：纯统计/展示修正，不改变任何旧训练、checkpoint、valid/test 预测或结论。**

### 9.1 `representation_refinement` 的 root 计数 bug

`bf_checks.representation_refinement` 以
`subdiv[k] = (roots, full_signatures)` 存储被细分的 raw108 组，随后写

```python
roots_sub = sum(len(v) for v in subdiv.values())   # 错：len((roots, sigs)) == 2
```

它计算的是 `2 × 细分组数`，而不是 root 数。正确写法是

```python
roots_sub = sum(len(roots) for roots, _ in subdiv.values())
```

CPU 重算 train 表示（只读 train；没有读取 test；`analysis/representation_check.json`
覆盖写入，`analysis/erratum_representation_fix.json` 记录 old/new 与恒等检查）：

| 量 | 旧（bug） | 正确 |
|---|---:|---:|
| 细分的 raw108 组 | 2,524 | 2,524（不变） |
| 被细分组覆盖的 roots | 5,048 | **131,168** |
| root 覆盖率 | 2.18% | **56.62%**（131,168 / 231,664） |
| 被细分组覆盖的分子 | 9,989 / 10,000 | 9,989 / 10,000（不变，99.89%） |
| 多成员组被细分比例 | 38.7% | 38.7%（不变） |

新增恒等检查（`erratum_representation_fix.json: identities`，全部为 true）：

- `n_molecules_in_subdivided_groups <= n_roots_in_subdivided_groups <= n_roots`；
- `roots_sub == sum(group sizes)`；
- 两个 fraction 分别能 round-trip 回对应计数；
- `roots_sub != 2 × n_groups_subdivided`（专门锁住本次 bug 的形式）。

细分组大小：min 2、median 11、mean 52.0、max 10,529 roots。

**口径影响。** 旧报告“新签名只涉及 2.18% roots”这句话是错误的；实际是多数 roots
（56.62%）所在的 raw108 组被新内容/多重度签名细分。这使“新增列只影响极少数 root”
这一打折理由不再成立，但它**不**改变已验证的训练事实：新训练的每个 epoch 的 valid
都不优于继承的 epoch 0，best 仍是 P3B 函数。也不能由此反推“信息充分”或“信息无用”；
root raw108 碰撞本身仍不证明 full P3 通道看不见差异（§2 的 A/B 例）。

**旧值不得再次引用。** `2.18%` / `5,048` 只作为 erratum 的历史值出现。

### 9.2 `descriptor_summary` 的展示口径

`descriptor_summary` 之前对 `feat[40:42]` 做 `np.argmax`，把 `ring_bonds` 区块
（两个**连续**标量：ring-bond count 与 ring-bond fraction）当成 one-hot。现有代码改为
从权威 layout `structural_relation_objects_v1.atom_blocks()` 读取
`in_ring`、`ring_bonds` 等区块，并输出
`ring_bond_count: float(...)` 与 `ring_bond_fraction: float(...)`。
**这只是 README/例子里的描述展示**；模型输入仍是 raw108 向量本身，未做任何改动，
也没有因此重跑或改动旧实验。

### 9.3 版本

- 旧训练与报告：commit `784d0ad`；旧报告提交 `5255cda`。
- 本 erratum + 修正代码：紧随本轮的独立 commit（见 `git log`）。
- 修正后的机器可读记录：`analysis/representation_check.json`、
  `analysis/cpu_checks.json`（13/13 仍通过）、
  `analysis/erratum_representation_fix.json`。
