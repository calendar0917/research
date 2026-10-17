# DEVIATIONS — cscl-bjoc-sac-v1

按严重度排序。所有条目均在 `REPORT.md` 中对应披露。

---

## D-1 冻结被两次重写（amendment 1 / 2）

协议原文要求"freeze 之后代码不可改"。实际上 `protocol_freeze.json` 被**重写两次**：

| amendment | 时间（UTC） | 触发 | 改动 |
|---|---|---|---|
| 1 | 2026-10-17T09:05Z | smoke 在 20-update SSL 探针**之后、任何臂训练之前**抛 `TypeError` | `cmd_smoke` 一行：`dict(float)` → `{"SSL": float(...)}`（预算投影的报告 bug） |
| 2 | 2026-10-17T09:47Z | 两个 seed 的 `train` 启动 9 秒后抛 `AttributeError` | `cmd_train` 一行：梯度探针 batch 的 `{k: v.to(device)}` → `sac.move_batch(...)`（batch dict 里混有 int） |

- 两次都发生在**任何监督训练步、soup、预测之前**（失败进程分别停在 SSL 探针之后与
  第一个 optimizer step 之前），因此不存在"用旧代码训了一半"的轨迹。
- `sac_v1.py`（模型/关系/训练器）在两次 amendment 中**逐字节未变**，SSL 检查点由
  未改动的上游 `bjoc_masked_pretraining_v1.pretrain_masked` 产生，仍然有效。
- 每次重冻前都重跑 12/12 CPU gates；amendment 2 之后又补了一个
  synthetic 评分路径测试（共 18 个测试）。
- **这不是"无代价"**：它削弱了"单次冻结"的强度，任何读者都应当把本轮的 freeze 理解为
  "amendment 链的最后一环"。链本身记录在 `protocol_freeze.json` 的
  `amendment`/`amendment_reason` 字段与 `AMENDMENTS.md`。

## D-2 未计入 GPU ledger 的 GPU 时间

`gpu_budget_ledger.json` 只记录 SSL/train 轨迹（2.101 h）。未计入但已发生的 GPU 时间：

| 项 | 估计 | 说明 |
|---|---|---|
| `sac-smoke`（失败） | ~21 s | 20-update SSL 后崩在报告行 |
| `sac-smoke2` | ~22 s | 20-update SSL + 每臂 1 epoch（512 分子） |
| `sac-tr-s0` / `sac-tr-s1`（失败） | ~9 s + ~9 s | 崩在梯度探针，0 个 optimizer step |
| 两次 `'&&'` 误传的 `rr run` | ~1 s | 立刻 exit 2，未进 GPU |
| 合计 | **< 0.02 h** | 相对 4.0 h 上限可忽略，但如实列出 |

## D-3 无法通过 `rr deploy` 的远端 untracked 文件

远端 `res` 的 p3 轮工作树里有一个 **untracked** 的
`results/p3_bond_relation_bridge_v1/holdout_read.lock`（本地已 tracked）。
它挡住了 `git merge --ff-only`，deploy 失败。

处理：先核对远端文件与本地已提交版本 **sha256 完全相同**（`9d8c6950eb03…`），
再在远端把它改名为 `holdout_read.lock.rr-untracked-backup`（保留，未删除），
然后 deploy 成功。属于对**非权威、生成型、内容完全一致**的 untracked 文件的一次性维护，
没有触碰任何 tracked 文件。**唯一一次直接 SSH 到 `res`** 就是这条 `sha256sum`/`mv`；
所有实验执行仍然只经 `rr run`。

## D-4 `rr run` 不解释 shell 运算符

第一次启动正式训练时用了
`rr run res ... -- python ... ssl ... '&&' python ... train ...`，
`&&` 被当作 Python 参数，两个 run 立刻 exit 2（无 GPU 工作）。
改为**每个阶段一次 `rr run`**：`ssl → train → predict` 每 seed 顺序执行，
两个 seed 分别在 `res-gpu0`/`res-gpu1` 并行。协议要求的
"同一 seed 内 SSL 先于训练、预测在训练之后"没有被破坏。

## D-5 split S 复用同一个 10k 池（不是盲测）

本轮只允许官方 train 10k，因此 split S 是对**历史上已被多轮使用**的同一个池子的新划分：

- 9,996 行**全部**在某个历史 zone 出现过
  （`frac_rows_seen_in_any_historical_zone = 1.0`）；
- 新 Locked 对应旧 split F：fit 786 / locked 106 / monitor 108；
  新 Monitor 对应旧 F：fit 787 / locked 105 / monitor 108；
- Fit 内部有 6,423 行曾出现在旧 F 的 Fit 中。

影响：**跨轮比较无效**（本轮 P3 的 0.0868 与历史轮次的数字不可直接比），
但**轮内三臂对比不受污染**（同一 Fit、同一 Zone、标签只读一次）。
另外，`ZINC` 官方 test 从未加载，本轮也没有"test MAE 0.060"这类外部锚点。

## D-6 SACU 不是等 FLOP、也不是行和匹配的 null

`T_s^U = [m_s/C(n,2)]·Σ_{e<f} q_s(e)q_s(f)/sqrt(max(1,m_s))`：

- 每分子用**真实** `m_s`（而不是 R_s 的对数），所以缩放因子与 SACR 不同；
- 求和覆盖**全部**无序键对（不是 R_s），FLOP 约为 SACR 的 3.4 倍；
- 结果张量的数值尺度、范数、梯度流都与 SACR 不同。

因此 `SACR − SACU` 混合了"关系内容"与"数值尺度/优化条件"两种因素，
**不能**解释为真实连接性的因果效应。协议 §10 已预先声明这一点，
本报告没有把它写成因果结论。

## D-7 尾部（y ≤ −3）分析是 post-hoc

主表里的 `tail_mae` / `tail_delta` 是冻结评分代码的一部分（描述性），
但 `REPORT.md` §4.3 里给尾部 Δ 配的 **bootstrap CI 是读标签之后的 post-hoc 计算**，
且只有 n=67 个分子、Monitor 上方向相反。它只用于说明"Locked 的尾部改善不可靠"，
不作为任何判定依据。

## D-8 并发写同一份预测清单

两个 seed 的 `predict` 同时在同一台机器的同一目录写
`prediction_manifest_S.json`。写入在 `fcntl.flock` 互斥下完成；
随后 `freeze-predictions` 对 **28/28** 文件重算 sha256，0 处不匹配。

## D-9 本地结果目录是从远端拷回的副本

本地 `results/sac_v1/` 用 `rsync` 从 `res` 的权威目录取回，
并逐项对照 `evidence_manifest.json` 校验：**88/88** artifact 的 sha256 与字节数一致
（该清单是在写 REPORT/DEVIATIONS 之前于远端生成的；两份 Markdown 落盘后在本地重跑
`finalize`，最终清单为 **91** 项，同样逐项哈希）。
`.lock` 辅助文件（`prediction_manifest_S.json.lock`、`gpu_budget_ledger.json.lock`）
未拷回；`holdout_read.lock` 作为标签读取凭证保留并提交。

## D-10 关系清单记录的是轮次开始前的 commit

`relations_manifest_S.json` 的 `git_commit` 是 `d0ffa18…`（本轮第一个 commit 之前的 HEAD），
因为关系是在首次提交前构建的。其内容哈希已被 `protocol_freeze.json` 绑定，
provenance 不依赖该字段。

## D-11 发现（不是本轮偏离）：p3 轮的 salt 元数据 bug

构建 split S 时把 salt **显式**传入，并新增回归测试：用保存的 metadata salt 重算 split、
用另一 salt 做探针、按各历史 split 的 **effective** salt 逐行复现。结果确认：

> `p3_bond_relation_bridge_v1` 的 `split_F.json` 里记录的 salt 是
> `cscl-p3-bond-relation-bridge-v1`，但实际生效的 salt 是
> `cscl-bond-relation-attribution-v1`（builder 的默认值）——这就是 p3 轮
> DEVIATIONS D-1 "split F 与 split E 逐行相同"的根因。

本轮 `split_S.json` 同时记录 `salt`（实际使用）、`salt_source`、
`salt_reproduced_from_saved_metadata`、`salt_is_load_bearing`，
并把该 bug 的复现写成了 gate 的一部分。
