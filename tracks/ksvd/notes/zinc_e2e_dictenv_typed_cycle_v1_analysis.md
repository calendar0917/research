# 分析报告 — ZINC 类型化静态环对象（typed chordless cycle）v1

轮次 `zinc-e2e-dictenv-typed-cycle-v1`（任务包
`/home/calendar/Downloads/zinc_typed_cycle_plan.md` +
`zinc_typed_cycle_agent_prompt.md`，handoff `zinc_typed_cycle_handoff.zip` 已
vendored 为 `experiments/luyin16/typed_cycle_reference/v1_20261002/`，审计 revision
`7fa46ab300b43c6457499e45408296c5a4bd4c5c`）。

- 预注册：`notes/zinc_e2e_dictenv_typed_cycle_v1_preregistration.md`，
  训练前冻结 sha `a9b58bac5d5f`（preflight 强制校验）。
- 代码 revision：实现 `8edbb24`，离线诊断修复 `52091ca`，命名规范化 `1cb6ba5`。
- 正式 run：`20261002-092747-a66ae9c0`（promoted record
  `records/runs/20261002-092747-a66ae9c0.json`），单 seed 0，local CPU。
- 官方 test 从未实例化（`official_test_loaded = false`，test_access=blocked）。

## 1. 结论

**`TYPED_CYCLE_STOP`**（soup official-valid MAE `M_S = 0.120779`）。冻结分档规则为
`≤0.115` 值得后续 / `0.115–0.120` 有限信号 / `>0.120` 关闭；实测 `0.120779`
落在 `>0.120`，因此**封档**：不加宽、不追加 seed、不做匹配对照、不改
lambda/环长/hash/threshold，不读官方 test。

- 最好单 epoch valid `0.127393` @ 269；Top-5 soup 成员 `[269, 273, 305, 306,
  318]`，成员 valid `0.127393 / 0.129021 / 0.128967 / 0.130100 / 0.129744`。
- 同口径 soup train MAE `0.054857`，train→valid gap `+0.065922`。
- 精确参数 `133002`（= 冻结 Small `106925` + 环编码器 `24816` + reader 新增
  `1261`；新增任务字典参数 `0`）。
- 训练 wall `2962.0 s`（`9.256 s/epoch`，320 epoch，8 CPU threads），peak RSS
  `2418 MB`。

关键判读：新增环对象确实**活**且**承重**（见 §4），但**没有把绝对性能推进新的
区间**。`0.120779` 仅比 0.120 的有限信号边界高 `0.000779`，相对未匹配的 Small
背景 latent-bridge `0.121058` 只差 `-0.000279`（基本持平），且**差于**同族
Full m=3 的 `0.119154`。因此本轮回答的是“这个静态对象能不能救绝对性能”——答案
是不能，至少在这一个 seed-0 轨迹与冻结协议下。

## 2. 阶段 A（已通过，购买门槛）

在真实 train-only 数据上对 `typed_cycle_reference.cheap_property_probe` 做了
一次廉价性质筛查（冻结 3 折 OOF，lambda 固定 0.001，无 λ/宽度/子集搜索）：

- 对象普查：10000 图 / 27810 个 chordless cycle / 313334 个二面体视图；
  无环图 55（比例 0.0055）；每图环数 mean 2.781 / p50 3 / p95 4.05 / max 10；
  长度分布 `{3:629, 4:166, 5:8519, 6:18052, 7:346, 8:96, 9:1, 10:1}`；每次对象
  枚举 + 特征 wall `43.06 s`。
- 筛查：OOF base MAE `0.362618` → typed `0.331320`，**OOF gain `0.031298`**，
  3/3 折为正（fold gains `0.032793 / 0.029569 / 0.031532`）。
- 判定 **`BUY_ONE_TYPED_CYCLE_SCREEN`**（门槛：OOF gain ≥ 0.003 且 ≥2/3 折为
  正）。NPZ sha `a7959476...3166d53`；split fingerprint `58c69506...faf28a`。

注意：阶段 A 的“便宜探针有信号”并不等于完整模型有绝对增益——这正是阶段 C 要
检验的，而阶段 C 给出了否定。

## 3. 阶段 B（8 项验收门，全通过）

模型 = 冻结的 Small `LatentBridgeSEM108`（同 seed 参数逐位相等）+ 一个共享
类型化环编码器（338→64→48），其每环 latent 由**同一个** `D_L[48,96]` /
`V_L[96,48]` 任务字典解码；按环池化 `[ΣE, ΣE², 0]`（97 维）拼到冻结的 302 维
统一读出后进入**同一个** reader（399→13→13→1）。无第二条字典、无消息传递、
无注意力、无 raw-ring→reader 旁路、无节点/边写回。

| 门 | 结果 |
|---|---|
| G1 包含性 | 133002 参数；共享参数与同 seed Small 逐位相等；新 reader 列初值为 0，真实 batch plain/masked 预测差 ≤1e-5 |
| G2 不变性 | 重标号 / 环旋转反向 / 端点交换 / 图顺序 / 跨图拼批 / 视图置换：最大差 3e-8–5e-7；整数 ring 视图集合严格相等 |
| G3 反例 | 旧输入块（sem110、phi65、untyped cycles、edge counts）在两个完整图上**确实碰撞**；新 PyTorch 环汇总不碰撞（差 `0.003770`）；组成相同/排列不同的六环例子差 `0.000738` |
| G4 连接 | 每 forward 恰好 2 次共享 bridge 调用；改变 ring 对象不改变节点 `E` |
| G5 冷启动梯度 | step0 环 MLP 梯度 = 0（零初始化，正确），新 reader 列梯度 `0.3056`；1 次 Adam 后环 MLP 梯度 `3.55e-4`、共享 `D_L 0.0992` / `V_L 0.1043` 均非零 |
| G6 有限性 | 梯度/环输出/ISTA 有限；24 步 smoke 通过（final `1.0056`） |
| G7 `zero_ring` | 只归零 bridge 解码后的环 `E`/pool，不设共享 `zero_code`；节点 `E` 完全相同 |
| G8 test 阻断 | 官方 test 构造被阻断，无 test 缓存文件 |

计时门：`0.1063 s/step`，`8.399 s/epoch`，预测 `320·epoch·1.15 + 15 min =
3991 s < 4 h`，正式训练获授权。

## 4. 阶段 C（单 seed-0 完整 CPU 筛查）终点与推理诊断

**绝对结果**：soup valid `0.120779` → `TYPED_CYCLE_STOP`。

**对象是否活**（推理-only，冻结 soup 状态，未用于选模）：

- `zero_ring`：ΔMAE `+0.279311`（MAE `0.120779 → 0.400091`），预测 RMS `0.666285`
  ——去掉环对象会让预测明显变差，说明模型确实依赖它。
- 固定 seed 11/22 的**跨分子、同环长环对象置换**（保留每图环数量/长度分布与
  全局对象多重集，交换 2×1723=3446 个对象，占 2786 个环的 0.6184/侧）：
  ΔMAE `+0.215760` / `+0.208004`，预测 RMS `0.385486` / `0.372687`——把环对象
  换到别的分子上会显著改变预测，说明环内容（而不仅是环计数）被使用。
- MAE-only 任务梯度：环编码器 `0.338665`；共享 `D_L 0.240076` / `V_L 0.244196`。
- 环码密度：mean 非零 `91.18/96`（比例 `0.9498`）；节点码 `87.97/96`
  （`0.9164`）。两者都近稠密，因此这是**软阈值任务字典**，**不能**称为稀疏字典
  优势。

**判读**：机制层面，类型化环对象是被任务字典真正读出、真正承重的通道；但它没有
带来绝对性能提升。训练误差低（soup train `0.054857`）、通道活、消融损失大，
**都不能改判**分档——分档首先由绝对性能决定。

## 5. 背景（未匹配，仅供上下文，不是增量）

- Small latent-bridge soup `0.121058`（本轮 `-0.000279`）。
- Full m=3 soup `0.119154`（本轮 `+0.001625`）。
- Sem108 soup `0.123705`。

这些都是不同参数量/单 seed 的历史背景，不能作为因果增量。

## 6. 明确区分

- **本包已做的合成审计**：`typed_cycle_reference` 的 `validate_typed_cycle.py`
  与 `validate_cheap_probe.py`（vendored，all_gates_passed=true）——是合成/参考
  层审计，不是我们的真实结果。
- **我新跑的真实 train-only 性质检查**：阶段 A 的 10000 图 train-only OOF 廉价
  筛查（§2），使用真实仓库输入 builder。
- **真实 PyTorch 验收**：阶段 B 的 8 项验收门（§3），全部在真实模型、真实
  batch、真实训练路径上通过。
- **正式 valid 结果**：阶段 C 的 soup valid `0.120779`（§4）。

## 7. limitations

- 单 seed 0，无匹配对照；无法把“环分支的加入”与别的变化区分开，也不提供 seed
  方差估计。
- 推理干预只证明**依赖**，不证明**训练增益**；near-dense 码是软阈值字典而非稀疏
  优势。
- 未匹配背景（0.121058 / 0.119154 / 0.123705）不能作为增量。
- 官方 test 未读。

## 8. 后续决策

按预注册分档规则 `TYPED_CYCLE_STOP`：封档，不购买下一轮**同族**变体（不加宽环
编码器/reader、不加 seed、不做环长/hash/正则/threshold 调参救援）。
若未来要重开，必须走**新预注册**，且应先回答“为什么便宜探针的 OOF gain
（0.031）没有转化为完整模型的绝对增益”这一 gap——例如：是图级池化
`[ΣE, ΣE², count]` 丢失了对象身份，还是冻结 Small 主体 + 零初始化 reader 列的
优化路径让新通道只能起边际作用；这需要带匹配对照的新设计，而不是同族加宽。
