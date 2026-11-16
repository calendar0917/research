# RUN_NOTES — gnn-route-clarification-8h-v1

执行记录：工程故障、修复与重跑。协议（`configs/frozen.yaml` / `protocol_freeze.json` /
`analysis/pairs_summary.json` 里的 pair catalog 与样本）没有因为任何故障而改动。

## 时间线（UTC）

| 事件 | 时间 |
|---|---|
| t0（任务书开始） | 2026-11-16T17:06:46Z |
| 12 格 stage-1 完成 | 2026-11-16T18:16Z |
| stage2_A / stage3_D / oracle 完成 | 2026-11-16T20:01Z |
| 本地账目 + oracle + 报告完成 | 2026-11-16T20:45Z |
| 软截止 7h / 硬截止 8h | 2026-11-17T00:06Z / 01:06Z |

实际 GPU 时间 2.68 卡·小时（按单元 `seconds_total` 求和），墙钟 2.9h；低于
14 GPUh 软预算。等待主要来自屏障与下面三次重启，不是额外训练。

## 故障与修复（都只改工程实现，不改模型/损失/评估协议）

1. **stage-3 冻结表示缓存用错行号**（`IndexError: index 6524 is out of bounds ... size 902`）
   - 原因：`cache_representation` 存的是**节点级** `h`，stage-3 却把图 id 当行 id。
   - 修复：缓存增加 `atom_ptr`，按图 gather 自己的节点行；评估循环用 `len(row_ids)`。
   - commit `692d0bdb`。

2. **stage-3 teacher 头未加载**（同一 commit）
   - 原因：只把 T checkpoint 的 `trunk.*` 载入，`head` 仍是随机初始化；teacher 因此是
     随机头的输出，D 的 fidelity 失去意义。
   - 修复：`model.load_state_dict(state["model_state"], strict=True)` 载入完整 T。
   - 影响：修复前的 D 结果全部作废并重跑（`stage3_D` 两格在修复后重算；旧目录被
     新签名覆盖）。

3. **resume 签名不匹配导致队列崩溃**（`resume signature mismatch; refusing to mix trajectories`）
   - 原因：旧 worker 在代码修订后留下 pre-fix 的 `resume_state.pt`，新代码把签名不一致
     当致命错误。
   - 修复：签名不一致时丢弃该单元的旧 partial state 并从头重跑（打印 `[resume] ...
     discarding the stale partial state`），不再杀整个队列。
   - 影响：`stage2_A:ogbg-mollipo:A:R3` 重训一次（667.8s），其余单元 skip。

4. **oracle `TargetOverride` 缺 `row_ids_sorted`**（`AttributeError`）
   - 修复：代理方法转发到 base bundle。commit `36aa5eba` 之后。

5. **WC_LOGP 与 MOLWT oracle 共用输出目录**
   - 原因：`results/oracle/<dataset>/<model>/selected` 不含 property，两个 worker 同时写
     L 时互相覆盖 tmp/checkpoint（`FileNotFoundError ... final_predictions.npz.tmp`）。
   - 修复：oracle 输出改为 `results/oracle/<dataset>/<model>/<property>/selected`，
     分析端同步读取。commit `f6c3099b`。
   - 影响：清理了被污染的 `results/oracle`，6 个 oracle 单元全部重跑；这是本轮唯一
     一次**数据产物**层面的返工。

## 重跑清单

| 单元 | 原因 | 结果 |
|---|---|---|
| stage3_D:ogbg-molesol:D | teacher 头 bug | 重跑，最终 Val 1.0568 / fidelity 0.2244 |
| stage3_D:ogbg-mollipo:D | teacher 头 bug | 重跑，最终 Val 0.7059 / fidelity 0.2438 |
| stage2_A:ogbg-mollipo:A:R3 | resume 签名 | 重跑，Val 0.7665 |
| oracle:WC_LOGP:{L,T,A} | 目录冲突 | 全部重跑并落入独立目录 |
| oracle:MOLWT:{L,T,A} | 目录冲突 | 全部重跑并落入独立目录 |

没有因为故障而改变 seed、split、epoch 数、recipe 集合、门限或解释样本。

## 解释/账目执行注记

- `gr_ledgers --job all`：L/T/A/D 的 96 对账目；`--job oracle`：L/T/A × WC_LOGP/MOLWT。
- 所有账目在 FP64 下写 `state_unchanged=True`；端点复现 ≤2.3e-8（A）与 ≤6.7e-16（T）。
- IG 求积阶梯固定 32→64→128→256；未过 pair 保留并计入 `n_unresolved`，没有用通过
  子集宣布优胜。真实 Lipo T 账目 22/96 未闭合，oracle WC_LOGP 6/96，MOLWT 0/96。
- `mmp_explanation` 的 variable-vs-Δy 只是局部效应一致性代理，不是原子真值。
- MolHIV 未启动 → 报告中明确 `BUDGET_NOT_RUN`；multi-mapping 如实记 `NOT_RUN`。
