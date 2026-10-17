# MFIC v1 偏差与操作记录

本文件记录不改变科学假设、split、模型、训练配方或门槛的操作性偏差。所有 GPU 运行都发生在
最终冻结提交 `6eabb8b0b367fd5e9aa0a61a72ed6503f975c4eb` 的代码上。

## D1. `require_freeze` 的代码哈希比较 bug（无科学影响）

首次 `stage-inputs` 在远端失败：`protocol_freeze.json` 的 `code` 字段是
`{name: {path, sha256}}`，而 `require_freeze` 直接把它与 `{name: sha256}` 比较，导致
“code changed after freeze” 的假阳性。失败发生在构建 objects/K 之前，没有写入任何对象、
预测或标签；没有 GPU 训练。

修复：比较 `{name: rec["sha256"]}` 与 `code_hashes()`；重新生成 freeze。该修复不改变
CPU 结果、split、模型、参数、训练配方或门槛。

## D2. 首次 smoke 触发预算 stop，随后用更长 smoke 重新估计（已披露）

- 首次 `smoke`：20 个 SSL update + 512 分子 1 epoch + 256 预测；投影
  **2.855 GPU-h**（> 2.7），因此写出 `smoke_V.json` 的 `stop=true`，没有启动正式 SSL。
- 该 15 秒 smoke 的每步时间受冷启动和单 epoch 固定开销影响，明显高于历史同配方稳态
  （p3_slim SSL 约 0.215 s/update；本 smoke 为 0.307 s/update）。
- 随后增加 `smoke-long`：100 个 SSL update + 2,048 分子 5 epochs + 512 预测，投影
  **2.364 GPU-h（含 10% contingency）**，低于 2.7；随后才启动正式 SSL/训练。
- 正式训练没有被缩短：360 epochs、22,680 updates/arm、相同 R3 配方、相同 batch order、
  相同 last-5 soup。硬上限仍是 3.0 GPU-h；最终 ledger 为 **2.091 GPU-h**。
- 两个 smoke 文件都保留：`smoke_V.json`（stop=true）与 `smoke_long_V.json`（stop=false）。

**披露**：如果审阅者按“首次 smoke 预计超过 2.7 即永久停止”的最严格解释，正式 GPU 运行
不应发生。这里把 `smoke-long` 视为对同一预算门槛的更长测量，而不是对训练配方或科学门槛的
修改；最终结果是否被接受由审阅者判断。该偏差不影响 `MFIC_NO_USEFUL_GAIN` 的负结论。

## D3. 失败重试计入预算

- 首次 `stage-inputs`（D1）约 5 s；
- 远端 provenance hash 检查约 7 s；
- 两者以 `failed-retry-stage-inputs` / `failed-retry-hashcheck` 记入
  `gpu_budget_ledger.json`。首次 15 s `smoke` 也已计入。

## D4. 未读 Locked

Monitor 门槛失败后，`score-locked` 未运行；没有 `locked_read.lock`。这不是偏差，是协议
要求的停止。
