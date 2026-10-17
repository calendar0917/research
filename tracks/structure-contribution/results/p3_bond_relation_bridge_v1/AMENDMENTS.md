# cscl-p3-bond-relation-bridge-v1 — 预执行修正记录

所有修正都发生在**任何正式 GPU 训练、任何 Monitor/Locked 标签读取之前**；不改变
协议语义、不改变模型/核/配方/判定规则。

## amendment-1 — 计算 checkout 上的机器本地输入重建（2026-10-17）

**症状**：`p3-bond-relation-bridge-v1` 首次 GPU smoke 在远端立即失败：
`FileNotFoundError: .../objects_fit_F.pt`。

**原因**：`objects_*.pt` / `pairs_*.pt` 被 `.gitignore` 排除（`tracks/*/results/**/*.pt`），
是本轮在**本机**由官方 PyG 图确定性生成的、无标签的中间产物；远端 compute checkout
只有提交进 Git 的 JSON 证据，没有这些二进制文件。上一轮在远端重跑 `objects`/`kernels`
阶段，但本轮这两个阶段被 `require(not freeze_path().exists(), ...)` 保护（freeze 已存在）。

**修正**：新增 `stage-inputs` 子命令，仅做：
1. 依据**冻结的** `split_F.json` 行号，用 `LabelGuard` 重建 `objects_{fit,monitor,locked}_F.pt`；
2. 断言行序完全一致、`guard.access_attempts == []`（不读任何标签）；
3. 用 Fit-only 重建 standardiser 并与冻结的 `prep_fit_F.json` 逐值比较（要求 ≤1e-12；
   本机实测 0.0）；
4. 重建 R/U pair list 并与冻结的 `pairs_manifest_F.json` 的**每区 pair 总数**逐一比较
   （fit R=276,825 / U=2,495,586；monitor 34,550 / 309,187；locked 34,855 / 315,990）；
5. 写 `staged_inputs_F.json` 记录远端文件哈希与校验结果。

**为什么不是“改协议”**：核、模型、参数、训练配方、数据划分、评分与判定规则一行未改；
改的只是“把已经被冻结哈希绑定的确定性中间产物放到计算节点上”的方式。`torch.save`
的字节在不同 checkout 上不保证逐字节相同，因此校验用**语义等价**（行序、standardiser、
每区 pair 总数），而不是文件字节哈希。

**重新冻结**：`protocol_freeze.json` 记为 `amendment: 1`，旧 freeze 归档为
`protocol_freeze_amendment1.json`。此后仍不得在正式训练开始后修改代码。
