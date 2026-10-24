# GOS-v1 工作假设与候选表（启动前冻结）

定向问题：在 ≤500k 参数、单模型、raw-y MAE 的既有载体上，**把显式全图结构关系
（键—键路径关联）与显式键对象内容放进一次全局混合**，能否超过 DCC-L 的
valid 0.07599 / test 0.06732？

## 载体（一次前向，一个 reader）

- 复用 DCC-L 的 P3 encoders + global composer + local composer（296,560 参数，
  实际计数核对通过）；不用原图/线图邻接递归消息传递。
- 每条**无向物理键**一个全局 token：`h0_b = Linear([c_global_b, c_local_b])` 192→48。
- 原始 513 维 `z0`（幅值/数量统计）**不经图级归一化**直接进 reader。
- 静态关系 `r_bc`：12 个 `log1p` lifted 路径关联统计 + 最小端点距离 + 共享端点数
  + 自关系 + 不可达 = 16 维，共享 `16→48→24`。
  lifted 定义 `(S_l(u,p)+S_l(u,q)+S_l(v,p)+S_l(v,q))/2 = M S_l M^T`，
  是键—键的结构关联统计，不是原图键—键简单路径精确计数（有先例：DBT 全图
  attention + atom-pair 路径计数、CKGConv 连续核、PPGT AdaRMSN/静态关系）。
- 3 层、宽 48、4 头、FFN ×2；`AdaRMSNorm(x)=x/rms(x)*rms(alpha*x+beta)`（借 PPGT）。
- 读出 `[z0, Σh/mean_train_n_bonds, Σh²/mean_train_n_bonds]`（609 维）→128→128→1。

## 候选

| 代号 | 混合运算 | 参数（实测） | 购买理由 | 已知抗辩 |
|---|---|---:|---|---|
| C0 | 无（同前端 + 513→128→128→1） | 378,993 | 判断“压缩 reader”本身的影响 | 不是性能候选，只是载体锚 |
| A0 | `softmax_c((q·k−0.5‖k‖²)/√d + β(r))`，`g=1+tanh(γ(r))` | 460,017 | 幅值敏感 + 静态关系统计同时进入内容交互 | DBT 已有全图 attention 却没赢 DCC；只有整体增益才算数 |
| K0 | `Σ_c tanh(K(r_bc))·V(N(h_c))/mean_n_bonds` | 452,457 | 有符号、非竞争累计是否更适合显式统计 | CKGConv 已有连续核先例；符号本身不新 |
| P0（条件性） | 端点绑定的真实路径内容 `Σ_p φ(atom,bond,…)` | — | 仅当 A/K 有竞争力时才做 | 旧 compact-v4 有序路径（先逐位置平均）为负结果 |

## 决策规则（写在此处，事后不改）

1. 与 C0 同 epoch 曲线比较；DCC-L 四 seed valid 0.07599 是绝对锚。
2. 若某路线在 ≥3 个窗口持续优于 C0 ≥0.003，且 >0.01 优于同阶段 C0 的情况收窄，
   则买第二 seed 确认；否则停。
3. 只在“valid 有意义、持续、且第二 seed 支持”时读 test（reporting-only，一次）。
4. 目标 test <0.060 是**待检验**目标，不用 valid-test 固定偏移外推。

## 预算

墙钟 7h 目标 / 8h 硬；GPU 16 A100·h 硬、14 软。C0/A0 先并行，K0 随后；
赢家第二 seed；剩余只做一次机制对照（A：普通规范化注意力；K：行 softmax 正权核）。
