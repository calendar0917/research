# Beam8/Mutagenicity typed relabel-invariance audit

> 日期：2026-08-13  
> 状态：typed classification 后发现潜在 binary-slot/typed-payload 不一致而注册。

## 1. 范围

确定性选择前 512 张 Mutagenicity 图（不读取 graph labels），每图做 3 个固定随机节点重编号。
对 BASE `s8/o2 Beam8/R1/m1.5` 检查：

- relabeled patch node sets 映射回原 ID 后的 ordered chain match；
- typed edge one-hot + node histogram patch token 的 ordered row match；
- RAW compact TRUE graph representation exact match；
- patch-token multiset match，用于区分 chain 不稳与 token canonicalization 不稳。

## 2. Gate

四项 match rate 必须全部为 1.0。任何一项失败，现有 typed classification 只作为实现诊断，
不得作为 Beam8 typed 方法结论；必须先采用 node/bond-attributed stable preorder 与 typed
canonical slots，再重新做无标签 gate 和分类。

