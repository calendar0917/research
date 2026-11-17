# 决策：`lcrc-explanation-fidelity-alignment-v1`

协议 `f9e283bdb38b7e790bd2bac4f37f38a2fcd05a80fa7144a476324be9892e6040`，提交 `fb561973ec15b1614e9ae086a86901e644772a36`。

## 状态

- `LOCAL_FAITHFUL_AND_RELEVANT`
- `NO_LCRC_SPECIFIC_ADVANTAGE`

## 支撑与反对

- B LCRC-LOCAL：oracle enrichment `0.0318` [0.0270, 0.0367]；fidelity advantage `0.0745` [0.0476, 0.1019]；oracle 支持度 `OK`，忠实性支持度 `OK`。
- C LCRC-MASK50：oracle enrichment `0.0218` [0.0169, 0.0268]；fidelity advantage `0.3303` [0.3002, 0.3594]；oracle 支持度 `OK`，忠实性支持度 `OK`。
- A LCRC-GRAD：oracle enrichment `0.0106` [0.0056, 0.0155]；fidelity advantage `0.2107` [0.1863, 0.2342]；oracle 支持度 `OK`，忠实性支持度 `OK`。
- E GINE-GRAD：oracle enrichment `0.0302` [0.0252, 0.0352]；fidelity advantage `0.2816` [0.2579, 0.3061]；oracle 支持度 `OK`，忠实性支持度 `OK`。
- 有限掩码 C−A 配对差 `0.0113` `[0.0061, 0.0168]`。
- 跨模型配对差 `-0.0004` `[-0.0070, 0.0062]`。
- 门禁全部通过：`True`。
- Test 复审：B 的 enrichment `0.0340`、忠实性 advantage `0.0948`，与 Valid 同号。

## 本轮**没有**主张的内容

- 没有主张 LCRC 比 GINE“更不忠实”：跨族门控语义不同，`|delta0|` 不可比。
- 没有主张 `m=0` 的删除效应等于 Shapley / 因果贡献；它只是一次完全门控移除。
- 没有主张 WC_LOGP 的原子加性分解就是化学机制；它只是冻结的生成式 oracle。
- 没有把 823 对辅助结果当独立验证：编辑区是结构差异，非 Train 端点只有 11 个分量。
- 没有事后更换 split、指标、阈值或候选范围。

## 什么会改变这个判断

1. 在**新的**预注册里，把选择目标改成尺寸归一的剂量/删除响应后，若 C 类方法在 k=6 内也明显超过 A（≥0.02）且同族忠实性同步上升，则应把 `FINITE_MASK_IMPROVES_ALIGNMENT` 升级为成立。
2. 若把候选池扩大到 k>6 或允许不连通子图后，B 的外部优势消失，则 `LOCAL_FAITHFUL_AND_RELEVANT` 需降级为池内结论。
3. 若在更多非 Train 端点上 B 的编辑区重叠优势反号，则辅助实验的正面描述不成立（当前非 Train 切片已是负号但 LOW_SUPPORT/CIs 跨 0）。

## 下一步

按本轮预注册条款，本轮结束后停止；不自动开始训练、不自动做架构搜索。若继续，起点是“同一候选池 + 尺寸归一的剂量/删除选择目标”，而不是继续优化 LCRC 的梯度/局部输出读出；新一步必须重新冻结协议。
