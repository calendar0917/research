# 带类型环对象：参考验证与 Agent 交接

这是 source-derived NumPy 参考包与新实验设计，不是仓库的可直接启动 PyTorch 实验。没有 ZINC 数据或 checkpoint 随包发布。

审计仓库 revision：`7fa46ab300b43c6457499e45408296c5a4bd4c5c`。

## 本地参考验证

```bash
python validate_typed_cycle.py
python validate_cheap_probe.py
```

依赖 NumPy / SciPy。输出 `typed_cycle_audit.json`、`cheap_probe_acceptance.json`。前者验证对象区分能力、不变性、旧函数包含及合成梯度；后者验证廉价检查器的正/零信号、分组隔离和特征不变性。它们不是 ZINC MAE 结果。

## 真实 train-only 检查

先由仓库 Agent 提取官方训练集 10k 图的 label-free 原始特征，保存为 NPZ，然后运行：

```bash
python cheap_property_probe.py train_ring_probe.npz --output train_ring_probe_report.json
```

NPZ 字段：`X_base`、`X_typed`、`y`、`group_ids`、`split="train"`、`official_test_loaded=false`、`label_free_features=true`。具体基底特征和预算见 prompt。CLI 拒绝其他 split；声明字段仍必须由提取脚本的 provenance 验证，不能仅靠填写布尔值证明无泄漏。

## 文件说明

- `typed_cycle_reference.py`：无弦环枚举、二面体视图、共享环编码器、反例。
- `latent_dictionary_bridge_reference.py`：现有任务字典的 NumPy 参考。
- `existing_spine_reference.py`：此前宽度审计用的骨干参考，覆盖环境输入之后的路径；不是完整 PyTorch 源码。
- `cheap_probe_features.py`：廉价检查使用的有界静态图特征。
- `cheap_property_probe.py`：train-only 分组嵌套 ridge 检查。
- `validate_typed_cycle.py`、`validate_cheap_probe.py`：参考验收。
- `zinc_typed_cycle_plan.md`：决策依据和边界。
- `zinc_typed_cycle_agent_prompt.md`：可直接交给执行 Agent 的任务。
- `source_manifest.json`：读过的仓库入口与审计 scope。
- `SHA256SUMS`：交接文件校验。

禁止把参考 audit 的数值标为真实 PyTorch 或 ZINC 性能验收。正式实现必须在真实 train batch 上重新通过 prompt 中的门禁。
