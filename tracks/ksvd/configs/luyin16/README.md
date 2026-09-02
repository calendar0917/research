# luyin16 配置

本目录保存 luyin16 阶段的可复现实验配置。每个配置必须明确：

- 数据集、任务和 split；
- `protocol_id` 与数据版本；
- data/model seed；
- 字典训练范围（必须是 train-only）；
- 结构、属性和融合分支；
- 输出目录与评估指标。

当前配置：

- `mentor_concept_v1.yaml`：完整开发协议，默认不看 test；
- `mentor_concept_v1_dev.yaml`：8000 图、K32 的中等规模开发运行；用于确认信号和成本，不作为最终协议；
- `mentor_concept_v1_k32_official_valid.yaml`：冻结 development 的 K32 设置，在完整 official train/valid 上复核；仍不看 test；
- `mentor_concept_v1_k32_topology_official_valid.yaml`：与 K32 official-valid 严格匹配，但字典 patch 只含邻接拓扑；
- `mentor_concept_v1_smoke.yaml`：300 图、K8 的管线烟测，不产生论文结论。
- `mentor_typed_reconstruction_proxy_v1.yaml`：独立定义的 624 维 typed-reconstruction proxy；不声称等同导师上游特征。
- `mentor_typed_slot_proxy_v2.yaml`：自然的 `8×64 + 28×4 = 624` typed-slot proxy；不声称等同导师上游特征。
- `mentor_typed_slot_proxy_v2_k64_dev.yaml`：同一 typed-slot object，但使用导师目录线索对应的 K64/T8/6-iteration 容量；仅作 development。
- `mentor_typed_slot_proxy_v2_k64_official_valid.yaml`：K64/T8 在完整 official train/valid 上的最终容量复核；仍不看 test。

本概念复现的 `S_v1/T_v1` 有独立 schema，不冒充导师未知的 69/624 维特征。

对应 runner 统一放在 [`experiments/luyin16/`](../../experiments/luyin16/)，可复用代码从 `ksvd_research` 导入；不要把新 luyin16 脚本继续添加到历史 `code/` 平铺目录。
