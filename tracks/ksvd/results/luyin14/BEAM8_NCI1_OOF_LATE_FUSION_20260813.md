# Beam8/NCI1 strict OOF late fusion

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_NCI1_OOF_LATE_FUSION_PROTOCOL_20260813.md`  
> 判定：`NCI1_BEAM8_OOF_FUSION_BELOW_GATE`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| FEATURE_STATS | 0.7017 ± 0.0043 | 0.7017 |
| INIT_BAG | 0.6779 ± 0.0080 | 0.6779 |
| INIT_TRUE | 0.6801 ± 0.0052 | 0.6800 |
| FUSION_INIT_BAG | 0.7025 ± 0.0049 | 0.7024 |
| FUSION_INIT_TRUE | 0.7027 ± 0.0050 | 0.7027 |
| FUSION_INIT_SHUFFLED | 0.7027 ± 0.0050 | 0.7027 |
| FUSION_FINAL_TRUE | 0.7027 ± 0.0050 | 0.7027 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| fusion_increment | +0.0010 | 2/1/0 |
| relation_specific | +0.0000 | 0/3/0 |
| chain_specific | +0.0002 | 1/2/0 |
| ksvd_update | +0.0000 | 0/3/0 |

## Frozen checks

- fusion_increment：`False`；
- relation_specific_fusion：`False`；
- chain_specific_fusion：`False`；
- ksvd_update：`False`；

## Boundary

- meta head 只读取 outer-train inner-OOF base logits；outer-test 不参与权重选择。
- 本轮失败时，NCI1 上停止扩大 Beam8 分类融合模型。
