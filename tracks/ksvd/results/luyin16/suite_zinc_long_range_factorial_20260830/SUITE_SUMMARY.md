# luyin16 ZINC 长程与归因 suite

本报告由 suite 的最后一个任务自动生成。所有 radius 任务使用同一套冻结 patch 对象，
字典只在对应 official train patches 上学习；test 仅对晋级视图做 train+valid refit。

## 数据源

- 状态：`available`；split sizes：`{'test': 1000, 'train': 10000, 'val': 1000}`
- policy：`official_pyg_zinc_only`
- dataset URL：`https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1`
- split URL：`https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/{}.index`
- raw 文件数：`8`（详见 preflight JSON）

## Radius 诊断

| run | status | train truncation | train pair coverage | best validation views |
|---|---|---:|---:|---|
| `zinc_radius1` | `full` | 0.0000 | 0.2391 | `global_all_plus_local_typed_raw_plus_typed_ksvd_final` (0.5436), `global_all_plus_typed_ksvd_final` (0.5455), `global_all_plus_typed_ksvd_init` (0.5512), `global_all_plus_local_topology_attributes_late` (0.5717), `global_all_plus_local_typed_raw` (0.5732) |
| `zinc_radius2` | `full` | 0.0001 | 0.4952 | `global_all_plus_local_typed_raw_plus_typed_ksvd_final` (0.5430), `global_all_plus_typed_ksvd_final` (0.5465), `global_all_plus_typed_ksvd_init` (0.5494), `global_all_plus_local_typed_raw` (0.5794), `global_all_plus_local_topology_attributes_late` (0.5847) |
| `zinc_radius3` | `full` | 0.0615 | 0.6787 | `global_all_plus_local_typed_raw_plus_typed_ksvd_final` (0.5225), `global_all_plus_topology_ksvd_init` (0.5261), `global_all_plus_typed_ksvd_final` (0.5357), `global_all_plus_topology_ksvd_final` (0.5380), `global_all_plus_typed_ksvd_init` (0.5432) |
| `zinc_radius3_wide_development` | `development` | 0.0000 | 0.6998 | `global_all_plus_typed_ksvd_final` (0.7425), `global_all_plus_local_typed_raw_plus_typed_ksvd_final` (0.7630), `global_all_plus_local_topology_attributes_late` (0.7647), `global_attributes_plus_local_attributes_raw` (0.7667), `global_all_plus_topology_ksvd_final` (0.7691) |
| `zinc_radius3_wide_full` | `full` | 0.0000 | 0.6987 | `global_all_plus_local_typed_raw_plus_typed_ksvd_final` (0.5256), `global_all_plus_topology_ksvd_init` (0.5261), `global_all_plus_topology_ksvd_final` (0.5337), `global_all_plus_typed_ksvd_final` (0.5419), `global_all_plus_typed_ksvd_init` (0.5437) |

## 固定预算视图对照

| run | global_all | local typed raw | typed KSVD init | typed KSVD final | global + typed final |
|---|---:|---:|---:|---:|---:|
| `zinc_radius1` | 0.6122 | 0.5920 | 0.6423 | 0.5867 | 0.5455 |
| `zinc_radius2` | 0.6122 | 0.6032 | 0.8162 | 0.8091 | 0.5465 |
| `zinc_radius3` | 0.6122 | 0.5836 | 1.0330 | 1.0613 | 0.5357 |
| `zinc_radius3_wide_development` | 0.8249 | 0.7953 | 1.2691 | 1.2522 | 0.7425 |
| `zinc_radius3_wide_full` | 0.6122 | 0.5815 | 1.0238 | 1.0602 | 0.5419 |

## 解释规则

- radius-1/2/3 固定 `max_nodes=12`，因此可直接比较感受野；wide-development 不与 full 数字混表。
- radius-3-wide-full 与 radius-3-full 使用同一 official split 和训练预算，仅改变 `max_nodes: 12→20`，用于归因截断。
- `local_*_raw` 检查统计对象；`*_ksvd_init` 与 `*_ksvd_final` 在相同初始化下检查 K-SVD 更新归因。
- `global_structure` 含全局距离/直径等长程统计；`global_attributes` 与局部 attributes 用于属性—结构解耦。
- screen-only 视图只做一次固定参数 validation 快筛；晋级视图才做多 seed 和 test。
