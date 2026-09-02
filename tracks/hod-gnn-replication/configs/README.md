# configs/

数据集列表、超参、suite 配置。保持可复现、可 diff。

- `method_dataset_matrix.yaml` — 方法 × 数据集 实验矩阵
- `seed_registry.yaml` — 各方法/数据集的 seed 来源与取值
- `dataset_registry.yaml` — 数据集元数据（来源、hash、split 信息）
- `environment-lock.yml` — 锁定 conda 环境