# `code/`：历史复现区

这里保留 KSVD 轨道已有的约 400 个脚本，作为历史实验、审计和数字复现材料。平铺结构不适合作为新的长期入口，因此新实验不应继续直接向这里增加文件；本次整理没有移动、删除或批量改写这些历史实现。

## 新代码从哪里开始

可复用逻辑从稳定包导入：

```bash
uv sync
uv run python -c "from ksvd_research.core import Graph, ksvd; print(Graph, ksvd)"
```

主要边界如下：

| 路径 | 职责 |
|------|------|
| `src/ksvd_research/core/` | 图对象和 KSVD/稀疏编码核心 |
| `src/ksvd_research/data/` | OGB-MolHIV、TUDataset 适配器 |
| `src/ksvd_research/sampling/` | B0/RW、coverage 和 patch cover |
| `src/ksvd_research/features/` | patch 向量化和 canonical slot |
| `src/ksvd_research/evaluation/` | 图级编码和过程指标 |
| `experiments/luyin16/` | luyin16 阶段唯一新增实验入口 |
| `tests/` | pytest 风格的维护测试 |

当前 `src/ksvd_research` 中的核心模块是显式兼容层，指向已验证的 `code/` 实现；这样可以先稳定导入路径，再按依赖和测试逐步迁移实现。历史脚本仍按旧方式运行，例如：

```bash
uv run python -m tracks.ksvd.code.test_canonical_slots
```

旧脚本不是维护测试套件。维护测试运行：

```bash
uv run pytest tracks/ksvd/tests
```

新增实验请放到相应的 `experiments/<stage>/`，配置放 `configs/<stage>/`，结果放 `results/<stage>/`；不要把尚未协议锁定的 runner 重新放回 `code/` 平铺目录。
