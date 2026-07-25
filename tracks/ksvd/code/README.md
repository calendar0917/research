# code/

| 模块 | 作用 |
|------|------|
| `graph.py` | 无向图、诱导子图、合成 ring+chords |
| `sample.py` | B0 一阶 / B1 均匀 RW / M0 偏置+降权 |
| `metrics.py` | edge_cover, edge_repeat, \|S\|, … |
| `vectorize.py` | 诱导邻接 pad→m（阶段1用） |
| `run_stage0.py` | 阶段0入口 |

```bash
cd tracks/ksvd
python -m code.run_stage0 --config configs/stage0.yaml
```

依赖：Python 3.10+ 标准库。有 PyYAML 则完整解析 yaml，否则用内置默认配置。
