# configs/

数据集列表、超参、suite 配置。保持可复现、可 diff。

Python 依赖以仓库根目录 `pyproject.toml` 和 `uv.lock` 为唯一来源：

```bash
uv sync --frozen
```

`requirements-molhiv.txt` 仅为旧命令兼容提示，不再单独维护版本。
