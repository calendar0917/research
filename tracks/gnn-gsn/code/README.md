# code/

本轨无自有实现（历史 `common.py` / `run_*.py` / `official_gsn/` 随旧仓 `../../paper` 遗失，见 TRACK.md）。

现行权威实现 → `tracks/gsn-replication/code/`：

- `run_strict.py` — 严格协议主入口（10 seeds × 10×10 CV，val 选 epoch）
- `run_official.py` — 官方乐观协议旁路
- `setup_data.py` — 数据布局/计数缓存/迁移
- `compat/` — 无 GPU/conda 机冒烟用 shim
- `vendor/` — 官方仓 graph-substructure-networks（git-ignored，服务器打包传入）
