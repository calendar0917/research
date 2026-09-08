# configs

- `social_paper.json`：4 数据集 × {GSN-e, GSN-v} 共 8 个配置。
  - 论文 6 个（Table 5 逐项 + 附录 C.2 公共项 + README IMDBBINARY 命令），`in_paper: true`；
  - REDDIT-BINARY 2 个为论文外扩展（`--split random`，无 10fold_idx），`in_paper: false`。
- 键格式：`{DATASET}--{gsn-e|gsn-v}`；`_base_cli` 为公共参数；`_decay_steps/_lr` 为数据集级。
- 每个配置的 CLI 由 `code/run_official.py::build_cli` 组装，`--dry-run` 可查看。
- 修改配置后：config 内容 sha256 会进结果 JSON（config_sha256），便于审计。
