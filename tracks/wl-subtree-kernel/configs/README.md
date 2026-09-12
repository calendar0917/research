# configs/

本轨无独立配置文件：全部参数走 CLI（`code/run_wl_subtree_kernel.py --help`），
等价于「run 参数即配置」，避免双份真相。

固定默认值（与原版一致）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--datasets` | REDDIT-BINARY COLLAB | TUDataset 名 |
| `--seeds` | 0 42 123 1024 2026 777 3407 999 111 888 | CV 种子 |
| `--n-splits` | 10 | 折数 |
| `--n-repeats` | 10 | 重复数 |
| `--wl-iter` | 3 | WL 迭代 |
| `--wl-C` | 10.0 | SVC C |
| `--data-root` | `<repo>/data/TUD` | 数据集根 |
