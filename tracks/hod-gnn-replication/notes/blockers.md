# blockers.md — 实验阻塞项记录

## 当前阻塞项

| # | 方法 | 数据集 | 问题 | 状态 | 解决思路 |
|---|------|--------|------|------|----------|
| 1 | GPS | MOLTOX21 | GraphGPS 官方仓库无 ogbg-moltox21 配置 | open | 从 HyMN 代码库获取配置 |
| 2 | Full/Random/Policy-Learn | Peptides | policy-learn 官方仓库无 Peptides 配置 | open | 从 HyMN 代码库获取配置 |
| 3 | 所有 | 所有 | 本机无 NVIDIA GPU（仅 AMD/ROCm，VRAM 4GB）；官方基线均为 CUDA 锁定 | open | 需 CUDA GPU 机器 |
| 4 | 所有 | 所有 | 本机 Python 3.14 过新，官方代码要求 PyTorch 1.12/1.13 + PyG 2.2 | 缓解 | 已建 .venv (py3.11 + torch 2.13 CPU + pyg 2.8) 用于数据/split 审计；真正训练仍需官方 env |
| 5 | 所有 | 所有 | 论文 seed 未披露 | 缓解 | 使用 [0,1,2,3] 临时替代，标注 provisional |
| 6 | 所有(各基线) | ZINC-12K | ZINC 数据源 deepchemdata S3 bucket 返回 403 / SSL 失败，无法从本机下载 | open | 需可用镜像或.mirror URL，或从有缓存的机器拷贝 |
| 7 | 所有 | 所有 | 官方训练入口需 CUDA 编译扩展 | 缓解 | GraphGPS 已确认 CPU 可跑（torch 1.13 CPU + pyg CPU wheels）；policy-learn 的 csrc 编译仍需验证；GraphViT 待验证 |
| 8 | GPS | MOLHIV | 完整训练 100 epochs ≈ 6.4h/seed (CPU) | 进行中 | 4 seeds 串行队列 (run_gps_molhiv_seeds.sh)，seed 0 运行中 |

## 已解决阻塞项

| # | 方法 | 数据集 | 问题 | 状态 | 解决方式 |
|---|------|--------|------|------|----------|
| 8 | 审计 wrapper | 所有 OGB | PyTorch≥2.6 `torch.load` weights_only 破坏 ogb .pt | resolved | PL-001 patch (`weights_only=False`) |