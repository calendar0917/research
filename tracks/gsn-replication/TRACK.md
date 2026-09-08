# Track: GSN 图分类复现（TG 官方仓，social 数据集）

| 项 | 内容 |
|----|------|
| slug | `gsn-replication` |
| 状态 | `active` |
| 创建 | 2026-09-08 |
| 主线位置 | Kernel → GNN → GSN 谱系：官方 GSN 在 TU social 数据集的严格复现 |

## 问题（一句话）

用 **Bouritsas 等官方实现**（graph-substructure-networks，原样不动）在 IMDB-BINARY /
IMDB-MULTI / COLLAB 上复现 GSN 论文 Table 1 的 GSN-e / GSN-v 数字，并（论文外）补
REDDIT-BINARY 同协议运行。

## 范围

- 做：官方仓 `6cce24a` + 论文超参（supplementary Table 5 + README IMDBBINARY 命令）；
  4 个数据集 × {GSN-e, GSN-v}；10 折 CV（powerful-gnns 的 `10fold_idx` 划分）；
  每配置一次运行（seed 0），记录 mean±std；
  服务器 conda 环境（graph-tool 官方计数路径）；本地 CPU 冒烟（shim 仅验证兼容性）。
- 不做：模型调参；GIN/其他基线（对照见 wl-subtree-kernel 轨，需要时单独重跑）；
  OGB/ZINC/SR 实验；官方代码改动。

## 协议 / 评估

| protocol_id | 角色 |
|-------------|------|
| **`gsn-strict-social-v1`** | **主协议（用户要求）**：10 seeds × 10×10 CV；每折 val 选 epoch，test 无泄漏 |
| `gsn-official-social-v1` | 旁路参考：官方乐观协议（README/Table 5 复现），数字仅作对照 |

- 严格协议指标语义：每 (seed, repeat, fold) 报告 **val-best epoch 的 test acc**；
  汇总 fold-level（千折 mean±std）与 seed-level（跨种子 mean±std）。
- 划分：`RepeatedStratifiedKFold(10, 10, random_state=seed)`（与 wl-subtree-kernel 轨同 API/同序，
  **同种子⇒同划分**，可横比）。注意 wl 轨种子列表为 0,42,…（本轨按用户给定 0,41,…，横比时需统一）。
- 超参：固定论文 Table 5（方法给定实现；严格化的是“评估/模型选择”而非重做超参搜索）。

## 进度

| 阶段 | 状态 |
|------|------|
| 来源登记与数据校验 | 完成（官方仓 HEAD+论文源+powerful-gnns 均已核对） |
| 配置导出（Table 5 → CLI 参数） | 完成 |
| strict runner（run_strict.py） | 完成；本地冒烟通过（val 选 epoch 路径已验） |
| 本地 CPU 冒烟（shim 验证管线） | 完成 |
| 服务器正式实验（strict 主表） | 待跑（需 GPU 服务器 + conda graph-tool；成本见 docs/README） |
| 汇总与交付 | 待开始 |

## 入口

| 路径 | 用途 |
|------|------|
| `code/run_strict.py` | **主入口**：10 seeds × 10×10 CV 严格协议（val 选 epoch） |
| `code/run_official.py` | 旁路参考：官方乐观协议单次复原 |
| `code/run_smoke_local.py` | 本地 CPU 冒烟（含 graph_tool/wandb shim） |
| `code/setup_data.py` | 数据布局准备/校验 |
| `configs/social_paper.json` | Table 5 的 6 个论文配置 + REDDIT-BINARY 扩展配置 |
| `env/` | 服务器 conda 环境（yaml + 命令） |
| `docs/README.md` | 人读操作手册（含预期数字） |
| `notes/README.md` | 来源/假设/偏差记录 |
| `code/vendor/` | 官方仓 + powerful-gnns（git-ignored，见 notes 登记 commit） |

## 关键事实（写论文/汇报前必读）

- **严格协议与官方协议的唯一区别在“epoch 选择”**：官方 main.py 无 val 时用 `argmax(test_accs)`
  选 epoch（测试泄漏，乐观）；本轨严格协议每折写 `val_idx` 文件 → 官方代码自动改用 val 选
  （main.py:333/398），test 只取 val-best 时点。超参不动。
- **GSN 主论文（arXiv:2006.09252v3）没有 REDDIT-BINARY**：Table 1 只含 MUTAG / PTC /
  Proteins / NCI1 / Collab / IMDB-B / IMDB-M。REDDIT-BINARY 结果只能标"论文外扩展"。
- 论文数字（Table 1）：Collab GSN-e **85.5±1.2** / GSN-v **82.7±1.5**；
  IMDB-B GSN-e **77.8±3.3** / GSN-v **76.8±2.0**；IMDB-M GSN-e **54.3±3.3** /
  GSN-v **52.6±3.6**。
- 官方 README 明示：不同 torch/CUDA 版本会影响种子级复现（差异通常很小）。

## 禁止

- 改动 vendor 官方代码（复现的意义在于原样）
- 与其它轨结果混表横比（参考对照需单开 deliverable）
- 把 REDDIT-BINARY 结果写进"复现论文数字"类表述

## 下一步

1. 本地跑 `run_smoke_local.py` 验证管线兼容性（当前机器无 GPU/conda，用 shim）
2. 用户将 repo（含 vendor）拷贝到 GPU 服务器 → `env/setup_server.sh` 建 conda 环境
3. `code/run_official.py --config ...` 逐配置运行（先 IMDBBINARY GSN-e 验证 77.8±3.3）
