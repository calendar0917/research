# HOD-GNN 基线复现审计 — 来源登记

> 日期：2026-08-24  
> 任务来源：docs/luyin/luyin15.txt  
> 目标论文：https://arxiv.org/html/2510.02565v1

## 论文元数据

| 项 | 内容 |
|----|------|
| 标题 | On The Expressive Power of GNN Derivatives |
| 作者 | Yam Eitan, Moshe Eliasof, Yoav Gelberg, Fabrizio Frasca, Guy Bar-Shalom, Haggai Maron |
| 出处 | arXiv:2510.02565v1, Oct 2025 |
| 论文 seed 说明 | "averaged over four runs with different random seeds" — 未披露具体数值 |
| 基线代码基础 | "based on code provided in Southern et al. 2025 and Rampášek et al. 2022" |

## 方法 1: GPS (GraphGPS)

| 项 | 内容 |
|----|------|
| 论文名 | Recipe for a General, Powerful, Scalable Graph Transformer (NeurIPS 2022) |
| 官方仓库 | https://github.com/rampasek/GraphGPS |
| 最新 commit | `28015707cbab7f8ad72bed0ee872d068ea59c94b` (main, 2023-02-17) |
| Tag | `v1.0.0` (`e38969bcca46f850fe08f2062991f9ecd5c9711e`), `v1.2.0` (`40cfeed9b224d0e15cd73dafd4de512cdc965e12`) |
| 依赖 | PyTorch 1.13, PyG 2.2, GraphGym (PyG2), pytorch-lightning, yacs, ogb, wandb |
| 环境 | conda create -n graphgps python=3.10 |
| 配置入口 | `configs/GPS/` — zinc-GPS+RWSE.yaml, ogbg-molhiv-GPS+RWSE.yaml, peptides-func-GPS.yaml, peptides-struct-GPS.yaml |
| 注意 | 仓库没有 ogbg-moltox21 和 ogbg-molbace 的 GPS 配置。HOD-GNN 论文中 GPS 的 moltox21 结果 (75.70) 可能来自 Southern et al. 2025 (HyMN) 代码库 |
| seed 来源 | `run/run_experiments.sh` 使用 `for SEED in {0..9}`; 官方 seed 口径为 0–9 |
| 论文表格目标 | ZINC: 0.070±0.004, MOLTOX21: 75.70±0.40, MOLHIV: 78.80±1.01 |

## 方法 2: GraphViT

| 项 | 内容 |
|----|------|
| 论文名 | A Generalization of ViT/MLP-Mixer to Graphs (ICML 2023) |
| 官方仓库 | https://github.com/XiaoxinHe/Graph-ViT-MLPMixer |
| 最新 commit | `0d66dd1b0d8376e9252a73d60c10940ac626ca81` (main, 2023-12-09) |
| 依赖 | PyTorch 1.12.1, PyG, ogb, rdkit, yacs, tensorboard, networkx, einops, metis |
| 环境 | conda create -n graph_mlpmixer python=3.8 |
| 配置入口 | `core/config.py`; 运行方式 `python -m train.<dataset>` |
| 数据集模块 | ZINC: `train/zinc.py`, MOLHIV: `train/molhiv.py`, MOLTOX21: `train/moltox21.py`, Peptides-func: `train/peptides_func.py`, Peptides-struct: `train/peptides_struct.py` |
| 注意 | 没有 MOLBACE 数据集模块（与论文 "–" 一致） |
| seed 来源 | `core/config.py` 中 `cfg.seed = None` (默认); `cfg.train.runs = 4` 控制运行次数。官方没有固定 seed 值，需要显式设置 `cfg.seed` |
| 论文表格目标 | ZINC: 0.085±0.005, MOLTOX21: 78.51±0.77, MOLHIV: 77.92±1.49, Peptides-func: 69.19±0.85, Peptides-struct: 0.2474±0.0016 |

## 方法 3-5: Full / Random / Policy-Learn

| 项 | 内容 |
|----|------|
| 论文名 | Efficient Subgraph GNNs by Learning Effective Selection Policies (ICLR 2024) |
| 官方仓库 | https://github.com/beabevi/policy-learn |
| 最新 commit | `a84adff1cde410fd6ffd098ccd7e813cd60656f0` (main, 2024-06-13) |
| 依赖 | PyTorch, PyG, ogb, hydra, wandb, einops, numpy |
| 环境 | docker 镜像 + conda (见 docker/); 使用 hydra 管理配置 |
| 配置入口 | `yaml-files/` — `{dataset}-all.yaml` (Full), `{dataset}-random.yaml` (Random), `{dataset}-gumbel.yaml` (Policy-Learn) |
| 数据集 | zinc, alchemy, molbace, molesol, molhiv, moltox21 — **没有 Peptides 配置** |
| 注意 | Peptides 上的 Policy-Learn 结果来自 Southern et al. 2025 (HyMN) 代码库而非本 repo |
| seed 来源 | 官方 sweep yamls 使用 seeds [1, 2, 3, 4, 5]；`config.py` 中默认 seed=1337, split=0 |
| 论文表格目标 | 见下文 |

### Full 目标数字

| 数据集 | 指标 | 论文值 |
|--------|------|--------|
| ZINC-12K | MAE | 0.087±0.003 |
| MOLTOX21 | ROC-AUC | 76.25±1.12 |
| MOLBACE | ROC-AUC | 78.41±1.94 |
| MOLHIV | ROC-AUC | 76.54±1.37 |

### Random 目标数字

| 数据集 | 指标 | 论文值 |
|--------|------|--------|
| ZINC-12K | MAE | 0.102±0.003 |
| MOLTOX21 | ROC-AUC | 76.62±0.63 |
| MOLBACE | ROC-AUC | 78.14±2.36 |
| MOLHIV | ROC-AUC | 77.30±2.56 |

### Policy-Learn 目标数字

| 数据集 | 指标 | 论文值 |
|--------|------|--------|
| ZINC-12K | MAE | 0.097±0.005 |
| MOLTOX21 | ROC-AUC | 77.36±0.60 |
| MOLBACE | ROC-AUC | 78.39±2.28 |
| MOLHIV | ROC-AUC | 78.49±1.01 |
| Peptides-func | AP | 64.59±0.18 |
| Peptides-struct | MAE | 0.2475±0.0011 |

## 额外参考代码库

| 项 | 内容 |
|----|------|
| 仓库 | https://github.com/jks17/HyMN (Southern et al. 2025) |
| Commit | `adde55268307ff69527375757ec31a146d59ccae` (2025-01-07) |
| 用途 | HOD-GNN 实现基于此代码库；GPS 的 moltox21 结果和 Policy-Learn 的 Peptides 结果可能来自此库 |
| 说明 | 如果官方代码库无法运行特定数据集/方法组合，应优先检查此库作为 fallback |

## 缺失 seed 状态

HOD-GNN 论文未披露任何 seed 数值。对于所有方法，`paper-seeds` 协议使用 `[0, 1, 2, 3]` 作为临时替代，标注为 `paper-seeds-provisional`。

## 官方配置完整性清单

| 方法 | ZINC | MOLTOX21 | MOLBACE | MOLHIV | Peptides-func | Peptides-struct |
|------|------|----------|---------|--------|---------------|-----------------|
| GPS | 官方有 | 官方无† | 官方无 | 官方有 | 官方有 | 官方有 |
| GraphViT | 官方有 | 官方有 | 官方无 | 官方有 | 官方有 | 官方有 |
| Full | 官方有 | 官方有 | 官方有 | 官方有 | 官方无†† | 官方无†† |
| Random | 官方有 | 官方有 | 官方有 | 官方有 | 官方无†† | 官方无†† |
| Policy-Learn | 官方有 | 官方有 | 官方有 | 官方有 | 官方无†† | 官方无†† |

†: GPS 的 moltox21 配置不存在于官方 GraphGPS 仓库，可能需从 HyMN 代码库获取。
††: Full/Random/Policy-Learn 的 Peptides 配置不存在于官方 policy-learn 仓库，可能需从 HyMN 代码库获取。