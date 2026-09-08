# GSN 官方复现：来源、协议、假设、偏差

## 两个协议（本轨）

| | `gsn-strict-social-v1`（**主**） | `gsn-official-social-v1`（旁路参考） |
|---|---|---|
| 评估 | 10 seeds × 10×10 CV（每 seed 100 折） | 论文同款：单次 10 折 CV（seed 0） |
| 划分 | `RepeatedStratifiedKFold(10,10,random_state=seed)`（同 wl-kernel 轨 API/同序） | 官方 powerful-gnns 10fold_idx |
| epoch 选择 | **折内验证集**（train 分层切 10% val） | 官方：无 val 时直接用 test 选（**乐观/测试泄漏**） |
| 报告 | 每折 val-best epoch 的 test acc；fold-level + seed-level 聚合 | 官方 main.py "Best test mean/std" |
| 超参 | 固定论文 Table 5（不乐观化的部分） | 同左 |

> 之所以官方是“乐观协议”：官方 main.py:333 `best_idx = perf_opt(test_accs)`（无
> val 时）——即在测试集上选 epoch。本轨严格协议写 `val_idx-{k}.txt` 后官方代码自动
> 改用 `perf_opt(val_accs)`（main.py:333/398），**零改动**消除该泄漏。

## 严格协议要点（run_strict.py）

- 每 (seed, rep, fold)：`RepeatedStratifiedKFold(10,10,random_state=seed)` 取 train/test →
  `train_test_split(stratify, test_size=0.1, random_state=seed*1e5+rep*1e3+fold)` 切 val；
  写入 `data/social/<NAME>/10fold_idx/{train,test,val}_idx-{fold+1}.txt` 后调官方 main.py
  （`--split given --fold_idx [fold]`）。
- 种子列表（用户给定）：`0, 41, 123, 1024, 2026, 777, 3407, 999, 111, 888`。
  ⚠ wl-subtree-kernel 轨种子为 `0, 42, …`（42≠41）——与其横比前需统一种子列表。
- 每折训练用 `--seed {seed}`（官方随机种子全链路），仅划分随 (seed,rep,fold) 变。
- 磁盘：官方每 epoch 存 checkpoint（300 文件/折 × 1000 折 ≈ 数百 GB/配置）→ 默认每折
  结束即清理（`--keep-checkpoints` 保留）。
- 汇总：fold-level（全部 1000 折 mean±std）与 seed-level（10 个种子均值再聚合）。

## 成本参考（服务器）

单折训练时长取决于数据集（V100 参考：IMDB-B/M 数分钟~十几分钟/折；COLLAB、REDDIT-B
更慢）。建议：先 `--seeds 0 --n-repeats 1`（10 折）测单折时长 → 外推 → 多 GPU
`--workers N --devices N` 并行。8 配置 × 1000 折总计可能是**数十~数百 GPU 小时**，
请按预算分阶段跑（先 IMDB-BINARY 一个配置出主表，再补其余）。

## 来源登记

| 项 | 值 |
|----|----|
| 论文 | Improving GNN Expressivity via Subgraph Isomorphism Counting, Bouritsas et al. |
| arXiv | 2006.09252 **v3**（2021-07-05，最终版；PDF+TeX 已本地核对补充材料） |
| 官方代码 | https://github.com/gbouritsas/graph-substructure-networks commit `6cce24a2c0f59c183c388f3016d33502f63e8175`（vendor 目录，**零改动**） |
| 数据/划分 | https://github.com/weihua916/powerful-gnns commit `9a2ce8a`；`dataset.zip`（含 4 个目标数据集 raw `.txt`；`IMDBBINARY.txt` 与官方仓自带 md5 相同 `30ccc205…`） |
| 数据集布局 | `vendor/graph-substructure-networks/datasets/social/{IMDBBINARY,IMDBMULTI,COLLAB,REDDITBINARY}/`；`10fold_idx/` 仅前 3 者有 |

## 论文协议（明确项）

- 评估：GIN [16] 协议——10 折 CV，**报告 10 折平均测试精度最优 epoch 上的 10 折 mean±std**。
  官方 `main.py --fold_idx 0..9 --split given` 即此；`Best test mean: X +/- Y` 是唯一对应输出。
- 结构：social 数据集 family=clique(s)（K3..K5），`--induced False`（motif）。**clique 的 induced/non-induced 计数完全相同**（K_k 无多余边可加），故论文 Table 5 中 "same"（stage-1 用 triangle 定的 type）对本配置无歧义。
- 网络：4 层 GIN 风格 + 结构 id 进消息（Eq. 8/9）、JK（全层，线性投影）、readout=mean、id 一热编码（one_hot_unique）。
- 超参（supplementary Table 5，本轨 `configs/social_paper.json` 逐项落实）：

| dataset | 变体 | k | degree | lr | decay_steps | 论文数字 (Table 1) |
|---|---|---|---|---|---|---|
| IMDB-B | GSN-e | 5 | No | 1e-3 | 10 | **77.8±3.3** |
| IMDB-B | GSN-v | 4 | Yes | 1e-3 | 10 | **76.8±2.0** |
| IMDB-M | GSN-e | 5 | Yes | 1e-3 | 10 | **54.3±3.3** |
| IMDB-M | GSN-v | 3 | Yes | 1e-3 | 10 | **52.6±3.6** |
| Collab | GSN-e | 3 | No | 1e-2 | 50 | **85.5±1.2** |
| Collab | GSN-v | 3 | No | 1e-2 | 50 | **82.7±1.5** |

（公共：batch 32、width 64、dropout 0、decay_rate 0.5、readout mean、epochs 300、iters 50）

## 假设清单（论文未写明，需在报告中声明）

1. `--num_iters 50 / --num_epochs 300` 对所有 social 数据集统一（官方 README 仅给出 IMDBBINARY 命令含 50；Table 5 未列 iters/epochs）。
2. REDDIT-BINARY：**论文 Table 1 无此项**（7 个数据集 = GIN/PPGN 交集；powerful-gnns 也无 REDDITBINARY 的 10fold_idx——这是被排除的直接原因）。本轨按 **GIN 论文对 REDDIT 的做法**用 `--split random`（StratifiedKFold(10, shuffle, random_state=split_seed=0)）。
   超参无依据，延续 IMDB-B（k=5/k=4），**不能引用论文数字**，只能作"同协议扩展"。
3. "substructure type: same" 按 motif（induced=False）落实（对 cliques 等价，见上）。
4. seed=0（论文 TUD 单次 10 折 CV；10 折 mean±std 已含折间方差）。

## 已知环境偏差（相对作者原始环境）

- 官方 README：torch 1.4.0/py3.7/PyG 1.4.3 + graph-tool（conda-forge），V100。
- 本复现：**python 3.11 + torch 2.5.1(+cu124) + PyG 2.6.1 + conda-forge graph-tool + numpy 2.x**。
- 官方 README 明示“不同 torch/CUDA 版本会影响种子级复现（差异通常很小）”；跨版本复现偏差应视为预期。
- 未改动 vendor 任何文件。唯一的非官方工件：
  a. 作者预计算缓存（PyG1.4 pickle）在 PyG2.6 下无法直接加载 → `setup_data.py --migrate-processed`
     在本地只迁移 tensor 属性并原址回写（仅 IMDBBINARY GSN-e 用得到；服务器正式跑建议
     `--purge-processed` 让 graph-tool 重算，走官方同一条代码路径）；
  b. 本地冒烟用 `code/compat/`（networkx 版 graph_tool + wandb stub），**仅本地**，服务器不用。

## 冒烟记录（本地 CPU，2026-09-08）

- pipeline 模式（作者缓存）：3 epoch 后 test 0.77，参数量 64908 ≈ 论文 Table 4 IMDB-B 65K ✓
- counts 模式（shim 计数 K3/K4）：identifiers 形状正确（20 节点 × 2 orbit），计数合理 ✓
- 兼容性结论：py3.12/numpy2/torch2.5/PyG2.6 全链路可跑（仅 deprecation warning）。
