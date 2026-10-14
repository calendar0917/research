# Benchmark audit — MoleculeNet Lipophilicity under the BJOC cross-task protocol

协议: `cscl-bjoc-cross-task-lipophilicity-v1`（track `structure-contribution`）
审计日期: 2026-10-14 · 全部数字在 GPU 训练前收集完毕。

本文件回答三个问题：

1. 数据集与 split 是否真的被验证、可复现？
2. 哪些公开数字与我们的协议**可直接比较**，哪些不能？
3. 在“相同 split / 相同度量 / 相同监督预算 / 单模型”的严格协议下，公开 SOTA 是否可被复述与复现？

> 纪律：凡未能从可获取的原始出处核实到数字与协议细节的条目，一律标
> `NOT VERIFIED FOR DIRECT COMPARISON`，不进入排名表，也不用于任何“领先/落后”结论。

---

## 1. 数据集来源与结构审计

| 项目 | 值 | 证据 |
| --- | --- | --- |
| 数据集 | MoleculeNet `Lipophilicity` | Wu et al., *Chem. Sci.* 2018, DOI `10.1039/C7SC02664A` |
| 任务 | 单任务回归，logD (pH 7.4) | MoleculeNet 原文 §3.1.6 |
| 分子数 | 4200 | 原文 §3.1.6 / DeepChem `lipo_datasets.py` docstring |
| 官方 URL | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv` | `deepchem/molnet/load_function/lipo_datasets.py`（`LIPO_URL`） |
| 下载校验 | 282606 bytes，sha256 `aed41590…69bacd` | 本轮下载日志 |
| 列 | `CMPD_CHEMBLID`, `exp`, `smiles` | CSV header |
| RDKit 解析 | 4200/4200 成功，0 失败 | 本轮 `prepare-dataset` |
| 重度重复 | 4200 个不同 SMILES；规范化 SMILES 0 重复 | 本轮审计 |
| 多组分分子 | 1 个（row 1646，4 组分，含 3 个孤立原子） | 本轮审计（保留，不丢弃） |
| 连接表 | 元素 B/Br/C/Cl/F/I/N/O/P/S/Si/Se；最大度 4；charge −1/0/+1 | 本轮化学审计 |
| 标签范围 | 全数据 mean 2.186, std 1.203, min −1.5, max 4.5 | 数据集级描述统计（见下方说明） |
| 许可/来源 | ChEMBL 沉积数据（Hersey 2015, DOI `10.6019/chembl3301361`），DeepChem 公开重分发 | `lipo_datasets.py` 参考文献 |

**标签统计说明**：上表的 mean/std/range 是对公开 CSV 的**数据集级描述统计**，
在任何训练之前计算，只用于审计与报告；不参与特征、标准化、选模或任何决策。
模型只使用 train split 的 `exp`（train-only 标准化）；valid/test 的逐分子标签在
评分命令运行前留在本地 vault，计算环境不可达。

## 2. Split 审计

| 项目 | 值 |
| --- | --- |
| Splitter | DeepChem **2.8.0** `ScaffoldSplitter.split`（确定性，无 RNG） |
| Scaffold 定义 | Bemis–Murcko `MurckoScaffoldSmiles(includeChirality=False)` |
| 比例 | 0.8 / 0.1 / 0.1 |
| 复现方式 | 隔离 venv 直接运行 deepchem 2.8.0（见 `upstream/UPSTREAM_PROVENANCE.md`） |
| 结果尺寸 | **3360 / 420 / 420** |
| Scaffold 组 | 2408 个（4200 个分子全部有 scaffold） |
| 跨集合 scaffold 泄漏 | **0 组** |
| 跨集合规范化 SMILES | **0** |
| 确定性 | 无 seed 参数，算法纯粹由 `(len(set), first_index)` 排序决定 |
| 与参考逐索引一致性 | `reference_match: true`（`split_lipo_v1.json`） |
| 冻结证据 | `split_lipo_v1.json`（行号哈希 ×3、train_y 哈希）、`upstream/deepchem_2.8.0_reference_split.json` |

DeepChem 自己的 loader 对 Lipophilicity 明确写有 *"Scaffold splitting is recommended
for this dataset"*（`lipo_datasets.py`），因此 scaffold 划分不是我们挑出来的有利协议。
注意：DeepChem 2.8.0 的 scaffold 划分与 GROVER / Chemprop 论文中"多次随机 scaffold
划分取均值"的做法**不是同一个 split**（见 §4）。

各 split 的 label 描述统计（仅审计，不用于决策）：

| split | n | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| train | 3360 | 2.163 | 1.211 | −1.50 | 4.50 |
| valid | 420 | 2.196 | 1.217 | −1.10 | 4.49 |
| test | 420 | 2.364 | 1.104 | −1.30 | 4.50 |

## 3. 我们的协议（比较时必须匹配的 6 个维度）

1. **数据**：仅 MoleculeNet Lipophilicity 的 4200 个分子；**无任何外部分子**参与
   预训练或训练（SSL 只用 train split 图结构）。
2. **Split**：上述 DeepChem 2.8.0 确定性 scaffold 3360/420/420，索引冻结并哈希。
3. **监督**：仅 train 标签；train-only 标准化；360 epoch、batch 128、Adam 1e-3→1e-5
   per-step cosine、最后 5 个 epoch 参数 soup；**无 validation 选模**。
4. **模型**：单模型（BJOC-Null，419,713 参数）；无 ensemble 头衔；无 GNN/attention。
5. **度量**：raw-scale RMSE（主）、MAE（辅），逐分子预测在读取标签前冻结哈希。
6. **开销**：GPU 小时与参数量全量记账（`gpu_budget_ledger.json`）。

## 4. 对比表

### 表 A — 同协议（in-protocol）结果

| 方法 | split | 度量 | 值 | 备注 |
| --- | --- | --- | --- | --- |
| ECFP4 (2048-bit) + RandomForest(500) | 本协议 | valid RMSE / MAE | 0.95976 / 0.75504 | 本仓库同索引训练（`baselines.json`） |
| ECFP4 (2048-bit) + RandomForest(500) | 本协议 | **test RMSE / MAE** | **0.87660 / 0.70321** | 同上 |
| ECFP4 + XGBoost (hist, 500×depth6) | 本协议 | valid RMSE / MAE | 0.93787 / 0.73806 | 同索引、同 train 标签 |
| ECFP4 + XGBoost (hist, 500×depth6) | 本协议 | **test RMSE / MAE** | **0.85973 / 0.68225** | 同上 |
| BJOC-Lipo-Scratch (Arm S) | 本协议 | valid RMSE / MAE（5 seeds 均值） | 0.80261 / 0.60856 | 419,713 参数，无预训练；test 0.77511 / 0.57682（±0.02781） |
| BJOC-Lipo-SSL (Arm P) | 本协议 | valid RMSE / MAE（5 seeds 均值） | **0.71351 / 0.52853** | 同上模型 + train-only masked SSL 编码器初始化；test **0.73444 / 0.54328**（±0.01667） |
| RDKit Crippen logP | — | — | 描述性参考 | 只作化学基线，绝不作为模型输入；本轮未将其与 RMSE 并列排名 |

表 A 的口径：同一 split 索引、同一逐分子标签、同一 raw-scale 度量、单模型、无
验证集选模。Arm S/P 的 valid 数字来自 Stage F/G（5 seeds × 420 分子），test 数字
来自 Stage H 的一次性冻结读取（10 个预测文件先 SHA-256 冻结，见
`artifact_manifest_pre_test.json` 与 `prediction_manifest_test.json`）。
ECFP 基线在 valid/test 上均优于 MoleculeNet 原文的 random-split 数字（0.799/0.655 不可比），
说明 split 难度差异是真实存在的，不能跨 split 引用。

### 表 B — 无外部分子、但 split 不同的公开数字（不能直接排名）

| 来源 | 方法 | split | 度量 | 值 | 可比性 |
| --- | --- | --- | --- | --- | --- |
| MoleculeNet 原文（DOI 10.1039/C7SC02664A） | XGBoost | **random** | RMSE | **0.799** | 不同 split |
| MoleculeNet 原文 | GraphConv | **random** | RMSE | **0.655** | 不同 split |
| GROVER 论文 Table 1（arXiv:2007.02835）复现 | D-MPNN | 3 次随机 scaffold 8:1:1 均值 | RMSE | **0.653 ± 0.046** | 不同 splitter/seed |
| 同上 | AttentiveFP | 同上 | RMSE | 0.650 ± 0.030 | 同上 |
| 同上 | MPNN | 同上 | RMSE | 0.672 ± 0.051 | 同上 |
| 同上 | GraphConv | 同上 | RMSE | 0.712 ± 0.049 | 同上 |
| 同上 | Weave | 同上 | RMSE | 0.813 ± 0.042 | 同上 |
| 同上 | N-GRAM | 同上 | RMSE | 0.876 ± 0.033 | 同上 |
| 同上 | TF_Robust | 同上 | RMSE | 0.909 ± 0.060 | 同上 |
| 同上 | SchNet | 同上 | RMSE | 0.909 ± 0.098 | 同上 |
| 同上 | MGCN | 同上 | RMSE | 1.113 ± 0.041 | 同上 |

这些数字的证据是 GROVER 论文可获取的 HTML 版本 Table 1（回归部分，Lipo 列）与
MoleculeNet PMC 全文；两者都使用与 DeepChem 2.8.0 不同的 split 生成方式
（GROVER：3 个随机 seed 的 scaffold 划分；MoleculeNet：random split），因此
**`NOT VERIFIED FOR DIRECT COMPARISON`**：可用于描述任务难度区间，不得用于
"我们超过/落后 XX" 的结论。

### 表 C — 公开领先数字（使用了额外数据/预训练；登记但不可直接比较）

| 来源 | 方法 | 外部分子 | split | 度量 | 值 | 可比性 |
| --- | --- | --- | --- | --- | --- | --- |
| GROVER (arXiv:2007.02835) | GROVERbase (自监督预训练) | ~11M | 3 次随机 scaffold 8:1:1 均值 | RMSE | **0.563 ± 0.030** | 额外数据 + 不同 split |
| GROVER | GROVERlarge | ~11M | 同上 | RMSE | **0.560 ± 0.035** | 同上 |
| GROVER Table 13（消融） | GROVER without pretraining | 0 | 同上 | RMSE | 0.643 | 不同 split |
| Uni-Mol (arXiv:2110.01191) / SMI-TED | — | 大规模 | — | RMSE | `NOT VERIFIED` | 2026-10-14 访问尝试：arXiv 元数据查询空结果 / 项目页 HTTP 503；数字与 split 细节无法核实 |
| Chemprop D-MPNN 原文 | D-MPNN | 0 | 原文自定义随机 scaffold | RMSE | `NOT VERIFIED` | 本地可获取文本中未能定位该列数字；GROVER 复现值见 Table B |

**同协议公开 SOTA 检索结论**：截至 2026-10-14，没有检索到任何公开结果在
**完全相同的 DeepChem 2.8.0 确定性 scaffold split（3360/420/420）**、且不使用
外部分子、单模型条件下报告过 Lipophilicity RMSE。因此：

- 本轮**不会**声称公开榜单 SOTA；
- 与 GROVER 0.560 一类数字的差距只能作为"不同协议下的任务上界参考"；
- 若最终决策词为 `SOTA_CANDIDATE_REQUIRES_EXTERNAL_VERIFICATION`，其含义是
  "在本协议内达到可复核的最优单模型水平、但仍缺少同协议外部复现"，而不是
  "超过 GROVER"。

## 5. 审计方法学备注

- 论文数字的提取方式：下载 arXiv HTML/PMC 全文后按 `<table>` 结构解析，
  官方 Table 1/Table 13 的 Lipo 列逐行核对（文件快照与解析脚本在本地
  临时目录，结论已逐数字抄录于上表；引用时以表格行列结构为准）。
- 对无法访问的来源（Cloudflare/HTTP 503/arXiv 查询空），保留失败证据并标记
  `NOT VERIFIED`，绝不猜测数值。
- 若后续把本协议结果与任何公开数字并列，必须同时标注该公开数字的 split 类型，
  并把 random-split 结果与 scaffold-split 结果分表呈现。
