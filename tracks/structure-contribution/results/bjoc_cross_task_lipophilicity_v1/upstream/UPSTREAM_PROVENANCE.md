# Upstream provenance — Lipophilicity split and RawGraph vocabulary

本文件记录本目录（`upstream/`）与 `dataset/` 中**上游来源**的可复现证据。
全部内容在任何 GPU 训练之前生成并冻结。

## 1. 数据集

| 项目 | 值 |
| --- | --- |
| 名称 | MoleculeNet Lipophilicity（Hersey 2015 ChEMBL 数据集，DeepChem 重分发） |
| 下载 URL | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv` |
| 下载日期 (UTC) | 2026-10-14 |
| 字节数 | 282606 |
| SHA-256 | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` |
| 行数 / 列 | 4200 / `CMPD_CHEMBLID,exp,smiles` |
| 标签列 | `exp`（octanol/water distribution coefficient，logD @ pH 7.4） |
| 解析失败 | 0 / 4200（RDKit `Chem.MolFromSmiles`） |
| 规范 SMILES 重复 | 0 |
| 原始 CSV 提交位置 | 不提交（本地 `data/lipophilicity/Lipophilicity.csv`，`**/data/**` 被 .gitignore 排除） |

数据许可与引用（见 `BENCHMARK_AUDIT.md` §1 展开）：MoleculeNet 论文 DOI
`10.1039/C7SC02664A`；原始 ChEMBL 沉积数据 ZENODO DOI `10.6019/chembl3301361`；
DeepChem 以 `deepchemdata.s3-us-west-1.amazonaws.com` 公开分发。

## 2. 上游 splitter 源码快照

本目录的 `deepchem_2.8.0_scaffold_splitter_excerpt.py` 是 **deepchem 2.8.0 官方
wheel** 中 `deepchem/splits/splitters.py` 的逐字摘录（两个 block：`_generate_scaffold`
与 `class ScaffoldSplitter`），仅去掉未引用部分、拼接为单文件；函数体未改动。

- 完整上游文件（wheel 内）: `deepchem/splits/splitters.py`，64330 bytes，
  sha256 `5adf9a5cbd1dca6419458f9ccf173dab1b0003228f820e436f8da4823b657d60`
  （与 wheel `RECORD` 条目
  `deepchem/splits/splitters.py,sha256=Wt-aXL0dymQZRY-czxc9qxsAAyKPgg5Db42kgjtlfWA,64330`
  的 base64url 摘要一致）。
- 本目录摘录文件 sha256:
  `9be01edce563bb25b88fe9be44e10de30cc3fa701c505563e1d95b007466281f`。
- 相关上游文件（同样与 wheel RECORD 哈希一致）:
  - `deepchem/data/data_loader.py`, 84152 bytes,
    sha256 `c71329c88d98ffd78a78506c72b4be09976554c41b4fc8ec81fdc41cc3aa2d27`；
    `CSVLoader` 在 `id_field=None` 时设 `id_field = feature_field`，而
    `_LipoLoader` 传 `feature_field="smiles"` ⇒ `dataset.ids` 就是 CSV 顺序的 smiles。
  - `deepchem/molnet/load_function/lipo_datasets.py`, 3318 bytes,
    sha256 `f16dde9daf4b8dcbca4c68096dbed058ea1267810ac32059824bbbdf5a3b8f8a`；
    `_LipoLoader.create_dataset` 使用 `tasks=['exp']`，
    `LIPO_URL` 与本轮下载 URL 完全一致。

安装方式：隔离 venv（`python -m venv /tmp/dcenv`）中 `pip install deepchem==2.8.0`
（PyPI 官方 wheel），另装 `rdkit`（生成参考 split 时版本 2026.09.1）。

## 3. 参考 split 的生成与校验

参考 split 由**直接运行 deepchem 2.8.0** 生成：

1. 用 `_LipoLoader`（`tasks=['exp']`, `feature_field='smiles'`, ECFP featurizer）
   从官方 CSV 创建 DeepChem `DiskDataset`；`dataset.ids` = CSV 顺序 smiles。
2. `deepchem.splits.ScaffoldSplitter().split(dataset, 0.8, 0.1, 0.1)`；
   splitter 的 `seed` 参数在该实现中不使用，算法是确定性的。
3. 输出 `train_idx / valid_idx / test_idx = 3360 / 420 / 420`。
4. 仅保留 **索引**：`y` 与 `scaffolds` 字段在写入
   `deepchem_2.8.0_reference_split.json` 时被主动丢弃（label-free）。

`code/bjoc_lipophilicity_adapter_v1.py` 中的
`generate_scaffold_dc280 / generate_scaffold_sets_dc280 / scaffold_split_dc280`
是上述算法的逐行重写；`prepare-dataset` 命令在生成 `split_lipo_v1.json` 时
逐索引比对参考 split（`reference_match: true`），并额外记录
`train/valid/test` 行号哈希、`train_y` 哈希、Bemis–Murcko 组数与跨集合泄漏检查
（`scaffold_groups_cross_split = 0`, `canonical_smiles_cross_split = 0`）。

重跑参考 split 需要的上游代码就是本目录的摘录文件；`rdkit` 版本不同不改变
Murcko scaffold 的语义（仅影响极少数异常输入），本轮 `prepare-dataset` 的
逐索引复现即是版本无关性的直接证据。

## 4. 本目录文件清单

| 文件 | 内容 | 生成方式 |
| --- | --- | --- |
| `deepchem_2.8.0_scaffold_splitter_excerpt.py` | 上游 splitter 摘录 | 从 deepchem 2.8.0 wheel 提取 |
| `deepchem_2.8.0_reference_split.json` | label-free 参考索引 | deepchem 2.8.0 直接运行（`y`/`scaffolds` 被剔除） |

`dataset/lipo_raw_v1.json.gz`（label-free RawGraph）与 `split_lipo_v1.json`
由 `prepare-dataset` 生成，位于上一级目录；其哈希绑定在
`protocol_freeze.json`。这些文件都不含 validation/test 标签。
