# Mentor 上游脚本 vendor 快照（v1, 2026-09-21）

本目录是导师提供的原始脚本的**只读、逐字节快照**，用于让 fresh clone 能够审计
被仓库已有文档引用的上游源码。

## 为什么在这里，而不是 `artifacts/`

这些文件最初来自仓库根目录的 `artifacts/`。但 `.gitignore` 明确排除了
`artifacts/**`（只保留 `artifacts/.gitkeep`），这与 `AGENT.md`、`README.md`
对 `artifacts/` 的定位一致：**本地临时导出，默认不提交**。

问题是：这些 `.py` 不是生成物，而是**不可再生的上游科研输入**，并且**已经被
Git-tracked 文档直接引用**：

- `tracks/ksvd/results/luyin16/MENTOR_ARTIFACT_TYPED_KSVD_REPRO_20260906.md:75-76`
  在审计中直接引用了 `artifacts/molhiv_online_structural_ksvd_full.py` 与
  `artifacts/run_molhiv_ksvd_three_diagnostics.py`。

如果被引用的文件本身不在版本控制内，该结论在 fresh clone 中**无法复核**
（引用悬空）。因此这里的处理是：**保持 `artifacts/` 继续被忽略，把上游源码
单独 vendor 进 ksvd track**，沿用仓库已有的两个先例：

- `tracks/gsn-replication/code/vendor/<upstream-repo>/`（vendor 外部官方代码）
- `tracks/ksvd/results/zinc_long_cycle_audit/upstream/` + `TARGET_PROVENANCE.md`
  （逐文件 sha256 + 溯源表）

`artifacts/` 原件保持不变，两者哈希一致，可交叉验证。

## 快照清单（sha256，逐字节与 `artifacts/` 原件一致）

| 文件 | 行数级别 | sha256 |
|---|---|---|
| `beam8_mentor_pipeline_20260809.py` | 导师 50 节点真实子图 Beam8 主线，单文件自包含 | `32acd598a661330d556cbadb0791b23ce193dad42b1a82089d30afcdd83fb4d0` |
| `molhiv_online_structural_ksvd_full.py` | MolHIV online structural K-SVD（OMP + mini-batch K-SVD），含 official-full 泄漏安全模式 | `45c1bdadbc7feb796bc248c3f672798bb94b044dd39163411772b5099f41bfaf` |
| `run_molhiv_ksvd_three_diagnostics.py` | 三诊断：A 重建→typed pooling、B 完整二阶矩、C 置换稳定性 | `213d060147f388e063fd5f2a68ae73fd64085f03ba9547510fa96416e97d38d9` |
| `run_molhiv_r2_atom_ring_context.py` | R2 共享原子 × 显式环上下文特征 | `94f753d5d2264cd67e18b641b7cf84db876da850933d73f9a3abbed1acdd23da` |
| `run_molhiv_no_ring_semantic_cross.py` | 无手写环的语义/结构交叉特征 | `c5d9ec76953d12cccf6acadb4adb76d67148139f9d13f0f77709955885ed6687` |
| `run_molhiv_no_ring_capacity_study.py` | 无环语义交叉的字典容量研究（控制器） | `b2b33d85201cb7475032c473819800c3cf73b81dac5e2dc070d10eb88b251c71` |
| `run_molhiv_auc_objective_study.py` | 固定 MolHIV K-SVD 特征上的 AUC 目标函数研究 | `58754f96fca55692367288df5a5b0a638f8a841060b08d74db37c94dc13e4548` |

复算方式：

```bash
cd tracks/ksvd/experiments/luyin16/mentor_upstream/v1_20260921
sha256sum *.py
```

## 整理原则：不做任何"清理"

这 7 个文件**逐字节原样保留**，未重命名、未改 import、未改任何硬编码路径。
原因：上面的已有审计文档是在"`artifacts/` 原文长什么样"的前提下做出的判断，
任何改动都会让该证据链的 sha256 对应关系失效。

脚本本身的缺陷写在本文件里，**不修代码**。

## 已核验

- 7 个文件均为合法 UTF-8，无 BOM 异常。
- 未发现凭据/密钥/token，未发现机器相关绝对路径（无 `/home/<user>/` 遗留）。
- 与 `artifacts/` 原件 sha256 逐个一致。
- `beam8_mentor_pipeline_20260809.py` 已在本 vendor 路径下**实际跑通**
  （`uv run python ... --pilot-size 5`，日期 2026-09-21）：脚本的
  `_repo_root()` 向上查找 `data/` 的逻辑对新路径有效，直接复用已有
  `data/subgraphs_50_20_10000_batch_0.npz` 缓存，未产生额外落盘文件。

## 已知限制（IMPORTANT）

### 1. 依赖图不闭合：7 个被引用的脚本从未提供

这些脚本通过 `importlib.util.spec_from_file_location` 在运行时加载**同目录**
的其他脚本。下列被引用的脚本**不在本快照中，且在仓库任何位置都不存在**：

| 缺失脚本 | 被谁引用 |
|---|---|
| `run_molhiv_shared_atom_classification_old_protocol.py` | `run_molhiv_ksvd_three_diagnostics.py`（`--classification-script`）、`run_molhiv_r2_atom_ring_context.py` |
| `run_molhiv_direct_typed_patch_pool.py` | `run_molhiv_ksvd_three_diagnostics.py`（208-D typed 描述的来源） |
| `run_molhiv_ensemble_multiview.py` | `run_molhiv_auc_objective_study.py`（特征视图构造） |
| `molhiv_ensemble_union.py` | 上述 ensemble 路径 |
| `run_molhiv_explicit_ring_oracle.py` | 环上下文 oracle 对照 |
| `molhiv_online_structural_ksvd_pilot.py` | `molhiv_online_structural_ksvd_full.py`（pilot 前身） |

另有一个被引用者 `run_mentor_subgraphs_grouped_ksvd_followup.py` **已存在于**
`tracks/ksvd/code/run_mentor_subgraphs_grouped_ksvd_followup.py`。

**后果**：本快照不是可完整重跑的实验包。除 `beam8_mentor_pipeline` 外，其余
脚本缺少关键依赖，只能作为**机制/协议的文字证据**来阅读。这也正是
`MENTOR_ARTIFACT_TYPED_KSVD_REPRO_20260906.md` 把当前实现定性为
"documented proxy 而非 byte-for-byte 复现"的根因之一。

### 2. 相对路径假定"脚本所在目录就是项目根"

多数脚本使用相对路径（`./results/...`）或
`ROOT = Path(__file__).resolve().parent`，例如：

- `run_molhiv_no_ring_capacity_study.py:21-28,84` →
  `results/molhiv_historical_m13_seed0/cache_M13_q999`、
  `results/molhiv_restored_legacy_v1/composition_69_exact.npz`
- `run_molhiv_no_ring_semantic_cross.py:56-79` → 同一批 `results/` 路径
- `run_molhiv_ksvd_three_diagnostics.py:1508,1588` → `--ksvd-script` 默认
  `./molhiv_online_structural_ksvd_full.py`

原作者是把它们放在自建的平铺工作目录里运行的。放到本仓后，这些路径**不再指向
真实数据**，必须显式传参覆盖。

### 3. 所需的 MolHIV 历史 results payload 本地不存在

上述脚本依赖 `tracks/ksvd/results/` 下的一批历史缓存
（`molhiv_historical_m13_seed0/cache_M13_q999`、
`molhiv_restored_legacy_v1/composition_69_exact.npz`、
`molhiv_historical_seed0_features_ablation_6views_xgb_10seed/auc_objective_summary.json`
等）。经核对，**这些目录当前都不存在**。因此除 `beam8_mentor_pipeline` 外，
其余 MolHIV 脚本在本机无法直接重跑。

`beam8_mentor_pipeline_20260809.py` 是唯一例外：其默认数据源
`data/subgraphs_50_20_10000_batch_0.pkl` 虽不存在，但脚本会回退到已有
`data/subgraphs_50_20_10000_batch_0.npz` 缓存，故可运行
（`data/ogb/ogbg_molhiv` 亦存在）。

### 4. 不声称复现

本目录的 vendor 行为**只提供可审计的源码溯源**，不构成对导师结果的复现声明。
现有对齐工作见 `tracks/ksvd/experiments/luyin16/mentor_artifact_typed_ksvd.py`、
`mentor_artifact_optuna.py`、`mentor_ring_context_proxy.py`，以及
`tracks/ksvd/results/luyin16/MENTOR_ARTIFACT_TYPED_KSVD_REPRO_20260906.md`。

## 相关文件

- `tracks/ksvd/results/luyin16/MENTOR_ARTIFACT_TYPED_KSVD_REPRO_20260906.md` — 直接引用本目录 `molhiv_online_structural_ksvd_full.py`、`run_molhiv_ksvd_three_diagnostics.py` 的审计报告
- `tracks/ksvd/experiments/luyin16/mentor_artifact_typed_ksvd.py` — 对齐后的 proxy 实现
- `tracks/ksvd/results/zinc_long_cycle_audit/upstream/TARGET_PROVENANCE.md` — 同类溯源约定
