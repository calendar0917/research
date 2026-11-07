# LCRC-RIE frozen pilot v1 — 使用说明

冻结的可行性探针：在**原始 LCRC-A（seed 0，epoch 40）完全冻结**的前提下，检验真实
motif–motif 拓扑关系是否提供超出“相同 motif 组成 + 相同容量 + 相同监督配方”的
MolPCBA Validation AP 增量。

* 研究定义与判定阈值：`PROTOCOL.md`（在本轮第一次 GPU 运行前冻结）
* 结论与逐任务诊断：`REPORT.md`
* 结论标签：`analysis/decision.json`

## 结构

```
code/rie_common.py     冻结定义：源 checkpoint、8 个 motif 家族、bin、165 维特征、
                       head 配方、训练配方、判定阈值、预算
code/rie_motifs.py     RDKit ChemicalFeatures(6) + SMARTS(2) 的 8 家族 motif 提取（去重）
code/rie_features.py   物理图距离、pair/bin 构造、z_i、pair feature、聚合、
                       SHUFFLE bin 排列、RMS 归一化
code/rie_extract.py    三个 stage：smoke / structures(CPU) / features(GPU, 唯一一次 A 前向)
code/rie_model.py      8,784 参数的 RelationHead（共享投影 + 任务权重，零残差初始化）
code/rie_train.py      preflight(门槛 13–16,20) + formal(REAL/SHUFFLE 配对 8 epoch)
code/rie_compare.py    epoch8 比较、逐任务 CSV、分层统计、GPU ledger、冻结判定
code/rie_audit.py      门槛 3–12,17–19 的 CPU 审计 + 可选 GPU 单 batch 行对应抽查
tests/test_rie_pilot.py  CPU-only 26 项单元测试
analysis/              冻结产物（JSON / CSV，入库）
runs/                  structures / features / heads / smoke（git-ignored）
```

## 远端执行（唯一授权流程）

```bash
git status --short                     # 必须干净
rr doctor res
rr deploy res --pool res-gpu1          # 绑定 host+pool+commit 的部署标记
```

固定六个 job，全部用同一个 commit 的代码（`--pool res-gpu1`，GPU0 不参与）：

```bash
R=tracks/structural-sufficiency/lcrc_pcba_rie_frozen_pilot_v1

# 1) CPU：motif / 物理图距离 / SHUFFLE bin 静态结构 + 1024 分子计时探针
#    rdkit 来自 pyproject 的 lipo 组（res 在线，从已提交的 uv.lock 安装）
rr run res --pool res-gpu1 rie-structures --result $R/runs --result $R/analysis -- \
  bash -c 'uv sync --frozen --group lipo -q && python $R/code/rie_extract.py --stage structures --workers 12 --chunk-size 2048'

# 2) GPU1：唯一一次冻结 A 前向 + REAL/SHUFFLE 关系特征 + RMS 归一化
rr run res --pool res-gpu1 rie-features --result $R/runs --result $R/analysis -- \
  bash -c 'python $R/code/rie_extract.py --stage features'

# 3) GPU1：smoke（CPython/GPU 数值 + 冻结身份）+ CPU 审计 + 单 batch 行对应抽查
#    `rr deploy` 的 uv sync 只装默认组，所以需要 rdkit 的 job 自己再装一次（幂等）
rr run res --pool res-gpu1 rie-audit --result $R/runs --result $R/analysis -- \
  bash -c 'uv sync --frozen --group lipo -q && python $R/code/rie_extract.py --stage smoke && python $R/code/rie_audit.py --with-gpu-spotcheck'

# 4) GPU1：head 预检（门槛 13–16、20）
rr run res --pool res-gpu1 rie-train-preflight --result $R/runs --result $R/analysis -- \
  bash -c 'uv sync --frozen --group lipo -q && python $R/code/rie_train.py --stage preflight'

# 5) GPU1：REAL / SHUFFLE 配对正式 8 epoch（epoch 0/2/4/8）
rr run res --pool res-gpu1 rie-train --result $R/runs --result $R/analysis -- \
  bash -c 'uv sync --frozen --group lipo -q && python $R/code/rie_train.py --stage formal'

# 6) CPU：epoch8 比较、逐任务 CSV、分层统计、GPU ledger、冻结判定
rr run res --pool res-gpu1 rie-compare --result $R/analysis -- \
  bash -c 'python $R/code/rie_compare.py'
```

查看与回收：

```bash
rr jobs ; rr status <exp> ; rr logs <exp> ; rr pull <exp>
```

运行顺序是**固定**的：

```
structures  ->  features  ->  audit(+gpu spotcheck)  ->  train --stage preflight
            ->  train --stage formal  ->  compare
```

任一 stage 的非零退出码即为停止信号；`analysis/*.json` 的 `all_passed` 字段是下一 stage
的准入条件（`rie_train` 会拒绝未通过的 cache / preflight）。

## 本地测试

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_rie_frozen_pilot_v1/tests/test_rie_pilot.py
```

26 项，约 1 秒，CPU-only。测试需要 `torch`（默认 `ksvd` 组）与 `rdkit`（`lipo` 组），
因此**不在** GitHub CI 的 fresh-clone-safe allowlist 内（allowlist 用 torch-free 最小环境）。

## 判定标签

| 标签 | 含义 |
|---|---|
| `PROMISING_RELATION_SIGNAL` | 四项阈值同时满足；仅表示值得考虑完整联合训练 |
| `WEAK_OR_MIXED_SIGNAL` | `D_base >= +0.002` 但未满足全部 PROMISING 条件 |
| `NO_RELATION_SIGNAL` | `D_base < +0.002`，或 REAL 对 SHUFFLE 无明显优势 |
| `INSUFFICIENT_RELATION_INTERVENTION` | Validation `REAL != SHUFFLE` 比例 < 15%（训练前停止） |
| `SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH` | 源 checkpoint 身份或 AP 复现失败 |
| `INCOMPLETE` / `INCOMPLETE_RESOURCE` | 预检/完整性失败，或硬预算内未完成 |

本轮**不**自动启动完整 LCRC-RIE 联合训练，也不含任何超参数搜索、额外 seed 或额外架构。

## 关键解释提醒

`D_base` 大而 `D_ctrl` 小意味着：新增的小型监督 readout 有用，但没有证据表明收益来自
真实 motif 距离关系。三个命题（REAL>A、REAL>SHUFFLE、高支持度 assay 上也成立）必须方向
一致才构成继续研究的理由。
