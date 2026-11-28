# molhiv-multigranular-structure-sharing-v0

MolHIV 上的**多粒度显式结构发现与预测价值研究 v0**：精确枚举属性诱导子图（size-2/3/4 全部连通诱导，
cycle-5/6 全部无弦环），只用 Train 拟合词表与支持度统计，然后在固定表示（线性 / XGBoost / GINE）
上量化"粗结构 / 细结构 / 真实父子关系 / 统计支持"各自的增量。

回答六个问题：见 [REPORT.md](REPORT.md) §0。

* 协议（执行前冻结）：[PROTOCOL.md](PROTOCOL.md)；机器可读版本 `configs/protocol.json`
* 结果与认识更新表：[REPORT.md](REPORT.md)
* 资源/环境/提交/数据哈希总账：`analysis/provenance.json`

## 关键数字

| 阶段 | 产物 | 结论一句话 |
|---|---|---|
| A | 5 420 485 个实例；COARSE 754 列 / FULL 3 652 列 | Test 上 FULL 词表只覆盖 76.8% 的实例 |
| B | 8 个线性臂（C=0.03） | `B0→B1→B2` 方向稳定（Test 上 +0.011 的 `B2−B1` CI 不跨 0），打乱层级/支持度不掉点 |
| C | 支持度四等分、12 个家族、36 张图、子群、scaffold bootstrap | 支持度正则确实压缩低支持系数（0.30×），但只值 +0.0013 |
| D | `XGB_STRUCT` | Test +0.0445 [+0.0193, +0.0697]，Valid −0.0038 |
| E | `G-M` / `G-STRUCT-U`（A100，0.35 GPU-hour） | Valid +0.0104 / Test +0.0103，CI 均跨 0 |

## 运行

```bash
cd tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0
PYTHONPATH=code uv run python -m msds_phaseA --workers 10 --chunk 250   # 枚举（~73 s, 10 worker）
PYTHONPATH=code uv run python -m run_phaseA_vocab                        # Train-only 词表
PYTHONPATH=code uv run python -m verify_canonical_partition              # 规范键划分检验
PYTHONPATH=code uv run python -m run_phaseB_linear                       # 8 个线性臂 × 3 个 C
PYTHONPATH=code uv run python -m run_phaseD_xgb                          # 历史复现 + XGB_STRUCT
PYTHONPATH=code uv run python -m build_phaseE_bundle                     # 冻结 Phase E 输入
PYTHONPATH=code uv run --group lipo python -m run_phaseC_analysis        # 表 + 图
PYTHONPATH=code uv run python -m record_provenance                       # 资源/哈希总账
PYTHONPATH=code uv run python -m pytest tests -q
```

GPU 部分通过 `rr`（见 `AGENT.md` 与 `remote-research-runner` skill），命令在 `REPORT.md` §9。
`rr` 只部署已提交修订，因此任何 GPU 运行前必须先 commit。

## 目录

```
code/           枚举、词表、设计矩阵、四个阶段的入口脚本、规范键检验、provenance
configs/        protocol.json（与 PROTOCOL.md 对应的机器可读配置）
analysis/       全部表（csv）、结果（json）、逐分子预测（predictions/*.npz）、图（figures/）
data/           本轮的中间缓存（npz，git-ignored）；原始 OGB 数据在仓库根 data/ogb/
tests/          不变量测试（枚举 vs 暴力、置换不变、FULL→COARSE、规范键划分、bond 只计一次）
```
