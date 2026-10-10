# CSCL-v0 标签盲数据审计结论(2026-10-10)

Round: cscl-v0 · 依据: `notes/v0_protocol.md` §8 · 数据: official-train 10000(仅 train)
脚本: `code/run_cscl_v0.py audit` · 落盘: `results/cscl_v0_units_audit/audit.json`

## 结论:**触发线全部未触发,RINGCHAIN-v0 划分可用,无需后备方案**

| 预注册触发线(v0_protocol §8) | 阈值 | 实测(fit_inner) | 判定 |
|---|---|---|---|
| singleton 类型占比 | >30% | **0.0%** | PASS |
| 零关系分子占比 | >50% | **0.60%** | PASS |
| 单元大小 P95 | >15 原子 | **9** | PASS |

## 关键统计(fit_inner 7200 / dev 2000)

| 量 | fit_inner | dev |
|---|---|---|
| 每分子单元数 mean / p50 / p95 / max | 5.51 / 5 / 8 / 13 | 5.57 / 6 / 8 / 12 |
| 单元大小 mean / p50 / p95 / max | 4.20 / 5 / 9 / 26 | 4.18 / 5 / 9 / 21 |
| 单元总数(环系, 链) | 39689 (16807, 22882) | 11136 (4658, 6478) |
| 每分子关系数 mean / p50 / p95 | 4.51 / 4 / 7 | 4.57 / 5 / 7 |
| 词表 known / total ids | 168 / 179 | —(复用 fit 词表) |
| UNK 单元占比 | 0.54% | 0.91% |
| top-10 类型对单元覆盖率 | 77.0% | 77.5% |

## 划分/对齐正确性

- 原子覆盖率与键唯一归属由测试保证(`test_partition_covers_all_atoms_exactly_once`,
  `test_every_bond_has_unique_correct_attribution`)。
- SMILES↔PyG 图对齐交叉校验:近似原子计数(含芳香小写与括号原子)
  **10000/10000 行完全一致(max_abs_diff=0)** —— committed canonical SMILES 表
  与 PyG subset 顺序确认为同一行序,分组划分键有效。
- 跨分子复用充分:top-10 类型覆盖 ~77% 单元;dev UNK 率 0.9% 很低。

## 对 v0 的含义

- 关系集合非退化:平均每分子 4.5 个单元间关系,B/C 臂的机制对照有实义对象。
- 词表规模 168 known 类型:embedding 表小,类型级 α 可直接做跨分子比较。
- 无需触发预注册后备分支(链单元分支切分);划分参数不做网格搜索。

## 后续

seed-0 冒烟(CPU)→ `res` GPU1 正式 seeds {0,1}(v0_protocol §4)。
