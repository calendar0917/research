# Track: Tree Models and Explicit Graph Statistics

| 项 | 内容 |
|----|------|
| slug | `tree-statistics-boundary` |
| 状态 | `active` |
| 创建 | 2026-11-04 |
| 主线位置 | 独立 track（不属于 KSVD/GSN 主线），研究"显式结构统计 + 树模型"的能力边界 |

## 问题（一句话）

不同图预测任务中，哪些显式结构统计足以支持树模型预测？性能受**局部模式数量、
结构范围、输入信息和训练样本量**怎样影响？

第一轮只回答 Lipophilicity 上的一个受控切片：
在一套固定的 76 列全局组成/拓扑统计 `G0` 之上，依次加入
rooted typed-WL 局部模式（二值 / 计数 / 更大半径）与标准 Morgan 计数指纹，
在固定 XGBoost 配方和三个已发布随机划分上，Validation/Test RMSE 如何变化，
以及这些变化在 N=1,000 / 2,000 / 3,360 时如何随样本量移动。

## 范围

做：

- 只做 Lipophilicity（ChemBench 已发布 `Lipop.random122/123/124`）；
- 只用 XGBoost 回归（一套冻结超参数，无搜索、无早停）；
- 五个预注册输入表示：`G0` / `WL2_BIN` / `WL2_COUNT` / `WL3_COUNT` / `MORGAN_COUNT`；
- 三档训练样本量（嵌套前缀）；
- 完整 CPU 正确性门槛 + 表达能力边界诊断。

不做：

- 不训练任何 GNN/MLP，不换 LightGBM/CatBoost；
- 不加 F3 结构关系特征，不加 Optuna/网格/早停；
- 不扩展到 ZINC / AQSOL / Peptides / MolHIV；
- 不生成新的 Validation/Test 划分；
- 不把 Test 分数用于选择臂、调参或决定下一轮。

## 协议 / 评估

- `protocol_id`：`tsb-lipo-feature-ladder-v1`（只增不改语义）
- 主指标：原始标签尺度 RMSE（越低越好）；辅助 MAE / R² / Train-Valid gap / 耗时
- Test 政策：`terminal`。45 个拟合全部在 Validation 评价；只有 15 个完整 Train
  模型在**冻结之后**一次性评分
- Test 暴露：`EXPOSED_PUBLIC_TEST`（Lipophilicity Test 标签在更早的
  structural-sufficiency 轮次已被读取，本轮不是盲测）
- 控制面：`uv run research run tsb_lipo_feature_ladder_v1`，运行落在
  `tracks/tree-statistics-boundary/runs/`，提升后进入 `records/runs/`

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 / 协议预注册 | 完成（`PROTOCOL.md`、`configs/`、`configs/feature_spec.json`） |
| 数据审计与特征实现 | 完成（3 套划分 hash 核对，五臂特征，CSR 缓存） |
| CPU 正确性门槛（11 项 + pytest 20 项） | 全部通过 |
| P1 完整 Train 15 模型 | 完成 |
| P2 样本量曲线 30 模型 | 完成 |
| P3 Test 前冻结 + 一次性 Test | 完成（`EXPOSED_PUBLIC_TEST`） |
| 报告（表 A–D、成本、限制） | `REPORT.md` |

## 第一轮结果入口

| 路径 | 用途 |
|------|------|
| `REPORT.md` | 表 A–D、成本、限制、异常、Test 使用记录 |
| `analysis/tables/` | 表格 CSV/JSON（报告数字唯一来源） |
| `analysis/test/test_summary.json` | 15 个完整 Train 模型的 Test 结果与哈希 |
| `analysis/freeze/pre_test_freeze.json` | Test 前冻结清单 |
| `analysis/post_freeze_edits.json` | 冻结后分析代码修改与未变项核验 |
| `records/runs/` | 不可变科学记录 |

## 入口

| 路径 | 用途 |
|------|------|
| `code/tsb_data.py` | 划分与标签（hash 校验，Test 默认不读） |
| `code/tsb_mol.py` | 唯一分子图属性提取（G0 与 WL 共用） |
| `code/tsb_g0.py` / `code/tsb_wl.py` / `code/tsb_morgan.py` | 三个特征块 |
| `code/tsb_features.py` | 缓存与五臂装配 |
| `code/tsb_model.py` / `code/tsb_runner.py` | XGBoost 与各 stage |
| `code/tsb_gates.py` / `tests/` | 正确性门槛（runner 与 pytest 共用） |
| `code/tsb_tables.py` | 表 A–D |
| `configs/tsb-lipo-feature-ladder-v1.yaml` | 机器可读冻结配置 |
| `protocols/` / `studies/` | 控制面 protocol / study |

## 禁止

- 与其它 track（KSVD/SSRC/WL-kernel）结果混表横比；输入与协议都不同
- 把本 track 的 run 记成 KSVD 结果，或在 `tracks/ksvd/` 下写记录
- 未经新的 `protocol_id` 修改臂定义、G0 列、词表规则、XGBoost 参数或样本量
- 用 Test 排名挑选下一轮方向

## 下一步

第一轮已按预算完成并停止。等待研究者给出下一条实验指令；本轮不自行启动第二轮。
