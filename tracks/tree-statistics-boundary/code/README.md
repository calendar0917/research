# code/ — `tsb-lipo-feature-ladder-v1`

依赖方向：`tsb_runner` → `tsb_features` / `tsb_model` / `tsb_gates` → 特征块
（`tsb_g0` / `tsb_wl` / `tsb_morgan`）→ `tsb_mol` → `tsb_data` → `tsb_common`。

| 模块 | 职责 |
|---|---|
| `tsb_common.py` | 冻结常量（划分哈希、词表大小、XGBoost 配方、种子）、指标、bootstrap、哈希工具 |
| `tsb_data.py` | 载入并校验三套已发布划分与标签；嵌套样本顺序；**默认不返回 Test 标签** |
| `tsb_mol.py` | 唯一的 RDKit 图属性提取（G0 与 typed-WL 共用）与 canonical token 精化 |
| `tsb_g0.py` | 76 列全局组成/拓扑统计（明确排除 217 描述符 / logP / TPSA / SA / QED） |
| `tsb_wl.py` | typed-WL 词表（每轮 ≤512 + OOV）、CSR 计数/二值矩阵、覆盖率 |
| `tsb_morgan.py` | Morgan radius-2、2048 桶计数指纹 + 哈希冲突诊断 |
| `tsb_features.py` | 特征缓存布局、五臂装配、manifest（哈希/维度/计时） |
| `tsb_model.py` | 唯一学习器：XGBoost 回归，参数漂移即报错 |
| `tsb_gates.py` | 11 项 CPU 正确性门槛 + 合成表达能力边界（runner 与 pytest 共用） |
| `tsb_runner.py` | 控制面 runner：`features / p0 / p1 / p2 / freeze / test / report` |
| `tsb_tables.py` | 表 A–D 与外部只读参照（纯后处理，不训练） |

运行方式见 `../REPORT.md` §13；缓存与模型二进制都不入库（见 `../.gitignore`）。
