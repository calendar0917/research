# Q 定点核对 — `zinc_cssd_nonlinear_binding_v1` 的 gid=3775 / gid=1424（2026-10-07）

**性质**：只读定点核对（旧产物 + 保存预测 + 冻结 Q soup 前向重放），复用
`zinc_long_cycle_audit` 与 `zinc_cycle_prototype_transfer_cpu_v1` 的历史审计结论，
**不训练任何新 Q 候选**，不接入 oracle / true c / 反解 k。核对脚本输出保存于
`results/zinc_cssd_basis_reuse_v1/q_spotcheck/`。被核对的 Q = 上轮共享 Q soup
（`Q_soup_state.pt`，8001 行 fit 池训练，`Q_meta.json` soup sha 校验通过）。

## 1. ID / 编号 / SMILES 源行映射（全部核对，无一凭文件名推断）

| gid（canonical_group_id） | position（=subset_index） | molecule_id | smi_line | canonical SMILES | n_nodes | 旧 8001 行 fit 池？ | 旧 select？ |
|---|---|---|---|---|---|---|---|
| 3775 | 3776 | train:3776 | 14677 | `COc1c(C=O)cc2c3c1CCCN3CCC2` | 17 | 否 | 是（select 内位置 380） |
| 1424 | 1424 | train:1424 | 40197 | `O=C(NCC1CCN(c2ncccn2)CC1)C(=O)Nc1cc2c3c(c1)CCC(=O)N3CCC2` | 33 | 否 | 是（select 内位置 135） |

- `molecule_id train:NNNN` 是 **0-based 位置编号**（与 subset_index 相同，label csv
  positional 断言通过）；`gid` 是 handoff `canonical_group_id`（语义完备 canonical
  certificate 组编号，**非位置编号**；gid≠position 从 position 3646 起普遍存在，
  属编号系统不同，非数据错位）。**历史审计的 train:3776 与本轮 gid=3775 是同一个
  分子**（position 3776，y=−20.2606、canonical SMILES、audit Table A 行逐项吻合）；
  gid=1424 与 audit train:1424 同为 position 1424（y=−20.7828 吻合）。两 gid 各自
  在数据集中只出现 1 次（无组重复）。
- 上轮 select 预测中两行均存在（所有 5 臂 seed0 同位置）。

## 2. 原始图 / 标签 / 分量构造 / k / c（与历史审计逐项对齐）

两行 y_stored、k=−6、c=−20.7898（c=(k−mu_cycle)/sigma_cycle，
mu_cycle≈−2.2e−5、sigma_cycle=0.28860，与 targets.npz 常量一致）：

| gid | y | g=y−c | ell=(logP−MU_LOGP)/sigma_logP | s=g−ell | label cycle 项 | 存储序 max basis | exact 最长简单环 | RDKit 最大环 |
|---|---|---|---|---|---|---|---|---|
| 3775 | −20.2606 | +0.5292 | −0.1746 | +0.7038 | −6 | **6（存储序无罚项！）** | 12 | 6 |
| 1424 | −20.7828 | +0.0070 | −0.5462 | +0.5532 | −6 | 12 | 12 | 6 |

历史审计已解释的部分，对两行的适用性：

- **cycle_basis 顺序依赖（audit Q9/§3）**：适用于 gid=3775（=audit train:3776，
  19 个顺序敏感标签例外之一：存储序 basis=6→无罚项，但标签 −6；该罚项是
  (图, 生成器节点序) 的函数，不是图的不变量）。**不适用于** gid=1424
  （存储序 basis=12 与标签 −6 自洽）。
- **长环尾部（audit Q1/Q5/Q6）**：两行都适用——y<−10 的 12/12 由长环罚项驱动；
  局部感受野外的严重度对 Q 不可见。
- **标签复现误差（audit Q3）**：gid=1424 属 11,981/12,000 精确复现行；
  gid=3775 属 19 个顺序敏感例外行（标签值本身与 GVAE 复制不一致，audit 已记录）。

## 3. topology25 输入与 Q 预测（原始值 + 旧轮 prep 逐位复现，见 q_spotcheck/spotcheck.json）

两行 T25（原始值；旧轮 prep 的 topo_fit_mean/scale 与 prep.npz 逐位相同）彼此只在
feat4（1.254 vs 3.353）、feat10（0.238 vs 2.231）、feat13（0.435 vs 2.548）、
feat14（0.238 vs 2.231；feat10≡feat14，见下）上不同，其余 21 维完全相同
（含 feat16=2.4619）。冻结 Q soup 前向重放：gid=3775 → q_raw=+0.0083；
gid=1424 → q_raw=+0.0069；两行 c 均 −20.79，|q−c| ≈ 20.80，即 Q 对这两个输入
给出 ~0 而非 −20.8。

**勘误（2026-10-07，定点输出落盘时发现）**：本节初稿的特征数值（4.66/6.57、
5.77/7.78、117.7/185.9、feat16=5196.8，“22 维相同”）未随任何保存产物产生，
与冻结 topology_features/T25_cache 逐位不符；已按 `q_spotcheck/spotcheck.json`
的可复现输出修正（差 4 维而非 3 维，feat16=2.4619）。q 重放值（+0.0083/+0.0069）
与其余各节全部数值不受影响。

附注（信息性，非本轮可修项）：旧 8001 行拟合池上 T25 列存在两对逐位重复列
（0≡15、10≡14）；该输入接口属冻结历史契约，本轮不据此改 Q 输入。

## 4. 旧 fit（8001 行池）中的完全相同 Q 输入：匹配数、c 冲突、相关预测

- **gid=1424：精确匹配 1 行**（position 1270），该行 **k=0、c≈+0.0001，
  与本行 k=−6 冲突**——同一 T25 向量在旧 Q 的训练池里携带的目标就是 ~0；
  Q 对两行输出同值（对 position 1270 = +0.0069，正是它学到的）。
  → **输入冲突（input conflict）**：25 维拓扑摘要无法区分这两个分子，
  任何 T25 的确定性函数都不可能同时输出 0 与 −20.79。这不是头部拟合遗漏。
- **gid=3775：精确匹配 0 行**——该 T25 类在旧 8001 行拟合池中不存在
  （它自己的单例类被划进了 select）。固定最近输入查询（L1，仅描述）：
  最近 fit 行 position 8325（L1 距离 4.11，k=0, c≈0）、次近 position 1270
  （8.22, k=0）、再次 position 2232（14.2, k=−2）。
  → **未覆盖类（unseen class）**：训练池中没有该输入的 −6 证据，Q 由邻近
  k=0 类外推到 ~0。
- 对照历史 prototype 审计（`zinc_cycle_prototype_transfer_cpu_v1`）：
  那轮的 Q 是 fulltrain（10k）对象，valid:0172 匹配到了 train:3776 单例类
  （覆盖行上的**函数拟合遗漏**）；**本轮的 Q 只在 8001 行 fit 池上训练，
  该单例类根本不在其训练集内**——两个结论对象不同、都成立，互不否定。

## 5. 定位结论（证据支持哪一类解释）

两行 Q≈0 而 c≈−20.79 的**已定位**成因组合：

1. gid=1424：**T25 输入冲突**（同向量、冲突 k）+ 长环尾部严重度超出 T25 表达力
   ——输入层不可分，非 ID/接线 bug，非头部拟合可在冻结 T25 接口内修复的遗漏。
2. gid=3775：**T25 类未覆盖**（单例类落在 select）+ 标签罚项本身顺序依赖
   （audit Q9 已回答）——同样不是 ID/接线 bug。

未发现任何数据/ID/接线 bug：subset_index/gid/smi_line/canonical SMILES/y/k/c
链条逐项一致；y、g、ell、s 恒等式在两行上精确成立；Q 输入构造与旧轮 prep
逐位复现（topo_fit_mean/scale 与 prep.npz 逐位相同）。因此**不需要最小修复**，
本轮按原目标继续：共享 Q 保持原配方（T_fit-only 重训一次），其 T25 顺序无关、
输入不可分与未覆盖限制如实写入报告。

## 6. 对 select 总绝对误差的贡献（与 confirm 分布分开报告）

- select（999 行，seed0 各臂）：gid=3775 与 gid=1424 的 |y−y_raw| 各 ≈20.8–21.0，
  **单行各占臂内 select 总绝对误差 ~13.6–14.6%，两行合计 ~27–29%**；
  select 内 |err|>5 的行共 3 行（两行即其中的 2）。
- confirm（1000 行，A/C00 seeds0/1）：两行不在 confirm；confirm 无 |err|>5 行
  （A_s1 分位 [p50 0.058, p90 0.175, p99 0.499, max 2.134]）。
- select MAE（0.144–0.153）与 confirm MAE（0.086–0.092）之间的差主要由这两行
  （及第 3 个 >5 尾行）造成：**两 holdout 的尾部构成不同**，不是训练过程差异。
- 全量 MAE 保持原样：所有样本都计入，未删除尾部，未以剔除后分数充当进步。
  （这两行的完整 y 误差分解 ≈ Q 缺失 20.80 + 体误差 ~0.05–0.1 的代数和，
  见第 2、3 节数值。）

## 7. 对本轮（zinc_cssd_basis_reuse_v1）的直接影响

- 本轮域划分后，两个尾部分子按 n_nodes 落入 source/target 池与否由冻结规则决定
  （不因误差/Q 表现调整）；T_eval 的构成同样不因这两行调整。
- Q 的局限（T25 输入冲突 / 未覆盖单例类 / 顺序依赖标签）对本轮两臂**对称存在**
  （共享同一 Q），主读出仍是完整 y_raw MAE；不据此调 Q、不加 cycle 输入、
  不重启 prototype 类候选（上轮已停止的决定不变）。

## 引用

- 旧产物：`results/zinc_cssd_nonlinear_binding_v1/{targets,prep,fold}.npz`、
  `Q_soup_state.pt`、`runs/*_s0/select_predictions.npz`、
  `runs/{A,C00}_s1/confirm_predictions.npz`（全部只读）。
- 历史审计：`notes/zinc_long_cycle_audit.md`（Q3/Q9、Table A/C）、
  `notes/zinc_cycle_prototype_transfer_cpu_v1_analysis.md`。
- 本核对输出：`results/zinc_cssd_basis_reuse_v1/q_spotcheck/`（JSON + 逐位可复现命令）。
