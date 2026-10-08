# zinc_cssd_g_component_attribution_v1 — 冻结 CSSD 消费者 g 组件泛化归因

状态：预注册（分组定义、指标与判读边界在**任何标签分析**之前冻结；export 阶段完全无标签）。
日期：2026-10-09。Track：ksvd；Study：zinc-context-gap。
代码：`tracks/ksvd/experiments/luyin16/zinc_cssd_g_component_attribution_v1.py`。
Runner：`zinc_cssd_g_component_attribution_v1`（非 terminal，official valid/test 永不加载）。

## 1. 科学问题（本轮唯一问题）

当前四个冻结消费者（`zinc_cssd_consumer_replacement_v1` 的 RAW_s0/RAW_s1/DICT_s0/DICT_s1，
g = ell + s）的**样本外 g 误差主要来自哪个组件**（ell 还是 s），该误差是否有具体、稳定、
可解释的粗粒度结构—化学条件，以及这是否为"字典作为结构—化学表示核心"的下一阶段设计
提供实质依据。本轮不是新的结构统计量搜索，不是模型训练，不产生新候选。

## 2. 对象与信息流（全部复用，零重训）

- 四个冻结 soup（RAW/DICT × seed 0/1）+ 冻结 CSSD 基底（U/common_rms/Dbar，radius-2
  untyped phi65、q=1、32 atoms、top-8、10-step tied IHT）+ 共享冻结 Q(topology25)。
- 部署信息流（`notes/zinc_cssd_graph_information_sufficiency_v1.md` §2 已以代码核实）：
  phi65（每 root 纯拓扑 65D，无化学）→ `cssd_decode`（alpha 被丢弃，消费者只见重构
  phi_hat）→ tuple 特征 [phi_hat65; onehot28(root); onehot28(neighbor); onehot4(bond)]
  + 真实 J incidence → M 编码器 → W_loc 注入 fusion[0] → + Sem108(108D)+size2 →
  bridge → R_G(814D) → ComponentReader 39→2 → (ĥ_ell, ĥ_s)，g_raw = ĥ_ell + ĥ_s = h
  → y_raw = h + q_raw。
- **绕过字典的真实通道**（如实列出）：化学 one-hot28/28/4、J incidence、Sem108+size2、
  静态 relation、global_context、topology25、reader、Q 全部不经过 CSSD；字典只供给
  tuple 特征中的 phi65 块。RAW 臂同骨架保留原始 phi65（局部信息严格更多、无基底重构损失）。

## 3. 复用与导出（避免重复前向）

- **fit 侧**：四个 run 的 `fit_predictions.npz` 已含逐行 `ell_hat/s_hat/h`——直接只读复用，
  不重前向。
- **dev 侧**：历史 dev npz 按设计无逐行分量预测；本轮对四个冻结 soup 在 1999 个 dev 行上
  做一次真实前向导出（`repl._predict_rows`，CPU，秒级），导出 `ell_hat/s_hat/h/gid`。
- 导出校验（无标签，先于标签分析）：
  1. h 重放对已存 dev npz `max|Δh| ≤ 1e-4`（fp32 保存精度；fit 侧同时校验 h/ell_hat/s_hat）；
  2. 严格重构恒等式 `max|h − (ell_hat + s_hat)| ≤ 1e-5`（预期 ~2.4e-7）；
  3. gid 对齐：导出行序 == dev_idx == 已存 dev npz gid；fit npz gid == targets gid[fit_idx]；
  4. 目标恒等式：y = g + c（≤1e-9）、g = ell + s（≤1e-9）；
  5. canonical-SMILES 组不跨 fit/dev（committed 表复核）。

## 4. 冻结的诊断（先于任何标签读取定义）

约定：e_ell = ell_hat − ell，e_s = s_hat − s，e_g = h − g（"预测 − 真值"符号，与 triage 一致）。

1. **组件合成表**：fit/dev × 四 run 的 MAE(ell)/MAE(s)/MAE(g)、signed bias(ell/s/g)、
   异号比例（e_ell·e_s < 0）、triangle gap = mean(|e_ell|+|e_s|−|e_g|)（≥0 为有利抵消）；
   两 seed 分开 + 两 seed 平均；指标先分 seed 算再平均（与 triage 相同）。
   k 组（k=0 / k=−1 / k=−2 / k≤−3）贡献 C_s/C_ell/C_g 以**全 split 行数 N** 归一，
   加回检查精确（残差 <1e-9）。
2. **k=0 常见分子子群**（dev 主、fit 辅）：三个单因子粗粒度分组（互斥、覆盖全部 k=0 行）：
   - 节点数三分位：界 = 全 10000 训练行节点数的 1/3、2/3 分位（无标签、预注册规则）；
   - 原子组成（优先级 halogen > S,P > N,O > C-only）：由原始 x id 映射
     C={0,4}, N={2,8,10,11,12,13}, O={1,7}, F={3}, S={5,14}, Cl={6}, Br={9}, I={15},
     P={16}；映射以 committed canonical SMILES 逐分子元素计数交叉验证（总偏差 ≤10 原子/
     元素，实测 ≤5）；
   - 键组成：triple（edge id 3）/ double-or-aromatic（edge id 2、无 3）/ single-only
     （仅 id 1）；id3 ⇔ '#'、id2 ⇔ '='或芳香环 已实证。
   每组：n、组内 MAE_s/MAE_ell/MAE_g（两 seed 平均/臂）、贡献 C_s（全 N 归一）、
   Σ|e_s| 组内份额、DICT−RAW 配对 ΔMAE_s / ΔMAE_ell。
3. **集中 vs 分散**：dev k=0（及全 dev）按两 seed 平均 |e_s|/|e_ell|/|e_g| 的 top-20 行
   份额（均匀参照 20/N）；子群 Σ|e_s| 份额表；**配对不确定性**：canonical-SMILES 组
   bootstrap 2000 次、seed 20261022，臂与 seed 共享组重采样、每次抽样先平均两 seed
   （与 consumer replacement bootstrap 同方案）——(a) k=0 dev 总体 ΔMAE_s/ΔMAE_ell
   的 CI；(b) 每个子群 Δ(DICT−RAW) 的 CI 与 子群−补集 MAE_s 对比的 CI。
   **不作因果解释**；只报样本量、贡献、配对不确定性与方向一致性。
4. **信息流归因陈述**（不新算）：汇总 phi65→CSSD→phi_hat 替换范围与绕过通道；引用
   既有 promoted 证据——triage §5（74.0% MAE_y = k=0 e_g；缺 dev 逐组件定位）、
   consumer replacement（DICT 承接 −0.00102 CI 跨零；唯一字典干预有害可换；
   RAW 局部信息严格更多但不更好）、graph information sufficiency（分支 B：W 见证
   无增量信号）、chemistry component supervision（新 COMP 模型 s 为 dev-gap leader，
   0.083 vs 0.054）、local dictionary component supervision（D 劣于 M，差距在 s 通道），
   说明哪些数据支持、哪些不支持"需要重新定义结构—化学字典"。

## 5. 判读边界（冻结）

- **dev 是历史开发比较集**（旧 select∪confirm，多次用于开发）：本轮只做探索性定位，
  不构成任何新确认；不包装成"全新确认"。
- fit 侧分量 MAE 只有域内含义；fit→dev 差值是泛化差的描述统计，不是因果分解。
- 组件 MAE 永不加总成 g 预算（抵消存在）；g 偏差只对 g 整体报告，不分解到组件。
- 若子群格局两 seed 不同向或 CI 覆盖零：明确写"数据无法区分竞争解释"，不制造候选。
- 禁止：新特征搜索、追加 W 统计、架构调参、重训 backbone、读取 official valid/test。

## 6. 预算与执行

全本地 CPU；export（四 soup × fit+dev 前向重放 + 导出）秒级；analyze 纯 numpy 秒级。
0 GPU；不触发 rr。产物：`dev_components_{run}.npz`、`structure_assignments.npz`、
`export.json`、`component_summary.json`、`subgroup_tables.csv`、`concentration.csv`、
`bootstrap.json`、`source_manifest.json`；REPORT 写入 `results/zinc_cssd_g_component_attribution_v1/`。
