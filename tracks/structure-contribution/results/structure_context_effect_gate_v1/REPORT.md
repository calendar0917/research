# REPORT — structure-context-effect-gate-v1 (Stage B)

**问题**：对一个冻结的 Lipophilicity 预测器（BJOC-SSL Arm P，seed 0/1），同一真实局部
结构 `S` 的模型内部响应，能否被真实上下文关系稳定解释，并在控制组成与非线性读出后
保留增量价值？

**标签**：`NO_JUSTIFIED_CONTEXT_EFFECT_METHOD`（按 `PROTOCOL.md` §7 预先冻结规则，
无事后阈值修改）。本轮**不训练任何新模型**，不读 valid/test 标签，不更新权重。

---

## 1. 冻结输入与边界（全部核验通过）

| 输入 | sha256（运行时复核一致） |
|---|---|
| Stage A `instances.npz` | `80970087965952baf63821e965418380aca88fc0463286bb3ff6bef10175d327` |
| `objects_train.pt` | `6d681cb0eaaab8c448564993c3740b24204e52e9e430f8c366189aa0061c7c86` |
| `soup_P_s0.pt` | `1a8292acf0477ce89899d48d913f0c50d651eab00c4d92d757580f847836a304` |
| `soup_P_s1.pt` | `505f840ea90783bdbd6761e2290e4f875670bcdf3f293a8aa5d9225582c542b0` |
| `standardizer_v1.json` | `b7527e857b365804c80cbd7bb3dada4feecf78a03080beea9d6b9cbdc15215c3` |

- 模型：`LipoConfig`，**419,713 参数**（与冻结 checkpoint 一致），`organization_mode=NULL`。
- 预测重放（valid，无标签）：seed0 max|Δ|=1.79e-6，seed1 max|Δ|=1.43e-6；重放文件 sha 与
  `prediction_manifest_valid.json` 一致。
- 边界：`labels_loaded=false`、`valid_test_labels_loaded=false`、`weights_updated=false`、
  未接触 ZINC official test/Locked；未修改任何 Stage A / BJOC / CSCL / HardConcept 冻结产物，
  未修改 `notes/research_core.md`。所有实现细节见 `code/structure_context_effect_gate_v1.py`，
  预注册见 `PROTOCOL.md`。

## 2. 分析对象与支持（在读取任何响应之前冻结）

- 93 个 Stage A C4 类型全部进入池；池 = 组内至少 2 个 CT、且每个 CT 至少由 2 个不同分子
  独立支持的实例。
- **有效实例 2521 条 Bond 记录 → 2020 个分析单位 `(mol, S, CT)`（同单位多键取均值）**，
  519 个分子、222 个 Murcko 骨架。
- 支持门槛（§3）全部通过：骨架 222 ≥ 30；跨骨架类型 43 ≥ 10；双骨架组 119 ≥ 30。
- 单位内 CT 变化特征 27/32（其余 5 维在 S 内恒定）；CT 不是 S 的复制。

## 3. 响应定义与正确性（响应 = 冻结模型 Bond 来源替换响应）

对目标 Bond 的编码 `b_i` 做预定义自然编码替换（REF-A/`anchor`、REF-B/`mean`、
REF-C/`random` 8 个同键类、非本分子 donor），只重算 `b→w→W/V/n_b→c→Z→reader`，
不触碰冻结模型参数。正确性诊断（两 seed）：

| 检查 | seed0 | seed1 |
|---|---|---|
| 无覆盖前向 vs `model.forward` 逐位一致 | = | = |
| 恒等覆盖 `identity_max_abs` | 0 | 0 |
| 非目标 Bond 变化 `b_unexpected_max` | 0 | 0 |
| 非相邻 Junction 变化 `w_unexpected_max` | 0 | 0 |
| `override_not_applied` / `localisation_violations` / `nonb_modified_violations` / `identity_violations` | 0 / 0 / 0 / 0（519 分子） | 同 |
| RDKit ↔ 模型 Bond 端点/编号/键型对应 | 3360 分子 / 90,505 原子 / 98,554 键 / 138,654 Junction 全部一致 | 同 |

因此**实现层面的响应定义是精确、可复现的**（`impl_valid=true`）。

## 4. 主结果（B0 = S only；B1 = S + 全局控制；B2 = B1 + 真实 CT；骨架分组 OOF ×5）

MAE（小写更好）；`Δ(2−1)=MAE(B1)−MAE(B2)`，CI 为骨架 bootstrap 2.5–97.5%；
`p95` 为 100 次 CT 打乱零对照（组内 `(S,stratum)` 置换）分布 95% 分位。

| seed | 参考 | B0 | B1 | B1d | B2 | B2x | Δ(2−1) | CI | p | p95 | Δ(2−1d) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | anchor | 0.0418 | 0.0448 | 0.0434 | 0.0440 | 0.0392 | **+0.00079** | (−0.00089, +0.00243) | 0.010 | 0.00050 | −0.00058 |
| 0 | mean | 0.0413 | 0.0451 | 0.0445 | 0.0442 | 0.0390 | +0.00087 | (−0.00087, +0.00263) | 0.139 | 0.00101 | +0.00027 |
| 0 | random | 0.0432 | 0.0466 | 0.0468 | 0.0455 | 0.0420 | +0.00105 | (−0.00045, +0.00257) | 0.059 | 0.00102 | +0.00121 |
| 1 | anchor | 0.0418 | 0.0444 | 0.0423 | 0.0423 | 0.0377 | **+0.00209** | (+0.00045, +0.00368) | 0.020 | 0.00185 | −0.00003 |
| 1 | mean | 0.0404 | 0.0439 | 0.0418 | 0.0408 | 0.0367 | +0.00311 | (+0.00109, +0.00544) | 0.010 | 0.00245 | +0.00099 |
| 1 | random | 0.0403 | 0.0428 | 0.0421 | 0.0402 | 0.0380 | +0.00262 | (+0.00132, +0.00398) | 0.010 | 0.00227 | +0.00198 |

要点：

1. **Δ(2−1) 六格全部为正**（0.0008–0.0031，约 B1 MAE 的 1.8%–7.0%），但**只有 seed1 的
   三个参考 CI 不含 0**；主口径 anchor 在 seed0 CI 跨 0。`improvement(seed0)=false`。
2. **B0 往往优于 B1/B2**：加入全局控制后 MAE 反而变差，B2 的增量不足以回到 B0 水平。
   即"组成控制"在本数据上不是被解释掉的混淆，而是损害预测的过参数化项。
3. **编码距离反证**：`B1d`（B1+替换编码距离）在 anchor 上完全吸收 B2 的增量
   （seed0 −0.00058、seed1 −0.00003，即 B1d ≥ B2）；mean/random 上 B2 仍略优
   （+0.0003–+0.0020）。主口径的"CT 增量"可被一个平凡距离协变量解释。
4. **`B2x`（精确 CT 身份 target encoding，描述性）稳定大幅更好**（+0.0046–+0.0072，
   OOF R² 0.30 vs B2 0.16–0.19），说明粗糙 32 维 CT 不是主要瓶颈；但精确 CT 身份
   信号可能来自高基数 CT 的记忆化，不能当作"真实关系"证据（§6 限制）。
5. 响应本身可预测性有限：B0 OOF R²≈0.18–0.20，B2≈0.16–0.19。

## 5. 反证与稳定性（全部预注册）

| 检查 | 结果 |
|---|---|
| **跨参考稳定性（同 seed）** | 最小 Spearman 0.810、最小符号一致率 0.783 → **通过**（响应对参考定义不敏感） |
| **跨 seed 复现（同参考）** | ρ=0.484/0.462/0.423，符号一致率 0.677/0.669/0.669 → **低于冻结阈值 0.5/0.7** |
| **全部 15 个 (参考×seed) 两两对** | 最小 ρ=0.390（anchor@seed1 vs random@seed0），最小符号一致率 0.627 → **不通过** |
| shuffle 零对照 | anchor：两 seed 的 Δ 均超过各自 p95（+0.00029 / +0.00024）；seed0 mean/random 未过（p=0.139/0.059） |
| 非平凡性 `nontrivial` | seed0 只有 random 过线（0/2 无效）；seed1 三参考全过 |
| 高频结构集中度 | 前 1/3/5 高频类型对总改善贡献 3%–44% / 12%–40% / 36%–98%（随 seed/参考变化很大） |
| 去前 3 高频类型重跑 | seed0 anchor +0.00100、mean +0.00077、random 崩到 +0.00006；seed1 +0.0025～+0.0032 |
| 去前 3 骨架重跑 | 六格 Δ 全为正（+0.00099～+0.00131），跨骨架结构上稳健 |
| 描述性方差分解 | 同一 S 内部、CT 之间的方差占 S 内方差 44%–56%（`η²_CT|S`≈0.44–0.56）——原始响应描述上确实随 CT 变化，但该响应跨 seed 不可复现到冻结要求 |
| 编码距离与响应相关 | −0.13–+0.04，距离不是原始响应的驱动因素（但它作为特征可替代 CT） |
| 独立性 | CV 按 Murcko 骨架分组；分子/骨架级去重；打乱对照保持组内分布 |

## 6. 决策轨迹（PROTOCOL §7，按顺序）

```
1. response_defined 失败（跨 seed 中位 ρ=0.462 < 0.5，符号一致率 0.669 < 0.7） -> NO_JUSTIFIED
   （实现层面 impl_valid=true：恒等=0、对应通过、重放通过）
2. reference_stable 失败（15 对全读法: min ρ=0.390、min sign=0.627）            -> NO_JUSTIFIED
   （同 seed 内读法通过: min ρ=0.810、min sign=0.783）
---（以下仅作参考，前两条已命中）---
3. any_positive = true（六格 Δ>0）
4. both_seeds = false（seed0 improvement 0/3，seed1 3/3）                       -> INCONCLUSIVE
5. support_ok = true
6. null_exceeded(anchor)=true/true；nontrivial(seed0)=false, (seed1)=true      -> INCONCLUSIVE
```

**最终标签 `NO_JUSTIFIED_CONTEXT_EFFECT_METHOD`。** 敏感性说明：若把"跨 seed 复现"
从 `response_defined` 中拆出、只当作效应层面的不确定性（即按任务描述中 INCONCLUSIVE 的
"两模型 seed 不一致"读法），结论会变为 `CONTEXT_EFFECT_INCONCLUSIVE`；**没有任何一种
读法会得到 `SIGNAL_PRESENT`**。两种读法对研究路线的实际含义一致：现阶段不能据此建立新方法。

## 7. 五个必答问题

1. **同一种 S 的模型响应是否确实存在可重复的上下文差异？**
   实现上完全可重复（恒等控制逐位为 0、查前向确定性）；但**两个独立训练的冻结 seed 之间
   的原始响应只有 ρ≈0.42–0.48、符号一致率≈67%**，低于预注册复现门槛。描述性上 CT 解释了
   S 内方差的 44%–56%，即"响应随上下文变化"在单模型内成立，但**不是跨模型可复现的稳定差异**。

2. **是否超过组成控制？**
   边际上超过但不可靠。六格 Δ(B2−B1) 全为正（1.8%–7.0%），seed1 三参考 CI 均>0；
   seed0 全部 CI 跨 0。且 B0（仅 S）在多数格优于 B1/B2，说明加入全局控制本身有害；
   主口径 anchor 的增量被 B1d（编码距离）完全吸收。

3. **真实 CT 比打乱对照多提供多少信息？**
   主口径 anchor：Δ 超出 shuffle p95 约 0.00024–0.00029（绝对预测单位 ~0.08–0.21%），
   两 seed 都过线但幅度很小；seed0 的 mean/random 不能过线（p=0.139/0.059）。
   精确 CT 身份的 `B2x` 大得多（+0.0046–+0.0072），但那是高基数身份信号，
   可能含记忆化，不能充当关系解释。

4. **是否在独立骨架组复现？**
   结构支持层面：是。CV 与 bootstrap 都按骨架分组，222 个骨架、43 个跨骨架类型、
   119 个双骨架组；去掉最高频前 3 个骨架后六格 Δ 仍全为正。模型 seed 层面：否——
   方向一致（Δ 全正）但效应量与置信度不一致，原始响应跨 seed 只中等相关。

5. **是否足以支撑一个跨骨架的新学习方法，还是应停止该论文命题？**
   **不足以支撑，应停止/重述当前命题。** 证据链：①响应定义在跨 seed 复现上不达预注册门槛；
   ②增量对全局控制不可靠（seed0 全部不显著），主口径增量被编码距离解释；③B0 常优于 B1/B2；
   ④打乱零对照的通过幅度小且随 seed/参考翻转。若坚持该方向，前置条件是把"响应"重定义为
   跨 seed 可复现的量（如多 seed 集成读出或直接对目标训练的可解释读出），再重新做本门禁；
   在此之前不应投入新方法论文。

## 8. 复现

```bash
uv run --group lipo python \
  tracks/structure-contribution/results/structure_context_effect_gate_v1/code/structure_context_effect_gate_v1.py all
uv run pytest -q tracks/structure-contribution/results/structure_context_effect_gate_v1/tests
```

产物：`summary.json`（含支持、B0/B1/B2/B1d/B2x、CI、shuffle、稳定性、决策轨迹）、
`analysis/`（`analysis_pool.npz`、`pool_manifest.json`、`bond_correspondence.json`、
`responses_seed{0,1}.npz` 与 diag、`features.npz`、`predictor_metrics.json`、
`shuffle_null.json`、`model_verification.json`）。测试 15/15 通过（前向一致性、恒等/局部性、
键对应、参考抽样、打乱保持组内分布、决策规则全部路径）。

## 9. 已知限制（如实报告，不隐藏）

1. 冻结模型是 `NULL` 组织模式，单键替换经全局均值通道影响所有 Bond，响应**不是**单键
   局部贡献；报告所有结论均限于"冻结模型的 Bond 来源替换响应"。
2. CT 仅是 radius-2 邻域 + 少量关系轴（32 维中 27 维在 S 内变化），长程上下文不在定义内。
3. 数据集高度同源，快照骨架分组后仍可能存在系列化合物共享局部环境。
4. 参考编码是模型内部反事实，不是真实化学编辑，结果不可外推为化学因果效应。
5. 只有一个性质（Lipophilicity）与一个模型族（BJOC-SSL Arm P），不外推。
6. 未使用 IG / 单点遮蔽一致性作为独立忠实度证据（本轮完全不用 IG）。
7. `B2x` 的 target encoding 按 CV 外折计算，但高基数 CT 仍可能记忆化，只能作描述性对照。
