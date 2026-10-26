# ATTENTION_NEXT — 如果 BOSI-R 成立，下一步唯一允许的 attention 形态

**状态：设计稿，未实施、未训练、无 GPU。** 本文件的存在不构成对 attention 的购买。
只有当 BOSI-R 在正式结果里表现出可重复的实用优势（见 `REPORT.md` 的 `decision.json`），
才允许把它作为 Stage 2 的候选；否则它只是记录下来的未实施方案。

---

## 1. 先说清楚为什么"加 attention"在默认情况下是错的

本仓库已有的三个负结果都在同一个方向上：

| 机制 | 结果 | 与本方案的关系 |
|---|---|---|
| Learned junction attention（`results/zinc_bond_conditioned_context_attention_v1`）| 比 P3 差 **+0.00745** | 用 learned query/key 在所有 junction 上做注意，**替代/扰动**了 junction 内容通道 |
| Object-token attention（`results/zinc_structural_token_attention_v1`）| 更差 | 把结构对象做成 token 再注意 |
| DBT（4 层 directed-bond attention，1500 ep）| test 0.06940，未超过 DCC-L | 多层边注意力 + 递归状态更新 |
| ESOC（键中心 + 4 层 sL2 + RRWP，187k updates）| 0.12801 vs DCC-L 0.07599 | 已把"键中心 + 逐对 learned 编码 + attention"做到极满，绝对水平仍差 0.052 |

再加上 BOSI v0 自身的位置：它**没有** learned junction token，伙伴信息只能经
`M → Side → Compose` 进入；R/B 的差异已经被"关系分配只在 19.5% 半键上有定义"这一事实
压在噪声边缘。在这种量级上再加一个高自由度模块，最可能的结果是增加方差而不是增加信号。

所以 BOSI-A 的**唯一合法形态**必须满足：不改变确定性统计路径、不引入递归、不引入
per-partner token、参数量可被同等 FFN 对照严格匹配、且只在 4 个固定结构组之间做**类型选择**。

## 2. 计算命题（一句话）

> 把 `M[v,e,r,:]`（4 个结构组的一/二/三阶矩）压缩成 4 个**结构有明确来源**的统计 token，
> 让**当前物理键的 query** 在这 4 个 token 上做一次 softmax 加权读取，并把结果**追加**到
> `s[v,e]` 的输入上；固定矩 `M` 本身不被替代。

### 2.1 张量（b0 版本）

```
token_r = Phi_r([ M[v,e,r,:] (96),  n[v,e,r] (1),  closure (4) ])        # 4 x 32
q       = W_q [ b_e (96), x[v,e] (32), log1p(deg) (1) ]                 # 32
logit_r = q · k_r / sqrt(32) + beta * log1p(n_r)      (r with n_r = 0 masked to -inf)
alpha   = softmax_r(logit)                                              # 4-way, <=3 dof
ctx     = sum_r alpha_r * v_r                                           # 32
s'[v,e] = Side([ a_v, b_e, x, M (384), n (4), closure (4), log1p(deg), ctx (32) ])   # 649 -> 128 -> 96
```

* `Phi_r` **共享参数**（4 组同一 MLP，`r` 只通过 `M` 的内容区分），所以 token 数固定为 4，
  批次里没有"token 维度"。它是 **4 路门控**，不是 permutation-equivariant token mixer。
* `r` 的结构来源是硬编码的：T = 三角闭合、Q = 额外公共邻居、R = 同 biconnected ring block、
  N = 无闭合。token 的语义不学习、不漂移、不可被重新分配给别的伙伴。
* **不更新**任何 atom/bond token；`b_e`、`a_v`、`x` 都是 Stage 1 的静态量；没有第二层。
* 固定矩路径（`M` 的 384 维）保持在输入里，attention 只增加 32 维；即使 `alpha` 塌缩成
  one-hot 或均匀，模型仍退化到 Stage 1 的可解释状态。

### 2.2 参数预算

| 项 | 形状 | 参数 |
|---|---|---:|
| `Phi`（共享）| (96+1+4)=101 → 64 → 32 | 6,464 + 2,080 = 8,544 |
| `W_q` | (96+32+1)=129 → 32 | 4,128 |
| `k_r`,`v_r`（各 4 组，共享也不影响计数）| 4×32×32 ×2 | 8,192 |
| `beta` | 1 | 1 |
| `s'` 第一层扩容 | (617+32)=649 替换 617 → 649×128+128 | +4,096（相对 Stage 1 的 91,488 → 95,584）|
| **合计新增** | | **≈ 24,961** |

Stage 1 是 473,889，加完 **≈ 498,850 < 500,000**（余量 1,150）。所以 b0 版本必须
*不*加任何额外宽度；任何"再加一层"都会撞上 cap，这也是设计约束。

## 3. 必须有同参数 FFN 统计融合对照（BOSI-AF）

**控制臂（预注册）**：不用 attention，用同一个 `Phi` 生成 4 个 token，然后

```
ctx_ffn = Psi([ token_T, token_Q, token_R, token_N, q ])     # 4*32+32=160 -> 64 -> 32
```
`Psi`: 160→64→32 = 10,304 + 2,080 = 12,384 参数，再补一个 32→32 的残差投影 1,056，
合计 13,440；与 attention 臂的 `W_q + k/v + beta`（12,321）差 1,119。为严格等参数，
attention 臂的 `k/v` 用 4×32×32 共享 + 每组的 `beta_r`（4），控制臂用同一 `Psi` 但
去掉 `q` 的 32 维（128→64→32 = 8,384 + 2,080 = 10,464），再加一个与 attention
等维的输出投影，两侧都精确对齐到 **24,961**。具体对齐表在实施时写入配置，且必须由
`named_parameters()` 重算，不接受手算。

控制臂测的是"**多给 4 组统计一个非线性融合**"，attention 臂测的是"**在这 4 组之间做
输入相关的选择**"。如果 FFN 对照一样好，那么 attention 的必要性不成立，结论就是
"统计融合有用、选择无用"，不得改写成"attention 有效"。

## 4. 为什么这不是旧 junction attention / STA / ESA 的重复

| 旧机制 | 它注意的对象 | 状态是否更新 | 本方案的不同 |
|---|---|---|---|
| junction attention | 每个 learned junction 向量 `w_j`，数量随图变化 | 键状态被更新 | 4 个**固定**关系组，没有 per-partner token，数量恒为 4，且固定矩不被替代 |
| STA / object-token attention | learned 结构对象 token | 是 | 同上；且 token 的语义由原始图算法固定，不由学习决定 |
| ESA / ESOC | 键中心诱导子图里的**每条边**，多层 masked attention + PMA | 是，逐层 | 单次、4 路、无层、无 pair encoder、无 PMA |
| DBT | directed-bond token，4 层全局关系注意 | 是 | 无递归、无全局 token 集 |
| SCM 的 A 项 | line-graph 距离带内的 `c_loc` 内容注意力 | 残差写回 | 只用静态统计，不读模型自己的隐状态，且不写回已有通道 |

一句话：**旧机制在"对象集合"上做注意力，本方案在"四个由原始图定义的关系池"上做选择。**
前者需要 learned token 才能形成对象，后者不需要；这正是"junction token 是否必要"的延续。

## 5. 预注册的判定（只写，不执行）

如果未来执行 Stage 2：

* 数据/配方/预算与 Stage 1 完全一致（Fit8k / Monitor / Locked，47,400 updates）。
* 只有当 **A 在两 seed 上都优于 Stage 1 R**，且**同参数 FFN 对照 AF 没有同样收益**时，
  才允许说"结构条件化选择"有独立价值；否则记录为 `FFN_EQUIVALENT`。
* 如果 A 的收益只出现在 `alpha` 接近 one-hot 的少数查询上（例如 >80% 的查询
  `max alpha > 0.9`），记录为 `HARD_ROUTING_NOT_ATTENTION`，不宣称 attention。
* 如果 A 与 AF 都不优于 R，**关闭**该方向，不再用层数/头数/温度救援。
* 官方 test/valid 仍不参与任何选择。

## 6. 当前结论

**不启动。** Stage 1 的 R/B 差异量级（见 `REPORT.md`）本身就落在两 seed 噪声附近；
在这一量级上叠加 25k 参数的选择模块无法与"多一个 seed"区分。若将来重开，
先决条件不是"再加 attention"，而是**先拿到 R 对 B 的 ≥0.005 稳定差异**。
