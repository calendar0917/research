# PAPER_SECTION — 结果段落（中文正文版 + English）

写作约束（先声明，避免审稿人和我们自己都误读）：

- 段落里**不允许**出现「所有家族都正向」「S2 独占 93%」「MolWt 不变」「唯一原子贡献」
  「LCRC 独有」这类表述；反例（酯、磺酸酯、芳醚反向、MolWt 尾部）**写在正文里**，
  不剔出主统计。
- 所有平均响应都用**分子权重**（`analysis/weighted_response.csv`），平铺值若要出现必须标注。
- 实验 logD、WC_LOGP、MolWt 三类参照始终分写，不合并成一个「真值」。
- 数字对应的文件在每段末的 `[来源]` 中给出。

---

## 中文版

### R1 方法与可追溯性

我们把被解释对象定义成一个**完全显式**的计算：对每个原子 $v$，模型只消费
逐原子特征（AF51）、逐键入射和 `b_sum`、一跳联合码计数 `nbr`、二跳的
`(path_cat, path_end63)`、closure 与端点内容项，以及 9 个逐原子标量；全图再加 42 维统计。
局部读出为 $h_v=\mathrm{LN}(\mathrm{MLP}(x_v))$、$q_v=f(h_v)$，预测为
$p(G)=\mu_y+s_y[\sum_v q_v+\rho(\bar g,\overline{h},\overline{h^2},\log(1+n))]$，
其中 $\overline{h^2}$ 是逐分量平方后再取平均。由此得到两个可以直接使用的性质：
（i）相同 `global42` 与相同「逐根统计签名」多重集在同一冻结参数下给出相同预测；
（ii）对有原子映射的改线，令 $U$ 为签名被改变的根集合，则
$\Delta p=s_y[\sum_{v\in U}\Delta q_v+\Delta\rho]$，未变根的 $q$ 不变。
第二条同时说明为什么不能做「原子贡献分摊」：$\rho$ 通过池化读全部根，
在已有记录里 $\Delta\rho$（+0.6393）与 $\Delta q$（+0.5203）同量级。
`[来源] METHOD.md §2–§4；dissection/analysis/attribution/TABLES.md]`

我们自己的一处边界证据也来自同一机制：5 对**非同构**分子拥有完全相同的逐根签名与
`global42`，模型给出的预测差 $\le 3\times10^{-7}$。因此「签名」是**对这个模型的计算输入充分**，
不是完整化学表示，也不保证签名不同就能被数值区分。
`[来源] s3_locality/analysis/alias_audit.json]`

### R2 实验 logD：偏好依赖家族与可匹配人群

在冻结的实验 logD 模型上，最强的「训练相关结构偏好」出现在**等剂量可匹配**的酰胺子群：
Test 上 62 个分子、110 个焦点对（分别约为该家族主队列的 38% 与 21%）。
该子群的端点差为 $J_C=0.146$，相对匹配对照的条件 S2 恢复 $R_2=+0.136$
（95% 区间 [0.033, 0.233]），条件标量恢复 $R_3=+0.031$（[0.004, 0.056]）；
真正的独立标量项 $g_3=-0.002$。偏好随训练出现（$E=+0.321$，相对同一初始化）。
`[来源] partA_clarifications.json::A1、A2、A3；mechanism.json]`

这一结论的适用范围必须写清楚：它是**相对对照**的无量纲陈述，不是原始 logD 幅度；
它不覆盖所有含酰胺的分子；换到其他家族，结果**异质**——磺酰胺只有 $E$ 明确
（$rC$ 区间跨 0）、芳醚**反向**（$rC=-0.433$）、芳胺只有 4 个分子且 $E$ 为负、
酯未成立（$rC=-0.187$，区间跨 0）、磺酸酯没有对照因而**未检验**（不是零效应）。
`[来源] 同上]`

### R3 计算参照：方向与幅度的任务相关性

因为改线分子没有实验 logD 标签，我们另建两个**计算**参照：Wildman–Crippen logP
（有真实尺度）与平均分子量（组成不变，因此参照变化严格为 0）。
在同批 1810 个改线对（Valid 966 + Test 844）上，WC_LOGP 模型的新图预测优于「预测不变」基线：
Test 上 amide $+0.108$ [0.094, 0.124]、sulfonamide $+0.116$ [0.074, 0.158]、
aryl\_ether $+0.119$ [0.094, 0.144]、arylamine $+0.314$ [0.258, 0.363]（四者六项 Bonferroni
区间不含 0），符号一致率 0.95–0.99，相对变化 $q$ 一致率 0.82–0.89；
新图对自身参照的 MAE 为 0.067–0.119 logP。
`[来源] partB_wc_logp_metrics.json::family_stats；summary_tables.json]`

**反例保留在正文**：酯族只有 8 个分子 / 46 对，$+0.063$ [−0.007, 0.138]，区间跨 0，
因此写作「未解决」；磺酸酯在 Valid/Test 上没有可用的改线与对照，从未被检验。
四项支持不等于「所有官能团都被正确响应」。
`[来源] 同上]`

不变量对照给出的是**有边界的**结论：MolWt 参照变化对全部 1810 对严格为 0，
模型仍移动，但很小——分子权重下逐家族平均 $|\Delta p|$ 为 0.90–1.92 g/mol，
即 0.0084–0.0177 个训练目标标准差（$\sigma=108.294$）；Test 合并为 1.33 g/mol（0.0123 σ）。
尾部不剔除：最大单视图 14.24 g/mol（0.132 σ，`sulfonamide:1275:1:0:d9-14`）。
我们**不**声称不变性，也**不**从零均值或不显著反推等价。
`[来源] weighted_response.csv；oracle_manifest.json::checks.molwt_invariant_*]`

必须强调：WC_LOGP 与 MolWt 是**计算估计函数**，不是新的测量；它们上面的正确性
**不能**追认实验 logD 模型的广泛化学正确性。实验 logD 侧的改线分子仍然没有标签。
`[来源] PROTOCOL.md §3、§7]`

### R4 口径与反例（方法学附注）

本轮统一了响应统计的口径：主量使用分子→实例→对照→donor 的层级权重，
平铺（pair/view）分位数只作描述并显式标注。MolWt 的参照变化为 0，因此在同一口径下
层级 delta-MAE 与分子加权平均 $|\Delta p|$ **必须相等**；24 个 cell 全部通过（差 0.0），
这条算术核对同时验证了参照值、聚合层级与新旧表的一致性。
旧文中的合并（pooled）数字（WC_LOGP 0.246 / 0.241 / 0.072；MolWt 1.81 g/mol）
经复核是**只取 T 视图、pair 加权**，现已改标为对应口径（T+K 平均 0.191 / 0.184 / 0.056；
分子权重 0.185 / 0.172 / 0.056；MolWt 分子权重 1.33 g/mol）。
`[来源] weighted_response.{csv,json}；WEIGHTING_NOTE.md §3–§4]`

---

## English version

### E1 Method and traceability

We define the object of explanation as a fully explicit computation. For each atom $v$ the
model consumes only per-atom features (AF51), the incident-bond sum `b_sum`, first-hop joint
code counts `nbr`, second-hop `(path_cat, path_end63)` counts with closure and endpoint-content
terms, and nine per-atom scalars, plus 42 whole-graph statistics. The local readout is
$h_v=\mathrm{LN}(\mathrm{MLP}(x_v))$, $q_v=f(h_v)$, and the prediction is
$p(G)=\mu_y+s_y[\sum_v q_v+\rho(\bar g,\overline{h},\overline{h^2},\log(1+n))]$, where
$\overline{h^2}$ is the mean of component-wise squares. Two properties follow directly.
(i) Equal `global42` and an equal multiset of per-root statistical signatures give equal
predictions under the same frozen parameters. (ii) For a rewiring with an atom mapping, with
$U$ the set of roots whose signature changed,
$\Delta p=s_y[\sum_{v\in U}\Delta q_v+\Delta\rho]$, while unchanged roots keep their $q_v$.
The second property also shows why a per-atom attribution is unavailable: $\rho$ pools all
roots, and in the recorded intervention $\Delta\rho$ (+0.6393) is the same size as
$\Delta q$ (+0.5203).
`[Source: METHOD.md §2–§4; dissection/analysis/attribution/TABLES.md]`

The same mechanism sets a boundary: five pairs of **non-isomorphic** molecules share the
identical per-root signature multiset and `global42`, and the model's predictions differ by
$\le 3\times10^{-7}$. The signature is therefore *sufficient for this model's computation*, not a
complete chemical representation, and unequal signatures do not guarantee numerical separation.
`[Source: s3_locality/analysis/alias_audit.json]`

### E2 Experimental logD: preference depends on family and on the matchable population

For the frozen experimental-logD model the strongest training-related structural preference
appears in the **dose-matchable** amide sub-population: 62 Test molecules and 110 focal pairs
(about 38% of molecules and 21% of pairs of that family's main queue). The endpoint difference
in that sub-population is $J_C=0.146$; the conditional-S2 recovery against the matched control is
$R_2=+0.136$ (95% interval [0.033, 0.233]) and the conditional-scalar recovery is
$R_3=+0.031$ ([0.004, 0.056]); the genuinely standalone scalar term is $g_3=-0.002$.
The preference arises with training ($E=+0.321$ relative to the same initialisation).
`[Source: partA_clarifications.json::A1, A2, A3; mechanism.json]`

The scope must be stated with the result: it is a dimensionless, control-relative statement,
not a raw logD magnitude; it does not cover all amide-bearing molecules; and other families are
**heterogeneous** — sulfonamide has a clear $E$ only ($rC$ interval covers 0), aryl-ether is
**reversed** ($rC=-0.433$), arylamine has four molecules with negative $E$, ester is not
established ($rC=-0.187$, interval covers 0), and sulfonate has no control and is therefore
**untested**, not null.
`[Source: same]`

### E3 Computational references: direction and magnitude of a task-relevant response

Because the rewired molecules have no experimental logD labels, we build two **computational**
references: Wildman–Crippen logP (with a real scale) and average molecular weight (exactly
composition-invariant, so its reference change is identically zero). On the same 1810 rewiring
pairs (966 Valid + 844 Test) the WC_LOGP model's new-graph prediction beats the fixed
no-change baseline on the Test split: amide $+0.108$ [0.094, 0.124], sulfonamide $+0.116$
[0.074, 0.158], aryl\_ether $+0.119$ [0.094, 0.144], arylamine $+0.314$ [0.258, 0.363] (all four
clear under a six-item Bonferroni interval), with sign agreement 0.95–0.99, relative-change $q$
agreement 0.82–0.89, and a new-graph MAE against its own reference of 0.067–0.119 logP.
`[Source: partB_wc_logp_metrics.json::family_stats; summary_tables.json]`

**Counter-examples stay in the text**: ester has only 8 molecules / 46 pairs and gives
$+0.063$ [−0.007, 0.138], an interval covering zero, and is therefore reported as unresolved;
sulfonate has no usable rewire or control on Valid/Test and was never tested. Four supported
families are not "all functional groups respond correctly".
`[Source: same]`

The invariant control supports a deliberately bounded statement. The MolWt reference change is
exactly zero for all 1810 pairs, yet the model still moves, by little: under molecule weighting
the per-family mean $|\Delta p|$ is 0.90–1.92 g/mol, i.e. 0.0084–0.0177 of the training target
standard deviation ($\sigma=108.294$), and the pooled Test value is 1.33 g/mol (0.0123 σ). We do
not trim the tail: the largest single view is 14.24 g/mol (0.132 σ,
`sulfonamide:1275:1:0:d9-14`). We claim **no** exact invariance and read **no** equivalence from a
zero mean or a non-significant test.
`[Source: weighted_response.csv; oracle_manifest.json::checks.molwt_invariant_*]`

Finally, WC_LOGP and MolWt are **computational estimated functions**, not new measurements; their
correctness cannot be used to retro-validate a broad chemical mechanism for the experimental
logD model, whose rewired molecules still carry no labels.
`[Source: PROTOCOL.md §3, §7]`

### E4 Scope of the resulting claim

Taken together, these results support a bounded and testable statement: LCRC predicts from
explicit local structural statistics and allows a specific connection change to be traced to the
statistical inputs, the local readout and the pooled readout; in the fixed Lipo panel the
experimental-logD preference depends on family and on the matchable population; on the same
structural panel, models trained on computational references follow the reference direction and
magnitude across several rewiring families while showing a small but non-zero response on a
composition-invariant target. This is task-relevant structural dependence, not a universal
chemical causal mechanism and not a unique per-atom contribution. We do not claim that this
capability is unique to LCRC, nor that it outperforms other explanatory models; without the
corresponding controls those are evidence gaps, not results.
`[Source: CLAIMS_EVIDENCE.md §1, §2]`
