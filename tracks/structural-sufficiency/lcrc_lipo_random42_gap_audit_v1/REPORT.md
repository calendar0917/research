# LCRC-A-R3 vs PNA-native — random42 Validation420 误差差距审计

- **Track:** `tracks/structural-sufficiency/lcrc_lipo_random42_gap_audit_v1/`
- **Scope:** frozen random42 **Validation420 only**；CPU-only；只读历史产物；无训练、无 checkpoint 选择、无 Test 标签/预测/统计/逐分子 Test 分析。
- **Commit:** 以本目录最新 commit 为准（`git log -1 -- tracks/structural-sufficiency/lcrc_lipo_random42_gap_audit_v1`）。

本报告全部为**事后描述**，不用于改写主指标，也不作为机制证明。分组阈值在查看结果前固定为验证集样本四分位数，未根据结果反复调整。

---

## 1. 输入、对齐与完整性

实际读取的冻结输入（均为本地现有文件，未重新推断、未重新训练）：

| 文件 | SHA-256 |
|---|---|
| `lcrc_lipo_esa_random42_v1/data/split_esa_random42_v1.json` | `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4` |
| `lcrc_lipo_esa_random42_v1/data/labels_train_valid_v1.json` | `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca` |
| `lcrc_lipo_esa_random42_v1/analysis/metrics_validation.json` | `c540f9ac89a391ea13c8235b7b14cb7faf10ae2a9d60616d2e4716659c576fdd` |
| `lcrc_lipo_esa_random42_v1/analysis/runs/lcrc_a_r3_s0/valid_predictions.npz` | `7f9ad3919d57755d5adc08b99f1209b18ab6629c14eb182592a858183d88e894` |
| `lcrc_lipo_esa_random42_v1/analysis/runs/pna_native_s0/valid_predictions.npz` | `735df74915d43803b63323e09d85089c6c2d284e04acffcf242eb587ae1ac905` |

完整性结果（`analysis/summary.json` 共 55 项检查，全部通过）：

- Validation420 的 global row ID 为 420 个唯一值，冻结顺序与 `split_esa_random42_v1.json` 完全一致。
- 两个模型的 `valid_predictions.npz` 内 `rows` 均与冻结 Validation420 顺序逐元素相等；两个 prediction 文件行 ID 的 SHA-256 均为 `8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44`。
- Train3360 row ID 哈希为 `0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee`；Train/Validation 无交集。
- Validation label 向量哈希为 `09bc1a1e1756bbd09dd17a5fc77a1b3c33e7010aef55fc352a4bba1d9822636f`；Train label 向量哈希为 `85a4c7c559f4b7fd58ad78b1b05d5a4160e0f60c351b7fa842eb0354aa47abd8`。
- 两个 prediction 文件的 SHA-256 与 `metrics_validation.json` 中历史记录一致。

数据边界说明：`split_esa_random42_v1.json` 作为冻结 split 文件整体被解析，但代码只按 Train3360/Validation420 的 row ID 取用对应的 SMILES；Test 的标签、预测、统计和逐分子 Test 分析均未打开或使用。`metrics_validation.json` 只含 train/validation 记录。

---

## 2. A. 指标重算

重算使用冻结 Validation420 标签与两个历史 `valid_predictions.npz` 的 `pred` 向量，未做任何后处理。

| 模型 | n | 历史 RMSE | 重算 RMSE | \|ΔRMSE\| | 历史 MAE | 重算 MAE | \|ΔMAE\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| LCRC-A-R3 | 420 | 0.5890072081930064 | 0.5890072081930064 | 0.0 | 0.4208143415034744 | 0.4208143415034744 | 0.0 |
| PNA-native | 420 | 0.5540342318996931 | 0.5540342318996931 | 0.0 | 0.4007184723844865 | 0.4007184723844865 | 0.0 |

两个 RMSE 的复现误差均为 **0.0**，满足 `1e-5` 要求；因此继续执行后续分析。

---

## 3. B. 逐分子平方误差差距

对每个 Validation420 分子定义：

`d_i = (pred_LCRC_i - y_i)^2 - (pred_PNA_i - y_i)^2`

`d_i > 0` 表示 PNA 在该分子上的平方误差更小。

### 3.1 总体

| 量 | 数值 |
|---|---:|
| SSE LCRC | 145.71038634739426 |
| SSE PNA | 128.92065064900686 |
| SSE 差值（LCRC − PNA） | **16.789735698387407** |
| 相对 SSE 降低（PNA 相对 LCRC） | 11.5227% |
| RMSE LCRC | 0.5890072081930064 |
| RMSE PNA | 0.5540342318996931 |
| RMSE 差值（LCRC − PNA） | **0.03497297629331331** |
| 相对 RMSE 降低（PNA 相对 LCRC） | 5.9376% |
| PNA 更优分子数 | **209 / 420（49.7619%）** |
| LCRC 更优分子数 | 211 / 420（50.2381%） |
| 平局 | 0 |
| `mean(d_i)` | 0.039975561186636685 |
| `median(d_i)` | **−0.00038023128067303367** |
| `std(d_i)` | 0.7445117322799051 |
| 正 `d_i` 总量 | 77.51387381148646 |
| 负 `d_i` 总量 | −60.724138113099066 |

要点：PNA 只在 **209/420** 个分子上更优，且 `median(d_i)` 略为负；总体 SSE 优势来自正尾部的非对称贡献。

### 3.2 前 5/10/20 个正 `d_i` 分子的贡献与移除后 RMSE

| 移除的 top-k 正 `d_i` | Σd（SSE 单位） | 占净 SSE 差 16.7897 的比例 | 占正 `d_i` 总量 77.5139 的比例 | 剩余 n | 移除后 SSE 差 | 移除后 RMSE LCRC | 移除后 RMSE PNA | 移除后 RMSE 差（LCRC−PNA） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 20.64183721698337 | **122.94%** | 26.63% | 415 | −3.852101518595962 | 0.5380802843321859 | 0.5466375076053346 | **−0.008557223273148673** |
| 10 | 30.284299863419648 | **180.37%** | 39.07% | 410 | −13.49456416503224 | 0.5136304186429885 | 0.5447290868767672 | **−0.031098668233778715** |
| 20 | 41.77540567369138 | **248.82%** | 53.89% | 400 | −24.985669975303974 | 0.4827420569943982 | 0.5436028591991057 | **−0.060860802204707465** |

`top-k` 的 Σd 可以超过净差值，因为其余分子合计对 LCRC 有利（净负贡献）。这也意味着：移除 top-5 后，Validation420 的 RMSE 排名反转为 **LCRC 更好**。

前 5 个分子：

| row_id | y_logD | pred LCRC | pred PNA | abs err LCRC | abs err PNA | `d_i` | max train Tanimoto | scaffold seen |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1630 | 3.90 | 1.136624 | 3.082469 | 2.763376 | 0.817531 | 6.967887 | 0.333333 | False |
| 2924 | 3.08 | 0.733992 | 2.143570 | 2.346008 | 0.936430 | 4.626851 | 0.454545 | False |
| 822 | 3.44 | 1.310462 | 2.705747 | 2.129538 | 0.734253 | 3.995803 | 0.266667 | False |
| 1153 | 3.37 | 1.701731 | 2.905735 | 1.668269 | 0.464265 | 2.567579 | 0.666667 | False |
| 3634 | 1.81 | 4.067704 | 3.426635 | 2.257704 | 1.616635 | 2.483718 | 0.485714 | True |

前 5 个分子中 4 个 scaffold-unseen、4 个位于 largest-size Q4、3 个位于最低训练相似度 Q1。完整逐分子表见 `analysis/per_molecule_valid.csv`。

**图 1**：`analysis/figures/01_gap_concentration.png`（累计 `d_i` 与逐次移除 top-d 后的 RMSE）。

**Q1 答案：是，差距显著由少数大误差分子驱动。** top-5 正 `d_i` 已经达到净 SSE 差的 **122.9%**；移除这 5 个分子后 LCRC RMSE 反而比 PNA 低 0.008557。PNA 仅在 49.76% 的分子上更优，`median(d_i)` 为负，因此这不是一个“多数分子一致占优”的差距。

---

## 4. C. 训练相似度与 scaffold 分组

### 4.1 方法

- 使用冻结 split 中的 Train3360 与 Validation420 SMILES。
- Morgan fingerprint：radius 2，2048 bits；相似度为 Tanimoto。
- 对每个 Validation 分子，计算其与全部 3360 个 Train 分子的最大 Tanimoto 相似度。
- Bemis–Murcko scaffold 使用 canonical SMILES 表示；空 scaffold 回退到该分子的 canonical SMILES。验证集空 scaffold 回退数为 0；训练集为 5。
- 训练集唯一 scaffold 数 2045；验证集唯一 scaffold 数 371。
- scaffold-seen：该验证分子的 scaffold 字符串出现在 Train3360 的 scaffold 集合中。

Validation 最大训练相似度分布：mean 0.6445，median 0.6973，IQR 0.4592–0.8072，min 0.2267，max 1.0。

### 4.2 训练相似度四分位组

| 组 | 相似度范围（min–max） | n | RMSE LCRC | RMSE PNA | RMSE 差 | MAE LCRC | MAE PNA | SSE 差贡献 | 占净 SSE 差 | `mean(d_i)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 lowest | 0.226667–0.458333 | 105 | 0.814079 | 0.757302 | 0.056777 | 0.626865 | 0.588107 | 9.367957 | 55.80% | 0.089219 |
| Q2 | 0.459459–0.696970 | 105 | 0.569532 | 0.542309 | 0.027223 | 0.402855 | 0.387641 | 3.178156 | 18.93% | 0.030268 |
| Q3 | 0.697674–0.807018 | 105 | 0.515544 | 0.495691 | 0.019853 | 0.383846 | 0.370611 | 2.107961 | 12.56% | 0.020076 |
| Q4 highest | 0.807692–1.000000 | 105 | 0.367206 | 0.338379 | 0.028827 | 0.269691 | 0.256515 | 2.135662 | 12.72% | 0.020340 |

补充：Q4 中有 20 个 Validation 分子的最大 Tanimoto 为 1.0，但均不是 Train3360 的 canonical-SMILES 精确重复；它们是 Morgan(2,2048) 位串等价的非同一分子。该 20 个分子上 LCRC RMSE 0.181124、PNA RMSE 0.288987，SSE 差 −1.014155（LCRC 更好）。排除这 20 个后，Q4（n=85）RMSE 为 LCRC 0.398558、PNA 0.348986，差 0.049571，`mean(d_i)` 0.037057。

### 4.3 Scaffold-seen / unseen

| 组 | n | RMSE LCRC | RMSE PNA | RMSE 差 | MAE LCRC | MAE PNA | SSE 差贡献 | 占净 SSE 差 | `mean(d_i)` | PNA 更优数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scaffold seen | 209 | 0.433788 | 0.430336 | 0.003452 | 0.308548 | 0.307093 | 0.623503 | 3.71% | 0.002983 | 110/209（52.63%） |
| scaffold unseen | 211 | 0.710058 | 0.653884 | 0.056173 | 0.532017 | 0.493456 | 16.166233 | 96.29% | 0.076617 | 99/211（46.92%） |

**图 2**：`analysis/figures/02_group_rmse.png`（各分组 RMSE 对比）。

**Q2 答案：**

- **scaffold-seen 组差距很小**：RMSE 差仅 0.003452，只占净 SSE 差的 3.71%。
- **scaffold-unseen 组差距大得多**：RMSE 差 0.056173，占净 SSE 差的 96.29%。注意 unseen 组中 PNA 更优的分子比例只有 46.92%，所以该组优势同样来自少数大误差分子，而不是普遍占优。
- **训练相似度四分位组全部为正**（PNA 的 RMSE 更低）：Q1 0.056777、Q2 0.027223、Q3 0.019853、Q4 0.028827；差距在最低相似度 Q1 最大，但并非随相似度单调下降。Q4 排除 20 个 fingerprint-equivalent 分子后差距为 0.049571。组间差异主要反映误差尾部位置，不能解释为相似度阈值机制。

---

## 5. D. 基本化学结构分组

描述符由 RDKit 从验证集 SMILES 计算，逐分子值全部保存在 `analysis/per_molecule_valid.csv`。全体验证集描述符的中位数（IQR）如下：

| 描述符 | median | IQR |
|---|---:|---:|
| heavy-atom count | 27 | 21–31.25 |
| ring count | 3 | 3–4 |
| aromatic atom fraction | 0.5455 | 0.4444–0.6667 |
| TPSA | 80.00 | 56.85–99.36 |
| HBD | 2 | 1–2 |
| HBA | 5 | 3–6 |
| MolLogP | 3.21925 | 2.3687–4.0192 |

分组使用固定的验证集样本四分位边界；边界值归入较低组。所有请求分组的 n 均 ≥ 87，未出现 `n < 30` 的不稳定分组。

### 5.1 按 heavy-atom count 四分位

| 组 | heavy atoms（min–max） | n | RMSE LCRC | RMSE PNA | RMSE 差 | MAE LCRC | MAE PNA | SSE 差贡献 | 占净 SSE 差 | `mean(d_i)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 lowest | 7–21 | 114 | 0.556926 | 0.635976 | −0.079050 | 0.397462 | 0.440090 | −10.750033 | −64.03% | −0.094299 |
| Q2 | 22–27 | 114 | 0.568082 | 0.484030 | 0.084052 | 0.408153 | 0.355211 | 10.081208 | 60.04% | 0.088432 |
| Q3 | 28–31 | 87 | 0.552291 | 0.568996 | −0.016705 | 0.424729 | 0.424341 | −1.629578 | −9.71% | −0.018731 |
| Q4 highest | 32–49 | 105 | 0.669217 | 0.515810 | 0.153407 | 0.456671 | 0.387807 | 19.088139 | 113.69% | 0.181792 |

### 5.2 按 TPSA 四分位

| 组 | TPSA（min–max） | n | RMSE LCRC | RMSE PNA | RMSE 差 | MAE LCRC | MAE PNA | SSE 差贡献 | 占净 SSE 差 | `mean(d_i)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 lowest | 3.24–56.73 | 105 | 0.435199 | 0.500241 | −0.065042 | 0.318954 | 0.390385 | −6.388531 | −38.05% | −0.060843 |
| Q2 | 56.89–80.00 | 106 | 0.641429 | 0.584626 | 0.056803 | 0.489860 | 0.428838 | 7.382217 | 43.97% | 0.069644 |
| Q3 | 80.04–99.36 | 105 | 0.594814 | 0.605472 | −0.010659 | 0.429614 | 0.407103 | −1.343331 | −8.00% | −0.012794 |
| Q4 highest | 99.77–164.55 | 104 | 0.658251 | 0.518162 | 0.140088 | 0.444397 | 0.376045 | 17.139381 | 102.08% | 0.164802 |

### 5.3 按 MolLogP 四分位

| 组 | MolLogP（min–max） | n | RMSE LCRC | RMSE PNA | RMSE 差 | MAE LCRC | MAE PNA | SSE 差贡献 | 占净 SSE 差 | `mean(d_i)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 lowest | −1.9714–2.3681 | 105 | 0.547165 | 0.602278 | −0.055113 | 0.373599 | 0.398293 | −6.651628 | −39.62% | −0.063349 |
| Q2 | 2.3689–3.2179 | 105 | 0.523075 | 0.477602 | 0.045472 | 0.385914 | 0.353170 | 4.777817 | 28.46% | 0.045503 |
| Q3 | 3.2206–4.0186 | 105 | 0.606642 | 0.582782 | 0.023860 | 0.456762 | 0.439973 | 2.979853 | 17.75% | 0.028380 |
| Q4 highest | 4.0210–6.9372 | 105 | 0.668361 | 0.545287 | 0.123074 | 0.466983 | 0.411438 | 15.683694 | 93.41% | 0.149369 |

描述性观察：PNA 的 RMSE 优势集中在 largest-size Q4、最高 TPSA Q4 和最高 MolLogP Q4；这些组的误差差贡献分别相当于净 SSE 差的 113.69%、102.08% 和 93.41%。这与 B 节 top-d 分子的分布一致，但都是单变量事后分组，不能据此作机制结论；分组之间高度重叠。

---

## 6. E. 训练集局部近邻支持

仅使用 Train3360 的 canonical SMILES 与 Train3360 标签，未使用 Validation/Test。Morgan(2,2048) Tanimoto，排除自身，阈值 `sim >= 0.80`；最近邻平局按冻结 Train3360 顺序取第一个。

### 6.1 近邻数量

| 近邻数量 k | 训练分子数 | 占比 |
|---|---:|---:|
| 0 | 2568 | 76.43% |
| 1 | 447 | 13.30% |
| 2 | 152 | 4.52% |
| 3–4 | 135 | 4.02% |
| 5–9 | 56 | 1.67% |
| 10+ | 2 | 0.06% |

满足至少 1 / 3 / 5 个近邻的训练分子数分别为：

| 支持水平 | n | 占 Train3360 |
|---|---:|---:|
| ≥1 neighbor | **792** | 23.57% |
| ≥3 neighbors | **193** | 5.74% |
| ≥5 neighbors | **58** | 1.73% |
| ≥10 neighbors | 2 | 0.06% |

### 6.2 近邻标签离散度与最近邻标签差

`neighbor_label_std` 定义为所有 `sim >= 0.80` 近邻标签的总体标准差（ddof=0）；恰好 1 个近邻时为 0（这是构造性的，不代表标签一致）。标签差定义为 `y_i − y_nearest`，下表使用其绝对值 `|Δy|`。

| 支持水平 | n | label std mean | label std median | label std p75 | \|Δy\| mean | \|Δy\| median | \|Δy\| p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥1 neighbor | 792 | 0.1592 | 0.0000 | 0.2767 | 0.4580 | 0.3300 | 1.0200 |
| ≥3 neighbors | 193 | 0.4320 | 0.3996 | 0.5605 | 0.5510 | 0.4800 | 1.1620 |
| ≥5 neighbors | 58 | 0.5449 | 0.5394 | 0.6495 | 0.6341 | 0.5700 | 1.2730 |

有 ≥1 个近邻的 792 个分子上，最近邻绝对标签差分布为：`[0,0.1)` 166 个、`[0.1,0.25)` 160 个、`[0.25,0.5)` 173 个、`[0.5,1.0)` 210 个、`[1.0,2.0)` 77 个、`≥2.0` 6 个。sign 方向基本平衡（signed mean −0.0046），但离散度大。

### 6.3 指纹分辨率注意事项

- Morgan(2,2048) Tanimoto = 1.0 只表示位串等价，不表示分子相同。Validation420 中有 20 个 max-sim=1.0 的非同一分子；Train3360 中有 136 个分子拥有至少一个 sim=1.0 的近邻，对应 84 个 fingerprint-equivalent 训练对。
- 在这 136 个训练分子中，只有 17 个的最近邻绝对标签差为 0；其余仍有非零标签差。
- 因此不能把“Tanimoto 高”直接解释为“只有一个官能团发生改变”；本审计没有做这样的结构编辑映射。

**图 3**：`analysis/figures/03_train_neighbor_support.png`（近邻数量、标签差与标签离散度）。

**Q3 答案：按本次 ≥0.80 的定义，数据基础有限，且近邻标签差异不可忽略。**

- 规模上：只有 792/3360（23.57%）的训练分子有至少 1 个近邻，193 个（5.74%）达到 ≥3，58 个（1.73%）达到 ≥5；76.43% 的训练分子没有 ≥0.80 近邻。
- 标签上：有近邻的分子中最近邻 `|Δy|` 中位数为 0.33，p90 为 1.02；在 ≥5 近邻组中中位数为 0.57。近邻标签并不近似恒定。
- 分辨率上：sim=1.0 是 fingerprint 等价而非分子同一；训练集中 136 个 sim=1.0 分子里只有 17 个最近邻标签差为 0。
- 因此，若把“局部编辑差分监督”理解为需要大量“高相似且仅局部改变”的训练对，本审计的实测支持集偏小；同时高相似度本身不能证明单官能团编辑。这里只报告数据规模与标签差异，不实现或建议新的监督损失。

---

## 7. 产物与复现

| 产物 | SHA-256 |
|---|---|
| `analysis/per_molecule_valid.csv` | `8bec6ad413fb0f07f492287cc74c74113875c20efed297d3c9c8a13eb02ce661` |
| `analysis/subgroups.csv` | `e5cfeb2b9f92b58708b959f0a4872b6065ddc68d4138ae4ca875b159ab60c130` |
| `analysis/train_neighbor_support.json` | `60e532fd59fe3edb852cbabe14b7d4343a4989a783291c9c7b1dc7ecbf3bcd04` |
| `analysis/summary.json` | `fcace8e3a0dd4c26482de09ee1ba41aa2f52da6d5323554bb48b081d0456e6a5` |
| `analysis/figures/01_gap_concentration.png` | `6f9aa9c48b7c91938bfacfd489537743ca1bdecc41cbbb7af124a6c1615982b8` |
| `analysis/figures/02_group_rmse.png` | `9b9028bc352d1cbfbbacaf0551919403aab2e74b7f2a8b44f069eef1a115b032` |
| `analysis/figures/03_train_neighbor_support.png` | `307f11674f5be3dbd1b07449980b7eebc0cce83625c48f38257f956286e4d924` |

复现命令：

```bash
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_gap_audit_v1/code/run_gap_audit.py
```

脚本仅写入本 track 的 `analysis/` 目录；未修改 `lcrc_lipo_esa_random42_v1/` 的历史产物。执行环境：Python 3.12.15、numpy 2.1.3、pandas 2.2.3、rdkit 2026.09.1、matplotlib 3.11.1；`gpu_used=false`、`training_or_finetuning=false`、`test_artifacts_used=false`。
