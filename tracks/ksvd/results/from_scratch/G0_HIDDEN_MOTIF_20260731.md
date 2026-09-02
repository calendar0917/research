# G0 Hidden Motif Discovery 结果

> 日期：2026-07-31  
> 数字唯一源：`g0_hidden_motif_20260731.json`

## 1. 判定

**PASS_INITIALIZER_DISCOVERY_ONLY**

The exact four-prototype vocabulary is already recovered by deterministic maximin at n_iter=0; G0 validates the pipeline but does not establish a KSVD discovery contribution.

本轮每个 data seed 对每种初始化只运行一次；没有 restart selection。

## 2. 数据与 oracle controls

- oracle：10/10
- 完整图全连通：10/10
- train/test 均恰有 4 种 canonical patch：10/10
- 四种 motif 全覆盖：10/10

## 3. INIT 与 FINAL 对照

| condition | stage | strict seeds | atom cosine mean/min(seed) | exact motifs mean | occurrence macro-F1 mean/min | exact occurrence mean/min | test recon mean | pairwise stability min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic_maximin | INIT | 10/10 | 1.0000/1.0000 | 4.00 | 1.0000/1.0000 | 1.0000/1.0000 | 0.000000 | 1.0000 |
| deterministic_maximin | FINAL | 10/10 | 1.0000/1.0000 | 4.00 | 1.0000/1.0000 | 1.0000/1.0000 | 0.000000 | 1.0000 |
| fixed_random_columns | INIT | 0/10 | 0.8109/0.7165 | 2.10 | 0.3844/0.1281 | 0.5494/0.3444 | 0.437185 | 0.4665 |
| fixed_random_columns | FINAL | 10/10 | 1.0000/1.0000 | 4.00 | 1.0000/1.0000 | 1.0000/1.0000 | 0.000000 | 1.0000 |

## 4. 每个 data seed

| data seed | train motif counts | maximin unique init | maximin init/final atom | maximin init/final occurrence | random unique init | random init/final atom | random init/final occurrence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20260731 | [448, 269, 304, 179] | 4 | 1.000/1.000 | 1.000/1.000 | 3 | 0.917/1.000 | 0.683/1.000 |
| 20260732 | [419, 307, 299, 175] | 4 | 1.000/1.000 | 1.000/1.000 | 2 | 0.811/1.000 | 0.324/1.000 |
| 20260733 | [413, 299, 296, 192] | 4 | 1.000/1.000 | 1.000/1.000 | 3 | 0.832/1.000 | 0.189/1.000 |
| 20260734 | [418, 296, 298, 188] | 4 | 1.000/1.000 | 1.000/1.000 | 2 | 0.789/1.000 | 0.374/1.000 |
| 20260735 | [436, 296, 296, 172] | 4 | 1.000/1.000 | 1.000/1.000 | 2 | 0.717/1.000 | 0.372/1.000 |
| 20260736 | [412, 287, 322, 179] | 4 | 1.000/1.000 | 1.000/1.000 | 3 | 0.832/1.000 | 0.387/1.000 |
| 20260737 | [401, 309, 298, 192] | 4 | 1.000/1.000 | 1.000/1.000 | 1 | 0.726/1.000 | 0.128/1.000 |
| 20260738 | [424, 294, 301, 181] | 4 | 1.000/1.000 | 1.000/1.000 | 3 | 0.832/1.000 | 0.374/1.000 |
| 20260739 | [439, 285, 270, 206] | 4 | 1.000/1.000 | 1.000/1.000 | 2 | 0.760/1.000 | 0.391/1.000 |
| 20260740 | [406, 287, 315, 192] | 4 | 1.000/1.000 | 1.000/1.000 | 3 | 0.894/1.000 | 0.622/1.000 |

## 5. 科学解释

- `INIT` 与 `FINAL` 使用完全相同的初始字典；差异只来自 KSVD iterations。
- deterministic maximin 若在 INIT 已通过，说明四种 exact canonical prototypes 被初始化器直接枚举出来，不能宣称 KSVD 自动提炼了 motif。
- fixed random-column 是单次弱初始化诊断，不参与选择，也不用于救 primary 结果。
- 即使 G0 通过，oracle cell extractor 仍是强条件；随机游走覆盖与 patch 间关联尚未验证。

## 6. 下一步

若判定为 `PASS_INITIALIZER_DISCOVERY_ONLY`，进入 G0B：加入 within-motif variation / nuisance edges，使训练集不再只有四种唯一 canonical columns，再检查 INIT→FINAL 是否出现真正的 KSVD refinement。
