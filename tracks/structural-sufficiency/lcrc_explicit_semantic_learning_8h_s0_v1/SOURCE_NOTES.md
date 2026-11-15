# SOURCE_NOTES — lcrc-explicit-semantic-learning-8h-s0-v1

本文件记录本轮开始前对上一轮（`lcrc_lipo_context_small_s0_v1`）的机器核对，
以及对显式 code 支持度的独立计数。所有数字来自已拉回的 run 输出，未重训。

## 1. 上一轮到底排除了什么、哪些旧文字需要收窄

* 旧报告称 R2 的 Val 也改善。**不成立**：机器表中 R2 best-Val RMSE=0.7419，B best-Val RMSE=0.7371，R2 反而更差。旧结论 '缺容量/归纳偏置' 并没有被该轮唯一识别。
* 上一轮 S 代码里有 `bond_proj` 项，PROTOCOL 中 'no bond term' 只是文字不准；
  这既不影响该轮判定，也不构成本轮重训的理由。
* H2 的 size-standardised gap（Test 0.1159）与未标准化 gap（约 0.1170）是两件事，
  必须分别报告，并同时给出群体/补集的支持数；本文件不混用。

## 2. 真正进入模型的 code 支持度（Train 分子 document frequency）

* path_cat 词表 400，Train 出现的不同 code 数 = 23。
* (path_cat,path_end63) 联合词表 32000，Train 出现的不同联合 code 数 = 74。
* nbr/end 词表 80，Train 出现的不同 code 数 = 20 / 19。

| split | path 未见 token | path df1-5 | path df6-20 | path df>20 | 联合未见 | nbr 未见 | end 未见 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.0000 | 0.0010 | 0.0047 | 0.9943 | 0.0000 | 0.0000 | 0.0001 |
| valid | 0.0000 | 0.0048 | 0.0064 | 0.9888 | 0.0032 | 0.0000 | 0.0003 |
| test | 0.0045 | 0.0045 | 0.0091 | 0.9818 | 0.0091 | 0.0032 | 0.0003 |

* 支持度按 **Train 分子 document frequency**（每分子同 code 只计一次），不是 token/根出现数。
* Morgan-H2 的高覆盖与真实表项稀有性分开报告；即使所有单 code 都在 Train 出现过，
  仍然可以检验少样本参数共享，不能把 Morgan radius3 未见直接说成模型查到了未训练的表项。

## 3. 编码 schema

* 16 atom types、5 bond types、400 path codes、80 binary codes；`path_end63` 是历史字段名，
  实际为 80 类。绝不沿用 ZINC 的 189/63 词表。

（生成：`code/se_audit.py --notes`；机器文件：`analysis/p0_audit.json`。）
