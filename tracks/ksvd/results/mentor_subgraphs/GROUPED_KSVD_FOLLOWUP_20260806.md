# 导师 50-node 真实子图：分组划分 KSVD 重构 follow-up

> 日期：2026-08-06
> 协议：`tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_GROUPED_RECONSTRUCTION_PROTOCOL_20260806.md`
> 判定：`KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION`（分组重构通过，可进入关系令牌消融）

## 1. 数据与 selection（样本选择）

- 源数据 SHA-256：`936134e7876740d5efc478f6bdf9663e31440ab506ecee507b00f8e32a7f05cb`；
- 缓存图数：`10000`；pilot 子集：`500`；
- pilot 根节点数：`231`；密度配额：`{'lt5': 100, 'd5_10': 100, 'd10_15': 100, 'd15_25': 100, 'ge25': 100}`（五个平均度分层各 100 张）；
- 标签使用：`False`；patch 向量中的源 ID：`False`；字典中的源 ID：`False`。
- 划分：`3` 折，seed `20260807`；视角（views）：`['random_reference', 'root_candidate_grouped']`（随机参考 / 根候选分组）。

- pilot 冻结验证：checked=`True`；passed=`True`；
  pilot_indices_match=`True`，compared=`3000`，mismatches=`0`，missing_but_in_pilot=`0`（与 pilot 的 500 图索引完全一致）。

## 2. 总判定

- 划分/数据 gate：`True`；
- 优化 gate 通过的分支：`['s8_o2_BASE', 's8_o2_FAIR95', 's10_o3_BASE', 's10_o3_FAIR95', 's12_o4_BASE', 's12_o4_FAIR95']`；
- 对照/Pareto 通过的分支：`['s8_o2_BASE', 's8_o2_FAIR95', 's10_o3_BASE', 's10_o3_FAIR95', 's12_o4_BASE', 's12_o4_FAIR95']`；
- 就绪分支：`['s8_o2_BASE', 's8_o2_FAIR95', 's10_o3_BASE', 's10_o3_FAIR95', 's12_o4_BASE', 's12_o4_FAIR95']`；Pareto：`['s8_o2_BASE', 's8_o2_FAIR95', 's10_o3_BASE', 's10_o3_FAIR95', 's12_o4_BASE', 's12_o4_FAIR95']`。

> 六个分支全部通过，进入关系令牌消融阶段。

## 3. 划分视角与源暴露审计（split views & source exposure audit）

> 源暴露（exposure）衡量：测试图的节点/边在训练图集中被"见过"的比例。它是转导式重叠审计，不是 inductive 泛化证明。

| 视角 | 折 | 训练/测试 | 根 train/test/∩ | 节点可见 | 节点 Jaccard | 边可见 | 边 Jaccard | 节点暴露 均值/p10/最小 | 边暴露 均值/p10/最小 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_reference | 0 | 333/167 | 186/119/74 | 0.8901 | 0.6947 | 0.8327 | 0.5631 | 0.9601/0.8800/0.5000 | 0.8666/0.7191/0.0583 |
| random_reference | 1 | 334/166 | 185/112/66 | 0.9241 | 0.6605 | 0.8399 | 0.5516 | 0.9754/0.9200/0.7600 | 0.8754/0.7497/0.2655 |
| random_reference | 2 | 333/167 | 188/115/72 | 0.9035 | 0.6924 | 0.8259 | 0.5581 | 0.9679/0.9000/0.4400 | 0.8668/0.7230/0.3430 |
| root_candidate_grouped | 0 | 333/167 | 153/78/0 | 0.8990 | 0.6825 | 0.8323 | 0.5324 | 0.9626/0.9000/0.7200 | 0.8681/0.7194/0.4797 |
| root_candidate_grouped | 1 | 333/167 | 154/77/0 | 0.9117 | 0.6786 | 0.8212 | 0.5405 | 0.9665/0.9000/0.5200 | 0.8643/0.7608/0.0583 |
| root_candidate_grouped | 2 | 334/166 | 155/76/0 | 0.8933 | 0.6688 | 0.8062 | 0.5389 | 0.9587/0.8800/0.5200 | 0.8538/0.7058/0.3982 |

> 关键：root_candidate_grouped 的根交集严格为 0（同一根不跨折），但节点/边暴露率仍高达 ~89%/~82%——说明大量结构在训练中已见过，结论只能叫"root-candidate 隔离"，不能叫完全 inductive。

## 4. 规范 patch 向量重叠与 patch 源节点暴露

> patch 向量是 rooted-canonical 邻接比特向量（不含源 ID）；occurrence=测试 patch 向量在训练中按多重性见过多少；unique overlap=唯一向量 Jaccard；patch 源节点暴露=patch 覆盖的稳定槽位经 `ids.order` 映回源节点后与训练源节点比较。

| 视角 | 几何 | 检查点 | 模板出现率 | 唯一重叠 | 覆盖节点可见 | 根已覆盖 | 覆盖根可见 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_reference | s8_o2 | BASE | 0.7503 | 0.2080 | 0.9755 | 0.8339 | 0.5420 |
| random_reference | s8_o2 | FAIR95 | 0.7078 | 0.1856 | 0.9678 | 1.0000 | 0.6420 |
| random_reference | s10_o3 | BASE | 0.3382 | 0.0518 | 0.9741 | 0.8400 | 0.5420 |
| random_reference | s10_o3 | FAIR95 | 0.2885 | 0.0444 | 0.9678 | 1.0000 | 0.6420 |
| random_reference | s12_o4 | BASE | 0.1679 | 0.0174 | 0.9735 | 0.8300 | 0.5320 |
| random_reference | s12_o4 | FAIR95 | 0.1306 | 0.0133 | 0.9678 | 1.0000 | 0.6420 |
| root_candidate_grouped | s8_o2 | BASE | 0.7476 | 0.2079 | 0.9732 | 0.8340 | 0.0000 |
| root_candidate_grouped | s8_o2 | FAIR95 | 0.7054 | 0.1852 | 0.9626 | 1.0000 | 0.0000 |
| root_candidate_grouped | s10_o3 | BASE | 0.3423 | 0.0538 | 0.9709 | 0.8400 | 0.0000 |
| root_candidate_grouped | s10_o3 | FAIR95 | 0.2913 | 0.0457 | 0.9626 | 1.0000 | 0.0000 |
| root_candidate_grouped | s12_o4 | BASE | 0.1694 | 0.0180 | 0.9709 | 0.8300 | 0.0000 |
| root_candidate_grouped | s12_o4 | FAIR95 | 0.1320 | 0.0140 | 0.9626 | 1.0000 | 0.0000 |

> 读法：patch 越大，结构模板重复率越低（s8≈0.75 → s12≈0.17），说明大 patch 的精确模板更长尾；root-grouped 下"覆盖根可见"=0 是预期（根被隔离了）。

## 5. 重构结果（按测试图数加权的均值）

> 阶段（stage）对照：`RAW`=原始 patch 邻接直接拼接（上界）；`PCA3`=训练集 3 维主成分（稠密基线）；`RANDOM`=随机 patch 字典的稀疏编码；`INIT`=确定性的最远点初始字典、不更新；`FINAL`=同起点学习 25 轮的 KSVD 字典。观测（obs）=仅采样看到的边；全图（full）=拼到 50 节点完整图；校正（corrected）=残差边单独暴露后的 patch 压缩误差。

| 视角 | 分支 | 阶段 | patch 误差 | 观测 RMSE | 观测 F1 | 全图 RMSE | 全图准确率 | 全图召回 | 全图 F1 | 校正 RMSE | 校正召回 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_reference | s8_o2_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1447 | 0.9776 | 0.8966 | 0.9446 | 0.0000 | 1.0000 |
| random_reference | s8_o2_BASE | PCA3 | 0.3570 | 0.2941 | 0.9062 | 0.2257 | 0.9373 | 0.8464 | 0.8581 | 0.1701 | 0.9499 |
| random_reference | s8_o2_BASE | RANDOM | 0.2817 | 0.2313 | 0.9496 | 0.1991 | 0.9569 | 0.8678 | 0.8982 | 0.1338 | 0.9712 |
| random_reference | s8_o2_BASE | INIT | 0.3375 | 0.2647 | 0.9327 | 0.2113 | 0.9561 | 0.8702 | 0.8830 | 0.1495 | 0.9736 |
| random_reference | s8_o2_BASE | FINAL | 0.2251 | 0.1836 | 0.9765 | 0.1811 | 0.9678 | 0.8865 | 0.9232 | 0.1058 | 0.9900 |
| random_reference | s8_o2_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0716 | 0.9929 | 0.9801 | 0.9899 | 0.0000 | 1.0000 |
| random_reference | s8_o2_FAIR95 | PCA3 | 0.3853 | 0.3037 | 0.8907 | 0.2083 | 0.9419 | 0.9085 | 0.8820 | 0.1927 | 0.9284 |
| random_reference | s8_o2_FAIR95 | RANDOM | 0.3193 | 0.2484 | 0.9329 | 0.1757 | 0.9644 | 0.9413 | 0.9236 | 0.1566 | 0.9611 |
| random_reference | s8_o2_FAIR95 | INIT | 0.3581 | 0.2714 | 0.9297 | 0.1874 | 0.9654 | 0.9455 | 0.9204 | 0.1697 | 0.9653 |
| random_reference | s8_o2_FAIR95 | FINAL | 0.2515 | 0.1949 | 0.9682 | 0.1467 | 0.9784 | 0.9621 | 0.9586 | 0.1239 | 0.9819 |
| random_reference | s10_o3_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1618 | 0.9704 | 0.8899 | 0.9410 | 0.0000 | 1.0000 |
| random_reference | s10_o3_BASE | PCA3 | 0.3925 | 0.3110 | 0.8743 | 0.2506 | 0.9174 | 0.8004 | 0.8243 | 0.1884 | 0.9105 |
| random_reference | s10_o3_BASE | RANDOM | 0.3652 | 0.2823 | 0.9060 | 0.2372 | 0.9339 | 0.8323 | 0.8539 | 0.1689 | 0.9425 |
| random_reference | s10_o3_BASE | INIT | 0.4141 | 0.3114 | 0.8867 | 0.2505 | 0.9306 | 0.8291 | 0.8357 | 0.1849 | 0.9392 |
| random_reference | s10_o3_BASE | FINAL | 0.3071 | 0.2393 | 0.9387 | 0.2191 | 0.9458 | 0.8539 | 0.8847 | 0.1441 | 0.9640 |
| random_reference | s10_o3_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0657 | 0.9935 | 0.9824 | 0.9911 | 0.0000 | 1.0000 |
| random_reference | s10_o3_FAIR95 | PCA3 | 0.4201 | 0.3185 | 0.8574 | 0.2285 | 0.9258 | 0.8621 | 0.8498 | 0.2158 | 0.8797 |
| random_reference | s10_o3_FAIR95 | RANDOM | 0.3954 | 0.2941 | 0.8885 | 0.2114 | 0.9442 | 0.9048 | 0.8806 | 0.1971 | 0.9223 |
| random_reference | s10_o3_FAIR95 | INIT | 0.4457 | 0.3243 | 0.8739 | 0.2286 | 0.9395 | 0.9098 | 0.8661 | 0.2154 | 0.9273 |
| random_reference | s10_o3_FAIR95 | FINAL | 0.3269 | 0.2435 | 0.9306 | 0.1813 | 0.9602 | 0.9348 | 0.9225 | 0.1651 | 0.9523 |
| random_reference | s12_o4_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1662 | 0.9669 | 0.8888 | 0.9402 | 0.0000 | 1.0000 |
| random_reference | s12_o4_BASE | PCA3 | 0.4296 | 0.3241 | 0.8493 | 0.2678 | 0.9032 | 0.7692 | 0.7996 | 0.2051 | 0.8804 |
| random_reference | s12_o4_BASE | RANDOM | 0.4065 | 0.3032 | 0.8761 | 0.2576 | 0.9157 | 0.7918 | 0.8244 | 0.1914 | 0.9031 |
| random_reference | s12_o4_BASE | INIT | 0.4776 | 0.3448 | 0.8376 | 0.2787 | 0.9073 | 0.7722 | 0.7871 | 0.2154 | 0.8834 |
| random_reference | s12_o4_BASE | FINAL | 0.3710 | 0.2761 | 0.9002 | 0.2449 | 0.9254 | 0.8159 | 0.8474 | 0.1742 | 0.9271 |
| random_reference | s12_o4_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0609 | 0.9940 | 0.9847 | 0.9922 | 0.0000 | 1.0000 |
| random_reference | s12_o4_FAIR95 | PCA3 | 0.4552 | 0.3288 | 0.8289 | 0.2450 | 0.9116 | 0.8259 | 0.8224 | 0.2344 | 0.8412 |
| random_reference | s12_o4_FAIR95 | RANDOM | 0.4315 | 0.3086 | 0.8620 | 0.2310 | 0.9280 | 0.8621 | 0.8552 | 0.2197 | 0.8775 |
| random_reference | s12_o4_FAIR95 | INIT | 0.5006 | 0.3487 | 0.8318 | 0.2552 | 0.9184 | 0.8674 | 0.8252 | 0.2446 | 0.8827 |
| random_reference | s12_o4_FAIR95 | FINAL | 0.3911 | 0.2794 | 0.8888 | 0.2116 | 0.9394 | 0.8898 | 0.8819 | 0.1993 | 0.9052 |
| root_candidate_grouped | s8_o2_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1447 | 0.9776 | 0.8966 | 0.9446 | 0.0000 | 1.0000 |
| root_candidate_grouped | s8_o2_BASE | PCA3 | 0.3568 | 0.2939 | 0.9053 | 0.2257 | 0.9372 | 0.8464 | 0.8574 | 0.1700 | 0.9499 |
| root_candidate_grouped | s8_o2_BASE | RANDOM | 0.2908 | 0.2367 | 0.9461 | 0.2007 | 0.9565 | 0.8692 | 0.8950 | 0.1359 | 0.9726 |
| root_candidate_grouped | s8_o2_BASE | INIT | 0.3223 | 0.2536 | 0.9427 | 0.2072 | 0.9586 | 0.8674 | 0.8918 | 0.1442 | 0.9708 |
| root_candidate_grouped | s8_o2_BASE | FINAL | 0.2265 | 0.1849 | 0.9773 | 0.1816 | 0.9679 | 0.8864 | 0.9238 | 0.1067 | 0.9899 |
| root_candidate_grouped | s8_o2_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0716 | 0.9929 | 0.9801 | 0.9899 | 0.0000 | 1.0000 |
| root_candidate_grouped | s8_o2_FAIR95 | PCA3 | 0.3852 | 0.3036 | 0.8910 | 0.2083 | 0.9420 | 0.9082 | 0.8823 | 0.1927 | 0.9280 |
| root_candidate_grouped | s8_o2_FAIR95 | RANDOM | 0.3234 | 0.2524 | 0.9328 | 0.1782 | 0.9636 | 0.9369 | 0.9235 | 0.1592 | 0.9567 |
| root_candidate_grouped | s8_o2_FAIR95 | INIT | 0.3619 | 0.2743 | 0.9223 | 0.1889 | 0.9639 | 0.9461 | 0.9131 | 0.1712 | 0.9659 |
| root_candidate_grouped | s8_o2_FAIR95 | FINAL | 0.2504 | 0.1943 | 0.9679 | 0.1466 | 0.9783 | 0.9591 | 0.9582 | 0.1238 | 0.9789 |
| root_candidate_grouped | s10_o3_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1618 | 0.9704 | 0.8899 | 0.9410 | 0.0000 | 1.0000 |
| root_candidate_grouped | s10_o3_BASE | PCA3 | 0.3924 | 0.3109 | 0.8750 | 0.2506 | 0.9175 | 0.8004 | 0.8249 | 0.1884 | 0.9105 |
| root_candidate_grouped | s10_o3_BASE | RANDOM | 0.3578 | 0.2782 | 0.9101 | 0.2356 | 0.9347 | 0.8315 | 0.8575 | 0.1671 | 0.9416 |
| root_candidate_grouped | s10_o3_BASE | INIT | 0.4209 | 0.3155 | 0.8771 | 0.2524 | 0.9293 | 0.8114 | 0.8260 | 0.1873 | 0.9215 |
| root_candidate_grouped | s10_o3_BASE | FINAL | 0.3061 | 0.2378 | 0.9402 | 0.2185 | 0.9463 | 0.8536 | 0.8859 | 0.1433 | 0.9637 |
| root_candidate_grouped | s10_o3_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0657 | 0.9935 | 0.9824 | 0.9911 | 0.0000 | 1.0000 |
| root_candidate_grouped | s10_o3_FAIR95 | PCA3 | 0.4202 | 0.3185 | 0.8575 | 0.2285 | 0.9259 | 0.8626 | 0.8500 | 0.2159 | 0.8801 |
| root_candidate_grouped | s10_o3_FAIR95 | RANDOM | 0.3749 | 0.2816 | 0.8997 | 0.2051 | 0.9456 | 0.8999 | 0.8918 | 0.1909 | 0.9174 |
| root_candidate_grouped | s10_o3_FAIR95 | INIT | 0.4426 | 0.3213 | 0.8775 | 0.2268 | 0.9411 | 0.9098 | 0.8697 | 0.2135 | 0.9274 |
| root_candidate_grouped | s10_o3_FAIR95 | FINAL | 0.3277 | 0.2444 | 0.9290 | 0.1817 | 0.9599 | 0.9315 | 0.9209 | 0.1655 | 0.9490 |
| root_candidate_grouped | s12_o4_BASE | RAW | 0.0000 | 0.0000 | 1.0000 | 0.1662 | 0.9669 | 0.8888 | 0.9402 | 0.0000 | 1.0000 |
| root_candidate_grouped | s12_o4_BASE | PCA3 | 0.4296 | 0.3241 | 0.8491 | 0.2678 | 0.9031 | 0.7692 | 0.7994 | 0.2051 | 0.8804 |
| root_candidate_grouped | s12_o4_BASE | RANDOM | 0.4043 | 0.3029 | 0.8763 | 0.2573 | 0.9158 | 0.7898 | 0.8246 | 0.1912 | 0.9010 |
| root_candidate_grouped | s12_o4_BASE | INIT | 0.4779 | 0.3450 | 0.8388 | 0.2788 | 0.9072 | 0.7850 | 0.7884 | 0.2154 | 0.8962 |
| root_candidate_grouped | s12_o4_BASE | FINAL | 0.3693 | 0.2754 | 0.9003 | 0.2445 | 0.9255 | 0.8155 | 0.8475 | 0.1739 | 0.9267 |
| root_candidate_grouped | s12_o4_FAIR95 | RAW | 0.0000 | 0.0000 | 1.0000 | 0.0609 | 0.9940 | 0.9847 | 0.9922 | 0.0000 | 1.0000 |
| root_candidate_grouped | s12_o4_FAIR95 | PCA3 | 0.4552 | 0.3287 | 0.8291 | 0.2450 | 0.9116 | 0.8263 | 0.8227 | 0.2344 | 0.8417 |
| root_candidate_grouped | s12_o4_FAIR95 | RANDOM | 0.4267 | 0.3049 | 0.8625 | 0.2288 | 0.9278 | 0.8578 | 0.8558 | 0.2175 | 0.8732 |
| root_candidate_grouped | s12_o4_FAIR95 | INIT | 0.4987 | 0.3479 | 0.8309 | 0.2547 | 0.9182 | 0.8657 | 0.8243 | 0.2440 | 0.8810 |
| root_candidate_grouped | s12_o4_FAIR95 | FINAL | 0.3914 | 0.2792 | 0.8884 | 0.2114 | 0.9394 | 0.8899 | 0.8815 | 0.1992 | 0.9052 |

> 核心读数（root-grouped，看 FINAL 行）：每行 `FINAL < INIT` 且 `FINAL < RANDOM < PCA3`（patch 误差逐级下降）→ 学习字典确实优于初始字典和两个基线。

## 6. 字典健康与标量（dictionary health & scalars）

> 基于训练编码统计（字典在其上训练）；FINAL 同时记录测试编码。`nondead`=被用到的原子数；`effective`=按激活熵折算的有效原子数；`max share`=单原子最大激活占比（越小越均匀，防坍缩）；`coherence`=字典原子互相的相似度（越低越独立）；`train rel err`=训练相对误差。

| 视角 | 分支 | 阶段 | 字典标量 | 每图编码标量 | 非死亡原子 | 有效原子 | 最大激活占比 | 原子一致性 | 训练相对误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_reference | s8_o2_BASE | RANDOM | 672 | 59.6 | 23 | 22.22 | 0.0674 | 1.0000 | 0.5858 |
| random_reference | s8_o2_BASE | INIT | 672 | 59.6 | 24 | 21.65 | 0.0939 | 0.3513 | 0.6543 |
| random_reference | s8_o2_BASE | FINAL | 672 | 59.6 | 24 | 22.71 | 0.0954 | 0.6430 | 0.4609 |
| random_reference | s8_o2_FAIR95 | RANDOM | 672 | 70.0 | 24 | 21.54 | 0.1141 | 0.8737 | 0.6144 |
| random_reference | s8_o2_FAIR95 | INIT | 672 | 70.0 | 24 | 19.82 | 0.1477 | 0.3111 | 0.6436 |
| random_reference | s8_o2_FAIR95 | FINAL | 672 | 70.0 | 24 | 22.97 | 0.0847 | 0.7733 | 0.4825 |
| random_reference | s10_o3_BASE | RANDOM | 1080 | 39.6 | 24 | 22.53 | 0.0786 | 0.8151 | 0.6739 |
| random_reference | s10_o3_BASE | INIT | 1080 | 39.6 | 24 | 19.76 | 0.1366 | 0.2678 | 0.7362 |
| random_reference | s10_o3_BASE | FINAL | 1080 | 39.6 | 24 | 23.39 | 0.0699 | 0.7384 | 0.5785 |
| random_reference | s10_o3_FAIR95 | RANDOM | 1080 | 50.8 | 24 | 22.16 | 0.1158 | 0.8503 | 0.7005 |
| random_reference | s10_o3_FAIR95 | INIT | 1080 | 50.8 | 24 | 19.72 | 0.1427 | 0.2443 | 0.7664 |
| random_reference | s10_o3_FAIR95 | FINAL | 1080 | 50.8 | 24 | 23.40 | 0.0617 | 0.8216 | 0.5925 |
| random_reference | s12_o4_BASE | RANDOM | 1584 | 28.9 | 24 | 22.80 | 0.0840 | 0.7945 | 0.7268 |
| random_reference | s12_o4_BASE | INIT | 1584 | 28.9 | 24 | 18.13 | 0.1713 | 0.2014 | 0.8109 |
| random_reference | s12_o4_BASE | FINAL | 1584 | 28.9 | 24 | 23.47 | 0.0599 | 0.8133 | 0.6537 |
| random_reference | s12_o4_FAIR95 | RANDOM | 1584 | 38.8 | 24 | 22.99 | 0.0762 | 0.7771 | 0.7537 |
| random_reference | s12_o4_FAIR95 | INIT | 1584 | 38.8 | 24 | 19.79 | 0.1581 | 0.2028 | 0.8178 |
| random_reference | s12_o4_FAIR95 | FINAL | 1584 | 38.8 | 24 | 23.12 | 0.0728 | 0.8427 | 0.6741 |
| root_candidate_grouped | s8_o2_BASE | RANDOM | 672 | 59.6 | 19 | 18.33 | 0.0839 | 1.0000 | 0.6067 |
| root_candidate_grouped | s8_o2_BASE | INIT | 672 | 59.6 | 24 | 20.76 | 0.1284 | 0.3275 | 0.6400 |
| root_candidate_grouped | s8_o2_BASE | FINAL | 672 | 59.6 | 24 | 22.43 | 0.0992 | 0.6949 | 0.4598 |
| root_candidate_grouped | s8_o2_FAIR95 | RANDOM | 672 | 70.0 | 24 | 22.82 | 0.0715 | 0.8656 | 0.5969 |
| root_candidate_grouped | s8_o2_FAIR95 | INIT | 672 | 70.0 | 24 | 19.84 | 0.1290 | 0.3184 | 0.6610 |
| root_candidate_grouped | s8_o2_FAIR95 | FINAL | 672 | 70.0 | 24 | 22.90 | 0.0857 | 0.7499 | 0.4698 |
| root_candidate_grouped | s10_o3_BASE | RANDOM | 1080 | 39.6 | 24 | 22.47 | 0.1068 | 0.8986 | 0.6878 |
| root_candidate_grouped | s10_o3_BASE | INIT | 1080 | 39.6 | 24 | 19.45 | 0.1482 | 0.2616 | 0.7553 |
| root_candidate_grouped | s10_o3_BASE | FINAL | 1080 | 39.6 | 24 | 22.83 | 0.0737 | 0.7385 | 0.5744 |
| root_candidate_grouped | s10_o3_FAIR95 | RANDOM | 1080 | 50.8 | 23 | 21.87 | 0.0925 | 1.0000 | 0.6789 |
| root_candidate_grouped | s10_o3_FAIR95 | INIT | 1080 | 50.8 | 24 | 19.57 | 0.1490 | 0.2489 | 0.7589 |
| root_candidate_grouped | s10_o3_FAIR95 | FINAL | 1080 | 50.8 | 24 | 23.13 | 0.0769 | 0.7648 | 0.5949 |
| root_candidate_grouped | s12_o4_BASE | RANDOM | 1584 | 28.9 | 23 | 21.40 | 0.0800 | 1.0000 | 0.7192 |
| root_candidate_grouped | s12_o4_BASE | INIT | 1584 | 28.9 | 24 | 18.03 | 0.1630 | 0.2067 | 0.7963 |
| root_candidate_grouped | s12_o4_BASE | FINAL | 1584 | 28.9 | 24 | 22.82 | 0.0892 | 0.8820 | 0.6508 |
| root_candidate_grouped | s12_o4_FAIR95 | RANDOM | 1584 | 38.8 | 24 | 22.67 | 0.0956 | 0.7915 | 0.7460 |
| root_candidate_grouped | s12_o4_FAIR95 | INIT | 1584 | 38.8 | 24 | 18.38 | 0.1681 | 0.1990 | 0.8114 |
| root_candidate_grouped | s12_o4_FAIR95 | FINAL | 1584 | 38.8 | 24 | 23.43 | 0.0605 | 0.8778 | 0.6760 |

> 读法：FINAL 的 24/24 原子全部 nondead、max share 约 0.06–0.10（无坍缩）、train rel err 比 INIT 低 → 字典健康。

## 7. 冻结门槛（gates）

### 7.1 划分/数据 gate

- `random_reference`：passed=`True`；检查项=`{'exposure_finite_in_unit_interval': True, 'fold_count_matches': True, 'raw_identity_and_decomposition': True}`（暴露率有限且在[0,1]内、折数匹配、原始恒等与分解正确）
- `root_candidate_grouped`：passed=`True`；检查项=`{'exposure_finite_in_unit_interval': True, 'fold_count_matches': True, 'root_intersection_zero': True, 'raw_identity_and_decomposition': True}`（根交集为 0）

### 7.2 KSVD 优化 gate（以 root-grouped 为主）

| 分支 | FINAL<INIT 3/3 折 | 加权改善 | ≥0.05 | 非死亡≥20 | 最大占比<0.5 | 观测 RMSE 不恶化 | 正改善折数 | random 视角改善 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s8_o2_BASE | True | 0.2973 | True | True | True | True | 3 | 0.3323 | True |
| s8_o2_FAIR95 | True | 0.3081 | True | True | True | True | 3 | 0.2976 | True |
| s10_o3_BASE | True | 0.2726 | True | True | True | True | 3 | 0.2584 | True |
| s10_o3_FAIR95 | True | 0.2596 | True | True | True | True | 3 | 0.2663 | True |
| s12_o4_BASE | True | 0.2272 | True | True | True | True | 3 | 0.2231 | True |
| s12_o4_FAIR95 | True | 0.2151 | True | True | True | True | 3 | 0.2186 | True |

> 关键：FINAL 相对 INIT 的 patch 误差加权下降 21.5%–30.8%，且两种划分视角方向一致 → 学习字典带来稳定增益。

### 7.3 对照 / Pareto gate（root-grouped 加权平均 patch 误差）

| 分支 | FINAL 误差 | RANDOM 误差 | PCA3 误差 | 胜 RANDOM | 胜 PCA3 | 通过 | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|
| s8_o2_BASE | 0.2265 | 0.2908 | 0.3568 | True | True | True | True |
| s8_o2_FAIR95 | 0.2504 | 0.3234 | 0.3852 | True | True | True | True |
| s10_o3_BASE | 0.3061 | 0.3578 | 0.3924 | True | True | True | True |
| s10_o3_FAIR95 | 0.3277 | 0.3749 | 0.4202 | True | True | True | True |
| s12_o4_BASE | 0.3693 | 0.4043 | 0.4296 | True | True | True | True |
| s12_o4_FAIR95 | 0.3914 | 0.4267 | 0.4552 | True | True | True | True |

> 六个分支全部同时胜 RANDOM 与 PCA3，且保留在 Pareto 前沿上（不同 geometry 对应质量/编码长度的权衡，没有单一赢家）。

## 8. 密度分层汇总（per view × branch）

| 视角 | 分支 | 密度层 | 图数 | RAW 误差 | PCA3 | RANDOM | INIT | FINAL | FINAL 校正 | 观测 RMSE | 全图召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_reference | s8_o2_BASE | lt5 | 100 | 0.0000 | 0.4420 | 0.3265 | 0.5249 | 0.2691 | 0.2691 | 0.1804 | 0.8712 |
| random_reference | s8_o2_BASE | d5_10 | 100 | 0.0000 | 0.3972 | 0.3236 | 0.3490 | 0.2610 | 0.2610 | 0.2093 | 0.8180 |
| random_reference | s8_o2_BASE | d10_15 | 100 | 0.0000 | 0.3440 | 0.2792 | 0.2935 | 0.2211 | 0.2211 | 0.1902 | 0.8700 |
| random_reference | s8_o2_BASE | d15_25 | 100 | 0.0000 | 0.3165 | 0.2564 | 0.2713 | 0.2014 | 0.2014 | 0.1797 | 0.9138 |
| random_reference | s8_o2_BASE | ge25 | 100 | 0.0000 | 0.2854 | 0.2230 | 0.2487 | 0.1729 | 0.1729 | 0.1586 | 0.9597 |
| random_reference | s8_o2_FAIR95 | lt5 | 100 | 0.0000 | 0.4509 | 0.3820 | 0.4886 | 0.2706 | 0.2706 | 0.1725 | 0.9629 |
| random_reference | s8_o2_FAIR95 | d5_10 | 100 | 0.0000 | 0.4481 | 0.3781 | 0.3978 | 0.3028 | 0.3028 | 0.2205 | 0.9558 |
| random_reference | s8_o2_FAIR95 | d10_15 | 100 | 0.0000 | 0.3906 | 0.3266 | 0.3365 | 0.2666 | 0.2666 | 0.2143 | 0.9665 |
| random_reference | s8_o2_FAIR95 | d15_25 | 100 | 0.0000 | 0.3458 | 0.2843 | 0.3010 | 0.2332 | 0.2332 | 0.1996 | 0.9657 |
| random_reference | s8_o2_FAIR95 | ge25 | 100 | 0.0000 | 0.2911 | 0.2255 | 0.2668 | 0.1841 | 0.1841 | 0.1676 | 0.9594 |
| random_reference | s10_o3_BASE | lt5 | 100 | 0.0000 | 0.4513 | 0.4853 | 0.6329 | 0.3596 | 0.3596 | 0.2154 | 0.8775 |
| random_reference | s10_o3_BASE | d5_10 | 100 | 0.0000 | 0.4516 | 0.4119 | 0.4498 | 0.3633 | 0.3633 | 0.2710 | 0.7917 |
| random_reference | s10_o3_BASE | d10_15 | 100 | 0.0000 | 0.3839 | 0.3437 | 0.3520 | 0.2993 | 0.2993 | 0.2523 | 0.8209 |
| random_reference | s10_o3_BASE | d15_25 | 100 | 0.0000 | 0.3565 | 0.3132 | 0.3305 | 0.2748 | 0.2748 | 0.2411 | 0.8642 |
| random_reference | s10_o3_BASE | ge25 | 100 | 0.0000 | 0.3193 | 0.2719 | 0.3053 | 0.2383 | 0.2383 | 0.2169 | 0.9151 |
| random_reference | s10_o3_FAIR95 | lt5 | 100 | 0.0000 | 0.4637 | 0.4894 | 0.6260 | 0.3472 | 0.3472 | 0.2028 | 0.9233 |
| random_reference | s10_o3_FAIR95 | d5_10 | 100 | 0.0000 | 0.4744 | 0.4405 | 0.4824 | 0.3744 | 0.3744 | 0.2588 | 0.9136 |
| random_reference | s10_o3_FAIR95 | d10_15 | 100 | 0.0000 | 0.4301 | 0.3960 | 0.4068 | 0.3419 | 0.3419 | 0.2648 | 0.9410 |
| random_reference | s10_o3_FAIR95 | d15_25 | 100 | 0.0000 | 0.3945 | 0.3567 | 0.3738 | 0.3118 | 0.3118 | 0.2599 | 0.9473 |
| random_reference | s10_o3_FAIR95 | ge25 | 100 | 0.0000 | 0.3379 | 0.2945 | 0.3393 | 0.2593 | 0.2593 | 0.2313 | 0.9488 |
| random_reference | s12_o4_BASE | lt5 | 100 | 0.0000 | 0.4855 | 0.4732 | 0.6992 | 0.4259 | 0.4259 | 0.2287 | 0.8315 |
| random_reference | s12_o4_BASE | d5_10 | 100 | 0.0000 | 0.4988 | 0.4725 | 0.5293 | 0.4387 | 0.4387 | 0.3028 | 0.7539 |
| random_reference | s12_o4_BASE | d10_15 | 100 | 0.0000 | 0.4228 | 0.3965 | 0.4142 | 0.3618 | 0.3618 | 0.2931 | 0.7849 |
| random_reference | s12_o4_BASE | d15_25 | 100 | 0.0000 | 0.3887 | 0.3627 | 0.3833 | 0.3302 | 0.3302 | 0.2860 | 0.8311 |
| random_reference | s12_o4_BASE | ge25 | 100 | 0.0000 | 0.3521 | 0.3275 | 0.3620 | 0.2983 | 0.2983 | 0.2700 | 0.8779 |
| random_reference | s12_o4_FAIR95 | lt5 | 100 | 0.0000 | 0.4929 | 0.4713 | 0.6807 | 0.4153 | 0.4153 | 0.2216 | 0.8576 |
| random_reference | s12_o4_FAIR95 | d5_10 | 100 | 0.0000 | 0.5129 | 0.4825 | 0.5472 | 0.4437 | 0.4437 | 0.2882 | 0.8450 |
| random_reference | s12_o4_FAIR95 | d10_15 | 100 | 0.0000 | 0.4655 | 0.4390 | 0.4598 | 0.4014 | 0.4014 | 0.2990 | 0.9003 |
| random_reference | s12_o4_FAIR95 | d15_25 | 100 | 0.0000 | 0.4294 | 0.4071 | 0.4252 | 0.3702 | 0.3702 | 0.3014 | 0.9186 |
| random_reference | s12_o4_FAIR95 | ge25 | 100 | 0.0000 | 0.3756 | 0.3579 | 0.3902 | 0.3248 | 0.3248 | 0.2868 | 0.9277 |
| root_candidate_grouped | s8_o2_BASE | lt5 | 100 | 0.0000 | 0.4411 | 0.3670 | 0.4770 | 0.2654 | 0.2654 | 0.1781 | 0.8698 |
| root_candidate_grouped | s8_o2_BASE | d5_10 | 100 | 0.0000 | 0.3972 | 0.3289 | 0.3363 | 0.2617 | 0.2617 | 0.2098 | 0.8194 |
| root_candidate_grouped | s8_o2_BASE | d10_15 | 100 | 0.0000 | 0.3439 | 0.2814 | 0.2893 | 0.2241 | 0.2241 | 0.1927 | 0.8695 |
| root_candidate_grouped | s8_o2_BASE | d15_25 | 100 | 0.0000 | 0.3165 | 0.2559 | 0.2690 | 0.2054 | 0.2054 | 0.1831 | 0.9142 |
| root_candidate_grouped | s8_o2_BASE | ge25 | 100 | 0.0000 | 0.2854 | 0.2206 | 0.2400 | 0.1758 | 0.1758 | 0.1610 | 0.9592 |
| root_candidate_grouped | s8_o2_FAIR95 | lt5 | 100 | 0.0000 | 0.4507 | 0.3783 | 0.5080 | 0.2648 | 0.2648 | 0.1685 | 0.9545 |
| root_candidate_grouped | s8_o2_FAIR95 | d5_10 | 100 | 0.0000 | 0.4480 | 0.3868 | 0.3977 | 0.3009 | 0.3009 | 0.2188 | 0.9525 |
| root_candidate_grouped | s8_o2_FAIR95 | d10_15 | 100 | 0.0000 | 0.3906 | 0.3357 | 0.3394 | 0.2669 | 0.2669 | 0.2153 | 0.9649 |
| root_candidate_grouped | s8_o2_FAIR95 | d15_25 | 100 | 0.0000 | 0.3457 | 0.2896 | 0.3015 | 0.2347 | 0.2347 | 0.2012 | 0.9645 |
| root_candidate_grouped | s8_o2_FAIR95 | ge25 | 100 | 0.0000 | 0.2911 | 0.2265 | 0.2630 | 0.1844 | 0.1844 | 0.1677 | 0.9590 |
| root_candidate_grouped | s10_o3_BASE | lt5 | 100 | 0.0000 | 0.4511 | 0.4506 | 0.6506 | 0.3589 | 0.3589 | 0.2135 | 0.8763 |
| root_candidate_grouped | s10_o3_BASE | d5_10 | 100 | 0.0000 | 0.4514 | 0.4084 | 0.4538 | 0.3593 | 0.3593 | 0.2675 | 0.7906 |
| root_candidate_grouped | s10_o3_BASE | d10_15 | 100 | 0.0000 | 0.3838 | 0.3419 | 0.3566 | 0.2983 | 0.2983 | 0.2506 | 0.8206 |
| root_candidate_grouped | s10_o3_BASE | d15_25 | 100 | 0.0000 | 0.3564 | 0.3124 | 0.3328 | 0.2742 | 0.2742 | 0.2398 | 0.8655 |
| root_candidate_grouped | s10_o3_BASE | ge25 | 100 | 0.0000 | 0.3192 | 0.2756 | 0.3106 | 0.2398 | 0.2398 | 0.2174 | 0.9149 |
| root_candidate_grouped | s10_o3_FAIR95 | lt5 | 100 | 0.0000 | 0.4639 | 0.4001 | 0.6228 | 0.3503 | 0.3503 | 0.2056 | 0.9176 |
| root_candidate_grouped | s10_o3_FAIR95 | d5_10 | 100 | 0.0000 | 0.4745 | 0.4234 | 0.4800 | 0.3763 | 0.3763 | 0.2607 | 0.9088 |
| root_candidate_grouped | s10_o3_FAIR95 | d10_15 | 100 | 0.0000 | 0.4302 | 0.3920 | 0.4054 | 0.3411 | 0.3411 | 0.2644 | 0.9383 |
| root_candidate_grouped | s10_o3_FAIR95 | d15_25 | 100 | 0.0000 | 0.3944 | 0.3575 | 0.3714 | 0.3114 | 0.3114 | 0.2602 | 0.9448 |
| root_candidate_grouped | s10_o3_FAIR95 | ge25 | 100 | 0.0000 | 0.3379 | 0.3018 | 0.3334 | 0.2592 | 0.2592 | 0.2311 | 0.9478 |
| root_candidate_grouped | s12_o4_BASE | lt5 | 100 | 0.0000 | 0.4854 | 0.4577 | 0.7067 | 0.4171 | 0.4171 | 0.2246 | 0.8343 |
| root_candidate_grouped | s12_o4_BASE | d5_10 | 100 | 0.0000 | 0.4989 | 0.4735 | 0.5299 | 0.4372 | 0.4372 | 0.3019 | 0.7520 |
| root_candidate_grouped | s12_o4_BASE | d10_15 | 100 | 0.0000 | 0.4227 | 0.3997 | 0.4115 | 0.3623 | 0.3623 | 0.2939 | 0.7827 |
| root_candidate_grouped | s12_o4_BASE | d15_25 | 100 | 0.0000 | 0.3887 | 0.3651 | 0.3816 | 0.3309 | 0.3309 | 0.2864 | 0.8314 |
| root_candidate_grouped | s12_o4_BASE | ge25 | 100 | 0.0000 | 0.3522 | 0.3256 | 0.3597 | 0.2989 | 0.2989 | 0.2704 | 0.8771 |
| root_candidate_grouped | s12_o4_FAIR95 | lt5 | 100 | 0.0000 | 0.4925 | 0.4567 | 0.6766 | 0.4164 | 0.4164 | 0.2212 | 0.8556 |
| root_candidate_grouped | s12_o4_FAIR95 | d5_10 | 100 | 0.0000 | 0.5129 | 0.4796 | 0.5454 | 0.4437 | 0.4437 | 0.2876 | 0.8437 |
| root_candidate_grouped | s12_o4_FAIR95 | d10_15 | 100 | 0.0000 | 0.4655 | 0.4374 | 0.4588 | 0.4025 | 0.4025 | 0.3000 | 0.8996 |
| root_candidate_grouped | s12_o4_FAIR95 | d15_25 | 100 | 0.0000 | 0.4294 | 0.4049 | 0.4240 | 0.3699 | 0.3699 | 0.3009 | 0.9210 |
| root_candidate_grouped | s12_o4_FAIR95 | ge25 | 100 | 0.0000 | 0.3756 | 0.3548 | 0.3887 | 0.3246 | 0.3246 | 0.2864 | 0.9295 |

> 读法：所有分支在所有密度层上 FINAL 都优于 INIT（稀疏层误差最高、密集层最低）；大 patch 在稀疏图上更难压缩，是全图误差的主要来源。

## 9. 边界（不可推而广之的点）

- 密度分层与 root_candidate 只用于划分/审计，不进入 patch 向量；源 ID 只用于暴露审计的记账。
- root-grouped 视角是"根候选分组"；`root_candidate` 尚未由生成方确认是真实采样根。
- patch 向量重叠是结构模板重复（canonical 邻接比特向量），不是源身份泄漏。
- 残差校正只隔离 patch 压缩误差；残差的比特成本不计入编码器，也不参与字典训练。
- 本协议只判定 KSVD 是否值得进入关系令牌消融；不自动证明语义 motif，不自动开始 Transformer 调参。
