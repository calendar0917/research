# IMDB-BINARY direct n-hop sampler feasibility audit

> 日期：2026-08-01  
> 性质：pre-KSVD sampler/ordering diagnostic  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_NHOP_SAMPLER_AUDIT_PROTOCOL_20260801.md`

## 1. Direct ego size feasibility

| radius | min/median/mean/p90/p95/max | <7 | =7 | >7 | equals whole graph |
|---:|---|---:|---:|---:|---:|
| 1 | 2/8.0/10.76/21.0/30.0/136 | 0.3156 | 0.1381 | 0.5464 | 0.1831 |
| 2 | 12/20.0/24.89/41.0/55.0/136 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

`radius=1` 若不截断就不是固定 7-node signal；`radius=2` 若接近整图，就不再是 local patch。

## 2. Selector substrate and relabel audit

`set match` 检查抽象节点集合；`ranked match` 检查排序邻接；`canonical match` 检查 exact rooted canonical 输出。多排序不自动保证这些指标。

| selector | cutoff tie | set match | ranked match | canonical match | clique mass | dominant mass | effective count | within-graph unique median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n_id | 0.8594 | 0.1741 | 0.6394 | 0.8396 | 0.5589 | 0.5589 | 6.7553 | 0.2143 |
| n_degree | 0.7705 | 0.3175 | 0.9611 | 0.9698 | 0.6221 | 0.6221 | 4.8278 | 0.1667 |
| n_signature | 0.7404 | 0.3440 | 0.9903 | 0.9952 | 0.6418 | 0.6418 | 4.7002 | 0.1538 |

WALK s=7 reference：clique/dominant mass `0.3359`，effective count `18.9290`，within-graph unique median `0.5000`。

## 3. Matched raw-stratified signal exposure

使用与 R0-P 相同且已经看过的 split seeds，因此只用于 matched diagnosis。

| feature | mean BA | split-seed std | per-seed means |
|---|---:|---:|---|
| stats | 0.7003 | 0.0054 | [0.7020, 0.7060, 0.6930] |
| walk_canonical_mean_std | 0.6297 | 0.0065 | [0.6220, 0.6380, 0.6290] |
| stats_plus_walk_canonical_mean_std | 0.6913 | 0.0024 | [0.6930, 0.6930, 0.6880] |
| n_id_canonical_mean_std | 0.6323 | 0.0061 | [0.6320, 0.6400, 0.6250] |
| stats_plus_n_id_canonical_mean_std | 0.6953 | 0.0040 | [0.7010, 0.6920, 0.6930] |
| n_degree_canonical_mean_std | 0.6450 | 0.0043 | [0.6390, 0.6470, 0.6490] |
| stats_plus_n_degree_canonical_mean_std | 0.6853 | 0.0082 | [0.6890, 0.6740, 0.6930] |
| n_signature_canonical_mean_std | 0.6467 | 0.0041 | [0.6520, 0.6420, 0.6460] |
| stats_plus_n_signature_canonical_mean_std | 0.6930 | 0.0090 | [0.6930, 0.6820, 0.7040] |

## 4. Frozen worth-dictionary checks

### n_id

- [ ] `canonical_relabel_match_ge_0_99`
- [ ] `dominant_mass_no_worse_than_walk`
- [ ] `effective_count_no_worse_than_walk`
- [ ] `standalone_ba_exceeds_walk_by_0_02`
- [ ] `conditional_not_below_stats`
- standalone gain over WALK：`0.0027`；
- conditional gain over STATS：`-0.0050`；
- worth dictionary gate：`FAIL`。

### n_degree

- [ ] `canonical_relabel_match_ge_0_99`
- [ ] `dominant_mass_no_worse_than_walk`
- [ ] `effective_count_no_worse_than_walk`
- [ ] `standalone_ba_exceeds_walk_by_0_02`
- [ ] `conditional_not_below_stats`
- standalone gain over WALK：`0.0153`；
- conditional gain over STATS：`-0.0150`；
- worth dictionary gate：`FAIL`。

### n_signature

- [x] `canonical_relabel_match_ge_0_99`
- [ ] `dominant_mass_no_worse_than_walk`
- [ ] `effective_count_no_worse_than_walk`
- [ ] `standalone_ba_exceeds_walk_by_0_02`
- [ ] `conditional_not_below_stats`
- standalone gain over WALK：`0.0170`；
- conditional gain over STATS：`-0.0073`；
- worth dictionary gate：`FAIL`。

## 5. Decision

> **FAIL_DIRECT_CAPPED_NHOP_SAMPLER**

Do not train n-hop KSVD. Direct capped ego selection did not jointly solve invariance, substrate diversity, and conditional signal.

本结果不说明所有 n-hop 表示都失败；它只判断最直接的 fixed-7 capped ego route 是否比当前 WALK 更适合继续。
