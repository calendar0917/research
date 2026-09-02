# luyin14 relation-conditioned structured pursuit screen

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_STRUCTURED_PURSUIT_PROTOCOL_20260812.md`  
> 判定：`STRUCTURED_PURSUIT_NO_GO`

## 1. Gate

- TRUE structured > independent：`False`；
- TRUE structured > SHUFFLED structured：`False`；
- STATS+TRUE structured > STATS：`False`；
- support/reconstruction diagnostic：`False`。

## 2. Balanced accuracy 与 paired delta

| dataset | independent | TRUE | SHUFFLED | TRUE-independent | W/T/L | TRUE-SHUFFLED | W/T/L | STATS+TRUE-STATS | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 0.672 | 0.678 | 0.667 | +0.006 | 7/0/2 | +0.011 | 6/0/3 | -0.034 | 0/0/9 |
| IMDB-MULTI | 0.473 | 0.465 | 0.467 | -0.008 | 3/0/6 | -0.002 | 5/0/4 | -0.004 | 4/0/5 |
| MUTAG | 0.718 | 0.699 | 0.714 | -0.018 | 4/0/5 | -0.014 | 4/0/5 | -0.096 | 0/1/8 |
| PTC_MR | 0.523 | 0.520 | 0.500 | -0.003 | 6/0/3 | +0.020 | 5/1/3 | -0.022 | 2/0/7 |

## 3. 编码机制诊断

| dataset | support agreement independent→TRUE | changed patches | recon relative change | FINAL TRUE-INIT TRUE |
|---|---:|---:|---:|---:|
| IMDB-BINARY | 0.356→0.488 | 0.822 | +2.714 | -0.010 (2/0/7) |
| IMDB-MULTI | 0.327→0.414 | 0.530 | +2.250 | +0.002 (6/0/3) |
| MUTAG | 0.118→0.263 | 0.910 | +0.662 | -0.036 (3/0/6) |
| PTC_MR | 0.097→0.222 | 0.630 | +0.331 | +0.014 (7/0/2) |

## 4. 结论

固定 structured pursuit 未建立可迁移增益。结合 rich readout 与 occurrence 证据，普通 adjacency-patch KSVD 的下游路线应停止。
