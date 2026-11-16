# Structural query learning pilot — sq-pilot-small-v1

官方 OGB `ogbg-molesol` / `ogbg-mollipo`，官方 Train/Validation/Test 划分，单训练 seed=0，`test_policy: always`（Test 作为本轮开发评估集）。本文件由 `stage=aggregate` 依据两个 GPU worker 的产物自动生成。

## 主表（每任务每方法：Val 选中状态，原始单位）

| dataset | method | selected state | #queries | #feat | #trees | Train RMSE | Val RMSE | Test RMSE | Test MAE | fit s | complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | F | H1@779 | 172 | 693 | 800 | 0.2875 | 1.0901 | 1.0400 | 0.8023 | 6.7 | yes |
| ogbg-molesol | R | H2@800 | 236 | 949 | 800 | 0.2012 | 1.1191 | 1.0841 | 0.8026 | 6.9 | yes |
| ogbg-molesol | A | H1@800 | 236 | 949 | 800 | 0.2617 | 1.0871 | 1.0907 | 0.8132 | 6.8 | yes |
| ogbg-molesol | E | H1@798 | - | 2059 | 800 | 0.3700 | 0.9270 | 0.8159 | 0.6539 | 10.6 | yes |
| ogbg-molesol | G | peak_lr=0.001@179 | - | - | - | 0.3299 | 0.9353 | 0.9354 | 0.7288 | 244.8 | yes |
| ogbg-molesol | L | R3@248 | - | - | - | 0.3462 | 0.8686 | 0.8570 | 0.6686 | 131.8 | yes |
| ogbg-mollipo | F | H2@800 | 203 | 817 | 800 | 0.3434 | 0.8313 | 0.7758 | 0.6104 | 6.8 | yes |
| ogbg-mollipo | R | H2@800 | 264 | 1061 | 800 | 0.3265 | 0.8019 | 0.7588 | 0.5893 | 7.7 | yes |
| ogbg-mollipo | A | H2@799 | 267 | 1073 | 800 | 0.2965 | 0.7911 | 0.7525 | 0.5892 | 8.4 | yes |
| ogbg-mollipo | E | H2@800 | - | 2059 | 800 | 0.4728 | 0.7747 | 0.7628 | 0.5905 | 11.1 | yes |
| ogbg-mollipo | G | peak_lr=0.001@168 | - | - | - | 0.2691 | 0.6091 | 0.6456 | 0.4847 | 923.6 | yes |
| ogbg-mollipo | L | R3@275 | - | - | - | 0.2631 | 0.7052 | 0.7139 | 0.5252 | 545.0 | yes |

## F / R / A 对照与配对不确定性

| dataset | comparison | split | RMSE A | RMSE ref | ΔRMSE (A−ref) | rel Δ | CI95 low | CI95 high | drop-1 Δ | flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | A - F | valid | 1.0871 | 1.0901 | -0.0029 | -0.0027 | -0.0761 | 0.0749 | -0.0101 | stable |
| ogbg-molesol | A - F | test | 1.0907 | 1.0400 | 0.0507 | 0.0487 | -0.0422 | 0.1418 | 0.0419 | stable |
| ogbg-molesol | A - R | valid | 1.0871 | 1.1191 | -0.0319 | -0.0285 | -0.1038 | 0.0389 | -0.0331 | stable |
| ogbg-molesol | A - R | test | 1.0907 | 1.0841 | 0.0066 | 0.0061 | -0.0607 | 0.0715 | 0.0013 | stable |
| ogbg-molesol | A - L | valid | 1.0871 | 0.8686 | 0.2185 | 0.2515 | 0.0284 | 0.4304 | 0.2735 | stable |
| ogbg-molesol | A - L | test | 1.0907 | 0.8570 | 0.2337 | 0.2726 | 0.0602 | 0.4169 | 0.2557 | stable |
| ogbg-mollipo | A - F | valid | 0.7911 | 0.8313 | -0.0402 | -0.0484 | -0.0842 | 0.0027 | -0.0298 | stable |
| ogbg-mollipo | A - F | test | 0.7525 | 0.7758 | -0.0233 | -0.0301 | -0.0595 | 0.0157 | -0.0189 | stable |
| ogbg-mollipo | A - R | valid | 0.7911 | 0.8019 | -0.0109 | -0.0135 | -0.0527 | 0.0303 | -0.0036 | stable |
| ogbg-mollipo | A - R | test | 0.7525 | 0.7588 | -0.0063 | -0.0083 | -0.0456 | 0.0320 | -0.0019 | stable |
| ogbg-mollipo | A - G | valid | 0.7911 | 0.6091 | 0.1820 | 0.2988 | 0.1222 | 0.2373 | 0.1940 | stable |
| ogbg-mollipo | A - G | test | 0.7525 | 0.6456 | 0.1069 | 0.1655 | 0.0512 | 0.1661 | 0.1139 | stable |

## 协议轴结论

- PERFORMANCE 轴：**PERFORMANCE_BEHIND**
- QUERY-LEARNING 轴：**CONTROL_INCOMPLETE**
- EXECUTABLE-EXPORT 轴：**EXPORT_PASS**
- PROMISING_SMALL_DATA_PILOT：**False**

## 每任务查询规模与导出

| dataset | B | discovered | R filled | trees | nodes | queries.json bytes | trees bytes | replay max abs err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 172 | 64 | 64/64 | 800 | 9338 | 287567 | 957227 | 0.00000636 |
| ogbg-mollipo | 203 | 64 | 61/64 | 799 | 37661 | 334212 | 3396531 | 0.00000499 |

## 资源账本（每个 worker 独占一块 GPU）

| dataset | discovery s | R s | XGB heads s | G s | L s | 合计 s | XGB device | torch device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 16.0 | 0.7 | 31.0 | 244.8 | 131.8 | 424.4 | cuda:0 | cuda |
| ogbg-mollipo | 33.2 | 7.2 | 34.0 | 923.6 | 545.0 | 1543.0 | cuda:0 | cuda |

- 合计 GPU·s = 1967.3，GPU·h = 0.55；两 worker 的 stage 时间之和 = 0.33 h（两个 worker 并行执行，实际墙钟取较大者）；本轮预算 8 h 墙钟 / 6 GPU·h。
