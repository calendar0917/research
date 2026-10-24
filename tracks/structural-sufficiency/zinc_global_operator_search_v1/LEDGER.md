# GOS-v1 实验账本

比分单位：官方 ZINC-subset（train 10,000 / valid 1,000），raw-y MAE，单模型，
部署参数 ≤500k，无预测融合 / seed 平均 / checkpoint soup。
`valid` 是开发集（已被历史多轮使用），不是盲测。test 只在冻结后一次性读取。

初始化：所有 run 都从 seed 配对的 train-only masked-SSL 编码器
（`pretrain_encoder_s0.pt` / `..._s1.pt`）加载 atom/bond/junction encoders，
其余模块随机初始化（同一 seed ⇒ 相同初态 sha，见各行 init 前缀）。
配方：`RECIPE B`（AdamW lr 1e-3→1e-5、wd 1e-5、batch 128、500 步 warmup、
per-step cosine、clip 5、raw-y L1、600 epoch = 47,400 updates、全量 valid 每 epoch、
strict best-valid 选点）。除 A-reg 一行外全部同配方。

| # | run | route | seed | commit | params | best-valid | ep | final | GPU·h | 备注 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | A_s0 | A | 0 | 19652e0 | 460,017 | 0.075919 | 402 | 0.076529 | 0.833 | 首个主结果 |
| 2 | C0_s0 | C0 | 0 | 19652e0 | 378,993 | 0.080462 | 419 | 0.081799 | 0.528 | 载体对照 |
| 3 | K_s0 | K | 0 | b245e88 | 452,457 | 0.081812 | 329 | 0.083661 | 0.869 | 有符号非竞争核 |
| 4 | A_s1 | A | 1 | b245e88 | 460,017 | — | — | — | — | 第二 seed（进行中） |
| 5 | C0_s1 | C0 | 1 | b245e88 | 378,993 | — | — | — | — | 载体第二 seed（进行中） |
| 6 | Asm_s0 | Asm | 0 | c79fe40 | 459,717 | — | — | — | — | 机制对照：普通归一化 attention |
| 7 | Areg_s0 | A | 0 | c79fe40 | 460,017 | — | — | — | — | 唯一修正版：wd 1e-3 + dropout 0.05 |

失败/浪费：无训练失败。`gos-timingA` 2 epoch 探针 0.003 GPU·h；
`gos-prep`（关系缓存构建，CPU）不计 GPU。

## 参考锚（历史，同一数据/配方/指标）

| 参考 | params | valid（seed 均值/range） | test 均值 | 来源 |
|---|---:|---|---:|---|
| DCC-L (B-R0, 4 seeds) | 494,193 | 0.075990 ± 0.002137 (0.073837–0.078537) | 0.067319 | `zinc_performance_search_v1` |
| DCC-L s0 | 494,193 | 0.078537 | 0.066267 | 同上 |

## 关键单调性检查（写在同一 epoch 上，不用“候选 ep100 vs 参考最终”）

见 `analysis/main_table.md` 的 matched-epochs 表。

## 终端 test 冻结规则（在 test 读取前写定，2026-10-25T18:10Z）

冻结家族 = **route A + 配方 B**，seed 0/1（预注册主配置），checkpoint = 各自 best-valid。
`Areg_s0`（同架构、R2 配方：wd 1e-3 + dropout 0.05，唯一一次有诊断依据的修正：
A 的 online train MAE 在 best epoch 已降到 0.013、训练末期 0.0005，valid 在 ep400 后进入平台）
的处置规则：

- 若 `Areg_s0` valid **≤ 0.0735**（比 A_s0 好 ≥0.0024，即超过本家族 seed 差 0.0013 的 ~2 倍且
  超过参考家族 seed sd），则作为同家族第三 checkpoint 一起冻结（但仍只有 1 个 seed，
  只能称"配方方向的单 seed 迹象"）。
- 否则 `Areg_s0` 只作 valid 诊断，不读 test，家族仍是 A_s0 + A_s1。

`N0`（同载体 + 键内容进 reader、无全局混合）是机制对照，**不是**候选，不进 test。
C0/K/Asm 同样不读 test。冻结文件 `analysis/test_freeze.json` 在第一次读 test 前写入，
之后不允许改 seed / 改 checkpoint / 改配方。
