# Next experiments — 结果解读

入口：`python -m code.run_next` → `results/next.json`

实现增量：

- `readout_X(..., mode="rich")`：std/sum/energy/分位数/winner 直方图  
- `make_distant_wedge`：长路径两端三角 vs 同端双三角  
- MUTAG：共享 \(D\) 下 struct / concat / 能量门控 / MLP  

---

## A. 三角任务 — 读出

| variant | Acc |
|---------|-----|
| shared basic + LR | 0.855 ± … |
| shared **rich** + LR | **0.860** |
| basic/rich + MLP | 0.78 / 0.76（更差） |

**解读**：rich 略升；小数据上 **MLP 不如 LR**（过拟合）。读出增强有限，共享字典仍是主贡献。

---

## B. 「多跳」distant triangles — 任务被局部信号破解

| 方法 | Acc（约） |
|------|-----------|
| degree / oracle 手写 | 0.73 |
| **B0 / B0_wide patch-pool** | **1.00** |
| **B0 shared KSVD** | **0.985–0.995** |
| B1 / M0 pool & KSVD | 0.80–0.95（**低于 B0**） |

**解读（重要）**：

1. 类1「同端两个三角」→ 锚点度数/局部 denser；类0「两端各一三角」→ 局部更均匀。  
2. **1-hop 星形已足够**，不是合格的 multi-hop 探针。  
3. RW 在此 **有害**：把判别性局部 smear 掉。  
4. 结论：**不是「RW 失败」**，是 **任务没逼出 RW**；且说明「乱加 RW 可能降分」。

**合格多跳任务（下轮）**：仅 \(C_4\) vs 更长环（1-hop 诱导只是 wedge，**看不到环**），或 SR 风格硬例。

---

## C. MUTAG 融合（诚实共享 \(D\)）

| 方法 | Acc |
|------|-----|
| attr / degree / attr+deg（基线） | ~0.86 / 0.88 / … |
| struct only（rich） | **0.692**（略升） |
| concat s+attr | 0.809（**仍 < attr**） |
| gate 能量缩放 + attr | **0.814**（略好于硬拼接） |
| MLP concat | 0.72（差） |

**解读**：门控略减伤害，**仍未超过纯属性**。结构通道在 MUTAG 上继续是弱/噪声项。

---

## 累计认知（做实到哪）

| 命题 | 证据 |
|------|------|
| 每图字典不可比 | 共享 \(D\)：合成 0.55→0.86 |
| Patch 里有结构信号 | patch-pool / oracle |
| 读出 rich 略有用 | +0.005 量级 |
| 小 MLP 不自动更好 | 多处更差 |
| RW 非自由午餐 | distant 上 B0 > M0 |
| MUTAG 结构未赢属性 | 门控后仍 < attr |

---

## 下一步（仍不上 molhiv）

1. **重做 multi-hop**：`C_4` vs `C_{\ge 6}` 单环图，禁止 local-stats 泄漏，强制 \(m\) 小的 B0 失败。  
2. 共享字典 + **仅邻接 Y**（关掉 append_local_stats）系统扫 \(m\)。  
3. MUTAG：结构作 **可选残差**（验证集选是否拼接），避免默认伤害。  

---

## 一句话

读出小修、门控小修；**最大新信息**是：在错误任务上 RW 会输给一阶。先设计「B0 必挂、更大感受野才分」的合成任务，再谈 RW 价值。
