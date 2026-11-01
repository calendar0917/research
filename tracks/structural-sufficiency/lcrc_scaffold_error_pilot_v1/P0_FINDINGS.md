# P0_FINDINGS — scaffold Test 逐分子误差归因

**复现**：旧 scaffold 9 个 checkpoint 在 CPU 重新推理；valid/test 的 RMSE、MAE 与旧 `metrics.json` 全部一致（最大差 <1e-5）。未重训，未修改旧 checkpoint。

**1. MAE 收益广泛吗？** 只有“略过半”。G2L-ALL 在 scaffold Test 三个 seed 中 ΔAE>0 的比例为 53.6%、56.4%、50.7%（valid 55.0/51.9/51.4%）；三 seed 平均 MAE 0.53978→0.52491（+0.01487）。但中位 ΔAE 仅 +0.002~+0.036，5%/95% 分位约 -0.36/+0.41：多数分子只是小幅改善，同时不少分子小幅变差。

**2. RMSE 损失集中吗？** 是。Test 上 G2L 的 gross SSE damage 为 44.9/60.9/54.5；top 5% 损失分子贡献 23.9/29.8/27.0%，top 10% 为 36.6/45.0/40.8%，top 20% 为 51.5/59.9/55.2%。seed 1/2 的净 SSE 恶化只有 +11.0/+4.3，而 top 5% 损失已达 37.1/30.7（3.4/7.1 倍）。剔除 top 5% 损失分子后 G2L 在三个 seed 都反超 BASE；按 top BASE 误差剔除则不会。此剔除只用于来源诊断，不能替代完整测试指标。

**3. 化学特征？** 没有强单调关系：true logD、重原子、芳香环、杂原子的 |ρ| 多 <0.13。Test 上唯一较明显的线索是 max Tanimoto to Train：ρ(ΔSE)=+0.150；[0.2,0.3) 组（n=50）平均 ΔAE=-0.075、ΔSE=-0.215，而 [0.7,1.0] 组为正。Top 20 平均损失分子来自 20 个不同 Bemis–Murcko scaffold，没有共同骨架。

**4. 仍是猜想。** “G2L 在训练集附近化学空间有帮助、在远处放大误差”与 test 相似度模式一致，但 valid 的 [0.2,0.3) 组方向相反（ΔSE=+0.272），未复现。极端 |BASE err|>2.5 的分子在 test 全部变差，但 valid 的 2 个反而改善，故“放大最困难分子”不是跨 split 结论。严重损失也不是固定分子：top 5% 三 seed 交集仅 2/21，top 10% 为 8/42，损失秩相关 0.22–0.31；更准确的说法是“每次训练有少数候选特异性失败分子”，row 231 是唯一三 seed 都进入 top5% 的突出例子。

**5. 数据/指标/实现错误？** 未发现新错误。row ID、标签、checkpoint 哈希、Train-only normalization 与旧 scaffold 阶段一致，指标可复现。唯一警告：scaffold Test 仅 420 行，top-k 结论必须由 P1 新划分复核。
