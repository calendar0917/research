# 短中间判断（审计完成时）

## 最可能解释 1：局部 junction 身份被 Null 分组稀释（H3）
- 支持：P3 的 Null 分组对每条键只给 `n_b × graph_mean(w)`，局部 junction 内容被全图均值替代；radius-2 未见环境覆盖 25.1% 分子却承担 49.8% 绝对误差；ring≥4 且 radius-2 未见环境的 47 个分子 MAE 0.314。
- 反对：J 已经注入完整 typed path 内容却更差；旧 organized 单臂在 official valid 上也不稳定；M/S 全干替换更差。说明“加局部内容”本身不等于“保住全局 P3 的归纳偏置”。
- 什么会改变判断：如果 dual-channel 的 organized 臂不优于同参数 Null-local 对照，则局部 ownership 不是可用的 MAE 杠杆，应停止结构接口路线。

## 最可能解释 2：共享的难例/尾部误差，不是 seed 噪声（H1/H4）
- 支持：7 个 checkpoint 的 signed error 相关 0.52–0.93；top 25% 分子承担 69.4% 绝对误差；P3 seed 代数方差只占 consensus MSE 7.7%；`y<=-3` 只占 6.5% 行但占 16.2% 误差。
- 反对：seed 平均仍有 0.00446 的 MAE 改善（CI 不含 0），所以 seed 不是零；但它是第二阶，不是平台主因。
- 什么会改变判断：如果 seed 间 top-100 难例集合几乎不重叠，才需要把 seed 分歧当主因；当前不是。

## 样本量验证是否购买
不购买。J 的 exact path 内容已覆盖 99.6% valid path occurrence；3k/6k 标签曲线约 0.30 GPU-h，只会再次测到同一个共享难例，无法区分“标签不够”与“ownership/readout 用不上”。若未来某个臂在 rare subgroup 出现正效应但总体被稀释，再购买曲线。

## 下一步唯一主干预
P3 dual-channel local/global composer：保留 P3 Null 全局通路，新增 organized 局部 composer（`Σ_{j∋b} w_j`）并把 `sum(c_loc)`、`sum(c_loc^2)` 接入 reader；同参数、同初始化的 Null-local 对照隔离 ownership。494,193 参数，无递归局部 MP。单 seed 两臂，预注册 gate；失败即关闭，不买 seed 1。
