# 下一轮：冻结 Full 表示，直接拟合图级原型字典读出

仓库审计起点：`calendar0917/research@339dbea63ade4fa1ab2e51c32bd8f04fd52964d7`。

这是一个**不重训骨干**的低成本筛查任务。包内完成了 NumPy 数值/合成/规模验证；没有 ZINC 数据、checkpoint 或真实 ZINC 新 MAE。

## 现在能支持的判断

最近几轮具有一致模式：字典与新增对象接受任务梯度，推理移除会严重损害预测，但最终 MAE 仍约0.12。Full 扩容主要增加训练拟合；typed-cycle 的 train-only ridge 增益没有转化为主模型收益。

后一个结果暴露了筛查协议的问题：0.3626→0.3313回答的是“弱回归器能否用新统计”，不是“新方案能否改善已有约0.119模型”。两者不能相互替代。合成反例也只证明可区分，不证明标签相关性。

本包合成检验演示了这个差别：候选相对弱线性参考改善0.05319，但相对强参考反而差0.00106。这不是 ZINC 分析；它验证新的购买门必须以实际强模型为基准。

旧仓库已有 MLP 容量、额外集合统计、低秩交互头和优化协议的负结果。本轮不重跑这些，也不直接宣布读出就是瓶颈。只购买一个用现有 checkpoint 即可完成的、训练目标可精确检查的函数族：**图级原型字典 + 带L2正则的 MAE**。

## 一个模型，两级字典

保持现有 Full m=3 的所有特征计算、局部任务字典与静态 pair，全部冻结。取得原 reader 之前的实际814维图表示 R。

用训练集 R 的代表坐标作为图级原型字典 C，K=256。它是训练集选出的固定代表字典，不是新增的端到端可学习矩阵；现有局部字典沿用已训练状态。

对 R 做固定 asinh、训练集拟合的标准化和四块平衡，得到 z。通过高斯核与各原型匹配，再用 Nyström 归一化，得到图级字典坐标 ψ。

`原图 → 现有局部字典/静态组合 → R → 图级原型字典 → 一个标量读出`。

直接拟合 y，不拟合旧模型残差；部署时删除旧 MLP 读出，不加旧预测，不做预测平均，没有第二个骨干。Nyström 变换可折叠到每个原型的标量值中，最终读出是 `offset + Σk kernel(z,Ck)*valuek`。

它仍属于字典主路径，但本轮不主张稀疏性、K-SVD优势或比稠密函数族更好。新增模型的训练策略是阶段式冻结拟合，不是从头端到端训练。

## 为什么这次计算更有价值

- 不改变表示，先查现有字典表示是否支持可兑现的改进；不再每次重建新对象再等320轮。
- 原型和正则都只由train决定；最终与同一个冻结Full在同样1000个valid分子上配对比较。
- MAE + L2 是凸目标，求解器输出 primal-dual gap，避免把求解未收敛误报为路线失败。
- 冻结特征一次导出，后续拟合是小矩阵运算。计算目标15–30分钟、任务总预算60分钟；实际吞吐由Agent测量，不能用合成数组计时代替ZINC耗时。

## 已完成的轻量验证

见 `graph_dictionary_acceptance.json`：

- 一维解析 MAE 最优解和独立 epigraph 优化核对；目标差约3.9e-9。
- Nyström 核矩阵恒等式独立核对；差约1.8e-10。
- 折叠读出与原坐标公式差约1.9e-12；分批、行顺序与重复原型处理通过。
- 合成非线性交互任务能够学习；这仅验证函数与求解器可用，**不是 ZINC 性能前景证据**。
- 10k×814合成相关数组、256原型、257读出系数的规模检查完成且达到最优性门。复杂度在本参考环境可承受，不推算实际ZINC质量。

PyTorch导出和部署脚手架仅做了源码核对与语法检查，尚未在本环境执行；正式Agent必须重新验收。

## 运行参考

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python validate_graph_dictionary.py
python run_cached_head.py fit --train full_train_R.npz --out fitted
python run_cached_head.py evaluate --model fitted/model.npz --valid full_valid_R.npz --out evaluated.json
```

第一条是合成验收。后两条要求真实Full特征缓存；正式仓库运行必须由新 `research run` runner调用这些API，不能绕开控制面。

`extract_full_features.py` 提供 source-audited 的导出入口；只接受train/valid，缺Full checkpoint会直接拒绝，不重训。

## 冻结购买门

新head拟合冻结后，仅评估一次official valid。相对**重新重放的同一Full**：

- MAE改善≥0.006；
- 新模型MAE≤0.113；
- 按固定ID划分的五个诊断组至少四组改善。

全部满足，才称本轮单checkpoint下有可继续的信号。否则停止这一个固定原型/核/目标配置，不加K、不扫核带宽、不换新对象、不启动320轮。解算未达到精度或预算不足，标为INCOMPLETE，不记成科学负结果。

这一次使用的是历史上反复使用的official valid，且Full的soup曾由它选择；因此结果仍是探索性信号，不是独立确认或多seed结论。负结果也不证明814维表示的理论上限。

完整任务见 `AGENT_PROMPT.md`；把整个交接包交给Agent，先读取该文件。只有包内代码和合成证据，不包含用户原始数据或大模型产物。

参考：Nyström归一化的官方说明 https://scikit-learn.org/1.9/modules/kernel_approximation.html ，Williams & Seeger (2001) https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf 。这些来源支持计算方法，不支持本方案的ZINC效果。
