# 图神经网络科研素养 Resources

## Knowledge

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
  官方实验实践指南，尤其用于检查预处理、数据泄漏、随机性和交叉验证。
- [scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
  官方 API 与实践说明，用于把预处理和模型绑定在同一训练边界内，降低交叉验证中的泄漏风险。
- [Open Graph Benchmark paper](https://arxiv.org/abs/2005.00687)
  图学习 benchmark 的原始论文，用于理解数据集、任务、split 和公平比较为何是研究结论的一部分。
- [XGBoost: Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
  XGBoost 官方原理教程，用于建立决策树集成、加法训练、二阶近似和正则化的基础理解。
- [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
  交互式 GNN 基础教程，用于理解图表示、消息传递、聚合、感受野和归纳偏置。
- [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
  GIN 原始论文，用于理解消息传递 GNN 的表达力与 Weisfeiler-Lehman 测试之间的关系。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  Transformer 原始论文，用于理解 attention 架构的基本动机；不把它当作图学习的直接教材。
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
  统计学习经典教材，提供线性模型、树、集成、核方法、模型评估和无监督学习的统一基础；官方页面提供全文。
- [Deep Learning](https://www.deeplearningbook.org/)
  Goodfellow、Bengio、Courville 的深度学习教材，用于神经网络、优化、正则化和表示学习基础。
- [Deep Learning, Chapter 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)
  深度学习教材的优化章节，用于理解经验风险、surrogate loss、梯度下降与 minibatch。
- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
  官方决策树文档，用于理解分段常数函数、贪心切分、过拟合、停止条件和剪枝。
- [scikit-learn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
  官方集成学习文档，用于理解 bagging、随机森林、梯度提升和弱学习器的加法组合。
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
  XGBoost 原始论文，用于理解正则化目标、加法训练、二阶近似、稀疏感知和系统实现。
- [Reproducibility in Machine Learning: Experiences from NeurIPS 2019 Reproducibility Program](https://arxiv.org/abs/2003.12206)
  经验性讨论机器学习复现中的报告与实验问题，用于建立“结果不是单个数字”的意识。
- [uv: Projects](https://docs.astral.sh/uv/concepts/projects/)
  uv 官方项目机制：pyproject、lock 与 sync 的设计，用于环境锁定与按项目隔离。
- [uv: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
  uv 官方 PyTorch 集成：torch 的 CUDA 变体、专用 index 与 GPU 扩展配置。
- [uv: Installing and managing Python](https://docs.astral.sh/uv/guides/install-python/)
  官方解释器管理：自动下载、复用系统解释器与版本锁定规则。
- [conda: Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
  conda 官方环境文档：创建、-p 位置、激活、导出与 lockfile，以及和 pip 混用的注意事项。
- [conda-forge](https://conda-forge.org/)
  社区 conda 频道，确认某项依赖是否存在官方二进制构建（用于判断是否必须读 conda 路线）。

## Wisdom (Communities)

暂不预设社区渠道。后续如果某个问题需要真实研究者经验，将根据具体方向筛选高信噪比的讨论社区或研究群体。

## Gaps

- 尚未按当前课题核验第一批主资源。
- 尚未确定用户最常见的 AI 报告类型和最希望审计的代码入口。
