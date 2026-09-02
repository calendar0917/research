# HOD-GNN 基线复现与可信度审计：实现计划

> 目的：把本文件交给另一个实现模型后，可以直接按文档完成代码、环境、实验和结果登记。
>
> 日期：2026-08-24  
> 任务来源：docs/luyin/luyin15.txt  
> 目标论文：https://arxiv.org/html/2510.02565v1  
> 本地精读：docs/literature/deep/hod-gnn.md

## 1. 任务定义

本 track 是 HOD-GNN 论文的旁路复现实验，不扩展 gnn-gsn 归档轨，也不与 ksvd 主线混合。

本轮重点不是重新实现 HOD-GNN，而是审计论文中与 HOD-GNN 比较的基线结果。优先使用各方法的官方实验代码、官方配置和官方数据处理；只有在官方代码无法运行时，才允许做最小兼容性 patch，并且必须保存 patch、原始 commit 和偏差说明。

本轮负责的方法：

1. GPS
2. GraphViT
3. Full
4. Random
5. Policy-Learn

其中 Full、Random、Policy-Learn 来自 Bevilacqua 等人的 Efficient Subgraph GNNs by Learning Effective Selection Policies（ICLR 2024）；GraphViT 为 He 等人 2023 年工作；GPS 为 Rampášek 等人 2022 年工作。虽然 GPS/GraphViT 不严格属于“2023 年以后”，但它们是录音中明确分配的任务，必须纳入本轮。

## 2. 实验范围

只复现 HOD-GNN 论文实际报告过的表格单元。论文中的 “–” 不补跑，不用其他论文的数字填充。

| 方法 | ZINC-12K | MOLTOX21 | MOLBACE | MOLHIV | Peptides-func | Peptides-struct |
|---|---:|---:|---:|---:|---:|---:|
| GPS | ✓ | ✓ | – | ✓ | – | – |
| GraphViT | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| Full | ✓ | ✓ | ✓ | ✓ | – | – |
| Random | ✓ | ✓ | ✓ | ✓ | – | – |
| Policy-Learn | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

注意：

- GPS 与 GraphGPS+LapPE 是不同实验单元，不能互相替代。
- Full、Random、Policy-Learn 是子图选择策略/模型变体，应尽可能共用其官方代码，只切换 policy 配置。
- Peptides 上论文只报告了 GraphViT 和 Policy-Learn；Full/Random 不得为了补全表格而强行运行。
- MOLBACE 上 GPS/GraphViT 在论文表格中为缺失值，不运行主协议。

## 3. 交付目录

建议目录如下：

~~~text
tracks/hod-gnn-replication/
├── TRACK.md
├── configs/
│   ├── method_dataset_matrix.yaml
│   ├── seed_registry.yaml
│   ├── dataset_registry.yaml
│   └── environment-lock.yml
├── code/
│   ├── run_gps.py
│   ├── run_graphvit.py
│   ├── run_subgraph_policy.py
│   ├── audit_splits.py
│   ├── collect_run.py
│   └── summarize_results.py
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── SOURCE_LEDGER.md
│   ├── PROTOCOL.md
│   ├── DEVIATIONS.md
│   └── REPRO_REPORT.md
├── results/
│   ├── registry.jsonl
│   ├── paper-seeds/
│   ├── official-seeds/
│   └── artifacts/
└── notes/
    ├── blockers.md
    └── decisions.md
~~~

如果仓库当前还没有 track 入口，可以先执行：

~~~bash
./scripts/init_track.sh hod-gnn-replication "HOD-GNN 基线复现审计"
~~~

然后把本文件放回 tracks/hod-gnn-replication/docs/IMPLEMENTATION_PLAN.md，并补齐上述目录。

## 4. 原始代码优先原则

这是本任务最重要的实现要求。

### 4.1 不要从头重写算法

实现模型必须先完成以下工作，再写 wrapper：

1. 找到 GPS 官方代码仓库、GraphViT 官方代码仓库、Full/Random/Policy-Learn 官方代码仓库。
2. 记录仓库 URL、commit、tag、分支和原始 README。
3. 确认官方代码能否直接处理论文中的数据集。
4. 尽量直接调用官方训练入口。
5. wrapper 只负责传参、设置 seed、记录日志和归档输出。

不允许把官方方法替换成“看起来相似”的 PyG 实现后称为复现。例如：

- 不能用一个自写 Transformer 代替 GraphViT；
- 不能用普通 GPS 代替 GraphViT；
- 不能用自写的 ego-net sampler 代替 Full/Random/Policy-Learn；
- 不能把 GPS+LapPE 的结果写成 GPS 结果。

### 4.2 允许的最小 patch

只有以下情况可以 patch：

- PyTorch/PyG API 已删除或改名；
- 数据下载 URL 失效但数据内容可验证一致；
- 旧版 Python 语法与当前解释器不兼容；
- 官方代码缺少非核心的日志、checkpoint 或 seed 参数。

每个 patch 必须保存：

~~~text
source_commit.txt
compatibility.patch
patch_reason.md
~~~

如果需要重写核心模型、采样策略、损失函数或数据划分，主协议应标记为 unavailable，改写后的结果只能放到 exploratory，不能写成严格复现。

## 5. 两套实验协议

必须按以下顺序执行：先原论文 seed，后官方 seed。两套协议的结果不能混合统计。

### 5.1 paper-seeds

目标：尽可能重现 HOD-GNN 论文表格中的结果。

对于每个方法和数据集，查找 seed 的顺序：

1. HOD-GNN 论文引用的代码和配置；
2. 各基线论文的官方代码默认 seed；
3. 官方 shell/python 运行脚本；
4. W&B 公共运行记录；
5. 补充材料或作者仓库；
6. 最后再向导师/作者确认。

HOD-GNN 论文只写明“4 个不同随机种子”，没有在正文或附录明确列出 seed 数值。因此不能直接把 [0, 1, 2, 3] 称为原论文 seed。找不到精确 seed 时，配置必须写成：

~~~yaml
paper_seeds:
  values: null
  status: unresolved
  source: "paper says four runs but does not disclose exact values"
~~~

若需要推进实验，可使用 [0, 1, 2, 3] 作为临时替代，但协议名称必须是 paper-seeds-provisional，不能冒充严格论文复现。

### 5.2 official-seeds

目标：使用数据集官方 split 和官方规定的 seed 口径复跑同一套模型配置。

数据集处理要求：

- OGB：使用 OGB 数据包提供的 scaffold split；保存 split 文件或 get_idx_split() 输出的 hash。
- ZINC：使用官方预定义 train/validation/test split。
- Peptides：使用 LRGB 官方 split。
- 若数据集只有固定 split、没有官方 model seed，必须分开记录“官方 split”和“模型 seed”，不能假装存在一个官方 seed。

配置建议：

~~~yaml
dataset: molhiv
official_split:
  source: ogb
  name: scaffold
  checksum: "..."
official_model_seeds:
  values: [ ... ]
  source: "..."
~~~

官方协议只替换 seed/split 口径，不重新调参。若要调参，另开 official-tuned，不进入本任务主表。

## 6. 数据集和指标

指标必须与 HOD-GNN 论文一致：

| 数据集 | 任务 | 指标 | 方向 |
|---|---|---|---|
| ZINC-12K | 图回归 | MAE | 越低越好 |
| MOLTOX21 | 分子性质分类 | ROC-AUC | 越高越好 |
| MOLBACE | 分子性质分类 | ROC-AUC | 越高越好 |
| MOLHIV | 分子性质分类 | ROC-AUC | 越高越好 |
| Peptides-func | 多标签分类 | Average Precision | 越高越好 |
| Peptides-struct | 多目标回归 | MAE | 越低越好 |

必须记录：

- 原始数据 URL 或下载来源；
- 文件 hash；
- 数据版本；
- 图数量；
- 节点/边特征维度；
- train/valid/test 数量；
- split hash；
- 是否使用官方预处理；
- 是否访问 test 标签进行调参。

## 7. 训练和评估规则

### 7.1 论文公共设置

HOD-GNN 附录中给出的公共设置如下，作为交叉检查：

- OGB：100 epochs；
- ZINC：2000 epochs；
- Peptides：250 epochs；
- AdamW；
- linear warmup + cosine decay；
- 根据 validation 最优 epoch 选择 test 结果；
- MOLHIV batch size 128；
- 其他 OGB 数据集 batch size 32；
- ZINC batch size 32。

但如果某个基线官方代码有自己的训练设置，优先保留基线原配置，并在 DEVIATIONS.md 中明确说明差异。不要用 HOD-GNN 的 HOD 配置覆盖基线配置。

### 7.2 随机性记录

每个运行至少记录以下 seed：

- global seed；
- Python random seed；
- NumPy seed；
- PyTorch CPU/CUDA seed；
- dataloader worker seed；
- sampler/subgraph seed；
- policy network seed（Policy-Learn）；
- split seed 或 split hash。

Random 和 Policy-Learn 必须单独记录采样随机性，否则无法判断结果差异来自模型参数还是子图采样。

### 7.3 确定性和硬件

记录：

- Python/PyTorch/PyG/CUDA 版本；
- GPU 型号和显存；
- 是否启用 AMP；
- 是否启用 deterministic algorithms；
- cudnn deterministic/benchmark 配置；
- 峰值显存；
- 每 epoch 时间；
- 总训练时间。

## 8. 分阶段执行流程

### Phase 0：来源登记

完成 SOURCE_LEDGER.md：

- 论文元数据；
- 官方仓库 URL；
- commit/tag；
- 依赖版本；
- 默认配置入口；
- 原始 seed 来源；
- 数据预处理入口；
- 论文表格中的目标数字。

出口条件：五个方法的代码来源和 seed 状态均已登记。

### Phase 1：环境和 split 审计

完成独立环境或锁定依赖，运行 audit_splits.py，生成：

~~~text
results/artifacts/dataset_manifest.json
results/artifacts/split_manifest.json
~~~

出口条件：所有主数据集 split 可验证，且无 train/test 泄漏。

### Phase 2：Smoke test

每个方法先选 ZINC，运行 10–20 个 batch：

- forward/backward 成功；
- loss finite；
- checkpoint 可保存和加载；
- sampler 输出合法；
- 指标计算正确；
- 显存可接受。

Smoke test 不计入正式均值。

### Phase 3：paper-seeds 正式实验

建议顺序：

1. GPS + ZINC；
2. GraphViT + ZINC；
3. Full + ZINC；
4. Random + ZINC；
5. Policy-Learn + ZINC；
6. 再扩展到 OGB；
7. 最后运行 Peptides 上的 GraphViT 和 Policy-Learn。

每个运行完成后立即写入 results/registry.jsonl，不能等所有实验结束后补记。

### Phase 4：official-seeds 正式实验

保持代码 commit、超参数和训练规则不变，只切换官方 split/seed 协议。结果写入 results/official-seeds/。

### Phase 5：汇总和审计

生成：

- 论文结果 vs 复现结果表；
- mean/std/CI；
- 每个 seed 的原始结果；
- paired seed 差值；
- 成功/失败/OOM 统计；
- 显存和运行时间；
- 方法排名稳定性；
- 协议偏差表。

## 9. 结果登记格式

每次运行一行 JSONL，至少包含：

~~~json
{
  "method": "policy-learn",
  "dataset": "molhiv",
  "protocol": "paper-seeds",
  "seed": 0,
  "sampler_seed": 0,
  "code_repo": "...",
  "code_commit": "...",
  "config_hash": "...",
  "split_hash": "...",
  "metric_name": "roc_auc",
  "best_epoch": 73,
  "valid_metric": 0.812,
  "test_metric": 0.784,
  "peak_memory_mib": 12345,
  "train_seconds": 4567,
  "status": "success",
  "failure_reason": null,
  "patch_id": null
}
~~~

允许的 status：

~~~text
success
oom
runtime_error
data_error
unavailable
implementation_drift
~~~

## 10. 失败、OOM 和兼容性规则

### OOM

如果论文配置 OOM：

1. 先记录原始 batch size、图规模、峰值显存和 traceback；
2. 不直接降低 batch size 后冒充严格复现；
3. 可以额外做 resource-adjusted 实验；
4. 主协议标记为 oom-under-paper-config。

### 运行失败

同一 seed 允许在完全相同配置下重跑一次。第二次仍失败时，不换 seed，不删除记录，将失败原因写入 registry 和 notes/blockers.md。

### 核心代码无法复原

如果需要重写核心模型、采样器、损失函数或数据划分，则：

- 主协议标记 unavailable；
- 改写版本只能放 exploratory；
- 不得在最终报告中称其为官方复现。

## 11. 统计和最终判定

每个方法/数据集/协议至少输出：

- mean；
- standard deviation；
- median；
- 95% confidence interval；
- 有效运行数；
- 失败运行数；
- 每个 seed 的原始结果。

论文使用 Welch’s t-test，阈值为较宽松的 p < 0.2。可以复现该统计口径，但不能只依赖 p 值。还必须报告配对 seed 差值和置信区间。

最终状态使用以下标签：

~~~text
reproduced
partially_reproduced
not_reproduced
unsupported
inconclusive_due_to_missing_seed
~~~

不能因为一个 seed 结果好，就判定方法复现成功；也不能因为某个 seed OOM，就删除该 seed 后计算均值。

## 12. 最终交付物

实现模型完成后必须生成：

1. configs/seed_registry.yaml；
2. configs/method_dataset_matrix.yaml；
3. docs/SOURCE_LEDGER.md；
4. docs/DEVIATIONS.md；
5. results/registry.jsonl；
6. results/paper_vs_reproduced.csv；
7. docs/REPRO_REPORT.md；
8. 每个方法的运行命令和复现说明；
9. OOM/失败/缺失 seed 清单；
10. 论文表格与复现表格的逐单元对照。

最终报告必须回答：

- 哪些基线能在论文 seed 下复现；
- 哪些基线只能在官方 split 下运行；
- 哪些结果依赖某个特殊 seed；
- Full/Random/Policy-Learn 的差异是否稳定；
- GraphViT 在 Peptides 上是否能复现论文结果；
- 论文中 HOD-GNN 的相对排名是否在重新运行后保持；
- 哪些结论受缺少精确 seed、代码漂移或资源限制影响。

## 13. 实现模型的第一轮任务

不要一开始就跑完整矩阵。第一轮只完成以下内容：

1. 创建 track 和目录；
2. 找到五个方法的官方代码；
3. 写完 SOURCE_LEDGER.md；
4. 写完 seed_registry.yaml；
5. 锁定 Python/PyTorch/PyG/CUDA 环境；
6. 校验 ZINC 和 MOLHIV split；
7. 完成五个方法的 ZINC smoke test；
8. 生成一份空的 registry.jsonl schema；
9. 只在 smoke test 全部通过后，开始第一组 paper-seeds 正式运行。

不要在 seed 来源未确认、split 未校验、官方代码未定位之前进行大规模实验。

