# MolHIV rooted-WL 强方案冻结终端测试：2026-09-01

## 披露与协议

MolHIV official test 在仓库更早的路线中已经被查看过，因此本轮只能称为
**冻结后受控 terminal evaluation**，不能声称 test 完全 untouched。

本轮仍严格执行：

1. test 编码前预注册两个候选：`T+A` 主分支与 centered challenger；
2. 预注册固定 `0.5*T+A + 0.5*centered` probability late fusion；
3. 两个 view 分别在完整 official-train 三折 scaffold 上进行 12-trial Optuna；
4. 只在参数、view、五个 model seeds 和 ensemble rule 全部冻结后编码 test；
5. test 不参与参数、view、ensemble weight 或停止条件选择；
6. terminal ledger 已写入，入口会拒绝重复评估。

固定表示为 full all-center radius-2 induced ego、topology-only rooted-WL 和
strict chemistry；没有 K-SVD、attention 或 learned test-time fusion。

## Test 前冻结结果

| view | official-train scaffold CV | official-valid 5-seed mean | valid seed ensemble |
|---|---:|---:|---:|
| `T+A` | 0.7756 | 0.7945 | 0.7951 |
| centered | **0.7907** | **0.7962** | **0.7976** |

centered 在 train scaffold-CV 与 official-valid 上都略高，因此它是有真实竞争力的
预注册 challenger；但 test 中仍同时报告所有冻结候选，不根据 test 改选。

## Official test

### 只用 official train 拟合

| candidate | 5-seed mean AUC | seed/model ensemble AUC |
|---|---:|---:|
| `T+A` | 0.7644 | 0.7663 |
| centered | 0.7644 | 0.7663 |
| fixed 50/50 late fusion | **0.7789** | **0.7804** |

### official train+valid refit

| candidate | 5-seed mean AUC | seed/model ensemble AUC |
|---|---:|---:|
| `T+A` | 0.7610 | 0.7628 |
| centered | **0.7716** | **0.7732** |
| fixed 50/50 late fusion | **0.7825** | **0.7839** |

最终应引用的本轮最好数字是：

> clean rooted-WL 双分支、train+valid refit、固定 50/50 late fusion 的十模型
> ensemble official-test ROC-AUC = **0.7839**。

若只比较五个配对 seed 的平均 AUC，则为 **0.7825**；若要求单一表示，centered
为 **0.7716**，其五 seed probability ensemble 为 **0.7732**。

## 结果解释

### 1. 超参数对绝对表现重要

此前固定参数 official-valid 上 `T+A/centered` 约为 `0.7865/0.7872`。完整
train-scaffold 搜索后变为 `0.7945/0.7962`，说明“一组固定 XGBoost 参数比较所有
维度表示”会低估最终性能。

但 test 仍显著低于 valid：

- `T+A`：valid mean `0.7945` → train+valid test `0.7610`；
- centered：valid mean `0.7962` → train+valid test `0.7716`。

因此超参数能改善拟合，却没有消除 MolHIV 的 valid--test scaffold mismatch。

### 2. centered 单分支获得了一部分 test 增量

train+valid refit 下，centered 相对 `T+A` 为 `+0.0106`，并在 4/5 seeds 上更高。
这修正了“binding 完全没有预测作用”的过强表述：适当正则化并加入 valid 分布后，
centered interaction 对 test 有可见贡献。

更准确的结论仍是：

> binding 有真实、可利用的信息，但作为单一高维 feature-level fusion，其收益强烈
> 依赖训练 scaffold 与正则化，稳定性不足。

### 3. 最有价值的信息是 late fusion

固定 50/50 ensemble 在 train+valid refit 下：

- 相对 centered 单分支五 seed mean：`+0.0109`；
- 相对 centered seed ensemble：`+0.0107`；
- 五个 paired seeds 全部高于对应的两个单分支。

这说明 `T+A` 低容量 marginals expert 与 centered interaction expert 学到的是不同的
排序误差。此前把所有 interaction 直接拼入一个模型、或用 sparse residual 强迫其修正
同一 baseline，都没有利用这种互补性；独立训练后在预测层融合反而有效。

## 与历史结果对照

历史 radius-2 mentor proxy 的 train+valid test 五 seed mean：

- valid 阶段主候选 `S+R_raw`：`0.7804`；
- 预留诊断候选 `S+R_final`：`0.8022`。

本轮 clean late-fusion paired mean `0.7825`，略高于当时 valid 主候选 `0.7804`；
但低于历史诊断最大值 `0.8022`。由于 `S+R_final` 当时不是 valid 选出的主模型，而且
旧 proxy 存在截断、排序与类别碰撞等对象问题，不能根据已见 test 反向把它升级为当前
主路线，也不能将 `0.8022` 当作 clean 方法结论。

## 下一步判定

1. 不再基于 MolHIV official test 修改 view、权重或超参数；
2. 当前最值得继续的不是更大的 feature-level joint，而是**独立专家 + late fusion**；
3. 下一协议应在新的 repeated scaffold outer splits 或外部数据集上学习/验证融合，不能
   再使用当前 test；
4. 最小下一模型应为：

   ```text
   expert M: rooted-WL T+A marginals XGBoost
   expert B: rooted-WL T+A+centered binding XGBoost
   cross-fitted logits -> constrained late fusion
   ```

5. 首先比较固定 50/50、OOF 学到的单一非负权重和两输入 logistic stack；只有融合权重
   在多个 outer scaffold splits 中稳定，才考虑更多 expert 或 neural attention；
6. K-SVD 仍不是当前增量来源，不进入下一阶段。

冻结记录：

- [`structural_role_terminal_audit/frozen_manifest.json`](structural_role_terminal_audit/frozen_manifest.json)
- [`structural_role_terminal_audit/terminal_test.json`](structural_role_terminal_audit/terminal_test.json)
- [`structural_role_terminal_audit/terminal_test.md`](structural_role_terminal_audit/terminal_test.md)
