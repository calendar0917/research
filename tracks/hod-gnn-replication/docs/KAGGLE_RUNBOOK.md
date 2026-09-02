# Kaggle 运行手册

本 track 已准备好可推送的 notebook：`kaggle/hod-gnn-run.ipynb`，元数据在
`kaggle/kernel-metadata.json`。脚本会在 Kaggle GPU + Internet 环境内安装旧版
PyTorch/PyG、固定 commit 的官方仓库，并按方法×数据集×seed 单元执行。

## 本地控制

```bash
export KAGGLE_API_TOKEN='由 Kaggle 生成的 token'
cd tracks/hod-gnn-replication
./code/kaggle_ctl.sh check
./code/kaggle_ctl.sh set --method gps --dataset molhiv --smoke
./code/kaggle_ctl.sh wait
./code/kaggle_ctl.sh sync
```

`kaggle_ctl.sh` 会在本机没有 CLI 时尝试安装 `kaggle`；token 只从环境变量读取，
不会写入 notebook、metadata、registry 或日志。正式运行时去掉 `--smoke`，并在
下一轮把上轮输出作为 Kaggle Dataset 挂载后使用 `--resume`。

### 混合调度（本地 CPU + Kaggle GPU）

当前工作机没有 NVIDIA GPU，但已经准备好 Python 3.10 的 CPU/GPS 环境。可把
相对较小的 GPS/MOLHIV 单元放在本地运行：

```bash
./code/run_local_profile.sh
```

本地结果写入 `results/local/`，不会覆盖 Kaggle 的 `results/kaggle/`。本地 runner
复用 `code/vendor/GraphGPS` 和 `.venv-gps`，并保留单元级 registry/日志；当前配置下
GPS/MOLHIV 每个 seed 预计需要数小时。ZINC（默认 2000 epoch）、GraphViT 以及
Full/Random/Policy-Learn 依赖 GPU 或 CUDA 扩展，按单个高耗时单元提交 Kaggle，避免
整矩阵一次运行触发会话时限。

## 断点与日志

每个运行单元是一个 `(method, dataset, protocol, seed)`，并以 smoke/max-epochs
等参数生成配置 hash。`--resume` 只跳过同一配置下已经 `success` 或
`unavailable` 的单元；失败、OOM、异常退出和中断会自动重试，避免把失败误当成
完成。一般断点粒度是实验单元，而非 epoch：三个官方仓库没有统一的 checkpoint 接口。
GPS/ZINC 是例外：GraphGPS 官方代码本身提供 checkpoint。正式非-smoke 运行会把
checkpoint 写入 `hod-gnn-results/checkpoints/gps-zinc-seedN/`，并开启
`train.auto_resume`；若会话在 2000 epoch 前结束，下一轮挂载上轮结果后会从最近
的 checkpoint 继续，而不是从 epoch 0 重跑。该目录必须作为 Kaggle Dataset 挂载，
单纯重新 push notebook 不会自动保留 `/kaggle/working`。

输出目录 `hod-gnn-results/` 中：

- `run.log`：人类可读的总日志；
- `events.jsonl`：runner/setup/单元开始、每分钟心跳、退出码和耗时；
- `registry.jsonl`：每个单元的终态、配置 hash、代码 commit、环境、日志 hash；
- `run_state.json`：最后一次计划/当前单元，Kaggle 强制终止后可定位续跑位置；
- `logs/`：每个官方训练进程的完整原始输出；
- `hod-gnn-results.zip`：上述结果的下载包。

ROC-AUC 和 Average Precision 在 registry/CSV 中统一使用 evaluator 的 `0..1`
尺度；论文表格里写成百分数的目标会在汇总前除以 100，避免把 `0.61` 和 `78.8`
直接比较。

如果一轮仍有失败单元，`run_state.json` 的状态会是
`complete_with_failures`（而不是假装全量完成）；下一轮继续使用相同参数加
`--resume` 即可重试剩余单元。

默认只把 epoch、metric、错误和心跳透传到 notebook；需要完整输出时加
`--live-log`。正式实验建议每个 Kaggle 会话只选择一组方法/数据集，保持单元日志
和 GPU 时限清晰可审计。

## 当前可预期的限制

- Kaggle 的 GPU 型号和会话时长是动态的；脚本记录 GPU 型号、CUDA/PyTorch 版本和
  每个单元的耗时。
- GPS 的 MOLTOX21、Policy-Learn 的 Peptides 在对应官方仓库没有可用配置，主协议
  会登记为 `unavailable`，不会用另一个方法冒充结果。
- HOD-GNN 论文未公开精确 seed；`paper-seeds-provisional` 的 `[0,1,2,3]` 仍会
  明确标注 provisional，不能称为严格论文 seed。
