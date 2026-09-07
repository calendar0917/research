# AI 使用规范

本文件是本仓库中 AI/Agent 的工作规则。仓库目标是维护可复现、可审计的 GNN/KSVD 科研过程。

## 开始工作（AI 快捷流程）

1. 先运行 `git status --short`；保留用户已有修改，不执行覆盖性整理。
2. 运行 `uv run research context`（约 100–200 行）。
   只读取 context 指向的 `STATE.yaml`、当前 `Study`、`Protocol` 与相关
   `Claim`/`Decision`；不要默认读取全部 luyin16 result Markdown。
3. 需要环境诊断或数据核对时运行 `uv run research doctor`。
4. 明确任务属于哪个 track、哪个 stage 和哪个 `protocol_id`，再改代码或运行实验。

### 运行控制平面（tracks/ksvd）

- 运行 scratch/screen/confirm/terminal 实验统一走
  `uv run research run <runner> [--study ...] [--purpose "..."] [--set k=v] [--mode ...]`；
  每个 run 自动落盘 `runs/YYYY/MM/DD/<run_id>/`（git-ignored，含
  manifest/config.resolved/metrics/stdout/stderr/artifacts）。
- 只有 `uv run research promote <run_id>` 才进入 `records/runs/`（Git 追踪、
  不可变）。scratch 实验不会产生 Git 文件。
- 协议声明 `test_policy: terminal` 时，非 terminal run 禁止访问 official test
  （控制面强制，manifest 记录 `test_access`）。
- 比较只看相同 `protocol_id` + dataset/split fingerprint：不一致默认
  `INCOMPARABLE`，显式 `--override` 才会有 rank 并带 warning。
- 重复 run（同 runner/config/protocol/seed/code state）默认拒绝，需 `--force`。

## 目录边界

- `tracks/<track>/code/`：代码；稳定且跨轨复用的逻辑才进入 `lib/`。
- `tracks/<track>/configs/`：数据集、划分、种子和实验配置，必须可 diff。
- `tracks/<track>/notes/`：定义、决策和实验意图，不堆大段结果表。
- `tracks/<track>/results/`：该 track 的数字唯一来源；原始大文件和缓存不入 Git。
- `docs/luyin/`：录音原文档案，不改写、不删除；整理结论写入 track notes 或 results。
- `data/`、`artifacts/`：本地数据和生成物，默认不提交。

## 实验纪律

- 每次运行通过 `research run` 走控制平面；新实验入口以 runner 形式注册，
  遵守 CLI → runner → feature/model/data/evaluation 的依赖方向。
- 每个实验先写清问题、数据集、split、seed、指标、输入特征和停止条件。
- 字典、标准化参数、特征选择和模型选择只能使用训练部分；验证/测试只做编码或评估。
- 不把不同协议、不同数据集或不同阶段的结果混入同一主表。
- 结果必须记录配置、代码版本、环境版本和输出路径；未经运行和核对不得声称结果。
- luyin16 的实验只进入 `tracks/ksvd/{notes,configs/luyin16,results/luyin16}`，先做结构独立性审计，再做融合。

## 环境

- 活跃 KSVD 主线统一使用 `uv run ...`；依赖只改根目录 `pyproject.toml`，并提交对应 `uv.lock`。
- Python 版本由 `.python-version` 约束；不要在 Python 3.14 上假定 PyTorch/PyG 可用。
- HOD-GNN 的官方旧环境是独立复现环境，不与 KSVD 锁文件强行合并。
- 数据根目录使用环境变量（如 `PROJECT_DATA_DIR`），不得写机器相关绝对路径。
- 数据集下载使用数据集官网或维护方提供的官方 URL；ZINC 只允许 PyG 官方
  dataset/split URL，不借助清华 PyPI 镜像，并记录 split 大小与 raw SHA-256。
- KSVD 自测脚本使用 `uv run python -m tracks.ksvd.code.test_<name>`；它们不是 pytest 风格测试函数。

## 修改与安全

- 优先做小而可回滚的提交；移动文件前先建立索引并确认引用关系。
- 不运行 `git reset --hard`、`git clean`、递归删除或覆盖用户文件。
- 不为了“整洁”删除历史实验；先标记 `legacy`/`archived` 并保留摘要和校验信息。
- `TRACK.md` 保持为 Track Charter；长实验流水不再追加到 TRACK，写入
  `records/`（claim/decision）与 `runs/`/`reports/`。
- 新增依赖、数据集或实验入口时，同时更新相应 README、配置和协议说明。

## 交付格式

完成任务时说明：改了什么、验证了什么、哪些仍受环境/数据/GPU限制，以及下一步入口。对不确定的录音转写、数据集名称和论文数字必须标注“待核对”。

尽可能用中文阐释方案、架构、流程，讲解清晰。
