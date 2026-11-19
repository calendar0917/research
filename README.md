# research — 可增长科研工作台

图学习科研工作台：结论必须能回到代码版本、配置、数据身份与运行记录。

长期研究关注**显式、可追溯的图计算**、**结构解释的忠实性**，以及**有竞争力的
预测表现**。研究方向、模型和数据集随任务变化，本仓库**不设全局 `current_model`**，
也不把 LCRC、SSRC、KSVD 或其他任何模型固定为全仓主线。`tracks/` 是可选的
研究组织单位，不代表当前主线。

## 目录与代码组织

| 路径 | 用途 |
|------|------|
| `tracks/<name>/` | 可选研究单元：代码、配置、证据、报告按轨隔离 |
| `tracks/<name>/TRACK.md` | 该轨的可选 charter（问题、范围、入口）；不是进度表 |
| `tracks/ksvd/` | KSVD 轨及其控制平面（见下） |
| `docs/` | 人读导航、文献笔记、录音、交付物 |
| `lib/` | 跨轨复用且无叙事的代码（按需填充） |
| `scripts/` | 仓库脚本（如 `init_track.sh`） |
| `templates/` | 脚手架模板 |
| `data/`、`artifacts/` | 本地数据与临时生成物，默认不提交 |
| `skills/remote-research-runner/` | 远端计算 policy + `rr` CLI |

代码按 `CLI → runner → feature/model/data/evaluation` 的方向依赖。稳定且跨轨复用
的逻辑才进入 `lib/`；轨内实验代码留在该轨 `code/` 或实验目录。

## 环境与基本运行

```bash
# 按锁文件安装环境（Python 版本见 .python-version）
uv sync --frozen

# 环境诊断 / KSVD 轨的科研记录一致性检查
uv run research doctor
uv run research verify

# 快速测试子集（不代表 fresh-clone 安全，见 .github/workflows/）
uv run pytest -q -m "not slow" tracks/ksvd/tests
```

依赖由 `.python-version`、`pyproject.toml`、`uv.lock` 共同锁定；修改依赖后运行
`uv lock`，普通复现不要跳过锁文件。需要数据或 checkpoint 的测试不属于快速子集；
CI 使用显式的 fresh-clone-safe allowlist（`.github/workflows/research-integrity.yml`）。

远端计算统一走 `rr`（见 `skills/remote-research-runner/SKILL.md`），不要手工
`ssh` + `sbatch`/`nohup`。本地是 source of truth，远端只作 compute checkout。

## `research` CLI 的适用范围

`uv run research context | run | promote | compare | verify` 目前是
**`tracks/ksvd` 的控制平面**（默认 `TRACK_ROOT=tracks/ksvd`，可用
`RESEARCH_TRACK_ROOT` 显式指向别的 track）。

- `uv run research context` 只描述 KSVD 轨内的 STATE / study / protocol / claims，
  是 **KSVD 专属工具**，不能用来推断整个仓库的研究方向或“当前主力模型”。
- 只有处理 KSVD 控制平面任务时才使用 `run`/`promote` 等专属流程。
- 其它 track 直接读自己的 `TRACK.md`、`PROTOCOL.md`、`REPORT.md` 和证据目录。

## 证据与可复现

- 正式实验事前写清：问题、对照、数据与划分、评估、停止规则；按需冻结
  `PROTOCOL.md` 或配置，事后不改口径。
- 运行事实必须可追溯：代码版本（commit SHA）、配置、环境、结果、必要的
  数据/协议哈希与身份校验。
- 每个正式实验至少留一份简明结果说明（观察、结论、局限、继续或停止）；
  机器可读记录与报告分开保存。
- 历史实验、报告、checkpoint 与冻结结果**保留、不改写、不移动**；新结论不得
  覆盖旧结论，必要时以纠错/ERRATA 形式追加。
- 大体积数据、`runs/` 等本地现场默认不提交；不可再生的上游源码按 `AGENT.md`
  的规则逐字节 vendor 并记录 sha256。
- **默认直接在 `main` 开发和提交**，不为每个实验开分支。复现依赖 commit SHA、
  配置、数据身份与运行记录，而不是长期存活的分支。

## 新建 Track（可选）

```bash
./scripts/init_track.sh <slug> "<一句话标题>"
```

脚本只创建该轨目录及静态模板入口，不再生成 `docs/tracks/` 镜像页；新的实验或
模型**不必**新建 track。

## 文档入口

| 文件 | 用途 |
|------|------|
| `AGENT.md` | Agent 执行规则（启动安全、上下文定位、记录规范、安全边界） |
| `docs/research_guide.md` | 稳定科研方法与文档定位原则 |
| `docs/index.md` | 导航链接 |
| `tracks/<name>/TRACK.md` | 该轨问题、范围、入口 |
