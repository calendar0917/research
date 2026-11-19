# AI / Agent 执行规则

本文件是仓库级 Agent 规则。科研内容与实验细节在各 track 内，不在这里堆积。

## 1. 启动安全与 Git 工作区

- 先运行 `git status`，确认当前分支、HEAD 和未提交文件。
- 保留用户已有的修改和未跟踪文件。目标文件已有用户修改时，不覆盖；记录冲突并
  停止该文件的修改。
- 默认直接在 `main` 开发和提交，不为每个实验新建分支。复现依赖 commit SHA、
  配置、数据身份和运行记录。
- 只有确实需要隔离风险（并行冲突、可能失败的迁移等）时才用分支；用完后按下面
  的安全条件清理。
- 禁止 `git reset --hard`、`git clean -fd`、`git push --force`；不改写已有 Git 历史。
- 分支清理只允许在全部条件可验证时进行：
  1. `git fetch origin --prune`，确认最新 `origin/main`；
  2. 枚举本地和远端的非 `main` 分支；
  3. 只清理 tip 已被 `main` 完整包含（无独有提交）的分支；
  4. 远端分支还须确认不是默认分支、不是受保护分支、没有未关闭 PR；
  5. 使用普通删除（`git branch -d` / `git push origin --delete`），不得 force；
  6. 条件无法验证或存在独有提交时保留，并记录原因。
- 保留所有 tags；不删除提交、实验目录或科研文件。

## 2. 上下文定位（按任务，而不是按全局状态）

- 以用户当前明确表达的任务和研究对象为准。
- 用 `rg --files`、`rg`、相关路径的 `git log -- <path>` 发现代码和证据。
- 优先读取与任务相关的 `DECISION.md`、`REPORT.md`、`PROTOCOL.md`，随后读取必要代码。
- 只加载少量相关文件；不默认阅读整个文档树、大型 `TRACK.md` 或 `STATE.yaml`。
- 不根据目录名称、`active` 标签、最近提交或旧 guide 单独推断“当前主力模型”或
  “全仓研究方向”。仓库不设全局 `current_model`。
- 不默认运行 KSVD 的 `uv run research context`；它是 `tracks/ksvd` 的专属工具。
  只有处理 KSVD 控制平面任务时才使用 `research context/run/promote` 等流程。
- 若用户未指定具体研究对象、证据又不足以确认焦点：给出相关候选和依据，不要伪造
  确定性。
- 遵守研究者的明确授权：不自主启动新训练、新实验，或扩大对 test 数据的访问。

## 3. 代码与实验记录规范

- Track 是可选的组织单位；新实验或模型不必自动新建 track。
- 新实验默认最少留下三样东西：
  1. **事前可核验的实验定义**：问题、对照、数据与划分、评估、停止规则；正式实验
     按需求冻结协议或配置。
  2. **可追溯的运行事实**：代码版本、配置、环境、结果和必要的身份校验/哈希。
  3. **一份简明结果说明**：观察、结论、局限、继续或停止的决定。
- 正式实验可以用 `PROTOCOL.md` + `REPORT.md`，机器记录单独保存。不要为满足模板
  默认同时创建 `PLAN.md`、`STUDY.md`、`LEDGER.md`、`DECISION.md` 和额外状态摘要。
- `TRACK.md` 是 charter（问题/范围/入口），不是实验流水账；新结果写入实验目录的
  `REPORT.md` / 决策记录，不追加到 `TRACK.md`。
- 控制平面（如 KSVD `tracks/ksvd/`）需要的文件结构保持现有格式与兼容性。
- 数据集、split、seed、特征与评估配置必须可 diff；不要写机器相关绝对路径，使用
  环境变量（如 `PROJECT_DATA_DIR`）。
- 共享代码只放无叙事的通用逻辑；模型与实验代码留在对应 track/实验目录。

## 4. 必要科研安全边界

- 字典、标准化参数、特征选择和模型选择只能使用训练部分；验证/测试只做编码或评估。
- 不混用不同协议、数据集或阶段的结果；比较只在同一 metric semantics 与协议身份
  下进行（不一致即 INCOMPARABLE）。
- 结果必须记录配置、代码版本、环境版本和输出路径；未经实际运行和核对，不得声称
  结果。
- official/test 数据只在研究者授权时访问，并记录访问方式与一次性/终局规则。
- 历史结果、冻结 checkpoint、纠错记录只读保留；需要修正时追加 ERRATA，不改写。
- 不为了“整洁”删除历史实验；先标记 legacy/archived 并保留摘要和校验信息。
- 不可再生的上游源码 vendor 进对应 track，逐字节保存并写 sha256 溯源，不就地修改
  被引用的原件。

## 5. 环境、测试与远端计算

- 活跃代码统一 `uv run ...`；依赖只改根 `pyproject.toml` 并提交 `uv.lock`。
  Python 版本由 `.python-version` 约束。
- 快速本地子集：`uv run pytest -q -m "not slow" tracks/ksvd/tests`。`slow` 只标记
  已知耗时用例，不代表其余用例 fresh-clone 安全（许多仍需要 `data/`、`results/`
  或本地 checkpoint）。
- CI 走显式 fresh-clone-safe allowlist（`.github/workflows/research-integrity.yml`，
  CPU-only、最小环境）；未按同样方式验证前，不要往 allowlist 加文件。
- `uv run research verify` 检查 KSVD 轨的 STATE 与 durable records 一致性。
- 远端执行一律走 `rr`（`rr doctor/deploy/run/jobs/status/logs/cancel/pull`），
  先读 `skills/remote-research-runner/SKILL.md`。主机：`res`（单机）与 `res-2`
  （Slurm，离线 uv）。远端只 fetch 不 push；不要在远端手工改科研文件。

## 6. 任务完成后的汇报

说明：改了什么文件、做了哪些验证、哪些受环境/数据/GPU 限制而未执行、下一步入口。
对不确定的录音转写、数据集名称和论文数字标注“待核对”。不要用新规划文档代替
实际修改。
