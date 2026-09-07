# research — 可增长科研工作台

上层研究框架。旧仓 `../paper` **只读归档**，本目录从零按 track 增长。

## 怎么用

```bash
# 新建课题（track）
./scripts/init_track.sh ksvd "KSVD 结构字典学习"

# 按 uv.lock 安装活跃 KSVD 环境（清华 PyPI 镜像）
uv sync --frozen

# AI 启动默认流程（不要先读全部 TRACK / guide / 结果历史）
git status --short
uv run research context              # 当前研究位置：STATE/Study/Protocol/claims
uv run research doctor               # 环境诊断（可选）

# ksvd 控制平面（tracks/ksvd）
uv run research run zinc_patch_path_pooling --study zinc-context-gap --purpose "..."
uv run research show <run_id>
uv run research compare --study zinc-context-gap
uv run research promote <run_id>     # 晋级为 Git-tracked durable record
uv run research runs --source all    # local + promoted

# 人读入口
cat AGENT.md                        # AI/Agent 使用规范
cat docs/research_guide.md          # 全局规矩 + 当前进度
cat tracks/<name>/TRACK.md          # 该课题 charter
```

真理层级（facts source）：`runs/` = 本地执行真相（Git-ignored）；
`records/runs/` = durable promoted facts（Git-tracked，fresh clone 可读）；
`records/{claims,decisions}` = 判断/决策；`STATE.yaml` = 导航指针；
`results/luyin16/` = legacy evidence archive（只读保留，新实验不再写入）。

## 目录

| 路径 | 用途 |
|------|------|
| `tracks/` | **增长单元**：每个课题独立 |
| `docs/` | 人读导航 + 全局文献深读 + 交付物 |
| `lib/` | 跨轨可复用代码（暂空，需要再填） |
| `scripts/init_track.sh` | 从模板初始化新课题 |
| `templates/track/` | 课题脚手架 |
| `data/` | 大数据集（建议 gitignore） |
| `artifacts/` | 临时导出 |
| `AGENT.md` | AI/Agent 使用规范 |

活跃 KSVD 代码统一通过 `uv run python ...` 执行。HOD-GNN 复现保留其独立的官方环境说明。

环境由 `.python-version`、`pyproject.toml` 和 `uv.lock` 共同锁定。修改依赖后运行 `uv lock`，普通复现不要跳过锁文件。

## 已有 track

| track | 状态 | 说明 |
|-------|------|------|
| [gnn-gsn](tracks/gnn-gsn/TRACK.md) | archived | Kernel–GNN–GSN；实现在 `../paper` |
| [ksvd](tracks/ksvd/TRACK.md) | **active** | KSVD 结构字典学习 |
| [hod-gnn-replication](tracks/hod-gnn-replication/TRACK.md) | **active** | HOD-GNN 论文基线复现审计；独立官方环境 |
| 录音 | — | [docs/luyin/](docs/luyin/) |

## 原则（短）

1. **Track = 可开可关的研究单元**（问题、协议、结果、状态）
2. **共享只放无叙事物**（数据加载、协议原语、schema）
3. **结果不跨轨混表**；跨轨对比单独写 deliverable
4. **一事一处**；进度只改 guide §进度 与各 `TRACK.md`
5. **控制面不自信地给错答案**：metric 不串、protocol 内容（hash）不串、
   代码版本（code_state_hash）不串、run 不丢、promoted evidence 换机器可读、
   test 权限由控制面统一裁决。
