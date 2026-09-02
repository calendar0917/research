# decisions.md — 实验决策记录

## 2026-08-24

### 决策 1: seed 协议命名
- 由于 HOD-GNN 论文未披露具体 seed 数值，使用 `[0,1,2,3]` 作为临时替代
- 协议名称必须为 `paper-seeds-provisional`，不能冒充严格论文复现
- 来源: IMPLEMENTATION_PLAN.md §5.1

### 决策 2: 官方代码优先
- 优先使用各方法的官方代码仓库，不重写
- 仅在 API 兼容性或数据 URL 失效时做最小 patch
- 来源: IMPLEMENTATION_PLAN.md §4

### 决策 3: 实验范围
- 只复现论文实际报告过的表格单元
- 论文中的 "–" 不补跑
- Full/Random 不在 Peptides 上运行（论文中无报告）
- 来源: IMPLEMENTATION_PLAN.md §2

### 决策 4: 缺少官方配置的 fallback
- GPS 的 MOLTOX21 和 Policy-Learn 的 Peptides 缺少官方配置
- 优先从 HyMN 代码库 (Southern et al. 2025) 获取配置
- 如果仍不可用，标记为 unavailable
- 来源: SOURCE_LEDGER.md

### 决策 5: CPU 环境可行性（2026-08-24）
- 官方仓库 (GraphGPS) 可以在 CPU 上运行：py3.10 + torch 1.13 CPU + pyg 2.2 + pyg 编译扩展 (CPU wheels from data.pyg.org/whl/torch-1.13.0+cpu)
- 每 epoch MOLHIV (41127 graphs, batch 32) ≈ 230s CPU（vs GPU ~几秒）
- 100 epochs ≈ 6.4 小时/seed；4 seeds ≈ 25.6 小时（串行）
- 用户选择：先完整跑 GPS + MOLHIV (100 epochs × 4 seeds)
- 依赖坑：uv 安装时 torch 会被 PL/performer 依赖顶到 2.13，必须最后 force-reinstall torch 1.13 CPU；numpy 需 <2 (1.24.4)；setuptools<80 才有 pkg_resources

### 决策 6: 队列自动收集
- `code/run_gps_molhiv_seeds.sh`：等待 seed 0（外部启动）→ 收集 → 顺序跑 seed 1-3 → 每个完成自动写 registry.jsonl
- 收集脚本 `code/collect_gps_results.py` 解析日志中每个 epoch 的 train/val/test dict，按验证集最优选 epoch