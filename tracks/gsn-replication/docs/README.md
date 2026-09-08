# GSN 复现操作手册（服务器版）

**两个入口**：`run_strict.py` = 严格协议主表（10 seeds × 10×10 CV，val 选 epoch）；
`run_official.py` = 官方乐观协议旁路参考。以下按主表写，旁路在第 5 节。

## 1. 拷贝到服务器

> ⚠ 完整分步手册（从放代码到出结果）见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。

本地（已含 vendor 官方仓 + powerful-gnns 数据源，**vendor 被 git-ignore**，必须整体拷贝）：

```bash
# 只拷贝本 track 相关（含 vendor）：
rsync -av ~/code/research/tracks/gsn-replication/ user@server:~/research/gsn-replication/
# 或整仓（含其余 track）：
rsync -av ~/code/research/ user@server:~/research/
```

服务器需满足：python 3.11、conda、NVIDIA GPU（paper 里 Collab 用了 32GB V100；现代 16GB+ 即可）。网络可访问
GitHub / download.pytorch.org / conda-forge（若不可直连，用 7897 代理或清华镜像）。

## 2. 环境

```bash
cd ~/research/gsn-replication
bash env/setup_server.sh            # 建 conda 环境 gsn，装 graph-tool + torch2.5.1+cu124 + PyG2.6
conda activate gsn
```

以 `env/gsn-server.yaml` 为准；`setup_server.sh` 是逐条命令的透明版本。

## 3. 数据布局

```bash
cd ~/research/gsn-replication
python code/setup_data.py            # 校验/补齐 4 个数据集到 vendor 官方仓布局
# 首次跑 IMDBBINARY--gsn-e 之前，二选一：
python code/setup_data.py --purge-processed      # A. 删作者缓存，用 graph-tool 重算（推荐，完全官方路径）
# python code/setup_data.py --migrate-processed  # B. 迁移作者缓存（PyG 旧 pickle → 新 Data）
```

## 4. 严格协议主实验（10 seeds × 10×10 CV）

```bash
cd ~/research/gsn-replication

# 0) 先测单折时长（每个数据集跑一个 seed 的 10 折；服务器不需要 --pythonpath）
python code/run_strict.py --config IMDBBINARY--gsn-e --seeds 0 --n-repeats 1
# 单折时长 t → 1000 折 ≈ t×1000；多 GPU 并行：
python code/run_strict.py --config IMDBBINARY--gsn-e --workers 4 --devices 4

# 1) 全量（8 配置 × 10 seeds × 100 折 = 8000 折；逐配置跑）
for cfg in IMDBBINARY--gsn-e IMDBBINARY--gsn-v IMDBMULTI--gsn-e IMDBMULTI--gsn-v \
           COLLAB--gsn-e COLLAB--gsn-v REDDITBINARY--gsn-e REDDITBINARY--gsn-v; do
  python code/run_strict.py --config $cfg --workers 4 --devices 4 2>&1 | tee logs/strict-$cfg.log
done
python code/audit_results.py
```

- 断点续跑：`--resume`（跳过已成功 fold）；`--force` 全重跑；`--purge-processed` 重算计数。
- 每折 checkpoint 默认清理（防数百 GB）；`--keep-checkpoints` 保留。
- REDDIT-BINARY 无官方 10fold_idx：strict 协议对所有数据集统一用随机划分
  （`RepeatedStratifiedKFold(10,10,random_state=seed)`，与 wl-subtree-kernel 轨同源）。
- 期望：严格协议数字会比论文低（val 选 epoch 损失的“选择收益”），差值本身即乐观偏差度量。
- 汇总：`results/strict/<config>/summary.json`（fold-level + seed-level），
  每折明细 `seed*_r*f*.json`。

## 5. 官方乐观协议（旁路参考，可跳过）

```bash
python code/setup_data.py            # 校验/补齐 4 个数据集到 vendor 官方仓布局
# 首次跑 IMDBBINARY--gsn-e 之前，二选一：
python code/setup_data.py --purge-processed      # A. 删作者缓存，用 graph-tool 重算（推荐）
# python code/setup_data.py --migrate-processed  # B. 迁移作者缓存（PyG 旧 pickle → 新 Data）
python code/run_official.py --config IMDBBINARY--gsn-e   # seed 0, folds 0-9，单次 10 折
```

## 6. 汇总与核对

```bash
python code/audit_results.py         # 读 results/*.json，与论文数字并排打印
```

## 7. 本地冒烟（可选，无 GPU/conda 的机器）

```bash
cd ~/code/research/tracks/gsn-replication
uv run python code/setup_data.py --migrate-processed
uv run python code/run_smoke_local.py --mode pipeline --epochs 3 --iters 10
uv run python code/run_smoke_local.py --mode counts   --epochs 2 --iters 8
```

冒烟用 `code/compat/` shim（networkx 版 graph_tool + wandb stub），只验证管线兼容性，
**冒烟后必须删除 shim 生成的 `processed/global/complete_graph_4.pt` 等缓存**，服务器
用真 graph-tool 重算。数值仅作 range sanity。

## 预期数字（论文 Table 1 / README）

| dataset | GSN-e 论文 | GSN-v 论文 | 说明 |
|---|---|---|---|
| IMDB-B | 77.8 ± 3.3 | 76.8 ± 2.0 | 官方 README 命令即此配置 |
| IMDB-M | 54.3 ± 3.3 | 52.6 ± 3.6 | 论文 Table 1 |
| Collab | 85.5 ± 1.2 | 82.7 ± 1.5 | 论文 Table 1 |
| REDDIT-B | —（论文无） | — | 本轨扩展；GIN 论文基线 92.4±2.5 可作参考 |

容差：官方 README 声明跨 torch/CUDA 版本种子级数字会有小差异；
1-2 个百分点内波动视为正常，异常 >3pp 先查：计数缓存来源、fold 划分文件、超参。
