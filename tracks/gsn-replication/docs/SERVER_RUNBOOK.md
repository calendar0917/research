# GSN 服务器操作手册（从放代码到出结果）

> 目标：在 GPU 服务器上用 conda 环境完成严格协议 gsn-strict-social-v1
> （4 数据集 × {GSN-e, GSN-v} × 10 seeds × 10×10 CV）。
> 下面每一步都可在自己的机器上直接执行（方括号处替换成你的服务器地址）。

---

## 0. 跑之前确认三件事

```bash
# ① GPU 与显存（COLLAB 论文在 32GB V100 上跑过；8-16GB 卡也能跑，见 FAQ）
ssh user@your-server
nvidia-smi
nvidia-smi -L          # 看有几张卡：后面 --workers/--devices 用

# ② conda≥23（graph-tool 只能 conda-forge 装）
conda --version

# ③ 磁盘 ≥50GB（processed 计数缓存 + logs；checkpoint 默认已自动清理）
df -h ~
```

## 1. 本地：打包上传（含 vendor！）

vendor（官方仓 + powerful-gnns 数据源）被 git 忽略，**打包时必须带上**：

```bash
# 本地机器
cd ~/code/research/tracks
tar czf gsn-replication.tar.gz gsn-replication        # 约 280MB
scp gsn-replication.tar.gz user@your-server:~/research/   # 或 rsync -avz 整个目录
```

服务器上落位并校验：

```bash
ssh user@your-server
mkdir -p ~/research && cd ~/research
tar xzf gsn-replication.tar.gz
cd gsn-replication
ls code/vendor/graph-substructure-networks/main.py      # 必须存在！
ls code/vendor/powerful-gnns/dataset | head             # 4 个数据集的 raw 来源
```

## 2. 建 conda 环境（约 15 分钟，主要耗在 torch 下载）

```bash
cd ~/research/gsn-replication
bash env/setup_server.sh            # 默认 torch 2.5.1+cu124；旧卡驱动换 cu118/cu121
```

等它打印出 `torch 2.5.1+cu124 cuda: True` 即成功。若 conda/pip 网络慢：
conda 改用清华镜像、pip 的 torch 仍必须用官方 `--index-url`
（`+cu124` 版本只在 pytorch 官方 index 有）。手动分步等价命令见
`env/setup_server.sh` 注释与 `env/gsn-server.yaml`。

## 3. 激活并验收环境

```bash
conda activate gsn
python -c "import torch, graph_tool, torch_geometric; \
print(torch.cuda.get_device_name(0), graph_tool.__version__)"
```

## 4. 数据布局 + 首次计数缓存

```bash
cd ~/research/gsn-replication
python code/setup_data.py            # 校验/补齐 4 数据集到 vendor 官方仓布局
```

## 5. 预热：单进程跑 1 个 seed（1×10 折），**同时做两件事**

① 触发各配置的 graph-tool 计数缓存（首次生成很贵，且后续多进程并行时
必须已存在，否则多进程会重复计算抢写同一缓存文件）
② 测单折时长，外推总预算

```bash
# 每个配置先跑一次（先 IMDBBINARY 最小的，确认没问题再往下）
python code/run_strict.py --config IMDBBINARY--gsn-e --seeds 0 --n-repeats 1
# 完成后看 summary：n_folds_total=10；记录单折耗时（log 里或 elapsed）
```

注意第 5 步**不要加 --workers**（必须单进程）。确认数字合理
（IMDB-B GSN-e 论文 77.8±3.3，严格协议低 2~5 个百分点属正常），再进第 6 步。

## 6. 全量：8 个配置 × 10 seeds × 100 折

推荐 tmux 挂在后台（断线不杀进程）：

```bash
tmux new -s gsn
conda activate gsn
cd ~/research/gsn-replication && mkdir -p logs

# 第一张卡数已知后改 --workers/--devices（例：4 张卡 → 4/4；单卡 → 1/1）
python code/run_strict.py --config IMDBBINARY--gsn-e --workers 4 --devices 4 \
    2>&1 | tee logs/strict-IMDBBINARY--gsn-e.log
```

按此逐个跑（先小后大）：

```bash
# IMDBBINARY--gsn-e  IMDBBINARY--gsn-v  IMDBMULTI--gsn-e  IMDBMULTI--gsn-v
# COLLAB--gsn-e  COLLAB--gsn-v  REDDITBINARY--gsn-e  REDDITBINARY--gsn-v
```

- 断点/中断后继续：跑同样的命令即可（已成功的 fold 自动跳过，`--force` 全重跑）
- `Ctrl-b d` 退出 tmux；回来 `tmux attach -t gsn`

## 7. 看结果

```bash
python code/audit_results.py
# 严格协议表：fold-level（千折 mean±std）+ seed-level
# 明细在 results/strict/<config>/summary.json 与 seed*_r*f*.json
```

## FAQ

| 现象 | 处理 |
|---|---|
| `ImportError: ... libgomp ... GOMP_5.0 not found` | torch 自带旧 libgomp 遮蔽了 graph-tool 需要的：重跑 setup_server.sh（自动写激活钩子）；旧环境则手动执行：`mkdir -p $CONDA_PREFIX/etc/conda/activate.d` 并把 `export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1` 写入其中，再 `conda deactivate && conda activate gsn` |
| `ModuleNotFoundError: graph_tool` | graph-tool 没有 pip 版；`conda install -y -c conda-forge graph-tool`（或重跑 setup_server.sh） |
| `torch.cuda.is_available()==False` | 驱动太旧：换 `bash env/setup_server.sh cu118` 重装 torch；或 `nvidia-smi` 看驱动 |
| 首次跑报 `older version of PyG` | 作者预计算缓存是 PyG1.4 pickle：`python code/setup_data.py --purge-processed`（graph-tool 重算，推荐）或 `--migrate-processed` |
| OOM（COLLAB/REDDIT-B） | 这是超参不可改（batch 32 论文值）。换大显存卡，或单卡串行（--workers 1）逐数据集跑 |
| 只让某几张卡跑 | `CUDA_VISIBLE_DEVICES=2,3,5,7 python code/run_strict.py ... --devices 4` |
| 多进程时 processed 抢写 | 它只发生在"首次缓存未生成"。始终先做第 5 步预热再并行 |
| 磁盘越跑越大 | 每折 checkpoint 默认自动清理；`--keep-checkpoints` 是选项不是默认 |
| 想先小规模验证 | `--seeds 0 --n-repeats 1`（10 折）或 `--n-splits 2`（2 折冒烟） |
