> **Status note (2026-10-02):** this is the original raw survey and is kept as
> historical evidence. Some conclusions have been superseded by runtime probes:
> `res` can reach GitHub again, and the live source of truth for node drivers /
> pools is now `rr doctor res-2 --refresh` (which probes each node via a tiny
> Slurm job rather than trusting static `gres.conf`). Use this file for the
> network/storage/software inventory, not as the current driver map.

# 服务器档案与迁移可行性调查

调查日期：2026-09-29
调查方式：全部结论来自实测（命令输出），未实测项显式标注。

---

## 1. res —— 当前默认执行主机

| 项 | 值 |
|---|---|
| SSH | `ssh res` → ProxyJump `pve-old` → `121.48.165.123:6003` |
| 登录用户 | `hxy` |
| hostname | `a100-2` |
| OS | Ubuntu 22.04.2 LTS (jammy) |
| 内核 | 6.8.0-138-generic |
| glibc | **2.35** |
| CPU | 2 × Xeon Gold 5320 @2.20GHz，104 逻辑核 |
| 内存 | 503 GiB |
| GPU | **2 × A100-SXM4-40GB**（同机） |
| 驱动 | **550.163.01**（CUDA 12.4） |
| 系统盘 | `/dev/sda2` 187G，用 31% |
| 数据盘 | `/dev/sdb1` 44T → `/home`，**已用 99%（仅剩 491G）** |
| 外网 | **有**（pypi.org 200、download.pytorch.org 200、清华源可达） |
| GitHub | **不通**（`https://github.com` 超时） |
| 仓库 | `/home/hxy/cy/research` |
| 仓库体积 | 32 GB（其中 `.venv` 5.9 G、`data` 2.1 G） |
| uv | `~/.local/bin/uv` 0.12.13（二进制 48 MB） |
| uv Python | `~/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu`（110 MB） |
| uv cache | `~/.cache/uv` **5.8 GB** |
| 当前占用 | GPU0 37263/40960 MiB @99%；GPU1 35125/40960 MiB @97%（zc 与 wujiayu 各占一张） |

### res 上的实际环境（实测 import 全部成功）

```
python  3.12.14
torch   2.5.1+cu124      torch.version.cuda = 12.4    cudnn 90100
torch_geometric 2.6.1    ogb 1.3.6        networkx 3.4.2
numpy 2.1.3              scipy 1.14.1     sklearn 1.5.2
optuna 4.2.1             pandas 2.2.3     xgboost 2.1.3
pynauty 2.8.8.1          pyarrow 25.0.1   openpyxl 3.1.5
matplotlib 3.11.1        pytest 8.3.4     ruff 0.8.6
grakel   —— 未安装（属可选 wl-kernel group，default-groups 不含它）
```

---

## 2. res-2 —— 目标主机（uestc_hpc 集群）

### 2.1 拓扑

```
本地 ──ssh res-2──> mgt01 (登录/管理节点, 无 GPU)
                       │  Slurm 21.08.6  (slurmctld + slurmdbd + slurmd)
                       ├─> c01 … c08   计算节点 (A100 40GB × 19)
                       └─> io01 / io02 存储节点

/share  73TB XFS，挂在 mgt01 本地 (/dev/sdb)，由 mgt01 做 NFS 服务端
/data1  io01:/data1  NFS 73TB，用 1%
/data2  io02:/data2  NFS 73TB，用 1%
```

| 项 | 值 |
|---|---|
| SSH | `ssh res-2` → ProxyJump `pve-old` → `10.20.26.206:10022` |
| 登录用户 | `snsun` |
| 登录节点 | `mgt01` |
| OS | Ubuntu 20.04.4 LTS (focal) |
| 内核 | 5.13.0-30-generic |
| glibc | **2.31** |
| CPU | 2 × Xeon Gold 5320 @2.20GHz，104 逻辑核 |
| 内存 | 251 GiB，Swap 2 GiB（**已 100% 用满，长期现象**） |
| 家目录 | `/share/home/snsun`（已用 70 G） |
| `/share` 容量 | 73 TB，**已用 83%，剩 13 TB** |
| 外网 | **完全没有**（无 DNS 解析，无任何出站） |
| 内部源 | `http://mgt01:9999`（仅 apt .deb 镜像，无 PyPI） |
| uv | **未安装** |
| Slurm 分区 | `gpu` → c01-c08（默认）；`normal` → mgt01 |
| 配额 | `snsun`：`GrpTRES=cpu=16,gres/gpu=2`（**最多 16 核 + 2 张卡**） |

### 2.2 计算节点（关键：**异构**）

| 节点 | GPU 型号 | 驱动 | 驱动支持的最高 CUDA | 状态 |
|---|---|---|---|---|
| c01 | **A100-SXM4-40GB** | 510.47.03 | 11.6 | MIXED |
| c02 | (未实探) | 510.47.03 | 11.6 | MIXED |
| c03 | (未实探) | 510.108.03 | 11.6 | MIXED |
| c04 | (未实探) | 510.108.03 | 11.6 | MIXED |
| c05 | (未实探) | **525.85.12** | **12.0** | MIXED |
| c06 | **A100-PCIE-40GB** | **525.85.12** | **12.0** | MIXED |
| c07 | A100-PCIE-40GB | 510.108.03 | 11.6 | MIXED |
| c08 | A100-PCIE-40GB | 510.108.03 | 11.6 | MIXED |

- 卡型与驱动均不统一；`gres.conf` 里的 `#Type=a10 / #Type=a40` 注释**已过时**，实测 c01 是 SXM4、c06/07/08 是 PCIE。
- c01 有 `/usr/local/cuda` → `cuda-11.6`；`/share/apps/cuda/cuda-11.2` 另有 nvcc。
- 每个节点内存配置 240 GB，但实测多个节点只剩 2-6 GB 空闲（别人的作业吃满）。

### 2.3 软件栈

- Slurm 21.08.6，非交互 SSH **不会自动加载** Slurm PATH，需要 `source /share/config/zz-hpc-env.sh`
- 共享 Conda：`/share/apps/anaconda3`（Python 3.9.7）；envs：`pytorch-1.11`（torch 1.11+cu113）、`tensorflow-2.4.0`
- 用户自己的 Conda：`/share/home/snsun/enter`（6 GB 包缓存）；`.virtualenvs/mae_main` 是**空 venv**
- 已有的离线包仓库：`/share/home/snsun/Packet/`（含 torch-1.12.1+cu113、timm、torchvision 等 cp39 wheel）
- Singularity、Docker、gcc/g++/gfortran、cmake、make、git 2.25.1、tmux 可用
- 无 `uv`、无 `pipx`、无 `conda-pack`

### 2.4 计算节点访问规则

```
$ ssh c01
Access denied by pam_slurm_adopt: you have no active jobs on this node
```

→ **必须先有 Slurm 分配**才能上计算节点。`srun` 交互 / `sbatch` 批处理。

### 2.5 存储性能（实测）

| 场景 | 结果 |
|---|---|
| mgt01 本地写 3000 小文件 | 0.64 ms/文件 |
| mgt01 本地读 3000 小文件 | 1.62 ms/文件 |
| 计算节点 c01 经 NFS 写 3000 小文件 | 1.80 ms/文件 |
| 计算节点 c01 经 NFS 读 3000 小文件 | 0.60 ms/文件 |
| 计算节点 c01 顺序写 | 219 MB/s |
| mgt01 顺序写 | 2.7 GB/s |

推论：5.8 GB / 25278 文件的 venv 放 NFS 可行，单节点冷启动 import 约 20-40 s，热启后走页缓存正常。

---

## 3. 兼容性探测结论

### 3.1 glibc —— ✅ 通过

| 二进制 | 需要的最高 GLIBC 符号 | res-2 提供 | 结论 |
|---|---|---|---|
| `uv` 0.12.13 | GLIBC_2.17 | 2.31 | OK |
| uv Python 3.12.14 | GLIBC_2.17 | 2.31 | OK |
| `.venv` 内全部 `.so` | 最高 GLIBC_2.27（PIL / xgboost） | 2.31 | OK |

**没有 glibc 前向兼容问题。**（虽然 res 是 2.35，但 wheel 都是 manylinux_2_17/2_27 量级。）

### 3.2 离线重建 —— ✅ 通过（这是整个方案的关键）

在 res 上把仓库 tracked 文件导出到 `/home/hxy/offline_probe`，然后：

```
$ uv sync --frozen --offline --python 3.12
… exit=0
.venv/bin/python -c "import torch" → torch 2.5.1+cu124
```

- 结果：venv 5.8 GB，25278 文件，全部必需包 import 成功
- **含义**：uv cache 自足；不需要 wheelhouse，不需要网络
- **搬运清单 = uv 二进制 + uv Python + uv cache + 仓库源码**

### 3.3 CUDA 驱动 —— ❌ **唯一致命阻断**

- `torch 2.5.1+cu124` 内含 CUDA 12.4 runtime，要求驱动 **≥ 525.60.13**（CUDA 12.x 最低驱动）
- res-2：**只有 c05 / c06 是 525.85.12**；c01/c02/c03/c04/c07/c08 全是 510.x（最高 CUDA 11.6）
- 在 510 的节点上，cu124 的 torch 会在初始化时报
  `CUDA driver version is insufficient for CUDA runtime version`，`torch.cuda.is_available()` 为 False

**这是必须显式决策的点，无法绕过。**

### 3.4 网络互通 —— ✅ 有惊喜

| 路径 | 结果 |
|---|---|
| res → res-2:10022 | **通** |
| res-2 → res:6003 | **通** |
| res → 外网 | 通（pypi / download.pytorch.org） |
| res-2 → 外网 | 不通 |
| res → github.com | 不通 |

→ **res 可以作为 res-2 的"下载代理 + 直传通道"**，无需经本地机器中转 8 GB 数据。
→ 但 GitHub 从 res 也不通，所以 `deploy.sh` 的 git 推送路径在两边都不可用。

### 3.5 数据 —— 需一次性同步 ~2.1 GB

`data/ZINC` 383 M、`data/TUD` 1.4 G、`data/ogb` 326 M。仓库用 `resolve_path("data/ZINC")` 解析为**仓库内相对路径**，所以必须落在 `<repo>/data/` 下。

---

## 4. 当前 skill 的假设 vs res-2 的现实

| # | skill 的隐含假设 | res-2 现实 | 破坏级别 |
|---|---|---|---|
| 1 | 远程有外网，`uv sync --frozen` 直接可用 | 完全无外网无 DNS | **阻断** |
| 2 | 同一台机器上的 2 张卡，用 `CUDA_VISIBLE_DEVICES=0/1` | Slurm 分配，卡号由调度器给，且驱动异构 | **阻断** |
| 3 | 进程可直接在 SSH 会话里跑 | 计算节点 `pam_slurm_adopt` 拒绝直连 | **阻断** |
| 4 | 长任务用 `setsid+nohup` 保活 | 作业必须由 Slurm 托管；退出分配即被杀 | **需重写** |
| 5 | `git push` + 远程 `git fetch` 部署 | 两边都连不上 GitHub | **需替换** |
| 6 | `uv` 已安装 | 未安装 | 需引导 |
| 7 | 资源随时可用 | 排队 + 配额 2 GPU / 16 CPU | 需排队感知 |
| 8 | venv 在本地盘 | venv 在 NFS（25278 文件） | 性能可接受，需实测确认 |
| 9 | `.exit` 文件是完成信号 | 应改用 `sacct` 的 ExitCode | 需适配 |
| 10 | GPU 是同一型号/驱动 | SXM4 vs PCIE，510 vs 525 | **provenance 必须记录** |

---

## 5. 迁移方案（分阶段，先不执行）

### 阶段 0：离线引导（一次性，~10 分钟传输）

```
res 侧打包:
  ~/.local/bin/uv                                 48 M
  ~/.local/share/uv/python/cpython-3.12.14-...    110 M
  ~/.cache/uv                                     5.8 G
        ↓  (直连 res → res-2，或经本地中转)
res-2 侧落位:
  /share/home/snsun/opt/uv/bin/uv
  /share/home/snsun/opt/uv/python/cpython-3.12.14-...
  /share/home/snsun/opt/uv/cache/
     并在 .bashrc 或 wrapper 里 export
       PATH=$HOME/opt/uv/bin:$PATH
       UV_PYTHON_INSTALL_DIR=$HOME/opt/uv/python
       UV_CACHE_DIR=$HOME/opt/uv/cache
```

风险：几乎为零；不动现有任何东西。可先做这一步验证离线重建在 res-2 上成立。

### 阶段 1：冻结环境 + 固定到 c05/c06（零科学风险）

- `uv sync --frozen --offline` 在 res-2 上重建**与 res 完全一致**的 venv
- 所有正式运行加 `--nodelist=c05,c06`（或 Slurm constraint），确保落在驱动 525 的节点
- 好处：环境 bit-identical 于 res，`uv.lock` 不动，科学上零风险
- 代价：只有 c05/c06 的 8 张卡可用，且这两台是共享热点；你自己配额 2 张

**建议先走这条路把链路打通。**

### 阶段 2：cu118 第二执行域（覆盖其余 6 台）

- 构建 `torch==2.5.1+cu118`（需驱动 ≥ 450，510 可满足）
- res 已确认可达 `https://download.pytorch.org/whl/cu118`，且
  `torch-2.5.1+cu118-cp312-cp312-linux_x86_64.whl` 存在
- 本质是**新增一个执行域（execution regime）**，不是"同一个环境的拷贝"
- 按本仓库既有纪律，必须：
  - 作为独立的 **infrastructure commit**（与科学 commit 分离）
  - 不能与 res 的 cu124 结果直接比小数值；需要该域自己的 baseline
  - 在 manifest / meta 里记录 `hostname + driver_version + torch cu` 三元组
- 备选（最优但不可控）：让管理员把 510 的 6 台升到 ≥525。
  `/share/install-2023/` 里管理员已备好 `NVIDIA-Linux-x86_64-525.85.12.run`。

### 阶段 3：数据同步 + 冒烟

- `data/{ZINC,TUD,ogb}` 2.1 GB 一次性 rsync
- `uv run research doctor` 必须在 res-2 上通过（`data.zinc` 是 blocking check）
- GPU 冒烟：`sbatch` 一个 1 卡 240 秒的 torch CUDA 自检，落在 c05 或 c06

---

## 6. skill 优化方案

### 6.1 核心改动：引入"执行域 (execution regime)"抽象

新增 `references/execution-regimes.md`，把三件事绑定成一个域：

```
regime id              host    scheduler  driver      torch cu   baseline 可复用?
res-cu124              res     none       550         cu124      是
res2-cu124-525         res-2   slurm      525 (c05/6) cu124      需重建
res2-cu118-510         res-2   slurm      510         cu118      需重建
```

`status.sh` / `run_remote.sh` 的输出必须打印当前 regime id。

### 6.2 分层：`lib.sh` → 增加 host profile

```
RESEARCH_HOST_PROFILE=res        → scheduler=none,  repo=/home/hxy/cy/research
RESEARCH_HOST_PROFILE=res-2      → scheduler=slurm, repo=/share/home/snsun/.../research
```

由 `SSH_ALIAS` 自动推断，允许显式覆盖。profile 里同时定义：

- `REMOTE_REPO`
- `SCHEDULER`（none | slurm）
- `SLURM_PARTITION` / `SLURM_NODELIST`（res-2 阶段 1 = `c05,c06`）
- `MAX_CONCURRENT_GPUS`（res=2，res-2=2 by quota）
- `REMOTE_ENV` 前缀（res-2 要带 `UV_*` 三个变量）

### 6.3 执行器改造

| 脚本 | res（不变） | res-2（新增分支） |
|---|---|---|
| `run_remote.sh` | `ssh … uv run …` | `ssh … sbatch --wait …` 或 `srun` |
| `launch_remote.sh` | `setsid+nohup`，写 `.pid/.exit/.log/.meta` | 生成 `.slurm` 脚本 → `sbatch --parsable` → 把 jobid 写进 `.pid`；`--output` 指向 run_dir 的 `<tag>.log` |
| `wait_remote.sh` | 轮询 `.exit` | 轮询 `sacct -j <jobid> --format=State,ExitCode`；同时区分 `PENDING(running/reason)` 与终态 |
| `tail_remote.sh` | 读 log + `kill -0 pid` | 读 log + `squeue -j <jobid>` |
| `status.sh` | 列 run dir | 额外列出 `squeue -u $USER` 的 PENDING/RUNNING 与排队原因 |
| `preflight.sh` | SSH/uv/A100/torch/doctor | 增补：Slurm 可达、配额、**逐节点驱动版本**、`UV_CACHE_DIR` 是否存在且非空、data.zinc |
| `deploy.sh` | `git push` + 远程 `fetch` | **git bundle 通路**：本地 `git bundle create` → rsync → 远程 `git fetch <bundle>` → `git merge --ff-only` → `uv sync --frozen --offline` |
| `sync_path.sh` | rsync | 不变；新增可选的 `--via res` 直传模式 |
| `pull_results.sh` | rsync | 不变 |

新增 `scripts/bootstrap_offline.sh`：幂等的一键离线引导（已在 res-2 落位则跳过），并做完整性校验（`uv --version`、`uv python list`、`uv sync --frozen --offline --dry-run`）。

### 6.4 新增硬性守卫

1. **并发护栏**：res-2 配额 2 GPU。`launch_remote.sh` 在提交前查 `squeue -u $USER` 的 GPU 占用；第 3 个直接报错并说明配额，而不是静默排队。
2. **节点/驱动护栏**：cu124 regime 下，job 若落到 510 节点会静默失败。在 `.slurm` 里加 `#SBATCH --nodelist=...`，并在作业开头断言驱动版本，不匹配则立即以非零退出并打印实际 driver。
3. **provenance 护栏**：`.meta` 必须记录 `hostname`、`driver_version`、`torch.version.cuda`、Slurm jobid、`SLURM_JOB_PARTITION`。异构集群里不记录这几项的结果不可比。
4. **离线护栏**：`deploy.sh` 在 res-2 上若发现 cache 为空/缺失，直接失败并指向 `bootstrap_offline.sh`，绝不退化成联网尝试。
5. **官方测试纪律不变**：`test_policy=terminal`、官方 ZINC/MolHIV 测试已用尽，迁移不改变这条。

### 6.5 SKILL.md 需要更新的段落

- "Defaults" 段落：增加 `res-2` profile 与 regime 概念
- "Long training runs"：res-2 上不再是 `setsid`，而是 **`sbatch` 天生 detach**；`launch/wait/tail` 语义改为 jobid 语义
- "Deploy committed code"：增加离线 bundle 通路
- "GPU baseline rule"：明确 **res 的 baseline 不能搬到 res-2**，即使 torch/cu 相同，驱动 550→525 也是新域
- "Failure handling"：新增
  - `squeue` 显示 `AssocGrpGRES` → 配额满，不是错误，等待或减少并发
  - 作业落到错误驱动节点 → 检查 `--nodelist` 与驱动断言
  - `uv sync --frozen` 失败 → **不要**尝试联网，检查 cache

### 6.6 不做的事

- 不把 cu124 与 cu118 的结果混在一张表里比较
- 不改 `uv.lock` 来完成迁移（阶段 1 完全不动 lock；阶段 2 的 cu118 走独立 regime 文档 + 独立 lock/环境）
- 不把 `/home/hxy` 硬编码路径的 venv 直接 `cp` 过去（25278 文件中 27 个脚本有绝对路径；走离线 `uv sync` 重建更干净）
- 不在登录节点 mgt01 上跑任何训练
