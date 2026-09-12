"""
kaggle_run.py — HOD-GNN 基线复现审计 · 单脚本执行器（Kaggle/任意 Linux）

自包含：环境安装 → 克隆官方仓库（固定 commit）→ 按矩阵跑实验 →
解析日志 → 写 registry.jsonl → 生成汇总 CSV。

用法（Kaggle notebook 内）:
    !python kaggle_run.py --plan                 # 先看计划（不执行）
    !python kaggle_run.py --method gps --dataset molhiv --smoke   # 冒烟
    !python kaggle_run.py --method gps --dataset molhiv           # 正式
    !python kaggle_run.py --method all --dataset all              # 全矩阵
    !python kaggle_run.py --method gps --dataset zinc --max-epochs 50  # resource-adjusted

参数:
    --method       gps|graphvit|full|random|policy_learn|all
    --dataset      zinc|molhiv|moltox21|molbace|peptides-func|peptides-struct|all
    --protocol     paper-seeds-provisional|official-seeds (default paper-seeds-provisional)
    --seeds        int list (default 按 seed_registry)
    --smoke        每方法 2 epoch 冒烟
    --max-epochs   覆盖 epoch 数（resource-adjusted，非严格复现，自动标注）
    --plan         只打印将要运行的单元，不执行
    --no-setup     跳过环境安装（重复会话提速）
    --resume       合并 /kaggle/input/ 下的 registry.jsonl（跳过已完成单元）

结果输出: /kaggle/working/hod-gnn-results/  （下载整个文件夹）
    registry.jsonl          所有运行记录
    paper_vs_reproduced.csv 汇总表
    logs/                   原始训练日志（审计用）

协议说明:
    - paper-seeds-provisional: 论文未披露 seed，[0,1,2,3] 为临时替代
    - official-seeds: 各方法官方 seed 口径（GPS 0-9 / policy-learn 1-5 / GraphViT 0-3）
    - 只运行 method_dataset_matrix 中 run:true 的单元
"""

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import selectors
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ============================================================
# 常量
# ============================================================

WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", ".")).resolve()
# Kaggle 默认把 runner、vendor 和结果都放在 /kaggle/working；本地复用时
# 可以把官方仓库、解释器和结果目录分别指向已有资源，避免重新下载/安装。
RES = Path(os.environ.get("HOD_GNN_RESULTS_DIR", str(WORK / "hod-gnn-results"))).resolve()
LOGS = RES / "logs"
REGISTRY = RES / "registry.jsonl"
EVENTS = RES / "events.jsonl"
STATE = RES / "run_state.json"
VENDOR = Path(os.environ.get("HOD_GNN_VENDOR_DIR", str(WORK / "vendor"))).resolve()
VENV = Path(os.environ.get("HOD_GNN_VENV_DIR", str(WORK / ".venv310"))).resolve()
# Do not resolve the interpreter symlink: a venv's ``bin/python`` symlink is
# what activates its site-packages.  Resolving it would invoke the bare uv
# interpreter and silently lose torch/PyG imports in local runs.
VENV_PY = Path(os.environ.get("HOD_GNN_VENV_PY", str(VENV / "bin" / "python")))

# Set by main().  Keeping this state in one place makes the same runner usable
# from a notebook, a shell, and a resumed Kaggle session.
RUN_ID = None
LIVE_LOG = False
RUNTIME_INFO = {}
UNIT_TIMEOUT_SECONDS = None

# 官方仓库（固定 commit）
REPOS = {
    "gps": {
        "url": "https://github.com/rampasek/GraphGPS.git",
        "commit": "28015707cbab7f8ad72bed0ee872d068ea59c94b",
        "dir": "GraphGPS",
    },
    "graphvit": {
        "url": "https://github.com/XiaoxinHe/Graph-ViT-MLPMixer.git",
        "commit": "0d66dd1b0d8376e9252a73d60c10940ac626ca81",
        "dir": "GraphViT",
    },
    "policy": {
        "url": "https://github.com/beabevi/policy-learn.git",
        "commit": "a84adff1cde410fd6ffd098ccd7e813cd60656f0",
        "dir": "policy-learn",
    },
}

# 方法 × 数据集 矩阵（run:false = 论文表格为 "-"，不跑）
MATRIX = {
    "gps": {"zinc": True, "moltox21": True, "molbace": False, "molhiv": True,
            "peptides-func": False, "peptides-struct": False},
    "graphvit": {"zinc": True, "moltox21": True, "molbace": False, "molhiv": True,
                 "peptides-func": True, "peptides-struct": True},
    "full": {"zinc": True, "moltox21": True, "molbace": True, "molhiv": True,
             "peptides-func": False, "peptides-struct": False},
    "random": {"zinc": True, "moltox21": True, "molbace": True, "molhiv": True,
               "peptides-func": False, "peptides-struct": False},
    "policy_learn": {"zinc": True, "moltox21": True, "molbace": True, "molhiv": True,
                     "peptides-func": True, "peptides-struct": True},
}

# 数据集 split hash（本 track 已用官方 ogb/LRGB 校验；zinc 为固定 split）
SPLIT_HASH = {
    "molhiv": "feda8af43ac4b55b656ba2e02727d03560e898f3edad160a615797f6ad321179",
    "moltox21": "cc8e66d463fa3923b7a8dd769bdd88dcd8c6f88530af0c78cd098381719b8875",
    "molbace": "b0bfcf9052e25ed1c2365caa7df668bfdca662ee476d66f57ff60c18c7efdd32",
    "zinc": "fixed-split-zinc12k",
    "peptides-func": "lrgb-official",
    "peptides-struct": "lrgb-official",
}

# 指标
METRIC = {"zinc": "mae", "moltox21": "auc", "molbace": "auc", "molhiv": "auc",
          "peptides-func": "ap", "peptides-struct": "mae"}
MODE = {"mae": "min", "auc": "max", "ap": "max"}
METRIC_NAME = {"mae": "mae", "auc": "roc_auc", "ap": "average_precision"}

# seed 默认值（paper-seeds-provisional 一律 0-3）
SEED_DEFAULT_PAPER = [0, 1, 2, 3]
SEED_DEFAULT_OFFICIAL = {
    "gps": list(range(10)), "graphvit": [0, 1, 2, 3],
    "full": [1, 2, 3, 4, 5], "random": [1, 2, 3, 4, 5], "policy_learn": [1, 2, 3, 4, 5],
}

PAPER_TARGETS = {  # 论文数字（HOD-GNN 表 1/2）；AUC/AP 统一存为 0..1
    ("gps", "zinc"): (0.070, 0.004), ("gps", "moltox21"): (0.7570, 0.0040), ("gps", "molhiv"): (0.7880, 0.0101),
    ("graphvit", "zinc"): (0.085, 0.005), ("graphvit", "moltox21"): (0.7851, 0.0077), ("graphvit", "molhiv"): (0.7792, 0.0149),
    ("graphvit", "peptides-func"): (0.6919, 0.0085), ("graphvit", "peptides-struct"): (0.2474, 0.0016),
    ("full", "zinc"): (0.087, 0.003), ("full", "moltox21"): (0.7625, 0.0112),
    ("full", "molbace"): (0.7841, 0.0194), ("full", "molhiv"): (0.7654, 0.0137),
    ("random", "zinc"): (0.102, 0.003), ("random", "moltox21"): (0.7662, 0.0063),
    ("random", "molbace"): (0.7814, 0.0236), ("random", "molhiv"): (0.7730, 0.0256),
    ("policy_learn", "zinc"): (0.097, 0.005), ("policy_learn", "moltox21"): (0.7736, 0.0060),
    ("policy_learn", "molbace"): (0.7839, 0.0228), ("policy_learn", "molhiv"): (0.7849, 0.0101),
    ("policy_learn", "peptides-func"): (0.6459, 0.0018), ("policy_learn", "peptides-struct"): (0.2475, 0.0011),
}

# 官方 config 入口
GPS_CFG = {
    "zinc": "configs/GPS/zinc-GPS+RWSE.yaml",
    "molhiv": "configs/GPS/ogbg-molhiv-GPS+RWSE.yaml",
    "moltox21": None,  # 官方仓库无配置 → unavailable
    "peptides-func": "configs/GPS/peptides-func-GPS.yaml",
    "peptides-struct": "configs/GPS/peptides-struct-GPS.yaml",
}
GV_MODULE = {
    "zinc": "train.zinc", "molhiv": "train.molhiv", "moltox21": "train.moltox21",
    "peptides-func": "train.peptides_func", "peptides-struct": "train.peptides_struct",
}
POLICY_YAML = {"zinc": "zinc", "molhiv": "molhiv", "moltox21": "moltox21", "molbace": "molbace"}
SELECTION_TYPE = {"full": "all", "random": "random", "policy_learn": "gumbel"}

# ============================================================
# 工具
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_json(path, payload):
    """原子写状态文件，避免 Kaggle 会话中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def command_string(cmd):
    return shlex.join([str(x) for x in cmd]) if isinstance(cmd, (list, tuple)) else str(cmd)


def emit_event(event_name, **fields):
    """写一条机器可读事件，同时输出一行易读摘要。"""
    payload = {
        "timestamp": utc_now(),
        "event": event_name,
        "run_id": RUN_ID,
        "pid": os.getpid(),
        **fields,
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())
    summary = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"[event] {event_name}" + (f" {summary}" if summary else ""), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def sh(cmd, cwd=None, timeout=1800, **kw):
    cmd_text = command_string(cmd)
    started = time.time()
    print(f"[sh] {cmd_text}", flush=True)
    emit_event("command_start", command=cmd_text, cwd=str(cwd or WORK), timeout_seconds=timeout)
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as ex:
        emit_event("command_timeout", command=cmd_text, cwd=str(cwd or WORK),
                   duration_seconds=round(time.time() - started, 3))
        print(f"[sh] TIMEOUT after {timeout}s: {cmd_text}", flush=True)
        raise ex
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
    emit_event("command_end", command=cmd_text, cwd=str(cwd or WORK),
               returncode=r.returncode, duration_seconds=round(time.time() - started, 3),
               stdout_tail=r.stdout[-1000:] if r.stdout else None,
               stderr_tail=r.stderr[-1000:] if r.stderr else None)
    return r


def run(cmd, cwd, log_path, timeout=86400, label=None, env=None):
    """运行一个训练单元并记录可审计的心跳、耗时、退出码和日志 hash。

    训练输出完整保存在 per-unit log；默认只把心跳和关键结果行透传到
    notebook，设置 ``--live-log`` 才会把每一行都透传，避免 Kaggle 输出过大。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    label = label or log_path.stem
    cmd_text = command_string(cmd)
    started_at = utc_now()
    emit_event("unit_process_start", unit=label, command=cmd_text, cwd=str(cwd),
               log=str(log_path.relative_to(WORK)), timeout_seconds=timeout)
    tail = deque(maxlen=30)
    line_count = 0
    timed_out = False
    returncode = None
    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
        selector = selectors.DefaultSelector()
        selector.register(p.stdout, selectors.EVENT_READ)
        last_heartbeat = t0
        while True:
            ready = selector.select(timeout=5)
            if ready:
                line = p.stdout.readline()
                if line:
                    lf.write(line)
                    tail.append(line.rstrip())
                    line_count += 1
                    stripped = line.strip()
                    # 训练日志常常只在 epoch 结束才输出；透传 metric/epoch 行
                    # 能让用户判断进程确实在推进，而不刷爆 Kaggle notebook。
                    if LIVE_LOG or re.search(r"(?i)(epoch|train:|val:|test:|best|error|traceback|oom)", stripped):
                        print(f"[{label}] {stripped[-500:]}", flush=True)
                elif p.poll() is not None:
                    break
            if p.poll() is not None and not ready:
                break
            elapsed = time.time() - t0
            if elapsed - (last_heartbeat - t0) >= 60:
                last_heartbeat = time.time()
                emit_event("unit_heartbeat", unit=label, elapsed_seconds=round(elapsed, 1),
                           lines=line_count, log_bytes=log_path.stat().st_size if log_path.exists() else 0)
            if elapsed > timeout:
                timed_out = True
                emit_event("unit_timeout", unit=label, elapsed_seconds=round(elapsed, 1))
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                break
        selector.close()
        returncode = p.returncode
    duration = time.time() - t0
    meta = {
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round(duration, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "line_count": line_count,
        "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "log_sha256": sha256_file(log_path),
        "command": cmd_text,
        "tail": list(tail)[-10:],
    }
    emit_event("unit_process_end", unit=label, returncode=returncode,
               duration_seconds=meta["duration_seconds"], timed_out=timed_out,
               lines=line_count, log_bytes=meta["log_bytes"])
    return returncode, duration, meta


def load_done(include_external=True):
    """读取本地及 Kaggle input registry，返回最后一次成功/不可用的单元。

    runtime_error、oom、interrupted 等终态不会被视为完成，因此 ``--resume``
    会自动重试失败单元；这样中断后不需要手工编辑 registry。
    """
    done = {}
    candidates = [REGISTRY]
    if include_external:
        candidates += sorted(glob.glob("/kaggle/input/**/registry.jsonl", recursive=True))
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    key = (e["method"], e["dataset"], e["protocol"], e["seed"])
                    if e.get("status") in {"success", "unavailable"}:
                        done[key] = e
                except Exception:
                    pass
    return done


def detect_gpu():
    """探测实际 GPU 型号（Kaggle 免费 tier 在 T4/P100 间随机分配）。
    返回 (gpu_name, torch_version, torch_cuda_version)。"""
    try:
        if not VENV_PY.exists():
            return ("CPU-only (venv not installed)", "unknown", {})
        r = sh([str(VENV_PY), "-c",
                "import torch, sys; "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'); "
                "print(torch.__version__); "
                "print(torch.version.cuda or 'none'); "
                "print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)"],
               timeout=120)
        lines = r.stdout.strip().splitlines()
        gpu_name = lines[0] if lines else "unknown"
        torch_ver = lines[1] if len(lines) > 1 else "unknown"
        return gpu_name, torch_ver, {
            "cuda_version": lines[2] if len(lines) > 2 else "unknown",
            "gpu_memory_bytes": int(lines[3]) if len(lines) > 3 and lines[3].isdigit() else None,
        }
    except Exception as ex:
        return f"unknown ({ex})", "unknown", {}


def append_registry(entry):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("timestamp", utc_now())
    entry.setdefault("run_id", RUN_ID)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())
    emit_event("registry_append", method=entry.get("method"), dataset=entry.get("dataset"),
               protocol=entry.get("protocol"), seed=entry.get("seed"),
               status=entry.get("status"))
    print(f"[registry] {entry['method']}/{entry['dataset']}/{entry['protocol']}/seed={entry['seed']} "
          f"{entry.get('metric_name')}={entry.get('test_metric')} [{entry['status']}]")


# ============================================================
# 环境安装（幂等）
# ============================================================

def ensure_venv():
    """官方基线代码需要 torch 1.13（仅支持 cp37-cp310）。
    本机/Kaggle 可能是 3.11/3.12+，因此强制用 Python 3.10 建 venv，
    所有 pip 与运行都走该 venv。优先用 uv（自带 pip 管理，避免 ensurepip 问题）。"""
    if VENV_PY.exists():
        print(f"[venv] use {VENV_PY}")
        return str(VENV_PY)
    import shutil
    uv = shutil.which("uv")
    py310 = shutil.which("python3.10")
    if uv:
        print("[venv] using uv venv --python 3.10")
        sh([uv, "venv", "--python", "3.10", str(VENV)], timeout=600)
    elif py310:
        print("[venv] using python3.10 -m venv")
        sh([py310, "-m", "venv", str(VENV)], timeout=300)
    if not VENV_PY.exists():
        print("[ERROR] cannot create Python 3.10 venv")
        sys.exit(1)
    print(f"[venv] created {VENV_PY}")
    return str(VENV_PY)


def pip_install(*pkgs, timeout=1200, extra=None, lock_torch=True):
    """在 venv 内安装。优先 uv（venv 可能没有 pip）。
    关键：默认追加 torch==1.13.0/torchvision==0.14.0 约束，防止 uv
    解析依赖时把已装的 torch 1.13 升级到 2.x（ABI 破坏）。"""
    import shutil
    if lock_torch:
        pkgs = list(pkgs) + ["torch==1.13.0", "torchvision==0.14.0"]
    uv = shutil.which("uv")
    if uv:
        r = sh([uv, "pip", "install", "--python", str(VENV_PY), "-q", *pkgs, *(extra or [])],
               timeout=timeout)
    else:
        r = sh([str(VENV_PY), "-m", "pip", "install", "-q", *pkgs, *(extra or [])],
               timeout=timeout)
    return r.returncode


def setup_env(methods=None):
    """安装与官方仓库匹配的 torch/pyg + 各方法依赖。
    torch 1.13 + pyg 2.2（GraphGPS 官方）；venv 为 Python 3.10。"""
    methods = set(methods or REPOS)
    print(f"=== setup_env methods={sorted(methods)} ===")
    ensure_venv()
    pyv = subprocess.run([str(VENV_PY), "--version"], capture_output=True, text=True).stdout.strip()
    print(f"[venv] python: {pyv}")
    # 与 GraphGPS 官方一致的 CUDA torch（cu117；Kaggle GPU 镜像有 CUDA 117 运行库）
    pip_install("torch==1.13.0", "torchvision==0.14.0",
                extra=["--index-url", "https://download.pytorch.org/whl/cu117"], timeout=900)
    # 官方 cu117 wheel 依赖 libcudart.so.11 等 CUDA 11 运行库；
    # Kaggle 镜像默认只有 CUDA 12/13，需装 NVIDIA 的 cu11 pip 包补齐 .so
    # (nvidia-nvrtc-cu11 在 PyPI 不存在；torch 1.13 运行不需要 libnvrtc)
    pip_install("nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11", "nvidia-cufft-cu11",
                "nvidia-curand-cu11", "nvidia-cusolver-cu11", "nvidia-cusparse-cu11",
                timeout=900)
    # 把 venv 里所有 nvidia/*/lib 加入 LD_LIBRARY_PATH（每个包一个 lib 目录）
    nvidia_root = VENV_PY.parent.parent / "lib" / "python3.10" / "site-packages" / "nvidia"
    nvidia_lib_dirs = []
    if nvidia_root.exists():
        for d in nvidia_root.glob("*/lib"):
            nvidia_lib_dirs.append(str(d))
    if nvidia_lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        print(f"[env] LD_LIBRARY_PATH += {'; '.join(sorted(nvidia_lib_dirs))}")
    # pyg 编译扩展（CUDA 117 版本；直接用精确 wheel URL，
    # 避免 uv 从 PyPI 解析源码版导致编译失败）
    PYG_INDEX = "https://data.pyg.org/whl/torch-1.13.0+cu117"
    pyg_wheels = [
        f"{PYG_INDEX}/pyg_lib-0.4.0%2Bpt113cu117-cp310-cp310-linux_x86_64.whl",
        f"{PYG_INDEX}/torch_scatter-2.1.1%2Bpt113cu117-cp310-cp310-linux_x86_64.whl",
        f"{PYG_INDEX}/torch_sparse-0.6.17%2Bpt113cu117-cp310-cp310-linux_x86_64.whl",
        f"{PYG_INDEX}/torch_cluster-1.6.1%2Bpt113cu117-cp310-cp310-linux_x86_64.whl",
    ]
    pip_install(*pyg_wheels, timeout=900)
    # 其余依赖（注意顺序：numpy<2、setuptools<80 避免 pkg_resources/ABI 问题）
    # GraphGPS 的 logger 仍调用 mean_squared_error(..., squared=False)。
    # scikit-learn 1.5+ 删除了这个参数，故固定到仍兼容官方代码的版本。
    # Policy-Learn 的训练脚本无条件导入 matplotlib，即使不启用可视化也需要它。
    pip_install("numpy<2", "setuptools<80", "scikit-learn<1.5", "matplotlib",
                "torch-geometric==2.2.0", "pytorch-lightning<1.7",
                "torchmetrics<1.0", "yacs", "performer-pytorch", "tensorboardX",
                "ogb", "wandb", "einops", "networkx", "rdkit",
                "hydra-core", "omegaconf", "tqdm", timeout=1800)
    # GraphViT 需要 METIS 图划分；只跑 GPS 时不要无谓安装。
    if "graphvit" in methods:
        pip_install("metis", timeout=900)
        # pip 的 metis 包只是 ctypes wrapper，真正的 libmetis 由 conda/系统提供。
        # Kaggle 镜像不保证预装它；尽量复用已有库，否则在 Linux 镜像中安装
        # libmetis-dev，并把绝对路径传给 wrapper 的 METIS_DLL。
        import glob as _glob
        import re as _re
        metis_candidates = []
        configured_metis = os.environ.get("METIS_DLL")
        if configured_metis and os.path.exists(configured_metis):
            metis_candidates.append(configured_metis)
        for pattern in [
            "/usr/lib/**/libmetis.so*", "/usr/local/lib/**/libmetis.so*",
            "/opt/conda/lib/**/libmetis.so*", str(VENV / "**" / "libmetis.so*"),
        ]:
            metis_candidates.extend(_glob.glob(pattern, recursive=True))
        if not metis_candidates and shutil.which("ldconfig"):
            ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
            metis_candidates.extend(_re.findall(r"=>\s*(/[^\s]*libmetis\.so[^\s]*)", ld.stdout))
        if not metis_candidates and shutil.which("apt-get"):
            print("[metis] libmetis not found; installing libmetis-dev")
            sh(["apt-get", "update", "-qq"], timeout=600)
            sh(["apt-get", "install", "-y", "-qq", "libmetis-dev"], timeout=600)
            for pattern in ["/usr/lib/**/libmetis.so*", "/usr/local/lib/**/libmetis.so*"]:
                metis_candidates.extend(_glob.glob(pattern, recursive=True))
        if metis_candidates:
            os.environ["METIS_DLL"] = sorted(set(metis_candidates), key=len)[0]
            print(f"[metis] METIS_DLL={os.environ['METIS_DLL']}")
        else:
            print("[metis] WARNING: libmetis was not found; GraphViT may fail at import")
    # 兜底：强制重装 torch 1.13.0+cu117，确保后续依赖安装没有把它升级掉
    pip_install("torch==1.13.0", "torchvision==0.14.0",
                extra=["--index-url", "https://download.pytorch.org/whl/cu117",
                       "--force-reinstall", "--no-deps"], timeout=900)
    # 验证环境（打印实际版本，自证没有被升级）
    chk = sh([str(VENV_PY), "-c",
              "import torch, torch_geometric; print('TORCH_VER', torch.__version__); "
              "print('CUDA_OK', torch.cuda.is_available()); "
              "print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); "
              "print('PYG', torch_geometric.__version__)"],
             timeout=300)
    if chk.returncode != 0:
        print("[ERROR] venv verification failed — torch/pyg not importable")
        sys.exit(1)
    print("=== setup_env done ===")


def clone_repos(methods=None):
    methods = set(methods or REPOS)
    VENDOR.mkdir(parents=True, exist_ok=True)
    repo_keys = set()
    for method in methods:
        repo_keys.add("gps" if method == "gps" else "graphvit" if method == "graphvit" else "policy")
    for key, repo in REPOS.items():
        if key not in repo_keys:
            continue
        dest = VENDOR / repo["dir"]
        if (dest / ".git").exists():
            print(f"[repo] {key}: already cloned")
            continue
        sh(["git", "clone", "-q", repo["url"], str(dest)], timeout=600)
        if repo["commit"] != "HEAD":
            sh(["git", "-C", str(dest), "checkout", "-q", repo["commit"]], timeout=120)
        print(f"[repo] {key}: {repo['commit'][:12]}")
    # policy-learn 编译 csrc（需 nvcc；Kaggle GPU 镜像通常有）
    # 需要 --no-build-isolation：uv 构建隔离环境里没有 torch，setup.py 会失败
    if "policy" in repo_keys and (VENDOR / "policy-learn" / "setup.py").exists():
        import shutil as _shutil
        uv = _shutil.which("uv")
        if uv:
            r = sh([uv, "pip", "install", "--python", str(VENV_PY), "-q", "-e",
                    str(VENDOR / "policy-learn"),
                    "--no-build-isolation", "torch==1.13.0", "torchvision==0.14.0"],
                   timeout=1800)
        else:
            r = sh([str(VENV_PY), "-m", "pip", "install", "-q", "-e",
                    str(VENDOR / "policy-learn"), "--no-build-isolation"], timeout=1800)
        print(f"[policy-learn] csrc build returncode={r.returncode}")


# ============================================================
# 运行
# ============================================================

def gps_checkpoint_root(dataset, seed):
    """Persistent GraphGPS output root for a resumable unit."""
    return RES / "checkpoints" / f"gps-{dataset}-seed{seed}"


def checkpoint_epochs(path):
    return sorted(
        int(p.stem) for p in path.rglob("*.ckpt") if p.stem.isdigit()
    ) if path.exists() else []


def restore_gps_checkpoint(dataset, seed):
    """Restore a prior Kaggle output checkpoint into the active RES tree.

    Kaggle mounts uploaded result datasets below ``/kaggle/input``.  We copy
    only the named GPS/ZINC seed tree and never overwrite a newer local copy.
    The same function also accepts ``HOD_GNN_CHECKPOINT_INPUT`` for local or
    scripted orchestration tests.
    """
    dest = gps_checkpoint_root(dataset, seed)
    dest.mkdir(parents=True, exist_ok=True)
    if checkpoint_epochs(dest):
        print(f"[ckpt] existing {dest} epochs={checkpoint_epochs(dest)[-5:]}")
        emit_event("checkpoint_ready", dataset=dataset, seed=seed,
                   path=str(dest), epochs=checkpoint_epochs(dest))
        return dest

    roots = []
    explicit = os.environ.get("HOD_GNN_CHECKPOINT_INPUT")
    if explicit:
        roots.append(Path(explicit))
    roots += [Path(p) for p in glob.glob("/kaggle/input/**/checkpoints", recursive=True)]
    candidates = []
    wanted = f"gps-{dataset}-seed{seed}"
    for root in roots:
        if not root.exists():
            continue
        direct = root / wanted
        if direct.exists():
            candidates.append(direct)
        candidates += [Path(p) for p in glob.glob(str(root / "**" / wanted), recursive=True)]

    # Kaggle Dataset uploads commonly package directories as one zip.  Extract
    # only this unit's checkpoint tree so a 200 MB results archive does not
    # needlessly duplicate logs and prior summaries in the working directory.
    archives = []
    for root in [Path(explicit)] if explicit else []:
        archives += [Path(p) for p in glob.glob(str(root / "**" / "*.zip"), recursive=True)]
    archives += [Path(p) for p in glob.glob("/kaggle/input/**/*.zip", recursive=True)]
    for archive in archives:
        if not archive.exists():
            continue
        try:
            import zipfile
            with zipfile.ZipFile(archive) as zf:
                members = [n for n in zf.namelist()
                           if f"checkpoints/{wanted}/" in n and not n.endswith("/")]
                if members:
                    print(f"[ckpt] extracting {len(members)} files from {archive}")
                    prefix = next((n.split(f"checkpoints/{wanted}/", 1)[0]
                                   for n in members if f"checkpoints/{wanted}/" in n), "")
                    for name in members:
                        rel = name.split(f"checkpoints/{wanted}/", 1)[1]
                        target = dest / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(name) as src, open(target, "wb") as out:
                            shutil.copyfileobj(src, out)
                    emit_event("checkpoint_extracted", dataset=dataset, seed=seed,
                               source=str(archive), destination=str(dest),
                               files=len(members), epochs=checkpoint_epochs(dest))
                    break
        except (OSError, zipfile.BadZipFile) as ex:
            print(f"[ckpt] archive read failed {archive}: {ex}")
    # Fallback for the previous non-checkpointed runner, whose active output
    # lived under vendor/GraphGPS/results/<config>/<seed>.
    for p in glob.glob("/kaggle/input/**/vendor/GraphGPS/results/zinc-GPS+RWSE/%s" % seed,
                       recursive=True):
        candidates.append(Path(p).parent.parent.parent / ".." / ".." / ".." / wanted)
        candidates.append(Path(p))

    seen = set()
    for src in candidates:
        src = src.resolve()
        if src in seen or not src.exists() or not checkpoint_epochs(src):
            continue
        seen.add(src)
        print(f"[ckpt] restoring {src} -> {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)
        emit_event("checkpoint_restored", dataset=dataset, seed=seed,
                   source=str(src), destination=str(dest),
                   epochs=checkpoint_epochs(dest))
        break
    if not checkpoint_epochs(dest):
        print(f"[ckpt] no prior checkpoint for {dataset}/seed={seed}; start at epoch 0")
        emit_event("checkpoint_missing", dataset=dataset, seed=seed, path=str(dest))
    return dest

def run_gps(dataset, seed, smoke, max_epochs):
    cfg = GPS_CFG.get(dataset)
    if cfg is None:
        return None, "unavailable", "no official config in GraphGPS repo (needs HyMN fallback)", {}
    repo = VENDOR / "GraphGPS"
    cmd = [str(VENV_PY), "main.py", "--cfg", cfg, "wandb.use", "False", "seed", str(seed)]
    # ZINC's official GPS config is 2000 epochs and a single Kaggle session may
    # end before one seed completes.  Keep the official max_epoch/scheduler,
    # but put GraphGPS checkpoints directly under RES so the next Kaggle
    # session can mount the previous output and continue from the last saved
    # epoch.  Checkpoint cadence only affects I/O, not the optimization path.
    checkpoint_dir = None
    if dataset == "zinc" and not smoke and max_epochs is None:
        checkpoint_dir = restore_gps_checkpoint(dataset, seed)
        cmd += [
            "out_dir", str(checkpoint_dir),
            "train.auto_resume", "True",
            "train.ckpt_clean", "False",
            "train.ckpt_period", "25",
        ]
    if smoke:
        cmd += ["optim.max_epoch", "2"]
    if max_epochs:
        cmd += ["optim.max_epoch", str(max_epochs)]
    log = LOGS / f"gps-{dataset}-{seed}.log"
    timeout = UNIT_TIMEOUT_SECONDS or 86400
    rc, secs, meta = run(cmd, repo, log, timeout=timeout,
                         label=f"gps/{dataset}/seed={seed}")
    if checkpoint_dir is not None:
        meta["checkpoint_dir"] = str(checkpoint_dir.relative_to(WORK))
        meta["checkpoint_epochs"] = checkpoint_epochs(checkpoint_dir)
    if rc != 0:
        status = "oom" if any("out of memory" in x.lower() for x in meta.get("tail", [])) else "runtime_error"
        return log, status, f"exit code {rc}", meta
    return log, "success", None, meta


def run_graphvit(dataset, seed, smoke, max_epochs):
    mod = GV_MODULE.get(dataset)
    if mod is None:
        return None, "unavailable", "no dataset module in GraphViT repo", {}
    repo = VENDOR / "GraphViT"
    cmd = [str(VENV_PY), "-m", mod, f"cfg.seed={seed}", "cfg.train.runs=1"]
    if smoke:
        cmd.append("cfg.train.epochs=2")
    if max_epochs:
        cmd.append(f"cfg.train.epochs={max_epochs}")
    log = LOGS / f"graphvit-{dataset}-{seed}.log"
    rc, secs, meta = run(cmd, repo, log, label=f"graphvit/{dataset}/seed={seed}")
    if rc != 0:
        status = "oom" if any("out of memory" in x.lower() for x in meta.get("tail", [])) else "runtime_error"
        return log, status, f"exit code {rc}", meta
    return log, "success", None, meta


def run_policy(policy, dataset, seed, smoke, max_epochs):
    yaml_key = POLICY_YAML.get(dataset)
    if yaml_key is None:
        return None, "unavailable", "no config in policy-learn repo (needs HyMN fallback)", {}
    repo = VENDOR / "policy-learn"
    env = dict(os.environ, WANDB_MODE="disabled")
    cmd = [str(VENV_PY), "train.py", f"dataset={yaml_key}",
           f"selection_type={SELECTION_TYPE[policy]}", f"seed={seed}"]
    if smoke:
        cmd.append("epochs=2")
    if max_epochs:
        cmd.append(f"epochs={max_epochs}")
    log = LOGS / f"{policy}-{dataset}-{seed}.log"
    rc, secs, meta = run(cmd, repo, log, label=f"{policy}/{dataset}/seed={seed}", env=env)
    if rc != 0:
        status = "oom" if any("out of memory" in x.lower() for x in meta.get("tail", [])) else "runtime_error"
        return log, status, f"exit code {rc}", meta
    return log, "success", None, meta


# ============================================================
# 解析（best-effort，原始日志全保留）
# ============================================================

def parse_gps(log):
    """GraphGPS: 每 epoch 输出 train/val/test dict，按 val 最优选 epoch。"""
    epochs = []
    cur = {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"^(train|val|test):\s*(\{.*\})", line)
            if m:
                split = m.group(1)
                try:
                    d = ast.literal_eval(m.group(2))
                except (SyntaxError, ValueError):
                    continue
                cur[split] = d
                if split == "test":
                    epochs.append(cur)
                    cur = {}
    return epochs


def pick_best(epochs, metric, mode):
    best = None
    best_val = None
    for e in epochs:
        val = e.get("val", {}).get(metric)
        if val is None:
            continue
        if best_val is None or (val < best_val if mode == "min" else val > best_val):
            best_val = val
            best = e
    return best, best_val


def parse_generic(log):
    """GraphViT / policy-learn: 尝试常见格式提取 best val / test。
    找不到就返回 None，靠人工审计原始日志。"""
    text = open(log, encoding="utf-8", errors="replace").read()
    best = {}
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    separator = r"[^0-9.+-eE]*"
    for key, pat in [
        ("best_epoch", r"best\s*(?:epoch|iter)[^\d]*(\d+)"),
        ("best_val", rf"best\s*val{separator}{number}"),
        ("best_test", rf"best\s*test{separator}{number}"),
        ("test_mae", rf"test[_-]*(?:mae|loss){separator}{number}"),
        ("test_auc", rf"test[_-]*auc{separator}{number}"),
        ("test_ap", rf"test[_-]*ap{separator}{number}"),
        ("val_auc", rf"val[_-]*auc{separator}{number}"),
        ("val_mae", rf"val[_-]*(?:mae|loss){separator}{number}"),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            best[key] = float(m.group(1))
    return best


# ============================================================
# 汇总
# ============================================================

def summarize():
    import statistics
    entries = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8", errors="replace") as f:
            entries = [json.loads(l) for l in f if l.strip()]
    # 同一个单元重试后 registry 会保留历史（这是审计需要），汇总只取
    # 最后一次终态，避免一次失败+一次成功被统计成两个 seed。
    latest = {}
    for e in entries:
        key = (e.get("method"), e.get("dataset"), e.get("protocol"), e.get("seed"))
        latest[key] = e
    entries = list(latest.values())
    rows = []
    for e in entries:
        if e["status"] != "success" or e.get("test_metric") is None:
            continue
        rows.append(e)
    # 分组统计
    from collections import defaultdict
    groups = defaultdict(list)
    for e in rows:
        groups[(e["method"], e["dataset"], e["protocol"])].append(e["test_metric"])
    print("\n=== SUMMARY (mean±std, n, paper target, verdict) ===")
    out = []
    for (m, d, p), vals in sorted(groups.items()):
        n = len(vals)
        mean = sum(vals) / n
        std = statistics.stdev(vals) if n > 1 else 0.0
        tgt = PAPER_TARGETS.get((m, d))
        tgt_s = f"{tgt[0]}±{tgt[1]}" if tgt else "—"
        verdict = "?"
        if tgt:
            diff = abs(mean - tgt[0])
            comb = (std ** 2 + tgt[1] ** 2) ** 0.5
            verdict = "reproduced" if diff <= comb else ("partial" if diff <= 2 * comb else "not_reproduced")
        print(f"{m:12s} {d:14s} {p:26s} {mean:.4f}±{std:.4f} (n={n})  paper={tgt_s}  {verdict}")
        out.append([m, d, p, f"{mean:.4f}", f"{std:.4f}", str(n), tgt_s, verdict])
    import csv
    csv_path = RES / "paper_vs_reproduced.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "protocol", "repro_mean", "repro_std", "n", "paper", "verdict"])
        w.writerows(out)
    print(f"\n[csv] {csv_path}")


# ============================================================
# 主流程
# ============================================================

def unit_config_hash(method, dataset, protocol, seed, args):
    payload = {
        "method": method, "dataset": dataset, "protocol": protocol, "seed": seed,
        "smoke": bool(args.smoke), "max_epochs": args.max_epochs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_plan(args, done):
    methods = ["gps", "graphvit", "full", "random", "policy_learn"] if args.method == "all" else [args.method]
    datasets = ["zinc", "moltox21", "molbace", "molhiv", "peptides-func", "peptides-struct"] \
        if args.dataset == "all" else [args.dataset]
    plan = []
    for m in methods:
        for d in datasets:
            if not MATRIX.get(m, {}).get(d, False):
                continue
            if args.seeds:
                seeds = args.seeds
            elif args.protocol == "official-seeds":
                seeds = SEED_DEFAULT_OFFICIAL.get(m, SEED_DEFAULT_PAPER)
            else:
                seeds = SEED_DEFAULT_PAPER
            for s in seeds:
                key = (m, d, args.protocol, s)
                expected_hash = unit_config_hash(m, d, args.protocol, s, args)
                previous = done.get(key)
                plan.append({"method": m, "dataset": d, "seed": s,
                             "config_hash": expected_hash,
                             "done": bool(previous and previous.get("config_hash") == expected_hash)})
    return plan


class Tee:
    """同时写文件与 stdout，保证 notebook 可见且可审计。"""
    def __init__(self, path, stdout=None):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = stdout if stdout is not None else sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.f.write(data)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def process_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def acquire_lock(lock):
    """获取进程锁，并自动清理上次被 Kaggle 强杀留下的 stale lock。"""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            old_pid = int(lock.read_text().strip())
        except (OSError, ValueError):
            old_pid = -1
        if process_alive(old_pid):
            print(f"another kaggle_run instance (pid={old_pid}) holds lock; exit")
            return False
        try:
            lock.unlink()
            emit_event("stale_lock_removed", stale_pid=old_pid)
        except OSError:
            return False
        return acquire_lock(lock)


def save_state(status, args, plan, current=None, extra=None):
    completed = sum(1 for p in plan if p.get("done"))
    payload = {
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "updated_at": utc_now(),
        "status": status,
        "args": vars(args),
        "plan_total": len(plan),
        "plan_done_at_start": completed,
        "plan_remaining_at_last_update": sum(1 for p in plan if not p.get("done")),
        "current": current,
        "plan": plan,
    }
    if extra:
        payload.update(extra)
    atomic_write_json(STATE, payload)


def collect_runtime_info():
    gpu_name, torch_ver, extra = detect_gpu()
    return {
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        **extra,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def package_results():
    """打包结果但保留工作目录，便于同一 Kaggle 会话继续跑。"""
    import zipfile
    zip_path = RES.with_name("hod-gnn-results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in RES.rglob("*"):
            if f.is_file() and f.name != ".lock":
                zf.write(f, str(f.relative_to(WORK)))
    return zip_path


def main():
    global RUN_ID, LIVE_LOG, RUNTIME_INFO, UNIT_TIMEOUT_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--protocol", default="paper-seeds-provisional",
                    choices=["paper-seeds-provisional", "official-seeds"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--unit-timeout-seconds", type=int, default=None,
                    help="主动结束单元并打包 checkpoint，避免 Kaggle 硬超时；推荐 ZINC 使用 21600")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--no-setup", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已有 success/unavailable；同时读取 /kaggle/input/**/registry.jsonl")
    ap.add_argument("--live-log", action="store_true",
                    help="把每一行训练输出透传到 notebook（默认只透传 epoch/metric/错误和每分钟心跳）")
    ap.add_argument("--keep-work", action="store_true",
                    help="完成后保留 vendor/.venv/data；默认清理以减小 Kaggle output")
    args = ap.parse_args()

    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-p{os.getpid()}"
    LIVE_LOG = args.live_log
    UNIT_TIMEOUT_SECONDS = args.unit_timeout_seconds
    RES.mkdir(parents=True, exist_ok=True)
    # 进程锁：Kaggle 可能执行 notebook 两次，防止两个进程并发互踩 venv
    lock = RES / ".lock"
    if not acquire_lock(lock):
        sys.exit(0)
    # 若同一解释器里重复调用 main()，不要把 Tee 套在旧 Tee 上；否则每一行
    # 会被写到外层 stdout 两次。Kaggle notebook 使用子进程时通常不会触发，
    # 但这个保护也覆盖本地 notebook/测试环境。
    base_stdout = sys.stdout.stdout if isinstance(sys.stdout, Tee) else sys.stdout
    sys.stdout = Tee(RES / "run.log", stdout=base_stdout)
    try:
        print(f"=== kaggle_run start: run_id={RUN_ID} args={args} ===", flush=True)
        emit_event("runner_start", args=vars(args), work=str(WORK), results=str(RES))

        LOGS.mkdir(parents=True, exist_ok=True)
        done = load_done(include_external=args.resume) if args.resume else {}
        plan = build_plan(args, done)
        save_state("planned", args, plan)

        print(f"\n=== PLAN: {len(plan)} units "
              f"({sum(1 for p in plan if p['done'])} already done, "
              f"{sum(1 for p in plan if not p['done'])} to run) ===")
        for p in plan[:50]:
            print(f"  [{'x' if p['done'] else ' '}] {p['method']:12s} {p['dataset']:14s} seed={p['seed']}")
        if len(plan) > 50:
            print(f"  ... and {len(plan) - 50} more")
        emit_event("plan_ready", units=len(plan), done=sum(1 for p in plan if p["done"]))
        if args.plan:
            save_state("plan_only", args, plan)
            return

        pending_methods = {p["method"] for p in plan if not p["done"]}
        if not args.no_setup and pending_methods:
            emit_event("setup_start")
            setup_env(pending_methods)
            clone_repos(pending_methods)
            emit_event("setup_end")
        RUNTIME_INFO = collect_runtime_info()
        emit_event("runtime_detected", **RUNTIME_INFO)

        for p in plan:
            if p["done"]:
                print(f"\n[skip] {p['method']}/{p['dataset']}/seed={p['seed']} already in registry")
                continue
            m, d, s = p["method"], p["dataset"], p["seed"]
            current = {"method": m, "dataset": d, "protocol": args.protocol, "seed": s,
                       "started_at": utc_now()}
            save_state("running", args, plan, current=current)
            emit_event("unit_start", method=m, dataset=d, protocol=args.protocol, seed=s)
            print(f"\n=== RUN {m} {d} seed={s} protocol={args.protocol} "
                  f"{'SMOKE' if args.smoke else ''} {'max-epochs=' + str(args.max_epochs) if args.max_epochs else ''} ===")
            log, status, reason, meta = None, "runtime_error", None, {}
            try:
                if m == "gps":
                    log, status, reason, meta = run_gps(d, s, args.smoke, args.max_epochs)
                elif m == "graphvit":
                    log, status, reason, meta = run_graphvit(d, s, args.smoke, args.max_epochs)
                else:
                    log, status, reason, meta = run_policy(m, d, s, args.smoke, args.max_epochs)
            except KeyboardInterrupt:
                status, reason = "interrupted", "runner interrupted; retry this unit with --resume"
                emit_event("unit_interrupted", method=m, dataset=d, seed=s)
            except Exception as ex:
                status, reason = "runtime_error", f"{type(ex).__name__}: {ex}"
                emit_event("unit_exception", method=m, dataset=d, seed=s,
                           error=reason, traceback=traceback.format_exc()[-4000:])

            repo_key = "gps" if m == "gps" else "graphvit" if m == "graphvit" else "policy"
            entry = {
                "method": m, "dataset": d, "protocol": args.protocol, "seed": s,
                "code_repo": REPOS[repo_key]["url"], "code_commit": REPOS[repo_key]["commit"],
                "config_hash": unit_config_hash(m, d, args.protocol, s, args),
                "split_hash": SPLIT_HASH.get(d), "metric_name": METRIC_NAME.get(METRIC.get(d)),
                "status": status, "failure_reason": reason,
                "log": str(log.relative_to(WORK)) if log else None,
                "smoke": args.smoke, "resource_adjusted": args.max_epochs is not None,
                "paper_seeds_provisional_note": "paper does not disclose seeds; provisional 0-3" if args.protocol == "paper-seeds-provisional" else None,
                **RUNTIME_INFO, **{k: v for k, v in meta.items() if k != "tail"},
            }
            if meta.get("tail"):
                entry["log_tail"] = meta["tail"]
            if status == "success" and log:
                metric = METRIC.get(d)
                if m == "gps":
                    epochs = parse_gps(log)
                    best, best_val = pick_best(epochs, metric, MODE[metric])
                    if best:
                        entry["best_epoch"] = best.get("test", {}).get("epoch")
                        entry["valid_metric"] = round(best_val, 6)
                        value = best["test"].get(metric)
                        entry["test_metric"] = round(value, 6) if value is not None else None
                else:
                    g = parse_generic(log)
                    tgt_key = "test_mae" if metric == "mae" else "test_auc" if metric == "auc" else "test_ap"
                    entry["best_epoch"] = g.get("best_epoch")
                    entry["valid_metric"] = g.get("best_val")
                    entry["test_metric"] = g.get(tgt_key)
                    if entry["test_metric"] is None:
                        entry["parse_note"] = "generic parser found no metric; audit raw log"
            entry = {k: v for k, v in entry.items() if v is not None}
            append_registry(entry)
            p["done"] = status in {"success", "unavailable"}
            save_state("running", args, plan, current=None,
                       extra={"last_unit": {"method": m, "dataset": d, "seed": s,
                                            "status": status, "finished_at": utc_now()}})
            emit_event("unit_end", method=m, dataset=d, protocol=args.protocol, seed=s, status=status)

        summarize()
        zip_path = package_results()
        final_status = "complete" if all(p.get("done") for p in plan) else "complete_with_failures"
        save_state(final_status, args, plan, current=None,
                   extra={"zip": str(zip_path.relative_to(WORK))})
        emit_event("runner_end", status=final_status, zip=str(zip_path.relative_to(WORK)),
                   remaining=sum(1 for p in plan if not p.get("done")))
        if not args.keep_work:
            # 只清理本 runner 创建的大目录；registry、events、state、logs 和 zip 保留。
            for d in [VENV, VENDOR]:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
        print(f"\n=== DONE. Results in {RES} (zip: {zip_path.name}) ===")
        print("Kaggle: 右侧 'Output' 下载 hod-gnn-results.zip；下一轮挂载后加 --resume")
    except KeyboardInterrupt:
        emit_event("runner_interrupted")
        try:
            save_state("interrupted", args, plan if "plan" in locals() else [],
                       current=current if "current" in locals() else None)
        finally:
            raise
    except Exception as ex:
        emit_event("runner_failed", error=f"{type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-4000:])
        if "plan" in locals():
            save_state("failed", args, plan, current=current if "current" in locals() else None,
                       extra={"error": f"{type(ex).__name__}: {ex}"})
        raise
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
