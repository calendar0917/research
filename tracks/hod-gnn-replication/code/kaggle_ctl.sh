#!/usr/bin/env bash
# kaggle_ctl.sh — 通过 Kaggle CLI 控制 HOD-GNN 复现实验
#
# 用法:
#   ./kaggle_ctl.sh set  --method gps --dataset molhiv [--smoke] [--max-epochs 50] [--protocol official-seeds] [--resume]
#       修改 notebook 里 RUN_CMD 并 push（不重复 push 的用 --no-push）
#   ./kaggle_ctl.sh status           查看运行状态
#   ./kaggle_ctl.sh wait             轮询直到 complete/error
#   ./kaggle_ctl.sh output           下载运行输出（结果文件夹）
#   ./kaggle_ctl.sh sync             下载结果到本地 results/kaggle/
#   ./kaggle_ctl.sh smoke            快捷: gps/molhiv 冒烟
#   ./kaggle_ctl.sh check            只检查 CLI 认证和 kernel 可见性，不提交运行
#
# 环境: 需要 KAGGLE_API_TOKEN 或 ~/.kaggle/kaggle.json
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE_DIR="$HERE/kaggle"
NB="$KAGGLE_DIR/hod-gnn-run.ipynb"
META="$KAGGLE_DIR/kernel-metadata.json"
KERNEL="calendar917/hod-gnn-replication"
DEST_DIR="$HERE/results/kaggle"
PYTHON="${PYTHON:-python3}"
KAGGLE_CMD="${KAGGLE_CMD:-}"

ensure_kaggle_cli() {
  export PATH="$HOME/.local/bin:$PATH"
  if command -v kaggle >/dev/null 2>&1; then
    KAGGLE_CMD="$(command -v kaggle)"
    return 0
  fi
  # PEP 668 会禁止系统 Python 的 pip --user；使用独立 venv，避免污染仓库。
  local cli_venv="${KAGGLE_CLI_VENV:-${TMPDIR:-/tmp}/hod-gnn-kaggle-cli}"
  if [[ ! -x "$cli_venv/bin/kaggle" ]]; then
    echo "[kaggle] CLI 未安装，创建临时 venv: $cli_venv"
    "$PYTHON" -m venv "$cli_venv" || return 1
    "$cli_venv/bin/pip" install -q kaggle || return 1
  fi
  if [[ -x "$cli_venv/bin/kaggle" ]]; then
    KAGGLE_CMD="$cli_venv/bin/kaggle"
  else
    echo "[kaggle] 安装后仍找不到 kaggle CLI；请手工安装 kaggle" >&2
    return 1
  fi
}

# 认证
if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "[auth] KAGGLE_API_TOKEN 未设置；尝试 kaggle.json" >&2
fi

cmd="$1"; shift || true

set_run_cmd() {
  local extra="${RUN_CMD_EXTRA:-}"
  # 从剩余参数构造 RUN_CMD（安全地嵌入到 notebook cell）
  local runcmd="$*"
  if [ -z "$runcmd" ]; then
    runcmd="--method gps --dataset molhiv --smoke"
  fi
  "$PYTHON" - "$NB" "$runcmd" <<'EOF'
import json, sys
nb_path, run_cmd = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path))
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    joined = "".join(c["source"])
    if "RUN_CMD" in joined:
        c["source"] = [f'# 由 kaggle_ctl.sh 设置\nRUN_CMD = "{run_cmd}"\n',
                       'import subprocess, sys, os\n',
                       'os.chdir("/kaggle/working")\n',
                       '# 由父 notebook 单路转发子进程输出，避免 Kaggle 对继承 stdout 做双重采集。\n',
                       'env = dict(os.environ, PYTHONUNBUFFERED="1")\n',
                       'p = subprocess.Popen([sys.executable, "kaggle_run.py"] + RUN_CMD.split(),\n',
                       '                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n',
                       '                     text=True, bufsize=1, env=env)\n',
                       'for line in iter(p.stdout.readline, ""):\n',
                       '    print(line, end="", flush=True)\n',
                       'r = p.wait()\n',
                       'print("EXIT CODE:", r, flush=True)\n']
json.dump(nb, open(nb_path, "w"), indent=1)
print(f"[ctl] RUN_CMD = {run_cmd}")
EOF
}

push_kernel() {
  local out rc
  out="$(cd "$KAGGLE_DIR" && "$KAGGLE_CMD" kernels push 2>&1)"
  rc=$?
  printf '%s\n' "$out"
  # Kaggle CLI 2.x 可能在 API 返回错误时仍给出 0；不要让控制脚本误报成功。
  if [[ $rc -ne 0 ]] || grep -Eiq 'kernel push error|maximum batch gpu session count|error:' <<<"$out"; then
    echo "[ctl] push failed; the kernel was not submitted" >&2
    return 1
  fi
}

case "$cmd" in
  set)
    ensure_kaggle_cli
    set_run_cmd "$@"
    echo "[ctl] pushing..."
    push_kernel
    ;;
  status)
    ensure_kaggle_cli
    "$KAGGLE_CMD" kernels status "$KERNEL"
    ;;
  wait)
    ensure_kaggle_cli
    echo "[ctl] waiting for $KERNEL ..."
    for i in $(seq 1 240); do
      st="$("$KAGGLE_CMD" kernels status "$KERNEL" 2>&1)"
      echo "[ctl] $st"
      case "$st" in
        *COMPLETE*|*complete*|*Error*|*error*|*canceled*) break ;;
      esac
      sleep 60
    done
    ;;
  output)
    ensure_kaggle_cli
    mkdir -p "$KAGGLE_DIR/output"
    "$KAGGLE_CMD" kernels output "$KERNEL" -p "$KAGGLE_DIR/output"
    ;;
  sync)
    ensure_kaggle_cli
    mkdir -p "$KAGGLE_DIR/output" "$DEST_DIR"
    "$KAGGLE_CMD" kernels output "$KERNEL" -p "$KAGGLE_DIR/output"
    if [ -d "$KAGGLE_DIR/output/hod-gnn-results" ]; then
      cp -r "$KAGGLE_DIR/output/hod-gnn-results/." "$DEST_DIR/"
      echo "[ctl] synced results -> $DEST_DIR"
      ls "$DEST_DIR"
    else
      echo "[ctl] no hod-gnn-results dir in output yet"
    fi
    ;;
  smoke)
    ensure_kaggle_cli
    set_run_cmd "--method gps --dataset molhiv --smoke"
    push_kernel
    ;;
  check)
    ensure_kaggle_cli
    echo "[kaggle] CLI: $("$KAGGLE_CMD" --version 2>/dev/null || true)"
    echo "[kaggle] kernel: $KERNEL"
    "$KAGGLE_CMD" kernels status "$KERNEL"
    ;;
  *)
    echo "usage: $0 {set|status|wait|output|sync|smoke|check}"
    exit 1
    ;;
esac
