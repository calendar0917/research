#!/usr/bin/env bash
# GSN 复现：GPU 服务器 conda 环境（透明命令版；等价 env/gsn-server.yaml）
# 用法:
#   bash env/setup_server.sh            # 默认 cu124
#   bash env/setup_server.sh cu118      # driver 较旧时
set -euo pipefail

CUDA_TAG="${1:-cu124}"
ENV_NAME="gsn"

# conda 函数加载（非交互 bash 必需）
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda 不在 PATH。请先安装/初始化 conda（或 source conda.sh）" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

echo "==> 创建 conda env: $ENV_NAME (python 3.11)"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

echo "==> graph-tool（子图同构计数，必须 conda-forge，无 pip 版）"
echo "    同时让 conda 管 numpy/scipy/sklearn（避免 pip 覆盖与 graph-tool 冲突）"
conda install -y -c conda-forge graph-tool numpy scipy scikit-learn

echo "==> torch $CUDA_TAG + PyG（若 GPU 驱动不兼容，换 cu121/cu118）"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  "torch==2.5.1+$CUDA_TAG"
pip install "torch-geometric==2.6.1"

# 可选加速（torch-scatter/sparse 非必需，PyG 2.6 支持纯 torch 回退）：
# pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "==> 其余依赖（networkx/ogb/tqdm/wandb 等）"
pip install networkx ogb tqdm wandb tensorboardX

echo "==> libgomp 冲突修复（torch 自带旧 libgomp 会遮蔽 graph-tool 需要的 GOMP_5.0）"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/gsn-libgomp.sh" <<'EOF'
# torch 自带 libgomp（仅 GOMP<=4.5 符号）会先被加载，导致 conda-forge graph-tool
# 报 'version GOMP_5.0 not found'。预加载 conda 的 libgomp 使全部库共用同一 runtime。
if [ -f "$CONDA_PREFIX/lib/libgomp.so.1" ]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+ $LD_PRELOAD}"
fi
EOF
echo "   已写入 \$CONDA_PREFIX/etc/conda/activate.d/gsn-libgomp.sh（重新 activate 生效）"

if [ -f "$CONDA_PREFIX/lib/libgomp.so.1" ]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+ $LD_PRELOAD}"
fi

echo "==> 验证"
python - <<'PY'
import torch, torch_geometric, graph_tool, networkx, ogb
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("pyg", torch_geometric.__version__, "graph-tool", graph_tool.__version__)
print("nx", networkx.__version__)
assert torch.cuda.is_available(), "CUDA 不可用：检查驱动/nvidia-smi"
PY
echo "==> 完成。激活: conda activate $ENV_NAME"
