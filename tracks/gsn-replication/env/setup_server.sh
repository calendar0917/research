#!/usr/bin/env bash
# GSN 复现：GPU 服务器 conda 环境（透明命令版；等价 env/gsn-server.yaml）
# 用法: bash env/setup_server.sh [cuda-tag]    e.g. cu124 | cu121 | cu118
set -euo pipefail

CUDA_TAG="${1:-cu124}"
ENV_NAME="gsn"

echo "==> 创建 conda env: $ENV_NAME (python 3.11, graph-tool from conda-forge)"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

echo "==> graph-tool（子图同构计数，必须 conda-forge，无 pip 版）"
conda install -y -c conda-forge graph-tool

echo "==> torch $CUDA_TAG + PyG（若 GPU 型号与 cu124 不匹配，改 CUDA_TAG 或改 index-url）"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  "torch==2.5.1+$CUDA_TAG"
pip install "torch-geometric==2.6.1"

# 可选加速（torch-scatter/sparse 非必需，PyG 2.6 支持纯 torch 回退）：
# pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "==> 其余依赖"
pip install networkx numpy scipy scikit-learn ogb tqdm wandb tensorboardX

echo "==> 验证"
python - <<'PY'
import torch, torch_geometric, graph_tool, networkx, ogb
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("pyg", torch_geometric.__version__, "graph-tool", graph_tool.__version__)
print("nx", networkx.__version__)
assert torch.cuda.is_available(), "CUDA 不可用：检查驱动/nvidia-smi"
PY
echo "==> 完成。激活: conda activate $ENV_NAME"
