# 环境管理：conda 为主（与组内工作流一致）

服务器（导师账号 hxy，无 sudo）决策：直接用导师的 conda 二进制创建 py3.12 命名环境，环境放在默认 `~/enter/envs/` 与导师/组内环境共存（用户明确认为共享无妨）；只需保证命名清晰、不改 base。理由：组内惯例是 conda，学习成本低；本次项目依赖（torch 2.5.1 + torch-geometric）有 PyPI 轮子，`pip install torch --index-url https://download.pytorch.org/whl/cu124` 即可的 CUDA 变体；conda-forge 处理有原生二进制依赖的论文时也更顺。uv 仍保留为备选/纯 pip 项目的选项（理论：解释器/包/位置三层，uv 自管解释器+项目 .venv；conda 环境自带+envs_dirs）。

**Evidence**：用户确认接受与导师环境共置于 `~/enter/envs`；所建环境命名 `research-toolchain` 以区分既有 4 个环境。
