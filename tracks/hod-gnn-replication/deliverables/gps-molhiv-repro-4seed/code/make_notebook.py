"""生成 kaggle/hod-gnn-run.ipynb（内嵌 kaggle_run.py）。"""
import json
from pathlib import Path

script = open('code/kaggle_run.py').read()

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# HOD-GNN 基线复现审计 · Kaggle 执行器\n',
            '\n',
            '## 用法\n',
            '1. 依次运行下方 cell。\n',
            '2. 修改 RUN_CMD 选择要跑的方法/数据集。\n',
            '3. 每轮会话结束：右侧 Output 下载整个 `hod-gnn-results` 文件夹。\n',
            '4. 下次运行时把上次下载的文件夹上传为 dataset 挂载到本 notebook，并加 `--resume`，会自动跳过已完成单元。\n',
            '5. `run_state.json` 记录当前单元；`events.jsonl` 记录阶段/心跳/退出码；`logs/` 保留每个单元的完整原始日志。\n',
            '\n',
            '## 常用命令\n',
            '```\n',
            '--plan  \n',
            '--method gps --dataset molhiv --smoke\n',
            '--method gps --dataset molhiv\n',
            '--method gps --dataset zinc\n',
            '--method all --dataset all --resume\n',
            '--method gps --dataset molhiv --smoke --no-setup --resume\n',
            '```\n',
            '\n',
            '## 注意\n',
            '- 每会话 GPU 约 9 小时上限；全矩阵需要多轮，用 `--resume` 衔接。\n',
            '- `--resume` 只跳过同一配置下的 `success`/`unavailable`；失败、OOM、中断或改了 smoke/epoch 的单元会自动重试。\n',
            '- 一般断点粒度是“方法×数据集×seed”；GPS/ZINC 例外，会把官方 checkpoint 写到 `hod-gnn-results/checkpoints/` 并在挂载后按 epoch 续训。\n',
            '- 默认每 60 秒输出一次心跳和关键 metric 行；需要完整透传时加 `--live-log`。\n',
            '- moltox21(GPS) 与 peptides(policy-learn) 官方仓无配置，脚本会自动标记 unavailable。\n',
            '- 正式实验不要开 `--smoke`。\n',
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': ['%%writefile kaggle_run.py\n' + script]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 修改这里：选择要跑的单元\n',
            'import subprocess, sys, os\n',
            'os.chdir("/kaggle/working")\n',
            'RUN_CMD = "--method gps --dataset molhiv --smoke"\n',
            '# 由父 notebook 单路转发子进程输出，避免 Kaggle 对继承 stdout 做双重采集。\n',
            'env = dict(os.environ, PYTHONUNBUFFERED="1")\n',
            'p = subprocess.Popen([sys.executable, "kaggle_run.py"] + RUN_CMD.split(),\n',
            '                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n',
            '                     text=True, bufsize=1, env=env)\n',
            'for line in iter(p.stdout.readline, ""):\n',
            '    print(line, end="", flush=True)\n',
            'r = p.wait()\n',
            'print("EXIT CODE:", r, flush=True)\n',
        ]
    },
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 4,
}
output_path = Path('kaggle') / 'hod-gnn-run.ipynb'
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    f.write(json.dumps(nb, indent=1))
print(f'written {output_path}')
