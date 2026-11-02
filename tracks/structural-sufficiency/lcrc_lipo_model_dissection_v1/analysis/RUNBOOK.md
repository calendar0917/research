# RUNBOOK — 精确入口与逐步命令

所有正式 GPU 训练都通过 `rr`（`skills/remote-research-runner`）提交；本地只做
CPU 冻结、评分、回溯与报告。`res` = a100-2，进程后端 pool `res-gpu0`/`res-gpu1`，
每个 run 一张 A100；最多两个并发。

```bash
# 0. 环境与远端预检（已完成，记录在此以便复现）
rr doctor res                      # 10/10 ok
rr deploy res                      # 部署当前 commit

# 1. 结构 cache（label-free，可重建；远端 49 s）
rr run res lipo-prep --pool res-gpu0 --cpus 8 \
  --result tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/cache/manifest.json -- \
  uv run python tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_prepare.py

# 2. CPU 高风险核验（62 checks）
uv run python tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_audit.py
uv run pytest -q tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/tests

# 3. Stage A（18 runs，两个 shard 各 9 个 cell）
R=tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/runs
rr run res lipo-A0 --pool res-gpu0 --gpus 1 --cpus 8 --result $R -- \
  uv run python tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_sweep.py --stage A --shard 0 --shards 2
rr run res lipo-A1 --pool res-gpu1 --gpus 1 --cpus 8 --result $R -- \
  uv run python tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_sweep.py --stage A --shard 1 --shards 2

# 4. 拉回 + stage 冻结（仅 Valid）+ 一次性 Test 评分（CPU）
rr pull lipo-A0 && rr pull lipo-A1
rsync -a .rr/pulled/<run_id>/result/$R/ $R/
uv run python .../code/dissect_freeze.py --stage A
uv run python .../code/dissect_score.py --freeze tracks/.../analysis/freeze/A_r122.json

# 5. Stage B / Stage C 同上（--stage B / C，21 / 15 runs）
# 6. COMPACT 选择（只读 Valid，脚本拒绝任何 Test 产物）
uv run python .../code/dissect_compact.py
#    -> analysis/compact_selection.json；提交后再 deploy，才允许跨划分 run
# 7. 跨划分 123/124（FULL、COMPACT、B5，去重，最多 18 runs）
rr run res lipo-X0 --pool res-gpu0 --gpus 1 --cpus 8 --result $R -- \
  uv run python .../code/dissect_sweep.py --stage cross --shard 0 --shards 2
rr run res lipo-X1 --pool res-gpu1 --gpus 1 --cpus 8 --result $R -- \
  uv run python .../code/dissect_sweep.py --stage cross --shard 1 --shards 2
uv run python .../code/dissect_freeze.py --stage cross
uv run python .../code/dissect_score.py --freeze tracks/.../analysis/freeze/cross_r123.json
uv run python .../code/dissect_score.py --freeze tracks/.../analysis/freeze/cross_r124.json

# 8. 表与回溯
uv run python .../code/dissect_tables.py            # + scaffold bootstrap
uv run python .../code/dissect_render.py            # markdown 片段
uv run python .../code/dissect_intervene.py         # 固定权重干预 + LOO 路径干预 + 图
```

恢复语义：

* `dissect_sweep.py` 跳过 `runs/<run>/train.json` 中 `status=="completed"` 且
  `device` 以 `cuda` 开头的 cell；中断后重跑同一 shard 即可，不会重复购买。
* `dissect_score.py` 对同一 checkpoint 只评分一次；重复调用会拒绝（除 `--repair`，
  且会写入 `analysis/test_access_log.json`）。
* `dissect_freeze.py` 在发现任何不一致时以非零退出，且不会写 “ok” 的 freeze。

已知工程细节：

* 远端 checkout 位于 `/home/hxy/cy/research`（独立文件系统）。`runs/` 与 `cache/`
  被 track 级 `.gitignore` 排除，避免远端 run 产物把远端 worktree 弄脏而阻塞后续
  deploy（本轮曾因此修复过一次）。
* 本机 122 的 420 个 Valid row 与 cache 顺序一致；`collate` 的 `bond_index`/
  `path_end` 是**分子内局部索引**，单分子子 cache 直接复用即可。
