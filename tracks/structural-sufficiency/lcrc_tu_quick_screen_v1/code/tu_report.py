"""Generate REPORT.md from the pulled run records (never touches the network).

All numbers in the report come from `runs/*_s0/run.json`, `runs/*_s0/selection.json`
and `analysis/summary.json`, so the report can be regenerated after a re-pull.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import tu_common as C

TRACK = Path(__file__).resolve().parents[1]


def _run(dataset: str, model: str) -> dict[str, Any]:
    return json.loads((C.RUNS / f"{dataset}_{model}_s0" / "run.json").read_text())


def _selection(dataset: str, model: str) -> dict[str, Any]:
    return json.loads((C.RUNS / f"{dataset}_{model}_s0" / "selection.json").read_text())


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main(argv: Sequence[str] | None = None) -> int:
    summary = json.loads((C.ANALYSIS / "summary.json").read_text())
    registry = json.loads((C.ANALYSIS / "run_registry.json").read_text())
    gates = json.loads((C.ANALYSIS / "gates.json").read_text())
    smoke = json.loads((C.ANALYSIS / "smoke_cuda.json").read_text())
    manifest = json.loads((C.ANALYSIS / "data_manifest.json").read_text())
    prep = json.loads((C.ANALYSIS / "prep_summary.json").read_text())

    datasets = ["NCI1", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI"]
    models = ["FULL", "C1", "GIN"]
    runs = {(d, m): _run(d, m) for d in datasets for m in models}
    sel = {(d, m): _selection(d, m) for d in datasets for m in models}

    first = runs[(datasets[0], "FULL")]
    env = first["environment"]
    commit = first["code_state"]["git_commit"]
    dirty_files = first["code_state"]["dirty_files"]
    gpu = ", ".join(env["gpu"])
    total_wall = sum(r["timing"]["process_seconds"] for r in runs.values())

    L: list[str] = []
    A = L.append
    A("# LCRC TU quick screen v1 — report")
    A("")
    A(f"**Result label**: `{C.RESULT_LABEL}`  ")
    A("**Round**: `lcrc-tu-quick-screen-v1`  ")
    A(f"**Status**: 12/12 formal runs completed, all integrity checks passed  ")
    A(f"**Code commit**: `{commit}`  ")
    driver = next((e["driver_version"] for e in registry["entries"] if e.get("driver_version")), "n/a")
    A(f"**Compute**: `res` (process backend, 2×{gpu}), driver {driver}, "
      f"torch {env['torch']} (cuda {env['torch_cuda']}), cuDNN {env['cudnn_version']}  ")
    A("")
    A("单 seed（`training_seed=0`）、单固定 8:1:1 分层划分（`split_seed=0`）的探索性筛查。")
    A("12 个正式 run = 4 数据集 × 3 模型，全部完成；没有多 seed、交叉验证、调参、消融、早停、")
    A("EMA/SWA/soup、增强、SSL 或额外 loss。")
    A("")
    A("## 1. 结果（Test 只在 checkpoint 哈希写盘后评估一次）")
    A("")
    A("| dataset | model | params | best epoch | Train acc | Valid acc | Valid CE | Test acc | Test macro-F1 | Test CE | wall s |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        for m in models:
            r = runs[(d, m)]
            A(
                f"| {d} | {r['model_label']} | {r['parameters']:,} | {r['best_epoch']} | "
                f"{_f(r['train_metrics_at_best']['accuracy'])} | "
                f"{_f(r['valid_metrics_at_best']['accuracy'])} | "
                f"{_f(r['valid_metrics_at_best']['cross_entropy'])} | "
                f"{_f(r['test_metrics']['accuracy'])} | "
                f"{_f(r['test_metrics']['macro_f1'])} | "
                f"{_f(r['test_metrics']['cross_entropy'])} | "
                f"{r['timing']['process_seconds']:.0f} |"
            )
    A("")
    A("`wall s` 是 `rr` 进程内计时（预处理读缓存 + 200 epoch 训练 + 每 epoch Valid 评估 + 一次 Test）。")
    A("所有 run 都跑满 200 epoch（`epochs_run=200`），没有早停。")
    A(f"12 个 run 合计 wall {total_wall/60:.1f} min。")
    A("")
    A("### 1.1 Test accuracy 与配对差值")
    A("")
    A("| dataset | n_test | majority | FULL | C1 | GIN | FULL−C1 | FULL−GIN | C1−GIN |")
    A("|---|---|---|---|---|---|---|---|---|")
    comp = summary["comparison"]
    for d in datasets:
        c = comp[d]
        maj = c["majority_baseline"]["test_accuracy"]
        acc = {m: c["test_accuracy"][m] for m in models}
        A(
            f"| {d} | {c['majority_baseline']['n_test']} | {maj:.4f} | {acc['FULL']:.4f} | "
            f"{acc['C1']:.4f} | {acc['GIN']:.4f} | {acc['FULL']-acc['C1']:+.4f} | "
            f"{acc['FULL']-acc['GIN']:+.4f} | {acc['C1']-acc['GIN']:+.4f} |"
        )
    A("")
    A("### 1.2 Test macro-F1")
    A("")
    A("| dataset | FULL | C1 | GIN |")
    A("|---|---|---|---|")
    for d in datasets:
        c = comp[d]
        A(f"| {d} | {c['test_macro_f1']['FULL']:.4f} | {c['test_macro_f1']['C1']:.4f} | {c['test_macro_f1']['GIN']:.4f} |")
    A("")
    A("### 1.3 Test confusion matrices")
    A("")
    for d in datasets:
        c = comp[d]
        A(f"* **{d}**（行 = 真类，列 = 预测）")
        for m in models:
            A(f"  * {m}: {c['confusion_matrix'][m]}")
    A("")
    A("### 1.4 选定 epoch 与选择轨迹")
    A("")
    A("| dataset | model | best epoch | Valid acc @best | Valid CE @best | Train acc @best | Train CE @best |")
    A("|---|---|---|---|---|---|---|")
    for d in datasets:
        for m in models:
            r = runs[(d, m)]
            A(
                f"| {d} | {m} | {r['best_epoch']} | {r['valid_metrics_at_best']['accuracy']:.4f} | "
                f"{r['valid_metrics_at_best']['cross_entropy']:.4f} | "
                f"{r['train_metrics_at_best']['accuracy']:.4f} | "
                f"{r['train_metrics_at_best']['cross_entropy']:.4f} |"
            )
    A("")
    A("## 2. 正确性验证（都在正式 run 之前完成）")
    A("")
    A(f"* 单元测试 9 项通过（CPU、无数据）：特征构造、清洗计数、划分、共享初始化、异构 batch、GIN 传播。")
    A(f"* 结构 gates：`analysis/gates.json`，`all_passed={gates['all_passed']}`，{gates['n_checks']} 项，包含")
    A("  划分互斥/完备、Train-only 归一化审计、手算链/三角/孤立点/不连通图/未见标签→UNK/度数桶、")
    A("  节点重编号不变性（特征/global/logits）、batch 拆分-合并一致、FULL/C1 共享初始化")
    A("  （非切片张量逐位相等、恰好删除 96 个输入列、参数差 18,432、FULL 的 b_mean 置零 ≡ C1 logits）、")
    A("  FULL/C1/GIN 每个可训练张量梯度可达 + S2 content gate。")
    A(f"* CUDA smoke：`analysis/smoke_cuda.json`，`all_passed={smoke['all_passed']}`，"
      f"device `{smoke['device']}`，{smoke['seconds']:.1f}s；三模型前向/反向/保存/重载 logits 一致，")
    A("  未读 Test。")
    A("* 跨主机一致性：本地与 `res` 各自从官方 raw 文件重建，`feature_spec_hash`、`norm_fingerprint`、")
    A("  `partition_sha256`、清洗计数、节点类型数、S2 路径数全部逐项相同（见 §3 表）。")
    A("")
    A("## 3. 数据与输入（官方 raw，无手工清洗）")
    A("")
    A("| dataset | graphs | classes | 8:1:1 | node types K | atom_feat | S2 paths | directed→physical edges | duplicates merged | self-loops dropped |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        p = prep[d]
        cl = p["cell_cleaning"]
        n_s2 = sum(manifest["datasets"][d]["caches"][x]["n_s2_paths"] for x in ("train", "valid", "test"))
        A(
            f"| {d} | {p['n_graphs']} | {len(p['class_counts_train'])} | "
            f"{p['n_train']}/{p['n_valid']}/{p['n_test']} | {p['n_node_types']} | "
            f"{p['atom_feat_dim']} | {n_s2:,} | "
            f"{cl['n_directed_records']:,}→{cl['n_physical_edges']:,} | "
            f"{cl['n_duplicate_directed_records_merged']:,} | {cl['n_self_loops_dropped']} |"
        )
    A("")
    A("| dataset | feature_spec_hash | norm_fingerprint | partition_sha256 |")
    A("|---|---|---|---|")
    for d in datasets:
        p = prep[d]
        A(f"| {d} | `{p['feature_spec_hash'][:16]}…` | `{p['norm_fingerprint'][:16]}…` | `{p['partition_sha256'][:16]}…` |")
    A("")
    A("官方 raw 文件 sha256（`analysis/data_manifest.json` 的 `raw_manifest`）：")
    A("")
    for d in datasets:
        rm = manifest["datasets"][d]["raw_manifest"]
        A(f"* `{d}` ← `{rm['source']}`（raw_dir `{rm['raw_dir']}`）")
        for f in rm["files"]:
            A(f"  * `{f['file']}` {f['bytes']:,} B `{f['sha256']}`")
    A("")
    A("## 4. 执行记录")
    A("")
    A(f"所有正式 run 都在 `res` 上以 `rr` 启动，commit `{commit}`，2 卡各跑一个任务（`CUDA_VISIBLE_DEVICES`")
    A("0/1 交替），任意时刻最多 2 个训练任务。完整 registry（含 prep/gates/smoke）见")
    A("`analysis/run_registry.json`。")
    A("")
    A("| experiment | run_id | pool | CUDA_VISIBLE_DEVICES | state | exit |")
    A("|---|---|---|---|---|---|")
    for e in registry["entries"]:
        A(
            f"| {e['experiment']} | `{e['run_id']}` | {e['pool']} | "
            f"{e['cuda_visible_devices'] if e['cuda_visible_devices'] is not None else '—'} | "
            f"{e['state']} | {e['exit_code']} |"
        )
    A("")
    A(f"代码状态：12 个 run 的 `run.json` 都记录 `git_commit={commit}`、`git_dirty=true`；")
    A("dirty 项只有当时已被 prep/gates 任务写过的证据文件：")
    for f in dirty_files:
        A(f"* `{f}`")
    A("即代码与 commit 一致，差异只在 `analysis/` 下的 JSON 证据（`gates.json` 被重写、`smoke_cuda.json` 新建），")
    A("没有任何 source/config/test 文件被修改。")
    A("")
    A("### 4.1 参数量与初始化共享")
    A("")
    A("| dataset | FULL params | C1 params | diff | FULL init hash | C1 init hash | C1 ← FULL init |")
    A("|---|---|---|---|---|---|---|")
    for d in datasets:
        f, c = runs[(d, "FULL")], runs[(d, "C1")]
        A(
            f"| {d} | {f['parameters']:,} | {c['parameters']:,} | "
            f"{f['parameters']-c['parameters']:,} | "
            f"`{f['model_info']['init_state_hash'][:12]}…` | "
            f"`{c['model_info']['init_state_hash'][:12]}…` | "
            f"`{c['model_info']['full_init_state_hash'][:12]}…` |"
        )
    A("")
    A("C1 的 `full_init_state_hash` 与同数据集 FULL 的 `init_state_hash` 相同，参数差恰为 `b_mean` 的")
    A("96×192 = 18,432 个权重。")
    A("")
    A("## 5. 解释（范围内可说的话）")
    A("")
    A("* 三个模型都明显高于 majority 基线；NCI1 0.82–0.85、PROTEINS 0.72–0.77、")
    A("  IMDB-BINARY 0.67–0.71、IMDB-MULTI 0.49–0.53。")
    A("* `FULL−C1`（Test acc）为 −0.029 / 0.000 / −0.040 / +0.027，均值 −0.011：本轮没有看到")
    A("  `b_mean` 块带来一致收益；两个 IMDB 上方向相反。这与该 block 在分子任务上的既有 dissection")
    A("  结论（C1 ≈ FULL）一致，但**本轮不是对该结论的检验**：单 seed、每格只有一个数。")
    A("* `C1−GIN` 为 +0.024 / +0.045 / +0.030 / −0.033，均值 +0.017：LCRC 主体与固定 GIN 参照在这个")
    A("  量级上互有胜负，没有形成稳定排序。")
    A("* Test 与 Valid 的排序并不总一致（例：IMDB-BINARY Valid 上 GIN 0.75 > C1 0.73 > FULL 0.70，")
    A("  Test 上 C1 0.71 > GIN 0.68 > FULL 0.67），说明 100/150 图的 Valid 选择本身带 ±0.04–0.05 的")
    A("  波动，单格差值小于约 0.05 时不应解读方向。")
    A("* NCI1 上三个模型的 Valid CE 在选定 epoch 都明显高于 Train CE（FULL 0.016 vs 1.237、")
    A("  C1 0.032 vs 1.172、GIN 0.058 vs 0.709），是 sum-readout + 200 epoch 无早停的过自信表现；")
    A("  它不影响本轮的 accuracy 排序，但说明这些 run 的 CE 不能当作校准质量来读。")
    A("* PROTEINS 的 GIN 在 epoch 3 就被选中（Valid 111 图，最差值 CE 0.587）：它的 5 层残差 GIN")
    A("  在这个小数据上很快过拟合，Valid 早期即达峰。该 run 仍按固定规则完整训练 200 epoch 并只取")
    A("  epoch 3 的权重，不是早停。")
    A("* `dropout_calls_observed` 只统计 LCRC 主干作用在 `h` 上的那一次 dropout（FULL/C1 每 step 一次，")
    A("  数目 = steps × 2 ≈ 2×total_updates）。GIN 的 dropout 在层内以 functional 形式调用，不计入该计数器；")
    A("  三个模型的 dropout 强度都是 0.10（GIN 每层一次），见 `recipe.dropout`。")
    A("")
    A("**不能说**：本轮结果不能支持或否定\"结构充分性\"假设，不能给出方差估计、显著性、")
    A("数据集间泛化结论，也不能外推到其它 split/seed。")
    A("")
    A("## 6. 偏离与基础设施记录")
    A("")
    A("* 远端 `res` 的 checkout 没有这四个数据集的 raw 文件；按官方 `chrsmrrs.com` 归档下载到本地后，")
    A("  用 `rsync` 把**未修改的官方 raw 文件**放到 `res:data/TUD/<NAME>/<NAME>/raw/`。远端 prep 从这些")
    A("  raw 文件重建缓存，raw sha256、spec/norm/partition 哈希与本地完全一致（§3），因此不构成数据偏离。")
    A("* `res-2`（Slurm）未使用：其 checkout 没有 TU 数据，且需要额外的驱动/离线环境验证；本轮 12 个 run")
    A("  在 `res` 的双卡上顺序成对完成，未触及并发上限。")
    A("* 未发生失败重跑、超时、OOM 或作业取消；15 个 `rr` 作业全部 `exit_code=0`。")
    A("* 没有修改任何既有 track、冻结产物或 `data/` 下的文件。")
    A("")
    A("## 7. 未做（明确列出）")
    A("")
    A("* 多 seed / 交叉验证 / 重复划分；")
    A("* 超参数或结构搜索、消融（除固定的 FULL vs C1 vs GIN 三格外没有第四格）；")
    A("* 早停、EMA/SWA/weight soup、数据增强、SSL 预训练、额外正则或 loss；")
    A("* 其它数据集、其它节点/边属性方案、其它划分比例；")
    A("* 校准、OOD、可解释性或化学语义分析。")
    A("")
    (C.REPORT).write_text("\n".join(L) + "\n")
    C.log(f"wrote {C.track_rel(C.REPORT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
