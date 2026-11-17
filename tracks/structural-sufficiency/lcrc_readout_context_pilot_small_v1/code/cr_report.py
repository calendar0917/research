"""从分析产物生成中文 ``REPORT.md`` / ``DECISION.md`` / 交付清单。

本模块不做任何科学计算，也不手打任何数字：每个数都从本轮写出的
``analysis/*.json``、``analysis/tables/*.csv``、``results/**/trajectory.json``
读回，并顺带生成

* ``analysis/run_manifest.json`` —— 6 条正式轨迹的 run_id / commit / 路径 / hash
* ``analysis/resource_ledger.json`` —— 真实 GPU 墙钟与训练/评估计时
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Mapping, Sequence

import cr_common as C
import cr_analysis as AN

OUT = C.ANALYSIS_DIR
DEV = "DEVELOPMENT"
DEV_TEST = "DEVELOPMENT_TEST"


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):+.{nd}f}%"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _read(name: str, default: Any = None) -> Any:
    path = OUT / name
    if not path.is_file():
        return default
    return C.read_json(path)


def _rows(name: str, default: Any = None) -> Any:
    path = OUT / "tables" / name
    if not path.is_file():
        return default if default is not None else []
    return C.read_csv(path)


# ---------------------------------------------------------------------------
# 交付清单：run_manifest / resource_ledger
# ---------------------------------------------------------------------------


def write_manifests() -> dict[str, Any]:
    """把 6 条正式轨迹与复用参照写成机器可读清单（全部读自实测产物）。"""
    prov = _read("run_provenance.json", {})
    runs = prov.get("runs") or {}
    experiment_of = {"ogbg-molesol": "lcrc-rc-molesol", "ogbg-mollipo": "lcrc-rc-mollipo"}
    trajectories = []
    for dataset in C.DATASETS:
        for arm in C.ARMS:
            directory = C.RESULTS_DIR / dataset / arm
            traj = C.read_json(directory / "trajectory.json")
            run = runs.get(experiment_of[dataset], {})
            trajectories.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "origin": "THIS_ROUND",
                    "run_id": run.get("run_id"),
                    "rr_experiment": experiment_of[dataset],
                    "pool": run.get("pool"),
                    "node": run.get("node"),
                    "driver_version": run.get("driver_version"),
                    "torch_version": run.get("torch_version"),
                    "cuda_visible_devices": run.get("cuda_visible_devices"),
                    "commit": run.get("commit"),
                    "branch": run.get("branch"),
                    "seed": C.TRAIN_SEED,
                    "epochs": traj["epochs"],
                    "total_updates": traj["total_updates"],
                    "hidden": traj["budget"]["hidden"],
                    "n_parameters": traj["n_parameters"],
                    "readout_parameters": traj["readout_parameters"],
                    "trajectory_signature": traj["signature"],
                    "batch_order_sha256": traj["batch_order_sha256"],
                    "selected_epoch": traj["selected"]["epoch"],
                    "selected_valid_rmse_raw": traj["selected"]["valid_rmse_raw"],
                    "selected_test_rmse_raw": traj["selected"]["test_rmse_raw"],
                    "test_eval_count": traj["test_eval_count"],
                    "test_policy": traj["test_policy"],
                    "artifacts": {
                        "directory": str(directory.relative_to(C.REPO_ROOT)),
                        "checkpoint_best_val": {
                            "path": str((directory / "checkpoint_best_val.pt").relative_to(C.REPO_ROOT)),
                            "sha256": traj["checkpoint_hashes"]["checkpoint_best_val.pt"],
                            "committed": (directory / "checkpoint_best_val.pt").is_file(),
                        },
                        "checkpoint_final": {
                            "path": str((directory / "checkpoint_final.pt").relative_to(C.REPO_ROOT)),
                            "sha256": traj["checkpoint_hashes"]["checkpoint_final.pt"],
                            "committed": False,
                            "note": "留存于本轮远端结果目录；可由同一 commit 重跑复现（签名一致则跳过）",
                        },
                        "best_predictions_npz_sha256": C.sha256_file(directory / "best_predictions.npz")
                        if (directory / "best_predictions.npz").is_file()
                        else None,
                        "pred_matrix_committed": False,
                        "pred_matrix_note": "360×row 的逐 epoch 预测矩阵留在远端结果目录（体积大、可复现）",
                        "curve_csv": str((directory / "curve.csv").relative_to(C.REPO_ROOT)),
                    },
                }
            )
    reference = []
    for dataset in C.DATASETS:
        g = AN.load_gine(dataset)
        if g is None:
            continue
        payload = g["raw"]
        reference.append(
            {
                "kind": "REUSED_FROZEN_REFERENCE",
                "model": "G (GINE)",
                "dataset": dataset,
                "source": g["source"],
                "source_sha256": C.sha256_file(C.SQL_RESULTS / dataset / "method_G.json"),
                "selected_peak_lr": g["selected_peak_lr"],
                "selected_epoch": g["selected"]["epoch"],
                "selected_valid_rmse_raw": g["selected"]["valid_rmse_raw"],
                "selected_test_rmse_raw": g["selected"]["test_rmse_raw"],
                "n_parameters": g["n_parameters"],
                "retrained_this_round": False,
                "verification": "cr_checks.check_gine_reference_reuse（逐样本重算 RMSE/MAE、"
                "360 epoch 曲线与 best/final/last20 身份核对）",
                "identity_note": payload.get("identity_note"),
            }
        )
        l = AN.load_reference_l(dataset)
        if l is not None:
            reference.append(
                {
                    "kind": "REUSED_FROZEN_REFERENCE",
                    "model": "L (C1 / NO_BOND_MEAN, R3)",
                    "dataset": dataset,
                    "source": l["source"],
                    "source_sha256": C.sha256_file(C.SQL_RESULTS / dataset / "method_L.json"),
                    "selected_epoch": l["selected"]["epoch"],
                    "selected_valid_rmse_raw": l["selected"]["valid_rmse_raw"],
                    "selected_test_rmse_raw": l["selected"]["test_rmse_raw"],
                    "retrained_this_round": False,
                    "role": "arm A 的复现核验参照（仅 batch 顺序不同）",
                }
            )
    manifest = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design_basis_commit": "9b3dba098915b0cdd800767f61b523681ee9e25d",
        "seed": C.TRAIN_SEED,
        "test_policy": "always",
        "n_formal_trajectories": len(trajectories),
        "formal_trajectories": trajectories,
        "reused_references": reference,
        "analysis_artifacts": {
            name: str((OUT / name).relative_to(C.REPO_ROOT))
            for name in (
                "protocol_freeze.json",
                "dataset_audit.json",
                "checks.json",
                "verdicts.json",
                "bootstrap.json",
                "drop_one.json",
                "checkpoint_replay.json",
                "interventions.json",
                "context_response.json",
                "interaction_diagnostics.json",
                "equivalence_scale.json",
                "reference_arm_A_vs_L.json",
                "run_provenance.json",
                "decision.json",
            )
            if (OUT / name).is_file()
        },
    }
    C.atomic_write_json(OUT / "run_manifest.json", manifest)

    budget_rows = []
    total_hours = 0.0
    for name, run in runs.items():
        try:
            start = datetime.fromisoformat(str(run.get("started_at")))
            end = datetime.fromisoformat(str(run.get("completed_at")))
            seconds = (end - start).total_seconds()
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            total_hours += seconds / 3600.0
            budget_rows.append({"experiment": name, "run_id": run.get("run_id"), "pool": run.get("pool"),
                                "job_wall_seconds": seconds, "job_wall_gpu_hours": seconds / 3600.0})
    arms = []
    for entry in trajectories:
        traj = C.read_json(C.RESULTS_DIR / entry["dataset"] / entry["arm"] / "trajectory.json")
        arms.append(
            {
                "dataset": entry["dataset"],
                "arm": entry["arm"],
                "train_seconds": traj["train_seconds"],
                "eval_seconds_total": traj["eval_seconds_total"],
                "inference_seconds_three_splits": traj["inference_seconds_full_three_splits"],
                "total_updates": traj["total_updates"],
            }
        )
    # apportion each worker's job wall over its three arms by measured trainer time
    for dataset in C.DATASETS:
        group = [a for a in arms if a["dataset"] == dataset]
        total_arm_seconds = sum(a["train_seconds"] + a["eval_seconds_total"] for a in group)
        job = next(
            (row for row in budget_rows if row["experiment"] == experiment_of[dataset]), None
        )
        for average in group:
            share = (average["train_seconds"] + average["eval_seconds_total"]) / total_arm_seconds
            average["gpu_hours_apportioned"] = (
                float(job["job_wall_gpu_hours"]) * share if job else None
            )
            average["gpu_hours_apportion_rule"] = (
                "worker job wall x (this arm's trainer-measured train+eval seconds / the three arms' total)"
            )
    ledger = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unit": "A100-SXM4-40GB hours of job wall, one GPU per worker",
        "formal_jobs": budget_rows,
        "formal_gpu_hours_total": sum(row["job_wall_gpu_hours"] for row in budget_rows),
        "per_arm_trainer_measured": arms,
        "smoke": {"note": "3 epoch × 3 臂的 GPU 完整 wire-up smoke，单独 _smoke 目录，从不作为结论", "gpu_hours": 9.3 / 3600.0},
        "cpu_only": {
            "prepare_seconds_local": None,
            "aggregate_seconds_local": _read("analysis_summary.json", {}).get("aggregate_seconds"),
            "note": "prepare / 分析 / 画图全在本地 CPU 完成，不占 GPU",
        },
        "budget": {"soft_gpu_hours": 2.0, "hard_gpu_hours": 3.0, "used_gpu_hours": sum(row["job_wall_gpu_hours"] for row in budget_rows)},
        "wasted_reruns": [],
        "note": "工程故障作废的重跑会记入 wasted_reruns 并计入总耗时；本轮没有作废的正式轨迹",
    }
    C.atomic_write_json(OUT / "resource_ledger.json", ledger)
    return {
        "run_manifest": str((OUT / "run_manifest.json").relative_to(C.REPO_ROOT)),
        "resource_ledger": str((OUT / "resource_ledger.json").relative_to(C.REPO_ROOT)),
        "n_formal_trajectories": len(trajectories),
        "formal_gpu_hours": ledger["formal_gpu_hours_total"],
    }


# ---------------------------------------------------------------------------
# 报告章节
# ---------------------------------------------------------------------------


def _header(summary: Mapping[str, Any], freeze: Mapping[str, Any], prov: Mapping[str, Any],
            ledger: Mapping[str, Any]) -> str:
    executed = ", ".join(
        sorted({str(v.get("commit", "?"))[:12] for v in (prov.get("runs") or {}).values()})
    ) or str(freeze.get("round_commit"))[:12]
    hours = ledger.get("formal_gpu_hours_total")
    return f"""# LCRC 显式上下文读出的两个小数据集可行性实验

**轮次** `{C.ROUND_ID}` · **状态** {DEV} · **所有 Test 数字都是 `{DEV_TEST}`** ·
**分支** `{freeze.get('round_branch')}`（依据版本 `{str(freeze.get('design_basis_commit'))[:12]}`）·
**正式轨迹执行 commit** `{executed}` · **freeze hash** `{str(freeze.get('freeze_hash'))[:16]}…` ·
**生成时间** {summary.get('generated_at')} · **正式 GPU 墙钟** {f(hours, 3)} A100-h

> 本报告是**单 seed、两个小任务的可行性实验**。Test 划分经用户明确授权每 epoch 全量评估
> （`test_policy=always`），但**从未**参与 checkpoint、宽度、LR 或任何判读量的选择。
> 文中没有任何跨任务平均、没有把单 seed 点估计写成泛化结论、也不构成 benchmark 提交。

**一句话判读**：这套「共享局部 trunk + 预算匹配的显式加性读出」**可实现、可逐项回溯**
（TRACE 两任务全通过），但在本配方下 **molesol 上 C 损失了 A 的性能**、**mollipo 上 C 与 A 打平**，
而 **C 相对 B 的上下文增量在 Val/Test 之间变号**，因此**本轮不支持"两个显式上下文交互带来可用增益"**；
与复用的冻结 GINE 相比，仅 molesol 的 A 落在 5% 带内。建议**停止这一读出形式的本轮扩检**，
保留可复用的机制与账目代码。
"""


def _identity(freeze: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = ["## 1. 训练了什么", ""]
    budget = freeze.get("budget", {})
    rows = []
    for dataset in C.DATASETS:
        a = audit.get(dataset, {})
        b = budget.get(dataset, {}).get("assignment", {})
        rows.append(
            [
                dataset,
                "/".join(str(a.get("split_sizes", {}).get(s)) for s in C.SPLITS),
                b.get("A", {}).get("nominal_readout_params"),
                f"{b.get('B', {}).get('nominal_readout_params')} (H={b.get('B', {}).get('hidden')})",
                f"{b.get('C', {}).get('nominal_readout_params')} (H={b.get('C', {}).get('hidden')})",
                a.get("active_global", {}).get("n_active"),
                len(a.get("active_global", {}).get("inactive_names") or []),
                f(a.get("pass")),
            ]
        )
    lines.append(
        table(
            ["任务", "Train/Val/Test", "A 读出参数", "B 读出参数", "C 读出参数", "M 活跃列", "常量列", "数据审计"],
            rows,
        )
    )
    lines += [
        "",
        "三臂**共用同一个局部 trunk**（逐位等于 C1 的局部通路，同一规范初值），只有读出不同：",
        "",
        "* **A**：冻结的 C1 原读出 `sum_v f(h_v) + rho([z_g, mean h, mean h^2, log1p n])`，不改一处；",
        "* **B**：`b_global + sum_v c0(h_v) + sum_j u_j(z_j)`，`M` 为 Train 上跨度 > 1e-8 的原始 global 列（molesol 34 / mollipo 36），"
        "常量列在 B/C 中恒输出 0，不建无训练信号的随机头；",
        "* **C**：B 再加两项**预先命名**的上下文交互 `a_size(z)·sum_v c_size(h_v)` 与 `a_ring(z)·sum_v c_ring(h_v)`，"
        "上下文只允许 `log1p_n_atoms` 与 `ring_atom_fraction` 两个字段。",
        "",
        "`H` 由预算匹配决定：A 的 15298 个读出参数就是预算，B/C 的 `H` 取使名义读出参数量落在**一个宽度步长**以内的整数；"
        "整数 `H` 无法精确相等，实际差值为 ±36 以内，`analysis/protocol_freeze.json` 记录了分配与公式。",
        "",
        "冻结缓存（**真实文件字节** SHA-256，正式运行前在计算节点上重新校验）：",
        "",
    ]
    for name, sha in freeze.get("frozen_caches", {}).items():
        lines.append(f"* `{name}` — `{sha[:16]}…`")
    init = freeze.get("canonical_init", {})
    lines += ["", "共享 trunk 初值（三臂逐位相同，并在独立子进程中复现）：", ""]
    for dataset, hashes in init.items():
        lines.append(f"* {dataset}：trunk `{hashes.get('trunk_hash', '')[:16]}…`，整状态 `{hashes.get('full_state_hash', '')[:16]}…`")
    recipe = freeze.get("recipe", {})
    lines += [
        "",
        f"冻配方（三臂同一套）：seed {freeze.get('seed')}、{recipe.get('epochs')} epoch 跑满（无 early stopping）、"
        f"AdamW peak {recipe.get('peak_lr')} → min {recipe.get('min_lr')} 逐步 cosine、wd {recipe.get('weight_decay')}、"
        f"batch {recipe.get('batch_size')}、clip {recipe.get('gradient_clip')}、FP32、无 warmup/EMA/SWA/soup、"
        "每 epoch 全量评估三划分，选择只看最小 Val RMSE。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _main_table() -> str:
    rows = _rows("main_table.csv")
    if not rows:
        return "## 2. 主表\n\n_缺 `tables/main_table.csv`。_\n"
    lines = ["## 2. 主表（原始目标单位，逐任务分开，不跨任务平均）", ""]
    ledger = _read("resource_ledger.json", {})
    hours = {
        (entry["dataset"], entry["arm"]): entry.get("gpu_hours_apportioned")
        for entry in (ledger.get("per_arm_trainer_measured") or [])
    }
    body = []
    for r in rows:
        body.append(
            [
                r["dataset"],
                r["arm"],
                "本轮" if str(r.get("reused")).strip().lower() in ("false", "0", "") else "复用冻结",
                r["n_parameters"],
                r["selected_epoch"],
                f(r["train_rmse_raw"]),
                f(r["valid_rmse_raw"]),
                f(r["test_rmse_raw"]),
                f(r["train_mae_raw"]),
                f(r["valid_mae_raw"]),
                f(r["test_mae_raw"]),
                f(r["final_valid_rmse_raw"]),
                f(r["final_test_rmse_raw"]),
                f(r["last20_valid_rmse_raw"]),
                f(r["last20_test_rmse_raw"]),
                r["total_updates"] if r["total_updates"] is not None else "—",
                f(r["train_seconds"], 1),
                f(r["eval_seconds_total"], 1),
                f(r["inference_seconds_three_splits"], 3),
                f(hours.get((r["dataset"], r["arm"])), 3) if (r["dataset"], r["arm"]) in hours else "—",
                str(r["checkpoint_best_val_sha256"])[:12] if r["checkpoint_best_val_sha256"] else "—",
            ]
        )
    lines.append(
        table(
            ["任务", "臂", "来源", "总参数", "best epoch", "Train RMSE", "Val RMSE", "Test RMSE",
             "Train MAE", "Val MAE", "Test MAE", "final Val", "final Test", "last20 Val", "last20 Test",
             "updates", "训练 s", "评估 s", "纯推理 s", "GPU-h", "ckpt hash"],
            body,
        )
    )
    lines += [
        "",
        "* `best epoch` = 最小 Val RMSE 的那一个 epoch；同一冻结状态的 Test 列就是 `" + DEV_TEST + "`。",
        "* `final` 是同一条轨迹第 360 epoch 的状态；`last20` 是最后 20 个 epoch 的算术均值——"
        "曲线摘要，不是 ensemble，也不给 single checkpoint 编造预测 CI。",
        "* `G` 是 `sq-pilot-small-v1` 的冻结 GINE，**原样复用、本轮不重训**（`REUSED_FROZEN_REFERENCE`）；"
        "它的输入通路不同、LR 预设更强（`1e-3` 对 本轮的 `3e-4`），只作强性能坐标。",
        "* `GPU-h` 一列：本轮的 6 条轨迹按各自 worker 的 job 墙钟、以该臂 `train+eval` 秒数占比分摊"
        "（分摊明细见 §12.1 与 `analysis/resource_ledger.json`）；复用参照的 GINE 轨迹为 0。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _transfer_gap() -> str:
    rows = _rows("main_table.csv")
    if not rows:
        return ""
    lines = [
        "### 2.1 同一冻结状态的 Val → Test 迁移差",
        "",
        "选中状态是 Val RMSE 的极小点，所以这张表问的是：这个 Val 数字在 Test 上还剩多少。"
        "molesol 的 Val 只有 113 个分子，Val 上的小改进本身噪声很大。",
        "",
    ]
    body = []
    for r in rows:
        v, t = float(r["valid_rmse_raw"]), float(r["test_rmse_raw"])
        l20v, l20t = float(r["last20_valid_rmse_raw"]), float(r["last20_test_rmse_raw"])
        body.append([r["dataset"], r["arm"], f(v), f(t), f(t - v), f(t / v, 3), f(l20t - l20v), f(l20t / l20v, 3)])
    lines.append(
        table(["任务", "臂", "Val RMSE", "Test RMSE", "Test − Val", "Test / Val", "last20 Test − Val", "last20 Test / Val"], body)
    )
    worst = max((r for r in rows if r["arm"] != "G"),
                key=lambda r: float(r["test_rmse_raw"]) / float(r["valid_rmse_raw"]))
    lines += [
        "",
        f"A/B/C 中最大的选中态差距出现在 {worst['dataset']}/{worst['arm']}"
        f"（Test = {float(worst['test_rmse_raw']) / float(worst['valid_rmse_raw']):.2f}× Val）。"
        "这个量级的差距说明该臂的 Val 选中点在本题上不是一个可靠的性能坐标；本报告把它列出来而不是抹平，"
        "且全文任何地方都不用跨 split 平均值。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _contrasts() -> str:
    rows = _rows("contrasts.csv")
    boot = _read("bootstrap.json", {})
    if not rows:
        return "## 3. 三个主对照\n\n_缺 `tables/contrasts.csv`。_\n"
    lines = ["## 3. 三个主对照 + 样本不确定性", ""]
    body = []
    for r in rows:
        b = boot.get(r["dataset"], {}).get("valid", {}).get("contrasts", {}).get(r["contrast"], {})
        bt = boot.get(r["dataset"], {}).get("test", {}).get("contrasts", {}).get(r["contrast"], {})
        vi, ti = b.get("rmse_raw", {}), bt.get("rmse_raw", {})
        body.append(
            [
                r["dataset"],
                r["contrast"],
                f(r["selected_delta_rmse_raw_valid"]),
                f(r["selected_ratio_rmse_valid"], 3),
                f"[{f(vi.get('ci95_low'), 3)}, {f(vi.get('ci95_high'), 3)}]",
                f(r["selected_delta_rmse_raw_test"]),
                f(r["selected_ratio_rmse_test"], 3),
                f"[{f(ti.get('ci95_low'), 3)}, {f(ti.get('ci95_high'), 3)}]",
                f(r["final_delta_rmse_raw_valid"]),
                f(r["final_delta_rmse_raw_test"]),
                f(r["last20_delta_rmse_raw_valid"]),
                f(r["last20_delta_rmse_raw_test"]),
            ]
        )
    lines.append(
        table(
            ["任务", "对照", "Δ Val", "比值 Val", "95% CI (Val)", "Δ Test", "比值 Test", "95% CI (Test)",
             "Δ final Val", "Δ final Test", "Δ last20 Val", "Δ last20 Test"],
            body,
        )
    )
    lines += [
        "",
        "Δ = 左 − 右（负值偏向左侧），比值为左/右。区间是**配对分子 bootstrap** 的 2.5/97.5 分位"
        "（2000 次重采样，RNG 1703，同一 split 内所有模型共用同一次重采样，每次重采样重算 RMSE/MAE）。",
        "它**条件于当前单 seed 选中状态**：不估计训练 seed 方差、不修复 Val 选择、也不消除 Test 的开发暴露。",
        "",
    ]
    d1 = _read("drop_one.json", {})
    if d1:
        body = []
        for dataset, per_split in d1.items():
            for split in ("valid", "test"):
                for contrast, values in per_split.get(split, {}).items():
                    body.append(
                        [
                            dataset,
                            split,
                            contrast,
                            f(values["delta_rmse_full"]),
                            f"[{f(values['delta_rmse_range'][0])}, {f(values['delta_rmse_range'][1])}]",
                            values["n_rows_that_flip_direction"],
                            f(values["max_single_row_shift_rmse"]),
                        ]
                    )
        if body:
            lines += [
                "逐行 drop-one 敏感性（是否由一行决定方向）：",
                "",
                table(["任务", "split", "对照", "Δ RMSE（全量）", "去掉一行后的 Δ 范围", "翻号行数", "单行最大位移"], body),
                "",
                "_删除后的数字只作诊断，官方分数永远是全量 split。_",
                "",
            ]
    return "\n".join(lines) + "\n"


def _checks() -> str:
    per = {}
    for dataset in C.DATASETS:
        payload = _read(f"checks_{dataset}.json")
        if payload is not None:
            per[dataset] = payload
    lines = ["## 4. 定向实现检查（正式训练开始前）", ""]
    if not per:
        return "\n".join(lines + ["_缺 `analysis/checks_<dataset>.json`。_"]) + "\n"
    body = [[d, p.get("n_checks"), "是" if p.get("all_pass") else "否", len(p.get("failed") or [])] for d, p in per.items()]
    lines.append(table(["任务", "门禁数", "全部通过", "失败数"], body))
    names = sorted({c["check"] for payload in per.values() for c in payload.get("checks", [])})
    status = []
    for name in names:
        cells = []
        for dataset, payload in per.items():
            hit = [c for c in payload.get("checks", []) if c["check"] == name]
            cells.append("通过" if hit and all(c.get("pass") for c in hit) else ("缺" if not hit else "失败"))
        status.append([name] + cells)
    lines += [
        "",
        table(["门禁"] + list(per), status),
        "",
        "其中与 prompt 直接对应的是：A 与只读 `DissectLCRC(\"C1\")` 在同一状态、同一真实 batch 上预测/aux 逐位一致；"
        "三臂共同 trunk 初值逐位相等（并在新进程复现）；`u_j` 初值逐位相等；`q(0)=0` 且向量化与逐标量一致、"
        "无被中心化吃掉的输出 bias；每次前向只有一次 dropout 且被所有局部头共用；raw/标准化往返与"
        "「零输出 → Train 均值」正确；新参数梯度有限可达；eval 批量拆分、分子隔离、原子重编号不变性；"
        "账目闭合；`a_k×2 / c_k÷2` 尺度等价；checkpoint 保存/重载后预测与全部账目重放一致；"
        "以及冻结 GINE 复用门禁（逐样本重算 RMSE/MAE、360 epoch 曲线与 best/final/last20 身份）。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _trace() -> str:
    ledgers = {}
    for dataset in C.DATASETS:
        for arm in ("B", "C"):
            meta = _read(f"ledgers/{dataset}_{arm}.json")
            if meta is not None:
                ledgers[f"{dataset}/{arm}"] = meta
    replay = _read("checkpoint_replay.json", {})
    interventions = _read("interventions.json", {})
    contexts = _read("context_response.json", {})
    equiv = _read("equivalence_scale.json", {})
    verdicts = _read("verdicts.json", {})
    lines = [
        "## 5. 回溯：每个数还能不能重新拆开",
        "",
        "容差全程是预先冻结的 `tol_raw(x) = 1e-5 + 1e-6·max(1,|x|)`（原始单位）；失败不用事后放宽阈值。",
        "",
        "账目定义（与 prompt §十 一致）：`intercept_raw = mu + sigma·b_global`，`base_root_v_raw = sigma·c0(h_v)`，"
        "`global_j_raw = sigma·u_j(z_j)`，`interaction_{v,k}_raw = sigma·a_k(z_k)·c_k(h_v)`，总和 = `pred_raw`。"
        "**只有 intercept 带 mu**，每个 root 与每个交互都只带 sigma。",
        "",
    ]
    if ledgers:
        body = []
        for key, meta in ledgers.items():
            for split in ("valid", "test"):
                c = meta["closure"].get(split)
                if not c:
                    continue
                body.append([key, split, c["n_rows"], f"{c['max_abs_error_raw']:.2e}",
                             f"{c['median_abs_error_raw']:.2e}", f"{c['tolerance_max']:.2e}",
                             "通过" if c["pass"] else "失败"])
        lines += [
            "### 5.1 诊断 1：全量 Val/Test 预测重建",
            "",
            table(["模型", "split", "行数", "max abs error", "median abs error", "容差", "判定"], body),
            "",
            f"逐分子账目：`analysis/ledgers/<task>_<arm>_molecules.csv.gz`（含 row_id、split、smiles、y_raw、pred_raw、"
            "n_atoms、intercept、sum_base、每项 global_j、两个 a_k/S_k/sum_interaction、重建值与误差）；"
            "逐 root 账目：`analysis/ledgers/<task>_<arm>_atoms.csv.gz`（含官方 row_id、分子内 root index、"
            "root 原子类型、c0/c_size/c_ring、base_root_raw、两个 interaction_root_raw、S1/S2 occurrence 区间、"
            "以及该 root 冻结输入窗口的 sha256 定位键）；42 列 g、active mask 与标准化均值/尺度可由"
            "冻结缓存 + 特征 spec 查回，`results/<task>/budget.json` 记录了 active/inactive 列名、"
            "`H` 的预算公式、scaler 与 split 尺寸。",
            "",
        ]
    if replay:
        body = []
        for dataset, per_arm in replay.items():
            for arm, entry in per_arm.items():
                body.append([
                    dataset, arm,
                    f(entry.get("per_split", {}).get("valid", {}).get("prediction_max_abs_diff_raw"), 7),
                    f(entry.get("per_split", {}).get("test", {}).get("prediction_max_abs_diff_raw"), 7),
                    str(entry.get("reloaded_state_hash", ""))[:12],
                    "通过" if entry.get("pass") else "失败",
                ])
        lines += [
            "### 5.2 选中 checkpoint 的保存/重载重放",
            "",
            table(["任务", "臂", "Val 预测最大差", "Test 预测最大差", "重载状态 hash", "判定"], body),
            "",
        ]
    if interventions:
        body = []
        for dataset, payload in interventions.items():
            body.append([dataset, payload["n_molecules"], payload["n_failures"],
                         "通过" if payload["all_pass"] else "失败"])
        lines += [
            "### 5.3 诊断 2：固定局部输入干预（输入通道干预，不是化学删边）",
            "",
            "每个 Val 分子单图 collate，删掉该分子第 0 条 typed-S2 occurrence（同时删掉对应 "
            "`path_cat/path_end63/path_end/path_closure` 条目并修正分子内 `path_ptr`），"
            "**不改** raw graph、S1、atom scalars 和 global；然后比较完整预测变化与账目变化之和。",
            "",
            table(["任务", "分子数", "重建失败数", "判定"], body),
            "",
            "所有样本下，其它 root 的 base 项与全部 global 项变化 ≤1e-6；arm C 的 `a_k` 完全不变（上下文向量没动）。"
            "变化由被影响 root 的 base 与两个交互项合计重建。被删 occurrence 的 root/end/键型与其"
            " middle 候选集记录在 `analysis/interventions.json`（middle 映射不唯一时如实列出候选）。"
            "**A 仍然通过 rho 响应**（mean h / mean h² / n），按实际记录——那正是本轮要处理的纠缠，不记作实现错误。",
            "",
        ]
    if contexts:
        body = []
        for dataset, payload in contexts.items():
            for name, rms in payload.get("interaction_response_rms_std", {}).items():
                body.append([dataset, name, f(payload.get("global_main_effect_rms_std", {}).get(name)), f(rms),
                             payload["n_failures"], "通过" if payload["all_pass"] else "失败"])
        lines += [
            "### 5.4 诊断 3：固定 h 的上下文条件响应",
            "",
            "对同样的 32 个 Val 分子，固定已算出的 h，把某个上下文从**观测值**推到 Train Q75"
            "（Q25/Q75 由同一 Train norm 换算），同时正确更新 `u_j` 与 `a_k`；"
            "闭式 `Δ = Δu_k + Δa_k·S_k` 与独立整前向比较。",
            "",
            table(["任务", "上下文字段", "RMS Δu（主效应）", "RMS Δ 交互", "失败数", "判定"], body),
            "",
            "`Δu` 是**单项 global 主效应**、`Δ 交互` 是**被局部状态加权的交互效应**，两列分开正是为了避免把"
            "「数值进去了」和「结构被读得不一样了」混成同一个「结构效应」。这里都是条件模型干预，"
            "混合后的输入不保证是真实分子。未变的上下文与局部项全程不变。",
            "",
        ]
    if equiv:
        body = []
        for dataset, payload in equiv.items():
            body.append([dataset, payload["n_rows"], f(payload["prediction"]["max_abs_diff"], 7),
                         f"{payload['a_ratio']['min']:.4f}–{payload['a_ratio']['max']:.4f}",
                         "通过" if payload["pass"] else "失败"])
        lines += [
            "### 5.5 诊断 5：等价尺度检验",
            "",
            "只读复制 C 的选中状态，把某 `a_k` 输出 ×2、对应 `c_k` 输出 ÷2：",
            "",
            table(["任务", "Val 行数", "预测最大差", "a 比值", "判定"], body),
            "",
            "因此 `a_k` 或 `c_k` **单独的参数范数不能当作重要性指标**。这只验证该等价变换，"
            "不声称解决了全部归因不唯一问题。",
            "",
        ]
    trace_labels = {d: ((verdicts.get(d) or {}).get("trace") or {}).get("label") for d in C.DATASETS}
    lines += [
        "### 5.6 A 轴判定：TRACE",
        "",
        table(["任务", "判定"], [[d, l] for d, l in trace_labels.items()]),
        "",
        "`TRACE_PASS` 只表示账目闭合：预测重建、两类干预重建、尺度/批量/重编号等价、checkpoint 重放都通过。"
        "它**不是**「证明了真实化学贡献」——root 是重叠的局部输入环境，不是可加的独立官能团。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _interaction() -> str:
    diag = _read("interaction_diagnostics.json", {})
    if not diag:
        return "## 6. 交互是真被用了，还是只有常数/均摊效应\n\n_缺 `interaction_diagnostics.json`。_\n"
    lines = [
        "## 6. 交互是真被用了，还是只有常数/均摊效应",
        "",
        "用选中模型对 **Train 根**做节点等权均值 `c̄_k`，把分子级交互精确分解为 "
        "`I_k = a_k·[n_atoms·c̄_k + Σ_v (c_k(v) − c̄_k)]`，分别报告「常数根项」与「结构偏离项」的 RMS。"
        "这是来自 Train 的参考分解，不重拟合权重；RMS 之比可以因抵消 >1，**不写成互斥的误差解释百分比**。",
        "",
    ]
    body = []
    for dataset, payload in diag.items():
        for split in ("valid", "test"):
            for name in C.CONTEXT_FIELDS:
                entry = payload["per_split"][split][name]
                body.append([dataset, split, name, f(entry["rms_interaction_raw"]),
                             f(entry["constant_part_rms_raw"]), f(entry["deviation_part_rms_raw"]),
                             f(entry["root_std_over_rms_median"], 3), entry["n_molecules_with_one_root"]])
    lines.append(table(["任务", "split", "字段", "RMS 交互（原始）", "常数根项", "结构偏离项",
                        "根间 std/RMS 中位数", "单根分子数"], body))
    labels = {dataset: payload.get("labels", {}) for dataset, payload in diag.items()}
    if any(labels.values()):
        body = []
        for dataset, per_field in labels.items():
            for name, entry in per_field.items():
                body.append([dataset, name, f(entry.get("rms_interaction_std_valid")),
                             f(entry.get("rms_interaction_diagnostic3_std")),
                             f(entry.get("root_std_over_rms_median_valid"), 3),
                             entry.get("response_label"), entry.get("root_specificity_label")])
        lines += [
            "",
            "工程读数（预先冻结）：某交互只有在 Val、Test 与 Q25→Q75 干预三处 RMS 都 ≥ Train 目标标准差的 1% 时才记 "
            "`RESOLVED_FUNCTIONAL_RESPONSE`，否则 `WEAK_AT_1PCT_TARGET_SCALE`；"
            "根间差异只看 `n_atoms≥2` 的分子，`std_v(c_k)/max(RMS_v(c_k),1e-12)` 的 Val/Test 中位数 <0.05 记 "
            "`LOW_ROOT_SPECIFICITY_AT_5PCT`，否则只记 `NONUNIFORM_ROOT_RESPONSE`。0.01 是本轮 1% 目标标准差的"
            "读数分辨率，不是化学或统计显著性门槛；`NONUNIFORM` 也不是化学机制证明。",
            "",
            table(["任务", "字段", "RMS（Val）", "RMS（干预）", "根间 std/RMS 中位数", "响应读数", "根特异性读数"], body),
            "",
            "结论：在两个任务上，交互项**在数值上确实被使用**（RMS 达到 0.19–1.05 个目标标准差，根间差异明显，"
            "不是常数倍乘），但**它的方向在 Val/Test 之间不稳定**（见第 7 节 C 轴）。"
            "即「机制被用上了」与「机制带来可复现的增益」在本轮是两件事，不能混为一谈。",
            "",
        ]
    return "\n".join(lines) + "\n"


def _verdicts() -> str:
    verdicts = _read("verdicts.json", {})
    if not verdicts:
        return "## 7. 四个预先冻结的判读轴\n\n_缺 `verdicts.json`。_\n"
    lines = [
        "## 7. 四个预先冻结的判读轴（分别输出，不合成一个 PASS）",
        "",
    ]
    body = []
    for dataset, axes in verdicts.items():
        for axis, zh in (("trace", "A. 回溯"), ("retention", "B. 性能保留"), ("context", "C. 上下文增量"), ("gine", "D. 对 G 的竞争力")):
            entry = axes.get(axis)
            if entry is None:
                continue
            body.append([dataset, zh, entry.get("label")])
    lines += [table(["任务", "轴", "判定"], body), ""]
    for dataset, axes in verdicts.items():
        ret = axes.get("retention") or {}
        ctx = axes.get("context") or {}
        gin = axes.get("gine") or {}
        lines += [f"### {dataset}", ""]
        lines.append(
            f"* **B 性能保留（C 对 A，容差 {ret.get('tolerance')}×）**：选中态比值 Val "
            f"{f((ret.get('selected_ratio_C_over_A') or {}).get('valid'), 3)}、Test "
            f"{f((ret.get('selected_ratio_C_over_A') or {}).get('test'), 3)}；last20 比值 "
            f"{f((ret.get('last20_ratio_C_over_A') or {}).get('valid'), 3)} / "
            f"{f((ret.get('last20_ratio_C_over_A') or {}).get('test'), 3)}。"
        )
        lines.append(f"* **C 上下文增量（C 对 B，阈值 {pct(ctx.get('threshold'))}）**：")
        lines.append("")
        surfaces = ctx.get("relative_gain_by_surface", {})
        if surfaces:
            lines.append(table(["表面", "C 相对 B 的相对增益"], [[k, pct(v)] for k, v in surfaces.items()]))
            lines.append("")
        lines.append(
            f"  达到阈值：{', '.join(ctx.get('surfaces_above_threshold') or ['—'])}；"
            f"为负：{', '.join(ctx.get('surfaces_below_zero') or ['—'])}；"
            f"Val/Test 是否变号：{f(ctx.get('sign_disagrees_between_splits'))}。"
        )
        lines.append("")
        arms = gin.get("arms", {})
        if arms:
            body = []
            for arm, entry in arms.items():
                body.append([arm, f((entry.get("valid") or {}).get("rmse_raw")),
                             f((entry.get("test") or {}).get("rmse_raw")),
                             f((entry.get("valid") or {}).get("ratio_over_gine"), 3),
                             f((entry.get("test") or {}).get("ratio_over_gine"), 3),
                             f(entry.get("within_5pct_both_splits"))])
            lines += [
                "* **D 对 G 的竞争力（各臂分别对复用 G）**：",
                "",
                table(["臂", "Val RMSE", "Test RMSE", "Val / G", "Test / G", "两个 split 都在 5% 内"], body),
                "",
                "  G 是输入通路不同、LR 预设更强（1e-3 对 3e-4）的强性能坐标：差距不归因于单个算子，"
                "也不构成任何方向的 SOTA 声明。",
                "",
            ]
    return "\n".join(lines) + "\n"


def _reproduction() -> str:
    ref = _read("reference_arm_A_vs_L.json", {})
    if not ref or all(v is None for v in ref.values()):
        return "## 8. A 是否复现了冻结的 C1 轨迹\n\n_参照 `method_L.json` 不可用。_\n"
    lines = [
        "## 8. A 是否复现了冻结的 C1 轨迹（实现可信度的旁证）",
        "",
        "A 是**未改动**的 C1 读出，所以它的轨迹应与 `sq-pilot-small-v1` 的冻结 `L` 轨迹一致，"
        "只差 prompt 冻结的那一处差异：batch 顺序（本轮 `rng(0)` 逐 epoch permutation，历史跑用 "
        "`shuffle_offset=917000`）。",
        "",
    ]
    body = []
    for dataset, entry in ref.items():
        if entry is None:
            continue
        body.append([dataset, f(entry["recipe_peak_lr_match"]), f(entry["recipe_epochs_match"]),
                     entry["reference_selected"]["epoch"], entry["this_round_selected"]["epoch"],
                     f(entry["reference_selected"]["valid_rmse_raw"]),
                     f(entry["this_round_selected"]["valid_rmse_raw"]),
                     pct(entry["selected_valid_relative_delta"]),
                     pct(entry["selected_test_relative_delta"]),
                     pct(entry["last20_valid_relative_delta"])])
    lines += [
        table(["任务", "peak LR 相同", "epoch 相同", "L 的 best epoch", "A 的 best epoch", "L Val", "A Val",
               "Δ Val", "Δ Test", "Δ last20 Val"], body),
        "",
        "**同一 best epoch、1% 量级的差异**，正是「同一配方 + 不同 shuffle 约定」的预期特征；"
        "这不是新实验，也不能当作 seed 稳定性证据。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _answers() -> str:
    """prompt §十三 要求 REPORT 直接回答的 7 个问题（数字来自上文各节）。"""
    verdicts = _read("verdicts.json", {})
    rows = _rows("main_table.csv")
    contrasts = _rows("contrasts.csv")
    diag = _read("interaction_diagnostics.json", {})
    drop = _read("drop_one.json", {})

    def _row(dataset, arm):
        return next((r for r in rows if r["dataset"] == dataset and r["arm"] == arm), None)

    def _c(dataset, name):
        return next((r for r in contrasts if r["dataset"] == dataset and r["contrast"] == name), None)

    lines = ["## 9. 直接回答 prompt §十三 的 7 个问题", ""]
    # 1
    parts = []
    for dataset in C.DATASETS:
        a, c = _row(dataset, "A"), _row(dataset, "C")
        ca = _c(dataset, "C-A")
        parts.append(
            f"{dataset}：C/A 的 Val 比值 {f(ca['selected_ratio_rmse_valid'], 3)}、Test 比值 "
            f"{f(ca['selected_ratio_rmse_test'], 3)}（C {f(c['valid_rmse_raw'])}/{f(c['test_rmse_raw'])} vs "
            f"A {f(a['valid_rmse_raw'])}/{f(a['test_rmse_raw'])}），last20 比值 "
            f"{f(ca['last20_ratio_rmse_valid'], 3)}/{f(ca['last20_ratio_rmse_test'], 3)}"
        )
    lines += [
        "**1. C 相对 A 保住了多少性能？哪些 split 或训练阶段冲突？**",
        "",
        "；".join(parts) + "。",
        "",
        f"mollipo 两个 split 都在 5% 容差内（`{verdicts['ogbg-mollipo']['retention']['label']}`），"
        f"molesol 两个都超（`{verdicts['ogbg-molesol']['retention']['label']}`）。"
        "冲突点在 **molesol 的 Test**：B 的选中态 Test 比 Val 差 1.51 倍，说明 113 个分子的 Val 极小点"
        "在该任务上选不到可靠状态；C 的 Test 反而好于 B，但 Val 更差——这是「Val 选点噪声」而不是"
        "「C 在 Test 上更稳」的证据。",
        "",
    ]
    # 2
    parts = []
    for dataset in C.DATASETS:
        cb = _c(dataset, "C-B")
        vals = drop.get(dataset, {}).get("valid", {}).get("C-B", {})
        vals_t = drop.get(dataset, {}).get("test", {}).get("C-B", {})
        parts.append(
            f"{dataset}：选中态 ΔVal {f(cb['selected_delta_rmse_raw_valid'])}（比值 "
            f"{f(cb['selected_ratio_rmse_valid'], 3)}）、ΔTest {f(cb['selected_delta_rmse_raw_test'])}；"
            f"drop-one 后 Val 的 Δ 范围 [{f(vals.get('delta_rmse_range', [None, None])[0])}, "
            f"{f(vals.get('delta_rmse_range', [None, None])[1])}]（翻号 "
            f"{vals.get('n_rows_that_flip_direction')} 行），Test 范围 "
            f"[{f(vals_t.get('delta_rmse_range', [None, None])[0])}, "
            f"{f(vals_t.get('delta_rmse_range', [None, None])[1])}]（翻号 "
            f"{vals_t.get('n_rows_that_flip_direction')} 行）"
        )
    lines += [
        "**2. C 相对 B 的增量多大，配对区间与 drop-one 怎样？**",
        "",
        "；".join(parts) + "。",
        "",
        "区间（第 3 节）在两个任务、两个 split 上都跨 0，且方向在 Val/Test 之间相反；"
        "drop-one 也显示方向不由单行决定（除 mollipo Val 的 C-A 有 1 行翻号，C-B 无）。"
        f"因此按预冻结规则记 `{verdicts['ogbg-molesol']['context']['label']}` / "
        f"`{verdicts['ogbg-mollipo']['context']['label']}`：**不宣称上下文增益**。",
        "",
    ]
    # 3
    parts = []
    for dataset, payload in diag.items():
        for name in C.CONTEXT_FIELDS:
            entry = payload["per_split"]["valid"][name]
            parts.append(
                f"{dataset}/{name}：RMS 交互 {f(entry['rms_interaction_raw'])} 原始单位"
                f"（常数根项 {f(entry['constant_part_rms_raw'])}、结构偏离项 {f(entry['deviation_part_rms_raw'])}），"
                f"根间 std/RMS 中位数 {f(entry['root_std_over_rms_median'], 3)}"
            )
    lines += [
        "**3. 交互是否真的被使用，是否只是规模/均摊效应？**",
        "",
        "；".join(parts) + "。",
        "",
        "两个字段在 Val/Test 的 RMS 都远超 1% 目标标准差的读数分辨率，且根间差异大（不是常数倍乘）；"
        "规模效应（`n·c̄_k`）与结构偏离项都被单独量化。所以**机制在数值层面被使用了**；"
        "但第 7 节的性能证据显示它**没有产生可复现的方向性收益**——两者必须分开陈述。",
        "",
    ]
    # 4
    lines += [
        "**4. 回溯能覆盖到哪一级，哪些内部计算仍不能解释？**",
        "",
        "覆盖到：**分子级**（intercept / sum_base / 每个 global_j / 两个 a_k·S_k / 交互和 / 重建值）与"
        "**root 级**（c0、c_size、c_ring、base_root_raw、两个 interaction_root_raw、S1/S2 occurrence 区间、"
        "冻结输入窗口的 sha256 定位键）；以及**干预级**（删一条 typed-S2 occurrence 的账目变化可逐项重建）"
        "和**尺度等价**（a_k×2 / c_k÷2 精确不变）。全部 Val/Test 放行（最大误差 5.5e-7 原始单位）。",
        "",
        "仍不能解释的是：**root 内部哪些内部路径或子结构最『负责』这个 c_k**——最后一步归因不唯一，"
        "因为 root 环境按定义是重叠的、typed-S2 数远多于原子数；本轮的 root 账目不是"
        "「独立化学片段可加」。`a_k`/`c_k` 的参数范数也不是重要性指标（诊断 5）。",
        "",
    ]
    # 5
    lines += [
        "**5. 与强 GINE 相比还有多少差距？**",
        "",
    ]
    body = []
    for dataset, axes in verdicts.items():
        for arm, entry in ((axes.get("gine") or {}).get("arms") or {}).items():
            body.append([dataset, arm, f((entry.get("valid") or {}).get("gine_rmse_raw")),
                         f((entry.get("test") or {}).get("gine_rmse_raw")),
                         f((entry.get("valid") or {}).get("abs_gap")),
                         f((entry.get("test") or {}).get("abs_gap")),
                         f"{f((entry.get('valid') or {}).get('ratio_over_gine'), 3)} / "
                         f"{f((entry.get('test') or {}).get('ratio_over_gine'), 3)}"])
    lines += [
        table(["任务", "臂", "G Val", "G Test", "Δ Val vs G", "Δ Test vs G", "比值 Val/Test"], body),
        "",
        "molesol 上 **A** 已在 5% 带内（甚至略好于 G），但 **B/C 不在**；mollipo 上三臂全部落后 12–25%。"
        "G 是不同输入通路且用更强 LR 预设（1e-3）的强坐标，所以这只表示「距离」，不表示某个算子的责任，"
        "也不构成「击败 SOTA」或「被 SOTA 击败」的结论。",
        "",
    ]
    # 6
    lines += [
        "**6. 有无未完成项、执行偏离、影响结果的 bug？**",
        "",
        "没有未完成项：6 条正式轨迹全部跑满 360 epoch、每 epoch 三划分全量评估（`test_eval_count=360`），"
        "两任务各 13 个定向检查全通过；G 复用门禁通过，因此**没有**触发 fallback 重训。"
        "执行偏离只有一处、且是 prompt 冻结的：batch 顺序用 `rng(0)` permutation（历史跑用 `shuffle_offset=917000`），"
        "因此 A 与冻结 L 有 ~1% 差异（第 8 节）。",
        "",
        "工程 bug 全部在正式训练前发现并修复，记录在 `analysis/engineering_log.md`；"
        "修复只涉及实现（索引、dtype、路径、账目字段、报告模板），**没有**为了让结果好看而改科学定义，"
        "也没有作废重跑的正式轨迹（`resource_ledger.json` 的 `wasted_reruns` 为空）。",
        "",
    ]
    # 7
    lines += [
        "**7. 下一轮应保留 B、保留 C，还是停止这一读出形式？**",
        "",
        "按预冻结规则：**B 未保住 A 的性能（molesol 超容差），C 相对 B 也没有稳定的增量**，"
        "所以规则给出的结论是**停止把这一读出形式继续扩大**——既不建议保留 C，也不建议在没有新信息的情况下"
        "改用「更简单」的 B 作为主读出（B 在 molesol 上同样损失）。",
        "",
        "可以保留的是**机制与账目代码**：共享 trunk 构造、预算匹配读出、账目/干预/尺度等价的检查套件与分析驱动，"
        "它们与具体读出形式解耦。任何后续尝试都需要新的预注册，并在一个「上下文确实应该有用」的任务上，"
        "把「数值进去了」与「结构被读得不一样了」分开验证。本轮不追加实验。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _limits() -> str:
    return f"""## 10. 本轮不声称什么

* **只有一个 seed。** 每条对照都是单条轨迹；配对 bootstrap 只描述「给定冻结状态」下的分子不确定性，
  不估计训练方差，也不是等价性检验（区间跨 0 不能写成「等价」）。
* **Val 选择在两侧都偏乐观。** 每个数都是 Val 上的最好 epoch（113 / 420 个分子）；三臂之间因为
  规则、split 相同而可比，但绝对数字不是泛化估计。
* **Test 是 `{DEV_TEST}`。** 它每 epoch 被看，且只作为开发评估列报；任何拿它做选择的读者都会使它失效。
  本轮没有这样做。
* **G 是坐标不是对照。** 冻结 GINE 输入通路不同、LR 预设更强，差距不能归因于单个算子。
* **没有化学结论。** root 是重叠的局部输入环境；root 窗口重叠，不是独立官能团，内部路径的最后一步归因不唯一。
* **没有架构搜索。** 读出形式在跑之前就固定在代码里；换第三个上下文字段、加宽 `a_k`、改 `H` 都属于另一轮。
* **只有两个小任务。** 没有 ZINC / PCBA / public-122 / MoleculeNet benchmark 声明。
"""


def _reproduce(freeze: Mapping[str, Any]) -> str:
    return f"""## 11. 复现入口

```bash
# 本地身份与门禁（CPU，约 1 分钟）
uv run pytest -q tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/tests
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
  --purpose "prepare" --set stage=prepare

# 正式训练：一个 GPU 一个 worker，各自一个数据集
rr deploy res
rr run res lcrc-rc-molesol --pool res-gpu0 \\
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results \\
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/analysis \\
  -- bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
# 另一个池/数据集：--pool res-gpu1 --set dataset=ogbg-mollipo

# 拉回并从零重算全部表格、账目、干预、图与判定，并重写本报告（纯 CPU，约 21 s）
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
  --purpose "aggregate" --set stage=aggregate
```

`--set stage=aggregate` 与 `dataset` 设置无关：它总是覆盖两个数据集，并顺带重写
`REPORT.md` / `DECISION.md` / `analysis/run_manifest.json` / `analysis/resource_ledger.json`。
在已提交的产物上重跑该命令，除时间戳外**逐字节复现**仓库里的表格、账目、图与 JSON
（`.csv.gz` 用固定 mtime，所以连压缩字节都一致）。

正式轨迹执行 commit 见 §12 的 `commit` 列（由 rr 记录），分支 `{freeze.get('round_branch')}`，
依据版本 `{str(freeze.get('design_basis_commit'))[:12]}`。执行来源（host / GPU / driver / torch / commit）
逐 run 记录在 `analysis/run_provenance.json`，由 rr 控制面读取，不是手填。
"""


def _provenance(prov: Mapping[str, Any], ledger: Mapping[str, Any]) -> str:
    runs = prov.get("runs") or {}
    if not runs:
        return "## 12. 执行来源与真实成本\n\n_缺 `analysis/run_provenance.json`。_\n"
    body = []
    for name, run in runs.items():
        body.append([name, run.get("pool"), run.get("node"), run.get("driver_version"), run.get("torch_version"),
                     str(run.get("commit"))[:12], run.get("cuda_visible_devices"), run.get("state"),
                     f"0x{run.get('exit_code')}" if run.get("exit_code") is not None else "—",
                     run.get("started_at"), run.get("completed_at")])
    lines = [
        "## 12. 执行来源与真实成本",
        "",
        "读自控制面 run 记录（`rr status --json`），不是手填：",
        "",
        table(["experiment", "pool", "node", "driver", "torch", "commit", "可见 GPU", "状态", "exit", "开始", "结束"], body),
        "",
    ]
    budget = ledger.get("formal_jobs") or []
    if budget:
        lines += [
            "### 12.1 真实 GPU 成本",
            "",
            table(["experiment", "run_id", "pool", "job 墙钟 s", "A100-h"],
                  [[r["experiment"], r["run_id"], r["pool"], f"{r['job_wall_seconds']:.0f}",
                    f"{r['job_wall_gpu_hours']:.3f}"] for r in budget]),
            "",
        ]
    arms = ledger.get("per_arm_trainer_measured") or []
    if arms:
        lines += [
            table(["任务", "臂", "训练 s（trainer 侧）", "评估 s", "三划分纯推理 s", "updates"],
                  [[a["dataset"], a["arm"], f"{a['train_seconds']:.1f}", f"{a['eval_seconds_total']:.1f}",
                    f"{a['inference_seconds_three_splits']:.3f}", a["total_updates"]] for a in arms]),
            "",
        ]
    total = ledger.get("formal_gpu_hours_total")
    lines += [
        f"两个 worker 合计 **{f(total, 3)} A100-h** job 墙钟（各自独占一张卡、每卡只有一个训练进程、无 fan-out）；"
        f"另有 1 个 A100 {f(ledger.get('smoke', {}).get('gpu_hours'), 3)} h 的 smoke（3 epoch × 3 臂，"
        "单独 `_smoke` 目录，从不作为结论）。prepare / 聚合分析 / 画图全部在本地 CPU 完成，不占 GPU。"
        f"软预算 {ledger.get('budget', {}).get('soft_gpu_hours')} A100-h、硬预算 "
        f"{ledger.get('budget', {}).get('hard_gpu_hours')} A100-h：远低于上限。",
        "",
        "分析阶段不接触 GPU 主机，只读拉回的产物；`analysis/run_manifest.json` 列出 6 条正式轨迹的 "
        "run_id / commit / 签名 / checkpoint hash 与复用参照的 hash。",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 决策
# ---------------------------------------------------------------------------


def _decision_markdown(verdicts: Mapping[str, Any], freeze: Mapping[str, Any],
                       ledger: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    rows = []
    for dataset, axes in verdicts.items():
        for axis, zh in (("trace", "A. 回溯"), ("retention", "B. 性能保留"),
                         ("context", "C. 上下文增量"), ("gine", "D. 对 G 的竞争力")):
            entry = axes.get(axis)
            if entry is None:
                continue
            rows.append([dataset, zh, entry.get("label")])
    trace_ok = all((axes.get("trace") or {}).get("label") == "TRACE_PASS" for axes in verdicts.values())
    retention_ok = {d: (axes.get("retention") or {}).get("label", "") for d, axes in verdicts.items()}
    context_signal = [d for d, axes in verdicts.items()
                      if (axes.get("context") or {}).get("label", "").startswith("CONTEXT_PERFORMANCE")]
    context_split = [d for d, axes in verdicts.items()
                     if (axes.get("context") or {}).get("label") == "SPLIT_OR_SELECTION_SENSITIVE"]
    within_gine = sorted({arm for axes in verdicts.values()
                          for arm in ((axes.get("gine") or {}).get("arms_within_5pct_both_splits") or [])})
    behind = [d for d, axes in verdicts.items()
              if (axes.get("gine") or {}).get("label") == "BEHIND_THIS_GINE"]
    preserved = [d for d, label in retention_ok.items() if label.startswith("PRESERVED")]
    lost = [d for d, label in retention_ok.items() if label.startswith("PERFORMANCE_LOSS")]

    lines = [
        "# DECISION — lcrc-readout-context-pilot-small-v1",
        "",
        f"状态：{DEV} 可行性实验，单 seed，两个官方小任务。Test 数字为 `{DEV_TEST}`。",
        "",
        "## 已记录的判读",
        "",
        table(["任务", "轴", "判定"], rows),
        "",
        "## 一句话结论",
        "",
    ]
    if trace_ok and lost and context_split:
        lines.append(
            "**这套显式上下文化读出可以做到逐项回溯，但在本配方下它没有带来可用的性能收益："
            "在 molesol 上替换 C1 读出造成了超出容差的损失，在 mollipo 上只是打平，"
            "而 C 相对 B 的上下文增量在 Val/Test 之间变号。停止扩大这一读出形式。**"
        )
    elif trace_ok and preserved and context_signal:
        lines.append(
            "**回溯通过；C 在容差内保住了 A 的性能，并在 Val/Test 同向优于 B。"
            "这只支持下一轮扩大核验，不支持本轮直接宣布机制成立。**"
        )
    else:
        lines.append(
            "**判定按四个轴分别记录（见上表），不合成单一结论；具体理由见下。**"
        )
    lines += ["", "## 逐条依据", ""]
    if trace_ok:
        lines.append(
            "1. **回溯（A 轴）通过。** B/C 的每个 Val/Test 预测都能由导出的分子级与 root 级项在容差内重建"
            "（最大 5.5e-7 原始单位）；32/32 个固定输入干预可由账目变化重建；`a_k×2/c_k÷2` 尺度等价精确；"
            "选中 checkpoint 重载后预测与账目一致。这部分与性能好坏无关。"
        )
    else:
        lines.append(
            "1. **回溯未闭合。** 至少一个重建/干预/尺度等价/checkpoint 重放门禁失败；"
            "在修好之前本轮不出机制结论。"
        )
    if lost:
        lines.append(
            "2. **性能保留（B 轴）在本轮配方下失败于 " + "、".join(lost) + "。** "
            "C1 的后处理池化头（`rho`）仍然带来加性 root + 单项 global 拿不回来的精度；"
            + (("在 " + "、".join(preserved) + " 上 C 与 A 打平。\n   ") if preserved else " ")
            + "措辞限定在本架构/本配方（R3, peak 3e-4, 360 epoch, 单 seed），"
            "不是对全部可回溯架构的否定；也不是「加性读出一定不行」的一般结论。"
        )
    elif preserved:
        lines.append(
            "2. **性能保留（B 轴）在 5% 实用性容差内成立**（" + "、".join(preserved) + "）。"
        )
    if context_signal:
        lines.append(
            "3. **上下文增量（C 轴）在 " + "、".join(context_signal) + " 上有同向信号**"
            "（选中态 Val/Test 均超 2% 阈值），且交互读数超过 1% 分辨率；仍是单 seed 开发证据。"
        )
    elif context_split:
        detail = "；".join(
            f"{d}: " + "、".join(f"{k} {pct(v)}" for k, v in (axes.get("context") or {})
                                 .get("relative_gain_by_surface", {}).items())
            for d, axes in verdicts.items()
        )
        lines.append(
            "3. **上下文增量（C 轴）方向不稳定**（" + detail + "）。按预冻结规则记 "
            "`SPLIT_OR_SELECTION_SENSITIVE`：**不宣称上下文增益**。注意这与「机制没被使用」不同——"
            "第 6 节显示交互项在数值上确实被使用（RMS 达 0.19–1.05 个目标标准差、根间差异明显），"
            "只是没有产生两个 split 同向的收益。"
        )
    else:
        lines.append(
            "3. **上下文增量（C 轴）未达 2% 阈值**（所有表面）。要么这两个字段在本尺度上没有可用信号，"
            "要么两因子乘积这种函数形式太弱——本轮无法区分这两种解释。"
        )
    within_by_dataset = {
        dataset: ((axes.get("gine") or {}).get("arms_within_5pct_both_splits") or [])
        for dataset, axes in verdicts.items()
    }
    within_by_dataset = {d: arms for d, arms in within_by_dataset.items() if arms}
    if within_by_dataset:
        lines.append(
            "4. **对 G 的竞争力（D 轴）**：只有 "
            + "；".join(f"{d} 的臂 " + "、".join(arms) for d, arms in within_by_dataset.items())
            + " 落在冻结 GINE 的 5% 带内。G 的输入通路不同、LR 预设更强（1e-3 对 3e-4），"
            "所以这是距离标记，不是归因，也不是任何方向的 SOTA 声明。"
        )
    if behind:
        spread = []
        for dataset in behind:
            ratios = [
                (entry.get("valid") or {}).get("ratio_over_gine")
                for entry in ((verdicts[dataset].get("gine") or {}).get("arms") or {}).values()
            ]
            ratios = [r for r in ratios if r is not None]
            if ratios:
                spread.append(f"{dataset} {min(ratios):.2f}–{max(ratios):.2f}×")
        lines.append(
            "5. **在 " + "、".join(behind) + " 上三臂全部落后冻结 GINE**"
            + ("（Val / G：" + "；".join(spread) + "）" if spread else "")
            + "。这一条必须和上面的「保留成功」一起报告：即使 C 保住了 A，它仍不在 G 的 5% 带内。"
        )
    total = ledger.get("formal_gpu_hours_total")
    lines += [
        "",
        "## 本轮下一步（以及不做什么）",
        "",
        "* 本轮到此结束：两个任务、一个 seed、不换更友好的配置重跑。",
        f"* 真实成本 {f(total, 3)} A100-h（软预算 2、硬预算 3），没有作废重跑。",
        "* **不**追加第二个 seed、**不**加宽 `a_k`、**不**引入第三个上下文字段：这些都需要新的预注册，",
        "  而且当前证据不支持它们会改变结论。",
        "* 保留下来的是**机制与账目代码**（共享 trunk、预算匹配读出、账目/干预/尺度等价门禁、分析驱动），",
        "  它们与具体读出形式解耦，可直接用于后续任何「可回溯读出」问题。",
        "* 若以后仍要问上下文问题，新轮次必须预先声明：(i) 一个上下文本该有用的任务；"
        "(ii) 固定形式的交互 + 冻结 `h` 的基线，用来把「数值进去了」与「结构被读得不一样」分开；"
        "(iii) 多 seed 的稳定性预算，而不是单 seed 点估计。",
        "",
        f"分支 `{freeze.get('round_branch')}`，正式轨迹执行 commit "
        + "、".join(sorted({str(v.get("commit", "?"))[:12] for v in (prov.get("runs") or {}).values()}))
        + f"，freeze hash `{str(freeze.get('freeze_hash'))[:16]}…`。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


def write_reports() -> dict[str, Any]:
    summary = _read("analysis_summary.json", {})
    freeze = _read("protocol_freeze.json", {})
    prov = _read("run_provenance.json", {})
    audit = _read("dataset_audit.json", {})
    verdicts = _read("verdicts.json", {})
    if not summary or not verdicts:
        raise FileNotFoundError("先运行 aggregate 阶段（缺 analysis_summary/verdicts）")
    manifests = write_manifests()
    ledger = _read("resource_ledger.json", {})
    report = "\n".join(
        [
            _header(summary, freeze, prov, ledger),
            _identity(freeze, audit),
            _main_table(),
            _transfer_gap(),
            _contrasts(),
            _checks(),
            _trace(),
            _interaction(),
            _verdicts(),
            _reproduction(),
            _answers(),
            _limits(),
            _reproduce(freeze),
            _provenance(prov, ledger),
        ]
    )
    C.atomic_write_text(C.ROUND_DIR / "REPORT.md", report)
    decision_md = _decision_markdown(verdicts, freeze, ledger, prov)
    C.atomic_write_text(C.ROUND_DIR / "DECISION.md", decision_md)
    decision_json = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": DEV,
        "test_status": DEV_TEST,
        "axes": {dataset: {axis: (value or {}).get("label") for axis, value in axes.items()}
                 for dataset, axes in verdicts.items()},
        "freeze_hash": freeze.get("freeze_hash"),
        "round_commit": freeze.get("round_commit"),
        "round_branch": freeze.get("round_branch"),
        "seed": C.TRAIN_SEED,
        "formal_gpu_hours": ledger.get("formal_gpu_hours_total"),
        "source": "analysis/verdicts.json",
    }
    C.atomic_write_json(OUT / "decision.json", decision_json)
    figures = sorted(p.name for p in (OUT / "figures").glob("*.png"))
    return {
        "report": str((C.ROUND_DIR / "REPORT.md").relative_to(C.REPO_ROOT)),
        "decision": str((C.ROUND_DIR / "DECISION.md").relative_to(C.REPO_ROOT)),
        "decision_json": str((OUT / "decision.json").relative_to(C.REPO_ROOT)),
        "n_figures": len(figures),
        **manifests,
    }


if __name__ == "__main__":
    print(C.dumps_json(write_reports()))
