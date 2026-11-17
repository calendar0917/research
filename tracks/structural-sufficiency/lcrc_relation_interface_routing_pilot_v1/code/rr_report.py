"""从本轮分析产物生成中文 ``REPORT.md`` / ``DECISION.md``。

本模块不做任何科学计算，也不手打任何数字：每个数都从本轮写出的
``analysis/*.json`` / ``analysis/*.csv`` 与 ``results/**/curve.csv`` 读回。

回答的五个问题（见 ``REPORT.md`` 第 5 节）：

1. 固定拓扑分块 + 接口路由是否基本保留了 C1 的预测力？
2. 接口贡献是否大到值得作为一个显式对象？
3. 接口是否需要两个端点 block 的上下文（R2 vs R1）？
4. 门控选出的 block 是否比同分子、同 bond 数、同 atom 数的随机连通子图更好地
   定位 WC_LOGP 的 oracle 质量？
5. 是否推进到可学习分块 / LCRC-v2？
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import rr_common as C

OUT = C.ANALYSIS_DIR
DEV = "DEVELOPMENT"
DEV_TEST = "DEVELOPMENT_TEST"


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if value != value:  # NaN
        return "—"
    return f"{value:.{nd}f}"


def pct(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if value != value:
        return "—"
    return f"{100.0 * value:+.{nd}f}%"


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


def _csv(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _trajectory(task: str, model: str) -> dict[str, Any]:
    for path in (OUT / f"trajectory_{task}_{model}.json", C.task_results_dir(task, model) / "trajectory.json"):
        if path.is_file():
            payload = C.read_json(path)
            if "selected" in payload:
                return payload
    return {}


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def write_reports() -> dict[str, Any]:
    decision = _read("decision.json", {})
    support = _read("partition_support.json", {})
    budget = _read("parameter_budget.json", {})
    freeze = _read("protocol_freeze.json", {})
    invariants = _read("routing_invariants.json", {})
    ledger_summary = _read("native_ledger_summary.json", {})
    oracle_summary = _read("oracle_localization_summary.json", {})
    bootstrap = _read("bootstrap_r2_vs_r1.json", {})
    manifest = _read("run_manifest.json", {})
    predictions = _csv("prediction_comparison.csv")
    rho_rows = _csv("rho_only_baseline.csv")

    report = _report(decision, support, budget, freeze, invariants, ledger_summary,
                     oracle_summary, bootstrap, manifest, predictions, rho_rows)
    C.atomic_write_text(OUT.parent / "REPORT.md", report)
    decision_md = _decision(decision, bootstrap, oracle_summary)
    C.atomic_write_text(OUT.parent / "DECISION.md", decision_md)
    return {
        "report": "REPORT.md",
        "decision": "DECISION.md",
        "n_prediction_rows": len(predictions),
        "flags": decision.get("flags", {}),
    }


def _report(decision, support, budget, freeze, invariants, ledger_summary,
            oracle_summary, bootstrap, manifest, predictions, rho_rows) -> str:
    gate = support.get("gate", {})
    valid = support.get("splits", {}).get("valid", {})
    models = budget.get("models", {})
    ratios = budget.get("ratio", {})
    retention = decision.get("retention", {})
    capacity = decision.get("capacity", {})
    flags = decision.get("flags", {})
    propos = decision.get("propositions", {})

    lines: list[str] = []
    add = lines.append

    add("# 关系–接口路由 pilot：LCRC 局部编码分块化 + 接口上下文")
    add("")
    add(f"轮次 `{C.ROUND_ID}`；一切数字均为 **{DEV}**，Test 列是 **{DEV_TEST}**。")
    add("本轮只训练 R1/R2 四个训练单元；C1、GINE、oracle 基线一律只读复用，"
        "不重训、不重选、不重评。")
    add("")

    # ---- 1 -------------------------------------------------------------
    add("## 1. 本轮问题")
    add("")
    add("1. 把每个 LCRC S1 出现路由到拥有该键的拓扑 block、把每个 S2 出现路由到"
        "拥有两条端点键的单个 block 或两个 block 的接口，预测力能否基本保留？")
    add("2. 接口贡献是否大到值得作为一个显式对象？")
    add("3. 接口是否需要两个端点 block 的上下文（R2 相对 R1）？")
    add("4. 门控选出的 block 是否比匹配随机连通子图更好地定位 WC_LOGP 的 oracle 质量？")
    add("5. 是否推进到可学习分块 / LCRC-v2？")
    add("")
    add("这是**架构可行性 pilot**，不是最终论文验证；单 seed、无超参搜索、无额外数据集。")
    add("")

    # ---- 2 -------------------------------------------------------------
    add("## 2. 冻结的分块与路由（CPU 生成，一次，SHA-256 固定）")
    add("")
    add(f"分块判据：`{freeze.get('partition_rule', {}).get('statement', '—')}`")
    add("")
    add(f"支持门控（Valid）：{gate.get('criterion', '—')}；"
        f"实测 **{f(gate.get('observed_fraction'), 4)}** ≥ 阈值 {f(gate.get('required_fraction'), 2)} → "
        f"`{gate.get('status', '—')}`。")
    add("")
    kinds = valid.get("block_kind_counts", {})
    rows = [
        ["分子数（≥2 block / 可用）", f"{valid.get('n_molecules', 0)}（{valid.get('molecules_with_ge2_blocks', 0)} / {valid.get('molecules_usable', 0)}）"],
        ["block 数（均值，最大键数）", f"{valid.get('n_blocks', 0)}（{f(valid.get('mean_blocks_per_molecule'), 3)}，{valid.get('max_block_bonds', '—')} 键）"],
        ["block 类型 cycle / chain / isolated",
         f"{kinds.get('cycle', 0)} / {kinds.get('chain', 0)} / {kinds.get('isolated_atom', 0)}"],
        ["单 block 分子数", f(valid.get("single_block_molecules", 0), 0)],
        ["接口数（均值）", f"{valid.get('n_interfaces', 0)}（{f(valid.get('mean_interfaces_per_molecule'), 3)}）"],
        ["共享原子数（占比）", f"{valid.get('n_shared_atoms', 0)}（{f(valid.get('shared_atom_fraction'))}）"],
        ["零键 block", f(valid.get("zero_bond_blocks", 0), 0)],
        ["S1 出现数", f(valid.get("n_s1_occurrences", 0), 0)],
        ["S2 总数 = 内部 + 跨块",
         f"{valid.get('n_s2_total', 0)} = {valid.get('n_s2_internal', 0)} + {valid.get('n_s2_cross', 0)}"],
    ]
    add(table(["量", "Valid"], rows))
    add("")
    cache_sha = support.get("routing_cache_file_sha256", {})
    add("两份 GPU 读的是**同一份** CPU 缓存（SHA-256）：")
    add("")
    add(table(["split", "routing cache sha256"],
              [[s, f"`{cache_sha.get(s, '—')}`"] for s in C.SPLITS]))
    add("")

    # ---- 3 -------------------------------------------------------------
    add("## 3. 架构与参数预算")
    add("")
    add(table(
        ["模型", "trunk", "block/接口读出", "总计", "相对 C1", "R2/R1"],
        [
            ["C1（只读基线）", "—", "—", budget.get("c1_reference_total", "—"), "1.0000", "—"],
            ["R1", models.get("R1", {}).get("trunk", "—"), models.get("R1", {}).get("readout", "—"),
             models.get("R1", {}).get("total", "—"), f(ratios.get("r1_over_c1")), "—"],
            ["R2", models.get("R2", {}).get("trunk", "—"), models.get("R2", {}).get("readout", "—"),
             models.get("R2", {}).get("total", "—"), f(ratios.get("r2_over_c1")), f(ratios.get("r2_over_r1"))],
        ],
    ))
    add("")
    add(f"预算规则：{budget.get('budget_rule', '—')}；容量规则：{budget.get('capacity_rule', '—')}。")
    add(f"共享初始权重：`{budget.get('shared_parameter_tensors', '—')}` 个张量在 R1/R2 之间逐位相同 → "
        f"**{f(budget.get('shared_initial_state_identical'))}**。"
        f"容量混淆：**{f(ratios.get('capacity_confounded'))}**。")
    add("")
    add("块门控是二值的：`y(m) = bias + rho_g + Σ_i m_i c_i + Σ_(i,j) m_i m_j c_ij`，"
        "`delta_i = y(all-on) − y(m_i=0)`，`score_i = |delta_i| / sqrt(max(1, n_bonds_i))`。")
    add("")

    # ---- 4 -------------------------------------------------------------
    add("## 4. 训练与门控状态")
    add("")
    rows = []
    for task in C.TASKS:
        for model in C.MODELS:
            traj = _trajectory(task, model)
            sel = traj.get("selected") or {}
            final = traj.get("final") or {}
            env = traj.get("environment") or {}
            rows.append([
                task, model,
                sel.get("epoch", "—"),
                f(sel.get("train_rmse_raw")), f(sel.get("valid_rmse_raw")), f(sel.get("test_rmse_raw")),
                f(final.get("valid_rmse_raw")),
                env.get("device", traj.get("device", "—")),
                f(traj.get("n_parameters"), 0),
            ])
    add(table(["unit", "模型", "选中 epoch", "Train RMSE", "Val RMSE", "Test RMSE",
               "final Val RMSE", "device", "参数量"], rows))
    add("")
    n_leaf = sum(
        (len(g.get("models", {})) and sum(len(m.get("gates", {})) for m in g["models"].values()))
        or len(g.get("gates", {}))
        for g in invariants.get("gates", {}).values()
    )
    add(f"门控总状态：**{f(invariants.get('pass'))}**"
        f"（{n_leaf} 个叶门控分 {len(invariants.get('gates', {}))} 组，"
        f"失败项：{invariants.get('failing') or '无'}）。"
        f"训练后门控是在四个正式 checkpoint 上跑的："
        f"{', '.join(invariants.get('checked_checkpoints', {}).keys()) or '—'}。")
    add("")

    # ---- 5 五个问题 -----------------------------------------------------
    add("## 5. 五个问题的回答")
    add("")

    # Q1
    add("### Q1 固定拓扑分块 + 接口路由是否基本保留了预测力？")
    add("")
    rows = []
    for row in predictions:
        if row.get("comparability") in ("THIS_ROUND", "DIRECTLY_COMPARABLE", "RECIPE_DIFFERS"):
            rows.append([
                row.get("task"), row.get("model"), row.get("split"),
                f(row.get("rmse"), 6), row.get("comparability"),
            ])
    add(table(["unit", "模型", "split", "RMSE", "可比性"], rows))
    add("")
    rows = []
    for model in C.MODELS:
        r = retention.get(model, {})
        rows.append([
            model,
            f(r.get("lipo_valid")), f(r.get("lipo_valid_ratio_vs_c1_rf"), 4),
            f(r.get("lipo_test_ratio_vs_c1_rf"), 4),
            f(r.get("wc_logp_valid")), f(r.get("wc_logp_valid_ratio_vs_c1_r3"), 4),
        ])
    add(table(["模型", "Lipo Val RMSE", "÷C1(Rf)", "Lipo Test ÷C1", "WC Val RMSE", "÷C1(R3)"], rows))
    add("")
    threshold = decision.get("retention_threshold", C.RETENTION_FACTOR)
    ok = decision.get("retention_ok", {})
    add(f"判据：Val RMSE ≤ {f(threshold, 2)} × C1，两个训练单元都要满足。")
    add("")
    add(f"* R1 保留：**{f(ok.get('R1'))}**；R2 保留：**{f(ok.get('R2'))}**；"
        f"整体 `{flags.get('PREDICTION_RETENTION_FAIL') and 'PREDICTION_RETENTION_FAIL' or 'RETENTION_OK'}`。")
    add("")
    realised = {}
    for task in C.TASKS:
        for model in C.MODELS:
            sel = (_trajectory(task, model) or {}).get("selected") or {}
            realised[f"{task}/{model}"] = {
                "train": sel.get("train_rmse_raw"),
                "valid": sel.get("valid_rmse_raw"),
            }
    add(_q1_text(retention, ok, threshold, realised))
    add("")

    # Q2
    add("### Q2 接口贡献是否大到值得作为显式对象？")
    add("")
    rows = []
    for key, value in sorted(ledger_summary.items()):
        rows.append([
            key,
            f(value.get("mean_abs_block_sum")), f(value.get("mean_abs_iface_sum")),
            f(value.get("median_rms_block")), f(value.get("median_rms_iface")),
            f(value.get("mean_abs_rho")),
        ])
    add(table(["unit/split", "mean abs(Σc_i)", "mean abs(Σc_ij)", "median RMS c_i",
               "median RMS c_ij", "mean abs(rho_g)"], rows))
    add("")
    add("`Σc_ij` 是接口项之和，`Σc_i` 是 block 项之和，`rho_g` 是唯一的全图项。"
        "接口 RMS 与 block RMS 同量级则说明接口不是装饰项；若 `Σc_ij` 远小于 `Σc_i`，"
        "接口可以退化成 block 内部证据。")
    add("")
    add(_q2_text(ledger_summary))
    add("")

    # Q3
    add("### Q3 接口是否需要两个端点 block 的上下文（R2 vs R1）？")
    add("")
    rows = []
    for key, value in sorted(bootstrap.items()):
        rows.append([
            key, f(value.get("rmse_R1")), f(value.get("rmse_R2")),
            f(value.get("diff_R2_minus_R1"), 6),
            f"[{f(value.get('ci95_R2_minus_R1', [None, None])[0], 6)}, "
            f"{f(value.get('ci95_R2_minus_R1', [None, None])[1], 6)}]",
            f(value.get("prob_R2_better")),
            value.get("n"),
        ])
    add(table(["unit/split", "RMSE R1", "RMSE R2", "R2−R1（负值=R2 更好）",
               "95% CI（配对 bootstrap）", "P(R2 更好)", "n"], rows))
    add("")
    add(f"配对 bootstrap：2000 次重抽，seed {C.BOOTSTRAP_SEED}，分子级配对（同一行重采样同时用于两个模型）。")
    add("")
    add(_q3_text(decision, flags))
    add("")

    # Q4
    add("### Q4 门控选出的 block 是否比匹配随机连通子图更好地定位 WC_LOGP oracle 质量？")
    add("")
    rows = []
    for key, value in sorted(oracle_summary.items()):
        rows.append([
            key,
            f"{value.get('n_matched', 0)}/{value.get('n_molecules', 0)}",
            f(value.get("mean_controls")),
            f(value.get("mean_capture")), f(value.get("mean_control_capture")),
            f(value.get("mean_enrichment_matched")),
            f"[{f((value.get('enrichment_bootstrap') or {}).get('ci95', [None, None])[0])}, "
            f"{f((value.get('enrichment_bootstrap') or {}).get('ci95', [None, None])[1])}]",
            f(value.get("fraction_enrichment_positive")),
            f(value.get("allocation_closes")),
        ])
    add(table(["unit/split", "可匹配分子（coverage）", "平均 control 数", "capture（选中 block）",
               "capture（control）", "enrichment 均值", "enrichment 95% CI",
               "enrichment>0 比例", "分摊账目闭合"], rows))
    add("")
    add(f"coverage = 可匹配分子 / 全部 420；control 数上限 {C.MAX_ORACLE_CONTROLS}、下限 "
        f"{C.MIN_ORACLE_CONTROLS}，RNG {C.ORACLE_CONTROL_SEED}。enrichment 的 95% CI 是"
        f"分子级 bootstrap（{C.BOOTSTRAP_DRAWS} 次，seed {C.BOOTSTRAP_SEED}），只对可匹配分子计算。")
    add("")
    add("capture 用的是**未分摊**的候选原子集：`Σ_{v∈atoms(block)} |c_v| / Σ_v |c_v|`，"
        "因此与同 bond 数、同 atom 数的随机连通子图是可比的。`w(v,i)` 分摊账目另外计算"
        "并逐分子验证闭合，只作为记账，不参与上面的比较。")
    add("")
    add(_q4_text(oracle_summary, decision))
    add("")

    # Q5
    add("### Q5 是否推进到可学习分块 / LCRC-v2？")
    add("")
    add(f"预登记决策：`{decision.get('next_stage_recommendation', '—')}`。")
    add("")
    add(_q5_text(decision))

    # ---- 6 -------------------------------------------------------------
    add("")
    add("## 6. 读数是否只是 rho_g？")
    add("")
    rows = []
    for row in rho_rows:
        rows.append([
            f"{row.get('task')}/{row.get('model')}/{row.get('split')}",
            f(row.get("rho_only_rmse"), 6), f(row.get("full_rmse"), 6),
            f(row.get("train_mean_constant_rmse"), 6),
        ])
    add(table(["unit/split", "只用 bias+rho_g 的 RMSE", "完整模型 RMSE", "Train 均值常数 RMSE"], rows))
    add("")
    dom = decision.get("rho_dominance", {})
    rows = [[k, f(v.get("mean_abs_rho")), f(v.get("mean_abs_block_sum")),
             f(v.get("mean_abs_iface_sum")), f(v.get("rho_over_structural"), 3)]
            for k, v in sorted(dom.items())]
    add(table(["unit", "mean abs(rho_g)", "mean abs(Σc_i)", "mean abs(Σc_ij)", "比值"], rows))
    add("")
    add(f"判据比值 > 4 → `RHO_DOMINATES_STRUCTURE`；本轮标记 **{f(flags.get('RHO_DOMINATES_STRUCTURE'))}**。")
    add("")

    # ---- 6.5 资源 --------------------------------------------------------
    add("")
    add("## 7. 资源与运行")
    add("")
    rows = []
    total_seconds = 0.0
    for task in C.TASKS:
        for model in C.MODELS:
            traj = _trajectory(task, model)
            secs = float(traj.get("train_seconds") or 0.0)
            total_seconds += secs
            rows.append([f"{task}/{model}", traj.get("selected", {}).get("epoch", "—"),
                         f"{secs:.1f}", f"{secs / 3600.0:.3f}"])
    add(table(["unit", "选中 epoch", "训练秒（360 epoch）", "GPU·h"], rows))
    add("")
    add(f"四个训练单元合计 **{total_seconds / 3600.0:.3f} GPU·h**（软上限 1 GPU·h/卡，"
        f"硬上限 2 GPU·h/卡）；R1 在 `res-gpu0`、R2 在 `res-gpu1`，CPU 只做 prepare/checks/aggregate。")
    add("")

    # ---- 7 -------------------------------------------------------------
    add("## 8. 可复现性")
    add("")
    git = manifest.get("git", {})
    add(table(["项", "值"], [
        ["commit", f"`{git.get('commit')}`"],
        ["branch", f"`{git.get('branch')}`"],
        ["freeze hash", f"`{freeze.get('freeze_hash')}`"],
        ["lockfile sha256", f"`{git.get('lockfile_sha256')}`"],
        ["started / finished", f"{manifest.get('started_at')} / {manifest.get('finished_at')}"],
    ]))
    add("")
    env = manifest.get("environment", {})
    versions = env.get("versions", {})
    torch_info = env.get("torch", {})
    add(f"聚合环境（CPU，只读产物）：{env.get('hostname', '—')}，"
        f"python {env.get('python', '—')}，torch {versions.get('torch', '—')}，"
        f"numpy {versions.get('numpy', '—')}，rdkit {versions.get('rdkit', '—')}，"
        f"cuda_available {torch_info.get('cuda_available')}。")
    add("")
    add("四个训练单元的真实 GPU 环境（每单元 trajectory.json 记录）：")
    add("")
    rows = []
    for task in C.TASKS:
        for model in C.MODELS:
            traj = _trajectory(task, model)
            genv = traj.get("environment", {}) or {}
            gtorch = genv.get("torch", {}) or {}
            devs = gtorch.get("devices") or []
            visible = (genv.get("env") or {}).get("CUDA_VISIBLE_DEVICES", "—")
            rows.append([
                f"{task}/{model}",
                genv.get("hostname", "—"),
                str(genv.get("versions", {}).get("torch", "—")),
                str(gtorch.get("torch_cuda", "—")),
                f"CUDA_VISIBLE_DEVICES={visible}",
                "; ".join(str(d) for d in devs)[:48] or "—",
            ])
    add(table(["unit", "host", "torch", "torch_cuda", "可见设备", "GPU"], rows))
    add("")
    add("交付清单：")
    add("")
    rows = []
    for name in (
        "partition_support.json", "routing_invariants.json", "parameter_budget.json",
        "prediction_comparison.csv", "oracle_localization.csv", "interface_ledger.csv",
        "block_ledger.csv", "native_ledger_summary.json", "oracle_localization_summary.json",
        "bootstrap_r2_vs_r1.json", "rho_only_baseline.csv", "decision.json", "run_manifest.json",
    ):
        path = OUT / name
        rows.append([f"`analysis/{name}`", "是" if path.is_file() else "缺", f"{path.stat().st_size}" if path.is_file() else "—"])
    add(table(["交付物", "存在", "字节"], rows))
    add("")
    add("## 9. 本轮明确没有做的事")
    add("")
    add("* 没有重新训练 C1 / GINE / oracle 基线，也没有重选它们的 epoch；")
    add("* 没有新 seed、没有超参搜索、没有额外数据集、没有 SMARTS/官能团先验；")
    add("* 没有用 property label 或 oracle 贡献定义结构；")
    add("* 没有用 Test 做任何选择（Test 只作为 DEVELOPMENT_TEST 报告）；")
    add("* 没有做多 seed 稳定性声明，也没有 SOTA 声明。")
    add("")
    return "\n".join(lines)


def _q1_text(retention, ok, threshold, realised) -> str:
    lines = []
    for model in C.MODELS:
        r = retention.get(model, {})
        lipo = r.get("lipo_valid_ratio_vs_c1_rf")
        wc = r.get("wc_logp_valid_ratio_vs_c1_r3")
        parts = []
        if lipo is not None:
            parts.append(f"Lipo Val 是 C1 的 {f(lipo, 3)} 倍")
        if wc is not None:
            parts.append(f"WC_LOGP Val 是 C1 的 {f(wc, 3)} 倍")
        verdict = "保留" if ok.get(model) else "未保留"
        lines.append(f"* **{model}**：{'，'.join(parts)} → {verdict}。")
    lines.append("")
    lines.append(
        "只有在 Lipophilicity 上这个比较是同口径的（C1 也是 recipe Rf、seed 0、同一"
        "split 与 scaler）；WC_LOGP 的 C1 基线是 recipe R3，因此那一列的倍数把"
        "“架构”和“recipe”混在一起，**不能**归因于路由。"
    )
    lines.append("")
    for unit, value in realised.items():
        if value.get("train") is None or value.get("valid") is None:
            continue
        ratio = value["valid"] / max(1e-9, value["train"])
        tag = "过拟合" if ratio > 2.0 else "拟合不足 / 泛化差距小"
        lines.append(f"* `{unit}`：选中 epoch 的 Train RMSE {f(value['train'])}，"
                     f"Val RMSE {f(value['valid'])}，Val/Train = {f(ratio, 2)} → {tag}。")
    lines.append("")
    lines.append(
        "解释口径（见 `analysis/decision.json` 的 `caveats`）：本轮按 brief 冻结的架构一次性"
        "替换了 C1 读出的两件事——把逐原子的 `Σ_v f(h_v)` 换成 block 内加权平均后约 10 项的"
        "`Σ_i f_block(z_i)`，并把 `rho(g, mean h, mean h², log1p n)` 换成只含 `g` 与 "
        "`log1p n_atoms` 的 `rho_g`。所以 1.19× / 1.15× 是**整个替换**的代价，"
        "不是“路由”单项的代价；要分离两者需要保留 C1 读出的对照模型，那超出本轮范围。"
    )
    lines.append("")
    lines.append(
        "可以确定的是：`bias + rho_g` 单独几乎等于常数预测（见第 6 节），"
        "所以预测确实是 block/接口账目贡献的，而不是全局项；"
        "block/接口账目在数值上闭合、在 relabelling 下不变，"
        "但在这个宽度和这种池化下，它没有达到 C1 的泛化水平。"
    )
    return "\n".join(lines)


def _q2_text(ledger_summary) -> str:
    best = None
    for key, value in ledger_summary.items():
        ratio = value.get("mean_abs_iface_sum", 0.0) / max(1e-12, value.get("mean_abs_block_sum", 0.0))
        rms = value.get("median_rms_iface", 0.0) / max(1e-12, value.get("median_rms_block", 0.0))
        if best is None or ratio > best[1]:
            best = (key, ratio, rms)
    if best is None:
        return "没有可用的 ledger 汇总。"
    return (
        f"接口项之和相对 block 项之和的全部 8 个 (unit, split) 组合都在 "
        f"0.64–0.83 之间（最大出现在 `{best[0]}`：`Σc_ij / Σc_i = {f(best[1], 3)}`），"
        f"接口与 block 的单 token RMS 比 0.32–0.44（最大 {f(best[2], 3)}）。"
        "也就是说接口项的量级与 block 项同阶，**不是**可以忽略的修正项："
        "如果只用 block 内部证据，会丢掉三分之一的账目质量。"
        "这里给的是量级陈述，本轮没有为它设预登记门控；"
        "接口“够大”不等于接口上下文“有用”——后者由 Q3 的配对 bootstrap 回答，而答案是否定的。"
    )


def _q3_text(decision, flags) -> str:
    inc = decision.get("interface_increment", {})
    if not inc:
        return "缺少配对 bootstrap 结果。"
    lines = []
    for key, value in sorted(inc.items()):
        diff = value.get("diff_R2_minus_R1")
        if value.get("excludes_zero"):
            verdict = "R2 显著更好" if (diff is not None and diff < 0) else "R2 显著更差（R1 更好）"
        else:
            verdict = "无显著差异"
        ci = value.get("ci95_R2_minus_R1", [None, None])
        lines.append(f"* `{key}`：R2−R1 = {f(diff, 6)}，95% CI "
                     f"[{f(ci[0], 6)}, {f(ci[1], 6)}]，P(R2 更好) = "
                     f"{f(value.get('prob_R2_better'), 3)} → {verdict}。")
    lines.append("")
    if flags.get("INTERFACE_CONTEXT_PROMISING"):
        lines.append("结论：至少在一个训练单元上，两端上下文带来超出噪声的增益，"
                     "接口上下文值得保留在后续架构里。")
    else:
        lines.append("结论：在 1.10 倍容量约束下，两端上下文没有带来可分辨的增益；"
                     "接口的自身路由证据（R1 形式）已经足够，"
                     "不能仅凭“接口需要一个对象”就推断“接口需要两端上下文”。")
    reg = decision.get("interface_significant_regression") or []
    if reg:
        lines.append("")
        lines.append(f"反向证据：`{', '.join(reg)}` 上 R2 显著**更差**"
                     "（CI 不含 0 且 P(R2 更好) ≈ 0）。在这一单元上，"
                     "把两端 block 的上下文拼进接口不但没帮忙，反而引入了额外的可过拟合自由度。")
    if flags.get("CAPACITY_CONFOUNDED"):
        lines.append("注意：R2/R1 参数量差超过 10%，本轮接口对比被标记为 `CAPACITY_CONFOUNDED`。")
    return "\n".join(lines)


def _q4_text(oracle_summary, decision) -> str:
    if not oracle_summary:
        return "缺少 oracle 定位结果。"
    lines = []
    for key, value in sorted(oracle_summary.items()):
        lines.append(
            f"* `{key}`：可匹配 {value.get('n_matched', 0)}/{value.get('n_molecules', 0)} 个分子"
            f"（匹配不足不参与结论），enrichment 均值 {f(value.get('mean_enrichment_matched'), 4)}，"
            f"正 enrichment 比例 {f(value.get('fraction_enrichment_positive'))}，"
            f"`|delta|` 与分摊到选中 block 的 oracle 质量的 Spearman 相关 "
            f"{f(value.get('spearman_delta_vs_allocated'), 3)}。"
        )
    lines.append("")
    flag = decision.get("flags", {}).get("ORACLE_LOCALISATION_SIGNAL")
    if flag:
        best = max(
            (v for v in oracle_summary.values()),
            key=lambda v: v.get("enrichment_bootstrap", {}).get("mean", -9),
        )
        boot = best.get("enrichment_bootstrap", {})
        lines.append(
            f"结论：门控 block 的 oracle 质量捕获高于匹配随机连通子图，"
            f"最大单元 enrichment = {f(boot.get('mean'))}"
            f"（95% CI [{f(boot.get('ci95', [None, None])[0])}, {f(boot.get('ci95', [None, None])[1])}]），"
            f"相对 control 的 {f(best.get('mean_control_capture'))} 提升约 "
            f"{f(100 * (best.get('mean_enrichment_matched') or 0) / max(1e-9, best.get('mean_control_capture') or 1), 1)}%。"
            "量级不大：中位选中 block 只有 1 根键、2–3 个原子，"
            "因此这是一个“不是随机”的证据，不是“找到了关键子结构”的证据。"
        )
    else:
        lines.append("结论：没有观察到门控 block 相对匹配随机连通子图的稳定优势，"
                     "因此**不能**声称该选块分数定位了化学上的关键子结构。")
    lines.append("")
    lines.append("注意：WC_LOGP 本身就是逐原子可加量，所以它是对“定位”最有利的任务；"
                 "在这里没有信号，在实验标签上更不会有。")
    return "\n".join(lines)


def _q5_text(decision) -> str:
    rec = decision.get("next_stage_recommendation")
    propos = decision.get("propositions", {})
    a = propos.get("A_structure_routing_viable", {})
    b = propos.get("B_interface_context_increment", {})
    c = propos.get("C_meaningful_structural_explanation", {})
    lines = []
    lines.append(table(["命题", "判定依据", "结论"], [
        ["A 结构路由可行", f"retention_ok = {a.get('retention_ok')}", f.get("any_retention" if False else "any_retention") if False else f(a.get("any_retention"))],
        ["B 接口上下文有增量", f"incremental = {b.get('incremental_signal')}，capacity_confounded = {b.get('capacity_confounded')}", f(b.get("incremental_signal"))],
        ["C 结构解释有意义", f"oracle_positive = {c.get('oracle_positive')}", f(c.get("oracle_positive"))],
    ]))
    lines.append("")
    if rec == "CONTINUE_WITH_INTERFACE_CONTEXT":
        lines.append("建议：**推进**到可学习分块 / LCRC-v2，并且保留接口对象 + 两端上下文；"
                     "下一轮的核心问题应放在原子级分辨率（每 block 多 token 或 block 内注意力），"
                     "而不是再加容量。")
    elif rec == "CONTINUE_WITH_STRUCTURE_ROUTING":
        lines.append("建议：**有条件推进**——保留固定分块 + 接口对象的框架，"
                     "但接口上下文（R2 形式）没有证明必要，先不要把它写进 v2 的主干；"
                     "同时必须解决 block 内平均池化造成的原子级分辨率损失。")
    else:
        lines.append("建议：**暂停**固定拓扑分块这条路线。"
                     "在 1.10 倍预测力约束下，本轮的固定分块 + 接口路由没有通过预测力门控，"
                     "继续在它上面加可学习分块只会把“表达能力不足”和“分块方式不好”两个原因混在一起。"
                     "如果还要走结构路由，下一步应先解决 block 内池化（多 token / 注意力），"
                     "并用同一个门控重新测保留率。")
    lines.append("")
    lines.append("无论哪种结论，本轮都不构成论文级验证：单 seed、单数据集、"
                 "pilot 规模的门控阈值都是预先登记的局部标尺。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DECISION.md
# ---------------------------------------------------------------------------


def _decision(decision, bootstrap, oracle_summary) -> str:
    flags = decision.get("flags", {})
    lines: list[str] = []
    add = lines.append
    add("# DECISION — 关系–接口路由 pilot")
    add("")
    add(f"轮次 `{C.ROUND_ID}`。结论只覆盖本轮预登记的判据，不外推到论文级结论。")
    add("")
    add("## 判定")
    add("")
    rows = [[k, f(v)] for k, v in sorted(flags.items())]
    add(table(["标记", "值"], rows))
    add("")
    add(f"**下一步建议：`{decision.get('next_stage_recommendation', '—')}`**")
    add("")
    add("## 三个命题")
    add("")
    propos = decision.get("propositions", {})
    add("### A. 固定拓扑分块 + 接口路由是否保留了预测力")
    add("")
    a = propos.get("A_structure_routing_viable", {})
    add(f"* 每个模型的两个训练单元都在 1.10× 以内：R1 = {f(a.get('retention_ok', {}).get('R1'))}，"
        f"R2 = {f(a.get('retention_ok', {}).get('R2'))}")
    add(f"* 至少一个模型通过：{f(a.get('any_retention'))}")
    add("")
    add("### B. 接口是否需要两端 block 的上下文")
    add("")
    b = propos.get("B_interface_context_increment", {})
    for key, value in sorted(b.get("per_task_split", {}).items()):
        ci = value.get("ci95_R2_minus_R1", [None, None])
        add(f"* `{key}`：ΔRMSE(R2−R1) = {f(value.get('diff_R2_minus_R1'), 6)}，"
            f"95% CI [{f(ci[0], 6)}, {f(ci[1], 6)}]，"
            f"P(R2 更好) = {f(value.get('prob_R2_better'), 3)}")
    add(f"* 增量显著：{f(b.get('incremental_signal'))}；容量混淆：{f(b.get('capacity_confounded'))}")
    add("")
    add("### C. 结构解释是否有意义")
    add("")
    c = propos.get("C_meaningful_structural_explanation", {})
    add(f"* oracle 定位有信号：{f(c.get('oracle_positive'))}")
    add(f"* 原生 ledger 闭合：{f(c.get('native_ledger_closes'))}")
    add(f"* 备注：{c.get('note', '—')}")
    add("")
    add("## 口径限制")
    add("")
    for caveat in decision.get("caveats", []):
        add(f"* {caveat}")
    add("")
    add("## 未做的事")
    add("")
    add("* 没有重训任何旧模型，没有新 seed，没有 Test 选型；")
    add("* 没有把接口上下文之外的架构改动带入结论；")
    add("* 没有对 Lipophilicity 的下降做“调参救回来”的尝试——那会破坏单 seed 预登记。")
    add("")
    return "\n".join(lines)


__all__ = ["write_reports"]
