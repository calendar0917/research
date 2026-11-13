"""REPORT.md generator.

The numbers come from ``analysis/*.json``; the interpretation text is written
here so that the same report can be regenerated after any re-analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lcpr_common as C


def _load(name: str, default: Any = None) -> Any:
    path = C.ANALYSIS / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _f(x: Any, fmt: str = "{:.4f}") -> str:
    try:
        return fmt.format(float(x))
    except Exception:
        return "" if x is None else str(x)


def _pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):+.2f}%"
    except Exception:
        return ""


def gpu_hours() -> dict[str, Any]:
    """Charged GPU hours from the queue logs (authoritative for cost)."""
    total = 0.0
    jobs: list[dict[str, Any]] = []
    for path in sorted(list(C.ANALYSIS.glob("queue_*.jsonl"))
                       + list((C.ANALYSIS / "queue_logs").glob("queue_*.jsonl"))):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gh = float(rec.get("gpu_hours_charged") or 0.0)
            total += gh
            jobs.append({"purpose": rec.get("purpose"), "job": rec.get("job"),
                         "status": rec.get("status"), "gpu_hours": gh,
                         "wall_seconds": rec.get("wall_seconds")})
    return {"total_gpu_hours": total, "jobs": jobs}


def build() -> str:
    rows = (_load("run_table.json") or {}).get("runs", [])
    trace = _load("selection_trace.json", {}) or {}
    fam = _load("family_summary.json", {}) or {}
    boot = _load("paired_comparisons.json", {}) or {}
    tails = _load("tail_summary.json", {}) or {}
    mech = _load("mechanism_summary.json", {}) or {}
    budget = gpu_hours()
    idx = {(r["candidate"], r["dataset"]): r for r in rows if not r.get("smoke")}
    lipo = {c: r for (c, d), r in idx.items() if d == "Lipophilicity"}
    base_name = trace.get("lipophilicity_base")
    base = lipo.get(base_name or "", None)

    xgb = {"valid_rmse": 0.6532, "test_rmse": 0.6477}
    dmp = {c: r for c, r in lipo.items() if str(c).startswith("DMPNN")}
    dmp_best = min(dmp.values(), key=lambda r: r["valid"]["rmse"]) if dmp else None

    lines: list[str] = []
    a = lines.append
    a("# LCRC 性能路线（8 小时 / public122 / 种子 0）")
    a("")
    a("> 本轮结论均为 `DEVELOPMENT_TEST` / `TEST_USED_FOR_ROUTE_SELECTION`：Test 全量参与每轮评估并参与路线筛选，"
      "因此不是独立确认，也不是跨种子稳定性证据。所有新训练 seed = 0，单种子。")
    a("")
    a(f"- 协议：`PROTOCOL.md`；冻结配置：`configs/experiment.yaml`；协议 ID `{C.ROUND_ID}`")
    a(f"- 新训练次数（不含 ≤2 轮 smoke）：{len([r for r in rows if not r.get('smoke')])}")
    a(f"- 计费 GPU 小时（wall × GPU，含评估/失败/取消）：{budget['total_gpu_hours']:.2f} A100·h"
      f"（软上限 {C.GPU_SOFT_CAP_HOURS}，硬上限 16）")
    a("")

    # ---------------- Q1 ----------------
    a("## Q1 当前 C1 与简单基线 / 合理 D-MPNN 参考的距离，以及便宜改动能解释多少")
    a("")
    if base is not None:
        recipe_tbl = (trace.get("recipes") or {}).get("table") or []
        a("### 1.1 配方筛选（C1 结构不变，只改 lr / weight decay）")
        a("")
        a(_table(["recipe", "peak_lr", "wd", "Valid RMSE", "Test RMSE", "Valid MAE", "Test MAE"],
                 [[r["recipe"],
                   {"R0": "3e-4", "R1": "1e-3", "R2": "3e-4", "R3": "1e-3"}[r["recipe"]],
                   {"R0": "1e-3", "R1": "1e-3", "R2": "1e-2", "R3": "1e-2"}[r["recipe"]],
                   _f(r["valid_rmse"]), _f(r["test_rmse"]),
                   _f(r.get("valid_mae")), _f(r.get("test_mae"))] for r in recipe_tbl]))
        a("")
        a(f"冻结规则选出 `R* = {trace['recipes'].get('best')}`（仅按 Valid RMSE）。"
          f"本轮所有 C1 形状的候选臂都在 `R*` 下训练，基线统一为 `{base_name}`。")
        a("")
        a("### 1.2 与参考坐标的距离（Lipophilicity random122，同 split 同行）")
        a("")
        ref_rows = [[ "本冻结 C1（历史 checkpoint，replay 校验）", "", _f(0.702536), _f(0.576167), "—"],
                    [f"{base_name}（本轮重训，R*）", _f(base["parameters"], "{:.0f}"),
                     _f(base["valid"]["rmse"]), _f(base["test"]["rmse"]), "0"],
                    ["ECFP4+RDKit+XGBoost（冻结外部参考）", "—", _f(xgb["valid_rmse"]),
                     _f(xgb["test_rmse"]), "—"],
                    ["D-MPNN（chemprop v2.2.1，最优 Valid 峰 lr）", 
                     _f(dmp_best["parameters"], "{:.0f}") if dmp_best else "—",
                     _f(dmp_best["valid"]["rmse"]) if dmp_best else "—",
                     _f(dmp_best["test"]["rmse"]) if dmp_best else "—",
                     _f(dmp_best["gpu_hours"], "{:.2f} h") if dmp_best else "—"]]
        a(_table(["坐标", "参数量", "Valid RMSE", "Test RMSE", "GPU·h"], ref_rows))
        a("")
        if dmp_best and base is not None:
            gap_v = (base["valid"]["rmse"] - dmp_best["valid"]["rmse"]) / base["valid"]["rmse"]
            gap_t = (base["test"]["rmse"] - dmp_best["test"]["rmse"]) / base["test"]["rmse"]
            a(f"- 与 D-MPNN 的相对差距：Valid {_pct(gap_v)}，Test {_pct(gap_t)}（正数=本模型更差）。")
        if base is not None:
            gv = (base["valid"]["rmse"] - xgb["valid_rmse"]) / base["valid"]["rmse"]
            gt = (base["test"]["rmse"] - xgb["test_rmse"]) / base["test"]["rmse"]
            a(f"- 与 ECFP+XGBoost 的相对差距：Valid {_pct(gv)}，Test {_pct(gt)}（正数=本模型更差）。")
        a("")

        a("### 1.3 便宜改动能解释多少（分解到配方 / 预算 / 容量）")
        a("")
        rows_dec = []
        for cand in ("C1_R0", "C1_R1", "C1_R2", "C1_R3", "WIDE_LOCAL", "LONG"):
            rec = lipo.get(cand)
            if rec is None:
                continue
            d_v = (rec["valid"]["rmse"] - base["valid"]["rmse"]) / base["valid"]["rmse"]
            d_t = (rec["test"]["rmse"] - base["test"]["rmse"]) / base["test"]["rmse"]
            rows_dec.append([cand, rec["recipe"], _f(rec["parameters"], "{:.0f}"),
                             _f(rec["valid"]["rmse"]), _f(rec["test"]["rmse"]),
                             _pct(-d_v), _pct(-d_t),
                             _f(rec.get("gpu_hours"), "{:.2f}")])
        a(_table(["臂", "recipe", "参数", "Valid RMSE", "Test RMSE", "Valid 增益", "Test 增益", "GPU·h"],
                 rows_dec))
        a("")
        a("正增益 = 相对 `%s` 更好。`LONG` 与 `R*` 的差异只反映训练预算（720 vs 360 epoch，"
          "**从零重训**，不是同一条轨迹的前缀延长），`WIDE_LOCAL` 的差异只反映 local-MLP 容量（192→384），"
          "两者都保留了 step 0 与 C1 完全一致的初始化。" % base_name)
        a("")

    # ---------------- Q2 ----------------
    a("## Q2 两个机制行为是否能转成真实预测增益")
    a("")
    a("### 2.1 dropout/二阶池化的训练-评估偏移（M 家族）")
    a("")
    m_arms = [c for c in lipo if c.startswith("M_")]
    if m_arms:
        a(_table(["臂", "定义", "Valid RMSE", "Test RMSE", "Valid 增益", "Test 增益", "清晰信号"],
                 [[c, {"M_UNBIASED": "d=m·h/(1-p)，训练期二阶矩乘 (1-p)",
                       "M_CLEANPOOL": "rho 用干净 h 的均值/二阶矩",
                       "M_NODROP": "p=0（正则强度对照）"}.get(c, ""),
                   _f(lipo[c]["valid"]["rmse"]), _f(lipo[c]["test"]["rmse"]),
                   _gain_str(base, lipo[c], "valid"), _gain_str(base, lipo[c], "test"),
                   _yes(fam, c)] for c in sorted(m_arms)]))
        a("")
    if "m_panel" in mech:
        mp = mech["m_panel"].get("splits", {})
        a(_table(["split", "mask/clean 二阶矩比", "MC 预测均值 − eval 均值", "MC 标准差均值"],
                 [[s, _f(v["summary"]["mean_h2_ratio_mask_over_clean"], "{:.3f}"),
                   _f(v["summary"]["pred_mc_mean_minus_eval_mean"], "{:+.4f}"),
                   _f(v["summary"]["pred_mc_std_mean"], "{:.4f}")] for s, v in mp.items()]))
        a("")
        a("这是**同一份权重**下 256 个固定 mask 的蒙特卡洛统计：量出的是「训练期池化统计与 eval 期的"
          "系统性差别」，而不是网络学到了什么。")
        a("")
    a("### 2.2 S2 配对（J）块在低支撑行上的泛化（S 家族）")
    a("")
    s_arms = [c for c in lipo if c.startswith("S_")]
    if s_arms:
        a(_table(["臂", "定义", "Valid RMSE", "Test RMSE", "Valid 增益", "Test 增益", "清晰信号"],
                 [[c, {"S_SELECT_K1": "select, κ", "S_UNIFORM_K1": "uniform, a_bar(κ)",
                       "S_SELECT_K4": "select, 4κ", "S_UNIFORM_K4": "uniform, a_bar(4κ)"}.get(c, ""),
                   _f(lipo[c]["valid"]["rmse"]), _f(lipo[c]["test"]["rmse"]),
                   _gain_str(base, lipo[c], "valid"), _gain_str(base, lipo[c], "test"),
                   _yes(fam, c)] for c in sorted(s_arms)]))
        a("")
    if "s_panel" in mech:
        sp = mech["s_panel"].get("splits", {})
        a(_table(["split", "ΔJ 平均（规则→边缘）", "J 完全不变的行占比",
                  "SSE 变化（规则→边缘）", "|Δpred| 平均"],
                 [[s, _f(v["summary"]["delta_j_norm_mean"], "{:.4f}"),
                   _f(v["summary"]["delta_j_norm_zero_frac"], "{:.3f}"),
                   _f(v["summary"]["delta_sse_alpha0_minus_1"], "{:+.4f}"),
                   _f(v["summary"]["delta_pred_abs_mean"], "{:.4f}")] for s, v in sp.items()]))
        a("")

    # ---------------- Q3 ----------------
    a("## Q3 计算式任务相关性（WC_LOGP）作为辅助监督")
    a("")
    a_arms = [c for c in lipo if c.startswith("A_")]
    if a_arms:
        a(_table(["臂", "辅助目标", "Valid RMSE", "Test RMSE", "Valid 增益", "Test 增益", "清晰信号"],
                 [[c, {"A_LOGP": "WC_LOGP", "A_MW": "MolWt（不变对照）",
                       "A_SHUFFLE": "WC_LOGP 行置换（纯正则对照）"}.get(c, ""),
                   _f(lipo[c]["valid"]["rmse"]), _f(lipo[c]["test"]["rmse"]),
                   _gain_str(base, lipo[c], "valid"), _gain_str(base, lipo[c], "test"),
                   _yes(fam, c)] for c in sorted(a_arms)]))
        a("")
        a("判据是**三者之间的排序**，而不是任何一支是否「比基线好一点」：只有当 `A_LOGP` 同时优于 "
          "`A_MW`（同参数量、同计算量的无关标量）和 `A_SHUFFLE`（同目标、零信息）时，"
          "才能说「任务相关性本身」起了作用；否则一切只是优化负担/正则。")
        a("")

    # ---------------- Q4 ----------------
    a("## Q4 增益是单点/Lipo 专属，还是可迁移；哪些路线停止、哪些仍未测")
    a("")
    transfer_rows = []
    for (c, d), rec in sorted(idx.items()):
        if d == "Lipophilicity":
            continue
        conflict = idx.get((c, "Lipophilicity"))
        transfer_rows.append([d, c, rec["recipe"], _f(rec["valid"].get("rmse", rec["valid"].get("roc_auc"))),
                              _f(rec["test"].get("rmse", rec["test"].get("roc_auc"))),
                              _f(conflict["valid"]["rmse"]) if conflict and conflict["valid"].get("rmse") else ""])
    if transfer_rows:
        a(_table(["任务", "臂", "recipe", "Valid", "Test", "Lipo Valid（同一臂）"], transfer_rows))
        a("")
    a("### 尾部分布（Lipo Test，行集合仅由基线误差定义）")
    a("")
    if tails.get("arm_tail_test"):
        rows_tail = []
        for c, v in tails["arm_tail_test"].items():
            k1 = v.get("exclude_top_1") or v.get("exclude_top_0.01") or {}
            k5 = v.get("exclude_top_5") or v.get("exclude_top_0.05") or {}
            rows_tail.append([c, _f(v["full_rmse"]), _pct(v.get("gain_full")),
                              _f(k1.get("arm_rmse")), _pct(k1.get("gain")),
                              _f(k5.get("arm_rmse")), _pct(k5.get("gain")),
                              "是" if (k1.get("tail_only_gain") or k5.get("tail_only_gain")) else "否"])
        a(_table(["臂", "全量 RMSE", "全量增益", "去最大 1%", "增益 1%", "去最大 5%", "增益 5%", "仅尾部增益"],
                 rows_tail))
        a("")

    a("## 配对 bootstrap（Lipo，2000 次，seed %s）" % C.BOOTSTRAP_SEED)
    a("")
    bt = (boot or {}).get("splits", {})
    if bt:
        rr = []
        for split, per in bt.items():
            for cand, res in per.items():
                rr.append([split, cand, _f(res["delta"], "{:+.5f}"),
                           f"[{_f(res['ci_low'], '{:+.5f}')}, {_f(res['ci_high'], '{:+.5f}')}]",
                           _f(res["p_two_sided"], "{:.3f}"), res.get("n_clusters")])
        a(_table(["split", "臂", "ΔRMSE（基线−臂，正=更好）", "95% CI", "p（双侧）", "簇数"], rr))
        a("")
        a("区间只条件于单种子与本轮的候选/配方/Test 筛选，不构成独立确认。")
        a("")

    bench = _load("bench.json", {}) or {}
    if bench:
        a("## 成本/实用性坐标（同一 Test 行集，batch 128，10 batch 预热 + 5 次重复）")
        a("")
        c1 = bench.get("c1_inference", {})
        dm = bench.get("dmpnn_inference", {})
        a(_table(["模型", "参数量", "Test 推理中位耗时 (s)", "范围 (s)", "训练 wall (min)"],
                 [["C1（本轮基线）", _f(bench.get("parameters", {}).get("c1"), "{:.0f}"),
                   _f(c1.get("median_seconds"), "{:.4f}"),
                   f"[{_f(c1.get('min_seconds'), '{:.4f}')}, {_f(c1.get('max_seconds'), '{:.4f}')}]",
                   _f(base.get("wall_seconds"), "{:.1f}") if base else ""],
                  ["D-MPNN", _f(bench.get("parameters", {}).get("dmpnn"), "{:.0f}"),
                   _f(dm.get("median_seconds"), "{:.4f}") if "median_seconds" in dm else dm.get("error", ""),
                   f"[{_f(dm.get('min_seconds'), '{:.4f}')}, {_f(dm.get('max_seconds'), '{:.4f}')}]"
                   if "median_seconds" in dm else "",
                   _f((dmp_best or {}).get("wall_seconds"), "{:.1f}") if dmp_best else ""]]))
        a("")
        a("两个模型输入与配方不同，这只是**实用性坐标**，不是架构的受控比较。")
        a("")

    a("## 预算与运行时")
    a("")
    a(_table(["用途", "任务", "状态", "GPU·h", "wall (s)"],
             [[j["purpose"], (j["job"] or {}).get("candidate", "") + ":" + (j["job"] or {}).get("dataset", ""),
               j["status"], _f(j["gpu_hours"], "{:.3f}"), _f(j["wall_seconds"], "{:.0f}")]
              for j in budget["jobs"]]))
    a("")
    return "\n".join(lines) + "\n"


def _gain_str(base: Mapping[str, Any] | None, rec: Mapping[str, Any], split: str) -> str:
    if base is None:
        return ""
    try:
        return _pct((base[split]["rmse"] - rec[split]["rmse"]) / base[split]["rmse"])
    except Exception:
        return ""


def _yes(fam: Mapping[str, Any], cand: str) -> str:
    arm = (fam.get("arms") or {}).get(cand)
    if not arm:
        return ""
    return "是" if arm.get("clear_signal") else "否"


def run(stage: str = "all") -> Path:
    text = build()
    path = C.TRACK / "REPORT.md"
    path.write_text(text, encoding="utf-8")
    C.log(f"[report] wrote {path.relative_to(C.TRACK)} ({len(text)} chars)")
    return path
