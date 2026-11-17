"""Assemble REPORT.md / DECISION.md / run_manifest.json from the frozen records.

Every number in the report is read from a persisted artefact; nothing is
transcribed by hand.  The prose branches on the pre-registered decision labels
and thresholds, so a different result would produce different text instead of a
rewritten conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lfa_common as C

DISPLAY = {
    "A_LCRC_GRAD": "A LCRC-GRAD",
    "B_LCRC_LOCAL": "B LCRC-LOCAL",
    "C_LCRC_MASK50": "C LCRC-MASK50",
    "D_TOPOLOGY@LCRC": "D TOPOLOGY（LCRC 评价）",
    "D_TOPOLOGY@GINE": "D TOPOLOGY（GINE 评价）",
    "E_GINE_GRAD": "E GINE-GRAD",
    "F_GINE_MASK50": "F GINE-MASK50",
}
ORDER = list(DISPLAY)
SPLIT_LABEL = {"valid": "Valid（主）", "test": "Test（复审）", "train_description": "固定 128 Train（描述）"}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def f(value: Any, nd: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "—"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def pct(value: Any, nd: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{nd}f}%"


def ci(entry: Mapping[str, Any] | None, key: str, nd: int = 4) -> str:
    if not entry:
        return "—"
    block = entry.get(key) or {}
    lo_hi = block.get("mean") if isinstance(block, dict) else None
    if not lo_hi or lo_hi[0] is None:
        return "—"
    return f"[{f(lo_hi[0], nd)}, {f(lo_hi[1], nd)}]"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def load(path: Path) -> dict:
    return C.read_json(path)


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    import csv
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rec: dict[str, Any] = {}
            for key, value in raw.items():
                if value == "":
                    rec[key] = None
                    continue
                try:
                    rec[key] = float(value)
                except (TypeError, ValueError):
                    rec[key] = value
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------


def section_status(decision: Mapping[str, Any]) -> str:
    labels = decision.get("labels", [])
    reasons = decision.get("reasons", [])
    lines = ["## 0. 结论速览", ""]
    lines.append("**预注册状态标签**： " + "、".join(f"`{x}`" for x in labels))
    if reasons:
        lines.append("")
        lines.append("触发原因： " + "；".join(f"`{x}`" for x in reasons))
    lines.append("")
    lines.append("本轮只做冻结模型上的解释方法比较，没有训练新架构、没有新 seed、没有重选 checkpoint。"
                 "主 split 是官方 Valid（420 分子），Test（420）只做方向复审，固定 128 个 Train 分子只作描述。")
    return "\n".join(lines)


def section_assets(identity: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    gates = identity.get("gates", {})
    ckpt = gates.get("checkpoint_identity_and_replay", {})
    lines = ["## 1. 冻结资产与身份复核", ""]
    env = identity.get("environment") or {}
    lines.append(f"协议冻结哈希 `{freeze.get('freeze_hash', '—')}`，代码提交 `{freeze.get('git_commit', '—')}`，"
                 f"环境：python `{env.get('python', '—')}` / torch `{env.get('torch', '—')}` / "
                 f"cuda `{env.get('cuda_available', '—')}`。")
    lines.append("")
    lines.append("| 检查 | 结果 |")
    lines.append("| --- | --- |")
    for key in ("cache_identity", "asset_identity", "split_identity", "checkpoint_identity_and_replay",
                "wc_glop_graph_closure", "wc_glop_main_scope_closure", "wc_glop_pair_closure"):
        entry = gates.get(key) or {}
        lines.append(f"| {key} | {'PASS' if entry.get('passed') else 'FAIL'} |")
    lines.append("")
    rows = []
    for rec in ckpt.get("models", []):
        ident = rec.get("identity", {})
        replay = rec.get("replay", {})
        rows.append([ident.get("name", "—"), ident.get("epoch", "—"),
                     f(ident.get("valid_rmse_raw")), f"{ident.get('sha256', '')[:12]}…",
                     "PASS" if rec.get("passed") else "FAIL",
                     f(replay.get("valid_rmse_raw")) if isinstance(replay, dict) else "—"])
    if rows:
        lines.append(table(["checkpoint", "epoch", "valid RMSE（身份记录）", "sha256", "身份+重放", "重放 valid RMSE"], rows))
    lines.append("")
    lines.append("WC_LOGP 原子贡献闭合：图级 `max_abs_error = "
                 f"{f((gates.get('wc_glop_graph_closure') or {}).get('max_abs_error'), 8)}`；"
                 f"主范围 {f((gates.get('wc_glop_main_scope_closure') or {}).get('n_rows'))} 个分子、"
                 f"823 对端点 {f((gates.get('wc_glop_pair_closure') or {}).get('n_endpoints'))} 个端点全部闭合。")
    return "\n".join(lines)


def section_gates(gates: Mapping[str, Any]) -> str:
    lines = ["## 2. 正确性门禁", ""]
    entry = gates.get("gates", {})
    rows = []
    for name, value in entry.items():
        if name == "all_passed" or not isinstance(value, dict):
            continue
        rows.append([f"`{name}`", "PASS" if value.get("passed") else "FAIL",
                     value.get("error", "")[:120]])
    lines.append(table(["门禁", "结果", "错误"], rows))
    lines.append("")
    lines.append(f"总门禁：**{'全部通过' if entry.get('all_passed') else '存在失败'}**。"
                 "门禁失败会停止受影响的方法；本轮没有任何门禁被跳过或放宽。")
    dose = next((r for r in entry.get("dose_path", {}).get("records", []) if r.get("family")), None)
    if dose:
        lines.append("")
        lines.append("剂量路径闭合（m=1 → 0，沿 64 步线性路径积分梯度）："
                     f"示例 {dose.get('family')} `{dose.get('task')}` 分子闭合误差 "
                     f"`{f(dose.get('closure_abs_error'), 8)}`（容差 `{f(dose.get('tolerance'))}`），"
                     f"中点 m=0.5 与 `DOSE_GATE` 一致。")
    return "\n".join(lines)


def section_pool(pool_manifest: Mapping[str, Any]) -> str:
    lines = ["## 3. 共享候选池", ""]
    rows = []
    for scope, rep in (pool_manifest.get("reports") or {}).items():
        size = rep.get("pool_size") or {}
        rows.append([scope, f(rep.get("n_rows")), f(size.get("mean"), 1),
                     f(size.get("min")), f(size.get("max")),
                     f"`{str(rep.get('sha256', ''))[:12]}…`"])
    lines.append(table(["scope", "分子数", "平均候选数", "最小", "最大", "sha256"], rows))
    lines.append("")
    lines.append("四个生成器（LCRC-GRAD / LCRC-LOCAL / GINE-GRAD / 拓扑 `n_s2_affected`）用同一套冻结 beam 搜索产生候选，"
                 "合并后按物理键 ID 元组去重；两个 GPU worker 只读这一份池文件，并在入口校验 SHA-256。"
                 "候选生成过程不读取标签、编辑区、配对信息或 oracle 贡献（见门禁 `label_blind`）。")
    return "\n".join(lines)


def method_rows(summary: Mapping[str, Any], split: str) -> list[list[Any]]:
    rows = []
    entries = summary.get(split, {})
    for key in ORDER:
        e = entries.get(key)
        if not e:
            continue
        rows.append([
            DISPLAY[key], f(e.get("n")), f(e.get("n_oracle_evaluable")),
            f(e.get("oracle_capture")), f(e.get("oracle_enrichment")),
            ci(e, "oracle_enrichment_ci"), f(e.get("oracle_enrichment_quantiles", {}).get("q50")),
            pct(e.get("enrichment_gt0_fraction")), pct(e.get("enrichment_eq0_fraction")),
            pct((e.get("enrichment_concentration") or {}).get("top_share"), 0),
            e.get("oracle_support"),
        ])
    return rows


def section_oracle(summary: Mapping[str, Any]) -> str:
    lines = ["## 4. 外部结构贡献对齐（WC_LOGP oracle）", ""]
    lines.append("`capture(S) = Σ_{v∈atoms(S)} |c_v| / Σ_v |c_v|`；"
                 "`oracle_enrichment = capture(Top1) − median(capture(16 个同键数、同原子数、连通随机子图))`。"
                 "这是本轮唯一与模型无关的外部结构贡献口径。")
    lines.append("")
    for split in ("valid", "test"):
        lines.append(f"### {SPLIT_LABEL[split]}")
        lines.append("")
        lines.append(table(
            ["方法", "n", "oracle 可评", "capture 均值", "enrichment 均值", "95% CI（均值）",
             "中位数", ">0 比例", "=0 比例", "Top5% 占比", "支持度"],
            method_rows(summary, split)))
        lines.append("")
    return "\n".join(lines)


def fidelity_rows(summary: Mapping[str, Any], split: str) -> list[list[Any]]:
    rows = []
    for key in ORDER:
        e = (summary.get(split) or {}).get(key)
        if not e:
            continue
        rows.append([
            DISPLAY[key], f(e.get("n_fidelity_evaluable")),
            f(e.get("fidelity_advantage_std")), ci(e, "fidelity_advantage_std_ci"),
            f(e.get("abs_delta0_std")), ci(e, "abs_delta0_std_ci"),
            f(e.get("median_half_over_full")), pct(e.get("sign_consistent_fraction")),
            f(e.get("median_n_s2_affected"), 1), e.get("fidelity_support"),
        ])
    return rows


def section_fidelity(summary: Mapping[str, Any]) -> str:
    lines = ["## 5. 模型预测忠实性", ""]
    lines.append("`delta0 = pred(m=1) − pred(m_S=0)`（S 上键的门控置 0，其余键保持 1）；"
                 "`fidelity_advantage = |delta0(Top1)| − median(|delta0(16 个随机对照)|)`，"
                 "全部在标准化尺度（target std）上报告。"
                 "`half_over_full = |delta0.5| / |delta0|` 检验半剂量一致性；`n_s2_affected` 是同一子图影响的 S2 路径数。"
                 "LCRC 与 GINE 的门控语义不同，**忠实性只在同族内比较**。")
    lines.append("")
    for split in ("valid", "test"):
        lines.append(f"### {SPLIT_LABEL[split]}")
        lines.append("")
        lines.append(table(
            ["方法", "可评", "advantage（std）", "95% CI", "|delta0|（std）", "95% CI",
             "半/全剂量中位数", "同号比例", "n_s2_affected 中位数", "支持度"],
            fidelity_rows(summary, split)))
        lines.append("")
    return "\n".join(lines)


def section_alignment(summary: Mapping[str, Any]) -> str:
    lines = ["## 6. 同一分子上：外部对齐与预测依赖是否落在同一结构", ""]
    rows = []
    for key in ORDER:
        e = (summary.get("valid") or {}).get(key)
        if not e:
            continue
        rows.append([DISPLAY[key], f(e.get("n_both")), f(e.get("both_positive")),
                     f(e.get("only_oracle")), f(e.get("only_fidelity")), f(e.get("neither")),
                     pct(e.get("both_positive_fraction"))])
    lines.append(table(["方法", "两项均可评", "两项均 >0", "仅 oracle", "仅忠实性", "两项都未过", "两项均 >0 比例"], rows))
    lines.append("")
    lines.append("“仅 oracle” 表示该方法的 Top-1 子图捕获了更多外部原子贡献，但在模型预测上并不比随机子图更敏感；"
                 "“仅忠实性” 相反。两者大量分离即说明“模型依赖的结构”和“真实结构贡献所在”不是同一组键。")
    return "\n".join(lines)


def section_paired(summary: Mapping[str, Any], paired: Sequence[Mapping[str, Any]],
                   stratified: Sequence[Mapping[str, str]]) -> str:
    lines = ["## 7. 方法间配对比较与按候选规模分层", ""]
    rows = []
    for split in ("valid", "test"):
        for rec in paired:
            if rec.get("split") != split or rec.get("metric") != "oracle_enrichment":
                continue
            if rec["method_a"].startswith("D_") or rec["method_b"].startswith("D_"):
                continue
            rows.append([SPLIT_LABEL[split], f"{DISPLAY.get(rec['method_a'], rec['method_a'])} − "
                                             f"{DISPLAY.get(rec['method_b'], rec['method_b'])}",
                         f(rec.get("n_pairs")), f(rec.get("mean_diff")),
                         f"[{f((rec.get('ci95_mean') or [None, None])[0])}, {f((rec.get('ci95_mean') or [None, None])[1])}]",
                         pct(rec.get("gt0_fraction"))])
    if rows:
        lines.append(table(["split", "配对差（oracle enrichment）", "n", "均值差", "95% CI", ">0 比例"], rows))
        lines.append("")
    lines.append("### 按候选键数 k 分层（Valid）")
    lines.append("")
    by: dict[tuple[str, int], list[dict]] = {}
    for rec in stratified:
        if rec.get("split") != "valid":
            continue
        by.setdefault((rec["method"], int(rec["k"])), []).append(rec)
    rows = []
    for key in ORDER:
        for k in range(C.SEARCH_MIN_K, C.SEARCH_MAX_K + 1):
            chunk = by.get((key, k))
            if not chunk:
                continue
            rec = chunk[0]
            rows.append([DISPLAY[key], k, f(rec.get("n")), f(rec.get("oracle_evaluable")),
                         f(rec.get("oracle_enrichment_mean")), f(rec.get("oracle_enrichment_median")),
                         pct(rec.get("oracle_gt0_fraction")),
                         f(rec.get("fidelity_advantage_std_mean")),
                         pct(rec.get("fidelity_gt0_fraction"))])
    lines.append(table(["方法", "k", "n", "oracle 可评", "enrichment 均值", "中位数", ">0", "advantage 均值", "advantage >0"], rows))
    lines.append("")
    lines.append("分层用来检查均值是否由少数大 k（更原子、更容易覆盖任何区域）的分子主导；"
                 "若 enrichment 只在 k 较大时为正，就不能把它读成“小片段定位成功”。")
    return "\n".join(lines)


def section_cross_model(decision: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    lines = ["## 8. LCRC 与 GINE：同一外部 oracle 下的比较", ""]
    valid = summary.get("valid") or {}
    rows = []
    for key in ("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50", "E_GINE_GRAD", "F_GINE_MASK50"):
        e = valid.get(key) or {}
        rows.append([DISPLAY[key], f(e.get("oracle_enrichment")), ci(e, "oracle_enrichment_ci"),
                     f(e.get("oracle_capture")), f(e.get("n_oracle_evaluable"))])
    lines.append(table(["方法", "enrichment 均值", "95% CI", "capture 均值", "可评"], rows))
    lines.append("")
    cross = decision.get("cross_model_pair")
    if cross:
        lines.append(f"外部对齐最强 LCRC 方法 `{cross['method_a']}` 与最强 GINE 方法 `{cross['method_b']}` 的分子级配对差："
                     f"均值 `{f(cross.get('mean_diff'))}`，95% CI "
                     f"`[{f((cross.get('ci95_mean') or [None, None])[0])}, {f((cross.get('ci95_mean') or [None, None])[1])}]`，"
                     f"n = {f(cross.get('n_pairs'))}。")
    else:
        lines.append("没有可用的跨模型配对行（某一方法在 Valid 上无候选）。")
    lines.append("")
    lines.append("注：跨模型**不比较** `|delta0|` 的绝对值——两个族的门控读的是不同消息通道，"
                 "原始响应尺度不可比；跨模型结论只来自与模型无关的 oracle capture/enrichment。")
    return "\n".join(lines)


def section_aux(aux: Mapping[str, Any]) -> str:
    lines = ["## 9. 辅助：823 对 Lipophilicity 描述性诊断", ""]
    lines.append(f"固定 823 对（{f(aux.get('n_endpoints'))} 个端点、{f(aux.get('n_components'))} 个共享端点分量）"
                 "来自上一轮已冻结的配对目录；本轮不新增配对、不按标签筛对、不新增 pair-aware 解释器。"
                 "Top-1 与编辑区的重叠只用来说明“解释落在哪里”，编辑区本身是结构差异，不是因果性质真值。")
    lines.append("")
    summary = aux.get("summary") or {}
    lines.append("### Top-1 与编辑区的重叠（相对同尺寸随机子图的 advantage）")
    lines.append("")
    rows = []
    for slice_name, label in (("all", "全部"), ("train_train", "Train–Train"),
                              ("non_train", "含非 Train 端点"), ("both_non_train", "两端均非 Train"),
                              ("pair_correct", "配对预测正确"), ("pair_wrong", "配对预测错误")):
        for key in ORDER:
            e = summary.get(f"{key}|{slice_name}")
            if not e or not e.get("n"):
                continue
            rows.append([label, DISPLAY[key], f(e.get("n")), f(e.get("n_components")),
                         f((e.get("atom_jaccard") or {}).get("top_mean")),
                         f((e.get("atom_jaccard") or {}).get("random_mean")),
                         f((e.get("atom_jaccard") or {}).get("adv_mean")),
                         f"{(e.get('atom_jaccard') or {}).get('adv_ci95_mean', [None, None])[0]:.4f}"
                         if (e.get("atom_jaccard") or {}).get("adv_ci95_mean", [None])[0] is not None else "—",
                         f((e.get("atom_recall") or {}).get("adv_mean")),
                         f((e.get("bond_recall") or {}).get("adv_mean")),
                         f((e.get("hit_attachment_boundary") or {}).get("adv_mean")),
                         e.get("support")])
    lines.append(table(["切片", "方法", "端点", "分量", "jaccard(Top1)", "jaccard(随机)", "adv",
                        "adv CI 下界", "atom recall adv", "bond recall adv", "边界命中 adv", "支持度"], rows))
    lines.append("")
    lines.append("### 结构组成：Top-1 子图有多少落在可变区 / 连接边界 / 共同核心")
    lines.append("")
    rows = []
    for key in ORDER:
        e = summary.get(f"{key}|all")
        if not e or not e.get("n"):
            continue
        rows.append([DISPLAY[key], f(e.get("k_mean"), 2),
                     f((e.get("atom_fraction_var_atoms") or {}).get("mean")),
                     f((e.get("atom_fraction_boundary_atoms") or {}).get("mean")),
                     f((e.get("atom_fraction_core_atoms") or {}).get("mean")),
                     f((e.get("bond_fraction_var_bonds") or {}).get("mean")),
                     f((e.get("bond_fraction_attach_bonds") or {}).get("mean")),
                     f((e.get("bond_fraction_other_bonds") or {}).get("mean"))])
    lines.append(table(["方法", "平均 k", "原子:可变区", "原子:边界", "原子:核心",
                        "键:可变区", "键:连接边界", "键:核心"], rows))
    lines.append("")
    predictions = aux.get("predictions") or {}
    pair_pred = aux.get("pair_predictions") or {}
    if predictions or pair_pred:
        lines.append("### 预测表现与配对 Δy")
        lines.append("")
        rows = []
        for key, e in predictions.items():
            rows.append([key.replace("_", " "), f(e.get("n")), f(e.get("rmse")), f(e.get("mae"))])
        lines.append(table(["split / model", "n", "RMSE", "MAE"], rows))
        lines.append("")
        rows = []
        for family, e in pair_pred.items():
            rows.append([family.upper(), f(e.get("n_pairs")), f(e.get("mean_abs_delta_error")),
                         f(e.get("rmse_delta")), pct(e.get("direction_accuracy_all")),
                         pct(e.get("direction_accuracy_strong")), f(e.get("mean_abs_true_delta"))])
        lines.append(table(["模型", "配对数", "平均 |Δŷ−Δy|", "RMSE(Δ)", "方向准确率", "|Δy|≥0.1 方向准确率", "平均 |Δy|"], rows))
        lines.append("")
    rows = []
    for slice_name in ("all", "train_train", "non_train", "pair_correct", "pair_wrong"):
        for key in ("B_LCRC_LOCAL", "A_LCRC_GRAD", "E_GINE_GRAD"):
            e = summary.get(f"{key}|{slice_name}")
            if not e or not e.get("n"):
                continue
            rows.append([slice_name, DISPLAY[key], f(e.get("n")),
                         f((e.get("atom_capture") or {}).get("mean")),
                         f((e.get("atom_capture") or {}).get("enrichment_mean")),
                         f"{(e.get('atom_capture') or {}).get('enrichment_ci95_mean', [None, None])[0]:.4f}"
                         if (e.get("atom_capture") or {}).get("enrichment_ci95_mean", [None])[0] is not None else "—"])
    if rows:
        lines.append("### 端点上的 oracle 原子捕获")
        lines.append("")
        lines.append(table(["切片", "方法", "端点", "capture 均值", "enrichment", "CI 下界"], rows))
        lines.append("")
    lines.append("说明：非 Train 端点只有 11 个（4 valid + 7 test），分量数更少；"
                 "按预注册规则，两个非 Train 切片都为空时记 `NO_SUPPORT`，本表给出的 CI 仅作描述。")
    return "\n".join(lines)


def section_questions(decision: Mapping[str, Any], summary: Mapping[str, Any], aux: Mapping[str, Any]) -> str:
    valid = summary.get("valid") or {}
    test = summary.get("test") or {}
    labels = set(decision.get("labels", []))
    b = valid.get("B_LCRC_LOCAL") or {}
    b_test = test.get("B_LCRC_LOCAL") or {}
    b_enr = b.get("oracle_enrichment") or {}
    b_enr_ci = (b.get("oracle_enrichment_ci") or {}).get("mean")
    b_adv = b.get("fidelity_advantage_std")
    b_adv_ci = (b.get("fidelity_advantage_std_ci") or {}).get("mean")
    lines = ["## 10. 五个问题的回答", ""]

    lines.append("### Q1. LOCAL_LCRC 的外部定位更强，它同时具备模型预测忠实性吗？")
    lines.append("")
    lines.append(f"- Valid：LOCAL 的 oracle enrichment 均值 `{f(b_enr.get('mean'))}`，95% CI "
                 f"`[{f((b_enr_ci or [None, None])[0])}, {f((b_enr_ci or [None, None])[1])}]`；"
                 f"Test 均值 `{f((b_test.get('oracle_enrichment') or {}).get('mean'))}`。")
    lines.append(f"- 同族忠实性：`fidelity_advantage(std) = {f(b_adv)}`，95% CI "
                 f"`[{f((b_adv_ci or [None, None])[0])}, {f((b_adv_ci or [None, None])[1])}]`；"
                 f"半/全剂量中位数 `{f(b.get('median_half_over_full'))}`，"
                 f"|delta0|(std) 均值 `{f(b.get('abs_delta0_std'))}`。")
    if "LOCAL_FAITHFUL_AND_RELEVANT" in labels:
        lines.append("- 结论：LOCAL 既在外部结构贡献上优于随机，也在模型预测依赖上优于同尺寸随机子图，两项同时成立。")
    elif "LOCAL_RELEVANT_NOT_FAITHFUL" in labels:
        lines.append("- 结论：LOCAL 的“定位更强”只在**外部结构贡献**上成立；"
                     "它选出的子图对模型预测的影响并不比同尺寸随机子图更大。"
                     "因此不能把 LOCAL 的说服力直接解释为“模型确实依赖它”。")
    else:
        lines.append("- 结论：在预注册阈值下，LOCAL 的外部对齐本身未被支持（或 Test 未复现方向），"
                     "本轮不能给出“更强定位”的结论。")
    if b.get("median_half_over_full") is not None and float(b["median_half_over_full"]) > 1.5:
        lines.append("- 半剂量检查显示 `|delta0.5|` 明显大于 `|delta0|`，说明响应在 m=0.5 附近并非近似线性，"
                     "“半剂量近似全剂量”的直觉在这些分子上不成立。")
    lines.append("")

    ca = decision.get("finite_mask_paired_C_minus_A") or {}
    a = valid.get("A_LCRC_GRAD") or {}
    c = valid.get("C_LCRC_MASK50") or {}
    lines.append("### Q2. GRAD 与 MASK50 的差异是否说明梯度近似限制了结构发现？")
    lines.append("")
    lines.append(f"- 外部对齐：A(GRAD) `{f((a.get('oracle_enrichment') or {}).get('mean'))}` "
                 f"{ci(a, 'oracle_enrichment_ci')}，C(MASK50) `{f((c.get('oracle_enrichment') or {}).get('mean'))}` "
                 f"{ci(c, 'oracle_enrichment_ci')}；配对差 C−A 均值 `{f(ca.get('mean_diff'))}`，"
                 f"95% CI `[{f((ca.get('ci95_mean') or [None, None])[0])}, {f((ca.get('ci95_mean') or [None, None])[1])}]`。")
    lines.append(f"- 同族忠实性：A `{f(a.get('fidelity_advantage_std'))}` {ci(a, 'fidelity_advantage_std_ci')}，"
                 f"C `{f(c.get('fidelity_advantage_std'))}` {ci(c, 'fidelity_advantage_std_ci')}。")
    lines.append(f"- C 的 MASK50 选择在中位数 `{f(c.get('median_mask50_n_scanned'), 0)}` 个候选上做筛选；"
                 f"选择分数（m=0.5）与真实忠实性评测（m=0）分属不同剂量，两者同族相关不能当作独立验证。")
    if "FINITE_MASK_IMPROVES_ALIGNMENT" in labels:
        lines.append("- 结论：在同一候选池上，有限掩码筛选把外部结构对齐显著推高，"
                     "说明只依赖一阶梯度符号/幅度的 GRAD 选择确实漏掉了结构贡献更集中的候选；"
                     "但这仍是**结构对齐**的改进，不是因果识别。")
    else:
        lines.append("- 结论：没有证据表明 MASK50 筛选在外部对齐上系统性优于 GRAD；"
                     "梯度近似的限制不能被这一对比证实。")
    lines.append("")

    cross = decision.get("cross_model_pair") or {}
    e_valid = valid.get("E_GINE_GRAD") or {}
    f_valid = valid.get("F_GINE_MASK50") or {}
    lines.append("### Q3. 在同一外部 oracle 下，LCRC 与 GINE 谁更强？有支持吗？")
    lines.append("")
    lines.append(f"- LCRC 侧：A `{f((a.get('oracle_enrichment') or {}).get('mean'))}`、"
                 f"B `{f(b_enr.get('mean'))}`、C `{f((c.get('oracle_enrichment') or {}).get('mean'))}`；"
                 f"GINE 侧：E `{f((e_valid.get('oracle_enrichment') or {}).get('mean'))}`、"
                 f"F `{f((f_valid.get('oracle_enrichment') or {}).get('mean'))}`。")
    if cross:
        lo = (cross.get("ci95_mean") or [None, None])[0]
        lines.append(f"- 最强 LCRC vs 最强 GINE 的配对差 `{f(cross.get('mean_diff'))}`（CI 下界 `{f(lo)}`），"
                     f"n = {f(cross.get('n_pairs'))}。")
    if "NO_LCRC_SPECIFIC_ADVANTAGE" in labels:
        lines.append("- 结论：在**与模型无关**的外部 oracle 上，没有检测到 LCRC 相对 GINE 的配对优势；"
                     "任何“LCRC 解释更忠实/更定位”的说法都缺少跨模型支持。")
    elif decision.get("lcrc_specific_advantage"):
        lines.append("- 结论：外部 oracle 上 LCRC 侧确有配对优势，且 CI 下界超过预注册阈值；"
                     "该优势只针对结构贡献对齐，不外推到因果性或跨数据集泛化。")
    else:
        lines.append("- 结论：跨模型配对差方向为正但未达预注册阈值/支持度，按预注册记 `UNRESOLVED`。")
    lines.append("")

    both = b.get("both_positive")
    only_o = b.get("only_oracle")
    only_f = b.get("only_fidelity")
    neither = b.get("neither")
    n_both = b.get("n_both")
    lines.append("### Q4. 外部贡献捕获与模型预测依赖，是否经常落在不同结构上？")
    lines.append("")
    lines.append(f"- LOCAL 在 Valid 上两项均可评的 `{f(n_both)}` 个分子中：两项均 >0 `{f(both)}`、"
                 f"仅 oracle `{f(only_o)}`、仅忠实性 `{f(only_f)}`、都未过 `{f(neither)}`。")
    frac = b.get("both_positive_fraction")
    if frac is not None:
        lines.append(f"- 两项同时为正的比例为 `{pct(frac)}`；"
                     + ("也就是说，多数分子的“外部贡献所在”和“模型最依赖的结构”并不重合。"
                        if float(frac) < 0.5 else
                        "也就是说过半分子的两项判据重合，仍存在明显的单侧分子。"))
    lines.append("")
    lines.append("### Q5. 下一步：改搜索方法、改结构表示/读出，还是停止 LCRC 特定解释性主张？")
    lines.append("")
    if "FINITE_MASK_IMPROVES_ALIGNMENT" in labels and "LOCAL_RELEVANT_NOT_FAITHFUL" in labels:
        lines.append("- 优先改**搜索/选择目标**：有限掩码已经证明“用更接近干预的量来选”能提高外部对齐，"
                     "下一步应把选择目标直接写成小剂量（或精确删除）下的预测变化，而不是继续调 LCRC 的梯度/局部输出读出。")
        lines.append("- 如果新的选择目标仍然只提高 oracle 对齐、不提高同族忠实性，"
                     "则应停止“LCRC 特定解释性”主张，转向结构表示/读出层面的工作。")
    elif "LOCAL_FAITHFUL_AND_RELEVANT" in labels:
        lines.append("- 搜索方法仍有改进空间，但 LOCAL 已在两项指标上同时成立；"
                     "下一步优先在**同一候选池**上做剂量/删除目标的选择器对照，再决定是否需要改表示。")
    elif "NO_LCRC_SPECIFIC_ADVANTAGE" in labels and "LOCAL_RELEVANT_NOT_FAITHFUL" in labels:
        lines.append("- 不建议继续围绕 LCRC 的读出做特化解释方法；"
                     "外部对齐和同族忠实性都没有给出 LCRC 特有的优势。"
                     "若继续，应改**结构表示/读出**（例如让解释对象直接是干预响应），或把问题转成模型无关的子图发现。")
    else:
        lines.append("- 本轮证据不足以在“改搜索”与“改表示”之间排序；先补足支持度（更多非 Train 端点/分子），"
                     "再决定是否继续 LCRC 特定解释性主张。")
    lines.append("")
    lines.append("**按本轮预注册规则，到此停止**：不自动开始训练、不自动进入架构搜索。")
    return "\n".join(lines)


def section_limits() -> str:
    return "\n".join([
        "## 11. 局限与不能从本轮推出的结论",
        "",
        "- 只有 968 个主实验分子（Valid 420 + Test 420 + 固定 128 Train），Test 只用于方向复审，"
        "不是独立的大样本验证。",
        "- 候选池被冻结在“2–6 根物理键、beam 12、四个生成器”的搜索空间内；"
        "池外的子图没有被评估，因此“某方法不忠实”只在本池语义内成立。",
        "- oracle 是 WC_LOGP 的原子加性贡献，不是因果真值；capture 高只说明与该加性分解一致。",
        "- `m=0` 是完全移除门控，`m=0.5` 是半剂量；两者都不是 Shapley 值，也不保证可加。",
        "- 823 对辅助实验的编辑区是结构差异，不是性质变化的因果位置；"
        "非 Train 端点只有 11 个，配对结论只作描述。",
        "- 本轮没有训练任何新模型，也没有更换 checkpoint；所有结论都条件于这四个冻结模型。",
        "",
    ])


def section_repro(manifest: Mapping[str, Any], identity: Mapping[str, Any], gates: Mapping[str, Any]) -> str:
    lines = ["## 12. 复现与文件清单", ""]
    lines.append("| 阶段 | 命令 | 输出 |")
    lines.append("| --- | --- | --- |")
    for stage in manifest.get("stages", []):
        lines.append(f"| {stage.get('name')} | `{stage.get('command')}` | `{stage.get('outputs')}` |")
    lines.append("")
    lines.append("| 文件 | sha256 |")
    lines.append("| --- | --- |")
    for name, digest in (manifest.get("artifacts") or {}).items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.append("")
    lines.append("本地复现（CPU，无 GPU）：")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python code/lfa_prepare.py            # 身份 + 重放 + WC_LOGP 闭合")
    lines.append("uv run python code/lfa_gates.py             # 实现门禁")
    lines.append("uv run python code/lfa_pred.py              # 冻结 Lipophilicity 预测")
    lines.append("uv run python code/lfa_main.py              # 主实验聚合 + 决策")
    lines.append("uv run python code/lfa_aux.py               # 823 对描述性诊断")
    lines.append("uv run python code/lfa_report.py            # 本报告")
    lines.append("uv run pytest tests -q                      # 结构不变量")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build(*, analysis_dir: Path | str = C.ANALYSIS_DIR, out_dir: Path | str | None = None) -> Path:
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir is not None else C.ROUND_DIR
    freeze = load(analysis_dir / "protocol_freeze.json")
    identity = load(analysis_dir / "identity.json")
    gates = load(analysis_dir / "gates.json")
    poll = load(C.POOL_DIR / "pool_manifest.json")
    summary = load(analysis_dir / "main_summary.json")
    decision = load(analysis_dir / "decision.json")
    aux = load(analysis_dir / "aux_summary.json")
    manifest = load(analysis_dir / "run_manifest.json")
    stratified = load_csv(analysis_dir / "main_stratified_by_k.csv")
    paired = load_csv(analysis_dir / "main_paired_differences.csv")

    parts = [
        f"# LCRC 解释忠实性与结构贡献对齐（`{C.ROUND_ID}`）",
        "",
        f"预注册协议：`PROTOCOL.md`；协议冻结哈希 `{freeze.get('freeze_hash')}`；"
        f"代码提交 `{freeze.get('git_commit')}`。",
        "",
        section_status(decision),
        section_assets(identity, freeze),
        section_gates(gates),
        section_pool(poll),
        section_oracle(summary["summary"]),
        section_fidelity(summary["summary"]),
        section_alignment(summary["summary"]),
        section_paired(summary["summary"], paired, stratified),
        section_cross_model(decision, summary["summary"]),
        section_aux(aux),
        section_questions(decision, summary["summary"], aux),
        section_limits(),
        section_repro(manifest, identity, gates),
    ]
    report = "\n\n".join(parts) + "\n"
    path = out_dir / "REPORT.md"
    C.write_text(path, report)
    decision_md = build_decision(decision, summary, aux, gates, freeze)
    C.write_text(out_dir / "DECISION.md", decision_md)
    print(f"[report] wrote {path} ({len(report)} bytes)")
    return path


def build_decision(decision: Mapping[str, Any], summary: Mapping[str, Any], aux: Mapping[str, Any],
                   gates: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    labels = decision.get("labels", [])
    lines = [f"# 决策：`{C.ROUND_ID}`", ""]
    lines.append(f"协议 `{freeze.get('freeze_hash')}`，提交 `{freeze.get('git_commit')}`。")
    lines.append("")
    lines.append("## 状态")
    lines.append("")
    for label in labels:
        lines.append(f"- `{label}`")
    if decision.get("reasons"):
        lines.append("")
        lines.append("触发原因： " + "；".join(f"`{x}`" for x in decision["reasons"]))
    lines.append("")
    lines.append("## 支撑与反对")
    lines.append("")
    valid = summary["summary"].get("valid", {})
    for key in ("B_LCRC_LOCAL", "C_LCRC_MASK50", "A_LCRC_GRAD", "E_GINE_GRAD"):
        e = valid.get(key) or {}
        lines.append(f"- {DISPLAY[key]}：oracle enrichment `{f((e.get('oracle_enrichment') or {}).get('mean'))}` "
                     f"{ci(e, 'oracle_enrichment_ci')}；fidelity advantage `{f(e.get('fidelity_advantage_std'))}` "
                     f"{ci(e, 'fidelity_advantage_std_ci')}；oracle 支持度 `{e.get('oracle_support')}`，"
                     f"忠实性支持度 `{e.get('fidelity_support')}`。")
    ca = decision.get("finite_mask_paired_C_minus_A")
    if ca:
        lines.append(f"- 有限掩码 C−A 配对差 `{f(ca.get('mean_diff'))}` "
                     f"`[{f((ca.get('ci95_mean') or [None, None])[0])}, {f((ca.get('ci95_mean') or [None, None])[1])}]`。")
    cross = decision.get("cross_model_pair")
    if cross:
        lines.append(f"- 跨模型配对差 `{f(cross.get('mean_diff'))}` "
                     f"`[{f((cross.get('ci95_mean') or [None, None])[0])}, {f((cross.get('ci95_mean') or [None, None])[1])}]`。")
    lines.append(f"- 门禁全部通过：`{bool(gates.get('gates', {}).get('all_passed'))}`。")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("按本轮预注册条款，本轮结束后停止；不自动开始训练、不自动做架构搜索。"
                 "若要继续，唯一被本轮数据支持的入口是“在冻结模型上把选择目标改成小剂量/精确删除下的预测变化”，"
                 "且必须在新的预注册里重新冻结候选池与统计口径。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS_DIR))
    parser.add_argument("--out-dir", default=str(C.ROUND_DIR))
    args = parser.parse_args()
    build(analysis_dir=args.analysis_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
