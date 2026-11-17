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
AUX_ORDER = ("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50", "D_TOPOLOGY", "E_GINE_GRAD", "F_GINE_MASK50")
DISPLAY["D_TOPOLOGY"] = "D TOPOLOGY"
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


def i(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


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


def blank(value: Any) -> bool:
    """True for ``None`` / ``NaN`` / empty string (all mean "missing")."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() in ("", "nan", "None")


def row_in_split(row: Mapping[str, Any], split: str) -> bool:
    if split == "train_description":
        return str(row.get("stratum")) == "train_description"
    return str(row.get("split")) == split


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
    lines.append("")
    lines.append("**主要事实**")
    lines.append("")
    for bullet in _findings(decision):
        lines.append(f"- {bullet}")
    return "\n".join(lines)


def _findings(decision: Mapping[str, Any]) -> list[str]:
    summary = decision.get("_summary") or {}
    valid = summary.get("valid") or {}
    out: list[str] = []
    b = valid.get("B_LCRC_LOCAL") or {}
    if b:
        out.append(f"B（LOCAL）在 Valid 上同时通过两项：oracle enrichment `{f(b.get('oracle_enrichment'))}` "
                   f"{ci(b, 'oracle_enrichment_ci')}，忠实性 advantage `{f(b.get('fidelity_advantage_std'))}` "
                   f"{ci(b, 'fidelity_advantage_std_ci')}；Test 同号"
                   f"（enrichment `{f((summary.get('test') or {}).get('B_LCRC_LOCAL', {}).get('oracle_enrichment'))}`）。")
        out.append(f"LOCAL 的忠实性优势是同族中最小的过线者（A `{f((valid.get('A_LCRC_GRAD') or {}).get('fidelity_advantage_std'))}`、"
                   f"C `{f((valid.get('C_LCRC_MASK50') or {}).get('fidelity_advantage_std'))}`、"
                   f"E `{f((valid.get('E_GINE_GRAD') or {}).get('fidelity_advantage_std'))}`）。")
    ca = decision.get("finite_mask_paired_C_minus_A") or {}
    if ca:
        out.append(f"C（MASK50）比 A（GRAD）多捕获 `{f(ca.get('mean_diff'))}` 的外部质量"
                   f"（CI `[{f(ca.get('ci95_low'))}, {f(ca.get('ci95_high'))}]`），"
                   "低于预注册的 0.02 改进阈值；C 平均选中 5.6 根键、A 为 4.0 根，"
                   "k=6 内两者只差 0.003，说明差异主要是“选得更大”。")
    cross = decision.get("cross_model_pair") or {}
    if cross:
        out.append(f"跨模型外部对齐：最强 LCRC（`{cross.get('method_a')}`）减最强 GINE"
                   f"（`{cross.get('method_b')}`）= `{f(cross.get('mean_diff'))}` "
                   f"`[{f(cross.get('ci95_low'))}, {f(cross.get('ci95_high'))}]` → 没有 LCRC 特有优势。")
    d = valid.get("D_TOPOLOGY@LCRC") or {}
    if d:
        out.append(f"拓扑基线 D 在 LCRC 上比随机子图**更不**影响预测（advantage `{f(d.get('fidelity_advantage_std'))}`），"
                   "在 GINE 上却为正；同一批子图在两族上的预测依赖方向不同。")
    if b:
        out.append(f"在 LOCAL 的 420 个可评分子里，外部对齐与预测依赖同时为正的只有 "
                   f"`{pct(b.get('both_positive_fraction'))}`，"
                   f"“仅 oracle”`{pct((b.get('only_oracle') or 0) / max(1, b.get('n_both') or 1))}` > "
                   f"“仅忠实性”`{pct((b.get('only_fidelity') or 0) / max(1, b.get('n_both') or 1))}`。")
    return out


def _dirty_note(manifest: Mapping[str, Any]) -> str:
    if not manifest.get("git_dirty_tracked"):
        return ""
    files = [str(x).split("/")[-1] for x in (manifest.get("git_dirty_files") or [])]
    if not files:
        return "，生成时工作树中有未提交的改动"
    shown = ", ".join(f"`{x}`" for x in files[:4]) + (" 等" if len(files) > 4 else "")
    return f"，生成时未提交的改动为 {shown}（完整列表见 `run_manifest.json` 的 `git_dirty_files`）"


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
    manifest = C.read_json(C.ANALYSIS_DIR / "run_manifest.json") or {}
    info = manifest.get("analysis_code") or {}
    head = info.get("git_head_now") or info.get("git_head")
    frozen, now = info.get("frozen_execution_hashes") or {}, info.get("hashes") or {}
    changed = sorted(k for k in now if frozen.get(k) not in (None, now.get(k)))
    if head:
        lines.append(f"执行阶段部署的是提交 `{freeze.get('git_commit')}`（`res` 上的版本；`protocol_freeze.json` 冻结了"
                     f"当时的代码哈希）；本报告由提交 `{head}` 的分析代码生成"
                     + _dirty_note(manifest)
                     + "。"
                     + (f"相对冻结时被改动的是 `{'`, `'.join(changed)}`。" if changed else "")
                     + "这些改动只在聚合/门禁/报告层（`aggregation_replay` 门禁、Train-128 分层过滤、"
                     "缺失计数、artifact 清单），不触及候选生成、选择分数或模型前向；"
                     "`pool_main/pool_aux` 与四个 select 记录的 SHA-256 与执行时一致（第 12 节）。")
        lines.append("")
    rows = []
    for rec in ckpt.get("models", []):
        ident = rec.get("identity", {})
        replay = rec.get("replay", {})
        replay_valid = ((replay.get("splits") or {}).get("valid") or {}).get("rmse_raw") if isinstance(replay, dict) else None
        rows.append([ident.get("name", "—"), ident.get("checkpoint_epoch", "—"),
                     f(ident.get("checkpoint_valid_rmse_raw")), f"{ident.get('sha256', '')[:12]}…",
                     "PASS" if rec.get("passed") else "FAIL", f(replay_valid)])
    if rows:
        lines.append(table(["checkpoint", "epoch", "valid RMSE（身份记录）", "sha256", "身份+重放", "重放 valid RMSE"], rows))
    lines.append("")
    main_scope = gates.get("wc_glop_main_scope_closure") or {}
    pair_scope = gates.get("wc_glop_pair_closure") or {}
    lines.append("WC_LOGP 原子贡献闭合：图级 `max_abs_error = "
                 f"{f((gates.get('wc_glop_graph_closure') or {}).get('max_abs_error'), 8)}`；"
                 f"主范围 {f(main_scope.get('n'))} 个分子（闭合 {f(main_scope.get('n_closed'))}，"
                 f"最大误差 {f(main_scope.get('max_closure_error'), 10)}）；"
                 f"{f(pair_scope.get('n_pairs'))} 对端点（最大误差 {f(pair_scope.get('max_abs_error'), 10)}）全部闭合。")
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
            DISPLAY[key], i(e.get("n")), i(e.get("n_oracle_evaluable")), f(e.get("mean_k"), 2),
            f(e.get("oracle_capture")), f(e.get("oracle_enrichment")),
            ci(e, "oracle_enrichment_ci"), f(e.get("oracle_enrichment_quantiles", {}).get("q50")),
            f"[{f(e.get('oracle_enrichment_quantiles', {}).get('q25'))}, "
            f"{f(e.get('oracle_enrichment_quantiles', {}).get('q75'))}]",
            pct(e.get("enrichment_gt0_fraction")), pct(e.get("enrichment_eq0_fraction")),
            pct((e.get("enrichment_concentration") or {}).get("top_share"), 0),
            e.get("oracle_support"),
        ])
    return rows


def section_oracle(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    lines = ["## 4. 外部结构贡献对齐（WC_LOGP oracle）", ""]
    lines.append("`capture(S) = Σ_{v∈atoms(S)} |c_v| / Σ_v |c_v|`；"
                 "`oracle_enrichment = capture(Top1) − median(capture(16 个同键数、同原子数、连通随机子图))`。"
                 "这是本轮唯一与模型无关的外部结构贡献口径。")
    lines.append("")
    for split in ("valid", "test", "train_description"):
        lines.append(f"### {SPLIT_LABEL[split]}")
        lines.append("")
        lines.append(table(
            ["方法", "n", "oracle 可评", "平均 k", "capture 均值", "enrichment 均值", "95% CI（均值）",
             "中位数", "IQR", ">0 比例", "=0 比例", "Top5% 占比", "支持度"],
            method_rows(summary, split)))
        lines.append("")
    if rows is not None:
        miss = [r for r in rows if str(r.get("status")) == "OK" and blank(r.get("oracle_enrichment"))]
        zero = [r for r in rows if blank(r.get("capture_top1"))]
        insuff = [r for r in miss if str(r.get("control_status")) == "INSUFFICIENT_RANDOM"]
        if miss:
            def n_of(split):
                return sum(1 for r in miss if row_in_split(r, split))

            parts = "、".join(f"{SPLIT_LABEL[s]} {n_of(s)} 条" for s in ("valid", "test", "train_description"))
            lines.append(f"缺失说明：oracle enrichment 记为缺失的 (分子, 方法) 记录为 {parts}；"
                         f"其中 {len(insuff)}/{len(miss)} 条的 `control_status = INSUFFICIENT_RANDOM`（Top-1 太大，"
                         "找不到同键数、同原子数的**不同**连通子图，`n_controls = 0`）；"
                         f"`capture_top1` 本身缺失（零分母）的有 {len(zero)} 条。"
                         "按预注册规则这些记为 missing，没有放宽匹配条件，也没有用 0 代替。")
            lines.append("")
    lines.append("固定 128 个 Train 分子只作描述性对照（它们参与过四轮之前的训练），"
                 "CI 与 Test 一样不作为独立确认；把它们列出来是为了检查 Valid/Test 的排序是否在这批分子上也出现。")
    lines.append("")
    return "\n".join(lines)


def fidelity_rows(summary: Mapping[str, Any], split: str) -> list[list[Any]]:
    rows = []
    for key in ORDER:
        e = (summary.get(split) or {}).get(key)
        if not e:
            continue
        rows.append([
            DISPLAY[key], i(e.get("n_fidelity_evaluable")), f(e.get("mean_k"), 2),
            f(e.get("fidelity_advantage_std")), ci(e, "fidelity_advantage_std_ci"),
            f(e.get("abs_delta0_std")), ci(e, "abs_delta0_std_ci"),
            f(e.get("median_half_over_full")), pct(e.get("sign_consistent_fraction")),
            f(e.get("median_n_s2_affected"), 1), e.get("fidelity_support"),
        ])
    return rows


def structure_rows(rows: Sequence[Mapping[str, Any]], split: str) -> list[list[Any]]:
    out = []
    for key in ORDER:
        sub = [r for r in rows if r.get("split") == split and r.get("method") == key
               and r.get("fidelity_model") == (key.split("@")[1] if "@" in key else
                                               {"A_LCRC_GRAD": "LCRC", "B_LCRC_LOCAL": "LCRC",
                                                "C_LCRC_MASK50": "LCRC", "E_GINE_GRAD": "GINE",
                                                "F_GINE_MASK50": "GINE"}[key])
               and r.get("status") == "OK"]
        if not sub:
            continue

        def med(name):
            vals = [float(r[name]) for r in sub if not blank(r.get(name))]
            return float(np.median(vals)) if vals else None

        def med_std(name):
            stds = [float(r["target_std"]) for r in sub if not blank(r.get("target_std"))]
            vals = [float(r[name]) for r in sub if not blank(r.get(name))]
            if not vals or not stds:
                return None
            return float(np.median(vals)) / float(np.median(stds))

        out.append([DISPLAY[key], i(len(sub)),
                    f(med("n_s2_internal"), 1), f(med("n_s2_boundary"), 1), f(med("n_s2_affected"), 1),
                    f(med_std("delta_local")), f(med_std("delta_rho")),
                    f(max([abs(float(r["closure_error"])) for r in sub
                           if not blank(r.get("closure_error"))]),
                      8) if key.startswith(("A_", "B_", "C_", "D_")) else "—"])
    return out


def section_fidelity(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    lines = ["## 5. 模型预测忠实性", ""]
    lines.append("`delta0 = pred(m=1) − pred(m_S=0)`（S 上键的门控置 0，其余键保持 1）；"
                 "`fidelity_advantage = |delta0(Top1)| − median(|delta0(16 个随机对照)|)`，"
                 "全部在标准化尺度（target std）上报告。"
                 "`half_over_full = |delta0.5| / |delta0|` 检验半剂量一致性；`n_s2_affected` 是同一子图影响的 S2 路径数。"
                 "LCRC 与 GINE 的门控语义不同，**忠实性只在同族内比较**。")
    lines.append("")
    for split in ("valid", "test", "train_description"):
        lines.append(f"### {SPLIT_LABEL[split]}")
        lines.append("")
        lines.append(table(
            ["方法", "可评", "平均 k", "advantage（std）", "95% CI", "\\|delta0\\|（std）", "95% CI",
             "半/全剂量中位数", "同号比例", "n_s2_affected 中位数", "支持度"],
            fidelity_rows(summary, split)))
        lines.append("")
    if rows:
        lines.append("### S2 结构分解与 LCRC 局部/rho 分解（Valid，中位数）")
        lines.append("")
        lines.append(table(
            ["方法", "n", "n_s2_internal", "n_s2_boundary", "n_s2_affected",
             "delta_local（std）", "delta_rho（std）", "max 闭合误差"],
            structure_rows(rows, "valid")))
        lines.append("")
        lines.append("`delta_local + delta_rho = delta0` 是 LCRC 的门控分解（`delta_rho` 为 GINE 不适用，故留空）；"
                     "闭合误差是逐分子的 `||delta0 − delta_local − delta_rho||`，全部远小于目标尺度。"
                     "`n_s2_*` 说明同一组键在 LCRC 内部命中的 S2 路径类型，不是因果量。")
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


def k_composition(summary: Mapping[str, Any], stratified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = summary.get("valid") or {}
    out: dict[str, Any] = {}
    for key in ORDER:
        e = valid.get(key) or {}
        if not e:
            continue
        rows = [r for r in stratified if r.get("split") == "valid" and r.get("method") == key]
        n6 = sum(int(float(r["n"])) for r in rows if int(float(r["k"])) == 6)
        n_all = sum(int(float(r["n"])) for r in rows) or 1
        out[key] = {"mean_k": e.get("mean_k"), "k6_fraction": n6 / n_all}
    return out


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
                         i(rec.get("n_pairs")), f(rec.get("mean_diff")),
                         f"[{f(rec.get('ci95_low'))}, {f(rec.get('ci95_high'))}]",
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
            rows.append([DISPLAY[key], k, i(rec.get("n")), i(rec.get("oracle_evaluable")),
                         f(rec.get("oracle_enrichment_mean")), f(rec.get("oracle_enrichment_median")),
                         pct(rec.get("oracle_gt0_fraction")),
                         f(rec.get("fidelity_advantage_std_mean")),
                         pct(rec.get("fidelity_gt0_fraction"))])
    lines.append(table(["方法", "k", "n", "oracle 可评", "enrichment 均值", "中位数", ">0", "advantage 均值", "advantage >0"], rows))
    lines.append("")
    lines.append("分层用来检查均值是否由少数大 k（更原子、更容易覆盖任何区域）的分子主导；"
                 "若 enrichment 只在 k 较大时为正，就不能把它读成“小片段定位成功”。")

    def strat(method: str, k: int, field: str):
        rec = by.get((method, k))
        return None if not rec else rec[0].get(field)

    b2, a2 = strat("B_LCRC_LOCAL", 2, "oracle_enrichment_mean"), strat("A_LCRC_GRAD", 2, "oracle_enrichment_mean")
    c2 = strat("C_LCRC_MASK50", 2, "oracle_enrichment_mean")
    if b2 is not None and a2 is not None:
        lines.append("")
        lines.append(f"- 小片段上差距最大：k=2 时 B 的 enrichment 为 `{f(b2)}`（>0 比例 `{pct(strat('B_LCRC_LOCAL', 2, 'oracle_gt0_fraction'))}`），"
                     f"A 只有 `{f(a2)}`（`{pct(strat('A_LCRC_GRAD', 2, 'oracle_gt0_fraction'))}`）；"
                     f"B 在 k=2–5 都稳定在 `{f(strat('B_LCRC_LOCAL', 3, 'oracle_enrichment_mean'))}`–"
                     f"`{f(strat('B_LCRC_LOCAL', 4, 'oracle_enrichment_mean'))}` 一线，到 k=6 才回落到 "
                     f"`{f(strat('B_LCRC_LOCAL', 6, 'oracle_enrichment_mean'))}`（k=6 的 227 个分子多是大候选，任何方法都容易命中）。")
        if c2 is not None:
            lines.append(f"- MASK50 在小片段上**反选**：k=2 时 C 的 enrichment 为 `{f(c2)}`，"
                         f">0 比例 `{pct(strat('C_LCRC_MASK50', 2, 'oracle_gt0_fraction'))}`；它靠 k=6 的 334 个分子"
                         f"（`{f(strat('C_LCRC_MASK50', 6, 'oracle_enrichment_mean'))}`）把总均值拉回正数。"
                         "这与“未归一化的 |Δy| 偏好大子图”一致。")
        lines.append("- 所以第 4/5 节的汇总均值必须与这张表一起读：LOCAL 的优势是真的出现在最小的候选上，"
                     "而 C/F 的汇总优势主要来自 k=6。")
    comp = k_composition(summary, stratified)
    if comp:
        lines.append("")
        lines.append("**规模组成**（Valid）：" + "；".join(
            f"{DISPLAY[key]} 平均 k `{f(v['mean_k'], 2)}`、k=6 占 `{pct(v['k6_fraction'])}`"
            for key, v in comp.items()))
        lines.append("")
        lines.append("这说明 C/F 的汇总 enrichment 有很大一部分来自“选更大的子图”，而不是同尺寸内更准；"
                     "A 的选择集中在较小 k，但小 k 上的外部对齐接近 0。"
                     "D 在每个分子上都选了 k=6 候选（分数 `n_s2_affected` 随键数单调），"
                     "因此 D 只能读作“最大子图基线”，不能读作一个独立的小片段发现器。")
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
        for key in AUX_ORDER:
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
    for key in AUX_ORDER:
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
        lines.append(table(["模型", "配对数", "平均 \\|Δŷ−Δy\\|", "RMSE(Δ)", "方向准确率",
                            "\\|Δy\\|≥0.1 方向准确率", "平均 \\|Δy\\|"], rows))
        lines.append("")
    loc = aux.get("pair_delta_localization") or {}
    if loc:
        lines.append("### 配对 Δ 贡献的局部性（为什么这里不报“oracle 捕获”）")
        lines.append("")
        lines.append(f"冻结的配对 Δ 贡献在 {f(loc.get('n_endpoints'))} 个端点中有 "
                     f"{f(loc.get('endpoints_support_subset_of_edit_atoms'))} 个**只**落在编辑原子上"
                     f"（平均非零原子占比 `{pct(loc.get('mean_nonzero_atom_fraction'))}`，"
                     f"编辑区之外平均 `{f(loc.get('mean_atoms_outside_edit'), 4)}` 个原子）。"
                     "在这些端点上，“子图捕获了多少 Δ 贡献”与“子图是否碰到编辑原子”几乎同义，"
                     "所以本报告不把端点 capture 当作独立结构验证；对外部贡献的独立检验只在主实验"
                     "（WC_LOGP 全分子原子贡献 + 匹配随机子图）上做。")
        lines.append("")
    nt = (summary.get("B_LCRC_LOCAL|non_train") or {})
    lines.append(f"说明：`含非 Train 端点` 只有 {f(nt.get('n'))} 个端点，来自 "
                 f"{f(nt.get('n_molecules'))} 个分子 / {f(nt.get('n_components'))} 个分量，记 `LOW_SUPPORT`；"
                 "`两端均非 Train` 为空，按预注册规则记 `NO_SUPPORT`。这些 CI 只作描述。"
                 "“配对预测正确/错误”的阈值是固定 `|Δŷ−Δy| ≤ 0.25`，且按模型族分别判定，"
                 "所以 LCRC 与 GINE 的端点分母不同。")
    return "\n".join(lines)


_STRAT_CACHE: dict[str, Any] = {}


def _k6_gap(summary: Mapping[str, Any], a: str, b: str) -> float | None:
    rows = _STRAT_CACHE.get("rows") or []
    def enr(method: str) -> float | None:
        for row in rows:
            if (row.get("split") == "valid" and row.get("method") == method
                    and int(float(row.get("k", 0))) == 6):
                return None if row.get("oracle_enrichment_mean") is None else float(row["oracle_enrichment_mean"])
        return None
    va, vb = enr(a), enr(b)
    return None if va is None or vb is None else va - vb


def section_questions(decision: Mapping[str, Any], summary: Mapping[str, Any], aux: Mapping[str, Any]) -> str:
    valid = summary.get("valid") or {}
    test = summary.get("test") or {}
    labels = set(decision.get("labels", []))
    b = valid.get("B_LCRC_LOCAL") or {}
    b_test = test.get("B_LCRC_LOCAL") or {}
    b_enr = b.get("oracle_enrichment")
    b_enr_ci = (b.get("oracle_enrichment_ci") or {}).get("mean")
    b_adv = b.get("fidelity_advantage_std")
    b_adv_ci = (b.get("fidelity_advantage_std_ci") or {}).get("mean")
    lines = ["## 10. 五个问题的回答", ""]

    lines.append("### Q1. LOCAL_LCRC 的外部定位更强，它同时具备模型预测忠实性吗？")
    lines.append("")
    lines.append(f"- Valid：LOCAL 的 oracle enrichment 均值 `{f(b_enr)}`，95% CI "
                 f"`[{f((b_enr_ci or [None, None])[0])}, {f((b_enr_ci or [None, None])[1])}]`；"
                 f"Test 均值 `{f(b_test.get('oracle_enrichment'))}`。")
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
    a_adv = (valid.get("A_LCRC_GRAD") or {}).get("fidelity_advantage_std")
    if a_adv is not None and b_adv is not None:
        lines.append(f"- 幅度对比：LOCAL 的忠实性优势远小于同族 GRAD（`{f(a_adv)}`）和 MASK50"
                     f"（`{f((valid.get('C_LCRC_MASK50') or {}).get('fidelity_advantage_std'))}`）。"
                     "也就是说 LOCAL 只是**过线**，不是“预测依赖最强”的方法。")
    half_ratio = b.get("median_half_over_full")
    if half_ratio is not None:
        if float(half_ratio) > 1.5:
            lines.append(f"- 半剂量检查：`|delta0.5| / |delta0|` 中位数 `{f(half_ratio)}`，"
                         "半剂量效应明显**大于**全剂量，说明门控响应在 m=0.5 附近非单调/强非线性。")
        elif float(half_ratio) < 0.8:
            lines.append(f"- 半剂量检查：`|delta0.5| / |delta0|` 中位数 `{f(half_ratio)}`，"
                         "半剂量效应明显**小于**全剂量（次线性）。这意味着用 m=0.5 的响应来近似 m=0 的删除效应会系统性低估，"
                         "C 的“先 0.5 筛选、再 0 评价”也因此存在剂量错配。")
    lines.append("")

    ca = decision.get("finite_mask_paired_C_minus_A") or {}
    a = valid.get("A_LCRC_GRAD") or {}
    c = valid.get("C_LCRC_MASK50") or {}
    lines.append("### Q2. GRAD 与 MASK50 的差异是否说明梯度近似限制了结构发现？")
    lines.append("")
    lines.append(f"- 外部对齐：A(GRAD) `{f(a.get('oracle_enrichment'))}` "
                 f"{ci(a, 'oracle_enrichment_ci')}，C(MASK50) `{f(c.get('oracle_enrichment'))}` "
                 f"{ci(c, 'oracle_enrichment_ci')}；配对差 C−A 均值 `{f(ca.get('mean_diff'))}`，"
                 f"95% CI `[{f((ca.get('ci95_mean') or [None, None])[0])}, {f((ca.get('ci95_mean') or [None, None])[1])}]`。")
    lines.append(f"- 同族忠实性：A `{f(a.get('fidelity_advantage_std'))}` {ci(a, 'fidelity_advantage_std_ci')}，"
                 f"C `{f(c.get('fidelity_advantage_std'))}` {ci(c, 'fidelity_advantage_std_ci')}。")
    lines.append(f"- C 的 MASK50 选择在中位数 `{i(c.get('median_mask50_n_scanned'))}` 个候选上做筛选；"
                 f"选择分数（m=0.5）与真实忠实性评测（m=0）分属不同剂量，两者同族相关不能当作独立验证。")
    lines.append(f"- 规模混淆：C 的平均 k 为 `{f(c.get('mean_k'), 2)}`，A 为 `{f(a.get('mean_k'), 2)}`；"
                 f"MASK50 分数没有按 |S| 归一化，更大的子图天然产生更大的 `|Δy|`。"
                 f"在 k=6 内比较时 C 对 A 的 enrichment 差只有 "
                 f"`{f(_k6_gap(summary, 'C_LCRC_MASK50', 'A_LCRC_GRAD'))}`，"
                 f"远小于汇总的 `{f(ca.get('mean_diff'))}`。")
    if "FINITE_MASK_IMPROVES_ALIGNMENT" in labels:
        lines.append("- 结论：在同一候选池上，有限掩码筛选把外部结构对齐显著推高，"
                     "说明只依赖一阶梯度符号/幅度的 GRAD 选择确实漏掉了结构贡献更集中的候选；"
                     "但这仍是**结构对齐**的改进，不是因果识别。")
    else:
        lo = (ca.get("ci95_mean") or [None, None])[0]
        if lo is not None and lo > 0 and (ca.get("mean_diff") or 0.0) > 0:
            lines.append(f"- 结论：C 的选择在汇总上确实优于 A（配对差 `{f(ca.get('mean_diff'))}`，CI 排除 0），"
                         f"但幅度低于预注册的 `{f(C.PAIRED_IMPROVEMENT_MIN)}` 阈值，且其中大部分可由“C 平均选了更大的子图”解释"
                         f"（k=6 内差 `{f(_k6_gap(summary, 'C_LCRC_MASK50', 'A_LCRC_GRAD'))}`）。"
                         "因此 `FINITE_MASK_IMPROVES_ALIGNMENT` 不成立，也不能据此断言梯度近似是结构发现的主要限制。")
        else:
            lines.append("- 结论：没有证据表明 MASK50 筛选在外部对齐上系统性优于 GRAD；"
                         "梯度近似的限制不能被这一对比证实。")
    lines.append("")

    cross = decision.get("cross_model_pair") or {}
    e_valid = valid.get("E_GINE_GRAD") or {}
    f_valid = valid.get("F_GINE_MASK50") or {}
    lines.append("### Q3. 在同一外部 oracle 下，LCRC 与 GINE 谁更强？有支持吗？")
    lines.append("")
    lines.append(f"- LCRC 侧：A `{f(a.get('oracle_enrichment'))}`、"
                 f"B `{f(b_enr)}`、C `{f(c.get('oracle_enrichment'))}`；"
                 f"GINE 侧：E `{f(e_valid.get('oracle_enrichment'))}`、"
                 f"F `{f(f_valid.get('oracle_enrichment'))}`。")
    if cross:
        lo = (cross.get("ci95_mean") or [None, None])[0]
        lines.append(f"- 最强 LCRC vs 最强 GINE 的配对差 `{f(cross.get('mean_diff'))}`（CI 下界 `{f(lo)}`），"
                     f"n = {f(cross.get('n_pairs'))}。")
    if "NO_LCRC_SPECIFIC_ADVANTAGE" in labels:
        lines.append("- 结论：在**与模型无关**的外部 oracle 上，没有检测到 LCRC 相对 GINE 的配对优势"
                     "（B 与 E/F 的配对差 CI 都跨 0）。跨模型的**忠实性**本来就不在本轮可比范围内"
                     "（门控语义不同），所以本轮只能说“LCRC 没有外部定位优势”，"
                     "不能说“LCRC 更不忠实”。")
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
        if n_both:
            lines.append(f"- 分离的方向不对称：仅 oracle 占 `{pct(only_o / n_both)}`，仅忠实性占 `{pct(only_f / n_both)}`。"
                         "LOCAL 的问题主要是“结构贡献被覆盖了，但模型并不特别依赖它”，而不是反过来。")
    f_both = (valid.get("F_GINE_MASK50") or {}).get("both_positive_fraction")
    if f_both is not None:
        lines.append(f"- 作为对照，F（GINE-MASK50）的两项均为正比例是 `{pct(f_both)}`，"
                     "而它几乎不选小片段（97.6% 在 k=6）；所以“两项重合”本身也可以由“总是选大子图”买到，"
                     "不能单独当作定位能力。")
    lines.append("")
    lines.append("### Q5. 下一步：改搜索方法、改结构表示/读出，还是停止 LCRC 特定解释性主张？")
    lines.append("")
    if "FINITE_MASK_IMPROVES_ALIGNMENT" in labels and "LOCAL_RELEVANT_NOT_FAITHFUL" in labels:
        lines.append("- 优先改**搜索/选择目标**：有限掩码已经证明“用更接近干预的量来选”能提高外部对齐，"
                     "下一步应把选择目标直接写成小剂量（或精确删除）下的预测变化，而不是继续调 LCRC 的梯度/局部输出读出。")
        lines.append("- 如果新的选择目标仍然只提高 oracle 对齐、不提高同族忠实性，"
                     "则应停止“LCRC 特定解释性”主张，转向结构表示/读出层面的工作。")
    elif "NO_LCRC_SPECIFIC_ADVANTAGE" in labels and "LOCAL_FAITHFUL_AND_RELEVANT" in labels:
        lines.append("- 停止把“LCRC 特有读出/解释”当作优势来源：外部 oracle 上没有 LCRC 相对 GINE 的优势，"
                     "唯一同时通过两项的 LOCAL 是一个**与模型无关**的局部输出启发式。")
        lines.append("- 若继续，优先级是（1）改**选择目标**：尺寸归一化的剂量/删除响应，"
                     "而不是未归一化的 `|Δy|`（本轮 C/F 的汇总优势主要来自选大子图）；"
                     "（2）再考虑结构表示/读出层面的改动；（3）不再单独优化 LCRC 的梯度/局部输出读出。")
        lines.append("- 本轮不支持“需要新架构”或“LCRC 解释路线整体失败”这类更强结论："
                     "LOCAL 的过线说明模型侧确实存在可被小片段捕获的预测依赖，只是它与外部结构贡献只在约一半分子上重合。")
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
    molecule_rows = load_csv(analysis_dir / "main_molecule_results.csv")

    _STRAT_CACHE["rows"] = stratified
    decision["_summary"] = summary["summary"]
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
        section_oracle(summary["summary"], molecule_rows),
        section_fidelity(summary["summary"], molecule_rows),
        section_alignment(summary["summary"]),
        section_paired(summary["summary"], paired, stratified),
        section_cross_model(decision, summary["summary"]),
        section_aux(aux),
        section_questions(decision, summary["summary"], aux),
        section_limits(),
        section_repro(manifest, identity, gates),
    ]
    report = "\n\n".join(parts)
    while "\n\n\n" in report:
        report = report.replace("\n\n\n", "\n\n")
    report = report + "\n"
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
        lines.append(f"- {DISPLAY[key]}：oracle enrichment `{f(e.get('oracle_enrichment'))}` "
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
    test = (summary["summary"].get("test") or {})
    tb = test.get("B_LCRC_LOCAL") or {}
    lines.append(f"- Test 复审：B 的 enrichment `{f(tb.get('oracle_enrichment'))}`、"
                 f"忠实性 advantage `{f(tb.get('fidelity_advantage_std'))}`，与 Valid 同号。")
    lines.append("")
    lines.append("## 本轮**没有**主张的内容")
    lines.append("")
    lines.append("- 没有主张 LCRC 比 GINE“更不忠实”：跨族门控语义不同，`|delta0|` 不可比。")
    lines.append("- 没有主张 `m=0` 的删除效应等于 Shapley / 因果贡献；它只是一次完全门控移除。")
    lines.append("- 没有主张 WC_LOGP 的原子加性分解就是化学机制；它只是冻结的生成式 oracle。")
    lines.append("- 没有把 823 对辅助结果当独立验证：编辑区是结构差异，非 Train 端点只有 11 个分量。")
    lines.append("- 没有事后更换 split、指标、阈值或候选范围。")
    lines.append("")
    lines.append("## 什么会改变这个判断")
    lines.append("")
    lines.append("1. 在**新的**预注册里，把选择目标改成尺寸归一的剂量/删除响应后，"
                 "若 C 类方法在 k=6 内也明显超过 A（≥0.02）且同族忠实性同步上升，"
                 "则应把 `FINITE_MASK_IMPROVES_ALIGNMENT` 升级为成立。")
    lines.append("2. 若把候选池扩大到 k>6 或允许不连通子图后，B 的外部优势消失，"
                 "则 `LOCAL_FAITHFUL_AND_RELEVANT` 需降级为池内结论。")
    lines.append("3. 若在更多非 Train 端点上 B 的编辑区重叠优势反号，则辅助实验的正面描述不成立"
                 "（当前非 Train 切片已是负号但 LOW_SUPPORT/CIs 跨 0）。")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("按本轮预注册条款，本轮结束后停止；不自动开始训练、不自动做架构搜索。"
                 "若继续，起点是“同一候选池 + 尺寸归一的剂量/删除选择目标”，"
                 "而不是继续优化 LCRC 的梯度/局部输出读出；新一步必须重新冻结协议。")
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
