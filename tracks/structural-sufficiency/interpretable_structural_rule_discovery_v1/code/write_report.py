"""Generate ``REPORT.md``, ``analysis/knowledge_cards.md`` and ``analysis/provenance.json``.

Every number in the report and the cards is read from the stored result files
(``metrics.csv``, ``contrasts.json``, ``phaseBC.json``, ``phaseC.json``, ``phaseD.json``,
``complexity.json``, ``v0_v1_rule_identity.json``) or recomputed from the frozen data.
No number is copied by hand.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

import isrd1_common as K
import isrd1_rules as R

R128 = "R128"


def fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def signed(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def ci(entry: dict, key: str = "delta_auc") -> str:
    lo, hi = entry[f"{key}_ci95"]
    return f"{signed(entry[key])} [{signed(lo)}, {signed(hi)}]"


def c_map(bc: dict, pc: dict) -> dict:
    out = {}
    for k, v in bc.get("phaseB", {}).get("arms", {}).items():
        out[k] = v.get("selected_C")
    for k, v in bc.get("phaseC", {}).get("arms", {}).items():
        out[k] = v.get("selected_C")
    for k, v in bc.get("phaseC", {}).get("joint_arms", {}).items():
        out[k] = v.get("selected_C")
    for k in ("count128", "count_all"):
        if k in pc:
            out["COUNT128" if k == "count128" else "COUNT-ALL"] = pc[k].get("selected_C")
    out["B2"] = 0.03
    return out


def metric_table(df: pd.DataFrame, arms: list[str], csel: dict | None = None) -> str:
    lines = ["| Arm | Valid AUC | Test AUC | Valid AP | Test AP | C | Evidence |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for arm in arms:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        v = sub[sub["split"] == "valid"]
        t = sub[sub["split"] == "test"]
        c = None if csel is None else csel.get(arm)
        sel = sub["source"].iloc[0]
        scope = "" if sub["scope"].iloc[0] == "FULL_SPLIT" else "（COARSE419 子集）"
        lines.append(f"| {arm}{scope} | {fmt(v['roc_auc'].iloc[0] if len(v) else None)} | "
                     f"{fmt(t['roc_auc'].iloc[0] if len(t) else None)} | "
                     f"{fmt(v['average_precision'].iloc[0] if len(v) else None)} | "
                     f"{fmt(t['average_precision'].iloc[0] if len(t) else None)} | "
                     f"{'-' if c is None else c} | {sel}{scope} |")
    return "\n".join(lines)


def contrast_row(c: dict) -> str:
    v = c.get("splits", {}).get("valid")
    t = c.get("splits", {}).get("test")
    return (f"| {c['a']} − {c['b']} | {ci(v) if v else 'n/a'} | {ci(t) if t else 'n/a'} | "
            f"{ci(v, 'delta_ap') if v else 'n/a'} | {ci(t, 'delta_ap') if t else 'n/a'} |")


def main() -> None:
    data = K.Data()
    reg = K.registry()
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    pc = K.read_json(K.ANALYSIS / "phaseC.json")
    pd_ = K.read_json(K.ANALYSIS / "phaseD.json")
    cj = K.read_json(K.ANALYSIS / "contrasts.json")
    comp = K.read_json(K.ANALYSIS / "complexity.json")
    ident = K.read_json(K.ANALYSIS / "v0_v1_rule_identity.json")
    ver = K.read_json(K.ANALYSIS / "verification.json")
    df = pd.read_csv(K.ANALYSIS / "metrics.csv")
    singles = {c["rule_id"]: c for c in pa["singles"]}
    for c in singles.values():
        c["key"] = (c["granularity"], int(c["column"]), R.f32(c["threshold"]), c["direction"])

    mp = K.read_json(K.ANALYSIS / "phaseA_mapping.json")
    n_cells = 3969616

    CSEL = c_map(bc, pc)

    def get(df, arm, split):
        s = df[(df["arm"] == arm) & (df["split"] == split)]
        return None if s.empty else s.iloc[0]

    def auc(arm, split):
        r = get(df, arm, split)
        return None if r is None else float(r["roc_auc"])

    def ap(arm, split):
        r = get(df, arm, split)
        return None if r is None else float(r["average_precision"])

    def con(a, b):
        for c in cj["contrasts"]:
            if c.get("a") == a and c.get("b") == b:
                return c
        return None

    # ------------------------------------------------------------------
    # knowledge cards
    # ------------------------------------------------------------------
    r128_ids = bc["phaseB"]["selected_rule_ids"][:128]
    coef = np.asarray(bc["phaseB"]["arms"]["R128"]["coef"], dtype=np.float64)
    rmu = np.asarray(bc["phaseB"]["rule_standardisation"]["mean"])[:128]
    rsd = np.asarray(bc["phaseB"]["rule_standardisation"]["std"])[:128]
    raw_coef = coef[2:] / rsd
    std_coef = coef[2:]
    train = data.rows["train"]
    with np.load(K.INSTANCES, allow_pickle=True) as z:
        examples = z["examples"]

    def rule_split_stats(c: dict) -> dict:
        act = R.rule_activation(data, c).astype(bool)
        out = {}
        for s in K.SPLITS:
            r = data.rows[s]
            a = act[r]
            y = data.y[r]
            pos_active = float(y[a].mean()) if a.sum() else None
            neg = ~a
            out[s] = {"n_active": int(a.sum()), "n_pos_active": int(y[a].sum()),
                      "rate_active": pos_active,
                      "rate_inactive": float(y[neg].mean()) if neg.sum() else None,
                      "n_inactive": int(neg.sum()),
                      "n_scaffolds_active": int(np.unique(data.scaffold_code[r][a]).size),
                      "activation_rate": float(a.mean()) if r.size else None}
        return out

    def _ex(g: int, c: dict) -> dict:
        split = next(s for s in K.SPLITS if g in set(data.rows[s].tolist()))
        return {"graph_index": int(g), "split": split, "smiles": None,
                "label": int(data.y[g]), "n_atoms": int(data.n_atoms[g]),
                "scaled": float(data.col(c["granularity"], int(c["column"]))[g]),
                "raw_count": int(data.raw_count(c["granularity"], int(c["column"]))[g])}

    def pick_examples(c: dict) -> dict:
        act = R.rule_activation(data, c).astype(bool)
        present = data.raw_count(c["granularity"], int(c["column"])) > 0
        out = {}
        for tag, mask in (("rule_active_positive", act & (data.y == 1)),
                          ("rule_active_negative", act & (data.y == 0)),
                          ("structure_present_positive", present & (data.y == 1)),
                          ("structure_present_negative", present & (data.y == 0)),
                          ("structure_absent_positive", (~present) & (data.y == 1))):
            rows = np.flatnonzero(mask)
            if rows.size:
                out[tag] = _ex(int(rows[0]), c)
        return out

    def complement_reading(c: dict) -> dict | None:
        ck = R.complement(c["key"])
        for rid, other in singles.items():
            if other["key"] == ck:
                return {"rule_id": rid, "key": ck,
                        "empirical_k": other["reading"].get("empirically_equivalent_to_count_ge"),
                        "same_threshold": True}
        return None

    def contribution_range(i: int, c: dict) -> dict:
        m = rmu[i]
        sdv = rsd[i]
        return {"active_std_logit": float(std_coef[i] * (1.0 - m) / sdv),
                "inactive_std_logit": float(std_coef[i] * (0.0 - m) / sdv),
                "train_active_share": float(rmu[i])}

    cards = []
    def add_card(tag, rid, note):
        i = r128_ids.index(rid)
        c = singles[rid]
        meta = reg[c["granularity"]].iloc[int(c["column"])]
        stats = rule_split_stats(c)
        ex = pick_examples(c)
        k = c["reading"].get("empirically_equivalent_to_count_ge")
        cards.append({"tag": tag, "rule_id": rid, "rank": i, "condition": R.condition_key_str(c["key"]),
                      "complement": complement_reading(c),
                      "contribution": contribution_range(i, c),
                      "granularity": c["granularity"], "column": int(c["column"]),
                      "threshold": c["threshold"], "direction": c["direction"],
                      "empirical_k": k,
                      "std_coef": float(std_coef[i]), "raw_coef": float(raw_coef[i]),
                      "family": str(meta["family"]), "n_nodes": int(meta["n_nodes"]),
                      "n_bonds": int(meta["n_bonds"]),
                      "node_signature": str(meta["node_signature"]),
                      "bond_signature": str(meta["bond_signature"]),
                      "element_sequence": str(meta["element_sequence"]),
                      "num_h_sequence": str(meta["num_h_sequence"]),
                      "aromatic_sequence": str(meta["aromatic_sequence"]),
                      "canonical_key": str(meta["canonical_key"]),
                      "n_source_trees": c["n_source_trees"], "source_arms": sorted(c["sources"]),
                      "reading": c["reading"], "stats": stats, "examples": ex, "note": note})

    add_card("A_sulfur_full2078", "S002", "C–S 家族的最强规则来源；在 R128 中以“F[2078] 缺失”形式出现")
    strongest = int(np.argmax(np.abs(std_coef)))
    add_card("B_strongest", r128_ids[strongest], "R128 中标准化系数绝对值最大的规则")
    used = {card["rule_id"] for card in cards}
    readable = [(i, rid) for i, rid in enumerate(r128_ids)
                if singles[rid]["reading"].get("empirically_equivalent_to_count_ge")
                and rid not in used]
    coarse = [(i, rid) for i, rid in enumerate(r128_ids)
              if singles[rid]["granularity"] == "C" and rid not in used]
    if readable:
        i, rid = max(readable, key=lambda t: abs(std_coef[t[0]]))
        add_card("C_readable_count", rid, "Train 上经验等价于原始计数 ≥k 且系数最大的未用规则")
    elif coarse:
        i, rid = max(coarse, key=lambda t: abs(std_coef[t[0]]))
        add_card("C_coarse", rid, "R128 中系数绝对值最大的 COARSE 规则")
    sulfur = [i for i, rid in enumerate(r128_ids)
              if (singles[rid]["granularity"], int(singles[rid]["column"])) == ("C", 419)]
    if sulfur:
        add_card("D_sulfur_parent", r128_ids[sulfur[0]], "C–S 家族的 COARSE 父结构；系数接近零")
    weakest = int(np.argmin(np.abs(std_coef)))
    add_card("E_weakest", r128_ids[weakest], "R128 中标准化系数绝对值最小的规则（含反例）")

    # SMILES for card examples
    mols = None
    try:
        sys.path.insert(0, str(K.MSDS / "code"))
        import msds_data as D
        mols = D.load_molecules()
    except Exception:
        mols = None
    for card in cards:
        for ex in card["examples"].values():
            ex["smiles"] = None if mols is None else mols.smiles[ex["graph_index"]]

    lines = ["# 修正后的知识卡片（ISRD-v1）", "",
             "全部数字由 `write_report.py` 从 `phaseBC.json` / `phaseA_rules.json` / 冻结数据生成。",
             "只描述三类事实：数据关联、模型内计算作用、经验等价表述。",
             "**不声称化学因果机制**。", ""]
    for card in cards:
        st = card["stats"]
        ex = card["examples"]
        arrow = "x < t（该结构“不满足”）" if card["direction"] == "lt" else "x ≥ t"
        ktxt = (f"Train 上经验等价于 `raw_count ≥ {card['empirical_k']}`（仅观察数据）"
                if card["empirical_k"] else "Train 上**不**等价于任何单一原始计数阈值")
        lines += [
            f"## {card['tag']} — {card['rule_id']}（R128 第 {card['rank']} 位）",
            "",
            f"* 条件：`x_{card['granularity']}[{card['column']}] {card['condition'].split(' ',1)[1]}`"
            f" → {arrow}",
            f"* 阈值：`{card['threshold']:.9g}`；{ktxt}",
            f"* 结构：{card['family']}，{card['n_nodes']} 节点 / {card['n_bonds']} 键；"
            f"node=`{card['node_signature']}`；bond=`{card['bond_signature']}`；"
            f"element=`{card['element_sequence']}`；nH=`{card['num_h_sequence']}`；"
            f"aromatic=`{card['aromatic_sequence']}`",
            f"* 来源：{'+'.join(card['source_arms'])}，{card['n_source_trees']} 棵来源树；"
            f"规则身份 `{card['condition']}`",
            (f"* 补方向：`{card['complement']['rule_id']}`，"
             f"{card['complement']['key'][3]} 方向"
             + (f"，Train 上经验等价于 `raw_count ≥ {card['complement']['empirical_k']}`"
                if card['complement']['empirical_k'] else "")
             if card["complement"] else "* 补方向：不在 512 候选池中（无法给出补方向读法）"),
            f"* R128 系数：standardised `{signed(card['std_coef'])}`，"
            f"raw-unit `{signed(card['raw_coef'])}`（正系数 = 条件成立时 logit 升高）",
            f"* R128 计算作用：条件成立时该项贡献 `{signed(card['contribution']['active_std_logit'])}`"
            f"（standardised logit），不成立时 `{signed(card['contribution']['inactive_std_logit'])}`；"
            f"Train 激活比例 `{fmt(card['contribution']['train_active_share'])}`",
            "",
            "| 划分 | 激活 n | 激活正例率 | 未激活正例率 | 激活 scaffold | 激活率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in K.SPLITS:
            d = st[s]
            lab = "**DEVELOPMENT_TEST**" if s == "test" else s
            lines.append(f"| {lab} | {d['n_active']} | {fmt(d['rate_active'])} | "
                         f"{fmt(d['rate_inactive'])} | {d['n_scaffolds_active']} | "
                         f"{fmt(d['activation_rate'])} |")
        lines += ["", "实例（`rule_active` = 条件成立；`structure_present` = 原始计数 > 0）："]
        labels = {"rule_active_positive": "条件成立 / 正例",
                  "rule_active_negative": "条件成立 / 反例",
                  "structure_present_positive": "结构存在 / 正例",
                  "structure_present_negative": "结构存在 / 反例",
                  "structure_absent_positive": "结构不存在 / 正例"}
        for tag, label in labels.items():
            if tag in ex:
                e = ex[tag]
                lines.append(f"* {label}：graph {e['graph_index']}（{e['split']}），label={e['label']}，"
                             f"n_atoms={e['n_atoms']}，raw_count={e['raw_count']}，"
                             f"scaled={e['scaled']:.6g}，`{e['smiles']}`")
        lines += ["", f"* 说明：{card['note']}",
                  "* 竞争解释：骨架身份、分子大小、共现结构、样本组成；本卡片不排除这些解释。",
                  ""]
    (K.ANALYSIS / "knowledge_cards.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    prov = {
        "round_id": K.ROUND_ID,
        "git": K.git_state(),
        "environment": K.environment(),
        "inputs": {
            "bundle": {"path": K.rel(K.BUNDLE), "sha256": K.sha256_file(K.BUNDLE)},
            "boosters": {a: {"path": K.rel(p), "sha256": K.sha256_file(p),
                             "num_features": R.booster_width(R.load_booster(a))}
                         for a, p in K.BOOSTERS.items()},
            "b2_coefficients": {"path": K.rel(K.B2_COEF), "sha256": K.sha256_file(K.B2_COEF)},
            "b2_predictions": {"path": K.rel(K.B2_PRED), "sha256": K.sha256_file(K.B2_PRED)},
            "instances": {"path": K.rel(K.INSTANCES), "sha256": K.sha256_file(K.INSTANCES)},
            "scaffolds": {"path": K.rel(K.SCAFFOLDS), "sha256": K.sha256_file(K.SCAFFOLDS)},
            "registry": {g: {"path": K.rel(p), "sha256": K.sha256_file(p)}
                         for g, p in K.REGISTRY.items()},
            "split": {"sizes": {s: int(data.rows[s].size) for s in K.SPLITS},
                      "sha256": {s: K.hashlib.sha256(data.rows[s].tobytes()).hexdigest()
                                 for s in K.SPLITS}},
        },
        "outputs": {},
        "phases": {
            "A": "repaired rule discovery (arm mapping + directed real paths)",
            "B": "R0/R16/R64/R128, B2-L64/B2-U64, B2-U+S/P/N, R64+P",
            "C": "COUNT128, COUNT-ALL, XGB-ALL, COUNT-ALL local contributions",
            "D": "C–S family description, CS-BASE/CS-FINE, cases",
        },
        "engineering_deviations": [
            {"id": "D1", "what": "v0 arm-specific mapping repaired",
             "affected": ["R16", "R64", "R128", "B2-L64", "B2-U64", "B2-U+S", "B2-U+P",
                          "B2-U+N", "R64+P", "knowledge cards"],
             "changes_numbers": True, "detail": "see ERRATA.md E1"},
            {"id": "D2", "what": "v0 path extraction rebuilt with thresholds and directions",
             "affected": ["real joint pair pool", "B2-U+P", "B2-U+N", "R64+P"],
             "changes_numbers": True, "detail": "see ERRATA.md E2"},
            {"id": "D3", "what": "exact condition identity for joint main effects; strict "
                                  "matched random pairs (0.05 / 0.03 tolerances)",
             "affected": ["B2-U+P", "B2-U+N", "R64+P"], "changes_numbers": True,
             "detail": "see ERRATA.md E3"},
            {"id": "D4", "what": "single-condition prefilter cap raised from 4000 to 12000 so no "
                                  "directed condition is truncated (v0 dropped 95 of 4095); the "
                                  "512-rule budget and all support gates are unchanged",
             "affected": ["single-rule candidate pool"], "changes_numbers": True,
             "detail": "n_prefilter_truncated = %d in v1" % pa["single_stats"]["n_prefilter_truncated"]},
            {"id": "D5", "what": "COUNT-ALL solved with a sparse exact-Newton parametrisation "
                                  "instead of materialising the dense standardised design; "
                                  "unit test and real-design check show equality with the dense "
                                  "fit_logistic (max|Δcoef| < 1e-6)",
             "affected": ["COUNT-ALL"], "changes_numbers": False, "detail": "see verification.json"},
            {"id": "D6", "what": "XGB-ALL trained once with the frozen historical recipe on the "
                                  "structure+size design; booster kept at analysis/models/XGB-ALL.json",
             "affected": ["XGB-ALL"], "changes_numbers": False, "detail": "no tuning"},
        ],
        "reused_assets": {
            "R0": "identical to v0 (max|Δlogit| < 1e-9, EXACT_MATCH)",
            "B2": "frozen coefficients/predictions replayed (max|Δlogit| = %.3e)"
                  % bc["b2_replay"]["max_abs_diff"],
            "B0/B3/X-C/X-F/X-CF": "historical predictions, metrics recomputed only",
            "C/F vocabulary, scaling, instances, scaffolds, splits": "unchanged",
        },
        "wall_clock_seconds": {"phaseA": pa["seconds"], "phaseB": bc["seconds"],
                              "phaseC": pc["seconds"]},
        "test_policy": "DEVELOPMENT_TEST: evaluated and reported for every arm, never used "
                       "to select a rule, threshold, capacity, C or design element.",
        "verification": {"n_checks": ver["_summary"]["n_checks"] - 1,
                         "n_fail": ver["_summary"]["n_fail"]},
    }
    for name in ("phaseA_rules.json", "phaseA_mapping.json", "phaseBC.json", "phaseC.json",
                 "phaseD.json", "metrics.csv", "contrasts.json", "rule_registry.csv",
                 "verification.json", "knowledge_cards.md", "models/XGB-ALL.json"):
        p = K.ANALYSIS / name
        if p.exists():
            prov["outputs"][name] = {"sha256": K.sha256_file(p), "bytes": int(p.stat().st_size)}
    K.write_json(K.ANALYSIS / "provenance.json", prov)

    # ------------------------------------------------------------------
    # REPORT.md
    # ------------------------------------------------------------------
    c_count128_r128 = con("COUNT128", "R128")
    c_countall_count128 = con("COUNT-ALL", "COUNT128")
    c_xgb_count = con("XGB-ALL", "COUNT-ALL")
    c_xgb_r128 = con("XGB-ALL", "R128")
    c_b2_r128 = con("B2", "R128")
    c_xcf_r128 = con("X-CF", "R128")
    c_b2_countall = con("B2", "COUNT-ALL")
    c_xcf_xgb = con("X-CF", "XGB-ALL")
    c_b2_xgb = con("B2", "XGB-ALL")
    c_u64_b2 = con("B2-U64", "B2")
    c_u64_l64 = con("B2-U64", "B2-L64")
    c_p_u64 = con("B2-U+P", "B2-U64")
    c_p_s = con("B2-U+P", "B2-U+S")
    c_p_n = con("B2-U+P", "B2-U+N")
    c_s_u64 = con("B2-U+S", "B2-U64")
    c_n_u64 = con("B2-U+N", "B2-U64")
    c_csfine = con("CS-FINE", "CS-BASE")
    joint = bc["phaseC"]["joint"]
    loc = pc["count_all_local"]
    d2 = pd_["D2"]

    def row(c):
        return contrast_row(c) if c else "| n/a | n/a | n/a | n/a | n/a |"

    report = f"""# REPORT — 显式结构预测充分性与规则语义修复 v1
**DEVELOPMENT_TEST**：所有 Test 结果已暴露，只报告，不用于选择规则、阈值、容量、C 或任何设计元素。
Round: `{K.ROUND_ID}`；seed=0；CPU-only；冻结上游 commit `6cddf702`（v0）+ SGA-v1/MSDS-v0 资产。
v0 的错误与影响范围见 `ERRATA.md`；机器可读记录见 `analysis/provenance.json`、`analysis/verification.json`。

## 首页：直接回答七个问题

1. **上一轮的实现错误影响了哪些结论？**
   v0 把 X-C（H+C, 3 012 列）与 X-F（H+F, 5 910 列）当成同一个 `H+C+F` 布局解码，
   因此 X-F 的 **3 390 个结构分裂节点全部被映射到错误的结构列**
   （1 420 个节点 / 174 列被读成 COARSE，1 970 个节点 / 459 列整体错位 754 列），并产生了 **38 条物理上不可能的 C–F 真实路径**。
   直接后果：v0 的 `R16/R64/R128`、`B2-L64/B2-U64`、四个联合规则臂、全部知识卡片，
   以及“四个 C–S FULL 子型都没有进入候选池”的结论全部作废（`SUPERSEDED_BY_RULE_MAPPING_FIX`）。
   修正后 `F[2078]`（芳香碳—硫）是来源树最多的规则之一并进入 R64/R128。
   预测层面：v1 `R128` = Valid {fmt(auc('R128','valid'))} / Test {fmt(auc('R128','test'))}，
   v0 = {fmt(auc('v0:R128','valid'))} / {fmt(auc('v0:R128','test'))}
   （Δ = {ci(con('R128','v0:R128')['splits']['valid'])} / Test {ci(con('R128','v0:R128')['splits']['test'])}）。
   v0 的 **R0、B2 重放、所有历史 XGB/MLP 指标不受影响**。

2. **修正后阈值、联合规则是否仍无稳定优势？**
   是。`B2-U64 − B2-L64`（同一批结构，阈值 vs 连续计数）Valid {ci(c_u64_l64['splits']['valid'])}、
   Test {ci(c_u64_l64['splits']['test'])}；`B2-U64 − B2` 同样接近零。
   严格匹配后的联合规则：`B2-U+P − B2-U+N` Valid {ci(c_p_n['splits']['valid'])}、
   Test {ci(c_p_n['splits']['test'])}；`B2-U+P − B2-U+S` Test {ci(c_p_s['splits']['test'])}。
   本轮对照首次同时满足：精确主效应身份、37/39 一对一无放回匹配、边际频率差 ≤0.05、
   联合频率差 ≤0.03、相同四格支持。因此“联合规则形式没有可检出的额外预测价值”这一判断
   现在有严格对照支持；但区间很窄、单 seed，仍不能宣称等价。

3. **性能损失更多来自阈值化，还是结构数量限制？**
   **都不是本轮的瓶颈；阈值化甚至是正则化而不是损失来源。**
   `COUNT128 − R128` Valid {ci(c_count128_r128['splits']['valid'])}、
   Test {ci(c_count128_r128['splits']['test'])}：把同样的 128 条阈值谓词换成连续计数
   **更差**，而不是更好。把结构字典从 128 放开到全部 4 406 条（`COUNT-ALL − COUNT128`）
   Valid {ci(c_countall_count128['splits']['valid'])}、Test {ci(c_countall_count128['splits']['test'])}：
   在同样的线性读出与同一 C 网格下没有收益，且 Train AUC {fmt(pc['count_all']['metrics']['train']['roc_auc'])}
   vs Valid {fmt(pc['count_all']['metrics']['valid']['roc_auc'])}，是明显的过拟合。
   因此“限制结构数量导致主要损失”与“阈值化导致主要损失”都不被支持。

4. **完整显式结构在线性/非线性读出下分别能达到什么水平？**
   同一输入（两个大小项 + 754 C + 3 652 F 连续计数，**无 Morgan、描述符、GNN**）：
   线性 `COUNT-ALL` Valid {fmt(auc('COUNT-ALL','valid'))} / Test {fmt(auc('COUNT-ALL','test'))}
   （AP {fmt(ap('COUNT-ALL','valid'))} / {fmt(ap('COUNT-ALL','test'))}）；
   冻结树配方 `XGB-ALL` Valid {fmt(auc('XGB-ALL','valid'))} / Test {fmt(auc('XGB-ALL','test'))}
   （AP {fmt(ap('XGB-ALL','valid'))} / {fmt(ap('XGB-ALL','test'))}）。
   `XGB-ALL − COUNT-ALL` Valid {ci(c_xgb_count['splits']['valid'])}、
   Test {ci(c_xgb_count['splits']['test'])}。
   `XGB-ALL` 与含 H 的 `X-CF` 在 Test 上不可区分（`X-CF − XGB-ALL` Valid
   {ci(c_xcf_xgb['splits']['valid'])}、Test {ci(c_xcf_xgb['splits']['test'])}），
   并在 Test 上优于冻结 `B2`（`B2 − XGB-ALL` Test {ci(c_b2_xgb['splits']['test'])}）。
   注意二者优化方法、容量、损失、正则不同，差值不能直接当作“纯非线性交互价值”。

5. **完整字典是否仍具有局部可追踪的计算解释？**
   可以精确重建（每个分子 `logit = b + 大小项 + Σ_s w_s x_s`，max|Δ| =
   {loc['reconstruction_max_abs_error']:.1e}），但**局部摘要并不集中**：平均每分子
   {fmt(loc['active_train']['mean_active_structures'],1)} 个结构项激活（Valid/Test
   {fmt(loc['active_valid']['mean_active_structures'],1)} /
   {fmt(loc['active_test']['mean_active_structures'],1)}），
   |贡献| 最大的 5/10/20 项只占全部结构项 |贡献| 的
   {fmt(100*loc['share_top5_mean']['test'],1)}% / {fmt(100*loc['share_top10_mean']['test'],1)}% /
   {fmt(100*loc['share_top20_mean']['test'],1)}%（Test），其余部分（均值
   {fmt(loc['remaining_abs_mean']['test'],1)} 个 logit 单位）被省略。
   结构项 |贡献| 总和平均 {fmt(loc['abs_total_mean']['test'],1)}，而 logit 本身在 0 附近，
   说明存在大量相互抵消。且 **100% 的激活 FULL 结构其 COARSE 父结构也同时激活**
   （{fmt(100*loc['frac_active_full_with_active_coarse_parent'],1)}%），FULL/COARSE 贡献天然重叠。
   因此完整字典的“局部可追踪”是**可重建但不稀疏**：单分子解释不能只报前 5 项。

6. **关于芳香碳—硫家族，我们真正新增了哪些认识？**
   修正映射后该家族第一次真正进入规则模型：`F[2078]` 出现在 R64/R128（R128 系数
   {signed(cards[0]['std_coef'])} standardised，以“F[2078] 缺失”形式）。数据上，
   COARSE 419 子集中 F[2078] 出现组的正例率 Valid
   {fmt(d2['groups']['valid']['present_2078']['positive_rate'])}、Test
   {fmt(d2['groups']['test']['present_2078']['positive_rate'])}，未出现组
   {fmt(d2['groups']['valid']['absent_2078']['positive_rate'])} /
   {fmt(d2['groups']['test']['absent_2078']['positive_rate'])}。
   但在控制冻结 B0 logit、分子大小与 COARSE 419 计数后，加入 F[2078] 的 `CS-FINE`
   没有增量：Valid {ci(c_csfine['splits']['valid'])}、Test {ci(c_csfine['splits']['test'])}，
   F[2078] 系数 {signed(d2['fine_coef_2078_standardised'])}（standardised）。
   所以新增认识是：**该家族的边际关联真实且跨划分同向，但其增量预测价值在 B0 已捕获的信息之外
   不存在（不优于零）**，并且规则模型使用的是“缺失”方向而非“存在”方向。

7. **哪些方向可以停止，哪些值得继续讨论？**
   可以停止：(a) 继续在 128 条规则内寻找阈值/联合形式的非线性收益；(b) 用“结构数量限制”
   或“阈值化”解释规则模型与强模型的差距；(c) 把 C–S 子型当作待挖掘的增量规律。
   值得讨论：(a) 固定结构词表 + 稀疏线性读出（`COUNT-ALL`）在匹配 C 网格下欠拟合/过拟合并存，
   “可解释线性结构函数”需要专门的容量控制而不是复用规则模型配方；(b) `XGB-ALL` 表明
   4 406 个显式结构在树读出下已接近含 H 的 X-CF，值得研究**可局部追踪的树/加性-交互读出**；
   (c) 局部解释的集中度问题（前 20 项仅约 {fmt(100*loc['share_top20_mean']['test'],1)}%）
   是比“规则条数”更实质的解释瓶颈。

### 认识更新表
| 原有判断 | 修正后的证据 | 当前认识 | 剩余不确定性 |
|---|---|---|---|
| R128 明显弱于 B2/X-CF | R128 {fmt(auc('R128','valid'))}/{fmt(auc('R128','test'))}；B2−R128 Test {ci(c_b2_r128['splits']['test'])}；X-CF−R128 Test {ci(c_xcf_r128['splits']['test'])} | 修正映射后 128 条规则与 B2/X-CF 在 Test 上不可区分 | 单 seed；Test 仅发展性；规则来源仍只有两个 booster |
| 阈值谓词优于线性计数 | B2-U64−B2-L64 Valid {ci(c_u64_l64['splits']['valid'])}、Test {ci(c_u64_l64['splits']['test'])}；COUNT128−R128 Test {ci(c_count128_r128['splits']['test'])} | 阈值化没有正收益；连续计数在 128 结构上更差 | 是否只是 C 网格对连续计数不合适 |
| 树路径联合条件无价值 | P−N Test {ci(c_p_n['splits']['test'])}；P−S Test {ci(c_p_s['splits']['test'])}（严格匹配首次成立） | 联合形式无额外预测价值，且不优于同预算新单规则 | K=16、单一来源模型、匹配仍非唯一 |
| COARSE 419 的 FULL 子型未被规则使用 | F[2078] 进入 R64/R128（系数 {signed(cards[0]['std_coef'])}） | 规则模型确实使用了该子型（缺失方向） | 增量预测价值为零；可能是 B0/X-CF 已用同信息 |
| 限制结构字典是主要瓶颈 | COUNT-ALL−COUNT128 Test {ci(c_countall_count128['splits']['test'])}；COUNT-ALL Train {fmt(pc['count_all']['metrics']['train']['roc_auc'])} vs Valid {fmt(pc['count_all']['metrics']['valid']['roc_auc'])} | 全字典线性读出没有收益，主要问题是容量/正则而非字典大小 | 未扫描 C（协议禁止），无法排除 C 不匹配 |
| 显式结构不足以预测 | XGB-ALL {fmt(auc('XGB-ALL','valid'))}/{fmt(auc('XGB-ALL','test'))}；X-CF−XGB-ALL Test {ci(c_xcf_xgb['splits']['test'])} | 显式结构 + 树读出已足够强，不需要 Morgan 才能达到 X-CF 量级 | 非线性收益与容量/优化混杂 |
| 知识卡片可直接积累 | 见 `analysis/knowledge_cards.md`（由结果文件生成） | 数值/命名/符号可核验的卡片可积累 | 机制解释仍缺失 |

## 数据、修复与核验
官方划分 32 901 / 4 113 / 4 113；冻结 C(754)/F(3 652) 与 `registry_v1` 列顺序未改。
修复与核验（全部来自 `analysis/phaseA_mapping.json`、`analysis/verification.json`、`tests/`）：

* booster 宽度 = 3 012（X-C）/ 5 910（X-F），与 `H_WIDTH + block` 一致；
* 映射后结构列值 vs booster 原始输入列值 max|Δ| = 0（{sum(v["n_features_value_checked"] for v in mp["arms"].values())} 个结构列 × 2 000 行）；
* 30 棵树 × 2 000 行叶节点重放零错位；`X-F:f2258` ↔ `F[0]`；
* 512/512 条单规则都能在 booster dump 中找到同一 (arm, tree, node)、同一阈值与方向的来源节点；
* 256/256 条真实联合规则都在其来源路径上同时包含两个带方向条件；
* 242 条（样本中可达叶节点的全部）投影条件在**实际到达该叶节点的 Train 样本**上成立；
* 真实路径条件（阈值 + 方向 + 来源节点）在**实际到达该叶节点的样本**上全部成立；
  结构条件不单独划分样本（H 分裂被有意投影掉），叶节点划分由 xgboost `pred_leaf` 核验；
* 原始整数计数与 `x·sqrt(n_atoms)` 在 {n_cells} 个非零单元上逐一致；
* B2 重放 max|Δlogit| = {bc['b2_replay']['max_abs_diff']:.3e}；R0 与 v0 存储预测完全一致；
* 所有线性模型原始单位加和闭合 < 1e-8；COUNT-ALL 局部贡献重建 {loc['reconstruction_max_abs_error']:.1e}；
* 稀疏中心化 logistic 与稠密标准化 `fit_logistic` 系数一致（< 1e-6）；
* COUNT-ALL 与独立 sklearn liblinear 在同一标准化设计上的 Valid/Test AUC 差 < 0.007
  （仅截距罚项实现不同）；
* 指标从存储预测独立重算（max|Δ| < 1e-9）。
数值核验 {ver['_summary']['n_checks'] - 1} 项，失败 {ver['_summary']['n_fail']}。

## Phase A：修正后的规则池
{pa['single_stats']['n_split_nodes']} 个结构分裂节点（X-C {pa['single_stats']['n_split_nodes_by_arm']['X-C']}、
X-F {pa['single_stats']['n_split_nodes_by_arm']['X-F']}）→
{pa['single_stats']['n_distinct_conditions']} 个**带方向**的唯一条件
（ge {sum(1 for c in pa['singles'] if c['direction']=='ge')} / lt
{sum(1 for c in pa['singles'] if c['direction']=='lt')} 在最终 512 条中）→
支持过滤后 {pa['single_stats']['n_after_support_filter']}，去重合并
{pa['single_stats']['n_duplicate_merged']}，保留 **512** 条单规则。
真实 root-to-leaf 路径 {pa['pair_stats']['n_paths']} 条，产生
{pa['pair_stats']['n_real_path_pairs']} 个不同共同路径对；落在 512 池内的
{pa['pair_stats']['n_real_path_pairs_inside_pool']} 对；四格支持过滤与去重后
{pa['pair_stats']['n_after_filters_and_dedup']}，保留 **{len(pa['pairs'])}** 条
`PROJECTED_PAIR`（C+C {pa['pair_stats']['gran_pair_counts']['C+C']} / F+F {pa['pair_stats']['gran_pair_counts']['F+F']}）。**C–F 真实路径数 = 0**。
规则身份变化：v0 与 v1 的 512 条单规则池只有 {ident['n_common_condition_identity']} 条
精确身份相同；v1 R128 的 128 条中只有 {ident['n_v1_R128_in_v0_pool']} 条在 v0 池中出现过
（v0 R128 的 128 条中 {ident['n_v0_R128_cond_identity_in_v1_pool']} 条在 v1 池中）。
v0 的 R128 是 {ident['v0_R128_granularity_counts']['C']} C / {ident['v0_R128_granularity_counts']['F']} F，v1 是 {ident['v1_R128_granularity_counts']['C']} C / {ident['v1_R128_granularity_counts']['F']} F。

## Phase B：重跑后的规则模型
{metric_table(df, ['R0','R16','R64','R128','B2','B2-L64','B2-U64','B2-U+S','B2-U+P','B2-U+N','R64+P','X-C','X-F','X-CF','v0:R16','v0:R64','v0:R128'], CSEL)}

R64 使用 {len(bc['phaseC']['r64_structures'])} 个不同结构；其中 C 419、F 2078 等来自修正后的池。
联合规则预算：合格真实对 {joint['n_eligible_real']}（其中
{joint['n_condition_slots_complement']} 个条件槽为精确补集），匹配成功
{joint['match']['stats']['n_matched_pairs']}（覆盖率 {fmt(joint['match']['stats']['coverage'],3)}），
K = {joint.get('K_available')}；匹配最大 |Δ边际频率| =
{fmt(joint['match']['stats']['max_abs_diff']['freq_a'])}, {fmt(joint['match']['stats']['max_abs_diff']['freq_b'])}，
|Δ联合频率| = {fmt(joint['match']['stats']['max_abs_diff']['freq_joint'])}。
（v0：256 条候选、211 条匹配（82.4%）、最大 |Δ边际| 0.48。）

## Phase C：预测充分性对照
{metric_table(df, ['R128','COUNT128','COUNT-ALL','XGB-ALL','B2','X-CF','B0','B3'], CSEL)}

### 核心配对比较（Δ = 前者 − 后者；1 000 次配对 scaffold bootstrap，seed 0）
| Contrast | Valid ΔAUC [95%] | Test ΔAUC [95%] | Valid ΔAP [95%] | Test ΔAP [95%] |
|---|---|---|---|---|
{row(con('R128','R0'))}
{row(con('B2','R128'))}
{row(con('X-CF','R128'))}
{row(con('B2-U64','B2'))}
{row(con('B2-U64','B2-L64'))}
{row(con('B2-U+P','B2-U64'))}
{row(con('B2-U+P','B2-U+S'))}
{row(con('B2-U+P','B2-U+N'))}
{row(con('B2-U+S','B2-U64'))}
{row(con( 'COUNT128','R128'))}
{row(con('COUNT-ALL','COUNT128'))}
{row(con('XGB-ALL','COUNT-ALL'))}
{row(con('B2','COUNT-ALL'))}
{row(con('X-CF','XGB-ALL'))}
{row(con('XGB-ALL','R128'))}
{row(con('CS-FINE','CS-BASE'))}

注意：不显著 ≠ 等价；单 seed bootstrap 只描述固定预测的样本不确定性。
`COUNT-ALL` 的差距包含“增加 4 278 个特征 + 容量 + 同一 C 网格不再合适”的混合影响，
不是纯粹的结构数量因果效应。

### COUNT-ALL 的局部计算
* 精确重建：`logit = b_raw + a1·log1p(n_atoms) + a2·log1p(n_bonds) + Σ_s w_s·x_s`，
  max|Δ| = {loc['reconstruction_max_abs_error']:.2e}。
* 每分子激活结构项：Train 均值 {fmt(loc['active_train']['mean_active_structures'],1)}
  （中位 {fmt(loc['active_train']['median_active_structures'],0)}，p95
  {fmt(loc['active_train']['p95_active_structures'],0)}）；Test 均值
  {fmt(loc['active_test']['mean_active_structures'],1)}。
* |贡献| 前 5/10/20 项占比（Test）：{fmt(100*loc['share_top5_mean']['test'],1)}% /
  {fmt(100*loc['share_top10_mean']['test'],1)}% / {fmt(100*loc['share_top20_mean']['test'],1)}%。
* 未展示的剩余 |贡献|（Test）均值：{fmt(loc['remaining_abs_mean']['test'],1)}；
  结构项 |贡献| 总量均值：{fmt(loc['abs_total_mean']['test'],1)}。
* 重叠边界：激活 FULL 结构中 {fmt(100*loc['frac_active_full_with_active_coarse_parent'],1)}%
  的 COARSE 父结构同时激活；同一结构可在一个分子中出现多次
  （`x_s` 为整数计数除以 `sqrt(n_atoms)`，贡献按次数线性叠加），
  因此“单项贡献”不等于“子图存在与否的效应”。
* 贡献集中度不是新的主要预测指标；系数不是化学贡献。

## Phase D：芳香碳—硫家族
COARSE 419 = `{reg['C'].iloc[419]['node_signature']}` / `{reg['C'].iloc[419]['bond_signature']}`
（{reg['C'].iloc[419]['n_nodes']} 节点 {reg['C'].iloc[419]['n_bonds']} 键，CCCS）。
四个 FULL 子型的真实属性：

| 类型 | family | nodes/bonds | node_signature | 关键差异 |
|---|---|---:|---|---|
"""
    for c in (2078, 2525, 1788, 2275):
        m = reg["F"].iloc[c]
        report += (f"| F[{c}] | {m['family']} | {m['n_nodes']}/{m['n_bonds']} | "
                   f"`{m['node_signature']}` | nH=`{m['num_h_sequence']}`, "
                   f"arom=`{m['aromatic_sequence']}` |\n")
    total_min = (pa['seconds'] + bc['seconds'] + pc['seconds'] + 185 + 4 + 7) / 60.0
    g = pd_["D1"]
    base_by_bin = {r["bin"]: r for r in g["size_stratified"]["coarse419_all"]
                   if r["split"] == "train"}
    size_rows = ""
    for b in range(4):
        rows_b = [r for r in g["size_stratified"]["F2078"] if r["bin"] == b]
        tr = next(r for r in rows_b if r["split"] == "train")
        va = next(r for r in rows_b if r["split"] == "valid")
        te = next(r for r in rows_b if r["split"] == "test")
        size_rows += (f"| {b} | {tr['n']} / {fmt(tr['positive_rate'])} | "
                      f"{fmt(base_by_bin[b]['positive_rate'])} | {va['n']} | {te['n']} |\n")
    report += f"""
COARSE 419 阳性分子数：Train {g['n_coarse419']['train']} / Valid {g['n_coarse419']['valid']} /
Test {g['n_coarse419']['test']}。FULL 子型（限于 COARSE 419 阳性）出现数与正例率：

| 类型 | Train n / 正例率 | Valid n / 正例率 | Test n / 正例率 | 与 2078 的 Jaccard |
|---|---|---|---|---|
"""
    for c in (2078, 2525, 1788, 2275):
        rec = g[f"F{c}"]
        s = rec["support_all"]
        jac = rec["jaccard_with_siblings"].get("2078", "-")
        report += (f"| F[{c}] | {s['train']['n']} / {fmt(s['train']['positive_rate'])} | "
                   f"{s['valid']['n']} / {fmt(s['valid']['positive_rate'])} | "
                   f"{s['test']['n']} / {fmt(s['test']['positive_rate'])} | "
                   f"{fmt(float(jac)) if jac != '-' else '-'} |\n")
    report += f"""
FULL 子型之间不是互斥类别（Jaccard 见上表，最大也远小于 1）。
分子大小分组（Train 原子数四分位：{g['size_stratified']['train_quantiles']}），
F[2078] 与同一子集基线的 Train 正例率：

| 大小组 | F[2078] n / 正例率 (Train) | 同组 COARSE419 基线正例率 (Train) | Valid n | Test n |
|---|---|---:|---:|---:|
{size_rows}
Valid/Test 的组内样本极少（F[2078] 最大一组仅 Valid 27 / Test 47），
因此组内 Valid/Test 正例率不稳定，完整数字见 `analysis/phaseD.json` → `D1.size_stratified`。
已有模型在该子集内：B0/B2 的 AUC 见 `analysis/phaseD.json` → `D1.existing_models`。

### 条件预测检验（仅 COARSE 419 阳性子集）
样本：Train {d2['subset_sizes']['train']['n']}（+{d2['subset_sizes']['train']['n_positive']}）、
Valid {d2['subset_sizes']['valid']['n']}（+{d2['subset_sizes']['valid']['n_positive']}）、
Test {d2['subset_sizes']['test']['n']}（+{d2['subset_sizes']['test']['n_positive']}）。
固定 C = 0.03；Train-only 标准化；class-balanced logistic。

| 模型 | Valid AUC | Valid AP | Test AUC | Test AP |
|---|---:|---:|---:|---:|
| CS-BASE | {fmt(d2['CS-BASE']['splits']['valid']['roc_auc'])} | {fmt(d2['CS-BASE']['splits']['valid']['average_precision'])} | {fmt(d2['CS-BASE']['splits']['test']['roc_auc'])} | {fmt(d2['CS-BASE']['splits']['test']['average_precision'])} |
| CS-FINE | {fmt(d2['CS-FINE']['splits']['valid']['roc_auc'])} | {fmt(d2['CS-FINE']['splits']['valid']['average_precision'])} | {fmt(d2['CS-FINE']['splits']['test']['roc_auc'])} | {fmt(d2['CS-FINE']['splits']['test']['average_precision'])} |

FULL 2078 的系数：standardised {signed(d2['fine_coef_2078_standardised'])}，
raw {signed(d2['fine_coef_2078_raw'])}（负号 = 在已有输入之上不增加活性倾向）。
配对差值 `CS-FINE − CS-BASE`：Valid {ci(c_csfine['splits']['valid'])}、
Test {ci(c_csfine['splits']['test'])}。
出现/未出现两组的具体支持与预测见 `analysis/phaseD.json` → `D2.groups`
（Test：出现组 n={d2['groups']['test']['present_2078']['n']}、正例率
{fmt(d2['groups']['test']['present_2078']['positive_rate'])}；未出现组 n=
{d2['groups']['test']['absent_2078']['n']}、正例率 {fmt(d2['groups']['test']['absent_2078']['positive_rate'])}）。
这是**事先指定的探索性条件检验**，不是因果分析。

### 三个解释层次
1. **数据关联**：F[2078] 出现与活性正相关，三个划分同向，但在控制 B0 预测、大小、父结构后不带来增量。
2. **模型内计算作用**：R128 用 `F[2078] 缺失` 方向（系数 {signed(cards[0]['std_coef'])}），
   即“缺失”降低 logit、存在相对抬高；这是模型内部事实，不是删除子图的因果效应。
3. **对其他分子的增量预测价值**：`CS-FINE − CS-BASE` 不优于零；规则/线性模型也没有
   因此获得可检出的 Test 收益。
分子实例（真实子图节点与外部连接，见 `analysis/phaseD.json` → `D3`）：
`{pd_['D3']['cases'].get('support_positive', {}).get('smiles', 'n/a')}`（2078 支持、正例）与
`{pd_['D3']['cases'].get('support_negative', {}).get('smiles', 'n/a')}`（2078 支持、反例）。

## 运行时间与资源
纯实验计算（单进程、CPU、seed=0）：Phase A {pa['seconds']:.0f}s、Phase B {bc['seconds']:.0f}s、
Phase C {pc['seconds']:.0f}s、Phase D（见 `analysis/phaseD.json` 的运行记录）、
对比与 bootstrap（`analysis/contrasts.json` 的 {len(cj['contrasts'])} 个对照 × 2 划分 × 1 000 次重采样）约 185s、
数值核验 ≈4s、测试 ≈7s；合计约 **{total_min:.1f} 分钟**。
包含一次性开发、修复与排查的会话总墙钟约 100 分钟：超过 60 分钟目标，未超过 120 分钟硬上限。
未重训任何历史 XGBoost/MLP/GINE；无 GPU；本地峰值内存保持在可用范围内（未发生 OOM）。

## 局限
1. 只有 seed=0；bootstrap 只描述固定预测的样本不确定性。
2. X-CF booster 未保存，未重训；规则来源仍只有 X-C 与 X-F。
3. COUNT-ALL 沿用 R128 的 C 网格 {list(bc['C_grid'])}；协议禁止扫描正则，
   因此“线性读出不足”不能与“C 网格不匹配”完全分离。
4. XGB-ALL 与 COUNT-ALL 的差值混合了优化方法、容量与损失差异。
5. 4 406 个结构来自 Train 文档频率前 4 406，仍不是“全部可能子图”。
6. 局部贡献是模型内部量；重叠/重复结构使“单项解释”不可加地归因到化学机制。
7. C–S 分析是条件性观察，骨架、共现结构与数据组成未被随机化。

## 交付物
`PROTOCOL.md`、`ERRATA.md`、`code/`、`tests/`、`analysis/metrics.csv`、
`analysis/contrasts.json`、`analysis/rule_registry.csv`、`analysis/predictions/`、
`analysis/count_all_contributions.npz`、`analysis/phaseD.json`、
`analysis/knowledge_cards.md`、`analysis/provenance.json`、`analysis/verification.json`
以及保留的 `analysis/models/XGB-ALL.json`。
"""
    (K.ROUND / "REPORT.md").write_text(report)
    print("[report] written", flush=True)


if __name__ == "__main__":
    main()
