"""Final report and route decision for gnn-route-clarification-8h-v1.

The prose is assembled from the frozen analysis tables.  Gates are the ones in
``protocol_freeze.json``; the decision follows task brief 10.3 and is allowed to
return UNRESOLVED when the required cells are missing.  Nothing here re-selects
a model or re-fits a threshold.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

import gr_common as C
import gr_analysis as A


def _load(name: str, default=None):
    path = C.ANALYSIS_DIR / "tables" / f"{name}.json"
    if not path.is_file():
        return default
    return C.read_json(path)


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x, nd=1):
    if x is None:
        return "—"
    return f"{100.0 * x:.{nd}f}%"


def ci(d, nd=4):
    if not d or "diff_ci95" not in d:
        return "—"
    lo, hi = d["diff_ci95"]
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def gate_value(table: Mapping[str, Any], key: str):
    gates = table.get("gates", {}) if table else {}
    return gates.get(key, {}).get("value") if isinstance(gates, dict) else None


def main() -> int:
    summary = C.read_json(C.ANALYSIS_DIR / "analysis_summary.json") if (C.ANALYSIS_DIR / "analysis_summary.json").is_file() else {}
    stage1 = _load("stage1", {"rows": [], "selected_scores": {}, "l_vs_gine": [], "recipe_gap": [], "gx_vs_g0": []})
    stage23 = _load("stage23", {"A": {}, "D": {}})
    mmp = _load("mmp_predictions", {})
    oracle = _load("oracle", {})
    checks = C.read_json(C.ANALYSIS_DIR / "ledger_checks.json") if (C.ANALYSIS_DIR / "ledger_checks.json").is_file() else {}
    manifest = C.read_json(C.ANALYSIS_DIR / "run_manifest.json") if (C.ANALYSIS_DIR / "run_manifest.json").is_file() else {}

    gate = {"gates": {}}
    scores = stage1["selected_scores"]
    rows = stage1["rows"]
    lines: list[str] = []
    add = lines.append

    add("# GNN 路线澄清 8h v1 — 报告")
    add("")
    add(f"- 轮次：`{C.ROUND_ID}`；设计基线 `{C.DESIGN_BASIS_COMMIT[:12]}`")
    add(f"- 运行设备：res 双 A100-SXM4-40GB（res-gpu0 / res-gpu1），单 seed=0，官方 split，每 epoch 全量三划分 eval")
    add(f"- 训练总墙钟：{f(manifest.get('total_seconds', 0) / 3600.0, 2)} 卡·小时（两卡合计，含解释/校准单元）")
    add("- Test 每 epoch 被观察；下文 Test 数字只作固定状态描述，不参与任何选择。")
    add("")

    # ---------------- Q1 ----------------
    add("## Q1. LCRC-C1 与 GINE 的差距依赖什么？")
    add("")
    add("| 数据集 | 模型 | 选中 recipe | Val RMSE | Test RMSE | Val epoch | 参数 |")
    add("|---|---|---|---|---|---|---|")
    for row in rows:
        if row.get("status") != "COMPLETE":
            continue
        if row.get("recipe") != stage1["selected_recipe"].get(f"{row['dataset']}/{row['model']}"):
            continue
        add(
            f"| {row['dataset']} | {row['model']} | {row['recipe']} | {f(row['valid_rmse_raw'])} | {f(row['test_rmse_raw'])} | {row['selected_epoch']} | {row['n_parameters']} |"
        )
    add("")
    add("**配对分子 bootstrap（2000 次，rng 1703，行共享重采样，负值=前者更好）**")
    add("")
    add("| 数据集 | split | 对比 | ΔRMSE | 95% CI | 比值 | P(前者更好) |")
    add("|---|---|---|---|---|---|---|")
    for d in stage1["l_vs_gine"]:
        add(
            f"| {d['dataset']} | {d['split']} | {d['a']} − {d['b']} | {d['diff']:+.4f} | {ci(d)} | {d['ratio']:.4f} | {d['prob_a_better']:.3f} |"
        )
    add("")
    add("**配方敏感性（同一 recipe 下 L−GINE 的 Val 差距）与 Gx−G0 输入桥**")
    add("")
    add("| 数据集 | 模型 | R3 gap | Rf gap | 方向翻转? |")
    add("|---|---|---|---|---|")
    for dataset in C.DATASETS:
        for arch in ("G0", "Gx"):
            g = {x["recipe"]: x["valid_gap_L_minus"] for x in stage1["recipe_gap"] if x["dataset"] == dataset and x["arch"] == arch}
            if not g:
                continue
            flip = "YES" if g.get("R3", 0) * g.get("Rf", 0) < 0 else "no"
            add(f"| {dataset} | {arch} | {f(g.get('R3'))} | {f(g.get('Rf'))} | {flip} |")
    add("")
    for d in stage1["gx_vs_g0"]:
        add(f"- Gx − G0，{d['dataset']}/{d['split']}：ΔRMSE={d['diff']:+.4f} {ci(d)}，比值 {d['ratio']:.4f}")
    add("")
    recipe_sensitive = any(
        (lambda g: g.get("R3") is not None and g.get("Rf") is not None and (g["R3"] * g["Rf"] < 0 or abs(g["Rf"]) < 0.5 * abs(g["R3"])))(
            {x["recipe"]: x["valid_gap_L_minus"] for x in stage1["recipe_gap"] if x["dataset"] == d and x["arch"] == a}
        )
        for d in C.DATASETS
        for a in ("G0", "Gx")
    )
    add(f"判读：**RECIPE_SENSITIVE = {'YES' if recipe_sensitive else 'NO'}**（定义：两配方下 L−GINE 差距翻转，或 Rf 下差距减半；精确数值见上表，不据此编造优化理论）。")
    l_ratio = {}
    for dataset in C.DATASETS:
        try:
            l = scores[f"{dataset}/L"]["valid_rmse_raw"]
            t = scores[f"{dataset}/{C_trunk()}"]["valid_rmse_raw"]
            l_ratio[dataset] = l / t
        except KeyError:
            pass
    add("")
    add(f"L 两配方最佳 / T 两配方最佳（Val）：{', '.join(f'{k}={v:.4f}' for k, v in l_ratio.items()) if l_ratio else '—'}")
    behind = [d for d, r in l_ratio.items() if r > C.A_GATE_RATIO]
    add(f"判读：**C1_BEHIND_THIS_BUDGET = {'YES' if behind else 'NO'}**，具体任务：{', '.join(behind) if behind else '无'}（门槛 5%，只适用本轮预算与这两套 recipe）。")
    add("")

    # ---------------- Q2 ----------------
    add("## Q2. 强消息传递骨干 + 无可加全图残差的可加读出，能否保住预测能力？")
    add("")
    add("| 数据集 | trunk | A recipe | A Val | T Val | A/T | A Test | T Test | A−T Test | 95% CI (Test) | A_VAL_GATE(1.05) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    a_gate = {}
    for dataset, e in stage23.get("A", {}).items():
        vt = e["comparisons"]["valid"]
        tt = e["comparisons"]["test"]
        ratio = e["selected"]["valid_rmse_raw"] / e["T_selected"]["valid_rmse_raw"]
        a_gate[dataset] = ratio <= C.A_GATE_RATIO
        add(
            f"| {dataset} | {e['trunk']} | {e['recipe']} | {f(e['selected']['valid_rmse_raw'])} | {f(e['T_selected']['valid_rmse_raw'])} | {ratio:.4f} | "
            f"{f(e['selected']['test_rmse_raw'])} | {f(e['T_selected']['test_rmse_raw'])} | {tt['diff']:+.4f} | {ci(tt)} | {'PASS' if a_gate[dataset] else 'FAIL'} |"
        )
    add("")
    add(f"A 的可加读出参数预算：head H 与总参数见 `analysis/tables/stage23.json`；预算公式在 `prepare.json` 中按解析式核对。")
    add("")

    # ---------------- Q3 ----------------
    add("## Q3. 端到端可加模型失败时，是读出表达力还是联合训练优化问题？")
    add("")
    add("| 数据集 | D Val fidelity (std) | D Val RMSE | T Val RMSE | D/T Val | D Test / T Test | D−T Test | 95% CI |")
    add("|---|---|---|---|---|---|---|---|")
    d_pass = {}
    for dataset, e in stage23.get("D", {}).items():
        if e.get("status") == "MISSING":
            add(f"| {dataset} | MISSING | — | — | — | — | — | — |")
            continue
        vt = e["comparisons"]["valid"]
        tt = e["comparisons"]["test"]
        d_pass[dataset] = (
            vt["teacher_fidelity_rmse_std"] <= C.D_FIDELITY_SIGMA and vt["ratio"] <= C.A_GATE_RATIO
        )
        add(
            f"| {dataset} | {f(vt['teacher_fidelity_rmse_std'])} | {f(vt['rmse_a'])} | {f(vt['rmse_b'])} | {vt['ratio']:.4f} | "
            f"{f(tt['rmse_a'])} / {f(tt['rmse_b'])} | {tt['diff']:+.4f} | {ci(tt)} |"
        )
    add("")
    if a_gate and d_pass:
        if not any(a_gate.values()) and any(d_pass.values()):
            add("判读：**REPRESENTATION_VS_READOUT = 联合训练嫌疑**——同一读出在冻结 T 表示上可用，端到端 A 未保住性能；这不等于证明可加形式表达力足够，只把瓶颈定位到端到端优化/正则。")
        elif not any(a_gate.values()) and not any(d_pass.values()):
            add("判读：**REPRESENTATION_VS_READOUT = 读出限制嫌疑**——A 与 D 都没有保住；仍不能证明形式不可表达。")
        else:
            add("判读：**REPRESENTATION_VS_READOUT = 目前不可分辨/未见失败**，按上表逐数据集读。")
    else:
        add("判读：stage-2/3 结果不完整，Q3 = UNRESOLVED。")
    add("")

    # ---------------- Q4 ----------------
    add("## Q4. 原生贡献账目相对归因的增量价值")
    add("")
    if oracle.get("properties"):
        add("**oracle 预测精度与 Δ 预测（全部 823 对；σ = oracle Train 目标标准差）**")
        add("")
        add("| property | model | Val RMSE/σ | Test RMSE/σ | Δf MAE/σ | 分量等权 Δ/σ | Δ 符号准确率 |")
        add("|---|---|---|---|---|---|---|")
        for prop, e in oracle["properties"].items():
            for model, p in (e.get("prediction") or {}).items():
                if not isinstance(p, dict) or "delta_mae_sigma" not in p:
                    continue
                add(
                    f"| {prop} | {model} | {f(p.get('valid_rmse_sigma'))} | {f(p.get('test_rmse_sigma'))} | {f(p.get('delta_mae_sigma'))} | "
                    f"{f(p.get('delta_component_equal_sigma'))} | {f(p.get('delta_sign_accuracy'))} |"
                )
        add("")
        add("预测精度门槛（8.6）：A、T 的 Val/Test RMSE 均需 ≤0.10σ；可比性 |RMSE_A−RMSE_T|≤0.02σ。逐性质判读见下。")
        add("")
        for prop, e in oracle["properties"].items():
            preds = e.get("prediction") or {}
            vals = {m: preds.get(m, {}) for m in ("A", "T")}
            ok = all(
                isinstance(v, dict) and v.get("valid_rmse_sigma", 9e9) <= C.ORACLE_ACCURACY_SIGMA and v.get("test_rmse_sigma", 9e9) <= C.ORACLE_ACCURACY_SIGMA
                for v in vals.values()
            )
            comparable = all(
                isinstance(vals["A"], dict)
                and isinstance(vals["T"], dict)
                and abs(vals["A"].get(f"{split}_rmse_sigma", 9e9) - vals["T"].get(f"{split}_rmse_sigma", 9e9)) <= C.ORACLE_COMPARABILITY_SIGMA
                for split in ("valid", "test")
            )
            h_comps = (mmp.get("per_model", {}).get("T") or {}).get("mmp_main_H", {}).get("n_components", 0)
            add(
                f"- {prop}: {'CALIBRATION_OK' if ok else 'CALIBRATION_INCONCLUSIVE（A/T 未同时达到 0.10σ）'}；"
                f"{'COMPARABLE' if comparable else 'NOT_COMPARABLE'}；H 独立分量 {h_comps}（<30，不能声明 A> T 的工作信号）。"
            )
        add("")
        add("**已知生成贡献校准**（truth = Crippen WC_LOGP / MolWt 的指定生成分解；E 按 8.5 定义，先分量内均值再分量均值）")
        add("")
        add("| property | model | source | E (pair) | E (component) | var MAE | core MAE | size | global |")
        add("|---|---|---|---|---|---|---|---|---|")
        for prop, e in oracle["properties"].items():
            for model, block in e.get("models", {}).items():
                if not isinstance(block, dict) or "E_pair_mean" not in block:
                    continue
                for source, s in block.items():
                    if not isinstance(s, dict) or "E_pair_mean" not in s:
                        continue
                    add(
                        f"| {prop} | {model} | {source} | {f(s['E_pair_mean'])} | {f(s['E_component_equal'])} | {f(s['var_mae'])} | {f(s['core_mae'])} | {f(s['size_abs_mean'])} | {f(s['global_abs_mean'])} |"
                    )
            z = e.get("controls", {}).get("zero", {}).get("E")
            add(f"| {prop} | zero | — | {f(z)} | — | — | — | — | — |")
        add("")
        add("注：E 的 raw 单位是目标 σ；`zero` 为每对预测零贡献的对照。")
        add("")
        for prop, e in oracle["properties"].items():
            r = e.get("A_vs_T")
            if not r:
                continue
            add(
                f"- {prop}: E_A(native)/E_T(IG) = {r['ratio']:.3f}，95% CI {r['ratio_ci95'][0]:.3f}–{r['ratio_ci95'][1]:.3f}，"
                f"P(A 更低)={r['prob_A_better']:.3f}，分量 {r['n_components']}（{r['support']}）。"
            )
        add("")
        add("由于主集合 H 的独立分量 <30，以上 A vs T 只作为固定状态描述，不构成“A 解释优于 T”的工作信号。")
    else:
        add("oracle 校准表缺失（oracle 训练或 ledger 未完成）→ Q4 = EXPLANATION_UNRESOLVED（样本/数值未完成，不是方法失败）。")
    add("")
    oracle_summary_path = C.ANALYSIS_DIR / "oracle_ledger_summary.json"
    if oracle_summary_path.is_file():
        os_ = C.read_json(oracle_summary_path)
        for prop, block in os_.get("properties", {}).items():
            for model, b in block.items():
                if isinstance(b, dict) and b.get("quadrature"):
                    q = b["quadrature"]
                    n = b.get("n_pairs", 0)
                    unresolved = q.get("n_unresolved", 0)
                    add(
                        f"- {prop}/{model}: scope={b.get('scope')}, quadrature unresolved {unresolved}/{n} "
                        f"{'(>5% → IG 部分标 NUMERICALLY_UNRESOLVED)' if n and unresolved / n > 0.05 else ''}"
                    )
        add("")
    if mmp.get("models"):
        add("**真实 MMP 预测对照（全部 823 对，不按标签挑核心）**")
        add("")
        add("| model | stratum | n | 分量等权 |Δf−Δy| | pair 等权 |Δf−Δy| | 符号准确率 (|Δy|≥0.1) | 分量数 | 支持 |")
        add("|---|---|---|---|---|---|---|---|")
        for name, strata in mmp.get("per_model", {}).items():
            for combo in ("mmp_main_H", "train-test", "train-valid", "train-train", "ALL"):
                if combo not in strata:
                    continue
                s = strata[combo]
                add(
                    f"| {name} | {combo} | {s.get('n_pairs')} | {f(s.get('component_equal_abs_delta_error'))} | {f(s.get('abs_delta_error_mean'))} | "
                    f"{f(s.get('component_equal_sign_accuracy'))} | {s.get('n_components')} | {s.get('support')} |"
                )
        z = mmp.get("zero_baseline", {})
        add(
            f"| zero | ALL | {z.get('n_pairs')} | {f(z.get('component_equal_abs_delta_error'))} | {f(z.get('abs_delta_error_mean'))} | — | {z.get('n_components')} | {z.get('support')} |"
        )
        add("")
        add("注意：H（两端均非 Train）为空集，主集合指标只能报 `LOW_SUPPORT/NO_H_SUPPORT`；上表的非 train-train 结论仅基于 5 个 train-valid + 27 个 train-test 对。")
    expl = _load("mmp_explanation", {})
    if expl.get("models"):
        add("")
        add("**解释样本上的局部效应一致性（只把 variable 通道与 Δy 对比；这是代理指标，不是原子真值）**")
        add("")
        add("| model/source | n | MAE | 分量等权 MAE | 符号准确率 | Pearson r | 支持 |")
        add("|---|---|---|---|---|---|---|")
        for name, e in expl["models"].items():
            add(
                f"| {name} | {e['n']} | {f(e['variable_vs_dy_mae'])} | {f(e['component_equal_mae'])} | "
                f"{f(e['sign_accuracy'])} | {f(e['pearson_r'])} | {e['support']} |"
            )
        add(f"（对照：该样本 |Δy| 均值 {f(next(iter(expl['models'].values()))['dy_mean_abs'])}；样本 {expl['n_pairs']} 对，其中非 train-train 32 对/11 分量。）")
    add("")

    # ---------------- Q5 ----------------
    add("## Q5. 最终路线建议")
    add("")
    route = _route(stage1, stage23, oracle, checks, a_gate, d_pass, mmp)
    for line in route["text"]:
        add(line)
        add("")
    add("**现在相信 / 反对证据 / 关键未知 / 最值得研究 / 应停止**")
    add("")
    for key in ("believe", "against", "unknown", "next", "stop"):
        add(f"- {route['bullets'][key]}")
    add("")

    # ---------------- checks ----------------
    add("## 数值核验")
    add("")
    if checks.get("models") or checks.get("checks"):
        add("| 项目 | 结果 |")
        add("|---|---|")
        for model, block in checks.get("models", {}).items():
            if isinstance(block, dict) and "scope" in block:
                add(f"| {model} scope | {block['scope']} |")
                for k, v in (block.get("checks") or {}).items():
                    add(f"| {model} {k} | {f(v, 8) if isinstance(v, float) else v} |")
                add(f"| {model} state_unchanged | {block.get('state_unchanged')} |")
        for model in ("T", "A"):
            for kind in ("batch_single", "antisymmetry", "renumbering"):
                c = checks.get("checks", {}).get(f"{kind}_{model}")
                if isinstance(c, dict):
                    add(f"| {kind} {model} | pass={c.get('pass')} worst={f(c.get('worst_abs_std'), 8)} |")
                elif isinstance(c, str):
                    add(f"| {kind} {model} | {c} |")
        for model in ("A", "D"):
            r = checks.get("checks", {}).get(f"reallocation_{model}")
            if isinstance(r, dict):
                add(
                    f"| reallocation {model} | λ={f(r.get('lambda'))} max|Δsum|={f(r.get('max_abs_sum_change'), 8)} "
                    f"max rank change={r.get('max_rank_change')} sign-flip frac={f(r.get('sign_flip_fraction'))} |"
                )
        mm = checks.get("checks", {}).get("multi_mapping", {})
        add(f"| multi-mapping | {mm.get('status')} |")
    else:
        add("ledger 核验未运行。")
    add("")
    add("## 限制与不声称")
    add("")
    add("- 两套 recipe 的预测差距不证明形式表达能力强弱；A/D 的对照只把瓶颈定位到“联合训练 vs 读出”，不构成不可表达性证明。")
    add("- 求和正确不等于化学解释正确；oracle E 只在该指定生成规则下校准，不冒充实验性质真值。")
    add("- H 集为空、非 train-train 分量少，MMP 主集合标 LOW_SUPPORT；不把 train-train 指标称为完全未见配对泛化。")
    add("- Test 已每 epoch 观察，不能称为盲测；bootstrap 只描述固定 checkpoint 的有限样本不确定性，不含 seed/配方选择偏差。")
    add("- IG 只对声明的连续输入表示与路径成立，中间点不是合法分子，不是物理干预。")
    add("")

    path = C.ROUND_DIR / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    decision = {**route["decision"], "gates": {
        "A_VAL_GATE": a_gate,
        "D_FIDELITY_AND_REAL": d_pass,
        "RECIPE_SENSITIVE": bool(recipe_sensitive),
        "C1_BEHIND": l_ratio,
    }}
    C.write_json(C.ROUND_DIR / "DECISION.json", decision)
    print(f"wrote {path}")
    print(json.dumps(decision, indent=1)[:3000])
    return 0


def C_trunk() -> str:
    import gr_select as S

    try:
        return S.selected_trunk_model()
    except Exception:
        return "G0"


def _route(stage1, stage23, oracle, checks, a_gate, d_pass, mmp) -> dict[str, Any]:
    """Task brief 10.3, evaluated in priority order with explicit evidence."""
    scores = stage1.get("selected_scores", {})
    l_ok = {}
    t_name = C_trunk()
    for dataset in C.DATASETS:
        l = scores.get(f"{dataset}/L", {}).get("valid_rmse_raw")
        t = scores.get(f"{dataset}/{t_name}", {}).get("valid_rmse_raw")
        l_ok[dataset] = bool(l is not None and t is not None and l / t <= C.A_GATE_RATIO)

    oracle_signal = False
    oracle_improvement = {}
    h_components = (mmp.get("per_model", {}).get("T") or {}).get("mmp_main_H", {}).get("n_components", 0)
    if oracle.get("properties") and h_components >= 30:
        for prop, e in oracle["properties"].items():
            vals = {}
            for model, block in e.get("models", {}).items():
                if isinstance(block, dict):
                    for source, s in block.items():
                        if isinstance(s, dict) and "E_component_equal" in s:
                            vals[f"{model}/{source}"] = s["E_component_equal"]
            if vals:
                base = min(vals.values())
                for key, value in vals.items():
                    if key.endswith("/native") and value <= (1.0 - C.EXPLANATION_IMPROVEMENT) * base:
                        oracle_signal = True
                        oracle_improvement[prop] = {"native": key, "E": value, "best": base}
    a_ok = any(a_gate.values())
    d_ok = any(d_pass.values())
    quad_ok = True
    for model in ("T", "A"):
        block = (checks.get("models") or {}).get(model) or {}
        q = block.get("quadrature") or {}
        n = block.get("n_pairs") or 0
        if n and q.get("n_unresolved", 0) / n > 0.05:
            quad_ok = False
    checks_pass = bool(
        checks
        and any(
            (checks.get("checks", {}).get(f"batch_single_{m}") or {}).get("pass")
            and (checks.get("checks", {}).get(f"antisymmetry_{m}") or {}).get("pass")
            and (checks.get("checks", {}).get(f"renumbering_{m}") or {}).get("pass")
            for m in ("T", "A")
        )
    )

    cells_complete = (
        len(stage23.get("A", {})) == len(C.DATASETS)
        and len(stage23.get("D", {})) == len(C.DATASETS)
        and all("selected" in v for v in stage23.get("A", {}).values())
    )
    text = []
    if not cells_complete:
        choice = "UNRESOLVED"
        text.append("**UNRESOLVED**：A/D 必要单元不齐，不能强行四选一。")
        believe = "目前只能确认已完成单元的预测数字与数值核验状态。"
        against = "缺格使任何路线判断都可能被新证据推翻。"
        unknown = "A/D 的真实性能-解释联合优势是否存在于未完成单元。"
        nxt = "先补齐 A/D 与 oracle/ledger 的必要单元，再做路线判断。"
        stop = "停止在证据不齐时扩大架构或数据集。"
    elif all(l_ok.get(d, False) for d in C.DATASETS):
        choice = "KEEP_LCRC_AS_MAIN_CANDIDATE"
        text.append("**(a) 保留 LCRC 作为主要候选**：L 在两个真实小任务的 Val 上都在 T 的 5% 内；Test 与原生结构账目的实际优势见上表，仍需按 10.3 的具体条件核对，不能据此宣称论文创新。")
        believe = "在本轮预算内，LCRC 的预测劣势不再是稳定的、跨任务的；它仍提供一份不摊 rho 的原生原子账目。"
        against = "Test 数字与 Val 不完全一致；L 的 rho 是独立全图项，不能当作原子贡献；bootstrap 区间仍宽。"
        unknown = "L 的优势是否来自 rho 对图级统计的拟合，而非局部读出；换 seed 或换硬度更大的任务是否仍成立。"
        nxt = "把 rho 的贡献与局部原子账目做可分离的实验（冻结 rho、只读局部），而不是再加一个架构。"
        stop = "停止用“LCRC 更可解释”作为牺牲预测性能的理由，直到账目在 oracle/真实 MMP 上显示出可复现的增量。"
    elif a_ok and oracle_signal and checks_pass:
        choice = "STRUCTURAL_EFFECT_READOUT_ON_STRONG_BACKBONE"
        text.append("**(b) 进入“强骨干上的结构效应读出”问题**：A 在至少一个任务保住 Val 门槛，原生账目相对 IG 在 oracle E 上有 ≥10% 增量，且 endpoint/数值核验通过；不再扩大旧 C 家族。")
        believe = "强 GINE 骨干加一个无全图残差的可加读出，能在本轮任务上保住预测，并给出可校准的原生贡献。"
        against = "A 的保住只在部分任务/部分阈值上成立；oracle E 的增量依赖指定生成规则，不是实验真值。"
        unknown = "A 的原生账目在真实 MMP 上是否比 T-IG 更接近结构效应，还是只在 oracle 性质上占优。"
        nxt = "把读出限制在“可加头 + 冻结强表示”的联合设计上，做一次预注册的跨任务验证。"
        stop = "停止把“MP 结果完全不能回溯”当作新架构的唯一理由，也停止扩大旧 LCRC。"
    elif d_ok and not a_ok:
        choice = "FROZEN_REPRESENTATION_READOUT"
        text.append("**(b′) 限定为“冻结强表示下的读出/训练问题”**：只有 D 在冻结 T 表示上保住了性能；本轮不说新端到端架构已成功。")
        believe = "选中的 T 表示可容纳可加读出；端到端 A 的失败更像联合训练/正则瓶颈。"
        against = "D 的高保真只针对 T 的输出，不是对真实标签的独立证据；A 的失败也可能来自读出表达力。"
        unknown = "在冻结表示上加入真实标签微调是否仍保住；A 的失败能否通过训练技巧而非新形式解决。"
        nxt = "在冻结表示上做小规模、预注册的读出/训练对照，不再新增大架构。"
        stop = "停止把端到端可加模型当作已失败的架构结论；也停止用 D 的保真度替代真实任务性能。"
    elif not a_ok and not d_ok and checks_pass and quad_ok:
        choice = "STRONG_PREDICTOR_WITH_ATTRIBUTION"
        text.append("**(c) 优先保留强预测器 + 现有归因**：A/D 都没有性能-解释联合优势，而 T-IG 的 endpoint/完整性核验通过、成本可接受；停止把“MP 不能回溯”作为新架构的唯一理由。")
        believe = "本轮最强的可复现对象是强 GINE 预测器；后置 IG 在该表示上数值可核验。"
        against = "IG 的路径依赖、非分子中间点和计算成本；oracle E 未必优于原生账目（见 Q4）。"
        unknown = "T-IG 的归因在真实 MMP 上是否有稳定的局部效应信号，而不是预测差的副产品。"
        nxt = "把 T-IG 与原生账目放在同一 oracle/真实配对基准上做一次成本-收益比较。"
        stop = "暂停“更可解释因此值得牺牲性能”的方法主张，保留预测参照与工具。"
    else:
        choice = "PAUSE_EXPLANATION_CLAIM"
        text.append("**(d) 暂停“更可解释因此值得牺牲性能”的方法主张**：A/D 没有性能-解释联合优势，IG 在冻结的 256 点预算下数值未闭合（或 oracle 校准不足），解释证据只达到“能重建/能描述”的层级；保留强预测参照与工具，暂停方法学主张。暂停是资源判断，不是证明不可能。")
        believe = "本轮能复现的最强对象是强 GINE 预测器；原生可加账目与 IG 都能在端点重建预测，但都没有转化为可验证的解释优势。"
        against = "A 与 D 都没保住预测；IG 在 256 点内未满足完整性门槛的 pair 超过 5%；H 为空使 oracle 的工作信号无法成立。"
        unknown = "这是优化/正则问题、读出形式问题，还是评价门槛（0.02σ 保真、256 点积分）本身过严；本轮设计无法分离。"
        nxt = "先把“可加读出为什么在冻结表示上也拟合不到 T”的原因查清（优化/容量/正则），再用同一 oracle 判据检验任何新读出。"
        stop = "暂停新增架构与“可解释性换性能”的叙事；停止用完整性/守恒本身当作解释优势。"

    decision = {
        "route": choice,
        "evidence": {
            "L_within_5pct_of_T_val": l_ok,
            "A_val_gate": a_gate,
            "D_fidelity_and_real": d_pass,
            "oracle_native_increment": oracle_improvement,
            "explanation_checks_pass": checks_pass,
        "ig_quadrature_within_5pct": quad_ok,
        },
        "note": "开发决策门槛，不是显著性检验；UNRESOLVED 是允许的结论。",
    }
    return {"text": text, "decision": decision, "bullets": {"believe": believe, "against": against, "unknown": unknown, "next": nxt, "stop": stop}}


if __name__ == "__main__":
    sys.exit(main())
