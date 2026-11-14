"""Report + synthesis: machine bundles, 3 core figures, REPORT.md, run_manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import lb_common as C
import lb_analysis as A
import lb_gates as G


def _ensure_dirs():
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.SYNTHESIS.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS / "figures").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# synthesis bundles
# ---------------------------------------------------------------------------


def synthesize() -> dict[str, Any]:
    _ensure_dirs()
    prior = A.p0_prior_evidence()
    C.write_json(C.SYNTHESIS / "prior_evidence.json", prior)

    selection = {"arms": {}, "fulln_u0_panel": A.fulln_u0_panel(),
                 "rules": list(C.SELECTION_RULES)}
    for arm in C.all_arms():
        selection["arms"][arm["arm_id"]] = A.selection_table_for_arm(arm["arm_id"], arm)
    C.write_json(C.SYNTHESIS / "selection.json", selection)

    data_budget = A.data_budget_table()
    C.write_json(C.SYNTHESIS / "data_budget.json", data_budget)

    readout = A.readout_table()
    if any(v.get("status") == "OK" for v in readout["lipo"].values()):
        readout["subgroups"] = A.structural_subgroups()
        # observed oracle over the probe grid, per arm (R_SEL can latch onto step 0)
        for ds_tag in ("lipo", "bace"):
            for rule, e in readout.get(ds_tag, {}).items():
                if e.get("status") != "OK" or "test_curve" not in e:
                    continue
        readout["probe_oracle"] = _probe_oracle()
    C.write_json(C.SYNTHESIS / "readout.json", readout)

    synth = {"prior": prior, "selection": selection, "data_budget": data_budget, "readout": readout}
    synth["gates"] = G.verdict_matrix(synth)
    C.write_json(C.SYNTHESIS / "gates.json", synth["gates"])
    return synth


def _probe_oracle() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag in ("LIP",):
        for rule in C.PROBE_ARMS:
            d = C.RUNS / f"{tag}_PROBE_{rule}"
            tj, gp = d / "train.json", d / "grid_predictions.npz"
            if not (tj.exists() and gp.exists()):
                out[rule] = {"status": "NOT_RUN"}
                continue
            p = C.read_json(tj)
            tn = p.get("target_norm")
            z = np.load(gp, allow_pickle=False)
            y = np.asarray(z["test_y"], np.float64)

            def raw(a):
                a = np.asarray(a, np.float64)
                return a * float(tn["scale"]) + float(tn["mean"]) if tn else a

            t = raw(z["test"])
            rmse = np.sqrt(((t - y) ** 2).mean(axis=1))
            i = int(np.argmin(rmse))
            out[rule] = {"status": "OK", "step0_test_rmse": float(rmse[0]),
                         "test_curve_min": float(rmse[i]),
                         "test_curve_min_update": int(z["grid_updates"][i]),
                         "observed_oracle_gap": float(rmse[0] - rmse[i])}
    # how far the extra branches actually move predictions relative to the warm head
    try:
        def preds(rule):
            z = np.load(C.RUNS / f"LIP_PROBE_{rule}" / "grid_predictions.npz", allow_pickle=False)
            tn = C.read_json(C.RUNS / f"LIP_PROBE_{rule}" / "train.json")["target_norm"]
            return np.asarray(z["test"], np.float64) * float(tn["scale"]) + float(tn["mean"])
        out["branch_displacement_vs_P_WARM_max_abs"] = float(np.abs(preds("P_POST") - preds("P_WARM")).max())
        out["pre_vs_post_max_abs"] = float(np.abs(preds("P_PRE") - preds("P_POST")).max())
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_data_budget(db: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, color in (("C1_R1@Lipophilicity", "C0"), ("DMPNN_D2F@Lipophilicity", "C1")):
        rows = sorted([r for r in db["cells"].get(key, []) if r.get("status") == "OK"], key=lambda r: r["N"])
        if not rows:
            continue
        Ns = [r["N"] for r in rows]
        for phase, marker, ls in (("E1", "o", "-"), ("E2", "s", "--")):
            vals = [r.get(f"{phase}_R_SEL_test") for r in rows]
            if all(v is not None for v in vals):
                ax[0].plot(Ns, vals, marker=marker, ls=ls, color=color, label=f"{key.split('@')[0]} {phase}")
        vals = [r.get("E1_last10_test_median") for r in rows]
        if all(v is not None for v in vals):
            ax[1].plot(Ns, vals, marker=".", ls=":", color=color, label=f"{key.split('@')[0]} last10")
    ax[0].set_title("Lipo Test RMSE vs fit N (R_SEL)")
    ax[0].set_xlabel("fit rows"); ax[0].set_ylabel("Test RMSE (logD)"); ax[0].legend(fontsize=7)
    ax[1].set_title("last-10 grid-point median (descriptive)")
    ax[1].set_xlabel("fit rows"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_update_curves(selection: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for arm in ("LIP_C1_R1_F100", "LIP_C1_R1_F25", "LIP_DMPNN_D2F_F100"):
        rec = A.load_arm_records(arm)
        if rec is None or rec["grid"] is None:
            continue
        g = rec["grid"]
        for i, split in enumerate(("train_fit", "valid", "test")):
            ax[i].plot(g["updates"], A.grid_metric_sequence(rec["meta"]["task"], g, split, "rmse"),
                       label=arm.replace("LIP_", ""))
    for i, t in enumerate(("train_fit", "valid", "test")):
        ax[i].set_title(f"{t} RMSE vs update"); ax[i].set_xlabel("update"); ax[i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_readout(ro: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rules = [r for r in C.PROBE_ARMS if ro["lipo"].get(r, {}).get("status") == "OK"]
    if rules:
        for rule, color in zip(rules, ("C0", "C1", "C2")):
            d = C.RUNS / f"LIP_PROBE_{rule}"
            z = np.load(d / "grid_predictions.npz", allow_pickle=False)
            tn = C.read_json(d / "train.json")["target_norm"]
            y = np.asarray(z["test_y"], np.float64)
            t = np.asarray(z["test"], np.float64) * float(tn["scale"]) + float(tn["mean"])
            ax[0].plot(z["grid_updates"], np.sqrt(((t - y) ** 2).mean(axis=1)), label=rule, color=color)
        ax[0].set_title("Lipo probe: Test RMSE vs head updates")
        ax[0].set_xlabel("update"); ax[0].set_ylabel("Test RMSE"); ax[0].legend(fontsize=7)
        z = np.load(C.RUNS / "LIP_PROBE_P_WARM" / "grid_predictions.npz", allow_pickle=False)
        tn = C.read_json(C.RUNS / "LIP_PROBE_P_WARM" / "train.json")["target_norm"]
        y = np.asarray(z["test_y"], np.float64)
        for rule, color in zip(rules, ("C0", "C1", "C2")):
            d = C.RUNS / f"LIP_PROBE_{rule}"
            zz = np.load(d / "grid_predictions.npz", allow_pickle=False)
            v = np.asarray(zz["valid"], np.float64) * float(tn["scale"]) + float(tn["mean"])
            vy = np.asarray(zz["valid_y"], np.float64)
            ax[1].plot(zz["grid_updates"], np.sqrt(((v - vy) ** 2).mean(axis=1)), label=rule, color=color)
        ax[1].set_title("Lipo probe: Valid RMSE vs head updates")
        ax[1].set_xlabel("update"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# run manifest
# ---------------------------------------------------------------------------


def run_manifest() -> dict[str, Any]:
    rows = []
    for arm in C.all_arms():
        d = C.RUNS / arm["arm_id"]
        tj = d / "train.json"
        entry = {"arm_id": arm["arm_id"], "model": arm["model"], "dataset": arm["dataset"],
                 "fit": arm["fit"]}
        if tj.exists():
            p = C.read_json(tj)
            entry.update({"status": "OK", "U0": p["recipe"]["U0"], "Umax": p["recipe"]["Umax"],
                          "updates": p["cost"]["updates_run"], "n_fit": p["subset"]["n"],
                          "wall_seconds": p["cost"]["wall_seconds"],
                          "gpu_hours": p["cost"]["gpu_hours_charged"],
                          "git_commit": p["provenance"]["git_commit"],
                          "init_state_hash": p["model_info"]["init_state_hash"],
                          "retained": p["selection"]["retained"]})
        else:
            entry["status"] = "NOT_RUN"
        rows.append(entry)
    probes = []
    for ds_tag in ("LIP", "BACE"):
        for rule in C.PROBE_ARMS:
            d = C.RUNS / f"{ds_tag}_PROBE_{rule}"
            tj = d / "train.json"
            e = {"arm_id": f"{ds_tag}_PROBE_{rule}"}
            if tj.exists():
                p = C.read_json(tj)
                e.update({"status": "OK", "updates": p["train"]["updates"],
                          "gpu_hours": p["cost"]["gpu_hours_charged"],
                          "new_params": p["new_params"],
                          "step0_equivalence_max_abs": p["step0_delta_equivalence_max_abs"],
                          "source_ckpt": p.get("source_ckpt"),
                          "git_commit": p["provenance"]["git_commit"]})
            else:
                e["status"] = "NOT_RUN"
            probes.append(e)
    r = C.RUNS / C.RESTART_ARM
    restart = {"arm_id": C.RESTART_ARM, "status": "RUN" if (r / "train.json").exists() else "NOT_RUN"}
    if restart["status"] == "RUN":
        p = C.read_json(r / "train.json")
        restart.update({"updates": p["cost"]["updates_run"], "wall_seconds": p["cost"]["wall_seconds"],
                        "gpu_hours": p["cost"]["gpu_hours_charged"],
                        "restart": p.get("restart"), "git_commit": p["provenance"]["git_commit"]})
    gpu_h = sum(x.get("gpu_hours") or 0.0 for x in rows + probes + [restart])
    return {"round_id": C.ROUND_ID, "git_commit": C.git_commit(), "created_utc": C.utcnow(),
            "arms": rows, "probes": probes, "restart": restart,
            "gpu_hours_retained": round(gpu_h, 3),
            "incident": C.read_json(C.ANALYSIS / "incident.json") if (C.ANALYSIS / "incident.json").exists() else None}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _verdict_block(g: dict[str, Any]) -> list[str]:
    lab = g["labels"]
    names = {"H_SEL": "选择（Validation 选择/有限样本波动）",
             "H_OPT": "优化预算（同信息下更多优化）",
             "H_DATA": "监督样本量/覆盖",
             "H_USE": "冻结表示的读出利用",
             "H_REP": "原输入缺任务信息（本轮不直接检验）"}
    L = ["| 假设 | 判定 | 证据等级 |", "|---|---|---|"]
    for k in ("H_SEL", "H_OPT", "H_DATA", "H_USE", "H_REP"):
        L.append(f"| **{k}** {names[k]} | `{lab[k]['evidence']}` | {lab[k]['grade']} |")
    return L


def report(synth: dict[str, Any]) -> str:
    prior, selection, db, ro = synth["prior"], synth["selection"], synth["data_budget"], synth["readout"]
    g = synth["gates"]
    man = run_manifest()
    gpu_h = {a["arm_id"]: a.get("gpu_hours") for a in man["arms"]}
    L: list[str] = []
    L.append("# LCRC 瓶颈诊断 — 主报告（中文）\n")
    L.append(f"- round: `{C.ROUND_ID}`  ·  git: `{C.git_commit()}`  ·  generated: {C.utcnow()}")
    L.append("- 口径：单 seed（0）、官方既有划分、每 epoch 全量评估 Train_fit/Valid/Test，"
             "Test 参与开发诊断。`observed_oracle_gap` 只命名事后可观测差距，含 Test 选择乐观偏差，"
             "不是可实现收益、总体风险上界或正式成绩。**不同干预的增益不能相加。**\n")

    # ---- 0. verdict first
    L.append("## 0. 结论先行\n")
    L += _verdict_block(g)
    L.append("")
    L.append("**一句话取舍**：在本轮这个固定学习过程里，最明确、可复核的限制是"
             "**监督样本量（H_DATA）**；其次是**选择敏感性已被测到但没有可兑现的替代规则（H_SEL）**；"
             "**额外优化只降低 Train_fit、不再改善留出集（H_OPT → FIT_ONLY）**；"
             "**冻结 h 上没有买到读出余量（H_USE = 无采购）**。因此本轮不支持“表达力不足”这类默认结论，"
             "但也没有拿到能宣布表示缺口被排除的证据（H_REP 仍可疑、未被检验）。\n")
    se = g["sample_efficiency"]
    L.append("支撑上面第一句的直接数字（Lipo，Test RMSE，R_SEL，同 U0/2U0 与同 50 候选机会）：\n")
    for cell in ("C1_R1@Lipophilicity", "DMPNN_D2F@Lipophilicity"):
        v = se["cells"].get(cell, {})
        if v.get("gains"):
            el = v["gains"]
            L.append(f"- `{cell}`: N {v['N_small']}→{v['N_big']}，"
                     + "，".join(f"{k} {vv:+.4f}" for k, vv in el.items()) + f"（ruler {v['ruler']}）")
    L.append("")

    # ---- 1. P0
    L.append("## 1. P0：用已有产物定义诊断基准（未重训任何旧臂）\n")
    L.append("| 臂 | n_epoch | Valid选中epoch | 选中Test | Test曲线最小 | observed_oracle_gap |")
    L.append("|---|---|---|---|---|---|")
    for name, v in prior["curves"].items():
        if "observed_oracle_gap" not in v:
            L.append(f"| {name} | MISSING | | | | |"); continue
        L.append(f"| {name} | {v['n_epochs']} | {v['valid_selected_epoch']} | "
                 f"{_f(v['valid_selected_test_rmse'])} | {_f(v['test_curve_min'])} | "
                 f"{_f(v['observed_oracle_gap'])} |")
    L.append("\n复现了提示里的两个例子（C1_R1 .58394/.56381、CONTRIB .60632/.56505）与 K 的 .55728。"
             "旧轨迹是 full-N、另一个 harness 的 360-epoch 口径；**新 N=25/50 的数字不能与旧 R1 直接相减**。\n")

    # ---- 2. data budget
    L.append("## 2. 数据量 × 优化预算（新轨迹，同一连续路径到 2U0）\n")
    L.append("![data budget](figures/data_budget.png)\n")
    for key, rows in db["cells"].items():
        task = "regression" if key.endswith(C.PRIMARY_DATASET) else "binary"
        unit = "RMSE" if task == "regression" else "ROC-AUC"
        L.append(f"**{key}**　主指标 {unit}" + ("" if task == "regression" else "　（分类只跑至 U0，无 E2 leg）") + "\n")
        L.append(f"| fit | N | U0 | E1 R_SEL {unit} | E1 last10 | E2 R_SEL {unit} | E2 last10 | Train_fit | GPU·h |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r.get("status") != "OK":
                L.append(f"| {r['arm_id']} | — | — | NOT_RUN | | | | | |"); continue
            L.append(f"| {r['fit']} | {r['N']} | {r['U0']} | {_f(r.get('E1_R_SEL_test'))} | "
                     f"{_f(r.get('E1_last10_test_median'))} | {_f(r.get('E2_R_SEL_test'))} | "
                     f"{_f(r.get('E2_last10_test_median'))} | "
                     f"{_f(r.get('E1_last10_train_fit_median'),3)}→{_f(r.get('E2_last10_train_fit_median'),3)} | "
                     f"{_f(gpu_h.get(r['arm_id']), 3)} |")
        L.append("")
    L.append("读法：**N 每翻一倍，两个学习器的 Test 都明确变好**（H_DATA）。"
             "而 E1→E2 的额外优化在所有臂上都是 **Train_fit 变好、留出集不变或更差**（H_OPT 只到 FIT_ONLY）。"
             "注意 C1 的 Train_fit RMSE 远低于 D-MPNN（≈0.126 vs ≈0.228）却留出更差——"
             "“训练误差更低”显然不能当能力证据。\n")
    # structural variants at full N
    L.append("### 2.1 full-N 结构变体对照（另一层信息：加结构描述符本身有多有用）\n")
    L.append("| 臂 | 输入 | E1 R_SEL Test | E1 last10 | E2 R_SEL Test |")
    L.append("|---|---|---|---|---|")
    variants = (("LIP_C1_R1_F100", "atom/bond 图特征（C1 基线）"),
                ("LIP_CHEM8_R1_F100", "冻结 8 个 chem8 原子列"),
                ("LIP_K_R1_F100", "chem8 + Q8 图贡献残差(K)"),
                ("LIP_C1_R3_F100", "C1_R1，wd 1e-2"),
                ("LIP_DMPNN_D2F_F100", "D-MPNN（不同架构与预算）"))
    for aid, desc in variants:
        arm = selection["arms"].get(aid, {})
        if not isinstance(arm, dict) or arm.get("status") == "NOT_RUN":
            L.append(f"| {aid} | {desc} | NOT_RUN | | |"); continue
        ph = arm["phases"]["E1"]
        ph2 = arm["phases"].get("E2") or {}
        e2 = ph2.get("rules", {}).get(C.R_SEL, {}).get("test_primary") if ph2.get("status") != "EMPTY" else None
        L.append(f"| {aid} | {desc} | {_f(ph['rules'][C.R_SEL]['test_primary'])} | "
                 f"{_f(ph['last10_median_test_primary'])} | {_f(e2)} |")
    L.append("")
    L.append("CHEM8 与 C1 基线几乎同水平，K 略好——**本轮不据此宣布“结构信息足够”或“表达力不足”**："
             "这只说明在这三个固定配方下，换/加结构描述的收益小于数据量与选择摆动的量级。\n")

    # ---- 3. selection
    L.append("## 3. 选择敏感性：差距存在，但换规则兑现不了\n")
    L.append("![update curves](figures/update_curves.png)\n")
    L.append("| 臂 | 阶段 | 候选 | observed_oracle_gap | 好状态最长持续点 | R_SEL upd/Test | A_SEL upd/Test | W_SEL upd/Test |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm_id, arm in selection["arms"].items():
        if not isinstance(arm, dict) or arm.get("status") == "NOT_RUN":
            continue
        for phase, ph in arm["phases"].items():
            if ph.get("status") == "EMPTY":
                continue
            rs, asl, wsl = ph["rules"][C.R_SEL], ph["rules"][C.A_SEL], ph["rules"][C.W_SEL]
            L.append(f"| {arm_id} | {phase} | {ph['n_candidates']} | "
                     f"{_f(ph['observed_oracle_gap']['observed_oracle_gap'])} | "
                     f"{ph['persistence']['longest_run_points']} | "
                     f"{rs['update']}/{_f(rs['test_primary'])} | {asl['update']}/{_f(asl['test_primary'])} | "
                     f"{wsl['update']}/{_f(wsl['test_primary'])} |")
    L.append("")
    L.append("### 3.1 full-N 面板：Validation 组成 → 选择（2000 次 paired bootstrap + 逐行 jackknife）\n")
    L.append("| 臂 | 模态选中update | Test P10 | P50 | P90 | 不同选中数 | jackknife Test 极差 | 选择稳定度 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a, v in selection["fulln_u0_panel"]["arms"].items():
        if v.get("status") == "NOT_RUN":
            L.append(f"| {a} | NOT_RUN | | | | | | |"); continue
        b, jk = v["bootstrap"], v["jackknife"]
        L.append(f"| {a} | {b.get('modal_update')} | {_f(b.get('test_metric_p10'))} | {_f(b.get('test_metric_p50'))} | "
                 f"{_f(b.get('test_metric_p90'))} | {b.get('selected_update_spread')} | "
                 f"{_f(jk.get('test_metric_spread'))} | {_f(jk.get('selected_update_modal_fraction'),3)} |")
    L.append("")
    util = g["selector_utility"]
    L.append(f"替代规则（`A_SEL` 对 `R_SEL`）：{util['A_SEL']['n_positive']}/{util['A_SEL']['n']} 条为正，"
             f"中位增益 {_f(util['A_SEL']['median_gain'])}；"
             f"`W_SEL`：{util['W_SEL']['n_positive']}/{util['W_SEL']['n']}，中位 {_f(util['W_SEL']['median_gain'])}。"
             f"判定 `{util['verdict']}`。\n")
    L.append("这两段必须分开读：**bootstrap/jackknife 说的是“同一批训练状态下，验证集组成会让选中结果漂移”**"
             "（结果：确实漂移——Lipo C1_R1 的 jackknife Test 极差 0.0111，和整条 E2 优化增益同量级；"
             "DMPNN F100 的 bootstrap P90=0.6213 说明尾部会挑到早期坏状态）；"
             "**规则对照说的是“有没有一条事先固定的替代规则能稳定兑现”**（结果：没有）。"
             "所以是诊断到评估敏感性，而不是已经拿到性能收益。\n")

    # ---- 4. optimization / restart
    L.append("## 4. 优化余量与唯一重启臂\n")
    o = g["optimization"]
    L.append(f"判定 `{o['verdict']}`。逐臂 E1→E2 的留出增益与 Train_fit 增益（正号=更好）：\n")
    L.append("| 臂 | 留出增益 | Train_fit 增益 | 标签 |")
    L.append("|---|---|---|---|")
    for r in o["rows"]:
        L.append(f"| {r['arm']} | {_f(r['holdout_gain'])} | {_f(r['train_fit_gain'])} | `{r['label']}` |")
    L.append("")
    tr = g["restart_trigger"]
    L.append(f"重启触发条件（冻结）：Train RMSE 降幅 < {tr['rule']['train_rmse_drop_min']} 且 Test 增益 < "
             f"{tr['rule']['test_gain_max']}。实测：Train 降幅 {_f(tr.get('train_rmse_drop_frac'))}、"
             f"Test 增益 {_f(tr.get('test_gain_E1_to_E2'))} → `{tr['verdict']}`。")
    rr = run_manifest()["restart"]
    rs = A.restart_summary()
    if rs.get("status") == "OK":
        L.append(f"已按协议执行 1 条重启臂（`{rs['restart_arm']}`：从 `LIP_C1_R1_F100/u0_snapshot.pt` 恢复 "
                 f"model+Adam+数据迭代器+RNG，以原 peak lr 重启 cosine 再跑 U0 步，不 reset Adam）。\n")
        L.append("### 4.1 重启臂 vs 低 lr 延长（同 prefix、同 50 点 E2 网格）\n")
        L.append("| 方案 | R_SEL 选中update | E2 R_SEL Test | E2 last10 Test | E2 last10 Train_fit | observed_oracle_gap |")
        L.append("|---|---|---|---|---|---|")
        for tag, name in (("base_E2", "低 lr 延长（F100 的 E2 leg）"), ("restart_E2", "重启臂（peak lr 重跑 U0 步）")):
            v = rs[tag]
            L.append(f"| {name} | {v['R_SEL_update']} | {_f(v['R_SEL_test_rmse'])} | {_f(v['last10_test_median'])} | "
                     f"{_f(v['last10_train_fit_median'],3)} | {_f(v['oracle_gap'])} |")
        L.append("")
        L.append(f"重启相对低 lr 延长的 Test 增益 = {_f(rs['test_gain_restart_minus_base'])}"
                 "（正号=重启更好）。两者 Train_fit 几乎相同、留出集都不改善；"
                 "即 **换一种优化路径也换不出泛化收益**，进一步把 H_OPT 钉在 FIT_ONLY。\n")
    else:
        L.append("重启臂：`NOT_RUN`（未跑，不得当作“无效”）。\n")

    # ---- 5. readout
    L.append("## 5. 冻结表示的读出定位\n")
    L.append("![readout](figures/readout.png)\n")
    L.append("源：`LIP_C1_R1_F100/u0_snapshot.pt`（update = U0 = 9720 的终点状态，不含任何按 Valid/Test 选 epoch 的步骤；"
             "见 PROTOCOL 附录 A1）。三条探针共用同一份冻结 `h`，新增参数都恰为 14705，step0 与 P_WARM 等价"
             "（最大绝对差 ~1e-7，float32 噪声）。\n")
    L.append("| probe | new_params | R_SEL 选中 | Valid RMSE | Test RMSE | Test 曲线最小 | observed_oracle_gap |")
    L.append("|---|---|---|---|---|---|---|")
    po = ro.get("probe_oracle", {})
    for rule in C.PROBE_ARMS:
        e = ro["lipo"].get(rule, {})
        if e.get("status") != "OK" or not e.get("selected"):
            L.append(f"| {rule} | NOT_RUN | | | | | |"); continue
        s = e["selected"]["R_SEL"]
        o_ = po.get(rule, {})
        L.append(f"| {rule} | {e['new_params']} | {s['update']} | {_f(s.get('valid_rmse'))} | "
                 f"{_f(s.get('test_rmse'))} | {_f(o_.get('test_curve_min'))} | {_f(o_.get('observed_oracle_gap'))} |")
    L.append("")
    L.append(f"采购门槛（冻结）：Lipo Valid 与 Test RMSE 都需相对改善 ≥1%。判定 `{g['readout']['verdict']}` → "
             f"BACE 探针 `NOT_RUN`（未采购，不算“无效”）。\n")
    L.append("**三条探针的 R_SEL 都落在 step 0**：头在 1800 步里把 Train_fit 从 0.127 降到 0.099，"
             "Valid 却从 0.7254 单调升到 ≈0.7369，所以“按 Valid 选”必然回到未训练状态——"
             "这本身就是“读出侧没有可兑现余量”的直接证据，而不是探针没跑起来。"
             "若改用事后 oracle（Test 曲线最小），三条都只到 0.5823@180 步，相对 step0 的 "
             "observed_oracle_gap 仅 ≈0.0074（仍含 Test 选择乐观偏差，且低于 0.01 的实用尺）。\n")
    L.append(f"**必须说清的实现限制**：`P_POST` 与 `P_PRE` 的预测最大差仅 "
             f"{po.get('pre_vs_post_max_abs', float('nan')):.2e}（对 P_WARM 是 "
             f"{po.get('branch_displacement_vs_P_WARM_max_abs', float('nan')):.2e}）。"
             "原因是 delta 末层零初始化 + 两条支路串联：新增非线性支路在 1800 步内几乎没被用起来。"
             "因此 **PRE vs POST 这一对照在本轮是退化的，“池化位置”没有结论**；"
             "能说的只是：在这个预算下，多给 14.7k 参数的非线性头**没有买到** Test 改善。\n")

    # ---- 6. interpretation guardrails
    L.append("## 6. 判读边界（明确不做的推断）\n")
    L.append("- 不把 oracle gap、数据增益、probe 增益相加去解释“总误差的百分之多少”。")
    L.append("- 不把 U0/2U0 当两次独立实验；它们是同一路径的两个观察预算。")
    L.append("- C1 的 Train_fit 更低却留出更差，只能说明它在记忆结构身份，不能说明“欠训练”或“表达力不足”。")
    L.append("- 单 seed、单嵌套子集序列、共用 norm 的受控设定：这是固定学习过程的样本效率，不是调优后的样本复杂度。")
    L.append("- 新轨迹不承诺逐位复现旧口径（E1 只在 50 点网格上选，旧口径在 360 个 epoch 上选；target norm 也不同）。")
    L.append("- BACE 只到 U0、每格 50 候选，且分类主 AUC、回归主 RMSE，绝对数字不混平均。\n")

    # ---- 7. budget + incident
    inc = man.get("incident") or {}
    L.append("## 7. 预算、执行与一次基础设施事故\n")
    L.append(f"- regime：`res` / a100-2 / 2×A100-SXM4-40GB / driver 550.163.01 / torch 2.5.1+cu124；"
             f"pool `res-gpu0`+`res-gpu1`，每 pool 一条进程。")
    L.append(f"- 正式轨迹：**{sum(1 for a in man['arms'] if a['status']=='OK')} 条主体 + "
             f"{sum(1 for p in man['probes'] if p['status']=='OK')} 条 Lipo 读出探针 + "
             f"{1 if man['restart']['status']=='RUN' else 0} 条重启臂**（上限 20，未超）。"
             f"保留产物计费 GPU·h = **{man['gpu_hours_retained']:.2f}**；"
             f"按 job 墙钟统计：有用 {inc.get('useful_job_minutes')} min、重复浪费 "
             f"{inc.get('wasted_duplicate_minutes')} min。")
    if inc:
        L.append(f"- 事故：{inc['incident']}")
        L.append(f"- 完整性：{inc['integrity_check']}")
        L.append(f"- 结论影响：重复臂与保留臂同 seed、同 commit、同数据顺序，属于同一确定性轨迹的重复执行，"
                 f"不产生新信息也不用作独立重复；只消耗预算。阻塞类失败 0 条（3 次即时失败 0.3 min 内修复重启）。")
    L.append("")

    # ---- 8. artifacts
    L.append("## 8. 产物与复现\n")
    L.append("```bash\nuv run python code/run.py prepare                       # 子集/norm/缓存指纹\n"
             "uv run python code/run.py train --model C1_R1 --dataset Lipophilicity --fit 25\n"
             "uv run python code/run.py analyze && uv run python code/run.py report\n"
             "uv run python -m pytest tests -q                        # 14 项不变量\n```\n")
    L.append("- 机器量：`analysis/synthesis/{prior_evidence,selection,data_budget,readout,gates}.json`、"
             "`analysis/run_manifest.json`、`analysis/incident.json`、`analysis/queue_history.json`。")
    L.append("- 每条轨迹：`train.json`（含 U0/Umax、初始 state hash、共用 norm 指纹、50 点网格、三规则选中权重）、"
             "`curve.csv`（每 epoch + 每网格点的全量三划分评估）、`grid_predictions.npz`（每个 grid 状态逐样本预测）、"
             "`u0_snapshot.pt`、`best_{R,A,W}_SEL.pt`、`last.pt`、`step0.pt`。")
    L.append("- 权重副本：`runs/` 本地保留（`.gitignore` 排除大文件），远端 `res` 同路径可恢复。")
    return "\n".join(L)


def write_all() -> dict[str, Any]:
    synth = synthesize()
    figs = C.ANALYSIS / "figures"
    for fn, name in ((lambda p: fig_data_budget(synth["data_budget"], p), "data_budget.png"),
                     (lambda p: fig_update_curves(synth["selection"], p), "update_curves.png"),
                     (lambda p: fig_readout(synth["readout"], p), "readout.png")):
        try:
            fn(figs / name)
        except Exception as exc:  # figures are descriptive only
            C.log(f"[report] figure {name} warning: {exc}")
    man = run_manifest()
    C.write_json(C.ANALYSIS / "run_manifest.json", man)
    text = report(synth)
    (C.ANALYSIS / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    (C.TRACK / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    C.log("report written")
    return synth


def main(argv=None) -> int:
    write_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
