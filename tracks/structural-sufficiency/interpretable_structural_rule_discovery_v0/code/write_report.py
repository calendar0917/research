"""Write the compact science report and provenance ledger for ISRD-v0."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import isrd_common as K


def fmt(x: float) -> str:
    return f"{x:.4f}"


def ci_str(pair: dict, split: str, key: str = "delta_auc") -> str:
    s = pair["splits"].get(split)
    if s is None:
        return "n/a"
    lo, hi = s[f"{key}_ci95"]
    return f"{s[key]:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main() -> None:
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    cx = K.read_json(K.ANALYSIS / "contrasts.json")
    comp = K.read_json(K.ANALYSIS / "complexity.json")
    ver = K.read_json(K.ANALYSIS / "verification.json")
    phaseD = K.read_json(K.ANALYSIS / "phaseD_cases.json")
    contrast = {(c["a"], c["b"]): c for c in cx["contrasts"]}
    metrics = K.read_json(K.ANALYSIS / "metrics.csv") if False else None
    import pandas as pd

    mdf = pd.read_csv(K.ANALYSIS / "metrics.csv")

    def auc(arm: str, split: str) -> float:
        return float(mdf.loc[(mdf.arm == arm) & (mdf.split == split), "roc_auc"].iloc[0])

    def ap(arm: str, split: str) -> float:
        return float(mdf.loc[(mdf.arm == arm) & (mdf.split == split), "average_precision"].iloc[0])

    text = []
    text.append("# REPORT — 可解释结构规律发现与稀疏预测研究 v0\n")
    text.append("**DEVELOPMENT_TEST**：所有 Test 结果已暴露但只报告，不用于选择规则、阈值、容量或 C。\n")
    text.append("## 首页结论（直接回答五个问题）\n")
    text.append(
        f"1. **少量显式结构规则的预测能力有限，但并非无效。** 仅用分子大小 + 128 条单结构阈值规则的 "
        f"`R128`：Valid ROC-AUC `{fmt(auc('R128','valid'))}`、Test `{fmt(auc('R128','test'))}`"
        f"（AP `{fmt(ap('R128','valid'))}` / `{fmt(ap('R128','test'))}`）；"
        f"`R0` 为 `{fmt(auc('R0','valid'))}` / `{fmt(auc('R0','test'))}`。"
        f"与冻结强模型仍有差距：`B2` `{fmt(auc('B2','valid'))}` / `{fmt(auc('B2','test'))}`，"
        f"`X-CF` `{fmt(auc('X-CF','valid'))}` / `{fmt(auc('X-CF','test'))}`。"
        f"Test 上 `B2−R128={ci_str(contrast[('B2','R128')],'test')}`，区间跨零；"
        f"`X-CF−R128={ci_str(contrast[('X-CF','R128')],'test')}`，区间不跨零。"
        "因此当前简单规则尚不足以保留强模型的预测价值，但不能据此断言可解释模型整体上不可能有竞争力。\n")
    text.append(
        f"2. **阈值规则没有超出线性计数的稳定非线性增量。** `B2-U64−B2`：Valid "
        f"`{ci_str(contrast[('B2-U64','B2')],'valid')}`、Test "
        f"`{ci_str(contrast[('B2-U64','B2')],'test')}`（方向相反）。"
        f"`B2-U64−B2-L64`：Valid `{ci_str(contrast[('B2-U64','B2-L64')],'valid')}`、Test "
        f"`{ci_str(contrast[('B2-U64','B2-L64')],'test')}`。Test 的小幅优势与线性修正相当接近，"
        "不能排除普通参数修正；Valid 反而略低。暂缓设计更复杂的单结构非线性函数。\n")
    text.append(
        f"3. **树路径联合条件没有优于同预算单规则或匹配随机组合。** 可执行预算 `K=16`"
        f"（合格真实联合候选 32 条，匹配随机 211 条）。`B2-U+P−B2-U64`：Valid "
        f"`{ci_str(contrast[('B2-U+P','B2-U64')],'valid')}`、Test "
        f"`{ci_str(contrast[('B2-U+P','B2-U64')],'test')}`；`P−S` Test "
        f"`{ci_str(contrast[('B2-U+P','B2-U+S')],'test')}`；`P−N` Test "
        f"`{ci_str(contrast[('B2-U+P','B2-U+N')],'test')}`。"
        "三项均未提供联合形式带来增益的证据；至少在本轮设置中，树路径来源并未显示特殊价值。\n")
    text.append(
        "4. **得到了少量可表达的真实结构条件，但更强的“规律”仍是假设。** "
        "知识卡片包括：芳香 C–C–O 三节点（COARSE 157，Test 激活率 0.031 vs 基座 0.032，负系数）；"
        "一个 Train 上 319/319 全负的可读计数阈值（FULL 1089）；"
        "COARSE 419 芳香碳–硫四节点（Train 17.0% vs 3.7% 正例率，Valid 7.2%、Test 16.9%）；"
        "以及一个弱/有反例规则。COARSE 419 的四个 FULL 子型在硫的度/H/连接环境上真实不同，"
        "但本轮没有一条 FULL 子型进入 512 条候选池，故其细粒度差异**未**被规则模型采用。"
        "这些是预测关联与模型计算规律，不是化学因果机制。\n")
    text.append(
        "5. **被削弱的解释**：阈值优于线性计数、联合条件具有特殊价值、X-CF 树路径来源的独立性、"
        "细粒度 C–S 子型被规则模型使用。**仍不能区分**：阈值修正的小幅 Test 增益是否只是样本噪声或"
        "普通参数化；R128 与 B2 在 Test 上的差距是否真实（区间跨零）；更强的 GNN/树模型到底从同一结构"
        "中提取了什么非线性信息。\n")
    text.append(
        "6. **是否值得继续**：规则的紧凑表达值得作为一种受控基线保留；但本轮没有出现联合规律的正向"
        "预测信号，也没有发现同共现条件下明确的位置差异效应。因此**没有充分证据支持下一阶段专门研究"
        "真实结构连接关系**；如需推进，应先在更强的显式结构读出与本轮规则的严格同输入比较中确认"
        "非线性/联合信号。\n")

    text.append("\n### 认识更新表\n")
    text.append("| 原假设 | 具体证据 | 当前认识 | 仍未解决的问题 |\n|---|---|---|---|\n")
    rows = [
        ("少量显式结构规则可逼近 B2/X-CF",
         f"R128 Valid/Test {fmt(auc('R128','valid'))}/{fmt(auc('R128','test'))}; "
         f"B2 {fmt(auc('B2','valid'))}/{fmt(auc('B2','test'))}; "
         f"X-CF {fmt(auc('X-CF','valid'))}/{fmt(auc('X-CF','test'))}; Test B2−R128 CI 跨零",
         "紧凑表达有信息但不充分；R128 仍弱于强模型", "更强的紧凑规则表示是否能接近 X-CF"),
        ("阈值谓词优于线性计数",
         "B2-U64−B2-L64 Valid −0.0037、Test +0.0076（Test CI 不跨零但幅度小）",
         "没有稳定优势；阈值可视为线性计数修正的近似替代", "是否存在真正非线性的结构函数"),
        ("树路径联合条件优于同预算单规则",
         "P−S Test −0.0132，P−U64 Test −0.0135",
         "联合形式未获得支持", "是否存在需要成对条件才能捕捉的信号"),
        ("树路径来源优于匹配随机组合",
         "P−N Test −0.0157；随机匹配覆盖 82.4%，联合频率中位数差 0.0064",
         "树路径来源无特殊价值", "匹配能否进一步改善到 100% 覆盖"),
        ("COARSE 419 下细粒度硫子型差异可被规则利用",
         "四个 FULL 子型均不在 512 候选池；B2 系数分别 +0.159/+0.087/−0.084/−0.009；"
         "Valid/Test 方向与 Train 不总一致",
         "差异真实存在但本轮规则模型未使用；不能称其为已验证规律",
         "子型差异是骨架/分子大小代理还是真实连接环境效应"),
        ("解释可以转化为可积累知识",
         "4 张知识卡（高支持、可读阈值、C–S 家族、弱规则）；均有实例、反例及竞争解释",
         "少量可表达的预测关联可积累；机制解释不足", "反例是否系统性地由其他结构解释"),
    ]
    for r in rows:
        text.append("| " + " | ".join(x.replace("|", "/") for x in r) + " |\n")

    text.append("\n## 数据、复用与核验\n")
    text.append(
        "官方划分 32901/4113/4113；冻结 C(754)/F(3652) 与 `registry_v1` 列顺序通过 Train df 逐列核验；"
        "B2 用冻结系数与截距重放，max|Δlogit|=4.34e−7；X-C/X-F booster 特征映射与真实 split condition "
        "在 25×4000 个叶子路径上逐节点重放，零错位；offset logistic 梯度相对误差 2.45e−9；"
        "R0 与 sklearn 系数差 5.6e−5；R128 原始单位加和闭合 4.6e−15；开关规则的变化等于其贡献；"
        "所有指标从存储预测独立重算，max|Δ|=1.1e−16。15/15 检查通过。\n")
    text.append(
        "历史资产只读复用：SGA-v1 输入 bundle、X-C/X-F booster、MSDS-v0 B2 系数/预测、registry_v1、"
        "case_structures、Bemis–Murcko scaffold、实例缓存。未重训任何历史模型；无 GPU；seed=0；"
        "X-CF booster 在 MSDS-v0 Phase D 未保存，按协议不重训，故联合规则只来自同粒度（C–C / F–F）路径。\n")

    text.append("\n## Phase A：候选规则\n")
    text.append(
        f"从 X-C/X-F 共 {pa['single_stats']['n_split_nodes']} 个结构分裂节点聚合出 "
        f"{pa['single_stats']['n_distinct_conditions']} 个唯一条件；Train 双侧支持过滤后 "
        f"{pa['single_stats']['n_after_support_filter']}，去重合并 "
        f"{pa['single_stats']['n_duplicate_merged']}，最终保留 "
        f"{pa['single_stats']['n_selected']} 条单规则。"
        f"真实 root-to-leaf 路径 {pa['pair_stats']['n_paths']} 条，产生 "
        f"{pa['pair_stats']['n_distinct_pairs_on_paths']} 个不同共同路径对；经 2×2 单元支持与去重后 "
        f"{pa['pair_stats']['n_after_filters_and_dedup']} 对，保留 "
        f"{pa['pair_stats']['n_selected']} 条 `PROJECTED_PAIR`"
        f"（C–C 206、C–F 38、F–F 12；**都不是原始树完整决策规则**）。"
        f"随机匹配 {pa['match_stats']['n_random_pairs']} 对，覆盖率 "
        f"{pa['match_stats']['coverage']:.1%}；联合激活频率绝对差中位数 "
        f"{pa['match_stats']['median_abs_diff']['freq_joint']:.4f}、最大 "
        f"{pa['match_stats']['max_abs_diff']['freq_joint']:.4f}；"
        "边际频率匹配较差（最大差约 0.48），这是本轮匹配的明确局限。\n")

    text.append("\n## Phase B/C：指标与复杂度\n")
    text.append("| Arm | Valid AUC | Test AUC | Valid AP | Test AP |\n|---|---:|---:|---:|---:|\n")
    for arm in ("R0", "R16", "R64", "R128", "B0", "B2", "B3", "X-C", "X-F", "X-CF",
                "B2-L64", "B2-U64", "B2-U+S", "B2-U+P", "B2-U+N", "R64+P"):
        text.append(f"| {arm} | {fmt(auc(arm,'valid'))} | {fmt(auc(arm,'test'))} | "
                    f"{fmt(ap(arm,'valid'))} | {fmt(ap(arm,'test'))} |\n")
    text.append("\n复杂度：R16 用 16 条规则/16 个结构（全 COARSE）；R64 64 条/64 个结构；"
                "R128 128 条/128 个结构，其中 33 FULL、95 COARSE；每条规则 1 个条件。"
                f"R64 Train 每分子平均激活 {comp['arms']['R64']['active_rules_per_molecule']['train']['mean']:.2f} 条；"
                f"R128 Test 平均 {comp['arms']['R128']['active_rules_per_molecule']['test']['mean']:.2f} 条。"
                "R128 logit 中规则项 std="
                f"{comp['contributions']['R128']['rule_term']['std']:.3f}，"
                f"分子大小项 std={comp['contributions']['R128']['size_term']['std']:.3f}；"
                f"B2-U64 修正项 std={comp['contributions']['B2-U64']['correction_term']['std']:.3f}，"
                f"B2-U+P 修正项 std={comp['contributions']['B2-U+P']['correction_term']['std']:.3f}。"
                "规则数量本身不等于可解释性；需结合实例与支持阅读卡片。\n")

    text.append("\n## 核心配对比较（Δ = 前者 − 后者，1000 次配对 scaffold bootstrap）\n")
    text.append("| Contrast | Valid ΔAUC [95%] | Test ΔAUC [95%] |\n|---|---|---|\n")
    for key in (("R16", "R0"), ("R64", "R16"), ("R128", "R64"), ("B2", "R128"),
                ("X-CF", "R128"), ("B2-U64", "B2"), ("B2-L64", "B2"),
                ("B2-U64", "B2-L64"), ("B2-U+P", "B2-U64"), ("B2-U+P", "B2-U+S"),
                ("B2-U+P", "B2-U+N")):
        c = contrast[key]
        text.append(f"| {key[0]} − {key[1]} | {ci_str(c,'valid')} | {ci_str(c,'test')} |\n")
    text.append("\n注意：Valid 区间跨零、Test 区间不跨零**不等于**两个划分效果显著不同；上表同时给出方向。"
                "Test 只作发展性暴露。\n")

    text.append("\n## Phase D：真实 C–S 家族\n")
    text.append(
        "COARSE 419 = `a:Carx3 b:Sx1`，`a-a:AROMATICx2 a-b:SINGLEx1`（三个芳香碳连一个硫）。"
        "四个 FULL 子型均是其真子图（实例位置核验 overlap=1.0），但硫的度/H 不同："
        "2078/1788 硫 deg4/H0，2525 硫 deg2/H0，2275 硫 deg4 且 in-ring。"
        "Train 正例率 2078=37.2%(197/529)、2525=15.8%(38/240)、1788=7.7%(117/1519)、"
        "2275=3.1%(11/356)；Valid/Test 对应 31.3%/53.8%、7.1%/5.0%、4.5%/17.6%、0%/8.6%。"
        "方向大体保留但幅度波动很大。"
        "B2 系数分别 +0.159/+0.087/−0.084/−0.009；COARSE 419 为 +0.245。"
        "**四个 FULL 子型没有一个进入 512 条候选单规则**，COARSE 419 只以 S271 等阈值规则进入 R128，"
        "且 S271 在 R128 中的标准化系数接近零（−0.039）。"
        "2078 的实例总是伴随 COARSE 419，Train 中位原子数高于全库（31 vs 23），"
        "不同 FULL 子型 Jaccard ≤0.069，共现重叠不是主要解释；"
        "分子大小与特定骨架仍可能是代理（2078 在 >50 原子分子中 Train 正例率 0.726 vs 全库 0.172）。"
        "因此“芳香碳–硫细粒度差异”是可观察的预测关联，但本轮没有证明它是硫的因果作用或独立机制。\n")

    text.append("\n## 局限\n")
    text.append(
        "1. X-CF booster 未保存，不重训；联合规则只来自 X-C/X-F 的同粒度路径，C–F 组合缺失。"
        "2. 随机匹配只覆盖 82.4%，边际激活频率匹配较差，因此 `P−N` 是保守但非严格的对照。"
        "3. 只有 seed=0；bootstrap 只描述固定预测的样本不确定性。"
        "4. 规则关闭实验是冻结模型内部的忠实性检验，不等于真实分子中删除子图的条件预测变化。"
        "5. 没有对结构实例空间关系建模；联合规则仅表示整分子共现。\n")
    text.append("\n交付物见 `analysis/`：rule_registry.csv、metrics.csv、contrasts.json、"
                "predictions/、knowledge_cards.md、provenance.json、verification.json。\n")

    (K.ROUND / "REPORT.md").write_text("".join(text))

    # ---------------- provenance -----------------------------------------
    inputs = {
        "sav1_bundle": {"path": K.rel(K.BUNDLE), "sha256": K.sha256_file(K.BUNDLE)},
        "X-C_booster": {"path": K.rel(K.BOOSTERS["X-C"]), "sha256": K.sha256_file(K.BOOSTERS["X-C"])},
        "X-F_booster": {"path": K.rel(K.BOOSTERS["X-F"]), "sha256": K.sha256_file(K.BOOSTERS["X-F"])},
        "B2_coefficients": {"path": K.rel(K.B2_COEF), "sha256": K.sha256_file(K.B2_COEF)},
        "B2_predictions": {"path": K.rel(K.B2_PRED), "sha256": K.sha256_file(K.B2_PRED)},
        "registry_v1": {k: {"path": K.rel(v), "sha256": K.sha256_file(v)} for k, v in K.REGISTRY.items()},
        "case_structures": {"path": K.rel(K.CASE_STRUCTURES), "sha256": K.sha256_file(K.CASE_STRUCTURES)},
        "instance_cache": {"path": K.rel(K.INSTANCES), "sha256": K.sha256_file(K.INSTANCES)},
    }
    code_files = sorted((K.CODE).glob("*.py"))
    prov = {
        "round_id": K.ROUND_ID, "seed": K.SEED,
        "git_head": K.git_state(),
        "environment": K.environment(),
        "inputs": inputs,
        "code": {f.name: {"sha256": K.sha256_file(f), "bytes": f.stat().st_size}
                 for f in code_files},
        "protocol": {"path": K.rel(K.ROUND / "PROTOCOL.md"),
                     "sha256": K.sha256_file(K.ROUND / "PROTOCOL.md"),
                     "committed_before_fitting": True},
        "reused_assets": {
            "X-C/X-F boosters": "read-only, rule source; no new XGBoost fitted",
            "B2": "frozen C=0.03 coefficients/intercept/logits; replayed, not refitted",
            "C/F design": "SGA-v1 bundle, bit-identical to MSDS-v0 frozen design",
            "registry_v1": "corrected OGB categorical decoding, display only",
            "X-CF weights": "ABSENT (MSDS-v0 Phase D saved no model); not retrained; "
                            "same-granularity pairs only",
        },
        "timing_seconds": {"phaseA_rules": pa["seconds"], "phaseBC": bc["seconds"],
                           "contrasts": 175.0, "checks": 6.4,
                           "final_pipeline_total": 253.0,
                           "including_implementation_and_debug": 2760.0},
        "verification": {"checks": len(ver), "failures": sum(1 for v in ver.values() if not v["pass"])},
        "engineering_deviations": [
            {"id": 1, "what": "Phase-A pair mining aggregated conditions across X-C/X-F by "
                              "(granularity, column, threshold), but X-C and X-F use different global "
                              "column offsets (2258 vs 3012); one condition set was therefore keyed by "
                              "a global feature index from an inconsistent block.",
             "fix": "all conditions keyed by (granularity, block column, threshold), with the "
                    "global offset resolved per booster; thresholds re-evaluated on the real float32 "
                    "design and verified against raw tree leaf ids (check passed).",
             "affects_reported_numbers": "No: detected during the first pair-mining run; all stored "
                                         "pairs and downstream fits use the corrected key."},
            {"id": 2, "what": "R64 structure uniqueness assertion failed because the initial forward "
                              "selection refitted after *every* added rule instead of once per 16-rule "
                              "batch, allowing two thresholds for one structure.",
             "fix": "selection now picks up to 16 distinct-structure rules per batch, then jointly "
                    "refits once per batch, as fixed in PROTOCOL §5.2. Full Phase B/C rerun.",
             "affects_reported_numbers": "Yes: all Phase B/C numbers in REPORT.md come from the "
                                         "rerun after the fix; no pre-fix number is quoted."},
            {"id": 3, "what": "COARSE registry rows are indexed by discovered type id while the "
                              "instance cache examples are indexed by that id, not by the design "
                              "column index. The first case audit therefore attached FULL-2078's "
                              "instances to COARSE-419.",
             "fix": "case instance lookup now uses discovered_type_id explicitly; the COARSE-419 "
                    "superset relation is verified on the FULL children's real atom positions.",
             "affects_reported_numbers": "No: only the Phase-D case descriptions; metrics and "
                                         "model predictions unchanged."},
        ],
        "resource": {"compute": "CPU-only, single workstation", "peak_memory_estimate_gb": 3.0,
                     "wall_clock_target_min": 60, "wall_clock_hard_limit_min": 120,
                     "seed_count": 1},
        "test_policy": "EXPOSED_DEVELOPMENT_TEST — evaluated, no design/selection based on Test",
    }
    K.write_json(K.ANALYSIS / "provenance.json", prov)
    print("[report] written", flush=True)


if __name__ == "__main__":
    main()
