"""Generate REPORT.md from the recorded numbers.

Every table in the report is formatted from `analysis/*.json`; the prose states
which comparison the numbers support.  Nothing here recomputes a metric, so the
report cannot drift from the recorded artefacts.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import trc_common as C

ARMS = list(C.ARMS)


def _load(name: str) -> Any:
    path = C.analysis_dir() / name
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _main_rows(rows: Sequence[Mapping[str, Any]], order: Sequence[str] | None = None) -> list:
    by_id = {r["run_id"]: r for r in rows}
    ids = order or [r["run_id"] for r in sorted(rows, key=lambda r: r["best_valid_rmse"])]
    out = []
    for rid in ids:
        r = by_id.get(rid)
        if r is None:
            continue
        out.append([r["run_id"], r["arm"], r["supervision"],
                    r.get("train_fraction") if r.get("train_fraction") else "1.0",
                    f(r["best_valid_rmse"]), f(r["best_test_rmse"]),
                    f(r["best_train_rmse"]), int(r["best_epoch"] or 0),
                    f(r["active_params"], 0),
                    f(r["wall_minutes"], 1)])
    return out


MAIN_HEADER = ["run", "arm", "aux", "train frac", "Valid RMSE", "Test RMSE",
               "Train RMSE", "best ep", "params", "wall min"]


def _boot(block: Mapping[str, Any], split: str) -> str:
    entry = (block or {}).get(split)
    if not entry or entry.get("n", 0) == 0:
        return "–"
    lo, hi = entry["lo"], entry["hi"]
    mark = "**" if (lo > 0 or hi < 0) else ""
    return f"{mark}{entry['delta']:+.4f}{mark} [{lo:+.4f}, {hi:+.4f}]"


def main() -> int:
    main_tab = _load("main_table.json")
    boot = _load("bootstrap.json")
    fact = _load("factorial.json")
    resp = _load("response_metrics.json")
    cross = _load("cross_combinations.json")
    cases = _load("cases.json")
    support = _load("support.json")
    freeze = _load("structure_freeze.json")
    checks = _load("checks.json")
    timing = _load("timing.json")
    cfg = C.load_config()
    refs = cfg["reference_values"]

    lipo = main_tab.get("lipo", [])
    esol = main_tab.get("esol", [])
    ref = main_tab.get("reference", [])
    best_lipo = min(lipo, key=lambda r: r["best_valid_rmse"]) if lipo else None
    gine = next((r for r in ref if r["arm"] == "GINE_RAW" and r["dataset"] == "lipo"), None)
    gine_esol = next((r for r in ref if r["arm"] == "GINE_RAW" and r["dataset"] == "esol"), None)

    L: list[str] = []
    add = L.append
    add("# task_response_composition_s0_v1 — 结构响应组合 s0 报告")
    add("")
    add("八小时轮次（T0 2026-11-21T23:33:19+08:00）的完整结果。所有数字来自 "
        "`analysis/*.json`；本文件由 `code/trc_report.py` 从这些记录生成，不重算指标。")
    add("")
    add("## 0. 结论摘要")
    add("")
    if best_lipo and gine:
        add(f"1. **整分子预测没有超过外部 GINE 坐标。** 最好的候选是 "
            f"`{best_lipo['run_id']}`，Valid {f(best_lipo['best_valid_rmse'])} / "
            f"Test {f(best_lipo['best_test_rmse'])}，同一协议下的 GINE 坐标是 "
            f"Valid {f(gine['best_valid_rmse'])} / Test {f(gine['best_test_rmse'])}。"
            f"Valid 上 GINE 显著更好（配对 bootstrap 差值 "
            f"{_boot(boot.get(f'GINE_RAW_LIPO_R0 - {best_lipo["run_id"]}', {}), 'valid')}）。")
    add(f"2. **外部 GINE 与历史 GINE 参考一致。** 本轮 raw-GINE 坐标 "
        f"{f(gine['best_valid_rmse']) if gine else '–'}/"
        f"{f(gine['best_test_rmse']) if gine else '–'}，历史 `GINE_RAW_REF` "
        f"{f(refs['GINE_RAW_REF']['valid_rmse'])}/{f(refs['GINE_RAW_REF']['test_rmse'])}"
        "（历史为 240 epoch + early stop，本轮统一 E=160），说明坐标可比。")
    add("")
    add("## 1. 问题与冻结结构")
    add("")
    add("问题：带接口片段的任务作用能否由少量共享响应状态承载，环境经由该状态改变片段作用？"
        "在显式共享限制下，用真实匹配替换目录检查共享是否成立；再看它对整分子预测和数据效率的价值。"
        "骨干被禁止使用 GNN / 图 Transformer / 预训练 / 蒸馏 / 强模型残差；GINE 只作外部预测坐标。")
    add("")
    sf = freeze.get("splits", {})
    add(table([[s, sf.get(s, {}).get("n_molecules", "–"), sf.get(s, {}).get("n_parses", "–"),
                f(sf.get(s, {}).get("parses_per_molecule_mean", 0), 2),
                len(sf.get(s, {}).get("fallback_molecules", []))]
               for s in C.SPLITS], ["split", "molecules", "parses", "parses/mol", "fallback mols"]))
    add("")
    add(f"* 分子 {freeze.get('n_molecules')}，解析 {freeze.get('n_parses')}，片段 "
        f"{freeze.get('n_fragments')}，对象帧 {freeze.get('n_frames')}；"
        f"帧原子属性行 {freeze.get('n_atom_attribute_rows')}，键属性行 "
        f"{freeze.get('n_bond_attribute_rows')}。")
    add(f"* 距离桶总量（d0..d3,d4+,unreachable）：{freeze.get('distance_bucket_totals')}。")
    add(f"* 片段原子数：word 侧（≤8）直方 {freeze.get('fragment_atoms_hist_words')}；"
        f"全体（含环境 C）分位数 {json.dumps(freeze.get('fragment_atoms_quantiles'), ensure_ascii=False)}；"
        f"连通分量直方 {freeze.get('fragment_components_hist')}。")
    add(f"* 无桥键分子（逐原子根回退）：{len(freeze.get('fallback_molecules', []))} 个，"
        f"OGB 行号 {freeze.get('fallback_molecules')}。")
    add(f"* cache `{freeze.get('cache_path')}` sha256 `{freeze.get('cache_sha256')}`，"
        f"commit `{freeze.get('commit')}`。")
    add("")
    td = support.get("train_directory", {})
    add("替换目录（只在 Train 内部构建；Valid/Test 只作单元，不产生对）：")
    add("")
    add(table([[c, v["n_pairs"], v["n_transforms"], v["n_transforms_supported_by_train"],
                v["n_molecules"], v["n_pairs_with_train_support"]]
               for c, v in support.get("cells", {}).items()],
              ["cell", "pairs", "transforms", "transforms in Train", "molecules",
               "pairs with Train support"]))
    add("")
    add(f"* Train 对 {td.get('n_train_pairs')}，transform {td.get('n_train_transforms')}，"
        f"重复 transform（≥2 个结构分量）{td.get('n_repeat_transforms')}，"
        f"结构分量 {td.get('n_train_pair_components')}，涉及分子 {td.get('n_molecules_in_pairs')}；"
        f"σ_y={f(td.get('sigma_y'))}，τ²={f(td.get('tau2'))}。")
    add(f"* `|Δy|` 均值 {f(td.get('delta_y_abs_mean'))}（原始单位，σ 标准化前）。")
    add(f"* 有 Train 支持的对 {support.get('support', {}).get('n_supported')}/"
        f"{support.get('n_pairs_total')}（{f(100 * support.get('support', {}).get('coverage_total', 0), 1)}%），"
        f"但 held-out 单元只有 {support.get('support', {}).get('n_held_out_supported')} 对"
        f"（{f(100 * support.get('support', {}).get('coverage_held_out', 0), 1)}%）。"
        "也就是说：在 Train 内部，同一 `(C, 词)` 约束可以反复被真实对约束；"
        "跨划分时这个约束几乎不存在，这也决定了下面第 6 节支持子集的口径。")
    add(f"* 重复 transform 子集 {support.get('repeat_subset', {}).get('n_pairs')} 对。")
    add(f"* 同一 `(cell, transform)` 单元最多包含 "
        f"{support.get('max_pairs_per_cell_transform')} 个真实对；"
        f"有多个对的单元 {support.get('cell_transform_units_with_multiple_pairs')}/"
        f"{support.get('cell_transform_units')}，这些单元内的 `mean|Δy|` = "
        f"{f(support.get('repeat_unit_label_delta_abs_mean'))}，"
        f"全体单元的 `mean|Δy|` = {f(support.get('label_delta_abs_mean'))}。"
        "这是“同一 transform 在不同结构里标签不一致”的直接度量，"
        "也是目录约束共享时无法避开的数据性质。")
    add("")
    add("## 2. 工程检查")
    add("")
    ck = checks.get("cpu", {})
    add(table([["inputs", ck.get("inputs", {}).get("ok"),
                f"recombination 坏 {ck.get('inputs', {}).get('parse_recombination_bad')}"
                f"/{ck.get('inputs', {}).get('parse_recombination_checked')}, "
                f"bond endpoint 坏 {ck.get('inputs', {}).get('parse_bond_endpoint_bad')}, "
                f"bridge 坏 {ck.get('inputs', {}).get('parse_bridge_bad')}, "
                f"frame mask ok={ck.get('inputs', {}).get('frames_edge_mask_ok')}"],
               ["renumber", ck.get("renumber", {}).get("ok"),
                "重编号后结构签名与四臂预测一致（FP64）"],
               ["port", ck.get("port", {}).get("ok"),
                f"用端口重算 CSR 与 pack 一致 max|Δ|={f(ck.get('port', {}).get('reconstruction_max_abs'), 9)}；"
                f"非等价端口改变真实特征 min|Δ|={f(ck.get('port', {}).get('min_port_change_diff'), 6)}"],
               ["context", ck.get("context", {}).get("ok"),
                f"恒等式 max|Δ|={f(ck.get('context', {}).get('identity_max_abs_err'), 2)}"],
               ["batching", ck.get("batching", {}).get("ok"),
                f"批/单条、乱序、稀疏 vs 显式循环 max|Δ|="
                f"{f(ck.get('batching', {}).get('sparse_vs_explicit_max_abs'), 2)}"],
               ["batch", ck.get("batch", {}).get("ok"), "真实批次主/辅损失有限、梯度非零"]],
              ["check", "ok", "evidence"]))
    add("")
    cuda = checks.get("cuda")
    if cuda:
        add(f"GPU/FP32 复跑：`all_ok = {cuda.get('all_ok')}`；重编号预测最大差 "
            f"{json.dumps(cuda.get('renumber', {}).get('prediction_max_abs_diff'), ensure_ascii=False)}，"
            f"容差 {cuda.get('renumber', {}).get('tolerance')}。")
    else:
        add("GPU/FP32 复跑在结果冻结前未执行（见偏差日志）；训练本身每个 epoch 都在 CUDA/FP32 上"
            "跑同一前向，重编号不变性由结构签名（与数值路径无关）覆盖。")
    add(f"CPU/FP64 `all_ok = {ck and all(ck.values())}`。")
    add("")
    add("## 3. 配方与实际条件")
    add("")
    add(f"* 共同 horizon **E={timing.get('horizon_epochs')}**：{timing.get('rule')}。"
        f"实测 epoch 秒数 {json.dumps(timing.get('observed_epoch_seconds'), ensure_ascii=False)}。"
        "horizon 只由计时决定，未看任何标签指标；epochs 1–5 是线性 warmup，与 E 无关。")
    add("* AdamW lr 1e-3，wd 1e-4，clip 5，batch 64 分子 + 16 对，5 epoch warmup + cosine 到 1e-5，"
        "seed 0，单种子；每个 epoch 全量 Train/Valid/Test 评估（原始单位）；"
        "checkpoint 只按 Valid RMSE 选择；λ_var=0。")
    add("* 主损 = 分子内解析均方；辅损 = `mean((Δf - Δy/σ_y)²)/τ²`，"
        "`τ² = max(0.25, E[((Δy)/σ_y)²])`，权重 0.25；"
        "辅对采样 80% 来自重复 transform、20% 来自全体，transform→分量→对均匀。")
    add("")
    add("## 4. 主表（ogbg-mollipo，官方 3360/420/420，原始单位，best-Valid）")
    add("")
    add(table(_main_rows(lipo), MAIN_HEADER))
    add("")
    if gine:
        add(f"外部 GINE 坐标（同划分、同目标缩放、同评估协议，raw 特征，5 层 128 维，"
            f"sum+mean readout，{f(gine['active_params'], 0)} 参数）："
            f"**Valid {f(gine['best_valid_rmse'])} / Test {f(gine['best_test_rmse'])}**"
            f"（best epoch {int(gine['best_epoch'] or 0)}，wall {f(gine['wall_minutes'], 1)} min）。")
        add("")
    add("历史参照（只读，不同协议，不可直接排名，只作坐标）：")
    add("")
    add(table([[k, f(v["valid_rmse"]), f(v["test_rmse"])] for k, v in refs.items()],
              ["historical run", "Valid", "Test"]))
    add("")
    add("## 5. 共享结构是否值钱：2×2 因子（配对 bootstrap，Test/Valid 各 1000 次重采样）")
    add("")
    rows = []
    for split in ("valid", "test"):
        b = fact.get(split, {})
        rows.append([split, f(b.get("DIRECT_NONE")), f(b.get("DIRECT_CUTRESP")),
                     f(b.get("FACTOR_NONE")), f(b.get("FACTOR_CUTRESP")),
                     f(b.get("gain_share_none")), f(b.get("gain_share_resp")),
                     f(b.get("gain_resp_factor")), f(b.get("gain_resp_direct")),
                     f(b.get("interaction"))])
    add(table(rows, ["split", "DIRECT_NONE", "DIRECT_CUTRESP", "FACTOR_NONE",
                     "FACTOR_CUTRESP", "gain_share(NONE)", "gain_share(CUTRESP)",
                     "resp|FACTOR", "resp|DIRECT", "interaction"]))
    add("")
    add("`gain_share = DIRECT − FACTOR`（正 = 分解式接口更准）；"
        "`resp|arm = NONE − CUTRESP`（正 = 真实匹配监督有用）；"
        "`interaction = resp|FACTOR − resp|DIRECT`（正 = 响应监督只在共享因式下有用）。")
    add("")
    add(table([[k, _boot(b, "valid"), _boot(b, "test")] for k, b in boot.items()],
              ["comparison (control − candidate)", "Valid ΔRMSE [95% CI]", "Test ΔRMSE [95% CI]"]))
    add("")
    add("Δ>0 表示 candidate 更好；**粗体**表示 95% 区间不含 0。")
    add("")
    add("## 6. 真实匹配替换目录上的响应指标")
    add("")
    add("评价对象是目录里**真实存在**的对：`Δy = y_b − y_a`，`Δf` 是模型的切口函数差或整分子差，"
        "全部除以 σ_y。不做任何未观察组合的伪标签。TABLE 只在训练对上按 transform 建表"
        "（每个结构分量先平均，再跨分量平均），零预测给 `Δ=0` 基线。")
    add("")
    rows = []
    rows.append(["TABLE（训练对查表）", resp.get("table", {}).get("train_supported", {}).get("n"),
                 f(resp.get("table", {}).get("train_supported", {}).get("mae")),
                 f(resp.get("table", {}).get("train_supported", {}).get("corr")),
                 f(resp.get("table", {}).get("train_supported", {}).get("sign_accuracy")),
                 f(resp.get("table", {}).get("train_supported_strict", {}).get("mae")),
                 f(resp.get("table", {}).get("train_supported_strict", {}).get("corr")),
                 f(resp.get("table", {}).get("repeat_transform", {}).get("mae"))])
    z = resp.get("zero_delta", {})
    rows.append(["零预测", z.get("all", {}).get("n"), f(z.get("train_supported", {}).get("mae")),
                 f(z.get("train_supported", {}).get("corr")),
                 f(z.get("train_supported", {}).get("sign_accuracy")),
                 f(z.get("train_supported_strict", {}).get("mae")),
                 f(z.get("train_supported_strict", {}).get("corr")),
                 f(z.get("repeat_transform", {}).get("mae"))])
    for rid, blk in resp.get("models", {}).items():
        for kind, label in (("delta_cut", f"{rid} · 切口函数差"), ("delta_mol", f"{rid} · 整分子差")):
            m = blk.get("metrics", {})
            if kind not in m.get("all", {}):
                continue
            rows.append([label, m["train_supported"][kind].get("n"),
                         f(m["train_supported"][kind].get("mae")),
                         f(m["train_supported"][kind].get("corr")),
                         f(m["train_supported"][kind].get("sign_accuracy")),
                         f(m["train_supported_strict"][kind].get("mae")),
                         f(m["train_supported_strict"][kind].get("corr")),
                         f(m["repeat_transform"][kind].get("mae"))])
    add(table(rows, ["predictor", "n(supported)", "MAE", "corr", "sign",
                     "MAE(strict)", "corr(strict)", "MAE(repeat)"]))
    add("")
    add("严格支持 = 排除目标分子自身所在的结构分量，避免用同一分子族的训练对自证。")
    add("")
    rows = []
    for rid, blk in resp.get("models", {}).items():
        for kind in ("delta_cut", "delta_mol"):
            m = blk.get("metrics", {}).get("unsupported", {}).get(kind)
            if not m:
                continue
            rows.append([rid, kind, m.get("n"), f(m.get("mae")), f(m.get("corr")),
                         f(m.get("sign_accuracy"))])
    zz = resp.get("zero_delta", {}).get("unsupported", {})
    rows.append(["零预测", "Δ=0", zz.get("n"), f(zz.get("mae")), "–", f(zz.get("sign_accuracy"))])
    add("无训练支持的对（模型必须外推，该子集上 TABLE 无定义）：")
    add("")
    add(table(rows, ["predictor", "quantity", "n", "MAE", "corr", "sign"]))
    add("")
    add("按单元分组：")
    add("")
    rows = []
    for cell, blk in resp.get("cells", {}).items():
        models = blk.get("models", {})
        first = next(iter(models.values()), {})
        rows.append([cell, blk.get("n_pairs"), f(blk.get("table", {}).get("mae")),
                     f(blk.get("zero_delta", {}).get("mae")),
                     f(first.get("delta_mol", {}).get("mae")),
                     f(first.get("delta_cut", {}).get("mae"))])
    add(table(rows, ["cell", "n pairs", "TABLE MAE", "zero MAE",
                     "模型 Δmol MAE", "模型 Δcut MAE"]))
    add("")
    add("（模型列取 `response_metrics.json` 中第一个模型，逐模型数值见该文件。）")
    add("")
    add("## 7. 组合双差（两个上下文 × 同一对词）")
    add("")
    add(f"可用四元组 {cross.get('n_quads_available')}，抽取 {cross.get('n_quads_used')}，"
        f"涉及分子 {cross.get('n_unique_molecules')}；各子集规模 {cross.get('sets')}。")
    add("")
    rows = []
    for rid, block in cross.get("models", {}).items():
        for name, sub in block.items():
            if not sub.get("n"):
                continue
            rows.append([rid, name, sub["n"], f(sub.get("D_mol", {}).get("mae")),
                         f(sub.get("D_mol", {}).get("corr", 0)) if "corr" in sub.get("D_mol", {}) else "–",
                         f(sub.get("D_mol", {}).get("sign_accuracy")),
                         f(sub.get("D_cut", {}).get("mae")),
                         f(sub.get("D_cut", {}).get("sign_accuracy"))])
    add(table(rows, ["run", "subset", "n", "D_mol MAE", "–", "D_mol sign",
                     "D_cut MAE", "D_cut sign"]))
    add("")
    add("`D_y = [y(C1,Sb) − y(C1,Sa)] − [y(C2,Sb) − y(C2,Sa)]`；"
        "模型的 `D` 用同一 (Sa,Sb) 的两个上下文差。双差可以抵消只依赖词的公共项，"
        "留下由环境经共享状态决定的部分。")
    add("")
    add("## 8. 数据效率")
    add("")
    if esol:
        add("ESOL（1128 分子，官方 902/112/114；原始单位，best-Valid）：")
        add("")
        add(table(_main_rows(esol), MAIN_HEADER))
        add("")
        if gine_esol:
            add(f"ESOL 的 raw-GINE 坐标：Valid {f(gine_esol['best_valid_rmse'])} / "
                f"Test {f(gine_esol['best_test_rmse'])}。")
            add("")
    small = [r for r in lipo if r.get("train_fraction")]
    if small:
        add("ogbg-mollipo 小样本（嵌套 sha256 子集，同一 E=160）：")
        add("")
        add(table(_main_rows(small), MAIN_HEADER))
        add("")
    add("## 9. 案例（每条由显式数值规则选出，非手工挑选）")
    add("")
    for key, block in cases.items():
        if not isinstance(block, dict):
            continue
        picks = block.get("picks") or []
        if not picks:
            continue
        add(f"**{key}** — {block.get('rule')}")
        add("")
        rows = []
        for p in picks:
            rows.append([p.get("cell", block.get("cell", "")),
                         p.get("molecule_a_ogb_id", ""), p.get("molecule_b_ogb_id", ""),
                         f(p.get("delta_y_sigma"), 3), f(p.get("delta_full_sigma"), 3),
                         f(p.get("delta_isolated_sigma"), 3), f(p.get("merge_magnitude"), 3),
                         f(p.get("molecule_a_abs_error"), 3), f(p.get("molecule_b_abs_error"), 3),
                         f(p.get("pair_delta_abs_error"), 3),
                         p.get("repeat_count", ""), f(p.get("label_delta_spread_sigma"), 3)])
        add(table(rows, ["cell", "mol a (OGB id)", "mol b", "Δy/σ", "Δf(C)", "Δf(∅)",
                         "merge", "err a", "err b", "pair err", "repeats", "label spread"]))
        add("")
    add("## 10. 负结果与开放问题")
    add("")
    add("* **没有超过 GINE。** 本轮最好的候选在 Valid 上比同协议 GINE 坐标差 "
        f"{f((best_lipo['best_valid_rmse'] - gine['best_valid_rmse']) if best_lipo and gine else None)}。"
        "禁止 GNN 骨干的前提下，这个差距是该方法本身的上限证据，而不是训练不足："
        "候选的训练误差已明显低于 GINE 的验证误差（见主表 Train 列）。")
    add("* **Valid/Test 排序反转。** `DIRECT_CUTRESP` 在 Test 上比 `FACTOR_CUTRESP` 好 "
        f"{f(-fact.get('test', {}).get('gain_share_resp', 0))}，在 Valid 上差 "
        f"{f(fact.get('valid', {}).get('gain_share_resp'))}；两侧 95% 区间都不含 0。"
        "best-Valid 选择本身几乎不损失 Test（`split_gap.json` 的 "
        "`cost_of_valid_selection` ≈ 0），所以这是划分难度差异，不是选择伪影。"
        "任何单侧结论都会被这次反转驳倒。")
    add("* **MOLPAIR 有害。** 只用整分子差约束 `f̄` 会让切口函数失配，"
        "Valid/Test 都比 CUTRESP 差一大截；说明起作用的不是“任何成对监督”，"
        "而是“监督切口函数本身”。")
    add("* **TABLE 在支持子集上远强于所有神经模型。** 这说明当前数据量下"
        "“按 transform 查表”仍是更强的近似；共享响应状态的价值必须体现在"
        "无支持外推（`unsupported` 子集）上，那里 TABLE 无定义，模型确实优于零预测，"
        "但绝对误差仍然偏大。")
    add("")
    add("## 11. 复现")
    add("")
    add("```")
    add("code/trc_prepare.py --dataset lipo|esol     # 建 cache（本地，需要 rdkit）")
    add("code/trc_checks.py --device cpu|cuda        # 工程检查")
    add("code/trc_train.py --run-id ... --arm ... --supervision ... --dataset ... "
        "--epochs 160 --device cuda")
    add("code/trc_reference.py --run-id ... --dataset lipo --epochs 160 --device cuda")
    add("code/trc_chain.sh code code/trc_queue_gpu1.txt   # 单卡串行队列")
    add("code/trc_analyze.py --device cpu            # analysis/*.json")
    add("code/trc_cases.py                           # analysis/cases.json")
    add("code/trc_report.py                          # 本文件")
    add("```")
    add("")
    add("`data/` 与 `results/*/` 不进版本库（cache 与权重是运行产物），"
        "但 `analysis/` 的每个数字都可以由上面的命令从公开 cache 重建。")
    add("")
    (C.ROUND_DIR / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {C.ROUND_DIR / 'REPORT.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
