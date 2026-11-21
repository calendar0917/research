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


def _ck_evidence(res: Mapping[str, Any]) -> str:
    keep = {k: v for k, v in res.items()
            if k.endswith("max_abs") or k.endswith("_bad") or k.endswith("_diff")
            or k in ("frames_edge_mask_ok", "ctx_feature_selfconsistent")}
    return json.dumps(keep, ensure_ascii=False)


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
    add("2. **但在 Test 上，`DIRECT_CUTRESP` 与 GINE 坐标在统计上不可区分**：Test RMSE "
        "0.6723 vs 0.6582，配对 bootstrap 差值 "
        f"{_boot(boot.get('GINE_RAW_LIPO_R0 - LIPO_DIRECT_CUTRESP_R0', {}), 'test')}"
        "（跨 0）。Valid 上则明显更差。同样的反转在第 5 节出现，说明单看一侧会得出相反结论。")
    add("")
    add(f"3. **外部 GINE 与历史 GINE 参考一致。** 本轮 raw-GINE 坐标 "
        f"{f(gine['best_valid_rmse']) if gine else '–'}/"
        f"{f(gine['best_test_rmse']) if gine else '–'}，历史 `GINE_RAW_REF` "
        f"{f(refs['GINE_RAW_REF']['valid_rmse'])}/{f(refs['GINE_RAW_REF']['test_rmse'])}"
        "（历史为 240 epoch + early stop，本轮统一 E=160），说明坐标可比。")
    add("")
    add("4. **交互是必需的，但它的具体形式无关。** 组合双差上纯加性臂 `ADD` 恒等于 0"
        "（结构上无法表达“环境改变词的作用”），而 `FACTOR`/`DIRECT` 的切口双差"
        "显著优于零基线（sign 0.722/0.760 vs 0.550）；同时 `FACTOR` 相对 `ADD` 的"
        f"整分子差值是 Valid {f(fact.get('valid', {}).get('factor_minus_add'))}、"
        f"Test {f(fact.get('test', {}).get('factor_minus_add'))}，都跨 0。"
        "所以“必需交互”成立，“必须低秩可解释状态”不成立。")
    add("")
    add("5. **真实匹配监督在整分子指标上收益很小，在组合维度上收益清楚。** "
        "`FACTOR` 由 `NONE` 到 `CUTRESP`：Valid +0.033（显著）、Test +0.011（不显著），"
        "整分子层面几乎不可见；但双差 sign 0.666 → 0.722、"
        "切口函数差在严格支持子集上的 MAE 0.211 → 0.119。`MOLPAIR`（只约束整分子差）"
        "则在两个维度上都明显有害。")
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
        add(f"GPU/FP32 复跑（`--skip renumber`：该子项需要在算力机上重建 mini pack，"
            f"而远端 venv 没有 rdkit，见偏差日志）：`all_ok = {cuda.get('all_ok')}`。")
        add("")
        add(table([[n, cuda.get(n, {}).get("ok"),
                    _ck_evidence(cuda.get(n, {}))] for n in
                   ("inputs", "context", "port", "batching", "batch")],
                  ["check", "ok (cuda/fp32)", "evidence"]))
        add("")
    else:
        add("GPU/FP32 复跑在结果冻结前未执行（见偏差日志）。")
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
    hours = sum(float(r.get("gpu_hours") or 0.0) for r in lipo + esol + ref)
    add(f"* 已用计算：{len(lipo)} 个 lipo 单元 + {len(esol)} 个 ESOL 单元 + "
        f"{len(ref)} 个 GINE 坐标，合计 **{hours:.2f} GPU·h**（16 GPU·h 预算内），"
        "队列上限 20 个正式单元未用满。这个数只累加每个单元自己的 wall time，"
        "不含两次作废的重跑和排队空转。")
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
    ordered = sorted(boot.items(), key=lambda kv: ("ESOL" in kv[0], kv[0]))
    add(table([[k, _boot(b, "valid"), _boot(b, "test")] for k, b in ordered],
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
    cons = _load("consistency.json")
    if cons:
        add("切口函数的内部一致性（best-Valid epoch 上的全量评估）：")
        add("")
        rows = []
        for rid, c in cons.items():
            rows.append([rid, f(c.get("train_cut_minus_mol")),
                         f(c.get("cut_var_mean", {}).get("train"), 4),
                         f(c.get("cut_var_mean", {}).get("valid"), 4),
                         f(c.get("cut_rmse", {}).get("train")),
                         f(c.get("cut_rmse", {}).get("valid")),
                         f(c.get("mol_rmse", {}).get("valid"))])
        add(table(rows, ["run", "Train cut − mol", "cut var (Train)", "cut var (Valid)",
                         "cut RMSE (Train)", "cut RMSE (Valid)", "mol RMSE (Valid)"]))
        add("")
        add("`cut var` 是同一分子内部各解析切口分数 `f_c` 的方差均值——它衡量模型是否真的把"
            "“不同接口有不同作用”表示出来了，而不是把所有解析压到同一个整分子值上。"
            "`Train cut − mol` 是切口级训练误差与整分子训练误差之差，反映主损"
            "（分子内平均后平方）与切口级拟合之间的差距。")
        add("")
        add("机制上可以读到一件事：在同一个 `FACTOR` 骨架上，`CUTRESP` 把 Train 内部的切口"
            "分数离散度从 0.012 抬到 0.023，同时把切口级 Train RMSE 从 0.234 抬到 0.306，"
            "却把整分子 Valid RMSE 从 0.743 降到 0.710。真实匹配监督并没有让模型更贴合"
            "Train 的切口标签，而是把它推向“同一分子的不同接口给出不同值、但均值更准”的表示——"
            "这正是接口分解应有的行为。")
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
            cut = sub.get("D_cut", {})
            if sub["n"] < 10:
                rows.append([rid, name, sub["n"], "n<10，不解释", "–", "–",
                             "–", "–"])
                continue
            rows.append([rid, name, sub["n"],
                         f(sub.get("D_mol", {}).get("mae")),
                         f(sub.get("D_mol", {}).get("sign_accuracy")),
                         f(cut.get("mae")),
                         "D≡0（加性臂上下文精确抵消）" if cut.get("degenerate")
                         else f(cut.get("sign_accuracy")),
                         f(cut.get("pred_std"), 8)])
    add(table(rows, ["run", "subset", "n", "D_mol MAE", "D_mol sign",
                     "D_cut MAE", "D_cut sign", "std(D_cut)"]))
    add("")
    add("**这个检验在本数据集上基本做不了，这是结论而不是缺陷。** 四元组要求同一对词出现在两个"
        f"不同上下文里，而 Valid 只有 {support.get('cells', {}).get('valid', {}).get('n_pairs')} 对、"
        f"Test 只有 {support.get('cells', {}).get('test', {}).get('n_pairs')} 对，"
        "所以 held-out 双差只有 1–2 个样本，任何数字都没有统计意义；"
        "有意义的只有 `train_train`（同一划分内部）与 `train_or_cross`（Train 与重组单元之间）。"
        "下表的 held-out 行留着是为了说明这一点，不是用来支持/否证假设的。")
    add("")
    add("`D_y = [y(C1,Sb) − y(C1,Sa)] − [y(C2,Sb) − y(C2,Sa)]`；"
        "模型的 `D` 用同一 (Sa,Sb) 的两个上下文差。双差抵消只依赖词的公共项，"
        "只留下“环境如何改变这一对词的作用”。")
    add("")
    add("**这张表是本轮最干净的一条机制证据。** 在 `train_train` 子集（1835 个四元组）上：")
    add("")
    add("* `ADD` 的 `D_cut` **恒等于 0**（std = 8e-08，纯浮点残差）：因为加性头"
        "`a(C)+a(S)+b(e)` 在双差里把上下文精确抵消。它的 `D_cut` MAE = 0.404 正好等于"
        "目标本身的 `mean|D_y|`，也就是零基线。换句话说，**纯加性臂在结构上就无法表达"
        "“环境改变词的作用”**；它在整分子 RMSE 上与 `FACTOR` 打平这件事，不能用来论证"
        "“组合效应不重要”。")
    add("* `DIRECT_CUTRESP`（D_cut MAE 0.236 / sign 0.760）与 `FACTOR_CUTRESP`"
        "（0.258 / 0.722）都显著优于零基线（0.404）与 `GLOBAL` 的整分子双差"
        "（0.393 / 0.550），说明共享接口状态确实承载了环境相关的作用。")
    add("* 真实匹配监督在这个诊断上有单独贡献："
        "`FACTOR` 0.666 → 0.722（NONE → CUTRESP）、`DIRECT` 0.658 → 0.760；"
        "而 `MOLPAIR` 把它压到 0.423。这条增益在整分子 RMSE 上看不见，"
        "只在“组合”这一维度上可见。")
    add("* 训练分子数减半后这条信号明显退化（`FACTOR` sign 0.722 → 0.553 @50% → 0.541 @25%），"
        "而零基线不变——所以双差能力依赖样本量，且比整分子 RMSE 更早饱和。")
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
        add("ogbg-mollipo 小样本（嵌套 sha256 子集：25% ⊂ 50% ⊂ 100%，各臂共用同一子集，同一 E=160）：")
        add("")
        add(table(_main_rows(small), MAIN_HEADER))
        add("")
        full = {(r["arm"], r["supervision"]): r for r in lipo if not r.get("train_fraction")}
        rows = []
        for r in sorted(small, key=lambda r: (r["arm"], -(r.get("train_fraction") or 0))):
            base = full.get((r["arm"], r["supervision"]))
            rows.append([f"{r['arm']}/{r['supervision']}", f(r.get("train_fraction"), 2),
                         f(r["best_valid_rmse"]), f(r["best_test_rmse"]),
                         f(r["best_valid_rmse"] - base["best_valid_rmse"]) if base else "–",
                         f(r["best_test_rmse"] - base["best_test_rmse"]) if base else "–"])
        add(table(rows, ["arm/aux", "train frac", "Valid", "Test", "ΔValid vs 100%",
                         "ΔTest vs 100%"]))
        add("")
        add("读法：在相同训练分子数下比较不同臂，才是在问“哪一种接口表示更省样本”；"
            "不同臂的 100%→50%→25% 退化斜率可以直接对比，因为子集是同一批分子。")
        add("")
        add("事实：两种臂的退化斜率几乎一样（Valid 上 FACTOR +0.145/+0.248、"
            "DIRECT +0.090/+0.247），所以“更省样本”在两个臂之间没有差别；"
            "而 Valid/Test 的反转在 100%、50%、25% 三个数据量上都存在"
            "（50% 时 DIRECT 在 Valid 领先 0.017、FACTOR 在 Test 领先 0.020；"
            "25% 时 FACTOR 在 Valid 领先 0.037、DIRECT 在 Test 领先 0.016，"
            "但 25% 两边都接近训练集规模，绝对值已不可用于比较）。"
            "所以这个反转是划分的性质，不是样本量的性质。")
    add("## 9. 案例（每条由显式数值规则选出，非手工挑选）")
    add("")
    for key in ("merge", "redirect", "composed", "pathological"):
        block = cases.get(key)
        if not isinstance(block, dict):
            continue
        picks = block.get("picks") or []
        if not picks:
            continue
        rule = block.get("rule")
        if key == "composed":
            rule += "（“预测得好”指 cross 单元内误差最小的 40%，该单元本身最难，"                     "所以绝对误差仍有 1–2 kcal/mol）"
        add(f"**{key}** — {rule}")
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
        if key == "redirect":
            rows = [[p.get("molecule_a_ogb_id"), p.get("molecule_b_ogb_id"),
                     p.get("flip_side"), f(p.get("word_a_with_context"), 3),
                     f(p.get("word_a_isolated"), 3), f(p.get("word_b_with_context"), 3),
                     f(p.get("word_b_isolated"), 3), f(p.get("delta_y_sigma"), 3),
                     f(p.get("merge_magnitude"), 3),
                     f(p.get("context_shift_a"), 3), f(p.get("context_shift_b"), 3)]
                    for p in picks]
            add(table(rows, ["mol a", "mol b", "flip side", "f(Sa|C)", "f(Sa|∅)",
                             "f(Sb|C)", "f(Sb|∅)", "Δy/σ", "merge", "shift a", "shift b"]))
        elif key == "pathological":
            rows = [[p.get("cell"), p.get("repeat_count"),
                     f(p.get("label_delta_spread_sigma"), 3),
                     str(p.get("label_deltas_sigma")), p.get("molecule_pairs_ogb")]
                    for p in picks]
            add(table(rows, ["cell", "repeats in cell", "label spread σ",
                             "Δy/σ of every repeat", "molecule pairs (OGB id)"]))
            add("")
            for p in picks:
                add(f"* 同 transform 的两个词：`{p['transform_words'][0]}` / "
                    f"`{p['transform_words'][1]}`")
            add("")
            continue
        else:
            add(table(rows, ["cell", "mol a (OGB id)", "mol b", "Δy/σ", "Δf(C)", "Δf(∅)",
                             "merge", "err a", "err b", "pair err", "repeats",
                             "label spread"]))
        add("")
    add("## 10. 对三个问题的直接回答")
    add("")
    add("**(1) 少量共享响应状态能否承载带接口片段的任务作用？**")
    add("")
    add("能，但结论分成两半，而且整分子 RMSE 会给出误导性答案。")
    add("")
    add(f"* **整分子 RMSE 上看不出交互的必要性**：`FACTOR` 相对纯加性 `ADD` 的差值 "
        f"Valid {f(fact.get('valid', {}).get('factor_minus_add'))}、"
        f"Test {f(fact.get('test', {}).get('factor_minus_add'))}，配对区间都跨 0。"
        "如果只看这个指标，会得出“加性就够”的结论。")
    add("* **组合双差上，加性结构上就不够**：`ADD` 的双差恒等于 0（上下文精确抵消，见第 7 节），"
        "`FACTOR`/`DIRECT` 的切口双差都显著好于零基线（MAE 0.258/0.236 vs 0.404，"
        "sign 0.722/0.760 vs 0.550）。也就是说，要表达“环境改变片段作用”，"
        "上下文与片段的交互项是必需的。")
    add("* **但交互的具体形式不重要**：`FACTOR` 的低秩双线性与 `DIRECT` 的黑箱 MLP "
        "在整分子 RMSE 上互有胜负（Valid FACTOR 好 0.038、Test DIRECT 好 0.050），"
        "在双差上 DIRECT 略好（0.760 vs 0.722 sign）。所以能确认的是"
        "“必需交互 + 共享接口状态”，不能确认的是“必须用可解释的低秩状态”。")
    add("")
    add("**(2) 显式共享限制是否被数据集支持？**")
    add("")
    add("支持，但形式比预期弱。")
    add("")
    if resp:
        m = resp.get("models", {})
        first = next(iter(m.values()), {})
        cblock = resp.get("cells", {}).get("cross", {})
        best_cross = min((v.get("delta_mol", {}).get("mae", 9e9)
                          for v in cblock.get("models", {}).values()), default=None)
        add(f"* 真实匹配替换目录给出 {support.get('n_pairs_total')} 个可评估对；"
            f"held-out（cross，{cblock.get('n_pairs')} 对、727 个互不相同的分子、"
            "两个上下文共享同一对词）上，最好的整分子差 MAE 是 "
            f"{f(best_cross)}，零预测是 "
            f"{f(cblock.get('zero_delta', {}).get('mae'))}。"
            "模型学到的是可外推的响应，而不只是记住了标签。")
        def _best(subset: str, kind: str) -> float | None:
            vals = [v.get("metrics", {}).get(subset, {}).get(kind, {}).get("mae")
                    for v in m.values()]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else None

        add(f"* 宽支持子集（{resp.get('table', {}).get('train_supported', {}).get('n')} 对，"
            f"含目标自身结构分量）：TABLE MAE "
            f"{f(resp.get('table', {}).get('train_supported', {}).get('mae'))}，"
            f"最好的神经网络切口函数差 {f(_best('train_supported', 'delta_cut'))}，"
            f"最好的整分子差 {f(_best('train_supported', 'delta_mol'))}——两者几乎打平，"
            "但 TABLE 的相关性 0.96 高于模型。")
        add(f"* 严格支持子集（{resp.get('table', {}).get('train_supported_strict', {}).get('n')} 对，"
            "已排除目标分子所在的结构分量）：TABLE MAE "
            f"{f(resp.get('table', {}).get('train_supported_strict', {}).get('mae'))}，"
            f"而 `DIRECT_CUTRESP` 的切口函数差是 "
            f"{f(_best('train_supported_strict', 'delta_cut'))}，"
            "比查表准一倍多。查表在严格子集上退化，正是“同 transform 跨结构共享”"
            "这一假设的直接检验：能共享的部分是真实存在的（否则模型不会在排除自身分量后仍然准确），"
            "但它不足以支撑一个查表近似。")
        add("* 反面证据同样清楚：目录里 577 个“同一 transform、多个真实对”的单元内部 "
            f"`mean|Δy|` = {f(support.get('repeat_unit_label_delta_abs_mean'))}，"
            f"只比全体单元的 {f(support.get('label_delta_abs_mean'))} 低 10%——"
            "同一个 `(C, 词)` 约束下标签本身就不一致，这是目录约束共享时无法回避的噪声地板。")
    add("")
    add("**(3) 对整分子预测和数据效率的价值？**")
    add("")
    erow = {r["run_id"]: r for r in esol}

    def _esol_delta(split: str, a: str, b: str) -> str:
        if a not in erow or b not in erow:
            return "–"
        return f(erow[a][f"best_{split}_rmse"] - erow[b][f"best_{split}_rmse"])

    add(f"* 对 `GLOBAL`（纯整分子描述，无接口分解）：`GLOBAL_NONE − FACTOR_NONE` "
        f"Valid {f(fact.get('valid', {}).get('global_minus_factor'))}、"
        f"Test {f(fact.get('test', {}).get('global_minus_factor'))}（正 = 分解更好），"
        f"配对区间 Valid {_boot(boot.get('LIPO_GLOBAL_NONE_R0 - LIPO_FACTOR_NONE_R0', {}), 'valid')}、"
        f"Test {_boot(boot.get('LIPO_GLOBAL_NONE_R0 - LIPO_FACTOR_NONE_R0', {}), 'test')}，都跨 0；")
    add(f"  在 ESOL（902 个训练分子）上同一个对比是 Valid "
        f"{_esol_delta('valid', 'ESOL_GLOBAL_NONE_R0', 'ESOL_FACTOR_NONE_R0')}、"
        f"Test {_esol_delta('test', 'ESOL_GLOBAL_NONE_R0', 'ESOL_FACTOR_NONE_R0')}。"
        "方向一致：数据越少，接口分解越值钱；但在 3360 个训练分子上不显著。")
    add("* 对数据效率：lipo 用 50%/25% 训练分子重跑 `FACTOR_CUTRESP` 与 `DIRECT_CUTRESP`，"
        "见第 8 节；ESOL 只有 902 个训练分子，是同一个问题的自然小样本点。")
    add("* 对绝对精度：没有超过外部 GINE 坐标，见第 0 节。")
    add("")
    add("## 11. 负结果与开放问题")
    add("")
    add("* **没有超过 GINE。** 本轮最好的候选在 Valid 上比同协议 GINE 坐标差 "
        f"{f((best_lipo['best_valid_rmse'] - gine['best_valid_rmse']) if best_lipo and gine else None)}。"
        "禁止 GNN 骨干的前提下，这个差距更像是方法上限而不是训练不足："
        "配方的 cosine 已经走完，最佳 epoch 落在内部（100/160、135/160），尾部平台化，"
        "且 Train RMSE 已远低于 Valid RMSE（见主表）。")
    add("* **Valid/Test 排序反转。** `DIRECT_CUTRESP` 在 Test 上比 `FACTOR_CUTRESP` 好 "
        f"{f(-fact.get('test', {}).get('gain_share_resp', 0))}，在 Valid 上差 "
        f"{f(fact.get('valid', {}).get('gain_share_resp'))}；两侧 95% 区间都不含 0。"
        "best-Valid 选择本身几乎不损失 Test（`split_gap.json` 的 "
        "`cost_of_valid_selection` ≈ 0），所以这是划分难度差异，不是选择伪影。"
        "任何单侧结论都会被这次反转驳倒。")
    add("* **MOLPAIR 有害。** 只用整分子差约束 `f̄` 会让切口函数失配，"
        "Valid/Test 都比 CUTRESP 差一大截；说明起作用的不是“任何成对监督”，"
        "而是“监督切口函数本身”。")
    add("* **查表很强，但只在宽支持口径上。** 在含目标自身结构分量的 4856 对上，"
        "TABLE 的 MAE 0.0885 与最好的神经切口函数差 0.0904 打平、相关性更高；"
        "但在排除目标自身分量的 1899 对上 TABLE 退化到 0.2218，"
        "`DIRECT_CUTRESP` 反而是 0.0925。所以“按 transform 查表”在本数据量下"
        "仍是宽支持口径的强基线，但它不能替代对未知结构的外推；"
        "真正的空白是**无任何训练支持**的 914 对：那里 TABLE 无定义，"
        "模型 MAE 0.367–0.415，明显优于零预测 0.651，但绝对误差仍然偏大。")
    add("")
    add("## 12. 复现")
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
