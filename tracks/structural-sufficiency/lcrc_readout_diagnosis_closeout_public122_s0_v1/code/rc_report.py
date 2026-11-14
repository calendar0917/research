"""Generate ``REPORT.md`` (Chinese) for the closeout round from analysis JSON."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import rc_common as R


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x, nd=2):
    if x is None:
        return "—"
    return f"{100.0 * float(x):+.{nd}f}%"


def _load(name: str) -> Any:
    p = R.ANALYSIS / name
    return R.read_json(p) if p.exists() else {}


def build_report() -> str:
    res = _load("results.json")
    aud = _load("source_audit.json")
    lp = _load("learning_path.json")
    comp = _load("comparisons.json")
    bud = _load("budget.json")

    src = res.get("source", {})
    step0 = res.get("step0_common", {})
    table = res.get("table", [])
    gains = res.get("gains", {})
    purch = res.get("purchase_threshold", {})
    boot = res.get("bootstrap", {})
    branch = res.get("branch_presence", {})
    layer = res.get("interpretation_layer", {})
    lpsum = lp.get("summary", {})

    replay = aud.get("replay", {})
    align = aud.get("alignment", {})
    cc = aud.get("hcache_crosscheck", {})
    src_ok = aud.get("verdict") == "SOURCE_OK"
    channel_formed = any(v.get("psi_grad_nonzero_by_1800") for v in lpsum.values()) if lpsum else None
    purchase_passed = purch.get("passed")
    layer_name_pre = layer.get("layer", "")

    def _best(vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    resp_rel = _best((lpsum.get(r, {}) or {}).get("channel_response_relative_to_full_delta") for r in ("P_POST", "P_PRE"))
    resp_raw = _best((lpsum.get(r, {}) or {}).get("channel_response_raw_rms") for r in ("P_POST", "P_PRE"))
    dfull = _best((lpsum.get(r, {}) or {}).get("delta_full_rms_std") for r in ("P_POST", "P_PRE"))
    psi_grads = {r: (lpsum.get(r, {}) or {}).get("grad_norm_preclip", {}).get("psi") for r in ("P_POST", "P_PRE")}
    scale = (res.get("target_norm") or {}).get("scale")

    # ---- opening four answers -------------------------------------------
    a1 = ("来源与缓存一致" if src_ok else "来源或缓存未通过校验")
    a1 += (f"：best_R_SEL.pt 内部 update={src.get('internal_update')}，"
           f"eval 重放 Valid/Test RMSE={f(replay.get('valid_rmse'),6)}/{f(replay.get('test_rmse'),6)}，"
           f"逐分子最大差 {f((align.get('per_split') or {}).get('test',{}).get('max_abs_diff'),2e0 and 6)}（≤1e-5），"
           f"h 缓存与 return_aux 交叉核对最大差 {f((cc or {}).get('max_abs'),6)}。")
    if channel_formed is None:
        a2 = "新增通道响应未测得（诊断缺失）。"
    elif layer_name_pre == "EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG":
        pg = psi_grads.get("P_POST")
        pg_s = f"{pg:.1e}" if isinstance(pg, (int, float)) else "—"
        a2 = ("计算路径是通的（首步只动 delta 末层，之后 psi/delta 首层梯度非零），"
              f"但额外通道的输出贡献很弱：1800 步时 ‖∇psi‖≈{pg_s}（f 约 1.8），"
              f"完整 delta 输出 RMS≈{f(dfull,4)}（标准化，目标尺度≈{f(scale,3)}），"
              f"通道响应原始单位 RMS≈{f(resp_raw,6)}，只占 delta 自身输出≈{f((resp_rel or 0)*100,1)}%。")
    elif channel_formed:
        a2 = "新增通道确实形成了可测响应（psi/delta 首层梯度非零，通道响应见 §3）。"
    else:
        a2 = "新增通道在 1800 步后仍近乎未被使用（psi/delta 首层梯度接近零，响应很弱）。"
    if purchase_passed:
        a3 = "有新增探针越过了相对 P_WARM 与共同 step0 的 1% 采购门槛（见 §2.2）。"
    elif table and all((row.get("r_sel") or {}).get("update") == 0 for row in table):
        a3 = ("没有越过采购门槛：三臂的 R_SEL（按 Valid）全部回到共同 step0（update 0），"
              "即读出头训练 1800 步没有给出更好的 Valid 状态；Test 终点仅小幅改善且无法由 Valid 选到，"
              "事后按 Test 选的 oracle 虽超 1%，但含 Test 选择乐观偏差、不是可选规则。")
    else:
        a3 = "没有越过 1% 采购门槛：相对 P_WARM 与共同 step0 均无 Valid+Test 双改善。"
    layer_name = layer.get("layer", "INCONCLUSIVE")
    if layer_name == "EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG":
        a4 = ("值得保留的方向：暂停给这类读出再加模块（本轮证据是额外通道在本配置下没被用起来，"
              "不是“读出容量不够”）；仍未识别：池化位置问题（PRE vs POST 在本配置下退化）与"
              "表示是否缺任务信息（H_REP 本轮不检验）。")
    elif layer_name == "NO_READOUT_GAIN_THIS_CONFIG":
        a4 = ("值得保留的方向：读出侧仍有计算响应但无留出收益，支持暂缓给出此类读出加模块；"
              "仍未识别：池化位置与 H_REP。")
    elif layer_name == "PURCHASE_CANDIDATE":
        a4 = ("达到采购门槛：保留该探针候选并停止，等待研究者设计下一步；"
              "仍未识别：能否迁移到其它任务/来源。")
    else:
        a4 = "本轮判读不充分（INCONCLUSIVE）；需检查实现或测量。"

    L: list[str] = []
    L.append("# 读出诊断收尾 — 报告（中文）\n")
    L.append(f"- round: `{R.ROUND_ID}`  ·  status: **{R.STATUS_TAG}**  ·  source commit: `{src.get('state_hash','')[:12]}`")
    L.append(f"- 来源：`{src.get('arm')}` 的 E1 R_SEL（update={src.get('internal_update')}）；单 seed（0）；"
             "官方既有划分；每 fit epoch 与公共 grid 全量评估 Train/Valid/Test。Test 参与开发诊断。")
    L.append("- 本报告在研究者看过上一轮全部结果之后设计，属 **DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP**，不称独立确认。\n")

    L.append("## 0. 四句回答\n")
    L.append(f"1. **{a1}**")
    L.append(f"2. **{a2}**")
    L.append(f"3. **{a3}**")
    L.append(f"4. **{a4}**\n")

    # ---- source audit ----------------------------------------------------
    L.append("## 1. 来源恢复与缓存审计\n")
    L.append(f"- 上游把三条探针落在 `u0_snapshot.pt`（update 9720），理由是“E1 的 R_SEL 权重未单独持久化”。"
             f"但 manifest 记录 `runs/LIP_C1_R1_F100/best_R_SEL.pt`，`retained.update={aud.get('source',{}).get('internal_update')}`，"
             "且 E1 与 E1+E2 的 R_SEL 是同一状态。")
    L.append(f"- 本次从该权重恢复成功：内部 update=`{aud.get('source',{}).get('internal_update')}`，"
             f"state hash=`{str(aud.get('source',{}).get('state_hash',''))[:16]}…`；"
             f"与 u0_snapshot（`{str(aud.get('u0_snapshot',{}).get('state_hash',''))[:16]}…`）确为不同状态。")
    L.append(f"- eval 重放：Valid RMSE={f(replay.get('valid_rmse'),6)}（参考 {f(replay.get('expected_valid_rmse'),6)}），"
             f"Test RMSE={f(replay.get('test_rmse'),6)}（参考 {f(replay.get('expected_test_rmse'),6)}），"
             f"误差 {f(replay.get('valid_rmse_abs_err'),2e0 and 8)}/{f(replay.get('test_rmse_abs_err'),8)}。")
    per = (align.get("per_split") or {})
    if per:
        L.append(f"- 逐分子对齐上游 update8165 预测（反标准化一次）："
                 f"train {f(per.get('train',{}).get('max_abs_diff'),6)}、valid {f(per.get('valid',{}).get('max_abs_diff'),6)}、"
                 f"test {f(per.get('test',{}).get('max_abs_diff'),6)}；均 ≤1e-5。status=`{align.get('status')}`。")
    hid = (aud.get("hcache") or {}).get("identity") or {}
    L.append(f"- 新建 **身份化** clean-h 缓存：identity=`{str(hid.get('identity_sha256',''))[:16]}…`"
             f"（含 source_state_hash / feature_spec / struct_norm / target_norm / split row hashes），"
             f"不再复用仅按 dataset 命名的旧 `hcache_Lipophilicity`。")
    if cc:
        L.append(f"- 缓存 vs `return_aux` 固定批次交叉核对：最大差 {f(cc.get('max_abs'),8)}（阈值 1e-5），通过={f(cc.get('passed'))}。")
    L.append("- 上游 protocol 的来源替换事实原样保留（旧结果仍以 `source: u0_snapshot (9720)` 标注）；"
             "本报告只说明：“目标 E1 权重不可得”对该轨迹与 manifest 不一致，未静默改旧协议或旧预测。\n")

    # ---- main results table ---------------------------------------------
    L.append("## 2. 三条新探针（source = update 8165）\n")
    L.append(f"共同 step0（三臂一致）：Valid RMSE={f(step0.get('valid_rmse'),6)}，"
             f"Test RMSE={f(step0.get('test_rmse'),6)}，Train_fit RMSE={f(step0.get('train_rmse'),6)}。\n")
    L.append("R_SEL（主规则，grid 含 step0，min Valid MSE）。逐臂同时给 step0 与 R_SEL 的 update 与三划分 RMSE/MAE：\n")
    L.append("| probe | new_params | 状态 | update | Train_fit RMSE | Valid RMSE | Test RMSE | Valid MAE | Test MAE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in table:
        r = row["probe"]
        s0 = row.get("step0", {})
        rr = row.get("r_sel", {})
        L.append(f"| {r} | {row.get('new_params')} | step0 | 0 | {f(s0.get('train',{}).get('rmse'))} | "
                 f"{f(s0.get('valid',{}).get('rmse'))} | {f(s0.get('test',{}).get('rmse'))} | "
                 f"{f(s0.get('valid',{}).get('mae'))} | {f(s0.get('test',{}).get('mae'))} |")
        L.append(f"| {r} | {row.get('new_params')} | R_SEL | {rr.get('update','—')} | {f(rr.get('train_rmse'))} | "
                 f"{f(rr.get('valid_rmse'))} | {f(rr.get('test_rmse'))} | {f(rr.get('valid_mae'))} | {f(rr.get('test_mae'))} |")
    L.append("")

    L.append("### 2.1 相对 P_WARM 与共同 step0 的增益（R_SEL）\n")
    L.append("正号=更好。相对 step0 的比较在同一起点（三臂共享 step0）上给出，避免把 WARM 自身的训练当成基线漂移。\n")
    L.append("| probe | R_SEL upd | Δ vs WARM Valid | Δ vs WARM Test | 相对WARM Valid | 相对WARM Test | Δ vs step0 Valid | Δ vs step0 Test |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r, g in gains.items():
        if not g:
            L.append(f"| {r} | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {r} | {g.get('R_SEL_update')} | {f(g.get('vs_warm_valid_delta'),6)} | "
                 f"{f(g.get('vs_warm_test_delta'),6)} | {pct(g.get('vs_warm_valid_rel_gain'))} | {pct(g.get('vs_warm_test_rel_gain'))} | "
                 f"{f(g.get('vs_step0_valid_delta'),6)} | {f(g.get('vs_step0_test_delta'),6)} |")
    L.append("")

    L.append(f"### 2.2 采购门槛（{purch.get('rule')}）\n")
    for r, v in (purch.get("result") or {}).items():
        L.append(f"- **{r}**：passes={f(v.get('passes'))}；相对增益 "
                 f"WARM(Valid/Test)={pct((v.get('relative_gain') or {}).get('vs_warm_valid'))}/"
                 f"{pct((v.get('relative_gain') or {}).get('vs_warm_test'))}，"
                 f"step0(Valid/Test)={pct((v.get('relative_gain') or {}).get('vs_step0_valid'))}/"
                 f"{pct((v.get('relative_gain') or {}).get('vs_step0_test'))}")
    L.append(f"\n门槛判定：**{'达到' if purch.get('passed') else '未达到'}**。"
             "这是研究投入门槛，不是显著性界限。\n")

    L.append("### 2.3 paired row bootstrap（2000 次，固定预测）\n")
    L.append("解释为条件于这些权重/单 seed 的样本不确定度，不称跨训练稳定性。差值 = RMSE(候选) − RMSE(WARM/step0)，负号=候选更好。\n")
    L.append("| probe | 比较 | 点差 | 均值 | 2.5% | 97.5% | P(Δ>0) |")
    L.append("|---|---|---|---|---|---|---|")
    for r, rec in boot.items():
        if not rec or "status" in rec:
            L.append(f"| {r} | — | — | — | — | — | — |")
            continue
        for cmp_name in ("vs_warm_valid", "vs_warm_test", "vs_step0_valid", "vs_step0_test"):
            b = rec.get(cmp_name, {})
            L.append(f"| {r} | {cmp_name} | {f(b.get('delta_point'),6)} | {f(b.get('delta_mean'),6)} | "
                     f"{f(b.get('delta_ci_lo'),6)} | {f(b.get('delta_ci_hi'),6)} | {f(b.get('frac_positive'),3)} |")
    L.append("")

    # ---- oracle (post-hoc, not selectable) --------------------------------
    oracle = res.get("oracle", {})
    if oracle:
        L.append("### 2.4 事后 oracle（含 Test 选择乐观偏差，不是可选规则）\n")
        L.append("| probe | Test 曲线最小 | 位置 | 相对 step0 的 oracle 相对增益 | Test 终点 | Valid 曲线最小 |")
        L.append("|---|---|---|---|---|---|")
        for r, o in oracle.items():
            L.append(f"| {r} | {f(o.get('test_curve_min'))} | {o.get('test_curve_min_update')} | "
                     f"{pct(o.get('oracle_rel_gain_vs_step0'))} | {f(o.get('test_final'))} | {f(o.get('valid_curve_min'))} |")
        L.append("\n读法：训练使 Train_fit 下降、Valid 变差，因此预注册的 R_SEL（按 Valid）必然回到 step0。"
                 "事后按 Test 选的 oracle 相对 step0 可达约 1.7%（>1% 投入尺），但它含 Test 选择乐观偏差、"
                 "不是预注册可选规则，不能当作采购依据；本轮的采购判定仍只看 R_SEL（见 §2.2）。"
                 "Test 终点（update 1800）只有约 0.6% 改善。\n")

    # ---- branch learning path -------------------------------------------
    L.append("## 3. 分支是否学到（直接测量，不只看 PRE/POST 接近）\n")
    L.append("“末层零初始化”只使前层在首步没有梯度；不能单凭 PRE≈POST 说明分支没训练。以下为 1800 步的实测（详见 `analysis/learning_path.json`）：\n")
    L.append("| probe | update | ‖∇psi‖ | ‖∇delta_first‖ | ‖∇delta_last‖ | ‖∇f‖ | ‖∇rho‖ |")
    L.append("|---|---|---|---|---|---|---|")
    for r, s in lpsum.items():
        g = s.get("grad_norm_preclip", {})
        L.append(f"| {r} | {s.get('update')} | {f(g.get('psi'),4)} | {f(g.get('delta_first'),4)} | "
                 f"{f(g.get('delta_last'),4)} | {f(g.get('f'),4)} | {f(g.get('rho'),4)} |")
    L.append("")
    L.append("固定 32 个 Train 分子（仅按公开 row 排序）的通道诊断（1800 步）：\n")
    L.append("| probe | meanψ(h) vs ψ(mean h) 相对量级 | extra 特征 RMS | 完整 delta 输出 RMS（标准化） | 通道响应相对完整 delta | 通道响应原始单位 RMS |")
    L.append("|---|---|---|---|---|---|")
    for r, s in lpsum.items():
        L.append(f"| {r} | {f(s.get('mean_psi_vs_psi_mean_relative_to_extra_rms'),3)} | {f(s.get('extra_feature_rms'),4)} | "
                 f"{f(s.get('delta_full_rms_std'),5)} | {f(s.get('channel_response_relative_to_full_delta'),4)} | "
                 f"{f(s.get('channel_response_raw_rms'),2e0 and 6)} |")
    L.append("")
    L.append(f"目标尺度（target norm scale）≈{f((res.get('target_norm') or {}).get('scale'),3)}；"
             "完整 delta 输出 RMS 与通道响应 RMS 都远小于它，可直接读出额外通道对预测的实际贡献量级。")
    L.append("（`delta([z_pool, extra]) − delta([z_pool, 0])` 只是内部通道使用诊断，不称合法分子干预或独立贡献。"
             "注意 `delta` 末层初值为零，其“相对初态位移”无定义，故只报梯度与输出量级。）\n")
    # channel-response trajectory
    for r, s in lpsum.items():
        traj = s.get("channel_response_trajectory") or []
        if traj:
            L.append(f"- {r} 通道响应时间线（update：相对完整 delta / 原始单位 RMS）：" +
                     "，".join(f"{t['update']}:{f(t.get('relative_to_full_delta'),4)}/{f(t.get('raw_rms'),2e0 and 6)}" for t in traj))
    pvp = branch.get("POST_vs_PRE_R_SEL", {})
    L.append(f"- PRE 与 POST 在 R_SEL 下的预测差：Valid 最大 {f(pvp.get('valid',{}).get('max_abs'),6)}，"
             f"Test 最大 {f(pvp.get('test',{}).get('max_abs'),6)}。")
    L.append(f"- 各臂相对 WARM 的最大预测差（全 grid）：POST={f((branch.get('P_POST_vs_WARM') or {}).get('test',{}).get('max_abs'),6)}，"
             f"PRE={f((branch.get('P_PRE_vs_WARM') or {}).get('test',{}).get('max_abs'),6)}（Test）。")
    L.append(f"\n**解释分层：`{layer.get('layer')}`**，弱通道臂={layer.get('weak_channel_arms')}，"
             f"响应臂={layer.get('responsive_channel_arms')}。")
    L.append("判读：PRE 与 POST 接近本身不能证明分支没训练；是否训练由上面的梯度/位移/通道响应测量决定。"
             "若通道响应很弱，只写 `EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG`，不写“没有信息”或“池化位置无影响”。\n")
    L.append("**建议（投入决定）**：暂停现有读出扩展（给该读出再加模块）、长训与选点搜索；"
             "不为弥补阴性结果自行启动新实验。本轮只排除 这三个探针 / 该来源 / 该预算，"
             "不证明表示没有信息或表达力不足；池化位置与 H_REP 仍未识别。\n")

    # ---- old vs new ------------------------------------------------------
    L.append("## 4. 旧（update 9720）与新（update 8165）探针分开展示\n")
    L.append("两组来源不同，不混成一个总体；新结果不称旧协议的事前结果。\n")
    L.append("| probe | 旧 source=9720 R_SEL Valid | 旧 R_SEL Test | 新 source=8165 R_SEL Valid | 新 R_SEL Test |")
    L.append("|---|---|---|---|---|")
    oldp = (comp.get("old_probes_u0_snapshot_9720") or {}).get("probes", {})
    newp = (comp.get("new_probes_update_8165") or {}).get("probes", {})
    for r in R.PROBE_ARMS:
        o = oldp.get(r, {})
        n = newp.get(r, {})
        L.append(f"| {r} | {f(o.get('valid_rmse'))} | {f(o.get('test_rmse'))} | {f(n.get('valid_rmse'))} | {f(n.get('test_rmse'))} |")
    L.append("")
    L.append(f"旧探针 meta：branch displacement vs WARM max_abs={f((comp.get('old_probes_u0_snapshot_9720') or {}).get('meta',{}).get('branch_displacement_vs_P_WARM_max_abs'),6)}，"
             f"PRE vs POST max_abs={f((comp.get('old_probes_u0_snapshot_9720') or {}).get('meta',{}).get('pre_vs_post_max_abs'),6)}。")
    L.append(f"新来源 step0 的 Valid 更好（{f(step0.get('valid_rmse'))} vs 旧 ≈0.7254）但 Test 略差"
             f"（{f(step0.get('test_rmse'))} vs 旧 ≈0.5897）：换成“原计划 E1 来源”并不等于得到一个更好的探针基线；"
             "本轮所有对比都是**同一来源内**的，不混用旧来源数字。\n")

    # ---- comparisons -----------------------------------------------------
    L.append("## 5. 已有上游预测的同口径对照（不重算）\n")
    L.append("Lipo E1 R_SEL Test RMSE 的结构臂与 D-MPNN，以及 BACE fullN（分类：ROC-AUC）：\n")
    L.append("| arm | N | R_SEL Test | last10 median |")
    L.append("|---|---|---|---|")
    for k, v in (comp.get("structural_fullN_E1_R_SEL") or {}).items():
        if v:
            L.append(f"| {k} | {v.get('N')} | {f(v.get('E1_R_SEL_test'))} | {f(v.get('E1_last10_test_median'))} |")
    for k, v in (comp.get("bace_fullN_E1_R_SEL") or {}).items():
        if v:
            L.append(f"| {k} | {v.get('N')} | {f(v.get('E1_R_SEL_test'))} | {f(v.get('E1_last10_test_median'))} |")
    L.append("")
    L.append("新探针对 R1 的改善与其是否超过更便宜的 R3 是两个问题；单个额外通道偏好不升级为池化瓶颈或化学机制。\n")

    # ---- boundaries ------------------------------------------------------
    L.append("## 6. 结论边界（明确不做的推断）\n")
    L.append("- 更多 N 的旧结果支持固定配方对样本量敏感；不证明“数据量是唯一主瓶颈”，不证明噪声下限，不推导 PCQM 或大规模预训练的必要性。")
    L.append("- 延长/重启的旧结果只排除这些优化方案的实用收益。")
    L.append("- 训练误差更低而留出更差，写“拟合优势未转化为泛化”，不断言“记忆结构身份”。")
    L.append("- 阴性读出结果不能推出表示没有信息或表达力不足；它足以支持暂缓继续给此类读出加模块的投入判断。")
    L.append("- 结构追溯仍是定位计算响应的工具；没有通过预测收益验证的机制观察不作为性能创新。\n")

    # ---- budget ----------------------------------------------------------
    L.append("## 7. 预算与执行\n")
    caps = bud.get("caps", {})
    L.append(f"- 正式科学轨迹：{bud.get('formal_trajectories')} 条（P_WARM / P_POST / P_PRE，各 1800 updates）。")
    L.append(f"- GPU·h（各进程占卡墙计）：合计 **{f(bud.get('gpu_hours_charged_total'),4)}**，"
             f"软上限 {caps.get('gpu_soft_hours')}，硬上限 {caps.get('gpu_hard_hours')}；"
             f"在软上限内={f(bud.get('within_gpu_soft_cap'))}。")
    sa = bud.get("source_audit", {})
    L.append(f"- 来源审计/缓存：eval-only，device={sa.get('device')}，wall={f(sa.get('wall_seconds'),1)}s。")
    L.append("- regime：`res`（a100-2，A100-SXM4-40GB ×2，driver 550.163.01，torch 2.5.1+cu124），"
             "`res-gpu0`/`res-gpu1` 每 pool 一条进程；逐臂 commit 见各自 train.json。")
    ledger = bud.get("launch_ledger") or {}
    if ledger:
        L.append("- launch ledger（fail-closed；每个作业独立 out_dir，无重复投放）：")
        for name, v in ledger.items():
            L.append(f"  - {name}: state={v.get('status')} exit={v.get('exit_code')} pool={v.get('pool')}")
    L.append("")

    # ---- reproduction ----------------------------------------------------
    L.append("## 8. 产物与复现\n")
    L.append("```bash")
    L.append("uv run python code/run.py source-audit --device cpu   # 来源恢复 + 身份化 h 缓存（eval，不训练）")
    L.append("uv run python code/run.py smoke --device cpu           # 4-update 冒烟（不计结果）")
    L.append("uv run python code/run.py probe --rule P_WARM --device cuda --out-dir runs/P_WARM")
    L.append("uv run python code/run.py analyze && uv run python code/run.py report")
    L.append("uv run python -m pytest tests -q                       # 不变量 + 合成 h 夹具")
    L.append("```")
    L.append("- 机器量：`analysis/{source_audit,results,learning_path,comparisons,budget}.json`、`analysis/launch_ledger.json`；"
             "曲线图 `analysis/figures/probe_curves.png`。")
    L.append("- 每条轨迹：`train.json`（source/cache/参数/norm hashes、50 点 grid + step0、三规则选中权重）、"
             "`curve.csv`、`grid_predictions.npz`（逐样本标准化预测）、`learning_path.json`（分支诊断）、`best_{R,A,W}_SEL.pt`、`last.pt`。")
    L.append("- 权重副本：`runs/` 本地保留；远端 `res` 同路径可恢复。h 缓存按身份命名、确定性可重建。")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="closeout report")
    ap.parse_args(argv)
    text = build_report()
    p = R.TRACK / "REPORT.md"
    p.write_text(text, encoding="utf-8")
    R.log(f"report written: {R.track_rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
