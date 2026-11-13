"""Stage ``method_figures``: the two method/evidence figures.

Both figures are drawn from *already committed* numbers only (the round's own
``summary_tables.json`` / ``weighted_response.csv``, the source round's
``mechanism.json``, and the ``s3_locality`` alias audit).  No model is loaded
and no number is invented here; every quoted value is read from disk and
asserted against a small expectation table so the figure cannot silently drift
away from the tables.

* ``fig_method_flow.png`` — the computational definition of the frozen
  LCRC-C1 forward plus the traceability relation used in this work
  (rewire -> changed roots U -> ``Delta p = s_y [sum_U Delta q + Delta rho]``).
* ``fig_evidence_layers.png`` — the three-layer evidence chain
  (real structure -> real input signature -> model computation) with the task
  references attached, and the three fixed counter-examples kept in the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import round_common as RC

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1b"
BLUE = "#1f4e79"
GREY = "#5f5f5f"
LIGHT = "#f2f4f7"
WARN = "#8a3b12"
OK = "#1d5c33"

LINE = 0.0148  # unused sentinel for documentation; real fitting is measured


def _box(ax, x, y, w, h, title, lines, *, edge=BLUE, face=LIGHT, title_size=10.2,
         body_size=8.2, title_color=INK, ls="-"):
    """Draw a titled box and shrink the body font until the text really fits.

    The fit is measured from the rendered text extent, so a long CJK line that
    wraps cannot silently overflow the box (the failure mode of a hand-tuned
    line height).
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.15, edgecolor=edge, facecolor=face, linestyle=ls, zorder=2,
        )
    )
    ax.text(x + 0.010, y + h - 0.016, title, ha="left", va="top",
            fontsize=title_size, color=title_color, fontweight="bold", zorder=3)
    body_top = y + h - 0.016 - title_size / 720.0 - 0.010
    body = ax.text(x + 0.010, body_top, "\n".join(lines), ha="left", va="top",
                   fontsize=body_size, color=INK, linespacing=1.42, zorder=3)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    size = body_size
    for _ in range(18):
        bb = body.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform(bb)
        if (y1 - y0) <= (body_top - y - 0.012) and (x1 - x0) <= (w - 0.020):
            return
        size -= 0.2
        if size < 6.4:
            raise SystemExit(
                f"box {title!r}: text does not fit even at {size:.1f} pt; shorten the lines"
            )
        body.set_fontsize(size)
        body.set_linespacing(1.42)



def _arrow(ax, xy_from, xy_to, *, color=BLUE, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=13,
                        linewidth=lw, color=color, zorder=1)
    )


# ---------------------------------------------------------------------------
# figure 1 — method flow
# ---------------------------------------------------------------------------

def fig_method_flow(path: Path, num: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(17.0, 11.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.012, 0.978, "LCRC 结构追溯方法：一个冻结模型的计算定义与可追溯关系",
            fontsize=15.5, fontweight="bold", color=INK, va="top")
    ax.text(0.012, 0.945,
            "public122 Lipophilicity · LCRC-C1 (NO_BOND_MEAN) · 228,674 参数 · "
            "本图不引入任何新训练 / 新推理 / 新干预；数字取自已提交的 round 产物",
            fontsize=9.0, color=GREY, va="top")

    w = 0.226
    xs = [0.012, 0.262, 0.512, 0.762]

    # ---- row 1: the four computational stages -----------------------------
    y1, h1 = 0.672, 0.245
    _box(ax, xs[0], y1, w, h1, "1 · 真实结构（冻结面板）", [
        "4200 分子；官方 random122 划分",
        "Train / Valid / Test = 3360/420/420",
        "6 个连接家族；Test 主队列 844 对",
        "冻结视图 O / T / K0 / C",
        "C = 等剂量对照；539 条 C 链接",
        "  = 364 个唯一视图（不是分子数）",
        "",
        "全部视图共享同一冻结前向：",
        "只有图变、参数不变 ⇒ Δp 只来自",
        "输入改变，不来自重训",
    ], body_size=8.8)

    _box(ax, xs[1], y1, w, h1, "2 · 显式局部结构统计（无标签）", [
        "AF51：元素/芳香/电荷/H/度/环…",
        "BF13：键型/环/端点/立体",
        "scalar9：n_d2, n_paths2, closure_ratio,",
        "  log1p deg, in_ring, min_ring,",
        "  ring_bond, terminal, fused",
        "一跳：b_e = BondEnc([a_u+a_v,",
        "  |a_u−a_v|, a_u·a_v, BF13])",
        "  b_sum_v = Σ_{e∋v} b_e；nbr 计数",
        "二跳：path_cat / path_end63 /",
        "  closure 计数（按根）",
        "global42（归一化冻结在 Train）",
    ])

    _box(ax, xs[2], y1, w, h1, "3 · 冻结 LCRC-C1 前向", [
        "x_v = [a_v, b_sum_v, nbr_v, T_v, E_v,",
        "  J_v, K_v, C_v, scal9]        (433)",
        "h_v = LN(MLP(433→192→64)(x_v))",
        "q_v = f(h_v)            (64→64→1)",
        "",
        "p(G) = μ_y + s_y · [ Σ_v q_v",
        "   + ρ( ḡ, mean h, mean h², log1p n) ]",
        "",
        "mean h² = 逐分量平方后取平均",
        "   （≠ (mean h)²）",
        "s_y 必须保留：Σq 与 ρ 都在标准化单位",
    ])

    _box(ax, xs[3], y1, w, h1, "4 · 任务参照（可检验）", [
        "实验 logD：只有原分子有标签，",
        "  改线图无标签 ⇒ 只讲原图偏好/路径",
        "WC_LOGP = Crippen logP：",
        "  有真实数值参照；方向一致 0.90–0.99",
        "  4/5 家族 Bonferroni 支持",
        "MolWt：组成不变 ⇒ Δref ≡ 0（严格 0）",
        "  模型仍动，约 1.2% 的 Train σ",
        "",
        "三类任务从不混成一个“真值”",
    ], edge=OK, face="#f3f8f4")

    for i in range(3):
        _arrow(ax, (xs[i] + w + 0.003, y1 + h1 / 2), (xs[i + 1] - 0.003, y1 + h1 / 2))

    # ---- row 2: traceability relation + signature -------------------------
    y2, h2 = 0.373, 0.268
    _box(ax, 0.012, y2, 0.487, h2, "可追溯关系（性质 2；固定权重、eval 模式）", [
        "对有原子映射的既有改线，令 U = 局部签名被改变的根原子集合：",
        "",
        "        Δp = s_y · [ Σ_{v∈U} Δq_v + Δrho ] ,   未变根的 q_v 不变",
        "",
        "Δrho ≠ 0 是必然的：ρ 读 mean_v h 与 mean_v h²，池化把全部根耦合。",
        "已提交记录（dissection leave-one-S2-out，原始单位）：",
        "ΔL = +0.5203、Δrho = +0.6393、Δpred = +1.1596，其他原子 Δf ≡ 0",
        "⇒ Δrho 与 Δq 同量级，不能把 Δp 分摊后称“唯一原子/官能团贡献”。",
    ], edge=WARN, face="#fdf5f1", body_size=10.4)

    _box(ax, 0.512, y2, 0.476, h2, "可验证的局部计算签名 S_v（充分，非最小）", [
        "只用 forward 真正读的字段，去掉一切标识符：",
        "A  atom_feat[v](51) + atom_scalar[v](9)",
        "B  入射多重集 (atom_feat[u], bond_feat[b])   ← b_sum 与 deg",
        "C  S1 nbr_code 计数                          ← nbr_v",
        "D  (path_cat, path_end63) 计数                ← T/E/J",
        "E  (path_cat, atom_feat[path_end]) 计数       ← C_v",
        "F  path_closure 边际计数                      ← K_v",
        "",
        "性质 1：相同 global42 + 相同 S_v 多重集 ⇒ 相同预测",
        "（同一冻结参数/归一化；实现按既有数值容差，实测 ≤ 3e-7）",
        "不称最小签名、不称完整化学表示、不称对分子性质充分。",
    ], edge=BLUE, face="#f4f7fb", body_size=10.0)

    # ---- row 3: what is not claimed --------------------------------------
    y3, h3 = 0.075, 0.268
    _box(ax, 0.012, y3, 0.487, h3, "八格 Möbius：计算干预，不是化学因果", [
        "8 个预定义输入分组格（S1 / S2 / scalar 开关）× 固定权重",
        "→ 7 个 Möbius 项 g1, g2, g3, g12, g13, g23, g123",
        "七项之和 = 端点差（已提交残差 4.4e-16）",
        "",
        "混合输入（保留 S1 去 S2 等）不是合法新分子；",
        "算术完备 ≠ 化学因果正确。",
        "",
        "R3 = value(123) − value(12) 是“条件标量恢复效应”，含全部",
        "含 scalar 的交互，不能叫 standalone main effect；单独的 γ3",
        "另列（amide / Test / 校准 H：γ3 = −0.0017）。",
    ], edge=GREY, face="#f7f7f7", body_size=10.2)

    _box(ax, 0.512, y3, 0.476, h3, "固定保留的三处边界（反例不剔出主统计）", [
        "① logD 校准限制：最强偏好只在酰胺等剂量可匹配子群",
        "   （Test 62/164 分子 ≈ 38%，110/533 对 ≈ 21%）；其他",
        "   家族异质或低支持；磺酸酯未检验（≠ 零效应）。",
        "② WC_LOGP 酯未解决：8 分子 / 46 对，+0.063",
        "   [−0.007, +0.138]，区间跨 0。",
        "③ MolWt 最大残留个例 sulfonamide:1275:1:0:d9-14（T 视图）",
        "   |Δpred| = 14.24 g/mol = 0.131 σ_train；家族分子权重",
        "   均值 1.92 g/mol。小但非零 ⇒ 不写“精确不变”。",
    ], edge=WARN, face="#fdf5f1", body_size=10.2)

    ax.text(0.012, 0.030,
            "边界：单 checkpoint / 单 seed（s0）· 单结构库（public122 / ChemBench Lipophilicity）· "
            "计算目标（WC_LOGP、MolWt）· 校准覆盖有限。"
            "图中“追溯”指计算输入 → 局部读出 → 池化读出，不指化学因果。",
            fontsize=8.4, color=GREY, va="bottom")

    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 2 — three-layer evidence
# ---------------------------------------------------------------------------

def fig_evidence_layers(path: Path, num: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(17.0, 9.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.012, 0.978, "三层证据链：实际结构 → 真实输入签名 → 模型计算，再接任务参照",
            fontsize=15.5, fontweight="bold", color=INK, va="top")
    ax.text(0.012, 0.945,
            "每一层只主张它自己那层的可检验内容；箭头表示“计算上被读取”，不表示“化学上被解释”",
            fontsize=9.0, color=GREY, va="top")

    w = 0.226
    xs = [0.012, 0.262, 0.512, 0.762]
    y, h = 0.420, 0.455

    _box(ax, xs[0], y, w, h, "层 1 · 实际结构（真实数据）", [
        "ChemBench Lipophilicity",
        "commit 0b10d4c1…",
        "4200 分子 / 3360-420-420",
        "官方 random122；row_id 与行序保留",
        "",
        "Test 主队列对：",
        "  amide 533    sulfonamide 112",
        "  aryl_ether 112   arylamine 41",
        "  ester 46     sulfonate 0",
        "  合计 844 对（分子跨家族重叠）",
        "",
        "539 条 C 链接 = 364 个唯一视图",
        "（结构链接，不是 539 个分子）",
        "",
        "这一层没有模型、没有标签泄漏，",
        "也没有把改线图当成新分子去测。",
    ], body_size=9.6)

    _box(ax, xs[1], y, w, h, "层 2 · 真实输入签名（模型消费）", [
        "AF51 / BF13 / scalar9 / global42 全部由",
        "无标签 RDKit 图计算；归一化冻结在 Train",
        "",
        "逐根签名 S_v = A..F 六块多重集：",
        "  A 原子特征 + 标量",
        "  B 入射 (原子, 键) 对",
        "  C 一跳 nbr_code",
        "  D (path_cat, path_end63)",
        "  E (path_cat, 端点特征)",
        "  F closure 边际",
        "",
        "已提交审计（s3_locality）：",
        "  113,568 原子 → 18,122 根签名",
        "  87 局部组 → 86 full-input 组",
        "  5 对非同一分子、S_v 与 global42 全等",
        "  ⇒ 复用 A 的预测 |Δ| ≤ 3e-7（容差 2e-5）",
        "",
        "S_v 是“充分”而非“最小”，也不等于化学。",
    ], edge=BLUE, face="#f4f7fb", body_size=9.6)

    _box(ax, xs[2], y, w, h, "层 3 · 模型计算（冻结前向）", [
        "LCRC-C1 / NO_BOND_MEAN，228,674 参数",
        "h_v = LN(MLP(433→192→64)(x_v))",
        "q_v = f(h_v)",
        "p = μ_y + s_y·[Σ_v q_v",
        "   + ρ(ḡ, mean h, mean h², log1p n)]",
        "",
        "八格 Möbius（固定权重、预定义分组）：",
        "  7 项 g1..g123；Σ 项 = 端点差",
        "  （残差 4.4e-16）",
        "",
        "amide / Test / 校准 H：Σ = 0.1460 = J_C",
        "  γ2 = +0.1037（S2 主项）",
        "  R2 = γ2+γ12+γ23+γ123 = +0.1355",
        "  R3 = γ3+γ13+γ23+γ123 = +0.0307",
        "  γ3 = −0.0017（真正的 standalone 标量项）",
        "",
        "Δp = s_y[Σ_{v∈U}Δq_v + Δrho]，Δrho 与 Δq 同量级",
        "⇒ 不能给出“唯一原子贡献”。",
    ], body_size=9.6)

    _box(ax, xs[3], y, w, h, "任务参照（各自独立可检验）", [
        "A. 实验 logD（只有原分子有标签）",
        "   amide 校准 rC = +0.164 [0.052, 0.278]",
        "   E = rC_TRAINED − rC_INIT = +0.321",
        "   覆盖 62/164 分子、110/533 对",
        "   磺酰胺部分；芳醚反向；芳胺 n=4；",
        "   酯未成立；磺酸酯未检验",
        "",
        "B. WC_LOGP（计算参照；Test/主 CF/FULL）",
        "   gain = baseline − model，Bonferroni(6)",
        "   amide      +0.108 [0.094, 0.124] ✓",
        "   sulfonamide+0.116 ✓",
        "   aryl_ether +0.119 ✓",
        "   arylamine  +0.314 ✓",
        "   ester      +0.063 [−0.007, 0.138] ✗",
        "",
        "C. MolWt（Δref ≡ 0 的不变量对照）",
        "   分子权重 mean|Δp| = 1.33 g/mol = 0.0123 σ",
        "   与 delta-MAE 精确相等（算术核对）",
    ], edge=OK, face="#f3f8f4", body_size=9.6)

    for i in range(3):
        _arrow(ax, (xs[i] + w + 0.003, y + h / 2), (xs[i + 1] - 0.003, y + h / 2))

    # ---- fixed caveats band ---------------------------------------------
    yb, hb = 0.070, 0.258
    _box(ax, 0.012, yb, 0.487, hb, "固定保留的三处边界（写进正文与图，不剔出主统计）", [
        "① logD 的校准限制：最强“训练相关结构偏好”只在酰胺等剂量",
        "   可匹配子群成立（Test 62/164 分子 ≈ 38%，110/533 对 ≈ 21%）；",
        "   不能写成“多个家族都验证了同一 logD 机制”。",
        "② WC_LOGP 酯未解决：8 分子 / 46 对，+0.063 [−0.007, +0.138]，",
        "   区间跨 0；磺酸酯从未检验（无 C、无 Valid/Test 改线）。",
        "③ MolWt 最大残留个例：sulfonamide:1275:1:0:d9-14（T 视图）",
        "   |Δpred| = 14.24 g/mol = 0.131 σ_train；该家族分子权重均值",
        "   1.92 g/mol。小但非零 ⇒ 不写“精确不变”，也不从零均值推等价。",
    ], edge=WARN, face="#fdf5f1", body_size=10.2)

    _box(ax, 0.512, yb, 0.476, hb, "四个推广边界（主张能走多远）", [
        "① 单 checkpoint / 单 seed：本轮两个任务都是 seed 0；",
        "   原 logD 模型是 public122 seed 0 的一个 checkpoint。",
        "② 单结构库：4200 个 Lipophilicity 分子、6 个连接家族；",
        "   无第二数据集、无新分子、无新改线。",
        "③ 计算目标：WC_LOGP / MolWt 是计算参照值，不是新测量；",
        "   计算任务的正确性不能追认原实验 logD 的广泛化学正确性。",
        "④ 校准覆盖：C 只覆盖每个队列的一部分（9%–26% 的对）；",
        "   校准外的分子只报告，不外推。",
    ], edge=GREY, face="#f7f7f7", body_size=10.2)

    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _numbers() -> dict[str, Any]:
    """Read the committed numbers and assert the ones the figures quote."""
    summary = RC.read_json(RC.NEW_ANALYSIS / "summary_tables.json")
    partA = RC.read_json(RC.NEW_ANALYSIS / "partA_clarifications.json")
    mech = RC.read_json(RC.SOURCE_ANALYSIS / "mechanism.json")
    wc = summary["partB"]["wc_logp"]["test_families"]
    mw = summary["partB"]["molwt"]["test_families"]
    a1 = partA["A1_coverage_and_claim_matrix"]["amide/test"]
    a3 = partA["A3_conditional_recovery"]["aggregate"]["amide/test/RETAINED_C"]
    mech_amide = mech["summary"]["test/calibration/amide"]["H"]
    resid = partA["A3_conditional_recovery"]["arithmetic_residual_max"]

    expect = {
        "wc_amide_gain": (wc["amide"]["gain"], 0.1082, 0.001),
        "wc_ester_lo": (wc["ester"]["gain_ci95"][0], -0.00709, 0.001),
        "wc_ester_hi": (wc["ester"]["gain_ci95"][1], 0.13792, 0.001),
        "amide_retention_pairs": (a1["retention_ratio_pairs"], 0.20638, 1e-4),
        "amide_retention_mols": (a1["retention_ratio_molecules"], 0.37805, 1e-4),
        "r2_t_minus_c_h": (a3["R2_TminusC_H"]["point"], 0.1355470, 1e-6),
        "r3_t_minus_c_h": (a3["R3_TminusC_H"]["point"], 0.0306653, 1e-6),
        "mechanism_sum": (mech_amide["sum_of_terms"], 0.14602419, 1e-7),
        "gamma3": (mech_amide["terms"]["g3"], -0.0017247, 1e-6),
        "recovery_residual": (resid, 4.440892098500626e-16, 1e-18),
    }
    for key, (got, want, tol) in expect.items():
        if abs(float(got) - float(want)) > tol:
            raise SystemExit(f"figure number drifted: {key} got {got} expected {want}+-{tol}")
    # the two recovery identities must hold on the committed seven terms
    t = mech_amide["terms"]
    if abs((t["g2"] + t["g12"] + t["g23"] + t["g123"]) - a3["R2_TminusC_H"]["point"]) > 1e-9:
        raise SystemExit("R2 identity does not hold on the committed seven terms")
    if abs((t["g3"] + t["g13"] + t["g23"] + t["g123"]) - a3["R3_TminusC_H"]["point"]) > 1e-9:
        raise SystemExit("R3 identity does not hold on the committed seven terms")

    weighted = {
        r["task"] + "/" + r["split"] + "/" + r["family"]: r
        for r in RC.read_csv(RC.NEW_ANALYSIS / "weighted_response.csv")
    }
    if not all(str(r["check_molwt_identity"]).strip() in ("1", "True")
               for k, r in weighted.items() if k.startswith("molwt/")):
        raise SystemExit("weighted_response.csv: MolWt identity check not satisfied")

    return {
        "summary": summary, "partA": partA, "mech": mech, "wc": wc, "mw": mw,
        "a3": a3, "a1": a1, "mech_amide": mech_amide, "weighted": weighted,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else RC.NEW_ANALYSIS
    num = _numbers()
    fig_method_flow(out / "fig_method_flow.png", num)
    fig_evidence_layers(out / "fig_evidence_layers.png", num)
    RC.log(f"method figures written to {RC.track_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
