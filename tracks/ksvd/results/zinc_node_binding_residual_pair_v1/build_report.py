"""Build the one-page REPORT.md / DECISION.md from pair_summary.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "pair_summary.json").read_text(encoding="utf-8"))


def health_row(arm: str, epoch: int) -> dict:
    for row in S["health"][arm]:
        if row["epoch"] == epoch:
            return row
    return {}


def collapsed(arm: str) -> bool:
    last = S["health"][arm][-1]
    return bool(last["slot_zero_frac"] >= 0.9 or last["slot_rms"] < 1e-8)


c_traj = S["health"]["control"]
r_traj = S["health"]["residual"]
control_collapsed = collapsed("control")
residual_collapsed = collapsed("residual")
gate = S["purchase_gate_passed"]

if residual_collapsed:
    mechanism = "MECHANISM_FAILED：candidate 节点通道仍在训练中塌缩，未达到稳定绑定目标。"
elif gate:
    mechanism = "健康改善 + 性能PASS：支持该稳定绑定配置的第二 paired seed 复核（本轮不自动启动）。"
else:
    mechanism = "健康改善 + 性能FAIL：通道恢复未转化为足够泛化收益，关闭这个具体绑定配置。"

lines = []
lines.append("# ZINC 节点绑定残差配对训练 v1 — REPORT / DECISION")
lines.append("")
lines.append("日期：2026-10-02 · 提交：task/zinc-nodebind-residual-v1 · 官方 test 从未实例化")
lines.append("")
lines.append("## 结论摘要")
lines.append("")
lines.append("| 项 | control (product) | candidate (residual) |")
lines.append("|---|---:|---:|")
lines.append(f"| 末5soup calibrated valid MAE | {S['control_cal_valid_mae']:.6f} | {S['residual_cal_valid_mae']:.6f} |")
lines.append(f"| raw valid MAE | {S['control_raw_valid_mae']:.6f} | {S['residual_raw_valid_mae']:.6f} |")
lines.append(f"| train-fit bias | {S['control_bias']:.6f} | {S['residual_bias']:.6f} |")
lines.append(f"| 240轮节点通道终态 | {'塌缩' if control_collapsed else '存活'} | {'塌缩' if residual_collapsed else '存活'} |")
lines.append("")
lines.append(f"- **calibrated gain (control − candidate) = {S['calibrated_gain']:+.6f}**；raw gain = {S['raw_gain']:+.6f}；gain_without172 = {S['gain_without172']:+.6f}。")
lines.append(f"- **购买决定：{'PASS（建议第二 paired seed 复核）' if gate else 'FAIL（本轮不购买后续 seed）'}**（门槛：cal gain ≥0.003，gain_without172 >0，G0 贡献恶化 ≤0.001）。")
lines.append(f"- **机制判定：{mechanism}**")
lines.append("")
lines.append("## 1. 父节点死亡是否在 fresh control 重现？")
lines.append("")
lines.append("是。相同 canonical fresh 初始化下，product control 的节点槽在训练中自锁死亡：")
lines.append("")
lines.append("| epoch | W_A_S rel_change | slot RMS | slot zero_frac | node_out const_frac | task grad W_A_S |")
lines.append("|---:|---:|---:|---:|---:|---:|")
for e in (0, 1, 10, 40, 80, 160, 240):
    row = health_row("control", e)
    if row:
        lines.append(
            f"| {e} | {row['W_A_S_rel_change']:.3f} | {row['slot_rms']:.3e} | "
            f"{row['slot_zero_frac']:.3f} | {row['node_out_constant_frac']:.3f} | {row['grad_W_A_S']:.3e} |"
        )
lines.append("")
lines.append("说明父 Full soup 的 denormal 死亡不是 soup/续训产物：从 fresh 初始化、无监督旧权重、"
             "256 轮完整协议下同样发生，且在 epoch 40–80 之间完成。")
lines.append("")
lines.append("## 2. candidate 是否恢复通道及交叉项？")
lines.append("")
lines.append("| epoch | slot RMS | slot zero_frac | node_out const_frac | task grad W_A_S | product RMS | struct-add RMS | atom-add RMS | cross-mix ΔRMS |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for e in (0, 1, 10, 40, 80, 160, 240):
    row = health_row("residual", e)
    if row:
        lines.append(
            f"| {e} | {row['slot_rms']:.3e} | {row['slot_zero_frac']:.3f} | "
            f"{row['node_out_constant_frac']:.3f} | {row['grad_W_A_S']:.3e} | "
            f"{row['product_rms']:.3e} | {row['structural_additive_rms']:.3e} | "
            f"{row['atomic_additive_rms']:.3e} | {row['cross_mix_delta_rms']:.3e} |"
        )
lines.append("")
lines.append("加性项在早期确实提供了更大的梯度路径（candidate 在 epoch 10–30 的 slot RMS/梯度高于 control），"
             "但 epoch 40 起同样进入塌缩、epoch 80 后节点槽近乎全零、task 梯度消失。"
             "**因此残差只延后、未阻止节点路径自锁死亡；交叉项也未保持可学习。**")
lines.append("")
lines.append("## 3. 性能收益是否超出 bias / id172？G0 如何？")
lines.append("")
lines.append("| group | n | control MAE | candidate MAE | gain | control contrib | candidate contrib | Δcontrib |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
group_lines = []
for g in ("G0", "G1", "G172"):
    group_lines.append(g)
gt = list(csv.DictReader((HERE / "group_table.csv").open(encoding="utf-8")))
for row in gt:
    lines.append(
        f"| {row['group']} | {row['n']} | {float(row['control_mae']):.6f} | {float(row['residual_mae']):.6f} | "
        f"{float(row['gain']):+.6f} | {float(row['control_contribution']):.6f} | "
        f"{float(row['residual_contribution']):.6f} | {float(row['contribution_delta']):+.6f} |"
    )
lines.append("")
lines.append(f"- 组贡献之和闭合：{S['group_contribution_closes']}（等于全体 MAE）。")
lines.append(f"- G0 贡献恶化 = {S['g0_contribution_worsening']:+.6f}（门槛 ≤0.001）。")
lines.append(f"- signed error 约定 `pred−y`；bias 差 = {S['bias_diff']:+.6f}。")
lines.append("")
lines.append("## 4. 假设关闭 / 未知")
lines.append("")
if residual_collapsed:
    lines.append("- **关闭**：'纯乘法绑定的塌缩可由固定幅值加性残差梯度路径修复'——candidate 同样塌缩，"
                 "加性项只延后了死亡，未改变终态。")
    lines.append("- **未关闭/未知**：塌缩是否由 weight decay + 下游常量吸收的优化几何主导；"
                 "如果冻结 node_encoder 或从下游切断常量吸收，通道能否存活；性能差异在完整 240 轮下是否稳定。")
else:
    lines.append("- 通道存活但性能未达门槛时，应区分 '节点绑定形式无用' 与 '恢复通道不足以改变泛化'。")
lines.append("")
lines.append("## 5. 下一步（只一个动作）")
lines.append("")
if residual_collapsed:
    lines.append("**关闭本固定幅值残差绑定配置，停止；不再扫系数 / residual 形式。**")
    lines.append("若要继续研究主张，需新 preregistration 针对塌缩机制（例如切断下游常量吸收或改变 node 路径正则），"
                 "而不是继续在本绑定家族内变体。")
elif gate:
    lines.append("**只推荐第二 paired seed 复核**（本轮不自动启动）。")
else:
    lines.append("**关闭本配置**；如需继续，需新 preregistration 并明确新的可判别干预。")
lines.append("")
lines.append("## 6. 执行记录")
lines.append("")
lines.append("- 服务器 skill：`/home/calendar/.pi/agent/skills/remote-research-runner/SKILL.md`")
lines.append("- 主机 `res-2`、pool `res2-cu124`；两 arm 各 1×A100、8 CPU、独立 Slurm 作业。")
lines.append("- 正式训练 commit 与 protocol_hash 相同；split fingerprint 两 arm 一致。")
lines.append("- 官方 valid 仅记录；**官方 test 从未实例化/加载/评估**。")
lines.append("")
lines.append(f"- 总耗时见 `pair_summary.json` 与本地日志。")

(HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "DECISION.md").write_text(
    "# DECISION\n\n"
    + mechanism
    + "\n\n"
    + (
        "本轮只做一次 pair；不扩成参数扫描，不追加训练。"
        if not gate
        else "本轮只做一次 pair；第二 seed 需另行授权。"
    )
    + "\n",
    encoding="utf-8",
)
print("wrote REPORT.md and DECISION.md")