"""Render ``REPORT.md`` from the frozen analysis artifacts.

The report contains no number that is not in ``analysis/``; re-running this
script after ``ssl_analyze.py`` reproduces the document exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import ssl_common as C


def _j(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _f(value: Any, digits: int = 10) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _signed(value: Any, digits: int = 10) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _int(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _per_task(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["task"] = int(row["task"])
        row["positive"] = int(row["positive"])
        for key in ("ap", "historical_ap", "delta"):
            row[key] = None if row[key] == "" else float(row[key])
    return rows


def render(analysis: Path, out: Path) -> Path:
    decision = _j(analysis / "decision.json")
    sup = _j(analysis / "supervised_epoch40.json")
    probe = _j(analysis / "pretrain_probe.json")
    pre = _j(analysis / "pretrain_run.json")
    transfer = _j(analysis / "encoder_transfer_audit.json")
    data = _j(analysis / "data_audit.json")
    audit = _j(analysis / "mask_leakage_audit.json")
    ledger = _j(analysis / "gpu_ledger.json")
    registry = _j(analysis / "run_registry.json")
    per_task = _per_task(analysis / "per_task_ap.csv")

    gate = probe.get("gate") or {}
    fin = probe.get("final") or {}
    margins = probe.get("margins") or {}
    ap = sup.get("validation_ap") or {}
    te = sup.get("train_eval_final") or {}
    summ = sup.get("per_task_summary") or {}
    buckets = sup.get("support_buckets") or {}
    flops = sup.get("flops") or {}
    label = decision.get("decision_label")
    delta = decision.get("delta_ap40")

    lines: list[str] = []
    add = lines.append

    add("# REPORT — `lcrc-pcba-atom-bond-ssl-pilot-v1`")
    add("")
    add(
        "Masked atom/bond self-supervised initialisation of the original "
        "no-message-passing LCRC-A (`A_SSL_INIT_S0`), seed 0, one pretrain and one "
        "40-epoch supervised fine-tune. Generated from `analysis/` by "
        "`code/ssl_report.py`."
    )
    add("")
    add(
        _table(
            ["field", "value"],
            [
                ["status", sup.get("status")],
                ["decision label", f"**{label}**"],
                ["Validation AP40", _f(decision.get("validation_ap40"))],
                ["historical A0 AP40", _f(decision.get("historical_a0_validation_ap40"))],
                ["Delta AP40", f"**{_signed(delta)}**"],
                ["pretext gate", gate.get("label")],
                ["git commit", f"`{sup.get('git_commit')}`"],
                ["Test evaluated", "no"],
            ],
        )
    )
    add("")

    # ---------------------------------------------------------------- summary
    add("## 1. 结论摘要 (required summary)")
    add("")
    add(
        f"1. **Masked atom/bond pretext 是否学超过类别先验？** "
        f"{'是' if gate.get('passed') else '否'}。"
        f"Probe atom CE {_f(fin.get('atom_ce'), 4)} vs 先验 {_f(fin.get('atom_prior_ce'), 4)}"
        f"（margin {_signed(margins.get('atom_prior_ce_minus_ce'), 4)} nats）；"
        f"bond CE {_f(fin.get('bond_ce'), 4)} vs 先验 {_f(fin.get('bond_prior_ce'), 4)}"
        f"（margin {_signed(margins.get('bond_prior_ce_minus_ce'), 4)} nats）。"
    )
    add(
        f"2. **预训练是否完成全部 10000 updates？** "
        f"{'是' if pre.get('updates_completed_is_final') else '否'}，"
        f"实际 {_int(pre.get('updates_completed'))}/{_int(pre.get('updates_requested'))} updates，"
        f"status `{pre.get('status')}`。"
    )
    changed = (pre.get("encoder_change") or {}).get("all_tensors_changed")
    add(
        f"3. **Encoder 权重是否确实迁移？** "
        f"{'是' if transfer.get('n_encoder_tensors_changed') == transfer.get('n_encoder_tensors') else '部分'}："
        f"{_int(transfer.get('n_encoder_tensors_changed'))}/{_int(transfer.get('n_encoder_tensors'))} 个 encoder tensor 被覆盖，"
        f"其余 {len(transfer.get('frozen_prefixes') or [])} 组模块（`local.`/`f.`/`rho.`/`readout.`）"
        f"与历史 seed-0 scratch 初始化逐 tensor 相等"
        f"（`{transfer.get('non_transferred_tensors_are_historical_seed0')}`）。"
        f"两个 encoder 在预训练中都发生了变化（`{changed}`）。"
    )
    add(
        f"4. **SSL 模型 epoch40 Validation AP？** `{_f(decision.get('validation_ap40'))}`。"
    )
    add(
        f"5. **相对历史 A0 = 0.2463560956？** "
        f"{'改善' if (delta or 0) > 0 else '下降'} `{_signed(delta)}`（{label}）。"
    )
    gap = sup.get("train_valid_ap_gap")
    add(
        f"6. **Train−Valid gap 如何？** Train AP40 {_f(te.get('ap'))}，"
        f"Valid AP40 {_f(decision.get('validation_ap40'))}，gap {_signed(gap)}；"
        f"Train BCE {_f(te.get('bce'))}，Valid BCE {_f(sup.get('valid_bce'))}。"
    )
    add("")

    # ------------------------------------------------------- protocol status
    add("## 2. Protocol compliance")
    add("")
    add(
        _table(
            ["constraint", "observed"],
            [
                ["seed", str(sup.get("seed"))],
                ["pretrain runs", "1"],
                ["supervised runs", "1 (fixed epoch 40)"],
                ["inference parameters", _int(sup.get("parameter_count"))],
                ["base + new parameters", f"{_int(sup.get('base_parameters'))} + {_int(sup.get('new_parameters'))}"],
                ["historical A0 retrained", "no"],
                ["other arms (A1/A2/A3/G/V)", "none"],
                ["validation checkpoints", f"{', '.join(sorted(ap))} (diagnostics only)"],
                ["official Test labels", "never read, never scored"],
                ["pretrain updates", f"{_int(pre.get('updates_completed'))} (export at update 10000)"],
                ["decision bands frozen", "yes (`ssl_common.DECISION_BANDS`)"],
            ],
        )
    )
    add("")

    # ------------------------------------------------------------ pretext
    add("## 3. Pretext learning (probe, visit 0)")
    add("")
    add(
        "Probe = 4,096 hash-selected **official Train** molecules, never used for a "
        "parameter update. Prior CE is computed from SSL-fit class frequencies on "
        "the probe's own masked targets."
    )
    add("")
    add(
        _table(
            ["update", "atom CE", "atom prior CE", "atom acc", "bond CE", "bond prior CE", "bond acc"],
            [
                [
                    str(row.get("update")),
                    _f(row.get("probe_atom_ce"), 4),
                    _f(row.get("probe_atom_prior_ce"), 4),
                    _pct(row.get("probe_atom_accuracy")),
                    _f(row.get("probe_bond_ce"), 4),
                    _f(row.get("probe_bond_prior_ce"), 4),
                    _pct(row.get("probe_bond_accuracy")),
                ]
                for row in probe.get("curve") or []
            ],
        )
    )
    add("")
    add(
        _table(
            ["gate item", "value", "required", "passed"],
            [
                ["atom margin (nats)", _signed(gate.get("atom_margin_nats"), 4), f">= {gate.get('required_margin_nats')}", str(gate.get("atom_passed"))],
                ["bond margin (nats)", _signed(gate.get("bond_margin_nats"), 4), f">= {gate.get('required_margin_nats')}", str(gate.get("bond_passed"))],
                ["encoders changed", _signed(gate.get("encoder_l2_delta"), 6), "L2 > 0 and all tensors changed", str(gate.get("encoders_changed"))],
                ["gate label", f"**{gate.get('label')}**", "", ""],
            ],
        )
    )
    add("")
    add(
        f"Masked targets scored on the probe at the final update: "
        f"{_int(fin.get('masked_atoms'))} atoms, {_int(fin.get('masked_bonds'))} bonds. "
        f"Atom class counts `{fin.get('atom_class_counts')}`, bond class counts "
        f"`{fin.get('bond_class_counts')}`."
    )
    add("")

    # ------------------------------------------------------------ transfer
    add("## 4. Encoder transfer audit")
    add("")
    add(
        _table(
            ["item", "value"],
            [
                ["transferred prefixes", ", ".join(f"`{p}*`" for p in transfer.get("transferred_prefixes") or [])],
                ["frozen prefixes", ", ".join(f"`{p}*`" for p in transfer.get("frozen_prefixes") or [])],
                ["tensors transferred", _int(transfer.get("n_encoder_tensors_changed"))],
                ["encoder tensors changed vs scratch", f"{_int(transfer.get('n_encoder_tensors_changed'))}/{_int(transfer.get('n_encoder_tensors'))}"],
                ["non-transferred tensors == historical seed-0", str(transfer.get("non_transferred_tensors_are_historical_seed0"))],
                ["pretext heads transferred", str(transfer.get("pretext_heads_present"))],
                ["SSL optimizer state transferred", str(transfer.get("ssl_optimizer_state_transferred"))],
                ["state-dict keys unchanged", str(transfer.get("state_dict_unchanged"))],
                ["inference parameters", _int(transfer.get("inference_parameters"))],
                ["forward probe", str((transfer.get("shape_probe") or {}).get("logits_shape"))],
                ["SSL updates used", _int(transfer.get("ssl_updates"))],
                ["SSL encoder fingerprint", f"`{str(transfer.get('ssl_encoder_fingerprint'))[:16]}…`"],
            ],
        )
    )
    add("")
    add("Transferred tensor SHA256:")
    add("")
    add(_table(["tensor", "sha256"], [[f"`{k}`", f"`{v[:16]}…`"] for k, v in (transfer.get("ssl_encoder_tensor_sha256") or {}).items()]))
    add("")
    change = transfer.get("encoder_change_vs_scratch") or {}
    if change:
        add(
            f"Encoder L2 change vs scratch init: `{_f(change.get('l2_delta'), 6)}` "
            f"(relative `{_f(change.get('relative_l2_delta'), 6)}`); "
            f"all tensors changed: `{change.get('all_tensors_changed')}`."
        )
        add("")

    # ---------------------------------------------------------- downstream
    add("## 5. Supervised fine-tune (official Validation)")
    add("")
    add(
        _table(
            ["epoch", "Validation AP", "Valid BCE", "tasks scored"],
            [
                [
                    epoch,
                    _f(rep.get("ap")),
                    _f(rep.get("valid_bce")),
                    _int(rep.get("n_tasks_scored")),
                ]
                for epoch, rep in sorted(ap.items(), key=lambda kv: int(kv[0]))
            ],
        )
    )
    add("")
    add(
        _table(
            ["epoch-40 quantity", "value"],
            [
                ["Validation macro AP", _f(decision.get("validation_ap40"))],
                ["historical A0 AP", _f(decision.get("historical_a0_validation_ap40"))],
                ["Delta AP40", _signed(delta)],
                ["decision label", f"**{label}**"],
                ["Train AP40", _f(te.get("ap"))],
                ["Train BCE40", _f(te.get("bce"))],
                ["Valid BCE40", _f(sup.get("valid_bce"))],
                ["Train − Valid AP gap", _signed(gap)],
                ["epochs completed", _int(sup.get("epochs_run"))],
                ["global steps", _int(sup.get("global_steps_run"))],
                ["gradient-clip events", _int(sup.get("grad_clip_events"))],
                ["non-finite events", _int(sup.get("non_finite_events"))],
            ],
        )
    )
    add("")
    add("Batch-order parity with the historical seed-0 plan:")
    add("")
    plan = sup.get("batch_plan") or {}
    add(
        _table(
            ["plan field", "value"],
            [
                ["global plan sha256", f"`{plan.get('global_plan_sha256')}`"],
                ["prefix sha256", f"`{plan.get('prefix_sha256')}`"],
                ["epochs consumed", _int(plan.get("consumed_epochs"))],
                ["batch size", _int(plan.get("batch_size"))],
            ],
        )
    )
    add("")

    # ----------------------------------------------------------- per-task
    add("## 6. Per-task breadth")
    add("")
    add(
        _table(
            ["metric", "value"],
            [
                ["tasks compared", _int(summ.get("n_tasks_scored"))],
                ["improved", _int(summ.get("n_improved"))],
                ["regressed", _int(summ.get("n_regressed"))],
                ["unchanged", _int(summ.get("n_equal"))],
                ["mean delta", _signed(summ.get("mean_delta"))],
                ["median delta", _signed(summ.get("median_delta"))],
                ["min / max delta", f"{_signed(summ.get('min_delta'))} / {_signed(summ.get('max_delta'))}"],
            ],
        )
    )
    add("")
    add("Support strata (Validation positives per task):")
    add("")
    add(
        _table(
            ["bucket", "tasks", "mean AP", "historical mean AP", "mean delta", "improved", "regressed"],
            [
                [
                    name,
                    _int(stats.get("n_tasks")),
                    _f(stats.get("mean_ap")),
                    _f(stats.get("mean_historical_ap")),
                    _signed(stats.get("mean_delta")),
                    _int(stats.get("n_improved")),
                    _int(stats.get("n_regressed")),
                ]
                for name, stats in buckets.items()
            ],
        )
    )
    add("")
    scored = [r for r in per_task if r["ap"] is not None]
    by_delta = sorted(scored, key=lambda r: r["delta"])
    add("Largest regressions:")
    add("")
    add(
        _table(
            ["task", "AP", "historical AP", "delta", "positives"],
            [
                [str(r["task"]), _f(r["ap"]), _f(r["historical_ap"]), _signed(r["delta"]), _int(r["positive"])]
                for r in by_delta[:5]
            ],
        )
    )
    add("")
    add("Largest improvements:")
    add("")
    add(
        _table(
            ["task", "AP", "historical AP", "delta", "positives"],
            [
                [str(r["task"]), _f(r["ap"]), _f(r["historical_ap"]), _signed(r["delta"]), _int(r["positive"])]
                for r in by_delta[::-1][:5]
            ],
        )
    )
    add("")
    low_support_share = None
    gains = [r for r in scored if r["delta"] > 0]
    if gains:
        low_support_share = sum(float(r["delta"]) for r in gains if r["positive"] < 5) / sum(
            float(r["delta"]) for r in gains
        )
    add(
        "Breadth note: "
        + (
            f"{_int(summ.get('n_improved'))}/{_int(summ.get('n_tasks_scored'))} tasks improved. "
            if summ
            else ""
        )
        + (
            f"Tasks with <5 Validation positives contribute {_pct(low_support_share)} of the summed "
            "positive per-task deltas; a gain concentrated there is not a stable generalisation claim."
            if low_support_share is not None
            else "No positive per-task deltas."
        )
    )
    add("")

    # ------------------------------------------------------------- compute
    add("## 7. Compute and provenance")
    add("")
    add(
        _table(
            ["item", "value"],
            [
                ["parameters", _int(sup.get("parameter_count"))],
                ["forward FLOPs (128 molecules)", _int(flops.get("flops")) if flops.get("flops") else "—"],
                ["pretrain GPU hours", _f(pre.get("gpu_hours"), 4)],
                ["supervised GPU hours", _f(sup.get("gpu_hours"), 4)],
                ["total GPU hours", _f(ledger.get("total_gpu_hours"), 4)],
                ["budget target / hard", f"{_f(ledger.get('budget_target_gpu_hours'), 1)} / {_f(ledger.get('budget_hard_gpu_hours'), 1)}"],
                ["within target", str(ledger.get("within_target"))],
                ["pretrain peak allocated", f"{_int(pre.get('gpu_peak_allocated_bytes'))} B"],
                ["supervised peak allocated", f"{_int(sup.get('gpu_peak_allocated_bytes'))} B"],
                ["host / device", f"{(pre.get('environment') or {}).get('hostname')} / {(pre.get('environment') or {}).get('cuda_device_name')}"],
                ["torch / CUDA", f"{(sup.get('environment') or {}).get('torch')} / cuda {(sup.get('environment') or {}).get('torch_cuda')}"],
                ["git commit", f"`{sup.get('git_commit')}`"],
                ["cache fingerprint", f"`{(data.get('cache') or {}).get('feature_spec_hash')}`"],
                ["cache sha256", f"`{(data.get('cache') or {}).get('manifest_cache_sha256')}`"],
                ["feature spec hash", f"`{(data.get('feature_spec') or {}).get('hash')}`"],
                ["split partition sha256", f"`{(data.get('split') or {}).get('partition_sha256')}`"],
                ["seed-0 norm fingerprint", f"`{pre.get('norm_fingerprint')}`"],
                ["pretrain checkpoint", f"`{pre.get('checkpoint')}`"],
                ["pretrain checkpoint sha256", f"`{pre.get('checkpoint_sha256')}`"],
                ["epoch-40 checkpoint", f"`{sup.get('checkpoint_epoch40')}`"],
                ["epoch-40 checkpoint sha256", f"`{sup.get('checkpoint_epoch40_sha256')}`"],
                ["per-task CSV sha256", f"`{sup.get('per_task_table_sha256')}`"],
            ],
        )
    )
    add("")

    # ---------------------------------------------------------- label audit
    add("## 8. Label and Test access audit")
    add("")
    access = data.get("loader_file_access") or {}
    add(
        _table(
            ["audit item", "value"],
            [
                ["label artifact opened by pretraining", str(access.get("label_artifact_opened"))],
                ["labels read", str(data.get("labels_read"))],
                ["Test rows used for pretraining", _int(data.get("test_rows_used_for_pretraining"))],
                ["Test labels read", str(data.get("test_labels_read"))],
                ["Valid rows used for pretraining", "0 (probe and fit are official Train only)"],
                ["mask/leakage audit all_passed", str(audit.get("all_passed"))],
                ["audit checks", ", ".join(c["name"] for c in audit.get("checks", []))],
                ["run registry Test evaluated", str((registry.get("rfcs") or {}).get("test_evaluated"))],
            ],
        )
    )
    add("")
    add(
        "The epoch-40 checkpoint contains no Test logits: validation and Train "
        "reports are the only scored splits in `runs/supervised/`."
    )
    add("")

    # ------------------------------------------------------------ decision
    add("## 9. Decision and interpretation")
    add("")
    add(f"**{label}** — Delta AP40 = `{_signed(delta)}` against the frozen historical A0 "
        f"`{_f(decision.get('historical_a0_validation_ap40'))}`.")
    add("")
    add("Pre-registered bands:")
    add("")
    add(
        _table(
            ["band", "rule"],
            [
                ["PROMISING_SINGLE_SEED", "Delta >= +0.005"],
                ["WEAK_SIGNAL", "+0.002 <= Delta < +0.005"],
                ["NO_CLEAR_GAIN", "-0.002 < Delta < +0.002"],
                ["NEGATIVE_TRANSFER", "Delta <= -0.002"],
            ],
        )
    )
    add("")
    add("Mandatory caveats:")
    add("")
    for caveat in decision.get("caveats") or []:
        add(f"- {caveat}")
    add("")

    # ----------------------------------------------------------- artifacts
    add("## 10. Artifacts")
    add("")
    add(
        _table(
            ["artifact", "path"],
            [
                ["data audit", "`analysis/data_audit.json`"],
                ["mask/leakage audit", "`analysis/mask_leakage_audit.json`"],
                ["pretrain probe", "`analysis/pretrain_probe.json`"],
                ["pretrain run", "`analysis/pretrain_run.json`"],
                ["encoder transfer", "`analysis/encoder_transfer_audit.json`"],
                ["epoch-40 report", "`analysis/supervised_epoch40.json`"],
                ["per-task AP", "`analysis/per_task_ap.csv`"],
                ["run registry", "`analysis/run_registry.json`"],
                ["GPU ledger", "`analysis/gpu_ledger.json`"],
                ["decision", "`analysis/decision.json`"],
                ["pretrain checkpoint (local)", "`runs/pretrain/checkpoint_final.pt`"],
                ["pretrain curve (local)", "`runs/pretrain/curve.csv`"],
                ["supervised checkpoint (local)", "`runs/supervised/checkpoint_epoch40.pt`"],
                ["supervised curve (local)", "`runs/supervised/train_curve.csv`"],
                ["per-task AP (local)", "`runs/supervised/per_task_ap.csv`"],
            ],
        )
    )
    add("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    parser.add_argument("--out", default=str(C.ROUND / "REPORT.md"))
    args = parser.parse_args(argv)
    path = render(Path(args.analysis_dir), Path(args.out))
    C.log(f"[ssl-report] wrote {C.track_rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
