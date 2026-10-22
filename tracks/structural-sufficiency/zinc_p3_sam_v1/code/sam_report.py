"""Local aggregation for ``zinc_p3_sam_v1``.

Reads the two pulled arm directories and writes the main table, paired bootstrap,
group description, curve figure and the machine-readable summary used by
REPORT.md.  It never trains or selects anything: the checkpoints were already
frozen and scored once on the official test by the per-arm runner.

Reuse, not reimplementation: the frozen reference rows/labels and the audited
``signature_unseen`` / ``y<=-3`` flags come from
``zinc_restricted_adaptation_v1/analysis/reference_preds_{valid,test}.csv`` and
the paired bootstrap is the existing ``ot_runner.paired_bootstrap``.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_OT = _SS / "zinc_structural_object_tokens_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_OT), str(_V2), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ot_runner as OTR  # noqa: E402
import sam_core as S  # noqa: E402

EXP = S.EXP
ARM_DIRS = {"C": EXP / "analysis" / "P3_Control", "S": EXP / "analysis" / "P3_SAM"}
RUN_DIRS = {"C": EXP / "runs" / "C_control_s0", "S": EXP / "runs" / "S_sam_s0"}
HISTORICAL = {
    "P3B_s0": {"params": 413505, "valid": 0.07622375367576023},
    "L_s0": {"params": 494193, "valid": 0.078537152},
    "DBT_s0": {"params": 452523, "valid": 0.077768402},
    "A_joint_BF_s0": {"params": 447505, "valid": 0.076731364, "test": 0.070067461},
}


def _load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return np.asarray(d["pred"], dtype=np.float64), np.asarray(d["row"], dtype=np.int64)


def _aux(pred: np.ndarray, y: np.ndarray) -> dict:
    err = np.abs(pred - y)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "p95_ae": float(np.quantile(err, 0.95)),
        "max_ae": float(err.max()),
        "max_ae_row": int(np.argmax(err)),
    }


def _order_hashes(arm: str) -> dict:
    return C.read_json(RUN_DIRS[arm] / "order_hashes.json")


def check_round_identity() -> dict:
    out: dict = {}
    ck = {arm: C.read_json(ARM_DIRS[arm] / "freeze.json") for arm in ("C", "S")}
    out["init_state_sha256"] = {arm: ck[arm]["init_state_sha256"] for arm in ("C", "S")}
    out["ssl_sha256"] = {arm: ck[arm]["ssl_sha256"] for arm in ("C", "S")}
    out["config_sha256"] = {arm: ck[arm]["config_sha256"] for arm in ("C", "S")}
    out["params"] = {arm: ck[arm]["params"] for arm in ("C", "S")}
    out["outer_updates"] = {arm: ck[arm]["outer_updates"] for arm in ("C", "S")}
    out["gradient_calculations"] = {arm: ck[arm]["gradient_calculations"] for arm in ("C", "S")}
    out["same_init"] = bool(len(set(out["init_state_sha256"].values())) == 1)
    out["same_ssl"] = bool(len(set(out["ssl_sha256"].values())) == 1)
    out["same_config"] = bool(len(set(out["config_sha256"].values())) == 1)
    out["same_params"] = bool(len(set(out["params"].values())) == 1)
    oh = {arm: _order_hashes(arm)["epochs"] for arm in ("C", "S")}
    out["row_orders_identical"] = bool(oh["C"] == oh["S"])
    out["n_epochs_compared"] = len(oh["C"])
    out["first_epoch_order_sha256"] = oh["C"][0]
    out["last_epoch_order_sha256"] = oh["C"][-1]
    out["outer_updates_equal_47400"] = bool(all(v == 47400 for v in out["outer_updates"].values()))
    out["gradient_counts_as_declared"] = bool(
        out["gradient_calculations"]["C"] == 47400
        and out["gradient_calculations"]["S"] == 94800
    )
    return out


def main() -> int:
    identity = check_round_identity()
    if not (identity["same_init"] and identity["same_ssl"] and identity["same_config"]
            and identity["same_params"] and identity["row_orders_identical"]
            and identity["outer_updates_equal_47400"]):
        raise RuntimeError(f"round identity check failed: {identity}")

    train = {arm: C.read_json(RUN_DIRS[arm] / "train.json") for arm in ("C", "S")}
    freeze = {arm: C.read_json(ARM_DIRS[arm] / "freeze.json") for arm in ("C", "S")}
    test_rep = {arm: C.read_json(ARM_DIRS[arm] / "test_report.json") for arm in ("C", "S")}
    diag = {arm: C.read_json(ARM_DIRS[arm] / "sam_diagnostics.json") for arm in ("C", "S")}
    timing = {arm: C.read_json(ARM_DIRS[arm] / "timing.json") for arm in ("C", "S")}

    ref_valid = OTR.read_reference("valid")
    ref_test = OTR.read_reference("test")
    pred_valid = {arm: _load_pred(RUN_DIRS[arm] / "predictions" / "valid_best.npz") for arm in ("C", "S")}
    pred_test = {arm: _load_pred(RUN_DIRS[arm] / "predictions" / "test_best.npz") for arm in ("C", "S")}
    for arm in ("C", "S"):
        for split, ref in (("valid", ref_valid), ("test", ref_test)):
            p, row = pred_valid[arm] if split == "valid" else pred_test[arm]
            if not np.array_equal(row, ref["rows"]):
                raise RuntimeError(f"{arm}/{split}: row identity mismatch")
        # frozen test MAE must equal the runner's own report
        if abs(_aux(pred_test[arm][0], ref_test["y"])["mae"] - test_rep[arm]["test"]["mae"]) > 1e-9:
            raise RuntimeError(f"{arm}: test MAE mismatch between npz and test_report.json")

    y_v, y_t = ref_valid["y"], ref_test["y"]
    aux = {arm: {"valid": _aux(pred_valid[arm][0], y_v),
                 "test": _aux(pred_test[arm][0], y_t)} for arm in ("C", "S")}

    # paired contrasts (molecule-level; not training-seed uncertainty)
    contrast = {}
    for split, y, preds, ref in (("valid", y_v, pred_valid, ref_valid),
                                 ("test", y_t, pred_test, ref_test)):
        ae = {arm: np.abs(preds[arm][0] - y) for arm in ("C", "S")}
        for name, p in ref["refs"].items():
            ae[name] = np.abs(p - y)
        contrast[split] = {
            "S_minus_C": OTR.paired_bootstrap(ae["S"], ae["C"]),
            "C_minus_P3B": OTR.paired_bootstrap(ae["C"], ae["P3B_s0"]),
            "S_minus_P3B": OTR.paired_bootstrap(ae["S"], ae["P3B_s0"]),
            "mean_delta_S_minus_C": float(ae["S"].mean() - ae["C"].mean()),
            "mean_delta_C_minus_P3B": float(ae["C"].mean() - ae["P3B_s0"].mean()),
            "mean_delta_S_minus_P3B": float(ae["S"].mean() - ae["P3B_s0"].mean()),
        }

    # descriptive four groups, unchanged audited flags
    group_rows = []
    for split, ref, preds in (("valid", ref_valid, pred_valid), ("test", ref_test, pred_test)):
        y = ref["y"]
        ae = {arm: np.abs(preds[arm][0] - y) for arm in ("C", "S")}
        for name, p in ref["refs"].items():
            ae[name] = np.abs(p - y)
        for gname, unseen_flag, ordinary_flag in OTR.GROUPS:
            mask = (ref["unseen"] == unseen_flag) & (ref["ordinary"] == ordinary_flag)
            item = {"split": split, "group": gname, "n": int(mask.sum()),
                    "mae": {k: float(v[mask].mean()) for k, v in ae.items()}}
            item["S_minus_C"] = float(item["mae"]["S"] - item["mae"]["C"])
            group_rows.append(item)

    main_rows = []
    for arm in ("C", "S"):
        f = freeze[arm]
        main_rows.append({
            "arm": arm,
            "name": train[arm]["arms_name"],
            "params": int(f["params"]),
            "best_valid": float(f["best_valid_mae"]),
            "best_epoch": int(f["best_epoch"]),
            "train_mae_eval_mode_at_best": float(f["best_train_mae_eval_mode"]),
            "final_valid": float(f["final_valid_mae"]),
            "train_mae_eval_mode_at_final": float(f["final_train_mae_eval_mode"]),
            "test_mae": float(test_rep[arm]["test"]["mae"]),
            "test_rmse": float(aux[arm]["test"]["rmse"]),
            "test_p95_ae": float(aux[arm]["test"]["p95_ae"]),
            "test_max_ae": float(aux[arm]["test"]["max_ae"]),
            "test_max_ae_row": int(aux[arm]["test"]["max_ae_row"]),
            "outer_updates": int(f["outer_updates"]),
            "gradient_calculations": int(f["gradient_calculations"]),
            "wall_seconds": float(train[arm]["wall_seconds"]),
            "formal_train_seconds": float(train[arm]["formal_train_seconds"]),
            "gpu_hours": train[arm]["gpu_hours"],
            "steady_epoch_seconds": float(timing[arm]["steady_epoch_seconds"]),
            "valid_tail_100_median": float(train[arm]["valid_tail_100"]["median"]),
            "valid_tail_100_min": float(train[arm]["valid_tail_100"]["min"]),
            "best_epoch_in_tail100": bool(train[arm]["valid_tail_100"]["best_epoch_in_tail"]),
            "start_epoch0_valid_diagnostic": float(train[arm]["epoch0_valid_diagnostic"]),
            "best_ckpt_sha256": f["best_sha256"],
            "final_ckpt_sha256": f["final_sha256"],
            "lr": float(f["recipe"]["lr"]),
            "weight_decay": float(f["recipe"]["weight_decay"]),
            "sam_rho": (float(f["sam"]["rho"]) if arm == "S" else None),
        })

    hist_rows = [
        {"name": "P3B_s0", "params": 413505, "best_valid": HISTORICAL["P3B_s0"]["valid"],
         "test_mae": float(np.abs(ref_test["refs"]["P3B_s0"] - y_t).mean()),
         "source": "reference_preds_test.csv (recomputed MAE)"},
        {"name": "L_s0", "params": 494193, "best_valid": HISTORICAL["L_s0"]["valid"],
         "test_mae": float(np.abs(ref_test["refs"]["L_s0"] - y_t).mean()),
         "source": "reference_preds_test.csv (recomputed MAE)"},
        {"name": "DBT_s0", "params": 452523, "best_valid": HISTORICAL["DBT_s0"]["valid"],
         "test_mae": float(np.abs(ref_test["refs"]["DBT_s0"] - y_t).mean()),
         "source": "reference_preds_test.csv (recomputed MAE)"},
        {"name": "A_joint_BF_s0", "params": 447505,
         "best_valid": HISTORICAL["A_joint_BF_s0"]["valid"],
         "test_mae": HISTORICAL["A_joint_BF_s0"]["test"],
         "source": "zinc_structural_object_tokens_v1 frozen REPORT/analysis; no prediction rows reused"},
    ]

    payload = {
        "protocol": S.PROTOCOL,
        "identity": identity,
        "main": main_rows,
        "historical_reference": hist_rows,
        "contrasts": contrast,
        "aux_metrics": aux,
        "groups": group_rows,
        "diagnostics": diag,
        "curves": {arm: [{"epoch": r["epoch"], "valid_mae": r["valid_mae"],
                          "online_train_mae": r["online_train_mae"],
                          "perturbed_train_mae": r["perturbed_train_mae"],
                          "lr": r["lr"], "seconds": r["seconds"]}
                         for r in train[arm]["curve"]] for arm in ("C", "S")},
        "official_test_exposed": True,
        "note": ("paired bootstrap resamples the 1,000 test molecules; it does not "
                 "represent across-seed training uncertainty"),
    }
    C.write_json(EXP / "analysis" / "summary.json", payload)

    # markdown main table
    def fmt(v, nd=6):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [
        "# zinc_p3_sam_v1 — main table",
        "",
        "| arm | params | best valid | ep | eval-train@best | final valid | test MAE | RMSE | p95 AE | max AE | updates | grads | wall s | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main_rows:
        lines.append(
            f"| **{r['arm']} {r['name']}** | {r['params']:,} | {fmt(r['best_valid'], 9)} | "
            f"{r['best_epoch']} | {fmt(r['train_mae_eval_mode_at_best'])} | "
            f"{fmt(r['final_valid'], 9)} | **{fmt(r['test_mae'], 9)}** | "
            f"{fmt(r['test_rmse'])} | {fmt(r['test_p95_ae'])} | {fmt(r['test_max_ae'])} | "
            f"{r['outer_updates']:,} | {r['gradient_calculations']:,} | "
            f"{r['wall_seconds']:.0f} | {fmt(r['gpu_hours'], 3)} |")
    lines += [
        "",
        "Same-round contrast (molecule-paired, 10,000 bootstrap resamples):",
        "",
        f"- valid S−C = {contrast['valid']['mean_delta_S_minus_C']:+.6f}, "
        f"95% CI [{contrast['valid']['S_minus_C']['ci95'][0]:+.6f}, "
        f"{contrast['valid']['S_minus_C']['ci95'][1]:+.6f}]",
        f"- test S−C = {contrast['test']['mean_delta_S_minus_C']:+.6f}, "
        f"95% CI [{contrast['test']['S_minus_C']['ci95'][0]:+.6f}, "
        f"{contrast['test']['S_minus_C']['ci95'][1]:+.6f}], "
        f"fraction of resamples S better = "
        f"{contrast['test']['S_minus_C']['fraction_candidate_better']:.3f}",
        f"- valid C−P3B = {contrast['valid']['mean_delta_C_minus_P3B']:+.6f}; "
        f"test C−P3B = {contrast['test']['mean_delta_C_minus_P3B']:+.6f}",
        "",
        "Historical rows are reused reference predictions (not retrained) and only "
        "for orientation; the C vs S contrast is the same-round evidence.",
        "",
        "| reference | params | valid | test | source |",
        "|---|---:|---:|---:|---|",
    ]
    for r in hist_rows:
        lines.append(f"| {r['name']} | {r['params']:,} | {fmt(r['best_valid'], 9)} | "
                     f"{fmt(r['test_mae'], 9)} | {r['source']} |")
    (EXP / "analysis" / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # groups markdown
    glines = ["# Fixed four groups (descriptive; no selection)", "",
              "`signature_unseen` × `y>−3` from the existing audited reference rows. "
              "Row 172 and all extreme labels stay in the overall numbers.", ""]
    for split in ("valid", "test"):
        glines += [f"## {split}", "",
                   "| group | n | P3B | L | DBT | C | S | S−C |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for item in group_rows:
            if item["split"] != split:
                continue
            m = item["mae"]
            glines.append(
                f"| {item['group']} | {item['n']} | {m['P3B_s0']:.6f} | {m['L_s0']:.6f} | "
                f"{m['DBT_s0']:.6f} | {m['C']:.6f} | {m['S']:.6f} | {item['S_minus_C']:+.6f} |")
        glines.append("")
    (EXP / "analysis" / "groups.md").write_text("\n".join(glines) + "\n", encoding="utf-8")

    # curves figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm, color in (("C", "#1f77b4"), ("S", "#d62728")):
            curve = train[arm]["curve"]
            ep = [r["epoch"] for r in curve]
            axes[0].plot(ep, [r["valid_mae"] for r in curve], color=color, lw=1.0,
                         label=f"{arm} {train[arm]['arms_name']}")
            axes[1].plot(ep, [r["online_train_mae"] for r in curve], color=color, lw=1.0,
                         label=f"{arm} normal online")
            if arm == "S":
                axes[1].plot(ep, [r["perturbed_train_mae"] for r in curve], color=color,
                             lw=0.7, ls="--", label="S perturbed (θ+ε)")
        axes[0].axhline(0.076223754, color="grey", ls=":", lw=1, label="historical P3B valid")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("valid raw-y MAE")
        axes[0].set_title("Full valid every epoch"); axes[0].legend(fontsize=7)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("online raw-y train MAE")
        axes[1].set_title("Normal fit vs SAM second-pass column")
        axes[1].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(EXP / "analysis" / "curves.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is descriptive only
        C.log(f"[report] curve figure skipped: {exc}")

    C.log(f"[report] wrote {EXP / 'analysis' / 'summary.json'}")
    C.log("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
