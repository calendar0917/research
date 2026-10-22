"""Local aggregation for the two arms (run after pulling the artifacts).

Writes ``analysis/main_table.{json,md}``, ``analysis/summary.json``,
``analysis/groups.md`` and ``analysis/curves.png`` from the two runs.  No
reference model is retrained; the P3B / L / DBT rows are the frozen numbers
used by every previous round of this track.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUNS = EXP / "runs"
REF_DIR = EXP.parents[0] / "zinc_restricted_adaptation_v1" / "analysis"

REFERENCES = [
    {"model": "P3B_s0", "params": 413_505, "init": "train-only masked SSL",
     "best_valid": 0.076223754, "best_epoch": 377, "train_eval_at_best": None,
     "final_valid": None, "test_mae": 0.069864841, "epochs": 600, "updates": 47_400,
     "gpu_hours": 0.30, "status": "reused"},
    {"model": "L_s0", "params": 494_193, "init": "train-only masked SSL",
     "best_valid": 0.078537152, "best_epoch": 489, "train_eval_at_best": 0.005510,
     "final_valid": None, "test_mae": 0.066266555, "epochs": 600, "updates": 47_400,
     "gpu_hours": 0.35, "status": "reused"},
    {"model": "DBT_s0", "params": 452_523, "init": "train-only masked SSL",
     "best_valid": 0.077768402, "best_epoch": 949, "train_eval_at_best": 0.015964,
     "final_valid": 0.081313816, "test_mae": 0.069401494, "epochs": 1500,
     "updates": 235_500, "gpu_hours": 3.6918, "status": "reused"},
    {"model": "old P3-BF_s0", "params": 447_505, "init": "supervised P3B best.pt",
     "best_valid": 0.076223756, "best_epoch": 0, "train_eval_at_best": 0.014680,
     "final_valid": 0.078363, "test_mae": 0.069864822, "epochs": 100, "updates": 7_900,
     "gpu_hours": 0.0645, "status": "reused (early-stopped, best = P3B)"},
]

ARMS = [
    ("A", "A_joint_BF_s0", "A_joint_BF"),
    ("B", "B_object_tokens_s0", "B_object_tokens"),
]


def load_arm(arm: str, run_name: str, candidate: str) -> dict | None:
    run = RUNS / run_name
    ana = ANALYSIS / candidate
    if not (run / "train.json").exists():
        return None
    train = C.read_json(run / "train.json")
    freeze = C.read_json(ana / "freeze.json") if (ana / "freeze.json").exists() else {}
    test = C.read_json(ana / "test_report.json") if (ana / "test_report.json").exists() else {}
    return {"arm": arm, "run": run, "ana": ana, "train": train,
            "freeze": freeze, "test": test}


def main() -> int:
    arms = [a for a in (load_arm(*spec) for spec in ARMS) if a]
    rows = [dict(r) for r in REFERENCES]
    for a in arms:
        t, s, f, tr = a["train"], a["train"]["scored"], a["freeze"], a["test"]
        test_mae = tr["test"]["mae"] if tr else None
        rows.append({
            "model": f"{a['train']['candidate']}_s0",
            "params": int(t["parameter_count"]),
            "init": ("train-only masked SSL atom/bond/junction only"
                     if a["arm"] == "A" else "train-only masked SSL atom encoder only"),
            "best_valid": float(s["best"]["valid_mae"]),
            "best_epoch": int(s["best"]["epoch"]),
            "train_eval_at_best": float(s["best"]["train_mae_eval_mode"]),
            "final_valid": float(s["final"]["valid_mae"]),
            "test_mae": test_mae,
            "epochs": int(t["recipe"]["epochs"]),
            "updates": int(t["recipe"]["actual_global_steps"]),
            "gpu_hours": t.get("gpu_hours"),
            "status": "completed" if test_mae is not None else "INCOMPLETE",
            "best_sha256": f.get("best_sha256"),
            "wall_seconds": t.get("wall_seconds"),
            "steady_epoch_seconds": t["plan"].get("steady_epoch_seconds"),
            "planned_epochs": t["plan"].get("epochs"),
        })
    C.write_json(ANALYSIS / "main_table.json", {"rows": rows})
    md = ["| model | params | init | best valid | ep | eval-train@best | final valid | test MAE | epochs | updates | GPU-h |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| {model} | {params:,} | {init} | {best_valid:.9f} | {best_epoch} | {te} | {fv} | {tm} | {epochs} | {updates:,} | {gh} |".format(
            te="—" if r["train_eval_at_best"] is None else f"{r['train_eval_at_best']:.6f}",
            fv="—" if r["final_valid"] is None else f"{r['final_valid']:.6f}",
            tm="—" if r["test_mae"] is None else f"{r['test_mae']:.9f}",
            gh="—" if r["gpu_hours"] is None else f"{r['gpu_hours']:.4f}",
            **{k: r[k] for k in ("model", "params", "init", "best_valid", "best_epoch",
                                 "epochs", "updates")}))
    (ANALYSIS / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = {"references": REFERENCES, "arms": {}}
    for a in arms:
        t, s, tr = a["train"], a["train"]["scored"], a["test"]
        item = {
            "candidate": t["candidate"], "params": t["parameter_count"],
            "best_valid": s["best"]["valid_mae"], "best_epoch": s["best"]["epoch"],
            "train_eval_at_best": s["best"]["train_mae_eval_mode"],
            "final_valid": s["final"]["valid_mae"],
            "final_train_eval": s["final"]["train_mae_eval_mode"],
            "epochs": t["recipe"]["epochs"], "updates": t["recipe"]["actual_global_steps"],
            "peak_lr": t["recipe"]["peak_lr"], "weight_decay": t["recipe"]["weight_decay"],
            "wall_seconds": t["wall_seconds"], "gpu_hours": t["gpu_hours"],
            "steady_epoch_seconds": t["plan"]["steady_epoch_seconds"],
            "planned_epochs": t["plan"]["epochs"],
            "best_sha256": a["freeze"].get("best_sha256"),
            "test_mae": tr["test"]["mae"] if tr else None,
            "test_contrasts": tr["overall"]["contrasts"] if tr else None,
            "groups_valid": C.read_json(a["ana"] / "groups_valid.json")
            if (a["ana"] / "groups_valid.json").exists() else None,
            "groups_test": tr["overall"]["by_group"] if tr else None,
        }
        if tr:
            for ref in ("P3B_s0", "L_s0", "DBT_s0"):
                item[f"delta_test_vs_{ref}"] = float(
                    tr["test"]["mae"] - float(np.float64(REFERENCES[
                        {"P3B_s0": 0, "L_s0": 1, "DBT_s0": 2}[ref]]["test_mae"])))
        summary["arms"][t["candidate"]] = item
    C.write_json(ANALYSIS / "summary.json", summary)

    # groups.md (analysis-only buckets; row 172 kept)
    lines = ["# Fixed four-group view (analysis only, row 172 kept)", "",
             "Groups: `signature_unseen` x `y > -3` from the frozen reference rows; "
             "the grouping never enters the model or the selection.", ""]
    ref = {}
    for split in ("valid", "test"):
        path = REF_DIR / f"reference_preds_{split}.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            recs = list(csv.DictReader(fh))
        y = np.asarray([float(r["y_raw"]) for r in recs])
        unseen = np.asarray([int(r["signature_unseen"]) for r in recs])
        ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in recs])
        ref[split] = {"y": y, "unseen": unseen, "ordinary": ordinary,
                      "pred": {name: np.asarray([float(r[f"pred_{name}"]) for r in recs])
                               for name in ("P3B_s0", "L_s0", "DBT_s0")}}
    arm_preds = {}
    for a in arms:
        cand = a["train"]["candidate"]
        arm_preds[(cand, "valid")] = np.load(a["run"] / "predictions" / "valid_best.npz")["pred"]
        if (a["run"] / "predictions" / "test_best.npz").exists():
            arm_preds[(cand, "test")] = np.load(a["run"] / "predictions" / "test_best.npz")["pred"]
    for split in ("valid", "test"):
        r = ref[split]
        for gname, uf, of in (("covered_ordinary", 0, 1), ("covered_tail", 0, 0),
                              ("unseen_ordinary", 1, 1), ("unseen_tail", 1, 0)):
            mask = (r["unseen"] == uf) & (r["ordinary"] == of)
            cells = []
            for name in ("P3B_s0", "L_s0", "DBT_s0"):
                cells.append(f"{float(np.abs(r['pred'][name][mask] - r['y'][mask]).mean()):.6f}")
            for (cand, sp), pred in arm_preds.items():
                if sp != split:
                    continue
                cells.append(f"{float(np.abs(pred[mask] - r['y'][mask]).mean()):.6f}")
            lines.append(f"| {split} | {gname} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    heads = ["split", "group", "n", "P3B", "L", "DBT"] + [a["train"]["candidate"] for a in arms]
    lines.insert(4, "| " + " | ".join(heads) + " |")
    lines.insert(5, "|" + "---|" * len(heads))
    (ANALYSIS / "groups.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for a in arms:
            curve = C.read_json(a["run"] / "train.json")["curve"]
            ep = [c["epoch"] for c in curve if c["valid_mae"] is not None]
            va = [c["valid_mae"] for c in curve if c["valid_mae"] is not None]
            ax.plot(ep, va, label=f"{a['train']['candidate']} valid")
        ax.set_xlabel("epoch")
        ax.set_ylabel("raw-y valid MAE")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ANALYSIS / "curves.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        C.log(f"[report] curve plot skipped: {exc}")
    C.log((ANALYSIS / "main_table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
