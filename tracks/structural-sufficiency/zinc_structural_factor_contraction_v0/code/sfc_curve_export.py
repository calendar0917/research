"""Export the corrected training curves to CSV and print report tables."""
from __future__ import annotations
import csv, json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
J = EXP / "analysis" / "t3_synth_corrected.json"
CSV = EXP / "analysis" / "t3_bce_curves.csv"
ARMS = ("C", "P", "S", "Srand", "SP")


def main() -> int:
    d = json.loads(J.read_text())
    rows = d["runs"]
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "arm", "seed", "step", "train_bce_batch",
                    "train_acc_batch", "sfc_grad_norm"])
        for r in rows:
            for c in r["curve"]:
                w.writerow([r["task"], r["arm"], r["seed"], c["step"],
                            f"{c['train_bce_batch']:.10g}", f"{c['train_acc_batch']:.6g}",
                            "" if c["sfc_grad_norm"] is None else f"{c['sfc_grad_norm']:.10g}"])
    print(f"-> {CSV} ({sum(len(r['curve']) for r in rows)} points)")

    checkpoints = (0, 50, 100, 200, 300, 399)
    for task in ("reach", "junction"):
        print(f"\n## {task} batch-BCE at step")
        print("| arm | seed | " + " | ".join(str(s) for s in checkpoints) + " |")
        print("|---|---|" + "---|" * len(checkpoints))
        for arm in ARMS:
            for seed in (0, 1):
                r = [x for x in rows if x["task"] == task and x["arm"] == arm
                     and x["seed"] == seed][0]
                cur = {c["step"]: c["train_bce_batch"] for c in r["curve"]}
                vals = " | ".join(f"{cur[s]:.3f}" for s in checkpoints)
                print(f"| {arm} | {seed} | {vals} |")
    print("\n## final train/unseen (consistent features)")
    print("| task | arm | train acc mean | train acc seeds | unseen acc mean | unseen acc seeds | train bce | unseen bce |")
    print("|---|---|---|---|---|---|---|---|")
    for task in ("reach", "junction"):
        for arm in ARMS:
            a = d["arms"][task][arm]
            print(f"| {task} | {arm} | {a['train_acc_mean']:.4f} | "
                  f"{[round(x,4) for x in a['train_acc_per_seed']]} | "
                  f"{a['unseen_acc_mean']:.4f} | "
                  f"{[round(x,4) for x in a['unseen_acc_per_seed']]} | "
                  f"{a['train_bce_mean']:.4f} | {a['unseen_bce_mean']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
