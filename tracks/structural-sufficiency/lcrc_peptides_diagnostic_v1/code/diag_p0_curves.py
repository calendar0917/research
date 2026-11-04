"""P0.1 — audit the three historical Fast v1 Peptides-func learning curves.

Reads only the copied Fast v1 artifacts under ``history/``.  Produces

* ``analysis/history_runs.csv``          one row per seed (recipe, hash, cost)
* ``analysis/learning_curves.csv``       epochs 1/20/40/60/80/100/120 snapshot
* ``analysis/learning_curve_full.csv``   every epoch of every seed
* ``analysis/learning_curve_summary.csv`` best epoch / last-20-epoch deltas
* ``analysis/learning_curves.png``       two-panel curve plot
"""

from __future__ import annotations

import csv
from typing import Any, Sequence

import numpy as np

import diag_common as C

SNAPSHOT_EPOCHS = (1, 20, 40, 60, 80, 100, 120)
LAST_WINDOW = 20


def read_curve(seed: int) -> list[dict[str, Any]]:
    path = C.history_curve_path(seed)
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                {
                    "epoch": int(raw["epoch"]),
                    "steps": int(raw["steps"]),
                    "lr": float(raw["lr"]),
                    "train_seconds": float(raw["train_seconds"]),
                    "valid_seconds": float(raw["valid_seconds"]),
                    "train_loss": float(raw["train_loss"]),
                    "is_best": raw["is_best"].strip().lower() == "true",
                    "valid_metric": float(raw["valid_metric"]),
                    "valid_aux": None if raw["valid_aux"].strip() == "" else float(raw["valid_aux"]),
                }
            )
    if not rows:
        raise RuntimeError(f"empty curve for seed {seed}")
    epochs = [r["epoch"] for r in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"seed {seed}: curve epochs are not 1..N contiguous")
    return rows


def audit_seed(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = C.history_run(seed)
    curve_path = C.history_curve_path(seed)
    curve_sha = C.sha256_file(curve_path)
    if curve_sha != run.get("curve_sha256"):
        raise RuntimeError(f"seed {seed}: curve.csv sha256 {curve_sha} != run.json {run.get('curve_sha256')}")
    ckpt_path = C.history_checkpoint_path(seed)
    ckpt_sha = C.sha256_file(ckpt_path)
    if ckpt_sha != run.get("checkpoint_sha256"):
        raise RuntimeError(f"seed {seed}: checkpoint sha256 mismatch")

    curve = read_curve(seed)
    metrics = np.asarray([r["valid_metric"] for r in curve], dtype=np.float64)
    losses = np.asarray([r["train_loss"] for r in curve], dtype=np.float64)
    lrs = np.asarray([r["lr"] for r in curve], dtype=np.float64)
    best_idx = int(np.argmax(metrics))
    best_epoch = int(curve[best_idx]["epoch"])
    best_metric = float(metrics[best_idx])
    if best_epoch != int(run["best_epoch"]):
        raise RuntimeError(f"seed {seed}: argmax valid epoch {best_epoch} != run.json {run['best_epoch']}")
    if abs(best_metric - float(run["best_metric"])) > 1e-12:
        raise RuntimeError(f"seed {seed}: argmax valid metric {best_metric} != run.json {run['best_metric']}")

    epochs_run = int(run["epochs_run"])
    if epochs_run != len(curve):
        raise RuntimeError(f"seed {seed}: epochs_run {epochs_run} != curve length {len(curve)}")

    n = len(metrics)
    tail = slice(max(0, n - LAST_WINDOW), n)
    last_window_epochs = [int(r["epoch"]) for r in curve[tail]]
    tail_best_epoch = int(curve[tail][int(np.argmax(metrics[tail]))]["epoch"])
    new_best_in_window = [int(r["epoch"]) for r in curve[tail] if r["is_best"] and int(r["epoch"]) > last_window_epochs[0]]

    summary = {
        "seed": seed,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_valid_mean_ap": best_metric,
        "best_valid_mean_ap_pct": 100.0 * best_metric,
        "valid_mean_ap_epoch_1": float(metrics[0]),
        "valid_mean_ap_epoch_100": float(metrics[99]) if n >= 100 else None,
        "valid_mean_ap_last": float(metrics[-1]),
        "valid_mean_ap_tail_min": float(metrics[tail].min()),
        "valid_mean_ap_tail_max": float(metrics[tail].max()),
        "valid_mean_ap_tail_argmax_epoch": tail_best_epoch,
        "valid_mean_ap_delta_last20": float(metrics[-1] - metrics[max(0, n - LAST_WINDOW)]),
        "new_best_in_last20": bool(new_best_in_window),
        "new_best_epochs_in_last20": ";".join(str(e) for e in new_best_in_window),
        "train_loss_epoch_1": float(losses[0]),
        "train_loss_epoch_100": float(losses[99]) if n >= 100 else None,
        "train_loss_last": float(losses[-1]),
        "train_loss_delta_last20": float(losses[-1] - losses[max(0, n - LAST_WINDOW)]),
        "lr_first": float(lrs[0]),
        "lr_epoch_100": float(lrs[99]) if n >= 100 else None,
        "lr_last": float(lrs[-1]),
        "train_seconds_total": float(sum(r["train_seconds"] for r in curve)),
        "valid_seconds_total": float(sum(r["valid_seconds"] for r in curve)),
        "curve_sha256": curve_sha,
        "checkpoint_sha256": ckpt_sha,
        "early_stop_reached": bool(epochs_run < int(run["recipe"]["max_epochs"])),
        "recipe": run["recipe"],
        "parameters": int(run["parameters"]),
        "seconds": float(run["seconds"]),
        "best_epoch_selection_consistent": True,
    }

    snapshot: list[dict[str, Any]] = []
    by_epoch = {r["epoch"]: r for r in curve}
    for ep in SNAPSHOT_EPOCHS:
        row = by_epoch.get(ep)
        snapshot.append(
            {
                "seed": seed,
                "epoch": ep,
                "present": row is not None,
                "train_loss": None if row is None else row["train_loss"],
                "valid_mean_ap": None if row is None else row["valid_metric"],
                "valid_mean_ap_pct": None if row is None else 100.0 * row["valid_metric"],
                "lr": None if row is None else row["lr"],
                "is_best": None if row is None else row["is_best"],
                "note": "" if row is not None else "MISSING_EPOCH_NOT_RUN",
            }
        )
    full = []
    for r in curve:
        full.append(
            {
                "seed": seed,
                "epoch": r["epoch"],
                "lr": r["lr"],
                "train_loss": r["train_loss"],
                "valid_mean_ap": r["valid_metric"],
                "valid_mean_ap_pct": 100.0 * r["valid_metric"],
                "is_best": r["is_best"],
                "train_seconds": r["train_seconds"],
                "valid_seconds": r["valid_seconds"],
            }
        )
    return {"summary": summary, "snapshot": snapshot, "full": full, "curve": curve, "run": run}


def write_csv(path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(results: list[dict[str, Any]], out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for res in results:
        seed = res["summary"]["seed"]
        curve = res["curve"]
        ep = [r["epoch"] for r in curve]
        axes[0].plot(ep, [r["train_loss"] for r in curve], color=colors[seed], label=f"seed {seed}")
        axes[1].plot(
            ep,
            [100.0 * r["valid_metric"] for r in curve],
            color=colors[seed],
            label=f"seed {seed}",
        )
        best_epoch = res["summary"]["best_epoch"]
        best_val = res["summary"]["best_valid_mean_ap_pct"]
        axes[1].scatter([best_epoch], [best_val], color=colors[seed], marker="o", s=28, zorder=5)
        axes[1].annotate(
            f"{best_val:.2f}",
            (best_epoch, best_val),
            textcoords="offset points",
            xytext=(-6, 6),
            fontsize=8,
            color=colors[seed],
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss (mean over batches)")
    axes[0].set_title("Fast v1 Peptides-func: train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Validation mean AP (%)")
    axes[1].set_title("Fast v1 Peptides-func: Validation mean AP")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    results = []
    run_rows = []
    snapshot_rows = []
    full_rows = []
    for seed in C.LEGACY_SEEDS:
        res = audit_seed(seed)
        results.append(res)
        summary = dict(res["summary"])
        summary.pop("recipe")
        run_rows.append(
            {
                "seed": seed,
                "parameters": res["summary"]["parameters"],
                "seconds": res["summary"]["seconds"],
                "epochs_run": res["summary"]["epochs_run"],
                "best_epoch": res["summary"]["best_epoch"],
                "best_valid_mean_ap": res["summary"]["best_valid_mean_ap"],
                "peak_lr": res["run"]["recipe"]["peak_lr"],
                "weight_decay": res["run"]["recipe"]["weight_decay"],
                "dropout": res["run"]["recipe"]["dropout"],
                "batch_size": res["run"]["recipe"]["batch_size"],
                "max_epochs": res["run"]["recipe"]["max_epochs"],
                "min_epochs": res["run"]["recipe"]["min_epochs"],
                "patience": res["run"]["recipe"]["patience"],
                "scheduler_horizon_epochs": res["run"]["recipe"]["scheduler_horizon_epochs"],
                "gpu_peak_allocated_bytes": res["run"].get("gpu_peak_allocated_bytes"),
                "gpu_peak_reserved_bytes": res["run"].get("gpu_peak_reserved_bytes"),
                "checkpoint_sha256": res["summary"]["checkpoint_sha256"],
                "curve_sha256": res["summary"]["curve_sha256"],
            }
        )
        snapshot_rows.extend(res["snapshot"])
        full_rows.extend(res["full"])

    write_csv(C.ANALYSIS / "history_runs.csv", run_rows)
    write_csv(C.ANALYSIS / "learning_curves.csv", snapshot_rows)
    write_csv(C.ANALYSIS / "learning_curve_full.csv", full_rows)
    summary_rows = []
    for res in results:
        row = {k: v for k, v in res["summary"].items() if k != "recipe"}
        summary_rows.append(row)
    write_csv(C.ANALYSIS / "learning_curve_summary.csv", summary_rows)
    plot(results, C.ANALYSIS / "learning_curves.png")

    payload = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "source": "history/peptide_seed{0,1,2}/{curve.csv,run.json,checkpoint.pt} (copied from res Fast v1 runs)",
        "checks": {
            "curve_sha256_matches_run_json": True,
            "checkpoint_sha256_matches_run_json": True,
            "argmax_curve_epoch_equals_recorded_best_epoch": True,
            "curve_length_equals_epochs_run": True,
        },
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "last_window_epochs": LAST_WINDOW,
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "learning_curve_audit.json", payload)
    for row in summary_rows:
        C.log(
            f"seed {row['seed']}: best {row['best_valid_mean_ap_pct']:.2f}% at epoch {row['best_epoch']}; "
            f"tail delta {100*row['valid_mean_ap_delta_last20']:+.2f} pp; "
            f"new best in last 20: {row['new_best_in_last20']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
