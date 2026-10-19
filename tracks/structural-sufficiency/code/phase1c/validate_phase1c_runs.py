"""Validate Phase 1c run provenance and data isolation before analysis/reporting."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1c/zinc"
OUT = REPO / "tracks/structural-sufficiency/results/phase1c/run_validation.json"

EXPECTED = {
    "phase1c_protocol_sha256": "545e58037386fde248a84b9ad104cbb7dab8ccd0f961a0392fdb933a14547ca5",
    "phase1c_config_sha256": "60c50f369983d97a7dec9c90f9f35d5c2497f5a40fdd8a07fce38c046bddeb82",
    "phase1c_cpu_gates_sha256": "aa91573e138f8910b92f96d31235ccb67a571320fd8edf2b0a0a14272ff75c4a",
    "size_features_sha256": "8a20858e8b660f2b7befee0ba36e17a339667393aa4d40cabb04a05c5e6fd434",
    "split_sha256": "fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49",
    "labels_sha256": "631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78",
    "cache_sha256": "76c1749277f1c4019f8b94b5a3a9bc7b9f83abb8b11b7feed60449f47a9ed84c",
}


def main() -> int:
    files = sorted(RUNS.glob("*.json"))
    if not files:
        raise SystemExit(f"no run JSONs in {RUNS}")
    rows = []
    errors: list[str] = []
    monitor_sets: dict[str, tuple[int, ...]] = {}
    for path in files:
        d = json.loads(path.read_text())
        prov = d["provenance"]
        name = f"{d['model']}_seed{d['seed']}" + (f"_{path.stem.split('_')[-1]}" if path.stem.endswith("_long") else "")
        rows.append(
            {
                "file": str(path.relative_to(REPO)),
                "model": d["model"],
                "seed": d["seed"],
                "params": d["parameters"],
                "flops_forward": d["flops"]["flops_forward"],
                "best_epoch": d["best_epoch"],
                "epochs_planned": d["config"]["epochs"],
                "epochs_run": len(d.get("curves", [])),
                "monitor_mae": d["best_metrics"]["monitor"]["mae"],
                "monitor_rmse": d["best_metrics"]["monitor"]["rmse"],
                "info_keys": sorted(d["information_access"]),
                "gpu_seconds": d.get("gpu_seconds"),
            }
        )
        if d["n_train"] != 8000 or d["n_monitor"] != 1000 or d["n_reserved_not_scored"] != 1000:
            errors.append(f"{name}: unexpected row counts")
        vault = prov["label_vault"]
        if vault.get("reserved_labels_retained") != 0:
            errors.append(f"{name}: reserved labels retained")
        if vault.get("official_valid_test_labels_retained") != 0:
            errors.append(f"{name}: official valid/test labels retained")
        if vault.get("labels_retained") != 9000:
            errors.append(f"{name}: expected 9000 Fit+Monitor labels")
        for key, expected in EXPECTED.items():
            if prov.get(key) != expected:
                errors.append(f"{name}: {key} mismatch")
        monitor_sets[name] = tuple(int(x) for x in d["monitor_mol_index"])
        keys = set(d["information_access"])
        if "mol_size_feat" not in keys:
            errors.append(f"{name}: missing mol_size_feat")
        if d["model"] == "S_cap_size" and any(k.startswith("root_edge") for k in keys):
            errors.append(f"{name}: S_cap_size saw root edges")
        if d["model"] == "R1_marginal_size" and "root_edge_index" in keys:
            errors.append(f"{name}: marginal saw root endpoints")
        if d["model"] in {"R1_size", "M_keep_edge"} and "root_edge_index" not in keys:
            errors.append(f"{name}: relation model missing root_edge_index")
    base = next(iter(monitor_sets.values()))
    for name, s in monitor_sets.items():
        if s != base:
            errors.append(f"{name}: monitor molecule order differs")

    out = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "n_runs": len(rows),
        "monitor_order_identical": all(s == base for s in monitor_sets.values()),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "n_runs", "monitor_order_identical", "errors")}, indent=2))
    print(f"wrote {OUT}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
