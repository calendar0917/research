"""Validate Phase 1d run provenance and data isolation before analysis/reporting."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1d/zinc"
RESULTS = REPO / "tracks/structural-sufficiency/results/phase1d"
OUT = RESULTS / "run_validation.json"

EXPECTED = {
    "size_features_sha256": "8a20858e8b660f2b7befee0ba36e17a339667393aa4d40cabb04a05c5e6fd434",
    "split_sha256": "fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49",
    "labels_sha256": "631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78",
    "cache_sha256": "76c1749277f1c4019f8b94b5a3a9bc7b9f83abb8b11b7feed60449f47a9ed84c",
}
PARAMS = {"R_incidence": 115_952, "R1_size_cap": 453_129, "R_incidence_cap": 436_841}


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = sorted(RUNS.glob("*.json"))
    if not files:
        raise SystemExit(f"no run JSONs in {RUNS}")
    rows = []
    errors: list[str] = []
    monitor_sets: dict[str, tuple[int, ...]] = {}
    current = {
        "phase1d_protocol_sha256": _sha(REPO / "tracks/structural-sufficiency/notes/phase1d_protocol.md"),
        "phase1d_config_sha256": _sha(REPO / "tracks/structural-sufficiency/configs/phase1d.yaml"),
        "phase1d_cpu_gates_sha256": _sha(RESULTS / "cpu_gates_phase1d.json"),
    }
    for path in files:
        d = json.loads(path.read_text())
        prov = d["provenance"]
        name = f"{d['model']}_seed{d['seed']}" + (f"_{path.stem.split('_')[-1]}" if path.stem.endswith("_smoke") else "")
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
                "scope": d.get("scope"),
            }
        )
        if d.get("protocol_id") != "ssrc-phase1d-zinc-v1":
            errors.append(f"{name}: wrong protocol_id")
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
        for key, expected in current.items():
            if expected is not None and prov.get(key) != expected:
                errors.append(f"{name}: {key} does not match current frozen file")
        if PARAMS.get(d["model"]) is not None and int(d["parameters"]) != PARAMS[d["model"]]:
            errors.append(f"{name}: unexpected parameter count {d['parameters']}")
        monitor_sets[name] = tuple(int(x) for x in d["monitor_mol_index"])
        keys = set(d["information_access"])
        if "mol_size_feat" not in keys or "root_edge_index" not in keys:
            errors.append(f"{name}: missing size/root-edge information key")
        if d["model"] == "R_incidence" and "root_edge_mol" in keys:
            errors.append(f"{name}: incidence model leaked marginal key")
        scope = d.get("scope") or {}
        for k in ("official_valid_test_labels_read", "internal_reserved_labels_read"):
            if scope.get(k) is not False:
                errors.append(f"{name}: scope.{k} is not False")
        if scope.get("reserved_labels_retained") not in (0, None):
            errors.append(f"{name}: scope reserved labels retained")

    base = next(iter(monitor_sets.values()))
    for name, s in monitor_sets.items():
        if s != base:
            errors.append(f"{name}: monitor molecule order differs")

    out = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "n_runs": len(rows),
        "monitor_order_identical": all(s == base for s in monitor_sets.values()),
        "current_file_hashes": current,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "n_runs", "monitor_order_identical", "errors")}, indent=2))
    print(f"wrote {OUT}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
