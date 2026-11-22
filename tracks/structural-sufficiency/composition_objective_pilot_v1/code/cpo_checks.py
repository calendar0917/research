"""Pre-training engineering gate for the composition-objective pilot.

Runs the audit-derived checks, the pair-definition checks, and a tiny CPU
trainer smoke for each of the three arms.  Writes ``analysis/checks.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import cpo_common as C


def _check(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def main(argv: Sequence[str] | None = None) -> int:
    cfg = C.load_config()
    pack = C.load_pack(cfg)
    checks: dict[str, Any] = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack["cache_sha256"],
        "cache_content_sha256": pack.get("content_sha256"),
        "test_policy": "no official Test label read",
    }

    audit_path = C.analysis_dir() / "loss_decomposition.json"
    audit = C.read_json(audit_path)
    identity = {}
    for name, arm in audit["arms"].items():
        identity[name] = {
            split: arm["splits"][split]["identity_max_abs"]
            for split in ("train", "valid")
        }
    checks["loss_identity"] = _check(
        "L_cut = L_graph + Var_cut per molecule",
        all(v < 1e-10 for arm in identity.values() for v in arm.values()),
        identity)

    prov = audit["pair_delta_provenance"]
    matches = {}
    for name, rec in prov.items():
        for scope, val in rec.get("matches_old_pair_metrics_json", {}).items():
            matches[f"{name}:{scope}"] = {
                "dmae": val["abs_diff_delta_mae"],
                "drmse": val["abs_diff_delta_rmse"],
            }
    checks["old_pair_metrics_are_whole_molecule_delta"] = _check(
        "recomputed whole-molecule delta equals the old pair_metrics.json",
        all(v["dmae"] < 1e-12 and v["drmse"] < 1e-12 for v in matches.values()) if matches else False,
        matches)

    checks["source_audit_complete"] = _check(
        "required source snippets located",
        all(rec.get("found") for rec in audit["source_audit"].values()),
        {k: rec.get("found") for k, rec in audit["source_audit"].items()})

    local = C.read_json(C.analysis_dir() / "local_context_support.json")
    checks["local_context_identity"] = _check(
        "renumbering invariance and complete-C reconstruction",
        bool(local["identity"]["renumber_invariance_ok"])
        and local["identity"]["complete_C_key_reconstruction_matches"]
        == local["identity"]["complete_C_key_reconstruction_checks"],
        local["identity"])

    # tiny CPU trainer smoke (not a result, just an engineering gate)
    smoke = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, family, arm in (("F_GRAPH", "parent", "FACTOR"),
                                  ("B_GRAPH", "csoi", "CSOI_ADD"),
                                  ("C_GRAPH", "csoi", "CSOI_GLOBAL")):
            rid = str(Path(tmp) / name)
            cmd = [sys.executable, str(C.CODE_DIR / "cpo_train.py"),
                   "--run-id", rid, "--family", family, "--arm", arm,
                   "--epochs", "1", "--device", "cpu", "--seed", "0",
                   "--train-subset", "16", "--valid-subset", "8", "--max-steps", "2"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            man_path = Path(rid) / "manifest.json"
            man = C.read_json(man_path) if man_path.is_file() else {}
            curves = C.read_csv(Path(rid) / "curves.csv") if (Path(rid) / "curves.csv").is_file() else []
            smoke[name] = {
                "returncode": int(proc.returncode),
                "manifest_main_loss": man.get("main_loss"),
                "n_curves": len(curves),
                "finite_curve": bool(curves) and all(
                    np.isfinite(float(r.get("valid_rmse", "nan"))) for r in curves),
                "test_keys_in_predictions": bool(
                    (Path(rid) / "predictions_best.npz").is_file() and any(
                        k.startswith("test") for k in np.load(
                            Path(rid) / "predictions_best.npz", allow_pickle=False).files)),
                "stderr_tail": proc.stderr[-500:],
            }
    checks["cpu_trainer_smoke"] = _check(
        "all three arms run finite graph-loss steps and write a checkpoint",
        all(v["returncode"] == 0 and v["manifest_main_loss"] == "graph_level_mean_squared_error"
            and v["finite_curve"] and not v["test_keys_in_predictions"]
            for v in smoke.values()),
        smoke)

    checks["all_ok"] = all(v["ok"] for k, v in checks.items()
                           if isinstance(v, dict) and "ok" in v)
    out_path = C.analysis_dir() / "checks.json"
    C.write_json(out_path, checks)
    print(json.dumps(checks, indent=2)[:4000])
    print(f"wrote {out_path}")
    return 0 if checks["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())