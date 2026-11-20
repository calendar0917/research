"""One-command runner for the frozen response-dictionary explanation check.

    uv run python tracks/structural-sufficiency/response_dictionary_explanation_check_s0_v1/code/ec_run.py

CPU only.  No training, no new seed, no K sweep.  The output directory defaults to
``results/`` next to the code directory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ec_common as EC  # noqa: E402
import ec_compression as ECOMP  # noqa: E402
import ec_explanation as EX  # noqa: E402
import ec_projection as EP  # noqa: E402
import rd_common as RC  # noqa: E402


def _versions() -> dict[str, str]:
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "rdkit": __import__("rdkit").__version__,
    }


def _worktree_status(out: Path) -> dict[str, Any]:
    """Record whether tracked code was clean at run time (results are new files)."""
    import subprocess

    proc = subprocess.run(["git", "-C", str(EC.REPO_ROOT), "status", "--porcelain"],
                          capture_output=True, text=True, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    result_prefix = str(out.relative_to(EC.REPO_ROOT)) + "/"
    non_results = [line for line in lines if result_prefix not in line]
    return {
        "tracked_worktree_clean_excluding_results": len(non_results) == 0,
        "n_status_lines": len(lines),
        "n_non_results_status_lines": len(non_results),
        "non_results_status": non_results[:20],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EC.ROUND_DIR / "results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    ctx = EC.prepare(device=args.device)
    proj = EP.run(ctx, out)
    exp = EX.run(ctx, proj, out)
    compression = ECOMP.build(ctx, proj)
    RC.write_json(out / "compression.json", compression)

    summary = {
        "round_id": ctx["config"]["round_id"],
        "protocol_hash": EC.protocol_hash(),
        "base_commit": ctx["config"]["base_commit"],
        "executed_commit": RC.git_commit(),
        "checkpoint": str(ctx["checkpoint"].relative_to(EC.REPO_ROOT)),
        "checkpoint_sha256": ctx["checkpoint_sha256"],
        "n_parameters": compression["n_model_parameters"],
        "projection": {
            "decision": proj["decision"],
            "checks": {
                "n_certificate_rows": len(proj["cert_rows"]),
                "n_fw64_pass": sum(1 for r in proj["cert_rows"] if r["pass_fw64"]),
                "n_slsqp_fw_pass": sum(1 for r in proj["cert_rows"] if r["pass_slsqp_fw"]),
                "n_slsqp_uniform_pass": sum(1 for r in proj["cert_rows"] if r["pass_slsqp_uniform"]),
                "n_both_qualified": sum(1 for r in proj["cert_rows"] if r["both_qualified"]),
                "n_unresolved_primary": sum(1 for r in proj["cert_rows"]
                                             if r["main_start"] == "fw64_fallback"),
                "n_primary_fw": sum(1 for r in proj["cert_rows"] if r["main_start"] == "fw"),
                "n_primary_uniform": sum(1 for r in proj["cert_rows"]
                                          if r["main_start"] == "uniform"),
                "n_ambiguous_alpha_only": sum(1 for r in proj["cert_rows"]
                                              if r["ambiguous_panel_solution"]),
            },
            "panel_underdetermination": proj["pua_summary"],
            "replay_fw64_max_abs_diff_vs_saved": max(
                proj["replay"]["checks"]["native_resp"][s]["max_abs_diff_raw_vs_saved"]
                for s in ("train", "valid", "test")),
            "fw64_predictions_max_abs_diff_vs_saved": max(
                value for key, value in
                EC.check_fw_predictions(ctx, {
                    f"{name}_{split}_pred_raw":
                        proj["predictions"][f"{name}|fw64"]["pred_raw"][split]
                    for name in EC.DICT_NAMES for split in EC.SPLITS}).items()
                if key != "max_abs_diff"),
        },
        "explanation": {
            "mapping": exp["mapping"],
            "coverage": exp["coverage"],
            "closure": exp["closure_summary"],
            "identity": exp["identity"],
            "D_RESPONSE_HP_CERTIFIED_ALL": exp["summary"]["D_RESPONSE_HP_CERTIFIED"]["ALL"],
            "labels": exp["summary"]["labels"],
        },
        "compression": compression,
        "seconds": time.perf_counter() - t0,
        "test_use": ctx["config"]["test_use"],
        "gpu_a100_hours": 0.0,
    }
    RC.write_json(out / "summary.json", summary)

    source_files = list(sorted(CODE_DIR.glob("ec_*.py"))) + [
        EC.PROTOCOL_PATH,
        EC.CONFIG_PATH,
    ]
    output_files = list(sorted(out.glob("*.csv"))) + list(sorted(out.glob("*.json"))) \
        + list(sorted(out.glob("*.npz"))) + list(sorted(out.glob("*.md")))
    manifest = {
        "round_id": ctx["config"]["round_id"],
        "protocol_hash": EC.protocol_hash(),
        "base_commit": ctx["config"]["base_commit"],
        "executed_commit": RC.git_commit(),
        "source_rounds": {
            "parent": ctx["config"]["parent_round"],
            "parent_commit": ctx["follow"].get("parent_commit"),
            "followup": ctx["config"]["followup_round"],
        },
        "code_sha256": {p.name: RC.sha256_file(p) for p in source_files},
        "config": ctx["config"],
        "data_identity": {
            "fragment_cache_sha256": RC.fragment_cache(ctx["config"]["dataset"])
            ["fragment_cache_sha256"],
            "official_cache_sha256": RC.dataset_cache(ctx["config"]["dataset"])["cache_sha256"],
            "checkpoint_sha256": ctx["checkpoint_sha256"],
            "pairs_sha256": RC.sha256_file(
                EC.REPO_ROOT / ctx["config"]["explanation"]["pairs_csv"]),
        },
        "regime": _versions(),
        "worktree_status": _worktree_status(out),
        "seconds": time.perf_counter() - t0,
        "gpu_a100_hours": 0.0,
        "artifacts": {p.name: RC.sha256_file(p) for p in output_files},
        "test_use": ctx["config"]["test_use"],
        "note": ("CPU-only frozen replay/linear algebra plus SciPy SLSQP on the unchanged panel "
                 "objective. No training, no optimizer update, no new seed, no K sweep."),
    }
    RC.write_json(out / "run_manifest.json", manifest)
    print(json.dumps(RC.jsonable({
        "projection": summary["projection"]["decision"]["label"],
        "panel": summary["projection"]["panel_underdetermination"]["label"],
        "coverage_interpretable": f"{len(exp['summary']['coverage'])} cells",
        "n_interpretable_pairs": exp["summary"]["n_pairs_interpretable"],
        "closure": exp["closure_summary"]["label"],
        "seconds": summary["seconds"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
