"""Regenerate manifest.json for zinc-task-dictionary-and-cycle-witness-seed0-v1.

Trains nothing; hashes the code, docs and every result file so the round can be
audited without trusting the report text.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # tracks/ksvd/results/<round> -> repo root

CODE_FILES = [
    "tracks/ksvd/experiments/luyin16/zinc_task_dictionary_and_cycle_witness_seed0_v1.py",
    "tracks/ksvd/experiments/luyin16/zinc_cycle_witness_seed0_v1.py",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/jm_flip_replay.py",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/make_manifest.py",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/PROTOCOL.md",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/METHOD_CONTRACT.md",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/protocol.json",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/REPORT.md",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/DECISION.md",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/ERRATA.md",
    "tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/EXECUTION.md",
]

SKIP_RESULT_FILES = {"manifest.json"}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def entry(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> int:
    analysis = json.loads((HERE / "analysis.json").read_text())
    bounds = json.loads((HERE / "cycle_bounds.json").read_text())
    smoke = json.loads((HERE / "smoke_checks.json").read_text())

    manifest: dict[str, object] = {
        "protocol_version": "zinc-task-dictionary-and-cycle-witness-seed0-v1",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": "task/zinc-task-dictionary-and-cycle-witness-seed0-v1",
        "training_revision": "77c40dda9d5b730a64ea653b63b053a11460993d",
        "remote_runs": {
            "ztdw-d": {
                "run_id": "ztdw-d-20261004-092910-07d6d373",
                "slurm_job_id": "55894",
                "node": "c05",
                "regime": "res-2 / res2-cu124 / c05 / A100-PCIE-40GB / driver 525.85.12 / torch 2.5.1+cu124",
            },
            "ztdw-m": {
                "run_id": "ztdw-m-20261004-092927-87de7d0e",
                "slurm_job_id": "55895",
                "node": "c05",
                "regime": "res-2 / res2-cu124 / c05 / A100-PCIE-40GB / driver 525.85.12 / torch 2.5.1+cu124",
            },
        },
        "anchor_hashes": {
            "new_fit_idx_sha256": analysis["fold"]["fit_idx_sha256"],
            "new_dev_idx_sha256": analysis["fold"]["dev_idx_sha256"],
            "shared_init_sha256": analysis["init_pair_check"]["shared_init_sha256"],
            "smoke_shared_state_sha256": smoke["shared_body"]["shared_state_hash"],
            "old_cpu_t25_fit_sha256": bounds["source"]["T25_sha256_fit"],
            "old_cpu_fit_idx_sha256": bounds["source"]["fit_idx_sha256"],
            "old_reported_irreducible_l1_per_row": bounds["old_reported"]["irreducible_l1_per_row"],
            "old_reported_group_kle3": bounds["old_reported"]["irreducible_per_group"]["k<=-3"][
                "irreducible_l1_per_row"
            ],
        },
        "code": {},
        "result_files": {},
    }
    for relative in CODE_FILES:
        path = REPO / relative
        if path.exists():
            manifest["code"][relative] = entry(path)
    for path in sorted(HERE.iterdir()):
        if path.is_dir() or path.name in SKIP_RESULT_FILES:
            continue
        manifest["result_files"][path.name] = entry(path)
    for path in sorted((HERE / "figures").glob("*.png")):
        manifest["result_files"][f"figures/{path.name}"] = entry(path)

    # schedule hash from both meta files (must agree).
    schedule_hashes = {
        arm: json.loads((HERE / f"{arm}_meta.json").read_text())["schedule_sha256"] for arm in ("D", "M")
    }
    assert schedule_hashes["D"] == schedule_hashes["M"], schedule_hashes
    manifest["anchor_hashes"]["batch_schedule_sha256"] = schedule_hashes["D"]

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(
        f"manifest.json: {len(manifest['code'])} code/doc files, "
        f"{len(manifest['result_files'])} result files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
