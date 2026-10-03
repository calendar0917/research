"""Identity manifest for the G0 transfer audit (read-only, local CPU)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path("/home/calendar/code/research")
SRC = REPO / "tracks/ksvd/results/zinc_topology_crossfit_diagnostic_v1"
ZJD = REPO / "tracks/ksvd/results/zinc_joint_dictionary_decision_v1"
HANDOFF = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    files = {
        "F_A_state.pt": SRC / "F_A_state.pt",
        "F_B_state.pt": SRC / "F_B_state.pt",
        "F_A.json": SRC / "F_A.json",
        "F_B.json": SRC / "F_B.json",
        "readouts.json": SRC / "readouts.json",
        "subfold_index.npz": SRC / "prep/subfold_index.npz",
        "subfold_split.json": SRC / "prep/subfold_split.json",
        "fold_A.npz": SRC / "prep/fold_A.npz",
        "fold_B.npz": SRC / "prep/fold_B.npz",
        "train.npz": HANDOFF / "train.npz",
        "zjd_split.json": ZJD / "prep/split.json",
        "train_topology_features.csv": REPO / "tracks/ksvd/results/zinc_topology_cache/train_topology_features.csv",
    }
    payload = {
        "protocol_version": "zinc-g0-transfer-audit-v1",
        "built_on": "task/zinc-g0-transfer-audit-v1",
        "source_task": "task/zinc-topology-crossfit-diagnostic-v1",
        "source_result_commit": "8134630",
        "source_science_commit": "240bd2b",
        "base_ref_commit": "ac8ef5b",
        "official_test_loaded": False,
        "official_valid_reevaluated": False,
        "artifacts": {k: {"path": str(p.relative_to(REPO)), "sha256": sha(p), "bytes": p.stat().st_size,
                          "present": True} for k, p in files.items()},
        "frozen_encoding": {
            "structural_code": "z = model.code(phi65) = [c~ ; alpha_res] width 32+1",
            "common_dim": 1, "residual_dim": 32, "support_eps": 1e-8,
            "support_size_note": "tied IHT s=8 -> every node has exactly 8 nonzeros",
            "node_struct_key": "residual support bitmask",
            "edge_struct_key": "unordered pair of endpoint support bitmasks",
            "descriptor_key": "dict_phi[65] rounded to 6 decimals, float32 bytes, 16-byte blake2b",
            "semantics": {"node": "dict_atom", "edge": "env_bond_type"},
            "real_edges": "deduped from env_bond_u/v occurrences; bond type asserted consistent",
        },
        "frequencies": {"unit": "distinct canonical training molecule",
                        "seen_threshold": 5, "high": 0.25, "low": 0.05},
        "git": {"head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "branch": subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True).stdout.strip()},
        "resource": {"device": "cpu", "threads": 8, "gpu": 0, "new_remote_runs": 0, "new_training": 0},
    }
    (OUT / "identity.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v["sha256"][:12] for k, v in payload["artifacts"].items()}, indent=2))
    print("git", payload["git"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
