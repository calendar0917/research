"""Hash manifest for zinc-frozen-chemistry-learned-cycle-v1 (local, CPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
FROZEN_DIR = RESULTS_DIR.parent / "zinc_full_cycle_target_decomposition_v1"
PREP_BLOB = RESULTS_DIR.parent / "zinc_joint_dictionary_decision_v1" / "prep" / "fold_objects.npz"
PROTOCOL = RESULTS_DIR.parent.parent / "protocols" / "zinc-frozen-chemistry-learned-cycle-v1.yaml"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    frozen_manifest = json.loads((FROZEN_DIR / "manifest.json").read_text())["files"]
    frozen_check = {}
    for name in sorted(frozen_manifest):
        if "O_seed" in name or "Y_seed" in name:
            path = FROZEN_DIR / name
            frozen_check[name] = {
                "sha256": sha_file(path),
                "matches_frozen_manifest": bool(sha_file(path) == frozen_manifest[name]["sha256"]),
            }

    result_files = sorted(
        p.name for p in RESULTS_DIR.iterdir() if p.is_file() and p.name != "manifest.json"
    )
    files = {name: {"sha256": sha_file(RESULTS_DIR / name), "bytes": (RESULTS_DIR / name).stat().st_size} for name in result_files}

    T = json.loads((RESULTS_DIR / "P_seed0.json").read_text())
    input_manifest = {
        "protocol_version": "zinc-frozen-chemistry-learned-cycle-v1",
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "frozen_source_commit": "87894a2",
        "frozen_dir": str(FROZEN_DIR.relative_to(RESULTS_DIR.parents[1])),
        "frozen_artifacts": frozen_check,
        "split": {
            "fit_idx_sha256": "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea",
            "dev_idx_sha256": "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376",
            "n_fit": 8000,
            "n_dev": 2000,
            "expected_dev_strata": {"k=0": 1926, "k=-1": 65, "k<=-2": 9},
        },
        "prep_blob_sha256": sha_file(PREP_BLOB),
        "decomp_sha256": sha_file(FROZEN_DIR / "target_decomposition.npz"),
        "protocol_sha256": sha_file(PROTOCOL),
        "mu_cycle": -0.0001334074230764987,
        "sigma_cycle": 0.28831626452532244,
        "c_is_forward_input": False,
        "note": "T25 is the existing Full topology25 model input after apply_prep (8000-fit standardization).",
    }
    with __import__("numpy").load(RESULTS_DIR / "T25_all.npz", allow_pickle=False) as z:
        input_manifest["topology25_all_sha256"] = str(z["sha256"][0])
    (RESULTS_DIR / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    manifest = {
        "protocol_version": "zinc-frozen-chemistry-learned-cycle-v1",
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "delivered_branch": "task/zinc-frozen-chemistry-learned-cycle-v1",
        "frozen_artifacts_unchanged": bool(all(v["matches_frozen_manifest"] for v in frozen_check.values())),
        "files": files,
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("files:", len(files), "frozen_unchanged:", manifest["frozen_artifacts_unchanged"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())