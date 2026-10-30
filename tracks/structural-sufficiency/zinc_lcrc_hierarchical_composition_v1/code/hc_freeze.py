"""Write the frozen provenance manifest for LCRC-HC3-v1.

Hashes the round code, protocol, configs, structure caches, CPU gates and every
upstream file the round reads.  Run after the CPU gates and before deployment.
"""

from __future__ import annotations

import json

import hc_common as C

FILES = {
    "protocol": C.ROUND / "PROTOCOL.md",
    "config": C.CONFIGS / "frozen_protocol.json",
    "code_common": C.HERE / "hc_common.py",
    "code_structure": C.HERE / "hc_structure.py",
    "code_models": C.HERE / "hc_models.py",
    "code_data": C.HERE / "hc_data.py",
    "code_train": C.HERE / "hc_train.py",
    "code_gates": C.HERE / "hc_gates.py",
    "code_erratum": C.HERE / "hc_erratum.py",
    "cpu_gates": C.GATES / "cpu_gates.json",
    "structure_train": C.CACHE / "hc3_train.pt",
    "structure_valid": C.CACHE / "hc3_valid.pt",
    "frozen_lc_models": C.LCRC_CODE / "lc_models.py",
    "frozen_lc_data": C.LCRC_CODE / "lc_data.py",
    "frozen_lc_common": C.LCRC_CODE / "lc_common.py",
    "frozen_norm": C.LCRC_NORM,
    "split_a": C.SPLIT_FILE,
    "ssl_seed0": C.LC.P3_SSL[0],
    "ssl_seed1": C.LC.P3_SSL[1],
    "lcrcA_s0_best": C.LCRC / "runs/lcrcA_s0/best.pt",
    "lcrcA_s1_best": C.LCRC / "runs/lcrcA_s1/best.pt",
    "lcrcA_s0_train_json": C.LCRC / "runs/lcrcA_s0/train.json",
    "lcrcA_s1_train_json": C.LCRC / "runs/lcrcA_s1/train.json",
    "dccl_per_molecule": C.DCCL_PERMOL,
    "cycle_audit_valid": C.CYCLE_AUDIT_VALID,
    "stage4_contrasts": C.SUFF / "zinc_compositional_research_tree_v1/analysis/contrasts.json",
}


def main() -> None:
    manifest = {
        "round": C.ROUND_ID,
        "created": C.now(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "files": {},
    }
    missing = []
    for name, path in FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        manifest["files"][name] = {"path": str(path.relative_to(C.REPO)),
                                   "sha256": C.sha256_file(path)}
    manifest["missing"] = missing
    C.write_json(C.CONFIGS / "freeze_manifest.json", manifest)
    C.log(f"wrote {C.CONFIGS / 'freeze_manifest.json'} ({len(manifest['files'])} files, "
          f"{len(missing)} missing)")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
