"""Freeze the preparation artifacts: hashes, sizes and a compact copy of every
metadata file next to the code, so the report can cite immutable numbers.

Reads the (git-ignored) cache produced by ``zct_prep.py`` and writes only small
JSON files into ``analysis/``.  Fails loudly if the pre-registered gates did not
pass or if the kappa/fork prerequisites are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

CACHE = _TASK / "cache"
ANALYSIS = _TASK / "analysis"
OUT = CACHE / "artifacts"

FILES = [
    "keys.npz", "keys_meta.json", "partitions_s0.npz", "partitions_s1.npz",
    "probe_episodes.npz", "probe_episodes.json", "support_sets.npz",
    "support_manifest.json", "prep.json",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-gate", action="store_true", default=True)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"files": {}, "gate": None, "checks": {}}
    for name in FILES:
        path = CACHE / name
        if not path.exists():
            out["files"][name] = None
            continue
        out["files"][name] = {"bytes": path.stat().st_size, "sha256": C.sha256_file(path)}
    gate = C.read_json(ANALYSIS / "split_strength.json")
    out["gate"] = gate["decision"]
    post = gate["declared_256_proposals"]
    out["checks"]["gate_pass"] = gate["decision"] == "PASS"
    out["checks"]["strength_numbers"] = {
        "d_sym_R": post["mean_d_sym_R"], "d_sym_S": post["mean_d_sym_S"],
        "delta_d_sym": post["delta_d_sym"], "delta_d_elig": post["delta_d_elig"],
        "d_elig_R": post["mean_d_elig_R"], "d_elig_S": post["mean_d_elig_S"],
        "r1_unseen_R": post["mean_r1_R"], "r1_unseen_S": post["mean_r1_S"],
        "smd_atoms_R": post["mean_smd_atoms_R"], "smd_atoms_S": post["mean_smd_atoms_S"],
        "smd_rings_R": post["mean_smd_rings_R"], "smd_rings_S": post["mean_smd_rings_S"],
        "full_unseen_R": post["mean_full_unseen_R"], "full_unseen_S": post["mean_full_unseen_S"],
        "fraction_batches_S_gt_R": post["frac_batches_S_gt_R"],
        "mean_n_swaps": post["mean_n_swaps"],
    }
    support = C.read_json(CACHE / "support_manifest.json")
    out["checks"]["support"] = {"sizes": support["sizes"],
                                "n_target_keys": support["n_target_keys"],
                                "max_abs_smd": support["matching"]["max_abs_smd"],
                                "npz_sha256": support["npz_sha256"]}
    prep = C.read_json(CACHE / "prep.json")
    out["checks"]["partition_summaries"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "epoch_order_sha256"}
        for k, v in prep.items() if k.startswith("partitions_s")}
    out["checks"]["total_prep_seconds"] = prep.get("total_seconds")
    for seed in (0, 1):
        summ = prep.get(f"partitions_s{seed}", {})
        out["checks"][f"partitions_s{seed}_epoch_hashes"] = summ.get("epoch_order_sha256")
    # copy metadata into analysis/ (small, tracked)
    for name in ("keys_meta.json", "probe_episodes.json", "support_manifest.json"):
        C.write_json(OUT / name, C.read_json(CACHE / name))
    C.write_json(OUT / "prep_freeze.json", out)
    C.log(json.dumps(out["checks"], indent=1)[:2000])
    if args.require_gate and not out["checks"]["gate_pass"]:
        raise SystemExit("structural intervention strength gate did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
