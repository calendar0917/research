"""Build the label-free structure caches for all public repeats.

Run once per execution host before the GPU matrix.  Every cache is verified
against the published split bytes before it is written.  For repeat 122 the
content fingerprints are additionally compared with the frozen
``lcrc_molnet_benchmark_pilot_v1`` caches, which proves the structure
pipeline semantics did not drift.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

import dissect_common as C
import dissect_data as D
import dissect_source as S

PILOT_CACHE = C.PILOT / "cache"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build label-free Lipophilicity structure caches")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    spec = D.frozen_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise SystemExit(f"feature spec hash {spec.hash()} != frozen")
    entries: dict[str, Any] = {}
    for repeat in C.REPEATS:
        src = S.load_lipop(repeat, labels="none")
        rows = src["rows"]
        for split in ("train", "valid", "test"):
            cache = D.load_or_build_split_cache(
                repeat, split, rows[split], src["smiles"], spec, rebuild=args.rebuild
            )
            key = f"r{repeat}_{split}"
            entries[key] = {
                "repeat": repeat,
                "split": split,
                "n_molecules": int(cache["n_molecules"]),
                "n_atoms": int(cache["meta"]["n_atoms_total"]),
                "n_bonds": int(cache["meta"]["n_bonds_total"]),
                "n_s1_codes": int(cache["meta"]["n_s1_codes"]),
                "n_s2_paths": int(cache["meta"]["n_s2_paths"]),
                "content_fingerprint": cache["meta"]["content_fingerprint"],
                "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
                "cache_path": C.track_rel(D.cache_path(repeat, split)),
                "cache_sha256": C.sha256_file(D.cache_path(repeat, split)) if D.cache_path(repeat, split).exists() else None,
                "labels_present": bool(cache["meta"]["labels_present"]),
                "feature_spec_hash": cache["meta"]["feature_spec_hash"],
            }
            C.log(f"[r{repeat} {split}] n={cache['n_molecules']} fp={cache['meta']['content_fingerprint'][:16]}")

    # cross-check against the frozen pilot caches for repeat 122
    pilot_check: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        p = PILOT_CACHE / f"Lipophilicity_{split}.pt"
        if not p.exists():
            pilot_check[split] = {"available": False}
            continue
        pilot = torch.load(p, map_location="cpu", weights_only=False)
        ours = entries[f"r122_{split}"]["content_fingerprint"]
        theirs = pilot["meta"]["content_fingerprint"]
        pilot_check[split] = {
            "available": True,
            "pilot_fingerprint": theirs,
            "dissection_fingerprint": ours,
            "identical": bool(ours == theirs),
        }
        if ours != theirs:
            raise SystemExit(f"repeat122/{split}: rebuilt structure cache differs from frozen pilot cache")

    payload = {
        "schema": "lcrc-lipo-dissection-cache-manifest-v1",
        "created_utc": C.utcnow(),
        "feature_spec_hash": spec.hash(),
        "cache_version": D.CACHE_VERSION,
        "entries": entries,
        "pilot_repeat122_crosscheck": pilot_check,
        "pilot_crosscheck_all_identical": all(v.get("identical", False) for v in pilot_check.values()),
    }
    C.write_json(C.ANALYSIS / "cache_manifest.json", payload)
    C.log("wrote cache_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
