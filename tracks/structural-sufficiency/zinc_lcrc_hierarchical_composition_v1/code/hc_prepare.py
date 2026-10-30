"""Build the deterministic HC3 structure caches (CPU, no labels, no test)."""

from __future__ import annotations

import json

import hc_common as C
import hc_structure as HS


def main() -> None:
    out = {"round": C.ROUND_ID, "created": C.now(), "structure_version": C.STRUCTURE_VERSION,
           "official_test_loaded": False, "caches": {}}
    for split in ("train", "valid"):
        cache = C.load_cache(split)
        obj = HS.load_split(split, cache=cache, build=True)
        out["caches"][split] = {
            "path": str(HS.struct_cache_path(split)),
            "sha256": C.sha256_file(HS.struct_cache_path(split)),
            "coverage": HS.coverage_stats(obj, cache),
        }
        C.log(f"[prep {split}] {out['caches'][split]['coverage']}")
    C.write_json(C.ANALYSIS / "prep_manifest.json", out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
