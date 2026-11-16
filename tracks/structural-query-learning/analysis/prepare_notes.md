# Preparation notes (sq-pilot-small-v1)

- provenance: `verify-only`; protocol `sq-pilot-small-v1`; commit `d3a71fffdac261fc8c6a7affd4297c214635c4a3`
- The GPU host consumes only these committed label-free caches: it needs neither the OGB raw archives nor RDKit.  `stage=prepare` and every worker re-verify the SHA-256 values pinned in `code/sq_common.py::EXPECTED_CACHE_SHA256`, the official split sizes, the split index hashes and the physical-bond/ring invariants.
- `ogbg-molesol`: 1128 molecules, splits {'train': 902, 'valid': 113, 'test': 113}, 14991 atoms, 15428 physical bonds, 0 multi-component molecules, ring buckets {'0': 6920, '1': 0, '2': 0, '3': 48, '4': 20, '5': 1113, '6': 6742, '7': 117, '8': 31, '9': 0}, labels [-11.600, 1.580]
  - cache: `ogbg_molesol_sq-pilot-cache-v1.npz` sha256 `3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a` (68927 bytes, committed)
  - ecfp: `ogbg_molesol_ecfp4_sq-pilot-cache-v1.npz` sha256 `653fb0190abc4af43822ec643bacbcbdbfdd7555f5e632574c9f61728d574f2e` (51598 bytes, committed)
  - lipo_raw: `ogbg_molesol_sq-pilot-lipo-raw-v1.npz` sha256 `a3ab9f7e22798a8f6fdc7f0913b1136b7cf59d4ffc7168faecc7f8b3094710b1` (54286 bytes, committed)
- `ogbg-mollipo`: 4200 molecules, splits {'train': 3360, 'valid': 420, 'test': 420}, 113568 atoms, 123899 physical bonds, 1 multi-component molecules, ring buckets {'0': 36121, '1': 0, '2': 0, '3': 658, '4': 316, '5': 13380, '6': 62215, '7': 700, '8': 27, '9': 151}, labels [-1.500, 4.500]
  - cache: `ogbg_mollipo_sq-pilot-cache-v1.npz` sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` (451095 bytes, committed)
  - ecfp: `ogbg_mollipo_ecfp4_sq-pilot-cache-v1.npz` sha256 `7990ce4c90a0fdb032897076e19221b62b1361fc98f36fea248cf097bb424ae4` (308526 bytes, committed)
  - lipo_raw: `ogbg_mollipo_sq-pilot-lipo-raw-v1.npz` sha256 `3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3` (406134 bytes, committed)

## Reproducing the caches

From a machine with the official OGB trees under `data/ogb` and RDKit:

```
uv run python tracks/structural-query-learning/code/sq_prepare.py
```

On any other machine (no OGB raw data, no RDKit) the caches are verified instead:

```
uv run python tracks/structural-query-learning/code/sq_prepare.py --verify-only
```

