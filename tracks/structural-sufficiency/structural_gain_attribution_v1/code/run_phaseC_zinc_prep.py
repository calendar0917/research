"""Phase C prep: build ``Z`` and ``Z-CF`` for PyG ZINC (subset=True) and freeze them.

Runs on the execution host (it needs only torch + numpy + the repo's own exact
enumeration code).  Train-only vocabulary and Train-only document frequencies; the
official 10 000 / 1 000 / 1 000 split is used as given.

Outputs (``--out``): ``zinc_design.npz`` + ``zinc_design_meta.json``
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import sga_common as K
import sga_zinc as Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(K.CACHE / "zinc_design.npz"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()

    data = Z.load_all()
    n = {k: data[k]["n_graphs"] for k in ("train", "valid", "test")}
    print(f"[zinc-prep] sizes {n} node_classes={len(data['node_classes'])} "
          f"edge_classes={data['edge_classes']}", flush=True)
    if n != {"train": 10000, "valid": 1000, "test": 1000}:
        raise SystemExit(f"unexpected ZINC split sizes: {n}")
    # official split -> global 0-based row ids in (train, valid, test) order
    rows = {"train": np.arange(0, n["train"], dtype=np.int64),
            "valid": np.arange(n["train"], n["train"] + n["valid"], dtype=np.int64),
            "test": np.arange(n["train"] + n["valid"],
                              n["train"] + n["valid"] + n["test"], dtype=np.int64)}
    y = np.concatenate([data[k]["y"] for k in ("train", "valid", "test")])
    node_count = np.asarray([g["x"].size for k in ("train", "valid", "test") for g in data[k]["graphs"]],
                            dtype=np.int64)

    Zmat, zmeta = Z.build_Z(data, rows["train"])
    print(f"[zinc-prep] Z {Zmat.shape} nnz={Zmat.nnz} in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    Cm, Fm, cfmeta = Z.build_ZCF(data, rows["train"], node_count, workers=args.workers)
    print(f"[zinc-prep] Z-CF C{Cm.shape} F{Fm.shape} in {time.perf_counter() - t1:.1f}s", flush=True)

    meta = {
        "round_id": K.ROUND_ID,
        "phase": "C prep: ZINC fixed representation",
        "created_utc": created,
        "split": {"sizes": n, "rows_sha256": {k: K.sha256_array(v) for k, v in rows.items()},
                  "source": "PyG ZINC(subset=True) official split, row order train/valid/test",
                  "no_re_split": True},
        "label": {
            "units": "PyG ZINC subset=True `y` as provided (upstream log-modulus transformed)",
            "train_mean": float(y[rows["train"]].mean()), "train_std": float(y[rows["train"]].std()),
            "valid_mean": float(y[rows["valid"]].mean()), "test_mean": float(y[rows["test"]].mean()),
            "min": float(y.min()), "max": float(y.max()),
        },
        "datasets": {k: {"file_sha256": data[k]["file_sha256"], "n_graphs": data[k]["n_graphs"],
                         "n_raw_edge_records": data[k]["n_raw_edge_records"],
                         "physical_bonds": data[k]["physical_bonds"],
                         "node_classes_observed": data[k]["node_classes"],
                         "edge_classes_observed": data[k]["edge_classes"]}
                     for k in ("train", "valid", "test")},
        "Z": zmeta,
        "Z_CF": cfmeta,
        "git": K.git_state(),
        "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }
    np.savez_compressed(
        out,
        Z_indptr=Zmat.indptr, Z_indices=Zmat.indices, Z_data=Zmat.data,
        C_indptr=Cm.indptr, C_indices=Cm.indices, C_data=Cm.data,
        F_indptr=Fm.indptr, F_indices=Fm.indices, F_data=Fm.data,
        y=y, n_atoms=node_count,
        rows_train=rows["train"], rows_valid=rows["valid"], rows_test=rows["test"],
    )
    meta["design"] = {"path": str(out), "sha256": K.sha256_file(out),
                      "size_bytes": int(out.stat().st_size),
                      "Z_shape": list(Zmat.shape), "C_shape": list(Cm.shape), "F_shape": list(Fm.shape)}
    (out.parent / "zinc_design_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, sort_keys=True, default=K._json_default) + "\n")
    print(f"[zinc-prep] wrote {out} {out.stat().st_size / 1e6:.1f} MB "
          f"sha={meta['design']['sha256'][:12]} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
