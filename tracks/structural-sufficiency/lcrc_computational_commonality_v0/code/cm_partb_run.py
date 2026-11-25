"""Part B driver: stats -> tables -> fit -> identity verify -> replacement.

``uv run python code/cm_partb_run.py all --device cuda``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import cm_common as C
import cm_partb as P


def stage_stats() -> dict[str, Any]:
    task = C.DC.SD.load_task()
    st = P.train_stats(task.cache, np.asarray(task.train_rows, dtype=np.int64))
    P.LOCAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        P.STATS_PATH,
        occurrence=st["occurrence"].astype(np.int64),
        doc_freq=st["doc_freq"].astype(np.int64),
        observed=st["observed"],
    )
    meta = {k: v for k, v in st.items() if k not in ("occurrence", "doc_freq", "observed")}
    comp = P.components(np.flatnonzero(st["observed"]))
    meta["component_support"] = {name: int(np.unique(comp[:, i]).size) for i, name in enumerate(P.COMP_NAMES)}
    meta["component_sizes"] = {name: int(size) for name, size in zip(P.COMP_NAMES, P.COMP_SIZES)}
    meta["n_types_observed"] = int(st["observed"].sum())
    meta["created_utc"] = C.utcnow()
    C.write_json(P.LOCAL / "train_type_stats.json", meta)
    return meta


def stage_tables() -> dict[str, Any]:
    model, task, meta = P.load_teacher("cpu")
    tab = P.teacher_tables(model, meta["layout"])
    tables = P.build_tables(tab)
    P.LOCAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P.TABLE_PATH, T_total=tab["T_total"].astype(np.float32),
                        T_joint=tab["T_joint"].astype(np.float32),
                        **{f"H_{k}": v.astype(np.float32) for k, v in tables.items()})
    amp = {
        "teacher": {"checkpoint_sha256": meta["checkpoint_sha256"], "epoch": meta["epoch"]},
        "T_total_rms": float(np.sqrt((tab["T_total"] ** 2).mean())),
        "T_joint_rms": float(np.sqrt((tab["T_joint"] ** 2).mean())),
        "T_type_rms": float(np.sqrt((tab["T_type"] ** 2).mean())),
        "T_end_rms": float(np.sqrt((tab["T_end"] ** 2).mean())),
        "T_joint_share_of_total_energy": float((tab["T_joint"] ** 2).sum() / (tab["T_total"] ** 2).sum()),
    }
    C.write_json(P.LOCAL / "teacher_amplitude.json", amp)
    return amp


def stage_fit() -> dict[str, Any]:
    return P.run_fit()


def stage_verify() -> dict[str, Any]:
    tab = np.load(P.TABLE_PATH, allow_pickle=False)
    tables = {k[2:]: np.asarray(tab[k]) for k in tab.files if k.startswith("H_")}
    return P.run_verify(dict(tab), tables, n_molecules=160)


def stage_replace() -> dict[str, Any]:
    tab = np.load(P.TABLE_PATH, allow_pickle=False)
    tables = {k[2:]: np.asarray(tab[k]) for k in tab.files if k.startswith("H_")}
    return P.run_replace(dict(tab), tables)


STAGES = {"stats": stage_stats, "tables": stage_tables, "fit": stage_fit,
          "verify": stage_verify, "replace": stage_replace}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    order = list(STAGES) if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()
    for name in order:
        ts = time.perf_counter()
        out = STAGES[name]()
        print(f"[partb:{name}] {time.perf_counter() - ts:.1f}s", flush=True)
        if name == "verify" and not out.get("passed", True):
            raise SystemExit("identity reconstruction failed; Part B stops here")
    print(f"[partb] total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
