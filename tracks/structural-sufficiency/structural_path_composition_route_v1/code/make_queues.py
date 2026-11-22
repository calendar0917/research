"""Emit D1 / FULL / DIAG queue files from the frozen selection lock."""

from __future__ import annotations

import json
from pathlib import Path

import psc_common as C
import psc_analysis as A

CODE = Path(__file__).resolve().parent


def line(run_id: str, arm: str, split: str, seed: int, epochs: int | None) -> str:
    return f"{run_id}\t{arm}\t{split}\t{seed}\t{epochs if epochs else '-'}"


def main() -> int:
    lock = C.read_json(A.ANALYSIS / "selection_lock.json")
    champ = lock["champion"]
    acra = lock["ACRA_B_star"]
    e_m = lock["E_m"]
    rows_d1, rows_full, rows_diag = [], [], []
    models_d1 = [champ, lock["O_star"], lock["R_star"], acra, "GINE_REF"]
    for arm in models_d1:
        e = e_m.get(arm)
        for seed in (0, 1):
            rows_d1.append(line(A.run_id(arm, "D1", seed), arm, "D1", seed, e))
    models_full = [champ, acra, "GINE_REF"]
    if lock.get("M_star"):
        models_full.append(lock["M_star"])
    for arm in models_full:
        e = e_m.get(arm) or e_m.get(champ)
        for seed in (0, 1, 2):
            rows_full.append(line(A.run_id(arm, "FULL", seed), arm, "FULL", seed, e))
    if lock.get("DIAG_twin"):
        arm = lock["DIAG_twin"]
        for seed in (0, 1, 2):
            rows_diag.append(line(A.run_id(arm, "D0", seed), arm, "D0", seed,
                                  e_m.get(arm) or e_m.get(champ)))
    (CODE / "queue_d1.txt").write_text("\n".join(rows_d1) + "\n", encoding="utf-8")
    (CODE / "queue_full.txt").write_text("\n".join(rows_full) + "\n", encoding="utf-8")
    (CODE / "queue_diag.txt").write_text("\n".join(rows_diag) + ("\n" if rows_diag else ""),
                                        encoding="utf-8")
    # split each stream over the two GPUs to balance wall time
    for name, rows in (("d1", rows_d1), ("full", rows_full), ("diag", rows_diag)):
        half = (len(rows) + 1) // 2
        (CODE / f"queue_{name}_gpu0.txt").write_text("\n".join(rows[:half]) + "\n"
                                                    if rows else "", encoding="utf-8")
        (CODE / f"queue_{name}_gpu1.txt").write_text("\n".join(rows[half:]) + "\n"
                                                    if rows else "", encoding="utf-8")
    print(json.dumps({"champion": champ, "d1": rows_d1, "full": rows_full,
                      "diag": rows_diag}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())