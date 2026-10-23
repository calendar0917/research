"""A3 provenance: what the k1 support bug did to the head inputs.

No training, no model, no labels beyond the F fit rows. Writes
``analysis/feature_fix.json`` and prints a short summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import zst_head as H  # noqa: E402
import ad_data as D  # noqa: E402


def main() -> int:
    man = json.load(open(Z.ANALYSIS / "split_manifest.json", encoding="utf-8"))
    roles = {k: np.asarray(v, dtype=np.int64) for k, v in man["roles"].items()}
    keys = Z.load_keys()
    cache = D.load_cache("train")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    mol1, mol2 = Z.molecule_key_sets(keys)
    n_key1, n_key2 = int(keys["n_key1"]), int(keys["n_key2"])
    atom_k1, atom_k2, _ = Z.atom_tables(keys, n_key2)
    sv = Z.per_atom_support(roles["F"], mol1, mol2, keys, n_key1, n_key2)
    sup1, sup2 = sv["sup1"], sv["sup2"]

    n1_new = sv["atom_n1"]
    n1_old = sup2[atom_k1].astype(np.float64)

    out = {"n_key1": n_key1, "n_key2": n_key2,
           "support_hash_fixed": Z.support_hash(sup1, sup2),
           "atom_level": {}, "context_columns": {}, "roles": {}}

    for role in ("C_fit", "C_dev", "Q_A", "Q_B", "Q_bg"):
        rows = roles[role]
        unseen = []
        allatoms = []
        for m in rows.tolist():
            s, e = int(atom_ptr[m]), int(atom_ptr[m + 1])
            allatoms.append(np.arange(s, e))
            k = np.arange(s, e)
            unseen.append(k[sv["atom_n2"][k] == 0])
        ua = np.concatenate(unseen) if unseen else np.zeros(0, dtype=np.int64)
        aa = np.concatenate(allatoms)
        out["roles"][role] = {
            "n_molecules": int(len(rows)), "n_atoms": int(len(aa)),
            "n_unseen_atoms": int(len(ua)),
            "n1_new_median_all_atoms": float(np.median(n1_new[aa])) if len(aa) else None,
            "n1_old_median_all_atoms": float(np.median(n1_old[aa])) if len(aa) else None,
            "unseen_atom_frac_where_n1_new_ne_n1_old": (
                float(np.mean(n1_new[ua] != n1_old[ua])) if len(ua) else None),
            "unseen_atom_max_abs_n1_change": (
                float(np.max(np.abs(n1_new[ua] - n1_old[ua]))) if len(ua) else None),
            "unseen_atom_frac_n1_new_zero": (
                float(np.mean(n1_new[ua] == 0)) if len(ua) else None),
            "unseen_atom_n1_new_median": float(np.median(n1_new[ua])) if len(ua) else None,
        }

    # effect on the two context columns that use n1 (indices 39/40 = min_n1/mean_n1)
    rows = np.concatenate([roles[r][:64] for r in ("C_fit", "C_dev", "Q_A")])
    zero = np.zeros(len(rows))
    f_new = H.Features(cache, rows, zero, sv["atom_n1"], sv["atom_n2"], atom_ptr)
    f_old = H.Features(cache, rows, zero, n1_old, sv["atom_n2"], atom_ptr)
    diff = np.abs(f_new.ctx - f_old.ctx)
    i_min_n1 = 1 + 3 + D.N_ATOM_TYPES + 2  # [f_base, logn, logb, logring, comp, logn_unseen, frac]
    changed = [int(i) for i in np.flatnonzero(diff.max(0) > 0)]
    out["context_columns"] = {
        "n_rows": int(len(rows)),
        "min_n1_mean_n1_indices": [i_min_n1, i_min_n1 + 1],
        "max_abs_diff_per_column": {str(i): float(diff[:, i].max()) for i in changed},
        "ctx_columns_changed": changed,
        "only_n1_columns_changed": bool(changed == [i_min_n1, i_min_n1 + 1]),
        "interpretation": f"columns {i_min_n1}/{i_min_n1 + 1} are min_n1/mean_n1 over "
                          "unseen atoms; the n2 columns and every other block are untouched",
    }
    Z.write_json(Z.ANALYSIS / "feature_fix.json", out)
    print(json.dumps(out, indent=1)[:2200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
