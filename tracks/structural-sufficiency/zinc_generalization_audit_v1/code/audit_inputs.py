"""Narrow input/alignment checks for the audit (no retraining, no test)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "zinc_induced_pattern_pilot_v1" / "code"))
import audit_common as C  # noqa: E402
import ip_data  # noqa: E402

ANALYSIS = C.ANALYSIS


def official_processed_val() -> dict:
    data, slices, _ = torch.load(
        C.REPO / "data" / "ZINC" / "subset" / "processed" / "val.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {"data": data, "slices": slices}


def main() -> int:
    valid = C.load_cache("valid")
    official = official_processed_val()
    data, slices = official["data"], official["slices"]
    n = int(valid["n_molecules"])
    ap = np.asarray(valid["atom_ptr"])
    bp = np.asarray(valid["bond_ptr"])
    at = np.asarray(valid["atom_type"])
    bi = np.asarray(valid["bond_index"])
    bt = np.asarray(valid["bond_type"])
    y = np.asarray(valid["labels"])
    y_off = np.asarray(data["y"]).reshape(-1)

    checks = {
        "official_split": "official PyG ZINC subset=True val (1000), processed local file",
        "official_test_loaded": False,
        "n_valid_rows": n,
        "prediction_row_checks": {},
    }

    # Predictions: row coverage, duplicates, finiteness, MAE parity.
    runs, y2, _ = C.load_all_predictions()
    if not np.array_equal(y, y2):
        raise RuntimeError("valid label arrays differ between cache loads")
    for name, r in runs.items():
        checks["prediction_row_checks"][name] = {
            "n_rows": int(len(r["row"])),
            "row_min": int(r["row"].min()),
            "row_max": int(r["row"].max()),
            "unique_rows": int(len(np.unique(r["row"]))),
            "finite": bool(np.isfinite(r["pred"]).all()),
            "recomputed_mae": r["mae_recomputed"],
            "stored_best_valid": float(r["info"]["fixed_objects"]["best"]["valid_mae"]),
            "mae_abs_diff": abs(r["mae_recomputed"] - float(r["info"]["fixed_objects"]["best"]["valid_mae"])),
        }

    # Official processed graph / label equality, all 1000 rows.
    bad_atom = []
    bad_edges = []
    bad_counts = []
    max_label_diff = 0.0
    for i in range(n):
        a0, a1 = int(slices["x"][i]), int(slices["x"][i + 1])
        b0, b1 = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        x = np.asarray(data["x"][a0:a1]).reshape(-1).astype(np.int64)
        if not np.array_equal(x, at[ap[i] : ap[i + 1]]):
            bad_atom.append(i)
        e1 = {}
        for (u, v), t in zip(
            np.asarray(data["edge_index"][:, b0:b1]).T.tolist(),
            np.asarray(data["edge_attr"][b0:b1]).reshape(-1).tolist(),
        ):
            if int(u) < int(v):
                e1[(int(u), int(v))] = int(t)
        e2 = {}
        for (u, v), t in zip(
            np.asarray(bi[bp[i] : bp[i + 1]]).tolist(),
            np.asarray(bt[bp[i] : bp[i + 1]]).tolist(),
        ):
            u, v, t = int(u), int(v), int(t)
            e2[(min(u, v), max(u, v))] = t
        if e1 != e2:
            bad_edges.append(i)
        if int(valid["n_atoms_graph"][i]) != a1 - a0 or int(valid["n_bonds_graph"][i]) != (b1 - b0) // 2:
            bad_counts.append(i)
        max_label_diff = max(max_label_diff, abs(float(y_off[i]) - float(y[i])))

    checks["official_graph_labels_vs_ad_valid_cache"] = {
        "n_checked": n,
        "atom_type_mismatch_rows": bad_atom[:10],
        "n_atom_type_mismatch": len(bad_atom),
        "edge_set_or_type_mismatch_rows": bad_edges[:10],
        "n_edge_mismatch": len(bad_edges),
        "node_or_bond_count_mismatch_rows": bad_counts[:10],
        "n_count_mismatch": len(bad_counts),
        "label_max_abs_diff_raw_y": float(max_label_diff),
        "ok": not bad_atom and not bad_edges and not bad_counts and max_label_diff == 0.0,
    }

    # Duplicate/missing graph rows in the cache (exact content fingerprint).
    hashes = [C.graph_sha256(valid, i) for i in range(n)]
    checks["graph_rows"] = {
        "n_rows": n,
        "n_unique_graph_sha256": int(len(set(hashes))),
        "duplicate_graph_content_rows": int(n - len(set(hashes))),
        "all_rows_present_0_to_n_minus_1": bool(np.array_equal(np.arange(n), np.arange(n))),
    }

    # Path-cache fidelity on three fixed molecules: exact re-enumeration from the
    # raw cached graph and exact match of every content key/root/multiplicity.
    path_valid = C.load_path_cache("valid")
    path_checks = {}
    for i in (0, 172, 917):
        obj = C.p3_object_from_cache(valid, i)
        recomputed = ip_data.enumerate_rooted_paths(obj)
        cached = C.cached_path_entries(path_valid, i)
        same_keys = True
        max_count_diff = 0
        for length in ip_data.LENGTHS:
            rk = recomputed[length]
            ck = cached[length]
            if set(rk) != set(ck):
                same_keys = False
                continue
            for key, roots in rk.items():
                c_roots = ck[key]
                if set(roots) != set(c_roots):
                    same_keys = False
                    continue
                for root, count in roots.items():
                    max_count_diff = max(max_count_diff, abs(int(count) - int(c_roots[root])))
        path_checks[str(i)] = {
            "same_pattern_key_set": bool(same_keys),
            "max_root_multiplicity_abs_diff": int(max_count_diff),
            "n_patterns_by_length_recomputed": {str(l): len(recomputed[l]) for l in ip_data.LENGTHS},
        }
    checks["path_cache_fidelity"] = {
        "rows_checked": [0, 172, 917],
        "per_row": path_checks,
        "ok": all(v["same_pattern_key_set"] and v["max_root_multiplicity_abs_diff"] == 0 for v in path_checks.values()),
    }

    checks["pass"] = bool(
        all(v["unique_rows"] == 1000 and v["row_min"] == 0 and v["row_max"] == 999 and v["finite"] and v["mae_abs_diff"] < 1e-5 for v in checks["prediction_row_checks"].values())
        and checks["official_graph_labels_vs_ad_valid_cache"]["ok"]
        and checks["graph_rows"]["n_unique_graph_sha256"] == 1000
        and checks["path_cache_fidelity"]["ok"]
    )
    C.write_json(ANALYSIS / "input_checks.json", checks)
    print(json.dumps(checks, indent=2)[:5000])
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
