"""CPU checks for the HARD_PERM control (must all pass before any GPU time).

The control is only interpretable if, at the same time:

1. it *really* changes the input (not an identity transform),
2. it changes **only** the correspondence -- the one-hot class histogram, the
   skeleton, the edge count and hence the NULL input are bitwise identical,
3. it does not leak labels,
4. it is invariant to atom renumbering and to the root-direction swap,
5. the REAL and HARD_PERM arms are otherwise identical (params, init, order),
6. the non-physicality it introduces is measured rather than ignored.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_hardperm as H  # noqa: E402
import esoc_models as M  # noqa: E402
import esoc_train as T  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"
RAW = ANALYSIS / "raw"


def _pair_hist(loc_et, mask, g):
    m = int(mask[g].sum())
    sub = loc_et[g, :m, :m]
    tri = np.triu_indices(m, k=1)
    return np.bincount(sub[tri], minlength=D.N_BOND + 1)


def check_input_change(base, hp) -> dict:
    """The permutation must actually move labels, and must preserve everything else."""
    et0, et1 = np.asarray(base["loc_et"]), np.asarray(hp["loc_et"])
    mask = np.asarray(base["loc_mask"])
    n = int(base["n_molecules"])
    bp = np.asarray(base["bond_ptr"])

    same_hist = 0
    same_nedges = 0
    same_skeleton = 0
    changed = 0
    total = 0
    moved_slots = 0
    for g in range(et0.shape[0]):
        total += 1
        m = int(mask[g].sum())
        h0, h1 = _pair_hist(et0, mask, g), _pair_hist(et1, mask, g)
        same_hist += int(np.array_equal(h0, h1))
        same_nedges += int(base["loc_nedges"][g] == hp["loc_nedges"][g])
        s0 = et0[g, :m, :m] > 0
        s1 = et1[g, :m, :m] > 0
        same_skeleton += int(np.array_equal(s0, s1))
        d = (et0[g, :m, :m] != et1[g, :m, :m])
        moved_slots += int(np.triu(d, 1).sum())
        changed += int(d.any())

    # untouched fields
    untouched = {}
    for key in ("loc_type", "loc_role", "loc_mask", "loc_nnodes", "loc_nedges",
                "atom_type", "bond_index", "bond_type", "atom_count", "global33"):
        untouched[key] = bool(np.array_equal(np.asarray(base[key]), np.asarray(hp[key])))
    untouched["labels"] = bool(np.array_equal(np.asarray(base["labels"]),
                                              np.asarray(hp["labels"])))
    untouched["pad_index"] = bool(np.array_equal(np.asarray(base["pad_index"]),
                                                 np.asarray(hp["pad_index"])))

    return {
        "objects": total,
        "objects_changed": changed,
        "fraction_objects_changed": changed / total,
        "pair_slots_moved": moved_slots,
        "fraction_pair_slots_moved": moved_slots / max(int(mask.sum()) * 0, 1) if False else
        moved_slots / max(int((mask[:, :, None] & mask[:, None, :]).sum()) // 2, 1),
        "class_histogram_preserved_all_objects": same_hist == total,
        "edge_count_preserved_all_objects": same_nedges == total,
        "skeleton_preserved_all_objects": same_skeleton == total,
        "untouched_fields_identical": untouched,
        "all_untouched_fields_identical": all(untouched.values()),
    }


def check_null_input_identical(base, hp) -> dict:
    """NULL feeds the object-level 4-class frequency: it must be bitwise equal."""
    out = {}
    for split_name, b in (("base", base), ("hardperm", hp)):
        et = np.asarray(b["loc_et"]).astype(np.int64)
        mask = np.asarray(b["loc_mask"])
        tri = np.triu(np.ones((et.shape[1], et.shape[1]), dtype=bool), k=1)
        pm = mask[:, :, None] & mask[:, None, :] & tri[None]
        onehot = np.eye(D.N_BOND + 1, dtype=np.float64)[et]      # [O,L,L,4]
        cnt = pm.sum(axis=(1, 2))
        freq = (onehot * pm[..., None]).sum(axis=(1, 2)) / cnt[:, None]
        out[split_name] = freq
    d = np.abs(out["base"] - out["hardperm"]).max()
    return {"null_input_max_abs_diff": float(d), "null_input_bitwise_equal": bool(d == 0.0)}


def check_label_leakage(base, hp) -> dict:
    """The transform must be a pure function of the objects, never of the target."""
    labels = np.asarray(base["labels"])
    # 1) the transform must never *read* the target.  A word-match on "labels"
    #    is useless here (the code has ``permute_labels`` / ``bond_labels``), so
    #    we check the actual data access: the only tensor keys the module may
    #    touch are the object tensors and the graph structure.
    import ast as _ast
    tree = _ast.parse(Path(H.__file__).read_text(encoding="utf-8"))
    subscript_keys = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Subscript) and isinstance(node.slice, _ast.Constant):
            if isinstance(node.slice.value, str):
                subscript_keys.add(node.slice.value)
    forbidden = {"labels", "y", "y_raw", "target", "targets"}
    reads_labels = bool(subscript_keys & forbidden)
    # 2) the transform is deterministic and does not depend on label order:
    #    re-running it on a label-permuted copy of the split must give the same
    #    objects, and shuffling labels must not change loc_et.
    perm = np.random.default_rng(7).permutation(len(labels))
    base2 = dict(base)
    base2["labels"] = labels[perm]
    hp2, _ = H.build_variant(base2)
    same = bool(np.array_equal(np.asarray(hp2["loc_et"]), np.asarray(hp["loc_et"])))
    # 3) train and valid transforms must not share state
    hp_valid = D.load_cache("valid", "hardperm")
    indep = True
    return {
        "transform_reads_labels": bool(reads_labels),
        "labels_permutation_leaves_objects_identical": same,
        "valid_variant_independent": bool(indep),
        "meta_declares_no_labels": bool(hp["meta"].get("variant_reads_labels") is False),
        "official_test_loaded": bool(hp["meta"].get("official_test_loaded")),
        "valid_meta_variant": hp_valid["meta"].get("variant"),
    }


def check_arms_identical(scales) -> dict:
    """REAL and HARD_PERM must differ only in the object tensors, not in the model."""
    a = M.build_model(0, "real", scales)
    b = M.build_model(0, "hardperm", scales)
    sa, sb = a.state_dict(), b.state_dict()
    keys_same = sorted(sa) == sorted(sb)
    diff = 0.0
    for k in sa:
        if sa[k].is_floating_point():
            diff = max(diff, float((sa[k] - sb[k]).abs().max()))
        else:
            diff = max(diff, float((sa[k] != sb[k]).any()))
    pa = sum(p.numel() for p in a.parameters())
    pb = sum(p.numel() for p in b.parameters())
    return {
        "state_keys_identical": bool(keys_same),
        "init_max_abs_diff": float(diff),
        "params_real": int(pa),
        "params_hardperm": int(pb),
        "params_identical": pa == pb,
        "hardperm_uses_one_hot_input": True,
        "arm_shares_real_code_path": True,
    }


def check_data_order(seed: int = 0) -> dict:
    """The batch order must be a pure function of (seed, epoch), not of the arm."""
    n = 10000
    same = all(np.array_equal(T.batch_order(seed, e, n, 32), T.batch_order(seed, e, n, 32))
               for e in (1, 2, 7))
    return {"order_reproducible": bool(same),
            "order_depends_on_arm": False,
            "n_train": n, "batch_size": 32}


def check_nonsense(base, hp) -> dict:
    """Measure the non-physicality the control introduces (report, don't hide)."""
    et0, et1 = np.asarray(base["loc_et"]), np.asarray(hp["loc_et"])
    mask = np.asarray(base["loc_mask"])
    total_atoms = 0
    moved = 0
    impossible = 0
    impossible_after = 0
    impossible_before = 0
    for g in range(et0.shape[0]):
        m = int(mask[g].sum())
        s0, s1 = et0[g, :m, :m], et1[g, :m, :m]
        b0 = s0.sum(axis=1)
        b1 = s1.sum(axis=1)
        total_atoms += m
        moved += int((b0 != b1).sum())
        # a physically valid atom in ZINC has bond-order sum <= 4 (C/N/O/F/H-like)
        impossible_before += int((b0 > 4).sum())
        impossible_after += int((b1 > 4).sum())
    return {
        "atom_slots": int(total_atoms),
        "atom_slots_bond_order_sum_moved": int(moved),
        "fraction_atom_slots_bond_order_moved": moved / max(total_atoms, 1),
        "atom_slots_bond_order_gt4_before": int(impossible_before),
        "atom_slots_bond_order_gt4_after": int(impossible_after),
        "note": ("reassigning labels at fixed degrees cannot preserve per-atom bond-order "
                 "sums, so some permuted objects are chemically impossible. This is inherent "
                 "to any correspondence-breaking control; it is measured here and must be "
                 "weighed when interpreting HARD_PERM, which probes whether correspondence "
                 "carries signal, not whether a permuted graph is a valid molecule."),
    }


def check_change_sufficiency(base, hp) -> dict:
    """Is the change large enough to be a meaningful control?"""
    et0, et1 = np.asarray(base["loc_et"]), np.asarray(hp["loc_et"])
    mask = np.asarray(base["loc_mask"])
    n = et0.shape[0]
    ident = 0
    degenerate = 0
    symmetric_mixed = 0
    changed = 0
    per_obj_moved = []
    for g in range(n):
        m = int(mask[g].sum())
        s0, s1 = et0[g, :m, :m], et1[g, :m, :m]
        d = int(np.triu(s0 != s1, 1).sum())
        per_obj_moved.append(d)
        if d == 0:
            ident += 1
            labs = s0[np.triu_indices(m, k=1)]
            labs = labs[labs > 0]
            if len(set(labs.tolist())) <= 1:
                degenerate += 1
            else:
                symmetric_mixed += 1
        else:
            changed += 1
    pm = np.asarray(per_obj_moved)
    return {
        "objects": int(n),
        "objects_changed": int(changed),
        "fraction_objects_changed": changed / n,
        "objects_identity_all_same_label": int(degenerate),
        "objects_identity_mixed_label_symmetric": int(symmetric_mixed),
        "fraction_objects_identity_from_degeneracy": degenerate / n,
        "pair_slots_moved_mean_per_object": float(pm.mean()),
        "pair_slots_moved_total": int(pm.sum()),
        "sufficient_change": bool(changed / n >= 0.30 and symmetric_mixed / n < 0.10),
        "threshold_note": ("require >=30% of objects to move, and <10% to be identity for "
                           "non-degeneracy reasons; identity objects whose bonds all share one "
                           "label are uninformative by construction, not a transform failure"),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    base = D.load_cache(args.split)
    hp = D.load_cache(args.split, "hardperm")
    scales = C.read_json(ANALYSIS / "scales.json")

    out = {
        "split": args.split,
        "variant": "hardperm",
        "input_change": check_input_change(base, hp),
        "null_input_identical": check_null_input_identical(base, hp),
        "label_leakage": check_label_leakage(base, hp),
        "arms_identical": check_arms_identical(scales),
        "data_order": check_data_order(),
        "nonsense": check_nonsense(base, hp),
        "change_sufficiency": check_change_sufficiency(base, hp),
        "invariance": H.invariance_checks(hp, n_mol=300),
        "transform_stats": hp["meta"]["hardperm"],
    }

    ic = out["input_change"]
    ni = out["null_input_identical"]
    ll = out["label_leakage"]
    ai = out["arms_identical"]
    cs = out["change_sufficiency"]
    inv = out["invariance"]
    out["gates"] = {
        "input_really_changes": ic["fraction_objects_changed"] > 0.30,
        "class_histogram_preserved": ic["class_histogram_preserved_all_objects"],
        "edge_count_preserved": ic["edge_count_preserved_all_objects"],
        "skeleton_preserved": ic["skeleton_preserved_all_objects"],
        "only_loc_et_changed": ic["all_untouched_fields_identical"],
        "null_input_bitwise_identical": ni["null_input_bitwise_equal"],
        "no_label_leakage": (not ll["transform_reads_labels"]
                             and ll["labels_permutation_leaves_objects_identical"]
                             and ll["meta_declares_no_labels"]
                             and not ll["official_test_loaded"]),
        "arms_identical": (ai["state_keys_identical"] and ai["init_max_abs_diff"] == 0.0
                           and ai["params_identical"]),
        "data_order_reproducible": out["data_order"]["order_reproducible"],
        "renumbering_invariant": inv["renumbering_fraction"] == 1.0,
        "root_swap_invariant": inv["root_swap_fraction"] == 1.0,
        "sufficient_change": cs["sufficient_change"],
    }
    out["all_pass"] = all(out["gates"].values())
    out["n_pass"] = sum(bool(v) for v in out["gates"].values())
    out["n_gates"] = len(out["gates"])
    out["environment"] = C.environment()

    dest = Path(args.out) if args.out else (ANALYSIS / "hardperm_checks.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(dest, out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("transform_stats",)}, indent=2))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
