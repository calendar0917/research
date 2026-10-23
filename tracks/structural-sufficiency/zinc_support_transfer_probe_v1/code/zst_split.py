"""Role construction + isolation gates for zinc-support-transfer-probe-v1.

Structure only. No label, prediction, error or model output is read here. The
assignment is written and hashed before any new-query error is computed.

Roles (whole molecules, never atoms):

    A, B  query key families        -> Q_A, Q_B
    C     correction-fit key family -> C_fit core (+ background molecules)
    D     correction-dev key family -> C_dev
    Q_bg  molecules carrying no selected key
    F     everything else           -> the one supervised base fit

Family order A -> B -> C -> D. Family B first only accepts radius-2 keys whose
radius-1 parent is unused by A, then relaxes once if the quota is short; the
remaining parent overlap is reported rather than engineered away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zst_common as Z  # noqa: E402
import ad_data as D  # noqa: E402

# (name, key target, molecule target, molecule cap)
FAMILIES = (
    ("A", 35, 300, 380),
    ("B", 35, 300, 380),
    ("C", 110, 900, 1000),
    ("D", 40, 300, 360),
)
CFIT_BACKGROUND = 250
QBG_SIZE = 400


def build_inverse(mol2, n_key2: int):
    inv = [[] for _ in range(int(n_key2))]
    sup = np.zeros(int(n_key2), dtype=np.int32)
    for m, s2 in enumerate(mol2):
        sup[s2] += 1
        for k in s2.tolist():
            inv[k].append(m)
    return sup, inv


def grow_family(order, inv, used, target_keys, target_mols, cap_mols,
                forbidden_parents=None, parent=None):
    """Add whole molecule sets of keys; every key all-own-molecules-unused."""
    selected: list[int] = []
    mols: set[int] = set()
    for k in order:
        if len(selected) >= target_keys and len(mols) >= target_mols:
            break
        if forbidden_parents is not None and int(parent[k]) in forbidden_parents:
            continue
        ms = inv[k]
        if any(used[m] or m in mols for m in ms):
            continue
        if len(mols) + len(ms) > cap_mols:
            continue
        selected.append(int(k))
        mols.update(ms)
    return selected, mols


def components(keys, inv):
    """Connected components of the selected keys under molecule co-occurrence."""
    keyset = set(int(k) for k in keys)
    par = {k: k for k in keyset}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    per_mol: dict[int, list[int]] = {}
    for k in keyset:
        for m in inv[k]:
            per_mol.setdefault(int(m), []).append(k)
    for ks in per_mol.values():
        for o in ks[1:]:
            ra, rb = find(ks[0]), find(o)
            if ra != rb:
                par[rb] = ra
    comp: dict[int, list[int]] = {}
    for k in keyset:
        comp.setdefault(find(k), []).append(k)
    return [sorted(v) for v in comp.values()]


def build_families(verbose: bool = True) -> dict:
    keys = Z.load_keys()
    mol1, mol2 = Z.molecule_key_sets(keys)
    n_key2 = int(keys["n_key2"])
    n_key1 = int(keys["n_key1"])
    sup2, inv = build_inverse(mol2, n_key2)
    sup1 = np.zeros(n_key1, dtype=np.int32)
    for s1 in mol1:
        sup1[s1] += 1
    atom_k1, atom_k2, parent = Z.atom_tables(keys, n_key2)

    cand = np.flatnonzero((sup2 >= Z.KEY_SUP_LO) & (sup2 <= Z.KEY_SUP_HI)
                          & (sup1[np.maximum(parent, 0)] >= Z.PARENT_MIN))
    rng = np.random.default_rng(int(Z.SPLIT_RNG_SEED))
    order = cand[rng.permutation(len(cand))]

    used = np.zeros(Z.N_TRAIN, dtype=bool)
    families: dict[str, dict] = {}
    parents_a: set[int] = set()
    for name, tk, tm, cap in FAMILIES:
        sel, mols = grow_family(order, inv, used, tk, tm, cap)
        pre_overlap = None
        if name == "B":
            sel_p, mols_p = grow_family(order, inv, used, tk, tm, cap,
                                        forbidden_parents=parents_a, parent=parent)
            pre_overlap = {
                "keys_parent_disjoint_from_A": len(sel_p),
                "molecules_parent_disjoint_from_A": len(mols_p),
                "used_after_parent_pass": int(used.sum()),
                "parent_pass_met_target": bool(len(sel_p) >= tk and len(mols_p) >= tm),
            }
            if len(sel_p) >= tk and len(mols_p) >= tm:
                sel, mols = sel_p, mols_p
        for m in mols:
            used[m] = True
        if name == "A":
            parents_a = set(int(parent[k]) for k in sel)
        families[name] = {"keys": sorted(int(k) for k in sel), "molecules": sorted(mols),
                          "target_keys": int(tk), "target_mols": int(tm), "cap": int(cap),
                          "parent_pass": pre_overlap}
        if verbose:
            print(f"[family {name}] keys={len(sel)} mols={len(mols)} "
                  f"(target {tk}/{tm}) parent={len(set(int(parent[k]) for k in sel))}", flush=True)

    # background molecules: carry no selected key at all
    selected_keys = set()
    for f in families.values():
        selected_keys |= set(f["keys"])
    pool = [m for m in range(Z.N_TRAIN)
            if not used[m] and not (set(int(k) for k in mol2[m]) & selected_keys)]
    # deterministic fixed-RNG order for the background draws
    brng = np.random.default_rng(int(Z.SPLIT_RNG_SEED) + 1)
    pool = [int(x) for x in brng.permutation(np.asarray(pool, dtype=np.int64))]
    if len(pool) < CFIT_BACKGROUND + QBG_SIZE:
        raise RuntimeError(f"only {len(pool)} background molecules available")
    cfit_bg = sorted(pool[:CFIT_BACKGROUND])
    q_bg = sorted(pool[CFIT_BACKGROUND:CFIT_BACKGROUND + QBG_SIZE])
    for m in cfit_bg + q_bg:
        used[m] = True

    f_rows = sorted(int(m) for m in np.flatnonzero(~used))
    roles = {
        "F": np.asarray(f_rows, dtype=np.int64),
        "C_fit": np.asarray(sorted(set(families["C"]["molecules"]) | set(cfit_bg)), dtype=np.int64),
        "C_dev": np.asarray(sorted(families["D"]["molecules"]), dtype=np.int64),
        "Q_A": np.asarray(sorted(families["A"]["molecules"]), dtype=np.int64),
        "Q_B": np.asarray(sorted(families["B"]["molecules"]), dtype=np.int64),
        "Q_bg": np.asarray(sorted(q_bg), dtype=np.int64),
    }
    target_keys = {k: np.asarray(families[k]["keys"], dtype=np.int64) for k in families}
    return {"roles": roles, "families": families, "target_keys": target_keys,
            "cfit_background": cfit_bg, "sup2": sup2, "inv": inv, "parent": parent,
            "mol2": mol2, "sup1": sup1, "n_key2": n_key2, "n_key1": n_key1}


def gates(built: dict, verbose: bool = True) -> dict:
    roles = built["roles"]
    sup2 = built["sup2"]
    mol2 = built["mol2"]
    n_key2 = built["n_key2"]
    tk = built["target_keys"]

    overlaps = {}
    names = list(roles)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(roles[names[i]].tolist()), set(roles[names[j]].tolist())
            overlaps[f"{names[i]}&{names[j]}"] = len(a & b)
    union = np.unique(np.concatenate([roles[r] for r in names]))
    support = {name: Z.key_support(roles[name], mol2, n_key2) for name in roles}

    per_key = {}
    for fam in ("A", "B", "C", "D"):
        sup_f = support["F"][tk[fam]]
        sup_fc = support["F"][tk[fam]] + support["C_fit"][tk[fam]]
        sup_fcd = sup_fc + support["C_dev"][tk[fam]]
        per_key[fam] = {
            "n_keys": int(len(tk[fam])),
            "max_support_in_F": int(sup_f.max()) if len(sup_f) else 0,
            "max_support_in_F_plus_Cfit": int(sup_fc.max()) if len(sup_fc) else 0,
            "max_support_in_F_plus_Cfit_plus_Cdev": int(sup_fcd.max()) if len(sup_fcd) else 0,
            "support_in_10k_min": int(sup2[tk[fam]].min()) if len(tk[fam]) else 0,
            "support_in_10k_max": int(sup2[tk[fam]].max()) if len(tk[fam]) else 0,
            "parent_keys": int(len(np.unique(built["parent"][tk[fam]]))),
        }
    parent_a = set(int(x) for x in built["parent"][tk["A"]])
    parent_b = set(int(x) for x in built["parent"][tk["B"]])
    checks = {
        "roles_disjoint": all(v == 0 for v in overlaps.values()),
        "roles_cover_train_10k": bool(len(union) == Z.N_TRAIN),
        "support_F_zero_A": per_key["A"]["max_support_in_F"] == 0,
        "support_F_zero_B": per_key["B"]["max_support_in_F"] == 0,
        "support_F_zero_C": per_key["C"]["max_support_in_F"] == 0,
        "support_F_zero_D": per_key["D"]["max_support_in_F"] == 0,
        "support_F_Cfit_zero_D": per_key["D"]["max_support_in_F_plus_Cfit"] == 0,
        "support_F_Cfit_Cdev_zero_A": per_key["A"]["max_support_in_F_plus_Cfit_plus_Cdev"] == 0,
        "support_F_Cfit_Cdev_zero_B": per_key["B"]["max_support_in_F_plus_Cfit_plus_Cdev"] == 0,
        "keys_A_disjoint_B": len(set(tk["A"].tolist()) & set(tk["B"].tolist())) == 0,
    }
    # gate m_F: molecules with >= 1 atom whose k2 has zero support in the base fit
    atom_k1, atom_k2, _ = Z.atom_tables(Z.load_keys(), n_key2)
    atom_ptr = np.asarray(D.load_cache("train")["atom_ptr"], dtype=np.int64)
    sup_f = support["F"]
    gate = {}
    for name in roles:
        rows = roles[name]
        any_unseen = 0
        frac = []
        for m in rows.tolist():
            s, e = int(atom_ptr[m]), int(atom_ptr[m + 1])
            u = sup_f[atom_k2[s:e]] == 0
            any_unseen += int(u.any())
            frac.append(float(u.mean()))
        gate[name] = {"n": int(len(rows)), "n_gated": int(any_unseen),
                      "gated_fraction": float(any_unseen / max(len(rows), 1)),
                      "mean_unseen_atom_fraction": float(np.mean(frac)) if len(frac) else 0.0}
    # background composition: which keys are also unseen in F
    qbg_unseen_keys = np.unique(np.concatenate(
        [np.asarray([k for k in mol2[m] if sup_f[k] == 0], dtype=np.int64) for m in roles["Q_bg"].tolist()]))
    comps = {fam: components(tk[fam], built["inv"]) for fam in ("A", "B", "C", "D")}
    out = {
        "overlaps": overlaps,
        "sizes": {r: int(len(roles[r])) for r in roles},
        "per_family_key_support": per_key,
        "parent_overlap_A_B": int(len(parent_a & parent_b)),
        "parent_overlap_A_B_fraction_of_A": float(len(parent_a & parent_b) / max(len(parent_a), 1)),
        "gate": gate,
        "q_bg_unseen_key_count": int(len(qbg_unseen_keys)),
        "q_bg_keys_unseen_in_F": int(qbg_unseen_keys.size),
        "components": {f: {"n_components": len(comps[f]),
                           "sizes": sorted((len(c) for c in comps[f]), reverse=True)[:8]}
                       for f in comps},
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }
    if verbose:
        print("[gates]", {k: v for k, v in checks.items()}, flush=True)
        print("[sizes]", out["sizes"], flush=True)
        print("[gate m_F]", gate, flush=True)
    return out, comps


def C_sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Z.CACHE))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = build_families()
    gate, comps = gates(built)
    if not gate["all_pass"]:
        raise SystemExit(f"split gates failed: {gate['checks']}")

    roles = built["roles"]
    np.savez_compressed(
        out_dir / "roles.npz",
        **{f"role_{k}": v for k, v in roles.items()},
        **{f"theme_{k}": v for k, v in built["target_keys"].items()},
        cfit_background=np.asarray(built["cfit_background"], dtype=np.int64),
    )
    np.savez_compressed(
        out_dir / "support_10k.npz",
        key_support_10k=built["sup2"].astype(np.int32),
        n_key1=np.asarray([built["n_key1"]]),
        n_key2=np.asarray([built["n_key2"]]),
    )
    manifest = {
        "round": "zinc-support-transfer-probe-v1",
        "rng_seed": int(Z.SPLIT_RNG_SEED),
        "candidate_rule": {"k2_support_lo": Z.KEY_SUP_LO, "k2_support_hi": Z.KEY_SUP_HI,
                           "parent_k1_support_min": Z.PARENT_MIN,
                           "n_candidates": int(len(np.flatnonzero(
                               (built["sup2"] >= Z.KEY_SUP_LO) & (built["sup2"] <= Z.KEY_SUP_HI)))),
                           "builder": "zct_keys.k2 = (atom_type, sorted(bond_type, k1(neighbour)))"},
        "roles": {k: v.tolist() for k, v in roles.items()},
        "target_keys": {k: v.tolist() for k, v in built["target_keys"].items()},
        "families": built["families"],
        "gates": gate,
        "components": {f: comps[f] for f in comps},
    }
    Z.write_json(out_dir / "split_manifest.json", manifest)

    # committed evidence: compact manifest (role arrays replaced by hashes) plus
    # the per-molecule role assignment
    import hashlib

    compact = {k: v for k, v in manifest.items()}
    compact["families"] = {
        name: {k: v for k, v in f.items() if k != "molecules"} | {"n_molecules": len(f["molecules"])}
        for name, f in manifest["families"].items()}
    compact["role_sizes"] = {k: int(len(v)) for k, v in roles.items()}
    compact["role_sha256"] = {
        k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.int64).tobytes()).hexdigest()
        for k, v in roles.items()}
    compact["roles_npz"] = "cache/roles.npz"
    compact["roles_npz_sha256"] = C_sha256(out_dir / "roles.npz")
    compact["support_10k_npz_sha256"] = C_sha256(out_dir / "support_10k.npz")
    Z.write_json(Z.ANALYSIS / "split_manifest.json", compact)
    role_of = np.full(Z.N_TRAIN, "unassigned", dtype=object)
    for k, v in roles.items():
        role_of[v] = k
    if (role_of == "unassigned").any():
        raise RuntimeError("some train molecules have no role")
    with open(Z.ANALYSIS / "role_assignment.csv", "w", encoding="utf-8") as fh:
        fh.write("molecule_id,role\n")
        for i in range(Z.N_TRAIN):
            fh.write(f"{i},{role_of[i]}\n")
    print(f"[split] F={len(roles['F'])} C_fit={len(roles['C_fit'])} C_dev={len(roles['C_dev'])} "
          f"Q_A={len(roles['Q_A'])} Q_B={len(roles['Q_B'])} Q_bg={len(roles['Q_bg'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
