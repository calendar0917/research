"""Local context support at radii 1/2/3 around the C-side attachment atom.

CPU-only, label-free structural analysis.  The canonical identity is the
parent's colour-encoded RDKit SMILES over raw atomic/bond attributes, with a
port marker at the C-side attachment atom.  The primary key appends the raw
cut-bond 3-tuple, matching the model's context identity; a structure-only key
without the cut bond is also reported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C
import trc_data as D


def _quantile_dict(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(list(values), np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "p50": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def _bond_suffix(bond_row: int, bond_feat: np.ndarray) -> str:
    if bond_row < 0:
        return "|b:null"
    return "|b:" + ",".join(str(int(x)) for x in bond_feat[bond_row].tolist())


def _build_parse_c_offsets(pack: Mapping[str, Any]) -> np.ndarray:
    """Per-parse offset into the per-molecule concatenated parse_C_atoms array."""
    n_mol = int(pack["n_molecules"])
    mol_parse_ptr = np.asarray(pack["mol_parse_ptr"], np.int64)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    frag_natoms = np.asarray(pack["frag_natoms"], np.int64)
    c_atoms_ptr = np.asarray(pack["parse_C_atoms_ptr"], np.int64)
    offsets = np.zeros(int(pack["n_parses"]), np.int64)
    for m in range(n_mol):
        p0, p1 = int(mol_parse_ptr[m]), int(mol_parse_ptr[m + 1])
        pos = int(c_atoms_ptr[m])
        for p in range(p0, p1):
            offsets[p] = pos
            cid = int(parse_C[p])
            pos += int(frag_natoms[cid]) if cid >= 0 else 0
        if pos != int(c_atoms_ptr[m + 1]):
            raise ValueError(f"parse_C_atoms offset mismatch at molecule {m}")
    return offsets


def _summarize(radius: str, train: Mapping[str, Any], valid: Mapping[str, Any],
               all_train_S: set[str], complete: bool = False,
               structure_only: bool = False) -> dict[str, Any]:
    train_set = set(train["keys"])
    train_s = all_train_S
    valid_keys = valid["keys"]
    valid_s = valid["S"]
    valid_mol = np.asarray(valid["mol"], np.int64)
    valid_contexts = set(valid_keys)
    seen_contexts = {k for k in valid_contexts if k in train_set}

    def mol_rates(joint: bool = False) -> dict[str, float]:
        groups: dict[int, list[bool]] = defaultdict(list)
        for i, m in enumerate(valid_mol.tolist()):
            if joint:
                ok = (valid_keys[i] in train_set) and ((valid_keys[i], valid_s[i]) in train["joint"])
            else:
                ok = valid_keys[i] in train_set
            groups[int(m)].append(bool(ok))
        if not groups:
            return {"n_molecules": 0}
        frac = np.asarray([np.mean(v) for v in groups.values()], np.float64)
        any_seen = np.asarray([any(v) for v in groups.values()], np.float64)
        return {
            "n_molecules": int(len(groups)),
            "mean_parse_seen_fraction": float(np.mean(frac)),
            "any_parse_seen_rate": float(np.mean(any_seen)),
        }

    # 2x2 support for valid real parses
    s_seen = np.asarray([s in train_s for s in valid_s], bool)
    c_seen = np.asarray([k in train_set for k in valid_keys], bool)
    support_2x2 = {}
    for a, b, name in ((True, True, "seen_context_seen_S"),
                       (True, False, "seen_context_unseen_S"),
                       (False, True, "unseen_context_seen_S"),
                       (False, False, "unseen_context_unseen_S")):
        m = (c_seen == a) & (s_seen == b)
        support_2x2[name] = {
            "n_parses": int(m.sum()),
            "fraction_of_valid_parses": float(m.mean()) if m.size else 0.0,
        }

    # reuse of seen valid contexts in Train: distinct Train molecules / parses
    reuse_mols = [len(train["molecules"][k]) for k in seen_contexts]
    reuse_parses = [train["parse_count"][k] for k in seen_contexts]
    reuse_s = [len(train["S_by_context"].get(k, set())) for k in seen_contexts]
    reuse_all_mols = [len(v) for v in train["molecules"].values()]

    return {
        "radius": radius,
        "complete_C": bool(complete),
        "structure_only": bool(structure_only),
        "definition": ("all C atoms" if complete else
                       f"C-side atoms within graph distance <= {radius} from the C-side port"),
        "train": {
            "n_parses": int(len(train["keys"])),
            "n_molecules": int(len(set(train["mol"]))),
            "n_distinct_contexts": int(len(train_set)),
            "n_distinct_S_types": int(len(train_s)),
            "n_distinct_joint_context_S": int(len(train["joint"])),
            "context_size_quantiles": _quantile_dict(train["sizes"]),
            "context_molecule_reuse": _quantile_dict(reuse_all_mols),
            "fraction_contexts_in_ge2_molecules": float(np.mean(np.asarray(reuse_all_mols) >= 2)) if reuse_all_mols else float("nan"),
            "fraction_contexts_in_ge5_molecules": float(np.mean(np.asarray(reuse_all_mols) >= 5)) if reuse_all_mols else float("nan"),
            "fraction_contexts_in_ge10_molecules": float(np.mean(np.asarray(reuse_all_mols) >= 10)) if reuse_all_mols else float("nan"),
        },
        "valid": {
            "n_parses": int(len(valid_keys)),
            "n_molecules": int(len(set(valid_mol.tolist()))),
            "n_distinct_contexts": int(len(valid_contexts)),
            "n_distinct_S_types": int(len(set(valid_s))),
            "context_size_quantiles": _quantile_dict(valid["sizes"]),
            "seen_parse_rate": float(c_seen.mean()) if c_seen.size else 0.0,
            "seen_unique_context_rate": float(len(seen_contexts) / max(1, len(valid_contexts))),
            "seen_molecule_any_context_rate": mol_rates(False).get("any_parse_seen_rate", float("nan")),
            "seen_molecule_mean_parse_fraction": mol_rates(False).get("mean_parse_seen_fraction", float("nan")),
            "joint_context_S_seen_parse_rate": float(np.mean(
                [(k, s) in train["joint"] for k, s in zip(valid_keys, valid_s)])) if valid_keys else 0.0,
            "joint_context_S_seen_molecule_any_rate": mol_rates(True).get("any_parse_seen_rate", float("nan")),
            "support_2x2": support_2x2,
            "seen_context_reuse_in_train": {
                "n_seen_contexts": int(len(seen_contexts)),
                "n_unseen_contexts": int(len(valid_contexts) - len(seen_contexts)),
                "train_molecules_per_seen_context": _quantile_dict(reuse_mols),
                "train_parses_per_seen_context": _quantile_dict(reuse_parses),
                "train_S_types_per_seen_context": _quantile_dict(reuse_s),
                "fraction_seen_contexts_in_ge2_train_molecules": float(np.mean(np.asarray(reuse_mols) >= 2)) if reuse_mols else float("nan"),
                "fraction_seen_contexts_in_ge5_train_molecules": float(np.mean(np.asarray(reuse_mols) >= 5)) if reuse_mols else float("nan"),
                "fraction_seen_contexts_in_ge10_train_molecules": float(np.mean(np.asarray(reuse_mols) >= 10)) if reuse_mols else float("nan"),
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--max-parses", type=int, default=0,
                    help="debug only: limit real parses")
    args = ap.parse_args(argv)

    cfg = C.load_config()
    pack = C.load_pack(cfg, args.dataset)
    raw = C.raw_cache(cfg, args.dataset)
    colors = D.ColorSpace.from_raw(raw)

    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat_all = np.asarray(raw["atom_feat"], np.uint8)
    bond_feat_all = np.asarray(raw["bond_feat"], np.uint8)
    bond_u_all = np.asarray(raw["bond_u"], np.int32)
    bond_v_all = np.asarray(raw["bond_v"], np.int32)

    parse_mol = np.asarray(pack["parse_mol"], np.int64)
    parse_kind = np.asarray(pack["parse_kind"], np.uint8)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    parse_S = np.asarray(pack["parse_S"], np.int64)
    parse_port_C = np.asarray(pack["parse_port_C"], np.int64)
    parse_bond_index = np.asarray(pack["parse_bond_index"], np.int64)
    frag_natoms = np.asarray(pack["frag_natoms"], np.int64)
    fragment_keys = np.asarray(pack["fragment_keys"]).astype(str)
    molecule_split = np.asarray(pack["molecule_split"], np.uint8)
    c_atoms = np.asarray(pack["parse_C_atoms"], np.int64)
    c_offsets = _build_parse_c_offsets(pack)
    mol_parse_ptr = np.asarray(pack["mol_parse_ptr"], np.int64)

    real_parses = np.nonzero(parse_kind == 0)[0]
    # Only Train and Valid are analyzed.  Test parse structure is not touched.
    real_parses = real_parses[molecule_split[parse_mol[real_parses]] != 2]
    if args.max_parses:
        real_parses = real_parses[:int(args.max_parses)]

    variants = ["radius_1", "radius_2", "radius_3", "complete_C"]
    struct_variants = [f"{v}_structure_only" for v in variants]
    all_variants = variants + struct_variants
    train: dict[str, dict[str, Any]] = {
        v: {"keys": [], "S": [], "mol": [], "sizes": [],
            "molecules": defaultdict(set), "parse_count": Counter(),
            "S_by_context": defaultdict(set), "joint": set()}
        for v in all_variants
    }
    valid: dict[str, dict[str, Any]] = {
        v: {"keys": [], "S": [], "mol": [], "sizes": []} for v in all_variants
    }
    all_train_S = {
        str(fragment_keys[int(parse_S[p])])
        for p in range(int(pack["n_parses"]))
        if int(molecule_split[int(parse_mol[p])]) == 0
    }
    complete_key_matches = 0
    complete_key_total = 0
    renumber_checks = 0
    renumber_ok = 0

    # group parses by molecule so the raw molecule is loaded once
    by_mol: dict[int, list[int]] = defaultdict(list)
    for p in real_parses.tolist():
        by_mol[int(parse_mol[p])].append(int(p))

    for row in sorted(by_mol):
        parses = by_mol[row]
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n_atoms = a1 - a0
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
        for e in range(b0, b1):
            u, v = int(bond_u_all[e]), int(bond_v_all[e])
            adj[u].append((v, e - b0))
            adj[v].append((u, e - b0))
        atom_feat = atom_feat_all[a0:a1]
        bond_feat = bond_feat_all[b0:b1]
        bond_u = bond_u_all[b0:b1]
        bond_v = bond_v_all[b0:b1]
        split = int(molecule_split[row])
        if split == 2:
            continue

        cache: dict[tuple[tuple[int, ...], int], str] = {}
        for p in parses:
            cid = int(parse_C[p])
            if cid < 0:
                continue
            c_size = int(frag_natoms[cid])
            c_atoms_this = c_atoms[c_offsets[p]:c_offsets[p] + c_size]
            c_set = set(int(x) for x in c_atoms_this.tolist())
            port = int(parse_port_C[p])
            if port not in c_set:
                raise ValueError(f"port {port} not in C atoms for parse {p}")
            dist = {port: 0}
            frontier = [port]
            while frontier:
                nxt: list[int] = []
                for u in frontier:
                    for v, _e in adj[u]:
                        if v in c_set and v not in dist:
                            dist[v] = dist[u] + 1
                            nxt.append(v)
                frontier = nxt
            bond_suffix = _bond_suffix(int(parse_bond_index[p]), bond_feat)

            for radius in (1, 2, 3):
                atoms = tuple(sorted(a for a, d in dist.items() if d <= radius))
                ck = (atoms, port)
                if ck not in cache:
                    idx = {a: i for i, a in enumerate(atoms)}
                    local_bonds = []
                    for e in range(len(bond_u)):
                        u, v = int(bond_u[e]), int(bond_v[e])
                        if u in idx and v in idx:
                            i, j = idx[u], idx[v]
                            local_bonds.append((min(i, j), max(i, j), bond_feat[e]))
                    key = D.canonical_key(atom_feat[list(atoms)], local_bonds,
                                          idx[port], colors)
                    cache[ck] = key
                key_struct = cache[ck]
                key = key_struct + bond_suffix
                vname = f"radius_{radius}"
                skey = str(fragment_keys[int(parse_S[p])])
                for vname, k in ((f"radius_{radius}", key),
                                 (f"radius_{radius}_structure_only", key_struct)):
                    target = train[vname] if split == 0 else valid[vname]
                    target["keys"].append(k)
                    target["S"].append(skey)
                    target["mol"].append(row)
                    target["sizes"].append(len(atoms))
                    if split == 0:
                        target["molecules"][k].add(row)
                        target["parse_count"][k] += 1
                        target["S_by_context"][k].add(skey)
                        target["joint"].add((k, skey))

            # complete C: use the parent's canonical C key directly
            key_struct = str(fragment_keys[cid])
            key = key_struct + bond_suffix
            skey = str(fragment_keys[int(parse_S[p])])
            for vname, k in (("complete_C", key),
                             ("complete_C_structure_only", key_struct)):
                target = train[vname] if split == 0 else valid[vname]
                target["keys"].append(k)
                target["S"].append(skey)
                target["mol"].append(row)
                target["sizes"].append(c_size)
                if split == 0:
                    target["molecules"][k].add(row)
                    target["parse_count"][k] += 1
                    target["S_by_context"][k].add(skey)
                    target["joint"].add((k, skey))

            # canonicalisation sanity: if the radius covers all C atoms, it must
            # reproduce the parent's C structure key
            if len(dist) == c_size:
                complete_key_total += 1
                # recompute complete structure key from the local context
                atoms = tuple(sorted(c_set))
                idx = {a: i for i, a in enumerate(atoms)}
                local_bonds = []
                for e in range(len(bond_u)):
                    u, v = int(bond_u[e]), int(bond_v[e])
                    if u in idx and v in idx:
                        i, j = idx[u], idx[v]
                        local_bonds.append((min(i, j), max(i, j), bond_feat[e]))
                k2 = D.canonical_key(atom_feat[list(atoms)], local_bonds, idx[port], colors)
                if k2 == str(fragment_keys[cid]):
                    complete_key_matches += 1

            # renumbering invariance on a small sample
            if renumber_checks < 60:
                atoms = tuple(sorted(dist))
                idx = {a: i for i, a in enumerate(atoms)}
                local_bonds = []
                for e in range(len(bond_u)):
                    u, v = int(bond_u[e]), int(bond_v[e])
                    if u in idx and v in idx:
                        i, j = idx[u], idx[v]
                        local_bonds.append((min(i, j), max(i, j), bond_feat[e]))
                k0 = D.canonical_key(atom_feat[list(atoms)], local_bonds, idx[port], colors)
                rng = np.random.default_rng(1234 + renumber_checks)
                perm = rng.permutation(len(atoms))
                inv = {int(old): int(new) for new, old in enumerate(perm.tolist())}
                atoms_p = [atoms[int(i)] for i in perm.tolist()]
                idx_p = {a: i for i, a in enumerate(atoms_p)}
                bonds_p = []
                for i, j, br in local_bonds:
                    a, b = atoms[i], atoms[j]
                    ii, jj = idx_p[a], idx_p[b]
                    bonds_p.append((min(ii, jj), max(ii, jj), br))
                k1 = D.canonical_key([atom_feat[a] for a in atoms_p], bonds_p,
                                     idx_p[port], colors)
                renumber_checks += 1
                renumber_ok += int(k0 == k1)

    results = {
        v: _summarize(v, train[v], valid[v], all_train_S,
                      complete=(v == "complete_C"))
        for v in variants
    }
    results_structure_only = {
        v: _summarize(v.replace("_structure_only", ""), train[v], valid[v], all_train_S,
                      complete=v.startswith("complete_C"), structure_only=True)
        for v in struct_variants
    }
    # cross-check the complete-C parse-weighted seen rate against the existing
    # Train-only support report
    support_path = C.CSOI_ROUND / "results" / "support.json"
    crosscheck = None
    if support_path.is_file():
        sup = C.read_json(support_path)["coverage"]
        crosscheck = {
            "existing_support_valid_C_seen_rate": sup["valid_C_seen_rate"],
            "recomputed_complete_C_seen_parse_rate_real_parses_only": results["complete_C"]["valid"]["seen_parse_rate"],
            "note": ("the existing rate includes any valid parse denominator; this analysis "
                     "uses real bridge parses only"),
        }

    out = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack["cache_sha256"],
        "cache_content_sha256": pack.get("content_sha256"),
        "cache_path_used": pack["cache_path_used"],
        "test_policy": "no labels used; structure and split membership only",
        "n_real_parses": int(real_parses.size),
        "n_train_real_parses": int(sum(1 for p in real_parses if molecule_split[int(parse_mol[p])] == 0)),
        "n_valid_real_parses": int(sum(1 for p in real_parses if molecule_split[int(parse_mol[p])] == 1)),
        "n_test_real_parses_not_analyzed": int(sum(1 for p in real_parses if molecule_split[int(parse_mol[p])] == 2)),
        "radius_definition": {
            "1": "C atoms with graph distance <=1 from the C-side cut endpoint, induced subgraph",
            "2": "C atoms with graph distance <=2 from the C-side cut endpoint, induced subgraph",
            "3": "C atoms with graph distance <=3 from the C-side cut endpoint, induced subgraph",
            "complete_C": "all atoms in C = V(G) \\ S, exactly the model's context object",
        },
        "identity": {
            "encoding": "parent canonical_key: RDKit canonical SMILES over colour-encoded raw attributes, port marker at C-side endpoint",
            "primary_key": "canonical structure + raw cut-bond 3-tuple (same semantics as the model's C key)",
            "structure_only": "canonical structure without the cut-bond suffix",
            "renumber_checks": renumber_checks,
            "renumber_ok": renumber_ok,
            "renumber_invariance_ok": bool(renumber_checks > 0 and renumber_ok == renumber_checks),
            "complete_C_key_reconstruction_checks": complete_key_total,
            "complete_C_key_reconstruction_matches": complete_key_matches,
        },
        "complete_C_support_crosscheck": crosscheck,
        "results_primary_key_with_cut_bond": results,
        "results_structure_only_key_without_cut_bond": results_structure_only,
    }
    out_path = C.analysis_dir() / "local_context_support.json"
    C.write_json(out_path, out)
    print(f"wrote {out_path}")
    for v in variants + struct_variants:
        r = (results if v in results else results_structure_only)[v]
        print(v, "train distinct", r["train"]["n_distinct_contexts"],
              "valid seen parse", round(r["valid"]["seen_parse_rate"], 4),
              "joint", round(r["valid"]["joint_context_S_seen_parse_rate"], 4),
              "reuse>=2", round(r["valid"]["seen_context_reuse_in_train"].get(
                  "fraction_seen_contexts_in_ge2_train_molecules", float("nan")), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())