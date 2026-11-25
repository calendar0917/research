"""From computed message attributions back to readable structural regularities.

Section 12 of ``PROTOCOL.md``.  Nothing here re-trains a structure vocabulary or
searches a response dictionary.  The input is the exact per-message task
attribution ``A_ji`` of the *actual* model computation (``cs_decompose.py``).

Three analyses, all fitted on Train only:

1. repeated (source structural record, relation type) combinations and whether
   the same structure carries the same-signed effect in different environments,
2. one fixed light regression tree per input set (``S`` source structure,
   ``C`` environment, ``SC`` both) with molecule-equal message weights,
3. up to five candidate regularities chosen from Train only, then checked for
   support and sign continuation on Valid/Test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import cs_data as D  # noqa: E402


# ---------------------------------------------------------------------------
# structural feature construction
# ---------------------------------------------------------------------------


def field_offsets(spec) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for f in spec.atom_feature_fields:
        out[str(f.name)] = (off, off + int(f.dim))
        off += int(f.dim)
    if off != spec.atom_feat_dim:
        raise RuntimeError(f"atom field table covers {off} != {spec.atom_feat_dim}")
    return out


def _cat(atom_feat: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    lo, hi = span
    return atom_feat[:, lo:hi].argmax(axis=1)


def atom_readable(atom_feat: np.ndarray, spec) -> dict[str, np.ndarray]:
    o = field_offsets(spec)
    out = {
        "atom_type": _cat(atom_feat, o["atom_type"]),
        "formal_charge": _cat(atom_feat, o["formal_charge"]),
        "n_hydrogens": _cat(atom_feat, o["n_hydrogens"]),
        "degree": _cat(atom_feat, o["degree"]),
        "in_ring": atom_feat[:, o["in_ring"][0]],
        "min_ring_size": _cat(atom_feat, o["min_ring_size"]),
        "ring_bond_count_over_4": atom_feat[:, o["ring_bond_count_over_4"][0]],
        "ring_bond_fraction": atom_feat[:, o["ring_bond_fraction"][0]],
        "is_terminal": atom_feat[:, o["is_terminal"][0]],
        "is_fused_or_spiro": atom_feat[:, o["is_fused_or_spiro"][0]],
    }
    lo, hi = o["incident_bond_class_fraction"]
    for i in range(hi - lo):
        out[f"incident_bond_class_fraction_{i}"] = atom_feat[:, lo + i]
    return out


def per_atom_code_counts(cache: Mapping[str, Any], code_key: str, ptr_key: str, vocab: np.ndarray) -> np.ndarray:
    """``counts[n_atoms, len(vocab)]`` of a per-atom structural code (S1 or S2)."""
    n_atoms = int(np.asarray(cache["atom_ptr"])[-1])
    ptr = np.asarray(cache[ptr_key], dtype=np.int64)
    codes = np.asarray(cache[code_key], dtype=np.int64)
    counts = np.zeros((n_atoms, int(vocab.size)), dtype=np.float32)
    index = {int(code): j for j, code in enumerate(vocab.tolist())}
    for atom in range(n_atoms):
        for code in codes[ptr[atom]:ptr[atom + 1]].tolist():
            j = index.get(int(code))
            if j is not None:
                counts[atom, j] += 1.0
    return counts


def top_codes(cache: Mapping[str, Any], code_key: str, ptr_key: str, n_top: int) -> np.ndarray:
    codes = np.asarray(cache[code_key], dtype=np.int64)
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts, kind="stable")[: int(n_top)]
    return uniq[order]


def structure_signature(cache: Mapping[str, Any], atom: int) -> str:
    """Exact structural identity of one atom's LCRC local record."""
    ptr_n = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    ptr_p = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr = np.sort(np.asarray(cache["nbr_code"], dtype=np.int64)[ptr_n[atom]:ptr_n[atom + 1]])
    path = np.asarray(cache["path_cat"], dtype=np.int64)[ptr_p[atom]:ptr_p[atom + 1]]
    end = np.asarray(cache["path_end63"], dtype=np.int64)[ptr_p[atom]:ptr_p[atom + 1]]
    clo = np.asarray(cache["path_closure"], dtype=np.int64)[ptr_p[atom]:ptr_p[atom + 1]]
    order = np.lexsort((end, clo, path)) if path.size else np.zeros(0, dtype=np.int64)
    a_feat = np.asarray(cache["atom_feat"])[atom]
    payload = b"|".join(
        [
            (a_feat * 1000).round().astype(np.int32).tobytes(),
            nbr.astype(np.int32).tobytes(),
            path[order].astype(np.int32).tobytes(),
            end[order].astype(np.int32).tobytes(),
            clo[order].astype(np.int8).tobytes(),
        ]
    )
    return C.sha256_bytes(payload)[:16]


# ---------------------------------------------------------------------------
# loading the decomposition artefacts
# ---------------------------------------------------------------------------


def load_messages(dataset: str, arm: str, split: str, bundle) -> dict[str, Any]:
    path = C.RESULTS_DIR / "decomposition" / f"{dataset.replace('ogbg-', '')}_{arm}_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    cache = bundle.caches[split]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_off = atom_ptr[:-1]
    atom_feat = np.asarray(cache["atom_feat"])
    readable = atom_readable(atom_feat, bundle.spec)

    msg_mol = z["msg_mol"]
    src_local = z["msg_src"] + atom_off[msg_mol]
    dst_local = z["msg_dst"] + atom_off[msg_mol]
    # physical bond position, for the real bond attributes
    phys = np.zeros(msg_mol.size, dtype=np.int64)
    for i in range(msg_mol.size):
        m = int(msg_mol[i])
        u, v = int(src_local[i]), int(dst_local[i])
        uu, vv = min(u, v), max(u, v)
        lo, hi = bond_ptr[m], bond_ptr[m + 1]
        bi = np.asarray(cache["bond_index"])[lo:hi] + atom_off[m]
        hit = np.nonzero((bi[:, 0] == uu) & (bi[:, 1] == vv))[0]
        phys[i] = lo + (int(hit[0]) if hit.size else -1)
    out = {
        "split": split,
        "mol_row": z["mol_row"],
        "mol_pos": z["mol_pos"],
        "msg_mol": msg_mol,
        "src_atom": src_local,
        "dst_atom": dst_local,
        "bond_type": z["msg_bond_type"],
        "rel": z["msg_rel"],
        "A": z["A_raw"],
        "A_extra": z["A_extra_raw"],
        "v_rms": z["v_rms"],
        "v_extra_rms": z["v_extra_rms"],
        "p_rms": z["p_rms"],
        "gamma_rms": z["gamma_rms"],
        "delta_abs": z["delta_msg_abs"],
        "node_degree": z["node_degree"],
        "node_n_hetero": z["node_n_hetero"],
        "y_raw": z["y_raw"],
        "pred_raw": z["pred_std"] * float(bundle.scaler["std"]) + float(bundle.scaler["mean"]),
        "n_molecules": int(z["mol_row"].size),
        "atom_feat": atom_feat,
        "readable": readable,
        "cache": cache,
        "phys": phys,
        "bond_feat": np.asarray(cache["bond_feat"])[phys] if phys.size else np.zeros((0, bundle.spec.bond_feat_dim), np.float32),
    }
    return out


def build_features(msg: Mapping[str, Any], spec, vocab_s1: np.ndarray, vocab_s2: np.ndarray) -> dict[str, Any]:
    """Readable S / C / SC feature matrices for every real message."""
    cache = msg["cache"]
    readable = msg["readable"]
    s1 = per_atom_code_counts(cache, "nbr_code", "nbr_ptr", vocab_s1)
    s2 = per_atom_code_counts(cache, "path_cat", "path_ptr", vocab_s2)
    src, dst = msg["src_atom"], msg["dst_atom"]
    names: list[str] = []
    cols: list[np.ndarray] = []
    sides: list[str] = []

    def push(name: str, values: np.ndarray, side: str) -> None:
        names.append(name)
        cols.append(np.asarray(values, dtype=np.float64))
        sides.append(side)

    for tag, ids in (("src", src), ("dst", dst)):
        side = "S" if tag == "src" else "C"
        for key, values in readable.items():
            push(f"{tag}_{key}", values[ids], side)
        for j in range(s1.shape[1]):
            push(f"{tag}_s1_{int(vocab_s1[j])}", s1[ids, j], side)
        for j in range(s2.shape[1]):
            push(f"{tag}_s2_{int(vocab_s2[j])}", s2[ids, j], side)

    rel = msg["rel"]
    for r, name in enumerate(C.RELATION_NAMES):
        push(f"bond_{name}", (rel == r).astype(np.float64), "S")

    n_r = np.zeros((dst.size, C.N_RELATIONS), dtype=np.float64)
    for i in range(dst.size):
        n_r[i, int(rel[i])] += 1.0
    dst_degree = msg["node_degree"][dst]
    indeg = np.maximum(dst_degree.astype(np.float64), 1.0)
    for r, name in enumerate(C.RELATION_NAMES):
        push(f"ctx_n_{name}", n_r[:, r], "C")
        push(f"ctx_relfrac_{name}", n_r[:, r] / indeg, "C")
    push("ctx_node_indegree", dst_degree, "C")
    push("ctx_node_n_hetero", msg["node_n_hetero"][dst], "C")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_atoms_all = (atom_ptr[1:] - atom_ptr[:-1]).astype(np.float64)
    push("ctx_molecule_n_atoms", n_atoms_all[msg["msg_mol"]], "C")
    g = np.asarray(cache["global"])
    push("ctx_molecule_ring_fraction", g[msg["msg_mol"], global_field(spec, "ring_atom_fraction")], "C")

    matrix = np.stack(cols, axis=1) if cols else np.zeros((0, 0))
    return {
        "names": names,
        "sides": np.asarray(sides),
        "matrix": matrix,
        "by_side": {
            "S": np.asarray([i for i, x in enumerate(sides) if x == "S"], dtype=np.int64),
            "C": np.asarray([i for i, x in enumerate(sides) if x == "C"], dtype=np.int64),
            "SC": np.arange(len(names), dtype=np.int64),
        },
    }


def global_field(spec, name: str) -> int:
    off = 0
    for f in spec.global_feature_fields:
        if str(f.name) == name:
            if int(f.dim) != 1:
                raise ValueError(f"global field {name} is not 1-d")
            return off
        off += int(f.dim)
    raise KeyError(name)


def molecule_weights(msg: Mapping[str, Any]) -> np.ndarray:
    """Message weights so that every molecule carries the same total weight."""
    deg = np.zeros(msg["n_molecules"], dtype=np.float64)
    np.add.at(deg, msg["msg_mol"], 1.0)
    w = 1.0 / np.maximum(deg[msg["msg_mol"]], 1.0)
    return w / w.sum() * msg["n_molecules"]


# ---------------------------------------------------------------------------
# repeated structure analysis
# ---------------------------------------------------------------------------


def repeated_structures(train: Mapping[str, Any], others: Mapping[str, Mapping[str, Any]], spec, *, min_support: int = 3) -> dict[str, Any]:
    cache = train["cache"]
    n_atoms = int(np.asarray(cache["atom_ptr"])[-1])
    sig = [structure_signature(cache, a) for a in range(n_atoms)]
    sig_arr = np.asarray(sig, dtype=object)
    keys = np.asarray([f"{sig_arr[a]}|{int(b)}" for a, b in zip(train["src_atom"], train["bond_type"])], dtype=object)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    A = train["A"]
    mols = train["msg_mol"]
    rows = train["mol_row"][mols]  # official row id of the message's molecule
    groups: dict[str, Any] = {}
    entropies = []
    for gid in np.nonzero(counts >= int(min_support))[0]:
        sel = inv == gid
        vals = A[sel]
        n_mol = int(np.unique(mols[sel]).size)
        n_row = int(np.unique(rows[sel]).size)
        signs = np.sign(vals)
        pos = float((signs > 0).mean())
        med = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        groups[str(uniq[gid])] = {
            "n_messages": int(sel.sum()),
            "n_molecules": n_mol,
            "n_unique_rows": n_row,
            "median": med,
            "q25": float(q1),
            "q75": float(q3),
            "iqr": float(q3 - q1),
            "positive_fraction": pos,
            "sign_change_fraction": float(min(pos, 1.0 - pos) * 2.0),
            "abs_mean": float(np.abs(vals).mean()),
            "unique_molecules_ge_2": bool(n_row >= 2),
        }
        entropies.append(abs(med) / (abs(q3 - q1) + 1e-9))
    n_types = len(groups)
    repeated_messages = int(sum(g["n_messages"] for g in groups.values()))
    summary = {
        "definition": "exact atom-level LCRC structural record (atom features + S1 code multiset + S2 path code/end/closure multiset) x real bond type",
        "signature_bits": 64,
        "min_support_messages": int(min_support),
        "n_train_messages": int(A.size),
        "n_repeated_types": int(n_types),
        "n_repeated_messages": repeated_messages,
        "repeated_message_fraction": float(repeated_messages / max(1, A.size)),
        "n_types_with_2plus_train_molecules": int(sum(1 for g in groups.values() if g["n_molecules"] >= 2)),
        "n_types_with_2plus_train_rows": int(sum(1 for g in groups.values() if g["n_unique_rows"] >= 2)),
        "median_abs_effect": float(np.median([abs(g["median"]) for g in groups.values()])) if groups else 0.0,
        "median_sign_change_fraction": float(np.median([g["sign_change_fraction"] for g in groups.values()])) if groups else 0.0,
        "fraction_types_with_sign_change_gt_0.5": float(np.mean([g["sign_change_fraction"] > 0.5 for g in groups.values()])) if groups else 0.0,
        "median_effect_to_iqr_ratio": float(np.median(entropies)) if entropies else 0.0,
    }
    coverage = {}
    for split, msg in others.items():
        if split == "train":
            continue
        cache_o = msg["cache"]
        n_atoms_o = int(np.asarray(cache_o["atom_ptr"])[-1])
        sig_o = np.asarray([structure_signature(cache_o, a) for a in range(n_atoms_o)], dtype=object)
        keys_o = np.asarray([f"{sig_o[a]}|{int(b)}" for a, b in zip(msg["src_atom"], msg["bond_type"])], dtype=object)
        present = np.isin(keys_o, uniq[counts >= int(min_support)])
        coverage[split] = {
            "n_messages": int(keys_o.size),
            "n_messages_with_repeated_train_type": int(present.sum()),
            "coverage": float(present.mean()) if keys_o.size else 0.0,
        }
    return {"summary": summary, "groups": groups, "valid_test_coverage": coverage}


# ---------------------------------------------------------------------------
# explanation trees
# ---------------------------------------------------------------------------


def fit_trees(
    train: Mapping[str, Any],
    valid: Mapping[str, Any],
    test: Mapping[str, Any],
    spec,
    vocab_s1: np.ndarray,
    vocab_s2: np.ndarray,
) -> dict[str, Any]:
    from sklearn.tree import DecisionTreeRegressor

    f_tr = build_features(train, spec, vocab_s1, vocab_s2)
    w_tr = molecule_weights(train)
    y_tr = train["A"]
    constant = C.weighted_median(y_tr, w_tr * (y_tr == y_tr))
    out: dict[str, Any] = {
        "constant_predictor_train_weighted_median": float(constant),
        "target": "A_ji in raw label units (Train message attribution; no molecule labels are used)",
        "weights": "each molecule carries the same total weight",
        "tree": {
            "max_depth": C.TREE_MAX_DEPTH,
            "min_samples_leaf": C.TREE_MIN_LEAF,
            "random_state": C.TREE_RANDOM_STATE,
            "implementation": "sklearn.tree.DecisionTreeRegressor",
        },
        "inputs": {},
    }
    for name in ("S", "C", "SC"):
        cols = f_tr["by_side"][name]
        tree = DecisionTreeRegressor(
            max_depth=C.TREE_MAX_DEPTH,
            min_samples_leaf=C.TREE_MIN_LEAF,
            random_state=C.TREE_RANDOM_STATE,
        )
        tree.fit(f_tr["matrix"][:, cols], y_tr, sample_weight=w_tr)
        entry: dict[str, Any] = {
            "n_features": int(cols.size),
            "feature_names": [f_tr["names"][i] for i in cols],
            "train": _tree_metrics(tree, f_tr["matrix"][:, cols], y_tr, w_tr, constant),
            "valid": None,
            "test": None,
        }
        for split, msg in (("valid", valid), ("test", test)):
            f = build_features(msg, spec, vocab_s1, vocab_s2)
            w = molecule_weights(msg)
            entry[split] = _tree_metrics(tree, f["matrix"][:, cols], msg["A"], w, constant)
        out["inputs"][name] = entry
    # skill relative to the stronger of S / C
    for split in ("valid", "test"):
        s = out["inputs"]["S"][split]["mae"]
        c = out["inputs"]["C"][split]["mae"]
        sc = out["inputs"]["SC"][split]["mae"]
        best_single = min(s, c)
        out.setdefault("skill_vs_stronger_single", {})[split] = {
            "mae_S": s,
            "mae_C": c,
            "mae_SC": sc,
            "stronger_single": "S" if s <= c else "C",
            "relative_mae_improvement": float(1.0 - sc / best_single) if best_single else None,
            "skill_SC_vs_constant": out["inputs"]["SC"][split]["skill"],
            "skill_S_vs_constant": out["inputs"]["S"][split]["skill"],
            "skill_C_vs_constant": out["inputs"]["C"][split]["skill"],
        }
    return out


def _tree_metrics(tree, X: np.ndarray, y: np.ndarray, w: np.ndarray, constant: float) -> dict[str, Any]:
    pred = tree.predict(X) if X.size else np.zeros(y.shape)
    mae = C.weighted_mae(pred, y, w)
    mae_const = C.weighted_mae(np.full_like(y, constant), y, w)
    return {
        "n_messages": int(y.size),
        "mae": float(mae),
        "mae_constant": float(mae_const),
        "skill": float(1.0 - mae / mae_const) if mae_const > 1e-12 else None,
        "low_effect_amplitude": bool(mae_const <= 1e-12),
        "target_abs_mean": float(np.average(np.abs(y), weights=w)) if y.size else 0.0,
        "target_std": float(np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w))) if y.size else 0.0,
        "target_q05": float(np.percentile(y, 5)) if y.size else 0.0,
        "target_q95": float(np.percentile(y, 95)) if y.size else 0.0,
        "pred_abs_mean": float(np.average(np.abs(pred), weights=w)) if y.size else 0.0,
    }


# ---------------------------------------------------------------------------
# candidate regularities
# ---------------------------------------------------------------------------

CANDIDATE_MAX = 5


def candidate_regularities(
    train: Mapping[str, Any],
    valid: Mapping[str, Any],
    test: Mapping[str, Any],
    spec,
    vocab_s1: np.ndarray,
    vocab_s2: np.ndarray,
    *,
    min_train_rows: int = 3,
    min_train_messages: int = 20,
) -> dict[str, Any]:
    """Choose candidates on Train only, then check support/sign on Valid/Test."""
    from sklearn.tree import DecisionTreeRegressor

    f_tr = build_features(train, spec, vocab_s1, vocab_s2)
    w_tr = molecule_weights(train)
    y_tr = train["A"]
    cols = f_tr["by_side"]["SC"]
    tree = DecisionTreeRegressor(
        max_depth=C.TREE_MAX_DEPTH, min_samples_leaf=C.TREE_MIN_LEAF, random_state=C.TREE_RANDOM_STATE
    )
    tree.fit(f_tr["matrix"][:, cols], y_tr, sample_weight=w_tr)
    leaf_tr = tree.apply(f_tr["matrix"][:, cols])
    splits = {
        "valid": {"features": build_features(valid, spec, vocab_s1, vocab_s2), "msg": valid, "weights": molecule_weights(valid)},
        "test": {"features": build_features(test, spec, vocab_s1, vocab_s2), "msg": test, "weights": molecule_weights(test)},
    }
    for name, block in splits.items():
        block["leaf"] = tree.apply(block["features"]["matrix"][:, cols])

    scored = []
    for leaf in np.unique(leaf_tr):
        sel = leaf_tr == leaf
        n_msg = int(sel.sum())
        n_rows = int(np.unique(train["mol_row"][train["msg_mol"][sel]]).size)
        if n_msg < int(min_train_messages) or n_rows < int(min_train_rows):
            continue
        vals = y_tr[sel]
        med = float(np.median(vals))
        spread = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        scored.append(
            {
                "leaf": int(leaf),
                "train_messages": n_msg,
                "train_rows": n_rows,
                "train_median": med,
                "train_iqr": spread,
                "train_positive_fraction": float((vals > 0).mean()),
                "train_abs_mean": float(np.abs(vals).mean()),
                "score": abs(med) * np.log1p(n_msg) / (1.0 + spread),
            }
        )
    scored.sort(key=lambda d: -d["score"])
    chosen = scored[:CANDIDATE_MAX]

    out: dict[str, Any] = {
        "selection_rule": (
            "leaves of the fixed SC tree on Train only, ordered by |median effect| * log1p(support) / (1+IQR); "
            "Valid/Test never enter the choice"
        ),
        "min_train_rows": int(min_train_rows),
        "min_train_messages": int(min_train_messages),
        "n_candidates_available": len(scored),
        "candidates": [],
    }
    for rank, cand in enumerate(chosen, start=1):
        condition = readable_condition(tree, cand["leaf"], f_tr["names"], cols, train, spec)
        entry = {
            "rank": rank,
            **cand,
            "condition": condition,
        }
        for name, block in splits.items():
            sel = block["leaf"] == cand["leaf"]
            msg = block["msg"]
            vals = msg["A"][sel]
            rows = np.unique(msg["mol_row"][msg["msg_mol"][sel]])
            entry[name] = {
                "n_messages": int(sel.sum()),
                "n_molecules": int(rows.size),
                "n_scaffolds": int(len({msg["scaffold"][int(r)] for r in rows})),
                "coverage_fraction": float(sel.mean()) if sel.size else 0.0,
                "median_effect": float(np.median(vals)) if vals.size else None,
                "abs_mean": float(np.abs(vals).mean()) if vals.size else None,
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size else None,
                "same_sign_fraction": float(np.mean(np.sign(vals) == np.sign(cand["train_median"]))) if vals.size else None,
                "sign_continues": bool(
                    vals.size and np.sign(np.median(vals)) == np.sign(cand["train_median"])
                ),
            }
        out["candidates"].append(entry)
    return out


def readable_condition(tree, leaf: int, names: Sequence[str], cols: np.ndarray, msg: Mapping[str, Any], spec) -> list[dict[str, Any]]:
    """Translate the decision path to a leaf into readable «feature op threshold» clauses."""
    t = tree.tree_
    feature, threshold = t.feature, t.threshold
    left, right = t.children_left, t.children_right

    def walk(node: int, path: list[tuple[int, bool, float]]) -> list[tuple[int, bool, float]] | None:
        if node == int(leaf):
            return path
        if left[node] < 0:
            return None
        f, thr = int(feature[node]), float(threshold[node])
        got = walk(int(left[node]), path + [(f, True, thr)])
        if got is not None:
            return got
        return walk(int(right[node]), path + [(f, False, thr)])

    got = walk(0, [])
    if got is None:
        raise RuntimeError(f"leaf {leaf} not reachable from the root")
    return [
        {"feature": names[int(cols[f])], "op": "<=" if is_left else ">", "threshold": round(thr, 6)}
        for f, is_left, thr in got
    ]


# ---------------------------------------------------------------------------
# explanation cases
# ---------------------------------------------------------------------------


def explanation_cases(
    bundle,
    arm: str,
    train: Mapping[str, Any],
    valid: Mapping[str, Any],
    test: Mapping[str, Any],
    candidates: Mapping[str, Any],
    spec,
) -> dict[str, Any]:
    """Three real molecules: a success, an environment-changed effect and a counterexample."""
    out: dict[str, Any] = {"created_utc": C.utcnow(), "cases": []}
    if not candidates["candidates"]:
        out["note"] = "no candidate regularity with enough Train support; no case study is manufactured"
        return out
    first = candidates["candidates"][0]
    cond_text = "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in first["condition"])

    # (a) success: a Valid molecule with several messages in a candidate leaf (the highest
    #     ranked candidate that has any Valid/Test support) whose sign matches the Train median
    best = None
    cand = None
    for c in candidates["candidates"]:
        for split, msg in (("valid", valid), ("test", test)):
            leaf = _leaf_for(msg, spec, candidates, split)
            sel = leaf == c["leaf"]
            for mol in np.unique(msg["msg_mol"][sel]):
                m = int(mol)
                mv = sel & (msg["msg_mol"] == m)
                if mv.sum() < 2:
                    continue
                vals = msg["A"][mv]
                if np.sign(np.median(vals)) == np.sign(c["train_median"]):
                    score = float(np.abs(vals).sum())
                    if best is None or score > best[0]:
                        best = (score, split, m)
                        cand = c
            if best is not None:
                break
        if best is not None:
            break
    if best is not None and cand is not None:
        _score, split, mol = best
        msg = valid if split == "valid" else test
        first, cond_text = cand, "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in cand["condition"])
        msgs_out = _molecule_messages(msg, mol, first, spec)
        case = {
            "kind": "SUCCESS_SIGN_CONTINUES",
            "split": split,
            "molecule_row": int(msg["mol_row"][mol]),
            "candidate_rank": int(first["rank"]),
            "candidate_condition": cond_text,
            "messages": msgs_out,
            "molecule_prediction_raw": float(msg["y_raw"][mol]),
            "molecule_prediction_of_this_molecule": float(msg["pred_raw"][mol]),
            "full_vectors": full_vectors(
                bundle, arm, split, int(msg["mol_pos"][mol]), int(msg["mol_row"][mol]),
                [(m["src_node"], m["dst_node"]) for m in msgs_out[:3]],
            ),
        }
        out["cases"].append(case)
    else:
        out["cases"].append(
            {
                "kind": "SUCCESS_ABSENT",
                "reason": "no candidate leaf with Valid/Test support carries a molecule-level effect whose sign matches the Train median",
            }
        )

    # (b) environment changes the effect: the same source structural signature with
    #     opposite-signed effects in two environments
    env = _environment_flip(bundle, arm, train, valid, test, spec)
    if env is not None:
        out["cases"].append(env)

    # (c) counterexample: the largest-|A| Valid message that does NOT belong to the
    #     top candidate leaf, described with its own structural condition
    alt = None
    for split, msg in (("valid", valid), ("test", test)):
        leaf = _leaf_for(msg, spec, candidates, split)
        miss = leaf != first["leaf"]
        if not miss.any():
            continue
        idx = int(np.argmax(np.abs(msg["A"]) * miss))
        alt = (float(abs(msg["A"][idx])), split, idx)
        break
    if alt is not None:
        _a, split, idx = alt
        msg = valid if split == "valid" else test
        mol = int(msg["msg_mol"][idx])
        out["cases"].append(
            {
                "kind": "COUNTEREXAMPLE_OUTSIDE_RULE",
                "split": split,
                "molecule_row": int(msg["mol_row"][mol]),
                "candidate_rank": 1,
                "candidate_condition": cond_text,
                "message": _message_detail(msg, idx, spec),
                "note": "large single-message effect that the top Train rule does not cover",
            }
        )

    # (d) the single largest-|A| real message on Valid, with its exact vectors: where the
    #     computation actually concentrates, whether or not any rule covers it
    msg = valid
    idx = int(np.argmax(np.abs(msg["A"])))
    mol = int(msg["msg_mol"][idx])
    det = _message_detail(msg, idx, spec)
    out["cases"].append(
        {
            "kind": "LARGEST_EFFECT_MESSAGE",
            "split": "valid",
            "molecule_row": int(msg["mol_row"][mol]),
            "message": det,
            "molecule_prediction_raw": float(msg["y_raw"][mol]),
            "molecule_prediction_of_this_molecule": float(msg["pred_raw"][mol]),
            "full_vectors": full_vectors(
                bundle, arm, "valid", int(msg["mol_pos"][mol]), int(msg["mol_row"][mol]),
                [(det["src_node"], det["dst_node"])],
            ),
        }
    )
    return out


def _leaf_for(msg: Mapping[str, Any], spec, candidates: Mapping[str, Any], split: str) -> np.ndarray:
    if "leaf" not in msg:
        raise KeyError("leaf must be attached before case selection")
    return msg["leaf"]


def _environment_flip(bundle, arm, train, valid, test, spec, *, min_abs: float = 0.05):
    for split, msg in (("valid", valid), ("test", test)):
        if "sig" not in msg:
            continue
        sig = msg["sig"]
        A = msg["A"]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(sig):
            groups[key].append(i)
        best = None
        for key, idxs in groups.items():
            if len(idxs) < 2:
                continue
            vals = A[np.asarray(idxs)]
            if not (vals.max() > min_abs and vals.min() < -min_abs):
                continue
            span = float(vals.max() - vals.min())
            if best is None or span > best[0]:
                best = (span, key, np.asarray(idxs))
        if best is not None:
            span, key, idxs = best
            hi = int(idxs[int(np.argmax(A[idxs]))])
            lo = int(idxs[int(np.argmin(A[idxs]))])
            pos = _message_detail(msg, hi, spec)
            pos_mol = int(msg["msg_mol"][hi])
            return {
                "kind": "ENVIRONMENT_CHANGES_EFFECT",
                "split": split,
                "molecule_row": int(msg["mol_row"][pos_mol]),
                "signature": str(key),
                "effect_span": float(span),
                "positive_message": pos,
                "negative_message": _message_detail(msg, lo, spec),
                "full_vectors": full_vectors(
                    bundle, arm, split, int(msg["mol_pos"][pos_mol]), int(msg["mol_row"][pos_mol]),
                    [(pos["src_node"], pos["dst_node"])],
                ),
                "note": (
                    "the same exact source-side structural record and bond type carry opposite-signed attributed "
                    "effects under different receiver environments; this is a context dependence of the computed "
                    "contribution, not a chemical causal claim"
                ),
            }
    return None


def full_vectors(bundle, arm: str, split: str, mol_pos: int, mol_row: int, pairs: Sequence[tuple[int, int]],
                 *, device: str = "cpu") -> dict[str, Any]:
    """Full per-message vectors for a few case molecules (the only full vectors stored)."""
    import torch

    import cs_data as D
    import cs_mechanism as ME

    model, bundle_loaded, _payload, _path = ME.load_best(bundle.dataset, arm, device=torch.device(device))
    batch = D.make_batch(bundle_loaded, split, np.asarray([int(mol_pos)]), torch.device(device))
    states = model.forward_states(batch)
    # the single-molecule batch re-indexes atoms from 0; the stored records use split-global ids
    offset = int(np.asarray(bundle_loaded.caches[split]["atom_ptr"])[int(mol_pos)])
    src_l = states["edges"]["src"].cpu().numpy()
    dst_l = states["edges"]["dst"].cpu().numpy()
    p = states["p"].cpu().numpy()
    v = states["v"].cpu().numpy()
    v_extra = states["v_extra"].cpu().numpy()
    gamma = states["gamma"].cpu().numpy()
    delta_msg = states["delta_msg"].cpu().numpy() if states["delta_msg"].numel() else np.zeros_like(p)
    out: dict[str, Any] = {"molecule_row": int(mol_row), "messages": []}
    for s_i, d_i in pairs:
        hit = np.nonzero((src_l + offset == int(s_i)) & (dst_l + offset == int(d_i)))[0]
        if hit.size == 0:
            continue
        e = int(hit[0])
        out["messages"].append(
            {
                "src_local": int(s_i),
                "dst_local": int(d_i),
                "p_ji": [round(float(x), 6) for x in p[e].tolist()],
                "gamma_i": [round(float(x), 6) for x in gamma[int(d_i) - offset].tolist()],
                "delta_i_r": [round(float(x), 6) for x in delta_msg[e].tolist()],
                "v_ji": [round(float(x), 6) for x in v[e].tolist()],
                "v_extra_ji": [round(float(x), 6) for x in v_extra[e].tolist()],
            }
        )
    out["note"] = "vectors are the exact tensors of this checkpoint's computation at the message's receiving node"
    return out


def _message_detail(msg: Mapping[str, Any], idx: int, spec) -> dict[str, Any]:
    cache = msg["cache"]
    readable = msg["readable"]
    src, dst = int(msg["src_atom"][idx]), int(msg["dst_atom"][idx])
    src_fields = {k: float(v[src]) for k, v in readable.items() if k in ("degree", "in_ring", "is_terminal", "ring_bond_fraction", "min_ring_size")}
    dst_fields = {k: float(v[dst]) for k, v in readable.items() if k in ("degree", "in_ring", "is_terminal", "ring_bond_fraction", "min_ring_size")}
    n_r = np.zeros(C.N_RELATIONS, dtype=np.int64)
    same_node = msg["dst_atom"] == dst
    for r in msg["rel"][same_node]:
        n_r[int(r)] += 1
    return {
        "molecule_row": int(msg["mol_row"][int(msg["msg_mol"][idx])]),
        "atom_type_src": int(readable["atom_type"][src]),
        "atom_type_dst": int(readable["atom_type"][dst]),
        "bond_type": int(msg["bond_type"][idx]),
        "relation": C.RELATION_NAMES[int(msg["rel"][idx])],
        "src_node": src,
        "dst_node": dst,
        "A_raw": float(msg["A"][idx]),
        "A_extra_raw": float(msg["A_extra"][idx]),
        "v_rms": float(msg["v_rms"][idx]),
        "p_rms": float(msg["p_rms"][idx]),
        "gamma_rms": float(msg["gamma_rms"][dst]),
        "delta_abs": float(msg["delta_abs"][idx]),
        "src_readable": src_fields,
        "dst_readable": dst_fields,
        "dst_relation_counts": {C.RELATION_NAMES[r]: int(n_r[r]) for r in range(C.N_RELATIONS)},
    }


def _molecule_messages(msg: Mapping[str, Any], mol: int, candidate: Mapping[str, Any], spec) -> list[dict[str, Any]]:
    sel = np.nonzero(msg["msg_mol"] == mol)[0]
    sel = sel[np.argsort(-np.abs(msg["A"][sel]))][:8]
    return [_message_detail(msg, int(i), spec) for i in sel]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(dataset: str = C.MAIN_DATASET, *, arms: Sequence[str] = ("A", "D")) -> dict[str, Any]:
    bundle = D.load_bundle(dataset)
    out = {
        "dataset": dataset,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "created_utc_note": "fitted on Train only; Valid/Test used for evaluation only",
        "vocabulary": {},
        "arms": {},
    }
    splits = {}
    for split in ("train", "valid", "test"):
        cache = bundle.caches[split]
        splits[split] = cache
    vocab_s1 = top_codes(splits["train"], "nbr_code", "nbr_ptr", C.TREE_TOP_CODES)
    vocab_s2 = top_codes(splits["train"], "path_cat", "path_ptr", C.TREE_TOP_CODES)
    out["vocabulary"] = {
        "top_s1_codes": [int(x) for x in vocab_s1.tolist()],
        "top_s2_codes": [int(x) for x in vocab_s2.tolist()],
        "rule": "most frequent S1 (neighbour code) and S2 (path code) records over Train atoms, capped at 32",
    }
    for arm in arms:
        try:
            msgs = {split: load_messages(dataset, arm, split, bundle) for split in ("train", "valid", "test")}
        except FileNotFoundError as exc:
            out["arms"][arm] = {"error": f"missing decomposition artefact: {exc}"}
            continue
        for split, msg in msgs.items():
            cache = msg["cache"]
            n_atoms = int(np.asarray(cache["atom_ptr"])[-1])
            sig = [structure_signature(cache, a) for a in range(n_atoms)]
            msg["sig"] = [f"{sig[a]}|{int(b)}" for a, b in zip(msg["src_atom"], msg["bond_type"])]
            msg["scaffold"] = scaffolds(bundle, split)
        entry: dict[str, Any] = {
            "n_messages": {split: int(m["A"].size) for split, m in msgs.items()},
            "attribution_amplitude": {
                split: {
                    "A_abs_mean": float(np.abs(m["A"]).mean()),
                    "A_rms": float(np.sqrt(np.mean(m["A"] ** 2))),
                    "A_q05": float(np.percentile(m["A"], 5)),
                    "A_q95": float(np.percentile(m["A"], 95)),
                    "A_extra_abs_mean": float(np.abs(m["A_extra"]).mean()),
                }
                for split, m in msgs.items()
            },
            "repeated_structures": repeated_structures(msgs["train"], msgs, bundle.spec),
            "trees": fit_trees(msgs["train"], msgs["valid"], msgs["test"], bundle.spec, vocab_s1, vocab_s2),
        }
        tree_train = build_features(msgs["train"], bundle.spec, vocab_s1, vocab_s2)
        from sklearn.tree import DecisionTreeRegressor

        cols = tree_train["by_side"]["SC"]
        tree = DecisionTreeRegressor(
            max_depth=C.TREE_MAX_DEPTH, min_samples_leaf=C.TREE_MIN_LEAF, random_state=C.TREE_RANDOM_STATE
        )
        tree.fit(tree_train["matrix"][:, cols], msgs["train"]["A"], sample_weight=molecule_weights(msgs["train"]))
        for split, m in msgs.items():
            f = tree_train if split == "train" else build_features(m, bundle.spec, vocab_s1, vocab_s2)
            m["leaf"] = tree.apply(f["matrix"][:, cols])
        entry["candidates"] = candidate_regularities(
            msgs["train"], msgs["valid"], msgs["test"], bundle.spec, vocab_s1, vocab_s2
        )
        entry["cases"] = explanation_cases(
            bundle, arm, msgs["train"], msgs["valid"], msgs["test"], entry["candidates"], bundle.spec
        )
        out["arms"][arm] = entry
    return out


def scaffolds(bundle, split: str) -> list[str]:
    smiles = bundle.smiles[split]
    out: list[str] = []
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

        for smi in smiles:
            mol = Chem.MolFromSmiles(str(smi))
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else "")
    except Exception:  # noqa: BLE001
        out = ["" for _ in smiles]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=C.MAIN_DATASET)
    parser.add_argument("--arms", default="A,D")
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "structure_rules.json"))
    parser.add_argument("--cases-out", default=str(C.ANALYSIS_DIR / "explanation_cases.json"))
    args = parser.parse_args(argv)
    report = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(), "datasets": {}}
    for dataset in [d for d in args.datasets.split(",") if d]:
        C.log(f"structure rules: {dataset}")
        report["datasets"][dataset] = run(dataset, arms=tuple(a for a in args.arms.split(",") if a))
    cases = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "datasets": {
            ds: {arm: block.get("cases", {}) for arm, block in payload["arms"].items()}
            for ds, payload in report["datasets"].items()
        },
    }
    C.write_json(Path(args.out), report)
    C.write_json(Path(args.cases_out), cases)
    C.log(f"structure rules written to {args.out} and {args.cases_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
