"""Structural preparation for the support-conditioned shrinkage round.

This module does the *local CPU* work that the frozen protocol demands before any
GPU training is purchased:

1. per-root inventory of the frozen LCRC two-hop path occurrences,
   ``D_v(c, e)`` — with multiplicity — and the two registered support keys
   ``K_FINE(v)`` / ``K_JOINT(v)``;
2. the exact integer residual test that decides whether a root can be
   intervened on at all (``R_{v,b}(c,e) = N_{v,b}(c,e) - n_c n_e / n_b``), the
   Frobenius norm ``r_v`` used as reference dose weight, and the *verification*
   that the linear map of ``R`` equals ``J_full - J_marg`` (that check lives in
   ``sc_model``/tests because it needs the actual operator);
3. canonical-structure groups from the published SMILES (RDKit, local only) so
   that support counts count *distinct fit structures*, never roots, paths or
   duplicated records;
4. the fixed weighted mid-CDF ``F_T``, the frozen alpha rules, the
   out-of-fold (Train-structure-only) feasibility gates E1-E5 and the frozen key
   selection rule.

Nothing here reads a Valid/Test label, a prediction or an error.  Support tables
are built from the current fit structures only; a smaller fit never sees the
counts of the structures it excludes.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sc_common as C


# ---------------------------------------------------------------------------
# key serialisation (canonical bytes; digest used as an index only)
# ---------------------------------------------------------------------------


def _u32(n: int) -> bytes:
    return struct.pack("<I", int(n))


def canonical_f32_bytes(row: np.ndarray) -> bytes:
    """Canonical float32 bytes: explicit little endian, ``-0.0`` folded to ``+0.0``."""
    a = np.ascontiguousarray(np.asarray(row, dtype=C.CANON_FLOAT_DTYPE))
    a = a.copy()
    a[a == 0.0] = np.float32(0.0)
    if not np.isfinite(a).all():
        raise ValueError("non-finite value in a canonical float32 block")
    return a.tobytes()


def joint_key_bytes(cat: np.ndarray, end: np.ndarray, cnt: np.ndarray) -> bytes:
    """``sorted tuple(c, e, D_v(c,e))`` -> length-prefixed little-endian int64 triples."""
    triples = np.stack([np.asarray(cat, np.int64), np.asarray(end, np.int64),
                        np.asarray(cnt, np.int64)], axis=1)
    if triples.size:
        order = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))
        triples = triples[order]
    return _u32(len(triples)) + np.ascontiguousarray(triples, dtype="<i8").tobytes()


def root_key_bytes(kind: str, atom_row: np.ndarray | None,
                   cat: np.ndarray, end: np.ndarray, cnt: np.ndarray) -> bytes:
    body = joint_key_bytes(cat, end, cnt)
    if kind == C.KEY_FINE:
        if atom_row is None:
            raise ValueError("K_FINE needs the 51 atom columns")
        return canonical_f32_bytes(atom_row) + body
    if kind == C.KEY_JOINT:
        return body
    raise KeyError(kind)


# ---------------------------------------------------------------------------
# per-root path inventory of one frozen structure cache
# ---------------------------------------------------------------------------


def path_root_index(cache: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(per_root_path_counts, path_root)`` from the per-atom ``path_ptr``."""
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    counts = np.diff(path_ptr)
    path_root = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    return counts, path_root


def root_activity(cat: np.ndarray, end: np.ndarray) -> tuple[bool, float]:
    """Integer activity test and ``r_v = ||R||_F`` for one root.

    For every ``bt2`` group ``b`` (``b = c % N_BOND_TYPES``) with
    ``N_{v,b}(c,e)`` the integer occurrence counts, ``n_c``/``n_e`` the row/column
    marginals and ``n_b`` the group total,

        R_{v,b}(c,e) = N_{v,b}(c,e) - n_c n_e / n_b

    is evaluated on the **small Cartesian product of the c values and the e
    values that actually occur in that group** (never on the full code space).
    Pairs with ``N = 0`` inside that product are part of R: their term
    ``-n_c n_e / n_b`` is generally non-zero, so a root can be active even when
    every occupied pair is "saturated".

    The root is inactive iff every integer ``n_b N - n_c n_e`` is zero, which is
    exactly ``R == 0``.  Only then do ``J_full`` and ``J_marg`` coincide.
    """
    if cat.size == 0:
        return False, 0.0
    c_all = np.asarray(cat, np.int64)
    e_all = np.asarray(end, np.int64)
    bond = c_all % C.N_BOND_TYPES
    r2 = 0.0
    active = False
    for g in np.unique(bond):
        mask = bond == g
        cg, eg = c_all[mask], e_all[mask]
        nb = int(cg.size)
        uc, ci = np.unique(cg, return_inverse=True)
        ue, ei = np.unique(eg, return_inverse=True)
        N = np.zeros((uc.size, ue.size), dtype=np.float64)
        np.add.at(N, (ci, ei), 1.0)
        nc = N.sum(axis=1)
        ne = N.sum(axis=0)
        num = nb * N - np.outer(nc, ne)         # integer-valued, in float64
        if np.any(num != 0.0):
            active = True
            R = num / float(nb)
            r2 += float(np.sum(R * R))
    return active, float(np.sqrt(r2))


def build_root_inventory(dataset: str, split: str, *, verbose: bool = True) -> dict[str, Any]:
    """Per-root keys, activity flag and dose weight for one split of one dataset."""
    cache = C.LD.load_cache(dataset, split)
    atom_feat = np.asarray(cache["atom_feat"], dtype=np.float32)
    cat = np.asarray(cache["path_cat"], dtype=np.int64)
    end = np.asarray(cache["path_end63"], dtype=np.int64)
    counts, path_root = path_root_index(cache)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    n_roots = int(atom_feat.shape[0])
    if n_roots != int(counts.size):
        raise ValueError(f"{dataset}/{split}: path_ptr length != atom count")
    if int(counts.sum()) != int(cat.size):
        raise ValueError(f"{dataset}/{split}: path block length != path_cat length")

    fine = np.zeros((n_roots, C.DIGEST_BYTES), dtype=np.uint8)
    joint = np.zeros((n_roots, C.DIGEST_BYTES), dtype=np.uint8)
    active = np.zeros(n_roots, dtype=bool)
    rnorm = np.zeros(n_roots, dtype=np.float64)
    #: per key kind: digest -> full key content (the digest is an index only)
    content: dict[str, dict[bytes, bytes]] = {C.KEY_FINE: {}, C.KEY_JOINT: {}}
    conflicts = {C.KEY_FINE: 0, C.KEY_JOINT: 0}

    for v in range(n_roots):
        p0, p1 = int(path_ptr[v]), int(path_ptr[v + 1])
        if p1 > p0:
            cv, ev = cat[p0:p1], end[p0:p1]
            pair = cv * C.N_NBR_CODES + ev
            uniq, cnt = np.unique(pair, return_counts=True)
            uc, ue = uniq // C.N_NBR_CODES, uniq % C.N_NBR_CODES
        else:
            cv = ev = cnt = uc = ue = np.zeros(0, np.int64)
        act, r = root_activity(cv, ev)
        active[v] = act
        rnorm[v] = r
        payload_joint = joint_key_bytes(uc, ue, cnt)
        payload_fine = root_key_bytes(C.KEY_FINE, atom_feat[v], uc, ue, cnt)
        joint[v] = C.digest_bytes(payload_joint)
        fine[v] = C.digest_bytes(payload_fine)
        for kind, payload, dig in ((C.KEY_JOINT, payload_joint, joint[v]),
                                   (C.KEY_FINE, payload_fine, fine[v])):
            key_bytes = dig.tobytes()
            prior = content[kind].setdefault(key_bytes, payload)
            if prior != payload:
                conflicts[kind] += 1
        if verbose and n_roots > 20000 and v % 20000 == 0:
            C.log(f"  [{dataset}/{split}] roots {v}/{n_roots}")
    return {
        "dataset": dataset, "split": split, "n_roots": n_roots,
        "content_map": content, "digest_conflicts": conflicts,
        "fine": fine, "joint": joint, "active": active, "r": rnorm,
        "n_paths": counts.astype(np.int64), "path_root": path_root,
        "n_paths_total": int(cat.size),
    }


def load_or_build_rootkeys(dataset: str) -> dict[str, dict[str, Any]]:
    """Per-split root inventory with a *global* key-id vocabulary per key kind."""
    path = C.rootkeys_path(dataset)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        out = {}
        for s in C.SPLITS:
            out[s] = {k: np.asarray(z[f"{s}__{k}"]) for k in
                      ("fine", "joint", "active", "r", "n_paths", "fine_id", "joint_id")}
            out[s]["fine"] = np.asarray(out[s]["fine"], dtype=np.uint8)
            out[s]["joint"] = np.asarray(out[s]["joint"], dtype=np.uint8)
            out[s]["active"] = np.asarray(out[s]["active"], dtype=bool)
            out[s]["r"] = np.asarray(out[s]["r"], dtype=np.float64)
            out[s]["fine_id"] = np.asarray(out[s]["fine_id"], dtype=np.int32)
            out[s]["joint_id"] = np.asarray(out[s]["joint_id"], dtype=np.int32)
            out[s]["n_paths"] = np.asarray(out[s]["n_paths"], dtype=np.int64)
        out["meta"] = {
            "dataset": dataset,
            "n_key_ids_fine": int(z["n_key_ids_fine"]),
            "n_key_ids_joint": int(z["n_key_ids_joint"]),
            "sha256": C.sha256_file(path),
        }
        return out

    per_split = {s: build_root_inventory(dataset, s) for s in C.SPLITS}
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"dataset": dataset}
    for kind, name in ((C.KEY_FINE, "fine"), (C.KEY_JOINT, "joint")):
        all_dig = np.concatenate([per_split[s][name] for s in C.SPLITS], axis=0)
        uniq, inv = np.unique(all_dig, axis=0, return_inverse=True)
        # the digest is only an index: confirm it is 1:1 with the serialised content
        contents: dict[bytes, bytes] = {}
        conflicts = 0
        for s in C.SPLITS:
            conflicts += int(per_split[s]["digest_conflicts"][kind])
            for dig, body in per_split[s]["content_map"][kind].items():
                prior = contents.setdefault(dig, body)
                if prior != body:
                    conflicts += 1
        meta[f"collision_check_{kind}"] = {
            "n_distinct_digests": int(uniq.shape[0]),
            "n_distinct_contents": int(len(contents)),
            "digest_conflicts": int(conflicts),
            "collision_free": bool(conflicts == 0 and len(contents) == int(uniq.shape[0])),
            "verified_by": "full serialised key content compared for every digest at build time",
        }
        offset = 0
        for s in C.SPLITS:
            n = per_split[s][name].shape[0]
            per_split[s][f"{name}_id"] = inv[offset: offset + n].astype(np.int32)
            offset += n
        payload[f"n_key_ids_{name}"] = np.int64(uniq.shape[0])
        meta[f"n_key_ids_{name}"] = int(uniq.shape[0])
    for s in C.SPLITS:
        for k in ("fine", "joint", "active", "r", "n_paths", "fine_id", "joint_id"):
            payload[f"{s}__{k}"] = per_split[s][k]
        per_split[s]["content_map"] = {}
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    per_split["meta"] = {**meta, "sha256": C.sha256_file(path)}
    return per_split


# ---------------------------------------------------------------------------
# canonical structure groups (local RDKit only)
# ---------------------------------------------------------------------------


_ATOM_MAP = re.compile(r":\d+")


def canonical_structure(smiles: str, *, rdkit=None) -> str:
    """Published SMILES -> atom-map-free canonical isomeric SMILES."""
    if rdkit is None:  # pragma: no cover - imported lazily on the prep host
        from rdkit import Chem
        rdkit = Chem
    mol = rdkit.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"published SMILES does not parse: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return rdkit.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_groups(dataset: str) -> dict[str, Any]:
    """Canonical-structure groups per split + the frozen 5-fold assignment."""
    import rdkit

    raw = np.load(C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/"
                  f"analysis/source/raw/lcrc_raw_v2/{dataset}.npz", allow_pickle=False)
    smiles = [str(x) for x in raw["smiles"]]
    published_canonical = [str(x) for x in raw["canonical_smiles"]]
    src = C.PD.load_source(dataset, labels="none")
    rows = {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS}
    canon_all = [canonical_structure(s) for s in smiles]
    vocab = {c: i for i, c in enumerate(sorted(set(canon_all)))}
    fold_of = {c: int(C.sha256_text(f"{C.FOLD_SALT}|{c}"), 16) % C.N_FOLDS for c in vocab}
    out: dict[str, Any] = {
        "dataset": dataset,
        "rdkit": str(rdkit.__version__),
        "definition": C.GROUP_RDKIT_SPEC,
        "n_structures_total": len(vocab),
        "atommapped_input": bool(any(_ATOM_MAP.search(s) for s in smiles)),
        "artifact_canonical_matches": float(np.mean([a == b for a, b in
                                                     zip(canon_all, published_canonical)])),
        "fold_rule": "fold = int(sha256('lcrc-support-fold-v1|' + canonical_smiles), 16) % 5",
    }
    for split in C.SPLITS:
        gid = np.asarray([vocab[canon_all[r]] for r in rows[split]], dtype=np.int32)
        out[split] = {
            "rows": rows[split],
            "group_id": gid,
            "fold": np.asarray([fold_of[canon_all[r]] for r in rows[split]], dtype=np.int8),
            "n_records": len(rows[split]),
            "n_groups": int(np.unique(gid).size),
        }
    return out


def save_groups(dataset: str, groups: Mapping[str, Any]) -> Path:
    path = C.groups_path(dataset)
    payload = {
        "meta_json": np.asarray(C.stable_json({k: v for k, v in groups.items()
                                               if k not in C.SPLITS})),
    }
    for s in C.SPLITS:
        for k, v in groups[s].items():
            payload[f"{s}__{k}"] = np.asarray(v)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_groups(dataset: str) -> dict[str, Any]:
    path = C.groups_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing canonical groups {path}")
    z = np.load(path, allow_pickle=False)
    import json as _json
    out: dict[str, Any] = _json.loads(str(z["meta_json"]))
    for s in C.SPLITS:
        out[s] = {k: np.asarray(z[f"{s}__{k}"]) for k in
                  ("rows", "group_id", "fold", "n_records", "n_groups")}
        out[s]["rows"] = [int(x) for x in out[s]["rows"]]
    out["sha256"] = C.sha256_file(path)
    return out


def check_group_identity(dataset: str, rootkeys: Mapping[str, Any],
                         groups: Mapping[str, Any]) -> dict[str, Any]:
    """Records that share a canonical structure must share the root-key multiset.

    This is checked with the full ``K_FINE`` serialisation (the finest key), so it
    is the strictest form.  If two records of one canonical structure disagree,
    the round stops and reports the identity problem instead of choosing a
    representative record to hide it.
    """
    problems: list[dict[str, Any]] = []
    checked = 0
    for split in C.SPLITS:
        cache = C.LD.load_cache(dataset, split)
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        atom_feat = np.asarray(cache["atom_feat"], dtype=np.float32)
        cat = np.asarray(cache["path_cat"], dtype=np.int64)
        end = np.asarray(cache["path_end63"], dtype=np.int64)
        path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
        gid = groups[split]["group_id"]
        by_group: dict[int, list[int]] = {}
        for rec, g in enumerate(gid):
            by_group.setdefault(int(g), []).append(rec)
        for g, recs in by_group.items():
            if len(recs) < 2:
                continue
            checked += 1
            base = None
            for rec in recs:
                a0, a1 = int(atom_ptr[rec]), int(atom_ptr[rec + 1])
                rows = []
                for v in range(a0, a1):
                    p0, p1 = int(path_ptr[v]), int(path_ptr[v + 1])
                    if p1 > p0:
                        pair = cat[p0:p1] * C.N_NBR_CODES + end[p0:p1]
                        uniq, cnt = np.unique(pair, return_counts=True)
                        uc, ue = uniq // C.N_NBR_CODES, uniq % C.N_NBR_CODES
                    else:
                        uc = ue = cnt = np.zeros(0, np.int64)
                    rows.append(root_key_bytes(C.KEY_FINE, atom_feat[v], uc, ue, cnt))
                multiset = tuple(sorted(map(C.sha256_bytes, rows)))
                if base is None:
                    base = multiset
                elif multiset != base:
                    problems.append({"split": split, "group_id": int(g), "records": recs})
    return {"status": "OK" if not problems else "IDENTITY_PROBLEM",
            "duplicate_groups_checked": int(checked), "problems": problems[:20],
            "n_problems": len(problems)}


# ---------------------------------------------------------------------------
# fit support table, reference weights, CDF
# ---------------------------------------------------------------------------


def fit_group_ids(groups: Mapping[str, Any], positions: Sequence[int]) -> np.ndarray:
    gid = np.asarray(groups["train"]["group_id"], dtype=np.int32)
    return np.asarray([int(gid[p]) for p in positions], dtype=np.int32)


def build_support_counts(dataset: str, rootkeys: Mapping[str, Any],
                         groups: Mapping[str, Any], positions: Sequence[int],
                         key: str, group_subset: Iterable[int] | None = None) -> dict[str, Any]:
    """Count distinct fit **structures** containing each key; unknown key -> 0.

    A canonical structure contributes **at most once** per key no matter how many
    roots, paths or duplicate records carry it.  ``group_subset`` restricts the
    table to a subset of canonical group ids (the out-of-fold tables); ``None``
    uses every group of the current fit.
    """
    name = "fine_id" if key == C.KEY_FINE else "joint_id"
    n_ids = int(rootkeys["meta"][f"n_key_ids_{'fine' if key == C.KEY_FINE else 'joint'}"])
    cache = C.LD.load_cache(dataset, "train")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    ids = rootkeys["train"][name]
    gid_of_rec = groups["train"]["group_id"]
    allowed = None if group_subset is None else set(int(g) for g in group_subset)

    pairs: set[tuple[int, int]] = set()
    n_groups = 0
    for pos in positions:
        g = int(gid_of_rec[int(pos)])
        if allowed is not None and g not in allowed:
            continue
        n_groups += 1
        a0, a1 = int(atom_ptr[int(pos)]), int(atom_ptr[int(pos) + 1])
        for k in np.unique(ids[a0:a1]):
            pairs.add((g, int(k)))
    counts = np.zeros(n_ids, dtype=np.int32)
    for _, k in pairs:
        counts[k] += 1
    return {"counts": counts, "n_groups": int(n_groups), "key": key,
            "n_occupied_keys": int((counts > 0).sum()),
            "n_group_key_pairs": int(len(pairs))}


def fit_root_slices(dataset: str, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Atom ranges (``atom_ptr``) of the fit records and the record of each root."""
    cache = C.LD.load_cache(dataset, "train")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    starts = atom_ptr[np.asarray(list(positions), dtype=np.int64)]
    stops = atom_ptr[np.asarray(list(positions), dtype=np.int64) + 1]
    n_roots = int((stops - starts).sum())
    rec_of_root = np.repeat(np.arange(len(positions), dtype=np.int64), (stops - starts))
    return np.stack([starts, stops], axis=1), rec_of_root


def reference_weights(r: np.ndarray, active: np.ndarray, rec_of_root: np.ndarray,
                      n_records: int) -> dict[str, Any]:
    """``w_v = r_v / sum_{u in G} r_u`` for active roots; inactive roots weigh 0."""
    r = np.asarray(r, dtype=np.float64)
    act = np.asarray(active, dtype=bool)
    w = np.zeros(r.shape, dtype=np.float64)
    per_rec = np.zeros(n_records, dtype=np.float64)
    np.add.at(per_rec, rec_of_root[act], r[act])
    ok = act & (per_rec[rec_of_root] > 0)
    w[ok] = r[ok] / per_rec[rec_of_root[ok]]
    return {"w": w, "per_record_total": per_rec,
            "W": float(w.sum()), "n_records_with_active_roots": int((per_rec > 0).sum()),
            "n_records_no_active": int((per_rec <= 0).sum())}


def support_alpha_tables(dataset: str, fit: str, key: str,
                         rootkeys: Mapping[str, Any], groups: Mapping[str, Any],
                         *, a_oof: float | None = None, oof: bool = False) -> dict[str, Any]:
    """c_T per root on every split, ``F_T`` and the frozen alpha tables.

    ``oof=True`` restricts the support table to the fit groups **outside** a fold
    and reports the held-out fold's weights (used for the feasibility gates and
    for the frozen ``a_oof`` constant only).
    """
    positions = C.fit_positions(dataset, fit)
    slices, rec_of_root = fit_root_slices(dataset, positions)
    starts, stops = slices[:, 0], slices[:, 1]
    n_fit_roots = int(rec_of_root.size)
    fit_r = np.zeros(n_fit_roots, dtype=np.float64)
    fit_active = np.zeros(n_fit_roots, dtype=bool)
    src_r = rootkeys["train"]["r"]
    src_act = rootkeys["train"]["active"]
    ptr = np.concatenate([[0], np.cumsum(stops - starts)]).astype(np.int64)
    for i in range(len(positions)):
        fit_r[ptr[i]: ptr[i + 1]] = src_r[starts[i]: stops[i]]
        fit_active[ptr[i]: ptr[i + 1]] = src_act[starts[i]: stops[i]]
    rw = reference_weights(fit_r, fit_active, rec_of_root, len(positions))

    name = "fine_id" if key == C.KEY_FINE else "joint_id"
    fit_key_ids = np.zeros(n_fit_roots, dtype=np.int64)
    for i in range(len(positions)):
        fit_key_ids[ptr[i]: ptr[i + 1]] = rootkeys["train"][name][starts[i]: stops[i]]

    if not oof:
        table = build_support_counts(dataset, rootkeys, groups, positions, key)
        counts = table["counts"]
        c_fit = counts[fit_key_ids]
        n_support = int(table["n_groups"])
        F = C.build_mid_cdf(c_fit[rw["w"] > 0], rw["w"][rw["w"] > 0], n_support)
        per_split_c = {s: table["counts"][rootkeys[s][name]] for s in C.SPLITS}
        return {"c_fit": c_fit, "F": F, "counts": counts, "n_groups": n_support,
                "weights": rw, "per_split_c": per_split_c, "oof": False,
                "fit_key_ids": fit_key_ids}

    # ---- out-of-fold queries: table from 4/5 of the fit structures ----------
    gid = np.asarray(groups["train"]["group_id"], dtype=np.int32)
    fit_groups = np.asarray([int(gid[p]) for p in positions], dtype=np.int32)
    fold_of_group = {}
    for p, g in zip(positions, fit_groups):
        fold_of_group[int(g)] = int(groups["train"]["fold"][int(p)])
    a_vals = np.zeros(n_fit_roots, dtype=np.float64)
    c_vals = np.zeros(n_fit_roots, dtype=np.int64)
    for fold in range(C.N_FOLDS):
        keep = [g for g, f in fold_of_group.items() if f != fold]
        table = build_support_counts(dataset, rootkeys, groups, positions, key,
                                     group_subset=keep)
        counts = table["counts"]
        c_fit = counts[fit_key_ids]
        F = C.build_mid_cdf(c_fit[rw["w"] > 0], rw["w"][rw["w"] > 0], int(table["n_groups"]))
        query = np.zeros(n_fit_roots, dtype=bool)
        for i, p in enumerate(positions):
            if fold_of_group[int(fit_groups[i])] == fold:
                query[ptr[i]: ptr[i + 1]] = True
        a_fold = C.alpha_rule(C.ARM_SELECT, c_fit, F, None)
        a_vals[query & fit_active] = a_fold[query & fit_active]
        c_vals[query & fit_active] = c_fit[query & fit_active]
    q = fit_active & (rw["w"] > 0)
    w_q = rw["w"][q]
    a_oof_value = float(np.sum(w_q * a_vals[q]) / np.sum(w_q)) if np.sum(w_q) > 0 else float("nan")
    return {"oof": True, "a": a_vals, "c": c_vals, "w": rw["w"], "active": fit_active,
            "a_oof": a_oof_value, "fit_key_ids": fit_key_ids}


def gate_report(dataset: str, fit: str, key: str, full: Mapping[str, Any],
                oof: Mapping[str, Any], groups: Mapping[str, Any]) -> dict[str, Any]:
    """The five frozen feasibility gates, evaluated on Train structures only."""
    positions = C.fit_positions(dataset, fit)
    gid = np.asarray(groups["train"]["group_id"], dtype=np.int32)
    fit_groups = np.asarray([int(gid[p]) for p in positions], dtype=np.int32)
    _, rec_of_root = fit_root_slices(dataset, positions)
    act = full["weights"]["w"] > 0

    active_groups = int(np.unique(fit_groups[rec_of_root[act]]).size)
    active_keys = int(np.unique(full["fit_key_ids"][act]).size)
    a_full = C.alpha_rule(C.ARM_SELECT, full["c_fit"], full["F"], None)
    spread_full = C.weighted_spread(a_full[act], full["weights"]["w"][act])

    q = act & np.isfinite(oof["a"])
    wq = oof["w"][q]
    spread_oof = C.weighted_spread(oof["a"][q], wq)
    unknown = float(wq[oof["c"][q] == 0].sum() / wq.sum()) if wq.size else float("nan")
    low = float(wq[oof["c"][q] <= C.LOW_C_MAX].sum() / wq.sum()) if wq.size else float("nan")
    high = float(wq[oof["c"][q] >= C.HIGH_C_MIN].sum() / wq.sum()) if wq.size else float("nan")

    checks = {
        "E1_active_groups": {"value": active_groups, "threshold": C.E1_MIN_ACTIVE_GROUPS,
                             "pass": active_groups >= C.E1_MIN_ACTIVE_GROUPS},
        "E1_active_keys": {"value": active_keys, "threshold": C.E1_MIN_ACTIVE_KEYS,
                           "pass": active_keys >= C.E1_MIN_ACTIVE_KEYS},
        "E2_spread_fullfit": {"value": spread_full["spread"],
                              "threshold": C.E2_MIN_SPREAD_FULLFIT,
                              "pass": spread_full["spread"] >= C.E2_MIN_SPREAD_FULLFIT},
        "E3_spread_oof": {"value": spread_oof["spread"], "threshold": C.E3_MIN_SPREAD_OOF,
                          "pass": spread_oof["spread"] >= C.E3_MIN_SPREAD_OOF},
        "E4_unknown_share": {"value": unknown, "threshold": C.E4_MAX_UNKNOWN_SHARE,
                             "pass": unknown <= C.E4_MAX_UNKNOWN_SHARE},
        "E5_low_share": {"value": low, "threshold": C.E5_MIN_LOW_SHARE,
                         "pass": low >= C.E5_MIN_LOW_SHARE},
        "E5_high_share": {"value": high, "threshold": C.E5_MIN_HIGH_SHARE,
                          "pass": high >= C.E5_MIN_HIGH_SHARE},
    }
    return {"dataset": dataset, "fit": fit, "key": key, "checks": checks,
            "pass": all(v["pass"] for v in checks.values()),
            "spread_full_detail": spread_full, "spread_oof_detail": spread_oof,
            "active_group_count": active_groups, "active_key_count": active_keys,
            "n_groups_fit": int(len(set(fit_groups.tolist()))),
            "a_oof": float(oof["a_oof"])}


def freeze_support(dataset: str, fit: str, key: str, rootkeys: Mapping[str, Any],
                   groups: Mapping[str, Any], *, a_oof: float) -> dict[str, Any]:
    """Persist the frozen per-root support counts, CDF and alpha tables."""
    full = support_alpha_tables(dataset, fit, key, rootkeys, groups)
    weights = full["weights"]["w"]
    act = weights > 0
    arms: dict[str, Any] = {}
    for arm in C.ARMS:
        if arm == C.ARM_BASE:
            continue
        a = C.alpha_rule(arm, full["c_fit"], full["F"], a_oof)
        a = np.where(act, a, 1.0)          # inactive roots always a = 1
        arms[arm] = a
    payload: dict[str, np.ndarray] = {
        "F_table": full["F"], "counts": full["counts"],
        "n_support": np.int64(full["n_groups"]), "a_oof": np.float64(a_oof),
        "key": np.asarray(key), "fit": np.asarray(fit), "dataset": np.asarray(dataset),
        "fit_mean_a_select": np.float64(np.sum(weights[act] * arms[C.ARM_SELECT][act])
                                        / max(np.sum(weights[act]), 1e-30)),
    }
    for s in C.SPLITS:
        payload[f"c_{s}"] = full["per_split_c"][s].astype(np.int32)
        payload[f"active_{s}"] = rootkeys[s]["active"]
        payload[f"r_{s}"] = rootkeys[s]["r"].astype(np.float64)
        for arm, a_fit in arms.items():
            a_split = C.alpha_rule(arm, full["per_split_c"][s], full["F"], a_oof)
            a_split = np.where(rootkeys[s]["active"], a_split, 1.0)
            payload[f"alpha_{arm}_{s}"] = a_split.astype(np.float32)
    path = C.support_path(dataset, fit, key)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"path": C.track_rel(path), "sha256": C.sha256_file(path),
            "fit_mean_a_select": float(payload["fit_mean_a_select"]),
            "n_support": int(full["n_groups"]), "a_oof": float(a_oof)}


def load_support(dataset: str, fit: str, key: str) -> dict[str, Any]:
    path = C.support_path(dataset, fit, key)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen support table {path}")
    z = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {k: np.asarray(z[k]) for k in z.files}
    out["key"] = str(z["key"])
    out["fit"] = str(z["fit"])
    out["dataset"] = str(z["dataset"])
    out["sha256"] = C.sha256_file(path)
    return out


def alpha_for_split(support: Mapping[str, Any], arm: str, split: str) -> np.ndarray:
    """Per-root alpha of one arm on one split (``None`` for BASE = no gate)."""
    if arm == C.ARM_BASE:
        return None
    return np.asarray(support[f"alpha_{arm}_{split}"], dtype=np.float32)


__all__ = [n for n in dir() if not n.startswith("_")]
