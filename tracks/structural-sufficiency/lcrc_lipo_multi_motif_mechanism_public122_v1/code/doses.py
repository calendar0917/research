"""Structure-only statistical-dose observables (no weights, no labels, no predictions).

Physically removing two and adding two bonds does **not** mean the model sees the
same amount of statistical change.  These observables quantify, on the rebuilt
input tables alone, how far a rewired view moves the frozen model's *input
statistics* away from the original molecule:

``d1n``
    sum over roots of the L1 distance between the per-root multisets of
    ``nbr_code`` (the typed-neighbour S1 table).
``d1b``
    sum over roots of the L1 distance between the per-root multisets of
    ``(AF51(neighbour) bytes hash, BF13(edge) bytes hash)``.  The frozen bond
    encoder consumes ``a_u + a_v``, ``|a_u - a_v|``, ``a_u * a_v`` and ``BF13``,
    so "AF51 is atom-wise identical" does **not** imply ``b_e`` / ``b_sum`` is
    unchanged; this is the first-order interaction term and must be recorded
    separately from the typed-neighbour table.
``d2j`` / ``d2c`` / ``d2k``
    per-root L1 distances of the S2 tables: the joint ``(path_cat, path_end63)``
    counts, the ``(path_cat, AF51(endpoint) bytes hash)`` content counts and the
    ``path_closure`` marginal counts.
``d3``
    sum over atoms of the frozen-Train-standardised L1 norm of the scalar9
    change.
``dr``
    number of roots whose S1 signature, any S2 table or the scalar9 row changed.

Nothing here is a model quantity; ``d1b_model_l1`` (recorded by the internal-path
stage) is the only weight-dependent companion and is labelled as such.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def _atom_span(cache: Mapping[str, Any], i: int) -> tuple[int, int]:
    return int(cache["atom_ptr"][i]), int(cache["atom_ptr"][i + 1])


def _bond_span(cache: Mapping[str, Any], i: int) -> tuple[int, int]:
    return int(cache["bond_ptr"][i]), int(cache["bond_ptr"][i + 1])


def molecule_tables(cache: Mapping[str, Any], i: int) -> dict[str, Any]:
    """Per-atom S1 / S2 / scalar tables of one molecule of a flat cache."""
    a0, a1 = _atom_span(cache, i)
    b0, b1 = _bond_span(cache, i)
    n_atoms = a1 - a0
    nbr_ptr = np.asarray(cache["nbr_ptr"])[a0 : a1 + 1]
    path_ptr = np.asarray(cache["path_ptr"])[a0 : a1 + 1]
    n0, p0 = int(nbr_ptr[0]), int(path_ptr[0])
    nbr_code = np.asarray(cache["nbr_code"])[n0 : int(nbr_ptr[-1])]
    path_cat = np.asarray(cache["path_cat"])[p0 : int(path_ptr[-1])]
    path_end63 = np.asarray(cache["path_end63"])[p0 : int(path_ptr[-1])]
    path_end = np.asarray(cache["path_end"])[p0 : int(path_ptr[-1])]
    path_closure = np.asarray(cache["path_closure"])[p0 : int(path_ptr[-1])]
    nbr_rows = [nbr_code[int(nbr_ptr[a]) - n0 : int(nbr_ptr[a + 1]) - n0] for a in range(n_atoms)]
    path_rows = [
        (
            path_cat[int(path_ptr[a]) - p0 : int(path_ptr[a + 1]) - p0],
            path_end63[int(path_ptr[a]) - p0 : int(path_ptr[a + 1]) - p0],
            path_end[int(path_ptr[a]) - p0 : int(path_ptr[a + 1]) - p0],
            path_closure[int(path_ptr[a]) - p0 : int(path_ptr[a + 1]) - p0],
        )
        for a in range(n_atoms)
    ]
    bond_index = np.asarray(cache["bond_index"])[b0:b1].reshape(-1, 2)
    bond_feat = np.asarray(cache["bond_feat"])[b0:b1]
    atom_feat = np.asarray(cache["atom_feat"])[a0:a1]
    incident: list[list[tuple[int, bytes]]] = [[] for _ in range(n_atoms)]
    for e in range(bond_index.shape[0]):
        u, v = int(bond_index[e, 0]), int(bond_index[e, 1])
        bkey = np.ascontiguousarray(bond_feat[e], dtype=np.float32).tobytes()
        incident[u].append((v, bkey))
        incident[v].append((u, bkey))
    return {
        "n_atoms": int(n_atoms),
        "atom_feat": atom_feat,
        "atom_scalar": np.asarray(cache["atom_scalar"])[a0:a1],
        "nbr_rows": nbr_rows,
        "path_rows": path_rows,
        "incident": incident,
    }


def af51_bytes(tbl: Mapping[str, Any], atom: int) -> bytes:
    return np.ascontiguousarray(np.asarray(tbl["atom_feat"])[int(atom)], dtype=np.float32).tobytes()


def _multiset_l1(a: Counter, b: Counter) -> int:
    total = 0
    for key in set(a) | set(b):
        total += abs(a.get(key, 0) - b.get(key, 0))
    return int(total)


def doses(
    tbl_o: Mapping[str, Any],
    tbl_v: Mapping[str, Any],
    *,
    scalar_mean: np.ndarray,
    scalar_scale: np.ndarray,
) -> dict[str, Any]:
    """All frozen structural-dose observables of one view against its own O view."""
    if int(tbl_o["n_atoms"]) != int(tbl_v["n_atoms"]):
        raise ValueError("dose: atom count differs between O and the view")
    n = int(tbl_o["n_atoms"])
    d1n = 0
    d1b = 0
    d2j = 0
    d2c = 0
    d2k = 0
    dr = 0
    d3 = 0.0
    n_s1_changed = 0
    n_s2_changed = 0
    n_scalar_changed = 0
    af_max = 0.0
    for a in range(n):
        s1a = Counter(int(x) for x in tbl_o["nbr_rows"][a])
        s1b = Counter(int(x) for x in tbl_v["nbr_rows"][a])
        c1 = _multiset_l1(s1a, s1b)
        d1n += c1
        b_a = Counter((af51_bytes(tbl_o, other), bkey) for other, bkey in tbl_o["incident"][a])
        b_b = Counter((af51_bytes(tbl_v, other), bkey) for other, bkey in tbl_v["incident"][a])
        c2 = _multiset_l1(b_a, b_b)
        d1b += c2
        oj = Counter(
            (int(c), int(e)) for c, e in zip(tbl_o["path_rows"][a][0], tbl_o["path_rows"][a][1])
        )
        vj = Counter(
            (int(c), int(e)) for c, e in zip(tbl_v["path_rows"][a][0], tbl_v["path_rows"][a][1])
        )
        j1 = _multiset_l1(oj, vj)
        d2j += j1
        oc = Counter(
            (int(c), af51_bytes(tbl_o, int(e)))
            for c, e in zip(tbl_o["path_rows"][a][0], tbl_o["path_rows"][a][2])
        )
        vc = Counter(
            (int(c), af51_bytes(tbl_v, int(e)))
            for c, e in zip(tbl_v["path_rows"][a][0], tbl_v["path_rows"][a][2])
        )
        j2 = _multiset_l1(oc, vc)
        d2c += j2
        ok = Counter(int(x) for x in tbl_o["path_rows"][a][3])
        vk = Counter(int(x) for x in tbl_v["path_rows"][a][3])
        j3 = _multiset_l1(ok, vk)
        d2k += j3
        so = (np.asarray(tbl_o["atom_scalar"])[a] - scalar_mean) / scalar_scale
        sv = (np.asarray(tbl_v["atom_scalar"])[a] - scalar_mean) / scalar_scale
        sc_changed = bool(np.any(so != sv))
        d3 += float(np.abs(sv - so).sum())
        af_max = max(af_max, float(np.abs(np.asarray(tbl_v["atom_feat"])[a] - np.asarray(tbl_o["atom_feat"])[a]).max()))
        if c1:
            n_s1_changed += 1
        if j1 or j2 or j3:
            n_s2_changed += 1
        if sc_changed:
            n_scalar_changed += 1
        if c1 or j1 or j2 or j3 or sc_changed:
            dr += 1
    return {
        "d1n": int(d1n),
        "d1b": int(d1b),
        "d2j": int(d2j),
        "d2c": int(d2c),
        "d2k": int(d2k),
        "d3": float(d3),
        "dr": int(dr),
        "n_atoms": int(n),
        "n_roots_s1_changed": int(n_s1_changed),
        "n_roots_s2_changed": int(n_s2_changed),
        "n_roots_scalar_changed": int(n_scalar_changed),
        "b_sum_changed": bool(d1b > 0),
        "af51_max_abs_diff": float(af_max),
    }


def caliper_ok(t: Mapping[str, Any], c: Mapping[str, Any], ratio: tuple[float, float], eps: float) -> tuple[bool, dict[str, Any]]:
    """The frozen dose caliper: exact on d1n / d1b / d2k, bounded ratio otherwise."""
    detail: dict[str, Any] = {
        "d1n_equal": int(t["d1n"]) == int(c["d1n"]),
        "d1b_equal": int(t["d1b"]) == int(c["d1b"]),
        "d2k_equal": int(t["d2k"]) == int(c["d2k"]),
        "ratios": {},
    }
    ok = detail["d1n_equal"] and detail["d1b_equal"] and detail["d2k_equal"]
    rest = ("d2j", "d2c", "d3", "dr")
    values_all_zero = all(float(t[k]) <= eps and float(c[k]) <= eps for k in rest)
    detail["rest_all_zero"] = bool(values_all_zero)
    if not ok:
        return False, detail
    if values_all_zero:
        detail["rest_pass"] = True
        return True, detail
    lo, hi = ratio
    rest_ok = True
    for k in rest:
        tv, cv = float(t[k]), float(c[k])
        if tv <= eps or cv <= eps:
            rest_ok = False
            detail["ratios"][k] = None
            continue
        r = tv / cv
        detail["ratios"][k] = float(r)
        if not (lo <= r <= hi):
            rest_ok = False
    detail["rest_pass"] = bool(rest_ok)
    return bool(ok and rest_ok), detail


__all__ = ["molecule_tables", "af51_bytes", "doses", "caliper_ok"]
