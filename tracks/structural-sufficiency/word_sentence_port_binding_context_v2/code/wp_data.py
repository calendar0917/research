"""Derived port/site features, verified loading and mini-batch packing.

The word partition itself is **inherited** from ``word_sentence_port_route_v1``
(same templates, same merge history, same content hash).  This module adds the
round-2 derived features and nothing else:

* ``site_*``      one entry per (word instance, template position): the raw 9
                  column atom row, its internal degree, its boundary incidence
                  count and the *compatible-site group* used by the MEAN arm
                  (``X_r == X_p`` on the original nine atom columns);
* ``edge_*``      the word's own internal bonds as directed message edges;
* ``pp_*``        every ordered site pair ``(p, a)`` inside a word together with
                  the template shortest-path distance ``d(p, a) in 0..5``.

No target value, split statistic or label ever enters the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import wp_common as C
import ws_words as W

ATOM_COLUMNS = C.ATOM_COLUMNS
BOND_COLUMNS = C.BOND_COLUMNS


# ---------------------------------------------------------------------------
# loading / verification
# ---------------------------------------------------------------------------


def load_old_words(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the inherited word artefacts and verify their frozen identity."""
    path = C.old_word_path(dataset)
    freeze_path = C.old_freeze_path(dataset)
    record_path = C.old_record_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"inherited word cache missing: {path}. Rebuild it with the frozen v1 rules "
            f"(word_sentence_port_route_v1 --prepare) before running this round.")
    if not record_path.is_file():
        raise FileNotFoundError(f"inherited structure record missing: {record_path}")
    record = C.read_json(record_path)
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    digest = C.content_sha256(payload)
    expected = record.get("cache_sha256")
    if digest != expected:
        raise ValueError(f"inherited word content hash changed: {digest} != {expected}")
    if freeze_path.is_file():
        host = C.read_json(freeze_path)
        if host.get("cache_sha256") != expected:
            raise ValueError("host word freeze disagrees with the committed record")
    return payload, record


def load_derived(dataset: str) -> dict[str, Any]:
    path = C.derived_path(dataset)
    record_path = C.derived_record_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"derived cache missing: {path} (run `prepare`)")
    if not record_path.is_file():
        raise FileNotFoundError(f"derived record missing: {record_path}")
    record = C.read_json(record_path)
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    digest = C.content_sha256(payload)
    if digest != record.get("derived_sha256"):
        raise ValueError(f"derived content hash mismatch: {digest} != {record.get('derived_sha256')}")
    return payload


def load_dataset(dataset: str) -> dict[str, Any]:
    words, record = load_old_words(dataset)
    derived = load_derived(dataset)
    raw = C.dataset_cache(dataset)
    return {"words": words, "derived": derived, "raw": raw, "record": record,
            "derived_record": C.read_json(C.derived_record_path(dataset))}


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def _split_arrays(words: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    prefix = f"word/{split}/"
    return {k[len(prefix):]: v for k, v in words.items() if k.startswith(prefix)}


def _base_arrays(words: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    prefix = f"base/{split}/"
    return {k[len(prefix):]: v for k, v in words.items() if k.startswith(prefix)}


def template_dist_table(words: Mapping[str, Any]) -> np.ndarray:
    """``(n_templates, 6, 6)`` template shortest-path distances (padded with 0)."""
    n = int(words["tmpl/n_templates"])
    n_atoms = np.asarray(words["tmpl/tmpl_n_atoms"])
    bond = np.asarray(words["tmpl/tmpl_bond"])
    n_bond_codes = int(bond[bond >= 0].max()) + 1 if bond.size else 0
    out = np.zeros((n, 6, 6), dtype=np.int8)
    for t in range(n):
        m = int(n_atoms[t])
        adj = np.zeros((m, m), dtype=bool)
        for i in range(m):
            for j in range(i + 1, m):
                if 0 <= bond[t, i, j] < n_bond_codes:
                    adj[i, j] = adj[j, i] = True
        # BFS
        dist = np.zeros((m, m), dtype=np.int8)
        for s in range(m):
            seen = np.full(m, -1, dtype=np.int64)
            seen[s] = 0
            queue = [s]
            while queue:
                node = queue.pop(0)
                for nxt in np.nonzero(adj[node])[0].tolist():
                    if seen[nxt] < 0:
                        seen[nxt] = seen[node] + 1
                        queue.append(nxt)
            dist[s] = seen.astype(np.int8)
        out[t, :m, :m] = dist
    return out


def derive_split(words: Mapping[str, Any], split: str, dist_table: np.ndarray) -> dict[str, np.ndarray]:
    """All round-2 derived per-split arrays (label free)."""
    return derive_core(_split_arrays(words, split), _base_arrays(words, split), dist_table)


def derive_core(arr: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray],
                dist_table: np.ndarray) -> dict[str, np.ndarray]:
    """Derived arrays from an arbitrary (word arrays, base arrays) pair."""
    word_ptr_atom = np.asarray(arr["word_ptr_atom"], dtype=np.int64)
    word_atoms = np.asarray(arr["word_atoms"], dtype=np.int64)
    word_n_atoms = np.asarray(arr["word_n_atoms"], dtype=np.int64)
    word_ptr_bond = np.asarray(arr["word_ptr_bond"], dtype=np.int64)
    word_bonds = np.asarray(arr["word_bonds"], dtype=np.int64)
    n_words = int(word_n_atoms.shape[0])
    site_ptr = word_ptr_atom
    n_sites = int(site_ptr[-1])
    if n_sites != int(word_atoms.shape[0]):
        raise ValueError("site count must equal the atom count")
    site_word = np.repeat(np.arange(n_words, dtype=np.int64), np.diff(site_ptr))
    site_atom = np.asarray(word_atoms, dtype=np.int64)
    atom_feat = np.asarray(base["atom_feat"], dtype=np.int64)
    x9 = atom_feat[site_atom]
    # compatible-site groups: identical original nine atom columns *within* a word
    key = np.concatenate([site_word[:, None], x9], axis=1)
    if n_sites:
        order = np.lexsort(tuple(key[:, j] for j in range(key.shape[1] - 1, -1, -1)))
        sorted_key = key[order]
        same = np.zeros(n_sites, dtype=bool)
        same[1:] = np.all(sorted_key[1:] == sorted_key[:-1], axis=1)
        gid = np.cumsum(~same) - 1
        site_group = np.empty(n_sites, dtype=np.int64)
        site_group[order] = gid
        sorted_word = sorted_key[:, 0]
    else:
        site_group = np.zeros(0, dtype=np.int64)
        same = np.zeros(0, dtype=bool)
        sorted_word = np.zeros(0, dtype=np.int64)
    n_groups = int(site_group.max()) + 1 if site_group.size else 0
    group_count = np.bincount(site_group, minlength=n_groups).astype(np.int32)
    # group ids are contiguous per word but the *sites* of a word are not group ordered,
    # so the per-word group counts are taken from the sorted (word, attributes) order.
    per_word_groups = np.bincount(sorted_word, weights=(~same).astype(np.float64),
                                  minlength=int(n_words)).astype(np.int64)
    word_group_ptr = np.concatenate([[0], np.cumsum(per_word_groups)]).astype(np.int64)
    if int(word_group_ptr[-1]) != n_groups:
        raise ValueError(f"compatible-group pointer inconsistent: {word_group_ptr[-1]} != {n_groups}")
    # internal edges (both directions), grouped by word
    bond_u = np.asarray(base["bond_u"], dtype=np.int64)
    bond_v = np.asarray(base["bond_v"], dtype=np.int64)
    graph_atom_ptr = np.asarray(arr["graph_atom_ptr"], dtype=np.int64)
    graph_bond_ptr = np.asarray(arr["graph_bond_ptr"], dtype=np.int64)
    if bond_u.size and int(bond_u.max()) >= int(graph_atom_ptr[-1]):
        raise ValueError("bond endpoints exceed the split atom range")
    atom_site = np.full(int(graph_atom_ptr[-1]), -1, dtype=np.int64)
    atom_site[site_atom] = np.arange(n_sites, dtype=np.int64)
    # `base/<split>/bond_u` is already offset to split-global atom ids
    eu = bond_u
    ev = bond_v
    sa = atom_site[eu]
    sb = atom_site[ev]
    internal_site_a = sa[word_bonds] if word_bonds.size else np.zeros(0, dtype=np.int64)
    internal_site_b = sb[word_bonds] if word_bonds.size else np.zeros(0, dtype=np.int64)
    if word_bonds.size and (np.any(internal_site_a < 0) or np.any(internal_site_b < 0)):
        raise ValueError("internal bond endpoint is not inside any word")
    word_of_bond = np.repeat(np.arange(n_words, dtype=np.int64), np.diff(word_ptr_bond))
    n_ib = int(word_bonds.shape[0])
    edge_src = np.empty(2 * n_ib, dtype=np.int64)
    edge_dst = np.empty(2 * n_ib, dtype=np.int64)
    edge_bond = np.empty(2 * n_ib, dtype=np.int64)
    edge_src[0::2] = internal_site_b
    edge_dst[0::2] = internal_site_a
    edge_src[1::2] = internal_site_a
    edge_dst[1::2] = internal_site_b
    edge_bond[0::2] = word_bonds
    edge_bond[1::2] = word_bonds
    edge_ptr = np.concatenate([[0], np.cumsum(2 * np.diff(word_ptr_bond))]).astype(np.int64)
    site_deg = np.bincount(edge_dst, minlength=n_sites).astype(np.int32)
    # per-word counts
    inc_ptr = np.asarray(arr["inc_ptr"], dtype=np.int64)
    pair_ptr = np.asarray(arr["pair_ptr"], dtype=np.int64)
    word_n_inc = np.diff(inc_ptr).astype(np.int32)
    word_n_pairs = np.diff(pair_ptr).astype(np.int32)
    word_n_ibonds = np.diff(word_ptr_bond).astype(np.int32)
    # site pairs (p, a) with template distance
    word_tmpl = np.asarray(arr["word_tmpl"], dtype=np.int64)
    pair_counts = (word_n_atoms ** 2).astype(np.int64)
    total_pairs = int(pair_counts.sum())
    owner = np.repeat(np.arange(n_words, dtype=np.int64), pair_counts)
    local = np.arange(total_pairs, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(pair_counts)[:-1]]), pair_counts)
    n_of_owner = word_n_atoms[owner]
    p_idx = local // n_of_owner
    a_idx = local % n_of_owner
    site_start = site_ptr[owner]
    pp_p = site_start + p_idx
    pp_a = site_start + a_idx
    pp_dist = dist_table[word_tmpl[owner], p_idx, a_idx].astype(np.int8)
    return {
        "site_ptr": site_ptr.astype(np.int64),
        "site_word": site_word,
        "site_atom": site_atom,
        "site_group": site_group,
        "site_deg": site_deg,
        "group_count": group_count,
        "word_group_ptr": word_group_ptr,
        "edge_ptr": edge_ptr,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_bond": edge_bond,
        "word_n_ibonds": word_n_ibonds,
        "word_n_inc": word_n_inc,
        "word_n_pairs": word_n_pairs,
        "pp_ptr": np.concatenate([[0], np.cumsum(pair_counts)]).astype(np.int64),
        "pp_p": pp_p.astype(np.int64),
        "pp_a": pp_a.astype(np.int64),
        "pp_dist": pp_dist,
    }


def build_derived(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    words, record = load_old_words(dataset)
    dist_table = template_dist_table(words)
    arrays: dict[str, np.ndarray] = {"tmpl/dist": dist_table}
    support = {}
    for split in ("train", "valid"):
        for key, value in derive_split(words, split, dist_table).items():
            arrays[f"derived/{split}/{key}"] = value
        support[split] = derived_support(arrays, split, words)
    arrays["tmpl/n_templates"] = np.asarray([int(words["tmpl/n_templates"])], dtype=np.int64)
    digest = C.content_sha256(arrays)
    freeze = {
        "dataset": dataset,
        "cache_version": C.CACHE_VERSION,
        "derived_path": str(C.derived_path(dataset)),
        "derived_sha256": digest,
        "inherited_word_record": str(C.old_record_path(dataset)),
        "inherited_word_cache_sha256": record.get("cache_sha256"),
        "inherited_template_table": {
            "n_templates": int(words["tmpl/n_templates"]),
            "n_multi_atom_words": int(words["tmpl/n_multi"]),
            "commit": record.get("commit"),
        },
        "definitions": {
            "site": "one entry per (word instance, template position); site order == word_atoms order",
            "compatible_group": "sites of one word with identical original 9-column atom rows",
            "site_pair_distance": "template shortest path, 0..5 (6-atom word maximum)",
            "boundary_count": "number of crossing physical bonds incident to the word",
        },
        "support": support,
        "labels_used": False,
        "split_sizes": {split: int(words[f"row_ids/{split}"].shape[0])
                        for split in ("train", "valid", "test")},
    }
    return arrays, freeze


def derived_support(arrays: Mapping[str, np.ndarray], split: str, words: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive, label-free support statistics of the derived features."""
    pre = f"derived/{split}/"
    site_group = arrays[pre + "site_group"]
    site_word = arrays[pre + "site_word"]
    group_count = arrays[pre + "group_count"]
    word_n_atoms = np.asarray(words[f"word/{split}/word_n_atoms"])
    word_n_inc = arrays[pre + "word_n_inc"]
    word_n_ibonds = arrays[pre + "word_n_ibonds"]
    word_tmpl = np.asarray(words[f"word/{split}/word_tmpl"])
    inc_ptr = np.asarray(words[f"word/{split}/inc_ptr"])
    graph_word_ptr = np.asarray(words[f"word/{split}/graph_word_ptr"])
    inc_own_site = np.asarray(words[f"word/{split}/inc_own_site"])
    inc_own_word = np.repeat(np.arange(word_n_atoms.shape[0]), np.diff(inc_ptr))
    site_group_of_inc = site_group[inc_ptr[:-1][inc_own_word] + inc_own_site]
    n_sites = int(site_word.shape[0])
    n_multi = int((word_n_atoms > 1).sum())
    sizes = {str(k): int(v) for k, v in zip(*np.unique(word_n_atoms, return_counts=True))}
    covers = np.zeros(n_sites, dtype=bool)
    covers[inc_ptr[:-1][inc_own_word] + inc_own_site] = True
    return {
        "n_word_instances": int(word_n_atoms.shape[0]),
        "n_atoms": n_sites,
        "n_multi_atom_words": n_multi,
        "word_size_histogram": sizes,
        "word_size_mean": float(word_n_atoms.mean()),
        "n_incidences": int(inc_own_site.shape[0]),
        "boundary_incidence_ratio": float(inc_own_site.shape[0]) / max(1, word_n_atoms.shape[0]),
        "internal_bonds_per_word_mean": float(word_n_ibonds.mean()),
        "site_coverage_by_boundary_incidence": float(covers.mean()),
        "incidence_count_histogram": {str(k): int(v) for k, v in
                                      zip(*np.unique(word_n_inc, return_counts=True))},
        "distinct_site_count_histogram": {str(k): int(v) for k, v in
                                          zip(*np.unique(np.minimum(word_n_inc, 6), return_counts=True))},
        "compatible_group_size_histogram": {str(k): int(v) for k, v in
                                            zip(*np.unique(group_count, return_counts=True))},
        "n_incidences_group_ge2": int(np.sum(
            group_count[site_group_of_inc] >= 2)) if site_group_of_inc.size else 0,
        "n_incidences_group_ge2_frac": float(np.mean(group_count[site_group_of_inc] >= 2))
        if site_group_of_inc.size else 0.0,
        "n_word_types": int(np.unique(word_tmpl).size),
        "n_graphs": int(graph_word_ptr.shape[0] - 1),
    }


# ---------------------------------------------------------------------------
# split containers and packing
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    name: str
    row_ids: np.ndarray
    labels: np.ndarray
    graph_n_atoms: np.ndarray
    graph_n_bonds: np.ndarray
    atom_feat: np.ndarray
    bond_feat: np.ndarray
    arrays: Mapping[str, np.ndarray]
    scratch: dict[str, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        arr = self.arrays
        n_words = int(arr["graph_word_ptr"][-1])
        n_atoms = int(arr["graph_atom_ptr"][-1])
        n_bonds = int(arr["graph_bond_ptr"][-1])
        n_inc = int(arr["inc_ptr"][-1])
        n_pair = int(arr["pair_ptr"][-1])
        n_groups = int(arr["group_count"].shape[0])
        self.scratch = {
            "word": np.full(n_words, -1, dtype=np.int64),
            "atom": np.full(n_atoms, -1, dtype=np.int64),
            "bond": np.full(n_bonds, -1, dtype=np.int64),
            "site": np.full(n_atoms, -1, dtype=np.int64),
            "group": np.full(n_groups, -1, dtype=np.int64),
            "inc": np.full(n_inc, -1, dtype=np.int64),
            "pair": np.full(n_pair, -1, dtype=np.int64),
        }

    def __len__(self) -> int:
        return int(self.row_ids.shape[0])


def _range_ids(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    counts = hi - lo
    if counts.size == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return np.repeat(lo, counts) + (np.arange(int(offsets[-1]), dtype=np.int64)
                                    - np.repeat(offsets[:-1], counts))


def merge_split_arrays(words: Mapping[str, Any], derived: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    out = dict(_split_arrays(words, split))
    pre = f"derived/{split}/"
    for key, value in derived.items():
        if key.startswith(pre):
            out[key[len(pre):]] = value
    return out


def make_split(data: Mapping[str, Any], split: str) -> SplitData:
    words, derived, raw = data["words"], data["derived"], data["raw"]
    arrays = merge_split_arrays(words, derived, split)
    base = _base_arrays(words, split)
    return SplitData(
        name=split,
        row_ids=np.asarray(words[f"row_ids/{split}"], dtype=np.int64),
        labels=np.asarray(words[f"labels/{split}"], dtype=np.float64),
        graph_n_atoms=np.asarray(base["graph_n_atoms"], dtype=np.int64),
        graph_n_bonds=np.asarray(base["graph_n_bonds"], dtype=np.int64),
        atom_feat=np.asarray(base["atom_feat"], dtype=np.int64),
        bond_feat=np.asarray(base["bond_feat"], dtype=np.int64),
        arrays=arrays,
    )


def pack_batch(data: SplitData, idx: Sequence[int]) -> dict[str, Any]:
    """Assemble one mini-batch with batch-local indices everywhere."""
    arr = data.arrays
    idx = np.asarray(idx, dtype=np.int64)
    G = int(idx.shape[0])
    gw = arr["graph_word_ptr"]
    ga = arr["graph_atom_ptr"]
    gb = arr["graph_bond_ptr"]
    sp = arr["site_ptr"]
    gp = arr["word_group_ptr"]
    ep = arr["edge_ptr"]
    ip = arr["inc_ptr"]
    pp = arr["pair_ptr"]
    w_lo, w_hi = gw[idx], gw[idx + 1]
    a_lo, a_hi = ga[idx], ga[idx + 1]
    b_lo, b_hi = gb[idx], gb[idx + 1]
    s_lo, s_hi = sp[w_lo], sp[w_hi]
    g_lo, g_hi = gp[w_lo], gp[w_hi]
    e_lo, e_hi = ep[w_lo], ep[w_hi]
    w_ids = _range_ids(w_lo, w_hi)
    a_ids = _range_ids(a_lo, a_hi)
    b_ids = _range_ids(b_lo, b_hi)
    s_ids = _range_ids(s_lo, s_hi)
    g_ids = _range_ids(g_lo, g_hi)
    e_ids = _range_ids(e_lo, e_hi)
    i_ids = _range_ids(ip[w_lo], ip[w_hi])
    p_ids = _range_ids(pp[w_lo], pp[w_hi])
    sw, sa, sb = data.scratch["word"], data.scratch["atom"], data.scratch["bond"]
    ss, sg = data.scratch["site"], data.scratch["group"]
    si, spr = data.scratch["inc"], data.scratch["pair"]
    n_w = int(w_ids.size)
    n_a = int(a_ids.size)
    n_b = int(b_ids.size)
    n_s = int(s_ids.size)
    n_g = int(g_ids.size)
    n_e = int(e_ids.size)
    n_i = int(i_ids.size)
    n_p = int(p_ids.size)
    sw[w_ids] = np.arange(n_w)
    sa[a_ids] = np.arange(n_a)
    sb[b_ids] = np.arange(n_b)
    ss[s_ids] = np.arange(n_s)
    sg[g_ids] = np.arange(n_g)
    si[i_ids] = np.arange(n_i)
    spr[p_ids] = np.arange(n_p)
    w_new = np.concatenate([[0], np.cumsum(w_hi - w_lo)])
    s_new = np.concatenate([[0], np.cumsum(s_hi - s_lo)])
    g_new = np.concatenate([[0], np.cumsum(g_hi - g_lo)])
    a_new = np.concatenate([[0], np.cumsum(a_hi - a_lo)])
    b_new = np.concatenate([[0], np.cumsum(b_hi - b_lo)])
    e_new = np.concatenate([[0], np.cumsum(e_hi - e_lo)])
    i_new = np.concatenate([[0], np.cumsum(ip[w_hi] - ip[w_lo])])
    p_new = np.concatenate([[0], np.cumsum(pp[w_hi] - pp[w_lo])])
    graph_of_word = np.repeat(np.arange(G), w_hi - w_lo)
    local_at = w_ids - w_lo[graph_of_word] + w_new[graph_of_word]
    # batch-local first site of every word: graph offset + offset inside the graph
    word_site_off = np.empty(n_w, dtype=np.int64)
    word_site_off[local_at] = s_new[graph_of_word] + (sp[w_ids] - sp[w_lo[graph_of_word]])
    word_natoms = arr["word_n_atoms"][w_ids].astype(np.int64)
    out: dict[str, Any] = {"n_graphs": G}
    out["atom_feat"] = data.atom_feat[a_ids]
    out["bond_feat"] = data.bond_feat[b_ids]
    out["row_ids"] = data.row_ids[idx]
    out["y"] = data.labels[idx]
    out["n_atoms_graph"] = data.graph_n_atoms[idx]
    out["n_bonds_graph"] = data.graph_n_bonds[idx]
    out["g_size"] = np.stack([np.log1p(data.graph_n_atoms[idx].astype(np.float64)),
                              np.log1p(data.graph_n_bonds[idx].astype(np.float64)),
                              np.log1p((w_hi - w_lo).astype(np.float64))], axis=1)
    # words
    out["word_tmpl"] = arr["word_tmpl"][w_ids]
    out["word_natoms"] = word_natoms
    out["graph_of_word"] = graph_of_word
    wpa = arr["word_ptr_atom"]
    wpb = arr["word_ptr_bond"]
    out["word_atoms"] = sa[arr["word_atoms"][_range_ids(wpa[w_ids], wpa[w_ids + 1])]]
    out["word_of_atom"] = np.repeat(np.arange(n_w), np.diff(wpa)[w_ids])
    out["word_ptr_atom"] = np.concatenate([[0], np.cumsum(np.diff(wpa)[w_ids])])
    out["word_bonds"] = sb[arr["word_bonds"][_range_ids(wpb[w_ids], wpb[w_ids + 1])]]
    out["word_of_bond"] = np.repeat(np.arange(n_w), np.diff(wpb)[w_ids])
    out["word_group_ptr"] = np.concatenate([[0], np.cumsum(np.diff(gp)[w_ids])])
    out["word_n_ibonds"] = arr["word_n_ibonds"][w_ids].astype(np.int64)
    out["word_n_inc"] = arr["word_n_inc"][w_ids].astype(np.int64)
    out["word_n_pairs"] = arr["word_n_pairs"][w_ids].astype(np.int64)
    # sites (site ids are contiguous per word, so the local order is a repeat)
    out["site_word"] = np.repeat(np.arange(n_w), np.diff(sp)[w_ids])
    out["site_atom"] = sa[arr["site_atom"][s_ids]]
    out["site_deg"] = arr["site_deg"][s_ids].astype(np.int64)
    out["site_group"] = sg[arr["site_group"][s_ids]]
    out["group_count"] = arr["group_count"][g_ids].astype(np.int64)
    out["n_groups"] = n_g
    # edges
    out["edge_src"] = ss[arr["edge_src"][e_ids]]
    out["edge_dst"] = ss[arr["edge_dst"][e_ids]]
    out["edge_bond"] = sb[arr["edge_bond"][e_ids]]
    # incidences
    inc_own_site = arr["inc_own_site"][i_ids]
    inc_other_word = sw[arr["inc_other"][i_ids]]
    out["inc_own_site"] = np.repeat(word_site_off, np.diff(ip)[w_ids]) + inc_own_site
    out["inc_other_word"] = inc_other_word
    out["inc_other_site"] = (word_site_off[inc_other_word]
                             + arr["inc_other_site"][i_ids])
    out["inc_own_atom"] = sa[arr["inc_own_atom"][i_ids]]
    out["inc_other_atom"] = sa[arr["inc_other_atom"][i_ids]]
    out["inc_bond"] = sb[arr["inc_bond"][i_ids]]
    out["word_of_inc"] = np.repeat(np.arange(n_w), np.diff(ip)[w_ids])
    # incidence pairs
    out["pair_a"] = si[arr["pair_a"][p_ids]]
    out["pair_b"] = si[arr["pair_b"][p_ids]]
    out["pair_R"] = arr["pair_R"][p_ids]
    out["pair_fields"] = arr["pair_fields"][p_ids]
    out["word_of_pair"] = np.repeat(np.arange(n_w), np.diff(pp)[w_ids])
    # site pairs (p, a): each word owns n_i^2 entries, so the shift is per word
    ppr = arr["pp_ptr"]
    j_ids = _range_ids(ppr[w_lo], ppr[w_hi])
    shift = np.repeat(word_site_off - sp[w_ids], np.diff(ppr)[w_ids])
    out["pp_p"] = arr["pp_p"][j_ids] + shift
    out["pp_a"] = arr["pp_a"][j_ids] + shift
    out["pp_d"] = arr["pp_dist"][j_ids].astype(np.int64)
    return out


def to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            dtype = torch.float32 if value.dtype.kind == "f" else torch.long
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# generic packer: arbitrary molecules + arbitrary partition (probes, D3)
# ---------------------------------------------------------------------------


def pack_molecules(molecules: Sequence[W.Molecule], partitions: Sequence[Sequence[Sequence[int]]],
                   table: W.TemplateTable, *, labels: Sequence[float] | None = None,
                   row_ids: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Rebuild the frozen array format from molecules and an explicit partition.

    ``partitions[g]`` is a list of atom lists (one per word instance).  Every
    word is re-ordered into its exact canonical template order, exactly as the
    frozen v1 builder does, so the same packing code path is exercised by the
    real data and by the synthetic probes.
    """
    states: dict[int, W.PartState] = {}
    graphs = list(range(len(molecules)))
    for g in graphs:
        mol = molecules[g]
        state = W.PartState()
        for atoms in partitions[g]:
            tid, ordered = _intern_piece(mol, atoms, table)
            state.pieces.append(list(ordered))
            state.tmpl.append(tid)
        covered = sorted(a for piece in state.pieces for a in piece)
        if covered != list(range(mol.n_atoms)):
            raise ValueError(f"graph {g}: partition does not cover every atom exactly once")
        states[g] = state
    payload = W.build_split(molecules, graphs, states, table)
    offsets = np.concatenate([[0], np.cumsum([m.n_atoms for m in molecules])])
    bu = np.concatenate([m.bond_u + offsets[g] for g, m in enumerate(molecules)])
    bv = np.concatenate([m.bond_v + offsets[g] for g, m in enumerate(molecules)])
    base: dict[str, np.ndarray] = {
        "atom_feat": np.concatenate([m.atom_feat for m in molecules], axis=0).astype(np.int16),
        "bond_u": bu.astype(np.int32),
        "bond_v": bv.astype(np.int32),
        "bond_feat": np.concatenate([m.bond_feat for m in molecules], axis=0).astype(np.int16),
        "graph_n_atoms": np.asarray([m.n_atoms for m in molecules], dtype=np.int32),
        "graph_n_bonds": np.asarray([len(m.bond_u) for m in molecules], dtype=np.int32),
    }
    out = {f"word/probe/{k}": v for k, v in payload.items()}
    out.update({f"base/probe/{k}": v for k, v in base.items()})
    out["row_ids/probe"] = np.asarray(row_ids if row_ids is not None else graphs, dtype=np.int64)
    out["labels/probe"] = np.asarray(labels if labels is not None else np.zeros(len(molecules)),
                                    dtype=np.float64)
    return out


def _intern_piece(mol: W.Molecule, atoms: Sequence[int], table: W.TemplateTable):
    local = list(atoms)
    pos = {a: i for i, a in enumerate(local)}
    labels = mol.atom_feat[local][:, W.WORDS_ATOM_COLS]
    lu, lv, lt = [], [], []
    for a in local:
        for nbr, bk in mol.adj[a]:
            if nbr in pos and pos[a] < pos[nbr]:
                lu.append(pos[a])
                lv.append(pos[nbr])
                lt.append(int(mol.bond_feat[bk, 0]))
    canon, order = W.canonicalize(labels, np.asarray(lu), np.asarray(lv), np.asarray(lt))
    tid = table.add_from_canon(canon, np.asarray(labels, dtype=np.int64),
                               list(zip(lu, lv, lt)), order)
    return tid, [local[i] for i in order]


def table_from_record(record: Mapping[str, Any]) -> W.TemplateTable:
    """Rebuild the frozen template table (tids, canonical strings) from a record."""
    table = W.TemplateTable()
    for row in record["template_table"]["templates"]:
        table._register(row["canon"], np.asarray(row["labels"], dtype=np.int64).reshape(-1, 3),
                        np.asarray(row["bu"], dtype=np.int64), np.asarray(row["bv"], dtype=np.int64),
                        np.asarray(row["btype"], dtype=np.int64))
        tmpl = table.templates[-1]
        if tmpl.tid != row["tid"] or tmpl.canon != row["canon"]:
            raise ValueError("frozen template table order mismatch")
    return table


def split_from_arrays(all_arrays: Mapping[str, np.ndarray], dist_table: np.ndarray,
                      split: str = "probe") -> SplitData:
    """SplitData for a synthetic/alternate partition packed in the frozen format."""
    word = {k[len(f"word/{split}/"):]: v for k, v in all_arrays.items()
            if k.startswith(f"word/{split}/")}
    base = {k[len(f"base/{split}/"):]: v for k, v in all_arrays.items()
            if k.startswith(f"base/{split}/")}
    derived = derive_core(word, base, dist_table)
    arrays = dict(word)
    arrays.update(derived)
    return SplitData(
        name=split,
        row_ids=np.asarray(all_arrays[f"row_ids/{split}"], dtype=np.int64),
        labels=np.asarray(all_arrays[f"labels/{split}"], dtype=np.float64),
        graph_n_atoms=np.asarray(base["graph_n_atoms"], dtype=np.int64),
        graph_n_bonds=np.asarray(base["graph_n_bonds"], dtype=np.int64),
        atom_feat=np.asarray(base["atom_feat"], dtype=np.int64),
        bond_feat=np.asarray(base["bond_feat"], dtype=np.int64),
        arrays=arrays,
    )
