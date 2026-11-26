"""The frozen strict real-edit catalogue, its components, its Fit support, and
the Fit-only pair sampler.

Everything here is a deterministic function of the committed
``catalogue_pairs.csv`` (SHA-256 pinned) plus the official split and the D0
Fit/Dev assignment.  No label is read by the index-building code: the row/label
columns are only attached in :func:`attach_labels`, which is called by the
evaluation code after all checkpoints are frozen.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C

PAIR_ID = "pair_id"


@dataclass
class PairIndex:
    rows: list[dict[str, Any]]                 # the 2133 both-official-Train pairs
    n_catalogue: int
    n_excluded_not_both_train: int
    component_ids: np.ndarray                  # [2133] component id in the 2133-edge graph
    component_sizes: dict[int, int]
    cell: np.ndarray                           # 'FF' | 'FD' | 'DD'
    fit_pair_mask: np.ndarray                  # [2133] True for Fit-Fit
    fit_component_ids: np.ndarray              # [2133] component id in the Fit-only graph (FF rows)
    fit_component_sizes: dict[int, int]
    support_other_component: np.ndarray        # [2133] bool
    support_two_components: np.ndarray         # [2133] bool
    n_fit_supporting_components: np.ndarray    # [2133] int
    orientation_sign: np.ndarray               # [2133] int
    dev_dev_index: np.ndarray                  # indices into rows for the 276 Dev-Dev pairs
    fit_dev_index: np.ndarray                  # indices for the 238 Fit-Dev pairs
    fit_fit_index: np.ndarray                  # indices for the 1619 Fit-Fit pairs
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rows)


def _union_find(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> dict[int, int]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for node in nodes:
        find(int(node))
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return {node: find(node) for node in list(parent)}


def _dense_component_labels(keys: Sequence[int]) -> tuple[np.ndarray, dict[int, int]]:
    uniq = sorted(set(int(k) for k in keys))
    index = {k: i for i, k in enumerate(uniq)}
    return np.asarray([index[int(k)] for k in keys], dtype=np.int64), {i: k for k, i in index.items()}


def build_pair_index(bundle, d0: Mapping[str, Any]) -> PairIndex:
    path = C.CATALOGUE
    got = C.sha256_file(path)
    if got != C.EXPECTED_SHA["catalogue_pairs"]:
        raise RuntimeError(f"catalogue sha256 {got} != frozen value")
    with path.open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    n_catalogue = len(raw)

    train = set(int(v) for v in bundle.train_idx.tolist())
    fit = set(int(v) for v in d0["fit_rows"].tolist())
    dev = set(int(v) for v in d0["dev_rows"].tolist())

    rows: list[dict[str, Any]] = []
    excluded = 0
    for r in raw:
        a, b = int(r["row_a"]), int(r["row_b"])
        if a not in train or b not in train:
            excluded += 1
            continue
        # Orientation audit.  The catalogue's canonical transformation direction
        # is fixed by ``transform_id = s_key_a > s_key_b``; therefore the stored
        # a->b direction *is* the canonical source->target direction for every
        # row, and the orientation sign is +1.  ``row_lo``/``row_hi`` are only the
        # unordered molecule-pair id and carry no orientation meaning.
        head = r["transform_id"].split("|")[0].split(">")
        if head[0] != r["s_key_a"] or head[1] != r["s_key_b"]:
            raise RuntimeError(f"transform orientation mismatch on {r['pair_id']}")
        if head[0] == head[1]:
            raise RuntimeError(f"degenerate transform on {r['pair_id']}")
        sign = 1
        row_order_sign = 1 if a == int(r["row_lo"]) else -1
        # one frozen cell label per pair: FF, DF (mixed), DD
        cell = "FF" if (a in fit and b in fit) else ("DD" if (a in dev and b in dev) else "DF")
        rows.append({
            "pair_id": r["pair_id"], "row_a": a, "row_b": b,
            "row_lo": int(r["row_lo"]), "row_hi": int(r["row_hi"]),
            "transform_id": r["transform_id"], "core_uid": r["core_uid"],
            "core_id": r["core_id"], "c_key": r["c_key"], "core_size": int(r["core_size"]),
            "s_key_a": r["s_key_a"], "s_key_b": r["s_key_b"],
            "s_size_a": int(r["s_size_a"]), "s_size_b": int(r["s_size_b"]),
            "cut_bond_type": int(r["cut_bond_type"]),
            "cut_bond_stereo": int(r["cut_bond_stereo"]),
            "cut_edge_a": int(r["cut_edge_a"]), "cut_edge_b": int(r["cut_edge_b"]),
            "cell": cell, "sign": sign, "row_order_sign": row_order_sign,
        })
    seen_pairs = set()
    for r in rows:
        key = frozenset((r["row_a"], r["row_b"]))
        if key in seen_pairs:
            raise RuntimeError(f"duplicate unordered molecule pair {sorted(key)}")
        seen_pairs.add(key)

    # molecule graph of the 2133 edges -> 391 components
    mol_comp = _union_find(list(range(0)), [(r["row_a"], r["row_b"]) for r in rows])
    comp_keys = [mol_comp[r["row_a"]] for r in rows]
    component_ids, _ = _dense_component_labels(comp_keys)
    component_sizes = {int(c): int((component_ids == c).sum()) for c in np.unique(component_ids)}
    cell = np.asarray([r["cell"] for r in rows], dtype=object)

    # Fit-only graph: the 1619 Fit-Fit edges -> 340 components
    ff_local = _union_find([], [(r["row_a"], r["row_b"]) for r in rows if r["cell"] == "FF"])
    fit_component_ids = np.full(len(rows), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        if r["cell"] == "FF":
            fit_component_ids[i] = ff_local[r["row_a"]]
    ff_keys = [int(v) for v in fit_component_ids[fit_component_ids >= 0].tolist()]
    dense_ff, _ = _dense_component_labels(ff_keys)
    fit_component_ids[fit_component_ids >= 0] = dense_ff
    fit_component_sizes = {int(c): int((fit_component_ids == c).sum())
                           for c in np.unique(dense_ff)}

    # exact-transform support inside Fit-Fit, excluding the evaluated pair's own
    # 2133-graph component
    transform_fit_components: dict[str, set[int]] = {}
    for i, r in enumerate(rows):
        if r["cell"] != "FF":
            continue
        transform_fit_components.setdefault(r["transform_id"], set()).add(int(component_ids[i]))

    support_other = np.zeros(len(rows), dtype=bool)
    support_two = np.zeros(len(rows), dtype=bool)
    n_support_comp = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        others = transform_fit_components.get(r["transform_id"], set()) - {int(component_ids[i])}
        n_support_comp[i] = len(others)
        support_other[i] = len(others) >= 1
        support_two[i] = len(others) >= 2

    orientation_sign = np.asarray([r["sign"] for r in rows], dtype=np.int64)

    index = np.arange(len(rows), dtype=np.int64)
    dev_dev = index[cell == "DD"]
    fit_dev = index[cell == "DF"]
    fit_fit = index[cell == "FF"]

    meta = {
        "catalogue_path": str(path.relative_to(C.REPO_ROOT)),
        "catalogue_sha256": got,
        "n_catalogue": n_catalogue,
        "n_excluded_not_both_official_train": excluded,
        "n_pairs_2133": int(len(rows)),
        "n_components_2133": int(len(np.unique(component_ids))),
        "n_fit_fit_pairs": int(fit_fit.size),
        "n_fit_fit_molecules": int(len(set([r["row_a"] for i, r in enumerate(rows) if cell[i] == "FF"]
                                           + [r["row_b"] for i, r in enumerate(rows) if cell[i] == "FF"]))),
        "n_fit_fit_components": int(len(np.unique(dense_ff))),
        "n_fit_dev_pairs": int(fit_dev.size),
        "n_fit_dev_dev_end_molecules": int(len({r["row_a"] if r["cell"] == "DF" else -1 for r in rows} - {-1})),
        "n_fit_dev_components": int(len(np.unique(component_ids[fit_dev]))),
        "n_dev_dev_pairs": int(dev_dev.size),
        "n_dev_dev_molecules": int(len(set([rows[i]["row_a"] for i in dev_dev.tolist()]
                                           + [rows[i]["row_b"] for i in dev_dev.tolist()]))),
        "n_dev_dev_components": int(len(np.unique(component_ids[dev_dev]))),
        "dev_dev_transform_seen_in_fit": int(sum(
            1 for i in dev_dev.tolist() if rows[i]["transform_id"] in transform_fit_components)),
        "dev_dev_supported_other_component": int(support_other[dev_dev].sum()),
        "dev_dev_supported_other_component_components": int(len(np.unique(component_ids[dev_dev][support_other[dev_dev]]))),
        "dev_dev_supported_other_component_transforms": int(len({
            rows[i]["transform_id"] for i in dev_dev.tolist() if support_other[i]})),
        "dev_dev_supported_two_components": int(support_two[dev_dev].sum()),
        "dev_dev_supported_two_components_components": int(len(np.unique(component_ids[dev_dev][support_two[dev_dev]]))),
        "dev_dev_supported_two_components_transforms": int(len({
            rows[i]["transform_id"] for i in dev_dev.tolist() if support_two[i]})),
        "orientation_all_plus_one": bool(np.all(orientation_sign == 1)),
        "n_sign_minus_one": int((orientation_sign == -1).sum()),
    }
    return PairIndex(
        rows=rows, n_catalogue=n_catalogue, n_excluded_not_both_train=excluded,
        component_ids=component_ids, component_sizes=component_sizes,
        cell=cell, fit_pair_mask=(cell == "FF"),
        fit_component_ids=fit_component_ids, fit_component_sizes=fit_component_sizes,
        support_other_component=support_other, support_two_components=support_two,
        n_fit_supporting_components=n_support_comp,
        orientation_sign=orientation_sign,
        dev_dev_index=dev_dev, fit_dev_index=fit_dev, fit_fit_index=fit_fit,
        meta=meta,
    )


def subset_masks(index: PairIndex) -> dict[str, np.ndarray]:
    """The frozen Dev-Dev evaluation subsets (booleans over the 276 Dev-Dev pairs)."""
    dd = index.dev_dev_index
    supported = index.support_other_component[dd]
    strong = index.support_two_components[dd]
    return {
        "dev_dev_all": np.ones(dd.size, dtype=bool),
        "dev_dev_supported119": supported,
        "dev_dev_strong97": strong,
        "dev_dev_unsupported157": ~supported,
    }


# ---------------------------------------------------------------------------
# Fit-only pair sampler (frozen schedule, seed 23017)
# ---------------------------------------------------------------------------


def pair_schedule(index: PairIndex, n_steps: int, pairs_per_step: int = 64,
                  seed: int = C.PAIR_SEED) -> np.ndarray:
    """``[n_steps, pairs_per_step]`` indices into ``index.rows`` (Fit-Fit only).

    One component is drawn uniformly from the 340 Fit-only components, then one
    Fit-Fit edge uniformly inside that component, with replacement.  The stream
    is generated once, so every family consumes a prefix of the same schedule.
    """
    fit_edges = index.fit_fit_index
    comps = np.asarray([int(index.fit_component_ids[i]) for i in fit_edges.tolist()], dtype=np.int64)
    order = np.argsort(comps, kind="stable")
    comps_sorted = comps[order]
    boundaries = np.searchsorted(comps_sorted, np.unique(comps_sorted), side="left")
    unique_comps, first = np.unique(comps_sorted, return_index=True)
    members = np.split(fit_edges[order], first[1:])
    n_comp = len(members)
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(n_steps), int(pairs_per_step)), dtype=np.int32)
    for step in range(int(n_steps)):
        draw = rng.integers(0, n_comp, size=int(pairs_per_step))
        for j, c in enumerate(draw.tolist()):
            group = members[c]
            out[step, j] = group[rng.integers(0, group.size)]
    return out


# ---------------------------------------------------------------------------
# label attachment (only called after all checkpoints are frozen)
# ---------------------------------------------------------------------------


def attach_labels(index: PairIndex, labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    a = np.asarray([r["row_a"] for r in index.rows], dtype=np.int64)
    b = np.asarray([r["row_b"] for r in index.rows], dtype=np.int64)
    y_a = labels[a]
    y_b = labels[b]
    # canonical direction: the stored a->b direction already is the transform's
    # own source->target direction (verified at index build time).
    delta = y_b - y_a
    return {"y_a": y_a, "y_b": y_b, "delta_y": delta, "row_a": a, "row_b": b}
