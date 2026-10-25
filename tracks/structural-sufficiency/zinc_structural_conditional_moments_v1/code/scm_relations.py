"""Line-graph distance cache for the frozen structural relation bands.

The relation is a deterministic, label-free function of the raw graph:

    d_L(e, f)  = shortest path length in the line graph of the molecule,
                 i.e. the number of bond-adjacency steps between physical
                 bonds e and f (d_L(e,e)=0; unreachable -> 255).

Physical edges that share an atom are line-graph neighbours.  This is the
minimal definition of "how close are these two physical bonds structurally"
and uses no atom/bond attributes, no labels and no learned quantity.

The cache is per molecule: one dense ``n_b x n_b`` uint8 matrix, flattened.
Molecules never share a relation matrix, so there is no cross-molecule path.

Two frozen band masks are derived from it once:

    mu_t[e,f] = 1[ BANDS[t][0] <= d_L(e,f) <= BANDS[t][1] ]

``prepare`` also emits the label-free structural statistics consumed by
``analysis/data_feasibility.json``: per-bond band sizes, emptiness rates,
row total-variation distance from the uniform distribution, coverage, and the
cross-band correlation.  No labels are loaded here at all.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch

import scm_common as S
import ad_data as D


def line_graph_distances(bond_index: np.ndarray, n_bonds: int) -> np.ndarray:
    """All-pairs line-graph distances for one molecule (uint8, 255=unreachable)."""
    inc: dict[int, list[int]] = {}
    for j, (u, v) in enumerate(np.asarray(bond_index, dtype=np.int64)):
        inc.setdefault(int(u), []).append(j)
        inc.setdefault(int(v), []).append(j)
    adj = [set() for _ in range(n_bonds)]
    for lst in inc.values():
        if len(lst) > 1:
            for a in lst:
                for b in lst:
                    if a != b:
                        adj[a].add(b)
    dist = np.full((n_bonds, n_bonds), 255, dtype=np.uint8)
    for src in range(n_bonds):
        dist[src, src] = 0
        q = deque([src])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[src, y] == 255:
                    dist[src, y] = dist[src, x] + 1
                    q.append(y)
    return dist


def build_split(split: str, cache: dict | None = None) -> dict:
    cache = D.load_cache(split) if cache is None else cache
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    n_mol = len(bond_ptr) - 1
    n_bonds = np.diff(bond_ptr).astype(np.int64)
    flat = []
    t0 = time.perf_counter()
    for r in range(n_mol):
        s, e = int(bond_ptr[r]), int(bond_ptr[r + 1])
        flat.append(line_graph_distances(bond_index[s:e], e - s).reshape(-1))
    dist = np.concatenate(flat, axis=0) if flat else np.zeros(0, dtype=np.uint8)
    # Every molecule contributes an n_b x n_b dense block, so the offset grows
    # by n_b**2 (not by n_b).
    ptr = np.concatenate([[0], np.cumsum(n_bonds * n_bonds)]).astype(np.int64)
    out = {
        "split": split,
        "ptr": ptr,
        "dist": dist,
        "n_bonds": n_bonds.astype(np.int16),
        "bands": [list(b) for b in S.BANDS],
        "source_bond_index_sha256": _sha_arrays(bond_index),
        "source_bond_ptr_sha256": _sha_arrays(bond_ptr),
        "n_molecules": int(n_mol),
        "prep_seconds": float(time.perf_counter() - t0),
        "definition": "line-graph shortest path; d(e,e)=0; unreachable=255",
    }
    return out


def _sha_arrays(a: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save_split(payload: dict, split: str) -> None:
    path = S.relation_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ptr=payload["ptr"],
        dist=payload["dist"],
        n_bonds=payload["n_bonds"],
        meta=np.array(
            [
                payload["source_bond_index_sha256"],
                payload["source_bond_ptr_sha256"],
                str([list(b) for b in S.BANDS]),
            ]
        ),
    )


def load_split(split: str) -> dict:
    path = S.relation_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing relation cache {path}; run scm_relations.py prep")
    z = np.load(path, allow_pickle=False)
    bands = eval(str(z["meta"][2]))  # controlled, written by save_split
    if [list(b) for b in bands] != [list(b) for b in S.BANDS]:
        raise RuntimeError(f"{path}: cached bands {bands} != frozen {S.BANDS}")
    return {
        "ptr": np.asarray(z["ptr"], dtype=np.int64),
        "dist": np.asarray(z["dist"], dtype=np.uint8),
        "n_bonds": np.asarray(z["n_bonds"], dtype=np.int64),
        "bands": bands,
        "source_bond_index_sha256": str(z["meta"][0]),
        "source_bond_ptr_sha256": str(z["meta"][1]),
    }


def row_matrix(store: dict, row: int) -> np.ndarray:
    n = int(store["n_bonds"][row])
    s = int(store["ptr"][row])
    return store["dist"][s:s + n * n].reshape(n, n)


def validate(store: dict) -> dict:
    """Structural invariants of the cached relation (no content, no labels)."""
    nb = store["n_bonds"].astype(np.int64)
    ptr = store["ptr"]
    if int(ptr[-1]) != int((nb * nb).sum()) or len(store["dist"]) != int(ptr[-1]):
        raise RuntimeError("relation cache pointer/length mismatch")
    if np.any(np.diff(ptr) < 0):
        raise RuntimeError("relation cache pointer is not monotone")
    max_self = 0
    max_overflow = 0
    for r, n in enumerate(nb):
        if n == 0:
            continue
        dmat = row_matrix(store, r)
        if dmat.shape != (n, n):
            raise RuntimeError(f"row {r}: shape {dmat.shape} != {(n, n)}")
        max_self = max(max_self, int(np.abs(np.diag(dmat)).max()))
        mu1, mu2 = band_masks(dmat)
        # bands are disjoint and exclude the self pair
        if int((mu1 & mu2).sum()):
            raise RuntimeError(f"row {r}: band supports overlap")
        max_overflow = max(max_overflow, int((mu1.sum(1) + mu2.sum(1)).max()) - int(n - 1))
    if max_self != 0:
        raise RuntimeError(f"non-zero self distance {max_self}")
    return {"max_abs_self_distance": max_self,
            "max_band_count_minus_other_bonds": max_overflow,
            "bands_disjoint": True,
            "no_cross_molecule_pairs": True}


def band_masks(dist: np.ndarray) -> list[np.ndarray]:
    return [((dist >= lo) & (dist <= hi)) for lo, hi in S.BANDS]


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------


class RelationCollator:
    """Wrap the frozen P3N collation and append the frozen relation masks."""

    def __init__(self, store: dict):
        self.store = store
        self.ptr = store["ptr"]
        self.dist = store["dist"]
        self.n_bonds = store["n_bonds"].astype(np.int64)

    def add(self, batch: dict, rows: np.ndarray) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        B = len(rows)
        nb = self.n_bonds[rows]
        bmax = max(int(nb.max()) if B else 1, 1)
        E = int(nb.sum())
        dist_pad = np.full((B, bmax, bmax), 255, dtype=np.uint8)
        mask = np.zeros((B, bmax), dtype=bool)
        mask[np.arange(B)[:, None], np.arange(bmax)[None, :]] = (
            np.arange(bmax)[None, :] < nb[:, None]
        )
        for i, row in enumerate(rows):
            n = int(nb[i])
            s = int(self.ptr[row])
            dist_pad[i, :n, :n] = self.dist[s:s + n * n].reshape(n, n)
        mu1 = (dist_pad >= S.BANDS[0][0]) & (dist_pad <= S.BANDS[0][1])
        mu2 = (dist_pad >= S.BANDS[1][0]) & (dist_pad <= S.BANDS[1][1])
        bond_local = np.concatenate([np.arange(c, dtype=np.int64) for c in nb]) if E else np.zeros(0, np.int64)
        out = dict(batch)
        out.update(
            {
                "bond_mask": torch.from_numpy(mask),
                "bond_local": torch.from_numpy(bond_local),
                "bond_bmax": int(bmax),
                "mu1": torch.from_numpy(mu1),
                "mu2": torch.from_numpy(mu2),
            }
        )
        return out


def make_collate(store: dict, base_collate):
    rc = RelationCollator(store)

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        batch = base_collate(cache, rows, candidate, valid=valid, walk_cols=walk_cols)
        return rc.add(batch, rows)

    return collate


def stats(store: dict) -> dict:
    """Label-free structural statistics for one split."""
    invariants = validate(store)
    nb = store["n_bonds"].astype(np.int64)
    n1_all, n2_all, tv1, tv2, cover1, cover2 = [], [], [], [], [], []
    diam, self_in = [], 0
    band_cross = 0
    for r in range(len(nb)):
        n = int(nb[r])
        dmat = row_matrix(store, r)
        mu1, mu2 = band_masks(dmat)
        self_in += int(np.trace(mu1)) + int(np.trace(mu2))
        n1 = mu1.sum(1).astype(np.int64)
        n2 = mu2.sum(1).astype(np.int64)
        n1_all.append(n1)
        n2_all.append(n2)
        band_cross += int((mu1 & mu2).sum())
        reach = int((dmat[dmat < 255]).max()) if n else 0
        diam.append(reach)
        if n > 1:
            u = np.full(n, 1.0 / n)
            p1 = mu1 / np.maximum(n1, 1)[:, None]
            p2 = mu2 / np.maximum(n2, 1)[:, None]
            tv1.append(0.5 * np.abs(p1 - u).sum(1))
            tv2.append(0.5 * np.abs(p2 - u).sum(1))
        cover1.append(n1 / max(n - 1, 1))
        cover2.append(n2 / max(n - 1, 1))
    n1 = np.concatenate(n1_all) if n1_all else np.zeros(0, np.int64)
    n2 = np.concatenate(n2_all) if n2_all else np.zeros(0, np.int64)
    tv1 = np.concatenate(tv1) if tv1 else np.zeros(0)
    tv2 = np.concatenate(tv2) if tv2 else np.zeros(0)
    c1 = np.concatenate(cover1) if cover1 else np.zeros(0)
    c2 = np.concatenate(cover2) if cover2 else np.zeros(0)

    def dist_summary(a: np.ndarray) -> dict:
        a = np.asarray(a)
        return {
            "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "p05": float(np.percentile(a, 5)) if a.size else None,
            "p95": float(np.percentile(a, 95)) if a.size else None,
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None,
        }

    def frac(a, cond):
        return float(cond(a).mean()) if a.size else None

    corr = float(np.corrcoef(n1, n2)[0, 1]) if n1.size > 1 and n1.std() > 0 and n2.std() > 0 else None
    # jaccard between the two band supports per query, via union/intersection sizes
    inter = []
    for r in range(len(nb)):
        n = int(nb[r])
        if not n:
            continue
        dmat = row_matrix(store, r)
        m1, m2 = band_masks(dmat)
        inter.append((m1 & m2).sum(1))
    inter = np.concatenate(inter) if inter else np.zeros(0)
    return {
        "n_molecules": int(len(nb)),
        "n_bonds_total": int(nb.sum()),
        "bonds_per_molecule": dist_summary(nb),
        "n_t": {"t1": dist_summary(n1), "t2": dist_summary(n2)},
        "n_t_zero_fraction": {"t1": frac(n1, lambda a: a == 0), "t2": frac(n2, lambda a: a == 0)},
        "n_t_one_fraction": {"t1": frac(n1, lambda a: a == 1), "t2": frac(n2, lambda a: a == 1)},
        "n_t_ge2_fraction": {"t1": frac(n1, lambda a: a >= 2), "t2": frac(n2, lambda a: a >= 2)},
        "n_t_max": {"t1": int(n1.max()) if n1.size else 0, "t2": int(n2.max()) if n2.size else 0},
        "row_total_variation_from_uniform": {
            "t1": dist_summary(tv1),
            "t2": dist_summary(tv2),
            "definition": "0.5*||mu_t/n_t - 1/N_bonds||_1 per bond",
        },
        "coverage_fraction_of_other_bonds": {"t1": dist_summary(c1), "t2": dist_summary(c2)},
        "n_t_correlation": corr,
        "band_support_intersection_fraction": (
            float(inter.sum() / max(n1.sum() + n2.sum() - inter.sum(), 1)) if inter.size else None
        ),
        "self_pairs_in_bands": int(self_in),
        "self_pairs_in_both_bands": int(band_cross),
        "structural_invariants": invariants,
        "line_graph_diameter": dist_summary(np.asarray(diam)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep"])
    p.add_argument("--splits", default="train,valid")
    args = p.parse_args(argv)
    if args.command == "prep":
        S.CACHE.mkdir(parents=True, exist_ok=True)
        out = {}
        for split in [s for s in args.splits.split(",") if s]:
            if split == "test":
                raise SystemExit("official test is not part of this round")
            payload = build_split(split)
            save_split(payload, split)
            out[split] = stats(payload)
            out[split]["cache_sha256"] = S.relation_path(split).stat().st_size
            path = S.relation_path(split)
            import hashlib

            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[split]["cache_sha256"] = h.hexdigest()
            S.write_json(S.ANALYSIS / f"relation_stats_{split}.json", out[split])
            print(f"[scm-relations] {split}: {out[split]['n_bonds_total']} bonds, "
                  f"n1 mean {out[split]['n_t']['t1']['mean']:.3f}, "
                  f"n2 mean {out[split]['n_t']['t2']['mean']:.3f}")
        S.write_json(S.ANALYSIS / "relation_stats.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def make_split_collate(stores: dict, base_collate):
    """One collate serving both splits (routes on ``cache['split']``).

    ``candidate`` is only a label for the frozen P3N layout, exactly as in the
    historical DCC-L runner.
    """
    collators = {s: RelationCollator(st) for s, st in stores.items()}

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        split = str(cache.get("split", ""))
        if split not in collators:
            raise RuntimeError(f"no relation store for split {split!r}")
        batch = base_collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        return collators[split].add(batch, rows)

    return collate
