"""Bond-pair structural relations ``R`` for the relation-composition probe.

The relation is **not** a random walk on the line graph.  With the raw,
unweighted, rewired-free molecular graph adjacency ``A`` in hand::

    P          = diag(degree)^(-1) A          (row-stochastic, zero rows -> 0)
    M_{b,i}    = 1/sqrt(2)  if atom i is an endpoint of physical bond b else 0
    R_{b,c}[k] = (M P^k M^T)_{b,c},           k = 0..20 (k=0 is M M^T)

For two bonds ``b={u,v}`` and ``c={s,t}`` this is ``0.5 * sum_{a in {u,v}}
sum_{d in {s,t}} (P^k)_{a,d}``, i.e. the average over the four endpoint pairs
times two (the two incidence entries 1/sqrt(2) each contribute 1/2).

Structural properties, checked in ``rc_checks.py``:
  * ``R[0]`` diagonal = 1 (the two incidence entries of a bond are 1/2 each),
  * two different bonds sharing exactly one endpoint -> 0.5,
  * two bonds with no shared endpoint -> 0,
  * flipping the stored endpoint order (u,v) -> (v,u) leaves ``R`` unchanged,
  * permuting atoms / renumbering bonds only permutes the corresponding axes.

Everything here is label-free and computed on CPU ahead of training.  Only
train/valid are built; official test is never constructed in this round.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

KSTEPS = 21  # k = 0 .. 20


# ---------------------------------------------------------------------------
# relation construction (float64 reference, float32 cache)
# ---------------------------------------------------------------------------
def random_walk_powers(edge_index: np.ndarray, n: int, ksteps: int = KSTEPS) -> np.ndarray:
    """``P^0 .. P^(ksteps-1)`` as ``[n, n, ksteps]`` float64.

    ``edge_index`` is one row per undirected physical bond.  Parallel edges, if
    any, are still one adjacency entry (the graph is *unweighted*).
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("edge index out of range")
    a = np.zeros((n, n), dtype=np.float64)
    if len(edge_index):
        a[edge_index[:, 0], edge_index[:, 1]] = 1.0
        a[edge_index[:, 1], edge_index[:, 0]] = 1.0
        a = (a > 0).astype(np.float64)
    deg = a.sum(axis=1)
    p = np.zeros_like(a)
    nz = deg > 0
    p[nz] = a[nz] / deg[nz, None]
    out = np.empty((n, n, ksteps), dtype=np.float64)
    q = np.eye(n, dtype=np.float64)
    for k in range(ksteps):
        out[:, :, k] = q
        q = q @ p
    return out


def bond_relation(edge_index: np.ndarray, n: int, ksteps: int = KSTEPS) -> np.ndarray:
    """``R`` of shape ``[E, E, ksteps]`` (float64) from the raw bond list."""
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    powers = random_walk_powers(edge_index, n, ksteps)
    if len(edge_index) == 0:
        return np.zeros((0, 0, ksteps), dtype=np.float64)
    u, v = edge_index[:, 0], edge_index[:, 1]
    out = np.empty((len(edge_index), len(edge_index), ksteps), dtype=np.float64)
    for k in range(ksteps):
        q = powers[:, :, k]
        out[:, :, k] = 0.5 * (q[np.ix_(u, u)] + q[np.ix_(u, v)]
                              + q[np.ix_(v, u)] + q[np.ix_(v, v)])
    return out


def bond_relation_direct(edge_index: np.ndarray, n: int, ksteps: int = KSTEPS) -> np.ndarray:
    """Independent four-endpoint enumeration of the same quantity (float64).

    Written as an explicit loop over bond pairs so it does not share code with
    the indexed implementation; used only by the checks.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    powers = random_walk_powers(edge_index, n, ksteps)
    e = len(edge_index)
    out = np.zeros((e, e, ksteps), dtype=np.float64)
    for b in range(e):
        ends_b = [int(edge_index[b, 0]), int(edge_index[b, 1])]
        for c in range(e):
            ends_c = [int(edge_index[c, 0]), int(edge_index[c, 1])]
            for k in range(ksteps):
                acc = 0.0
                for x in ends_b:
                    for y in ends_c:
                        acc += 0.5 * powers[x, y, k]
                out[b, c, k] = acc
    return out


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
ANALYSIS = _HERE.parent / "analysis"
CACHE_DIR = _HERE.parent / "cache"


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"rc_R{split}_k{KSTEPS}.pt"


def _raw_edges(cache: Mapping[str, Any], row: int) -> tuple[np.ndarray, int]:
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    n = int(atom_ptr[row + 1] - atom_ptr[row])
    bi = np.asarray(cache["bond_index"])[int(bond_ptr[row]):int(bond_ptr[row + 1])]
    return bi.astype(np.int64), n


def build_split(split: str, *, force: bool = False) -> dict:
    """Build (or load) the per-molecule ``R`` cache for ``train`` / ``valid``."""
    path = cache_path(split)
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)
    payload = build_from_cache(D.load_cache(split), split)
    C.atomic_torch_save(payload, cache_path(split))
    C.log(f"[rc_R {split}] built {n_mol if False else payload['stats']['n_molecules']} molecules, "
          f"{cache_path(split).stat().st_size / 1e6:.1f} MB")
    return payload


def build_from_cache(ad: Mapping[str, Any], label: str) -> dict:
    """Build ``R`` for an arbitrary ad-cache layout (used by the checks)."""
    n_mol = int(ad["n_molecules"])
    bond_ptr = np.asarray(ad["bond_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(ad["atom_ptr"], dtype=np.int64)
    n_bonds = np.diff(bond_ptr).astype(np.int64)
    n_atoms = np.diff(atom_ptr).astype(np.int64)
    offset = np.zeros(n_mol + 1, dtype=np.int64)
    offset[1:] = np.cumsum(n_bonds * n_bonds)
    r = np.zeros((int(offset[-1]), KSTEPS), dtype=np.float32)
    bond_deg = np.zeros((int(bond_ptr[-1]),), dtype=np.float32)
    t0 = time.perf_counter()
    for row in range(n_mol):
        bi, n = _raw_edges(ad, row)
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        deg = np.zeros(n, dtype=np.int64)
        if len(bi):
            deg = np.bincount(bi.reshape(-1), minlength=n)
        bond_deg[b0:b1] = (deg[bi[:, 0]] + deg[bi[:, 1]] - 2).astype(np.float32)
        rel = bond_relation(bi, n, KSTEPS)
        r[offset[row]:offset[row + 1]] = rel.reshape(-1, KSTEPS).astype(np.float32)
        if (row + 1) % 2000 == 0:
            C.log(f"[rc_R {label}] {row + 1}/{n_mol} {time.perf_counter() - t0:.1f}s")
    stats = {
        "mean_bonds": float(n_bonds.mean()),
        "mean_atoms": float(n_atoms.mean()),
        "n_molecules": int(n_mol),
    }
    return {
        "split": label,
        "ksteps": KSTEPS,
        "R": torch.from_numpy(r),
        "offset": torch.from_numpy(offset),
        "n_bonds": torch.from_numpy(n_bonds),
        "n_atoms": torch.from_numpy(n_atoms),
        "bond_deg": torch.from_numpy(bond_deg),
        "bond_ptr": torch.from_numpy(bond_ptr),
        "atom_ptr": torch.from_numpy(atom_ptr),
        "stats": stats,
        "meta": {
            "definition": "R_bc[k] = (M P^k M^T)_bc, M endpoint 1/sqrt(2), "
                          "P = diag(deg)^-1 A, k=0..20",
            "dtype_math": "float64 build -> float32 cache",
            "label_free": True,
            "official_test_loaded": False,
            "build_seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }


def load_split(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing relation cache {path}; run rc_runner.py prep")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------
class RCCollator:
    """P3N batch layout plus a dense padded bond-pair relation block.

    The P3N collate is reused verbatim (same encoders, same token order).  The
    only additions are ``rel_R`` ``[B, Emax, Emax, K]``, ``bond_mask``
    ``[B, Emax]``, ``log_line_deg`` ``[B, Emax]``, ``bond_local`` and ``n_bonds``
    ``[B]``.  Padding contributes zeros, never a real graph's statistics.

    ``payloads`` is a mapping ``split -> relation payload``; the split is taken
    from ``cache["split"]`` so the same callable serves train and valid.
    """

    def __init__(self, payloads, *, ksteps: int = KSTEPS):
        self.ksteps = int(ksteps)
        if isinstance(payloads, dict) and "R" in payloads:
            payloads = {payloads.get("split", "train"): payloads}
        self.payloads = {}
        for split, cache in dict(payloads).items():
            if int(cache["ksteps"]) != self.ksteps:
                raise ValueError("cache ksteps mismatch")
            self.payloads[split] = dict(
                R=np.asarray(cache["R"]),
                offset=np.asarray(cache["offset"], dtype=np.int64),
                n_bonds=np.asarray(cache["n_bonds"], dtype=np.int64),
                bond_deg=np.asarray(cache["bond_deg"], dtype=np.float32),
                bond_ptr=np.asarray(cache["bond_ptr"], dtype=np.int64),
            )

    def _payload(self, cache):
        split = str(cache.get("split", "")) if isinstance(cache, dict) else ""
        if split in self.payloads:
            return self.payloads[split]
        if len(self.payloads) == 1:
            return next(iter(self.payloads.values()))
        raise KeyError(f"no relation payload for split {split!r}")

    def __call__(self, cache, rows: Sequence[int], candidate: str, *,
                 valid: bool = False, walk_cols=None) -> dict[str, Any]:
        p = self._payload(cache)
        rows = np.asarray(rows, dtype=np.int64)
        base = T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        b = len(rows)
        e_counts = p["n_bonds"][rows]
        ad_counts = np.diff(np.asarray(cache["bond_ptr"], dtype=np.int64))[rows]
        if not np.array_equal(e_counts, ad_counts):
            raise RuntimeError("relation cache bond counts disagree with the ad cache")
        e_max = int(max(1, e_counts.max()))
        rel = np.zeros((b, e_max, e_max, self.ksteps), dtype=np.float32)
        mask = np.zeros((b, e_max), dtype=bool)
        logdeg = np.zeros((b, e_max), dtype=np.float32)
        for i, row in enumerate(rows):
            e = int(e_counts[i])
            s, t = int(p["offset"][row]), int(p["offset"][row + 1])
            rel[i, :e, :e] = p["R"][s:t].reshape(e, e, self.ksteps)
            mask[i, :e] = True
            b0 = int(p["bond_ptr"][row])
            logdeg[i, :e] = np.log1p(p["bond_deg"][b0:b0 + e])
        base.update(
            {
                "rel_R": torch.from_numpy(rel),
                "bond_mask": torch.from_numpy(mask),
                "log_line_deg": torch.from_numpy(logdeg),
                "n_bonds": torch.from_numpy(e_counts.astype(np.float32)),
                # local bond index inside each molecule, needed to scatter the
                # flat composer outputs into the padded [B, Emax, *] layout
                "bond_local": torch.from_numpy(
                    np.concatenate([np.arange(e, dtype=np.int64) for e in e_counts])
                ),
            }
        )
        return base


def make_collator(splits: Sequence[str] = ("train", "valid")) -> RCCollator:
    return RCCollator({s: load_split(s) for s in splits})


def prep(splits: Sequence[str] = ("train", "valid"), *, force: bool = False) -> dict:
    out = {}
    for split in splits:
        payload = build_split(split, force=force)
        out[split] = {
            "path": str(cache_path(split)),
            "stats": payload["stats"],
            "meta": payload["meta"],
        }
    C.write_json(_HERE.parent / "analysis" / "relation_cache.json", out)
    return out


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "show"])
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prep":
        print(C.json.dumps(prep(args.splits, force=args.force), indent=2))
    else:
        for split in args.splits:
            payload = load_split(split)
            print(split, payload["stats"], payload["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
