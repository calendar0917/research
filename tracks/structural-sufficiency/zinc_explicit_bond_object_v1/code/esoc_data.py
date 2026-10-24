"""ESOC-v1 static data: bond-centred typed induced objects.

One token per *undirected physical bond* ``b = {u,v}``.  For every bond the
local object is the full typed induced subgraph on

    P_b = {u, v} U N(u) U N(v)

with, for every node ``i`` in ``P_b``, its role ``(dist(i,u), dist(i,v))``
computed on the **original** graph (not on the induced subgraph), and for every
unordered pair ``i<j`` in ``P_b`` the real edge class
``0 = non-edge, 1/2/3 = single/double/triple``.

Everything cached here is a deterministic, label-blind function of the raw
atom/bond graph.  No label is read by the builder; labels are attached
separately (:func:`attach_labels`) and only for train/valid.  The official test
split is never built in this round.

Cache contents (per split)
--------------------------
``atom_type``    [A]      int8    raw atom-type index (0..20)
``bond_index``   [E,2]    int16   physical bonds, canonical u<v
``bond_type``    [E]      int8    1/2/3
``atom_count``   [M,21]   int16   atom-type histogram (readout)
``global33``     [M,33]   f32     frozen P3 label-free global topology vector
``loc_type``     [E,LMAX] int8    atom type of each local node (-1 pad)
``loc_role``     [E,LMAX] int8    role id = dist(i,u)*(cap+1)+dist(i,v) (-1 pad)
``loc_et``       [E,L,L]  int8    induced edge class (0 non-edge, 1..3 real)
``loc_mask``     [E,LMAX] bool    real local node
``loc_nnodes``   [E]      int8    |P_b|
``loc_nedges``   [E]      int8    number of real bonds inside P_b
``pad_index``    [M,EMAX] int32   global bond index per (molecule, slot), -1 pad
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as AD  # noqa: E402

EXP = _HERE.parent
CACHE = EXP / "cache"
ANALYSIS = EXP / "analysis"
RAW = ANALYSIS / "raw"   # generated per-run artefacts (gitignored)
CONFIG = EXP / "configs" / "esoc_v1.json"

VERSION = "esoc-v1"
EMAX = 41             # fixed bond-token slot count (train max; asserted >= valid)
LMAX = 8              # max |P_b| over train+valid (asserted at build time)
ROLE_CAP = 3          # distances 0..3 are separate ids, >3 collapsed
N_ROLE = (ROLE_CAP + 1) ** 2
N_ATOM = AD.N_ATOM_TYPES   # 21
N_BOND = AD.N_BOND_TYPES   # 3 (bond_type in {1,2,3})


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cache_path(split: str, variant: str | None = None) -> Path:
    if variant:
        return CACHE / f"esoc_{split}_{variant}.pt"
    return CACHE / f"esoc_{split}_v1.pt"


def _capped_distances(adj: list[list[int]], n: int, cap: int) -> np.ndarray:
    """All-pairs shortest-path distances clipped to ``cap+1`` (unreachable)."""
    inf = cap + 1
    d = np.full((n, n), inf, dtype=np.int16)
    for s in range(n):
        d[s, s] = 0
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        front = [s]
        for step in range(1, cap + 1):
            nxt = []
            for x in front:
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        d[s, y] = step
                        nxt.append(y)
            front = nxt
            if not front:
                break
    return d


def build_molecule_objects(atom_type: np.ndarray, bond_index: np.ndarray,
                           bond_type: np.ndarray, n_atoms: int) -> dict:
    """Local induced objects for one molecule (pure numpy, no labels).

    ``bond_index`` rows are the physical bonds; the stored order (u,v) vs (v,u)
    is irrelevant: roles are assigned from both root directions and the encoder
    averages the two, so a flip only permutes the two terms.
    """
    atom_type = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    bond_index = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    bond_type = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    n = int(n_atoms)
    e = int(bond_index.shape[0])
    if atom_type.shape[0] != n:
        raise ValueError("atom_type length != n_atoms")
    if bond_type.shape[0] != e:
        raise ValueError("bond_type length != n_bonds")

    adj = [set() for _ in range(n)]
    bond_map: dict[tuple[int, int], int] = {}
    for (u, v), t in zip(bond_index.tolist(), bond_type.tolist()):
        if u == v:
            raise ValueError("self-loop is not a physical bond")
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in bond_map:
            raise ValueError("duplicate physical bond")
        if t < 1 or t > N_BOND:
            raise ValueError(f"bond type {t} outside 1..{N_BOND}")
        bond_map[(a, b)] = int(t)
        adj[u].add(v)
        adj[v].add(u)

    dist = _capped_distances([sorted(a) for a in adj], n, ROLE_CAP)

    loc_type = np.full((e, LMAX), -1, dtype=np.int8)
    loc_role = np.full((e, LMAX), -1, dtype=np.int8)
    loc_et = np.zeros((e, LMAX, LMAX), dtype=np.int8)
    loc_mask = np.zeros((e, LMAX), dtype=bool)
    loc_nnodes = np.zeros(e, dtype=np.int16)
    loc_nedges = np.zeros(e, dtype=np.int16)

    for k, (u, v) in enumerate(zip(bond_index[:, 0].tolist(), bond_index[:, 1].tolist())):
        nodes = sorted({u, v} | adj[u] | adj[v])
        if len(nodes) > LMAX:
            raise ValueError(f"local object size {len(nodes)} > LMAX={LMAX}")
        m = len(nodes)
        loc_nnodes[k] = m
        for a, i in enumerate(nodes):
            loc_type[k, a] = np.int8(atom_type[i])
            ru = min(int(dist[u, i]), ROLE_CAP)
            rv = min(int(dist[v, i]), ROLE_CAP)
            loc_role[k, a] = np.int8(ru * (ROLE_CAP + 1) + rv)
            loc_mask[k, a] = True
        n_edges = 0
        for a in range(m):
            for b in range(a + 1, m):
                i, j = nodes[a], nodes[b]
                t = bond_map.get((i, j) if i < j else (j, i), 0)
                loc_et[k, a, b] = np.int8(t)
                loc_et[k, b, a] = np.int8(t)
                n_edges += int(t > 0)
        loc_nedges[k] = n_edges
    return {
        "loc_type": loc_type,
        "loc_role": loc_role,
        "loc_et": loc_et,
        "loc_mask": loc_mask,
        "loc_nnodes": loc_nnodes,
        "loc_nedges": loc_nedges,
    }


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------
def build_split(split: str) -> dict:
    if split not in ("train", "valid"):
        raise ValueError("ESOC-v1 builds train/valid only (official test is not loaded)")
    t0 = time.perf_counter()
    objs, obj_source = AD.load_p3_objects(split)
    n_mol = len(objs)
    if n_mol != AD.split_size(split):
        raise RuntimeError(f"{split}: {n_mol} molecules")

    atom_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    bond_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    atom_types, bond_indices, bond_types = [], [], []
    atom_counts, globals_ = [], []
    loc = {k: [] for k in ("loc_type", "loc_role", "loc_et", "loc_mask",
                            "loc_nnodes", "loc_nedges")}
    n_atoms_graph = np.zeros(n_mol, dtype=np.int16)
    n_bonds_graph = np.zeros(n_mol, dtype=np.int16)

    for i, obj in enumerate(objs):
        n = int(obj.n_atoms)
        at = AD._atom_types(obj)                     # argmax over the 21 one-hot slots
        bi = np.asarray(obj.bond_index, dtype=np.int64)
        bt = np.asarray(obj.bond_type, dtype=np.int64)
        if bt.size and (bt.min() < 1 or bt.max() > N_BOND):
            raise RuntimeError(f"{split}[{i}]: bond types {np.unique(bt)}")
        atom_ptr[i + 1] = atom_ptr[i] + n
        bond_ptr[i + 1] = bond_ptr[i] + len(bi)
        atom_types.append(at.astype(np.int8))
        bond_indices.append(bi.astype(np.int16))
        bond_types.append(bt.astype(np.int8))
        atom_counts.append(np.bincount(at, minlength=N_ATOM).astype(np.int16))
        globals_.append(np.asarray(obj.global_feat, dtype=np.float32))
        n_atoms_graph[i] = n
        n_bonds_graph[i] = len(bi)
        part = build_molecule_objects(at, bi, bt, n)
        for k in loc:
            loc[k].append(part[k])
        if (i + 1) % 2000 == 0:
            C.log(f"[esoc {split}] {i + 1}/{n_mol} {time.perf_counter() - t0:.1f}s")

    max_bonds = int(n_bonds_graph.max()) if n_mol else 0
    if max_bonds > EMAX:
        raise RuntimeError(f"{split}: {max_bonds} bonds exceeds the frozen slot width {EMAX}")
    pad_index = np.full((n_mol, EMAX), -1, dtype=np.int32)
    for i in range(n_mol):
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        pad_index[i, : b1 - b0] = np.arange(b0, b1, dtype=np.int32)

    data = {
        "version": VERSION,
        "split": split,
        "n_molecules": int(n_mol),
        "atom_ptr": atom_ptr,
        "atom_type": np.concatenate(atom_types).astype(np.int8),
        "bond_ptr": bond_ptr,
        "bond_index": np.concatenate(bond_indices, axis=0).astype(np.int16),
        "bond_type": np.concatenate(bond_types).astype(np.int8),
        "atom_count": np.stack(atom_counts, axis=0).astype(np.int16),
        "global33": np.stack(globals_, axis=0).astype(np.float32),
        "n_atoms_graph": n_atoms_graph,
        "n_bonds_graph": n_bonds_graph,
        "pad_index": pad_index,
        "max_bonds": max_bonds,
        "pad_width": int(EMAX),
        "max_local": int(LMAX),
        "role_cap": int(ROLE_CAP),
        **{k: np.concatenate(v, axis=0) for k, v in loc.items()},
        "meta": {
            "version": VERSION,
            "object_source": obj_source,
            "max_local_size": int(LMAX),
            "role_cap": int(ROLE_CAP),
            "role_vocab": int(N_ROLE),
            "atom_vocab": int(N_ATOM),
            "bond_vocab": int(N_BOND),
            "labels_read_by_builder": False,
            "official_test_loaded": False,
            "build_seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    if split in ("train", "valid"):
        data["labels"] = AD.official_labels(split).astype(np.float32)
        data["meta"]["labels_sha256"] = C.sha256_array(data["labels"])
    return data


def save_cache(data: dict, split: str, variant: str | None = None) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(split, variant)
    C.atomic_torch_save(dict(data), path)
    return path


def load_cache(split: str, variant: str | None = None) -> dict:
    path = cache_path(split, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"missing ESOC cache {path}; run `python esoc_data.py build --split {split}`"
        )
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("version") != VERSION:
        raise RuntimeError(f"{path}: version {d.get('version')} != {VERSION}")
    return d


def graph_scales(train: dict) -> dict:
    """Fixed train-only scalars used by the model (never per-graph normalisers)."""
    nb = np.asarray(train["n_bonds_graph"], dtype=np.float64)
    na = np.asarray(train["n_atoms_graph"], dtype=np.float64)
    nn = np.asarray(train["loc_nnodes"], dtype=np.float64)
    mol_of_bond = np.repeat(np.arange(len(nb), dtype=np.int64),
                            np.diff(np.asarray(train["bond_ptr"], dtype=np.int64)))
    mean_local = np.bincount(mol_of_bond, weights=nn, minlength=len(nb)) / np.maximum(nb, 1.0)
    counts = np.stack([
        np.log1p(na),
        np.log1p(nb),
        nb / np.maximum(na, 1.0),
        mean_local,
    ], axis=1)
    return {
        "max_atoms": int(np.asarray(train["n_atoms_graph"], dtype=np.int64).max()),
        "ref_bonds": float(nb.mean()),
        "ref_atoms": float(na.mean()),
        "count_mean": counts.mean(axis=0).astype(np.float32).tolist(),
        "count_scale": np.maximum(counts.std(axis=0), 1e-6).astype(np.float32).tolist(),
        "fit_split": "train",
        "valid_or_test_used": False,
    }


def attach_labels(data: dict) -> dict:
    """Attach labels to a *train/valid* cache built without them (rebuild path)."""
    split = data["split"]
    if split not in ("train", "valid"):
        raise ValueError("labels exist only for train/valid in this round")
    data = dict(data)
    data["labels"] = AD.official_labels(split).astype(np.float32)
    data["meta"] = dict(data.get("meta", {}))
    data["meta"]["labels_sha256"] = C.sha256_array(data["labels"])
    return data


def audit_cache(data: dict) -> dict:
    n_mol = int(data["n_molecules"])
    m = data
    checks = {
        "version": m["version"],
        "n_molecules": n_mol,
        "n_atoms": int(m["atom_type"].shape[0]),
        "n_bonds": int(m["bond_index"].shape[0]),
        "atom_ptr_last": int(m["atom_ptr"][-1]),
        "bond_ptr_last": int(m["bond_ptr"][-1]),
        "max_bonds": int(m["max_bonds"]),
        "max_local": int(m["max_local"]),
        "labels_finite": bool(np.isfinite(m["labels"]).all()),
        "global33_finite": bool(np.isfinite(m["global33"]).all()),
        "loc_mask_count": int(m["loc_mask"].sum()),
        "loc_type_range": [int(m["loc_type"][m["loc_mask"]].min()), int(m["loc_type"][m["loc_mask"]].max())],
        "loc_role_range": [int(m["loc_role"][m["loc_mask"]].min()), int(m["loc_role"][m["loc_mask"]].max())],
        "loc_et_unique": sorted(int(v) for v in np.unique(m["loc_et"])),
        "loc_nnodes_range": [int(m["loc_nnodes"].min()), int(m["loc_nnodes"].max())],
        "loc_nedges_range": [int(m["loc_nedges"].min()), int(m["loc_nedges"].max())],
        "label_mean": float(np.mean(m["labels"])),
        "label_std": float(np.std(m["labels"])),
    }
    if checks["atom_ptr_last"] != checks["n_atoms"]:
        raise RuntimeError("atom_ptr mismatch")
    if checks["bond_ptr_last"] != checks["n_bonds"]:
        raise RuntimeError("bond_ptr mismatch")
    if checks["loc_type_range"][0] < 0 or checks["loc_type_range"][1] >= N_ATOM:
        raise RuntimeError("atom type outside vocabulary")
    if checks["loc_role_range"][0] < 0 or checks["loc_role_range"][1] >= N_ROLE:
        raise RuntimeError("role outside vocabulary")
    return checks


# ---------------------------------------------------------------------------
# torch view for training / evaluation
# ---------------------------------------------------------------------------
class SplitTensors:
    """Whole split held on one device; batches are index gathers (no collate)."""

    def __init__(self, data: dict, device, with_labels: bool = True):
        def t(key, dtype):
            return torch.as_tensor(np.ascontiguousarray(data[key]), dtype=dtype, device=device)

        self.device = device
        self.n = int(data["n_molecules"])
        self.pad_index = torch.as_tensor(data["pad_index"], dtype=torch.long, device=device)
        self.loc_type = t("loc_type", torch.int16)
        self.loc_role = t("loc_role", torch.int16)
        self.loc_et = t("loc_et", torch.int8)
        self.loc_mask = t("loc_mask", torch.bool)
        self.loc_nnodes = t("loc_nnodes", torch.int16)
        self.loc_nedges = t("loc_nedges", torch.int16)
        self.bond_index = t("bond_index", torch.int16)
        self.n_atoms = t("n_atoms_graph", torch.long)
        self.atom_count = t("atom_count", torch.float32)
        self.global33 = t("global33", torch.float32)
        self.labels = t("labels", torch.float32) if with_labels else None

    def batch(self, rows: np.ndarray) -> dict:
        rows_t = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        idx = self.pad_index[rows_t]                      # [B, EMAX]
        bond_mask = idx >= 0
        gi = idx.clamp(min=0)
        out = {
            "loc_type": self.loc_type[gi].long(),
            "loc_role": self.loc_role[gi].long(),
            "loc_et": self.loc_et[gi].long(),
            "loc_mask": self.loc_mask[gi],
            "loc_nnodes": self.loc_nnodes[gi].float(),
            "loc_nedges": self.loc_nedges[gi].float(),
            "bond_mask": bond_mask,
            "bonds": self.bond_index[gi].long(),
            "n_atoms": self.n_atoms[rows_t],
            "atom_count": self.atom_count[rows_t],
            "global33": self.global33[rows_t],
        }
        if self.labels is not None:
            out["y"] = self.labels[rows_t]
        return out


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="ESOC-v1 static cache")
    p.add_argument("cmd", choices=["build", "audit"])
    p.add_argument("--split", default="train")
    args = p.parse_args(argv)
    if args.cmd == "build":
        data = build_split(args.split)
        path = save_cache(data, args.split)
        info = audit_cache(data)
        info["path"] = str(path.relative_to(_REPO))
        info["sha256"] = C.sha256_file(path)
        info["meta"] = data["meta"]
        print(json.dumps(info, indent=2))
        if args.split == "train":
            ANALYSIS.mkdir(parents=True, exist_ok=True)
            C.write_json(ANALYSIS / "scales.json", graph_scales(data))
        RAW.mkdir(parents=True, exist_ok=True)
        C.write_json(RAW / f"cache_{args.split}.json", info)
    else:
        print(json.dumps(audit_cache(load_cache(args.split)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def pairing_permutation(loc_et: np.ndarray, loc_mask: np.ndarray, rng) -> np.ndarray:
    """Marginal-preserving permutation of the pair-edge labels inside one object.

    The multiset of typed pair labels of the object is preserved exactly (so the
    object-level class-frequency vector, and hence the NULL arm, is unchanged);
    only the *correspondence* between the label and the atom pair moves.  Used by
    the frozen structure-sensitivity diagnostic.
    """
    out = np.array(loc_et, copy=True)
    n = int(loc_mask.sum())
    if n < 3:
        return out
    idx = np.flatnonzero(loc_mask)[:n]
    iu = np.triu_indices(n, k=1)
    pos = [(idx[a], idx[b]) for a, b in zip(*iu)]
    vals = np.asarray([out[a, b] for a, b in pos], dtype=out.dtype)
    perm = rng.permutation(len(vals))
    # reject the identity permutation: we want a genuinely different pairing
    if len(vals) > 1 and np.array_equal(perm, np.arange(len(vals))):
        perm = np.roll(perm, 1)
    for (a, b), v in zip(pos, vals[perm]):
        out[a, b] = v
        out[b, a] = v
    return out


def batch_from_specs(specs: list[dict], device="cpu") -> dict:
    """Pack raw molecule specs into the model's batch dict (checks / synthetics).

    Each spec: ``atom_type`` [n], ``bonds`` [e,2], ``bond_type`` [e],
    optional ``global33`` [33].  Used by the CPU gates only; training uses
    :class:`SplitTensors`.
    """
    import torch as _torch

    B = len(specs)
    parts = [build_molecule_objects(s["atom_type"], s["bonds"], s["bond_type"],
                                    len(np.asarray(s["atom_type"]).reshape(-1)))
             for s in specs]
    Emax = max(1, max(len(np.asarray(s["bonds"]).reshape(-1, 2)) for s in specs))
    L = LMAX
    loc_type = np.full((B, Emax, L), -1, dtype=np.int64)
    loc_role = np.full((B, Emax, L), -1, dtype=np.int64)
    loc_et = np.zeros((B, Emax, L, L), dtype=np.int64)
    loc_mask = np.zeros((B, Emax, L), dtype=bool)
    loc_nnodes = np.zeros((B, Emax), dtype=np.float32)
    loc_nedges = np.zeros((B, Emax), dtype=np.float32)
    bonds = np.zeros((B, Emax, 2), dtype=np.int64)
    bond_mask = np.zeros((B, Emax), dtype=bool)
    atom_count = np.zeros((B, N_ATOM), dtype=np.float32)
    g33 = np.zeros((B, 33), dtype=np.float32)
    n_atoms = np.zeros(B, dtype=np.int64)
    for i, (s, p) in enumerate(zip(specs, parts)):
        at = np.asarray(s["atom_type"], dtype=np.int64).reshape(-1)
        bi = np.asarray(s["bonds"], dtype=np.int64).reshape(-1, 2)
        e = len(bi)
        loc_type[i, :e] = p["loc_type"]
        loc_role[i, :e] = p["loc_role"]
        loc_et[i, :e] = p["loc_et"]
        loc_mask[i, :e] = p["loc_mask"]
        loc_nnodes[i, :e] = p["loc_nnodes"]
        loc_nedges[i, :e] = p["loc_nedges"]
        if e:
            bonds[i, :e, 0] = np.minimum(bi[:, 0], bi[:, 1])
            bonds[i, :e, 1] = np.maximum(bi[:, 0], bi[:, 1])
        bond_mask[i, :e] = True
        atom_count[i] = np.bincount(at, minlength=N_ATOM)[:N_ATOM]
        n_atoms[i] = len(at)
        if s.get("global33") is not None:
            g33[i] = np.asarray(s["global33"], dtype=np.float32).reshape(-1)[:33]

    def t(x, dt):
        return _torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)

    return {
        "loc_type": t(loc_type, _torch.long),
        "loc_role": t(loc_role, _torch.long),
        "loc_et": t(loc_et, _torch.long),
        "loc_mask": t(loc_mask, _torch.bool),
        "loc_nnodes": t(loc_nnodes, _torch.float32),
        "loc_nedges": t(loc_nedges, _torch.float32),
        "bond_mask": t(bond_mask, _torch.bool),
        "bonds": t(bonds, _torch.long),
        "n_atoms": t(n_atoms, _torch.long),
        "atom_count": t(atom_count, _torch.float32),
        "global33": t(g33, _torch.float32),
    }
