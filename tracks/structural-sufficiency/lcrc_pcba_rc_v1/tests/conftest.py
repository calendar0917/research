from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import rc_common as C  # noqa: E402,F401  (sets up the inherited code dirs)
import rc_dist as D  # noqa: E402
from feature_spec import lipo_v1_spec  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return lipo_v1_spec()


@pytest.fixture(scope="session")
def norm(spec):
    """Stand-in normaliser: the W2 *initialisation* is norm-independent.

    The recorded historical fingerprints include the norm buffers, so the
    hash-equality check against history is skipped for this fixture and covered
    by the real-task test instead.
    """
    return {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }


@pytest.fixture(scope="session")
def task():
    """The real frozen task; skipped when the PCBA cache is unavailable."""
    try:
        return D.load_task_gated()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PCBA cache unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def dist_cache():
    try:
        return D.load_cache()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distance cache unavailable: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# synthetic batches
# ---------------------------------------------------------------------------

import rc_checks as CHK  # noqa: E402

#: Edge lists (local atom ids) of the tiny molecules used by the tests.
TINY_GRAPHS = CHK.TINY_GRAPHS


def _local_edges(n_atoms_per, edges=None):
    """Per-molecule local edge lists for the default chain graphs."""
    out = []
    off = 0
    for n in n_atoms_per:
        if edges is None:
            out.append([(i, i + 1) for i in range(n - 1)])
        else:
            out.append(list(edges[len(out)]))
        off += n
    return out


def synthetic_batch(
    *,
    n_atoms_per=(5, 4),
    dtype=torch.float32,
    seed: int = 7,
    graphs=None,
    with_pairs: bool = True,
) -> dict:
    """A tiny but structurally valid batch, plus a valid distance-pair payload.

    ``graphs`` is an optional list of ``(n_atoms, [(u, v), ...])`` per molecule
    (local atom ids).  The packed pair list is derived from those edges with the
    frozen distance definition, so the batch is internally consistent.
    """
    gen = torch.Generator().manual_seed(int(seed))
    if graphs is not None:
        n_atoms_per = tuple(int(g[0]) for g in graphs)
    edges_local = _local_edges(n_atoms_per, None if graphs is None else [g[1] for g in graphs])
    n_atoms = int(sum(n_atoms_per))
    atom_batch = torch.repeat_interleave(torch.arange(len(n_atoms_per)), torch.tensor(n_atoms_per))

    bonds = []
    off = 0
    for g, n in enumerate(n_atoms_per):
        for u, v in edges_local[g]:
            bonds.append((off + int(u), off + int(v)))
        off += n
    bond_index = torch.tensor(bonds, dtype=torch.long).reshape(-1, 2)
    n_bonds = int(bond_index.shape[0])

    nbr_atom = bond_index.reshape(-1)
    nbr_code = torch.randint(0, 80, (nbr_atom.numel(),), generator=gen, dtype=torch.long)

    path_atom, path_end = [], []
    for g, n in enumerate(n_atoms_per):
        off = int(sum(n_atoms_per[:g]))
        for i in range(min(n - 1, 2)):
            path_atom.append(off + i)
            path_end.append(off + i + 1)
            path_atom.append(off + i + 1)
            path_end.append(off + i)
    path_atom_t = torch.tensor(path_atom, dtype=torch.long)
    path_end_t = torch.tensor(path_end, dtype=torch.long)
    n_path = int(path_atom_t.numel())

    batch = {
        "atom_feat": torch.randn(n_atoms, 51, generator=gen, dtype=dtype),
        "atom_scalar": torch.randn(n_atoms, 9, generator=gen, dtype=dtype),
        "atom_type": torch.randint(0, 16, (n_atoms,), generator=gen, dtype=torch.long),
        "bond_index": bond_index,
        "bond_feat": torch.randn(n_bonds, 13, generator=gen, dtype=dtype),
        "bond_type": torch.randint(1, 6, (n_bonds,), generator=gen, dtype=torch.long),
        "atom_batch": atom_batch,
        "bond_batch": atom_batch[bond_index[:, 0]],
        "atom_off": torch.tensor([int(sum(n_atoms_per[:g])) for g in range(len(n_atoms_per))], dtype=torch.long),
        "atom_counts": torch.tensor(n_atoms_per, dtype=torch.long),
        "global": torch.randn(len(n_atoms_per), 42, generator=gen, dtype=dtype),
        "n_graphs": len(n_atoms_per),
        "nbr_atom": nbr_atom,
        "nbr_code": nbr_code,
        "path_atom": path_atom_t,
        "path_end": path_end_t,
        "path_graph": atom_batch[path_atom_t],
        "path_cat": torch.randint(0, 400, (n_path,), generator=gen, dtype=torch.long),
        "path_end63": torch.randint(0, 80, (n_path,), generator=gen, dtype=torch.long),
        "path_closure": torch.randint(0, 2, (n_path,), generator=gen, dtype=torch.long),
    }
    if with_pairs:
        packed, counts = pair_payload(n_atoms_per, edges_local)
        batch["pair_bucket"] = torch.from_numpy(packed)
        batch["pair_counts"] = torch.from_numpy(counts)
    return batch


def pair_payload(n_atoms_per, edges_local) -> tuple[np.ndarray, np.ndarray]:
    """Packed bucket codes + per-molecule pair counts (frozen distance rule)."""
    parts = []
    counts = []
    for n, edges in zip(n_atoms_per, edges_local):
        codes = D.molecule_bucket_codes(int(n), np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        parts.append(codes)
        counts.append(int(codes.size))
    packed = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    return np.ascontiguousarray(packed, dtype=np.uint8), np.asarray(counts, dtype=np.int64)


def graph_batch(graph_names, *, seed: int = 11, dtype=torch.float32) -> dict:
    graphs = [TINY_GRAPHS[name] for name in graph_names]
    return synthetic_batch(graphs=graphs, seed=seed, dtype=dtype)


@pytest.fixture
def tiny_batch():
    return synthetic_batch()
