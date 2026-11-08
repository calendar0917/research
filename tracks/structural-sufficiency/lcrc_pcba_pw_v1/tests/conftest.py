"""Shared fixtures: a tiny structurally valid batch and the real pair index.

The synthetic batch is a PW batch in the sense that it carries the inherited
W2 fields *and* the five pair tensors, rebuilt from its own ``bond_index`` so it
is internally consistent by construction.  The real pair-index fixture uses the
production cache when it exists and otherwise builds a small subset cache in a
temporary directory (a subset cache keeps the full-length row pointer, so it is
still indexed by official row id).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pw_common as C  # noqa: E402,F401  (sets up the inherited code dirs)
import pw_pairs as D  # noqa: E402
import pw_checks as CHK  # noqa: E402
from feature_spec import lipo_v1_spec  # noqa: E402

#: ``(n_atoms, [(u, v), ...])`` per tiny molecule.
TINY_GRAPHS: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "chain4_plus_chord": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    "chain5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    "ring6": (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
    "two_components": (6, [(0, 1), (1, 2), (3, 4)]),
    "single_atom": (1, []),
}


@pytest.fixture(scope="session")
def spec():
    return lipo_v1_spec()


@pytest.fixture(scope="session")
def norm(spec):
    """Stand-in normaliser: the W2 *initialisation* is norm-independent."""
    return {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }


@pytest.fixture(scope="session")
def task():
    try:
        return D.load_task_gated()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PCBA cache unavailable: {type(exc).__name__}: {exc}")


def _small_rows(task, n_atoms_max: int = 20) -> np.ndarray:
    import lr_common as LC

    n = np.diff(np.load(Path(LC.CACHE_PATH) / "atom_ptr.npy"))
    small = np.where((n >= 2) & (n <= n_atoms_max))[0]
    tiny = np.where(n < 2)[0]
    gen = np.random.default_rng(0)
    return np.sort(np.concatenate([gen.choice(small, 250, replace=False), tiny]))


@pytest.fixture(scope="session")
def pair_cache(task, tmp_path_factory):
    """The production pair index if present, else a deterministic subset cache."""
    if C.PAIR_CACHE_META.is_file():
        cache = D.load_cache(C.PAIR_CACHE_DIR)
        if not bool(cache["meta"].get("rows_subset")):
            return cache
    rows = _small_rows(task)
    out = tmp_path_factory.mktemp("pair_cache")
    D.build_cache(out, rows=rows, verify_fraction=0.05, rc_crosscheck_rows=100)
    return D.load_cache(out)


@pytest.fixture(scope="session")
def pair_rows(pair_cache) -> list[int]:
    counts = np.asarray(pair_cache["pair_counts"])
    return [int(r) for r in np.where(counts > 0)[0][:128]]


@pytest.fixture(scope="session")
def real_batch(task, pair_cache, pair_rows):
    return D.collate_pw(task, pair_cache, pair_rows)


@pytest.fixture(scope="session")
def no_pair_batch(task, pair_cache):
    counts = np.asarray(pair_cache["pair_counts"])
    rows = [int(r) for r in np.where(counts == 0)[0][:6]]
    if not rows:
        pytest.skip("pair cache has no molecule without a legal pair")
    return D.collate_pw(task, pair_cache, rows)


# ---------------------------------------------------------------------------
# synthetic batches (inherited W2 fields + the PW pair payload)
# ---------------------------------------------------------------------------


def synthetic_batch(
    *,
    graphs: Sequence[tuple[int, list[tuple[int, int]]]] | None = None,
    n_atoms_per: tuple[int, ...] = (5, 4),
    dtype=torch.float32,
    seed: int = 7,
) -> dict:
    gen = torch.Generator().manual_seed(int(seed))
    if graphs is not None:
        n_atoms_per = tuple(int(g[0]) for g in graphs)
        edges_local = [list(g[1]) for g in graphs]
    else:
        edges_local = [[(i, i + 1) for i in range(n - 1)] for n in n_atoms_per]
    n_atoms = int(sum(n_atoms_per))
    atom_batch = torch.repeat_interleave(torch.arange(len(n_atoms_per)), torch.tensor(n_atoms_per))

    bonds = []
    for g, n in enumerate(n_atoms_per):
        off = int(sum(n_atoms_per[:g]))
        for u, v in edges_local[g]:
            bonds.append((off + int(u), off + int(v)))
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
    return CHK.rebuild_pairs_from_bonds(batch)


@pytest.fixture
def tiny_batch():
    return synthetic_batch()


@pytest.fixture
def tiny_no_pair_batch():
    return synthetic_batch(graphs=[TINY_GRAPHS["single_atom"], TINY_GRAPHS["single_atom"]], seed=5)
