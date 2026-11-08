from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sp_common as C  # noqa: E402
from feature_spec import lipo_v1_spec  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return lipo_v1_spec()


@pytest.fixture(scope="session")
def norm(spec):
    """Stand-in normaliser (the *initialisation* is norm-independent, but the
    recorded historical fingerprints include the norm buffers, so tests that
    compare hashes against history use the real task fixture instead)."""
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
        return C.load_task_gated()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PCBA cache unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def has_test_cache():
    return (C.S3_TEST_CACHE_DIR / "meta.json").is_file()


def _as_long(values, dtype=torch.long):
    return torch.as_tensor(values, dtype=dtype)


def synthetic_batch(
    *,
    n_atoms_per=(5, 4),
    s3_paths_per_root=(0, 1, 2, 0, 3, 1, 2, 0, 1),
    dtype=torch.float32,
    seed: int = 7,
) -> dict:
    """A tiny but structurally valid batch (no data files needed).

    ``s3_paths_per_root`` assigns S3 paths to batch-local atom ids; every path
    is rooted at that atom and ends at a *different* atom of the same molecule.
    """
    gen = torch.Generator().manual_seed(int(seed))
    n_atoms = int(sum(n_atoms_per))
    atom_batch = torch.repeat_interleave(torch.arange(len(n_atoms_per)), torch.tensor(n_atoms_per))
    # one bond chain inside each molecule, plus one skip bond where possible
    bonds = []
    for g, n in enumerate(n_atoms_per):
        off = int(sum(n_atoms_per[:g]))
        for i in range(n - 1):
            bonds.append((off + i, off + i + 1))
        if n >= 4:
            bonds.append((off, off + 3))
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
    path_atom = torch.tensor(path_atom, dtype=torch.long)
    path_end = torch.tensor(path_end, dtype=torch.long)
    n_path = int(path_atom.numel())

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
        "path_atom": path_atom,
        "path_end": path_end,
        "path_graph": atom_batch[path_atom],
        "path_cat": torch.randint(0, 400, (n_path,), generator=gen, dtype=torch.long),
        "path_end63": torch.randint(0, 80, (n_path,), generator=gen, dtype=torch.long),
        "path_closure": torch.randint(0, 2, (n_path,), generator=gen, dtype=torch.long),
    }

    s3_root, s3_end, s3_m1, s3_m2, s3_b1, s3_b2, s3_b3, s3_d = [], [], [], [], [], [], [], []
    for atom, n_paths in enumerate(s3_paths_per_root):
        if atom >= n_atoms or n_paths <= 0:
            continue
        g = int(atom_batch[atom])
        off = int(sum(n_atoms_per[:g]))
        n = int(n_atoms_per[g])
        candidates = [off + j for j in range(n) if off + j != atom]
        if not candidates:
            continue
        for k in range(int(n_paths)):
            s3_root.append(atom)
            s3_end.append(candidates[k % len(candidates)])
            s3_m1.append(int(torch.randint(0, 16, (1,), generator=gen)))
            s3_m2.append(int(torch.randint(0, 16, (1,), generator=gen)))
            s3_b1.append(int(torch.randint(1, 6, (1,), generator=gen)))
            s3_b2.append(int(torch.randint(1, 6, (1,), generator=gen)))
            s3_b3.append(int(torch.randint(1, 6, (1,), generator=gen)))
            s3_d.append(int(torch.randint(1, 4, (1,), generator=gen)))
    batch.update(
        {
            "s3_path_atom": torch.tensor(s3_root, dtype=torch.long),
            "s3_path_end": torch.tensor(s3_end, dtype=torch.long),
            "s3_mid1_type": torch.tensor(s3_m1, dtype=torch.long),
            "s3_mid2_type": torch.tensor(s3_m2, dtype=torch.long),
            "s3_bond1": torch.tensor(s3_b1, dtype=torch.long),
            "s3_bond2": torch.tensor(s3_b2, dtype=torch.long),
            "s3_bond3": torch.tensor(s3_b3, dtype=torch.long),
            "s3_dist": torch.tensor(s3_d, dtype=torch.long),
        }
    )
    return batch


@pytest.fixture
def tiny_batch():
    return synthetic_batch()


@pytest.fixture
def lone_atom_batch():
    """One molecule whose only atom has no S3 path at all."""
    return synthetic_batch(n_atoms_per=(1,), s3_paths_per_root=())
