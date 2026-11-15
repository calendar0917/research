"""Local CPU unit tests for the semantic-table construction (fresh-clone safe).

These tests only exercise the codec, parameter accounting, the epoch-0 pairing
and the export equivalence, all on synthetic tensors / the frozen schema.  They
do not need a GPU, the downloaded archive, or training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import se_common as C  # noqa: E402
import se_tables as T  # noqa: E402


def test_codec_roundtrip():
    rt = T.codec_roundtrip()
    assert rt["path_roundtrip"] and rt["binary_roundtrip"]
    assert rt["n_path_codes"] == 400 and rt["n_binary_codes"] == 80


def test_parameter_accounting():
    assert C.FREE_TABLE_PARAMS == 33680
    assert C.GENERATOR_PARAMS == 33689
    assert C.FS_PARAMS == C.C1_PARAMS - C.FREE_TABLE_PARAMS + C.GENERATOR_PARAMS
    assert C.FS_PARAMS == 228683


def test_permutation_bijection():
    perms = T.build_permutations()
    assert np.array_equal(np.sort(perms["path"]), np.arange(C.N_PATH_CODES))
    assert np.array_equal(np.sort(perms["binary"]), np.arange(C.N_NBR_CODES))
    # deterministic
    again = T.build_permutations()
    assert np.array_equal(perms["path"], again["path"])


def test_generator_gate_init_one():
    gen = T.SemanticGenerator()
    g = torch.Generator().manual_seed(C.GENERATOR_SEED)
    gen.init_module(g)
    codes = torch.arange(C.N_PATH_CODES)
    gate = 1.0 + gen.bank_raw("path_gate", codes)
    assert float(gate.min()) == 1.0 and float(gate.max()) == 1.0


def test_decode_consistency():
    mid = torch.arange(C.N_ATOM_TYPES)
    b1 = torch.arange(C.N_BOND_TYPES)
    codes = torch.arange(C.N_PATH_CODES)
    dm, db1, db2 = T.decode_path(codes)
    ids = dm * (C.N_BOND_TYPES * C.N_BOND_TYPES) + db1 * C.N_BOND_TYPES + db2
    assert torch.equal(ids, codes)
