from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import adapters as A  # noqa: E402
import cpu_gates as G  # noqa: E402
import lcrc_data as LD  # noqa: E402
import lcrc_model as LM  # noqa: E402
from feature_spec import legacy_zinc_spec, lipo_v1_spec  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return lipo_v1_spec()


@pytest.fixture(scope="session")
def legacy_spec():
    return legacy_zinc_spec()


@pytest.fixture(scope="session")
def raw_payload():
    return A.load_lipo_raw_artifact()


@pytest.fixture(scope="session")
def split():
    return A.load_lipo_frozen_split()


@pytest.fixture(scope="session")
def row_manifest():
    return A.load_lipo_row_manifest()


@pytest.fixture(scope="session")
def sample_indices(split, raw_payload):
    rows = raw_payload["rows"]
    fit = [int(x) for x in split["split"]["fit"]]
    n_atoms = np.asarray([len(rows[i]["atom_class"]) for i in fit])
    n_edges = np.asarray([len(rows[i]["edge_class"]) for i in fit])
    charged = np.asarray(
        [sum(1 for c in rows[i]["atom_charge"] if int(c) in (0, 1, 3, 4)) for i in fit]
    )
    picks = {
        fit[0],
        fit[1],
        fit[2],
        fit[int(np.argmax(n_atoms))],
        fit[int(np.argmax(n_edges))],
        fit[int(np.argmax(charged))],
    }
    rng = np.random.default_rng(20261030)
    for i in rng.choice(len(fit), size=min(10, len(fit)), replace=False):
        picks.add(fit[int(i)])
    return sorted(picks)


@pytest.fixture(scope="session")
def sample_graphs(sample_indices, raw_payload, spec):
    rows = raw_payload["rows"]
    return [A.lipo_row_to_graph(rows[i], i, spec) for i in sample_indices]


@pytest.fixture(scope="session")
def sample_cache(sample_graphs, spec):
    return LD.build_cache(sample_graphs, spec, split="fit_sample")


@pytest.fixture(scope="session")
def sample_norm(sample_cache, spec):
    return LD.fit_norm(sample_cache, spec)


@pytest.fixture(scope="session")
def sample_model(sample_cache, sample_norm, spec):
    model, _info = LM.build_model(spec, sample_norm, seed=0, device="cpu", load_ssl=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def synthetic_graphs(spec):
    return G.synthetic_cases(spec)


@pytest.fixture(scope="session")
def model_batch(sample_cache):
    return LD.collate(sample_cache, list(range(int(sample_cache["n_molecules"]))))
