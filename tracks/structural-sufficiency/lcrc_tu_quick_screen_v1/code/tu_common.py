"""Shared constants, paths, frozen recipe and IO helpers.

Round: ``lcrc-tu-quick-screen-v1``.

This module only records what was frozen *before* the first formal run:

* four TUDataset datasets and their fixed 8:1:1 stratified split sizes;
* ``split_seed = 0`` / ``training_seed = 0``;
* the single fixed training recipe (AdamW, 200 full epochs, per-step cosine);
* the three models (``FULL``, ``C1``, ``GIN``) and the 12-cell run matrix;
* the 28-dim TU global field list and the 9 per-node scalar fields.

Nothing here is fitted.  No label or fitted statistic enters a feature spec.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-tu-quick-screen-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_SINGLE_SEED_RANDOM_811"

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
for _p in (str(PREP_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Root that holds the official PyG ``TUDataset`` directories.  The official
#: raw files live in ``<root>/<NAME>/<NAME>/raw`` (PyG 2.6 layout) or in
#: ``<root>/<NAME>/raw``; :func:`resolve_dataset_root` picks whichever exists.
TU_ROOT = Path(os.environ.get("LCX_TU_ROOT", str(REPO / "data" / "TUD")))

ANALYSIS = TRACK / "analysis"
CACHE = TRACK / "cache"
RUNS = TRACK / "runs"
CONFIGS = TRACK / "configs"
TESTS = TRACK / "tests"
DATA_MANIFEST = ANALYSIS / "data_manifest.json"
SPLIT_FILE = ANALYSIS / "splits.json"

#: Official data description / download URL recorded in the manifest.
TU_DATASETS_DOCS = "https://chrsmrrs.github.io/datasets/docs/datasets/"
TU_PYG_DOCS = (
    "https://pytorch-geometric.readthedocs.io/en/latest/generated/"
    "torch_geometric.datasets.TUDataset.html"
)
TU_ARCHIVE_ROOT = "https://www.chrsmrrs.com/graphkerneldatasets"

SPLIT_SEED = 0
TRAINING_SEED = 0

#: The only split allowed in this round.  Sizes are 8:1:1 of the official
#: graph counts; ``sklearn.model_selection.train_test_split`` is applied twice
#: with ``random_state=0`` and stratification on the graph label.
DATASETS: dict[str, dict[str, Any]] = {
    "NCI1": dict(n_graphs=4110, n_classes=2, n_train=3288, n_valid=411, n_test=411,
                 node_type="label"),
    "PROTEINS": dict(n_graphs=1113, n_classes=2, n_train=890, n_valid=111, n_test=112,
                     node_type="label"),
    "IMDB-BINARY": dict(n_graphs=1000, n_classes=2, n_train=800, n_valid=100, n_test=100,
                        node_type="degree"),
    "IMDB-MULTI": dict(n_graphs=1500, n_classes=3, n_train=1200, n_valid=150, n_test=150,
                       node_type="degree"),
}

MODEL_NAMES = ("FULL", "C1", "GIN")
MODEL_LABELS = {
    "FULL": "LCRC-FULL",
    "C1": "LCRC-C1 (no b_mean)",
    "GIN": "GIN+global",
}
#: The exactly 12 authorised formal runs.
RUN_MATRIX: tuple[tuple[str, str], ...] = tuple(
    (dataset, model) for dataset in DATASETS for model in MODEL_NAMES
)

#: Fixed model widths (identical for FULL and C1; input width comes from the
#: TU feature spec).
MODEL_CONFIG: dict[str, Any] = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
}

#: Fixed training recipe (single seed, no early stopping, no soup/EMA/SSL).
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "peak_lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-4,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "batch_size": 64,
    "epochs": 200,
    "scheduler": "per_step_cosine",
    "gradient_clip": 5.0,
    "dropout": 0.10,
    "dropout_position": "shared local representation h, one call per forward",
    "loss": "CrossEntropyLoss (unweighted, C-dim logits)",
    "selection": "max Valid accuracy; ties -> lower Valid CE; ties -> earlier epoch",
    "ema": False,
    "swa": False,
    "soup": False,
    "ssl": False,
    "augmentation": False,
    "early_stopping": False,
    "hyperparameter_search": False,
    "cross_validation": False,
    "test_policy": "terminal: one Test evaluation after checkpoint hash is saved",
}

#: Fixed GIN reference recipe.
GIN_CONFIG: dict[str, Any] = {
    "hidden": 64,
    "n_layers": 5,
    "mlp_hidden": 128,
    "dropout": 0.10,
    "eps_init": 0.0,
    "train_eps": True,
    "readout_in": 157,
    "readout_hidden": 64,
    "norm": "LayerNorm after each conv, then SiLU, dropout, residual add",
}

#: The 28 TU global fields, in order: the first 20 topological fields of the
#: frozen ``lipo_v1`` global definition (``log1p_n_atoms`` .. 
#: ``frac_deg3plus_junctions``), the six minimum-cycle-size fractions, the
#: fused-cycle atom fraction (``cycle_degree >= 3``) and the number of distinct
#: node types.  All chemistry-specific fields are removed.
GLOBAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("log1p_n_atoms", "log1p(number of nodes)"),
    ("log1p_n_bonds", "log1p(number of physical edges after cleaning)"),
    ("log1p_n_junctions", "log1p(number of unordered neighbour pairs at a centre)"),
    ("cyclomatic_number", "n_bonds - n_atoms + n_components (first Betti number)"),
    ("n_components", "number of connected components (isolated nodes included)"),
    ("ring_atom_fraction", "fraction of nodes on a cycle"),
    ("ring_bond_fraction", "fraction of edges on a cycle"),
    ("n_ring_systems", "number of biconnected blocks with at least three edges"),
    ("diameter_max_finite", "maximum finite shortest-path distance inside a component (0 for a single node)"),
    ("mean_finite_pair_distance", "mean finite unordered-pair distance; unreachable pairs excluded"),
    ("frac_unreachable_pairs", "unreachable unordered pairs / all unordered pairs"),
    ("frac_finite_pairs_ge3", "finite unordered pairs at distance >= 3 / all unordered pairs"),
    ("bonds_per_atom", "n_bonds / n_atoms"),
    ("log1p_n_ring_atoms", "log1p(number of nodes on a cycle)"),
    ("max_degree", "maximum node degree"),
    ("terminal_atom_fraction", "fraction of degree-1 nodes"),
    ("junction_per_bond", "n_junctions / n_bonds"),
    ("frac_junction_centers_ring", "fraction of junction centres on a cycle"),
    ("frac_deg2_junctions", "fraction of junctions whose centre has degree 2"),
    ("frac_deg3plus_junctions", "fraction of junctions whose centre has degree >= 3"),
    ("ring_size3_fraction", "cycle-node fraction with minimum cycle length 3"),
    ("ring_size4_fraction", "cycle-node fraction with minimum cycle length 4"),
    ("ring_size5_fraction", "cycle-node fraction with minimum cycle length 5"),
    ("ring_size6_fraction", "cycle-node fraction with minimum cycle length 6"),
    ("ring_size7_fraction", "cycle-node fraction with minimum cycle length 7"),
    ("ring_size8plus_fraction", "cycle-node fraction with minimum cycle length >= 8"),
    ("fused_ring_atom_fraction", "fraction of nodes with cycle_degree >= 3"),
    ("distinct_atom_types", "number of distinct node types present"),
)

GLOBAL_DIM = len(GLOBAL_FIELDS)

#: The 9 per-node topology scalars, in order.
SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("log1p_n_d2", "log1p(number of nodes at graph distance exactly 2)"),
    ("log1p_n_paths2", "log1p(number of ordered non-backtracking 2-paths rooted at the node)"),
    ("closure_ratio", "closure-count / path-count (0 when there is no rooted 2-path)"),
    ("log1p_degree", "log1p(degree)"),
    ("in_cycle", "node lies on at least one cycle"),
    ("log1p_min_cycle_size", "log1p(smallest cycle through the node); 0 if acyclic"),
    ("log1p_cycle_degree", "log1p(number of incident cycle edges)"),
    ("is_terminal", "degree == 1"),
    ("cycle_degree_ge3", "cycle_degree >= 3"),
)
SCALAR_DIM = len(SCALAR_FIELDS)

#: Edge-type vocabulary: every physical edge is ``EDGE``; ``UNK`` is reserved.
BOND_TYPE_NAMES = ("EDGE", "UNK")
N_BOND_TYPES = len(BOND_TYPE_NAMES)
BOND_UNK_ID = 2
EDGE_TYPE = 1

#: Degree buckets used as synthetic node types for the two IMDB datasets.
DEGREE_BUCKET_MAX = 16  # bucket 16 means ">= 16"
N_DEGREE_BUCKETS = DEGREE_BUCKET_MAX + 1  # 0..16


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return sha256_bytes(a.tobytes())


def index_hash(rows: Iterable[int]) -> str:
    return sha256_array(np.asarray(list(rows), dtype=np.int64))


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{ROUND_ID}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(REPO), capture_output=True, text=True, check=True
        )
    except Exception:  # pragma: no cover - git missing
        return None
    return out.stdout.strip()


def code_state() -> dict[str, Any]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dirty_files": [x for x in (_git("status", "--porcelain") or "").splitlines() if x],
    }


def environment() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        out["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        out["cuda_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        out["cudnn_version"] = torch.backends.cudnn.version()
    try:
        import torch_geometric

        out["torch_geometric"] = torch_geometric.__version__
    except Exception:  # pragma: no cover
        out["torch_geometric"] = None
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        out["sklearn"] = None
    try:
        import networkx

        out["networkx"] = networkx.__version__
    except Exception:  # pragma: no cover
        out["networkx"] = None
    return out


def seed_everything(seed: int = TRAINING_SEED) -> dict[str, Any]:
    """Seed every RNG used by the run and record the CUDA determinism settings."""
    import random

    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.empty_cache()
    return {
        "python_random_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "torch_cuda_seed_all": int(seed) if torch.cuda.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "use_deterministic_algorithms": True,
        "use_deterministic_algorithms_warn_only": True,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
    }


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "RESULT_LABEL",
    "CODE_DIR",
    "TRACK",
    "REPO",
    "PREP",
    "PREP_CODE",
    "TU_ROOT",
    "ANALYSIS",
    "CACHE",
    "RUNS",
    "CONFIGS",
    "TESTS",
    "DATA_MANIFEST",
    "SPLIT_FILE",
    "TU_DATASETS_DOCS",
    "TU_PYG_DOCS",
    "TU_ARCHIVE_ROOT",
    "SPLIT_SEED",
    "TRAINING_SEED",
    "DATASETS",
    "MODEL_NAMES",
    "MODEL_LABELS",
    "RUN_MATRIX",
    "MODEL_CONFIG",
    "RECIPE",
    "GIN_CONFIG",
    "GLOBAL_FIELDS",
    "GLOBAL_DIM",
    "SCALAR_FIELDS",
    "SCALAR_DIM",
    "BOND_TYPE_NAMES",
    "N_BOND_TYPES",
    "BOND_UNK_ID",
    "EDGE_TYPE",
    "DEGREE_BUCKET_MAX",
    "N_DEGREE_BUCKETS",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "sha256_array",
    "index_hash",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "log",
    "code_state",
    "environment",
    "seed_everything",
]
