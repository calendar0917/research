"""Phase 1e shared loaders: frozen predictions, structures, and the label side.

Scope discipline
----------------
* Reads only the frozen Phase 1b/1c/1d artifacts (run JSONs, structure-only
  feature table, split) and the official **train** split structures.
* Never reads the internal Reserved split's labels or official ZINC valid/test.
* The label side is *reconstructed* from the official train target plus the
  upstream GVAE formula (see ``notes/phase1e_protocol.md`` §1).  The
  reconstruction is a diagnostic, not a model input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
for _p in (HERE, PHASE1B):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1C = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1D = REPO / "tracks/structural-sufficiency/results/phase1d"
RESULTS_1E = REPO / "tracks/structural-sufficiency/results/phase1e"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
RUNS_1C = REPO / "tracks/structural-sufficiency/runs/phase1c"
RUNS_1D = REPO / "tracks/structural-sufficiency/runs/phase1d"

SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
FEATURES_PATH = RESULTS_1D / "structural_features.npz"
SMILES_TABLE = REPO / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ZINC_ROOT = REPO / "data/ZINC"
ZINC_SMI = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/upstream/zinc250k.smi"
SASCORER_SRC = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/upstream/sascorer_gvae.py"
FPSCORES = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/upstream/fpscores.pkl.gz"
KSVD_AUDIT = REPO / "tracks/ksvd/results/zinc_long_cycle_audit"

#: Frozen Monitor-scored arms.  ``G_ref`` is the Phase 1b reference; the rest are
#: the Phase 1c/1d candidates and controls.
RUN_FILES: dict[str, Path] = {
    "G_ref_seed0": RUNS_1B / "zinc/G_ref_seed0.json",
    "S0_seed0": RUNS_1B / "zinc/S0_seed0.json",
    "R1_seed0": RUNS_1B / "zinc/R1_seed0.json",
    "S_cap_size_seed0": RUNS_1C / "zinc/S_cap_size_seed0.json",
    "R1_size_seed0": RUNS_1C / "zinc/R1_size_seed0.json",
    "R1_size_seed1": RUNS_1C / "zinc/R1_size_seed1.json",
    "R1_marginal_size_seed0": RUNS_1C / "zinc/R1_marginal_size_seed0.json",
    "M_keep_edge_seed0": RUNS_1C / "zinc/M_keep_edge_seed0.json",
    "R_incidence_seed0": RUNS_1D / "zinc/R_incidence_seed0.json",
    "R1_size_cap_seed0": RUNS_1D / "zinc/R1_size_cap_seed0.json",
}

#: Forward FLOPs at the frozen batch-64 convention (from the frozen run records).
FORWARD_FLOPS: dict[str, float] = {
    "G_ref_seed0": 1_185_100_000.0,
    "S0_seed0": 441_539_200.0,
    "R1_seed0": 877_730_176.0,
    "S_cap_size_seed0": 441_539_200.0,
    "R1_size_seed0": 646_843_008.0,
    "R1_size_seed1": 646_843_008.0,
    "R1_marginal_size_seed0": 646_843_008.0,
    "M_keep_edge_seed0": 694_913_664.0,
    "R_incidence_seed0": 645_911_200.0,
    "R1_size_cap_seed0": 2_491_880_704.0,
}

PARAMS: dict[str, int] = {
    "G_ref_seed0": 439_559,
    "S0_seed0": 56_963,
    "R1_seed0": 94_659,
    "S_cap_size_seed0": 115_883,
    "R1_size_seed0": 115_977,
    "R1_size_seed1": 115_977,
    "R1_marginal_size_seed0": 115_977,
    "M_keep_edge_seed0": 132_681,
    "R_incidence_seed0": 115_952,
    "R1_size_cap_seed0": 453_129,
}

#: Community normalization constants for the GVAE penalized-logP formula
#: (JTNN-era snippet; verified over the 249,456-molecule pool by
#: ``tracks/ksvd/results/zinc_long_cycle_audit`` stage ``verify`` to <=1.5 %
#: relative error).  They are hard-coded because the audit artifact is a local
#: (git-ignored) CSV; §1 of the protocol records this provenance.
COMMUNITY_CONSTANTS: dict[str, float] = {
    "logP_mean": 2.4570953396190123,
    "logP_std": 1.4498737504491803,
    "SA_mean": -3.0536628609296634,
    "SA_std": 0.8315096218718826,
    "cycle_mean": -0.048569687328769765,
    "cycle_std": 0.2859726330705808,
}

#: Pre-registered bands copied from Phase 1b/1d (do not retune).
NEGLIGIBLE_BAND = 0.02
ZINC_MEANINGFUL_BAND = 0.02

#: Phase 1e A4/B3 gate thresholds, frozen in notes/phase1e_protocol.md §2/§3.
MAX_UNPENALISED_SHARE = 0.05
MIN_PENALISED_ERROR_SHARE = 0.25
MAX_UNSTABLE_FRACTION = 0.05
MAX_RECON_ERROR_FRACTION = 0.05
MIN_PROXY_MISS_FRACTION = 0.05


# ---------------------------------------------------------------------------
# hashing / io
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# frozen Monitor table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorTable:
    rows: np.ndarray  # [N] int64 molecule_index in the official train subset
    y: np.ndarray  # [N] float64 frozen target
    pred: dict[str, np.ndarray]  # model key -> [N] float64 prediction


def load_monitor() -> MonitorTable:
    split = read_json(SPLIT_PATH)
    rows = np.asarray(split["rows"]["monitor"], dtype=np.int64)
    pred: dict[str, np.ndarray] = {}
    y: np.ndarray | None = None
    for key, path in RUN_FILES.items():
        payload = read_json(path)
        idx = np.asarray(payload["monitor_mol_index"], dtype=np.int64)
        if not np.array_equal(idx, rows):
            raise SystemExit(f"monitor molecule order mismatch in {path.name}")
        if y is None:
            y = np.asarray(payload["monitor_y"], dtype=np.float64)
        elif np.abs(np.asarray(payload["monitor_y"], dtype=np.float64) - y).max() > 1e-9:
            raise SystemExit(f"monitor_y mismatch in {path.name}")
        pred[key] = np.asarray(payload["monitor_pred_best"], dtype=np.float64)
    if y is None:
        raise SystemExit("no frozen run found")
    return MonitorTable(rows=rows, y=y, pred=pred)


@dataclass(frozen=True)
class StructureFeatures:
    rows: np.ndarray
    values: dict[str, np.ndarray]


def load_features() -> StructureFeatures:
    with np.load(FEATURES_PATH) as z:
        rows = np.asarray(z["rows"], dtype=np.int64)
        values = {k: np.asarray(z[k]) for k in z.files if k != "rows"}
    return StructureFeatures(rows=rows, values=values)


def align_features(feat: StructureFeatures, rows: Sequence[int]) -> dict[str, np.ndarray]:
    pos = {int(r): i for i, r in enumerate(feat.rows)}
    idx = np.asarray([pos[int(r)] for r in rows], dtype=np.int64)
    return {k: v[idx] for k, v in feat.values.items()}


# ---------------------------------------------------------------------------
# ring statistics
# ---------------------------------------------------------------------------


def cycle_excess(cycles: Sequence[Sequence[int]]) -> int:
    """GVAE ``cycle_scores`` excess: ``max(0, max_cycle_len - 6)``."""
    if not cycles:
        return 0
    longest = max(len(c) for c in cycles)
    return 0 if longest <= 6 else int(longest - 6)


def pyg_graph(atom_count: int, bond_pairs: Sequence[tuple[int, int]]):
    """NetworkX graph whose node iteration order is ``0..n-1``."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(int(atom_count)))
    g.add_edges_from((int(u), int(v)) for u, v in bond_pairs)
    return g


def pyg_cycle_basis(graph: tuple[int, Sequence[tuple[int, int]]]):
    """``nx.cycle_basis`` on the frozen PyG graph in its stored node order."""
    import networkx as nx

    n, bonds = graph
    return nx.cycle_basis(pyg_graph(n, bonds))


def load_zinc_pool():
    """Official ZINC subset train: graphs + target, in subset row order.

    The pool pickle is only used for the target vector; bond topology comes from
    the processed PyG ``train.pt`` so that Phase 1e sees exactly the structures
    the models saw.
    """
    import torch
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(ZINC_ROOT), subset=True, split="train")
    graphs: list[tuple[int, list[tuple[int, int]]]] = []
    ys = []
    for data in ds:
        n = int(data.x.numel())
        ei = data.edge_index.numpy()
        pairs = set()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                pairs.add((min(u, v), max(u, v)))
        graphs.append((n, sorted(pairs)))
        ys.append(float(data.y.reshape(-1)[0]))
    return graphs, np.asarray(ys, dtype=np.float64)


def load_canonical_smiles() -> np.ndarray:
    if not SMILES_TABLE.exists():
        raise SystemExit(
            f"{SMILES_TABLE} is missing; it is a local derived artifact. "
            "Regenerate it with "
            "`uv run python tracks/ksvd/experiments/luyin16/zinc_cssd_basis_reuse_v1.py "
            "--write-smiles-table` before running Phase 1e."
        )
    with np.load(SMILES_TABLE, allow_pickle=True) as z:
        return np.asarray(z["smiles"], dtype=object)


def _sascorer_modules():
    """Load the upstream GVAE SA scorer without polluting ``sys.path``."""
    import gzip
    import importlib.util
    import pickle

    spec = importlib.util.spec_from_file_location("_p1e_sascorer", SASCORER_SRC)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {SASCORER_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Reproduce ``readFragmentScores('fpscores')`` with an explicit path instead
    # of relying on the module's own working directory.
    with gzip.open(FPSCORES, "rb") as fh:
        table = pickle.load(fh)
    scores: dict[str, float] = {}
    for entry in table:
        for fp in entry[1:]:
            scores[fp] = float(entry[0])
    mod._fscores = scores
    return mod


def reconstruct_label_components(smiles: Sequence[str]) -> dict[str, np.ndarray]:
    """Recompute the GVAE ``logP``/``SA``/``cycle`` terms for the train subset.

    The chemical terms use the *canonical SMILES* molecule (the same molecule the
    label generator canonicalised to).  ``cycle_excess`` here is computed in the
    canonical SMILES atom order, which is a *reconstruction* of the label's
    ordering, not necessarily identical to it: the generator used the RDKit
    canonical order of its own release.  The label side actually used is
    recovered in ``recover_label_cycle`` by snapping, not by trusting this.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdmolops
    import networkx as nx

    sascorer = _sascorer_modules()
    n = len(smiles)
    logp = np.zeros(n)
    sa = np.zeros(n)
    excess = np.zeros(n, dtype=np.int64)
    atom_counts = np.zeros(n, dtype=np.int64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise SystemExit(f"RDKit parse failure at row {i}")
        atom_counts[i] = mol.GetNumAtoms()
        logp[i] = Descriptors.MolLogP(mol)
        sa[i] = -float(sascorer.calculateScore(mol))
        cycles = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        excess[i] = cycle_excess(cycles)
    return {"logP": logp, "SA": sa, "canonical_excess": excess, "n_atoms": atom_counts}


def recover_label_cycle(
    y: np.ndarray,
    logp: np.ndarray,
    sa: np.ndarray,
    *,
    fit_scale: bool = True,
) -> dict[str, Any]:
    """Reverse-engineer the label's integer cycle excess from the target itself.

    The chemical part ``(logP-mu)/sigma + (SA-mu)/sigma`` is computed with the
    published community constants; the residual is a multiple of ``1/sigma_c``.
    Fitting ``(sigma_c, mu_c)`` by snapping the residual to the integer lattice
    recovers the per-molecule cycle excess the generator actually used,
    independently of any node-ordering assumption.
    """
    from scipy.optimize import minimize

    c = COMMUNITY_CONSTANTS
    y_chem = (logp - c["logP_mean"]) / c["logP_std"] + (sa - c["SA_mean"]) / c["SA_std"]
    eff_norm = y - y_chem
    if fit_scale:
        def loss(p: np.ndarray) -> float:
            scale, offset = float(p[0]), float(p[1])
            e = eff_norm * scale + offset
            return float(np.mean((e - np.clip(np.round(e), -20, 0)) ** 2))

        best = minimize(loss, np.array([0.288, -0.049]), method="Nelder-Mead")
        sigma_c, mu_c = float(best.x[0]), float(best.x[1])
    else:
        sigma_c, mu_c = c["cycle_std"], c["cycle_mean"]
    eff = eff_norm * sigma_c + mu_c
    snapped = np.clip(np.round(eff), -20, 0)
    cycle_component = (snapped - mu_c) / sigma_c
    return {
        "sigma_cycle": sigma_c,
        "mu_cycle": mu_c,
        "y_chem": y_chem,
        "effective_cycle": eff,
        "snap_error": eff - snapped,
        "excess": (-snapped).astype(np.int64),
        "cycle_component": cycle_component,
        "y_without_cycle": y - cycle_component,
    }


# ---------------------------------------------------------------------------
# frozen KSVD long-cycle audit tables (optional cross-check, local-only)
# ---------------------------------------------------------------------------


def load_ksvd_label_table() -> dict[int, dict[str, str]] | None:
    path = KSVD_AUDIT / "train_cycle_audit_label.csv"
    if not path.exists():
        return None
    out: dict[int, dict[str, str]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["subset_index"])] = row
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def paired_bootstrap_mean(
    values: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 20261020,
    ci: float = 95.0,
) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    means = vals[idx].mean(axis=1)
    lo = float(np.percentile(means, (100.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 - (100.0 - ci) / 2.0))
    return {"mean": float(vals.mean()), "ci95": [lo, hi], "n": int(vals.size)}


def group_stats(ae: np.ndarray, resid: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"n": 0}
    a = np.asarray(ae, dtype=np.float64)[m]
    r = np.asarray(resid, dtype=np.float64)[m]
    return {
        "n": int(m.sum()),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((a**2).mean())),
        "median_ae": float(np.median(a)),
        "p90_ae": float(np.percentile(a, 90)),
        "mean_signed_residual": float(r.mean()),
        "abs_error_sum": float(a.sum()),
    }


def summarize(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(e) for e in entries]
