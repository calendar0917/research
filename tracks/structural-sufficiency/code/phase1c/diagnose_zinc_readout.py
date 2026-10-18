"""Phase 1c CPU diagnostic: ZINC residual/readout audit on frozen Phase 1b runs.

Scope
-----
* Uses only the frozen Phase 1b internal *Monitor* predictions and the
  Fit/Monitor label vault.  Reserved rows and official valid/test are never
  read or scored.
* Structural features (n_atoms, n_bonds, cyclomatic ring number, degree
  moments) are label-free and come from the frozen Phase 1b structure cache.
* The ridge readout probe is an exploratory *diagnostic*, not a model arm.  It
  fits on Fit only, selects vocabulary on Fit only, and evaluates on Monitor.
  It is not used to select any Phase 1c architecture after seeing GPU results.

Questions
---------
1. Do S0 / R1 / G_ref errors have a size-related residual (atoms, bonds,
   rings)?
2. Is the S0/R1 mean-only readout structurally unable to use graph size?
3. Does the patch representation already contain local degree/size
   information, and if so at which scale?
4. Do the training curves support "optimization-limited" or
   "representation/readout-limited"?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
sys.path.insert(0, str(PHASE1B))

from ssrc_data import (  # noqa: E402
    CACHE_DIR,
    LabelVault,
    load_cache,
    row_sha256,
)

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
OUT = REPO / "tracks/structural-sufficiency/results/phase1c/readout_diagnostics.json"
SPLIT = RESULTS_1B / "split_zinc_phase1b.json"
LABELS = RESULTS_1B / "zinc_labels_fit_monitor.json"
CACHE = CACHE_DIR / "zinc_phase1b.pt"

MODEL_FILES = {
    "S0_seed0": "S0_seed0.json",
    "S0_seed1": "S0_seed1.json",
    "R1_seed0": "R1_seed0.json",
    "R1_seed1": "R1_seed1.json",
    "M2_seed0": "M2_seed0.json",
    "G_ref_seed0": "G_ref_seed0.json",
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_structures(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    n_atoms = cache["mol_n_atoms"][rows].numpy().astype(np.float64)
    n_bonds = (cache["root_n_edges"][rows].numpy() // 2).astype(np.float64)
    rings = n_bonds - n_atoms + 1.0  # ZINC molecules are connected (checked in gates)
    # degree moments from the true root adjacency (label-free)
    src = cache["root_edge_src_flat"].numpy()
    dst = cache["root_edge_dst_flat"].numpy()
    starts = cache["root_edge_start"].numpy()
    deg_max = np.zeros(rows.size, dtype=np.float64)
    deg_mean = np.zeros(rows.size, dtype=np.float64)
    deg1 = np.zeros(rows.size, dtype=np.float64)
    for j, mol in enumerate(rows):
        a, b = int(starts[mol]), int(starts[mol + 1])
        u = src[a:b]
        v = dst[a:b]
        deg = np.bincount(np.concatenate([u, v]), minlength=int(n_atoms[j]))
        deg_max[j] = float(deg.max()) if deg.size else 0.0
        deg_mean[j] = float(deg.mean()) if deg.size else 0.0
        deg1[j] = float(np.mean(deg == 1)) if deg.size else 0.0
    return {
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "rings": rings,
        "deg_max": deg_max,
        "deg_mean": deg_mean,
        "frac_degree1": deg1,
    }


def _ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "beta": [float(b) for b in beta],
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n": int(y.size),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def _bin_stats(values: np.ndarray, y: np.ndarray, pred: np.ndarray, bins: Sequence[float]) -> list[dict[str, Any]]:
    edges = np.asarray(bins, dtype=np.float64)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if not mask.any():
            continue
        err = pred[mask] - y[mask]
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "mean_abs_err": float(np.mean(np.abs(err))),
            }
        )
    return out


def _quantile_bins(values: np.ndarray, n_bins: int = 5) -> list[float]:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = sorted(set(float(q) for q in qs))
    if len(edges) < 2:
        return [float(values.min()), float(values.max()) + 1.0]
    edges[0] = float(values.min()) - 1e-9
    edges[-1] = float(values.max()) + 1e-9
    return edges


def model_diagnostics(name: str, payload: Mapping[str, Any], struct: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(payload["monitor_pred_best"], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    err = pred - y
    abs_err = np.abs(err)
    n_atoms = struct["n_atoms"]
    out: dict[str, Any] = {
        "params": int(payload["parameters"]),
        "flops_forward": payload.get("flops", {}).get("flops_forward"),
        "best_epoch": int(payload["best_epoch"]),
        "epochs_run": len(payload.get("curves", [])),
        "monitor_mae": float(np.mean(abs_err)),
        "monitor_rmse": float(np.sqrt(np.mean(err**2))),
        "train_mae": float(payload["best_metrics"]["train"]["mae"]),
        "train_monitor_gap": float(payload["best_metrics"]["monitor"]["mae"] - payload["best_metrics"]["train"]["mae"]),
        "mean_signed_error": float(np.mean(err)),
        "corr_signed_error_n_atoms": _pearson(err, n_atoms),
        "corr_abs_error_n_atoms": _pearson(abs_err, n_atoms),
        "spearman_signed_error_n_atoms": _spearman(err, n_atoms),
        "spearman_abs_error_n_atoms": _spearman(abs_err, n_atoms),
        "ols_signed_error_on_n_atoms": _ols(err, np.stack([np.ones_like(n_atoms), n_atoms], axis=1)),
        "ols_abs_error_on_log_size": _ols(
            abs_err,
            np.stack([np.ones_like(n_atoms), np.log1p(n_atoms), struct["rings"]], axis=1),
        ),
        "ols_target_on_pred": _ols(y, np.stack([np.ones_like(y), pred], axis=1)),
        "bins": {},
        "curves": {
            "monitor_mae_best": float(payload["best_metrics"]["monitor"]["mae"]),
            "monitor_mae_last": float(payload["curves"][-1]["monitor_mae"]),
            "monitor_mae_best_epoch": int(payload["best_epoch"]),
            "monitor_mae_at_best_epoch": float(payload["curves"][int(payload["best_epoch"])]["monitor_mae"]),
            "train_mae_last": float(payload["curves"][-1].get("train_mae", float("nan"))),
            "train_loss_last": float(payload["curves"][-1].get("train_loss", float("nan"))),
            "best_epoch_at_budget_edge": bool(int(payload["best_epoch"]) >= len(payload["curves"]) - 2),
        },
    }
    for feature in ["n_atoms", "n_bonds", "rings", "deg_mean"]:
        vals = struct[feature]
        bins = _quantile_bins(vals, 5)
        out["bins"][feature] = {
            "edges": bins,
            "rows": _bin_stats(vals, y, pred, bins),
            "largest_minus_smallest_bias": None,
        }
        rows = out["bins"][feature]["rows"]
        if len(rows) >= 2:
            out["bins"][feature]["largest_minus_smallest_bias"] = float(rows[-1]["bias"] - rows[0]["bias"])
    return out


# ---------------------------------------------------------------------------
# patch-local information and mean-vs-sum readout probe
# ---------------------------------------------------------------------------


def patch_local_audit(cache: Mapping[str, Any], mol_rows: Sequence[int], max_mols: int = 300) -> dict[str, Any]:
    """Verify which local degree information is visible inside one E2 patch.

    A rooted radius-2 induced patch contains every edge incident to the root
    and to every distance-1 node (any neighbour of a distance-1 node has
    distance <= 2).  Distance-2 nodes have only the edges induced by the
    patch.
    """
    atom_dim = int(cache["meta"]["atom_dim"])
    node_start = cache["patch_node_start"].numpy()
    edge_start = cache["patch_edge_start"].numpy()
    psrc = cache["patch_edge_src"].numpy()
    pdst = cache["patch_edge_dst"].numpy()
    pfeat = cache["patch_node_feat_flat"].numpy()
    mol_patch_start = cache["mol_patch_start"].numpy()
    root_edges_src = cache["root_edge_src_flat"].numpy()
    root_edges_dst = cache["root_edge_dst_flat"].numpy()
    root_starts = cache["root_edge_start"].numpy()

    root_degree_visible = 0
    first_shell_degree_visible = 0
    first_shell_total = 0
    n_patches = 0
    patch_size_min, patch_size_max = 10**9, 0
    for mol in list(mol_rows)[:max_mols]:
        a, b = int(root_starts[mol]), int(root_starts[mol + 1])
        true_deg = np.bincount(
            np.concatenate([root_edges_src[a:b], root_edges_dst[a:b]]),
            minlength=int(cache["mol_n_atoms"][mol]),
        )
        p0, p1 = int(mol_patch_start[mol]), int(mol_patch_start[mol + 1])
        for p in range(p0, p1):
            ns, ne = int(node_start[p]), int(node_start[p + 1])
            es, ee = int(edge_start[p]), int(edge_start[p + 1])
            n_patches += 1
            patch_size_min = min(patch_size_min, ne - ns)
            patch_size_max = max(patch_size_max, ne - ns)
            # local degrees within the patch
            local_deg = np.bincount(
                np.concatenate([psrc[es:ee], pdst[es:ee]]), minlength=ne - ns
            )
            root_atom = int(cache["patch_root"][p])
            if local_deg[0] == int(true_deg[root_atom]):
                root_degree_visible += 1
            # distance-1 local nodes: node feature one-hot at atom_dim+2
            dist1 = np.flatnonzero(pfeat[ns:ne, atom_dim + 2] > 0.5)
            for li in dist1:
                # map local node to atom id: local node features do not carry
                # atom ids, but the patch is rooted and ordered; the full set
                # of distance-1 nodes is all true neighbours.  We verify the
                # total degree of the corresponding true atom by matching the
                # atom feature block and local degree instead of identity.
                first_shell_total += 1
                # local_deg includes all patch edges incident to this node,
                # and the patch contains all edges incident to distance-1
                # nodes; so local_deg is the true degree for this node.
                # We only assert it is finite/positive here; the structural
                # argument is recorded in the report.
                if local_deg[li] >= 1:
                    first_shell_degree_visible += 1
    return {
        "max_mols_checked": int(min(len(mol_rows), max_mols)),
        "n_patches_checked": int(n_patches),
        "root_degree_matches_true_degree_frac": float(root_degree_visible / max(n_patches, 1)),
        "first_shell_nodes_checked": int(first_shell_total),
        "first_shell_local_degree_positive_frac": float(first_shell_degree_visible / max(first_shell_total, 1)),
        "patch_node_count_min": int(patch_size_min),
        "patch_node_count_max": int(patch_size_max),
        "structural_argument": (
            "A rooted radius-2 induced patch contains every edge incident to the root "
            "and to every distance-1 node. Hence root and first-shell degrees are fully "
            "visible; distance-2 degrees are only partially visible. The number of patch "
            "tokens equals n_atoms, so the E2 multiset contains graph size, but mean "
            "pooling over tokens is invariant to token multiplicity and discards it."
        ),
    }


def _patch_key(patch_index: int, cache: Mapping[str, Any], atom_dim: int) -> str:
    """Coarse label-free local patch key (a coarsening of E2, for diagnosis only)."""
    ns, ne = int(cache["patch_node_start"][patch_index]), int(cache["patch_node_start"][patch_index + 1])
    es, ee = int(cache["patch_edge_start"][patch_index]), int(cache["patch_edge_start"][patch_index + 1])
    feats = cache["patch_node_feat_flat"][ns:ne].numpy()
    src = cache["patch_edge_src"][es:ee].numpy()
    dst = cache["patch_edge_dst"][es:ee].numpy()
    attr = cache["patch_edge_attr"][es:ee].numpy()
    root_type = int(np.argmax(feats[0, :atom_dim]))
    # incident edges (both directions stored); deduplicate by local neighbour
    nb: dict[int, tuple[int, ...]] = {}
    for k, (u, v) in enumerate(zip(src, dst)):
        if u == 0:
            nb[int(v)] = tuple(int(x) for x in attr[k])
        elif v == 0:
            nb[int(u)] = tuple(int(x) for x in attr[k])
    deg = len(nb)
    neigh = sorted((int(np.argmax(feats[j, :atom_dim])), a) for j, a in nb.items())
    # ring closure at the root: an edge between two distance-1 nodes
    dist1 = set(int(i) for i in np.flatnonzero(feats[:, atom_dim + 2] > 0.5))
    ring_at_root = any((int(u) in dist1 and int(v) in dist1) for u, v in zip(src, dst))
    # second-shell size (visible without identities)
    n_second = int(np.sum(feats[:, atom_dim + 3] > 0.5))
    return json.dumps([root_type, deg, neigh, bool(ring_at_root), n_second], separators=(",", ":"))


def readout_probe(
    cache: Mapping[str, Any],
    fit_rows: Sequence[int],
    monitor_rows: Sequence[int],
    y_fit: np.ndarray,
    y_monitor: np.ndarray,
    struct: Mapping[str, np.ndarray],
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge on coarse local-patch histograms: mean-only vs count/size-aware.

    This is an exploratory diagnostic, not a Phase 1c model arm.  Vocabulary and
    ridge coefficients are fitted on Fit only; Monitor is evaluation only.
    """
    atom_dim = int(cache["meta"]["atom_dim"])
    mol_patch_start = cache["mol_patch_start"].numpy()
    all_rows = [int(r) for r in fit_rows] + [int(r) for r in monitor_rows]
    keys_by_mol: list[Counter] = []
    for mol in all_rows:
        p0, p1 = int(mol_patch_start[mol]), int(mol_patch_start[mol + 1])
        keys_by_mol.append(Counter(_patch_key(p, cache, atom_dim) for p in range(p0, p1)))
    n_fit = len(fit_rows)
    vocab = sorted({k for c in keys_by_mol[:n_fit] for k in c})
    index = {k: i for i, k in enumerate(vocab)}
    K = len(vocab)

    def build(rows: list[Counter], add_oov: bool) -> sparse.csr_matrix:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for c in rows:
            for k, v in c.items():
                j = index.get(k)
                if j is None:
                    if add_oov:
                        j = K
                    else:
                        continue
                indices.append(j)
                data.append(float(v))
            indptr.append(len(indices))
        width = K + (1 if add_oov else 0)
        return sparse.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=(len(rows), width))

    fit_counts = build(keys_by_mol[:n_fit], add_oov=True)
    mon_counts = build(keys_by_mol[n_fit:], add_oov=True)
    fit_n = struct["n_atoms"][:n_fit]
    mon_n = struct["n_atoms"][n_fit:]
    size_fit = np.stack([fit_n / 10.0, struct["n_bonds"][:n_fit] / 10.0, struct["rings"][:n_fit]], axis=1)
    size_mon = np.stack([mon_n / 10.0, struct["n_bonds"][n_fit:] / 10.0, struct["rings"][n_fit:]], axis=1)
    # fractions: counts / n_atoms (mean pooling analogue)
    fit_frac = sparse.diags(1.0 / fit_n) @ fit_counts
    mon_frac = sparse.diags(1.0 / mon_n) @ mon_counts

    variants = {
        "mean_only_fraction": (fit_frac, mon_frac),
        "counts_sum": (fit_counts, mon_counts),
        "mean_plus_size": (
            sparse.hstack([fit_frac, sparse.csr_matrix(size_fit)]).tocsr(),
            sparse.hstack([mon_frac, sparse.csr_matrix(size_mon)]).tocsr(),
        ),
        "counts_plus_size": (
            sparse.hstack([fit_counts, sparse.csr_matrix(size_fit)]).tocsr(),
            sparse.hstack([mon_counts, sparse.csr_matrix(size_mon)]).tocsr(),
        ),
    }
    out: dict[str, Any] = {
        "vocab_size_fit": int(K),
        "oov_monitor_frac": float(mon_counts[:, K].nnz / len(y_monitor)),
        "alpha": float(alpha),
        "variants": {},
        "note": "Coarse local-patch key (root type, root degree, typed neighbour multiset, root ring closure, second-shell count); exploratory only.",
    }
    for name, (Xf, Xm) in variants.items():
        reg = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        reg.fit(Xf, y_fit)
        pred = reg.predict(Xm)
        err = pred - y_monitor
        out["variants"][name] = {
            "monitor_mae": float(np.mean(np.abs(err))),
            "monitor_rmse": float(np.sqrt(np.mean(err**2))),
            "corr_signed_error_n_atoms": _pearson(err, mon_n),
            "ols_signed_error_on_n_atoms": _ols(err, np.stack([np.ones_like(mon_n), mon_n], axis=1)),
        }
    return out


def main() -> int:
    split = json.loads(SPLIT.read_text())
    rows = {k: [int(x) for x in v] for k, v in split["rows"].items()}
    vault = LabelVault(LABELS, rows["fit"], rows["monitor"], rows["reserved"])
    vault.load()
    y_fit = vault.labels(rows["fit"], "train").numpy().astype(np.float64)
    y_monitor = vault.labels(rows["monitor"], "monitor").numpy().astype(np.float64)
    assert set(rows["fit"]) & set(rows["reserved"]) == set()
    assert set(rows["monitor"]) & set(rows["reserved"]) == set()

    cache = load_cache(CACHE)
    mon_struct = load_structures(cache, rows["monitor"])
    fit_struct = load_structures(cache, rows["fit"])

    payload: dict[str, Any] = {
        "protocol_id": "ssrc-phase1c-readout-diagnostic",
        "scope": {
            "labels_used": "Fit for ridge fit; Monitor for evaluation; Reserved never read",
            "official_valid_test_read": False,
            "penalized_logp_components_as_input": False,
        },
        "data": {
            "split_sha256": row_sha256(rows["monitor"]),
            "n_fit": len(rows["fit"]),
            "n_monitor": len(rows["monitor"]),
            "monitor_mean": float(np.mean(y_monitor)),
            "monitor_std": float(np.std(y_monitor)),
            "monitor_constant_mae": float(np.mean(np.abs(y_monitor - np.mean(y_fit)))),
        },
        "structure_summary": {
            k: {
                "min": float(v.min()),
                "max": float(v.max()),
                "mean": float(v.mean()),
                "q05": float(np.quantile(v, 0.05)),
                "q50": float(np.quantile(v, 0.50)),
                "q95": float(np.quantile(v, 0.95)),
            }
            for k, v in mon_struct.items()
        },
        "models": {},
    }
    for name, fname in MODEL_FILES.items():
        path = RUNS_1B / "zinc" / fname
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        assert [int(x) for x in run["monitor_mol_index"]] == rows["monitor"]
        payload["models"][name] = model_diagnostics(name, run, mon_struct, y_monitor)

    payload["patch_local_audit"] = patch_local_audit(cache, rows["monitor"], max_mols=300)
    payload["patch_local_audit"]["n_distinct_coarse_keys_fit"] = None  # filled by probe
    probe = readout_probe(cache, rows["fit"], rows["monitor"], y_fit, y_monitor, _concat_structs(fit_struct, mon_struct))
    payload["patch_local_audit"]["n_distinct_coarse_keys_fit"] = probe["vocab_size_fit"]
    payload["readout_probe"] = probe

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    print(f"== ZINC readout / scale diagnostic (Monitor n={len(y_monitor)}) ==")
    print(f"constant MAE {payload['data']['monitor_constant_mae']:.4f}")
    for name, m in payload["models"].items():
        print(
            f"{name:12s} MAE {m['monitor_mae']:.4f} RMSE {m['monitor_rmse']:.4f} "
            f"train {m['train_mae']:.4f} gap {m['train_monitor_gap']:+.4f} "
            f"corr(err,n) {m['corr_signed_error_n_atoms']:+.3f} "
            f"corr(|err|,n) {m['corr_abs_error_n_atoms']:+.3f} "
            f"bias(large-small) {m['bins']['n_atoms']['largest_minus_smallest_bias']:+.4f}"
        )
    print("\nby n_atoms (S0 / R1 / G_ref):")
    for i, row in enumerate(payload["models"]["S0_seed0"]["bins"]["n_atoms"]["rows"]):
        line = f"  n in [{row['lo']:.0f},{row['hi']:.0f}) n={row['n']:3d}"
        for name in ["S0_seed0", "R1_seed0", "G_ref_seed0"]:
            if name in payload["models"]:
                r = payload["models"][name]["bins"]["n_atoms"]["rows"][i]
                line += f" | {name.split('_')[0]} MAE {r['mae']:.3f} bias {r['bias']:+.3f}"
        print(line)
    print("\nreadout probe (Fit->Monitor, ridge alpha=1):")
    for name, v in payload["readout_probe"]["variants"].items():
        print(f"  {name:20s} MAE {v['monitor_mae']:.4f} corr(err,n) {v['corr_signed_error_n_atoms']:+.3f}")
    print(f"\nwrote {OUT}")
    return 0


def _concat_structs(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([a[k], b[k]]) for k in a}


if __name__ == "__main__":
    raise SystemExit(main())
