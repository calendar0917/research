"""Shared plumbing for the composition-objective pilot.

The parent round (``task_response_composition_s0_v1``) and the CSOI pilot
(``contextual_structural_object_interaction_pilot_v1``) are imported read-only.
Nothing in this module writes to either round.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
PARENT_ROUND = ROUND_DIR.parent / "task_response_composition_s0_v1"
CSOI_ROUND = ROUND_DIR.parent / "contextual_structural_object_interaction_pilot_v1"
PARENT_CODE = PARENT_ROUND / "code"
CSOI_CODE = CSOI_ROUND / "code"

for _path in (str(PARENT_CODE), str(CSOI_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trc_common as PC        # noqa: E402
import trc_data as PD          # noqa: E402
import trc_pairs as PPAIR      # noqa: E402
import trc_model as PM         # noqa: E402
import trc_train as PTR        # noqa: E402
import trc_prepare as PP       # noqa: E402
import csoi_common as CC       # noqa: E402
import csoi_data as CD         # noqa: E402
import csoi_model as CM        # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "cpo_v1.json"
SPLITS = PC.SPLITS
N_BUCKETS = PC.N_BUCKETS

# re-export the path-independent helpers
rmse = PC.rmse
mae = PC.mae
read_json = PC.read_json
write_json = PC.write_json
write_csv = PC.write_csv
sha256_file = PC.sha256_file
target_scaler = PC.target_scaler
official_feature_dims = PC.official_feature_dims
set_deterministic = PC.set_deterministic
environment = PC.environment
git_commit = PC.git_commit
lr_at = PTR.lr_at


def load_config() -> dict[str, Any]:
    import json

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def results_root() -> Path:
    override = os.environ.get("CPO_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(run_id: str) -> Path:
    raw = Path(run_id)
    if raw.is_absolute():
        path = raw
    elif len(raw.parts) > 1:
        path = REPO_ROOT / raw
    else:
        path = results_root() / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                rec[key] = value
                continue
            try:
                rec[key] = float(value)
            except (TypeError, ValueError):
                rec[key] = value
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# data identity
# ---------------------------------------------------------------------------


def pack_candidates(cfg: Mapping[str, Any], dataset: str = "lipo") -> list[Path]:
    """Our own prepared copy first, then the two read-only parent copies."""
    return [
        ROUND_DIR / "data" / f"csoi_cache_{dataset}.npz",
        CSOI_ROUND / "data" / f"csoi_cache_{dataset}.npz",
        PARENT_ROUND / "data" / f"trc_cache_{dataset}.npz",
    ]


def pack_path(cfg: Mapping[str, Any], dataset: str = "lipo") -> Path:
    for path in pack_candidates(cfg, dataset):
        if path.is_file():
            return path
    raise FileNotFoundError(
        "no prepared structural cache found; looked for:\n  "
        + "\n  ".join(str(p) for p in pack_candidates(cfg, dataset)))


def load_pack(cfg: Mapping[str, Any], dataset: str = "lipo") -> dict[str, Any]:
    path = pack_path(cfg, dataset)
    pack = CD.load_pack(cfg, dataset, path=path)
    pack["cache_path_used"] = str(path)
    return pack


def raw_cache(cfg: Mapping[str, Any], dataset: str = "lipo") -> dict[str, Any]:
    return PD.load_raw_cache(cfg, dataset)


def parent_module_provenance() -> dict[str, str]:
    files = ("trc_common.py", "trc_data.py", "trc_prepare.py", "trc_pairs.py",
             "trc_model.py", "trc_train.py")
    return {name: sha256_file(PARENT_CODE / name) for name in files}


def csoi_module_provenance() -> dict[str, str]:
    files = ("csoi_common.py", "csoi_data.py", "csoi_model.py")
    return {name: sha256_file(CSOI_CODE / name) for name in files}


def check_signature() -> dict[str, Any]:
    return {
        "commit": git_commit(),
        "parent_modules": parent_module_provenance(),
        "csoi_modules": csoi_module_provenance(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_id": load_config()["protocol_id"],
    }


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def quantiles(values: np.ndarray, points: Sequence[float] = (50, 90, 95, 99)) -> dict[str, float]:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "max": float("nan")}
    out = {f"p{int(p)}": float(np.percentile(v, p)) for p in points}
    out["mean"] = float(np.mean(v))
    out["max"] = float(np.max(v))
    out["min"] = float(np.min(v))
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.size < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    return pearson(rx, ry)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    # average ties
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i + 1
        while j < sx.size and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """RMSE(a) - RMSE(b) over resampled Valid molecules; >0 means b better."""
    a = np.asarray(pred_a, np.float64)
    b = np.asarray(pred_b, np.float64)
    y = np.asarray(labels, np.float64)
    n = y.size
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    point = rmse(a, y) - rmse(b, y)
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot), np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        stats[i] = rmse(a[idx], y[idx]) - rmse(b[idx], y[idx])
    return {
        "delta": float(point),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "n": int(n),
        "reps": int(n_boot),
        "seed": int(seed),
        "train_seed_uncertainty_included": False,
    }


def pair_delta_stats(delta_true: np.ndarray, delta_pred: np.ndarray) -> dict[str, float]:
    dl = np.asarray(delta_true, np.float64)
    dp = np.asarray(delta_pred, np.float64)
    if dl.size == 0:
        return {"n": 0}
    err = dp - dl
    sign = np.sign(dl) * np.sign(dp)
    return {
        "n": int(dl.size),
        "delta_mae": float(np.mean(np.abs(err))),
        "delta_rmse": float(np.sqrt(np.mean(err ** 2))),
        "delta_pearson": pearson(dp, dl),
        "direction_accuracy": float(np.mean(sign > 0)),
        "delta_bias": float(np.mean(err)),
        "delta_true_std": float(np.std(dl)),
    }


def cluster_bootstrap_pair_stats(delta_true: np.ndarray, delta_pred: np.ndarray,
                                 clusters: np.ndarray, n_boot: int = 2000,
                                 seed: int = 0) -> dict[str, Any]:
    """Cluster bootstrap for pair-delta metrics.

    Resampling units are clusters (e.g. the Valid molecule of a crossing pair,
    or a pair component), not individual pairs.  This is deliberately
    conservative: crossing pairs sharing a Valid molecule are dependent.
    """
    dl = np.asarray(delta_true, np.float64)
    dp = np.asarray(delta_pred, np.float64)
    cl = np.asarray(clusters)
    if dl.size == 0:
        return {"n": 0, "clusters": 0}
    uniq = np.unique(cl)
    members = {c: np.nonzero(cl == c)[0] for c in uniq.tolist()}
    point = pair_delta_stats(dl, dp)
    rng = np.random.default_rng(int(seed))
    boots = {"delta_mae": [], "delta_rmse": [], "direction_accuracy": []}
    for _ in range(int(n_boot)):
        pick = uniq[rng.integers(0, uniq.size, uniq.size)].tolist()
        idx = np.concatenate([members[c] for c in pick])
        s = pair_delta_stats(dl[idx], dp[idx])
        for key in boots:
            boots[key].append(s.get(key, float("nan")))
    out = dict(point)
    out.update({
        "n_clusters": int(uniq.size),
        "cluster_unit": "provided cluster label",
        "cluster_bootstrap": {
            key: {"lo": float(np.nanpercentile(vals, 2.5)),
                  "hi": float(np.nanpercentile(vals, 97.5))}
            for key, vals in boots.items()},
        "reps": int(n_boot),
        "seed": int(seed),
    })
    return out


# ---------------------------------------------------------------------------
# model construction / loading (read-only checkpoints)
# ---------------------------------------------------------------------------


def build_model(family: str, cfg: Mapping[str, Any], pack: Mapping[str, Any],
                arm: str, apply_support: bool = True):
    atom_dims, bond_dims = official_feature_dims()
    if family == "parent":
        model = PM.ResponseModel(cfg, atom_dims, bond_dims, arm)
    elif family == "csoi":
        model = CM.CSOIModel(cfg, atom_dims, bond_dims, arm)
    else:
        raise ValueError(f"unknown family {family}")
    model.load_pack(pack)
    zeroed = {}
    if apply_support:
        raw = raw_cache(cfg, "lipo")
        support = PM.category_support(raw, np.asarray(pack["train_idx"], np.int64),
                                      atom_dims, bond_dims)
        zeroed = PM.apply_atom_bond_support(model, support[0], support[1])
    return model, zeroed


class CutAdapter:
    """Uniform cut-response interface for a parent or CSOI model."""

    def __init__(self, model, family: str, pack: Mapping[str, Any], device: str = "cpu") -> None:
        self.model = model
        self.family = family
        self.pack = pack
        self.device = device
        self._prepared = None

    def prepare(self):
        import torch

        self.model.eval()
        with torch.no_grad():
            if self.family == "parent":
                self._prepared = self.model.frame_phi()
            else:
                self._prepared = self.model.frame_embeddings()
        return self._prepared

    def cuts(self, rows: np.ndarray) -> dict[str, Any]:
        import torch

        if self._prepared is None:
            self.prepare()
        with torch.no_grad():
            idx = torch.as_tensor(np.asarray(rows, np.int64), device=self.device)
            if self.family == "parent":
                return self.model.molecule_cuts(idx, self._prepared)
            return self.model.molecule_cuts(idx, self._prepared)

    def direct_pair_delta(self, pair_idx: np.ndarray) -> np.ndarray:
        import torch

        if self._prepared is None:
            self.prepare()
        sel = np.asarray(pair_idx, np.int64)
        pairs = {
            "C": torch.as_tensor(np.asarray(self.pack["pair_C"], np.int64)[sel], device=self.device),
            "Sa": torch.as_tensor(np.asarray(self.pack["pair_Sa"], np.int64)[sel], device=self.device),
            "Sb": torch.as_tensor(np.asarray(self.pack["pair_Sb"], np.int64)[sel], device=self.device),
            "bond": torch.as_tensor(np.asarray(self.pack["pair_cut_bond_row"], np.int64)[sel],
                                    device=self.device),
        }
        with torch.no_grad():
            if self.family == "parent":
                out = self.model.pair_cut_delta(pairs, self._prepared)
            else:
                out = self.model.pair_cut_delta(pairs, self._prepared)
        return out.detach().double().cpu().numpy()


def load_checkpoint(family: str, cfg: Mapping[str, Any], pack: Mapping[str, Any],
                    arm: str, run_path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    ckpt = torch.load(Path(run_path) / "state_best.pt", map_location="cpu", weights_only=False)
    model, _zeroed = build_model(family, cfg, pack, arm, apply_support=False)
    model.load_state_dict(ckpt["state"])
    model = model.to(device).float().eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# evaluation of cut outputs
# ---------------------------------------------------------------------------


def evaluate_cut_rows(adapter: CutAdapter, pack: Mapping[str, Any], rows: np.ndarray,
                      y_std: np.ndarray, scaler: Mapping[str, float],
                      batch: int = 32, collect_parses: bool = False) -> dict[str, Any]:
    """Per-molecule decomposition + aggregate metrics for one split."""
    rows = np.asarray(rows, np.int64)
    y_all = np.asarray(y_std, np.float64)
    mol_parse_ptr = np.asarray(pack["mol_parse_ptr"], np.int64)
    parse_S = np.asarray(pack["parse_S"], np.int64)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    parse_kind = np.asarray(pack["parse_kind"], np.uint8)
    frag_natoms = np.asarray(pack["frag_natoms"], np.int64)
    sigma = float(scaler["std"])
    shift = float(scaler["mean"])

    fbar_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ncut_parts: list[np.ndarray] = []
    lcut_parts: list[np.ndarray] = []
    lgraph_parts: list[np.ndarray] = []
    var_parts: list[np.ndarray] = []
    span_parts: list[np.ndarray] = []
    mean_s_parts: list[np.ndarray] = []
    mean_c_parts: list[np.ndarray] = []
    mixed_parts: list[np.ndarray] = []
    parse_records: list[dict[str, np.ndarray]] = []

    for start in range(0, rows.size, int(batch)):
        part = rows[start:start + int(batch)]
        res = adapter.cuts(part)
        f = res["f_cut"].detach().double().cpu().numpy()
        owner = res["owner"].detach().cpu().numpy()
        counts = res["counts"].detach().cpu().numpy().astype(np.float64)
        n = part.size
        y = y_all[part]

        pidx_parts = [np.arange(mol_parse_ptr[r], mol_parse_ptr[r + 1], dtype=np.int64)
                      for r in part.tolist()]
        pidx = np.concatenate(pidx_parts) if pidx_parts else np.zeros(0, np.int64)
        s_size = frag_natoms[parse_S[pidx]].astype(np.float64)
        c_size = np.where(parse_kind[pidx] == 0,
                          frag_natoms[np.maximum(parse_C[pidx], 0)].astype(np.float64),
                          0.0)

        fbar = np.bincount(owner, weights=f, minlength=n) / np.maximum(counts, 1.0)
        l_cut = np.bincount(owner, weights=(f - y[owner]) ** 2, minlength=n) / np.maximum(counts, 1.0)
        l_graph = (fbar - y) ** 2
        var = np.bincount(owner, weights=(f - fbar[owner]) ** 2, minlength=n) / np.maximum(counts, 1.0)
        fmin = np.full(n, np.inf)
        fmax = np.full(n, -np.inf)
        np.minimum.at(fmin, owner, f)
        np.maximum.at(fmax, owner, f)
        pos = np.zeros(n, bool)
        neg = np.zeros(n, bool)
        np.logical_or.at(pos, owner, f > 0)
        np.logical_or.at(neg, owner, f < 0)

        fbar_parts.append(fbar)
        y_parts.append(y)
        ncut_parts.append(counts.astype(np.int64))
        lcut_parts.append(l_cut)
        lgraph_parts.append(l_graph)
        var_parts.append(var)
        span_parts.append(fmax - fmin)
        mean_s_parts.append(np.bincount(owner, weights=s_size, minlength=n) / np.maximum(counts, 1.0))
        mean_c_parts.append(np.bincount(owner, weights=c_size, minlength=n) / np.maximum(counts, 1.0))
        mixed_parts.append(pos & neg)

        if collect_parses:
            parse_records.append({
                "mol": np.repeat(part, counts.astype(np.int64)),
                "pidx": pidx,
                "f_cut": f,
                "y": y[owner],
                "cut_sq": (f - y[owner]) ** 2,
                "S_size": s_size,
                "C_size": c_size,
                "kind": parse_kind[pidx],
            })

    fbar = np.concatenate(fbar_parts) if fbar_parts else np.zeros(0)
    y = np.concatenate(y_parts) if y_parts else np.zeros(0)
    ncut = np.concatenate(ncut_parts).astype(np.int64) if ncut_parts else np.zeros(0, np.int64)
    l_cut = np.concatenate(lcut_parts) if lcut_parts else np.zeros(0)
    l_graph = np.concatenate(lgraph_parts) if lgraph_parts else np.zeros(0)
    var = np.concatenate(var_parts) if var_parts else np.zeros(0)
    span = np.concatenate(span_parts) if span_parts else np.zeros(0)
    mean_s = np.concatenate(mean_s_parts) if mean_s_parts else np.zeros(0)
    mean_c = np.concatenate(mean_c_parts) if mean_c_parts else np.zeros(0)
    mixed = np.concatenate(mixed_parts) if mixed_parts else np.zeros(0, bool)

    eps = 1e-12
    cancel = var / (l_graph + var + eps)
    identity = l_cut - l_graph - var
    out: dict[str, Any] = {
        "n_molecules": int(y.size),
        "n_cuts": int(ncut.sum()),
        "n_parses_mean": float(ncut.mean()) if ncut.size else 0.0,
        "cut_mse_std": float(np.mean(l_cut)) if l_cut.size else float("nan"),
        "graph_mse_std": float(np.mean(l_graph)) if l_graph.size else float("nan"),
        "cut_var_std": float(np.mean(var)) if var.size else float("nan"),
        "cut_mse": float(np.mean(l_cut) * sigma ** 2) if l_cut.size else float("nan"),
        "graph_mse": float(np.mean(l_graph) * sigma ** 2) if l_graph.size else float("nan"),
        "cut_var": float(np.mean(var) * sigma ** 2) if var.size else float("nan"),
        "molecule_rmse": float(math.sqrt(np.mean(l_graph)) * sigma) if l_graph.size else float("nan"),
        "molecule_mae": float(np.mean(np.abs(fbar - y)) * sigma) if y.size else float("nan"),
        "molecule_rmse_std": float(math.sqrt(np.mean(l_graph))) if l_graph.size else float("nan"),
        "molecule_mae_std": float(np.mean(np.abs(fbar - y))) if y.size else float("nan"),
        "cancel_share": float(np.mean(cancel)) if cancel.size else float("nan"),
        "cut_span_std": float(np.mean(span)) if span.size else float("nan"),
        "cut_span": float(np.mean(span) * sigma) if span.size else float("nan"),
        "mixed_sign_fraction": float(np.mean(mixed)) if mixed.size else float("nan"),
        "cut_pred_min_std": float(np.min(fbar)) if fbar.size else float("nan"),
        "cut_pred_max_std": float(np.max(fbar)) if fbar.size else float("nan"),
        "identity_max_abs": float(np.max(np.abs(identity))) if identity.size else 0.0,
        "identity_mean_abs": float(np.mean(np.abs(identity))) if identity.size else 0.0,
        "var_quantiles": quantiles(var),
        "lgraph_quantiles": quantiles(l_graph),
        "lcut_quantiles": quantiles(l_cut),
        "cut_count_quantiles": quantiles(ncut.astype(np.float64)),
        "span_quantiles": quantiles(span),
        "mean_S_size_quantiles": quantiles(mean_s),
        "mean_C_size_quantiles": quantiles(mean_c),
        "corr_var_vs_lgraph": pearson(var, l_graph),
        "spearman_var_vs_lgraph": spearman(var, l_graph),
        "corr_var_vs_abs_graph_err": pearson(var, np.abs(fbar - y)),
        "spearman_var_vs_abs_graph_err": spearman(var, np.abs(fbar - y)),
        "var_share_of_lcut": float(np.sum(var) / max(1e-12, np.sum(l_cut))),
        "graph_share_of_lcut": float(np.sum(l_graph) / max(1e-12, np.sum(l_cut))),
        "sigma": sigma,
        "shift": shift,
        "per_molecule": {
            "f_bar_std": fbar, "y_std": y, "n_cuts": ncut,
            "L_cut": l_cut, "L_graph": l_graph, "Var_cut": var,
            "cut_span": span, "cancel_share": cancel,
            "mean_S_size": mean_s, "mean_C_size": mean_c,
        },
    }
    if collect_parses:
        out["parses"] = {
            key: np.concatenate([rec[key] for rec in parse_records])
            for key in parse_records[0]
        } if parse_records else {}
    return out


def prediction_file(run_path: Path, kind: str = "best") -> Path:
    return Path(run_path) / f"predictions_{kind}.npz"


def load_predictions(run_path: Path, kind: str = "best") -> dict[str, np.ndarray]:
    """Load only Train/Valid arrays.

    Test keys may be present in a legacy npz.  They are enumerated (the zip
    directory) but their arrays are never indexed or converted; the returned
    dict contains only ``train_*`` / ``valid_*`` entries.
    """
    path = prediction_file(run_path, kind)
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        allowed = [k for k in keys if k.startswith(("train_", "valid_"))]
        return {k: np.asarray(npz[k]) for k in allowed}


def predictions_by_molecule(pack: Mapping[str, Any],
                            preds: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = np.asarray(pack["row_ids"], np.int64)
    idx_of = {int(r): i for i, r in enumerate(rows.tolist())}
    out = np.full(int(pack["n_molecules"]), np.nan, np.float64)
    for split in ("train", "valid"):
        rid = np.asarray(preds[f"{split}_row_id"], np.int64)
        val = np.asarray(preds[f"{split}_pred"], np.float64)
        for r, v in zip(rid.tolist(), val.tolist()):
            out[idx_of[int(r)]] = v
    return out