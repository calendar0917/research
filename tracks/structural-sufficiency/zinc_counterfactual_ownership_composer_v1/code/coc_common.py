"""Shared, label-access-guarded utilities for COC-v1.

The module deliberately imports the frozen DCC-L, DCB and SCSB helpers instead of
reimplementing the data split, support table or grouped metrics.  That keeps the
comparison to the historical Fit8k L checkpoints on exactly the same rows,
support definition, penalty mask and metric semantics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]
CFG_PATH = ROUND / "configs" / "coc_v1.json"
ANALYSIS = ROUND / "analysis"
CACHE_DIR = ROUND / "cache"
RUNS = ROUND / "runs"

_V1 = SUFF / "zinc_arch_discovery_v1" / "code"
_DCB = SUFF / "zinc_dccl_bottleneck_v1" / "code"
_DCC = SUFF / "zinc_dual_channel_composer_v1" / "code"
_SCSB = SUFF / "zinc_support_conditioned_backoff_v1" / "code"
_AUDIT = SUFF / "zinc_dcc_information_audit_v1" / "code"
for _p in (str(_V1), str(_DCB), str(_DCC), str(_SCSB), str(_AUDIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import audit_common as AC  # noqa: E402
import dcb_train as DB  # noqa: E402
import dcc_models as DM  # noqa: E402
import dcc_runner as DR  # noqa: E402
import scb_common as S  # noqa: E402

# The guard refuses any torch.load of the official valid/test cache and records
# every path that is actually loaded.  It is installed at import time so every
# COC script (checks, training, scoring) runs under the same barrier.
AC.install_load_guard()

EXPECTED_PARAMS = 494193
_CODE_PATHS = ("code", "configs", "PROTOCOL.md", "DESIGN.md")
READER_LAYOUT = {
    "sum_c_glob": (0, 96),
    "sum_a": (96, 192),
    "sum_a_sq": (192, 288),
    "global33": (288, 321),
    "sum_c_loc": (321, 417),
    "sum_c_loc_sq": (417, 513),
}


# --------------------------------------------------------------------------- io
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=AC._json_default)
                    + "\n", encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cfg() -> dict:
    return read_json(CFG_PATH)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ------------------------------------------------------------------------ data
def split_rows() -> dict:
    """Frozen BJOC Fit8k/Monitor1k/Locked1k split, with hash assertions."""
    rows = DB.split_rows()
    cfg = load_cfg()["data"]
    fit = np.asarray(rows["fit8k"], dtype=np.int64)
    monitor = np.asarray(rows["monitor"], dtype=np.int64)
    locked = np.asarray(rows["locked"], dtype=np.int64)
    payload = {
        "fit": fit,
        "monitor": monitor,
        "locked": locked,
        "hashes": {
            "fit": sha256_array(fit),
            "monitor": sha256_array(monitor),
            "locked": sha256_array(locked),
        },
        "split_sha256": sha256_file(REPO / "tracks/structure-contribution/results/"
                                    "bjoc_predictive_bottleneck_v1/split_a.json"),
    }
    assert len(fit) == cfg["fit_rows"] == 8000
    assert len(monitor) == cfg["monitor_rows"] == 1000
    assert len(locked) == cfg["locked_rows"] == 1000
    assert payload["hashes"]["fit"] == cfg["fit_rows_sha256"]
    assert payload["hashes"]["monitor"] == cfg["monitor_rows_sha256"]
    assert payload["hashes"]["locked"] == cfg["locked_rows_sha256"]
    assert payload["split_sha256"] == cfg["split_sha256"]
    assert not (set(fit.tolist()) & set(monitor.tolist()))
    assert not (set(fit.tolist()) & set(locked.tolist()))
    assert not (set(monitor.tolist()) & set(locked.tolist()))
    assert len(set(fit.tolist()) | set(monitor.tolist()) | set(locked.tolist())) == 10000
    return payload


def code_dirty() -> bool:
    """Dirty state of the frozen scientific surface, not regenerated analysis files.

    ``analysis/`` outputs are written by the scoring scripts themselves; their
    content is protected by explicit sha256 fields in the freeze.
    """
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "--", *_CODE_PATHS],
        text=True, stderr=subprocess.DEVNULL)
    return bool(out.strip())


def cache_tensor_sha256(cache: Mapping[str, Any]) -> str:
    """Hash tensor content only; torch zip metadata is not scientific content."""
    h = hashlib.sha256()
    for key in sorted(cache):
        if key == "meta":
            continue
        value = cache[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(key.encode())
            h.update(repr(value).encode())
    return h.hexdigest()


def load_train_cache() -> dict:
    cache = AC.load_train_cache()
    cfg = load_cfg()["data"]
    path = REPO / cfg["train_cache"]
    got = sha256_file(path)
    accepted = cfg["train_cache_sha256_accepted"]
    assert got in accepted, (got, accepted)
    tensor_sha = cache_tensor_sha256(cache)
    assert tensor_sha == cfg["train_cache_tensor_sha256"], (tensor_sha, cfg["train_cache_tensor_sha256"])
    assert cache.get("split") == "train"
    assert int(cache["n_molecules"]) == 10000
    return cache


def norm_from_train_json(info: Mapping[str, Any]) -> dict:
    norm = dict(info["norm"])
    for k, v in list(norm.items()):
        if isinstance(v, list):
            norm[k] = np.asarray(v, dtype=np.float32)
    for k in ("global_mean", "global_scale"):
        if k not in norm:
            raise KeyError(f"norm missing {k}")
    return norm


def fit_only_norm(train_cache: Mapping[str, Any], fit_rows: np.ndarray) -> dict:
    """Use the exact DCB Fit-only standardisation implementation."""
    norm = DB.fit_only_norm(train_cache, np.asarray(fit_rows, dtype=np.int64))
    for seed in (0, 1):
        frozen = norm_from_train_json(load_frozen_train_json(seed))
        for key in ("global_mean", "global_scale", "root_scalar_mean",
                    "root_scalar_scale", "block_scalar_mean", "block_scalar_scale"):
            a = np.asarray(norm[key], dtype=np.float64)
            b = np.asarray(frozen[key], dtype=np.float64)
            if not np.array_equal(a, b):
                raise RuntimeError(f"fit-only norm {key} differs from frozen L seed {seed}")
    return norm


def load_frozen_train_json(seed: int) -> dict:
    return read_json(ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" /
                     f"fit8k_s{int(seed)}" / "train.json")


def load_frozen_ckpt(seed: int, tag: str) -> dict:
    path = ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{int(seed)}" / f"{tag}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def load_frozen_l_model(seed: int, tag: str, device) -> tuple[torch.nn.Module, dict, Path]:
    info = load_frozen_train_json(seed)
    norm = norm_from_train_json(info)
    ck_path = ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{int(seed)}" / f"{tag}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = DM.DualChannel(norm)
    model.organized_local = True
    C.load_state(model, ck["state_dict"], device)
    model.to(device)
    model.eval()
    return model, ck, ck_path


# --------------------------------------------------------------------- forward
def _bond_offsets(cache: Mapping[str, Any], rows: Sequence[int]) -> np.ndarray:
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    return np.concatenate([[0], np.cumsum([int(bp[int(r) + 1] - bp[int(r)]) for r in rows])]
                          ).astype(np.int64)


def capture_coc(model: torch.nn.Module, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Run COC forward while returning every intermediate tensor."""
    a, b, w = model.encoders(batch)
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    x_null = torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    c_glob = model.group_composer(x_null)
    Wl, Vl, _ = model.local_grouping(w, batch, organized=bool(model.organized_local))
    x_real = torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    c_real = model.local_composer(x_real)
    c_abs = model.local_composer(x_null)
    c_loc = c_real - c_abs
    B = int(batch["n_graphs"])
    z_glob = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
    zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc)
    zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc * c_loc)
    a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
    a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
    g = (batch["global"] - model.global_mean) / model.global_scale
    z = torch.cat([z_glob, a_sum, a_sq, g, zl, zl2], dim=1)
    pred = model.reader(z).view(-1)
    return {
        "a": a, "b": b, "w": w,
        "Wg": Wg, "Vg": Vg, "Wl": Wl, "Vl": Vl, "n_b": n_b,
        "x_real": x_real, "x_null": x_null,
        "c_real": c_real, "c_abs": c_abs, "c_loc": c_loc,
        "c_glob": c_glob, "z_glob": z_glob, "zl": zl, "zl2": zl2,
        "a_sum": a_sum, "a_sq": a_sq, "g": g, "z": z, "pred": pred,
    }


@torch.no_grad()
def eval_rows(model: torch.nn.Module, cache: Mapping[str, Any], rows: np.ndarray, *,
              device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = T1._to_device(DR.collate_p3n(cache, chunk, "P3N", valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, y, float(np.mean(np.abs(pred.astype(np.float64) - y))) if len(y) else float("nan")


# ------------------------------------------------------------------- metrics
def mae(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    return S.mae(pred, y, mask)


def w20_t100(curve: Sequence[Mapping[str, Any]], key: str = "monitor_mae") -> dict:
    return S.w20_t100(curve, key=key)


def bootstrap_mean(x: np.ndarray, *, n_boot: int = 10000, seed: int = 20261029) -> dict:
    return S.bootstrap_mean(x, n_boot=n_boot, seed=seed)


def drop_top_k(delta: np.ndarray, k: int) -> dict:
    return S.drop_top_k(delta, k)


# -------------------------------------------------------------------- groups
def prepare_monitor_groups(cache: Mapping[str, Any], fit_rows: np.ndarray,
                           monitor_rows: np.ndarray) -> tuple[dict, dict]:
    """Fit8k-only support table and Monitor group masks.  No Locked labels read."""
    _, per_atom = S.build_signatures(cache)
    support = S.build_support(cache, fit_rows, per_atom)
    monitor = S.group_payload(cache, support, monitor_rows, "train")
    from collections import Counter
    tiers = Counter(monitor["tier"].tolist())
    assert dict(tiers) == S.MONITOR_TIER_COUNTS, (dict(tiers), S.MONITOR_TIER_COUNTS)
    return support, monitor


def prepare_locked_groups(cache: Mapping[str, Any], support: Mapping[str, Any],
                          locked_rows: np.ndarray) -> dict:
    """Only called by the one-shot Locked scorer after a Monitor PASS + freeze."""
    return S.group_payload(cache, support, locked_rows, "train")


def group_scores(pred: np.ndarray, group: Mapping[str, Any]) -> dict:
    y = group["y"]
    out = {"overall": {"n": int(len(y)), "mae": mae(pred, y)},
           "natural_overall": S.natural_overall(pred, group),
           "groups": {}}
    for name, mask in group["masks"].items():
        out["groups"][name] = {
            "n": int(np.asarray(mask, dtype=bool).sum()),
            "mae": mae(pred, y, mask),
        }
    return out


# -------------------------------------------------------------- diagnostics
def select_diagnostic_rows(fit_rows: np.ndarray, n: int = 2000, seed: int = 20261029) -> np.ndarray:
    fit = np.asarray(fit_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(fit, size=min(int(n), len(fit)), replace=False).astype(np.int64))


def _participation_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / float(x.shape[0] - 1)
    ev = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2.0), 0.0, None)
    total = float(ev.sum())
    if total <= 0:
        return 0.0
    return float(total * total / float(np.sum(ev * ev)))


@torch.no_grad()
def representation_capture(model: torch.nn.Module, cache: Mapping[str, Any],
                           rows: np.ndarray, *, device, batch_size: int = 128) -> dict:
    """Capture c_loc / c_abs / c_real for a fixed row list."""
    rows = np.asarray(rows, dtype=np.int64)
    chunks: dict[str, list[np.ndarray]] = {}
    offsets = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = T1._to_device(DR.collate_p3n(cache, chunk, "P3N", valid=True), device)
            cap = capture_coc(model, batch)
            for k in ("c_loc", "c_abs", "c_real", "c_glob"):
                chunks.setdefault(k, []).append(cap[k].detach().cpu().numpy().astype(np.float64))
            offsets.append(_bond_offsets(cache, chunk))
    out = {k: np.concatenate(v, axis=0) if v else np.zeros((0, 96)) for k, v in chunks.items()}
    out["bond_ptr"] = np.concatenate([[0], np.cumsum([
        int(x[-1] - x[0]) for x in offsets
    ])]).astype(np.int64) if offsets else np.zeros(1, dtype=np.int64)
    # The per-chunk offsets above are local; rebuild exact molecule offsets for rows.
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    counts = np.asarray([int(bp[int(r) + 1] - bp[int(r)]) for r in rows], dtype=np.int64)
    out["bond_ptr"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out["rows"] = rows
    return out


def difference_diagnostics(cap: Mapping[str, Any]) -> dict:
    """Descriptive diagnostics of the learned counterfactual difference.

    The input is the output of ``representation_capture``.  All reported values
    are diagnostics, not performance evidence.
    """
    c_loc = np.asarray(cap["c_loc"], dtype=np.float64)
    c_abs = np.asarray(cap["c_abs"], dtype=np.float64)
    c_real = np.asarray(cap["c_real"], dtype=np.float64)
    ptr = np.asarray(cap["bond_ptr"], dtype=np.int64)
    nb = int(c_loc.shape[0])
    dim = int(c_loc.shape[1]) if c_loc.ndim == 2 else 96
    if nb == 0:
        return {"n_bonds": 0, "n_molecules": 0, "dim": dim}
    rms_loc = np.sqrt(np.mean(c_loc * c_loc, axis=1))
    rms_abs = np.sqrt(np.mean(c_abs * c_abs, axis=1))
    rms_real = np.sqrt(np.mean(c_real * c_real, axis=1))
    eps = 1e-12
    ratio = rms_loc / np.maximum(rms_abs, eps)
    ratio_real = rms_loc / np.maximum(rms_real, eps)
    # within-molecule centring
    within_parts = []
    var_per_mol = []
    common_energy = 0.0
    total_energy = float(np.sum(c_loc * c_loc))
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        if e <= s:
            continue
        x = c_loc[s:e]
        mu = x.mean(axis=0, keepdims=True)
        w = x - mu
        within_parts.append(w)
        common_energy += float(e - s) * float(np.sum(mu * mu))
        if e - s >= 2:
            var_per_mol.append(float(np.mean(np.var(x, axis=0))))
    within = np.concatenate(within_parts, axis=0) if within_parts else np.zeros((0, dim))
    # same for c_abs, for a like-for-like PR comparison
    abs_within_parts = []
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        if e <= s:
            continue
        x = c_abs[s:e]
        abs_within_parts.append(x - x.mean(axis=0, keepdims=True))
    abs_within = (np.concatenate(abs_within_parts, axis=0) if abs_within_parts
                  else np.zeros((0, dim)))
    return {
        "n_bonds": nb,
        "n_molecules": int(len(ptr) - 1),
        "dim": dim,
        "median_bond_rms_c_loc": float(np.median(rms_loc)),
        "mean_bond_rms_c_loc": float(np.mean(rms_loc)),
        "median_bond_rms_c_abs": float(np.median(rms_abs)),
        "mean_bond_rms_c_abs": float(np.mean(rms_abs)),
        "median_bond_rms_c_real": float(np.median(rms_real)),
        "mean_bond_rms_c_real": float(np.mean(rms_real)),
        "median_rms_ratio_to_c_abs": float(np.median(ratio)),
        "mean_rms_ratio_to_c_abs": float(np.mean(ratio)),
        "median_rms_ratio_to_c_real": float(np.median(ratio_real)),
        "mean_rms_ratio_to_c_real": float(np.mean(ratio_real)),
        "nonzero_fraction_1e-8": float(np.mean(np.abs(c_loc) > 1e-8)),
        "nonzero_fraction_1e-6": float(np.mean(np.abs(c_loc) > 1e-6)),
        "max_abs_c_loc": float(np.max(np.abs(c_loc))),
        "mean_square_amplitude_c_loc": float(np.mean(np.sum(c_loc * c_loc, axis=1))),
        "mean_square_amplitude_c_abs": float(np.mean(np.sum(c_abs * c_abs, axis=1))),
        "within_molecule_variance_mean": float(np.mean(var_per_mol)) if var_per_mol else 0.0,
        "within_molecule_variance_median": float(np.median(var_per_mol)) if var_per_mol else 0.0,
        "n_molecules_with_ge2_bonds": int(len(var_per_mol)),
        "participation_ratio_c_loc_pooled": _participation_ratio(c_loc),
        "participation_ratio_c_loc_within": _participation_ratio(within),
        "participation_ratio_c_abs_within": _participation_ratio(abs_within),
        "participation_ratio_c_real_within": _participation_ratio(
            np.concatenate([c_real[int(ptr[i]):int(ptr[i + 1])]
                            - c_real[int(ptr[i]):int(ptr[i + 1])].mean(axis=0, keepdims=True)
                            for i in range(len(ptr) - 1)
                            if int(ptr[i + 1]) > int(ptr[i])], axis=0)
            if nb else np.zeros((0, dim))),
        "common_energy_fraction_c_loc": float(common_energy / total_energy) if total_energy > 0 else 0.0,
    }


def representation_gate(diag: Mapping[str, Any], cfg: Mapping[str, Any], *,
                        post_training: bool = False) -> dict:
    if post_training:
        req = cfg["cpu_gates"]["post_training_mechanism_alive"]
        checks = {
            "median_rms_ge": diag.get("median_bond_rms_c_loc", 0.0) >= req["median_bond_rms_c_loc_ge"],
            "nonzero_fraction_ge": diag.get("nonzero_fraction_1e-8", 0.0) >= req["nonzero_fraction_ge"],
            "ratio_ge": diag.get("median_rms_ratio_to_c_abs", 0.0) >= req["median_rms_ratio_to_c_abs_ge"],
        }
    else:
        bad = cfg["cpu_gates"]["representation_degenerate"]
        checks = {
            "median_rms_ok": diag.get("median_bond_rms_c_loc", 0.0) >= bad["median_bond_rms_c_loc_lt"],
            "nonzero_fraction_ok": diag.get("nonzero_fraction_1e-8", 0.0) >= bad["nonzero_fraction_lt"],
            "ratio_ok": diag.get("median_rms_ratio_to_c_abs", 0.0) >= bad["median_rms_ratio_to_c_abs_lt"],
        }
    return {"pass": bool(all(checks.values())), "checks": checks,
            "post_training": bool(post_training)}


# --------------------------------------------------------------- code audit
def assert_no_official_test_tokens(code_dir: Path | None = None) -> dict:
    """Static scan for official valid/test cache paths.

    Tokens are assembled at runtime so this scanner's own source does not create
    a false positive.
    """
    code_dir = HERE if code_dir is None else Path(code_dir)
    forbidden = [
        "ad_" + "valid.pt",
        "ad_" + "test.pt",
        'load_cache("' + 'valid")',
        "load_cache('" + "valid')",
        'load_cache("' + 'test")',
        "load_cache('" + "test')",
    ]
    hits = []
    files = sorted(code_dir.glob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append({"file": path.name, "token": token})
    return {
        "scanned": [p.name for p in files],
        "forbidden_hits": hits,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def loaded_paths_audit() -> dict:
    return {
        "loaded": AC.loaded_paths(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
