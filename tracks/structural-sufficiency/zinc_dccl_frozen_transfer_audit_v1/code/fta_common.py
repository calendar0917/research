"""Shared, label-disciplined machinery for the DCC-L frozen-transfer audit.

This round only *reads* the four frozen DCC-L final-update checkpoints of
``zinc_dccl_bottleneck_v1`` and trains small 513-dim readers on already
exported representation tensors.  No trunk weight is ever optimised.

Access discipline (enforced, not just documented):

* official **valid / test** caches are refused by :func:`load_train_cache`;
* the internal **Locked1k** rows are never returned by :func:`split_rows`;
* the probe split is computed from a **structure fingerprint only** (no labels);
* ``y`` is read only for probe-fit training, probe-selection scoring, and the
  single post-freeze Monitor diagnostic.

Layout of ``z`` (identical to ``zinc_dcc_information_audit_v1`` A1)::

    z = [ c_glob_sum 96 | atom_sum 96 | atom_sq 96 | global33 33
          | c_loc_sum 96 | c_loc_sq 96 ]   # 513
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]

ROUND_ID = "zinc-dccl-frozen-transfer-audit-v1"
CONFIG_PATH = ROUND / "configs" / "fta_v1.json"
ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
Z_DIR = RUNS / "z"
HEAD_DIR = RUNS / "heads"
for _p in (ANALYSIS, RUNS, Z_DIR, HEAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)

_V1 = SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_V1), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_runner as DR  # noqa: E402

BLOCKS = {
    "c_glob_sum": (0, 96),
    "atom_sum": (96, 192),
    "atom_sq": (192, 288),
    "global33": (288, 321),
    "c_loc_sum": (321, 417),
    "c_loc_sq": (417, 513),
}
PROBE_BLOCKS = {
    "atom_global": ((96, 192), (192, 288), (288, 321)),
    "local_global": ((288, 321), (321, 417), (417, 513)),
    "full": ((0, 513),),
}
CONDITIONS = (("fit4k", 0), ("fit4k", 1), ("fit8k", 0), ("fit8k", 1))


# --------------------------------------------------------------------------- io
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def log(msg: str) -> None:
    print(f"[{ROUND_ID} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_npz(path: Path | str, **arrays) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


def atomic_torch_save(obj, path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def ledger_append(entry: dict) -> None:
    with open(RUNS / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ data access
def load_train_cache() -> dict:
    """Load the official-train structural cache; refuse valid/test by name and split."""
    cfg = load_config()
    path = REPO / cfg["data"]["train_cache"]
    name = path.name
    if "valid" in name or "test" in name:
        raise RuntimeError(f"refusing non-train cache: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("split") != "train" or int(cache["n_molecules"]) != 10000:
        raise RuntimeError(f"unexpected cache split/n: {cache.get('split')} {cache.get('n_molecules')}")
    got = sha256_file(path)
    want = cfg["data"]["train_cache_sha256"]
    if got != want:
        raise RuntimeError(f"train cache sha256 mismatch: {got} != {want}")
    return cache


def row_hash(rows) -> str:
    return hashlib.sha256(np.asarray(sorted(int(r) for r in rows), dtype=np.int64).tobytes()).hexdigest()


def split_rows() -> dict:
    """Return fit4k / fit8k / monitor rows.  Locked rows are deliberately dropped."""
    cfg = load_config()
    split_path = REPO / cfg["data"]["split"]
    subset_path = REPO / cfg["data"]["fit4k_subset"]
    if sha256_file(split_path) != cfg["data"]["split_sha256"]:
        raise RuntimeError("split_a.json sha256 mismatch")
    if sha256_file(subset_path) != cfg["data"]["fit4k_subset_sha256"]:
        raise RuntimeError("fit4k_subset.json sha256 mismatch")
    split = json.loads(split_path.read_text())
    out = {
        "fit4k": np.asarray(json.loads(subset_path.read_text())["rows"], dtype=np.int64),
        "fit8k": np.asarray(split["fit_rows"], dtype=np.int64),
        "monitor": np.asarray(split["monitor_rows"], dtype=np.int64),
    }
    # the locked rows are known to exist but are not returned, so no downstream
    # code path can accidentally score them.
    locked = np.asarray(split["locked_rows"], dtype=np.int64)
    expect = cfg["data"]["row_hashes"]
    for key, rows in out.items():
        if row_hash(rows) != expect[key]:
            raise RuntimeError(f"{key} row hash mismatch")
    if row_hash(locked) != expect["locked"]:
        raise RuntimeError("locked row hash mismatch (file drift)")
    if len(set(out["fit4k"]) & set(out["monitor"])) or len(set(out["fit8k"]) & set(out["monitor"])):
        raise RuntimeError("monitor overlaps a fit set")
    if not set(out["fit4k"].tolist()) <= set(out["fit8k"].tolist()):
        raise RuntimeError("fit4k is not nested in fit8k")
    return out


# ----------------------------------------------------------------- structure
def molecule_graphs(cache: dict, rows=None):
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    idx = range(len(ap) - 1) if rows is None else np.asarray(rows, dtype=np.int64)
    out = []
    for i in idx:
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        bonds = [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e])) for e in range(b0, b1)]
        out.append(([int(x) for x in at[a0:a1]], bonds))
    return out


def structure_fingerprint(graph) -> str:
    """Permutation-invariant, label-free canonical string hash of one molecule."""
    atoms, bonds = graph
    at = sorted(int(x) for x in atoms)
    bl = sorted((min(u, v), max(u, v), int(t)) for u, v, t in bonds)
    payload = (
        f"{len(at)}|"
        + ",".join(map(str, at))
        + "|"
        + ",".join(f"{u}-{v}-{t}" for u, v, t in bl)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def probe_split(cache: dict, fit4k_rows) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic, label-free 3k / 1k split of Fit4k (whole fingerprint groups)."""
    cfg = load_config()["probe_split"]
    salt = cfg["salt"]
    groups: dict[str, list[int]] = {}
    for r in fit4k_rows:
        groups.setdefault(structure_fingerprint(molecule_graphs(cache, [r])[0]), []).append(int(r))
    order = sorted(groups.items(), key=lambda kv: hashlib.sha256(f"{salt}:{kv[0]}".encode()).hexdigest())
    target = int(cfg["n_probe_fit_target"])
    fit: list[int] = []
    for _, rows in order:
        if len(fit) + len(rows) > target:
            break
        fit.extend(rows)
    pf = np.asarray(sorted(fit), dtype=np.int64)
    ps = np.asarray(sorted(set(int(r) for r in fit4k_rows) - set(fit)), dtype=np.int64)
    info = {
        "rule": cfg["rule"],
        "salt": salt,
        "n_fingerprint_groups": len(groups),
        "n_groups_with_duplicates": int(sum(1 for v in groups.values() if len(v) > 1)),
        "max_group_size": int(max(len(v) for v in groups.values())),
        "n_probe_fit": int(len(pf)),
        "n_probe_sel": int(len(ps)),
        "probe_fit_sha256": row_hash(pf),
        "probe_sel_sha256": row_hash(ps),
    }
    if info["probe_fit_sha256"] != cfg["probe_fit_sha256"]:
        raise RuntimeError(f"probe_fit hash {info['probe_fit_sha256']} != frozen")
    if info["probe_sel_sha256"] != cfg["probe_sel_sha256"]:
        raise RuntimeError(f"probe_sel hash {info['probe_sel_sha256']} != frozen")
    return pf, ps, info


def radius2_signatures(graph) -> list[str]:
    atoms, bonds = graph
    nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
    for u, v, t in bonds:
        nbrs[u].append((t, v))
        nbrs[v].append((t, u))
    one = [
        f"r1({int(atoms[i])};" + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
        for i in range(len(atoms))
    ]
    return [
        f"r2({one[i]};" + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
        for i in range(len(atoms))
    ]


def support_frequency(graphs) -> dict[str, int]:
    freq: dict[str, int] = {}
    for g in graphs:
        for k in set(radius2_signatures(g)):
            freq[k] = freq.get(k, 0) + 1
    return freq


def support_tiers(graphs, freq: dict[str, int]) -> dict[str, np.ndarray]:
    """Per-molecule unseen / low4 / supported atom fractions (frozen round-0 rule)."""
    unseen, low4, supported, min_sup = [], [], [], []
    for g in graphs:
        sigs = radius2_signatures(g)
        c = [freq.get(s, 0) for s in sigs]
        if not c:
            unseen.append(0.0); low4.append(0.0); supported.append(0.0); min_sup.append(0)
            continue
        unseen.append(float(np.mean([x == 0 for x in c])))
        low4.append(float(np.mean([1 <= x <= 4 for x in c])))
        supported.append(float(np.mean([x >= 5 for x in c])))
        min_sup.append(int(min(c)))
    return {
        "unseen_frac": np.asarray(unseen),
        "low4_frac": np.asarray(low4),
        "supported_frac": np.asarray(supported),
        "min_support": np.asarray(min_sup),
    }


def ring_stats(graph) -> dict:
    import networkx as nx

    atoms, bonds = graph
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    for u, v, _ in bonds:
        g.add_edge(u, v)
    cycles = nx.cycle_basis(g)
    sizes = sorted(len(c) for c in cycles)
    longest = int(sizes[-1]) if sizes else 0
    return {
        "n_atoms": int(len(atoms)),
        "n_bonds": int(len(bonds)),
        "n_components": int(nx.number_connected_components(g)),
        "cycle_rank": int(len(bonds) - len(atoms) + nx.number_connected_components(g)),
        "n_rings": int(len(cycles)),
        "max_ring_size": longest,
        "cycle_excess": 0 if longest <= 6 else int(longest - 6),
        "ring_sizes": sizes,
    }


# ---------------------------------------------------------------------- trunk
def trunk_paths(fit: str, seed: int) -> tuple[Path, Path]:
    cfg = load_config()
    root = REPO / cfg["trunks"]["root"]
    return root / f"{fit}_s{seed}" / cfg["trunks"]["checkpoint"], root / f"{fit}_s{seed}" / "train.json"


def build_trunk(fit: str, seed: int, device=None):
    """Rebuild the frozen DCC-L trunk and load its final-update state dict."""
    device = device or torch.device("cpu")
    ckpt_path, info_path = trunk_paths(fit, seed)
    want = {c["id"]: c["sha256"] for c in load_config()["trunks"]["candidates"]}[f"{fit}_s{seed}"]
    got = sha256_file(ckpt_path)
    if got != want:
        raise RuntimeError(f"{fit}_s{seed} checkpoint sha256 {got} != frozen {want}")
    info = json.loads(info_path.read_text())
    norm = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in info["norm"].items()}
    model, model_info = DR.make_builder("L")("P3N", norm, seed, device)
    if C.parameter_count(model) != int(load_config()["trunks"]["parameters"]):
        raise RuntimeError("parameter count drift")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("selection") != "final" or int(ckpt.get("step", -1)) != int(load_config()["trunks"]["updates"]):
        raise RuntimeError(f"unexpected checkpoint metadata: {ckpt.get('selection')} {ckpt.get('step')}")
    C.load_state(model, ckpt["state_dict"], device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {
        "condition": f"{fit}_s{seed}",
        "checkpoint": str(ckpt_path.relative_to(REPO)),
        "checkpoint_sha256": got,
        "state_hash": C.state_hash(model),
        "selection": ckpt["selection"],
        "step": int(ckpt["step"]),
        "epoch": int(ckpt["epoch"]),
        "parameters": C.parameter_count(model),
        "train_json_sha256": sha256_file(info_path),
        "norm_source": info["norm"].get("source"),
        "total_updates": int(info["total_updates"]),
        "epochs_run": int(info["epochs_run"]),
        "n_train_rows": int(info["n_train_rows"]),
    }
    return model, norm, meta


def _compose_z(model, batch) -> torch.Tensor:
    """Independent explicit recomputation of the 513-dim reader input."""
    a, b, w = model.encoders(batch)
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    c_glob = model.group_composer(
        torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    )
    Wl, Vl, _ = model.local_grouping(w, batch, organized=True)
    c_loc = model.local_composer(
        torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    )
    B = int(batch["n_graphs"])
    zc = c_glob.new_zeros((B, 96)).index_add_(0, batch["bond_batch"], c_glob)
    zl = c_loc.new_zeros((B, 96)).index_add_(0, batch["bond_batch"], c_loc)
    zl2 = c_loc.new_zeros((B, 96)).index_add_(0, batch["bond_batch"], c_loc * c_loc)
    a_sum = a.new_zeros((B, 96)).index_add_(0, batch["atom_batch"], a)
    a_sq = a.new_zeros((B, 96)).index_add_(0, batch["atom_batch"], a * a)
    g = (batch["global"] - model.global_mean) / model.global_scale
    return torch.cat([zc, a_sum, a_sq, g, zl, zl2], dim=1)


def export_z(model, cache, rows, *, batch_size: int = 256, check_batch: int = 64):
    """Run the real forward, capture the reader input, and verify it two ways."""
    rows = np.asarray(rows, dtype=np.int64)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inp, _out):
        if "z" in captured:
            raise RuntimeError("reader invoked more than once per batch")
        captured["z"] = inp[0].detach().clone()

    handle = model.reader.register_forward_hook(hook)
    zs, preds, z_re = [], [], []
    diff_e2 = 0.0
    try:
        with torch.no_grad(), C.isolated_rng():
            for s in range(0, len(rows), batch_size):
                chunk = rows[s:s + batch_size]
                batch = T1._to_device(T1.collate(cache, chunk, "P3N", valid=True), torch.device("cpu"))
                pred = model(batch)
                if "z" not in captured:
                    raise RuntimeError("reader hook did not fire")
                z_hook = captured.pop("z")
                z_compose = _compose_z(model, batch)
                diff_e2 = max(diff_e2, float((z_hook - z_compose).abs().max()))
                zs.append(z_hook.numpy())
                z_re.append(z_compose.numpy())
                preds.append(pred.detach().numpy().reshape(-1))
    finally:
        handle.remove()
    z = np.concatenate(zs, 0).astype(np.float32)
    z_compose = np.concatenate(z_re, 0).astype(np.float32)
    pred = np.concatenate(preds, 0).astype(np.float32)

    # E3: replay the *exported* z through the same checkpoint's reader
    with torch.no_grad():
        replay = model.reader(torch.from_numpy(z)).numpy().reshape(-1)
    diff_replay = float(np.abs(replay - pred).max())

    # E4: batch-size invariance on a fixed sub-sample
    probe = rows[: min(512, len(rows))]
    with torch.no_grad(), C.isolated_rng():
        small = []
        for s in range(0, len(probe), check_batch):
            batch = T1._to_device(T1.collate(cache, probe[s:s + check_batch], "P3N", valid=True),
                                  torch.device("cpu"))
            small.append(_compose_z(model, batch).numpy())
    z_small = np.concatenate(small, 0)
    z_big = z_compose[: len(probe)]
    diff_batch = float(np.abs(z_small - z_big).max())

    checks = {
        "E1_hook_vs_export_max_abs": 0.0,  # by construction the export *is* the hook
        "E2_hook_vs_independent_recompute_max_abs": diff_e2,
        "E3_reader_replay_max_abs": diff_replay,
        "E4_batch64_vs_batch256_max_abs": diff_batch,
    }
    return z, pred, checks


# --------------------------------------------------------------------- stats
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA on row-centred features."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = float(np.linalg.norm(X.T @ Y) ** 2)
    den = float(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
    return num / den if den > 0 else float("nan")


def participation_ratio(sv: np.ndarray) -> float:
    lam = np.asarray(sv, dtype=np.float64) ** 2
    s = lam.sum()
    return float(s * s / (lam * lam).sum()) if s > 0 else 0.0


def spectral_summary(X: np.ndarray, tol_rel=(1e-2, 1e-4, 1e-6)) -> dict:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    energy = sv ** 2
    total = energy.sum()
    cum = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    out = {
        "n_rows": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "sigma_max": float(sv[0]) if sv.size else 0.0,
        "trace_cov": float(total / max(1, X.shape[0] - 1)),
        "participation_ratio": participation_ratio(sv),
        "rank_95pct_energy": int(np.searchsorted(cum, 0.95) + 1) if total > 0 else 0,
        "rank_99pct_energy": int(np.searchsorted(cum, 0.99) + 1) if total > 0 else 0,
        "per_dim_var_mean": float(Xc.var(0).mean()),
    }
    for tol in tol_rel:
        out[f"numerical_rank_tol_{tol:g}"] = int((sv > tol * sv[0]).sum()) if sv.size else 0
    return out


def block_variance_summary(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64)
    total = 0.0
    per = {}
    for name, (a, b) in BLOCKS.items():
        trace = float(z[:, a:b].var(0, ddof=1).sum())
        per[name] = {"range": [a, b], "dim": b - a, "trace_cov": trace,
                     "per_dim_var_mean": trace / (b - a)}
        total += trace
    for name in per:
        per[name]["trace_share"] = per[name]["trace_cov"] / total if total > 0 else 0.0
    return {"blocks": per, "total_trace_cov": total}


def standardizer(z_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(z_fit, dtype=np.float64).mean(0)
    sd = np.asarray(z_fit, dtype=np.float64).std(0)
    sd = np.maximum(sd, 1e-6)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(z, dtype=np.float32) - mu) / sd).astype(np.float32)


def block_mask(block: str, dim: int = 513) -> np.ndarray:
    mask = np.zeros(dim, dtype=bool)
    for a, b in PROBE_BLOCKS[block]:
        mask[a:b] = True
    return mask
