"""cscl-structural-relation-composer-v1 driver.

Protocol: ``configs/structural_relation_composer_v1.yaml`` (frozen before any
formal GPU run).  Track: ``tracks/structure-contribution``.

Stages
------
``audit``     Phase A CPU audits: Full-Y information flow, the concrete
              organisation counterexample, label duplicate/isomorphism audit,
              historical dev error structure.  Reads official-train only.
``prep``      New grouped official-train split (fit ~8000 / monitor ~1000 /
              locked ~1000), fit-only Full-Y prep for the matched reference arm
              and the label-blind BJOC object cache + fit-only global stats.
``synthetic`` Phase D: known-ground-truth organisation experiment (CPU).
``train-f``   Full-Y matched reference, seed 0, 240 epochs, last-5 soup.
``train-bjoc`` BJOC arm (``organized`` / ``null``), seed 0/1, same recipe.
``eval-locked``  ONE frozen evaluation of all trained arms on the locked dev
              split (labels read only here), paired bootstrap + subgroups.
``mechanism`` Phase I diagnostics on frozen checkpoints.

Discipline
----------
* ``official valid`` and ``official test`` are never loaded anywhere in this
  driver.
* ``locked dev`` labels are read only inside ``eval-locked``; no selection,
  no preprocessing, no training statistic uses them.
* Augmented files land in ``tracks/structure-contribution/results/
  structural_relation_composer_v1/`` (checkpoints are never committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import cscl_units as cu  # noqa: E402

PROTOCOL_ID = "cscl-structural-relation-composer-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "structural_relation_composer_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"

# ---- frozen training recipe (identical for every formal arm) --------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SCALE_SEED = 0
H1_LAMBDA = float(rpv2.H1_LAMBDA)

# ---- frozen split ----------------------------------------------------------
SPLIT_SEED = "cscl-structural-relation-composer-v1"
FIT_TARGET = 8000
MONITOR_TARGET = 1000

# ---- artifacts -------------------------------------------------------------
SPLIT_JSON = OUT / "split_v1.json"
PREP_NPZ = OUT / "prep_fit7200.npz"
PREP_META = OUT / "prep_fit7200_meta.json"
OBJECTS_CACHE = OUT / "bjoc_objects_v1.pt"
NORM_JSON = OUT / "bjoc_global_norm.json"

ARM_FILES = {
    "full_y": {"soup": OUT / "full_y_s0_soup_state.pt", "json": OUT / "train_full_y_s0.json"},
    "null": {"soup": OUT / "bjoc_null_s0_soup_state.pt", "json": OUT / "train_bjoc_null_s0.json"},
    "organized": {"soup": OUT / "bjoc_organized_s0_soup_state.pt", "json": OUT / "train_bjoc_organized_s0.json"},
    "null_s1": {"soup": OUT / "bjoc_null_s1_soup_state.pt", "json": OUT / "train_bjoc_null_s1.json"},
    "organized_s1": {"soup": OUT / "bjoc_organized_s1_soup_state.pt", "json": OUT / "train_bjoc_organized_s1.json"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def build_split_v1(smiles: Sequence[str]) -> dict[str, Any]:
    """Deterministic grouped split of official-train 10000 rows.

    Groups are canonical SMILES; groups are ordered by
    ``sha256(SPLIT_SEED + '|' + smiles)`` and greedily filled into
    fit (~8000) then monitor (~1000); the remainder is the locked dev set.
    The y value never enters.  Duplicate molecules never cross a boundary.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{SPLIT_SEED}|{s}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(fit) < FIT_TARGET:
            fit.extend(rows)
        elif len(monitor) < MONITOR_TARGET:
            monitor.extend(rows)
        else:
            locked.extend(rows)
    out = {
        "seed": SPLIT_SEED,
        "fit_target": FIT_TARGET,
        "monitor_target": MONITOR_TARGET,
        "n_groups": len(groups),
        "fit": np.asarray(sorted(fit), dtype=np.int64),
        "monitor": np.asarray(sorted(monitor), dtype=np.int64),
        "locked": np.asarray(sorted(locked), dtype=np.int64),
    }
    out["overlap"] = {
        "fit_monitor": int(len(set(fit) & set(monitor))),
        "fit_locked": int(len(set(fit) & set(locked))),
        "monitor_locked": int(len(set(monitor) & set(locked))),
        "union": int(len(set(fit) | set(monitor) | set(locked))),
    }
    return out


def load_or_build_split(rows: Sequence[Any], log=print) -> dict[str, Any]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(smiles) == 10000
    split = build_split_v1(smiles)
    assert split["overlap"]["union"] == 10000
    assert split["overlap"]["fit_monitor"] == 0
    assert split["overlap"]["fit_locked"] == 0
    assert split["overlap"]["monitor_locked"] == 0
    return split


def split_to_json(split: Mapping[str, Any], path: Path) -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL_ID,
        "split_seed": split["seed"],
        "fit_target": split["fit_target"],
        "monitor_target": split["monitor_target"],
        "n_groups": split["n_groups"],
        "n_fit": int(len(split["fit"])),
        "n_monitor": int(len(split["monitor"])),
        "n_locked": int(len(split["locked"])),
        "overlap": split["overlap"],
        "fit_sha256_rows": hashlib.sha256(np.ascontiguousarray(split["fit"], dtype=np.int64).tobytes()).hexdigest(),
        "monitor_sha256_rows": hashlib.sha256(np.ascontiguousarray(split["monitor"], dtype=np.int64).tobytes()).hexdigest(),
        "locked_sha256_rows": hashlib.sha256(np.ascontiguousarray(split["locked"], dtype=np.int64).tobytes()).hexdigest(),
        "locked_labels_read": False,
    }
    write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Phase A audits
# ---------------------------------------------------------------------------


def _wl_hash(atom_types: Sequence[int], edges: Sequence[tuple[int, int, int]], rounds: int = 3) -> str:
    n = len(atom_types)
    colors = [f"a{int(t)}" for t in atom_types]
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, bt in edges:
        adj[u].append((v, int(bt)))
        adj[v].append((u, int(bt)))
    for _ in range(rounds):
        new = []
        for i in range(n):
            payload = colors[i] + "#" + ";".join(sorted(f"{bt}:{colors[j]}" for j, bt in adj[i]))
            new.append(hashlib.sha256(payload.encode()).hexdigest()[:24])
        colors = new
    return ";".join(sorted(colors))


def audit_labels(ds: Sequence[Any], smiles: Sequence[str], log=print) -> dict[str, Any]:
    """Training-domain duplicate / isomorphism / label-consistency audit."""
    y = np.asarray([float(d.y.reshape(-1)[0]) for d in ds], dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    dup_groups = {s: rows for s, rows in groups.items() if len(rows) > 1}
    max_spread = 0.0
    inconsistent = 0
    examples = []
    for s, rows in dup_groups.items():
        spread = float(y[rows].max() - y[rows].min())
        max_spread = max(max_spread, spread)
        if spread > 1e-9:
            inconsistent += 1
            if len(examples) < 5:
                examples.append({"smiles": s, "rows": rows, "y": [float(y[i]) for i in rows], "spread": spread})
    # same-graph (1-WL) groups with different y: a stronger invariance check
    wl_groups: dict[str, list[int]] = {}
    for i, d in enumerate(ds):
        bonds = cu.unique_undirected_bonds(
            [(int(a), int(b)) for a, b in d.edge_index.numpy().T],
            [int(t) for t in d.edge_attr.numpy().reshape(-1)],
        )
        key = _wl_hash([int(v) for v in d.x.numpy().reshape(-1)], bonds)
        wl_groups.setdefault(key, []).append(i)
    wl_multi = {k: rows for k, rows in wl_groups.items() if len(rows) > 1}
    wl_inconsistent = 0
    wl_examples = []
    for k, rows in wl_multi.items():
        spread = float(y[rows].max() - y[rows].min())
        if spread > 1e-9:
            wl_inconsistent += 1
            if len(wl_examples) < 5:
                wl_examples.append({"rows": rows, "y": [float(y[i]) for i in rows], "spread": spread})
    return {
        "n_rows": len(ds),
        "n_unique_smiles": len(groups),
        "n_duplicate_smiles_groups": len(dup_groups),
        "n_rows_in_duplicate_groups": int(sum(len(r) for r in dup_groups.values())),
        "duplicate_y_max_spread": max_spread,
        "n_duplicate_groups_with_inconsistent_y": inconsistent,
        "duplicate_inconsistent_examples": examples,
        "n_wl_groups": len(wl_groups),
        "n_wl_multi_groups": len(wl_multi),
        "n_wl_groups_with_inconsistent_y": wl_inconsistent,
        "wl_inconsistent_examples": wl_examples,
        "y_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "p05": float(np.quantile(y, 0.05)),
            "median": float(np.median(y)),
        },
    }


def audit_counterexample(log=print) -> dict[str, Any]:
    """Concrete organisation counterexample (audit A1.5 / Phase D D2).

    Static object systems (see ``src.counterexample_object_pair``): four bonds
    ``{P,P,Q,Q}`` and four junctions ``{X,X,Y,Y}``, every bond touching
    exactly two junctions.  System A pairs both X junctions onto the same P
    bond, system B distributes them across both bond types.  All marginals
    (bond multiset, junction multiset, per-bond counts) are identical; the
    junction-to-bond ownership differs.

    Additionally, the two-token molecular system ``X-Y-X-Y`` vs ``Y-X-Y-X``
    is recorded as a marginal-preserving relabel that is *not* observable at
    the graph level (a useful negative control for "ownership" claims).
    """
    a, b = src.counterexample_object_pair()
    batch = src.batch_object_instances([a, b])
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=0,
        organization_mode=src.ORGANIZED,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    with torch.no_grad():
        fo = model.forward_features(batch, organization_mode=src.ORGANIZED)
        fn = model.forward_features(batch, organization_mode=src.NULL)
    result = {
        "object_pair": {
            "marginal_signature_identical": src.object_marginal_signature(a) == src.object_marginal_signature(b),
            "per_bond_junction_tokens_A": src.per_bond_junction_tokens(a),
            "per_bond_junction_tokens_B": src.per_bond_junction_tokens(b),
            "target_incidence_A": a.target_incidence,
            "target_incidence_B": b.target_incidence,
            "target_marginal_A": a.target_marginal,
            "target_marginal_B": b.target_marginal,
            "organized_W_diff_max": float((fo["W"][:4] - fo["W"][4:]).abs().max()),
            "null_W_diff_max": float((fn["W"][:4] - fn["W"][4:]).abs().max()),
        },
        "molecular_negative_control": {
            "pair": "X-Y-X-Y vs Y-X-Y-X (two atom types, all single bonds)",
            "marginal_signature_identical": True,
            "note": (
                "the reversal swaps the two junction roles as well, so the sorted per-bond "
                "junction-content multiset is identical: this relabel is NOT observable at the "
                "graph level for either arm; ownership claims need the object-level witness above"
            ),
        },
        "full_y_cannot_recover": [
            "which local environments shared a centre atom (junction ownership)",
            "which bonds the local pair states belonged to (only distance-bucket moments survive)",
        ],
    }
    log(
        f"[audit] counterexample: marginals identical={result['object_pair']['marginal_signature_identical']}, "
        f"organized W diff={result['object_pair']['organized_W_diff_max']:.3g}, "
        f"null W diff={result['object_pair']['null_W_diff_max']:.3g}"
    )
    return result


def audit_error_structure(log=print) -> dict[str, Any]:
    """Historical internal-dev error structure (read-only, dev labels of the
    *historical* split -- not the new locked dev)."""
    dev = np.load(REPO_ROOT / "tracks/structure-contribution/results/strong_nongnn_predictor_v1/static_triplet_s0_dev_preds.npz")
    row, pred, y = dev["row"].astype(np.int64), dev["pred"].astype(np.float64), dev["y"].astype(np.float64)
    ds = cf.load_official_train()
    n_atoms, cyc = [], []
    for r in row:
        d = ds[int(r)]
        bonds = cu.unique_undirected_bonds(
            [(int(a), int(b)) for a, b in d.edge_index.numpy().T],
            [int(t) for t in d.edge_attr.numpy().reshape(-1)],
        )
        n_atoms.append(len(d.x))
        cyc.append(len(bonds) - len(d.x) + 1)
    n_atoms = np.asarray(n_atoms)
    cyc = np.asarray(cyc)
    err = np.abs(pred - y)
    out = {
        "n": int(len(row)),
        "mae": float(err.mean()),
        "corr_abs_err_y": float(np.corrcoef(err, y)[0, 1]),
        "y_low_tail": {
            "n": int((y <= -3).sum()),
            "error_mass_share": float(err[y <= -3].sum() / err.sum()),
            "mae": float(err[y <= -3].mean()) if (y <= -3).any() else None,
        },
        "by_atom_count_quartile": [],
        "by_cyclomatic_quartile": [],
    }
    for name, values in (("atoms", n_atoms), ("cyclomatic", cyc)):
        qs = np.quantile(values, [0.25, 0.5, 0.75])
        edges = [values.min() - 0.5, qs[0], qs[1], qs[2], values.max() + 0.5]
        bin_id = np.searchsorted(edges, values, side="right") - 1
        entry = []
        for k, label in enumerate(["q1", "q2", "q3", "q4"]):
            mask = bin_id == k
            entry.append(
                {
                    "bucket": label,
                    "range": [float(edges[k]), float(edges[k + 1])],
                    "n": int(mask.sum()),
                    "mae": float(err[mask].mean()) if mask.any() else None,
                    "share_error_mass": float(err[mask].sum() / err.sum()),
                }
            )
        out["by_atom_count_quartile" if name == "atoms" else "by_cyclomatic_quartile"] = entry
    log(f"[audit] historical dev error structure: mae={out['mae']:.5f}")
    return out


def cmd_audit(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "phase": "A",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "full_y_flow_reference": {
            "reader_input_layout": "unary 289 | pair moments 5x97=485 | global 32 | topology 8 = 814",
            "pair_pooling": "per hard distance bucket: sum(h), sum(h^2), log1p(count); C6_MASK zeroes counts",
            "flow": "atom x -> ctx/patch/anchor features -> E (144) -> pair state h (48) -> bucket moments -> reader",
            "cannot_recover": [
                "which pair states shared a centre atom (junction ownership)",
                "which bonds were attached to which local environment beyond distance-bucket marginals",
            ],
            "evidence": "tracks/structure-contribution/notes/distance_pooling_bottleneck_v1.md; static_triplet audits",
        },
        "counterexample": audit_counterexample(),
        "label_audit": audit_labels(ds, smiles),
        "historical_dev_error_structure": audit_error_structure(),
    }
    write_json(OUT / "phase_a_audit.json", payload)
    log(f"[audit] wrote {OUT / 'phase_a_audit.json'}")
    return 0


# ---------------------------------------------------------------------------
# prep: split + fit-only Full-Y prep + BJOC objects
# ---------------------------------------------------------------------------


def load_train_rows(log=print) -> list[Any]:
    train = list(torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False))
    rpv2.p1run.attach_env(train, "train")
    assert len(train) == 10000, len(train)
    log(f"[data] loaded encoded train rows={len(train)}")
    return train


def build_bjoc_objects(smiles: Sequence[str], log=print) -> list[sro.MolObjects]:
    """Label-blind BJOC objects for all 10000 official-train rows.

    Uses the raw PyG train split (x / edge_index / edge_attr only).  The row
    order is verified against the encoded cache by raw y equality before the
    cache is written (see :func:`verify_row_alignment`)."""
    ds = cf.load_official_train()
    assert len(ds) == 10000
    objs = []
    started = time.perf_counter()
    for i, d in enumerate(ds):
        objs.append(
            sro.build_mol_objects(
                i,
                d.x.numpy(),
                d.edge_index.numpy(),
                d.edge_attr.numpy(),
                smiles[i],
            )
        )
        if (i + 1) % 2000 == 0:
            log(f"[objects] {i + 1}/10000 ({time.perf_counter() - started:.0f}s)")
    return objs


def cmd_prep(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train_rows()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    split = load_or_build_split(train)
    split_json = split_to_json(split, SPLIT_JSON)
    log(f"[prep] split fit={split_json['n_fit']} monitor={split_json['n_monitor']} locked={split_json['n_locked']}")
    # label-blind rows: raw PyG train split; verify alignment against the
    # encoded cache by raw y equality (official-train labels only)
    dataset = cf.load_official_train()
    y_encoded = np.asarray([float(train[i].y.reshape(-1)[0]) for i in range(10000)])
    y_raw = np.asarray([float(dataset[i].y.reshape(-1)[0]) for i in range(10000)])
    assert float(np.abs(y_encoded - y_raw).max()) < 1e-9, "row alignment mismatch"
    del dataset

    # --- BJOC object cache (label-blind, all 10000 rows) -------------------
    if not OBJECTS_CACHE.exists() or args.force:
        objs = build_bjoc_objects(smiles)
        torch.save(objs, OBJECTS_CACHE)
        log(f"[prep] objects cached -> {OBJECTS_CACHE}")
    else:
        log("[prep] objects cache present, skipping rebuild")

    # --- fit-only global standardizer --------------------------------------
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    fit_rows = split["fit"]
    gfit = np.stack([objs[int(i)].global_feat for i in fit_rows], 0).astype(np.float64)
    mean = gfit.mean(0)
    scale = gfit.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    write_json(
        NORM_JSON,
        {
            "protocol": PROTOCOL_ID,
            "fit_only": True,
            "n_fit": int(len(fit_rows)),
            "global_mean": mean.tolist(),
            "global_scale": scale.tolist(),
            "names": list(sro.GLOBAL_FEAT_NAMES),
        },
    )
    log("[prep] global stats written")

    # --- Full-Y matched-arm prep (fit-only fit 8k) -------------------------
    if not PREP_NPZ.exists() or args.force:
        idx = {"fit_inner": fit_rows, "monitor": split["monitor"], "dev": split["locked"]}
        path, meta = rpv2.build_prep_fit7200(train, idx, OUT, smoke=False)
        if path != PREP_NPZ:
            path.rename(PREP_NPZ)
            (OUT / "prep_fit7200_meta.json").rename(PREP_META)
        log(f"[prep] Full-Y prep done in {meta['seconds']:.0f}s")
    else:
        log("[prep] Full-Y prep present, skipping rebuild")

    counts = {
        "n_atoms": int(sum(o.n_atoms for o in objs)),
        "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
        "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
    }
    write_json(OUT / "prep_summary.json", {**split_json, **counts, "objects_cache": str(OBJECTS_CACHE.relative_to(REPO_ROOT))})
    log(f"[prep] counts {counts}")
    return 0


# ---------------------------------------------------------------------------
# training: shared loop
# ---------------------------------------------------------------------------


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Parameter mean of the final ``SOUP_LAST`` epoch states (or fewer in smoke)."""
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    keys = states[0].keys()
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return members, soup


def train_bjoc_arm(
    arm: str,
    seed: int,
    device: torch.device,
    *,
    smoke: bool = False,
    log=print,
    objects_path: Path | None = None,
    norm_path: Path | None = None,
    full10k: bool = False,
) -> dict[str, Any]:
    """Formal BJOC training: raw-y L1 only, frozen recipe, last-5 soup.

    ``full10k=True`` is the conditional Phase-H candidate: all 10000
    official-train rows, all-10k train-only preprocessing statistics, same
    frozen architecture and training parameters (no monitor, no selection).
    """
    assert arm in (src.ORGANIZED, src.NULL), arm
    objs = torch.load(objects_path or OBJECTS_CACHE, map_location="cpu", weights_only=False)
    norm = json.loads((norm_path or NORM_JSON).read_text())
    # rows are recomputed deterministically from the raw table (same as prep)
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    rows = build_split_v1(smiles)
    fit_rows = [int(i) for i in rows["fit"]]
    mon_rows = [int(i) for i in rows["monitor"]]
    if full10k:
        fit_rows = list(range(10000))
        mon_rows = []
    # label access: read only the cached rows of fit/monitor (never locked)
    train = torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)
    y_all = np.asarray([float(train[i].y.reshape(-1)[0]) for i in range(10000)], dtype=np.float64)
    del train
    if smoke:
        fit_rows, mon_rows = fit_rows[:512], mon_rows[:128]
    # objects are indexed by dataset row; training indexes by position
    fit_objs = [objs[int(i)] for i in fit_rows]
    mon_objs = [objs[int(i)] for i in mon_rows]
    assert [o.mol_index for o in fit_objs] == [int(i) for i in fit_rows]
    assert [o.mol_index for o in mon_objs] == [int(i) for i in mon_rows]
    y_fit = torch.as_tensor(y_all[fit_rows], dtype=torch.float32)
    y_mon = y_all[mon_rows]

    if full10k and not (norm_path or NORM_JSON).exists():
        gfit = np.stack([o.global_feat for o in objs], 0).astype(np.float64)
        mean = gfit.mean(0)
        scale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        write_json(norm_path or NORM_JSON, {"protocol": PROTOCOL_ID, "fit_only": True, "n_fit": len(objs), "global_mean": mean.tolist(), "global_scale": scale.tolist(), "names": list(sro.GLOBAL_FEAT_NAMES)})
        norm = json.loads((norm_path or NORM_JSON).read_text())

    cfg = src.BJOCConfig()
    seed_everything(seed)
    model = src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=arm,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )
    init_sha = src.state_dict_hash(model)
    audit = src.parameter_audit(cfg, model)
    if not audit["parameter_exact"]:
        raise RuntimeError(f"parameter audit failed: {audit}")
    if smoke:
        log(f"[smoke] {arm} params={audit['actual_parameters']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    epochs = 3 if smoke else EPOCHS
    curve: list[dict[str, Any]] = []
    from collections import deque

    states: deque = deque(maxlen=SOUP_LAST)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in _shuffle_batches(len(fit_rows), BATCH_SIZE, gen):
            batch = _mv(sro.batch_objects([fit_objs[i] for i in indices]), device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y_fit[indices].to(device).view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float((pred.view(-1) - y_fit[indices].to(device).view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        if mon_rows:
            mon_pred = predict_bjoc(model, mon_objs, list(range(len(mon_rows))), device)
            mon_mae = float(np.abs(mon_pred - y_mon).mean())
        else:
            mon_mae = float("nan")
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if log and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
            log(f"[{arm}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(mon_mae) or not np.isfinite(curve[-1]["train_mae"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_bjoc(model, fit_objs, list(range(len(fit_rows))), device)
    mon_pred = (
        predict_bjoc(model, mon_objs, list(range(len(mon_rows))), device)
        if mon_rows
        else np.zeros(0, dtype=np.float64)
    )
    b_cal = float(np.median(y_all[fit_rows] - fit_pred))
    results = {
        "protocol": PROTOCOL_ID,
        "arm": f"BJOC-{arm}",
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs": epochs,
        "soup_rule": "parameter mean of the last 5 epochs (236..240)",
        "soup_members": members,
        "hyper": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "supervision": "raw y L1 only",
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "config": {
            "atom_hidden": cfg.atom_hidden,
            "atom_out": cfg.atom_out,
            "bond_hidden": cfg.bond_hidden,
            "bond_out": cfg.bond_out,
            "junc_hidden": cfg.junc_hidden,
            "junc_out": cfg.junc_out,
            "compose_hidden": cfg.compose_hidden,
            "compose_out": cfg.compose_out,
            "reader_hidden": cfg.reader_hidden,
            "reader_layers": cfg.reader_layers,
            "dropout": cfg.dropout,
            "organization_mode": arm,
        },
        "parameters": audit["actual_parameters"],
        "init_state_sha256": init_sha,
        "curve": curve,
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - y_all[fit_rows]).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()) if mon_rows else None,
        "soup_calibrated_monitor_mae": float(np.abs(mon_pred + b_cal - y_mon).mean()) if mon_rows else None,
        "full10k": bool(full10k),
        "b_fit": b_cal,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fit_only_discipline": {
            "global_stats": "fit rows only",
            "locked_labels_read": False,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not smoke:
        name = f"bjoc_{arm}_s{seed}" if not full10k else f"bjoc_{arm}_full10k_s{seed}"
        torch.save(soup, OUT / f"{name}_soup_state.pt")
        if mon_rows:
            np.savez_compressed(
                OUT / f"bjoc_{arm}_s{seed}_monitor_preds.npz",
                row=np.asarray(mon_rows, dtype=np.int64),
                pred=mon_pred.astype(np.float32),
                y=y_mon.astype(np.float32),
            )
        write_json(OUT / f"train_bjoc_{arm}{'_full10k' if full10k else ''}_s{seed}.json", results)
    log(
        f"[{arm}] soup monitor {results['soup_monitor_mae']:.5f} "
        f"(cal {results['soup_calibrated_monitor_mae']:.5f}) | wall {wall:.0f}s"
    )
    return results


def predict_bjoc(model: src.BJOC, objs: Sequence[sro.MolObjects], rows: Sequence[int], device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = [objs[int(r)] for r in rows[i : i + 256]]
            batch = _mv(sro.batch_objects(chunk), device)
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def cmd_train_bjoc(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    train_bjoc_arm(
        args.arm,
        int(args.seed),
        device,
        smoke=bool(args.smoke),
        full10k=bool(getattr(args, "full10k", False)),
        norm_path=(OUT / "bjoc_global_norm_full10k.json") if getattr(args, "full10k", False) else None,
    )
    return 0


# ---------------------------------------------------------------------------
# training: Full-Y matched reference (same recipe, last-5 soup)
# ---------------------------------------------------------------------------


def train_full_y_matched(seed: int, device: torch.device, *, smoke: bool = False, log=print) -> dict[str, Any]:
    """Full-Y on the new fit split, last-5 soup, raw y + disclosed label-free
    reconstruction.  Recipe constants are imported from the frozen feasibility
    driver so they cannot drift."""
    train = load_train_rows()
    prep = {k: z for k, z in np.load(PREP_NPZ, allow_pickle=False).items()}
    legacy = {k: z for k, z in np.load(rpv2.LEGACY_PREP, allow_pickle=False).items()}
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    rows = build_split_v1(smiles)
    fit_rows = [int(i) for i in rows["fit"]]
    mon_rows = [int(i) for i in rows["monitor"]]
    if smoke:
        fit_rows, mon_rows = fit_rows[:512], mon_rows[:128]
    rpv2.apply_prep_fit7200(train, prep, legacy)
    fit_data = [train[i] for i in fit_rows]
    mon_data = [train[i] for i in mon_rows]
    y_all = np.asarray([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    y_fit = torch.as_tensor(y_all[fit_rows], dtype=torch.float32)
    y_mon = y_all[mon_rows]

    seed_everything(seed)
    model = rpv2.build_model(prep, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rpv2.LR, weight_decay=rpv2.WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(int(seed) + rpv2.TRAIN_SHUFFLE_OFFSET)
    epochs = 3 if smoke else EPOCHS
    curve: list[dict[str, Any]] = []
    from collections import deque

    states: deque = deque(maxlen=SOUP_LAST)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        task_sum, rec_sum, n_mol, steps = 0.0, 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in rpv2._batches(len(fit_data), rpv2.BATCH_SIZE, gen, True):
            batch = rpv2._batch(fit_data, indices, y_fit, device)
            prediction, aux = model(batch, mask=rpv2.cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rpv2.GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            steps += 1
        mon_pred = rpv2.predict_raw(model, mon_data, device)
        mon_mae = float(np.abs(mon_pred - y_mon).mean())
        curve.append({"epoch": epoch, "train_task_mae": task_sum / max(n_mol, 1), "train_rec": rec_sum / max(steps, 1), "monitor_mae": mon_mae, "seconds": time.perf_counter() - t_ep})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if log and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
            log(f"[full_y] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(mon_mae):
            raise RuntimeError(f"non-finite monitor at epoch {epoch}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = rpv2.predict_raw(model, fit_data, device)
    mon_pred = rpv2.predict_raw(model, mon_data, device)
    b_cal = float(np.median(y_all[fit_rows] - fit_pred))
    results = {
        "protocol": PROTOCOL_ID,
        "arm": "Full-Y-matched",
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs": epochs,
        "soup_rule": "parameter mean of the last 5 epochs (236..240)",
        "soup_members": members,
        "hyper": {"lr": rpv2.LR, "weight_decay": rpv2.WEIGHT_DECAY, "grad_clip": rpv2.GRAD_CLIP, "batch_size": rpv2.BATCH_SIZE, "epochs": EPOCHS, "h1_lambda": H1_LAMBDA},
        "curve": curve,
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - y_all[fit_rows]).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()) if mon_rows else None,
        "soup_calibrated_monitor_mae": float(np.abs(mon_pred + b_cal - y_mon).mean()) if mon_rows else None,
        "full10k": bool(full10k),
        "b_fit": b_cal,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "prep_sha256": file_sha256(PREP_NPZ),
        "fit_only_discipline": {"standardizers": "fit rows only", "D_fit": "fit rows only", "U_subspace": "fit rows only", "locked_labels_read": False},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not smoke:
        torch.save(soup, OUT / "full_y_s0_soup_state.pt")
        np.savez_compressed(OUT / "full_y_s0_monitor_preds.npz", row=np.asarray(mon_rows, dtype=np.int64), pred=mon_pred.astype(np.float32), y=y_mon.astype(np.float32))
        write_json(OUT / "train_full_y_s0.json", results)
    log(f"[full_y] soup monitor {results['soup_monitor_mae']:.5f} | wall {wall:.0f}s")
    return results


def cmd_train_f(args: argparse.Namespace) -> int:
    train_full_y_matched(int(args.seed), torch.device(args.device), smoke=bool(args.smoke))
    return 0


# ---------------------------------------------------------------------------
# Phase D synthetic experiment
# ---------------------------------------------------------------------------


def _fit_object_arm(
    arm: str,
    cfg,
    train_instances,
    target_key: str,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_fn=None,
    target_scale=(0.0, 1.0),
):
    """Train one synthetic arm on a standardised target (train mean/std)."""
    from structural_relation_composer_v1 import build_bjoc

    torch.manual_seed(int(seed))
    model = build_bjoc(
        cfg,
        seed=int(seed),
        organization_mode=arm,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device=device,
    )
    mu, sd = target_scale
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(int(seed) + 7)
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_instances), generator=gen).tolist()
        for i in range(0, len(order), 64):
            chunk = [train_instances[j] for j in order[i : i + 64]]
            batch = (batch_fn or src.batch_object_instances)(chunk)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = (batch[target_key] - mu) / sd
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def _target_stats(instances, target_key: str, batch_fn=None):
    batch = (batch_fn or src.batch_object_instances)(instances)
    values = batch[target_key].numpy().astype(np.float64)
    return float(values.mean()), float(values.std() + 1e-9)


def run_synthetic(args: argparse.Namespace, log=print) -> dict[str, Any]:
    from structural_relation_composer_v1 import (
        NULL,
        ORGANIZED,
        batch_object_instances,
        build_bjoc,
        counterexample_object_pair,
        object_marginal_signature,
        object_targets,
        per_bond_junction_tokens,
        random_object_instance,
    )

    cfg = src.BJOCConfig(
        atom_hidden=32,
        atom_out=16,
        bond_hidden=32,
        bond_out=16,
        junc_hidden=32,
        junc_out=16,
        compose_hidden=32,
        compose_out=16,
        reader_hidden=48,
        reader_layers=2,
    )
    device = torch.device(args.device)
    results = []

    # ---- 1. explicit marginal-preserving ownership switch ------------------
    ca, cb = counterexample_object_pair()
    model = build_bjoc(
        cfg, seed=0, organization_mode=ORGANIZED,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_object_instances([ca, cb]).items()}
    with torch.no_grad():
        fro = model.forward_features(b, organization_mode=ORGANIZED)
        fnull = model.forward_features(b, organization_mode=NULL)
    counterexample = {
        "marginals_identical": object_marginal_signature(ca) == object_marginal_signature(cb),
        "per_bond_junction_tokens_A": per_bond_junction_tokens(ca),
        "per_bond_junction_tokens_B": per_bond_junction_tokens(cb),
        "target_incidence_A": ca.target_incidence,
        "target_incidence_B": cb.target_incidence,
        "target_marginal_A": ca.target_marginal,
        "target_marginal_B": cb.target_marginal,
        "organized_W_diff_max": float((fro["W"][:4] - fro["W"][4:]).abs().max()),
        "null_W_diff_max": float((fnull["W"][:4] - fnull["W"][4:]).abs().max()),
    }
    log(
        f"[synthetic] counterexample: marginals identical={counterexample['marginals_identical']} "
        f"W diff organized={counterexample['organized_W_diff_max']:.3g} null={counterexample['null_W_diff_max']:.3g}"
    )

    # ---- 2. object-instance fitting (incidence vs marginal targets) --------
    for seed in args.seeds:
        rng = np.random.default_rng(int(seed))
        train = [random_object_instance(rng, int(rng.integers(4, 8)), int(rng.integers(4, 10))) for _ in range(int(args.n_train))]
        test = [random_object_instance(rng, int(rng.integers(4, 8)), int(rng.integers(4, 10))) for _ in range(int(args.n_test))]
        for target_key in ("target_incidence", "target_marginal"):
            entry = {"seed": int(seed), "target": target_key, "n_train": len(train), "n_test": len(test)}
            mu, sd = _target_stats(train, target_key)
            entry["target_mean"] = mu
            entry["target_std"] = sd
            for arm in (ORGANIZED, NULL):
                model = _fit_object_arm(arm, cfg, train, target_key, seed=int(seed), epochs=args.epochs, device=device, target_scale=(mu, sd))
                with torch.no_grad():
                    tb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_object_instances(test).items()}
                    pred = model(tb).cpu().numpy()
                    y_true = (tb[target_key] - mu).numpy() / sd
                    mae_val = float(np.abs(pred - y_true).mean())
                entry[arm] = {"mae": mae_val}
            entry["mae_gap_organized_minus_null"] = entry[ORGANIZED]["mae"] - entry[NULL]["mae"]
            results.append(entry)
            log(
                f"[synthetic] seed={seed} target={target_key} MAE org={entry[ORGANIZED]['mae']:.4f} "
                f"null={entry[NULL]['mae']:.4f} gap={entry['mae_gap_organized_minus_null']:.4f}"
            )

    # ---- 3. molecular random graphs (secondary realism check) --------------
    molecular = []
    from structural_relation_composer_v1 import random_synthetic_instance, synthetic_instance_to_batch

    for seed in args.seeds[:1]:
        rng = np.random.default_rng(int(seed) + 1000)
        train = [random_synthetic_instance(rng, int(rng.integers(5, 10))) for _ in range(int(args.n_train))]
        test = [random_synthetic_instance(rng, int(rng.integers(5, 10))) for _ in range(int(args.n_test))]
        entry = {"seed": int(seed), "n_train": len(train), "n_test": len(test)}
        mu, sd = _target_stats(train, "target_incidence", batch_fn=synthetic_instance_to_batch)
        entry["target_mean"] = mu
        entry["target_std"] = sd
        for arm in (ORGANIZED, NULL):
            model = _fit_object_arm(arm, cfg, train, "target_incidence", seed=int(seed), epochs=args.epochs, device=device, batch_fn=synthetic_instance_to_batch, target_scale=(mu, sd))
            with torch.no_grad():
                tb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in synthetic_instance_to_batch(test).items()}
                pred = model(tb).cpu().numpy()
                y_true = (tb["target_incidence"] - mu).numpy() / sd
                mae_val = float(np.abs(pred - y_true).mean())
            entry[arm] = {"mae": mae_val}
        entry["mae_gap_organized_minus_null"] = entry[ORGANIZED]["mae"] - entry[NULL]["mae"]
        molecular.append(entry)
        log(
            f"[synthetic] molecular seed={seed} MAE org={entry[ORGANIZED]['mae']:.4f} "
            f"null={entry[NULL]['mae']:.4f} gap={entry['mae_gap_organized_minus_null']:.4f}"
        )

    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "synthetic",
        "device": str(device),
        "counterexample_object_pair": counterexample,
        "object_instance_results": results,
        "molecular_results": molecular,
    }
    write_json(OUT / "synthetic_phase_d.json", payload)
    return payload


def cmd_synthetic(args: argparse.Namespace) -> int:
    run_synthetic(args)
    return 0


# ---------------------------------------------------------------------------
# locked dev evaluation
# ---------------------------------------------------------------------------


def metrics(prefix: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - y)
    return {
        f"{prefix}_mae": float(err.mean()),
        f"{prefix}_median_abs": float(np.median(err)),
        f"{prefix}_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        f"{prefix}_p90_abs": float(np.quantile(err, 0.90)),
        f"{prefix}_p95_abs": float(np.quantile(err, 0.95)),
    }


def paired_bootstrap(delta: np.ndarray, n: int = 5000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_rows = len(delta)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        means[i] = delta[idx].mean()
    return {"mean": float(delta.mean()), "ci_lo": float(np.quantile(means, 0.025)), "ci_hi": float(np.quantile(means, 0.975)), "frac_better": float((delta < 0).mean())}


def cmd_eval_locked(args: argparse.Namespace) -> int:
    """ONE evaluation.  Reads locked-dev labels for the first time."""
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    rows = build_split_v1(smiles)
    locked = [int(i) for i in rows["locked"]]
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    norm = json.loads(NORM_JSON.read_text())
    device = torch.device(args.device)
    train_rows = torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)
    y_locked = np.asarray([float(train_rows[i].y.reshape(-1)[0]) for i in locked], dtype=np.float64)

    preds: dict[str, np.ndarray] = {}
    # BJOC arms
    for arm in ("null", "organized"):
        soup_path = OUT / f"bjoc_{arm}_s0_soup_state.pt"
        if not soup_path.exists():
            continue
        cfg = src.BJOCConfig()
        model = src.build_bjoc(
            cfg, seed=0, organization_mode=arm,
            global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
            global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
            device=device,
        )
        model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
        preds[arm] = predict_bjoc(model, objs, locked, device)
    # Full-Y arm needs the prep-applied rows
    if (OUT / "full_y_s0_soup_state.pt").exists():
        train = load_train_rows()
        prep = {k: z for k, z in np.load(PREP_NPZ, allow_pickle=False).items()}
        legacy = {k: z for k, z in np.load(rpv2.LEGACY_PREP, allow_pickle=False).items()}
        rpv2.apply_prep_fit7200(train, prep, legacy)
        cfg = None  # Full-Y model
        model = rpv2.build_model(prep, 0).to(device)
        model.load_state_dict(torch.load(OUT / "full_y_s0_soup_state.pt", map_location=device, weights_only=False))
        locked_data = [train[i] for i in locked]
        preds["full_y"] = rpv2.predict_raw(model, locked_data, device)

    ckpt_hashes = {}
    for f in sorted(OUT.glob("*_soup_state.pt")):
        ckpt_hashes[f.name] = file_sha256(f)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "eval_locked",
        "git_commit": git_commit(),
        "checkpoints_sha256": ckpt_hashes,
        "n_locked": len(locked),
        "locked_rows_sha256": hashlib.sha256(np.ascontiguousarray(locked, dtype=np.int64).tobytes()).hexdigest(),
        "locked_labels_read": True,
        "models": {},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    for name, p in preds.items():
        payload["models"][name] = {
            **metrics("locked", p, y_locked),
            "fit_calibrated": None,
        }
        np.savez_compressed(OUT / f"locked_preds_{name}.npz", row=np.asarray(locked, dtype=np.int64), pred=p.astype(np.float32), y=y_locked.astype(np.float32))
    if "organized" in preds and "null" in preds:
        delta = np.abs(preds["organized"] - y_locked) - np.abs(preds["null"] - y_locked)
        payload["organized_minus_null"] = paired_bootstrap(delta)
    if "organized" in preds and "full_y" in preds:
        delta = np.abs(preds["organized"] - y_locked) - np.abs(preds["full_y"] - y_locked)
        payload["organized_minus_full_y"] = paired_bootstrap(delta)
    if "null" in preds and "full_y" in preds:
        delta = np.abs(preds["null"] - y_locked) - np.abs(preds["full_y"] - y_locked)
        payload["null_minus_full_y"] = paired_bootstrap(delta)
    # per-molecule improvement fractions + subgroups
    if len(preds) >= 2:
        ds = cf.load_official_train()
        n_atoms = np.asarray([int(ds[i].x.shape[0]) for i in locked])
        n_bonds = np.asarray([int(ds[i].edge_index.shape[1]) // 2 for i in locked])
        cyc = n_bonds - n_atoms + 1
        subgroups = {
            "y_le_-3": y_locked <= -3.0,
            "atoms_le_median": n_atoms <= np.median(n_atoms),
            "atoms_gt_median": n_atoms > np.median(n_atoms),
            "cyclomatic_le_2": cyc <= 2,
            "cyclomatic_gt_2": cyc > 2,
        }
        payload["subgroups"] = {}
        for label, mask in subgroups.items():
            entry: dict[str, Any] = {"n": int(mask.sum()), "mae": {k: float(np.abs(p[mask] - y_locked[mask]).mean()) for k, p in preds.items()}}
            if "organized" in preds and "null" in preds:
                d = np.abs(preds["organized"][mask] - y_locked[mask]) - np.abs(preds["null"][mask] - y_locked[mask])
                entry["org_minus_null"] = paired_bootstrap(d)
            payload["subgroups"][label] = entry
    write_json(OUT / "locked_dev_eval.json", payload)
    for name, p in preds.items():
        log(f"[locked] {name}: MAE {np.abs(p - y_locked).mean():.5f}")
    return 0


# ---------------------------------------------------------------------------
# mechanism diagnostics
# ---------------------------------------------------------------------------


def _load_official_split(split: str):
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    return _load_zinc(cf.ZINC_ROOT, split)


def cmd_eval_valid(args: argparse.Namespace) -> int:
    """Conditional Phase-H: ONE frozen official-validation evaluation.

    Order (protocol H3): build label-blind valid objects, freeze every arm's
    predictions / checkpoint+preprocessing hashes in a single manifest, and
    only then read the official-validation labels once (for all arms).
    Official test is never touched.
    """
    norm_path = OUT / "bjoc_global_norm_full10k.json"
    arms = list(args.arms)
    device = torch.device(args.device)
    norm = json.loads(norm_path.read_text())
    ds = _load_official_split("val")
    if len(ds) != 1000:
        raise SystemExit(f"unexpected valid size {len(ds)}")
    objs = [
        sro.build_mol_objects(i, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        for i, d in enumerate(ds)
    ]
    cfg = src.BJOCConfig()
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "eval_valid_freeze",
        "arms": {},
        "norm": str(norm_path.relative_to(REPO_ROOT)),
        "norm_sha256": file_sha256(norm_path),
        "git_commit": git_commit(),
        "n_valid": len(objs),
        "official_test_loaded": False,
    }
    preds: dict[str, np.ndarray] = {}
    for arm in arms:
        soup_path = OUT / f"bjoc_{arm}_full10k_s0_soup_state.pt"
        if not soup_path.exists():
            raise SystemExit(f"missing frozen checkpoint {soup_path}")
        model = src.build_bjoc(
            cfg,
            seed=0,
            organization_mode=arm,
            global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
            global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
            device=device,
        )
        model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
        pred = predict_bjoc(model, objs, list(range(len(objs))), device)
        if not np.isfinite(pred).all():
            raise RuntimeError("non-finite official-valid predictions")
        pred_path = OUT / f"official_valid_preds_{arm}_full10k_s0.npz"
        np.savez_compressed(pred_path, row=np.arange(len(pred), dtype=np.int64), pred=pred.astype(np.float32))
        manifest["arms"][arm] = {
            "checkpoint": str(soup_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": file_sha256(soup_path),
            "predictions": str(pred_path.relative_to(REPO_ROOT)),
            "predictions_sha256": file_sha256(pred_path),
        }
        preds[arm] = pred.astype(np.float64)
        log(f"[eval-valid] frozen {arm} predictions (sha {manifest['arms'][arm]['predictions_sha256'][:12]})")
    write_json(OUT / "official_valid_freeze.json", manifest)
    log("[eval-valid] freeze manifest written; reading labels exactly once")
    y = np.asarray([float(d.y.reshape(-1)[0]) for d in ds], dtype=np.float64)
    payload = {
        **manifest,
        "labels_read_after_freeze": True,
        "historical_full_y_valid_mae": 0.118618,
        "note": (
            "historical Full-Y official-valid 0.118618 is a disclosed read-only reference from an "
            "earlier round (different seed/trajectory/matched split); official validation has been "
            "exposed before, so this is a controlled comparison, not a fresh blind test"
        ),
        "models": {arm: metrics("valid", p, y) for arm, p in preds.items()},
    }
    if "organized" in preds and "null" in preds:
        payload["organized_minus_null"] = paired_bootstrap(
            np.abs(preds["organized"] - y) - np.abs(preds["null"] - y)
        )
    write_json(OUT / "official_valid_eval.json", payload)
    for arm, m in payload["models"].items():
        log(f"[eval-valid] {arm}: MAE {m['valid_mae']:.6f}")
    return 0


def cmd_mechanism(args: argparse.Namespace) -> int:
    """Phase I: frozen-model sensitivity / counterfactual diagnostics."""
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    rows = build_split_v1(smiles)
    mon_rows = [int(i) for i in rows["monitor"]]
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    norm = json.loads(NORM_JSON.read_text())
    device = torch.device(args.device)
    train = torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)
    y_mon = np.asarray([float(train[i].y.reshape(-1)[0]) for i in mon_rows], dtype=np.float64)

    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "stage": "mechanism", "rows": "monitor (never locked/official)"}
    # gradient activity on fit (fresh pass, a few batches)
    if (OUT / "bjoc_organized_s0_soup_state.pt").exists():
        cfg = src.BJOCConfig()
        model = src.build_bjoc(
            cfg, seed=0, organization_mode="organized",
            global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
            global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
            device=device,
        )
        model.load_state_dict(torch.load(OUT / "bjoc_organized_s0_soup_state.pt", map_location=device, weights_only=False))
        fit_rows = [int(i) for i in rows["fit"]][:256]
        y_fit = torch.as_tensor([float(train[i].y.reshape(-1)[0]) for i in fit_rows], dtype=torch.float32, device=device)
        model.train()
        batch = _mv(sro.batch_objects([objs[i] for i in fit_rows]), device)
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), y_fit.view(-1))
        loss.backward()
        grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads.append({"name": name, "grad_norm": float(p.grad.norm()), "param_norm": float(p.detach().norm())})
        payload["gradient_activity"] = grads

        model.eval()

        def predict_batch(objs_chunk, **kwargs):
            batch = _mv(sro.batch_objects(objs_chunk), device)
            with torch.no_grad():
                if kwargs.get("mode") is not None:
                    return model(batch, organization_mode=kwargs["mode"]).cpu().numpy()
                return model(batch).cpu().numpy()

        monitor_objs = [objs[i] for i in mon_rows]
        base = predict_batch(monitor_objs)
        payload["base_monitor_mae"] = float(np.abs(base - y_mon).mean())
        counterfactuals: dict[str, Any] = {}
        # (1) full null-assignment counterfactual on the organized model
        alt = predict_batch(monitor_objs, mode=src.NULL)
        counterfactuals["null_assignment"] = {
            "mean_abs_dpred": float(np.abs(alt - base).mean()),
            "max_abs_dpred": float(np.abs(alt - base).max()),
            "mae": float(np.abs(alt - y_mon).mean()),
        }
        # (1b) graded ownership destruction: permute the junction codes w
        # within each molecule (junction multiset identical, ownership gone)
        # and across molecules (only the global multiset survives).
        def forward_with_w(batch, w_override):
            a = model.encode_atoms(batch)
            b = model.encode_bonds(a, batch)
            W, V, n_b = model.group_to_bonds(w_override, batch, src.ORGANIZED)
            compose_in = torch.cat(
                [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1
            )
            c = model.group_composer(compose_in)
            n_graphs = int(batch["n_graphs"])
            z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
            z.index_add_(0, batch["bond_batch"], c)
            a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
            a_sum.index_add_(0, batch["atom_batch"], a)
            a_sq = torch.zeros_like(a_sum)
            a_sq.index_add_(0, batch["atom_batch"], a * a)
            g = (batch["global_feat"].to(a.dtype) - model.global_mean) / model.global_scale
            reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
            return model.reader(reader_in).view(-1), w_override

        import random as _random

        def _permute_within(w, junc_batch, seed):
            groups: dict[int, list[int]] = {}
            for i, gidx in enumerate(junc_batch.tolist()):
                groups.setdefault(gidx, []).append(i)
            perm = list(range(len(w)))
            rnd = _random.Random(seed)
            for idxs in groups.values():
                shuffled = idxs[:]
                rnd.shuffle(shuffled)
                for i, j in zip(idxs, shuffled):
                    perm[i] = j
            return w[perm]

        for tag, seed in (("permute_w_within_molecule", 0), ("permute_w_within_molecule_seed1", 1)):
            batch = _mv(sro.batch_objects(monitor_objs), device)
            with torch.no_grad():
                a_ = model.encode_atoms(batch)
                b_ = model.encode_bonds(a_, batch)
                w_ = model.encode_junctions(a_, b_, batch)
                w_perm = _permute_within(w_, batch["junc_batch"], seed)
                p_perm, _ = forward_with_w(batch, w_perm)
                p_base, _ = forward_with_w(batch, w_)
                p_base_np = p_base.cpu().numpy()
                p_perm = p_perm.cpu().numpy()
                d_base = float(np.abs(p_base_np - base).max())
                assert p_base_np.shape == base.shape and d_base < 1e-4, (p_base_np.shape, base.shape, d_base)
            counterfactuals[tag] = {
                "mean_abs_dpred": float(np.abs(p_perm - base).mean()),
                "max_abs_dpred": float(np.abs(p_perm - base).max()),
                "mae": float(np.abs(p_perm - y_mon).mean()),
                "recompute_consistency_max": d_base,
            }
        # across-molecule permutation of w (global multiset preserved)
        batch = _mv(sro.batch_objects(monitor_objs), device)
        with torch.no_grad():
            a_ = model.encode_atoms(batch)
            b_ = model.encode_bonds(a_, batch)
            w_ = model.encode_junctions(a_, b_, batch)
            rnd = _random.Random(0)
            perm = list(range(w_.shape[0]))
            rnd.shuffle(perm)
            p_perm, _ = forward_with_w(batch, w_[perm])
        counterfactuals["permute_w_across_molecules"] = {
            "mean_abs_dpred": float(np.abs(p_perm.cpu().numpy() - base).mean()),
            "max_abs_dpred": float(np.abs(p_perm.cpu().numpy() - base).max()),
            "mae": float(np.abs(p_perm.cpu().numpy() - y_mon).mean()),
        }
        # (2) input masking counterfactuals: rebuild objects with masked features
        import copy

        specs = {
            "zero_bond_type_counts": ["bond_type_counts"],
            "zero_ring_features": ["in_ring", "min_ring_size", "ring_bonds"],
            "zero_radius2": ["radius2_types", "radius2_stats"],
            "zero_atom_type": ["atom_type"],
            "zero_role_features": ["roles"],
        }
        blocks = sro.atom_blocks()

        def apply_mask(f: np.ndarray, names: Sequence[str]) -> np.ndarray:
            f2 = f.copy()
            for name in names:
                f2[:, blocks[name]] = 0.0
            return f2

        for name, block_names in specs.items():
            try:
                masked = []
                for o in monitor_objs:
                    o2 = copy.copy(o)
                    o2.atom_feat = apply_mask(o.atom_feat, block_names)
                    masked.append(o2)
                p = predict_batch(masked)
                counterfactuals[name] = {"mean_abs_dpred": float(np.abs(p - base).mean()), "mae": float(np.abs(p - y_mon).mean())}
            except Exception as exc:  # noqa: BLE001
                counterfactuals[name] = {"error": repr(exc)}
        # (3) bond-feature masking (bond raw block)
        masked = []
        for o in monitor_objs:
            o2 = copy.copy(o)
            bf = o.bond_feat.copy()
            bf[:, :4] = 0.0
            o2.bond_feat = bf
            masked.append(o2)
        p = predict_batch(masked)
        counterfactuals["zero_bond_type_onehot"] = {"mean_abs_dpred": float(np.abs(p - base).mean()), "mae": float(np.abs(p - y_mon).mean())}
        # (4) global feature masking
        masked = []
        for o in monitor_objs:
            o2 = copy.copy(o)
            o2.global_feat = np.zeros_like(o2.global_feat)
            masked.append(o2)
        p = predict_batch(masked)
        counterfactuals["zero_global_features"] = {"mean_abs_dpred": float(np.abs(p - base).mean()), "mae": float(np.abs(p - y_mon).mean())}
        payload["counterfactuals"] = counterfactuals

    if (OUT / "bjoc_null_s0_soup_state.pt").exists():
        cfg = src.BJOCConfig()
        model_n = src.build_bjoc(
            cfg, seed=0, organization_mode="null",
            global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
            global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
            device=device,
        )
        model_n.load_state_dict(torch.load(OUT / "bjoc_null_s0_soup_state.pt", map_location=device, weights_only=False))
        model_n.eval()
        monitor_objs = [objs[i] for i in mon_rows]
        with torch.no_grad():
            b = _mv(sro.batch_objects(monitor_objs), device)
            base_n = model_n(b).cpu().numpy()
            alt_n = model_n(b, organization_mode=src.ORGANIZED).cpu().numpy()
        payload["null_model"] = {
            "base_monitor_mae": float(np.abs(base_n - y_mon).mean()),
            "real_assignment_swap_mean_abs_dpred": float(np.abs(alt_n - base_n).mean()),
        }
    write_json(OUT / "mechanism_diagnostics.json", payload)
    log(f"[mechanism] wrote {OUT / 'mechanism_diagnostics.json'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("prep")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("synthetic")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--n-pairs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.set_defaults(func=lambda a: (run_synthetic(a), 0)[1])

    p = sub.add_parser("train-f")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_train_f)

    p = sub.add_parser("train-bjoc")
    p.add_argument("--arm", choices=list(src.ORGANIZATION_MODES), required=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full10k", action="store_true")
    p.set_defaults(func=cmd_train_bjoc)

    p = sub.add_parser("eval-locked")
    p.set_defaults(func=cmd_eval_locked)

    p = sub.add_parser("eval-valid")
    p.add_argument("--arms", nargs="+", choices=list(src.ORGANIZATION_MODES), default=[src.ORGANIZED])
    p.set_defaults(func=cmd_eval_valid)

    p = sub.add_parser("mechanism")
    p.set_defaults(func=cmd_mechanism)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
