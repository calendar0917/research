"""ZINC CSSD g-component attribution v1: frozen-consumer ell/s component error
attribution on the four frozen soups of ``zinc_cssd_consumer_replacement_v1``.

Round (see ``tracks/ksvd/notes/zinc_cssd_g_component_attribution_v1.md``, frozen
before any labeled analysis): where does the out-of-sample error of the current
consumers' ``g = ell + s`` live, under which coarse structure/chemistry
conditions, concentrated or dispersed — and what that does or does not imply
for the next-stage "dictionary as the structure-chemistry representation core"
design.  This is an attribution round: no training, no feature search, no new
statistics, no architecture changes, no official valid/test.

Two stages:

* ``export`` (label-free, CPU, seconds): restore the four frozen soups
  (RAW/DICT x seed 0/1), replay-check against the promoted prediction npz
  (fit h/ell_hat/s_hat and dev h, ``<=1e-4`` fp32 save precision), verify the
  strict reader identity ``h == ell_hat + s_hat`` on the real forward output,
  and export the missing per-row dev component predictions
  (``dev_components_<run>.npz``).  The fit side is reused read-only from the
  promoted ``fit_predictions.npz`` (it already contains ell_hat/s_hat).  Also
  fixes the label-free structure groupings (node-count terciles, coarse atom
  composition, coarse bond composition) with their pre-registered validation
  cross-checks, saved to ``structure_assignments.npz``.  targets.npz is never
  opened in this stage.
* ``analyze`` (labels, pure numpy, seconds): component synthesis
  (MAE/bias/opposite-sign/cancellation per run, split and k-group), k=0
  subgroup tables with DICT-RAW paired deltas, concentration (top-20 shares,
  subgroup shares) and canonical-SMILES-group paired bootstrap CIs
  (2000 draws, seed 20261022, shared group resampling across arms and seeds).

Sign convention (identical to the triage round): ``e = prediction - truth``.
Component MAEs are never summed into a g budget (cancellation exists); the g
bias is reported for g only.  dev (1999 rows = the historical development set)
is exploratory localization, never a confirmation.

Usage::

    uv run research run zinc_cssd_g_component_attribution_v1 \
        --study zinc-context-gap --set model.stage=export
    uv run research run zinc_cssd_g_component_attribution_v1 \
        --study zinc-context-gap --set model.stage=analyze
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROTOCOL_VERSION = "zinc-cssd-g-component-attribution-v1"
RESULT_SLUG = "zinc_cssd_g_component_attribution_v1"
TRACK_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG

#: read-only frozen sources (paths identical to the analyzed round's constants)
SOURCE_DIR = TRACK_ROOT / "results" / "zinc_cssd_nonlinear_binding_v1"
CONSUMER_DIR = TRACK_ROOT / "results" / "zinc_cssd_consumer_replacement_v1"
SMILES_TABLE = TRACK_ROOT / "results" / "zinc_cssd_basis_reuse_v1" / "train_canonical_smiles.npz"

ARMS = ("RAW", "DICT")
SEEDS = (0, 1)
RUN_NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)

REPLAY_TOL = 1e-4    # fp32 save-precision tolerance for prediction replay
IDENTITY_TOL = 1e-5  # h == ell_hat + s_hat on the real forward output
TARGET_IDENTITY_TOL = 1e-9

TOP_N = 20
BOOT_DRAWS = 2000
BOOT_SEED = 20261022

# ---------------------------------------------------------------------------
# frozen structure groupings (pre-registered, label-free)
# ---------------------------------------------------------------------------

#: raw ZINC x-id -> element mapping, established empirically on the committed
#: train-only graphs and cross-checked against the committed canonical SMILES
#: element counts (atom-level total discrepancy <= 10 atoms per element over
#: all 10000 molecules; observed <= 5).  id 0/4 = plain/bracket C; the N and O
#: ids cover the charged bracket variants ([NH+], [NH2+], [NH3+], [N+], [N-],
#: [O-]); id 14 = [S-].  The cross-check is re-run at export time.
ATOM_ELEMENT_IDS: Mapping[str, tuple[int, ...]] = {
    "C": (0, 4),
    "N": (2, 8, 10, 11, 12, 13),
    "O": (1, 7),
    "F": (3,),
    "S": (5, 14),
    "Cl": (6,),
    "Br": (9,),
    "I": (15,),
    "P": (16,),
}
#: coarse mutually-exclusive atom-composition groups, first matching priority
ATOM_GROUP_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("halogen", ("F", "Cl", "Br", "I")),
    ("S_or_P", ("S", "P")),
    ("N_or_O", ("N", "O")),
    ("C_only", ()),
)
#: raw ZINC edge id semantics, established empirically:
#: 3 <=> '#' present, 2 <=> '=' or aromatic ring present, 1 = everything else
BOND_TRIPLE_ID = 3
BOND_DOUBLE_AROMATIC_ID = 2
BOND_GROUP_NAMES = ("triple", "double_or_aromatic", "single_only")
ATOM_GROUP_NAMES = tuple(name for name, _ in ATOM_GROUP_PRIORITY)

#: label-cycle groups (identical definition to the triage round)
GROUPS: tuple[tuple[str, Any], ...] = (
    ("k=0", lambda k: k == 0),
    ("k=-1", lambda k: k == -1),
    ("k=-2", lambda k: k == -2),
    ("k<=-3", lambda k: k <= -3),
)

#: the located long-cycle rows keep their historical localisation (triage §4)
LOCATED_GIDS = {
    3775: "T25 class uncovered (singleton class in select, outside the fit pool) "
          "+ label-penalty ordering artifact (q_spotcheck §5)",
    1424: "T25 input conflict (identical T25 vector carries k=0 at fit position "
          "1270) + long-cycle tail beyond T25 expressivity (q_spotcheck §5)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=TRACK_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return {"head": head, "dirty": bool(dirty), "dirty_files": dirty.splitlines()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _mae(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a)))


def round_split_idx(fold: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """fit = the saved fit fold; dev = sorted union(select, confirm)."""
    fit = np.sort(np.asarray(fold["fit_idx"], np.int64))
    dev = np.sort(np.union1d(
        np.asarray(fold["select_idx"], np.int64),
        np.asarray(fold["confirm_idx"], np.int64),
    ))
    return {"fit": fit, "dev": dev}


# ---------------------------------------------------------------------------
# pure-numpy analysis primitives (no torch / no data access)
# ---------------------------------------------------------------------------


def component_stats(e_ell: np.ndarray, e_s: np.ndarray, e_g: np.ndarray) -> dict[str, Any]:
    """Component synthesis for one run on one split (pure numpy).

    e_* = prediction - truth; component MAEs are never summed into a g budget.
    """
    e_ell = np.asarray(e_ell, np.float64)
    e_s = np.asarray(e_s, np.float64)
    e_g = np.asarray(e_g, np.float64)
    if not (e_ell.shape == e_s.shape == e_g.shape):
        raise RuntimeError("component_stats shape mismatch")
    identity_gap = float(np.max(np.abs(e_g - (e_ell + e_s)))) if e_g.size else 0.0
    return {
        "n": int(e_g.size),
        "MAE_ell": _mae(e_ell),
        "MAE_s": _mae(e_s),
        "MAE_g": _mae(e_g),
        "bias_ell": float(np.mean(e_ell)),
        "bias_s": float(np.mean(e_s)),
        "bias_g": float(np.mean(e_g)),
        "opposite_sign_fraction": float(np.mean(np.sign(e_ell) * np.sign(e_s) < 0)),
        "triangle_gap": float(np.mean(np.abs(e_ell) + np.abs(e_s) - np.abs(e_g))),
        "sum_component_MAE": _mae(e_ell) + _mae(e_s),
        "abs_s_share": float(
            np.abs(e_s).sum() / (np.abs(e_s).sum() + np.abs(e_ell).sum())
        ),
        "identity_gap_e_g_minus_e_ell_minus_e_s": identity_gap,
    }


def k_group_component_rows(
    k: np.ndarray,
    e_ell: np.ndarray,
    e_s: np.ndarray,
    e_g: np.ndarray,
    n_total: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-k-group component metrics; contributions normalised by the FULL
    split row count ``n_total`` (never the in-group n); exact add-back
    residuals (mirrors the triage round's contribution convention)."""
    rows: list[dict[str, Any]] = []
    sums = {"s": 0.0, "ell": 0.0, "g": 0.0}
    for name, pred in GROUPS:
        m = pred(k)
        cs = float(np.abs(e_s)[m].sum() / n_total)
        ce = float(np.abs(e_ell)[m].sum() / n_total)
        cg = float(np.abs(e_g)[m].sum() / n_total)
        rows.append({
            "group": name, "n": int(m.sum()),
            "MAE_s_in_group": _mae(e_s[m]),
            "MAE_ell_in_group": _mae(e_ell[m]),
            "MAE_g_in_group": _mae(e_g[m]),
            "C_s": cs, "C_ell": ce, "C_g": cg,
        })
        sums["s"] += cs
        sums["ell"] += ce
        sums["g"] += cg
    residual = {
        "groups_cover_all_rows": sum(int(pred(k).sum()) for _, pred in GROUPS) == int(k.size),
        "addback_C_s_residual": sums["s"] - _mae(e_s),
        "addback_C_ell_residual": sums["ell"] - _mae(e_ell),
        "addback_C_g_residual": sums["g"] - _mae(e_g),
    }
    return rows, residual


def atom_composition_of(x: np.ndarray) -> dict[str, bool]:
    """Coarse element presence of one molecule from its raw atom ids."""
    return {
        elem: bool(np.isin(x, ids).any()) for elem, ids in ATOM_ELEMENT_IDS.items()
    }


def atom_group_of(x: np.ndarray) -> str:
    """Coarse mutually-exclusive atom-composition group (first match wins)."""
    present = atom_composition_of(x)
    for name, elems in ATOM_GROUP_PRIORITY:
        if any(present[e] for e in elems):
            return name
    return "C_only"


def bond_group_of(edge_types: np.ndarray) -> str:
    """Coarse bond-composition group from raw edge type ids."""
    kinds = set(np.asarray(edge_types).tolist())
    if BOND_TRIPLE_ID in kinds:
        return "triple"
    if BOND_DOUBLE_AROMATIC_ID in kinds:
        return "double_or_aromatic"
    if kinds == {1}:
        return "single_only"
    raise RuntimeError(f"unexpected raw edge ids {sorted(kinds)}")


def node_tercile_bounds(node_counts: np.ndarray) -> tuple[float, float]:
    """Pre-registered rule: 1/3 and 2/3 quantiles over ALL 10000 train rows."""
    b1, b2 = np.quantile(np.asarray(node_counts, np.float64), [1.0 / 3.0, 2.0 / 3.0])
    return float(b1), float(b2)


def node_tercile_group(node_count: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    b1, b2 = bounds
    nc = np.asarray(node_count, np.float64)
    return np.where(nc <= b1, 0, np.where(nc <= b2, 1, 2)).astype(np.int64)


def smiles_elements(smiles: str) -> dict[str, int]:
    """Element counts from a canonical SMILES string (organic subset; aromatic
    lowercase folded onto the uppercase element; two-letter halogens handled).

    Presence/count heuristic used ONLY for the pre-registered cross-validation
    of the raw x-id mapping (not for any feature).
    """
    two_letter = {"Cl", "Br"}
    fold = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P"}
    out: dict[str, int] = {}
    i = 0
    while i < len(smiles):
        c = smiles[i]
        if c.isalpha():
            pair = smiles[i:i + 2]
            if pair in two_letter:
                out[pair] = out.get(pair, 0) + 1
                i += 2
                continue
            elem = fold.get(c, c)
            out[elem] = out.get(elem, 0) + 1
            i += 1
            continue
        i += 1
    return out


def element_mapping_crosscheck(
    raw_atom_ids: Sequence[np.ndarray], smiles: Sequence[str]
) -> dict[str, Any]:
    """Atom-level cross-validation of ATOM_ELEMENT_IDS vs SMILES counts."""
    per_element_ids = {e: 0 for e in ATOM_ELEMENT_IDS}
    per_element_smiles = {e: 0 for e in ATOM_ELEMENT_IDS}
    for x, smi in zip(raw_atom_ids, smiles):
        for elem, ids in ATOM_ELEMENT_IDS.items():
            per_element_ids[elem] += int(np.isin(x, ids).sum())
        counts = smiles_elements(str(smi))
        for elem in ATOM_ELEMENT_IDS:
            per_element_smiles[elem] += int(counts.get(elem, 0))
    diffs = {e: abs(per_element_ids[e] - per_element_smiles[e]) for e in ATOM_ELEMENT_IDS}
    return {
        "atoms_by_id_mapping": per_element_ids,
        "atoms_by_smiles": per_element_smiles,
        "abs_diff_per_element": diffs,
        "max_abs_diff": int(max(diffs.values())),
        "passed": bool(max(diffs.values()) <= 10),
    }


def group_resample_indices(
    group_codes: np.ndarray, n_draws: int, seed: int
) -> list[np.ndarray]:
    """Canonical-SMILES-group bootstrap: resample whole groups
    (``default_rng(seed)``); per-draw concatenated row indices.  Codes are
    remapped to a dense partition internally, so subgroup-restricted code
    arrays (with gaps) work unchanged."""
    codes = np.asarray(group_codes, np.int64)
    unique = np.unique(codes)
    n_groups = int(unique.size)
    dense = np.searchsorted(unique, codes)
    members = [np.flatnonzero(dense == g) for g in range(n_groups)]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(int(n_draws), n_groups))
    return [np.concatenate([members[g] for g in row]) for row in picks]


def paired_group_bootstrap(
    err_by_run: Mapping[str, np.ndarray],
    group_codes: np.ndarray,
    *,
    dict_arm: str = "DICT",
    raw_arm: str = "RAW",
    seeds: Sequence[int] = SEEDS,
    n_draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """DICT - RAW paired MAE bootstrap over canonical-SMILES groups.

    The SAME group resampling is shared by both arms and both seeds; each draw
    first averages the two seeds' MAE differences before the CI is taken (the
    consumer-replacement round's scheme).  The molecular bootstrap carries no
    training-seed or basis uncertainty.
    """
    draws = group_resample_indices(group_codes, n_draws, seed)
    diffs = np.empty(len(draws), np.float64)
    for i, rows in enumerate(draws):
        per_seed = []
        for s in seeds:
            d = _mae(err_by_run[f"{dict_arm}_s{s}"][rows])
            r = _mae(err_by_run[f"{raw_arm}_s{s}"][rows])
            per_seed.append(d - r)
        diffs[i] = float(np.mean(per_seed))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "n_draws": int(n_draws),
        "seed": int(seed),
        "directions_by_seed": [
            float(_mae(err_by_run[f"{dict_arm}_s{s}"]) - _mae(err_by_run[f"{raw_arm}_s{s}"]))
            for s in seeds
        ],
    }


def subgroup_contrast_bootstrap(
    e: np.ndarray,
    subgroup_mask: np.ndarray,
    group_codes: np.ndarray,
    *,
    n_draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Subgroup-vs-complement MAE contrast with a group bootstrap CI."""
    subgroup_mask = np.asarray(subgroup_mask, bool)
    if subgroup_mask.sum() == 0 or (~subgroup_mask).sum() == 0:
        raise RuntimeError("subgroup contrast needs both sides non-empty")
    draws = group_resample_indices(group_codes, n_draws, seed)
    diffs = np.empty(len(draws), np.float64)
    for i, rows in enumerate(draws):
        m = subgroup_mask[rows]
        diffs[i] = _mae(e[rows][m]) - _mae(e[rows][~m])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "n_subgroup": int(subgroup_mask.sum()),
        "n_complement": int((~subgroup_mask).sum()),
        "n_draws": int(n_draws),
        "seed": int(seed),
    }


def paired_delta_within(
    err_by_run: Mapping[str, np.ndarray], mask: np.ndarray, *, seeds: Sequence[int] = SEEDS
) -> float:
    """DICT - RAW MAE contrast within ``mask`` (two-seed mean of per-seed
    differences; the fixed-sample companion of the bootstrap)."""
    per_seed = []
    for s in seeds:
        per_seed.append(
            _mae(err_by_run[f"DICT_s{s}"][mask]) - _mae(err_by_run[f"RAW_s{s}"][mask])
        )
    return float(np.mean(per_seed))


# ---------------------------------------------------------------------------
# stage: export (label-free)
# ---------------------------------------------------------------------------


def _state_hash(path: Path) -> str:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    blob = b"".join(
        f"{k}|{tuple(v.shape)}|{v.detach().numpy().tobytes()}".encode()
        if hasattr(v, "numpy") else str(v).encode()
        for k, v in state.items()
    )
    return hashlib.sha256(blob).hexdigest()


def stage_export(*, out_dir: Path = RESULTS_DIR, device_name: str = "cpu", log: Any = print) -> dict[str, Any]:
    """Restore the four frozen soups, replay-check, verify the reader identity,
    export per-row dev component predictions; fix the label-free structure
    groupings with their validation checks.  targets.npz is never opened."""
    started = time.perf_counter()
    import torch

    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from tracks.ksvd.experiments.luyin16 import (
        zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
    )
    from tracks.ksvd.experiments.luyin16 import zinc_cssd_consumer_replacement_v1 as repl
    from tracks.ksvd.experiments.luyin16.zinc_cssd_graph_information_sufficiency_v1 import (
        _restore_model,
    )

    device = torch.device(device_name)
    ro = repl.load_round_objects()
    _prep_meta, fit_data, dev_data = repl.build_round_data(ro)
    parts_full = repl.frozen_basis_parts(ro["basis"])
    basis_parts = {k: parts_full[k] for k in ("U", "common_rms", "Dbar")}
    payload = prev.TuplePayload(ro["objects"]["payload_arrays"])
    kappa = float(ro["objects"]["kappa"]["kappa_M"])
    dev_idx = np.asarray(ro["dev_idx"], np.int64)
    fit_idx = np.asarray(ro["fit_idx"], np.int64)

    # -- label-free structure groupings (fixed here, before any label read) --
    raw_graphs = prev.load_raw_train_graphs()
    if len(raw_graphs) != 10000:
        raise RuntimeError("raw train graph count mismatch")
    node_counts = np.array([int(x.shape[0]) for x, _ei, _ea in raw_graphs], np.int64)
    atom_group = np.array([atom_group_of(x) for x, _ei, _ea in raw_graphs], dtype=object)
    bond_group = np.array([bond_group_of(ea) for _x, _ei, ea in raw_graphs], dtype=object)
    bounds = node_tercile_bounds(node_counts)
    tercile = node_tercile_group(node_counts, bounds)
    with np.load(SMILES_TABLE, allow_pickle=False) as z:
        smiles = np.asarray(z["smiles"])
    mapping_check = element_mapping_crosscheck([g[0] for g in raw_graphs], smiles)
    if not mapping_check["passed"]:
        raise RuntimeError(f"atom mapping cross-check failed: {mapping_check}")

    # canonical-SMILES groups must not cross fit/dev (historical property)
    fit_groups = {str(smiles[i]) for i in fit_idx.tolist()}
    dev_groups = {str(smiles[i]) for i in dev_idx.tolist()}
    if fit_groups & dev_groups:
        raise RuntimeError("canonical-SMILES groups cross fit/dev")

    np.savez_compressed(
        out_dir / "structure_assignments.npz",
        node_count=node_counts,
        node_tercile=tercile,
        node_tercile_bounds=np.array(bounds, np.float64),
        atom_group=atom_group.astype(str),
        bond_group=bond_group.astype(str),
        fit_idx=fit_idx,
        dev_idx=dev_idx,
        row_index=np.arange(10000, dtype=np.int64),
    )

    # -- frozen-soup replay + component export --
    replay: dict[str, Any] = {}
    # positional anchor: dev rows are ordered by dev_idx; local_mol_id IS the
    # global train row position (verified identity from the witness round).
    # gid (canonical_group_id) is a DIFFERENT id space - the exported gid
    # column is copied verbatim from the promoted dev npz and re-checked
    # against targets in the analyze stage.
    dev_row_positions = np.array(
        [int(r.local_mol_id.reshape(-1)[0].item()) for r in dev_data], np.int64
    )
    if not np.array_equal(dev_row_positions, dev_idx):
        raise RuntimeError("dev data rows are not ordered by dev_idx")
    for name in RUN_NAMES:
        arm, seed_s = name.rsplit("_s", 1)
        model, _run = _restore_model(
            arm, int(seed_s), basis_parts=basis_parts, payload=payload, kappa=kappa
        )
        saved_fit = dict(np.load(CONSUMER_DIR / "runs" / name / "fit_predictions.npz", allow_pickle=False))
        saved_dev = dict(np.load(CONSUMER_DIR / "runs" / name / "dev_predictions.npz", allow_pickle=False))
        fit_preds = repl._predict_rows(model, fit_data, device)
        dev_preds = repl._predict_rows(model, dev_data, device)
        entry = {
            "fit_h_max_abs_diff": float(np.abs(fit_preds["h"] - saved_fit["h"]).max()),
            "fit_ell_hat_max_abs_diff": float(np.abs(fit_preds["ell_hat"] - saved_fit["ell_hat"]).max()),
            "fit_s_hat_max_abs_diff": float(np.abs(fit_preds["s_hat"] - saved_fit["s_hat"]).max()),
            "dev_h_max_abs_diff": float(np.abs(dev_preds["h"] - saved_dev["h"]).max()),
            "fit_identity_max_abs": float(
                np.abs(fit_preds["h"] - (fit_preds["ell_hat"] + fit_preds["s_hat"])).max()
            ),
            "dev_identity_max_abs": float(
                np.abs(dev_preds["h"] - (dev_preds["ell_hat"] + dev_preds["s_hat"])).max()
            ),
            "dev_rows_match_dev_idx": bool((dev_row_positions == dev_idx).all()),
        }
        failures = {k: v for k, v in entry.items() if isinstance(v, float) and v > REPLAY_TOL}
        failures.update({
            k: v for k, v in entry.items()
            if k.endswith("identity_max_abs") and v > IDENTITY_TOL
        })
        if failures or not entry["dev_rows_match_dev_idx"]:
            raise RuntimeError(f"{name} export replay/identity failure: {failures}")
        np.savez_compressed(
            out_dir / f"dev_components_{name}.npz",
            gid=saved_dev["gid"].astype(np.int64),
            h=dev_preds["h"],
            ell_hat=dev_preds["ell_hat"],
            s_hat=dev_preds["s_hat"],
        )
        replay[name] = entry
        log(f"[export] {name}: dev_h_diff {entry['dev_h_max_abs_diff']:.2e}, "
            f"dev_identity {entry['dev_identity_max_abs']:.2e}")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "export",
        "label_free": True,
        "device": str(device),
        "basis_sha256": parts_full["hashes"],
        "soup_state_sha256": {
            name: _state_hash(CONSUMER_DIR / "runs" / name / "soup_state.pt")
            for name in RUN_NAMES
        },
        "structure_groupings": {
            "node_tercile_bounds": list(bounds),
            "node_tercile_sizes": [int((tercile == i).sum()) for i in range(3)],
            "atom_group_sizes": {g: int((atom_group == g).sum()) for g in ATOM_GROUP_NAMES},
            "bond_group_sizes": {g: int((bond_group == g).sum()) for g in BOND_GROUP_NAMES},
            "atom_mapping_crosscheck": mapping_check,
            "canonical_smiles_groups_fit": len(fit_groups),
            "canonical_smiles_groups_dev": len(dev_groups),
            "canonical_smiles_groups_cross_fit_dev": 0,
        },
        "replay_and_identity": replay,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(out_dir / "export.json", manifest)
    log(f"[export] done in {manifest['seconds']:.1f}s; "
        f"atom-mapping max|diff| = {mapping_check['max_abs_diff']} atoms over 10000 molecules")
    return manifest


# ---------------------------------------------------------------------------
# stage: analyze (labels)
# ---------------------------------------------------------------------------


def _load_targets() -> dict[str, np.ndarray]:
    return dict(np.load(SOURCE_DIR / "targets.npz", allow_pickle=False))


def _load_run_errors(out_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Per-run error arrays per split: {run: {split: {component: e, k: k}}}.

    fit side from the promoted fit_predictions.npz (read-only reuse of the
    saved ell_hat/s_hat/h); dev side from this round's label-free export.
    gid alignment and the strict component identity are re-verified here.
    """
    targets = _load_targets()
    fold = dict(np.load(SOURCE_DIR / "fold.npz", allow_pickle=False))
    splits = round_split_idx(fold)
    errors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    alignment: dict[str, Any] = {}
    for split, idx in splits.items():
        gid_t = targets["gid"][idx].astype(np.int64)
        ell_t, s_t, g_t = targets["ell"][idx], targets["s"][idx], targets["g"][idx]
        k_t, y_t, c_t = targets["k"][idx], targets["y"][idx], targets["c"][idx]
        alignment[f"{split}_y_equals_g_plus_c_max_abs"] = float(np.max(np.abs(y_t - (g_t + c_t))))
        alignment[f"{split}_g_equals_ell_plus_s_max_abs"] = float(np.max(np.abs(g_t - (ell_t + s_t))))
        if alignment[f"{split}_y_equals_g_plus_c_max_abs"] > TARGET_IDENTITY_TOL:
            raise RuntimeError(f"targets y != g + c on {split}")
        if alignment[f"{split}_g_equals_ell_plus_s_max_abs"] > TARGET_IDENTITY_TOL:
            raise RuntimeError(f"targets g != ell + s on {split}")
        for name in RUN_NAMES:
            if split == "fit":
                z = dict(np.load(CONSUMER_DIR / "runs" / name / "fit_predictions.npz", allow_pickle=False))
            else:
                z = dict(np.load(out_dir / f"dev_components_{name}.npz", allow_pickle=False))
            if not np.array_equal(z["gid"].astype(np.int64), gid_t):
                raise RuntimeError(f"gid misaligned for {name} on {split}")
            e_ell = np.asarray(z["ell_hat"], np.float64) - ell_t
            e_s = np.asarray(z["s_hat"], np.float64) - s_t
            e_g = np.asarray(z["h"], np.float64) - g_t
            ig = float(np.max(np.abs(e_g - (e_ell + e_s))))
            alignment[f"{split}_{name}_identity_gap"] = ig
            if ig > IDENTITY_TOL:
                raise RuntimeError(f"component identity violated for {name}/{split}: {ig}")
            errors.setdefault(name, {})[split] = {
                "e_ell": e_ell, "e_s": e_s, "e_g": e_g, "k": k_t,
            }
    return errors, alignment


def _two_seed_arm_mean(per_run: Mapping[str, Mapping[str, Any]], key: str, arm: str) -> float:
    return float(np.mean([per_run[f"{arm}_s{seed}"][key] for seed in SEEDS]))


def _top_n_share(
    e_by_run: Mapping[str, np.ndarray], arm: str, top_n: int
) -> dict[str, Any]:
    """Top-N concentration of one error component over the current rows
    (two-seed mean |e| per row), plus the per-row detail list."""
    mean_abs = np.mean(
        [np.abs(e_by_run[f"{arm}_s{seed}"]) for seed in SEEDS], axis=0
    )
    total = float(mean_abs.sum())
    order = np.argsort(-mean_abs, kind="stable")[:top_n]
    top_share = float(mean_abs[order].sum() / total)
    detail = [
        {"rank": int(r), "position": int(pos), "two_seed_mean_abs": float(mean_abs[pos])}
        for r, pos in enumerate(order.tolist(), start=1)
    ]
    return {
        "top_n": int(top_n),
        "n_rows": int(mean_abs.size),
        "uniform_share_reference": float(top_n / mean_abs.size),
        "top_share": top_share,
        "rows": detail,
    }


def _subgroup_tables(
    k0_err: Mapping[str, Mapping[str, np.ndarray]],
    k0_masks: Mapping[str, Mapping[str, np.ndarray]],
    k0_codes: np.ndarray,
    n_dev_total: int,
    n_k0: int,
    *,
    with_bootstrap: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """k=0 subgroup tables (dev primary) + bootstrap contrasts.

    ``k0_err`` maps run -> component -> e restricted to k=0 dev rows;
    ``k0_masks`` maps spec -> group -> boolean mask over the k=0 dev rows.
    """
    tables: dict[str, list[dict[str, Any]]] = {}
    boot: dict[str, Any] = {}
    for spec, groups in k0_masks.items():
        rows_out: list[dict[str, Any]] = []
        for gname, mask in groups.items():
            mask = np.asarray(mask, bool)
            if mask.sum() == 0:
                continue
            row: dict[str, Any] = {
                "spec": spec, "group": gname, "n": int(mask.sum()),
                "n_fraction": float(mask.sum() / n_k0),
            }
            for arm in ARMS:
                for comp in ("e_s", "e_ell", "e_g"):
                    per_seed = [_mae(k0_err[f"{arm}_s{seed}"][comp][mask]) for seed in SEEDS]
                    row[f"MAE_{comp[2:]}_{arm}"] = float(np.mean(per_seed))
                    row[f"MAE_{comp[2:]}_{arm}_by_seed"] = [float(v) for v in per_seed]
                # contribution of the group's |e_s| to the FULL dev N
                row[f"C_s_{arm}"] = float(np.mean([
                    np.abs(k0_err[f"{arm}_s{seed}"]["e_s"][mask]).sum() / n_dev_total
                    for seed in SEEDS
                ]))
                # share of the group's |e_s| within all k=0 |e_s| of the arm
                row[f"share_of_k0_abs_s_{arm}"] = float(np.mean([
                    np.abs(k0_err[f"{arm}_s{seed}"]["e_s"][mask]).sum()
                    / max(np.abs(k0_err[f"{arm}_s{seed}"]["e_s"]).sum(), 1e-300)
                    for seed in SEEDS
                ]))
            for comp in ("e_s", "e_ell"):
                comp_arr = {run: k0_err[run][comp][mask] for run in RUN_NAMES}
                row[f"Delta_MAE_{comp[2:]}_DICT_minus_RAW"] = paired_delta_within(
                    comp_arr, np.ones(int(mask.sum()), bool)
                )
                if with_bootstrap:
                    boot[f"{spec}/{gname}/{comp}"] = paired_group_bootstrap(
                        comp_arr, k0_codes[mask]
                    )
                row[f"Delta_MAE_{comp[2:]}_by_seed"] = [
                    float(_mae(k0_err[f"DICT_s{seed}"][comp][mask]) - _mae(k0_err[f"RAW_s{seed}"][comp][mask]))
                    for seed in SEEDS
                ]
            rows_out.append(row)
        tables[spec] = rows_out
        # subgroup-vs-complement MAE_s contrast per arm (fit-side stability view
        # is handled by the caller through fit tables; bootstrap here is dev)
        if with_bootstrap:
            for arm in ARMS:
                for gname, mask in groups.items():
                    mask = np.asarray(mask, bool)
                    if mask.sum() == 0 or (~mask).sum() == 0:
                        continue
                    e_two_seed = np.mean(
                        [np.abs(k0_err[f"{arm}_s{seed}"]["e_s"]) for seed in SEEDS], axis=0
                    )
                    boot[f"{spec}/{gname}/contrast_e_s/{arm}"] = subgroup_contrast_bootstrap(
                        e_two_seed, mask, k0_codes
                    )
    return tables, boot


def analyze(out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """The pre-registered labeled analysis.  dev is exploratory localization
    only (historical development set), never a confirmation."""
    started = time.perf_counter()
    out_dir = Path(out_dir)
    if not (out_dir / "structure_assignments.npz").exists():
        raise RuntimeError("run the label-free export stage first")
    errors, alignment = _load_run_errors(out_dir)
    assign = dict(np.load(out_dir / "structure_assignments.npz", allow_pickle=False))
    targets = _load_targets()
    fold = dict(np.load(SOURCE_DIR / "fold.npz", allow_pickle=False))
    splits = round_split_idx(fold)
    with np.load(SMILES_TABLE, allow_pickle=False) as z:
        smiles = np.asarray(z["smiles"])

    # canonical-SMILES group codes per split row (dense 0..G-1 partition)
    group_codes: dict[str, np.ndarray] = {}
    for split, idx in splits.items():
        labels = [str(smiles[i]) for i in idx.tolist()]
        order = {s: i for i, s in enumerate(sorted(set(labels)))}
        group_codes[split] = np.array([order[s] for s in labels], np.int64)

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "alignment": alignment,
        "splits": {},
        "k_groups": {},
        "k0_subgroups": {},
        "k0_concentration": {},
        "k0_bootstrap": {},
        "fit_stability": {},
        "seconds": 0.0,
    }

    for split, idx in splits.items():
        n_total = int(idx.size)
        k_split = errors[RUN_NAMES[0]][split]["k"]

        # ---- (1) per-run component synthesis + two-seed/arm means ----
        per_run = {
            name: component_stats(
                errors[name][split]["e_ell"], errors[name][split]["e_s"], errors[name][split]["e_g"]
            )
            for name in RUN_NAMES
        }
        keys = ("MAE_ell", "MAE_s", "MAE_g", "bias_ell", "bias_s", "bias_g",
                "opposite_sign_fraction", "triangle_gap", "abs_s_share")
        means = {key: {arm: _two_seed_arm_mean(per_run, key, arm) for arm in ARMS} for key in keys}
        summary["splits"][split] = {"n_total": n_total, "per_run": per_run, "two_seed_mean": means}

        # ---- k-group contributions (full-N normalisation, exact add-back) ----
        kgroup: dict[str, Any] = {}
        for name in RUN_NAMES:
            rows, resid = k_group_component_rows(
                k_split, errors[name][split]["e_ell"], errors[name][split]["e_s"],
                errors[name][split]["e_g"], n_total,
            )
            if not resid["groups_cover_all_rows"] or any(
                abs(resid[f"addback_C_{c}_residual"]) >= 1e-9 for c in ("s", "ell", "g")
            ):
                raise RuntimeError(f"k-group add-back failed for {name}/{split}: {resid}")
            kgroup[name] = {"rows": rows, "addback": resid}
        kgroup_arm_mean: dict[str, Any] = {}
        for arm in ARMS:
            per_seed_rows = [kgroup[f"{arm}_s{seed}"]["rows"] for seed in SEEDS]
            merged = []
            for gi in range(len(per_seed_rows[0])):
                row = {"group": per_seed_rows[0][gi]["group"]}
                for key in per_seed_rows[0][gi]:
                    if key == "group":
                        continue
                    row[key] = float(np.mean([r[gi][key] for r in per_seed_rows]))
                merged.append(row)
            kgroup_arm_mean[arm] = merged
        summary["k_groups"][split] = {"per_run": kgroup, "two_seed_mean": kgroup_arm_mean}

        if split != "dev":
            continue

        # ================= dev-only: k=0 attribution =================
        k0 = k_split == 0
        n_k0 = int(k0.sum())
        gid0 = targets["gid"][idx][k0].astype(np.int64)
        k0_codes = group_codes[split][k0]
        k0_err = {
            run: {c: errors[run][split][c][k0] for c in ("e_ell", "e_s", "e_g")}
            for run in RUN_NAMES
        }

        # ---- (2) subgroup tables: node tercile / atom group / bond group ----
        k0_masks = {
            "node_tercile": {f"tercile_{i}": (assign["node_tercile"][idx][k0] == i) for i in range(3)},
            "atom_group": {g: (assign["atom_group"][idx][k0] == g) for g in ATOM_GROUP_NAMES},
            "bond_group": {g: (assign["bond_group"][idx][k0] == g) for g in BOND_GROUP_NAMES},
        }
        tables, boot = _subgroup_tables(
            k0_err, k0_masks, k0_codes, n_total, n_k0, with_bootstrap=True
        )
        summary["k0_subgroups"] = {
            "n_k0": n_k0,
            "tables": tables,
            "definitions": {
                "node_tercile_bounds": [float(v) for v in assign["node_tercile_bounds"]],
                "atom_group_priority": [g for g, _ in ATOM_GROUP_PRIORITY],
                "bond_group_semantics": "id3<=>'#', id2<=>'=' or aromatic, id1-only = single",
            },
        }
        summary["k0_bootstrap"] = {
            "overall": {
                comp: paired_group_bootstrap(
                    {run: k0_err[run][comp] for run in RUN_NAMES}, k0_codes
                )
                for comp in ("e_s", "e_ell")
            },
            "subgroup_delta": {k: v for k, v in boot.items() if "/e_s" in k or "/e_ell" in k},
            "subgroup_contrast_e_s": {
                k: v for k, v in boot.items() if "/contrast_e_s/" in k
            },
        }

        # ---- (3) concentration vs dispersion (dev k=0 and full dev) ----
        conc: dict[str, Any] = {}
        for scope, sel in (("k0", k0), ("full", np.ones(n_total, bool))):
            scoped = {
                run: {c: errors[run][split][c][sel] for c in ("e_ell", "e_s", "e_g")}
                for run in RUN_NAMES
            }
            conc[scope] = {
                comp: {arm: _top_n_share({r: scoped[r][comp] for r in RUN_NAMES}, arm, TOP_N)
                       for arm in ARMS}
                for comp in ("e_s", "e_ell", "e_g")
            }
        # annotate top-k0 e_s rows with gid/k/group assignments
        k0_assign = {
            "node_tercile": assign["node_tercile"][idx][k0],
            "atom_group": assign["atom_group"][idx][k0],
            "bond_group": assign["bond_group"][idx][k0],
        }
        for arm in ARMS:
            detail = conc["k0"]["e_s"][arm]["rows"]
            for row in detail:
                pos = row["position"]
                row["gid"] = int(gid0[pos])
                row["node_tercile"] = str(k0_assign["node_tercile"][pos])
                row["atom_group"] = str(k0_assign["atom_group"][pos])
                row["bond_group"] = str(k0_assign["bond_group"][pos])
                if int(gid0[pos]) in LOCATED_GIDS:
                    row["located"] = LOCATED_GIDS[int(gid0[pos])]
        summary["k0_concentration"] = conc

        # ---- CSV outputs ----
        _write_subgroup_csv(out_dir, tables)
        _write_concentration_csv(out_dir, conc, gid0)

    # ---- source manifest + report manifest ----
    inputs = {
        "targets": SOURCE_DIR / "targets.npz",
        "fold": SOURCE_DIR / "fold.npz",
        "canonical_smiles": SMILES_TABLE,
        "export_json": out_dir / "export.json",
        "structure_assignments": out_dir / "structure_assignments.npz",
    }
    for name in RUN_NAMES:
        inputs[f"fit_predictions_{name}"] = CONSUMER_DIR / "runs" / name / "fit_predictions.npz"
        inputs[f"dev_components_{name}"] = out_dir / f"dev_components_{name}.npz"
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "analyze",
        "git_state": _git_state(),
        "inputs_sha256": {k: sha256_file(v) for k, v in inputs.items()},
        "prior_round_pointers": {
            "consumer_replacement": "results/zinc_cssd_consumer_replacement_v1/REPORT.md",
            "performance_triage": "results/zinc_cssd_performance_triage_v1/REPORT.md",
            "graph_information_sufficiency": "results/zinc_cssd_graph_information_sufficiency_v1/REPORT.md",
            "chemistry_component_supervision": "results/zinc_chemistry_component_supervision_seed0_v1/REPORT.md",
            "local_dictionary_component_supervision": "results/zinc_local_dictionary_component_supervision_seed0_v1/REPORT.md",
            "information_flow_note": "notes/zinc_cssd_graph_information_sufficiency_v1.md",
        },
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(out_dir / "source_manifest.json", manifest)
    summary["seconds"] = manifest["seconds"]
    _write_json(out_dir / "component_summary.json", summary)
    log(f"[analyze] done in {manifest['seconds']:.1f}s")
    return summary


def _write_subgroup_csv(
    out_dir: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    cols = [
        "spec", "group", "n", "n_fraction",
        "MAE_s_RAW", "MAE_s_DICT", "MAE_ell_RAW", "MAE_ell_DICT",
        "MAE_g_RAW", "MAE_g_DICT",
        "Delta_MAE_s_DICT_minus_RAW", "Delta_MAE_ell_DICT_minus_RAW",
        "MAE_s_RAW_by_seed", "MAE_s_DICT_by_seed",
        "Delta_MAE_s_by_seed", "Delta_MAE_ell_by_seed",
        "C_s_RAW", "C_s_DICT",
        "share_of_k0_abs_s_RAW", "share_of_k0_abs_s_DICT",
    ]
    with open(out_dir / "subgroup_tables.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rows in tables.values():
            for row in rows:
                w.writerow({k: row.get(k, "") for k in cols})


def _write_concentration_csv(
    out_dir: Path,
    conc: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    gid0: np.ndarray,
) -> None:
    with open(out_dir / "concentration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "component", "arm", "rank", "position", "gid",
                    "two_seed_mean_abs", "node_tercile", "atom_group", "bond_group", "located"])
        for scope in ("k0", "full"):
            for comp in ("e_s", "e_ell", "e_g"):
                for arm in ARMS:
                    for row in conc[scope][comp][arm]["rows"]:
                        gid = int(gid0[row["position"]]) if scope == "k0" else ""
                        w.writerow([
                            scope, comp, arm, row["rank"], row["position"], gid,
                            row["two_seed_mean_abs"],
                            row.get("node_tercile", ""), row.get("atom_group", ""),
                            row.get("bond_group", ""), row.get("located", ""),
                        ])
