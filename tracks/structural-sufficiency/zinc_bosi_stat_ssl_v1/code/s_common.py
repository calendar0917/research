"""Shared paths, frozen constants and I/O helpers for zinc_bosi_stat_ssl_v1.

This round adds **only** a self-supervised *initialisation* of three BOSI
modules (``half_encoder``, ``side``, ``compose``).  No model, data, split,
recipe or inference path is changed.  Every frozen artefact of the previous
rounds is loaded read-only and hash-verified:

* ``zinc_bosi_architecture_v0/cache/bosi_rel_train.pt``  relation cache
  (content sha256 ``c698a2f3...``), produced by the frozen BOSI v0 code;
* ``zinc_bosi_partner_content_v1/code/{bosi_common,bosi_data,bosi_models}.py``
  the frozen BOSI implementation (``arm="B"`` is the historical BOSI-B);
* ``bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt`` the
  train-only masked-SSL atom/bond checkpoints BOSI-B already used.

Official valid / test are never constructed, loaded, predicted or scored by any
module here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
_REPO = _SUFF.parents[1]

# the frozen BOSI implementation lives in the previous round; it is imported,
# never copied or edited, so the forward is byte-identical by construction.
_V1 = _SUFF / "zinc_bosi_partner_content_v1" / "code"
_V0 = _SUFF / "zinc_bosi_architecture_v0"
#: the frozen upstream feature builder (audit reference only)
_SRO = _REPO / "tracks/structure-contribution/code"
for _p in (str(_HERE), str(_V1), str(_SRO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bosi_common as BC  # noqa: E402  (also puts ad_common on the path)
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402

ROUND_ID = "zinc-bosi-stat-ssl-v1"
ROUND = _ROUND
V0 = _V0
V1 = _SUFF / "zinc_bosi_partner_content_v1"
CONFIGS = _ROUND / "configs"
# remote jobs write into a gitignored directory so the compute checkout never
# becomes dirty; the curated local copy still lives in analysis/.
ANALYSIS = Path(os.environ.get("BOSI_SSL_ANALYSIS_DIR", str(_ROUND / "analysis")))
RUNS = Path(os.environ.get("BOSI_SSL_RUNS_DIR", str(_ROUND / "runs")))
RUNS_SMOKE = _ROUND / "runs_smoke"
CACHE = _ROUND / "cache"

# ---------------------------------------------------------------------------
# frozen data / artefact identity
# ---------------------------------------------------------------------------
DATA_CACHE = BC.DATA_CACHE
REL_CACHE = BC.CACHE / "bosi_rel_train.pt"
REL_CACHE_CONTENT_SHA256 = "c698a2f30e0fe9f2fc51e79f37156e81da56f44b6a47843293dddb308ceda2b7"
DATA_CACHE_SHA256 = "d85d9875878a3cc0df27aeb8ad80e6cddf809e9d8265d77ef423a91bef4e19e3"

#: initial full-model state of the historical BOSI-B / BOSI-R runs per seed
HISTORICAL_INIT_SHA256 = {
    0: "ceeb598f3a51761ae46cf761dfe7de3b4c08d74c2c5b7c1a3c823d75e53e9a58",
    1: "c93ea85b10aa7c88e1bde9aee90be7d8857fab1418cf4cec5505dbd5dded39b4",
}
#: historical BOSI-B Monitor MAE (the frozen comparison level of this round)
HISTORICAL_B_MONITOR = {0: 0.1245047393636778, 1: 0.12240047505032271}
#: historical BOSI-B Locked MAE (BOSI v0 freeze / locked_read, hash-verified)
HISTORICAL_B_LOCKED = {0: 0.082449, 1: 0.081491}

# ---------------------------------------------------------------------------
# frozen pretext design (PROTOCOL.md §2); nothing here is searchable
# ---------------------------------------------------------------------------
ATOM_MASK_TOKEN = 27          # unused atom class in the frozen 28-way vocabulary
BOND_MASK_TOKEN = 0           # unused bond class in the frozen 4-slot vocabulary
ATOM_MASK_RATIO = 0.15
BOND_MASK_RATIO = 0.15
PRETRAIN_UPDATES = 7_560      # frozen full budget (== BJOC masked pretraining)
PRETRAIN_PILOT_UPDATES = 1_890  # frozen pilot budget (25 %, and a prefix of the stream)
PRETRAIN_BATCH = 128
PRETRAIN_LR = 1.0e-3
PRETRAIN_WD = 1.0e-5
PRETRAIN_CLIP = 5.0
PRETRAIN_SHUFFLE_OFFSET = 202  # as in bjoc_masked_pretraining_v1
HEAD_HIDDEN = 160
LOSS_WEIGHT_ATOM = 1.0
LOSS_WEIGHT_BOND = 1.0

#: fixed Fit8k probe set for the pilot mechanism read (never Monitor/Locked)
PROBE_N = 1024

#: pretraining checkpoint file names
def pretrain_state_name(seed: int) -> str:
    return f"stat_ssl_state_s{int(seed)}.pt"


# ---------------------------------------------------------------------------
# 108-d atom feature layout (frozen by structural_relation_objects_v1)
# ---------------------------------------------------------------------------
ATOM_TYPE_BLOCK = slice(0, 28)          # own atom type one-hot (type-dependent)
ATOM_TOPO_MAIN = slice(28, 42)          # degree, in_ring, min_ring, ring bonds
ATOM_BONDCOUNT_BLOCK = slice(42, 46)    # incident bond-type histogram
ATOM_NBR_BLOCK = slice(46, 74)          # radius-1 neighbour-type histogram
ATOM_D2_BLOCK = slice(74, 102)          # radius-2 atom-type histogram
ATOM_TOPO_TAIL = slice(102, 108)        # radius-2 scalar stats + roles
ATOM_TOPO_COLS = np.concatenate([np.arange(28, 42), np.arange(102, 108)])
BOND_TYPE_BLOCK = slice(0, 4)           # own bond type one-hot (type-dependent)
BOND_TOPO_COLS = np.arange(4, 7)        # is_ring / one-in-ring / both-in-ring
GLOBAL_BOND_FRAC = slice(19, 23)        # bond-type fractions
GLOBAL_DISTINCT_BOND = 31
GLOBAL_DISTINCT_ATOM = 32
GLOBAL_TYPE_IDX = np.array([19, 20, 21, 22, 31, 32])
GLOBAL_TOPO_COLS = np.array([i for i in range(33) if i not in set(GLOBAL_TYPE_IDX.tolist())])


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path | str) -> str:
    return C.sha256_file(path)


def write_json(path: Path | str, payload: Any) -> None:
    C.write_json(path, payload)


def log(msg: str) -> None:
    C.log(msg)


def now() -> str:
    return C.utcnow()


def load_relation_cache() -> dict:
    rel = __import__("torch").load(REL_CACHE, map_location="cpu", weights_only=False)
    h = BD.relation_content_hash(rel)
    if h != REL_CACHE_CONTENT_SHA256:
        raise SystemExit(f"relation cache content hash changed: {h}")
    return rel


def load_train_cache() -> dict:
    return BC.load_train_cache()


def structural_cache(cache: Mapping[str, Any]) -> dict:
    """The cache restricted to *structural* keys (no labels, no y anywhere).

    Everything the pretraining path is allowed to see is in this dict.  The
    pretraining trainer is handed this dict only, so ``cache["labels"]`` is
    unreachable by construction (and gate G1 re-checks it dynamically).
    """
    keys = ("split", "n_molecules", "atom_ptr", "atom_feat", "atom_type",
            "bond_ptr", "bond_feat", "bond_type", "bond_index", "global_feat")
    return {k: cache[k] for k in keys}


def labels(cache: Mapping[str, Any], rows) -> np.ndarray:
    """Raw-y of ``rows`` (the only label access in any analysis script)."""
    return BC.labels(cache, np.asarray(rows, dtype=np.int64))


def split_rows() -> dict:
    return BC.split_rows()


def fit_only_norm(cache: Mapping[str, Any]) -> dict:
    return BC.fit_only_norm(cache, split_rows()["fit8k"])


def ssl_encoder_path(seed: int) -> Path:
    return BC.ssl_path(int(seed))


def historical_b_run(seed: int) -> dict:
    """Frozen v0 entry for BOSI-B of this seed (never re-trained here)."""
    freeze = json.loads((V0 / "analysis" / "freeze.json").read_text())
    return freeze["runs"][f"B_s{int(seed)}"]


def historical_b_checkpoint(seed: int, ckpt: str = "best_monitor") -> Path:
    """v0 checkpoint path; its freeze stores paths relative to the v0 round."""
    entry = historical_b_run(seed)["checkpoints"][ckpt]
    p = Path(entry["file"])
    return p if p.is_absolute() else (V0 / p)


def historical_b_locked_pred(seed: int, ckpt: str = "best_monitor") -> Path:
    lr = json.loads((V0 / "analysis" / "locked_read.json").read_text())
    entry = lr["predictions"][f"B_s{int(seed)}"][ckpt]
    p = Path(entry["file"])
    return p if p.is_absolute() else (V0 / p)


def probe_rows() -> np.ndarray:
    """Fixed Fit8k-only probe rows for the pilot mechanism read."""
    rows = split_rows()["fit8k"]
    rng = np.random.default_rng(20261027)
    return np.sort(rng.choice(rows, size=PROBE_N, replace=False)).astype(np.int64)


def group_definitions(rows: np.ndarray, cache: Mapping[str, Any], rel: Mapping[str, Any]) -> dict:
    """The frozen label-free Locked groupings of BOSI v0 (unchanged)."""
    graphs = BC.molecules_from_cache(cache)
    fit_graphs = [graphs[int(r)] for r in split_rows()["fit8k"]]
    q_graphs = [graphs[int(r)] for r in rows]
    sup = BC.radius2_support(fit_graphs, q_graphs)
    tier = BC.support_tier(sup["unseen_frac"], sup["low4_frac"])
    ring_rank = np.asarray([_cycle_rank(g) for g in q_graphs])
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    alloc = np.zeros(len(rows))
    tq = np.zeros(len(rows))
    for i, r in enumerate(rows):
        hs = np.arange(hp[r], hp[r + 1])
        if not len(hs):
            continue
        counts = np.diff(pp)[hs]
        tgt = np.repeat(np.arange(len(hs)), counts)
        part = (np.concatenate([np.arange(pp[h], pp[h + 1]) for h in hs])
                if counts.sum() else np.zeros(0, np.int64))
        onehot = np.zeros((len(hs), 4), dtype=np.int64)
        if len(part):
            np.add.at(onehot, (tgt, pr[part]), 1)
        alloc[i] = float(np.mean((onehot > 0).sum(1) >= 2))
        tq[i] = float(np.mean((onehot[:, 0] + onehot[:, 1]) > 0))
    aq = np.quantile(alloc, [0.25, 0.5, 0.75])
    y = BC.labels(cache, rows)
    return {
        "all": np.ones(len(y), dtype=bool),
        "ordinary": y > -3.0,
        "tail": y <= -3.0,
        "low_support": tier == "low_support",
        "any_unseen": tier == "any_unseen",
        "all_supported": tier == "all_supported",
        "has_T_or_Q": tq > 0,
        "alloc_defined_top_quartile": alloc >= aq[-1],
        "rings_ge4": ring_rank >= 4,
    }


def _cycle_rank(graph) -> int:
    atoms, bonds = graph
    return int(len(bonds) - len(atoms) + 1)


def rel_or_abs(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROUND.resolve()))
    except ValueError:
        return str(p)


def resolve_path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else (ROUND / p)
