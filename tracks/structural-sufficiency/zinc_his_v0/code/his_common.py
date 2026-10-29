"""Shared plumbing for HIS-v0 (hierarchical interaction constraint, exploratory seed-0 screen).

HIS-v0 does **not** introduce a new architecture.  It reuses the frozen LCRC-A
body byte-for-byte (``zinc_local_relation_composition_v1/code/lc_*.py``,
read-only) and adds exactly one training-time regulariser on the *effective
linear response* of ``LocalContent.path_emb`` to the first LocalContent MLP
layer.  Inference, inputs, parameter count (237,519) and forward pass are
unchanged.

This module only provides paths, frozen-input verification and the frozen
valid-only group masks.  It contains no model, data or statistics logic.

Boundaries
----------
* official **train** labels + features: used (features shared with LCRC-A).
* official **valid** labels + features: used for evaluation / analysis only.
* official **test**: never constructed, never loaded, never scored.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROUND_ID = "zinc-his-v0"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
ANALYSIS = EXP_DIR / "analysis"
RUNS = EXP_DIR / "runs"
CONFIGS = EXP_DIR / "configs"

#: Read-only reuse of the frozen LCRC-A track (code, feature cache, norm,
#: frozen seed-0 checkpoint for the CPU audit).
LCRC_TRACK = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
_LCRC_CODE = LCRC_TRACK / "code"
if str(_LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(_LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402,F401
import lc_models as M  # noqa: E402,F401
import lc_train as T  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Frozen inputs (all verified by sha256 / fingerprint before any use)
# ---------------------------------------------------------------------------

FROZEN = {
    "train_cache_file_sha256": "8777494f7745ff5a9a8e3d2913cae8102ad84d267e54fb65235d7eff4c3a6cb2",
    "train_cache_content_sha256": "ded004e8a0c03af31413b0f8d0e8223d80c6ba424a2cbd34db42e16c9b85459b",
    "valid_cache_file_sha256": "aae2caa50d322a25ce75cbd8160e61a831b1488d43295e872db31d0bc7a6dbfe",
    "valid_cache_content_sha256": "f28a0ceed34444c39523e1d301a02350a67087c15b3de725850e45976212aef9",
    "ssl_s0_sha256": "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97",
    "norm_fingerprint": "8c3e88137f18ca6841758b5779b57d7362d9dd57c2c29db2067f822cafd1c89f",
    "feature_version": "lcrc-v1-atom108-s1joint-s2typedpath",
    "lcrc_a_s0_best_sha256": "06340bcd9e45af03e887e706031b6ea14c719cac2b68ba716ce3d686bb21fdad",
    "lcrc_a_s0_initial_state_sha256": "11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e",
    "lcrc_a_s0_best_epoch": 476,
    "lcrc_a_s0_best_valid": 0.07658163924043765,
    "params": 237519,
    "n_train": 10000,
    "n_valid": 1000,
}

LCRC_A_S0_BEST = LCRC_TRACK / "runs/lcrcA_s0/best.pt"

#: Frozen historical anchors (read-only, never retrained this round).
HIST = {
    "lcrc_a_s0_valid": 0.076582,
    "lcrc_a_s0_test": 0.066782,
    "dcc_l_valid_4seed_mean": 0.075990,
    "cjp_a_replay_s0_valid": 0.076116,
    "cjp_a_replay_s1_valid": 0.082301,
}

ROW172 = 172

#: ``historical_lca_s0_worst10`` frozen in
#: ``zinc_cjp_ssl_v1/configs/cjp_ssl_v1.json`` **before** any GPU result of that
#: round; reused verbatim here so the two rounds drop the same rows.
A_S0_WORST10 = (172, 917, 753, 759, 224, 447, 678, 659, 267, 260)

ANCHOR_CSV = REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"

#: ``torch.save`` container bytes of the *same* cache content as produced by the
#: remote checkout (they are the values recorded in the historical LCRC
#: ``train.json`` payloads).  Recorded for provenance only, never gating.
FROZEN_CACHE_FILE_SHA256_BY_HOST = {
    "remote_res": {
        "train": "1aef7f2135dbf320ee0751a03eae357fe0d6d527b80f6204fab78b02a0450c4e",
        "valid": "514e8c723dbd3d88adea8cfe176c3b8c37c849a0812e31c53ae434e4554c4cf3",
    },
    "local": {
        "train": "8777494f7745ff5a9a8e3d2913cae8102ad84d267e54fb65235d7eff4c3a6cb2",
        "valid": "aae2caa50d322a25ce75cbd8160e61a831b1488d43295e872db31d0bc7a6dbfe",
    },
}


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def write_json(path: Path | str, obj) -> None:
    L.write_json(path, obj)


def read_json(path: Path | str):
    return L.read_json(path)


def read_json_or_none(path: Path | str):
    return L.read_json_or_none(path)


def now() -> str:
    return L.now()


def log(msg: str) -> None:
    L.log(msg)


def sha256_file(path: Path | str) -> str:
    return L.C.sha256_file(path)


# ---------------------------------------------------------------------------
# Frozen-input verification
# ---------------------------------------------------------------------------

def verify_frozen_inputs(*, verbose: bool = True) -> dict:
    """Recompute every frozen sha256 / fingerprint this round depends on.

    Gating rule for the feature caches: the **content fingerprint** is the
    scientific invariant and is what gates.  The ``torch.save`` container file
    sha256 differs between checkouts that re-saved identical tensors (zip
    member ordering/timestamps); the historical LCRC runs used the remote
    container bytes recorded in :data:`FROZEN_CACHE_FILE_SHA256_BY_HOST`,
    which produced exactly the same content fingerprint.  The file sha256 is
    still recorded so any real content change is caught by the content gate.
    """
    checks = {}

    def record(name: str, got: str, want: str, *, gate: bool = True) -> None:
        checks[name] = {"got": got, "want": want, "ok": bool(got == want) if gate else None}

    record("train_cache_file", sha256_file(D.cache_path("train")),
           FROZEN["train_cache_file_sha256"], gate=False)
    record("valid_cache_file", sha256_file(D.cache_path("valid")),
           FROZEN["valid_cache_file_sha256"], gate=False)
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    record("train_cache_content", D.content_fingerprint(train), FROZEN["train_cache_content_sha256"])
    record("valid_cache_content", D.content_fingerprint(valid), FROZEN["valid_cache_content_sha256"])
    record("feature_version", str(train["meta"]["feature_version"]), FROZEN["feature_version"])
    record("valid_feature_version", str(valid["meta"]["feature_version"]), FROZEN["feature_version"])
    record("ssl_s0", sha256_file(M.ssl_path(0)), FROZEN["ssl_s0_sha256"])
    norm = D.load_norm()
    record("norm_fingerprint", D.norm_fingerprint(norm), FROZEN["norm_fingerprint"])
    if LCRC_A_S0_BEST.exists():
        record("lcrc_a_s0_best", sha256_file(LCRC_A_S0_BEST), FROZEN["lcrc_a_s0_best_sha256"])
    checks["n_train"] = {"got": int(train["n_molecules"]), "want": FROZEN["n_train"],
                         "ok": int(train["n_molecules"]) == FROZEN["n_train"]}
    checks["n_valid"] = {"got": int(valid["n_molecules"]), "want": FROZEN["n_valid"],
                         "ok": int(valid["n_molecules"]) == FROZEN["n_valid"]}
    ok = all(v["ok"] is not False for v in checks.values())
    if verbose:
        for k, v in checks.items():
            log(f"[frozen] {k}: {'ok' if v['ok'] else 'MISMATCH'} ({v['got']})")
    if not ok:
        bad = [k for k, v in checks.items() if v["ok"] is False]
        raise RuntimeError(f"frozen input mismatch: {bad}")
    return checks


# ---------------------------------------------------------------------------
# Frozen valid-only group masks
# ---------------------------------------------------------------------------

def load_anchor() -> dict:
    with open(ANCHOR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["row_id"]))
    return {
        "row_id": [int(r["row_id"]) for r in rows],
        "y": [float(r["y_raw"]) for r in rows],
        "r2_unseen_atom_frac": [float(r["r2_unseen_atom_frac"]) for r in rows],
        "n_rings": [int(float(r["n_rings"])) for r in rows],
    }


def valid_masks() -> dict:
    """Fixed analysis masks over the official valid split (label-derived only).

    The radii-2 unseen flag and the fixed worst-10 list were frozen in earlier
    rounds before any HIS-v0 result existed; nothing here is chosen this round.
    """
    import numpy as np

    anchor = load_anchor()
    y = np.asarray(anchor["y"], dtype=np.float64)
    unseen = np.asarray(anchor["r2_unseen_atom_frac"], dtype=np.float64) > 0.0
    n_rings = np.asarray(anchor["n_rings"], dtype=np.int64)
    worst = np.zeros(len(y), dtype=bool)
    for r in A_S0_WORST10:
        worst[int(r)] = True
    idx = np.arange(len(y))
    return {
        "all": np.ones(len(y), dtype=bool),
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "row172": idx == ROW172,
        "r2_unseen": unseen,
        "r2_unseen_ordinary": unseen & (y > -3.0),
        "seen_ordinary": (~unseen) & (y > -3.0),
        "ring_rich_ge4": n_rings >= 4,
        "a_s0_worst10_fixed": worst,
        "all_minus_row172": idx != ROW172,
        "all_minus_worst10": ~worst,
    }


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "ANALYSIS", "RUNS", "CONFIGS",
    "LCRC_TRACK", "FROZEN", "FROZEN_CACHE_FILE_SHA256_BY_HOST", "HIST", "ROW172",
    "A_S0_WORST10", "ANCHOR_CSV",
    "L", "D", "M", "T", "LCRC_A_S0_BEST",
    "track_rel", "write_json", "read_json", "read_json_or_none", "now", "log", "sha256_file",
    "verify_frozen_inputs", "load_anchor", "valid_masks",
]
