"""Shared plumbing for zinc-support-transfer-probe-v1.

Everything structural is reused, not re-implemented:

* keys            ``zinc_composition_transfer_v1/code/zct_keys.py`` (the old
                  radius-1/2 content-key builder, unchanged);
* features/data   ``zinc_arch_discovery_v1/code/ad_data.py`` (the frozen P3
                  cache: 108-dim atom descriptor, 33-dim global descriptor,
                  composition counts, labels);
* model/recipe    ``zinc_recipe_trunk_v2/code/ad2_models.P3NullModel`` (413,505
                  parameters) and ``zinc_composition_transfer_v1/code/zct_train
                  .build_model`` (SSL-only initialisation + 500k parameter cap).

The only new objects are the role split, the support maps (which are always
computed from *one specific supervised fit*), the correction heads, and the
evaluation code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_V2 = _SSUF / "zinc_recipe_trunk_v2" / "code"
_ZCT = _SSUF / "zinc_composition_transfer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_ZCT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import zct_keys as K  # noqa: E402

TASK = _TASK
CACHE = TASK / "cache"
ANALYSIS = TASK / "analysis"
RUNS = TASK / "runs"
PREDS = TASK / "predictions"

N_TRAIN = 10_000
KEY_SUP_LO, KEY_SUP_HI = 4, 25
PARENT_MIN = 50
SPLIT_RNG_SEED = 20261121

# role names used by the manifest
ROLES = ("F", "C_fit", "C_dev", "Q_A", "Q_B", "Q_bg")


# ---------------------------------------------------------------------------
# structural tables
# ---------------------------------------------------------------------------


def load_keys() -> dict:
    return K.load()


def molecule_key_sets(keys: dict, n: int = N_TRAIN):
    """Per-molecule unique k1 / k2 key arrays for the first ``n`` molecules."""
    off2 = np.asarray(keys["mol_k2_off"], dtype=np.int64)[: n + 1]
    off1 = np.asarray(keys["mol_k1_off"], dtype=np.int64)[: n + 1]
    flat2 = np.asarray(keys["mol_k2"], dtype=np.int64)
    flat1 = np.asarray(keys["mol_k1"], dtype=np.int64)
    mol2 = [flat2[off2[i]:off2[i + 1]] for i in range(n)]
    mol1 = [flat1[off1[i]:off1[i + 1]] for i in range(n)]
    return mol1, mol2


def key_support(rows, mol2, n_key2: int) -> np.ndarray:
    """Number of molecules in ``rows`` that contain each radius-2 key."""
    sup = np.zeros(int(n_key2), dtype=np.int32)
    for m in np.asarray(rows, dtype=np.int64).tolist():
        sup[mol2[m]] += 1
    return sup


def k1_support(rows, mol1, n_key1: int) -> np.ndarray:
    """Number of molecules in ``rows`` that contain each radius-1 key.

    Radius-1 and radius-2 keys live in two **independent id spaces** (the key
    builder allocates them separately), so the k1 count has to be built in the
    k1 space itself.  Summing the children's k2 counts would double count (one
    molecule can hold several k2 children of the same k1) and -- worse -- would
    make the number depend on the arbitrary numbering of the k2 space.

    Fix A3 (2026-10-23): the first version of this round wrote
    ``n1 = sup2[atom_k1[s:e]]``, i.e. it looked a radius-1 id up in the radius-2
    support table.  See ``analysis/cpu_checks.json`` for the renumbering
    demonstration.
    """
    sup = np.zeros(int(n_key1), dtype=np.int32)
    for m in np.asarray(rows, dtype=np.int64).tolist():
        sup[mol1[m]] += 1
    return sup


def atom_tables(keys: dict, n_key2: int):
    """Flat per-atom (train atoms first): k2 id, k1 id and k1-parent of k2."""
    atom_k2 = np.asarray(keys["atom_key2"], dtype=np.int64)
    atom_k1 = np.asarray(keys["atom_key1"], dtype=np.int64)
    parent = np.full(int(n_key2), -1, dtype=np.int64)
    parent[atom_k2] = atom_k1
    return atom_k1, atom_k2, parent


def per_atom_support(rows, mol1, mol2, keys: dict, n_key1: int, n_key2: int) -> dict:
    """Per-atom k1 and k2 support inside one supervised fit (fixed, A3).

    Each key space is counted in its own space; the returned ``atom_n1`` /
    ``atom_n2`` are flat per-atom arrays indexed exactly like ``cache["atom_feat"]``
    (train atoms first, then valid atoms for the global key tables).
    """
    sup1 = k1_support(rows, mol1, n_key1)
    sup2 = key_support(rows, mol2, n_key2)
    atom_k1, atom_k2, _ = atom_tables(keys, n_key2)
    return {"sup1": sup1, "sup2": sup2,
            "atom_n1": sup1[atom_k1].astype(np.float64),
            "atom_n2": sup2[atom_k2].astype(np.float64)}


def molecule_support_view(rows, mol1, mol2, keys: dict, n_key1: int, n_key2: int) -> dict:
    """Everything a corrector needs to know about one base fit's support."""
    sv = per_atom_support(rows, mol1, mol2, keys, n_key1, n_key2)
    return {"rows": np.asarray(rows, dtype=np.int64),
            "sup1": sv["sup1"], "sup2": sv["sup2"],
            "atom_n1": sv["atom_n1"], "atom_n2": sv["atom_n2"]}


def support_hash(sup1: np.ndarray, sup2: np.ndarray) -> str:
    """Provenance hash covering *both* support tables (A3: k1 is now used too)."""
    import hashlib

    h = hashlib.sha256()
    for a in (sup1, sup2):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# molecule-level coarse features (label free)
# ---------------------------------------------------------------------------


def coarse_features(cache: dict) -> dict:
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_atoms = np.diff(atom_ptr).astype(np.float64)
    n_bonds = np.diff(bond_ptr).astype(np.float64)
    return {
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "ring_rank": n_bonds - n_atoms + 1.0,
        "atom_comp": np.asarray(cache["atom_count_graph"], dtype=np.float64),
        "global": np.asarray(cache["global_feat"], dtype=np.float64),
    }


CTX_DIM = 1 + 3 + D.N_ATOM_TYPES + 2 + 4 + D.GLOBAL_DIM  # 64
ATOM_IN_DIM = D.ATOM_FEAT_DIM + 2  # 110
POOL_DIM = 16
S_EXTRA_DIM = 2
S_READ_IN = CTX_DIM + 2 * POOL_DIM + S_EXTRA_DIM  # 98
G_READ_IN = CTX_DIM
S_READ_HIDDEN = 64
G_READ_HIDDEN = 160
PSI_HIDDEN = 32


def molecule_context(cache_coarse: dict, rows, f_base, n_unseen, unseen_frac,
                     unseen_min_n1, unseen_mean_n1, unseen_min_n2, unseen_mean_n2):
    """The 64-dim coarse context shared by G and S (raw, not standardised)."""
    rows = np.asarray(rows, dtype=np.int64)
    n_atoms = cache_coarse["n_atoms"][rows]
    comp = cache_coarse["atom_comp"][rows] / np.maximum(n_atoms, 1.0)[:, None]
    return np.concatenate(
        [
            np.asarray(f_base, dtype=np.float64).reshape(-1, 1),
            np.log1p(cache_coarse["n_atoms"][rows]).reshape(-1, 1),
            np.log1p(cache_coarse["n_bonds"][rows]).reshape(-1, 1),
            np.log1p(cache_coarse["ring_rank"][rows]).reshape(-1, 1),
            comp,
            np.log1p(np.asarray(n_unseen, dtype=np.float64)).reshape(-1, 1),
            np.asarray(unseen_frac, dtype=np.float64).reshape(-1, 1),
            np.asarray(unseen_min_n1, dtype=np.float64).reshape(-1, 1),
            np.asarray(unseen_mean_n1, dtype=np.float64).reshape(-1, 1),
            np.asarray(unseen_min_n2, dtype=np.float64).reshape(-1, 1),
            np.asarray(unseen_mean_n2, dtype=np.float64).reshape(-1, 1),
            cache_coarse["global"][rows],
        ],
        axis=1,
    )


def atom_view(cache: dict, atom_rows, atom_n1, atom_n2):
    """The S atom block input: [x_v (108), log1p n1, log1p n2]."""
    x = np.asarray(cache["atom_feat"], dtype=np.float64)[np.asarray(atom_rows, dtype=np.int64)]
    return np.concatenate(
        [
            x,
            np.log1p(np.asarray(atom_n1, dtype=np.float64)).reshape(-1, 1),
            np.log1p(np.asarray(atom_n2, dtype=np.float64)).reshape(-1, 1),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_base_model(seed: int, device, ssl_path=None):
    import zct_train as ZT

    norm = D.load_norm()
    model, info = ZT.build_model(norm, seed, device, ssl_path=ssl_path)
    info["init_state_sha256"] = C.state_hash(model)
    return model, info


def load_base_model(path, seed: int, device):
    import torch

    from ad2_models import P3NullModel

    norm = D.load_norm()
    model = P3NullModel(norm).to(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    C.load_state(model, state, device)
    model.eval()
    return model, ckpt


def predict_rows(model, cache, rows, device, batch_size: int = 256) -> np.ndarray:
    """Raw-y pred for arbitrary train-cache rows, P3N collate, eval mode."""
    import torch

    import ad_train as T1

    pred, _ = T1.evaluate(model, cache, np.asarray(rows, dtype=np.int64), candidate="P3N",
                          device=device, batch_size=batch_size)
    return np.asarray(pred, dtype=np.float64)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def read_manifest(name: str = "split_manifest.json") -> dict:
    with open(CACHE / name, encoding="utf-8") as fh:
        return json.load(fh)


def load_roles(name: str = "roles.npz") -> dict:
    z = np.load(CACHE / name)
    return {k: np.asarray(z[k]) for k in z.files}


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    C.write_json(path, payload)
    print(f"[write] {path}", flush=True)
