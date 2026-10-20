"""Precompute masked structural views for the joint (clean + masked) runs.

This reuses the **frozen P3 masked-pretraining construction** verbatim
(``bjoc_masked_pretraining_v1.sample_mask`` / ``mask_raw_arrays`` /
``build_masked_sample``): 15% of atoms and 15% of physical bonds per molecule
are replaced by the dedicated mask tokens, and every feature tensor is then
rebuilt from the masked raw graph, so no field can leak the hidden type.  The
hidden raw codes are kept outside the model input as CE targets.

Only the *schedule* differs from the SSL pretraining: instead of rebuilding a
fresh masked molecule on every visit, ``--visits`` fixed mask snapshots are
built once on CPU and cycled one per epoch during supervised training.  That
keeps the auxiliary branch affordable inside the 500k/5.65 A100-h budget while
preserving the published mask ratio, mask source and rebuild pipeline.

Output (git-ignored, under the round's ``cache/``)::

    masked_v{V}.pt
      atom_feat_v / bond_feat_v / junc_feat_v / global_feat_v  [V, ...]
      mask_atom_ptr [V, N+1] , mask_atom_ids / mask_atom_target
      mask_bond_ptr [V, N+1] , mask_bond_ids / mask_bond_target
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

EXP2 = _HERE.parent
CACHE2 = EXP2 / "cache"
MASK_SEED = 0


def _modules():
    C.ensure_p3_modules()
    sys.path.insert(0, str(C.REPO / "tracks/structure-contribution/code"))
    import structural_relation_objects_v1 as sro  # noqa: E402
    import bjoc_masked_pretraining_v1 as mp  # noqa: E402
    import cscl_features as cf  # noqa: E402

    return sro, mp, cf


def cache_path(visits: int) -> Path:
    return CACHE2 / f"masked_v{int(visits)}.pt"


def prep(visits: int = 4, force: bool = False) -> dict:
    sro, mp, cf = _modules()
    out = cache_path(visits)
    if out.exists() and not force:
        C.log(f"[masked] {out} exists; use --force to rebuild")
        return torch.load(out, map_location="cpu", weights_only=False)["meta"]

    t0 = time.perf_counter()
    train = D.load_cache("train")
    bases, source = D.load_p3_objects("train")
    ds = cf.load_official_train()
    n_mol = len(bases)
    if len(ds) != n_mol:
        raise RuntimeError(f"raw dataset {len(ds)} vs objects {n_mol}")

    atom_ptr = np.asarray(train["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(train["bond_ptr"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    n_bonds = int(bond_ptr[-1])
    n_junc = int(np.asarray(train["junc_ptr"])[-1])
    a_dim = int(train["atom_feat"].shape[1])
    b_dim = int(train["bond_feat"].shape[1])
    j_dim = int(train["junc_feat"].shape[1])
    g_dim = int(train["global_feat"].shape[1])

    atom_feat_v = np.zeros((visits, n_atoms, a_dim), dtype=np.float32)
    bond_feat_v = np.zeros((visits, n_bonds, b_dim), dtype=np.float32)
    junc_feat_v = np.zeros((visits, n_junc, j_dim), dtype=np.float32)
    global_feat_v = np.zeros((visits, n_mol, g_dim), dtype=np.float32)
    mask_atom_ptr = np.zeros((visits, n_mol + 1), dtype=np.int64)
    mask_bond_ptr = np.zeros((visits, n_mol + 1), dtype=np.int64)
    atom_id_l: list[list[np.ndarray]] = []
    atom_tgt_l: list[list[np.ndarray]] = []
    bond_id_l: list[list[np.ndarray]] = []
    bond_tgt_l: list[list[np.ndarray]] = []

    # Alignment check: an unmasked rebuild must reproduce the frozen cache.
    for i in (0, 7, 123):
        d = ds[i]
        o = sro.build_mol_objects(
            int(i), d.x.reshape(-1).long().numpy(), d.edge_index.numpy(),
            d.edge_attr.reshape(-1).long().numpy(),
        )
        a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
        if not np.allclose(np.asarray(o.atom_feat, dtype=np.float32),
                           np.asarray(train["atom_feat"])[a0:a1], atol=1e-6):
            raise RuntimeError(f"raw rebuild does not match the cache at row {i}")

    for v in range(visits):
        visit_atom_ids, visit_atom_tgt, visit_bond_ids, visit_bond_tgt = [], [], [], []
        for i in range(n_mol):
            d = ds[i]
            sample = mp.build_masked_sample(
                bases[i], d.x.reshape(-1).long().numpy(), d.edge_index.numpy(),
                d.edge_attr.reshape(-1).long().numpy(), seed=MASK_SEED, visit_index=v,
            )
            a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
            b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
            j0, j1 = int(np.asarray(train["junc_ptr"])[i]), int(np.asarray(train["junc_ptr"])[i + 1])
            atom_feat_v[v, a0:a1] = sample.masked.atom_feat
            bond_feat_v[v, b0:b1] = sample.masked.bond_feat
            junc_feat_v[v, j0:j1] = sample.masked.junc_feat
            global_feat_v[v, i] = sample.masked.global_feat
            mask_atom_ptr[v, i + 1] = mask_atom_ptr[v, i] + sample.atom_ids.size
            mask_bond_ptr[v, i + 1] = mask_bond_ptr[v, i] + sample.bond_ids.size
            visit_atom_ids.append(sample.atom_ids.astype(np.int32) + a0)
            visit_atom_tgt.append(sample.atom_types.astype(np.int64))
            visit_bond_ids.append(sample.bond_ids.astype(np.int32) + b0)
            visit_bond_tgt.append(sample.bond_types.astype(np.int64))
        atom_id_l.append(visit_atom_ids)
        atom_tgt_l.append(visit_atom_tgt)
        bond_id_l.append(visit_bond_ids)
        bond_tgt_l.append(visit_bond_tgt)
        C.log(f"[masked] visit {v + 1}/{visits} {time.perf_counter() - t0:.1f}s")

    payload = {
        "visits": int(visits),
        "mask_seed": int(MASK_SEED),
        "atom_feat_v": atom_feat_v,
        "bond_feat_v": bond_feat_v,
        "junc_feat_v": junc_feat_v,
        "global_feat_v": global_feat_v,
        "mask_atom_ptr": mask_atom_ptr,
        # Per-visit ragged lists: ``mask_atom_ptr[v]`` indexes
        # ``mask_atom_ids[v]`` (offsets restart at zero for every visit).
        "mask_atom_ids": [np.concatenate(x).astype(np.int32) for x in atom_id_l],
        "mask_atom_target": [np.concatenate(x).astype(np.int64) for x in atom_tgt_l],
        "mask_bond_ptr": mask_bond_ptr,
        "mask_bond_ids": [np.concatenate(x).astype(np.int32) for x in bond_id_l],
        "mask_bond_target": [np.concatenate(x).astype(np.int64) for x in bond_tgt_l],
        "meta": {
            "source_objects": source,
            "construction": "bjoc_masked_pretraining_v1.build_masked_sample",
            "atom_mask_ratio": 0.15,
            "bond_mask_ratio": 0.15,
            "mask_token_atom": 27,
            "mask_token_bond": 0,
            "visits": int(visits),
            "n_molecules": int(n_mol),
            "masked_atoms_per_visit": int(mask_atom_ptr[-1, -1]),
            "masked_bonds_per_visit": int(mask_bond_ptr[-1, -1]),
            "prep_seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    CACHE2.mkdir(parents=True, exist_ok=True)
    C.atomic_torch_save(payload, out)
    C.write_json(CACHE2 / "masked_cache.json", payload["meta"])
    C.log(payload["meta"])
    return payload["meta"]


def load(visits: int = 4) -> dict:
    return torch.load(cache_path(visits), map_location="cpu", weights_only=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep"])
    p.add_argument("--visits", type=int, default=4)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    prep(visits=args.visits, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
