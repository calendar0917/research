"""Freeze the CJP-SSL v1 configuration (``configs/cjp_ssl_v1.json``).

The file records every frozen scientific choice, the SHA-256 of every reused
upstream artifact, the reference LCRC-A numbers and the GPU budget.  Nothing in
it may be edited after the first formal run.
"""

from __future__ import annotations

import argparse

import numpy as np

import cjp_common as C
import cjp_data as D

L = C.L


def freeze() -> dict:
    raws = D.load_raw_train()
    split = D.structure_split(raws)
    iso = D.verify_split_isomorphism(raws, split)
    norm = C.D.load_norm()
    ssl = {
        f"seed{seed}": {
            "path": C.track_rel(C.M.ssl_path(seed)),
            "sha256": C.sha256_file(C.M.ssl_path(seed)),
        }
        for seed in (0, 1)
    }
    lc = C.LC_DIR
    payload = {
        "round": C.ROUND_ID,
        "status": "frozen_before_gpu",
        "created": C.now(),
        "git_commit_at_freeze": C.git_commit(),
        "task": "Conditional Joint-Path Self-Supervision for LCRC-A",
        "reference": {
            "track": C.track_rel(lc),
            "lc_models_sha256": C.sha256_file(lc / "code/lc_models.py"),
            "lc_data_sha256": C.sha256_file(lc / "code/lc_data.py"),
            "lc_train_sha256": C.sha256_file(lc / "code/lc_train.py"),
            "lc_common_sha256": C.sha256_file(lc / "code/lc_common.py"),
            "lcrc_v1_config": C.track_rel(lc / "configs/lcrc_v1.json"),
            "lcrc_v1_config_sha256": C.sha256_file(lc / "configs/lcrc_v1.json"),
            "lca_s0_valid": 0.07658163605158916,
            "lca_s1_valid": 0.08050378351932158,
            "lca_seed_mean_valid": 0.078542710,
            "lca_epochs": 600,
            "lca_updates": 47400,
            "lca_param_count": 237519,
            "historical_lca_s0_test_mae_background_only": 0.066782,
            "official_test_never_read_this_round": True,
        },
        "data": {
            "source": "official PyG ZINC subset=True raw train x/edge_index/edge_attr",
            "labels_read_by_ssl": False,
            "official_valid_loaded_by_ssl": False,
            "official_test_loaded": False,
            "n_train": len(raws),
            "fingerprint": "typed Weisfeiler-Lehman hash (5 rounds, 128-bit)",
            "n_unique_fingerprints": int(split["n_unique_fingerprints"]),
            "n_duplicate_groups": int(split["n_duplicate_groups"]),
            "n_molecules_in_duplicate_groups": int(split["n_molecules_in_duplicate_groups"]),
            "fingerprint_sha256": split["fingerprint_sha"],
            "fingerprint_order_sha256": split["fingerprint_order_sha"],
            "n_fit": int(len(split["fit_index"])),
            "n_diag": int(len(split["diag_index"])),
            "fit_index_sha256": C.sha256_bytes(np.asarray(split["fit_index"], np.int64).tobytes()),
            "diag_index_sha256": C.sha256_bytes(np.asarray(split["diag_index"], np.int64).tobytes()),
            "duplicate_groups_isomorphic": iso["pass"],
            "lcrc_train_cache_sha256": C.sha256_file(C.D.cache_path("train")),
            "lcrc_valid_cache_sha256": C.sha256_file(C.D.cache_path("valid")),
            "norm_source": norm["source"],
            "norm_fingerprint": C.D.norm_fingerprint(norm),
            "ssl_checkpoints": ssl,
        },
        "masking": {
            "atom_mask_token": C.ATOM_MASK_TOKEN,
            "bond_mask_token": C.BOND_MASK_TOKEN,
            "max_paths_per_molecule": C.MAX_PATHS_PER_MOLECULE,
            "vertex_disjoint": True,
            "mask_rng": "numpy SeedSequence([seed, molecule_id, visit_index, 20261029])",
            "batch_rng": "numpy SeedSequence([seed, 20261030])",
            "masked_atom": "u (path end)",
            "masked_bond": "(w,u) both directed PyG records",
            "visible": ["type(v)", "type(w)", "bond(v,w)"],
            "edges_deleted": False,
            "topology_unchanged": True,
            "target": "3*raw_atom_type(u) + (raw_bond_type(w,u)-1) in [0,62]",
            "target_classes": 63,
        },
        "masked_feature_construction": {
            "order": [
                "modify raw x/edge_attr",
                "structural_relation_objects_v1.build_mol_objects on the masked raw graph",
                "SSL-only masked LCRC builder (S1/S2 visibility filter)",
            ],
            "s1_rule": "include (neighbour type, bond type) only if both are visible",
            "s2_rule": "include a typed 2-path only if type(w), type(u), bond1, bond2 are all visible",
            "topology_scalars": "degree / distance / total path count / closure ratio use true topology",
            "normalisation": "frozen train-only mean/std, never refitted",
        },
        "pretraining": dict(C.PRETRAIN),
        "model": {
            "body": "LCRC('A') unchanged",
            "atom_encoder_out": 96,
            "bond_encoder_out": 96,
            "localcontent_out": 64,
            "deployed_parameters": C.DEPLOY_PARAMETERS,
            "pretext_head": "Linear(192,128) -> SiLU -> Linear(128,63)",
            "marg_projection": "L_marg[a,b] = mean_b L[a,b] + mean_a L[a,b] - mean_ab L[a,b]",
            "frozen_in_pretraining": ["atom_encoder", "bond_encoder"],
            "excluded_from_pretraining": ["f", "rho"],
        },
        "supervised": dict(C.T.RECIPE),
        "arms": {
            "A-REPLAY": {"load_ssl": True, "local_state": None},
            "SCRATCH": {"load_ssl": False, "local_state": None},
            "MARG-SSL": {"load_ssl": True, "local_state": "MARG"},
            "JOINT-SSL": {"load_ssl": True, "local_state": "JOINT"},
        },
        "comparisons": [
            "A-REPLAY - SCRATCH",
            "MARG - A-REPLAY",
            "JOINT - MARG",
            "JOINT - A-REPLAY",
        ],
        "primary_metric": "official valid 1000 raw-y MAE",
        "diagnostic_groups": [
            "full", "valid_minus_row172", "minus_historical_lca_worst10",
            "ordinary_y_gt_-3", "tail_y_le_-3", "r2_unseen_ordinary",
            "seen_ordinary",
        ],
        "budget": {"gpu_hard_limit_hours": 16.0, "gpu_soft_limit_hours": 10.0,
                   "wall_clock_hard_limit_hours": 8.0, "gpus": ["res-gpu0", "res-gpu1"]},
    }
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="freeze CJP-SSL v1 config")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    payload = freeze()
    out = args.out or (C.CONFIGS / "cjp_ssl_v1.json")
    C.write_json(out, payload)
    C.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
