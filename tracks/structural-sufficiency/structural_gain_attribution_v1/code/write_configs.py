"""Emit configs/protocol.json: the machine-readable form of the frozen PROTOCOL.md.

Every value is read from the module that actually used it, so the JSON cannot drift
from the code that produced the results.
"""

from __future__ import annotations

import json

import sga_common as K

K.add_prev_code_path()


def main() -> None:
    import run_analysis_abc as AN
    import run_phaseA_xgb as A
    import run_phaseB_mlp as B
    import run_phaseC_zinc as C
    import sga_zinc as Z

    payload = {
        "round_id": K.ROUND_ID,
        "seed": K.SEED,
        "predecessor": "tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0",
        "evidence_labels": {
            "X0": "EXACT_MATCH",
            "X-CF": "SAME_SPLIT_REFERENCE",
            "new_arms": "SAME_SPLIT_REFERENCE",
            "historical_zinc": "HISTORICAL_REFERENCE",
            "test_split": "EXPOSED_DEVELOPMENT_TEST",
        },
        "molhiv": {
            "splits": {"train": 32901, "valid": 4113, "test": 4113},
            "n_positive": {"train": 1232, "valid": 81, "test": 130},
            "frozen_blocks": {
                "H": {"n_columns": 2258, "source": "MSDS-v0 historical features (median-imputed)"},
                "C": {"n_columns": 754, "source": "MSDS-v0 design_CF (COARSE, raw counts)"},
                "F": {"n_columns": 3652, "source": "MSDS-v0 design_CF (FULL, raw counts)"},
            },
            "new_representations": {
                "MC": {"n_columns": 2048,
                       "definition": "GetHashedMorganFingerprint(radius=2, nBits=2048) counts, "
                                     "raw (not scaled)"},
                "MU": {"n_columns": 4406,
                       "definition": "GetMorganFingerprint(radius=2) unfolded counts, Train document "
                                     "frequency Top-4406 (min df 5), ties by ascending env hash"},
                "WL": {"n_columns": 4406,
                       "definition": "attributed rooted 1-WL rounds 0/1/2 over OGB FULL atom+bond "
                                     "attributes, blake2b-16 signatures, per-round Train df, "
                                     "global Train df Top-4406"},
                "PRES": {"n_columns": 4406,
                         "definition": "1/sqrt(n_atoms) on the support of the raw C+F count vector"},
            },
            "xgboost_recipe": {
                "objective": "binary:logistic",
                "n_estimators": K.XGB_CLASSIFICATION_RECIPE["n_estimators"],
                "max_depth": K.XGB_CLASSIFICATION_RECIPE["max_depth"],
                "learning_rate": K.XGB_CLASSIFICATION_RECIPE["learning_rate"],
                "subsample": K.XGB_CLASSIFICATION_RECIPE["subsample"],
                "colsample_bytree": K.XGB_CLASSIFICATION_RECIPE["colsample_bytree"],
                "reg_lambda": K.XGB_CLASSIFICATION_RECIPE["reg_lambda"],
                "tree_method": K.XGB_CLASSIFICATION_RECIPE["tree_method"],
                "random_state": 0,
                "early_stopping": None,
                "n_jobs": 8,
            },
            "xgb_arms": {
                "X0": {"blocks": ["H"], "role": "reused historical model (EXACT_MATCH)"},
                "X-CF": {"blocks": ["H", "C", "F"], "role": "reused MSDS-v0 XGB_STRUCT"},
                **{arm: {"blocks": ["H", *blocks], "role": "new"}
                   for arm, blocks in A.ARMS.items()},
            },
            "mlp": {
                "architecture": "Linear->128->ReLU->Dropout(0.1)->128->ReLU->Dropout(0.1)->1",
                "hidden": B.HIDDEN, "dropout": B.DROPOUT,
                "extra_branch": "W_Q zero-initialised; W_H/H2/out initialised identically across arms",
                "optimizer": "AdamW", "lr": B.LR, "weight_decay": B.WEIGHT_DECAY,
                "batch_size": B.BATCH, "epochs": B.EPOCHS, "grad_clip": B.CLIP,
                "loss": "class-balanced BCEWithLogitsLoss(mean weight 1, Train positives only)",
                "selection": "best Valid ROC-AUC checkpoint, every epoch",
                "arms": {"N0": ["H"], "N-WL": ["H", "WL"], "N-CF": ["H", "C", "F"]},
            },
            "contrasts": [list(pair) for pair in AN.CONTRASTS_A],
            "contrasts_mlp": [list(pair) for pair in AN.CONTRASTS_B],
            "bootstrap": {"scheme": "paired scaffold bootstrap", "n_boot": 1000,
                          "tail": "percentile 2.5/97.5"},
        },
        "zinc": {
            "source": "PyG ZINC(subset=True) processed/{train,val,test}.pt",
            "splits": {"train": 10000, "valid": 1000, "test": 1000},
            "target": "log-modulus transformed y as shipped by PyG upstream",
            "Z": {"n_columns": None, "wl_rounds": Z.WL_ROUNDS, "wl_top_k": Z.WL_TOPK},
            "Z_CF": {"families": ["size-2", "size-3", "size-4", "cycle-5", "cycle-6"],
                     "coarse": "topology only", "full": "topology + native node/bond classes",
                     "gates": {"coarse_min_df": Z.COARSE_MIN_DF, "full_min_df": Z.FULL_MIN_DF,
                               "min_scaffold": None}},
            "xgb_recipe": {"objective": "reg:absoluteerror", **{
                k: K.XGB_REGRESSION_RECIPE[k] for k in ("n_estimators", "max_depth",
                                                        "learning_rate", "subsample",
                                                        "colsample_bytree", "reg_lambda",
                                                        "tree_method")}},
            "mlp": {"architecture": "same as MolHIV", "loss": "L1",
                    "target_scaling": "Train-only z-score"},
            "arms": ["ZX0", "ZX1", "ZN0", "ZN1"],
            "bootstrap": {"scheme": "paired molecule bootstrap", "n_boot": 1000,
                          "statistic": "delta_mae = MAE(b) - MAE(a); positive = arm a better"},
        },
        "phase_d": {
            "structure_classes": "per (arm, block, family): columns used and tree gain share",
            "subgroups": {"edges_from": "Train quantiles",
                          "size_bins": 5, "coverage_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
                          "never_seen_bins": [0.0, 0.02, 0.1, 0.3, 1.0],
                          "min_n_for_auc": 30},
            "cases": {"n_supported_types": 3, "n_granularity_families": 3,
                      "n_improved": 3, "n_worsened": 3,
                      "definition": "largest |p_XCF - p_X0| whose sign moves toward the true label "
                                    "(improved) or away from it (worsened)"},
        },
        "budget": {"wall_clock_hours": 4, "gpu_hours": 4, "new_xgb": 8, "new_mlp": 5},
        "decision_rules": {
            "P1": "quantity value = X-CF - X-PRES (and X-MC - X0 for the ordinary-count route)",
            "P2": "subgraph value = X-CF - max(X-MU, X-WL); requires CI excluding 0 on BOTH splits",
            "P3": "MLP: same-sign delta vs XGB, and >60% of the XGB AUC delta, is required for "
                  "'structure transfers'",
            "P4": "ZINC: improvement on both splits (delta_mae > 0, CI excluding 0) required",
            "test_rule": "Test never selects anything; a Valid/Test sign or CI conflict is reported "
                         "as SPLIT_CONFLICT and neither side is chosen",
        },
    }
    # fill the ZINC column count from the built design, when available
    meta_path = K.CACHE / "zinc_design_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        payload["zinc"]["Z"]["n_columns"] = meta["Z"]["width"]
        payload["zinc"]["Z"]["columns"] = meta["Z"]["columns"]
        payload["zinc"]["Z"]["n_selected_by_round"] = meta["Z"]["n_selected_by_round"]
        payload["zinc"]["Z_CF"]["n_full_columns"] = meta["Z_CF"]["n_selected_full"]
        payload["zinc"]["Z_CF"]["n_coarse_columns"] = meta["Z_CF"]["n_selected_coarse"]
        payload["zinc"]["Z_CF"]["n_instances"] = meta["Z_CF"]["n_instances"]
        payload["zinc"]["Z_CF"]["full_to_coarse_conflicts"] = meta["Z_CF"]["full_to_coarse"]
        payload["zinc"]["Z_CF"]["filters"] = meta["Z_CF"]["filters"]

    out = K.ROUND / "configs" / "protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"[configs] wrote {K.rel(out)} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
