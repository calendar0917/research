"""Write the frozen machine-readable protocol."""

from __future__ import annotations

import bp_common as C


def payload() -> dict:
    return {
        "protocol_id": C.PROTOCOL_ID,
        "result_label": C.RESULT_LABEL,
        "round_id": C.ROUND_ID,
        "scope": "LCRC benchmark positioning, 13 datasets, exploratory fast round",
        "model": {
            "name": "LCRC-C1 / NO_BOND_MEAN",
            "variant": "C1",
            "only_structural_change": "delete the 96-d b_mean = b_sum/deg local-input block from LCRC-A",
            "frozen_widths": dict(C.MODEL_CONFIG),
            "init": "seed-matched FULL state mapped to C1 by the frozen remap; no pretraining",
            "training": "from scratch, no SSL",
            "ensembles": False,
            "ema": False,
            "swa": False,
            "soup": False,
            "augmentation": False,
            "class_reweighting": False,
            "extra_loss": False,
            "normalization": "all fitted statistics use only the corresponding Train split",
            "checkpoint": "one per training task: Validation-selected state",
            "test": "one terminal evaluation after checkpoint hash is recorded",
        },
        "tu": {
            "datasets": list(C.TU_DATASETS),
            "loader": "torch_geometric.datasets.TUDataset(cleaned=False, use_node_attr=True, use_edge_attr=True)",
            "raw_source": "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip",
            "features": {
                "node_labels": "retained as the discrete node type; dataset-wide sorted vocabulary plus reserved UNK",
                "degree_datasets": "degree buckets 0..15, >=16 + reserved UNK",
                "continuous_node_attributes": "retained verbatim and standardized on the fold Train split only",
                "edge_labels": "deterministic dataset-wide sorted mapping to one-based LCRC bond types plus reserved UNK",
                "topology": "historical TU S1/S2, 9 scalars and 28 global fields; no target label",
            },
            "split": {
                "outer": "StratifiedKFold(n_splits=10, shuffle=True, random_state=20261104)",
                "outer_test": "current fold",
                "outer_train": "remaining 9 folds",
                "validation": "stratified 1/9 of outer_train; random_state=1000+fold_id",
                "model_seed": "100+fold_id",
                "no_inner_grid_search": True,
            },
            "recipe": dict(C.TU_RECIPE),
            "selection": "max Validation accuracy; tie -> lower Validation CE; tie -> earlier epoch",
            "early_stop": "minimum 40 epochs; stop after 25 consecutive epochs without strictly higher best Validation accuracy",
            "report": "mean and sample std of 10 outer Test accuracies in percent; per-fold macro-F1 retained but not used for selection",
        },
        "ogb": {
            "datasets": list(C.MOL_DATASETS),
            "source": "OGB GraphPropPredDataset official get_idx_split(); official SMILES mapping for label-free input construction",
            "input": "frozen lipo_v1 SMILES -> raw attributes -> LCRC molecular graph; no new RDKit target descriptor",
            "seeds": list(C.MOL_SEEDS),
            "recipe": dict(C.OGB_RECIPE),
            "stopping": dict(C.OGB_STOPPING),
            "loss": "BCEWithLogitsLoss with missing-label mask; MolTox21 NaNs are never converted to negatives",
            "metric": "official OGB Evaluator ROC-AUC on each epoch's Validation split; terminal Test with the same Evaluator",
            "report": "per-seed ROC-AUC; mean and sample std over 3 seeds",
        },
        "peptides_func": {
            "dataset": C.PEPTIDE_DATASET,
            "source": "official PyG LRGBDataset Peptides-func fixed train/val/test",
            "input": {
                "atom": "9 OGB-style discrete columns, one-hot with legal category counts 119,4,12,12,6,6,6,2,2",
                "bond": "3 discrete columns, one-hot with legal category counts 5,6,2",
                "s1_s2_types": "atom column 0 and bond column 0",
                "topology": "historical LCRC local statistics and 28 global structural fields",
            },
            "seeds": list(C.PEPTIDE_SEEDS),
            "recipe": dict(C.PEPTIDE_RECIPE),
            "loss": "BCEWithLogitsLoss over 10 independent labels",
            "metric": "macro mean of 10 average-precision scores (percent)",
            "selection": "max Validation mean AP",
            "batch_fallback": [64, 32, 16],
        },
        "correctness_gates": [
            "split pairwise disjoint and complete",
            "TU outer Test never accessed during training or checkpoint selection",
            "OGB official split indices match the cache rows",
            "train-only normalization",
            "target labels absent from structural caches",
            "node renumbering invariance",
            "single-graph vs batched forward consistency",
            "S2 hand-count equality on small graphs",
            "LCRC-C1 structure equals FULL minus b_mean",
            "loss, gradient and checkpoint reload",
            "Test metric independently recomputed from saved predictions",
            "Peptides AP and OGB ROC-AUC shapes/missing-value behavior",
        ],
        "budgets": dict(C.BUDGET),
        "large_graph_precheck": {
            "datasets": list(C.LARGE_GRAPH_DATASETS),
            "requirements": [
                "label-free cache before formal run",
                "per-graph node/edge/S2 distribution and maximum",
                "5-epoch train-throughput check without Test labels",
                "actual epoch time, CPU memory, GPU memory",
                "RESOURCE_BLOCKED if projected full task exceeds 25 minutes",
                "stop dataset if preprocessing exceeds 30 minutes",
            ],
            "forbidden": [
                "random S2 sampling",
                "silent high-order truncation",
                "dropping large graphs",
                "silent edge changes",
                "changing the S2 definition for speed",
            ],
        },
        "screenshot_reference": {
            "label": C.SCREENSHOT_LABEL,
            "values_pct": dict(C.REFERENCE),
            "usage": "visual positioning only; not a verified same-protocol competitor",
        },
    }


def main() -> int:
    C.write_json(C.PROTOCOL_JSON, payload())
    print(C.PROTOCOL_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
