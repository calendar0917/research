"""P0: per-molecule error attribution on the *old* scaffold checkpoints.

The old scaffold stage saved aggregate metrics only.  This module reloads the
nine read-only checkpoints, regenerates per-molecule predictions for scaffold
Validation (420) and scaffold Test (420), verifies that the recomputed RMSE/MAE
reproduce the old metrics, and then answers:

* is the G2L-ALL MAE gain broad or concentrated?
* is the G2L-ALL RMSE loss carried by a few molecules?
* are the damaged molecules a recurring set / a chemical class?

No checkpoint is modified and no model is trained here.
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import pilot_common as PC

import be_models as MODELS  # noqa: E402
import be_scaffold as OLD_SCAFFOLD  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

TOL = 1.0e-5
CANDIDATES = ("roleblind", "g2l_all")
CANDIDATE_DISPLAY = {"roleblind": "ROLEBLIND", "g2l_all": "G2L-ALL"}
TOP_KS = (5, 10, 20)
TOP_FRACTIONS = (0.05, 0.10)


# ---------------------------------------------------------------------------
# old scaffold views and predictions
# ---------------------------------------------------------------------------


def load_predictions() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, np.ndarray]], dict[str, Any]]:
    views = OLD_SCAFFOLD.build_scaffold_views()
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_test = np.asarray(views["y_test"], dtype=np.float64)
    old_metrics = PC.read_json(OLD_SCAFFOLD.SCAFFOLD_EVAL / "metrics.json")["models"]
    predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    reproduction: dict[str, Any] = {}
    for arm in PC.OLD_SCAFFOLD_ARMS:
        family = arm[len("scaffold_") : arm.rindex("_s")]
        seed = int(arm[arm.rindex("_s") + 2 :])
        model, _meta = MODELS.build_family(family, views["spec"], views["train_norm"], seed, "cpu")
        ckpt_path = OLD_SCAFFOLD.SCAFFOLD_RUNS / arm / "checkpoint.pt"
        MODELS.load_checkpoint(model, ckpt_path)
        pred_valid = NT.denormalise(
            MODELS.predict(model, views["valid_cache"], "cpu"), views["target_norm"]
        )
        pred_test = NT.denormalise(
            MODELS.predict(model, views["test_cache"], "cpu"), views["target_norm"]
        )
        metrics_valid = MET.all_metrics(pred_valid, y_valid)
        metrics_test = MET.all_metrics(pred_test, y_test)
        old = old_metrics[arm]
        reproduction[arm] = {
            "family": family,
            "seed": seed,
            "checkpoint": PC.track_rel(ckpt_path),
            "checkpoint_sha256": PC.sha256_file(ckpt_path),
            "old_valid_rmse": float(old["valid"]["rmse"]),
            "new_valid_rmse": float(metrics_valid["rmse"]),
            "valid_rmse_abs_diff": abs(float(metrics_valid["rmse"]) - float(old["valid"]["rmse"])),
            "old_valid_mae": float(old["valid"]["mae"]),
            "new_valid_mae": float(metrics_valid["mae"]),
            "old_test_rmse": float(old["test"]["rmse"]),
            "new_test_rmse": float(metrics_test["rmse"]),
            "test_rmse_abs_diff": abs(float(metrics_test["rmse"]) - float(old["test"]["rmse"])),
            "old_test_mae": float(old["test"]["mae"]),
            "new_test_mae": float(metrics_test["mae"]),
        }
        if reproduction[arm]["valid_rmse_abs_diff"] > TOL or reproduction[arm]["test_rmse_abs_diff"] > TOL:
            raise RuntimeError(f"{arm}: scaffold metrics do not reproduce")
        predictions[(family, seed)] = {"valid": pred_valid, "test": pred_test}
    return views, predictions, reproduction


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def _features(views: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    from rdkit.DataStructs import BulkTanimotoSimilarity

    RDLogger.DisableLog("rdApp.*")

    rows = views["split"]["rows"]
    smiles = views["split"]["smiles"]
    train_rows = [int(x) for x in rows["train"]]
    eval_rows = [int(x) for x in rows["valid"]] + [int(x) for x in rows["test"]]
    train_scaffolds = {
        MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles[row]), includeChirality=False)
        for row in train_rows
    }
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles[row]), radius=2, nBits=2048, useChirality=False
        )
        for row in train_rows
    ]
    features: dict[int, dict[str, Any]] = {}
    for row in eval_rows:
        mol = Chem.MolFromSmiles(smiles[row])
        if mol is None:
            raise ValueError(f"invalid SMILES at row {row}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, useChirality=False
        )
        similarities = BulkTanimotoSimilarity(fp, train_fps)
        nearest = int(np.argmax(similarities))
        heavy = int(mol.GetNumHeavyAtoms())
        arom_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        n_o_s_p = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ("N", "O", "S", "P"))
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        features[row] = {
            "n_atoms": int(mol.GetNumAtoms()),
            "n_heavy_atoms": heavy,
            "n_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "n_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "n_aromatic_atoms": int(arom_atoms),
            "fraction_aromatic_atoms": float(arom_atoms / max(heavy, 1)),
            "n_hetero_atoms": int(hetero),
            "n_N_O_S_P": int(n_o_s_p),
            "n_halogens": int(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ("F", "Cl", "Br", "I"))),
            "mw": float(Descriptors.MolWt(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "n_hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
            "n_hba": int(rdMolDescriptors.CalcNumHBA(mol)),
            "scaffold": str(scaffold),
            "scaffold_in_train": bool(scaffold in train_scaffolds),
            "max_train_tanimoto": float(max(similarities)),
            "nearest_train_row": int(train_rows[nearest]),
        }
    return features


def _numeric_feature_names() -> list[str]:
    return [
        "n_atoms",
        "n_heavy_atoms",
        "n_rings",
        "n_aromatic_rings",
        "n_aromatic_atoms",
        "fraction_aromatic_atoms",
        "n_hetero_atoms",
        "n_N_O_S_P",
        "n_halogens",
        "mw",
        "tpsa",
        "n_hbd",
        "n_hba",
        "max_train_tanimoto",
    ]


def _y_bin(value: float) -> str:
    edges = [-1.0e9, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})"
    return "unknown"


def _bin(values: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    out = np.full(values.shape, labels[-1], dtype=object)
    for index in range(len(edges) - 1):
        mask = (values >= edges[index]) & (values < edges[index + 1])
        out[mask] = labels[index]
    return out


def _spearman(x, y) -> tuple[float, float]:
    from scipy.stats import spearmanr

    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in qs}


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------


def _seed_row(
    split: str,
    candidate: str,
    seed: int,
    y: np.ndarray,
    base_pred: np.ndarray,
    cand_pred: np.ndarray,
    features: Mapping[int, Mapping[str, Any]],
    rows: list[int],
) -> dict[str, Any]:
    base_err = base_pred - y
    cand_err = cand_pred - y
    base_abs = np.abs(base_err)
    cand_abs = np.abs(cand_err)
    base_sq = base_err * base_err
    cand_sq = cand_err * cand_err
    delta_ae = base_abs - cand_abs
    delta_se = base_sq - cand_sq  # positive = candidate better
    loss = np.maximum(-delta_se, 0.0)  # positive = extra SSE from candidate
    recovery = np.maximum(delta_se, 0.0)
    order = np.argsort(-loss)
    n = int(y.size)
    base_rmse = float(np.sqrt(np.mean(base_sq)))
    cand_rmse = float(np.sqrt(np.mean(cand_sq)))
    base_mae = float(np.mean(base_abs))
    cand_mae = float(np.mean(cand_abs))
    net_sse_change = float(np.sum(cand_sq) - np.sum(base_sq))  # positive = candidate worse
    gross_damage = float(loss.sum())
    gross_recovery = float(recovery.sum())
    top = {}
    for k in TOP_KS:
        top[str(k)] = {
            "loss_sum": float(loss[order[:k]].sum()),
            "share_of_gross_damage": float(loss[order[:k]].sum() / gross_damage)
            if gross_damage > 0
            else 0.0,
        }
    for fraction in TOP_FRACTIONS:
        k = int(np.ceil(fraction * n))
        top[f"top{int(fraction * 100)}pct"] = {
            "k": k,
            "loss_sum": float(loss[order[:k]].sum()),
            "share_of_gross_damage": float(loss[order[:k]].sum() / gross_damage)
            if gross_damage > 0
            else 0.0,
            "multiple_of_net_change": float(loss[order[:k]].sum() / net_sse_change)
            if net_sse_change > 0
            else None,
        }

    def rmse_after_removal(score: np.ndarray, k: int) -> dict[str, float]:
        keep = np.ones(n, dtype=bool)
        keep[np.argsort(-score)[:k]] = False
        rb = float(np.sqrt(np.mean(base_sq[keep])))
        rc = float(np.sqrt(np.mean(cand_sq[keep])))
        return {"base_rmse": rb, "cand_rmse": rc, "cand_minus_base": rc - rb}

    removal = {}
    for fraction in TOP_FRACTIONS:
        k = int(np.ceil(fraction * n))
        removal[f"top{int(fraction * 100)}pct"] = {
            "k": k,
            "remove_by_candidate_loss": rmse_after_removal(loss, k),
            "remove_by_base_sq_error": rmse_after_removal(base_sq, k),
            "remove_by_candidate_sq_error": rmse_after_removal(cand_sq, k),
        }
    k5 = int(np.ceil(0.05 * n))
    top_loss_set = set(order[:k5].tolist())
    top_base_set = set(np.argsort(-base_sq)[:k5].tolist())
    jaccard = len(top_loss_set & top_base_set) / max(len(top_loss_set | top_base_set), 1)

    top_rows = []
    for index in order[:20]:
        row = int(rows[index])
        top_rows.append(
            {
                "row_id": row,
                "y_logD": float(y[index]),
                "base_err": float(base_err[index]),
                "cand_err": float(cand_err[index]),
                "delta_ae": float(delta_ae[index]),
                "delta_se": float(delta_se[index]),
                "base_abs_err": float(base_abs[index]),
                "cand_abs_err": float(cand_abs[index]),
                "max_train_tanimoto": float(features[row]["max_train_tanimoto"]),
                "scaffold": str(features[row]["scaffold"]),
            }
        )
    return {
        "split": split,
        "candidate": candidate,
        "seed": int(seed),
        "n": n,
        "frac_delta_ae_positive": float(np.mean(delta_ae > 0)),
        "frac_delta_se_positive": float(np.mean(delta_se > 0)),
        "mean_delta_ae": float(np.mean(delta_ae)),
        "median_delta_ae": float(np.median(delta_ae)),
        "mean_delta_se": float(np.mean(delta_se)),
        "median_delta_se": float(np.median(delta_se)),
        "base_mae": base_mae,
        "cand_mae": cand_mae,
        "delta_mae_base_minus_cand": float(base_mae - cand_mae),
        "base_rmse": base_rmse,
        "cand_rmse": cand_rmse,
        "delta_rmse_base_minus_cand": float(base_rmse - cand_rmse),
        "base_sse": float(base_sq.sum()),
        "cand_sse": float(cand_sq.sum()),
        "net_sse_change_cand_minus_base": net_sse_change,
        "gross_damage": gross_damage,
        "gross_recovery": gross_recovery,
        "top_loss": top,
        "removal": removal,
        "jaccard_top5pct_loss_vs_top5pct_base_error": float(jaccard),
        "quantiles_delta_ae": _quantiles(delta_ae),
        "quantiles_delta_se": _quantiles(delta_se),
        "top20_loss_rows": top_rows,
    }


def _recurrence(split: str, candidate: str, losses: np.ndarray, rows: list[int], features: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    n = losses.shape[1]
    out: dict[str, Any] = {"split": split, "candidate": candidate}
    for fraction in TOP_FRACTIONS + (0.20,):
        k = int(np.ceil(fraction * n))
        sets = [set(np.argsort(-losses[seed])[:k].tolist()) for seed in range(losses.shape[0])]
        pairwise = {
            f"s{a}_s{b}": len(sets[a] & sets[b])
            for a, b in ((0, 1), (0, 2), (1, 2))
        }
        out[f"top{int(fraction * 100)}pct"] = {
            "k": k,
            "pairwise_overlap_counts": pairwise,
            "all_three_count": len(sets[0] & sets[1] & sets[2]),
            "union_count": len(sets[0] | sets[1] | sets[2]),
            "all_three_rows": [
                {
                    "row_id": int(rows[i]),
                    "mean_loss": float(losses[:, i].mean()),
                    "scaffold": str(features[int(rows[i])]["scaffold"]),
                    "max_train_tanimoto": float(features[int(rows[i])]["max_train_tanimoto"]),
                }
                for i in sorted(sets[0] & sets[1] & sets[2])
            ],
        }
    out["spearman_loss_rank_pairs"] = {
        "s0_s1": _spearman(losses[0], losses[1])[0],
        "s0_s2": _spearman(losses[0], losses[2])[0],
        "s1_s2": _spearman(losses[1], losses[2])[0],
    }
    out["candidate_worse_all_three_seeds"] = {
        "count": int(np.all(losses > 0, axis=0).sum()),
        "fraction": float(np.all(losses > 0, axis=0).mean()),
    }
    top20 = set(np.argsort(-losses.mean(axis=0))[:20].tolist())
    out["top20_mean_loss_unique_scaffolds"] = len({features[int(rows[i])]["scaffold"] for i in top20})
    out["top20_mean_loss_scaffold_counts"] = dict(
        collections.Counter(str(features[int(rows[i])]["scaffold"]) for i in top20).most_common(5)
    )
    return out


def _feature_tables(
    split: str,
    candidate: str,
    y: np.ndarray,
    rows: list[int],
    features: Mapping[int, Mapping[str, Any]],
    mean_delta_ae: np.ndarray,
    mean_delta_se: np.ndarray,
) -> dict[str, Any]:
    numeric = _numeric_feature_names()
    correlations = {}
    for name in numeric:
        values = np.asarray([features[row][name] for row in rows], dtype=np.float64)
        rho_se, p_se = _spearman(values, mean_delta_se)
        rho_ae, p_ae = _spearman(values, mean_delta_ae)
        correlations[name] = {
            "spearman_rho_delta_se": rho_se,
            "p_delta_se": p_se,
            "spearman_rho_delta_ae": rho_ae,
            "p_delta_ae": p_ae,
        }
    correlations["true_logD"] = {
        "spearman_rho_delta_se": _spearman(y, mean_delta_se)[0],
        "p_delta_se": _spearman(y, mean_delta_se)[1],
        "spearman_rho_delta_ae": _spearman(y, mean_delta_ae)[0],
        "p_delta_ae": _spearman(y, mean_delta_ae)[1],
    }

    group_specs = {
        "true_logD_bin": _bin(
            y,
            [-1.0e9, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0e9],
            ["[-inf,-1)", "[-1,0)", "[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,inf)"],
        ),
        "heavy_atom_bin": _bin(
            np.asarray([features[row]["n_heavy_atoms"] for row in rows], dtype=np.float64),
            [0, 20, 25, 30, 35, 40, 50, 1.0e9],
            ["[0,20)", "[20,25)", "[25,30)", "[30,35)", "[35,40)", "[40,50)", "[50,inf)"],
        ),
        "aromatic_ring_bin": _bin(
            np.asarray([features[row]["n_aromatic_rings"] for row in rows], dtype=np.float64),
            [0, 1, 2, 3, 4, 1.0e9],
            ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,inf)"],
        ),
        "hetero_atom_bin": _bin(
            np.asarray([features[row]["n_hetero_atoms"] for row in rows], dtype=np.float64),
            [0, 2, 4, 6, 8, 1.0e9],
            ["[0,2)", "[2,4)", "[4,6)", "[6,8)", "[8,inf)"],
        ),
        "max_train_tanimoto_bin": _bin(
            np.asarray([features[row]["max_train_tanimoto"] for row in rows], dtype=np.float64),
            [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01],
            ["[0,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,0.7)", "[0.7,1.0]"],
        ),
    }
    groups: dict[str, Any] = {}
    for name, labels in group_specs.items():
        records = []
        for label in dict.fromkeys(labels.tolist()):
            mask = labels == label
            if not mask.any():
                continue
            records.append(
                {
                    "bin": str(label),
                    "n": int(mask.sum()),
                    "mean_delta_ae": float(mean_delta_ae[mask].mean()),
                    "mean_delta_se": float(mean_delta_se[mask].mean()),
                    "fraction_delta_ae_positive": float(np.mean(mean_delta_ae[mask] > 0)),
                    "sum_delta_se": float(mean_delta_se[mask].sum()),
                }
            )
        groups[name] = records
    return {
        "split": split,
        "candidate": candidate,
        "sign_convention": "ΔAE and ΔSSE are positive when the candidate is better",
        "correlations": correlations,
        "groups": groups,
    }


def _base_error_groups(
    split: str,
    candidate: str,
    y: np.ndarray,
    base_preds: Mapping[int, np.ndarray],
    cand_preds: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    losses = []
    delta_aes = []
    base_abs = []
    for seed in PC.PILOT_SEEDS:
        base_err = base_preds[seed] - y
        cand_err = cand_preds[seed] - y
        losses.append(cand_err * cand_err - base_err * base_err)
        delta_aes.append(np.abs(base_err) - np.abs(cand_err))
        base_abs.append(np.abs(base_err))
    mean_loss = np.mean(losses, axis=0)  # positive = candidate worse
    mean_delta_se = -mean_loss  # positive = candidate better (the round's definition)
    mean_delta_ae = np.mean(delta_aes, axis=0)  # positive = candidate better
    mean_base_abs = np.mean(base_abs, axis=0)
    edges = [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 1.0e9]
    labels = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.75)", "[0.75,1.0)", "[1.0,1.5)", "[1.5,2.5)", "[2.5,inf)"]
    bins = _bin(mean_base_abs, edges, labels)
    out = []
    for label in labels:
        mask = bins == label
        if not mask.any():
            continue
        out.append(
            {
                "bin": label,
                "n": int(mask.sum()),
                "mean_base_abs_error": float(mean_base_abs[mask].mean()),
                "mean_delta_ae": float(mean_delta_ae[mask].mean()),
                "mean_delta_se": float(mean_delta_se[mask].mean()),
                "fraction_delta_ae_positive": float(np.mean(mean_delta_ae[mask] > 0)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_per_molecule(
    views: Mapping[str, Any],
    predictions: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, Any]],
) -> None:
    smiles = views["split"]["smiles"]
    rows_by_split = views["split"]["rows"]
    y_by_split = {"valid": np.asarray(views["labels"]["y_valid"]), "test": np.asarray(views["y_test"])}
    columns = ["split", "row_id", "smiles", "y_logD", "scaffold"]
    for family in PC.OLD_SCAFFOLD_MODELS:
        for seed in PC.PILOT_SEEDS:
            display = PC.MODEL_DISPLAY[family]
            columns += [
                f"pred_{display}_s{seed}",
                f"err_{display}_s{seed}",
                f"abs_err_{display}_s{seed}",
                f"sq_err_{display}_s{seed}",
            ]
    with open(PC.P0_DIR / "per_molecule_scaffold.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split in ("valid", "test"):
            y = y_by_split[split]
            for index, row in enumerate(rows_by_split[split]):
                record = {
                    "split": split,
                    "row_id": int(row),
                    "smiles": str(smiles[int(row)]),
                    "y_logD": float(y[index]),
                    "scaffold": str(features[int(row)]["scaffold"]),
                }
                for family in PC.OLD_SCAFFOLD_MODELS:
                    display = PC.MODEL_DISPLAY[family]
                    for seed in PC.PILOT_SEEDS:
                        pred = float(predictions[(family, seed)][split][index])
                        error = pred - float(y[index])
                        record[f"pred_{display}_s{seed}"] = pred
                        record[f"err_{display}_s{seed}"] = error
                        record[f"abs_err_{display}_s{seed}"] = abs(error)
                        record[f"sq_err_{display}_s{seed}"] = error * error
                writer.writerow(record)


def _write_error_attribution(
    views: Mapping[str, Any],
    predictions: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, Any]],
) -> None:
    smiles = views["split"]["smiles"]
    rows_by_split = views["split"]["rows"]
    y_by_split = {"valid": np.asarray(views["labels"]["y_valid"]), "test": np.asarray(views["y_test"])}
    feature_names = _numeric_feature_names()
    columns = [
        "split",
        "row_id",
        "smiles",
        "y_logD",
        "scaffold",
        "candidate",
        "seed",
        "base_pred",
        "cand_pred",
        "base_err",
        "cand_err",
        "base_abs_err",
        "cand_abs_err",
        "base_sq_err",
        "cand_sq_err",
        "delta_ae",
        "delta_se",
        "loss",
        "recovery",
    ] + feature_names + ["true_logD_bin"]
    with open(PC.P0_DIR / "error_attribution.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split in ("valid", "test"):
            y = y_by_split[split]
            for index, row in enumerate(rows_by_split[split]):
                row = int(row)
                for candidate in CANDIDATES:
                    for seed in PC.PILOT_SEEDS:
                        base_pred = float(predictions[("base", seed)][split][index])
                        cand_pred = float(predictions[(candidate, seed)][split][index])
                        base_err = base_pred - float(y[index])
                        cand_err = cand_pred - float(y[index])
                        base_sq = base_err * base_err
                        cand_sq = cand_err * cand_err
                        record = {
                            "split": split,
                            "row_id": row,
                            "smiles": str(smiles[row]),
                            "y_logD": float(y[index]),
                            "scaffold": str(features[row]["scaffold"]),
                            "candidate": CANDIDATE_DISPLAY[candidate],
                            "seed": int(seed),
                            "base_pred": base_pred,
                            "cand_pred": cand_pred,
                            "base_err": base_err,
                            "cand_err": cand_err,
                            "base_abs_err": abs(base_err),
                            "cand_abs_err": abs(cand_err),
                            "base_sq_err": base_sq,
                            "cand_sq_err": cand_sq,
                            "delta_ae": abs(base_err) - abs(cand_err),
                            "delta_se": base_sq - cand_sq,
                            "loss": max(cand_sq - base_sq, 0.0),
                            "recovery": max(base_sq - cand_sq, 0.0),
                            "true_logD_bin": _y_bin(float(y[index])),
                        }
                        for name in feature_names:
                            record[name] = features[row][name]
                        writer.writerow(record)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _figures(
    views: Mapping[str, Any],
    predictions: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PC.P0_FIGURES.mkdir(parents=True, exist_ok=True)
    y_by_split = {"valid": np.asarray(views["labels"]["y_valid"]), "test": np.asarray(views["y_test"])}
    rows_by_split = {"valid": [int(x) for x in views["split"]["rows"]["valid"]], "test": [int(x) for x in views["split"]["rows"]["test"]]}
    colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

    # Figure 1: delta-AE vs delta-SSE, both candidates and both splits.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=False, sharey=False)
    for row_index, candidate in enumerate(CANDIDATES):
        for col_index, split in enumerate(("valid", "test")):
            ax = axes[row_index, col_index]
            y = y_by_split[split]
            for seed in PC.PILOT_SEEDS:
                base_err = predictions[("base", seed)][split] - y
                cand_err = predictions[(candidate, seed)][split] - y
                delta_ae = np.abs(base_err) - np.abs(cand_err)
                delta_se = base_err * base_err - cand_err * cand_err
                ax.scatter(delta_ae, delta_se, s=7, alpha=0.28, color=colors[seed], label=f"seed {seed}")
            ax.axhline(0, color="black", lw=0.8)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"{CANDIDATE_DISPLAY[candidate]} - BASE · scaffold {split}")
            ax.set_xlabel("ΔAE = |BASE error| - |candidate error|  (positive better)")
            ax.set_ylabel("ΔSSE = BASE² - candidate²  (positive better)")
            ax.grid(alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8, markerscale=1.8)
    fig.tight_layout()
    fig.savefig(PC.P0_FIGURES / "fig1_delta_ae_vs_delta_se.png", dpi=170)
    plt.close(fig)

    # Figure 2: concentration of gross damage for G2L-ALL.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, split in zip(axes, ("valid", "test")):
        y = y_by_split[split]
        for seed in PC.PILOT_SEEDS:
            base_err = predictions[("base", seed)][split] - y
            cand_err = predictions[("g2l_all", seed)][split] - y
            loss = np.maximum(cand_err * cand_err - base_err * base_err, 0.0)
            order = np.argsort(-loss)
            cumulative = np.cumsum(loss[order]) / max(loss.sum(), 1e-12)
            x = np.arange(1, len(loss) + 1) / len(loss)
            ax.plot(x, cumulative, color=colors[seed], lw=1.8, label=f"seed {seed}")
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="uniform")
        for q in TOP_FRACTIONS:
            ax.axvline(q, color="grey", lw=0.7, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_title(f"G2L-ALL gross damage concentration · scaffold {split}")
        ax.set_xlabel("fraction of molecules ranked by ΔSSE loss")
        ax.set_ylabel("cumulative share of gross damage")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PC.P0_FIGURES / "fig2_loss_concentration.png", dpi=170)
    plt.close(fig)

    # Figure 3: mean delta-SSE vs chemistry / error regime for G2L-ALL.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey="row")
    columns = [
        ("max_train_tanimoto", "max Tanimoto to scaffold Train"),
        ("true_logD", "true logD"),
        ("base_abs", "mean |BASE error|"),
    ]
    for row_index, split in enumerate(("valid", "test")):
        y = y_by_split[split]
        rows = rows_by_split[split]
        mean_loss = []
        mean_base_abs = []
        for seed in PC.PILOT_SEEDS:
            base_err = predictions[("base", seed)][split] - y
            cand_err = predictions[("g2l_all", seed)][split] - y
            mean_loss.append(cand_err * cand_err - base_err * base_err)
            mean_base_abs.append(np.abs(base_err))
        mean_loss = np.mean(mean_loss, axis=0)
        mean_delta_se = -mean_loss  # positive = G2L better
        mean_base_abs = np.mean(mean_base_abs, axis=0)
        for col_index, (key, label) in enumerate(columns):
            ax = axes[row_index, col_index]
            if key == "true_logD":
                values = y
            elif key == "base_abs":
                values = mean_base_abs
            else:
                values = np.asarray([features[row][key] for row in rows], dtype=np.float64)
            ax.scatter(values, mean_delta_se, s=8, alpha=0.25, color="#1f77b4")
            # Binned median for readability; the scatter above is the data.
            if key == "true_logD":
                edges = [-1.0e9, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0e9]
            elif key == "base_abs":
                edges = [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 1.0e9]
            else:
                edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
            centers, medians = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (values >= lo) & (values < hi)
                if mask.sum() >= 5:
                    centers.append(float(np.median(values[mask])))
                    medians.append(float(np.median(mean_delta_se[mask])))
            ax.plot(centers, medians, color="#d62728", lw=1.8, marker="o", ms=3, label="binned median")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"scaffold {split} · {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("mean ΔSSE = BASE² - G2L² (positive = G2L better)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PC.P0_FIGURES / "fig3_feature_association.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    PC.P0_DIR.mkdir(parents=True, exist_ok=True)
    views, predictions, reproduction = load_predictions()
    features = _features(views)
    _write_per_molecule(views, predictions, features)
    _write_error_attribution(views, predictions, features)
    _figures(views, predictions, features)

    summary: dict[str, Any] = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "git_commit": PC.git_commit(),
        "source": "old scaffold stage checkpoints, re-inferred on CPU; no training",
        "scope": "development distribution-shift evaluation on the same 4,200 Lipophilicity molecules",
        "sign_convention": (
            "ΔAE = |BASE error| - |candidate error|; ΔSSE = BASE² - candidate²; "
            "positive = candidate better. Gross damage/recovery and SSE-loss use the opposite sign."
        ),
        "reproduction": reproduction,
        "all_reproduced": bool(
            all(
                rec["valid_rmse_abs_diff"] <= TOL and rec["test_rmse_abs_diff"] <= TOL
                for rec in reproduction.values()
            )
        ),
        "per_seed": [],
        "recurrence": [],
        "feature_association": [],
        "base_error_regime": [],
    }
    if not summary["all_reproduced"]:
        raise SystemExit("P0 reproduction failed; stop before P1")

    for split in ("valid", "test"):
        y = np.asarray(views["labels"]["y_valid"] if split == "valid" else views["y_test"], dtype=np.float64)
        rows = [int(x) for x in views["split"]["rows"][split]]
        for candidate in CANDIDATES:
            losses = []
            for seed in PC.PILOT_SEEDS:
                base_pred = predictions[("base", seed)][split]
                cand_pred = predictions[(candidate, seed)][split]
                summary["per_seed"].append(
                    _seed_row(split, CANDIDATE_DISPLAY[candidate], seed, y, base_pred, cand_pred, features, rows)
                )
                base_err = base_pred - y
                cand_err = cand_pred - y
                losses.append(cand_err * cand_err - base_err * base_err)
            losses = np.asarray(losses)
            summary["recurrence"].append(
                _recurrence(split, CANDIDATE_DISPLAY[candidate], losses, rows, features)
            )
            mean_delta_ae = np.mean(
                [
                    np.abs(predictions[("base", seed)][split] - y)
                    - np.abs(predictions[(candidate, seed)][split] - y)
                    for seed in PC.PILOT_SEEDS
                ],
                axis=0,
            )
            mean_delta_se = -losses.mean(axis=0)
            summary["feature_association"].append(
                _feature_tables(split, CANDIDATE_DISPLAY[candidate], y, rows, features, mean_delta_ae, mean_delta_se)
            )
            summary["base_error_regime"].append(
                {
                    "split": split,
                    "candidate": CANDIDATE_DISPLAY[candidate],
                    "bins": _base_error_groups(
                        split,
                        CANDIDATE_DISPLAY[candidate],
                        y,
                        {seed: predictions[("base", seed)][split] for seed in PC.PILOT_SEEDS},
                        {seed: predictions[(candidate, seed)][split] for seed in PC.PILOT_SEEDS},
                    ),
                }
            )
    PC.write_json(PC.P0_DIR / "p0_summary.json", summary)
    PC.log("[p0] wrote per-molecule tables, summary and three figures")
    return summary


__all__ = ["run", "load_predictions"]
