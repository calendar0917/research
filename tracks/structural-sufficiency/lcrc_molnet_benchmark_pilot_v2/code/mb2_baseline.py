"""P2: matched traditional baseline -- ECFP4 + RDKit descriptors + XGBoost.

One frozen configuration, applied to all seven datasets on **exactly the same
Train/Valid/Test row ids** as the LCRC runs.  No hyper-parameter search, no
Validation-driven choice, no threshold tuning.  CPU only.

Feature block
-------------
* Morgan/ECFP4, radius 2, 2048 bits (``AllChem.GetMorganFingerprintAsBitVect``).
* 217 RDKit 2D descriptors, a *frozen* name list (:data:`RDKIT_DESCRIPTORS`),
  verified at run time to exist in the pinned RDKit build.
* Concatenated into one dense float32 matrix of width ``2048 + 217``.

Pre-declared preprocessing (all fitted on the **Train split only**)
-------------------------------------------------------------------
1. a descriptor that raises or returns a non-finite value becomes ``NaN``;
2. ``NaN`` is imputed with the descriptor's Train-split median;
3. a column whose Train-split standard deviation is exactly 0 is dropped;
4. no scaling (tree model).

Nothing is fitted on Validation or Test, and no feature is selected with a
label.  The rule set is fixed in :data:`PREPROCESSING` and written into every
per-dataset report.

Stages
------
``--stage features``  build and hash the label-free feature matrices
``--stage train``     fit the frozen XGBoost models, record Validation metrics
``--stage freeze``    write ``analysis/baseline_manifest.json`` (pre-Test gate)
``--stage test``      one-shot Test scoring, only after the freeze, only once

Tox21 gets one model per endpoint, each fitted on the Train rows that carry a
label for that endpoint.  An endpoint with a single class in those rows is
recorded as ``UNTRAINABLE`` -- no random predictions are fabricated.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import mb2_common as C
import mb2_metrics as MET
import mb2_source as S

FEATURE_DIR = C.CACHE / "baseline_features"
BASELINE_DIR = C.BASELINE_DIR
MODEL_DIR = BASELINE_DIR / "models"

#: Frozen RDKit descriptor list (217 names, i.e. every entry of `Descriptors._descList`), taken from ``Descriptors._descList``
#: of the pinned build recorded in ``analysis/baseline_manifest.json``.  Sorted.
RDKIT_DESCRIPTORS: tuple[str, ...] = (
    "AvgIpc", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI",
    "BCUT2D_MRLOW", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v",
    "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4",
    "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "ExactMolWt",
    "FractionCSP3", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "HallKierAlpha",
    "HeavyAtomCount", "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
    "MaxAbsEStateIndex", "MaxAbsPartialCharge", "MaxEStateIndex", "MaxPartialCharge",
    "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex", "MinPartialCharge", "MolLogP",
    "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    "NumAliphaticRings", "NumAmideBonds", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumAromaticRings", "NumAtomStereoCenters", "NumBridgeheadAtoms", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumHeterocycles", "NumRadicalElectrons", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumSpiroAtoms",
    "NumUnspecifiedAtomStereoCenters", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10",
    "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4",
    "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "Phi", "RingCount",
    "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
    "SMR_VSA8", "SMR_VSA9", "SPS", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4",
    "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "fr_Al_COO", "fr_Al_OH",
    "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2",
    "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2",
    "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde",
    "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_amidine",
    "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene",
    "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester",
    "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole",
    "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam",
    "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation",
    "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine",
    "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide",
    "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan",
    "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "qed",
)

PREPROCESSING: dict[str, Any] = {
    "descriptor_error_policy": "descriptor raising or returning a non-finite value -> NaN for that cell",
    "imputation": "per-column median fitted on the Train split only",
    "constant_column_rule": "drop columns whose Train-split std == 0 (bit columns included)",
    "scaling": "none",
    "label_used_for_any_feature_decision": False,
    "validation_or_test_used_for_any_feature_decision": False,
    "tox21_preprocessing_fit": "the dataset-level Tox21 Train split (not the per-endpoint subset)",
}


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------


def _mol_robust(smiles: str) -> tuple[Any, str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "strict"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "failed"
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:  # pragma: no cover - ring perception on a broken molecule
        return None, "failed"
    return mol, "unsanitized_fallback"


def feature_path(dataset: str) -> Path:
    return FEATURE_DIR / f"{dataset}.npz"


def build_features(dataset: str, *, rebuild: bool = False) -> dict[str, Any]:
    """Label-free ECFP4 + descriptor matrix for every source row of a dataset."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    path = feature_path(dataset)
    if path.exists() and not rebuild:
        with np.load(path, allow_pickle=False) as handle:
            matrix = handle["features"]
            methods = handle["methods"].astype(str)
            parse_ok = handle["parse_ok"]
        return {
            "path": C.track_rel(path),
            "sha256": C.sha256_file(path),
            "shape": list(matrix.shape),
            "n_parse_failed": int((~parse_ok.astype(bool)).sum()),
            "n_unsanitized_fallback": int((methods == "unsanitized_fallback").sum()),
            "cached": True,
        }

    available = {name for name, _ in Descriptors._descList}
    missing = [n for n in RDKIT_DESCRIPTORS if n not in available]
    if missing:
        raise SystemExit(f"RDKit build lacks pre-declared descriptors: {missing[:5]}")

    smiles = S.all_smiles(dataset)
    n = len(smiles)
    width = 2048 + len(RDKIT_DESCRIPTORS)
    matrix = np.full((n, width), np.nan, dtype=np.float32)
    methods = np.empty(n, dtype=object)
    parse_ok = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    desc_failures: dict[str, int] = {}
    for i, s in enumerate(smiles):
        mol, method = _mol_robust(s)
        methods[i] = method
        if mol is None:
            continue
        parse_ok[i] = True
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, C.BASELINE_CONFIG["ecfp_radius"], nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray

            ConvertToNumpyArray(fp, arr)
            matrix[i, :2048] = arr.astype(np.float32)
        except Exception:
            desc_failures["__ecfp__"] = desc_failures.get("__ecfp__", 0) + 1
        for j, name in enumerate(RDKIT_DESCRIPTORS):
            func = dict(Descriptors._descList)[name]
            try:
                value = float(func(mol))
            except Exception:
                value = float("nan")
                desc_failures[name] = desc_failures.get(name, 0) + 1
            matrix[i, 2048 + j] = value
    seconds = time.perf_counter() - t0
    C.ensure_dir(FEATURE_DIR)
    np.savez_compressed(
        path,
        features=matrix,
        methods=np.asarray(methods, dtype=str),
        parse_ok=parse_ok,
        descriptor_names=np.asarray(RDKIT_DESCRIPTORS, dtype=str),
    )
    nan_fraction = np.isnan(matrix).mean(axis=0)
    return {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "shape": list(matrix.shape),
        "n_rows": int(n),
        "n_features": int(width),
        "n_parse_failed": int((~parse_ok).sum()),
        "n_unsanitized_fallback": int((methods == "unsanitized_fallback").sum()),
        "ecfp_n_bits": 2048,
        "ecfp_radius": int(C.BASELINE_CONFIG["ecfp_radius"]),
        "n_descriptors": len(RDKIT_DESCRIPTORS),
        "descriptor_value_failures": {k: int(v) for k, v in sorted(desc_failures.items(), key=lambda kv: -kv[1])[:10]},
        "columns_all_nan": [int(i) for i in np.flatnonzero(nan_fraction >= 1.0)],
        "seconds": float(seconds),
        "cached": False,
    }


def load_features(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    info = build_features(dataset)
    with np.load(feature_path(dataset), allow_pickle=False) as handle:
        matrix = np.asarray(handle["features"], dtype=np.float32)
    return matrix, info


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def _preprocess_train_only(X_train: np.ndarray) -> dict[str, Any]:
    median = np.nanmedian(X_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float64)
    filled = np.where(np.isnan(X_train), median.astype(np.float32), X_train)
    std = filled.std(axis=0)
    keep = np.flatnonzero(std > 0)
    return {
        "median": median,
        "keep": keep,
        "n_dropped_constant": int(keep.size != X_train.shape[1]),
        "n_dropped_columns": int(X_train.shape[1] - keep.size),
        "imputed_cells_train": int(np.isnan(X_train).sum()),
        "median_sha256": C.sha256_bytes(np.ascontiguousarray(median).tobytes()),
        "keep_sha256": C.sha256_bytes(np.ascontiguousarray(keep, dtype=np.int64).tobytes()),
    }


def _apply(stats: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    median = np.asarray(stats["median"], dtype=np.float32)
    filled = np.where(np.isnan(X), median, X)
    return np.ascontiguousarray(filled[:, np.asarray(stats["keep"], dtype=np.int64)])


def _xgb(task: str):
    import xgboost as xgb

    params = dict(C.BASELINE_CONFIG["xgboost"])
    if task == "regression":
        return xgb.XGBRegressor(**params)
    return xgb.XGBClassifier(**params, eval_metric="logloss")


def fit_dataset(dataset: str, *, overwrite: bool = False) -> dict[str, Any]:
    meta = C.dataset_entry(dataset)
    out_path = BASELINE_DIR / f"{dataset}.json"
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{C.track_rel(out_path)} exists; the baseline is one-shot (use --overwrite only before freeze)")
    src = S.load_dataset(dataset, labels="train_valid")
    X, feat_info = load_features(dataset)
    rows = src["rows"]
    X_train = X[np.asarray(rows["train"], dtype=np.int64)]
    X_valid = X[np.asarray(rows["valid"], dtype=np.int64)]
    stats = _preprocess_train_only(X_train)
    Xtr = _apply(stats, X_train)
    Xva = _apply(stats, X_valid)
    y_train = np.asarray(src["y"]["train"], dtype=np.float64)
    y_valid = np.asarray(src["y"]["valid"], dtype=np.float64)

    C.ensure_dir(MODEL_DIR)
    t0 = time.perf_counter()
    models: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    if meta["task"] == "multitask":
        valid_logits = np.full(y_valid.shape, np.nan, dtype=np.float64)
        for j, name in enumerate(C.TOX21_TASKS):
            obs_tr = np.isfinite(y_train[:, j])
            obs_va = np.isfinite(y_valid[:, j])
            y_tr = y_train[obs_tr, j]
            classes = np.unique(y_tr)
            entry: dict[str, Any] = {
                "n_train_observed": int(obs_tr.sum()),
                "n_valid_observed": int(obs_va.sum()),
                "train_positives": int(y_tr.sum()),
                "train_negatives": int(obs_tr.sum() - y_tr.sum()),
            }
            if classes.size != 2:
                entry["status"] = "UNTRAINABLE_SINGLE_CLASS_IN_TRAIN"
                entry["note"] = "no model is fitted and no prediction is fabricated for this endpoint"
                per_task[name] = entry
                continue
            model = _xgb("binary")
            model.fit(Xtr[obs_tr], y_tr.astype(int))
            logits = model.predict(Xtr[obs_tr], output_margin=True)
            entry["train_roc_auc"] = float(MET.binary_metrics(logits, y_tr)["roc_auc"])
            logits_va = model.predict(Xva[obs_va], output_margin=True)
            valid_logits[obs_va, j] = logits_va
            entry["valid_roc_auc"] = float(
                MET.binary_metrics(logits_va, y_valid[obs_va, j])["roc_auc"]
            )
            entry["valid_pr_auc"] = float(
                MET.binary_metrics(logits_va, y_valid[obs_va, j])["pr_auc"]
            )
            entry["status"] = "TRAINED"
            models[name] = model
            per_task[name] = entry
        valid_metrics = MET.multitask_metrics(valid_logits, y_valid, list(C.TOX21_TASKS))
        for name in C.TOX21_TASKS:
            if per_task[name]["status"] != "TRAINED":
                per_task[name]["valid_roc_auc"] = per_task[name].get("valid_roc_auc")
    elif meta["task"] == "regression":
        model = _xgb("regression")
        model.fit(Xtr, y_train.reshape(-1))
        models["single"] = model
        valid_pred = np.asarray(model.predict(Xva), dtype=np.float64).reshape(-1, 1)
        valid_metrics = MET.regression_metrics(valid_pred[:, 0], y_valid.reshape(-1))
    else:
        flat = y_train.reshape(-1).astype(int)
        if np.unique(flat).size != 2:
            raise SystemExit(f"{dataset}: single-class Train split; the baseline is untrainable")
        model = _xgb("binary")
        model.fit(Xtr, flat)
        models["single"] = model
        valid_pred = np.asarray(model.predict(Xva, output_margin=True), dtype=np.float64).reshape(-1, 1)
        valid_metrics = MET.binary_metrics(valid_pred[:, 0], y_valid.reshape(-1))
    seconds = time.perf_counter() - t0
    if meta["task"] == "multitask":
        valid_pred = valid_logits

    model_files: dict[str, Any] = {}
    for name, model in models.items():
        suffix = "" if name == "single" else f"_{name}"
        path = MODEL_DIR / f"{dataset}{suffix}.ubj"
        model.save_model(str(path))
        model_files[name] = {"path": C.track_rel(path), "sha256": C.sha256_file(path), "size_bytes": int(path.stat().st_size)}

    np.savez(
        BASELINE_DIR / f"{dataset}_valid_predictions.npz",
        rows=np.asarray(rows["valid"], dtype=np.int64),
        pred=np.asarray(valid_pred, dtype=np.float64),
        y_true=np.asarray(y_valid, dtype=np.float64),
    )

    payload: dict[str, Any] = {
        "schema": "lcrc-molnet-v2-baseline-dataset-v1",
        "round_id": C.ROUND_ID,
        "dataset": dataset,
        "task": meta["task"],
        "label_columns": list(meta["label_columns"]),
        "units": meta["units"],
        "frozen_config": dict(C.BASELINE_CONFIG),
        "descriptor_list_sha256": C.sha256_json(list(RDKIT_DESCRIPTORS)),
        "descriptor_count": len(RDKIT_DESCRIPTORS),
        "rdkit_version": C.public_env_info()["rdkit"],
        "xgboost_version": __import__("xgboost").__version__,
        "features": feat_info,
        "preprocessing": {
            **PREPROCESSING,
            "n_dropped_columns": stats["n_dropped_columns"],
            "n_used_columns": int(len(stats["keep"])),
            "imputed_cells_train": stats["imputed_cells_train"],
            "median_sha256": stats["median_sha256"],
            "keep_sha256": stats["keep_sha256"],
        },
        "rows": {
            "train": [int(x) for x in rows["train"]],
            "valid": [int(x) for x in rows["valid"]],
            "train_sha256": C.row_sha256(rows["train"]),
            "valid_sha256": C.row_sha256(rows["valid"]),
        },
        "n_train": int(len(rows["train"])),
        "n_valid": int(len(rows["valid"])),
        "metrics": {"valid": valid_metrics},
        "tox21_per_endpoint": per_task if meta["task"] == "multitask" else None,
        "models": model_files,
        "fit_seconds": float(seconds),
        "hostname": __import__("platform").node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "test_labels_read": False,
        "validation_used_for_any_baseline_choice": False,
    }
    C.write_json(out_path, payload)
    C.log(
        f"[baseline {dataset}] {meta['task']} valid "
        f"{MET.primary_name(meta['task'])}="
        f"{MET.primary_value(meta['task'], valid_metrics):.5f} in {seconds:.1f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# freeze / test
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    missing: list[str] = []
    for key in C.DATASET_ORDER:
        path = BASELINE_DIR / f"{key}.json"
        if not path.exists():
            missing.append(key)
            continue
        payload = C.read_json(path)
        if payload.get("test_labels_read"):
            raise SystemExit(f"{key}: baseline report claims Test access before the freeze")
        entries[key] = {
            "report": C.track_rel(path),
            "report_sha256": C.sha256_file(path),
            "features_sha256": payload["features"]["sha256"],
            "features_shape": payload["features"]["shape"],
            "frozen_config_sha256": C.sha256_json(payload["frozen_config"]),
            "descriptor_list_sha256": payload["descriptor_list_sha256"],
            "preprocessing": payload["preprocessing"],
            "rows": payload["rows"],
            "n_train": payload["n_train"],
            "n_valid": payload["n_valid"],
            "metrics_valid": payload["metrics"]["valid"],
            "tox21_per_endpoint": payload["tox21_per_endpoint"],
            "models": payload["models"],
            "fit_seconds": payload["fit_seconds"],
            "untrainable_endpoints": [
                k
                for k, v in (payload["tox21_per_endpoint"] or {}).items()
                if v.get("status") != "TRAINED"
            ],
        }
    manifest = {
        "schema": "lcrc-molnet-v2-baseline-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "frozen_config": dict(C.BASELINE_CONFIG),
        "descriptor_list_sha256": C.sha256_json(list(RDKIT_DESCRIPTORS)),
        "descriptor_count": len(RDKIT_DESCRIPTORS),
        "preprocessing_rules": PREPROCESSING,
        "datasets": entries,
        "missing_datasets": missing,
        "frozen": True,
        "test_evaluation_rule": (
            "one shot per dataset, after this manifest is committed; no Test result may change the "
            "configuration, the imputation statistics, the dropped columns or the number of trees"
        ),
        "test_labels_read": False,
    }
    C.write_json(BASELINE_DIR / "baseline_manifest.json", manifest)
    C.write_json(C.ANALYSIS / "baseline_manifest.json", manifest)
    C.log(f"froze {len(entries)} baseline model(s); missing: {missing}")
    return manifest


def _baseline_test_predictions(dataset: str, X_test: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Re-fit is impossible after the freeze, so test scoring reloads the saved models."""
    import xgboost as xgb

    payload = C.read_json(BASELINE_DIR / f"{dataset}.json")
    meta = C.dataset_entry(dataset)
    # rebuild the frozen preprocessing statistics without touching Validation or
    # Test: medians and the column mask are re-derived from the Train rows only
    src = S.load_dataset(dataset, labels="train_valid")
    X = np.load(feature_path(dataset), allow_pickle=False)["features"]
    X_train = X[np.asarray(src["rows"]["train"], dtype=np.int64)]
    frozen_stats = _preprocess_train_only(X_train)
    if frozen_stats["median_sha256"] != payload["preprocessing"]["median_sha256"]:
        raise SystemExit(f"{dataset}: recomputed Train medians differ from the frozen report")
    if frozen_stats["keep_sha256"] != payload["preprocessing"]["keep_sha256"]:
        raise SystemExit(f"{dataset}: recomputed Train column mask differs from the frozen report")
    Xte = _apply(frozen_stats, X_test)

    tasks = list(C.TOX21_TASKS) if meta["task"] == "multitask" else ["single"]
    if meta["task"] == "multitask":
        pred = np.full((X_test.shape[0], 12), np.nan, dtype=np.float64)
    else:
        pred = np.zeros((X_test.shape[0], 1), dtype=np.float64)
    for name in tasks:
        suffix = "" if name == "single" else f"_{name}"
        path = C.REPO / payload["models"][name]["path"] if name in payload["models"] else None
        if path is None:
            continue
        if meta["task"] == "multitask":
            per = (payload["tox21_per_endpoint"] or {}).get(name, {})
            if per.get("status") != "TRAINED":
                continue
            model = xgb.XGBClassifier()
        elif meta["task"] == "regression":
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()
        model.load_model(str(path))
        if meta["task"] == "regression":
            pred[:, 0] = model.predict(Xte)
        elif meta["task"] == "binary":
            pred[:, 0] = model.predict(Xte, output_margin=True)
        else:
            col = C.TOX21_TASKS.index(name)
            pred[:, col] = model.predict(Xte, output_margin=True)
    return pred, tasks


def test_dataset(dataset: str, *, force: bool = False) -> dict[str, Any]:
    manifest_path = C.ANALYSIS / "baseline_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("analysis/baseline_manifest.json missing: freeze the baseline first")
    manifest = C.read_json(manifest_path)
    if dataset not in manifest["datasets"]:
        raise SystemExit(f"{dataset} is not in the frozen baseline manifest")
    out_path = BASELINE_DIR / f"{dataset}_test.json"
    if out_path.exists() and not force:
        raise SystemExit(f"{C.track_rel(out_path)} already exists: the Test evaluation is one-shot")
    entry = manifest["datasets"][dataset]
    if C.sha256_file(C.REPO / entry["report"]) != entry["report_sha256"]:
        raise SystemExit(f"{dataset}: baseline report changed since the freeze")

    meta = C.dataset_entry(dataset)
    src = S.load_dataset(dataset, labels="all")
    X, _ = load_features(dataset)
    rows = src["rows"]["test"]
    y_test = np.asarray(src["y"]["test"], dtype=np.float64)
    t0 = time.perf_counter()
    pred, _tasks = _baseline_test_predictions(dataset, X[np.asarray(rows, dtype=np.int64)])
    seconds = time.perf_counter() - t0
    metrics = MET.task_metrics(meta["task"], pred, y_test, None, list(C.TOX21_TASKS))
    ogb_block = None
    if dataset == "ogbg-molhiv":
        ogb_block = MET.ogb_evaluator_roc_auc(y_test.reshape(-1), pred[:, 0])
    np.savez(BASELINE_DIR / f"{dataset}_test_predictions.npz", rows=np.asarray(rows, dtype=np.int64), pred=pred, y_true=y_test)
    payload = {
        "schema": "lcrc-molnet-v2-baseline-test-v1",
        "round_id": C.ROUND_ID,
        "dataset": dataset,
        "task": meta["task"],
        "test_rows_sha256": C.row_sha256(rows),
        "n_test": int(len(rows)),
        "metrics": {"test": metrics},
        "ogb_evaluator": ogb_block,
        "baseline_manifest_sha256": C.sha256_file(manifest_path),
        "baseline_report_sha256": entry["report_sha256"],
        "one_shot": True,
        "evaluated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "seconds": float(seconds),
        "test_labels_read": True,
        "test_used_for_any_selection": False,
        "comparison_label": (
            "EXPOSED_TEST_EXPLORATORY" if dataset in C.EXPOSED_TEST_DATASETS else "FIRST_TEST_READ"
        ),
    }
    C.write_json(out_path, payload)
    C.log(
        f"[baseline test {dataset}] {meta['primary_metric']}="
        f"{MET.primary_value(meta['task'], metrics):.5f} ({payload['comparison_label']})"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 matched ECFP4+RDKit+XGBoost baseline (CPU)")
    parser.add_argument("--stage", required=True, choices=["features", "train", "all", "freeze", "test"])
    parser.add_argument("--datasets", default=",".join(C.DATASET_ORDER))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in C.DATASETS:
            raise SystemExit(f"unknown dataset {key!r}")
    C.ensure_dir(BASELINE_DIR)
    if args.stage == "freeze":
        freeze()
        return 0
    if args.stage == "test":
        for key in keys:
            test_dataset(key)
        return 0
    for key in keys:
        if args.stage in ("features", "all"):
            info = build_features(key, rebuild=args.rebuild)
            C.log(f"[features {key}] shape={info['shape']} fallback={info['n_unsanitized_fallback']} sha={info['sha256'][:12]}")
        if args.stage in ("train", "all"):
            fit_dataset(key, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
