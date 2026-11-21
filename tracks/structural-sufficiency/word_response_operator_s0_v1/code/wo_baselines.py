"""Deterministic references, structure support statistics and the GINE_RAW reference.

* ``ADD_WORD``  ridge on the shared raw count vector (the scalar-word reference).
* ``FP_KRR``    Morgan r=2 / 2048 bits / chirality / Tanimoto kernel ridge.
* ``GINE_RAW``  the previous round's plain GINE reference, from scratch on this
  round's Fit split, official 9/3 columns only, 240 epochs (external comparator).
* support statistics of the frozen word vocabulary over Fit / Valid / H.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

import wo_common as C


# ---------------------------------------------------------------------------
# support statistics (structure only)
# ---------------------------------------------------------------------------


def support_stats(data: Mapping[str, Any]) -> dict[str, Any]:
    cfg = C.load_config()
    record = data["record"]
    templates = data["templates"]
    df_fit = templates["df_fit"]
    seen_fit = {int(t) for t in np.nonzero(df_fit > 0)[0]}
    out: dict[str, Any] = {
        "n_templates": int(record["n_templates"]),
        "n_types_in_fit": int(len(seen_fit)),
        "n_types_unseen_in_fit": int(record["n_templates"] - len(seen_fit)),
        "df_smoothing": float(cfg["operator"]["df_smoothing"]),
        "splits": {},
    }
    for name in ("fit", "valid", "h"):
        split = data[name]
        a = split.arrays
        word_tmpl = a["word_tmpl"].astype(np.int64)
        word_mol = a["word_mol"].astype(np.int64)
        n_instances = int(len(word_tmpl))
        unseen = np.array([int(t) not in seen_fit for t in word_tmpl.tolist()], dtype=bool)
        port_hist = Counter(int(x) for x in a["word_n_boundary"].tolist())
        size_hist = Counter(int(x) for x in a["word_n_atoms"].tolist())
        incidence = Counter()
        cross = a["cross_mol"].astype(np.int64)
        for m in range(split.n_molecules):
            incidence[int(np.sum(cross == m))] += 1
        out["splits"][name] = {
            "n_molecules": int(split.n_molecules),
            "n_word_instances": n_instances,
            "n_distinct_types_present": int(len(set(word_tmpl.tolist()))),
            "unknown_type_instance_rate": float(np.mean(unseen)) if n_instances else 0.0,
            "unknown_type_types_present": int(len({int(t) for t, u in zip(word_tmpl, unseen) if u})),
            "boundary_port_histogram": {str(k): int(v) for k, v in sorted(port_hist.items())},
            "word_size_histogram": {str(k): int(v) for k, v in sorted(size_hist.items())},
            "crossing_bonds_per_molecule_histogram": {str(k): int(v) for k, v in
                                                      sorted(incidence.items())},
            "n_boundary_atoms": int(np.sum(a["graph_n_boundary"])),
            "boundary_atom_share": float(np.sum(a["graph_n_boundary"])
                                         / max(1, np.sum(a["graph_n_atoms"]))),
            "median_word_type_count": float(np.median(list(Counter(word_tmpl.tolist()).values())))
            if n_instances else 0.0,
        }
    # Murcko scaffold overlap of H against Fit
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        payload = C.dataset_cache(cfg["dataset"])

        def scaffold(rows):
            out_set = set()
            for row in np.asarray(rows, dtype=np.int64).tolist():
                mol = Chem.MolFromSmiles(str(payload["smiles"][row]))
                if mol is None:
                    continue
                out_set.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
            return out_set

        fit_scaffolds = scaffold(data["split"]["fit_row_ids"])
        h_rows = data["split"]["h_row_ids"]
        h_scaffolds = scaffold(h_rows)
        shared = h_scaffolds & fit_scaffolds
        out["murcko"] = {
            "n_fit_scaffolds": int(len(fit_scaffolds)),
            "n_h_scaffolds": int(len(h_scaffolds)),
            "n_shared_scaffolds": int(len(shared)),
            "h_molecule_share_in_fit_scaffold": (
                float(np.mean([(_scaffold_of(payload, r) in fit_scaffolds) for r in h_rows]))
                if h_rows else 0.0),
            "note": "pair-component holdout, NOT a scaffold holdout",
        }
    except Exception as exc:  # pragma: no cover - only when RDKit is absent
        out["murcko"] = {"unavailable": True, "reason": str(exc)}
    out["core_overlap"] = {
        "n_fit_pair_core_keys": int(data["split"].get("n_fit_core_keys", 0)),
        "n_evaluable_pairs": int(data["split"].get("n_evaluable_pairs", 0)),
        "definition": "exact retained-core key (core_id) never on a Fit pair",
    }
    C.write_json(C.analysis_dir() / "support.json", out)
    return out


def _scaffold_of(payload, row: int) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(str(payload["smiles"][int(row)]))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


# ---------------------------------------------------------------------------
# ADD_WORD
# ---------------------------------------------------------------------------


def add_word(data: Mapping[str, Any]) -> dict[str, Any]:
    fit, valid = data["fit"], data["valid"]
    z = fit.z_std.astype(np.float64)
    y = (fit.labels.astype(np.float64) - fit.target_mean) / fit.target_std
    lam = float(C.load_config()["add_word"]["ridge_lambda"])
    y_mean = float(y.mean())
    beta = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y_mean))
    beta0 = y_mean

    def predict(split):
        pred_std = beta0 + split.z_std.astype(np.float64) @ beta
        return pred_std * split.target_std + split.target_mean

    pred_fit = predict(fit)
    pred_valid = predict(valid)
    out = {
        "arm": "ADD_WORD",
        "n_features": int(z.shape[1]),
        "n_fit": int(fit.n_molecules),
        "ridge_lambda": lam,
        "beta0": float(beta0),
        "beta": [float(x) for x in beta.tolist()],
        "fit_rmse_raw": C.rmse(pred_fit, fit.labels),
        "valid_rmse_raw": C.rmse(pred_valid, valid.labels),
        "fit_mae_raw": C.mae(pred_fit, fit.labels),
        "valid_mae_raw": C.mae(pred_valid, valid.labels),
        "predictions": {
            "fit_row_ids": fit.row_ids, "fit_pred_raw": pred_fit,
            "valid_row_ids": valid.row_ids, "valid_pred_raw": pred_valid,
            "valid_target_raw": valid.labels,
        },
        "seconds": 0.0,
    }
    if data["h"].labels.size:
        h = data["h"]
        pred_h = (beta0 + h.z_std.astype(np.float64) @ beta) * h.target_std + h.target_mean
        out["h_rmse_raw"] = C.rmse(pred_h, h.labels)
        out["h_mae_raw"] = C.mae(pred_h, h.labels)
        np.savez_compressed(C.run_dir("lipo", "ADD_WORD") / "predictions_h.npz",
                            row_ids=h.row_ids, pred_raw=pred_h, target_raw=h.labels)
    path = C.run_dir("lipo", "ADD_WORD")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "predictions_best.npz", **{
        "fit_row_ids": out["predictions"]["fit_row_ids"],
        "fit_pred_raw": out["predictions"]["fit_pred_raw"],
        "valid_row_ids": out["predictions"]["valid_row_ids"],
        "valid_pred_raw": out["predictions"]["valid_pred_raw"],
        "valid_target_raw": out["predictions"]["valid_target_raw"],
    })
    payload = {k: v for k, v in out.items() if k != "predictions"}
    C.write_json(path / "run.json", C.jsonable(payload))
    return out


# ---------------------------------------------------------------------------
# FP_KRR
# ---------------------------------------------------------------------------


def _morgan_matrix(payload) -> tuple[np.ndarray, list[int], int]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    cfg = C.load_config()["fp_krr"]
    n_bits = int(cfg["n_bits"])
    radius = int(cfg["radius"])
    rows = []
    failed = []
    for row in range(len(payload["smiles"])):
        mol = Chem.MolFromSmiles(str(payload["smiles"][row]))
        if mol is None:
            failed.append(row)
            rows.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=bool(cfg["use_chirality"]))
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    matrix = np.stack(rows, axis=0)
    words = np.packbits(matrix, axis=1).view(np.uint64)
    if words.shape[1] * 8 < n_bits:
        pad = np.zeros((words.shape[0], 1), dtype=np.uint64)
        words = np.concatenate([words, pad], axis=1)
    return words, failed, radius


def fp_krr(data: Mapping[str, Any]) -> dict[str, Any]:
    cfg = C.load_config()["fp_krr"]
    alpha = float(cfg["alpha"])
    payload = C.dataset_cache("ogbg-mollipo")
    t0 = time.perf_counter()
    bits, failed, radius = _morgan_matrix(payload)
    if failed:
        out = {"arm": "FP_KRR", "unavailable": True,
               "reason": f"RDKit failed to parse {len(failed)} SMILES", "failed_rows": failed[:20],
               "seconds": time.perf_counter() - t0}
        path = C.run_dir("lipo", "FP_KRR")
        path.mkdir(parents=True, exist_ok=True)
        C.write_json(path / "run.json", C.jsonable(out))
        return out
    popcount = np.array([bin(int(w)).count("1") for w in bits[0]])  # dtype check only
    del popcount

    def dots(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)
        for slot in range(a.shape[1]):
            x = a[:, slot].astype(np.uint64)[:, None]
            y = b[:, slot].astype(np.uint64)[None, :]
            out += _popcount64(x & y)
        return out

    def counts(a: np.ndarray) -> np.ndarray:
        out = np.zeros(a.shape[0], dtype=np.int64)
        for slot in range(a.shape[1]):
            out += _popcount64(a[:, slot].astype(np.uint64))
        return out

    fit, valid = data["fit"], data["valid"]
    fit_bits = bits[fit.row_ids]
    valid_bits = bits[valid.row_ids]
    fit_count = counts(fit_bits)
    valid_count = counts(valid_bits)
    inter_fit = dots(fit_bits, fit_bits)
    union_fit = fit_count[:, None] + fit_count[None, :] - inter_fit
    K = inter_fit / np.maximum(union_fit, 1)
    y = (fit.labels.astype(np.float64) - fit.target_mean) / fit.target_std
    coef = np.linalg.solve(K + alpha * np.eye(K.shape[0]), y)
    inter = dots(valid_bits, fit_bits)
    union = valid_count[:, None] + fit_count[None, :] - inter
    k_valid = inter / np.maximum(union, 1)
    pred_std = k_valid @ coef
    pred = pred_std * fit.target_std + fit.target_mean
    inter_tr = inter_fit
    union_tr = union_fit
    k_fit = inter_tr / np.maximum(union_tr, 1)
    pred_fit = (k_fit @ coef) * fit.target_std + fit.target_mean
    out = {
        "arm": "FP_KRR",
        "radius": radius,
        "n_bits": int(cfg["n_bits"]),
        "use_chirality": bool(cfg["use_chirality"]),
        "alpha": alpha,
        "fit_rmse_raw": C.rmse(pred_fit, fit.labels),
        "valid_rmse_raw": C.rmse(pred, valid.labels),
        "fit_mae_raw": C.mae(pred_fit, fit.labels),
        "valid_mae_raw": C.mae(pred, valid.labels),
        "seconds": time.perf_counter() - t0,
        "n_fit": int(fit.n_molecules),
    }
    if data["h"].labels.size:
        h = data["h"]
        h_bits = bits[h.row_ids]
        h_count = counts(h_bits)
        inter_h = dots(h_bits, fit_bits)
        union_h = h_count[:, None] + fit_count[None, :] - inter_h
        k_h = inter_h / np.maximum(union_h, 1)
        pred_h = (k_h @ coef) * fit.target_std + fit.target_mean
        out["h_rmse_raw"] = C.rmse(pred_h, h.labels)
        out["h_mae_raw"] = C.mae(pred_h, h.labels)
        np.savez_compressed(C.run_dir("lipo", "FP_KRR") / "predictions_h.npz",
                            row_ids=h.row_ids, pred_raw=pred_h, target_raw=h.labels)
    path = C.run_dir("lipo", "FP_KRR")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "predictions_best.npz",
                        fit_row_ids=fit.row_ids, fit_pred_raw=pred_fit,
                        valid_row_ids=valid.row_ids, valid_pred_raw=pred,
                        valid_target_raw=valid.labels)
    C.write_json(path / "run.json", C.jsonable(out))
    return out


def _popcount64(x):
    x = x.copy()
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


# ---------------------------------------------------------------------------
# GINE_RAW external reference (previous round's implementation, Fit split)
# ---------------------------------------------------------------------------


def gine_raw(data: Mapping[str, Any], device: str, epochs: int = 240,
             deadline: float | None = None) -> dict[str, Any]:
    v2_code = C.REPO_ROOT / "tracks" / "structural-sufficiency" / "word_sentence_port_binding_context_v2" / "code"
    if str(v2_code) not in sys.path:
        sys.path.insert(0, str(v2_code))
    import wp_train as WT

    cfg = C.load_config()["gine_raw"]
    raw = C.dataset_cache("ogbg-mollipo")
    fit, valid = data["fit"], data["valid"]
    scaler = {"mean": float(data["record"]["scaler"]["mean"]),
              "std": float(data["record"]["scaler"]["std"])}
    model = WT.build_gine_model(raw, 0, device, raw_features=True)
    splits = WT.gine_data_lists(raw, scaler, splits=("train", "valid"))
    # this round's Fit molecules, located inside the canonical split order
    import sq_data as SD

    train_pos = {int(v): i for i, v in enumerate(SD.split_indices(raw, "train").tolist())}
    valid_pos = {int(v): i for i, v in enumerate(SD.split_indices(raw, "valid").tolist())}
    train_list = [splits["train"][train_pos[int(r)]] for r in fit.row_ids.tolist()]
    valid_list = [splits["valid"][valid_pos[int(r)]] for r in valid.row_ids.tolist()]
    recipe = WT.Recipe(epochs=int(epochs), batch=int(cfg["batch"]), peak_lr=float(cfg["peak_lr"]),
                       weight_decay=float(cfg["weight_decay"]),
                       warmup_epochs=int(cfg["warmup_epochs"]), seed=0)
    out_dir = C.run_dir("lipo", "GINE_RAW_REF")
    out_dir.mkdir(parents=True, exist_ok=True)
    WT.run_dir = lambda run_id: C.results_root() / f"lipo_{run_id}"
    result = WT.train(model=model, kind="gine", splits={"train": train_list, "valid": valid_list},
                      scaler=scaler, recipe=recipe, device=device, run_id="GINE_RAW_REF",
                      deadline=deadline)
    return result


def gine_predict(split_data, device: str = "cpu") -> np.ndarray:
    """Predictions from the frozen GINE_RAW_REF checkpoint on an arbitrary split."""
    import sys as _sys

    import torch

    v2_code = C.REPO_ROOT / "tracks" / "structural-sufficiency" / "word_sentence_port_binding_context_v2" / "code"
    if str(v2_code) not in _sys.path:
        _sys.path.insert(0, str(v2_code))
    import sq_data as SD
    import wp_train as WT

    raw = C.dataset_cache("ogbg-mollipo")
    scaler = {"mean": float(split_data.target_mean), "std": float(split_data.target_std)}
    model = WT.build_gine_model(raw, 0, device, raw_features=True)
    state = torch.load(C.run_dir("lipo", "GINE_RAW_REF") / "state_best.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    packed = SD.packed_split(raw, "train")
    counts = WT._gine_counts()
    import sq_gine as SG
    data_list = SG.build_data_list(packed, counts, scaler)
    train_pos = {int(v): i for i, v in enumerate(SD.split_indices(raw, "train").tolist())}
    rows = [data_list[train_pos[int(r)]] for r in split_data.row_ids.tolist()]
    out = WT.evaluate_gine(model, rows, device)
    return np.asarray(out["pred_std"], dtype=np.float64) * scaler["std"] + scaler["mean"]
