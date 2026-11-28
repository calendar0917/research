"""Build and freeze the label-free input bundle for Phase A/B (MolHIV).

Everything here is fitted on **Train only** (vocabulary, column order, scaling).
The bundle stores exactly the matrices the models consume plus the frozen
predictions of the two arms that are *reused* instead of refitted (X0, X-CF), so
the remote host never has to re-derive a representation.

Outputs
-------
``bundle/sav1_inputs.npz``       committed, label-free, float32 design blocks
``bundle/sav1_inputs_meta.json`` sizes, hashes, fit-on-Train declarations, checks
``data/design_blocks_local.npz`` git-ignored local mirror used by Phase D
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

import sga_common as K

K.add_prev_code_path()

import msds_data as D  # noqa: E402
import msds_features as FEAT  # noqa: E402
import sga_features as F  # noqa: E402
import sga_historical as HIST  # noqa: E402


def main() -> None:
    K.ensure_dirs()
    t0 = time.perf_counter()
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    train = splits["train"]
    n_atoms = molecules.n_atoms.astype(np.float64)
    print(f"[build] molecules={molecules.n_molecules} splits={ {k: int(v.size) for k, v in splits.items()} }",
          flush=True)

    # ---------------- frozen historical block H -------------------------------
    Hmat, Hmeta = HIST.load_H()
    roles = HIST.column_roles()
    if not bool(roles["all_bit_columns_kept"]):
        raise SystemExit("unexpected: some ECFP bit columns were dropped by the historical preprocessing")
    print(f"[build] H {Hmat.shape} bits=2048 descriptors={roles['n_descriptor_kept']}", flush=True)

    # ---------------- frozen C / F / PRES -------------------------------------
    frozen = F.load_frozen_structure()
    c_raw = frozen["c_raw"]
    f_raw = frozen["f_raw"]
    Cm = F.scale_rows(c_raw, n_atoms)
    Fm = F.scale_rows(f_raw, n_atoms)
    cf_check = F.check_against_frozen_design(Cm, Fm)
    if not cf_check["identical"]:
        raise SystemExit(f"rebuilt C/F differ from the frozen design matrix: {cf_check}")
    pres = F.scale_rows(F.binarise(sp.hstack([c_raw, f_raw], format="csr")), n_atoms)
    print(f"[build] C {Cm.shape} nnz={Cm.nnz} | F {Fm.shape} nnz={Fm.nnz} | PRES {pres.shape} nnz={pres.nnz}",
          flush=True)

    # ---------------- MC (folded Morgan counts) -------------------------------
    bits, morgan_meta = FEAT.build_morgan(molecules)
    mc_raw, mc_meta = F.build_mc(bits, molecules.smiles)
    MCm = F.scale_rows(mc_raw, n_atoms)
    print(f"[build] MC {MCm.shape} nnz={MCm.nnz} support_identity={mc_meta['support_identity_holds']}", flush=True)

    # ---------------- MU (unfolded Morgan counts, top-K) ----------------------
    mu_raw, mu_env, mu_meta, mu_fold_all = F.build_mu(molecules.smiles, train)
    MUm = F.scale_rows(mu_raw, n_atoms)
    print(f"[build] MU {MUm.shape} nnz={MUm.nnz} envs={mu_meta['n_distinct_environments_all']}", flush=True)

    # ---------------- WL (attributed rooted counts, top-K) --------------------
    wl_raw, wl_sel, wl_meta = F.build_wl(molecules, train)
    WLm = F.scale_rows(wl_raw, n_atoms)
    print(f"[build] WL {WLm.shape} nnz={WLm.nnz} sigs={wl_meta['n_distinct_signatures_all']}", flush=True)

    # ---------------- verification -------------------------------------------
    checks: dict[str, Any] = {
        "split_fingerprint": K.split_fingerprint(splits),
        "H": Hmeta,
        "H_column_roles": {k: v for k, v in roles.items() if k != "descriptor_names"},
        "morgan_binary_block": {
            "cache_sha256": morgan_meta["cache_sha256"],
            "impl": morgan_meta["impl"],
            "n_parse_failed": morgan_meta["n_parse_failed"],
        },
        "C_F_rebuild": cf_check,
        "zero_pattern_equals_positive_count": {
            name: bool(np.array_equal(np.diff(mat.tocsr().indptr) > 0, np.asarray(mat.sum(axis=1)).ravel() > 0))
            for name, mat in (("C", Cm), ("F", Fm), ("PRES", pres), ("MC", MCm), ("MU", MUm), ("WL", WLm))
        },
        "pres_values_are_1_over_sqrt_n": {
            "unique": sorted(set(np.round(pres.data / (1.0 / np.sqrt(np.maximum(n_atoms, 1.0))[
                np.repeat(np.arange(pres.shape[0]), np.diff(pres.indptr))])).tolist()))[:5],
            "note": "PRES_data / (1/sqrt(n_atoms)) must be exactly 1 for every stored cell",
            "max_abs_dev": float(np.abs(pres.data / (1.0 / np.sqrt(np.maximum(n_atoms, 1.0))[
                np.repeat(np.arange(pres.shape[0]), np.diff(pres.indptr))]) - 1.0).max()) if pres.nnz else 0.0,
        },
        "tables_fitted_on_train": {
            "H": "historical Train-only median imputation + Train-only zero-variance drop",
            "C_F_vocab": "frozen in MSDS-v0 on Train only; reused unchanged",
            "MC": "no fitted table (fixed 2048 buckets, generator settings fixed)",
            "MU": "Train document-frequency top-4406 environment ids",
            "WL": "Train document-frequency top-4406 (round, signature) keys",
            "scaling": "count / sqrt(max(n_atoms,1)); no Train statistic involved",
        },
        "blocks": {
            "H": list(Hmat.shape),
            "MC": list(MCm.shape),
            "MU": list(MUm.shape),
            "WL": list(WLm.shape),
            "C": list(Cm.shape),
            "F": list(Fm.shape),
            "PRES": list(pres.shape),
        },
        "mc_meta": mc_meta,
        "mu_meta": mu_meta,
        "wl_meta": wl_meta,
        "git": K.git_state(),
        "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }

    # cross-check: is the folded count block exactly the MU block folded to 2048?
    fold_ok, fold_stats = _fold_check(mc_raw, mu_fold_all)
    checks["mc_equals_fold_of_mu"] = fold_stats
    if not fold_ok:
        print(f"[build] WARNING folded-Morgan != fold(unfolded): {fold_stats}", flush=True)

    ids = {
        "rows_train": splits["train"], "rows_valid": splits["valid"], "rows_test": splits["test"],
        "y": molecules.label.astype(np.int8), "n_atoms": molecules.n_atoms,
        "mu_env_ids": mu_env.astype(np.int64), "wl_selected": wl_sel.astype(np.int64),
    }
    payload = {
        **HIST.pack_H(Hmat),
        "C_indptr": Cm.indptr, "C_indices": Cm.indices, "C_data": Cm.data,
        "F_indptr": Fm.indptr, "F_indices": Fm.indices, "F_data": Fm.data,
        "PRES_indptr": pres.indptr, "PRES_indices": pres.indices, "PRES_data": pres.data,
        "MC_indptr": MCm.indptr, "MC_indices": MCm.indices, "MC_data": MCm.data,
        "MU_indptr": MUm.indptr, "MU_indices": MUm.indices, "MU_data": MUm.data,
        "WL_indptr": WLm.indptr, "WL_indices": WLm.indices, "WL_data": WLm.data,
        **ids,
    }
    np.savez_compressed(K.INPUT_BUNDLE, **payload)
    K.write_json(K.CACHE / "design_blocks_local.npz.marker.json", {"path": K.rel(K.INPUT_BUNDLE),
                                                                 "sha256": K.sha256_file(K.INPUT_BUNDLE)})
    np.savez_compressed(K.CACHE / "design_blocks_local.npz", **payload)

    checks["input_bundle"] = {
        "path": K.rel(K.INPUT_BUNDLE),
        "sha256": K.sha256_file(K.INPUT_BUNDLE),
        "size_bytes": int(K.INPUT_BUNDLE.stat().st_size),
    }
    # frozen per-molecule predictions of the two reused arms
    checks["reused_arms"] = _frozen_predictions(molecules, splits)
    K.write_json(K.INPUT_BUNDLE_META, checks)
    print(f"[build] bundle {K.rel(K.INPUT_BUNDLE)} "
          f"{K.INPUT_BUNDLE.stat().st_size / 1e6:.1f} MB sha={checks['input_bundle']['sha256'][:12]}", flush=True)
    print(f"[build] zero-pattern checks {checks['zero_pattern_equals_positive_count']}", flush=True)
    print(f"[build] PRES 1/sqrt(n) check {checks['pres_values_are_1_over_sqrt_n']['max_abs_dev']}", flush=True)
    print(f"[build] done in {time.perf_counter() - t0:.1f}s", flush=True)


def _fold_check(mc_raw: sp.csr_matrix, fold_all: sp.csr_matrix):
    """Is the folded count block exactly ``all unfolded counts summed by id % 2048``?

    This is a property of RDKit's generator pair, not of the top-K selection: the
    check uses *all* unfolded environments, so it compares the two folding rules on
    the same support.  It cannot be reproduced from MU alone, because MU keeps only
    the top-4 406 environments.
    """
    diff = abs(fold_all.tocsr() - mc_raw)
    max_abs = float(abs(diff).max()) if diff.nnz else 0.0
    stats = {
        "rule_tested": "MC[b] == sum over ALL unfolded environments e with (e % 2048) == b of count(e)",
        "max_abs_diff": max_abs,
        "n_cells_differing": int(diff.nnz),
        "identical": bool(max_abs == 0.0),
        "note": "when it holds, RDKit's hashed (folded) and unfolded generators agree on the "
                "id -> bucket map id % 2048; MU cannot recover MC because MU keeps only top-4406",
    }
    return max_abs == 0.0, stats


def _frozen_predictions(molecules, splits) -> dict[str, Any]:
    """Per-molecule predictions of the two reused arms (X0 historical, X-CF)."""
    out: dict[str, Any] = {}
    hist = {}
    for name in ("valid", "test"):
        path = K.HIST_PRED_DIR / f"ogbg-molhiv_{name}_predictions.npz"
        with np.load(path) as z:
            rows = np.asarray(z["rows"], dtype=np.int64)
            pred = np.asarray(z["pred"]).reshape(-1).astype(np.float64)
            y_true = np.asarray(z["y_true"]).reshape(-1).astype(np.int8)
        if not np.array_equal(rows, splits[name]):
            raise SystemExit(f"historical {name} prediction rows differ from the official split")
        if not np.array_equal(y_true, molecules.label[rows]):
            raise SystemExit(f"historical {name} stored labels differ from raw graph labels")
        hist[name] = {"path": K.rel(path), "sha256": K.sha256_file(path),
                      "rows_match_official_split": True, "labels_match": True}
    out["X0_historical"] = hist

    path = K.MSDS_DATA / "phaseD_xgb_struct_predictions.npz"
    with np.load(path) as z:
        xcf = {name: (np.asarray(z[name], dtype=np.float64), np.asarray(z[f"{name}_rows"], dtype=np.int64))
               for name in ("train", "valid", "test")}
    ok = all(np.array_equal(rows, splits[name]) for name, (pred, rows) in xcf.items())
    out["X-CF_phaseD"] = {
        "path": K.rel(path), "sha256": K.sha256_file(path),
        "rows_match_official_split": bool(ok),
        "note": "XGB_STRUCT of MSDS-v0 Phase D is reused as X-CF; nothing is refitted",
    }
    if not ok:
        raise SystemExit("predecessor XGB_STRUCT predictions do not align with the official split")
    return out


if __name__ == "__main__":
    main()
