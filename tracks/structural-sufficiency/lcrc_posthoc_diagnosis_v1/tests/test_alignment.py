import numpy as np
import pandas as pd

import ph_common as C


def test_molecule_rows_and_labels_aligned():
    for dataset in C.MOLECULE_ORDER:
        base_rows = None
        base_y = None
        for seed in C.MOLECULE_SEEDS:
            for model in C.MODELS:
                z = C.load_mol(dataset, model, seed)
                rows = np.asarray(z["rows"], dtype=np.int64)
                y = np.asarray(z["y_true"], dtype=np.float64)
                assert np.isfinite(np.asarray(z["pred"], dtype=np.float64)).all()
                if base_rows is None:
                    base_rows, base_y = rows, y
                else:
                    assert np.array_equal(rows, base_rows)
                    assert np.allclose(y, base_y, rtol=0, atol=0, equal_nan=True)
        assert np.unique(base_rows).size == base_rows.size


def test_molhiv_rows_are_official_test_order():
    split = pd.read_csv(C.OGB_SOURCE / "split/scaffold/test.csv.gz", header=None).to_numpy().reshape(-1)
    z = C.load_mol("ogbg-molhiv", "A0", 0)
    assert np.array_equal(np.asarray(z["rows"], dtype=np.int64), split.astype(np.int64))


def test_esol_rows_are_frozen_test_split():
    split = pd.read_pickle(
        C.REPO
        / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/ESOL.random122.ind.pkl"
    )
    test_rows = [int(x) for x in list(split.iloc[2])]
    z = C.load_mol("ESOL", "A0", 0)
    assert np.asarray(z["rows"], dtype=np.int64).tolist() == test_rows


def test_nci1_gids_are_frozen_fold_tests():
    # The exact fold hashes are checked in ph_aux; here we only check that the
    # saved prediction gids are identical across model/seed for the same fold.
    for fold in range(5):
        base = None
        for seed in C.TU_SEEDS:
            for model in C.MODELS:
                z = C.load_npz(C.tu_run_dir("NCI1", fold, seed, model) / "test_predictions.npz")
                gids = np.asarray(z["gids"], dtype=np.int64)
                if base is None:
                    base = gids
                else:
                    assert np.array_equal(gids, base)
