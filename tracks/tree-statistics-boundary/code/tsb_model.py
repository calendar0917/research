"""The single learner of this round: XGBoost regression, frozen recipe.

No search, no early stopping, no per-arm tuning.  Every one of the 45 fits uses
exactly :data:`tsb_common.XGB_PARAMS`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import tsb_common as C


def make_model(config_params: dict[str, Any] | None = None):
    import xgboost as xgb

    params = dict(C.XGB_PARAMS if config_params is None else config_params)
    if params != C.XGB_PARAMS:
        raise ValueError(f"XGBoost parameters drifted from the frozen recipe:\n{params}")
    return xgb.XGBRegressor(**params)


def fit(
    X_train,
    y_train: np.ndarray,
    config_params: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    model = make_model(config_params)
    t0 = time.perf_counter()
    model.fit(X_train, np.asarray(y_train, dtype=np.float64).reshape(-1))
    return model, time.perf_counter() - t0


def predict(model, X) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(pred)):
        raise ValueError("XGBoost produced non-finite predictions")
    return pred


def save(model, path: Path) -> dict[str, Any]:
    C.ensure_dir(Path(path).parent)
    model.save_model(str(path))
    try:
        display = C.track_rel(path)
    except ValueError:  # diagnostics may write to a temp directory
        display = str(path)
    return {
        "path": display,
        "sha256": C.sha256_file(path),
        "size_bytes": int(Path(path).stat().st_size),
    }


def load(path: Path):
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def params_sha256() -> str:
    return C.sha256_json(C.XGB_PARAMS)


def version() -> str:
    import xgboost

    return str(xgboost.__version__)
