"""OGB-MolHIV adapter with the historical loader's behavior."""

from .._legacy import import_legacy


_legacy = import_legacy("data_molhiv")
MolhivBundle = _legacy.MolhivBundle
check_env = _legacy.check_env
degree_hist_features = _legacy.degree_hist_features
load_molhiv = _legacy.load_molhiv

__all__ = ["MolhivBundle", "load_molhiv", "degree_hist_features", "check_env"]
