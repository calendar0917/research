"""TUDataset adapter with the historical loader's behavior."""

from .._legacy import import_legacy


_legacy = import_legacy("data_tud")
load_tud = _legacy.load_tud
node_feature_readout = _legacy.node_feature_readout

__all__ = ["load_tud", "node_feature_readout"]
