# DECISION — zinc_graph_dictionary_readout_v1

**GRAPH_DICTIONARY_READOUT_STOP**

Fixed graph-dictionary readout MAE `0.248224` vs replayed Full `0.119154` (gain `-0.129070`); no frozen gate set was met.

This does not purchase a new full neural training run. It excludes only this fixed prototype dictionary / Gaussian kernel / MAE+L2 head configuration; it does not prove the 814-D representation is theoretically sufficient or insufficient.

No K increase, no bandwidth scan, no normaliser or object swap, no backbone training.
