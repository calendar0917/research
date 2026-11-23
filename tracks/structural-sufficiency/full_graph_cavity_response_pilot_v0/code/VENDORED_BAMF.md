# Vendored exact-inference source

`bamf_inference.py` in this directory is a **byte-for-byte copy** of the
validated BAMF-v1 junction-tree implementation:

| field | value |
|---|---|
| source path | `tracks/structural-sufficiency/zinc_bond_anchored_marginal_field_v1/code/bamf_inference.py` |
| source sha256 | `34024597b290451d931abe9d806d3497de1585722731bb09f21b1057ec39f2a7` |
| local sha256 | `34024597b290451d931abe9d806d3497de1585722731bb09f21b1057ec39f2a7` |
| copied | 2027-01-15, FGCR-v0 protocol freeze |

The file is not edited. `fgcr_inference.py` only wraps its
`_exact_beliefs` function so that `logZ` and per-node unnormalised log weights
are read from **one** calibrated belief pass. The original track and all other
historical files remain read-only.