# PEC-I1 — Stage A: zero-training Patch-B recoverability

Verdict: **LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT**

Label-free (no target `y` read).  PEC environment raw primitives vs S0
`patch_cont` blocks for the same rooted radius-2 patches.

Audited molecules: train 2000, valid 1000; roots 69544.

| block | dim | max abs err | mean abs err | exact fraction | unexplained |
|---|---:|---:|---:|---:|---:|
| atom_shell | 84 | 2.649e-08 | 2.271e-10 | 1.000000 | 0 |
| bond_shell | 24 | 2.751e-08 | 7.593e-10 | 1.000000 | 0 |
| root_atom | 28 | 0.000e+00 | 0.000e+00 | 1.000000 | 0 |
| incident | 4 | 1.987e-08 | 1.597e-09 | 1.000000 | 0 |
| scalars | 6 | 2.403e-01 | 7.724e-03 | 0.837628 | 67752 |

S0 structural-scalar coordinate 5 (patch-scoped mean molecule degree) has
no analytic map in PEC; its molecule-scope counterpart and a train-fit
ridge probe are reported.

* ridge probe valid R² = 0.316360, MAE = 0.032891, max |err| = 0.177174
* molecule-scope analogue MAE = 0.046242

Shell-pair taxonomy:

* S0 declares 6 classes, including `(0,0)`, which PEC cannot express.
* `(0,0)` train occurrences: 0 (structurally impossible).
* `(0,2)` train occurrences: 0 (also empty).

