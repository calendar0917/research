# Reference environment

This bundle is self-contained at the Python-source level.  It vendors the
upstream GRIT source at a pinned commit, two tiny pure-Python dependencies, and
two compatibility shims.  It does **not** vendor torch, CUDA, or PyG.

## Verified base

The CPU functional checks and the res A100 timing run use the research
repository's existing environment:

| component | verified value |
|---|---|
| Python | 3.12.14 |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| PyG | 2.6.1 |
| CUDA | 12.4 |
| yacs | 0.1.8 (vendored in `../deps`) |
| opt_einsum | 3.4.0 (vendored in `../deps`) |

The A100 host details and driver are recorded in `../timing.json` under
`environment`; the local CPU environment is recorded in
`../functional_checks.json`.

## Why the shims

The upstream GRIT code imports `torch_scatter` and `torch_sparse`.  Compiling
those old extensions against PyG 2.6.1 is unnecessary for ZINC-12k (max 37
nodes).  `../compat/torch_scatter.py` and `../compat/torch_sparse.py` delegate
the used operations to PyG 2.6.1's `scatter`, `coalesce`, and dense tensor
operations.  The RRWP computation, including the row-normalised random-walk
powers and the dense `N x N` pair tensor, is numerically unchanged; the
functional check compares it against an independent dense implementation.

## Reproduce the environment

If the base environment already has torch/PyG, nothing needs installing:

```bash
cd tracks/structural-sufficiency/zinc_grit_reference_preparation_v1
PYTHONPATH=. python verify_reference.py --device cpu
```

If `yacs`/`opt_einsum` are missing, the vendored `deps/` directory is already on
the runner's import path.  To rebuild it from a network-enabled machine:

```bash
uv pip install --target deps --no-deps yacs==0.1.8 opt_einsum==3.4.0
```

`patch(1)` is required only on the first run, to materialise `runtime/GRIT`
from `upstream/GRIT` plus `patches/0001-reference-no-test-compat.patch`.
`runtime/` is generated and ignored by git.
