# E2E-DictEnv-v0 — decision

```
E2E_DICTENV_ABSOLUTE_WEAK
```

* `M_S` (SparseDictEnv soup) = **0.145508** (band weak)
* `M_D` (DenseTiedEnv soup)  = **0.313047**
* `G_sparse = M_D - M_S` = **0.167539** (gate 0.003: True)
* `G_dict-use = M_zero - M_S` = **0.854839** (gate 0.01: True)
* `G_assign = M_shuffle - M_S` = **0.017289** (gate 0.01: True)
* dictionary health: True
* official test loaded: False

## Reading

The sparse dictionary-core does not have the absolute capacity this task needs, regardless of the dense control. STOP.
