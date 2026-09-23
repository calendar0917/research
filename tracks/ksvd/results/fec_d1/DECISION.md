# FEC-D1 — decision

```
FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC
```

* `M_B` (frozen FEC-S1 best replay): 0.136782
* `M_D` (Dict32 localized binding soup): 0.131537
* `M_P` (PCA32 localized binding soup): 0.130231
* `M_shuffle` (assignment shuffle): 0.144669
* `G_D`: 0.005245 (A=True)
* `G_dict-specific`: -0.001306 (B=False)
* `G_assign`: 0.013132 (C=True)

## Stop reason

Local R65 structure x chemistry refinement helps, but the sparse dictionary is not better than the dense PCA32 coordinate. STOP the dictionary performance route.

official_test_loaded = False
