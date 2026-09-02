# MolHIV radius-2 patch-object audit

Protocol: `luyin16-molhiv-patch-object-audit-v1`

## Decision

- relabel invariance: **FAIL**
- matched topology/attribute scope: **FAIL**
- modulo category encoding: **FAIL**
- proceed to fold attribution: **False**

## Relabel audit

- audited graphs: 256
- comparisons: 1280
- typed readout maximum drift: 0.36650833
- topology readout maximum drift: 0.40824829
- frozen prediction maximum drift: 0.34539089
- cached-feature reproduction maximum error: 2.932927e-08

## Object scope

- truncated patch fraction: 0.141206
- graphs with truncation: 0.929688
- legacy vs retained-node readout maximum relative L2: 0.074704
- mean dropped nodes per patch: 0.222111
- mean dropped edges per patch: 0.265452

## Category encoding

- atom raw categories: 55; colliding modulo bins: 16
- atom occurrences in colliding bins: 1.000000
- bond raw categories: 4; colliding modulo bins: 0

## Interpretation

This audit changes no model selection and never evaluates official test. A failed gate means the current object must be corrected before RAW/INIT/FINAL attribution.
