# Invalid (pre-fix) runs — not results

`LIPO_FACTOR_CUTRESP_R0_invalid_2026-11-21` and
`LIPO_DIRECT_CUTRESP_R0_invalid_2026-11-21` were cancelled at epochs 78 and 24
after two correctness defects were found (see `PROTOCOL.md` deviations 2 and 3):
`C` was not `V(G) \ S`, and the fragment CSR lost multi-frame object
multiplicity.  Their curves and weights live in the remote run directories
`/home/hxy/cy/rr-runs/trc-factor-cutresp-20261121-234935-d4617d57` and
`/home/hxy/cy/rr-runs/trc-direct-cutresp-20261121-234952-3f50925b` and are
**never** reported as measurements.
