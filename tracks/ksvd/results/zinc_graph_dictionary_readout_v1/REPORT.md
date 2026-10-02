# zinc_graph_dictionary_readout_v1 — frozen Full graph-dictionary readout

CPU only · official test never loaded · single frozen Full seed-0 soup.

## A. Provenance

- commit `28052f15f308fa68e42a5f5216d76c956cd6af60` · device `cpu` · threads `8` · seed `0`
- parent checkpoint `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb` (408,651 parameters, soup valid MAE `0.119154`)
- backbone state (reader excluded) `ad7e1baacce1d7e075301a622d17df55d5508e6d4b6dd5abbfcf8a135417f2e7`
- split fingerprint `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`
- train cache `51c6e93ac5f5f507102b0a4e3460c78bde47d466f64d7fd506409a8abf92d066` · valid cache `4a68c8f6566e98bb803b025e7292954dd325e90ce3c01ae9bd32a28ab8a28260` · frozen head `90193d0e56d17a655fc92e13bca4dd07ecbbb5cca513c9d3829a91cc4186e156`
- `official_test_loaded = false`

## B. Frozen representation replay

- scaffold valid replay MAE `0.119154` against published `0.119154` (delta `+0.00e+00`)
- reader replay of the captured 814-D tensor: max per-molecule `0.00e+00`
- train groups: typed_cycle_probe canonical certificate (canonical `True`)

## C. Fixed dictionary fit (train only)

- K=`256` seed `20261002` · asinh / train-fit standardiser / four-block balance
- head dev MAE by lambda: {"1e-05": 0.26647557522145177, "0.0001": 0.2671752782694034, "0.001": 0.30117523133761565}
- selected lambda `1e-05` (equal dev MAE breaks to the larger lambda)
- final head train MAE `0.221953` · solver iterations `660` · gap `9.47e-07`
- the valid split is not accepted by the fit stage and is opened only for the paired screen

## D. One paired screen on the frozen official valid

- original Full H0 readout MAE `0.119154`
- graph-dictionary readout MAE `0.248224`
- absolute gain `-0.129070` (gate >= 0.006) · new MAE <= 0.113 gate `False`
- per-molecule improvement fraction `0.2590` · median gain `-0.081157`
- five fixed-ID bins: [-0.129204, -0.112836, -0.133628, -0.094671, -0.175008] (0/5 improved)

## E. Deployment acceptance

Not purchased — the screen did not meet all frozen gates.

## F. Verdict

**GRAPH_DICTIONARY_READOUT_STOP**

> One backbone checkpoint; the official valid split has been reused across many historical rounds and selected the Full soup, so this is an exploratory screen, not an independent confirmation.

## G. Storage and capacity accounting

| item | count | storage | nature |
|---|---|---|---|
| frozen Full backbone parameters | 408,651 | 1,634,604 B (float32) | trained, frozen this round |
| prototype / normaliser / scaler buffers | mean 814, scale 814, keep 814, weights 814, centers 256x598, inverse_root 256x256, bandwidth 1, selected_rows 256, spectrum 256 | 1,773,478 B (float64) | fixed transforms, not trainable capacity |
| fitted readout coefficients | 257 | 2,056 B (float64) | the only fitted head values |
| head container `model.npz` | - | 1,779,346 B | prototype buffers + 257 coefficients + metadata |

Only the 257 coefficients are fit; the backbone and the prototype/scaler buffers are fixed.

## H. Evidence discipline

- One backbone checkpoint, one reused official-valid screen; no significance claim.
- A negative result excludes only this fixed dictionary / kernel / lambda family; it does not prove the 814-D representation is sufficient or insufficient.
- No old weak-ridge proxy, no unmatched historical delta, no inactive-ablation gain.

