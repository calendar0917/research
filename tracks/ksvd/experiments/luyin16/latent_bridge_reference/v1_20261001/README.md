# Latent bridge reference package (v1, 2026-10-01)

Vendored copies of the externally delivered design / NumPy reference package
for the `e2e_dictenv_latent_bridge_v1` round.  The originals live outside the
repository; these copies are byte-identical and are kept only for provenance
and reproducibility of the pre-registration math.  They are **not** imported by
the model code — the implementation in
`tracks/ksvd/experiments/luyin16/e2e_dictenv_latent_bridge_v1.py` is the
authoritative translation.

Source: `/home/calendar/Downloads/zinc_dictionary_next_step/`
(review revision `65d4b8fb82261e9ec0a94bbf02d7ccc36e14219f`).

| file | sha256 |
|---|---|
| `zinc_dictionary_next_step.md` | `355cedcae5668bad43d018357384a671b9c4aaaa3f3cc278af69659ef2e4b4c5` |
| `latent_dictionary_bridge_reference.py` | `1c08f29fd403ca2b5681d87689f3df6c42562ee8dc2c0317a7a8941f2893f1f9` |
| `validate_dictionary_bridge.py` | `dd61dd051835bfe193085d2437cdc0b607caee0f56d9191b36d6db6b943dc063` |
| `dictionary_bridge_audit.json` | `20554ff6f5088bce495b4ff906dacfa2dc5fb80fa204363d7029f2fa52825b1f` |

Status of the package: synthetic structural / numerical audit only.  No ZINC
data was loaded, no PyTorch backward was run, no MAE claim exists and the study
repository was not modified by the package.  Running
`python validate_dictionary_bridge.py` in a copy of this directory regenerates
`dictionary_bridge_audit.json` with a platform-dependent byte layout; the
vendored JSON is the delivered artifact.

The package contains no license header; it is internal research material for
this repository and is not redistributed.
