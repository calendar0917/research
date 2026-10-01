# Dictionary scaling reference package (v1, 2026-10-01)

Vendored copies of the externally delivered design / NumPy reference package
for the `e2e_dictenv_scale_v1` round, together with the executable task prompt.
The originals live outside the repository; these copies are byte-identical and
are kept only for provenance and reproducibility of the pre-registration math.
They are **not** imported by the model code — the implementation in
`tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py` is the authoritative
translation.

Source: `/home/calendar/Downloads/zinc_dictionary_scaling_handoff.zip`
(package root `dictionary_scaling/`; audit revision of the repository
`00c203624682a9ca46d54a225721e25015d8b28c`).

| file | sha256 |
|---|---|
| `zinc_dictionary_scale_plan.md` | `b98dd8f4d8e7a95dc5904403c9a9546607abc9876d3b488b9c972e32ec47c447` |
| `zinc_dictionary_scale_agent_prompt.md` | `a3f803455fd69295595a4b9b5831bbed4b60692b425f674a91e1d36321963ea0` |
| `dictionary_scale_reference.py` | `e42ccc0d4eca4071558b23153771b36cb1b01c9501c0791209a9838ee5fabe19` |
| `latent_dictionary_bridge_reference.py` | `1c08f29fd403ca2b5681d87689f3df6c42562ee8dc2c0317a7a8941f2893f1f9` |
| `validate_dictionary_scaling.py` | `191f089b6c63994a98bc8abca140d56551b79afbd16ef27a7727b8a87c88b93a` |
| `dictionary_scale_audit.json` | `d484b8de35f5bdd39e55869a8eb544491a53e360a9b28a6ca739f2bb717af8e9` |
| `source_manifest.json` | `7125ccb835d38ee743b0d2ab36134bca645af38bd15ce53ec53eb0c9c892fc9d` |
| `SHA256SUMS.json` | `4cd46589e18109f644800e4221fb51036b9d457ff75d9c609675b67b71837dc4` |
| `README.md` (package, kept as `PACKAGE_README.md`) | `92f177699ff66389373050f10c75eaa2845ddbe321271cb294f24dcf8f637f6f` |

Status of the package: synthetic structural / numerical audit only.  No ZINC
data was loaded, no PyTorch forward/backward was run, no MAE claim exists and
the study repository was not modified by the package.  Running
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_dictionary_scaling.py`
in a copy of this directory regenerates `dictionary_scale_audit.json` with a
platform-dependent byte layout; the vendored JSON is the delivered artifact.
The package contains no license header; it is internal research material for
this repository and is not redistributed.
