# Typed-cycle static-object reference package (v1, 2026-10-02)

Vendored copies of the externally delivered design / NumPy reference package
for the `zinc_e2e_dictenv_typed_cycle_v1` round, together with the executable
task prompt.  The originals live outside the repository; these copies are
byte-identical and are kept only for provenance and reproducibility of the
pre-registration math.  They are **not** imported by the model code — the
implementation in `tracks/ksvd/experiments/luyin16/e2e_dictenv_typed_cycle_v1.py`
is the authoritative translation.

Source: `/home/calendar/Downloads/zinc_typed_cycle_handoff.zip`
(package root; audit revision of the repository
`7fa46ab300b43c6457499e45408296c5a4bd4c5c`).

| file | sha256 |
|---|---|
| `typed_cycle_reference.py` | `f6dfaffc499ecc6fc788fc87fceb2cd46f6f76c2deaeea973f080454d811e67f` |
| `latent_dictionary_bridge_reference.py` | `1c08f29fd403ca2b5681d87689f3df6c42562ee8dc2c0317a7a8941f2893f1f9` |
| `existing_spine_reference.py` | `e42ccc0d4eca4071558b23153771b36cb1b01c9501c0791209a9838ee5fabe19` |
| `cheap_probe_features.py` | `f9d491a7e16e203afc10740cf37d38075a55a5b8f6d52cc29f449e45f1808d77` |
| `cheap_property_probe.py` | `76905c1c96cd32b2b263a9d67b01d9da92133f75021be78c2d2d92d010491111` |
| `validate_typed_cycle.py` | `e5e6d2260ab9c4bcfec1a44d8aa09c78e96bf2c086c61eb890e89a98168369dc` |
| `validate_cheap_probe.py` | `c8d4e4097b1368a07419098f086afdff581c5a7e776b275d923b52c8e085e803` |
| `typed_cycle_audit.json` | `df18d2675270d307c49fcef80ffbafe14ea553fad635a9c5bf67107557e5e423` |
| `cheap_probe_acceptance.json` | `62ae151ac68d2999409cf8a38424614d5c19133861eb6eaa99d51193b2e5b564` |
| `source_manifest.json` | `7e334210b0cfb94167605680f9e649955f5cf4a1aba554c3457be6f727ee55d0` |
| `zinc_typed_cycle_plan.md` | `bc6b0efae6f886619bc1b2be925b64c3fab359af85674c8b051e20293aed42bc` |
| `zinc_typed_cycle_agent_prompt.md` | `134f4172ebdf9d926321a3fa78ced2c8d538e22edd3f8c08216344ad5dfc3f8f` |
| `SHA256SUMS` (package) | `780866df5466199854287c168e834ce9f9cbe0772a045f5f8fde69bebf17b784` |
| `PACKAGE_README.md` (package README) | `151a08d225be43f48c716c37007a68b70fb8a2d7dc98272b54e3a8a085d8b0cc` |

Status of the package: source-derived NumPy structural / numerical audit only.
No ZINC data was loaded, no PyTorch forward/backward was run, no MAE claim
exists and the study repository was not modified by the package.  Re-running
`python validate_typed_cycle.py` in this directory rewrites
`typed_cycle_audit.json` with a platform-dependent byte layout (the delivered
file was restored after the local acceptance rerun); the vendored JSON is the
delivered artifact.  The package carries no license header; it is internal
research material for this repository and is not redistributed.
