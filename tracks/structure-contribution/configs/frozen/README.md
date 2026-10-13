# Frozen R0 replay references (cscl-bjoc-recipe-tuning-v1)

`bjoc_recipe_tuning_v1_valid_P{0,1}_pred.npy` are the frozen
masked-pretrained P predictions on the **official-valid structures only**
(1000 float32 values, labels never involved).

Why they are committed: the compute checkout on `res` does not carry the
git-ignored `results/bjoc_full10k_pretraining_transfer_v1/predictions/*.npz`
files, and the R0 replay must still be verifiable on the remote host.  A
SHA-256 of the prediction array is *not* portable across CPUs (float32
reduction ordering differs between the machine that produced the npz and the
res CPUs), so the array itself is committed as the frozen reference and the
replay compares it numerically under the pre-declared tolerance
`r0_replay_tolerance = 1e-4`.

Provenance:

| item | seed 0 | seed 1 |
|---|---|---|
| source npz | `results/bjoc_full10k_pretraining_transfer_v1/predictions/valid_P0.npz` | `.../valid_P1.npz` |
| source npz sha256 (frozen config) | `c6a3caba92224a097e73dd61c11388e31eaeaa20e7cdaff8901bb2ab350b6037` | `0b57560ff05e0f6da0279db3b038048bd46a26d62649a607cbecf91b9a722adb` |
| checkpoint | `soup_P_seed0.pt` sha `73118e9b...` | `soup_P_seed1.pt` sha `bf640cd8...` |
| generation device recorded in the P round | `cpu` | `cpu` |
| array sha256 (float32 bytes) | `e7fdb3ab2c22683cac11496731d9ed36354c58915ec7da5dd469ae0487e647d3` | `3d6403655a0a4dc020035285295b2694dd4dbd234f4d8af88b99eb66544b4830` |
| committed npy file sha256 | `8fba5053e8cf1f42b3929e882c47526a9c745eb7ca4dc3900566575d154b7ac4` | `e444020e4948c883de34d27e8ce246ec999997d57e47770f0bf63e5fe3b2bfb2` |

These arrays are evaluation references only: they are never used for training,
selection or retuning, and the official test was never loaded.
