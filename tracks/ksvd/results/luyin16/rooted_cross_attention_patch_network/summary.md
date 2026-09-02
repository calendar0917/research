# MolHIV Rooted Conditional Patch Network

Protocol: `molhiv-rooted-cross-attention-patch-network-v1`

This is a learnable patch-token screen, not a K-SVD/XGBoost feature sweep.

| variant | mean scaffold-fold AUC | fold AUC |
|---|---:|---|
| `structure` | 0.617102 | 0.612953, 0.580000, 0.658353 |
| `attribute` | 0.668323 | 0.706804, 0.675000, 0.623163 |
| `concat` | 0.633016 | 0.616532, 0.608190, 0.674327 |
| `cross_attention` | 0.629667 | 0.615628, 0.622931, 0.650441 |
| `cross_attention_shuffled_random` | 0.585460 | 0.500000, 0.627414, 0.628965 |
| `cross_attention_shuffled_size_matched` | 0.625525 | 0.559189, 0.662500, 0.654887 |

## Mechanism deltas

- `cross_attention_minus_concat`: mean `-0.003350`, wins `1/3`; fold deltas `[-0.0009042272624519843, 0.014741379310344804, -0.023886670183106085]`
- `cross_attention_minus_patch_pair_random`: mean `+0.044207`, wins `2/3`; fold deltas `[0.11562806118604474, -0.004482758620689586, 0.021475397483234127]`
- `cross_attention_minus_patch_pair_size_matched`: mean `+0.004141`, wins `1/3`; fold deltas `[0.05643885163137663, -0.03956896551724132, -0.004445784040388867]`
- `attribute_minus_structure`: mean `+0.051221`, wins `2/3`; fold deltas `[0.09385125461532662, 0.09499999999999997, -0.035189510963755444]`

The conditional model routes a permutation-invariant attribute set through a topology-conditioned expert gate. The shuffled control keeps both patch bags but breaks the same-patch pairing.

Official validation and test were not encoded or evaluated.
