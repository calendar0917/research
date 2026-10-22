# Kekule support tables

## Train vocabulary sizes

| vocabulary | types |
|---|---:|
| orig | 12108 |
| rep (orbit quotient) | 11744 |
| arom (ordinary aromaticisation) | 13195 |
| union (joint statistics) | 13168 |

## Merge / split accounting (train occurrences)

| statistic | value |
|---|---:|
| atom occurrences | 231664 |
| occurrences with >1 legal key | 50865 |
| orig keys with >1 rep image (context split) | 316 |
| rep merge classes | 686 |
| rep merged extra original keys | 699 |
| rep classes with >1 witness component | 1 |
| rep unwitnessed singleton keys | 2 |
| orig keys with >1 arom image (context split) | 1458 |
| arom merge classes | 929 |
| arom merged extra original keys | 1075 |
| arom classes with >1 witness component | 227 |
| arom unwitnessed singleton keys | 380 |

## Group support

| group | n | atoms | scheme | unseen atoms | pooled unseen frac | mean per-mol frac | fully supported |
|---|---:|---:|---|---:|---:|---:|---:|
| all | 1000 | 23083 | orig | 506 | 0.021921 | 0.024609 | 749/1000 |
| all | 1000 | 23083 | rep | 495 | 0.021444 | 0.024039 | 753/1000 |
| all | 1000 | 23083 | arom | 556 | 0.024087 | 0.026710 | 735/1000 |
| all | 1000 | 23083 | union | 477 | 0.020665 | 0.023304 | 763/1000 |
| seen_ordinary | 710 | 16818 | orig | 0 | 0.000000 | 0.000000 | 710/710 |
| seen_ordinary | 710 | 16818 | rep | 5 | 0.000297 | 0.000276 | 705/710 |
| seen_ordinary | 710 | 16818 | arom | 35 | 0.002081 | 0.001958 | 684/710 |
| seen_ordinary | 710 | 16818 | union | 0 | 0.000000 | 0.000000 | 710/710 |
| seen_tail | 39 | 834 | orig | 0 | 0.000000 | 0.000000 | 39/39 |
| seen_tail | 39 | 834 | rep | 0 | 0.000000 | 0.000000 | 39/39 |
| seen_tail | 39 | 834 | arom | 0 | 0.000000 | 0.000000 | 39/39 |
| seen_tail | 39 | 834 | union | 0 | 0.000000 | 0.000000 | 39/39 |
| unseen_ordinary | 225 | 4911 | orig | 433 | 0.088169 | 0.092183 | 0/225 |
| unseen_ordinary | 225 | 4911 | rep | 417 | 0.084911 | 0.088777 | 9/225 |
| unseen_ordinary | 225 | 4911 | arom | 443 | 0.090206 | 0.093873 | 12/225 |
| unseen_ordinary | 225 | 4911 | union | 404 | 0.082264 | 0.086380 | 14/225 |
| unseen_tail | 26 | 520 | orig | 73 | 0.140385 | 0.148775 | 0/26 |
| unseen_tail | 26 | 520 | rep | 73 | 0.140385 | 0.148775 | 0/26 |
| unseen_tail | 26 | 520 | arom | 78 | 0.150000 | 0.161474 | 0/26 |
| unseen_tail | 26 | 520 | union | 73 | 0.140385 | 0.148775 | 0/26 |

## Rescued / lost atom occurrences

| group | rescued rep | rescued arom | rescued union | lost rep | lost arom |
|---|---:|---:|---:|---:|---:|
| all | 19 | 34 | 29 | 8 | 84 |
| seen_ordinary | 0 | 0 | 0 | 5 | 35 |
| seen_tail | 0 | 0 | 0 | 0 | 0 |
| unseen_ordinary | 19 | 34 | 29 | 3 | 44 |
| unseen_tail | 0 | 0 | 0 | 0 | 5 |

## H-inference audit

| split | inferred H | audited | explicit-H capacity outside {0,1} | no solution | original missing | H-changing candidates |
|---|---:|---:|---:|---:|---:|---:|
| valid | 911 | 911 | 907 | 3 | 1 | 1 |
| train | 9156 | 9156 | 9112 | 42 | 2 | 2 |
