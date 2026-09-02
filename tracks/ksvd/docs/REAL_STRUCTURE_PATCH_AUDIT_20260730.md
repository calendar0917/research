# 真实纯结构数据的 patch substrate 审计（2026-07-30）

> 在解释 KSVD 前，先检查 sampler 是否给出了足够丰富的 motif population。若 patch 主要是单边、clique 或硬截断样本，字典失败不能单独归因于 KSVD，字典成功也可能只是规模/度原型量化。

每图最多 16 patches；每 patch 最多 12 nodes；sampling seed=0。

| 设置 | patches | size mean/median | edge-only | clique | tree | at cap | unique signatures | top-10 mass | within-graph unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY/cleaned/B0 | 7652 | 8.00/7.0 | 0.000 | 0.824 | 0.000 | 0.200 | 311 | 0.827 | 0.306 |
| IMDB-MULTI/cleaned/B0 | 4753 | 8.36/8.0 | 0.001 | 0.818 | 0.001 | 0.264 | 253 | 0.820 | 0.287 |
| REDDIT-BINARY/raw/B0 | 31914 | 2.88/2.0 | 0.619 | 0.653 | 0.902 | 0.017 | 213 | 0.948 | 0.283 |
| REDDIT-BINARY/raw/R2 | 31914 | 9.66/12.0 | 0.007 | 0.007 | 0.409 | 0.620 | 4188 | 0.454 | 0.635 |

## 详细分布

### IMDB-BINARY/cleaned/B0

- size counts：`{'2': 1, '3': 23, '4': 416, '5': 947, '6': 1331, '7': 1149, '8': 1002, '9': 511, '10': 428, '11': 316, '12': 1528}`；q75=10.0，q90=12.0。
- signature entropy=3.068，normalized=0.535。
- most common signatures：`[('n6|m15|d5,5,5,5,5,5|t20|c1|wle5be41ac873fd68e9803271420963b83', 1260), ('n7|m21|d6,6,6,6,6,6,6|t35|c1|wle372ad80e7a606d55dadcc076a7748e5', 1039), ('n5|m10|d4,4,4,4,4|t10|c1|wlbde135a811cfdf794cdaa83144f8362f', 917), ('n12|m66|d11,11,11,11,11,11,11,11,11,11,11,11|t220|c1|wla242e3d918bccecfe9af99a03d43b8dc', 893), ('n8|m28|d7,7,7,7,7,7,7,7|t56|c1|wl7195185881c25b224bb4a573eafb7cad', 852)]`。

### IMDB-MULTI/cleaned/B0

- size counts：`{'2': 4, '3': 17, '4': 269, '5': 507, '6': 826, '7': 587, '8': 420, '9': 371, '10': 246, '11': 253, '12': 1253}`；q75=12.0，q90=12.0。
- signature entropy=3.077，normalized=0.556。
- most common signatures：`[('n12|m66|d11,11,11,11,11,11,11,11,11,11,11,11|t220|c1|wla242e3d918bccecfe9af99a03d43b8dc', 870), ('n6|m15|d5,5,5,5,5,5|t20|c1|wle5be41ac873fd68e9803271420963b83', 779), ('n5|m10|d4,4,4,4,4|t10|c1|wlbde135a811cfdf794cdaa83144f8362f', 493), ('n7|m21|d6,6,6,6,6,6,6|t35|c1|wle372ad80e7a606d55dadcc076a7748e5', 489), ('n8|m28|d7,7,7,7,7,7,7,7|t56|c1|wl7195185881c25b224bb4a573eafb7cad', 351)]`。

### REDDIT-BINARY/raw/B0

- size counts：`{'2': 19759, '3': 6648, '4': 2351, '5': 1115, '6': 576, '7': 374, '8': 234, '9': 152, '10': 97, '11': 73, '12': 535}`；q75=3.0，q90=4.0。
- signature entropy=1.516，normalized=0.283。
- most common signatures：`[('n2|m1|d1,1|t0|c1|wl214f669fa084665b09f671ac302ce9ce', 19759), ('n3|m2|d2,1,1|t0|c1|wl7027fd65cc86d3efaefa1911a201c652', 5581), ('n4|m3|d3,1,1,1|t0|c1|wl68f38cbd23a2edb067d0f8d42838916c', 1730), ('n3|m3|d2,2,2|t1|c1|wl2fb3866368d0f21768e2c0e53c32d66c', 1067), ('n5|m4|d4,1,1,1,1|t0|c1|wl2fb48ebbfad1620deb8ade80c84f5b90', 751)]`。

### REDDIT-BINARY/raw/R2

- size counts：`{'2': 228, '3': 2200, '4': 2078, '5': 1748, '6': 1512, '7': 1194, '8': 967, '9': 889, '10': 707, '11': 604, '12': 19787}`；q75=12.0，q90=12.0。
- signature entropy=5.295，normalized=0.635。
- most common signatures：`[('n12|m12|d11,2,2,1,1,1,1,1,1,1,1,1|t1|c1|wl0e94bb54079e9f9705a78f94ea69492b', 2886), ('n12|m11|d11,1,1,1,1,1,1,1,1,1,1,1|t0|c1|wl51f40fdf60d75c9aea14fe2e6eb215bb', 2767), ('n3|m2|d2,1,1|t0|c1|wl7027fd65cc86d3efaefa1911a201c652', 2198), ('n4|m3|d3,1,1,1|t0|c1|wl68f38cbd23a2edb067d0f8d42838916c', 1535), ('n12|m13|d11,2,2,2,2,1,1,1,1,1,1,1|t2|c1|wldcce330219d3744ea56dbf9a82b36636', 1200)]`。

## 判断口径

1. edge-only/clique fraction 高：vocabulary 容易退化为 patch size/degree prototypes。
2. cap fraction 高：结果对 `max_nodes` 敏感，需做尺度复核。
3. within-graph unique fraction 低：增加 patch 数主要产生重复样本，不会增加结构覆盖。
4. 该审计无标签，只判断 sampler 是否为 vocabulary learning 提供了合理 substrate。
