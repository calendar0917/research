# 真实纯结构数据的 patch substrate 审计（2026-07-30）

> 在解释 KSVD 前，先检查 sampler 是否给出了足够丰富的 motif population。若 patch 主要是单边、clique 或硬截断样本，字典失败不能单独归因于 KSVD，字典成功也可能只是规模/度原型量化。

每图最多 16 patches；每 patch 最多 24 nodes；sampling seed=0。

| 设置 | patches | size mean/median | edge-only | clique | tree | at cap | unique signatures | top-10 mass | within-graph unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REDDIT-BINARY/raw/R2 | 31914 | 16.14/22.0 | 0.007 | 0.007 | 0.360 | 0.483 | 8167 | 0.282 | 0.680 |

## 详细分布

### REDDIT-BINARY/raw/R2

- size counts：`{'2': 228, '3': 2200, '4': 2078, '5': 1748, '6': 1512, '7': 1194, '8': 967, '9': 889, '10': 707, '11': 604, '12': 486, '13': 475, '14': 439, '15': 398, '16': 373, '17': 305, '18': 337, '19': 337, '20': 297, '21': 318, '22': 293, '23': 307, '24': 15422}`；q75=24.0，q90=24.0。
- signature entropy=6.631，normalized=0.736。
- most common signatures：`[('n3|m2|d2,1,1|t0|c1|wl7027fd65cc86d3efaefa1911a201c652', 2198), ('n4|m3|d3,1,1,1|t0|c1|wl68f38cbd23a2edb067d0f8d42838916c', 1535), ('n5|m4|d4,1,1,1,1|t0|c1|wl2fb48ebbfad1620deb8ade80c84f5b90', 992), ('n24|m24|d23,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1|t1|c1|wl5e434e5a811bb1b5eaf829817772872d', 891), ('n24|m25|d23,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1|t2|c1|wl35beffab419d578a38f117f67508852f', 705)]`。

## 判断口径

1. edge-only/clique fraction 高：vocabulary 容易退化为 patch size/degree prototypes。
2. cap fraction 高：结果对 `max_nodes` 敏感，需做尺度复核。
3. within-graph unique fraction 低：增加 patch 数主要产生重复样本，不会增加结构覆盖。
4. 该审计无标签，只判断 sampler 是否为 vocabulary learning 提供了合理 substrate。
