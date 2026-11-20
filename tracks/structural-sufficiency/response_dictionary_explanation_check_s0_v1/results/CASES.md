# Explanation check — auditable cases (max 12)

Selection rule (fixed before results): the first 6 interpretable pairs with at least one non-Train endpoint, then the first 6 interpretable Train–Train pairs, ordered by catalogue `pair_id`. Nothing is selected by effect size.

`C` is the shared complement side of the selected existing view; `S_A`/`S_B` are the small sides. `*` marks the cut bond. The word account uses `D_RESPONSE` and the actual `S_A` as reference. `HP_CERTIFIED` is the primary high-precision version; `HP_SLSQP_ALL` is the uncertified sensitivity that also replaces rows where SLSQP missed the 1e-8 gap certificate.

Interpretable pairs: **396/823** (48.1%). Word-account closure max |error| = 2.50e-15. Mapping: 823 pairs, 231 components; cell counts reproduced = True.

Train–Train cases describe the frozen function; they are not new-structure generalisation. Held-out cases are low-support: the whole interpretable set has 5 Test molecules / 5 Train–Test components and 3 Valid molecules / 3 Train–Valid components (the full 823-pair support is 7 Test / 7 components and 4 Valid / 4 components; the shared-C restriction removes 2 Test and 1 Valid molecule). Cases 1–6 use only 2 distinct Test molecules (rows 37 and 758) / 2 components, so they are not 6 independent held-out experiments. Cases 7–12 use component 28 (5 cases) and component 53 (1 case); they are also clustered, not 6 independent Train examples.

## Summary table

| # | pair | cell | component | Δy | Δlocal FW | Δfull FW | rem FW | Δlocal HP | Δfull HP | rem HP |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `p00028_00758` | Train-Test | 28 | +0.0900 | -0.1283 | -0.1109 | +0.0173 | -0.1283 | -0.1109 | +0.0173 |
| 2 | `p00037_00072` | Train-Test | 37 | +1.1700 | +0.2152 | +0.1736 | -0.0417 | +0.2152 | +0.1736 | -0.0417 |
| 3 | `p00037_01164` | Train-Test | 37 | +1.7800 | +0.4943 | +0.3811 | -0.1132 | +0.4943 | +0.3811 | -0.1132 |
| 4 | `p00037_01265` | Train-Test | 37 | +0.7200 | +0.1388 | +0.0298 | -0.1090 | +0.1388 | +0.0298 | -0.1090 |
| 5 | `p00037_01772` | Train-Test | 37 | +1.6400 | +0.4263 | +0.3627 | -0.0637 | +0.4263 | +0.3627 | -0.0637 |
| 6 | `p00037_02195` | Train-Test | 37 | +1.4600 | +0.4362 | +0.2133 | -0.2229 | +0.4362 | +0.2133 | -0.2229 |
| 7 | `p00028_01765` | Train-Train | 28 | +0.3000 | +0.1462 | +0.1370 | -0.0093 | +0.1462 | +0.1370 | -0.0093 |
| 8 | `p00028_02739` | Train-Train | 28 | +0.4000 | +0.3440 | +0.2577 | -0.0863 | +0.3440 | +0.2577 | -0.0863 |
| 9 | `p00028_03000` | Train-Train | 28 | -0.3000 | -0.3371 | -0.3871 | -0.0500 | -0.3371 | -0.3871 | -0.0500 |
| 10 | `p00028_03004` | Train-Train | 28 | +0.0500 | +0.0124 | -0.1239 | -0.1363 | +0.0124 | -0.1239 | -0.1363 |
| 11 | `p00028_03597` | Train-Train | 28 | -0.1300 | -0.3005 | -0.3868 | -0.0863 | -0.3005 | -0.3868 | -0.0863 |
| 12 | `p00053_04033` | Train-Train | 53 | +0.2000 | +0.3250 | +0.3456 | +0.0207 | +0.3250 | +0.3456 | +0.0207 |

## Case 1 — `p00028_00758` (Train-Test, component 28)

**Molecules.** `A` (test, row 758): `CC1(CC1)c2c(cnn2c3ccc(cc3)C(=O)O)C(=O)NC4C5CC6CC(CC4C6)C5`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `4` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]C1(C)CC1=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 11331 (`*C1(C)CC1`, 4 atoms, Train word=False, prototype=False); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | -0.1283 | 1.0000 | -0.1283 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = +0.0900; Δresp (native RESP full F) = -0.1532; error of native Δresp = -0.2432; native one-view Δlocal = -0.2128; native remainder = +0.0596. Dictionary vs native model: Δfull error = +0.0423 (HP) / +0.0423 (FW); Δlocal error = +0.0845 (HP) / +0.0845 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | -0.1283 | +0.0173 | -0.1109 | -0.2009 | -0.2183 |
| HP_CERTIFIED | -0.1283 | +0.0173 | -0.1109 | -0.2009 | -0.2183 |
| HP_SLSQP_ALL | -0.1255 | +0.0160 | -0.1096 | -0.1996 | -0.2155 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0027 (all); ΔΔfull = +0.0000 (HP) / +0.0014 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = -0.1004, remainder = +0.0068; D_FREQ Δfull = -0.1379, remainder = +0.0443.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 2 — `p00037_00072` (Train-Test, component 37)

**Molecules.** `A` (train, row 72): `Cn1cncc1c2c3C(=O)N(CC#C)C(=O)N(CC4CC4)c3nn2Cc5ccnc6ccc(Cl)cc56`  
`B` (test, row 37): `Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67`  
Catalogue edit: `11` → `11` on core `[1*]Cn1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O` (34 heavy atoms); class `[1*]C#C=>[1*]C1CC1`.

**Common context.** `C` = fragment 193 (33 atoms, key `f73b5cdb671c8a3f…`), cut bond `0/0/0`. `S_A` = 192 (`*CC#C`, 3 atoms, Train word=True, prototype=False); `S_B` = 194 (`*CC1CC1`, 4 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 1875 | `*CC1CCOCC1` | C | 0.3331 | +0.2479 | 0.3331 | +0.2479 |
| 1104 | `*C#Cc1ccccc1` | C | 0.3927 | +0.0652 | 0.3927 | +0.0652 |
| 3414 | `*CC(F)(F)F` | C | 0.1373 | -0.0515 | 0.1373 | -0.0515 |
| 21 | `*CC(O)CO` | C | 0.0315 | -0.0184 | 0.0315 | -0.0184 |
| 0 (D0) | `*C` | C | 0.0722 | -0.0152 | 0.0722 | -0.0152 |

**Numbers.** Δy = +1.1700; Δresp (native RESP full F) = +0.1948; error of native Δresp = -0.9752; native one-view Δlocal = +0.2625; native remainder = -0.0677. Dictionary vs native model: Δfull error = -0.0213 (HP) / -0.0213 (FW); Δlocal error = -0.0473 (HP) / -0.0473 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.2152 | -0.0417 | +0.1736 | -0.9964 | -0.9548 |
| HP_CERTIFIED | +0.2152 | -0.0417 | +0.1736 | -0.9964 | -0.9548 |
| HP_SLSQP_ALL | +0.2191 | -0.0439 | +0.1751 | -0.9949 | -0.9509 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 1.373e-01 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0039 (all); ΔΔfull = +0.0000 (HP) / +0.0016 (all); closure errors FW/HP = -1.11e-16/-1.11e-16. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.1815, remainder = -0.0575; D_FREQ Δfull = +0.2029, remainder = -0.0111.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 3 — `p00037_01164` (Train-Test, component 37)

**Molecules.** `A` (train, row 1164): `Cn1cncc1c2c3C(=O)N(CC#CCO)C(=O)N(CC4CC4)c3nn2Cc5ccnc6ccc(Cl)cc56`  
`B` (test, row 37): `Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67`  
Catalogue edit: `11` → `11` on core `[1*]Cn1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O` (34 heavy atoms); class `[1*]C#CCO=>[1*]C1CC1`.

**Common context.** `C` = fragment 193 (33 atoms, key `f73b5cdb671c8a3f…`), cut bond `0/0/0`. `S_A` = 2245 (`*CC#CCO`, 5 atoms, Train word=True, prototype=False); `S_B` = 194 (`*CC1CC1`, 4 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 1875 | `*CC1CCOCC1` | C | 0.3331 | +0.3409 | 0.3331 | +0.3409 |
| 1104 | `*C#Cc1ccccc1` | C | 0.3927 | +0.1748 | 0.3927 | +0.1748 |
| 3414 | `*CC(F)(F)F` | C | 0.1373 | -0.0132 | 0.1373 | -0.0132 |
| 308 | `*Cc1ccc(Cl)cc1` | C | 0.0186 | +0.0128 | 0.0186 | +0.0128 |
| 0 (D0) | `*C` | C | 0.0722 | +0.0049 | 0.0722 | +0.0049 |

**Numbers.** Δy = +1.7800; Δresp (native RESP full F) = +0.4029; error of native Δresp = -1.3771; native one-view Δlocal = +0.5184; native remainder = -0.1155. Dictionary vs native model: Δfull error = -0.0218 (HP) / -0.0218 (FW); Δlocal error = -0.0241 (HP) / -0.0241 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.4943 | -0.1132 | +0.3811 | -1.3989 | -1.2857 |
| HP_CERTIFIED | +0.4943 | -0.1132 | +0.3811 | -1.3989 | -1.2857 |
| HP_SLSQP_ALL | +0.4645 | -0.0916 | +0.3729 | -1.4071 | -1.3155 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 1.373e-01 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / -0.0298 (all); ΔΔfull = +0.0000 (HP) / -0.0082 (all); closure errors FW/HP = -2.22e-16/-2.22e-16. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.3880, remainder = -0.0768; D_FREQ Δfull = +0.3747, remainder = -0.0308.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 4 — `p00037_01265` (Train-Test, component 37)

**Molecules.** `A` (train, row 1265): `CC#CCN1C(=O)N(CC2CC2)c3nn(Cc4ccnc5ccc(Cl)cc45)c(c3C1=O)c6cncn6C`  
`B` (test, row 37): `Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67`  
Catalogue edit: `3` → `11` on core `[1*]Cn1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O` (34 heavy atoms); class `[1*]C#CC=>[1*]C1CC1`.

**Common context.** `C` = fragment 2426 (34 atoms, key `f14c89f01b323813…`), cut bond `0/0/0`. `S_A` = 2425 (`*C#CC`, 3 atoms, Train word=True, prototype=False); `S_B` = 206 (`*C1CC1`, 3 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 1875 | `*CC1CCOCC1` | C | 0.4591 | +0.2678 | 0.4591 | +0.2678 |
| 262 | `*C(F)F` | C | 0.4381 | -0.1031 | 0.4381 | -0.1031 |
| 0 (D0) | `*C` | C | 0.0755 | -0.0225 | 0.0755 | -0.0225 |
| 31 | `*C(=O)O` | C | 0.0041 | -0.0107 | 0.0041 | -0.0107 |

**Numbers.** Δy = +0.7200; Δresp (native RESP full F) = +0.0540; error of native Δresp = -0.6660; native one-view Δlocal = +0.1820; native remainder = -0.1280. Dictionary vs native model: Δfull error = -0.0242 (HP) / -0.0242 (FW); Δlocal error = -0.0432 (HP) / -0.0432 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.1388 | -0.1090 | +0.0298 | -0.6902 | -0.5812 |
| HP_CERTIFIED | +0.1388 | -0.1090 | +0.0298 | -0.6902 | -0.5812 |
| HP_SLSQP_ALL | +0.1501 | -0.1115 | +0.0386 | -0.6814 | -0.5699 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 3.339e-02 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0113 (all); ΔΔfull = +0.0000 (HP) / +0.0088 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.0151, remainder = -0.0706; D_FREQ Δfull = +0.0322, remainder = -0.1480.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 5 — `p00037_01772` (Train-Test, component 37)

**Molecules.** `A` (train, row 1772): `CN1C(=O)N(CC2CC2)c3nn(Cc4ccnc5ccc(Cl)cc45)c(c6cncn6C)c3C1=O`  
`B` (test, row 37): `Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67`  
Catalogue edit: `1` → `10` on core `[1*]n1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O` (33 heavy atoms); class `[1*]C=>[1*]CC1CC1`.

**Common context.** `C` = fragment 193 (33 atoms, key `f73b5cdb671c8a3f…`), cut bond `0/0/0`. `S_A` = 0 (`*C`, 1 atoms, Train word=True, prototype=True); `S_B` = 194 (`*CC1CC1`, 4 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 1875 | `*CC1CCOCC1` | C | 0.3331 | +0.3182 | 0.3331 | +0.3182 |
| 1104 | `*C#Cc1ccccc1` | C | 0.3927 | +0.1481 | 0.3927 | +0.1481 |
| 3414 | `*CC(F)(F)F` | C | 0.1373 | -0.0225 | 0.1373 | -0.0225 |
| 1715 | `*CC(O)CNC(C)C` | C | 0.0125 | -0.0123 | 0.0125 | -0.0123 |
| 0 (D0) | `*C` | C | 0.0722 | +0.0000 | 0.0722 | +0.0000 |

**Numbers.** Δy = +1.6400; Δresp (native RESP full F) = +0.3623; error of native Δresp = -1.2777; native one-view Δlocal = +0.4056; native remainder = -0.0434. Dictionary vs native model: Δfull error = +0.0004 (HP) / +0.0004 (FW); Δlocal error = +0.0207 (HP) / +0.0207 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.4263 | -0.0637 | +0.3627 | -1.2773 | -1.2137 |
| HP_CERTIFIED | +0.4263 | -0.0637 | +0.3627 | -1.2773 | -1.2137 |
| HP_SLSQP_ALL | +0.3978 | -0.0407 | +0.3571 | -1.2829 | -1.2422 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 1.373e-01 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / -0.0286 (all); ΔΔfull = +0.0000 (HP) / -0.0056 (all); closure errors FW/HP = -1.67e-16/-1.67e-16. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.3636, remainder = -0.0622; D_FREQ Δfull = +0.3685, remainder = +0.0217.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 6 — `p00037_02195` (Train-Test, component 37)

**Molecules.** `A` (train, row 2195): `CCN1C(=O)N(CC2CC2)c3nn(Cc4ccnc5ccc(Cl)cc45)c(c6cncn6C)c3C1=O`  
`B` (test, row 37): `Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67`  
Catalogue edit: `1` → `11` on core `[1*]Cn1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O` (34 heavy atoms); class `[1*]C=>[1*]C1CC1`.

**Common context.** `C` = fragment 2426 (34 atoms, key `f14c89f01b323813…`), cut bond `0/0/0`. `S_A` = 0 (`*C`, 1 atoms, Train word=True, prototype=True); `S_B` = 206 (`*C1CC1`, 3 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 1875 | `*CC1CCOCC1` | C | 0.4591 | +0.4043 | 0.4591 | +0.4043 |
| 262 | `*C(F)F` | C | 0.4381 | +0.0271 | 0.4381 | +0.0271 |
| 308 | `*Cc1ccc(Cl)cc1` | C | 0.0231 | +0.0141 | 0.0231 | +0.0141 |
| 31 | `*C(=O)O` | C | 0.0041 | -0.0095 | 0.0041 | -0.0095 |
| 0 (D0) | `*C` | C | 0.0755 | +0.0000 | 0.0755 | +0.0000 |

**Numbers.** Δy = +1.4600; Δresp (native RESP full F) = +0.2197; error of native Δresp = -1.2403; native one-view Δlocal = +0.4867; native remainder = -0.2671. Dictionary vs native model: Δfull error = -0.0064 (HP) / -0.0064 (FW); Δlocal error = -0.0506 (HP) / -0.0506 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.4362 | -0.2229 | +0.2133 | -1.2467 | -1.0238 |
| HP_CERTIFIED | +0.4362 | -0.2229 | +0.2133 | -1.2467 | -1.0238 |
| HP_SLSQP_ALL | +0.4335 | -0.2256 | +0.2079 | -1.2521 | -1.0265 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 3.339e-02 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / -0.0026 (all); ΔΔfull = +0.0000 (HP) / -0.0054 (all); closure errors FW/HP = 1.67e-16/1.67e-16. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.2143, remainder = -0.2182; D_FREQ Δfull = +0.2246, remainder = -0.2621.

**Scope.** Has a non-Train endpoint: held-out support is LOW (see the report); this is not a task-law validation.

## Case 7 — `p00028_01765` (Train-Train, component 28)

**Molecules.** `A` (train, row 1765): `CCc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `2` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]CC=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 126 (`*CC`, 2 atoms, Train word=True, prototype=False); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | +0.1462 | 1.0000 | +0.1462 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = +0.3000; Δresp (native RESP full F) = +0.1405; error of native Δresp = -0.1595; native one-view Δlocal = +0.1534; native remainder = -0.0128. Dictionary vs native model: Δfull error = -0.0036 (HP) / -0.0036 (FW); Δlocal error = -0.0071 (HP) / -0.0071 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.1462 | -0.0093 | +0.1370 | -0.1630 | -0.1538 |
| HP_CERTIFIED | +0.1462 | -0.0093 | +0.1370 | -0.1630 | -0.1538 |
| HP_SLSQP_ALL | +0.1565 | -0.0144 | +0.1421 | -0.1579 | -0.1435 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0103 (all); ΔΔfull = +0.0000 (HP) / +0.0051 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.1380, remainder = -0.0103; D_FREQ Δfull = +0.1405, remainder = -0.0128.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

## Case 8 — `p00028_02739` (Train-Train, component 28)

**Molecules.** `A` (train, row 2739): `Cc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `1` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]C=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 0 (`*C`, 1 atoms, Train word=True, prototype=True); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | +0.3440 | 1.0000 | +0.3440 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = +0.4000; Δresp (native RESP full F) = +0.2577; error of native Δresp = -0.1423; native one-view Δlocal = +0.3440; native remainder = -0.0863. Dictionary vs native model: Δfull error = +0.0000 (HP) / +0.0000 (FW); Δlocal error = +0.0000 (HP) / +0.0000 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.3440 | -0.0863 | +0.2577 | -0.1423 | -0.0560 |
| HP_CERTIFIED | +0.3440 | -0.0863 | +0.2577 | -0.1423 | -0.0560 |
| HP_SLSQP_ALL | +0.3440 | -0.0863 | +0.2577 | -0.1423 | -0.0560 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0000 (all); ΔΔfull = +0.0000 (HP) / +0.0000 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.2577, remainder = -0.0863; D_FREQ Δfull = +0.2577, remainder = -0.0863.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

## Case 9 — `p00028_03000` (Train-Train, component 28)

**Molecules.** `A` (train, row 3000): `OC(=O)c1ccc(cc1)n2ncc(C(=O)NC3C4CC5CC(CC3C5)C4)c2C(F)(F)F`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `26` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]C(F)(F)F=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 95 (`*C(F)(F)F`, 4 atoms, Train word=True, prototype=False); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | -0.3371 | 1.0000 | -0.3371 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = -0.3000; Δresp (native RESP full F) = -0.3883; error of native Δresp = -0.0883; native one-view Δlocal = -0.3418; native remainder = -0.0464. Dictionary vs native model: Δfull error = +0.0012 (HP) / +0.0012 (FW); Δlocal error = +0.0047 (HP) / +0.0047 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | -0.3371 | -0.0500 | -0.3871 | -0.0871 | -0.0371 |
| HP_CERTIFIED | -0.3371 | -0.0500 | -0.3871 | -0.0871 | -0.0371 |
| HP_SLSQP_ALL | -0.3371 | -0.0500 | -0.3871 | -0.0871 | -0.0371 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0000 (all); ΔΔfull = +0.0000 (HP) / +0.0000 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = -0.3977, remainder = -0.0180; D_FREQ Δfull = -0.3883, remainder = -0.0464.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

## Case 10 — `p00028_03004` (Train-Train, component 28)

**Molecules.** `A` (train, row 3004): `CC(C)c1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `3` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]C(C)C=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 120 (`*C(C)C`, 3 atoms, Train word=True, prototype=False); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | +0.0124 | 1.0000 | +0.0124 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = +0.0500; Δresp (native RESP full F) = -0.1192; error of native Δresp = -0.1692; native one-view Δlocal = +0.0264; native remainder = -0.1456. Dictionary vs native model: Δfull error = -0.0046 (HP) / -0.0046 (FW); Δlocal error = -0.0139 (HP) / -0.0139 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.0124 | -0.1363 | -0.1239 | -0.1739 | -0.0376 |
| HP_CERTIFIED | +0.0124 | -0.1363 | -0.1239 | -0.1739 | -0.0376 |
| HP_SLSQP_ALL | +0.0075 | -0.1330 | -0.1255 | -0.1755 | -0.0425 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / -0.0050 (all); ΔΔfull = +0.0000 (HP) / -0.0017 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = -0.1327, remainder = -0.1186; D_FREQ Δfull = -0.1192, remainder = -0.1456.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

## Case 11 — `p00028_03597` (Train-Train, component 28)

**Molecules.** `A` (train, row 3597): `OC(=O)c1ccc(cc1)n2ncc(C(=O)NC3C4CC5CC(CC3C5)C4)c2C6CCC6`  
`B` (train, row 28): `CSc1c(cnn1c2ccc(cc2)C(=O)O)C(=O)NC3C4CC5CC(CC3C5)C4`  
Catalogue edit: `26` → `2` on core `[1*]c1c(C(=O)NC2C3CC4CC(C3)CC2C4)cnn1-c1ccc(C(=O)O)cc1` (27 heavy atoms); class `[1*]C1CCC1=>[1*]SC`.

**Common context.** `C` = fragment 90 (27 atoms, key `ce4823a778008715…`), cut bond `0/0/0`. `S_A` = 1067 (`*C1CCC1`, 4 atoms, Train word=True, prototype=False); `S_B` = 89 (`*SC`, 2 atoms, Train word=True, prototype=True); same τ = False.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=15,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 89 | `*SC` | S | 1.0000 | -0.3005 | 1.0000 | -0.3005 |
| 45 (D0) | `*S(C)(=O)=O` | S | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

**Numbers.** Δy = -0.1300; Δresp (native RESP full F) = -0.4553; error of native Δresp = -0.3253; native one-view Δlocal = -0.3690; native remainder = -0.0863. Dictionary vs native model: Δfull error = +0.0684 (HP) / +0.0684 (FW); Δlocal error = +0.0684 (HP) / +0.0684 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | -0.3005 | -0.0863 | -0.3868 | -0.2568 | -0.1705 |
| HP_CERTIFIED | -0.3005 | -0.0863 | -0.3868 | -0.2568 | -0.1705 |
| HP_SLSQP_ALL | -0.3029 | -0.0863 | -0.3892 | -0.2592 | -0.1729 |

High precision vs FW64: `S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; no high-precision version can change this word account. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 0.000e+00 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / -0.0024 (all); ΔΔfull = +0.0000 (HP) / -0.0024 (all); closure errors FW/HP = 0.00e+00/0.00e+00. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = -0.3587, remainder = -0.0863; D_FREQ Δfull = -0.4191, remainder = -0.0863.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

## Case 12 — `p00053_04033` (Train-Train, component 53)

**Molecules.** `A` (train, row 53): `CCNC(=O)c1cc2c(c(cnc2[nH]1)c3cncc(c3)C(=O)O)n4ccc(n4)C(F)(F)F`  
`B` (train, row 4033): `CCCNC(=O)c1cc2c(c(cnc2[nH]1)c3cncc(c3)C(=O)O)n4ccc(n4)C(F)(F)F`  
Catalogue edit: `1` → `2` on core `[1*]CNC(=O)c1cc2c(-n3ccc(C(F)(F)F)n3)c(-c3cncc(C(=O)O)c3)cnc2[nH]1` (31 heavy atoms); class `[1*]C=>[1*]CC`.

**Common context.** `C` = fragment 154 (31 atoms, key `83f613c300e46aa9…`), cut bond `0/0/0`. `S_A` = 0 (`*C`, 1 atoms, Train word=True, prototype=True); `S_B` = 126 (`*CC`, 2 atoms, Train word=True, prototype=False); same τ = True.

**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) `Z=5,chg=5,arom=0,b0=0,b1=0,b2=0`; `s_D(C,·)` is the frozen readout, `contribution_k = α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.

| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |
|---:|---|---|---:|---:|---:|---:|
| 308 | `*Cc1ccc(Cl)cc1` | C | 0.0919 | +0.1881 | 0.0919 | +0.1881 |
| 1104 | `*C#Cc1ccccc1` | C | 0.1037 | +0.1580 | 0.1037 | +0.1580 |
| 31 | `*C(=O)O` | C | 0.0078 | -0.0181 | 0.0078 | -0.0181 |
| 21 | `*CC(O)CO` | C | 0.0096 | -0.0071 | 0.0096 | -0.0071 |
| 0 (D0) | `*C` | C | 0.7586 | +0.0000 | 0.7586 | +0.0000 |

**Numbers.** Δy = +0.2000; Δresp (native RESP full F) = +0.3624; error of native Δresp = +0.1624; native one-view Δlocal = +0.3982; native remainder = -0.0358. Dictionary vs native model: Δfull error = -0.0167 (HP) / -0.0167 (FW); Δlocal error = -0.0732 (HP) / -0.0732 (FW).

| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |
|---|---:|---:|---:|---:|---:|
| FW64 | +0.3250 | +0.0207 | +0.3456 | +0.1456 | +0.1250 |
| HP_CERTIFIED | +0.3250 | +0.0207 | +0.3456 | +0.1456 | +0.1250 |
| HP_SLSQP_ALL | +0.3391 | +0.0101 | +0.3492 | +0.1492 | +0.1391 |

High precision vs FW64: `S_B` is a projected (non-prototype) fragment. max |Δα(S_B)| = 0.000e+00 (HP_CERTIFIED) / 1.138e-01 (HP_SLSQP_ALL); ΔΔlocal = +0.0000 (HP) / +0.0141 (all); ΔΔfull = +0.0000 (HP) / +0.0035 (all); closure errors FW/HP = 3.89e-16/3.89e-16. `HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; `HP_SLSQP_ALL` is the uncertified all-rows sensitivity. Other selectors (HP): D_LATENT Δfull = +0.3445, remainder = +0.0124; D_FREQ Δfull = +0.3624, remainder = -0.0358.

**Scope.** Train–Train: describes the frozen function; not held-out evidence and not a new-structure generalisation claim.

