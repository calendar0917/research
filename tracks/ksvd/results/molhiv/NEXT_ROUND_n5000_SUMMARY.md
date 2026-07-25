# molhiv next-round (B/C/D/A)

n≈5000 · seeds=[0, 1, 2]

Size baseline valid **0.6564**

## B dict grid (s+size valid, seed0)

| name | s_only val | s+size val | Δ vs size |
|------|------------|------------|-----------|
| A6T2 | 0.5029 | 0.5932 | -0.0632 |
| A8T2 | 0.3995 | 0.5491 | -0.1073 |
| A8T3 | 0.5090 | 0.6145 | -0.0419 |
| A12T2 | 0.4671 | 0.5932 | -0.0632 |
| A12T3 | 0.4929 | 0.5622 | -0.0942 |
| A16T3 | 0.4340 | 0.5845 | -0.0719 |

## B multi-seed mean Δ(s+size − size) valid

- **best**: mean Δ=-0.0057 ± 0.0429
- **A8T2**: mean Δ=-0.0271 ± 0.0628

## C chem vs topo

- **topo**: s+size val=0.5491 Δ=-0.1073
- **chem**: s+size val=0.6536 Δ=-0.0028

## D ring_boost

- **boost=0.0**: s+size val=0.5491 Δ=-0.1073
- **boost=1.0**: s+size val=0.6307 Δ=-0.0257
- **boost=2.0**: s+size val=0.5576 Δ=-0.0988
- **boost=4.0**: s+size val=0.5484 Δ=-0.1080

## A node gate dual (smoke)

- **gine_only**: val=0.8767 test@best=0.7807
- **graph_concat**: val=0.9041 test@best=0.6848
- **node_gate**: val=0.8720 test@best=0.6744

Elapsed: 174.97s
