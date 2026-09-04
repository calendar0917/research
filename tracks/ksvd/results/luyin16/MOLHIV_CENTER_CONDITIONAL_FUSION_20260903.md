# luyin16-molhiv-center-conditional-fusion-v1

MolHIV evaluation of the centre-level conditional-fusion prototype.

- split sizes: {'train': 32901, 'valid': 4113, 'test': 4113}
- representation: topology-only rooted-WL radius 3, 3 WL rounds
- device: cpu; fixed budget 100 epochs
- test policy: best official-valid epoch, then train+valid refit; test was not used for selection

| phase | ROC-AUC | epoch |
|---|---:|---:|
| official train -> valid (best) | 0.814636 | 22 |
| official train+valid -> test | 0.701877 | 22 |

The model uses same-centre [s_v, a_v, s_v*a_v] fusion and sum/mean/std centre readout; relation propagation and attention are disabled.

Runtime: 347.6s.
