# E2E-DictEnv-v0 — decision (STOPPED at Stage-1)

```
E2E_DICTENV_MECHANISM_COLLAPSED
```

* Stage-1 sub-gates: `{'alpha_still_sparse': True, 'atoms_active': False, 'd_grad_nonzero': True, 'effective_atom_count': True, 'environment_rank': True, 'loss_finite': True, 'no_single_atom_dominance': True, 'train_mae_decreased': True}`
* active atoms after smoke: 23/32 (official train), 23/32 (512 subset); required ≥ 24
* active atoms at init: 25/32 (official train), 24/32 (512 subset)
* effective atom count 12.827, top-1 share 0.1250, environment rank 2.297
* correctness Gate 0: True
* official test loaded: False

## Reading

STOP per the frozen verdict table (case S1). The sparse tied-dictionary mechanism
is intact at initialization but the fixed 3-epoch train-only smoke drops it below the
preregistered ≥24/32 coverage threshold. No formal training and no performance or
dictionary-specificity claim is authorized. A new round with a corrected,
statistically-appropriate coverage gate may be preregistered separately; this round
must not be rescued.
