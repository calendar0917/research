# ZINC dictionary scaling handoff

This package proposes one unified Small and Full task-dictionary family for
`calendar0917/research`. It contains a Chinese rationale and an executable-task
prompt, plus a **NumPy-only synthetic reference** and its numerical audit.
No ZINC training result is produced by this package.

Read `zinc_dictionary_scale_plan.md`, then hand
`zinc_dictionary_scale_agent_prompt.md` to the implementing Agent.

Run the numerical reference with Python and NumPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_dictionary_scaling.py
```

`dictionary_scale_audit.json` records parameter accounting, initialization,
ISTA stability, function containment, invariance and task-signal checks.
The reference starts after the fixed semantic/slot stem and after the graph
auxiliary encoders. It is neither a complete PyTorch model nor a chemistry
featurizer. The implementing Agent must complete the real-data gates.

The block-replication initializer is an eval-mode containment witness only.
Formal Full training must use fresh width-specific initialization.

`source_manifest.json` lists source files audited at the pinned repository
revision. Repository source code and historical checkpoints are not included.
