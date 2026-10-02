# GPU Migration Rules

Use this reference when an experiment path is still hard-coded to CPU.

## Goal

Add explicit device handling without changing historical CPU behavior by default.

## Rules

1. Prefer a `device` argument/config such as `cpu`, `cuda`, or `auto`; preserve the existing default unless the experiment explicitly opts into CUDA.
2. Move the model and each batch to the resolved device.
3. Create temporary tensors on an existing tensor's device (`x.new_zeros(...)`, `torch.zeros(..., device=x.device)`) rather than relying on a global default device.
4. Seed CUDA when CUDA is used:

```python
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

5. Record `device`, GPU name/index, `torch.__version__`, and `torch.version.cuda` in experiment metadata.
6. Do not enable DDP initially. For this repository, prefer one independent seed/experiment per A100.
7. Treat CUDA as a new execution regime. Re-establish a matched GPU baseline before interpreting small deltas against historical CPU results.
8. Keep official-test freeze/unlock rules unchanged.
9. Test at least one short forward/backward CUDA smoke path before buying a full run.
10. If deterministic algorithms materially change quality, use them only for mechanism audits unless the protocol explicitly freezes deterministic mode as the new training regime.

## Common migration failure

Code that works on CPU often creates new CPU tensors inside `forward`. Fix the allocation site rather than calling `torch.set_default_device("cuda")` globally.
