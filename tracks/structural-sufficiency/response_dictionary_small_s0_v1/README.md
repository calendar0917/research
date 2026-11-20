# response_dictionary_small_s0_v1

Small official-split screen (seed 0): real conditional fragment–environment
responses (`RESP`) versus an exact matched additive control (`ADD`), followed by
a frozen-RESP dictionary compression stage. The frozen design is in
[`PROTOCOL.md`](PROTOCOL.md); results are in [`REPORT.md`](REPORT.md).

## Layout

```
code/rd_common.py      paths, hashes, frozen OGB cache identity, scaler
code/rd_fragment.py    bridge cuts, exact coloured-graph identity, fragment cache
code/rd_prepare.py     build the two fragment caches (CPU, content-addressed)
code/rd_model.py       shared encoder, RESP/ADD fusion, canonical init state
code/rd_train.py       frozen recipe, per-epoch full Train/Val/Test evaluation
code/rd_run.py         one (task, arm) training unit
code/rd_dictionary.py  Train-only words/panels, three selections, Frank–Wolfe
code/rd_dict_run.py    frozen-RESP dictionary stage over one best checkpoint
code/rd_checks.py      targeted CPU checks + 3-update-step smoke
code/rd_analyze.py     gains, paired bootstrap, w=0, labels, curves, report facts
configs/rd_small_s0.json  frozen configuration
data/                  fragment caches + one canonical init state (SHA-256 pinned)
results/<task>_<arm>/  curves, checkpoints, predictions, status, run log
```

## Commands

```bash
# reproducibility / correctness gate (CPU)
uv run python code/rd_checks.py --smoke

# (re)build the content-addressed fragment caches, CPU only
uv run python code/rd_prepare.py

# one formal unit (on res via rr, device cuda:0, pool maps the physical GPU)
uv run python code/rd_run.py --task lipo --arm RESP --device cuda:0

# dictionary stage after the RESP unit (CPU linear algebra + one frozen pass)
uv run python code/rd_dict_run.py --task lipo \
  --checkpoint results/lipo_RESP/best.pt \
  --predictions results/lipo_RESP/predictions_best.npz \
  --out results/lipo_RESP/dictionary

# aggregate all units and the dictionary stage
uv run python code/rd_analyze.py --results results --out .
```

`RD_RESULTS_DIR` overrides the result root (used for smoke runs). Test data is
read under `test_policy: always` and reported as `DEVELOPMENT_TEST`; it never
selects a checkpoint, dictionary or recipe.
