#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/calendar/.conda/envs/gsn-official/bin/python}
R=results/molhiv
F=$R/molhiv_n41127_scaffold_folds3_seed20260726.npz
TOK=$R/node_tokens_n41127_a32_t3_pca64_officialtrainfit.npz
LAT=$R/rawpatch_pca64_latents_n41127_officialtrainfit_traintest.npz
AUDIT=$R/official_test_latent_isolation_audit_v1.json
VOC=$R/scaffold_stable_vocabulary_n41127_official_test_seed0.json
GATE=$R/fixed_vocabulary_relation_gates_n41127_official_test_seed0.json
VALID_VOC=$R/scaffold_stable_vocabulary_n41127_official_valid_seed0.json
VALID_GATE=$R/fixed_vocabulary_relation_gates_n41127_official_valid_seed0.json
SUMMARY=$R/compact_official_test_summary_v1.json
MM=${KSVD_FACILITY_MEMMAP_DIR:-$R/.facility_memmap}
FREEZE=$R/compact_official_test_freeze_v1.json
FREEZE_SHA=$FREEZE.sha256

# The manifest, code and evidence are sealed before official-test rows are encoded.
$PY - "$FREEZE" "$FREEZE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
seal_path = Path(sys.argv[2])
if not manifest_path.is_file() or not seal_path.is_file():
    raise SystemExit("missing frozen manifest or external SHA256 seal")
expected_manifest_sha = seal_path.read_text(encoding="utf-8").split()[0]
actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if actual_manifest_sha != expected_manifest_sha:
    raise SystemExit("frozen manifest SHA256 seal mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "frozen_before_one_shot_official_test":
    raise SystemExit("manifest is not in the pre-test frozen state")
for section in ("code_audit", "frozen_inputs", "promotion_evidence"):
    for item in manifest[section].values():
        if isinstance(item, str):
            continue
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise SystemExit(f"{section} SHA256 mismatch: {path}")
print(f"terminal-test freeze preflight passed: {actual_manifest_sha}", flush=True)
PY

for OUTPUT in "$VOC" "$GATE" "$SUMMARY"; do
  if [[ -e "$OUTPUT" ]]; then
    echo "refusing to repeat official-test evaluation; output exists: $OUTPUT" >&2
    exit 1
  fi
done

$PY code/extract_molhiv_patch_metric_latents.py \
  --token-cache "$TOK" --max-graphs 0 --radius 2 --max-nodes 8 \
  --encode-splits train_test --output "$LAT"

# Independent split audit before any model receives the terminal heldout latents.
$PY - "$LAT" "$AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

latent_path, output_path = map(Path, sys.argv[1:])
with np.load(latent_path, allow_pickle=False) as d:
    offsets = np.asarray(d["offsets"], dtype=np.int64)
    train = np.asarray(d["train_indices"], dtype=np.int64)
    valid = np.asarray(d["valid_indices"], dtype=np.int64)
    test = np.asarray(d["test_indices"], dtype=np.int64)
    fit = np.asarray(d["fit_indices"], dtype=np.int64)
    latents = np.asarray(d["latents"], dtype=np.float32)

def rows(indices):
    return np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in indices
    ])

def sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

train_rows, valid_rows, test_rows = rows(train), rows(valid), rows(test)
if not np.array_equal(fit, train):
    raise SystemExit("fit indices are not exact official train")
if np.count_nonzero(latents[valid_rows]) != 0:
    raise SystemExit("official-valid latent rows are nonzero")
if np.mean(np.linalg.norm(latents[test_rows], axis=1) > 0.99) < 0.999:
    raise SystemExit("official-test latent rows are missing or unnormalized")
report = {
    "protocol_id": "molhiv_official_test_latent_isolation_audit_v1",
    "date": "2026-07-28",
    "fit_indices_equal_official_train": True,
    "fit_indices_sha256": sha(fit),
    "n_train_graphs": int(len(train)),
    "n_valid_graphs": int(len(valid)),
    "n_test_graphs": int(len(test)),
    "n_train_nodes": int(len(train_rows)),
    "n_valid_nodes": int(len(valid_rows)),
    "n_test_nodes": int(len(test_rows)),
    "official_valid_nonzero_count": int(np.count_nonzero(latents[valid_rows])),
    "official_test_nonzero_count": int(np.count_nonzero(latents[test_rows])),
    "official_test_normalized_fraction": float(np.mean(np.linalg.norm(latents[test_rows], axis=1) > 0.99)),
    "latent_cache": str(latent_path),
    "latent_cache_sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
}
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
PY

$PY code/run_molhiv_scaffold_stable_vocabulary.py \
  --latent-cache "$LAT" --fold-cache "$F" --fold 0 --evaluation-split official_test \
  --controls farthest,scaffold_facility --max-graphs 0 --seed 0 \
  --prototype-seed 20260728 --facility-seed 20260728 --facility-targets-per-graph 3 \
  --facility-memmap-dir "$MM" --n-prototypes 32 --epochs 30 --batch-size 128 \
  --hidden 64 --dropout 0.15 --sparsity 3 --temperature 0.25 --task-lr 0.001 \
  --weight-decay 0.0001 --save-predictions --output "$VOC"

$PY code/run_molhiv_fixed_vocabulary_relation_gates.py \
  --latent-cache "$LAT" --fold-cache "$F" --vocabulary-result "$VOC" --broad-result "$VOC" \
  --fold 0 --evaluation-split official_test --gate-types uniform --max-graphs 0 --seed 0 \
  --selector-seed 20260728 --relation-seed 20260728 --inner-splits 3 \
  --selector-top-nodes 3 --selector-c 0.1 --n-prototypes 32 --sparsity 3 \
  --relation-epochs 30 --relation-batch-size 256 --relation-lr 0.001 \
  --relation-weight-decay 1.0 --max-relation-residual 0.3125 \
  --standardization-clip 5.0 --save-predictions --output "$GATE"

$PY code/summarize_molhiv_compact_official_test.py \
  --vocabulary-result "$VOC" --gate-result "$GATE" --valid-vocabulary-result "$VALID_VOC" \
  --valid-gate-result "$VALID_GATE" \
  --output "$SUMMARY"
