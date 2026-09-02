#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/calendar/.conda/envs/gsn-official/bin/python}
R=results/molhiv
F=$R/molhiv_n41127_scaffold_folds3_seed20260726.npz
TOK=$R/node_tokens_n41127_a32_t3_pca64_officialtrainfit.npz
LAT=$R/rawpatch_pca64_latents_n41127_officialtrainfit_trainvalid.npz
VOC=$R/scaffold_stable_vocabulary_n41127_official_valid_seed0.json
GATE=$R/fixed_vocabulary_relation_gates_n41127_official_valid_seed0.json
MM=${KSVD_FACILITY_MEMMAP_DIR:-$R/.facility_memmap}
FREEZE=$R/label_free_pair_pca_official_valid_freeze_v1.json
FREEZE_SHA=$FREEZE.sha256

# Refuse to run unless the manifest was sealed before any official-valid
# encoding/evaluation and every audited code file still matches it.
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
if manifest.get("status") != "frozen_before_one_shot_official_valid":
    raise SystemExit("manifest is not in the pre-valid frozen state")
for name, expected in manifest["code_audit"].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"code audit mismatch: {name}")
print(f"freeze preflight passed: {actual_manifest_sha}", flush=True)
PY

for OUTPUT in "$VOC" "$GATE" \
  "$R/label_free_pair_pca_n41127_official_valid_seed0.json" \
  "$R/label_free_pair_pca_n41127_official_valid_seed1.json" \
  "$R/label_free_pair_pca_n41127_official_valid_seed2.json"; do
  if [[ -e "$OUTPUT" ]]; then
    echo "refusing to repeat official-valid evaluation; output exists: $OUTPUT" >&2
    exit 1
  fi
done

$PY code/extract_molhiv_patch_metric_latents.py --token-cache "$TOK" --max-graphs 0 --radius 2 --max-nodes 8 --encode-splits train_valid --output "$LAT"
$PY code/run_molhiv_scaffold_stable_vocabulary.py --latent-cache "$LAT" --fold-cache "$F" --fold 0 --evaluation-split official_valid --controls farthest,scaffold_facility --max-graphs 0 --seed 0 --prototype-seed 20260728 --facility-seed 20260728 --facility-targets-per-graph 3 --facility-memmap-dir "$MM" --n-prototypes 32 --epochs 30 --batch-size 128 --hidden 64 --dropout 0.15 --sparsity 3 --temperature 0.25 --task-lr 0.001 --weight-decay 0.0001 --save-predictions --output "$VOC"
$PY code/run_molhiv_fixed_vocabulary_relation_gates.py --latent-cache "$LAT" --fold-cache "$F" --vocabulary-result "$VOC" --broad-result "$VOC" --fold 0 --evaluation-split official_valid --gate-types uniform --max-graphs 0 --seed 0 --selector-seed 20260728 --relation-seed 20260728 --inner-splits 3 --selector-top-nodes 3 --selector-c 0.1 --n-prototypes 32 --sparsity 3 --relation-epochs 30 --relation-batch-size 256 --relation-lr 0.001 --relation-weight-decay 1.0 --max-relation-residual 0.3125 --standardization-clip 5.0 --save-predictions --output "$GATE"
for SEED in 0 1 2; do
  $PY code/run_molhiv_label_free_pair_pca.py --latent-cache "$LAT" --fold-cache "$F" --vocabulary-result "$VOC" --compact-base-result "$GATE" --fold 0 --evaluation-split official_valid --max-graphs 0 --seed "$SEED" --relation-seed 20260728 --n-prototypes 32 --sparsity 3 --pca-rank 144 --pca-types covariance_pca --relation-epochs 30 --relation-batch-size 256 --relation-lr 0.001 --relation-weight-decay 1.0 --max-relation-residual 0.25 --standardization-clip 5.0 --save-predictions --output "$R/label_free_pair_pca_n41127_official_valid_seed${SEED}.json"
done
