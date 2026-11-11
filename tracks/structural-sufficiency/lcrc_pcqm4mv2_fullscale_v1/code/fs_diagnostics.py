"""Read-only grouped diagnostics on the two full-Train best checkpoints.

For each selected best checkpoint the full official Validation split is scored
once and stratified by: exact-SMILES presence in the official Train split
(which is exactly the full cache row set), within-Validation duplicate strings,
presence of an UNK atom class, heavy-atom count and true-gap bin.

Descriptive only: the official metric is never recomputed after dropping rows,
and nothing here selects or changes a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fs_common as C  # noqa: E402
import fs_data as D  # noqa: E402
import ew_diagnostics as ED  # noqa: E402  (read-only stratification helpers)

import torch  # noqa: E402

OUT = C.ANALYSIS / "fs_diagnostics.json"
MD = C.ANALYSIS / "fs_diagnostics.md"


def run(device: torch.device, *, batch_size: int = 512) -> dict[str, Any]:
    task = D.load_task("PCQM_W2_FULL", verify_hashes=False, verbose=False)
    # Diagnostics need SMILES; the training loader deliberately skips them.
    task["valid_cache"] = D.load_cache(C.PARENT_CACHE / "pcqm_valid", with_smiles=True)
    full = D.load_cache(C.CACHE / D.FULL_CACHE_NAME, with_smiles=True)
    train_smiles = set(str(s) for s in full["smiles"])
    subset_smiles = set(str(s) for s in full["smiles"][:1_000_000])
    C.log(
        f"[diagnostics] unique Train SMILES {len(train_smiles):,}; "
        f"unique 1M-subset SMILES {len(subset_smiles):,}"
    )
    results: dict[str, Any] = {}
    for key, run_id in (("W2", "PCQM_W2_FULL"), ("W3", "PCQM_W3_FULL")):
        payload = C.read_json(C.RUNS / run_id / "run.json")
        checkpoint = Path(C.REPO / payload["best_checkpoint"])
        results[key] = ED.run_one(
            key,
            checkpoint,
            task,
            device,
            train_smiles=train_smiles,
            subset_smiles=subset_smiles,
            batch_size=int(batch_size),
        )
        C.log(
            f"[diagnostics:{key}] mae={results[key]['mae_eV']:.9f} eV "
            f"dups={results[key]['official_train_smiles']['n_valid_duplicate']:,} "
            f"unk={results[key]['by_unk']['n_unk_molecules']:,}"
        )
    out = {
        "schema": "lcrc-pcqm4mv2-fullscale-diagnostics-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "official_train_smiles": {
            "n_unique": len(train_smiles),
            "n_cache_rows": int(full["n_molecules"]),
            "source": "full Train cache SMILES (one per official Train row)",
        },
        "one_million_subset_smiles": {"n_unique": len(subset_smiles)},
        "results": results,
        "policy": "descriptive only; official Validation MAE is never recomputed after dropping rows",
    }
    C.write_json(OUT, C.json_safe(out))
    ED.to_markdown(results, MD)
    C.log(f"[diagnostics] wrote {C.track_rel(OUT)} and {C.track_rel(MD)}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    run(torch.device(args.device), batch_size=int(args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
