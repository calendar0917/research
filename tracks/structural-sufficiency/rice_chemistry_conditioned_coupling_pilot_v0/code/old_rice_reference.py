"""Read-only reference forward of the historical RICE-v0 code.

This helper is executed in a **separate process** so the new round can import
its own ``rice_*`` modules without a name collision.  It only reads the
historical round's source, data layer and frozen checkpoint; it never writes
there.

Modes
  initial  build the historical model with seed 0 and run its first forward on
           the first ``--n`` D0 Fit rows
  replay   load a historical best checkpoint and run it on D0 Dev rows

Outputs a small ``.npz`` with standardized predictions plus the unary / pair /
Phi / kappa tensors needed for a bitwise comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

OLD_CODE = (Path(__file__).resolve().parents[2]
            / "rooted_interaction_cumulant_encoder_pilot_v0" / "code")
if str(OLD_CODE) not in sys.path:
    sys.path.insert(0, str(OLD_CODE))

import rice_common as C          # noqa: E402
import rice_data as D            # noqa: E402
import rice_model as M           # noqa: E402
import rice_structure as S       # noqa: E402
import rice_tensor as T          # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "replay"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = C.load_config()
    bundle = D.load_bundle()
    if args.mode == "initial":
        rows = np.asarray(bundle.fit_rows[:args.n], np.int64)
        model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                    D.normalisers(bundle), config=cfg)
        init_hashes = info["init_hashes"]
        ledger = info["parameters"]
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for replay")
        state = torch.load(args.checkpoint, map_location=args.device,
                           weights_only=False)
        rows = np.asarray(bundle.eval_rows, np.int64)
        model = M.load_model_from_state(state["state"], bundle.atom_dims,
                                        bundle.bond_dims, D.normalisers(bundle),
                                        config=cfg, device=args.device)
        init_hashes = state.get("init_hashes", {})
        ledger = state.get("ledger", {})
    model.eval()
    structs = S.build_all(bundle, rows)
    parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, rows.size, 256):
            part = rows[start:start + 256]
            sub = {int(r): structs[int(r)] for r in part}
            batch = T.collate(part.tolist(), sub, bundle.raw, args.device)
            out = model(batch, arm="RICE_CUMULANT", return_aux=True)
            parts.setdefault("yhat_std", []).append(out["yhat_std"].cpu().numpy())
            parts.setdefault("u", []).append(out["u"].cpu().numpy())
            parts.setdefault("j", []).append(out["j"].cpu().numpy())
            for size in (2, 3, 4):
                parts.setdefault(f"phi_{size}", []).append(
                    out["phi"][size].cpu().numpy())
                parts.setdefault(f"kappa_{size}", []).append(
                    out["kappa"][size].cpu().numpy())
    payload = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    payload["rows"] = rows
    payload["pred_raw"] = (payload["yhat_std"] * float(bundle.scaler["std"])
                           + float(bundle.scaler["mean"]))
    payload["labels"] = bundle.labels[rows]
    np.savez(args.out, **payload)
    import json

    Path(str(args.out) + ".json").write_text(json.dumps({
        "mode": args.mode, "rows": int(rows.size),
        "init_hashes": init_hashes, "ledger": ledger,
        "scaler": bundle.scaler, "data_hashes": bundle.hashes,
        "git_commit": C.git_commit(),
    }, indent=1, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()