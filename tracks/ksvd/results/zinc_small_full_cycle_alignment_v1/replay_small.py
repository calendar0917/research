"""Read-only replay of the published Small (LATENT-BRIDGE-seed0) soup.

No fitting, no scaler refit, no backbone update.  Produces raw eval-mode
train/valid predictions for the frozen Small checkpoint so the Small/Full
calibration comparison uses the same measurement pipeline.
Official test never instantiated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CKPT = REPO / "tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/checkpoints/LATENT-BRIDGE-seed0_soup_state.pt"
SEED = 0
THREADS = 8
BATCH = 128


def build():
    sub = base._load_parent_subspace()
    dic = base._dictionary_tensor()
    m = lb.build_latent_bridge_model(dic, SEED, sub)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.float() for k, v in st.items()})
    m.eval()
    return m


@torch.no_grad()
def extract(model, data):
    loader = p1run.p1.make_env_loader(data, BATCH, False, SEED)
    P, Y = [], []
    for batch in loader:
        pred = model(batch, mask=cm.C6_MASK)
        if isinstance(pred, tuple):
            pred = pred[0]
        P.append(pred.view(-1).cpu().numpy())
        Y.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(P).astype(np.float64), np.concatenate(Y).astype(np.float64)


def main():
    torch.set_num_threads(THREADS)
    m = build()
    n_params = int(sum(p.numel() for p in m.parameters()))
    tr = p1run.load_split("train")
    va = p1run.load_split("valid")
    ptr, ytr = extract(m, tr)
    pva, yva = extract(m, va)
    # identity cross-check against the published handoff labels (Full cache labels)
    ht = np.load(REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff/train_reader.npz")
    hv = np.load(REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff/valid_reader.npz")
    info = {
        "checkpoint": str(CKPT.relative_to(REPO)),
        "checkpoint_sha256": "b14c92af9d62ae262eb10ab0e00646c74c96b5c44872ffd8b754fda7df1c8e54",
        "params": n_params,
        "raw_train_mae": float(np.mean(np.abs(ptr - ytr))),
        "raw_valid_mae": float(np.mean(np.abs(pva - yva))),
        "y_train_max_abs_diff_vs_handoff": float(np.max(np.abs(ytr - ht["y"]))),
        "y_valid_max_abs_diff_vs_handoff": float(np.max(np.abs(yva - hv["y"]))),
        "official_test_loaded": False,
    }
    np.savez_compressed(OUT / "small_replay.npz", ptr=ptr, pva=pva, ytr=ytr, yva=yva)
    (OUT / "small_replay.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()