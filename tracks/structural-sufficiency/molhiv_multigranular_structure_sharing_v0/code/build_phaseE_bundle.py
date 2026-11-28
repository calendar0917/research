"""Bundle the frozen Phase-E inputs into one small committed artefact.

The GPU hosts receive only the committed revision, so the label-free inputs a
Phase E arm needs are packed into ``bundle/phaseE_inputs.npz`` (~9 MB):

* ``C``/``F``  frozen structure design matrices (float32 CSR, frozen scaling)
* ``morgan``   packed binary Morgan bits (radius 2, 2048)
* ``n_atoms`` / ``n_bonds``
* ``support_weight`` (a_f, for G-STRUCT-W only)

Nothing here carries a label or a Valid/Test statistic.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import msds_common as C
import msds_data as D
import msds_features as FEAT
import run_phaseB_linear as LIN


def main() -> None:
    molecules = D.load_molecules()
    design = LIN.load_design()
    bits, morgan_meta = FEAT.build_morgan(molecules)
    packed = np.packbits(bits, axis=1)
    del bits
    # the two Train-standardised count columns, computed exactly as Phase B's M block
    n_atoms_f = molecules.n_atoms.astype(np.float64)
    n_bonds_f = (molecules.bond_offsets[1:] - molecules.bond_offsets[:-1]).astype(np.float64)
    train_rows = np.asarray(molecules.split["train"], dtype=np.int64)
    a = np.log1p(n_atoms_f)
    b = np.log1p(n_bonds_f)
    m_extra = np.stack([
        (a - a[train_rows].mean()) / a[train_rows].std(),
        (b - b[train_rows].mean()) / b[train_rows].std(),
    ], axis=1).astype(np.float32)
    n_bonds = (molecules.bond_offsets[1:] - molecules.bond_offsets[:-1]).astype(np.int32)
    out_dir = C.ROUND / "bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phaseE_inputs.npz"
    np.savez_compressed(
        path,
        C_indptr=design["C"].indptr, C_indices=design["C"].indices, C_data=design["C"].data.astype(np.float32),
        F_indptr=design["F"].indptr, F_indices=design["F"].indices, F_data=design["F"].data.astype(np.float32),
        n_rows=np.int64(design["C"].shape[0]), n_coarse=np.int64(design["C"].shape[1]),
        n_full=np.int64(design["F"].shape[1]),
        morgan_packed=packed, morgan_n_bits=np.int64(2048), m_extra=m_extra,
        n_atoms=molecules.n_atoms.astype(np.int32), n_bonds=n_bonds,
        support_weight=design["support_weight"],
        label=__import__("numpy").array([], dtype=np.int8),  # labels deliberately absent
    )
    payload = {
        "round_id": C.ROUND_ID,
        "path": C.rel(path),
        "sha256": C.sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "contents": {
            "C": list(design["C"].shape), "F": list(design["F"].shape),
            "morgan": "packed binary Morgan r2 2048",
            "m_extra": "Train-standardised log1p(n_atoms), log1p(n_bonds)",
            "n_atoms": "physical atom count",
            "n_bonds": "physical bond count (raw edge records, no double counting)",
            "support_weight": "a_f for G-STRUCT-W",
            "labels": "absent by construction",
        },
        "design_cache_sha256": design["sha256"],
        "vocab_sha256": design["vocab_sha256"],
        "morgan_meta": morgan_meta,
        "git": C.git_state(),
    }
    C.write_json(C.ANALYSIS / "phaseE_bundle.json", payload)
    print(f"wrote {C.rel(path)} ({payload['size_bytes'] / 1e6:.1f} MB) sha256={payload['sha256']}", flush=True)


if __name__ == "__main__":
    main()
