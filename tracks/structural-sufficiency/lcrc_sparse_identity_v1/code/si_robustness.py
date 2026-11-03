"""Post-hoc, clearly-labelled robustness diagnostics for the frozen decision.

The pre-registered analysis (``si_analysis.py``) is complete and unchanged.  This
script adds two things it did not cover, and neither enters the verdict:

1. **Scaffold-grouping diagnostic.**  On the official OGB scaffold split, the
   atom-typed Bemis-Murcko scaffold of every Validation and every Test molecule is
   unique, so the pre-registered "resample scaffold groups" bootstrap degenerates
   to a molecule-level bootstrap.  This is a property of the split, so it is
   measured and reported rather than repaired.
2. **Post-hoc descriptive bootstrap on generic frameworks** (atom types erased,
   ``MurckoScaffold.MakeScaffoldGeneric``).  Generic frameworks *do* repeat, so this
   interval additionally reflects framework-level clustering.  It is explicitly
   NOT the pre-registered analysis, it used a different grouping unit, and it must
   not be used to re-decide anything.

Nothing here re-scores a model, re-selects a checkpoint or touches the labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

import si_analysis as AN
import si_common as C


def generic_frameworks(smiles: Sequence[str]) -> list[str]:
    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    n_parse = 0
    n_generic_fail = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_parse += 1
            out.append(f"__PARSE_FAILED__{n_parse}")
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=MurckoScaffold.MakeScaffoldGeneric(mol)))
        except Exception:  # noqa: BLE001
            n_generic_fail += 1
            out.append(f"__GENERIC_FAILED__{n_generic_fail}")
    return out


def codes_from(groups: Sequence[str]) -> np.ndarray:
    uniq = sorted(set(groups))
    index = {g: i for i, g in enumerate(uniq)}
    return np.asarray([index[g] for g in groups], dtype=np.int64)


def grouping_summary(groups: Sequence[str], label: str) -> dict[str, Any]:
    counts = Counter(groups)
    sizes = np.asarray(sorted(counts.values()), dtype=np.int64)
    return {
        "unit": label,
        "n_molecules": len(groups),
        "n_groups": int(len(counts)),
        "max_group_size": int(sizes.max()),
        "mean_group_size": float(sizes.mean()),
        "n_singleton_groups": int((sizes == 1).sum()),
        "fraction_of_molecules_in_singleton_groups": float((sizes == 1).sum() / len(groups)),
        "top_groups": counts.most_common(5),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    import ptb_data as PD

    view = PD.molecule_view(C.DATASET)

    split_sources = {
        "valid": (
            view.valid_cache["smiles"],
            np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1),
        ),
        "test": (
            torch.load(
                C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/cache/ogbg-molhiv_test.pt",
                map_location="cpu",
                weights_only=False,
            )["smiles"],
            np.load(C.PREDICTIONS / f"{C.MODEL_A0}_s0_test.npz")["y_true"].reshape(-1),
        ),
    }
    out: dict[str, Any] = {
        "schema": "lcrc-sparse-identity-robustness-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "status": "POST_HOC_DESCRIPTIVE__NOT_PRE_REGISTERED__DOES_NOT_ENTER_THE_VERDICT",
        "pre_registered_analysis": "analysis/paired_bootstrap.json (atom-typed Bemis-Murcko scaffold groups)",
        "reason_for_this_file": "the pre-registered grouping unit is degenerate on this split (every held-out "
        "molecule has a unique atom-typed Bemis-Murcko scaffold), so a descriptive second grouping is reported "
        "alongside it for context only",
        "splits": {},
    }
    for split, (smiles, y) in split_sources.items():
        typed_groups = AN.scaffolds_for(smiles, key=split)["groups"]
        gen_groups = generic_frameworks(smiles)
        typed_codes = codes_from(typed_groups)
        gen_codes = codes_from(gen_groups)
        preds = {
            arm: {
                seed: np.load(C.PREDICTIONS / f"{arm}_s{seed}_{split}.npz")["pred"].reshape(-1)
                for seed in C.SEEDS
            }
            for arm in (C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD)
        }
        typed_boot = AN.paired_bootstrap(
            preds, y, typed_codes, seed=C.BOOTSTRAP_SEED, replicates=C.BOOTSTRAP_REPLICATES
        )
        gen_boot = AN.paired_bootstrap(
            preds, y, gen_codes, seed=C.BOOTSTRAP_SEED, replicates=C.BOOTSTRAP_REPLICATES
        )
        out["splits"][split] = {
            "pre_registered_scaffold_groups": grouping_summary(typed_groups, "atom-typed Bemis-Murcko scaffold"),
            "post_hoc_generic_frameworks": grouping_summary(gen_groups, "generic Bemis-Murcko framework"),
            "pre_registered_bootstrap_recomputed": {
                "n_groups": typed_boot["n_scaffold_groups"],
                "contrasts": typed_boot["contrasts"],
            },
            "post_hoc_generic_framework_bootstrap": {
                "n_groups": gen_boot["n_scaffold_groups"],
                "replicates_used": gen_boot["replicates_used"],
                "replicates_skipped_single_class": gen_boot["replicates_skipped_single_class"],
                "contrasts": gen_boot["contrasts"],
            },
        }
    out["interpretation"] = {
        "pre_registered": "atom-typed scaffold grouping has one molecule per group on both held-out splits, so its "
        "interval is a molecule-level (i.i.d.) interval and understates any scaffold-level clustering",
        "post_hoc": "generic frameworks repeat, so this interval additionally reflects framework-level clustering; "
        "it is reported for context and is not part of the pre-registered decision",
        "consequence_for_the_verdict": "none: the verdict is NO_PERFORMANCE_SIGNAL on the pre-registered "
        "three-seed mean ROC-AUC difference, which does not depend on either interval",
    }
    C.write_json(C.ANALYSIS / "paired_bootstrap_extra.json", out)
    for split, entry in out["splits"].items():
        pre = entry["pre_registered_scaffold_groups"]
        gen = entry["post_hoc_generic_frameworks"]
        C.log(
            f"{split}: typed groups {pre['n_molecules']}/{pre['n_groups']} unique "
            f"(max {pre['max_group_size']}); generic frameworks {gen['n_groups']} groups "
            f"(max {gen['max_group_size']})"
        )
        for name, v in entry["post_hoc_generic_framework_bootstrap"]["contrasts"].items():
            C.log(f"  post-hoc generic {name}: {v['point']:+.4f} ci95 {v['ci95_percentile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
