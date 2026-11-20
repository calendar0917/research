"""What the 136-word dictionary actually stores and what still needs the encoder.

This is a descriptive accounting of the frozen inference path, not a speed
benchmark and not a claim that the whole network became smaller.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build(ctx: Mapping[str, Any], proj: Mapping[str, Any]) -> dict[str, Any]:
    frozen, inventory, dicts = ctx["frozen"], ctx["inventory"], ctx["dicts16"]
    config = ctx["config"]
    names = ("D_RESPONSE", "D_LATENT", "D_FREQ")
    n_fragments = int(frozen.z.shape[0])
    n_word_fragments = int(len(inventory.word_fragments))
    n_candidates = int(len(inventory.word_mol_freq))
    n_env = int(len(inventory.env_mol_freq))
    latent = int(frozen.q.shape[1])
    n_words = int(sum(len(v) for v in dicts["D_RESPONSE"]["prototypes"].values()))
    panel = int(ctx["recipe"]["dictionary"]["panel_size"])
    n_classes = int(len(inventory.class_words))

    prototype_response = int(sum(len(v) for v in dicts["D_RESPONSE"]["prototypes"].values())
                               * panel)
    optional_alpha = 0
    for tau, _ in inventory.class_words.items():
        k = len(dicts["D_RESPONSE"]["prototypes"][tau])
        n_local = sum(1 for f in inventory.word_fragments if inventory.tau.get(int(f)) == tau)
        optional_alpha += n_local * k

    word_set = dicts["D_RESPONSE"]["word_set"]
    env_train = set(inventory.env_mol_freq)
    view_stats = {}
    for split in ("train", "valid", "test"):
        payload = frozen.per_split[split]
        s = np.asarray(payload["view_s_frag"], dtype=np.int64)
        c = np.asarray(payload["view_c_frag"], dtype=np.int64)
        stored_s = np.asarray([bool(word_set.get(int(f), False)) for f in s.tolist()], dtype=bool)
        stored_c = np.asarray([int(f) in env_train for f in c.tolist()], dtype=bool)
        fallback_flags = np.asarray(payload["fallback_frag"], dtype=np.int64) >= 0
        view_stats[split] = {
            "n_views": int(s.size),
            "n_molecules": int(fallback_flags.size),
            "n_fallback_molecules": int(fallback_flags.sum()),
            "fallback_molecule_fraction": float(fallback_flags.mean()),
            "stored_S_views": int(stored_s.sum()),
            "stored_S_distinct": int(len({int(f) for f in s.tolist() if bool(word_set.get(int(f), False))})),
            "unseen_S_views": int((~stored_s).sum()),
            "unseen_S_distinct": int(len({int(f) for f in s.tolist() if not bool(word_set.get(int(f), False))})),
            "stored_S_fraction": float(stored_s.mean()) if s.size else 0.0,
            "stored_C_views": int(stored_c.sum()),
            "stored_C_fraction": float(stored_c.mean()) if c.size else 0.0,
            "stored_both_fraction": float((stored_s & stored_c).mean()) if s.size else 0.0,
        }
    model = ctx["model"]
    total_params = int(sum(p.numel() for p in model.parameters()))
    encoder_params = int(sum(p.numel() for p in model.encoder.parameters()))
    return {
        "n_model_parameters": total_params,
        "n_encoder_parameters": encoder_params,
        "n_non_encoder_parameters": total_params - encoder_params,
        "n_fragments_cached": n_fragments,
        "n_word_fragments_observed_all_splits": n_word_fragments,
        "n_train_word_candidates": n_candidates,
        "n_env_fragments_observed_train": n_env,
        "n_stored_words": n_words,
        "n_classes": n_classes,
        "latent": latent,
        "panel_size": panel,
        "storage_floats_if_materialised": {
            "word_a_q_table": int(n_words * (latent + 1)),
            "prototype_response_matrix_P": prototype_response,
            "panel_environment_ids": int(n_classes * panel),
            "train_C_b_k_table": int(n_env * (latent + 1)),
            "optional_alpha_cache_all_observed_S": int(optional_alpha),
        },
        "storage_note": ("The 136 stored words 702 -> 136 do not remove the encoder. The encoder "
                         "(and the a/q heads) is still required for a new S to compute alpha; "
                         "k(C)/b(C) are still required for the C side. The tables above are "
                         "materialised only if an application chooses to cache them; the model "
                         "weights remain 198k parameters."),
        "inference_path": {
            "molecule_to_views": "bridge cuts + fixed <=4 view selection (structure processing)",
            "stored_S": "a_words(S), q_words(S) from the 136-word table (one-hot alpha)",
            "unseen_S": ("encoder z_S -> a(S), q(S) -> response on the 16-env Train panel -> "
                         "simplex projection to alpha over the stored prototypes; no GNN re-run "
                         "per environment"),
            "stored_C": "b(C), k(C) from a Train-environment table if cached",
            "unseen_C": "encoder z_C -> b(C), k(C)",
            "per_view_score": "b(C) + a_words(S) + <q_words(S), k(C), w>/sqrt(latent)",
            "molecule_F": "mean over the fixed views; fallback head for molecules without a cut",
            "alpha_storage": "for a stored prototype word alpha is one-hot; for an unseen S it is computed",
        },
        "view_stats": view_stats,
        "view_stats_note": ("`stored_S` means the S fragment is one of the 136 stored words; "
                            "`stored_C` means the C fragment is one of the 9191 Train environment "
                            "fragments. A pure lookup view needs both. The unseen-S fraction is "
                            "not a generalisation claim: these fragments exist in Val/Test but "
                            "were not Train words."),
    }
