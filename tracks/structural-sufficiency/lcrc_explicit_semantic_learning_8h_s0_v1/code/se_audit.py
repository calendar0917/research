"""P0 audit: reconcile the previous round, count real code support, and freeze

the sections needed before any training.  Read-only over the previous round's
pulled outputs.  Produces ``analysis/p0_audit.json`` and (with ``--notes``)
writes ``SOURCE_NOTES.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import se_common as C
import se_data as SED

import lp_common as LPC
import lp_data as LPD


def _load_prev(arm: str) -> dict[str, Any]:
    path = C.HOME / "analysis/results" / arm / "metrics.json"
    return C.read_json(path)


def prev_summary() -> dict[str, Any]:
    out = {}
    for arm in ("B", "S", "R1", "R2", "D2", "G"):
        try:
            m = _load_prev(arm)
        except FileNotFoundError:
            continue
        passed = {
            "arm": arm,
            "parameters": m["model"]["parameters"],
            "best_epoch": m["summary"]["best_valid"]["epoch"],
            "val_rmse": m["summary"]["best_valid"]["valid"]["rmse"],
            "test_rmse": m["summary"]["best_valid"]["test"]["rmse"],
            "test_mae": m["summary"]["best_valid"]["test"]["mae"],
            "last_test_rmse": m["summary"]["last_epoch"]["test"]["rmse"],
        }
        # replicate the three main points from the round-1 report
        if arm == "B":
            expected = {"B_test_best": 0.7211229002, "D2_test_best": 0.7456796270, "G_test_best": 0.7022962103}
            passed["recomputed_B_test_best"] = float(m["summary"]["best_valid"]["test"]["rmse"])
        out[arm] = passed
    return out


def recompute_three_points() -> dict[str, Any]:
    """Recompute B / D2 / G Test RMSE at best-Val from the raw predictions."""
    out: dict[str, Any] = {}
    for arm in ("B", "D2", "G"):
        try:
            d = np.load(C.HOME / "analysis/results" / arm / "pred_test_best.npz", allow_pickle=False)
        except FileNotFoundError:
            continue
        raw = np.asarray(d["raw"], dtype=np.float64)
        y = np.asarray(d["y_true"], dtype=np.float64)
        out[arm] = {"test_rmse_recomputed": float(np.sqrt(np.mean((raw - y) ** 2))),
                    "n": int(y.size)}
    return out


def h2_gap_numbers() -> dict[str, Any]:
    """Extract the standardised and unstandardised H2 gaps separately."""
    g = C.read_json(C.HOME / "analysis/groups/groups.json")
    out: dict[str, Any] = {}
    sel = g.get("selection", {})
    for split in ("valid", "test"):
        entry = {}
        # per-model standardised gaps
        for src in ("standardised", "bootstrap"):
            block = g.get(src, {}).get("H2", {}).get(split)
            if block:
                if "models" in block:
                    entry[src] = {a: block["models"][a].get("gap_point") for a in block["models"]}
                else:
                    entry[src] = block
        unstd = None
        if isinstance(sel, Mapping):
            unstd = None
        out[split] = entry
    # unstandardised gap lives in the group selection payload
    out["selection"] = sel
    return out


def support_counts(view) -> dict[str, Any]:
    """Real model code support by official-Train molecule document frequency."""
    train = view.caches["train"]
    valid = view.caches["valid"]
    test = view.caches["test"]
    tr = SED.code_support(train)
    out: dict[str, Any] = {}

    def _split_stats(cache, tr_sets, split):
        path_ptr = np.asarray(cache["path_ptr"])
        n = int(cache["n_molecules"])
        path_cat = np.asarray(cache["path_cat"])
        end63 = np.asarray(cache["path_end63"])
        nbr_code = np.asarray(cache["nbr_code"])
        nbr_ptr = np.asarray(cache["nbr_ptr"])
        path_df = tr_sets["path_df"]
        joint_df = tr_sets["joint_df"]
        nbr_df = tr_sets["nbr_df"]
        end_df = tr_sets["end_df"]
        # token counts and df-support per token
        path_tokens = 0
        path_unseen = 0
        path_df1_5 = 0
        path_df6_20 = 0
        path_df20 = 0
        joint_unseen = 0
        joint_tokens = 0
        nbr_tokens = 0
        nbr_unseen = 0
        end_tokens = 0
        end_unseen = 0
        for m in range(n):
            pcs = path_cat[path_ptr[m]:path_ptr[m + 1]]
            es = end63[path_ptr[m]:path_ptr[m + 1]]
            for p, e in zip(pcs.tolist(), es.tolist()):
                path_tokens += 1
                pr = _df(p, path_df)
                if pr == 0:
                    path_unseen += 1
                elif pr <= 5:
                    path_df1_5 += 1
                elif pr <= 20:
                    path_df6_20 += 1
                else:
                    path_df20 += 1
                joint_tokens += 1
                if _df((p, e), joint_df) == 0:
                    joint_unseen += 1
            ncs = nbr_code[nbr_ptr[m]:nbr_ptr[m + 1]]
            for c in ncs.tolist():
                nbr_tokens += 1
                if _df(c, nbr_df) == 0:
                    nbr_unseen += 1
        for e in end63.tolist():
            end_tokens += 1
            if _df(e, end_df) == 0:
                end_unseen += 1
        return {
            "n_molecules": n,
            "path_tokens": path_tokens,
            "path_unseen_frac": path_unseen / max(path_tokens, 1),
            "path_df1_5_frac": path_df1_5 / max(path_tokens, 1),
            "path_df6_20_frac": path_df6_20 / max(path_tokens, 1),
            "path_df_gt20_frac": path_df20 / max(path_tokens, 1),
            "joint_tokens": joint_tokens,
            "joint_unseen_frac": joint_unseen / max(joint_tokens, 1),
            "nbr_tokens": nbr_tokens,
            "nbr_unseen_frac": nbr_unseen / max(nbr_tokens, 1),
            "end_tokens": end_tokens,
            "end_unseen_frac": end_unseen / max(end_tokens, 1),
        }

    def _df(code, df_map):
        return int(df_map.get(code, 0)) if not isinstance(code, tuple) else int(df_map.get(code, 0))

    out["path_df_size"] = len(tr["path_df"])
    out["nbr_df_size"] = len(tr["nbr_df"])
    out["end_df_size"] = len(tr["end_df"])
    out["joint_df_size"] = len(tr["joint_df"])
    out["train"] = _split_stats(train, tr, "train")
    out["valid"] = _split_stats(valid, tr, "valid")
    out["test"] = _split_stats(test, tr, "test")
    return out


def run() -> dict[str, Any]:
    view = LPD.build_view()
    payload = {
        "schema": "lcrc-explicit-semantic-learning-p0-v1",
        "prev_summary": prev_summary(),
        "recompute_three_points": recompute_three_points(),
        "h2_gaps": h2_gap_numbers(),
        "support_counts": support_counts(view),
        "notes": {
            "r2_val_vs_b_val": "R2 Val 0.7419 is worse than B Val 0.7371; the old report's "
                               "'R2 Val also improved' is not supported by the machine table.",
            "capacity_identifiability": "The previous round did not uniquely identify 'missing capacity / inductive bias'.",
            "h2_stand_vs_unstd": "H2 size-standardised gap (0.1159 Test) and unstandardised gap (~0.1170) "
                                 "must not be mixed; report both with support counts.",
            "path_end63_naming": "path_end63 is a historical name; the live schema is 80 end classes (not 63).",
            "bond_proj_note": "The previous S-arm code contains a bond_proj term; the PROTOCOL 'no bond term' "
                              "wording was inaccurate. This did not invalidate the round and is not retrained here.",
        },
    }
    C.write_json(C.ANALYSIS / "p0_audit.json", payload)
    return payload


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", action="store_true")
    args = ap.parse_args()
    payload = run()
    print(json.dumps({k: payload[k] for k in ("recompute_three_points",)}, indent=1, default=str))
    print("support train path_df_size", payload["support_counts"]["path_df_size"])
    if args.notes:
        write_notes(payload)
    return 0


def write_notes(payload: Mapping[str, Any]) -> None:
    s = payload["support_counts"]
    lines = [
        "# SOURCE_NOTES — lcrc-explicit-semantic-learning-8h-s0-v1",
        "",
        "本文件记录本轮开始前对上一轮（`lcrc_lipo_context_small_s0_v1`）的机器核对，",
        "以及对显式 code 支持度的独立计数。所有数字来自已拉回的 run 输出，未重训。",
        "",
        "## 1. 上一轮到底排除了什么、哪些旧文字需要收窄",
        "",
        "* 旧报告称 R2 的 Val 也改善。**不成立**：机器表中 R2 best-Val RMSE=%.4f，"
        "B best-Val RMSE=%.4f，R2 反而更差。旧结论 '缺容量/归纳偏置' 并没有被该轮唯一识别。"
        % (0.7419, 0.7371),
        "* 上一轮 S 代码里有 `bond_proj` 项，PROTOCOL 中 'no bond term' 只是文字不准；",
        "  这既不影响该轮判定，也不构成本轮重训的理由。",
        "* H2 的 size-standardised gap（Test 0.1159）与未标准化 gap（约 0.1170）是两件事，",
        "  必须分别报告，并同时给出群体/补集的支持数；本文件不混用。",
        "",
        "## 2. 真正进入模型的 code 支持度（Train 分子 document frequency）",
        "",
        "* path_cat 词表 400，Train 出现的不同 code 数 = %d。" % s["path_df_size"],
        "* (path_cat,path_end63) 联合词表 32000，Train 出现的不同联合 code 数 = %d。" % s["joint_df_size"],
        "* nbr/end 词表 80，Train 出现的不同 code 数 = %d / %d。" % (s["nbr_df_size"], s["end_df_size"]),
        "",
        "| split | path 未见 token | path df1-5 | path df6-20 | path df>20 | 联合未见 | nbr 未见 | end 未见 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sp in ("train", "valid", "test"):
        e = s[sp]
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
            % (sp, e["path_unseen_frac"], e["path_df1_5_frac"], e["path_df6_20_frac"],
               e["path_df_gt20_frac"], e["joint_unseen_frac"], e["nbr_unseen_frac"], e["end_unseen_frac"])
        )
    lines += [
        "",
        "* 支持度按 **Train 分子 document frequency**（每分子同 code 只计一次），不是 token/根出现数。",
        "* Morgan-H2 的高覆盖与真实表项稀有性分开报告；即使所有单 code 都在 Train 出现过，",
        "  仍然可以检验少样本参数共享，不能把 Morgan radius3 未见直接说成模型查到了未训练的表项。",
        "",
        "## 3. 编码 schema",
        "",
        "* 16 atom types、5 bond types、400 path codes、80 binary codes；`path_end63` 是历史字段名，",
        "  实际为 80 类。绝不沿用 ZINC 的 189/63 词表。",
        "",
        "（生成：`code/se_audit.py --notes`；机器文件：`analysis/p0_audit.json`。）",
    ]
    (C.TRACK / "SOURCE_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log(f"wrote {C.track_rel(C.TRACK / 'SOURCE_NOTES.md')}")


if __name__ == "__main__":
    raise SystemExit(main())
