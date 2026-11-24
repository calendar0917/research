"""Human-readable decoding of selected structural words (report support only)."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

import tc_common as C

ATOM = {5: "C", 6: "N", 7: "O", 8: "F", 9: "Ne", 15: "P", 16: "S", 17: "Cl",
        35: "Br", 53: "I"}
HYB = ["sp", "sp2", "sp3", "sp3d", "sp3d2", "misc"]
BOND = ["single", "double", "triple", "aromatic", "misc"]


def atom_full(f: list[int]) -> str:
    sym = ATOM.get(int(f[0]), f"A{int(f[0])}")
    charge = int(f[3]) - 5
    ch = "" if charge == 0 else (f"{charge:+d}")
    h = int(f[4])
    hyb = HYB[int(f[6])] if int(f[6]) < len(HYB) else f"hyb{int(f[6])}"
    bits = [f"deg{int(f[2])}", f"H{h}", hyb]
    if int(f[7]):
        bits.append("arom")
    if int(f[8]):
        bits.append("ring")
    return f"{sym}{ch}({','.join(bits)})"


def atom_coarse(d: dict[str, Any]) -> str:
    sym = ATOM.get(int(d["atomic_num"]), f"A{int(d['atomic_num'])}")
    charge = int(d["formal_charge"]) - 5
    ch = "" if charge == 0 else f"{charge:+d}"
    return f"{sym}{ch}" + ("(arom)" if int(d["is_aromatic"]) else "")


def decode(structure: dict[str, Any]) -> str:
    root = int(structure["root_slot"])
    nodes = []
    gran = structure["gran"]
    for i, nd in enumerate(structure["node_fields"]):
        label = atom_full(nd["fields9"]) if gran == "FULL" else atom_coarse(nd)
        nodes.append(("*" if i == root else "") + label)
    edges = []
    for b in structure["internal_bonds"]:
        s, t = b["slots"]
        bt = (BOND[int(b["fields"][0])] if gran == "FULL"
              else BOND[int(b["bond_type"])])
        edges.append(f"{s}-{t}:{bt}")
    return " ".join(nodes) + " | " + ", ".join(edges)


def top_rows(path: Path, arm: str, limit: int) -> list[dict[str, Any]]:
    rows = [r for r in csv.DictReader(path.open()) if r["arm"] == arm]
    rows.sort(key=lambda r: int(r["rank"]))
    out = []
    for r in rows[:limit]:
        corr = ast.literal_eval(r["cond_corr_folds"])
        out.append({
            "rank": int(r["rank"]), "size": int(r["size"]), "gran": r["gran"],
            "df": int(r["df"]), "scaffold_df": int(r["scaffold_df"]),
            "score": float(r["score"]),
            "fold_corr": [float(x) for x in corr],
            "sign": "+" if sum(corr) > 0 else "-",
            "pattern": decode(ast.literal_eval(r["structure"])),
            "key": r["key"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    lines = ["# Selected structural words (D_TASK)", "",
             "Decoded from `analysis/selected_words_D0.csv` / `_D1.csv`.",
             "Full field indices follow the official OGB atom/bond feature order.",
             "`*` marks the rooted atom; edges are `slot-slot:bondtype`.", ""]
    for split in ("D0", "D1"):
        rows = top_rows(C.ANALYSIS_DIR / f"selected_words_{split}.csv", "D_TASK",
                        args.limit)
        lines += [f"## {split} top {len(rows)} by frozen score", "",
                  "| # | size/gran | df | scaffold df | score | fold corr | pattern |",
                  "|---:|---|---:|---:|---:|---|---|"]
        for r in rows[:10]:
            fc = ", ".join(f"{x:+.3f}" for x in r["fold_corr"])
            lines.append(f"| {r['rank']} | {r['size']} {r['gran']} | {r['df']} | "
                         f"{r['scaffold_df']} | {r['score']:.4f} | {fc} | "
                         f"`{r['pattern']}` |")
        lines += ["", "Remaining top words:", ""]
        for r in rows[10:]:
            fc = ", ".join(f"{x:+.3f}" for x in r["fold_corr"])
            lines.append(f"- **{r['rank']}** {r['size']} {r['gran']}, df={r['df']}, "
                         f"score={r['score']:.4f}, corr=[{fc}]  \n  `{r['pattern']}`")
        lines.append("")
    (C.ANALYSIS_DIR / "top_words.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {C.ANALYSIS_DIR / 'top_words.md'}")


if __name__ == "__main__":
    main()