"""Download official TU/OGB data into the repository data root.

Peptides-func is not downloadable from the execute host; it is transferred as a
dataset cache and this module only verifies its presence.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

import bp_common as C


def _download(url: str, dest: Path, timeout: int = 120) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)
    return {"url": url, "path": str(dest), "bytes": int(dest.stat().st_size), "sha256": C.sha256_file(dest)}


def ensure_tu(name: str) -> dict[str, Any]:
    target = C.TU_ROOT / name / name / "raw"
    core = [target / f"{name}_A.txt", target / f"{name}_graph_indicator.txt", target / f"{name}_graph_labels.txt"]
    if all(p.is_file() for p in core):
        return {"dataset": name, "status": "present", "raw_dir": C.track_rel(target)}
    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    with tempfile.TemporaryDirectory(prefix=f"tu_{name}_") as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / f"{name}.zip"
        info = _download(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir / "extract")
        hits = list((tmpdir / "extract").rglob(f"{name}_A.txt"))
        if len(hits) != 1:
            raise RuntimeError(f"{name}: expected one {name}_A.txt in the official zip, found {len(hits)}")
        src = hits[0].parent
        target.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, target / p.name)
        info["extracted_from"] = C.track_rel(src) if str(src).startswith(str(C.REPO)) else str(src)
    missing = [p.name for p in core if not p.is_file()]
    if missing:
        raise RuntimeError(f"{name}: download incomplete, missing {missing}")
    return {"dataset": name, "status": "downloaded", "raw_dir": C.track_rel(target), "download": info}


def ensure_ogb(name: str) -> dict[str, Any]:
    from ogb.graphproppred import GraphPropPredDataset

    ds = GraphPropPredDataset(name=name, root=str(C.OGB_ROOT))
    mapping = Path(ds.root) / "mapping" / "mol.csv.gz"
    if not mapping.is_file():
        raise RuntimeError(f"{name}: official mapping missing after load")
    return {
        "dataset": name,
        "status": "present",
        "root": C.track_rel(Path(ds.root)),
        "n_graphs": int(len(ds)),
        "mapping_sha256": C.sha256_file(mapping),
    }


def ensure_peptides() -> dict[str, Any]:
    root = C.LRGB_ROOT / "peptides-func"
    files = [root / "processed" / f"{s}.pt" for s in ("train", "val", "test")]
    if all(p.is_file() for p in files):
        return {
            "dataset": C.PEPTIDE_DATASET,
            "status": "present",
            "files": [{"file": C.track_rel(p), "bytes": int(p.stat().st_size), "sha256": C.sha256_file(p)} for p in files],
        }
    return {
        "dataset": C.PEPTIDE_DATASET,
        "status": "MISSING_EXTERNAL_TRANSFER_REQUIRED",
        "expected": [C.track_rel(p) for p in files],
        "reason": "Dropbox is unreachable from the execute host; the official processed cache must be transferred as data",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tu", nargs="*", default=None)
    parser.add_argument("--ogb", nargs="*", default=None)
    parser.add_argument("--peptides", action="store_true")
    args = parser.parse_args(argv)
    out: dict[str, Any] = {"generated_utc": C.utcnow(), "results": []}
    tu_names = args.tu if args.tu is not None else list(C.TU_DATASETS)
    ogb_names = args.ogb if args.ogb is not None else []
    for name in tu_names:
        try:
            out["results"].append(ensure_tu(name))
        except Exception as exc:  # noqa: BLE001
            out["results"].append({"dataset": name, "status": "DOWNLOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"})
    for name in ogb_names:
        try:
            out["results"].append(ensure_ogb(name))
        except Exception as exc:  # noqa: BLE001
            out["results"].append({"dataset": name, "status": "DOWNLOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"})
    if args.peptides:
        out["results"].append(ensure_peptides())
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(C.ANALYSIS / "data_fetch.json", out)
    print(C.stable_json(out))
    return 0 if all(r.get("status") in ("present", "downloaded") for r in out["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
