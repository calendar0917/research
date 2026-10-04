"""Regenerate manifest.json for this result directory (sha256 over sorted files)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "zinc-cycle-head-size-input-repair-seed0-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    payload = {"protocol_version": PROTOCOL_VERSION, "n_files": len(files), "files": files}
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
