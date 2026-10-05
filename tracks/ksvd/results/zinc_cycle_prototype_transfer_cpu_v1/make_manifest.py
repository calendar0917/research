"""zinc_cycle_prototype_transfer_cpu_v1 — manifest builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "__pycache__"}:
            if "__pycache__" in path.parts:
                continue
            files[str(path.relative_to(OUT))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    manifest = {
        "protocol_version": "zinc-cycle-prototype-transfer-cpu-v1",
        "files": files,
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
