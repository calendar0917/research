"""Convenience dispatcher for the round's fixed entry points.

Examples
--------
python code/run.py prepare --full
python code/run.py prepare --cache-only          # remote CPU prepare
python code/run.py gates
python code/run.py smoke --device cuda           # four arms x 2 epochs, separate outputs
python code/run.py train --arm A --device cuda   # one formal unit
python code/run.py evaluate                      # local CPU analysis from pulled predictions
python code/run.py post-gates                    # checkpoint / metric / RNG cross-checks
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "prepare":
        import prepare

        return prepare.main(rest)
    if cmd == "gates":
        import gates

        return gates.main(rest)
    if cmd == "post-gates":
        import gates

        return gates.main(["--post", *rest])
    if cmd == "smoke":
        import train

        return train.main(["--smoke-all", *rest])
    if cmd == "train":
        import train

        return train.main(rest)
    if cmd == "evaluate":
        import evaluate

        return evaluate.main()
    raise SystemExit(f"unknown command {cmd!r}; see --help")


if __name__ == "__main__":
    raise SystemExit(main())
