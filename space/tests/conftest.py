import locale
import os
import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPACE))

if sys.platform == "win32":
    # See bindings/python/tests/conftest.py (llvm-mingw libc++ and the CRT locale).
    locale.setlocale(locale.LC_CTYPE, "C")

os.environ.setdefault("LOOMCORE_SKIP_BENCH", "1")  # the benchmark has its own test; keep startup quick
