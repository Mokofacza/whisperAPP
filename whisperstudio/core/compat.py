"""
Numpy 2.x compatibility shim:
- Reimplements the removed binary mode of np.fromstring using np.frombuffer
- Only intercepts calls where `sep==''` (binary) AND input is bytes-like.
- Falls back to the original behavior for text parsing.
"""

from __future__ import annotations
from typing import Any

def apply_numpy_compat(np_module: Any | None = None) -> None:
    try:
        import numpy as _np
    except Exception:
        return

    np = np_module or _np

    # Jeśli nie ma atrybutu wersji albo to <2.0 – nie dotykamy.
    ver = getattr(np, "__version__", "0")
    try:
        major = int(ver.split(".", 1)[0])
    except Exception:
        major = 0
    if major < 2:
        return

    # Jeśli fromstring nie istnieje (albo już kompatybilne) – nic nie rób.
    if not hasattr(np, "fromstring"):
        return

    _orig_fromstring = np.fromstring

    def _fromstring_compat(string, dtype=float, count=-1, sep=""):
        """
        Kompatybilność binarna:
        - gdy `string` to bytes/bytearray/memoryview i sep in {"", None} -> użyj np.frombuffer
        - w pozostałych przypadkach deleguj do oryginalnego fromstring (parsowanie tekstowe)
        """
        try:
            if (sep == "" or sep is None) and isinstance(string, (bytes, bytearray, memoryview)):
                # W trybie binarnym fromstring==frombuffer — implementujemy nową ścieżkę.
                return np.frombuffer(string, dtype=dtype, count=count)
        except Exception:
            # Jeśli coś poszło nie tak, spróbuj oryginalną ścieżką (dla zgodności).
            pass
        return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)

    # Podmień tylko raz
    if getattr(np.fromstring, "__name__", "") != "_fromstring_compat":
        np.fromstring = _fromstring_compat  # type: ignore[attr-defined]
