"""
Hotfix: SoundCard ↔ NumPy 2.x (Windows Media Foundation backend)
W niektórych wersjach soundcard odwołuje się do numpy.fromstring (tryb binarny),
usuniętego w NumPy 2.x. Patchujemy TYLKO moduł soundcard.mediafoundation,
zamieniając wywołania na numpy.frombuffer + copy(), bez wpływu na resztę bibliotek.
"""

from __future__ import annotations

def patch_soundcard_numpy_fromstring() -> bool:
    try:
        import numpy as np
        import soundcard.mediafoundation as _sc_mf  # tylko na Windows

        def _fromstring_compat(buf, dtype=None, **kw):
            # równoważnik trybu binarnego: frombuffer + copy (soundcard oczekuje kopii)
            arr = np.frombuffer(buf, dtype=dtype)
            return arr.copy()

        patched = False
        for mod in (getattr(_sc_mf, "numpy", None), getattr(_sc_mf, "np", None)):
            if mod is not None and hasattr(mod, "fromstring"):
                try:
                    mod.fromstring = _fromstring_compat  # type: ignore[attr-defined]
                    patched = True
                except Exception:
                    pass
        return patched
    except Exception:
        return False
