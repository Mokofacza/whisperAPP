from __future__ import annotations
from typing import List, Optional, Tuple
import sounddevice as sd

try:
    import soundcard as sc
    HAVE_SOUNDCARD = True
except Exception:
    HAVE_SOUNDCARD = False

def list_input_devices() -> List[Tuple[int, dict]]:
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d))
    except Exception:
        pass
    return out

def default_input() -> Tuple[Optional[int], Optional[dict]]:
    try:
        d = sd.query_devices(kind="input")
        idx = None
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("name") == d.get("name") and dev.get("max_input_channels", 0) > 0:
                idx = i
                break
        return idx, d
    except Exception:
        return None, None

def list_loopback() -> List[str]:
    if not HAVE_SOUNDCARD:
        return []
    return [str(m) for m in sc.all_microphones(include_loopback=True) if getattr(m, "is_loopback", False)]

def get_loopback_microphone(preferred_name: str | None = None):
    if not HAVE_SOUNDCARD:
        return None
    if preferred_name:
        for m in sc.all_microphones(include_loopback=True):
            if preferred_name.lower() in str(m).lower() and getattr(m, "is_loopback", False):
                return m
    try:
        spk = sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.id), include_loopback=True)
        return mic
    except Exception:
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "is_loopback", False):
                return m
    return None
