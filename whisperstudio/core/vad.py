from __future__ import annotations
import numpy as np
from .constants import SR
from .utils import rms

def energy_ok(audio_float_16k: np.ndarray, thr: float = 0.008) -> bool:
    return rms(audio_float_16k) >= float(thr)

try:
    import webrtcvad
    _HAVE_WEBRTC = True
except Exception:
    _HAVE_WEBRTC = False

class WebRtcVad:
    def __init__(self, aggressiveness: int = 2):
        if not _HAVE_WEBRTC:
            raise RuntimeError("webrtcvad nie jest zainstalowany")
        self.vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech_ratio(self, audio_float_16k: np.ndarray, frame_ms: int = 30) -> float:
        x = np.clip(audio_float_16k, -1.0, 1.0)
        pcm = (x * 32768.0).astype(np.int16).tobytes()
        fbytes = int(SR * frame_ms / 1000) * 2
        n = len(pcm) // fbytes
        if n == 0:
            return 0.0
        speech = 0
        off = 0
        for _ in range(n):
            fr = pcm[off:off + fbytes]
            off += fbytes
            if len(fr) < fbytes:
                break
            if self.vad.is_speech(fr, SR):
                speech += 1
        return speech / max(1, n)
