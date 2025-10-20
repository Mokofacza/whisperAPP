from __future__ import annotations
import time
import platform
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import librosa
import sounddevice as sd

from .constants import SR
from .asr import transcribe_chunk
from .vad import energy_ok, WebRtcVad
from .utils import longest_common_prefix

# Fallback: soundcard (loopback z hotfixem fromstring→frombuffer)
try:
    import soundcard as sc
    HAVE_SC = True
except Exception:
    HAVE_SC = False

@dataclass
class StreamCfg:
    source: str = "mic"          # "mic" | "loopback"
    input_index: Optional[int] = None
    loopback_name: Optional[str] = None   # fragment nazwy do dopasowania
    dev_sr: Optional[int] = None
    chunk_sec: float = 6.0
    stride_sec: float = 1.5
    block_sec: float = 0.2
    vad: str = "energy"          # "energy" | "webrtc" | "off"
    energy_th: float = 0.008
    vad_aggr: int = 2
    silence_resets: int = 3
    min_chars: int = 3
    max_len: int = 225

# --- pomoc: przelicz RMS -> dBFS i na pasek 0..100 ---
def _rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0:
        return -120.0
    x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak
    rms = float(np.sqrt(np.mean(x * x) + eps))
    return 20.0 * np.log10(rms + eps)

def _db_to_bar(db: float, floor: float = -60.0, ceil: float = 0.0) -> int:
    db = max(floor, min(ceil, db))
    val = int(round((db - floor) / (ceil - floor) * 100.0))
    return max(0, min(100, val))

class LiveStreamer:
    def __init__(self, cfg: StreamCfg):
        self.cfg = cfg
        self._stop = False
        self.processor = None
        self.base_model = None
        self.device = None

    def stop(self):
        self._stop = True

    def _emit_increment(self, audio_16k: np.ndarray, prev_hyp: str, on_delta, c) -> str:
        hyp = transcribe_chunk(audio_16k, self.processor, self.base_model, self.device, c.max_len)
        if len(hyp.strip()) < c.min_chars:
            return prev_hyp
        p = longest_common_prefix(prev_hyp, hyp)
        tail = hyp[p:]
        if tail:
            on_delta(tail)
        return hyp

    def _find_sd_loopback_device(self, preferred: Optional[str] = None):
        try:
            devices = sd.query_devices()
            loopbacks = []
            for i, d in enumerate(devices):
                name = str(d.get("name", ""))
                if "loopback" not in name.lower():
                    continue
                hostapi_idx = int(d.get("hostapi", 0))
                host_name = sd.query_hostapis(hostapi_idx).get("name", "").lower()
                if "wasapi" not in host_name:
                    continue
                if int(d.get("max_input_channels", 0)) <= 0:
                    continue
                loopbacks.append((i, d))
            if not loopbacks:
                return None, None
            if preferred:
                pref = preferred.lower()
                for i, d in loopbacks:
                    if pref in str(d.get("name", "")).lower():
                        return i, d
            try:
                out_dev = sd.query_devices(kind="output")
                out_name = str(out_dev.get("name", "")).lower()
                for i, d in loopbacks:
                    if out_name and out_name in str(d.get("name", "")).lower():
                        return i, d
            except Exception:
                pass
            return loopbacks[0]
        except Exception:
            return None, None

    def run(
        self,
        processor,
        base_model,
        device,
        on_delta: Callable[[str], None],
        on_status: Callable[[str], None] = lambda s: None,
        on_level: Callable[[int], None] = lambda v: None,   # <── NOWE
    ):
        c = self.cfg
        self.processor = processor
        self.base_model = base_model
        self.device = device

        webrtc = None
        if c.vad == "webrtc":
            try:
                webrtc = WebRtcVad(c.vad_aggr)
            except Exception:
                on_status("webrtcvad niedostępny – używam energy gate")
                c.vad = "energy"

        prev_hyp = ""
        sil_cnt = 0

        # -------- MIC (sounddevice) --------
        if c.source == "mic":
            try:
                dev = sd.query_devices(c.input_index) if c.input_index is not None else sd.query_devices(kind="input")
                DEV_SR = int(dev.get("default_samplerate", 48000))
                CH = max(1, min(2, dev.get("max_input_channels", 1)))
            except Exception as e:
                on_status(f"Błąd wejścia: {e}")
                return

            chunk_s = int(DEV_SR * c.chunk_sec)
            stride_s = int(DEV_SR * c.stride_sec)
            blocksize = int(DEV_SR * c.block_sec) if c.block_sec > 0 else 0
            ring = np.zeros((0, CH), dtype=np.float32)
            samples_since_last = 0
            last_level = 0

            def cb(indata, frames, time_info, status):
                nonlocal ring, samples_since_last, last_level
                if status:
                    pass
                ring = np.concatenate([ring, indata.copy()], axis=0)
                if len(ring) > chunk_s * 2:
                    ring = ring[-chunk_s * 2 :, :]
                samples_since_last += frames
                # poziom z bieżącego bloku
                mono = (
                    indata.mean(axis=1).astype(np.float32)
                    if indata.ndim == 2 and indata.shape[1] > 1
                    else indata.reshape(-1).astype(np.float32)
                )
                last_level = _db_to_bar(_rms_db(mono))

            try:
                with sd.InputStream(
                    samplerate=DEV_SR,
                    channels=CH,
                    dtype="float32",
                    device=c.input_index,
                    callback=cb,
                    blocksize=blocksize,
                ):
                    while not self._stop:
                        time.sleep(0.01)
                        on_level(last_level)  # <── aktualizacja paska
                        if samples_since_last < stride_s or len(ring) < chunk_s:
                            continue
                        audio_dev = ring[-chunk_s:, :]
                        samples_since_last = 0
                        mono = (
                            audio_dev.mean(axis=1).astype(np.float32)
                            if audio_dev.ndim == 2 and audio_dev.shape[1] > 1
                            else audio_dev.reshape(-1).astype(np.float32)
                        )
                        audio_16k = librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR) if DEV_SR != SR else mono

                        allow = True
                        if c.vad == "energy":
                            allow = energy_ok(audio_16k, c.energy_th)
                        elif c.vad == "webrtc" and webrtc is not None:
                            allow = (webrtc.is_speech_ratio(audio_16k) >= 0.6)

                        if not allow:
                            sil_cnt += 1
                            if sil_cnt >= c.silence_resets:
                                if prev_hyp:
                                    on_delta("\n")
                                prev_hyp = ""
                                sil_cnt = 0
                            continue
                        else:
                            sil_cnt = 0

                        prev_hyp = self._emit_increment(audio_16k, prev_hyp, on_delta, c)
            except Exception as e:
                on_status(f"Błąd audio: {e}")
            return

        # -------- LOOPBACK: sounddevice/PortAudio (WASAPI) --------
        if c.source == "loopback" and platform.system().lower().startswith("win"):
            idx, info = self._find_sd_loopback_device(preferred=c.loopback_name)
            if idx is not None and info is not None:
                try:
                    DEV_SR = c.dev_sr or int(info.get("default_samplerate", 48000))
                    CH = max(1, min(2, int(info.get("max_input_channels", 2))))
                    dev_block  = int(DEV_SR * (c.block_sec if c.block_sec > 0 else 0.05))
                    dev_chunk  = int(DEV_SR * c.chunk_sec)
                    dev_stride = int(DEV_SR * c.stride_sec)
                    ring = np.zeros((0, CH), dtype=np.float32)
                    samples_since_last = 0
                    last_level = 0

                    def cb(indata, frames, time_info, status):
                        nonlocal ring, samples_since_last, last_level
                        if status:
                            pass
                        ring = np.concatenate([ring, indata.copy()], axis=0)
                        if len(ring) > dev_chunk * 2:
                            ring = ring[-dev_chunk * 2 :, :]
                        samples_since_last += frames
                        mono = (
                            indata.mean(axis=1).astype(np.float32)
                            if indata.ndim == 2 and indata.shape[1] > 1
                            else indata.reshape(-1).astype(np.float32)
                        )
                        last_level = _db_to_bar(_rms_db(mono))

                    with sd.InputStream(
                        samplerate=DEV_SR,
                        channels=CH,
                        dtype="float32",
                        device=idx,
                        blocksize=dev_block,
                        callback=cb,
                    ):
                        while not self._stop:
                            time.sleep(0.01)
                            on_level(last_level)
                            if samples_since_last < dev_stride or len(ring) < dev_chunk:
                                continue
                            audio_dev = ring[-dev_chunk:, :]
                            samples_since_last = 0
                            mono = (
                                audio_dev.mean(axis=1).astype(np.float32)
                                if audio_dev.ndim == 2 and audio_dev.shape[1] > 1
                                else audio_dev.reshape(-1).astype(np.float32)
                            )
                            audio_16k = librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR) if DEV_SR != SR else mono

                            allow = True
                            if c.vad == "energy":
                                allow = energy_ok(audio_16k, c.energy_th)
                            elif c.vad == "webrtc" and webrtc is not None:
                                allow = (webrtc.is_speech_ratio(audio_16k) >= 0.6)

                            if not allow:
                                sil_cnt += 1
                                if sil_cnt >= c.silence_resets:
                                    if prev_hyp:
                                        on_delta("\n")
                                    prev_hyp = ""
                                    sil_cnt = 0
                                continue
                            else:
                                sil_cnt = 0

                            prev_hyp = self._emit_increment(audio_16k, prev_hyp, on_delta, c)
                    return
                except Exception as e:
                    on_status(f"WASAPI loopback (sounddevice) nieudany: {e}")

        # -------- LOOPBACK fallback: soundcard --------
        if c.source == "loopback":
            if not HAVE_SC:
                on_status("Brak biblioteki 'soundcard' – loopback niedostępny.")
                return

            # wybór urządzenia loopback
            mic = None
            try:
                if c.loopback_name:
                    pref = c.loopback_name.lower()
                    for m in sc.all_microphones(include_loopback=True):
                        if getattr(m, "is_loopback", False) and pref in str(m).lower():
                            mic = m; break
                if mic is None:
                    spk = sc.default_speaker()
                    mic = sc.get_microphone(id=str(spk.id), include_loopback=True)
                if mic is None:
                    for m in sc.all_microphones(include_loopback=True):
                        if getattr(m, "is_loopback", False):
                            mic = m; break
            except Exception as e:
                on_status(f"Soundcard: problem z wyborem urządzenia: {e}")
                return

            if mic is None:
                on_status("Soundcard: brak mikrofonów loopback. Włącz 'Stereo Mix' lub zainstaluj VB-Cable.")
                return

            try:
                DEV_SR = c.dev_sr or SR
                CH = 2
                dev_block  = int(DEV_SR * (c.block_sec if c.block_sec > 0 else 0.05))
                dev_chunk  = int(DEV_SR * c.chunk_sec)
                dev_stride = int(DEV_SR * c.stride_sec)
                ring = np.zeros((0, CH), dtype=np.float32)
                samples_since_last = 0

                on_status("Używam fallbacku 'soundcard' (loopback, SR=16000).")

                with mic.recorder(samplerate=DEV_SR, channels=CH, blocksize=dev_block) as rec:
                    while not self._stop:
                        block = rec.record(numframes=dev_block)
                        if block is None or len(block) == 0:
                            continue
                        # poziom z aktualnego bloku
                        blk_mono = (
                            block.mean(axis=1).astype(np.float32)
                            if block.ndim == 2 and block.shape[1] > 1
                            else block.reshape(-1).astype(np.float32)
                        )
                        on_level(_db_to_bar(_rms_db(blk_mono)))

                        ring = np.concatenate([ring, block], axis=0)
                        if len(ring) > dev_chunk * 2:
                            ring = ring[-dev_chunk * 2 :, :]
                        samples_since_last += len(block)
                        if samples_since_last < dev_stride or len(ring) < dev_chunk:
                            continue
                        audio_dev = ring[-dev_chunk:, :]
                        samples_since_last = 0
                        mono = (
                            audio_dev.mean(axis=1).astype(np.float32)
                            if audio_dev.ndim == 2 and audio_dev.shape[1] > 1
                            else audio_dev.reshape(-1).astype(np.float32)
                        )
                        audio_16k = mono if DEV_SR == SR else librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR)

                        allow = True
                        if c.vad == "energy":
                            allow = energy_ok(audio_16k, c.energy_th)
                        elif c.vad == "webrtc" and webrtc is not None:
                            allow = (webrtc.is_speech_ratio(audio_16k) >= 0.6)

                        if not allow:
                            sil_cnt += 1
                            if sil_cnt >= c.silence_resets:
                                if prev_hyp:
                                    on_delta("\n")
                                prev_hyp = ""
                                sil_cnt = 0
                            continue
                        else:
                            sil_cnt = 0

                        prev_hyp = self._emit_increment(audio_16k, prev_hyp, on_delta, c)
            except Exception as e:
                on_status(f"Błąd loopback (soundcard): {e}")
            return
