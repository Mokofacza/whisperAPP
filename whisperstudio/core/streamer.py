# -*- coding: utf-8 -*-
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import librosa
import numpy as np
import sounddevice as sd

from .asr import transcribe_chunk
from .constants import SR
from .utils import longest_common_prefix
from .vad import WebRtcVad, energy_ok

# Fallback: soundcard (loopback, z naszym hotfixem fromstring→frombuffer)
try:
    import soundcard as sc
    HAVE_SC = True
except Exception:
    HAVE_SC = False


# ───────────────────────────────────────────────────────────────────────────────
# Konfiguracja streamera
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamCfg:
    source: str = "mic"                # "mic" | "loopback"
    input_index: Optional[int] = None
    loopback_name: Optional[str] = None   # fragment nazwy do dopasowania
    dev_sr: Optional[int] = None
    chunk_sec: float = 6.0
    stride_sec: float = 1.5
    block_sec: float = 0.2
    vad: str = "energy"                # "energy" | "webrtc" | "off"
    energy_th: float = 0.008
    vad_aggr: int = 2
    silence_resets: int = 3
    min_chars: int = 3
    max_len: int = 225
    # Autokalibracja progu energii (zgodna z energy_ok)
    auto_energy: bool = True
    auto_calib_sec: float = 1.5
    auto_mult: float = 2.5
    auto_floor: float = 0.002
    auto_ceil: float = 0.020


# ───────────────────────────────────────────────────────────────────────────────
# Pomocnicze: RMS/poziom, pre-emphasis, ograniczanie rozmiaru bloków
# ───────────────────────────────────────────────────────────────────────────────

def _rms_abs(x: np.ndarray, eps: float = 1e-12) -> float:
    """Bezwzględny RMS względem pełnej skali ±1.0."""
    if x.size == 0:
        return 0.0
    x = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(x * x) + eps))

def _rms_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    r = _rms_abs(x, eps)
    return 20.0 * np.log10(r + eps)

def _db_to_bar(db: float, floor: float = -60.0, ceil: float = 0.0) -> int:
    db = max(floor, min(ceil, db))
    return int(round((db - floor) / (ceil - floor) * 100.0))

def _preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.size < 2:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y

def _cap_block_seconds(sec: float, backend: str) -> float:
    """
    Ogranicz rozmiar bloku (w sekundach) do stabilnego zakresu.
    backend ∈ {"sd", "sc"} -> sounddevice / soundcard
    """
    if backend == "sc":          # soundcard: najlepiej na bardzo krótkich blokach
        return 0.05              # 50 ms
    # sounddevice: 20–250 ms
    sec = max(0.02, min(0.25, float(sec)))
    return sec


# ───────────────────────────────────────────────────────────────────────────────
# Główny streamer
# ───────────────────────────────────────────────────────────────────────────────

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

    # ── znajdź urządzenie loopback w sounddevice/WASAPI ────────────────────────
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
                if "wasapi" not in host_name or int(d.get("max_input_channels", 0)) <= 0:
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

    # ── AUTOKALIBRACJA (sounddevice) – RMS bezwzględny po pre-emphasis ────────
    def _auto_calibrate_sd(self, dev_index: Optional[int], sr: int, ch: int, block_s: int,
                           seconds: float, on_status, on_level) -> float:
        target = int(seconds * sr)
        got = 0
        r_list: List[float] = []

        def cb(indata, frames, time_info, status):
            nonlocal got, r_list
            mono = (indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.reshape(-1)).astype(np.float32)
            got += frames
            on_level(_db_to_bar(_rms_dbfs(mono)))
            mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=SR) if sr != SR else mono
            r_list.append(_rms_abs(_preemphasis(mono_16k)))

        try:
            with sd.InputStream(samplerate=sr, channels=ch, dtype="float32",
                                device=dev_index, callback=cb, blocksize=block_s):
                while not self._stop and got < target:
                    time.sleep(0.02)
        except Exception as e:
            on_status(f"AUTO: nie udało się skalibrować (sounddevice): {e}")
            return self.cfg.energy_th

        if not r_list:
            return self.cfg.energy_th
        noise = float(np.percentile(r_list, 95))  # odporny na pojedyncze „piknięcia”
        th = float(np.clip(noise * self.cfg.auto_mult, self.cfg.auto_floor, self.cfg.auto_ceil))
        on_status(f"AUTO: szum={20*np.log10(noise+1e-12):.1f} dBFS → próg={th:.4f}")
        return th

    # ── AUTOKALIBRACJA (soundcard) – RMS bezwzględny po pre-emphasis ──────────
    def _auto_calibrate_sc(self, mic, sr: int, ch: int, block_s: int,
                           seconds: float, on_status, on_level) -> float:
        target = int(seconds * sr)
        got = 0
        r_list: List[float] = []
        try:
            with mic.recorder(samplerate=sr, channels=ch, blocksize=block_s) as rec:
                while not self._stop and got < target:
                    blk = rec.record(numframes=block_s)
                    if blk is None or len(blk) == 0:
                        continue
                    mono = (blk.mean(axis=1) if blk.ndim == 2 and blk.shape[1] > 1 else blk.reshape(-1)).astype(np.float32)
                    got += len(mono)
                    on_level(_db_to_bar(_rms_dbfs(mono)))
                    mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=SR) if sr != SR else mono
                    r_list.append(_rms_abs(_preemphasis(mono_16k)))
        except Exception as e:
            on_status(f"AUTO: nie udało się skalibrować (soundcard): {e}")
            return self.cfg.energy_th

        if not r_list:
            return self.cfg.energy_th
        noise = float(np.percentile(r_list, 95))
        th = float(np.clip(noise * self.cfg.auto_mult, self.cfg.auto_floor, self.cfg.auto_ceil))
        on_status(f"AUTO: szum={20*np.log10(noise+1e-12):.1f} dBFS → próg={th:.4f}")
        return th

    # ── GŁÓWNA PĘTLA ──────────────────────────────────────────────────────────
    def run(
        self,
        processor,
        base_model,
        device,
        on_delta: Callable[[str], None],
        on_status: Callable[[str], None] = lambda s: None,
        on_level: Callable[[int], None] = lambda v: None,
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

        # ---------------- MIC (sounddevice) ----------------
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
            blocksize = int(DEV_SR * _cap_block_seconds(c.block_sec, "sd"))

            # ► autokalibracja progu
            if c.vad == "energy" and c.auto_energy:
                on_status(f"AUTO: kalibracja tła ({c.auto_calib_sec:.1f}s)… nie mów nic.")
                c.energy_th = self._auto_calibrate_sd(c.input_index, DEV_SR, CH, blocksize, c.auto_calib_sec, on_status, on_level)

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
                mono_blk = (indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.reshape(-1)).astype(np.float32)
                last_level = _db_to_bar(_rms_dbfs(mono_blk))

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
                        on_level(last_level)
                        if samples_since_last < stride_s or len(ring) < chunk_s:
                            continue
                        audio_dev = ring[-chunk_s:, :]
                        samples_since_last = 0
                        mono = (audio_dev.mean(axis=1) if audio_dev.ndim == 2 and audio_dev.shape[1] > 1 else audio_dev.reshape(-1)).astype(np.float32)
                        audio_16k = librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR) if DEV_SR != SR else mono
                        audio_16k = _preemphasis(audio_16k)

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

        # ---------------- LOOPBACK (sounddevice/WASAPI) ----------------
        if c.source == "loopback" and platform.system().lower().startswith("win"):
            idx, info = self._find_sd_loopback_device(preferred=c.loopback_name)
            if idx is not None and info is not None:
                try:
                    DEV_SR = c.dev_sr or int(info.get("default_samplerate", 48000))
                    CH = max(1, min(2, int(info.get("max_input_channels", 2))))
                    dev_block  = int(DEV_SR * _cap_block_seconds(c.block_sec, "sd"))
                    dev_chunk  = int(DEV_SR * c.chunk_sec)
                    dev_stride = int(DEV_SR * c.stride_sec)

                    # ► autokalibracja
                    if c.vad == "energy" and c.auto_energy:
                        on_status(f"AUTO: kalibracja tła ({c.auto_calib_sec:.1f}s)… wycisz źródło.")
                        c.energy_th = self._auto_calibrate_sd(idx, DEV_SR, CH, dev_block, c.auto_calib_sec, on_status, on_level)

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
                        mono_blk = (indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.reshape(-1)).astype(np.float32)
                        last_level = _db_to_bar(_rms_dbfs(mono_blk))

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
                            mono = (audio_dev.mean(axis=1) if audio_dev.ndim == 2 and audio_dev.shape[1] > 1 else audio_dev.reshape(-1)).astype(np.float32)
                            audio_16k = librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR) if DEV_SR != SR else mono
                            audio_16k = _preemphasis(audio_16k)

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

        # ---------------- LOOPBACK fallback (soundcard) ----------------
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
                dev_block  = int(DEV_SR * _cap_block_seconds(c.block_sec, "sc"))
                dev_chunk  = int(DEV_SR * c.chunk_sec)
                dev_stride = int(DEV_SR * c.stride_sec)
                ring = np.zeros((0, CH), dtype=np.float32)
                samples_since_last = 0

                on_status("Używam fallbacku 'soundcard' (loopback).")

                with mic.recorder(samplerate=DEV_SR, channels=CH, blocksize=dev_block) as rec:
                    while not self._stop:
                        block = rec.record(numframes=dev_block)
                        if block is None or len(block) == 0:
                            continue
                        blk_mono = (block.mean(axis=1) if block.ndim == 2 and block.shape[1] > 1 else block.reshape(-1)).astype(np.float32)
                        on_level(_db_to_bar(_rms_dbfs(blk_mono)))

                        ring = np.concatenate([ring, block], axis=0)
                        if len(ring) > dev_chunk * 2:
                            ring = ring[-dev_chunk * 2 :, :]
                        samples_since_last += len(block)
                        if samples_since_last < dev_stride or len(ring) < dev_chunk:
                            continue
                        audio_dev = ring[-dev_chunk:, :]
                        samples_since_last = 0
                        mono = (audio_dev.mean(axis=1) if audio_dev.ndim == 2 and audio_dev.shape[1] > 1 else audio_dev.reshape(-1)).astype(np.float32)
                        audio_16k = mono if DEV_SR == SR else librosa.resample(mono, orig_sr=DEV_SR, target_sr=SR)
                        audio_16k = _preemphasis(audio_16k)

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
