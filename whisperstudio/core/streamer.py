from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import librosa
import sounddevice as sd

from .constants import SR
from .asr import transcribe_chunk
from .vad import energy_ok, WebRtcVad
from .utils import longest_common_prefix

try:
    import soundcard as sc
    HAVE_SC = True
except Exception:
    HAVE_SC = False

@dataclass
class StreamCfg:
    source: str = "mic"          # mic | loopback
    input_index: Optional[int] = None
    loopback_name: Optional[str] = None
    dev_sr: Optional[int] = None
    chunk_sec: float = 6.0
    stride_sec: float = 1.5
    block_sec: float = 0.2
    vad: str = "energy"          # energy | webrtc | off
    energy_th: float = 0.008
    vad_aggr: int = 2
    silence_resets: int = 3
    min_chars: int = 3
    max_len: int = 225

class LiveStreamer:
    def __init__(self, cfg: StreamCfg):
        self.cfg = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(
        self,
        processor,
        base_model,
        device,
        on_delta: Callable[[str], None],
        on_status: Callable[[str], None] = lambda s: None
    ):
        c = self.cfg

        webrtc = None
        if c.vad == "webrtc":
            try:
                webrtc = WebRtcVad(c.vad_aggr)
            except Exception:
                on_status("webrtcvad niedostępny – używam energy gate")
                c.vad = "energy"

        prev_hyp = ""
        sil_cnt = 0

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

            def cb(indata, frames, time_info, status):
                nonlocal ring, samples_since_last
                if status:  # zachowaj ciszę
                    pass
                ring = np.concatenate([ring, indata.copy()], axis=0)
                if len(ring) > chunk_s * 2:
                    ring = ring[-chunk_s * 2 :, :]
                samples_since_last += frames

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

                        hyp = transcribe_chunk(audio_16k, processor, base_model, device, c.max_len)
                        if len(hyp.strip()) < c.min_chars:
                            continue
                        p = longest_common_prefix(prev_hyp, hyp)
                        tail = hyp[p:]
                        if tail:
                            on_delta(tail)
                        prev_hyp = hyp
            except Exception as e:
                on_status(f"Błąd audio: {e}")
            return

        if c.source == "loopback":
            if not HAVE_SC:
                on_status("Brak biblioteki 'soundcard'")
                return
            try:
                spk = sc.default_speaker()
                mic = sc.get_microphone(id=str(spk.id), include_loopback=True)
                DEV_SR = c.dev_sr or 48000
            except Exception as e:
                on_status(str(e))
                return

            CH = 2
            dev_block  = int(DEV_SR * (c.block_sec if c.block_sec > 0 else 0.05))
            dev_chunk  = int(DEV_SR * c.chunk_sec)
            dev_stride = int(DEV_SR * c.stride_sec)
            ring = np.zeros((0, CH), dtype=np.float32)
            samples_since_last = 0

            try:
                with mic.recorder(samplerate=DEV_SR, channels=CH) as rec:
                    while not self._stop:
                        block = rec.record(numframes=dev_block)
                        if block is None or len(block) == 0:
                            continue
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

                        hyp = transcribe_chunk(audio_16k, processor, base_model, device, c.max_len)
                        if len(hyp.strip()) < c.min_chars:
                            continue
                        p = longest_common_prefix(prev_hyp, hyp)
                        tail = hyp[p:]
                        if tail:
                            on_delta(tail)
                        prev_hyp = hyp
            except Exception as e:
                on_status(f"Błąd loopback: {e}")
            return
