# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable, List

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

SR = 16000  # stała próbkowania


# ========================= Helpers: device & filesystem =========================

def _pick_device(device_str: str = "auto") -> str:
    ds = (device_str or "auto").lower()
    if ds == "cpu":
        return "cpu"
    if torch.cuda.is_available() and ds in ("auto", "cuda", "gpu"):
        return "cuda"
    if torch.backends.mps.is_available() and ds in ("auto", "mps"):
        return "mps"
    return "cpu"


def _exists(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        pp = Path(p)
        return str(pp) if pp.exists() else None
    except Exception:
        return None


# Katalog uznajemy za "z wagami", kiedy zawiera którykolwiek z poniższych plików
_MODEL_FILES: tuple[str, ...] = (
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",   # shardy
    "tf_model.h5",
    "flax_model.msgpack",
)

# Katalog z procesorem/tokenizerem
_PROCESSOR_FILES: tuple[str, ...] = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
)

# LoRA – bardzo szerokie wzorce (również stare nazwy i shardy)
_LORA_PATTERNS: tuple[str, ...] = (
    "adapter_model*.safetensors",
    "adapter_model*.bin",
    "pytorch_lora_weights*.safetensors",
    "pytorch_lora_weights*.bin",
    "pytorch_model*.safetensors",     # niektóre eksporty LoRA
    "pytorch_model*.bin",             # starsze eksporty LoRA
    "adapter_model.safetensors.index.json",
    "pytorch_model.safetensors.index.json",
)


def _has_any(path: Path, names: Iterable[str]) -> bool:
    return any((path / n).exists() for n in names)


def _has_any_glob(path: Path, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if list(path.glob(pat)):
            return True
    return False


def _find_model_dir(root: Optional[str]) -> Optional[str]:
    """Znajdź katalog zawierający wagi (checkpoint)."""
    if not root:
        return None
    base = Path(root)
    if not base.is_dir():
        return None
    if _has_any(base, _MODEL_FILES):
        return str(base)
    cands = [p for p in base.rglob("*") if p.is_dir() and _has_any(p, _MODEL_FILES)]
    if cands:
        sel = max(cands, key=lambda p: p.stat().st_mtime)
        return str(sel)
    return None


def _find_processor_dir(root: Optional[str], prefer: Optional[str] = None) -> Optional[str]:
    """Znajdź katalog z preprocessor_config.json (tokenizer/FE)."""
    for candidate in (prefer, root):
        if candidate:
            p = Path(candidate)
            if p.is_dir() and _has_any(p, _PROCESSOR_FILES):
                return str(p)
    if root:
        base = Path(root)
        cands = [p for p in base.rglob("*") if p.is_dir() and _has_any(p, _PROCESSOR_FILES)]
        if cands:
            sel = max(cands, key=lambda p: p.stat().st_mtime)
            return str(sel)
    return None


def _find_lora_dir(root: Optional[str]) -> Optional[str]:
    """
    Znajdź katalog z adapterem LoRA.
    Warunki: jest adapter_config.json + przynajmniej jeden plik wag *.bin/*.safetensors (również shardy).
    Preferencje: lora_adapters_last -> lora_adapters -> root -> skan w głąb.
    """
    if not root:
        return None
    base = Path(root)
    if not base.is_dir():
        return None

    def _ok(p: Path) -> bool:
        if not (p / "adapter_config.json").exists():
            return False
        if _has_any_glob(p, _LORA_PATTERNS):
            return True
        if list(p.glob("*.safetensors")) or list(p.glob("*.bin")):
            return True
        return False

    for sub in ("lora_adapters_last", "lora_adapters"):
        cand = base / sub
        if cand.is_dir() and _ok(cand):
            return str(cand)

    if _ok(base):
        return str(base)

    candidates: List[Path] = []
    for p in base.rglob("*"):
        try:
            if p.is_dir() and _ok(p):
                candidates.append(p)
        except Exception:
            pass
    if candidates:
        sel = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(sel)
    return None


def _load_processor(dir_with_processor: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(dir_with_processor)


def _load_model(dir_with_weights: str, device: str) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(dir_with_weights)
    model.to(device)
    model.eval()
    return model


def _apply_lora(model, lora_dir: str):
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("Wybrano wariant LoRA, ale brak pakietu 'peft' (pip install peft).") from e
    peft_model = PeftModel.from_pretrained(model, lora_dir)
    try:
        model = peft_model.merge_and_unload()
    except Exception:
        model = peft_model
    return model


# ========================= Public API =========================

def load_model(
    *,
    variant: str,
    mode: str = "transcribe",
    language: str = "pl",
    device_str: str = "auto",
    base_dir: Optional[str] = None,       # np. whisper-small-pl-fullft / whisper-small-local
    lora_out_dir: Optional[str] = None,   # np. whisper-small-pl-lora
    merged_dir: Optional[str] = None,     # np. whisper-small-pl-merged
) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration, str, Dict]:
    """
    Zwraca: (processor, model, device, base_dict)
    Obsługiwane: small-local | small-lora | small-merged | small-full
    """
    device = _pick_device(device_str)
    v = (variant or "").strip().lower()
    if v not in {"small-local", "small-lora", "small-merged", "small-full"}:
        raise RuntimeError(f"Nieznany wariant modelu: {variant}")

    base_dir = _exists(base_dir)
    merged_dir = _exists(merged_dir)
    lora_out_dir = _exists(lora_out_dir)

    base_model_dir         = _find_model_dir(base_dir)
    base_processor_dir     = _find_processor_dir(base_dir)
    merged_model_dir       = _find_model_dir(merged_dir)
    merged_processor_dir   = _find_processor_dir(merged_dir)
    lora_dir_resolved      = _find_lora_dir(lora_out_dir)

    hf_small = "openai/whisper-small"

    selected_model_dir = None
    selected_processor_dir = None
    used_lora_dir = None

    if v == "small-merged":
        selected_model_dir = merged_model_dir or base_model_dir
        if selected_model_dir:
            selected_processor_dir = (
                _find_processor_dir(merged_dir, prefer=selected_model_dir)
                or _find_processor_dir(base_dir)
            ) or hf_small
            processor = _load_processor(selected_processor_dir)
            model = _load_model(selected_model_dir, device)
        else:
            selected_model_dir = hf_small
            selected_processor_dir = hf_small
            processor = _load_processor(hf_small)
            model = _load_model(hf_small, device)

    elif v == "small-full":
        selected_model_dir = base_model_dir or base_dir or hf_small
        selected_processor_dir = (_find_processor_dir(base_dir, prefer=selected_model_dir) or hf_small)
        processor = _load_processor(selected_processor_dir)
        model = _load_model(selected_model_dir, device)

    elif v == "small-lora":
        if not lora_dir_resolved:
            raise RuntimeError(
                "Wybrano LoRA, ale nie znaleziono adaptera (adapter_config.json + pliki wag *.bin/*.safetensors). "
                "Sprawdź np. 'whisper-small-pl-lora/lora_adapters_last' lub 'lora_adapters'."
            )
        used_lora_dir = lora_dir_resolved
        selected_model_dir = base_model_dir or hf_small
        selected_processor_dir = base_processor_dir or hf_small
        processor = _load_processor(selected_processor_dir)
        model = _load_model(selected_model_dir, device)
        model = _apply_lora(model, used_lora_dir)

    else:  # small-local
        selected_model_dir = base_model_dir or hf_small
        selected_processor_dir = base_processor_dir or selected_model_dir
        processor = _load_processor(selected_processor_dir)
        model = _load_model(selected_model_dir, device)

    # Debug-info: trasy wczytanych elementów (do wyświetlenia w UI)
    dbg = {
        "variant": v,
        "base_dir": base_dir,
        "merged_dir": merged_dir,
        "lora_root": lora_out_dir,
        "resolved": {
            "base_model_dir": base_model_dir,
            "base_processor_dir": base_processor_dir,
            "merged_model_dir": merged_model_dir,
            "merged_processor_dir": merged_processor_dir,
            "lora_dir": lora_dir_resolved,
        },
        "selected": {
            "model_dir": selected_model_dir,
            "processor_dir": selected_processor_dir,
            "lora_dir": used_lora_dir,
        },
        "device": device,
    }
    # log do konsoli – pomaga przy debugowaniu poza GUI
    print(f"[ASR] loaded: {dbg}")

    base = {"model": model, "task": mode, "language": language, "debug": dbg}
    return processor, model, device, base


@torch.no_grad()
def transcribe_chunk(
    audio_f32_mono: np.ndarray,
    processor: WhisperProcessor,
    base: Dict,
    device: str,
) -> str:
    """Prosta transkrypcja kawałka audio (float32 mono @ 16k)."""
    model: WhisperForConditionalGeneration = base["model"]
    task: str = base.get("task", "transcribe")
    language: str = base.get("language", "pl")

    peak = float(np.max(np.abs(audio_f32_mono))) or 1.0
    audio = (audio_f32_mono / peak).astype(np.float32, copy=False)

    fe = processor.feature_extractor(
        audio, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
    )
    input_features = fe.input_features.to(device)
    attention_mask = fe.attention_mask.to(device)

    ids = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        task=task,
        language=language,
        do_sample=False,
        max_new_tokens=224,
    )
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
