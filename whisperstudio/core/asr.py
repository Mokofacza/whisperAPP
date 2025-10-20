from __future__ import annotations
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from .constants import (
    SR,
    DEFAULT_MODEL_BASE,
    DEFAULT_LORA_DIR,
    DEFAULT_MERGED_DIR,
    DEFAULT_FULLFT_DIR,
)

def _resolve_device(device_str: str | None) -> torch.device:
    if device_str in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def _load_proc(repo_or_dir, language: str, task: str):
    try:
        return WhisperProcessor.from_pretrained(str(repo_or_dir), local_files_only=Path(str(repo_or_dir)).exists(),
                                               language=language, task=task)
    except Exception:
        return WhisperProcessor.from_pretrained(str(repo_or_dir), local_files_only=False,
                                               language=language, task=task)

def _load_whisper(repo_or_dir):
    try:
        return WhisperForConditionalGeneration.from_pretrained(str(repo_or_dir),
                                                               local_files_only=Path(str(repo_or_dir)).exists())
    except Exception:
        return WhisperForConditionalGeneration.from_pretrained(str(repo_or_dir), local_files_only=False)

@torch.no_grad()
def load_model(
    *,
    variant: str = "merged",          # "local" | "lora" | "merged" | "full"
    mode: str = "transcribe",
    language: str = "pl",
    device_str: str | None = None,
    base_dir: str | Path = DEFAULT_MODEL_BASE,
    lora_out_dir: str | Path = DEFAULT_LORA_DIR,
    merged_dir: str | Path = DEFAULT_MERGED_DIR,
):
    """
    Zwraca: processor, model (PEFT/merged/base), device, base_model (WhisperForConditionalGeneration)
    """
    device = _resolve_device(device_str)

    if variant == "local":
        processor = _load_proc(base_dir, language, mode)
        base_model = _load_whisper(base_dir).to(device).eval()
        model = base_model

    elif variant == "lora":
        lora_adapters = Path(lora_out_dir) / "lora_adapters"
        if not lora_adapters.exists():
            raise FileNotFoundError(f"Nie znaleziono adapterów LoRA: {lora_adapters}")
        processor = _load_proc(lora_out_dir, language, mode)
        base = _load_whisper(base_dir)
        base.config.forced_decoder_ids = None
        base.config.suppress_tokens = []
        try:
            base.generation_config.language = language
            base.generation_config.task = mode
        except Exception:
            pass
        model = PeftModel.from_pretrained(base, str(lora_adapters)).to(device).eval()
        base_model = model.base_model

    elif variant in ("merged", "full"):
        # merged_dir wskazuje katalog modelu scalonego lub pełnotrenowanego
        if not Path(merged_dir).exists():
            lbl = "Merged" if variant == "merged" else "Full Trained"
            raise FileNotFoundError(f"Nie znaleziono katalogu modelu ({lbl}): {merged_dir}")
        processor = _load_proc(merged_dir, language, mode)
        model = _load_whisper(merged_dir).to(device).eval()
        base_model = model

    else:
        raise ValueError(f"Nieznany wariant modelu: {variant}")

    # Ustawienia generacji (bez wymuszania)
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    try:
        base_model.generation_config.language = language
        base_model.generation_config.task = mode
    except Exception:
        pass

    return processor, model, device, base_model

@torch.no_grad()
def transcribe_chunk(audio_float_mono_16k, processor, base_model, device, max_len: int = 225) -> str:
    feats = processor(audio_float_mono_16k, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    ids = base_model.generate(feats, max_length=max_len, do_sample=False, num_beams=1)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
