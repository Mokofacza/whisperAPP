from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from .constants import (
    SR,
    DEFAULT_MODEL_BASE,
    DEFAULT_LORA_DIR,
    DEFAULT_MERGED_DIR,
)

@torch.no_grad()
def load_model(
    mode: str = "transcribe",
    language: str = "pl",
    device_str: str | None = None,
    use_merged: bool = True,
    base_dir: Path | str = DEFAULT_MODEL_BASE,
    lora_out_dir: Path | str = DEFAULT_LORA_DIR,
    merged_dir: Path | str = DEFAULT_MERGED_DIR,
):
    """
    Zwraca: processor, model (z PEFT lub scalony), device, base_model (WhisperForConditionalGeneration)
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    if use_merged:
        processor = WhisperProcessor.from_pretrained(
            str(merged_dir), local_files_only=True,
            language=language, task=mode
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(merged_dir), local_files_only=True
        ).to(device).eval()
        base_model = model
    else:
        processor = WhisperProcessor.from_pretrained(
            str(lora_out_dir), local_files_only=True,
            language=language, task=mode
        )
        base = WhisperForConditionalGeneration.from_pretrained(
            str(base_dir), local_files_only=True
        )
        base.config.forced_decoder_ids = None
        base.config.suppress_tokens = []
        base.generation_config.language = language
        base.generation_config.task = mode

        model = PeftModel.from_pretrained(base, str(Path(lora_out_dir) / "lora_adapters"))
        model = model.to(device).eval()
        base_model = model.base_model

    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    base_model.generation_config.language = language
    base_model.generation_config.task = mode

    return processor, model, device, base_model

@torch.no_grad()
def transcribe_chunk(audio_float_mono_16k, processor, base_model, device, max_len: int = 225) -> str:
    feats = processor(audio_float_mono_16k, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    ids = base_model.generate(feats, max_length=max_len, do_sample=False, num_beams=1)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
