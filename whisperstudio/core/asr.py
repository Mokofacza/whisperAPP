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


def _resolve_lora_adapters_dir(lora_out_dir: Path | str) -> Path:
    """
    Zwraca katalog z adapterami LoRA.
    Akceptuje zarówno `<lora_out_dir>` jak i `<lora_out_dir>/lora_adapters`.
    """
    p = Path(lora_out_dir)
    # Bezpośrednio w katalogu
    if (p / "adapter_config.json").exists():
        return p
    # W podkatalogu lora_adapters
    if (p / "lora_adapters" / "adapter_config.json").exists():
        return p / "lora_adapters"
    raise FileNotFoundError(
        f"Nie znaleziono adapterów LoRA. Oczekuję pliku 'adapter_config.json' w:\n"
        f" - {p}\n"
        f" - {p / 'lora_adapters'}"
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
    Zwraca: processor, model (Whisper lub PeftModel), device, generate_model
    - `processor`: WhisperProcessor
    - `model`: załadowany model (WhisperForConditionalGeneration lub PeftModel)
    - `device`: torch.device
    - `generate_model`: OBIEKT, na którym należy wywoływać .generate()
                        (dla LoRA = PeftModel, dla merged/HF = Whisper)
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    if use_merged:
        # HF lub scalony (merged) model – wszystko w jednym katalogu
        processor = WhisperProcessor.from_pretrained(
            str(merged_dir),
            local_files_only=True,
            language=language,
            task=mode,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(merged_dir),
            local_files_only=True,
        ).to(device).eval()

        generate_model = model  # generate wywołujemy na tym obiekcie
    else:
        # LoRA: processor musi pochodzić z modelu bazowego
        processor = WhisperProcessor.from_pretrained(
            str(base_dir),
            local_files_only=True,
            language=language,
            task=mode,
        )
        base = WhisperForConditionalGeneration.from_pretrained(
            str(base_dir),
            local_files_only=True,
        )

        # Ustawienia dekodera (często pomocne dla Whisper)
        base.config.forced_decoder_ids = None
        base.config.suppress_tokens = []
        base.generation_config.language = language
        base.generation_config.task = mode

        adapters_dir = _resolve_lora_adapters_dir(lora_out_dir)
        model = PeftModel.from_pretrained(base, str(adapters_dir))
        model = model.to(device).eval()

        # generate trzeba wykonywać na wrapperze PeftModel, żeby LoRA działała
        generate_model = model

    # Finalne ustawienia (nie zaszkodzi powtórzyć na obiekcie do generacji)
    generate_model.config.forced_decoder_ids = None
    generate_model.config.suppress_tokens = []
    generate_model.generation_config.language = language
    generate_model.generation_config.task = mode

    return processor, model, device, generate_model


@torch.no_grad()
def transcribe_chunk(audio_float_mono_16k, processor, generate_model, device, max_len: int = 225) -> str:
    """
    generate_model = obiekt zwrócony jako 4. element z load_model (Whisper lub PeftModel),
    czyli dokładnie ten, na którym trzeba wywoływać .generate().
    """
    feats = processor(
        audio_float_mono_16k,
        sampling_rate=SR,
        return_tensors="pt"
    ).input_features.to(device)

    ids = generate_model.generate(
        feats,
        max_length=max_len,
        do_sample=False,
        num_beams=1,
    )
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
