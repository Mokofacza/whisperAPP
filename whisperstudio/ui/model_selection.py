from __future__ import annotations
from typing import Literal
from ..core.constants import (
    DEFAULT_MODEL_BASE,
    DEFAULT_LORA_DIR,
    DEFAULT_MERGED_DIR,
    DEFAULT_FULLFT_DIR,
)

Variant = Literal["local", "lora", "merged", "full"]

_VARIANT: Variant = "merged"

_LABEL = {
    "local":  "Whisper Small — Local",
    "lora":   "Whisper Small — LoRA",
    "merged": "Whisper Small — Merged",
    "full":   "Whisper Small — Full Trained",
}

def set_variant(v: Variant) -> None:
    global _VARIANT
    _VARIANT = v

def get_variant() -> Variant:
    return _VARIANT

def get_label() -> str:
    return _LABEL[_VARIANT]

def get_model_args() -> dict:
    merged_dir = DEFAULT_MERGED_DIR if _VARIANT != "full" else DEFAULT_FULLFT_DIR
    return dict(
        variant=_VARIANT,
        base_dir=DEFAULT_MODEL_BASE,
        lora_dir=DEFAULT_LORA_DIR,   # używane tylko dla LoRA
        merged_dir=merged_dir,       # używane dla merged/full
    )
