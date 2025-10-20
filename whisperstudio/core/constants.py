from pathlib import Path

# Audio
SR = 16000

# Domyślne identyfikatory / ścieżki modeli.
# Mogą być zarówno lokalnym katalogiem, jak i ID repo na HF.
DEFAULT_MODEL_BASE = "openai/whisper-small"         # baza (HF id lub lokalny katalog)
DEFAULT_LORA_DIR   = "whisper-small-pl-lora"        # katalog z LoRA (opcjonalnie)
DEFAULT_MERGED_DIR = "whisper-small-pl-merged"      # katalog z modelem scalonym (opcjonalnie)
DEFAULT_FULLFT_DIR = "whisper-small-pl-fullft"      # katalog z pełnym FT (opcjonalnie)

def is_local_path(p: str | Path | None) -> bool:
    if p is None:
        return False
    try:
        return Path(p).exists()
    except Exception:
        return False
