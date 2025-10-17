from pathlib import Path

SR = 16000

DEFAULT_MODEL_BASE = Path("whisper-small-local").resolve()
DEFAULT_LORA_DIR   = Path("whisper-small-pl-lora").resolve()
DEFAULT_MERGED_DIR = Path("whisper-small-pl-merged").resolve()
DEFAULT_FULLFT_DIR = Path("whisper-small-pl-fullft").resolve()
