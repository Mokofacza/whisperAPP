from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

def merge_lora(base_dir: Path | str, lora_dir: Path | str, out_dir: Path | str) -> str:
    base_dir = Path(base_dir)
    lora_dir = Path(lora_dir)
    out_dir = Path(out_dir)

    base = WhisperForConditionalGeneration.from_pretrained(str(base_dir))
    model = PeftModel.from_pretrained(base, str(lora_dir))
    merged = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))

    # Jeśli w katalogu LoRA jest zapisany procesor/tokenizer – skopiuj
    try:
        proc_root = lora_dir.parent  # np. whisper-small-pl-lora
        WhisperProcessor.from_pretrained(str(proc_root)).save_pretrained(str(out_dir))
    except Exception:
        pass

    return str(out_dir)
