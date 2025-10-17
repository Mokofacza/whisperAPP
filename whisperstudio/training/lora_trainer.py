from __future__ import annotations
import subprocess, sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "train_whisper_lora.py"

def run_lora(
    data_dir: str,
    model_dir: str,
    output_dir: str,
    language: str = "pl",
    task: str = "transcribe",
    batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 10,
    eval_interval: int = 500,
    early_stop: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    args = [
        sys.executable, str(SCRIPT),
        "--data-dir", data_dir,
        "--model-dir", model_dir,
        "--output-dir", output_dir,
        "--language", language,
        "--task", task,
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--eval-interval", str(eval_interval),
        "--early-stop", str(early_stop),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
    ]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
