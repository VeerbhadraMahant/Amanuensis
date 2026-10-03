"""Merge a LoRA adapter into its base model and convert to CTranslate2 for faster-whisper (ADR-004).

Usage: uv run python -m amanuensis.training.merge_convert RUN_DIR [--config configs/training.yaml]
Writes RUN_DIR/merged_hf and RUN_DIR/ct2.
"""
import argparse
from pathlib import Path

from amanuensis.training.train_lora import TrainConfig, load_train_config


def merge_and_convert(cfg: TrainConfig, run_dir: Path) -> tuple[Path, Path]:
    import ctranslate2
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # Merge in fp32 on CPU: independent of GPU memory (the GPU may be needed for dictation).
    base = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)
    merged = PeftModel.from_pretrained(base, run_dir / "adapter").merge_and_unload()
    hf_dir, ct2_dir = run_dir / "merged_hf", run_dir / "ct2"
    merged.save_pretrained(hf_dir)
    processor = WhisperProcessor.from_pretrained(cfg.base_model, language=cfg.language, task=cfg.task)
    processor.save_pretrained(hf_dir)
    # transformers 5 writes one combined processor_config.json; the CT2 converter and faster-whisper
    # need the standalone preprocessor_config.json.
    processor.feature_extractor.save_pretrained(hf_dir)
    ctranslate2.converters.TransformersConverter(
        str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
    ).convert(str(ct2_dir), quantization=cfg.ct2_quantization, force=True)
    return hf_dir, ct2_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    a = p.parse_args()
    print(merge_and_convert(load_train_config(a.config), a.run_dir))
