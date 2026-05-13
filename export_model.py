#!/usr/bin/env python
"""
One-time script: export the HuggingFace sentiment model to ONNX format.

Run ONCE before starting the server on any new deployment:
    python export_model.py

The exported model is saved to ./sentiment_onnx/ (or ONNX_MODEL_PATH env var).
On subsequent starts ai_inference.py loads it automatically — no PyTorch needed
at runtime, memory footprint drops ~4×, inference is ~2× faster.

Requirements:
    pip install optimum[onnxruntime] onnxruntime
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
SAVE_PATH = os.getenv("ONNX_MODEL_PATH", "./sentiment_onnx")


def main() -> int:
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline as hf_pipeline
    except ImportError:
        logger.error(
            "optimum[onnxruntime] not found.\n"
            "Install with:  pip install 'optimum[onnxruntime]' onnxruntime"
        )
        return 1

    if os.path.isdir(SAVE_PATH) and os.listdir(SAVE_PATH):
        logger.info("ONNX model already present at %s — skipping export.", SAVE_PATH)
        logger.info("Delete the directory to force a re-export.")
        return 0

    logger.info("Exporting %s to ONNX …", MODEL_NAME)
    logger.info("This downloads ~500 MB on first run.")

    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            export=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        return 1

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    logger.info("Saved to %s", SAVE_PATH)

    # Smoke-test the exported model
    logger.info("Running smoke test …")
    try:
        pipe = hf_pipeline("text-classification", model=model, tokenizer=tokenizer)
        result = pipe(["I hate this", "I love this"], truncation=True, max_length=128)
        logger.info("Smoke test passed: %s", result)
    except Exception as exc:
        logger.warning("Smoke test failed (model may still work): %s", exc)

    logger.info(
        "\nExport complete.\n"
        "Set ONNX_MODEL_PATH=%s in your environment (or .env) "
        "so ai_inference.py picks it up on next start.",
        SAVE_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
