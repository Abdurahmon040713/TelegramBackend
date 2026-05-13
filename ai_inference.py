"""
AI inference abstraction layer.

Load order:
  1. ONNX Runtime  — load from ONNX_MODEL_PATH if the directory exists
  2. Transformers  — PyTorch pipeline as fallback

Both backends expose the same interface so callers never need to know which one
is running.  Import this module and call load_model() at startup; then call
analyze_batch() for inference.
"""
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "./sentiment_onnx")

_pipeline = None
_backend: str = "unavailable"


def load_model() -> bool:
    """Try ONNX first, fall back to transformers.  Returns True if any backend loaded."""
    global _pipeline, _backend

    # ── Attempt 1: ONNX Runtime ───────────────────────────────────────────────
    if os.path.isdir(ONNX_MODEL_PATH):
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer, pipeline as hf_pipeline

            logger.info("Loading ONNX model from %s …", ONNX_MODEL_PATH)
            model = ORTModelForSequenceClassification.from_pretrained(ONNX_MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH)
            _pipeline = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
            )
            _backend = "onnx"
            logger.info("AI model ready  [backend=ONNX, path=%s]", ONNX_MODEL_PATH)
            return True
        except Exception as exc:
            logger.warning("ONNX load failed — falling back to transformers: %s", exc)

    # ── Attempt 2: transformers + PyTorch ─────────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading transformers model %s …", MODEL_NAME)
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
        )
        _backend = "transformers"
        logger.info("AI model ready  [backend=transformers (PyTorch)]")
        return True
    except Exception as exc:
        logger.error("All model loading attempts failed: %s", exc)
        _backend = "unavailable"
        return False


def is_available() -> bool:
    return _pipeline is not None


def get_backend() -> str:
    return _backend


def analyze_batch(texts: List[str]) -> List[Dict]:
    """Run *texts* through the loaded pipeline.  Raises if model is not loaded."""
    if _pipeline is None:
        raise ValueError("AI model is not loaded — call load_model() at startup.")
    try:
        return _pipeline(texts, truncation=True, max_length=512)
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        raise
