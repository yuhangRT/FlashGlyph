"""
TrOCR evaluation backend (HuggingFace).

Uses `transformers.pipeline("image-to-text")` so you can swap checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

try:
    from transformers import pipeline
except Exception as exc:  # pragma: no cover
    pipeline = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_TROCR_MODEL_ID = "microsoft/trocr-base-printed"  # good default for English printed text


@dataclass
class TrOCRPipeline:
    pipe: object
    device: int

    def predict(self, images: List["PIL.Image.Image"], batch_size: int = 16) -> List[str]:
        outputs = self.pipe(images, batch_size=batch_size)
        preds: List[str] = []
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            for one in outputs:
                preds.append(one[0].get("generated_text", "").strip())
        elif isinstance(outputs, list):
            for one in outputs:
                preds.append(one.get("generated_text", "").strip())
        else:
            raise TypeError(f"Unexpected pipeline output type: {type(outputs)}")
        return preds


def build_pipeline(
    hf_model_id: str = DEFAULT_TROCR_MODEL_ID,
    device: Optional[int] = None,
):
    if pipeline is None:
        raise ImportError(
            "transformers is required for eval_trocr.py. "
            f"Original import error: {_IMPORT_ERROR}"
        )
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("image-to-text", model=hf_model_id, device=device)
    return TrOCRPipeline(pipe=pipe, device=device)
