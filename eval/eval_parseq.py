"""
PARSeq evaluation backend.

This file intentionally keeps the dependency surface minimal:
- Uses HuggingFace `transformers.pipeline("image-to-text")` so that you can swap
  different PARSeq checkpoints via `--hf_model_id`.
- If you have a non-HF PARSeq implementation, you can replace `build_pipeline()`.
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


DEFAULT_PARSEQ_MODEL_ID = "baudm/parseq"  # change if you use a different checkpoint


@dataclass
class ParseqPipeline:
    pipe: object
    device: int

    def predict(self, images: List["PIL.Image.Image"], batch_size: int = 16) -> List[str]:
        # transformers returns List[List[Dict]] or List[Dict] depending on version/config
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
    hf_model_id: str = DEFAULT_PARSEQ_MODEL_ID,
    device: Optional[int] = None,
):
    if pipeline is None:
        raise ImportError(
            "transformers is required for eval_parseq.py. "
            f"Original import error: {_IMPORT_ERROR}"
        )
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("image-to-text", model=hf_model_id, device=device)
    return ParseqPipeline(pipe=pipe, device=device)
