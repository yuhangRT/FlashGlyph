"""
External OCR evaluation for AnyText-benchmark style JSON + generated images.

Expected file naming:
  <img_dir>/<img_name_without_ext>_<sample_idx>.jpg

Expected JSON format (AnyText-benchmark / AnyWord style):
  {
    "data_root": "...",
    "data_list": [
      {
        "img_name": "xxx.jpg",
        "caption": "...",
        "annotations": [
          {"polygon": [[x,y],...], "text": "...", "valid": true, ...},
          ...
        ]
      }, ...
    ]
  }

This script evaluates each *text line* independently (and repeats over samples).
It reports:
- Word Acc: exact string match rate
- CER: character error rate (edit distance / |gt|)
- Char Acc: 1 - CER
- WER: word error rate (token edit distance / #tokens)

You can choose a backend:
- parseq  (HF image-to-text pipeline)
- trocr   (HF image-to-text pipeline)
- parseq+trocr  (run both and macro-average metrics)

Outputs:
- prints metrics
- optionally writes a JSON report for table provenance
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# make repo root importable (so we can import cldm.recognizer)
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(REPO_ROOT))

from cldm.recognizer import crop_image  # type: ignore

from eval_parseq import DEFAULT_PARSEQ_MODEL_ID, build_pipeline as build_parseq
from eval_trocr import DEFAULT_TROCR_MODEL_ID, build_pipeline as build_trocr


def _edit_distance(a: Sequence, b: Sequence) -> int:
    # classic DP, O(len(a)*len(b)) but strings are short in benchmarks
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[m]


def _normalize_text(s: str, lowercase: bool = False, strip: bool = True) -> str:
    if s is None:
        s = ""
    if strip:
        s = s.strip()
    if lowercase:
        s = s.lower()
    return s


@dataclass
class LineSample:
    img_key: str          # img_name without extension
    sample_idx: int
    line_idx: int
    gt: str
    polygon: np.ndarray   # (N,2) int32


def load_lines(input_json: str, max_lines: int = 20) -> List[Tuple[str, List[Tuple[str, np.ndarray]]]]:
    """Return list of (img_key, [(gt_text, polygon), ...])."""
    with open(input_json, "r", encoding="utf-8") as f:
        content = json.load(f)
    data_list = content.get("data_list", content)  # some dumps may omit wrapper
    items = []
    for entry in data_list:
        img_name = entry.get("img_name", "")
        img_key = os.path.splitext(os.path.basename(img_name))[0]
        anns = entry.get("annotations", [])
        lines = []
        for ann in anns:
            if ann.get("valid", True) is False:
                continue
            poly = ann.get("polygon", [])
            if not poly:
                continue
            text = ann.get("text", "")
            poly_np = np.asarray(poly, dtype=np.int32)
            if poly_np.ndim != 2 or poly_np.shape[1] != 2:
                continue
            lines.append((text, poly_np))
            if len(lines) >= max_lines:
                break
        if not lines:
            # keep at least one dummy line to avoid shape mismatch in some pipelines
            lines = [(" ", np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32))]
        items.append((img_key, lines))
    return items


def polygon_to_mask(poly: np.ndarray, img_wh: int = 512) -> np.ndarray:
    mask = np.zeros((img_wh, img_wh), dtype=np.uint8)
    pts = poly.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], color=255)
    return mask[..., None]  # HWC


def crop_patch(img_rgb: np.ndarray, mask_hwc: np.ndarray) -> Image.Image:
    # crop_image expects torch CHW in [0,255]
    src = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    patch = crop_image(src, mask_hwc)  # CHW float
    patch = patch.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()  # HWC uint8
    return Image.fromarray(patch)


@dataclass
class Metrics:
    n: int = 0
    word_correct: int = 0
    cer_sum: float = 0.0
    cer_den: int = 0
    wer_sum: float = 0.0
    wer_den: int = 0

    def update(self, pred: str, gt: str):
        self.n += 1
        if pred == gt:
            self.word_correct += 1

        # CER
        gt_chars = list(gt)
        pred_chars = list(pred)
        ed_c = _edit_distance(pred_chars, gt_chars)
        self.cer_sum += ed_c
        self.cer_den += max(1, len(gt_chars))

        # WER
        gt_tokens = gt.split()
        pred_tokens = pred.split()
        ed_w = _edit_distance(pred_tokens, gt_tokens)
        self.wer_sum += ed_w
        self.wer_den += max(1, len(gt_tokens))

    def as_dict(self) -> Dict[str, float]:
        cer = self.cer_sum / max(1, self.cer_den)
        wer = self.wer_sum / max(1, self.wer_den)
        return {
            "lines": float(self.n),
            "word_acc": float(self.word_correct) / max(1, self.n),
            "cer": float(cer),
            "char_acc": float(1.0 - cer),
            "wer": float(wer),
        }


def evaluate_backend(
    backend: str,
    img_dir: str,
    input_json: str,
    num_samples: int,
    batch_size: int,
    lowercase: bool,
    img_wh: int,
    hf_model_id: Optional[str] = None,
    device: Optional[int] = None,
) -> Dict[str, float]:
    t0 = time.time()
    items = load_lines(input_json=input_json)
    model_id: Optional[str] = None
    if backend == "parseq":
        model_id = hf_model_id or DEFAULT_PARSEQ_MODEL_ID
        pipe = build_parseq(hf_model_id=model_id, device=device)
    elif backend == "trocr":
        model_id = hf_model_id or DEFAULT_TROCR_MODEL_ID
        pipe = build_trocr(hf_model_id=model_id, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    metrics = Metrics()
    pil_batch: List[Image.Image] = []
    gt_batch: List[str] = []
    images_found = 0
    images_missing = 0

    def flush():
        nonlocal pil_batch, gt_batch
        if not pil_batch:
            return
        preds = pipe.predict(pil_batch, batch_size=batch_size)
        for p, g in zip(preds, gt_batch):
            p2 = _normalize_text(p, lowercase=lowercase)
            g2 = _normalize_text(g, lowercase=lowercase)
            metrics.update(p2, g2)
        pil_batch = []
        gt_batch = []

    for img_key, lines in items:
        # prebuild masks once per line
        masks = [polygon_to_mask(poly, img_wh=img_wh) for _, poly in lines]
        gts = [gt for gt, _ in lines]
        for sidx in range(num_samples):
            img_path = Path(img_dir) / f"{img_key}_{sidx}.jpg"
            if not img_path.exists():
                # also try png
                img_path = Path(img_dir) / f"{img_key}_{sidx}.png"
            if not img_path.exists():
                images_missing += 1
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                images_missing += 1
                continue
            images_found += 1
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[0] != img_wh or img_rgb.shape[1] != img_wh:
                img_rgb = cv2.resize(img_rgb, (img_wh, img_wh), interpolation=cv2.INTER_LINEAR)
            for line_idx, (gt, mask_hwc) in enumerate(zip(gts, masks)):
                pil = crop_patch(img_rgb, mask_hwc)
                pil_batch.append(pil)
                gt_batch.append(gt)
                if len(pil_batch) >= batch_size * 4:
                    flush()
    flush()
    out = metrics.as_dict()
    out["backend"] = backend
    out["model_name"] = model_id or ""
    out["images_found"] = float(images_found)
    out["images_missing"] = float(images_missing)
    out["input_items"] = float(len(items))
    out["num_samples_per_input"] = float(num_samples)
    out["elapsed_sec"] = float(time.time() - t0)
    return out


def macro_average(dicts: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    avg: Dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        avg[k] = float(sum(vals) / max(1, len(vals)))
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--backend", type=str, default="parseq+trocr", choices=["parseq", "trocr", "parseq+trocr"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--img_wh", type=int, default=512)
    parser.add_argument("--device", type=int, default=None, help="transformers pipeline device (0..N-1 or -1)")
    parser.add_argument("--parseq_model_id", type=str, default=None)
    parser.add_argument("--trocr_model_id", type=str, default=None)
    parser.add_argument("--out_json", type=str, default="", help="write metrics to json for table provenance")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    reports: List[Dict[str, float]] = []
    if args.backend in ("parseq", "parseq+trocr"):
        reports.append(
            evaluate_backend(
                backend="parseq",
                img_dir=args.img_dir,
                input_json=args.input_json,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                lowercase=args.lowercase,
                img_wh=args.img_wh,
                hf_model_id=args.parseq_model_id,
                device=args.device,
            )
        )
    if args.backend in ("trocr", "parseq+trocr"):
        reports.append(
            evaluate_backend(
                backend="trocr",
                img_dir=args.img_dir,
                input_json=args.input_json,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                lowercase=args.lowercase,
                img_wh=args.img_wh,
                hf_model_id=args.trocr_model_id,
                device=args.device,
            )
        )

    if args.backend == "parseq+trocr":
        keys = ["word_acc", "char_acc", "cer", "wer"]
        avg = macro_average(reports, keys)
        print("[parseq]", json.dumps({k: reports[0][k] for k in keys}, ensure_ascii=False, indent=2))
        print("[trocr ]", json.dumps({k: reports[1][k] for k in keys}, ensure_ascii=False, indent=2))
        print("[avg  ]", json.dumps(avg, ensure_ascii=False, indent=2))
        final = {
            "timestamp_utc": started_at,
            "backend": args.backend,
            "input_json": args.input_json,
            "img_dir": args.img_dir,
            "num_samples_per_input": args.num_samples,
            "batch_size": args.batch_size,
            "lowercase": args.lowercase,
            "img_wh": args.img_wh,
            "device": args.device,
            "avg": avg,
            "parseq": reports[0],
            "trocr": reports[1],
        }
    else:
        keys = ["word_acc", "char_acc", "cer", "wer"]
        print(json.dumps({k: reports[0][k] for k in keys}, ensure_ascii=False, indent=2))
        final = {
            "timestamp_utc": started_at,
            "backend": args.backend,
            "input_json": args.input_json,
            "img_dir": args.img_dir,
            "num_samples_per_input": args.num_samples,
            "batch_size": args.batch_size,
            "lowercase": args.lowercase,
            "img_wh": args.img_wh,
            "device": args.device,
            "metrics": reports[0],
        }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
