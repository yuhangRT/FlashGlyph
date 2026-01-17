#!/usr/bin/env python
# coding=utf-8

import argparse
import io
import json
import os
import sys
import multiprocessing as mp
import shutil
from hashlib import sha1
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont

try:
    import lmdb
except Exception:
    lmdb = None

try:
    import torch
except Exception:
    torch = None

from student_model_v2.dataset_anytext_v2 import (
    JsonlIndex,
    _infer_data_roots,
    draw_font_hint,
    draw_glyph,
    draw_glyph2,
)


META_KEY = b"__meta__"
META_VERSION = 1
DEFAULT_POLYGON = [[10, 10], [100, 10], [100, 100], [10, 100]]
_WORKER_CFG = {}


def make_lmdb_key(json_path, img_name):
    base = f"{Path(json_path).resolve()}::{img_name}"
    return sha1(base.encode("utf-8")).hexdigest().encode("ascii")


def expand_paths(paths, repo_root):
    expanded = []
    for entry in paths:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if p.suffix in {".list", ".txt"}:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        line_path = Path(line)
                        if not line_path.is_absolute():
                            line_path = (repo_root / line_path).resolve()
                        expanded.append(str(line_path))
            else:
                expanded.append(str(p))
    return expanded


def resolve_img_path(img_name, data_roots, data_root):
    if not img_name:
        return None
    p = Path(img_name)
    if p.is_absolute() and p.exists():
        return str(p)
    if data_roots:
        for root in data_roots:
            candidate = Path(root) / img_name
            if candidate.exists():
                return str(candidate)
        return None
    return str(Path(data_root) / img_name)


def build_annotations(item, max_chars):
    annotations = item.get("annotations", [])
    if not annotations:
        annotations = [
            {
                "polygon": DEFAULT_POLYGON,
                "text": " ",
                "color": [500, 500, 500],
                "language": "Latin",
            }
        ]
    results = []
    for ann in annotations:
        text = ann.get("text", "")
        if max_chars > 0:
            text = text[:max_chars]
        results.append(
            {
                "polygon": ann.get("polygon", DEFAULT_POLYGON),
                "text": text,
                "color": ann.get("color", [500, 500, 500]),
                "valid": ann.get("valid", True),
            }
        )
    return results


def to_uint8(array):
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def init_worker(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)
    if torch is not None and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    _WORKER_CFG["font"] = ImageFont.truetype(cfg["font_path"], size=60)


def process_item(task):
    key, img_path, annotations = task
    cfg = _WORKER_CFG
    try:
        img = Image.open(img_path).convert("RGB")
        if img.size != (cfg["resolution"], cfg["resolution"]):
            img = img.resize((cfg["resolution"], cfg["resolution"]))
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0

        glyphs = []
        gly_line = []
        font_hint_base = []

        font = cfg["font"]
        for ann in annotations:
            if ann.get("valid") is False:
                glyphs.append(np.zeros((1, cfg["resolution"], cfg["resolution"]), dtype=np.uint8))
                gly_line.append(np.zeros((1, 80, 512), dtype=np.uint8))
                font_hint_base.append(np.zeros((cfg["resolution"], cfg["resolution"], 1), dtype=np.uint8))
                continue

            polygon = np.array(ann["polygon"], dtype=np.float32)
            text = ann["text"]

            color_val = np.array(ann.get("color", [500, 500, 500]), dtype=np.float32)
            if color_val[0] < 500:
                color_val = color_val / 255.0
            else:
                color_val = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            glyph_color = (color_val * 255).astype(np.uint8)

            glyph = draw_glyph2(
                font,
                text,
                polygon,
                glyph_color,
                scale=cfg["glyph_scale"],
                width=cfg["resolution"],
                height=cfg["resolution"],
                add_space=cfg["add_space"],
                vert_ang=cfg["vert_ang"],
            )
            gly = draw_glyph(font, text)
            hint, _ = draw_font_hint(
                img_np,
                polygon,
                target_area_range=(1.0, 1.0),
                prob=1.0,
                randaug=False,
            )

            glyphs.append(to_uint8(glyph.cpu().numpy()))
            gly_line.append(to_uint8(gly.cpu().numpy()))
            font_hint_base.append(to_uint8(hint))

        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            glyphs=np.stack(glyphs, axis=0),
            gly_line=np.stack(gly_line, axis=0),
            font_hint_base=np.stack(font_hint_base, axis=0),
        )
        return key, payload.getvalue(), None
    except Exception as exc:
        return key, None, str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Build LMDB cache for AnyText2.")
    parser.add_argument("--dataset_json", nargs="+", required=True)
    parser.add_argument("--output_lmdb", required=True)
    parser.add_argument("--font_path", default="./font/Arial_Unicode.ttf")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=20)
    parser.add_argument("--glyph_scale", type=float, default=1.0)
    parser.add_argument("--add_space", action="store_true", default=True)
    parser.add_argument("--vert_ang", type=int, default=10)
    parser.add_argument("--wm_thresh", type=float, default=1.0)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--streaming_threshold_mb", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--commit_interval", type=int, default=200)
    parser.add_argument("--map_size_gb", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_tasks(json_paths, args):
    for json_path in json_paths:
        index = JsonlIndex(
            json_path=json_path,
            wm_thresh=args.wm_thresh,
            force_streaming=True,
            threshold_mb=args.streaming_threshold_mb,
            cache_dir=args.cache_dir,
        )
        data_root = index.data_root
        data_roots = _infer_data_roots(data_root, json_path=json_path)
        for i in range(len(index)):
            item = index[i]
            img_name = item.get("img_name")
            img_path = resolve_img_path(img_name, data_roots, data_root)
            if img_path is None:
                continue
            key = make_lmdb_key(json_path, img_name)
            annotations = build_annotations(item, args.max_chars)
            yield key, img_path, annotations


def main():
    if lmdb is None:
        raise RuntimeError("lmdb is required. Install with: pip install lmdb")
    if torch is None:
        raise RuntimeError("torch is required to build LMDB cache.")

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    json_paths = expand_paths(args.dataset_json, repo_root)

    output_path = Path(args.output_lmdb)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"LMDB path not empty: {output_path}. Use --overwrite to replace.")
        for child in output_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": META_VERSION,
        "resolution": args.resolution,
        "max_chars": args.max_chars,
        "font_path": str(Path(args.font_path).resolve()),
        "glyph_scale": args.glyph_scale,
        "add_space": args.add_space,
        "vert_ang": args.vert_ang,
        "wm_thresh": args.wm_thresh,
        "source_jsons": [str(Path(p).resolve()) for p in json_paths],
    }

    env = lmdb.open(
        str(output_path),
        map_size=int(args.map_size_gb) * (1024**3),
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    with env.begin(write=True) as txn:
        txn.put(META_KEY, json.dumps(meta, ensure_ascii=True).encode("utf-8"))

    cfg = {
        "resolution": args.resolution,
        "max_chars": args.max_chars,
        "glyph_scale": args.glyph_scale,
        "add_space": args.add_space,
        "vert_ang": args.vert_ang,
        "font_path": str(Path(args.font_path).resolve()),
    }

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        maxtasksperchild=200,
        initializer=init_worker,
        initargs=(cfg,),
    ) as pool:
        txn = env.begin(write=True)
        written = 0
        for key, value, error in pool.imap_unordered(process_item, iter_tasks(json_paths, args), chunksize=1):
            if error or value is None:
                continue
            txn.put(key, value)
            written += 1
            if written % args.commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()

    env.sync()
    env.close()
    print(f"LMDB cache built at {output_path}")


if __name__ == "__main__":
    main()
