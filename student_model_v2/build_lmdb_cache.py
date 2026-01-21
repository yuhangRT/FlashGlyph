#!/usr/bin/env python
# coding=utf-8

import argparse
import io
import json
import os
import sys
import time
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

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from student_model_v2.dataset_anytext_v2 import (
    JsonlIndex,
    _infer_data_roots,
    draw_font_hint,
    draw_glyph,
    draw_glyph2,
)


META_KEY = b"__meta__"
META_VERSION = 2
SUPPORTED_META_VERSIONS = {1, 2}
DEFAULT_POLYGON = [[10, 10], [100, 10], [100, 100], [10, 100]]
_WORKER_CFG = {}


def _coerce_positive_int(value):
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _parse_size(value):
    if isinstance(value, dict):
        w = _coerce_positive_int(value.get("width") or value.get("w"))
        h = _coerce_positive_int(value.get("height") or value.get("h"))
        if w and h:
            return w, h
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        w = _coerce_positive_int(value[0])
        h = _coerce_positive_int(value[1])
        if w and h:
            return w, h
    return None


def _infer_item_size(item):
    if not isinstance(item, dict):
        return None
    for key in ("img_size", "image_size", "size"):
        size = _parse_size(item.get(key))
        if size:
            return size
    for w_key, h_key in (
        ("img_width", "img_height"),
        ("image_width", "image_height"),
        ("width", "height"),
        ("img_w", "img_h"),
        ("w", "h"),
    ):
        w = _coerce_positive_int(item.get(w_key))
        h = _coerce_positive_int(item.get(h_key))
        if w and h:
            return w, h
    return None


def _as_polygon_array(polygon):
    poly = np.asarray(polygon, dtype=np.float32)
    if poly.size == 0:
        return poly.reshape(0, 2)
    if poly.ndim == 3 and poly.shape[1] == 1:
        poly = poly[:, 0, :]
    if poly.ndim == 1 and poly.size % 2 == 0:
        poly = poly.reshape(-1, 2)
    return poly


def _scale_polygon(polygon, src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    poly = _as_polygon_array(polygon)
    if poly.size == 0:
        return poly
    max_x = float(np.nanmax(poly[:, 0]))
    max_y = float(np.nanmax(poly[:, 1]))
    if max_x <= 1.5 and max_y <= 1.5:
        poly[:, 0] *= float(src_w)
        poly[:, 1] *= float(src_h)
    if src_w != dst_w or src_h != dst_h:
        poly[:, 0] *= float(dst_w) / float(src_w)
        poly[:, 1] *= float(dst_h) / float(src_h)
    poly[:, 0] = np.clip(poly[:, 0], 0, float(dst_w) - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, float(dst_h) - 1)
    return poly


def _scale_annotations(annotations, src_size, dst_size):
    if not annotations:
        return annotations
    scaled = []
    for ann in annotations:
        ann_copy = dict(ann)
        if "polygon" in ann_copy:
            ann_copy["polygon"] = _scale_polygon(ann_copy["polygon"], src_size, dst_size)
        scaled.append(ann_copy)
    return scaled


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
    if len(task) == 3:
        key, img_path, annotations = task
        item_size = None
    else:
        key, img_path, annotations, item_size = task
    cfg = _WORKER_CFG
    try:
        img = Image.open(img_path).convert("RGB")
        if cfg.get("scale_annotations", True):
            orig_w, orig_h = img.size
            src_size = item_size or (orig_w, orig_h)
            dst_size = (cfg["resolution"], cfg["resolution"])
            annotations = _scale_annotations(annotations, src_size, dst_size)
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
    parser.add_argument("--log_interval", type=int, default=30, help="Seconds between progress logs.")
    parser.add_argument("--chunksize", type=int, default=1, help="Multiprocessing chunksize.")
    parser.add_argument("--resume", action="store_true", help="Resume building into an existing LMDB.")
    parser.add_argument(
        "--allow_mismatch",
        action="store_true",
        help="Continue even if LMDB meta differs from current args (not recommended).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sources(json_paths, args):
    sources = []
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
        sources.append((json_path, index, data_root, data_roots))
    return sources


def iter_tasks(sources, args, stats, read_txn=None):
    for json_path, index, data_root, data_roots in sources:
        for i in range(len(index)):
            stats["seen"] += 1
            item = index[i]
            img_name = item.get("img_name")
            img_path = resolve_img_path(img_name, data_roots, data_root)
            if img_path is None:
                stats["missing"] += 1
                continue
            key = make_lmdb_key(json_path, img_name)
            if read_txn is not None and read_txn.get(key) is not None:
                stats["skipped"] += 1
                continue
            annotations = build_annotations(item, args.max_chars)
            item_size = _infer_item_size(item)
            yield key, img_path, annotations, item_size


def _normalize_paths(paths):
    return sorted(str(Path(p).resolve()) for p in paths)


def _load_existing_meta(env):
    with env.begin(write=False) as txn:
        raw = txn.get(META_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _validate_resume_meta(meta, args, json_paths):
    mismatches = []
    if int(meta.get("version", -1)) not in SUPPORTED_META_VERSIONS:
        mismatches.append("version")
    if int(meta.get("resolution", -1)) != int(args.resolution):
        mismatches.append("resolution")
    if int(meta.get("max_chars", -1)) != int(args.max_chars):
        mismatches.append("max_chars")
    if str(meta.get("font_path", "")) != str(Path(args.font_path).resolve()):
        mismatches.append("font_path")
    if float(meta.get("glyph_scale", -1)) != float(args.glyph_scale):
        mismatches.append("glyph_scale")
    if int(meta.get("vert_ang", -1)) != int(args.vert_ang):
        mismatches.append("vert_ang")
    meta_add_space = meta.get("add_space", None)
    if meta_add_space is None or bool(meta_add_space) != bool(args.add_space):
        mismatches.append("add_space")
    meta_wm = meta.get("wm_thresh", None)
    if meta_wm is None or float(meta_wm) != float(args.wm_thresh):
        mismatches.append("wm_thresh")
    meta_jsons = meta.get("source_jsons") or []
    if meta_jsons:
        expected = set(_normalize_paths(json_paths))
        actual = set(_normalize_paths(meta_jsons))
        if expected != actual:
            mismatches.append("source_jsons")
    return mismatches


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
    resume_mode = False
    if output_path.exists() and any(output_path.iterdir()):
        if args.overwrite:
            for child in output_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif args.resume:
            resume_mode = True
        else:
            raise RuntimeError(f"LMDB path not empty: {output_path}. Use --overwrite or --resume.")
    output_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(output_path),
        map_size=int(args.map_size_gb) * (1024**3),
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    existing_meta = None
    if resume_mode:
        existing_meta = _load_existing_meta(env)
        if existing_meta is None:
            raise RuntimeError("Resume requested but LMDB meta is missing or unreadable.")
        mismatches = _validate_resume_meta(existing_meta, args, json_paths)
        if mismatches and not args.allow_mismatch:
            raise RuntimeError(
                f"LMDB meta mismatch for resume: {', '.join(mismatches)}. "
                "Use --allow_mismatch to force resume."
            )
        if mismatches:
            print(f"[lmdb] resume with mismatches: {', '.join(mismatches)}", flush=True)
    else:
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
        with env.begin(write=True) as txn:
            txn.put(META_KEY, json.dumps(meta, ensure_ascii=True).encode("utf-8"))

    meta_version = int(existing_meta.get("version", META_VERSION)) if existing_meta else META_VERSION
    scale_annotations = meta_version >= 2
    cfg = {
        "resolution": args.resolution,
        "max_chars": args.max_chars,
        "glyph_scale": args.glyph_scale,
        "add_space": args.add_space,
        "vert_ang": args.vert_ang,
        "font_path": str(Path(args.font_path).resolve()),
        "scale_annotations": scale_annotations,
    }

    sources = build_sources(json_paths, args)
    total_est = sum(len(index) for _, index, _, _ in sources)
    stats = {"seen": 0, "missing": 0, "processed": 0, "written": 0, "errors": 0, "skipped": 0}
    data_path = output_path / "data.mdb"
    start_time = time.time()
    last_log = start_time

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        maxtasksperchild=200,
        initializer=init_worker,
        initargs=(cfg,),
    ) as pool:
        txn = env.begin(write=True)
        read_txn = env.begin(write=False) if resume_mode else None
        written = 0
        task_iter = iter_tasks(sources, args, stats, read_txn=read_txn)
        for key, value, error in pool.imap_unordered(process_item, task_iter, chunksize=args.chunksize):
            stats["processed"] += 1
            if error or value is None:
                stats["errors"] += 1
                continue
            txn.put(key, value)
            written += 1
            stats["written"] = written
            if written % args.commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
            now = time.time()
            if now - last_log >= args.log_interval:
                elapsed = max(now - start_time, 1e-6)
                seen = stats["seen"]
                rate = stats["written"] / elapsed
                data_gb = data_path.stat().st_size / (1024**3) if data_path.exists() else 0.0
                progress = (seen / total_est * 100.0) if total_est else 0.0
                eta = (total_est - seen) / (seen / elapsed) if seen else 0.0
                print(
                    f"[lmdb] {progress:.2f}% seen={seen}/{total_est} "
                    f"written={stats['written']} skipped={stats['skipped']} missing={stats['missing']} "
                    f"errors={stats['errors']} rate={rate:.2f} item/s "
                    f"data={data_gb:.2f}GB eta={eta/3600:.1f}h",
                    flush=True,
                )
                last_log = now
        txn.commit()

    env.sync()
    env.close()
    print(f"LMDB cache built at {output_path}")


if __name__ == "__main__":
    main()
