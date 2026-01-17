import io
import json
import math
import mmap
import os
import random
import struct
from decimal import Decimal
from hashlib import sha1
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from dataset_util import load

try:
    import ijson  # type: ignore
except Exception:
    ijson = None

try:
    import ujson as _ujson  # type: ignore
except Exception:
    _ujson = None


STREAMING_THRESHOLD_MB = 200
_INDEX_STRUCT = struct.Struct("Q")
_INDEX_ITEM_SIZE = _INDEX_STRUCT.size
LMDB_META_KEY = b"__meta__"
LMDB_META_VERSION = 1
_FONT_CACHE = {}
_FONT_CACHE_MAX = 100


def _get_cached_font(font_path, size):
    key = (font_path, int(size))
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_FONT_CACHE) >= _FONT_CACHE_MAX:
        _FONT_CACHE.clear()
    try:
        cached = ImageFont.truetype(font_path, size=int(size))
    except OSError:
        _FONT_CACHE.clear()
        try:
            cached = ImageFont.truetype(font_path, size=int(size))
        except OSError:
            return ImageFont.load_default()
    _FONT_CACHE[key] = cached
    return cached


def _get_font_variant(font, size):
    if isinstance(font, str):
        return _get_cached_font(font, size)
    try:
        return font.font_variant(size=int(size))
    except Exception:
        pass
    font_path = getattr(font, "path", None)
    if font_path:
        return _get_cached_font(font_path, size)
    return ImageFont.load_default()


def _json_loads(raw):
    if _ujson is not None:
        return _ujson.loads(raw)
    return json.loads(raw)


def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _add_suffix(path, suffix):
    return path.with_name(path.name + suffix)


def _meta_path_for(path):
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def _make_lmdb_key(json_path, img_name):
    base = f"{Path(json_path).resolve()}::{img_name}"
    return sha1(base.encode("utf-8")).hexdigest().encode("ascii")


def _read_lmdb_meta(lmdb_path):
    try:
        import lmdb  # type: ignore
    except Exception:
        return None, "lmdb not installed"
    lmdb_path = str(lmdb_path)
    if not os.path.exists(lmdb_path):
        return None, "lmdb path not found"
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=8,
        subdir=True,
    )
    with env.begin(write=False) as txn:
        raw = txn.get(LMDB_META_KEY)
    env.close()
    if not raw:
        return None, "lmdb meta missing"
    try:
        return json.loads(raw.decode("utf-8")), None
    except Exception:
        return None, "lmdb meta invalid"


def _build_font_hint_mask(polygon, height, width, target_area_range):
    if target_area_range[0] >= 1.0 and target_area_range[1] >= 1.0:
        return None
    polygon = np.array(polygon, dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    rect_width, rect_height = size
    long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
    if long_side <= 0:
        return None
    area_ratio = random.uniform(target_area_range[0], target_area_range[1])
    long_axis_mask_length = long_side * (1 - area_ratio)
    if long_axis_mask_length <= 0:
        return None
    angle_rad = np.radians(angle)
    rect_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    if rect_width < rect_height:
        rect_vector = np.array([-rect_vector[1], rect_vector[0]])
    start_offset = random.uniform(0, max(long_side - long_axis_mask_length, 0))
    start_point = center - rect_vector * (long_side / 2 - start_offset)
    mask_center = start_point + rect_vector * (long_axis_mask_length / 2)
    mask_vector = rect_vector * (long_axis_mask_length / 2)
    short_axis_vector = np.array([-rect_vector[1], rect_vector[0]]) * (short_side / 2)
    mask_corners = np.array(
        [
            mask_center - mask_vector - short_axis_vector,
            mask_center + mask_vector - short_axis_vector,
            mask_center + mask_vector + short_axis_vector,
            mask_center - mask_vector + short_axis_vector,
        ],
        dtype=np.int32,
    )
    mask = np.ones((height, width), dtype=np.float32)
    cv2.fillPoly(mask, [mask_corners], color=0)
    return mask


def apply_font_hint_base(font_hint_base, polygon, target_area_range, prob=1.0):
    if random.random() < (1 - prob):
        return np.zeros_like(font_hint_base)
    mask = _build_font_hint_mask(polygon, font_hint_base.shape[0], font_hint_base.shape[1], target_area_range)
    if mask is None:
        return font_hint_base
    if mask.ndim == 2 and font_hint_base.ndim == 3:
        mask = mask[..., None]
    return font_hint_base * mask


class IndexFile:
    def __init__(self, path):
        self.path = Path(path)
        self._fh = None
        self._mm = None
        self._pid = None
        self._count = self.path.stat().st_size // _INDEX_ITEM_SIZE

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._count:
            raise IndexError(idx)
        mm = self._get_mmap()
        return _INDEX_STRUCT.unpack_from(mm, idx * _INDEX_ITEM_SIZE)[0]

    def _get_mmap(self):
        pid = os.getpid()
        if self._mm is None or self._pid != pid:
            if self._mm is not None:
                self._mm.close()
                if self._fh is not None:
                    self._fh.close()
            self._fh = self.path.open("rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._pid = pid
        return self._mm


def _dataset_root():
    override = os.environ.get("ANYTEXT2_DATASET_ROOT")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "dataset"


def _infer_data_roots(data_root, json_path=None):
    roots = []
    if data_root:
        roots.append(Path(data_root))

    def _existing(paths):
        seen = set()
        existing = []
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            if p.exists():
                existing.append(p)
        return existing

    existing = _existing(roots)
    if existing:
        return [str(p) for p in existing]

    base = _dataset_root()
    data_root_str = str(data_root or "")
    if "/ocr_data/" in data_root_str:
        name = data_root_str.split("/ocr_data/")[1].split("/")[0]
        candidate = base / "ocr_data" / name / name / "images"
        existing = _existing([candidate])
        if existing:
            return [str(p) for p in existing]
    if "/wukong_word/" in data_root_str:
        name = data_root_str.split("/wukong_word/")[1].split("/")[0]
        candidate = base / name / name / "imgs"
        existing = _existing([candidate])
        if existing:
            return [str(p) for p in existing]
    if "/laion_word/" in data_root_str:
        candidates = [base / "laion" / f"laion_p{i}" / "imgs" for i in range(1, 6)]
        existing = _existing(candidates)
        if existing:
            return [str(p) for p in existing]

    if json_path:
        json_path = Path(json_path)
        rel_candidate = (json_path.parent / (data_root or "")).resolve()
        existing = _existing([rel_candidate])
        if existing:
            return [str(p) for p in existing]

    return [str(p) for p in roots if p] if roots else []


class JsonlIndex:
    def __init__(
        self,
        json_path,
        wm_thresh=1.0,
        force_streaming=True,
        threshold_mb=STREAMING_THRESHOLD_MB,
        cache_dir=None,
    ):
        self.json_path = Path(json_path)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.wm_thresh = wm_thresh
        self.threshold_mb = threshold_mb
        self.force_streaming = force_streaming
        self._fh = None
        self._pid = None

        if self.json_path.suffix == ".jsonl":
            self.jsonl_path = self._cache_path(self.json_path)
        else:
            self.jsonl_path = self._cache_path(self.json_path.with_suffix(".jsonl"))

        self.meta_path = self._cache_path(_meta_path_for(self.json_path))
        self.idx_path = _add_suffix(self.jsonl_path, ".idx")

        self._ensure_cache()
        self.data_root = self._load_meta()
        self._offsets = self._load_or_build_index()

        if self.wm_thresh < 1.0:
            self._offsets = self._load_or_build_filtered_index()

    def _ensure_cache(self):
        if self.json_path.suffix == ".jsonl":
            if not self.meta_path.exists():
                raise FileNotFoundError(f"Missing meta file for JSONL: {self.meta_path}")
            return

        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

        if not self.force_streaming and self.json_path.stat().st_size < self.threshold_mb * 1024 * 1024:
            return

        if ijson is None:
            raise RuntimeError("ijson is required for streaming JSON parsing.")

        if self._cache_is_fresh():
            return

        data_root = self._read_data_root()
        count = self._build_jsonl()
        self._write_meta(data_root, count)

    def _cache_is_fresh(self):
        if not (self.jsonl_path.exists() and self.meta_path.exists()):
            return False
        return self.jsonl_path.stat().st_mtime >= self.json_path.stat().st_mtime

    def _read_data_root(self):
        with self.json_path.open("rb") as f:
            for value in ijson.items(f, "data_root"):
                return value
        return ""

    def _build_jsonl(self):
        count = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("rb") as f, self.jsonl_path.open("w", encoding="utf-8") as out:
            for item in ijson.items(f, "data_list.item"):
                out.write(_json_dumps(item))
                out.write("\n")
                count += 1
        return count

    def _write_meta(self, data_root, count):
        payload = {"data_root": data_root, "count": count}
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_meta(self):
        with self.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("data_root", "")

    def _load_or_build_index(self):
        if self.idx_path.exists() and self.idx_path.stat().st_mtime >= self.jsonl_path.stat().st_mtime:
            return self._load_index(self.idx_path)
        self._build_index(self.idx_path)
        return self._load_index(self.idx_path)

    def _build_index(self, path):
        offset = 0
        with self.jsonl_path.open("rb") as f, path.open("wb") as out:
            for line in f:
                out.write(_INDEX_STRUCT.pack(offset))
                offset += len(line)

    def _load_index(self, path):
        return IndexFile(path)

    def _filtered_index_path(self):
        key = str(self.wm_thresh).replace(".", "_")
        return _add_suffix(self.jsonl_path, f".wm{key}.idx")

    def _cache_path(self, path):
        if not self.cache_dir:
            return path
        digest = sha1(str(path).encode("utf-8")).hexdigest()[:10]
        filename = f"{path.name}.{digest}"
        return self.cache_dir / filename

    def _load_or_build_filtered_index(self):
        filtered_path = self._filtered_index_path()
        if filtered_path.exists() and filtered_path.stat().st_mtime >= self.jsonl_path.stat().st_mtime:
            return self._load_index(filtered_path)

        offset = 0
        with self.jsonl_path.open("rb") as f, filtered_path.open("wb") as out:
            for line in f:
                if not line.strip():
                    offset += len(line)
                    continue
                item = _json_loads(line)
                if item.get("wm_score", 0) < self.wm_thresh:
                    out.write(_INDEX_STRUCT.pack(offset))
                offset += len(line)
        return self._load_index(filtered_path)

    def _get_handle(self):
        pid = os.getpid()
        if self._fh is None or self._pid != pid:
            if self._fh is not None:
                self._fh.close()
            self._fh = self.jsonl_path.open("rb")
            self._pid = pid
        return self._fh

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._offsets):
            raise IndexError(idx)
        fh = self._get_handle()
        fh.seek(self._offsets[idx])
        line = fh.readline()
        return _json_loads(line)


def draw_glyph(font, text):
    g_size = 50
    width, height = (512, 80)
    new_font = _get_font_variant(font, g_size)
    img = Image.new(mode="1", size=(width, height), color=0)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = draw.textbbox((0, 0), text=text, font=new_font)
    text_width = right - left
    text_height = bottom - top
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, font=new_font, fill=1)
    gly_line = torch.from_numpy(np.array(img)).float() / 255.0
    return gly_line.unsqueeze(0)


def _insert_spaces(text, num_spaces):
    return (" " * num_spaces).join(text)


def draw_glyph2(font, text, polygon, color, scale=1.0, width=512, height=512, add_space=True, vert_ang=10):
    def initialize_img(w, h, s):
        return Image.new("RGB", (int(w * s), int(h * s)), "white")

    def prepare_image(img):
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img.mean(dim=0, keepdim=True)

    color = tuple(color)
    try:
        polygon = np.array(polygon, dtype=np.float32)
        enlarge_polygon = polygon * scale
        rect = cv2.minAreaRect(enlarge_polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w, h = rect[1]
        angle = rect[2]

        if angle < -45:
            angle += 90
        angle = -angle
        if w < h:
            angle += 90

        vert = False
        if (abs(angle) % 90 < vert_ang or abs(90 - abs(angle) % 90) % 90 < vert_ang):
            box_w = max(box[:, 0]) - min(box[:, 0])
            box_h = max(box[:, 1]) - min(box[:, 1])
            if box_h >= box_w:
                vert = True
                angle = 0

        img = initialize_img(width, height, scale)
        image4ratio = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(image4ratio)
        min_dim = min(w, h)
        max_dim = max(w, h)

        def adjust_font_size(min_size, max_size, text):
            while min_size < max_size:
                mid_size = (min_size + max_size) // 2
                new_font = _get_font_variant(font, int(mid_size))
                bbox = draw.textbbox((0, 0), text=text, font=new_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= max_dim and text_h <= min_dim:
                    min_size = mid_size + 1
                else:
                    max_size = mid_size
            return max_size - 1

        optimal_font_size = adjust_font_size(1, int(min_dim), text)
        new_font = _get_font_variant(font, int(optimal_font_size))

        if add_space and not vert:
            max_spaces = 50
            lo, hi = 1, max_spaces
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                text_space = _insert_spaces(text, mid)
                bbox2 = draw.textbbox((0, 0), text=text_space, font=new_font)
                text_w, text_h = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                if text_w <= max_dim and text_h <= min_dim:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best > 0:
                text = _insert_spaces(text, best)

        left, top, right, bottom = draw.textbbox((0, 0), text=text, font=new_font)
        text_width = right - left
        text_height = bottom - top

        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        if not vert:
            text_y_center = rect[0][1] - (text_height / 2)
            draw.text(
                (rect[0][0] - text_width / 2, text_y_center - top),
                text,
                font=new_font,
                fill=tuple(color) + (255,),
            )
        else:
            box_w = max(box[:, 0]) - min(box[:, 0])
            x_s = min(box[:, 0]) + box_w // 2 - text_height // 2
            y_s = min(box[:, 1])
            for c in text:
                draw.text((x_s, y_s), c, font=new_font, fill=tuple(color) + (255,))
                _, _t, _, _b = draw.textbbox((0, 0), text=c, font=new_font)
                y_s += _b - _t

        rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))
        x_offset = int((img.width - rotated_layer.width) / 2)
        y_offset = int((img.height - rotated_layer.height) / 2)
        img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
        img = img.resize((width, height))
        return prepare_image(img)
    except Exception:
        img = initialize_img(width, height, scale)
        img = img.resize((width, height))
        return prepare_image(img)


def draw_font_hint(target_img, polygon, target_area_range=(1.0, 1.0), prob=1.0, randaug=False):
    height, width, _ = target_img.shape
    img = np.zeros((height, width), dtype=np.uint8)

    if random.random() < (1 - prob):
        return img[..., None] / 255.0, img[..., None] / 255.0

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(img, [pts], color=255)
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    rect_width, rect_height = size
    x, y, w, h = cv2.boundingRect(np.clip(polygon, 0, None))
    target_img_scaled = (target_img + 1.0) / 2.0
    cropped_ori_img = target_img_scaled[y : y + h, x : x + w]

    if randaug:
        cropped_ori_img = random_augment(cropped_ori_img, rot=(-10, 10), trans=(-10, 10), scale=(0.9, 1.1))

    cropped_gray = cv2.cvtColor((cropped_ori_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    thresholded = cv2.adaptiveThreshold(
        cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    thresholded_resized = np.zeros_like(img.squeeze())
    thresholded_resized[y : y + h, x : x + w] = (1 - thresholded / 255.0)

    area_ratio = random.uniform(target_area_range[0], target_area_range[1])
    long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
    long_axis_mask_length = long_side * (1 - area_ratio)
    angle_rad = np.radians(angle)
    rect_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    if rect_width < rect_height:
        rect_vector = np.array([-rect_vector[1], rect_vector[0]])
    start_offset = random.uniform(0, long_side - long_axis_mask_length)
    start_point = center - rect_vector * (long_side / 2 - start_offset)
    mask_center = start_point + rect_vector * (long_axis_mask_length / 2)
    mask_vector = rect_vector * (long_axis_mask_length / 2)
    short_axis_vector = np.array([-rect_vector[1], rect_vector[0]]) * (short_side / 2)
    mask_corners = np.array(
        [
            mask_center - mask_vector - short_axis_vector,
            mask_center + mask_vector - short_axis_vector,
            mask_center + mask_vector + short_axis_vector,
            mask_center - mask_vector + short_axis_vector,
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [mask_corners], color=0)
    img = img[..., None] / 255.0

    font_hint = img.squeeze() * thresholded_resized
    return font_hint[..., None], img


def random_rotate(image, angle_range):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_translate(image, translate_range):
    tx = random.uniform(translate_range[0], translate_range[1])
    ty = random.uniform(translate_range[0], translate_range[1])
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_scale(image, scale_range):
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    if scale >= 1:
        return scaled[(scaled.shape[0] - h) // 2 : (scaled.shape[0] + h) // 2,
                      (scaled.shape[1] - w) // 2 : (scaled.shape[1] + w) // 2]
    pad_h = (h - scaled.shape[0]) // 2
    pad_w = (w - scaled.shape[1]) // 2
    return cv2.copyMakeBorder(
        scaled,
        pad_h,
        h - scaled.shape[0] - pad_h,
        pad_w,
        w - scaled.shape[1] - pad_w,
        cv2.BORDER_REPLICATE,
    )


def random_augment(image, rot=(-10, 10), trans=(-5, 5), scale=(0.9, 1.1)):
    image = random_rotate(image, rot)
    image = random_translate(image, trans)
    return random_scale(image, scale)


def build_text_caption(n_lines, place_holder="*"):
    if n_lines <= 0:
        return ""
    return "Text says " + ", ".join([place_holder] * n_lines) + " . "


def rotate_point(point, center, angle):
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


def generate_random_rectangles(width, height, box_num):
    rectangles = []
    for _ in range(box_num):
        x = random.randint(0, width)
        y = random.randint(0, height)
        w = random.randint(16, 256)
        h = random.randint(16, 96)
        angle = random.randint(-45, 45)
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)
        p1 = rotate_point(p1, center, angle)
        p2 = rotate_point(p2, center, angle)
        p3 = rotate_point(p3, center, angle)
        p4 = rotate_point(p4, center, angle)
        rectangles.append((p1, p2, p3, p4))
    return rectangles


def draw_inv_mask(polygons, img_wh):
    img = np.zeros((img_wh, img_wh), dtype=np.float32)
    for p in polygons:
        pts = p.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], color=255)
    img = img[..., None]
    return img / 255.0


def draw_pos(polygon, img_wh, prob=1.0, target_area_range=(1.0, 1.0)):
    img = np.zeros((img_wh, img_wh), dtype=np.float32)
    rect = cv2.minAreaRect(polygon)
    center, size, angle = rect
    w, h = size
    small = False
    min_wh = 20 * img_wh / 512
    if w < min_wh or h < min_wh:
        small = True
    if random.random() < prob:
        pts = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
        random_value = random.random()
        kernel = np.ones((3, 3), dtype=np.uint8)
        if random_value < 0.7:
            pass
        elif random_value < 0.8:
            img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
        elif random_value < 0.9 and not small:
            img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
        elif random_value < 0.95:
            img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
        elif random_value < 1.0 and not small:
            img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
        if target_area_range[0] < 1.0 or target_area_range[1] < 1.0:
            area_ratio = random.uniform(target_area_range[0], target_area_range[1])
            long_side, short_side = max(w, h), min(w, h)
            long_axis_mask_length = long_side * (1 - area_ratio)
            angle_rad = np.radians(angle)
            rect_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            if w < h:
                rect_vector = np.array([-rect_vector[1], rect_vector[0]])
            start_offset = random.uniform(0, long_side - long_axis_mask_length)
            start_point = center - rect_vector * (long_side / 2 - start_offset)
            mask_center = start_point + rect_vector * (long_axis_mask_length / 2)
            mask_vector = rect_vector * (long_axis_mask_length / 2)
            short_axis_vector = np.array([-rect_vector[1], rect_vector[0]]) * (short_side / 2)
            mask_corners = np.array(
                [
                    mask_center - mask_vector - short_axis_vector,
                    mask_center + mask_vector - short_axis_vector,
                    mask_center + mask_vector + short_axis_vector,
                    mask_center - mask_vector + short_axis_vector,
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [mask_corners], color=0)
    img = img[..., None]
    return img / 255.0


def get_hint(positions, img_wh):
    if len(positions) == 0:
        return np.zeros((img_wh, img_wh, 1), dtype=np.float32)
    return np.sum(positions, axis=0).clip(0, 1)


class AnyTextMockDataset(Dataset):
    def __init__(
        self,
        size=100,
        resolution=512,
        max_lines=3,
        font_path="./font/Arial_Unicode.ttf",
        mask_img_prob=0.5,
        fix_masked_img_bug=True,
    ):
        self.size = size
        self.resolution = resolution
        self.max_lines = max_lines
        self.font_path = font_path
        self.font = ImageFont.truetype(font_path, size=60)
        self.mask_img_prob = mask_img_prob
        self.fix_masked_img_bug = fix_masked_img_bug

        self.sample_texts = [
            "Hello World",
            "AnyText2",
            "Text Generation",
            "Deep Learning",
            "Computer Vision",
            "LCM Distillation",
        ]
        self.sample_captions = [
            "A photo of *",
            "An image showing *",
            "Text says *",
            "Picture with words *",
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        n_lines = random.randint(1, self.max_lines)

        img = torch.rand(self.resolution, self.resolution, 3) * 2 - 1
        texts = [random.choice(self.sample_texts) for _ in range(n_lines)]
        colors = [torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32) for _ in range(n_lines)]

        glyphs = []
        gly_line = []
        positions = []
        for _ in range(n_lines):
            glyph = torch.zeros(1, self.resolution, self.resolution)
            y_start = random.randint(50, self.resolution - 100)
            y_end = y_start + random.randint(30, 60)
            x_start = random.randint(50, self.resolution - 200)
            x_end = x_start + random.randint(100, 150)
            glyph[0, y_start:y_end, x_start:x_end] = 1.0
            glyphs.append(glyph)
            gly_line.append(draw_glyph(self.font, random.choice(texts)))

            pos = torch.zeros(1, self.resolution, self.resolution)
            pos[0, y_start:y_end, x_start:x_end] = 1.0
            positions.append(pos)

        positions_np = [p.permute(1, 2, 0).numpy() for p in positions]
        hint_np = get_hint(positions_np, self.resolution)
        hint = torch.from_numpy(hint_np).permute(2, 0, 1)

        editing_mode = random.random() < self.mask_img_prob
        if editing_mode:
            mask = get_hint(positions_np, self.resolution)
            if self.fix_masked_img_bug:
                masked_img = (img.numpy() - mask * 10).clip(-1, 1)
            else:
                masked_img = img.numpy() * (1 - mask)
        else:
            if self.fix_masked_img_bug:
                masked_img = np.zeros_like(img.numpy()) - 1
            else:
                masked_img = np.zeros_like(img.numpy())

        inv_mask = torch.ones(1, self.resolution, self.resolution)
        for pos in positions:
            inv_mask = torch.where(pos > 0.5, torch.zeros_like(inv_mask), inv_mask)

        latent_size = self.resolution // 8
        masked_x = torch.zeros(latent_size, latent_size, 4)

        img_caption = random.choice(self.sample_captions).replace("*", texts[0])
        text_caption = build_text_caption(n_lines)

        return {
            "img": img,
            "hint": hint,
            "glyphs": glyphs,
            "gly_line": gly_line,
            "positions": positions,
            "masked_img": torch.from_numpy(masked_img).float(),
            "masked_x": masked_x,
            "img_caption": img_caption,
            "text_caption": text_caption,
            "texts": texts,
            "n_lines": n_lines,
            "font_hint": torch.zeros(1, self.resolution, self.resolution),
            "color": colors,
            "language": "Latin",
            "inv_mask": inv_mask,
        }


class RealAnyTextDataset(Dataset):
    def __init__(
        self,
        json_path="demodataset/annotations/demo_data.json",
        max_lines=5,
        max_chars=20,
        resolution=512,
        font_path="./font/Arial_Unicode.ttf",
        font_hint_prob=0.8,
        font_hint_area=(0.7, 1.0),
        color_prob=1.0,
        wm_thresh=1.0,
        glyph_scale=1.0,
        font_hint_randaug=True,
        mask_img_prob=0.5,
        fix_masked_img_bug=True,
        streaming=True,
        streaming_threshold_mb=STREAMING_THRESHOLD_MB,
        cache_dir=None,
        lmdb_path=None,
    ):
        self.data = None
        json_path_obj = Path(json_path)
        if not json_path_obj.is_absolute():
            json_path_obj = (Path(__file__).resolve().parent.parent / json_path_obj).resolve()
        self.json_path = str(json_path_obj)
        if streaming:
            index = JsonlIndex(
                json_path=self.json_path,
                wm_thresh=wm_thresh,
                force_streaming=True,
                threshold_mb=streaming_threshold_mb,
                cache_dir=cache_dir,
            )
            self.data_root = index.data_root
            self.data_list = index
        else:
            self.data = load(self.json_path)
            self.data_root = self.data["data_root"]
            self.data_list = [d for d in self.data["data_list"] if d.get("wm_score", 0) < wm_thresh]
        self.data_roots = _infer_data_roots(self.data_root, json_path=self.json_path)
        if self.data_roots:
            self.data_root = self.data_roots[0]

        self.font_path = font_path
        self.font = ImageFont.truetype(font_path, size=60)
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.resolution = resolution
        self.font_hint_prob = font_hint_prob
        self.font_hint_area = font_hint_area
        self.color_prob = color_prob
        self.glyph_scale = glyph_scale
        self.font_hint_randaug = font_hint_randaug
        self.mask_img_prob = mask_img_prob
        self.fix_masked_img_bug = fix_masked_img_bug

        self.lmdb_path = lmdb_path or ""
        if self.lmdb_path:
            lmdb_path_obj = Path(self.lmdb_path)
            if not lmdb_path_obj.is_absolute():
                lmdb_path_obj = (Path(__file__).resolve().parent.parent / lmdb_path_obj).resolve()
            self.lmdb_path = str(lmdb_path_obj)
        self._lmdb_env = None
        self._lmdb_pid = None
        self._lmdb_enabled = False
        self._lmdb_use_font_hint = False
        if self.lmdb_path:
            meta, err = _read_lmdb_meta(self.lmdb_path)
            if meta is None:
                print(f"[lmdb] disabled for {self.json_path}: {err}")
            else:
                expected_font = str(Path(self.font_path).resolve())
                if (
                    int(meta.get("version", -1)) == LMDB_META_VERSION
                    and int(meta.get("resolution", -1)) == int(self.resolution)
                    and int(meta.get("max_chars", -1)) == int(self.max_chars)
                    and str(meta.get("font_path", "")) == expected_font
                    and float(meta.get("glyph_scale", -1)) == float(self.glyph_scale)
                    and int(meta.get("vert_ang", -1)) == 10
                ):
                    self._lmdb_enabled = True
                    self._lmdb_use_font_hint = not self.font_hint_randaug
                else:
                    print(f"[lmdb] meta mismatch for {self.json_path}; disabled.")

    def __len__(self):
        return len(self.data_list)

    def _resolve_img_path(self, img_name):
        if self.data_roots:
            for root in self.data_roots:
                candidate = os.path.join(root, img_name)
                if os.path.exists(candidate):
                    return candidate
            return None
        return os.path.join(self.data_root, img_name)

    def _get_lmdb_env(self):
        if not self._lmdb_enabled:
            return None
        pid = os.getpid()
        if self._lmdb_env is None or self._lmdb_pid != pid:
            import lmdb  # type: ignore

            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
                subdir=True,
            )
            self._lmdb_pid = pid
        return self._lmdb_env

    def _load_lmdb_item(self, img_name):
        env = self._get_lmdb_env()
        if env is None:
            return None
        key = _make_lmdb_key(self.json_path, img_name)
        with env.begin(write=False) as txn:
            raw = txn.get(key)
        if not raw:
            return None
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        except Exception:
            return None

    def __getitem__(self, idx):
        item_dict = {}
        cur_item = self.data_list[idx]
        img_path = self._resolve_img_path(cur_item["img_name"])
        if img_path is None:
            for _ in range(4):
                alt_idx = random.randint(0, len(self.data_list) - 1)
                cur_item = self.data_list[alt_idx]
                img_path = self._resolve_img_path(cur_item["img_name"])
                if img_path is not None:
                    break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {cur_item.get('img_name')}")

        lmdb_item = None
        if self._lmdb_enabled:
            lmdb_item = self._load_lmdb_item(cur_item.get("img_name"))

        img = Image.open(img_path).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        item_dict["img"] = torch.from_numpy(img).float()

        item_dict["img_caption"] = cur_item.get("caption", "")
        item_dict["text_caption"] = ""

        annotations = cur_item.get("annotations", [])
        if len(annotations) == 0:
            annotations = [{
                "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
                "text": " ",
                "color": [500, 500, 500],
                "language": "Latin",
            }]

        if len(annotations) > self.max_lines:
            sel_idxs = random.sample(range(len(annotations)), self.max_lines)
            unsel_idxs = [i for i in range(len(annotations)) if i not in sel_idxs]
        else:
            sel_idxs = list(range(len(annotations)))
            unsel_idxs = []

        texts = []
        polygons = []
        colors = []
        languages = []

        invalid_polygons = []
        valid_sel_indices = []
        for i in sel_idxs:
            ann = annotations[i]
            if ann.get("valid", True) is False:
                invalid_polygons.append(np.array(ann["polygon"]))
                continue
            valid_sel_indices.append(i)
            polygons.append(np.array(ann["polygon"]))
            texts.append(ann["text"][: self.max_chars])
            lang = ann.get("language", "Latin")
            if lang == "Chinese":
                try:
                    from opencc import OpenCC
                    cc = OpenCC("t2s")
                    lang = "Chinese_tra" if cc.convert(ann["text"]) != ann["text"] else "Chinese_sim"
                except Exception:
                    lang = "Chinese_sim"
            languages.append(lang)

            if "color" in ann and random.random() < self.color_prob:
                color_val = np.array(ann["color"], dtype=np.float32)
                if color_val[0] < 500:
                    colors.append(color_val / 255.0)
                else:
                    colors.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            else:
                colors.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

        if len(texts) == 0:
            texts = [" "]
            polygons = [np.array([[10, 10], [100, 10], [100, 100], [10, 100]])]
            colors = [np.array([0.5, 0.5, 0.5], dtype=np.float32)]
            languages = ["Latin"]

        n_lines = len(texts)
        item_dict["text_caption"] = build_text_caption(n_lines)

        glyphs_np = lmdb_item.get("glyphs") if lmdb_item else None
        gly_line_np = lmdb_item.get("gly_line") if lmdb_item else None
        font_hint_base_np = lmdb_item.get("font_hint_base") if lmdb_item else None

        glyphs = []
        gly_line = []
        for idx, text in enumerate(texts):
            ann_idx = valid_sel_indices[idx] if idx < len(valid_sel_indices) else None
            if (
                ann_idx is not None
                and glyphs_np is not None
                and gly_line_np is not None
                and ann_idx < glyphs_np.shape[0]
                and ann_idx < gly_line_np.shape[0]
            ):
                glyph = torch.from_numpy(glyphs_np[ann_idx]).float() / 255.0
                gly_line_item = torch.from_numpy(gly_line_np[ann_idx]).float() / 255.0
                gly_line.append(gly_line_item)
                glyphs.append(glyph)
                continue
            gly_line.append(draw_glyph(self.font, text))
            glyph_color = (colors[idx] * 255).astype(np.uint8)
            glyphs.append(
                draw_glyph2(
                    self.font,
                    text,
                    polygons[idx],
                    glyph_color,
                    scale=self.glyph_scale,
                    width=self.resolution,
                    height=self.resolution,
                )
            )
        item_dict["glyphs"] = glyphs
        item_dict["gly_line"] = gly_line

        positions = []
        for polygon in polygons:
            positions.append(draw_pos(polygon, self.resolution, prob=1.0))
        item_dict["positions"] = positions

        font_hint_list = []
        if self.font_hint_prob > 0:
            for idx, polygon in enumerate(polygons):
                ann_idx = valid_sel_indices[idx] if idx < len(valid_sel_indices) else None
                if (
                    ann_idx is not None
                    and self._lmdb_use_font_hint
                    and font_hint_base_np is not None
                    and ann_idx < font_hint_base_np.shape[0]
                ):
                    base = font_hint_base_np[ann_idx].astype(np.float32) / 255.0
                    if base.ndim == 2:
                        base = base[..., None]
                    font_hint = apply_font_hint_base(
                        base, polygon, target_area_range=self.font_hint_area, prob=self.font_hint_prob
                    )
                else:
                    font_hint, _ = draw_font_hint(
                        img,
                        polygon,
                        target_area_range=self.font_hint_area,
                        prob=self.font_hint_prob,
                        randaug=self.font_hint_randaug,
                    )
                font_hint_list.append(torch.from_numpy(font_hint).permute(2, 0, 1).float())
        else:
            for _ in polygons:
                empty = np.zeros((self.resolution, self.resolution), dtype=np.float32)
                font_hint_list.append(torch.from_numpy(empty).unsqueeze(0).float())

        combined_font_hint = torch.zeros(1, self.resolution, self.resolution)
        for fh in font_hint_list:
            combined_font_hint = torch.maximum(combined_font_hint, fh)
        item_dict["font_hint"] = combined_font_hint

        hint_np = get_hint(positions, self.resolution)
        item_dict["hint"] = torch.from_numpy(hint_np).permute(2, 0, 1).float()

        if len(texts) > 0:
            for idx in unsel_idxs:
                invalid_polygons.append(np.array(annotations[idx]["polygon"]))

        editing_mode = random.random() < self.mask_img_prob
        if editing_mode:
            invalid_polygons = []
            pos_list = positions.copy()
            box_num = random.randint(0, 0)
            if box_num > 0:
                boxes = generate_random_rectangles(self.resolution, self.resolution, box_num)
                boxes = np.array(boxes)
                for i in range(box_num):
                    pos_list.append(draw_pos(boxes[i], self.resolution, prob=1.0))
            mask = get_hint(pos_list, self.resolution)
            if self.fix_masked_img_bug:
                masked_img = (img - mask * 10).clip(-1, 1)
            else:
                masked_img = img * (1 - mask)
        else:
            if self.fix_masked_img_bug:
                masked_img = np.zeros_like(img) - 1
            else:
                masked_img = np.zeros_like(img)
        item_dict["masked_img"] = torch.from_numpy(masked_img).float()

        inv_mask_np = draw_inv_mask(invalid_polygons, self.resolution)
        item_dict["inv_mask"] = torch.from_numpy(inv_mask_np).permute(2, 0, 1).float()

        latent_size = self.resolution // 8
        item_dict["masked_x"] = torch.zeros(latent_size, latent_size, 4)

        item_dict["texts"] = texts
        item_dict["language"] = languages[0] if languages else "Latin"
        item_dict["color"] = [torch.tensor(c, dtype=torch.float32) for c in colors]
        item_dict["n_lines"] = n_lines
        item_dict["positions"] = [torch.from_numpy(p).permute(2, 0, 1).float() for p in item_dict["positions"]]

        return item_dict


def collate_fn_anytext(batch):
    img = torch.stack([item["img"] for item in batch])
    masked_img = torch.stack([item["masked_img"] for item in batch])
    hint = torch.stack([item["hint"] for item in batch])
    masked_x = torch.stack([item["masked_x"] for item in batch])
    masked_x = masked_x.permute(0, 3, 1, 2).contiguous()
    inv_mask = torch.stack([item["inv_mask"] for item in batch])
    font_hint = torch.stack([item["font_hint"] for item in batch])

    max_lines = max(item["n_lines"] for item in batch)
    glyph_shape = batch[0]["glyphs"][0].shape
    gly_line_shape = batch[0]["gly_line"][0].shape
    pos_shape = batch[0]["positions"][0].shape

    glyphs_list = [[] for _ in range(max_lines)]
    gly_line_list = [[] for _ in range(max_lines)]
    positions_list = [[] for _ in range(max_lines)]
    colors_list = [[] for _ in range(max_lines)]
    texts_list = [[] for _ in range(max_lines)]

    n_lines_list = []
    img_captions = []
    text_captions = []
    languages = []

    for item in batch:
        n_lines = item["n_lines"]
        n_lines_list.append(n_lines)

        for i in range(max_lines):
            if i < len(item["glyphs"]):
                glyphs_list[i].append(item["glyphs"][i])
                gly_line_list[i].append(item["gly_line"][i])
                positions_list[i].append(item["positions"][i])
                colors_list[i].append(item["color"][i])
                texts_list[i].append(item["texts"][i])
            else:
                glyphs_list[i].append(torch.zeros(glyph_shape))
                gly_line_list[i].append(torch.zeros(gly_line_shape))
                positions_list[i].append(torch.zeros(pos_shape))
                colors_list[i].append(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32))
                texts_list[i].append("")

        img_captions.append(item["img_caption"])
        text_captions.append(item["text_caption"])
        languages.append(item["language"])

    glyphs = [torch.stack(glyphs_list[i]) for i in range(max_lines)]
    gly_line = [torch.stack(gly_line_list[i]) for i in range(max_lines)]
    positions = [torch.stack(positions_list[i]) for i in range(max_lines)]
    color = [torch.stack(colors_list[i]) for i in range(max_lines)]
    texts = [texts_list[i] for i in range(max_lines)]

    return {
        "img": img,
        "masked_img": masked_img,
        "hint": hint,
        "glyphs": glyphs,
        "gly_line": gly_line,
        "positions": positions,
        "masked_x": masked_x,
        "img_caption": img_captions,
        "text_caption": text_captions,
        "texts": texts,
        "n_lines": torch.tensor(n_lines_list),
        "font_hint": font_hint,
        "color": color,
        "language": languages,
        "inv_mask": inv_mask,
    }
