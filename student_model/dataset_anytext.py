"""
Mock Dataset for AnyText2 LCM-LoRA Training

This is a placeholder dataset that generates synthetic data matching the AnyText2 format.
Replace this with your actual dataset loader for production training.

Expected output format matches cldm/cldm.py:436-499 (get_input method)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import cv2
import os
import math
from dataset_util import load


# ============================================================================
# Helper functions copied from t3_dataset.py for real dataset processing
# ============================================================================

def random_rotate(image, angle_range):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def random_translate(image, translate_range):
    tx = random.uniform(translate_range[0], translate_range[1])
    ty = random.uniform(translate_range[0], translate_range[1])
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    translated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return translated


def random_scale(image, scale_range):
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    if scale >= 1:
        scaled = scaled[(scaled.shape[0]-h)//2: (scaled.shape[0]+h)//2, (scaled.shape[1]-w)//2: (scaled.shape[1]+w)//2]
    else:
        pad_h = (h - scaled.shape[0]) // 2
        pad_w = (w - scaled.shape[1]) // 2
        scaled = cv2.copyMakeBorder(scaled, pad_h, h - scaled.shape[0] - pad_h, pad_w, w - scaled.shape[1] - pad_w, cv2.BORDER_REPLICATE)
    return scaled


def random_augment(image, rot=(-10, 10), trans=(-5, 5), scale=(0.9, 1.1)):
    image = random_rotate(image, rot)
    image = random_translate(image, trans)
    image = random_scale(image, scale)
    return image


def insert_spaces(text, num_spaces):
    return (' ' * num_spaces).join(text)


def rotate_point(point, center, angle):
    """Rotate a point around a center by a given angle (in degrees)."""
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


def draw_glyph(font, text):
    """
    Render gly_line (simplified linear rendering).

    Args:
        font: PIL ImageFont object
        text: Text string to render

    Returns:
        gly_line: Tensor of shape (1, 80, 512) - normalized to [0, 1]
    """
    if isinstance(font, str):
        font = ImageFont.truetype(font, size=60)
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = draw.textbbox((0, 0), text=text, font=new_font)
    text_width = right - left
    text_height = bottom - top

    # Center the text
    x = (W - text_width) // 2
    y = (H - text_height) // 2

    draw.text((x, y), text, font=new_font, fill=1)

    gly_line = torch.from_numpy(np.array(img)).float() / 255.0
    gly_line = gly_line.unsqueeze(0)  # (1, H, W)
    return gly_line


def draw_glyph2(font, text, polygon, color, scale=0.7, width=512, height=512, add_space=True, vertAng=10):
    """
    Render text to polygon region with proper rotation and scaling.

    Args:
        font: PIL ImageFont object
        text: Text string to render
        polygon: numpy array of shape (4, 2) with four corner points
        color: numpy array of shape (3,) with RGB values [0-255]
        scale: Scale factor for glyph rendering
        width: Image width
        height: Image height
        add_space: Whether to add space between characters
        vertAng: Threshold angle for vertical text detection

    Returns:
        glyph: Tensor of shape (1, H, W) - normalized to [0, 1]
    """
    def initialize_img(width, height, scale):
        return Image.new("RGB", (int(width * scale), int(height * scale)), "white")

    def prepare_image(img):
        return torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1) / 255.0

    color = tuple(color)

    try:
        # Ensure polygon is correct format and type
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
        if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
            _w = max(box[:, 0]) - min(box[:, 0])
            _h = max(box[:, 1]) - min(box[:, 1])
            if _h >= _w:
                vert = True
                angle = 0

        img = initialize_img(width, height, scale)
        image4ratio = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(image4ratio)
        min_dim = min(w, h)
        max_dim = max(w, h)

        # Binary search for optimal font size
        def adjust_font_size(min_size, max_size, text):
            while min_size < max_size:
                mid_size = (min_size + max_size) // 2
                new_font = font.font_variant(size=int(mid_size))
                bbox = draw.textbbox((0, 0), text=text, font=new_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= max_dim and text_h <= min_dim:
                    min_size = mid_size + 1
                else:
                    max_size = mid_size
            return max_size - 1

        optimal_font_size = adjust_font_size(1, min_dim, text)
        new_font = font.font_variant(size=int(optimal_font_size))

        extra_space = 0
        if add_space:
            if vert:
                # Calculate total height with added space
                total_height = sum(draw.textbbox((0, 0), text=char, font=new_font)[3] -
                                   draw.textbbox((0, 0), text=char, font=new_font)[1]
                                   for char in text)
                if total_height < max_dim and len(text) > 1:
                    extra_space = (max_dim - total_height) // (len(text) - 1)
            else:
                # Handle horizontal text space addition
                for i in range(1, 100):
                    text_space = insert_spaces(text, i)
                    bbox2 = draw.textbbox((0, 0), text=text_space, font=new_font)
                    text_w, text_h = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                    if text_w > max_dim or text_h > min_dim:
                        text = insert_spaces(text, i - 1)
                        break

        left, top, right, bottom = draw.textbbox((0, 0), text=text, font=new_font)
        text_width = right - left
        text_height = bottom - top

        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        if not vert:
            text_y_center = rect[0][1] - (text_height / 2)
            draw.text((rect[0][0] - text_width / 2, text_y_center - top), text, font=new_font, fill=tuple(color)+(255,))
        else:
            x_s = min(box[:, 0]) + _w // 2 - text_height // 2
            y_s = min(box[:, 1])
            for c in text:
                draw.text((x_s, y_s), c, font=new_font, fill=tuple(color)+(255,))
                _, _t, _, _b = draw.textbbox((0, 0), text=c, font=new_font)
                char_height = _b - _t
                y_s += char_height + extra_space

        rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))
        x_offset = int((img.width - rotated_layer.width) / 2)
        y_offset = int((img.height - rotated_layer.height) / 2)
        img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)

        # Resize back to target size
        img = img.resize((width, height))

        return prepare_image(img)

    except Exception as e:
        print(f"An error occurred in draw_glyph2: {e}")
        img = initialize_img(width, height, scale)
        img = img.resize((width, height))
        return prepare_image(img)


def draw_font_hint(target_img, polygon, target_area_range=[1.0, 1.0], prob=1.0, randaug=False):
    """
    Generate font hint from image region using adaptive thresholding.

    Args:
        target_img: Image tensor/array in range [-1, 1], shape (H, W, 3)
        polygon: numpy array of shape (4, 2) with four corner points
        target_area_range: Range of area to preserve [min, max]
        prob: Probability of using font hint (1-prob = return empty)
        randaug: Whether to use random augmentation

    Returns:
        font_hint: Tensor of shape (H, W, 1) - range [0, 1]
        mask: Binary mask of shape (H, W, 1)
    """
    height, width, _ = target_img.shape
    img = np.zeros((height, width), dtype=np.uint8)

    if random.random() < (1 - prob):  # Empty font hint
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
    cropped_ori_img = target_img_scaled[y:y+h, x:x+w]
    if randaug:
        augmented_cropped = random_augment(cropped_ori_img, rot=(-10, 10), trans=(-10, 10), scale=(0.9, 1.1))
    else:
        augmented_cropped = cropped_ori_img
    augmented_cropped_gray = cv2.cvtColor((augmented_cropped * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    thresholded = cv2.adaptiveThreshold(augmented_cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresholded_resized = np.zeros_like(img.squeeze())
    thresholded_resized[y:y+h, x:x+w] = (1 - thresholded / 255.0)

    # Generate a random mask
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
    mask_corners = np.array([
        mask_center - mask_vector - short_axis_vector,
        mask_center + mask_vector - short_axis_vector,
        mask_center + mask_vector + short_axis_vector,
        mask_center - mask_vector + short_axis_vector
    ], dtype=np.int32)
    cv2.fillPoly(img, [mask_corners], color=0)
    img = img[..., None] / 255.0

    # Compute font hint
    font_hint = img.squeeze() * thresholded_resized
    return font_hint[..., None], img


# ============================================================================
# Mock Dataset (Original)
# ============================================================================

class AnyTextMockDataset(Dataset):
    """
    Mock dataset for AnyText2 LCM training.

    Generates synthetic data with the format expected by AnyText2:
    - Images with text overlays
    - Glyph images (text renderings)
    - Position masks
    - Text captions
    - Font hints (optional)
    - Color labels
    """

    def __init__(
        self,
        size=1000,
        resolution=512,
        max_lines=3,
        glyph_channels=1,
        position_channels=1,
        font_path='./font/Arial_Unicode.ttf'
    ):
        """
        Args:
            size: Number of samples in dataset
            resolution: Image resolution (assumes square images)
            max_lines: Maximum number of text lines per image
            glyph_channels: Number of channels for glyph images
            position_channels: Number of channels for position masks
            font_path: Path to font file for text rendering
        """
        self.size = size
        self.resolution = resolution
        self.max_lines = max_lines
        self.glyph_channels = glyph_channels
        self.position_channels = position_channels

        # Load font for gly_line generation (matches real dataset)
        self.font = ImageFont.truetype(font_path, size=60)

        # Sample text strings
        self.sample_texts = [
            "Hello World",
            "AnyText2",
            "Text Generation",
            "Deep Learning",
            "Computer Vision",
            "LCM Distillation",
        ]

        # Sample captions
        self.sample_captions = [
            "A photo of *",
            "An image showing *",
            "Text says *",
            "Picture with words *",
        ]

        # Sample colors (RGB, normalized 0-1)
        self.sample_colors = [
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Generate a mock sample matching AnyText2 format.

        Returns:
            Dictionary with all keys expected by ControlLDM.get_input()
        """
        # Random number of text lines (1 to max_lines)
        n_lines = random.randint(1, self.max_lines)

        # Generate base image (random background)
        img = self._generate_background()

        # Generate text data
        texts = [random.choice(self.sample_texts) for _ in range(n_lines)]
        colors = [random.choice(self.sample_colors) for _ in range(n_lines)]

        # Generate glyphs (text renderings)
        glyphs, gly_line = self._generate_glyphs(texts, n_lines)

        # Generate position masks
        positions = self._generate_positions(n_lines)

        # Generate control hint (position mask)
        hint = self._generate_hint(positions, n_lines)

        # Generate inverse mask
        inv_mask = self._generate_inv_mask(positions, n_lines)

        # Generate masked_x (for text editing/revamping)
        # IMPORTANT: masked_x should be in latent space (H/8, W/8, C), not pixel space
        latent_size = self.resolution // 8  # 512 -> 64
        masked_x = torch.randn(latent_size, latent_size, 4)  # (64, 64, 4) - NHWC format

        # Generate font hint (optional)
        font_hint = self._generate_font_hint()

        # Generate captions
        img_caption = random.choice(self.sample_captions).replace("*", texts[0])
        text_caption = random.choice(self.sample_captions)  # Keep placeholder

        # Language identifier (mock)
        language = "en" if random.random() > 0.5 else "zh"

        return {
            'img': img,  # (H, W, 3) normalized to [-1, 1]
            'hint': hint,  # (H, W, 1) position mask
            'glyphs': glyphs,  # List of (1, H, W) glyph images
            'gly_line': gly_line,  # List of (1, H, W) line renderings
            'positions': positions,  # List of (1, H, W) position masks
            'masked_x': masked_x,  # (1, H, W, 3) masked latent
            'img_caption': img_caption,  # str base caption
            'text_caption': text_caption,  # str with placeholder
            'texts': texts,  # List[str] text content per line
            'n_lines': n_lines,  # int number of lines
            'font_hint': font_hint,  # (H, W, 1) font hint image
            'color': [torch.tensor(c, dtype=torch.float32) for c in colors],  # List[Tensor]
            'language': language,  # str language code
            'inv_mask': inv_mask,  # (H, W, 1) inverse mask
        }

    def _generate_background(self):
        """Generate random background image."""
        # Create random colored background
        img = torch.rand(self.resolution, self.resolution, 3) * 2 - 1  # [-1, 1]
        return img

    def _generate_glyphs(self, texts, n_lines):
        """
        Generate glyph images (text renderings).

        Returns:
            glyphs: List of (1, H, W) tensors - binary text masks
            gly_line: List of (1, H, W) tensors - line renderings (1, 80, 512)
        """
        glyphs = []
        gly_line = []

        for i in range(n_lines):
            # Generate binary text mask (simplified: random rectangular regions)
            glyph = torch.zeros(1, self.resolution, self.resolution)

            # Random text region
            y_start = random.randint(50, self.resolution - 100)
            y_end = y_start + random.randint(30, 60)
            x_start = random.randint(50, self.resolution - 200)
            x_end = x_start + random.randint(100, 150)

            glyph[0, y_start:y_end, x_start:x_end] = 1.0

            glyphs.append(glyph)

            # Line rendering using draw_glyph() to match real dataset format
            # draw_glyph() returns (1, 80, 512) - gly_line format
            line = draw_glyph(self.font, texts[i])
            gly_line.append(line)

        return glyphs, gly_line

    def _generate_positions(self, n_lines):
        """
        Generate position masks for each text line.

        Returns:
            List of (1, H, W) position masks
        """
        positions = []

        for i in range(n_lines):
            # Generate position mask (similar to glyphs but with different format)
            pos = torch.zeros(1, self.resolution, self.resolution)

            # Random position
            y_start = random.randint(50, self.resolution - 100)
            y_end = y_start + random.randint(30, 60)
            x_start = random.randint(50, self.resolution - 200)
            x_end = x_start + random.randint(100, 150)

            pos[0, y_start:y_end, x_start:x_end] = 1.0

            positions.append(pos)

        return positions

    def _generate_hint(self, positions, n_lines):
        """
        Generate control hint by combining all positions.

        Returns:
            hint: (H, W, 1) combined position mask
        """
        hint = torch.zeros(self.resolution, self.resolution, 1)

        for pos in positions:
            # Add position to hint
            hint[:, :, 0] = torch.maximum(hint[:, :, 0], pos[0])

        return hint

    def _generate_inv_mask(self, positions, n_lines):
        """
        Generate inverse mask (1 outside text regions, 0 inside).

        Returns:
            inv_mask: (H, W, 1) inverse mask
        """
        inv_mask = torch.ones(self.resolution, self.resolution, 1)

        for pos in positions:
            # Set to 0 inside text regions
            inv_mask[:, :, 0] = torch.where(pos[0] > 0.5, 0.0, inv_mask[:, :, 0])

        return inv_mask

    def _generate_font_hint(self):
        """
        Generate font hint image (placeholder).

        Returns:
            font_hint: (1, H, W) font hint - format expected by embedding_manager
        """
        # Simplified: use random noise as font hint placeholder
        # Format must be (1, H, W) not (H, W, 1) for embedding_manager
        font_hint = torch.rand(1, self.resolution, self.resolution)
        return font_hint


def collate_fn_anytext(batch):
    """
    Custom collate function for AnyText2 dataset.

    Handles lists of tensors with varying lengths (glyphs, positions, etc.)
    """
    # Stack simple tensors
    img = torch.stack([item['img'] for item in batch])
    hint = torch.stack([item['hint'] for item in batch])
    masked_x = torch.stack([item['masked_x'] for item in batch])
    # masked_x needs to be in (B, C, H, W) format for ControlNet
    # Input is (B, H, W, C), so permute to (B, C, H, W)
    masked_x = masked_x.permute(0, 3, 1, 2).contiguous()
    inv_mask = torch.stack([item['inv_mask'] for item in batch])
    font_hint = torch.stack([item['font_hint'] for item in batch])

    # Collect lists (each item has n_lines tensors)
    # For simplicity, assume max_lines is known
    max_lines = 5  # Should match training config

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
        n_lines = item['n_lines']
        n_lines_list.append(n_lines)

        # Pad or truncate to max_lines
        for i in range(max_lines):
            if i < len(item['glyphs']):
                glyphs_list[i].append(item['glyphs'][i])
                gly_line_list[i].append(item['gly_line'][i])
                positions_list[i].append(item['positions'][i])
                colors_list[i].append(item['color'][i])
                texts_list[i].append(item['texts'][i])
            else:
                # Pad with empty tensors
                glyphs_list[i].append(torch.zeros(1, 512, 512))
                gly_line_list[i].append(torch.zeros(1, 512, 512))
                positions_list[i].append(torch.zeros(1, 512, 512))
                colors_list[i].append(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32))
                texts_list[i].append("")

        img_captions.append(item['img_caption'])
        text_captions.append(item['text_caption'])
        languages.append(item['language'])

    # Stack lists
    glyphs = [torch.stack(glyphs_list[i]) for i in range(max_lines)]
    gly_line = [torch.stack(gly_line_list[i]) for i in range(max_lines)]
    positions = [torch.stack(positions_list[i]) for i in range(max_lines)]
    color = [torch.stack(colors_list[i]) for i in range(max_lines)]
    texts = [texts_list[i] for i in range(max_lines)]

    return {
        'img': img,
        'hint': hint,
        'glyphs': glyphs,
        'gly_line': gly_line,
        'positions': positions,
        'masked_x': masked_x,
        'img_caption': img_captions,
        'text_caption': text_captions,
        'texts': texts,
        'n_lines': torch.tensor(n_lines_list),
        'font_hint': font_hint,
        'color': color,
        'language': languages,
        'inv_mask': inv_mask,
    }


# ============================================================================
# Real Dataset (Uses demodataset with real images and annotations)
# ============================================================================

class RealAnyTextDataset(Dataset):
    """
    Real dataset for AnyText2 LCM training using demodataset.

    Loads real images and annotations from the demo dataset, matching the
    AnyText2 format with proper text rendering and font hint generation.
    """

    def __init__(
        self,
        json_path='demodataset/annotations/demo_data.json',
        max_lines=5,
        max_chars=20,
        resolution=512,
        font_path='./font/Arial_Unicode.ttf',
        mask_img_prob=0.5,
        font_hint_prob=0.8,
        font_hint_area=[0.7, 1.0],
        color_prob=1.0,
        wm_thresh=1.0,
        glyph_scale=0.7,
        font_hint_randaug=True,
    ):
        """
        Args:
            json_path: Path to demo_data.json file
            max_lines: Maximum number of text lines per image
            max_chars: Maximum number of characters per text line
            resolution: Image resolution (assumes square images)
            font_path: Path to font file for text rendering
            mask_img_prob: Probability of masking the image (for text editing)
            font_hint_prob: Probability of using font hints
            font_hint_area: Range of area to preserve for font hints [min, max]
            color_prob: Probability of using color labels
            wm_thresh: Watermark filtering threshold (1.0 = filter all)
            glyph_scale: Scale factor for glyph rendering
            font_hint_randaug: Whether to use random augmentation for font hints
        """
        # Load JSON
        self.data = load(json_path)
        self.data_root = self.data['data_root']
        self.data_list = self.data['data_list']

        # Filter watermark samples
        self.data_list = [
            d for d in self.data_list
            if d.get('wm_score', 0) < wm_thresh
        ]

        # Load font
        self.font = ImageFont.truetype(font_path, size=60)

        # Store parameters
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.resolution = resolution
        self.mask_img_prob = mask_img_prob
        self.font_hint_prob = font_hint_prob
        self.font_hint_area = font_hint_area
        self.color_prob = color_prob
        self.glyph_scale = glyph_scale
        self.font_hint_randaug = font_hint_randaug

        # Sample texts for text_caption generation
        self.sample_texts = [
            "Hello World",
            "AnyText2",
            "Text Generation",
            "Deep Learning",
            "Computer Vision",
            "LCM Distillation",
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load a real sample from demodataset.

        Returns:
            Dictionary with all keys expected by ControlLDM.get_input()
        """
        item_dict = {}
        cur_item = self.data_list[idx]

        # 1. Load and preprocess image
        img_path = os.path.join(self.data_root, cur_item['img_name'])
        img = Image.open(img_path).convert('RGB')
        if img.size[0] != self.resolution:
            img = img.resize((self.resolution, self.resolution))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        # Convert to torch tensor for collate_fn
        item_dict['img'] = torch.from_numpy(img).float()

        # 2. Caption
        item_dict['img_caption'] = cur_item.get('caption', '')
        item_dict['text_caption'] = ''

        # 3. Process annotations
        annotations = cur_item.get('annotations', [])
        if len(annotations) == 0:
            # Empty text annotation
            annotations = [{
                'polygon': [[10, 10], [100, 10], [100, 100], [10, 100]],
                'text': ' ',
                'color': [500, 500, 500],  # Use special value for no color
                'language': 'Latin'
            }]

        # 4. Randomly select max_lines text annotations
        if len(annotations) > self.max_lines:
            sel_idxs = random.sample(range(len(annotations)), self.max_lines)
        else:
            sel_idxs = range(len(annotations))

        # 5. Extract fields from selected annotations
        texts = []
        polygons = []
        colors = []
        languages = []

        for i in sel_idxs:
            ann = annotations[i]
            # Check valid field
            if ann.get('valid', True) == False:
                continue
            polygons.append(np.array(ann['polygon']))
            texts.append(ann['text'][:self.max_chars])
            lang = ann.get('language', 'Latin')

            # Handle Chinese language conversion
            if lang == 'Chinese':
                try:
                    from opencc import OpenCC
                    cc = OpenCC('t2s')
                    lang = 'Chinese_tra' if cc.convert(ann['text']) != ann['text'] else 'Chinese_sim'
                except:
                    lang = 'Chinese_sim'  # Fallback if opencc not available
            languages.append(lang)

            # Handle color (500 means no color specified)
            if 'color' in ann and random.random() < self.color_prob:
                # Normalize from [0, 255] to [0, 1] if color is specified
                color_val = np.array(ann['color'])
                if color_val[0] < 500:  # Valid color
                    colors.append(color_val / 255.0)  # Normalize to [0, 1]
                else:
                    colors.append(np.array([0.5, 0.5, 0.5]))  # Default gray
            else:
                colors.append(np.array([0.5, 0.5, 0.5]))  # Default gray

        # Ensure at least one text
        if len(texts) == 0:
            texts = [' ']
            polygons.append(np.array([[10, 10], [100, 10], [100, 100], [10, 100]]))
            colors.append(np.array([0.5, 0.5, 0.5]))
            languages.append('Latin')

        # 6. Update text_caption
        n_lines = len(texts)
        item_dict['text_caption'] = f"Text says {', '.join(['*']*n_lines)} . "

        # 7. Render glyphs and gly_line
        item_dict['glyphs'] = []
        item_dict['gly_line'] = []
        for idx, text in enumerate(texts):
            gly_line = draw_glyph(self.font, text)
            # Convert color back to [0, 255] for draw_glyph2
            glyph_color = (colors[idx] * 255).astype(np.uint8)
            glyphs = draw_glyph2(
                self.font, text, polygons[idx], glyph_color,
                scale=self.glyph_scale,
                width=self.resolution,
                height=self.resolution
            )
            # draw_glyph2 returns (3, H, W), need to convert to (1, H, W)
            # Take the mean across RGB channels to get grayscale
            if glyphs.shape[0] == 3:
                glyphs = glyphs.mean(dim=0, keepdim=True)
            item_dict['glyphs'].append(glyphs)
            item_dict['gly_line'].append(gly_line)

        # 8. Generate position masks
        item_dict['positions'] = []
        for polygon in polygons:
            h, w = self.resolution, self.resolution
            pos = np.zeros((h, w), dtype=np.float32)
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(pos, [pts], color=1.0)
            # Convert to torch tensor (1, H, W)
            item_dict['positions'].append(torch.from_numpy(pos[None, ...]).float())

        # 9. Generate font_hint
        item_dict['font_hint'] = []
        if self.font_hint_prob > 0:
            for polygon in polygons:
                font_hint, _ = draw_font_hint(
                    img,  # img is already numpy array in [-1, 1] range
                    polygon,
                    target_area_range=self.font_hint_area,
                    prob=self.font_hint_prob,
                    randaug=self.font_hint_randaug
                )
                # font_hint is (H, W, 1) numpy array, convert to tensor
                item_dict['font_hint'].append(torch.from_numpy(font_hint).permute(2, 0, 1).float())
        else:
            # Empty font_hint
            for _ in polygons:
                empty = np.zeros((self.resolution, self.resolution), dtype=np.float32)
                item_dict['font_hint'].append(torch.from_numpy(empty).unsqueeze(0).float())

        # 10. Combine position masks into hint
        hint = np.zeros((self.resolution, self.resolution, 1), dtype=np.float32)
        for pos in item_dict['positions']:
            # pos is now torch tensor, convert to numpy for maximum operation
            pos_np = pos[0].numpy() if isinstance(pos[0], torch.Tensor) else pos[0]
            hint[:, :, 0] = np.maximum(hint[:, :, 0], pos_np)
        item_dict['hint'] = torch.from_numpy(hint).float()  # Convert to torch tensor

        # 11. Generate inv_mask and masked_x
        inv_mask = np.ones((self.resolution, self.resolution, 1), dtype=np.float32)
        for pos in item_dict['positions']:
            # pos is now torch tensor, convert to numpy for where operation
            pos_np = pos[0].numpy() if isinstance(pos[0], torch.Tensor) else pos[0]
            # Set to 0 inside text regions
            inv_mask[:, :, 0] = np.where(pos_np > 0.5, 0.0, inv_mask[:, :, 0])
        item_dict['inv_mask'] = torch.from_numpy(inv_mask).float()  # Convert to torch tensor

        # Generate masked_x (for text editing/revamping)
        # IMPORTANT: masked_x should be in latent space (H/8, W/8, C), not pixel space
        latent_size = self.resolution // 8  # 512 -> 64
        masked_x = torch.randn(latent_size, latent_size, 4)  # (64, 64, 4) - NHWC format
        item_dict['masked_x'] = masked_x

        # 12. Other fields
        item_dict['texts'] = texts
        item_dict['language'] = languages[0] if languages else 'Latin'
        # Convert color list from numpy to torch tensors
        item_dict['color'] = [torch.tensor(c, dtype=torch.float32) for c in colors]
        item_dict['n_lines'] = n_lines

        # Combine font_hint into single tensor (for batching)
        # The collate function expects a single (1, H, W) tensor
        if len(item_dict['font_hint']) > 0:
            # Combine all font hints by taking maximum
            combined_font_hint = torch.zeros(1, self.resolution, self.resolution)
            for fh in item_dict['font_hint']:
                # fh is (1, H, W), take maximum
                combined_font_hint = torch.maximum(combined_font_hint, fh)
            item_dict['font_hint'] = combined_font_hint
        else:
            # Empty font hint
            item_dict['font_hint'] = torch.zeros(1, self.resolution, self.resolution)

        return item_dict


# Example usage
if __name__ == "__main__":
    import sys

    # Test both datasets
    if len(sys.argv) > 1 and sys.argv[1] == '--real':
        print("Testing RealAnyTextDataset...")
        dataset = RealAnyTextDataset(
            json_path='demodataset/annotations/demo_data.json',
            resolution=512
        )
        print(f"Total samples: {len(dataset)}")

        # Test single sample
        sample = dataset[0]
        print("\nSample fields:")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
    else:
        # Create dataset
        dataset = AnyTextMockDataset(size=100, resolution=512)

    # Test collate function
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_anytext
    )

    # Get a batch
    batch = next(iter(dataloader))

    print("Mock dataset test successful!")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"img shape: {batch['img'].shape}")
    print(f"hint shape: {batch['hint'].shape}")
    print(f"Number of glyphs: {len(batch['glyphs'])}")
    print(f"glyphs[0] shape: {batch['glyphs'][0].shape}")
