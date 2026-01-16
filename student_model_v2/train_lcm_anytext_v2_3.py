# CUDA_VISIBLE_DEVICES=1,2 python student_model_v2/oom_guard.py --min-available-gb 4 accelerate launch --num_processes 2 student_model_v2/launch_from_yaml.py --config student_model_v2/train_config_template_v3.yaml
import argparse
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torchvision
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, PeftModel, get_peft_model
try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    Conv1D = None
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.cldm import ControlLDM
from cldm.model import create_model, load_state_dict

from student_model_v2.dataset_anytext_v2 import (
    AnyTextMockDataset,
    RealAnyTextDataset,
    collate_fn_anytext,
)
from student_model_v2.losses import MultiDomainTextLoss
from student_model_v2.lcm_utils_v2 import (
    add_noise,
    apply_cfg,
    compute_lcm_loss,
    make_lcm_schedule,
    predict_x0_from_eps,
    sample_timesteps,
)


def _worker_init_fn(worker_threads, cv2_threads, _):
    try:
        import cv2  # type: ignore
        if cv2_threads is not None:
            cv2.setNumThreads(int(cv2_threads))
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    if worker_threads:
        torch.set_num_threads(int(worker_threads))


def unwrap_controlldm(model):
    if isinstance(model, ControlLDM):
        return model
    if hasattr(model, "module"):
        return unwrap_controlldm(model.module)
    if hasattr(model, "base_model"):
        base = model.base_model
        while True:
            if isinstance(base, ControlLDM):
                return base
            if hasattr(base, "model") and isinstance(base.model, ControlLDM):
                return base.model
            if hasattr(base, "base_model"):
                base = base.base_model
                continue
            break
    raise TypeError(f"Unsupported model wrapper: {type(model)}")


class AnyText2ForwardWrapper:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.base_model = unwrap_controlldm(model)

        if hasattr(self.base_model, "control_model"):
            self.base_model.control_model.fast_control = False

        self._patch_apply_model()
        self.reset_state()

    def reset_state(self):
        self.base_model.control = None
        self.base_model.control_uncond = None
        self.base_model.is_uncond = False

    def _patch_apply_model(self):
        original_method = ControlLDM.apply_model
        model_instance = self.base_model

        def patched_apply_model(self, x_noisy, t, cond, *args, **kwargs):
            model_instance.control = None
            model_instance.control_uncond = None
            model_instance.is_uncond = False
            eps = original_method(model_instance, x_noisy, t, cond, *args, **kwargs)
            model_instance.control = None
            model_instance.control_uncond = None
            model_instance.is_uncond = False
            return eps

        import types
        model_instance.apply_model = types.MethodType(patched_apply_model, model_instance)

    def encode_text(self, batch, text_info=None):
        img_caption = [cap.replace("*", "") for cap in batch["img_caption"]]
        cond = {
            "c_crossattn": [[img_caption, batch["text_caption"]]],
            "text_info": text_info,
        }
        with torch.no_grad():
            return self.base_model.get_learned_conditioning(cond)

    def prepare_text_info(self, batch):
        return {
            "glyphs": batch["glyphs"],
            "positions": batch["positions"],
            "colors": batch["color"],
            "n_lines": batch["n_lines"],
            "language": batch["language"],
            "texts": batch["texts"],
            "img": batch["img"],
            "masked_x": batch["masked_x"],
            "gly_line": batch["gly_line"],
            "inv_mask": batch["inv_mask"],
            "font_hint": batch["font_hint"],
        }

    def forward(self, latents, t, text_emb, text_info, hint):
        self.reset_state()
        cond = {
            "c_concat": [hint],
            "c_crossattn": text_emb["c_crossattn"],
            "text_info": text_info,
        }
        return self.base_model.apply_model(latents, t, cond)


def build_lora_target_modules(model):
    target_modules = []
    for name, module in model.named_modules():
        if not (name.startswith("model.diffusion_model") or name.startswith("control_model")):
            continue
        if any(skip in name for skip in ["glyph_block", "position_block", "fuse_block_za"]):
            continue
        is_linear = isinstance(module, torch.nn.Linear)
        is_conv2d = isinstance(module, torch.nn.Conv2d)
        is_conv1d = Conv1D is not None and isinstance(module, Conv1D)

        if any(key in name for key in ["to_q", "to_k", "to_v", "to_out.0"]) and (is_linear or is_conv1d):
            target_modules.append(name)
            continue
        if "zero_convs" in name and is_conv2d:
            target_modules.append(name)
            continue

    target_modules = sorted(set(target_modules))
    return target_modules


def disable_checkpointing(model):
    if hasattr(model, "use_checkpoint"):
        model.use_checkpoint = False
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False


def compute_split_lengths(total_size, train_ratio, val_ratio):
    train_len = max(1, int(total_size * train_ratio))
    val_len = max(1, int(total_size * val_ratio))
    test_len = total_size - train_len - val_len
    if test_len < 1:
        test_len = 1
        train_len = max(1, total_size - val_len - test_len)
    if train_len + val_len + test_len != total_size:
        val_len = max(1, total_size - train_len - test_len)
        test_len = total_size - train_len - val_len
    return train_len, val_len, test_len


def build_cond_batch(batch, device, non_blocking=False):
    return {
        "img": batch["img"].to(device, non_blocking=non_blocking),
        "hint": batch["hint"].to(device, non_blocking=non_blocking),
        "glyphs": [g.to(device, non_blocking=non_blocking) for g in batch["glyphs"]],
        "gly_line": [g.to(device, non_blocking=non_blocking) for g in batch["gly_line"]],
        "positions": [p.to(device, non_blocking=non_blocking) for p in batch["positions"]],
        "masked_x": batch["masked_x"].to(device, non_blocking=non_blocking),
        "img_caption": batch["img_caption"],
        "text_caption": batch["text_caption"],
        "texts": batch["texts"],
        "n_lines": batch["n_lines"],
        "font_hint": batch["font_hint"].to(device, non_blocking=non_blocking),
        "color": [c.to(device, non_blocking=non_blocking) for c in batch["color"]],
        "language": batch["language"],
        "inv_mask": batch["inv_mask"].to(device, non_blocking=non_blocking),
    }


def build_uncond_batch(batch, device):
    batch_size = batch["img"].shape[0]
    max_lines = len(batch["texts"])

    null_glyphs = [torch.zeros_like(g, device=device) for g in batch["glyphs"]]
    null_positions = [torch.zeros_like(p, device=device) for p in batch["positions"]]
    null_hint = torch.zeros(batch["hint"].shape, device=device, dtype=batch["hint"].dtype)
    null_gly_line = [torch.zeros_like(g, device=device) for g in batch["gly_line"]]
    null_masked_x = torch.zeros(batch["masked_x"].shape, device=device, dtype=batch["masked_x"].dtype)
    null_font_hint = torch.zeros(
        batch["font_hint"].shape, device=device, dtype=batch["font_hint"].dtype
    )
    null_inv_mask = torch.ones(batch["inv_mask"].shape, device=device, dtype=batch["inv_mask"].dtype)
    null_img = torch.zeros(batch["img"].shape, device=device, dtype=batch["img"].dtype)

    null_img_caption = [""] * batch_size
    null_text_caption = [""] * batch_size
    null_texts = [[""] * batch_size for _ in range(max_lines)]
    null_n_lines = torch.zeros(batch_size, dtype=batch["n_lines"].dtype, device=device)
    null_colors = [torch.ones(batch_size, 3, device=device) * 0.5 for _ in range(max_lines)]
    null_language = [""] * batch_size

    return {
        "img": null_img,
        "hint": null_hint,
        "glyphs": null_glyphs,
        "gly_line": null_gly_line,
        "positions": null_positions,
        "masked_x": null_masked_x,
        "img_caption": null_img_caption,
        "text_caption": null_text_caption,
        "texts": null_texts,
        "n_lines": null_n_lines,
        "font_hint": null_font_hint,
        "color": null_colors,
        "language": null_language,
        "inv_mask": null_inv_mask,
    }


def make_uncond_cache_key(batch, device):
    max_lines = len(batch["texts"])
    glyph_shape = tuple(batch["glyphs"][0].shape) if batch["glyphs"] else None
    gly_line_shape = tuple(batch["gly_line"][0].shape) if batch["gly_line"] else None
    pos_shape = tuple(batch["positions"][0].shape) if batch["positions"] else None
    return (
        batch["img"].shape[0],
        max_lines,
        tuple(batch["img"].shape[1:]),
        tuple(batch["hint"].shape[1:]),
        glyph_shape,
        gly_line_shape,
        pos_shape,
        tuple(batch["font_hint"].shape[1:]),
        tuple(batch["masked_x"].shape[1:]),
        tuple(batch["inv_mask"].shape[1:]),
        str(device),
        str(batch["img"].dtype),
        str(batch["masked_x"].dtype),
    )


class UncondCache:
    def __init__(self):
        self.cache = {}

    def get(self, batch, wrapper, device):
        key = make_uncond_cache_key(batch, device)
        cached = self.cache.get(key)
        if cached is None:
            uncond_batch = build_uncond_batch(batch, device)
            text_info = wrapper.prepare_text_info(uncond_batch)
            text_emb = wrapper.encode_text(uncond_batch, text_info)
            cached = (uncond_batch, text_info, text_emb)
            self.cache[key] = cached
        return cached


def encode_img_and_masked_x(batch, wrapper, device, non_blocking=False):
    img = batch["img"]
    masked_img = batch.get("masked_img", img)
    img_nchw = img.permute(0, 3, 1, 2).to(device, non_blocking=non_blocking)
    masked_nchw = masked_img.permute(0, 3, 1, 2).to(device, non_blocking=non_blocking)
    stacked = torch.cat([img_nchw, masked_nchw], dim=0)
    latent_dist = wrapper.base_model.first_stage_model.encode(stacked)
    latents = latent_dist.sample() * wrapper.base_model.scale_factor
    latents_img, latents_masked = latents.chunk(2, dim=0)
    batch["masked_x"] = latents_masked
    return latents_img


def _ensure_nchw(tensor):
    if tensor.dim() == 3:
        if tensor.shape[0] in (1, 3):
            return tensor.unsqueeze(0)
        if tensor.shape[-1] in (1, 3):
            return tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.unsqueeze(0)
    if tensor.dim() == 4:
        if tensor.shape[1] in (1, 3):
            return tensor
        if tensor.shape[-1] in (1, 3):
            return tensor.permute(0, 3, 1, 2)
    return tensor


def _expand_to_rgb(tensor):
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    return tensor


def _to_01(tensor, assume_neg1_pos1):
    tensor = tensor.detach().float().cpu()
    if assume_neg1_pos1:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def make_preview_grid(img, masked_img, hint, pred_img, max_samples=4):
    img = _expand_to_rgb(_to_01(_ensure_nchw(img), assume_neg1_pos1=True))
    masked_img = _expand_to_rgb(_to_01(_ensure_nchw(masked_img), assume_neg1_pos1=True))
    hint = _expand_to_rgb(_to_01(_ensure_nchw(hint), assume_neg1_pos1=False))
    pred_img = _expand_to_rgb(_to_01(_ensure_nchw(pred_img), assume_neg1_pos1=True))

    n = min(max_samples, img.shape[0], masked_img.shape[0], hint.shape[0], pred_img.shape[0])
    tiles = []
    for i in range(n):
        tiles.extend([img[i], masked_img[i], hint[i], pred_img[i]])
    grid = torchvision.utils.make_grid(torch.stack(tiles), nrow=4)
    return grid


def log_train_images(step, batch, pred_x0_student, wrapper, output_dir, max_samples=4):
    if max_samples <= 0:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred_img = wrapper.base_model.decode_first_stage(pred_x0_student[:max_samples])
    grid = make_preview_grid(
        batch["img"],
        batch["masked_img"],
        batch["hint"],
        pred_img,
        max_samples=max_samples,
    )
    out_path = out_dir / f"step_{step:07d}.png"
    torchvision.utils.save_image(grid, out_path)
    return str(out_path)


def sanitize_hparams(config):
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (int, float, bool, str, torch.Tensor)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = ",".join(str(v) for v in value)
        else:
            sanitized[key] = str(value)
    return sanitized


def _parse_checkpoint_step(resume_path):
    if not resume_path:
        return None
    name = os.path.basename(os.path.normpath(resume_path))
    match = re.search(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _infer_latest_step(output_dir, image_dir):
    max_step = 0
    out_dir = Path(output_dir)
    if out_dir.exists():
        for entry in out_dir.glob("checkpoint-*"):
            step = _parse_checkpoint_step(entry.name)
            if step:
                max_step = max(max_step, step)
    img_dir = Path(image_dir)
    if img_dir.exists():
        for entry in img_dir.glob("step_*.png"):
            match = re.search(r"step_(\d+)", entry.name)
            if match:
                max_step = max(max_step, int(match.group(1)))
    return max_step or None


def warmup_cosine_scale(step, total_steps, warmup_steps, min_ratio):
    if total_steps <= 0:
        return 1.0
    warmup_steps = min(max(int(warmup_steps), 0), total_steps)
    min_ratio = max(0.0, min(float(min_ratio), 1.0))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps == warmup_steps:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA distillation for AnyText2 (v3)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--output_dir", type=str, default="student_model_v2/checkpoints")
    parser.add_argument("--dataset_json", type=str, nargs="+", default=["demodataset/annotations/demo_data.json"])
    parser.add_argument("--use_mock_dataset", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--resume_optimizer", action="store_true", default=False)
    parser.add_argument(
        "--resume_add_steps",
        type=int,
        default=0,
        help="Additional steps to run after resuming. 0 uses max_train_steps when resuming.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--loss_ffl_weight", type=float, default=0.05)
    parser.add_argument("--loss_grad_weight", type=float, default=0.05)
    parser.add_argument("--ffl_alpha", type=float, default=1.0)
    parser.add_argument("--ffl_patch_factor", type=int, default=1)
    parser.add_argument("--ffl_ave_spectrum", action="store_true", default=False)
    parser.add_argument("--ffl_log_matrix", action="store_true", default=False)
    parser.add_argument("--ffl_batch_matrix", action="store_true", default=False)
    parser.add_argument("--loss_mask_key", type=str, default="hint", choices=["hint", "positions", "inv_mask"])
    parser.add_argument("--loss_text_weight", type=float, default=5.0)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "warmup_cosine"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_min_ratio", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_epochs", type=int, default=0)
    parser.add_argument("--log_image_steps", type=int, default=0)
    parser.add_argument("--log_image_samples", type=int, default=4)
    parser.add_argument("--log_image_dir", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--eval_batches", type=int, default=1)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--eval_timeout", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--mp_context", type=str, default="", help="DataLoader multiprocessing context (spawn|forkserver).")
    parser.add_argument("--worker_num_threads", type=int, default=1)
    parser.add_argument("--cv2_num_threads", type=int, default=0)
    parser.add_argument("--allow_tf32", action="store_true", default=False)
    parser.add_argument("--cudnn_benchmark", action="store_true", default=False)
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--wm_thresh", type=float, default=1.0)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no_streaming", action="store_false", dest="streaming")
    parser.add_argument("--streaming_threshold_mb", type=int, default=200)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--auto_list_path", type=str, default="")
    args = parser.parse_args()

    if args.worker_num_threads and args.worker_num_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.worker_num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.worker_num_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.worker_num_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.worker_num_threads))

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision(args.matmul_precision)
    except Exception:
        pass

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    set_seed(args.seed)

    config_path = Path(args.config)
    ckpt_path = Path(args.teacher_ckpt)
    if not config_path.is_absolute():
        config_path = (Path(__file__).parent.parent / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).parent.parent / ckpt_path).resolve()

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.init_trackers("anytext2_lcm", config=sanitize_hparams(vars(args)))

    teacher = create_model(str(config_path))
    student = create_model(str(config_path))
    state_dict = load_state_dict(str(ckpt_path), location="cpu")
    teacher.load_state_dict(state_dict, strict=False)
    student.load_state_dict(state_dict, strict=False)
    del state_dict
    teacher.eval()
    disable_checkpointing(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    disable_checkpointing(student)
    for p in student.parameters():
        p.requires_grad = False

    target_modules = build_lora_target_modules(student)
    if len(target_modules) == 0:
        raise RuntimeError("No LoRA target modules found. Check model naming.")

    resume_path = args.resume_path.strip()
    if resume_path:
        resume_path = str((Path(__file__).parent.parent / resume_path).resolve()) if not os.path.isabs(resume_path) else resume_path
        student = PeftModel.from_pretrained(student, resume_path, is_trainable=True)
        resume_step_hint = _parse_checkpoint_step(resume_path)
        if accelerator.is_local_main_process:
            accelerator.print(f"[resume] Using LoRA weights from {resume_path}")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="DIFFUSION",
        )
        try:
            student = get_peft_model(student, lora_config)
        except Exception as exc:
            raise RuntimeError(
                "LoRA injection failed. zero_convs are Conv2d modules and must be supported. "
                "Please upgrade peft/transformers to a version that supports Conv2d LoRA."
            ) from exc
    student.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.learning_rate,
    )

    if args.use_mock_dataset:
        dataset = AnyTextMockDataset(size=1000, resolution=args.resolution)
    else:
        repo_root = Path(__file__).parent.parent

        def expand_paths(paths):
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
                                path_line = Path(line)
                                if not path_line.is_absolute():
                                    path_line = (repo_root / path_line).resolve()
                                expanded.append(str(path_line))
                    else:
                        expanded.append(str(p))
            return expanded

        json_paths = expand_paths(args.dataset_json)
        if args.auto_list_path:
            list_path = Path(args.auto_list_path)
            if not list_path.is_absolute():
                list_path = (repo_root / list_path).resolve()
            list_path.parent.mkdir(parents=True, exist_ok=True)
            with list_path.open("w", encoding="utf-8") as f:
                for p in json_paths:
                    f.write(f"{p}\n")
        cache_dir = args.cache_dir.strip() if args.cache_dir else ""
        if cache_dir:
            cache_dir = str((repo_root / cache_dir).resolve()) if not os.path.isabs(cache_dir) else cache_dir
        datasets = [
            RealAnyTextDataset(
                json_path=path,
                resolution=args.resolution,
                wm_thresh=args.wm_thresh,
                streaming=args.streaming,
                streaming_threshold_mb=args.streaming_threshold_mb,
                cache_dir=cache_dir or None,
            )
            for path in json_paths
        ]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    total_size = len(dataset)
    if total_size < 3:
        train_set = dataset
        val_set = None
        test_set = None
    else:
        train_len, val_len, test_len = compute_split_lengths(
            total_size, args.train_ratio, args.val_ratio
        )

        generator = torch.Generator().manual_seed(args.split_seed)
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len], generator=generator
        )

    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    mp_context = args.mp_context if args.num_workers > 0 and args.mp_context else None

    worker_init_fn = None
    if args.num_workers > 0:
        worker_init_fn = partial(_worker_init_fn, args.worker_num_threads, args.cv2_num_threads)
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_anytext,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_context,
        worker_init_fn=worker_init_fn,
    )
    val_loader = None
    test_loader = None
    eval_num_workers = max(0, int(args.eval_num_workers))
    eval_prefetch = args.prefetch_factor if eval_num_workers > 0 else None
    eval_mp_context = args.mp_context if eval_num_workers > 0 and args.mp_context else None
    eval_worker_init = worker_init_fn if eval_num_workers > 0 else None

    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn_anytext,
            num_workers=eval_num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=eval_prefetch,
            multiprocessing_context=eval_mp_context,
            worker_init_fn=eval_worker_init,
            timeout=args.eval_timeout,
        )
    if test_set is not None and len(test_set) > 0:
        test_loader = DataLoader(
            test_set,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn_anytext,
            num_workers=eval_num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=eval_prefetch,
            multiprocessing_context=eval_mp_context,
            worker_init_fn=eval_worker_init,
            timeout=args.eval_timeout,
        )

    student, optimizer, train_loader = accelerator.prepare(student, optimizer, train_loader)
    device = accelerator.device
    teacher.to(device)

    teacher_wrapper = AnyText2ForwardWrapper(teacher, device)
    student_wrapper = AnyText2ForwardWrapper(student, device)

    alphas_cumprod = teacher_wrapper.base_model.alphas_cumprod.to(device)
    schedule = make_lcm_schedule(args.num_inference_steps, num_train_timesteps=alphas_cumprod.shape[0])
    autocast_context = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    if args.max_epochs > 0:
        max_epochs = args.max_epochs
    else:
        max_epochs = math.ceil(args.max_train_steps / steps_per_epoch) if args.max_train_steps > 0 else 1
    total_updates = max_epochs * steps_per_epoch
    if args.max_train_steps > 0:
        total_updates = min(total_updates, args.max_train_steps)
    total_updates = max(1, total_updates)

    global_step = 0
    start_epoch = 0
    last_log_time = time.perf_counter()
    last_log_step = 0
    ema_loss = None

    non_blocking = bool(args.pin_memory)
    uncond_cache = UncondCache()
    log_image_dir = args.log_image_dir
    if not log_image_dir:
        if resume_path:
            log_image_dir = os.path.join(os.path.dirname(resume_path), "train_img")
        else:
            log_image_dir = os.path.join(args.output_dir, "train_img")

    criterion = MultiDomainTextLoss(
        ffl_weight=args.loss_ffl_weight,
        grad_weight=args.loss_grad_weight,
        ffl_alpha=args.ffl_alpha,
        ffl_patch_factor=args.ffl_patch_factor,
        ffl_ave_spectrum=args.ffl_ave_spectrum,
        ffl_log_matrix=args.ffl_log_matrix,
        ffl_batch_matrix=args.ffl_batch_matrix,
        text_weight=args.loss_text_weight,
    ).to(device)

    def get_text_mask(batch):
        key = args.loss_mask_key
        mask = None
        if key == "hint":
            mask = batch.get("hint")
        elif key == "positions":
            positions = batch.get("positions")
            if positions:
                mask = torch.stack(positions, dim=0).amax(dim=0)
        elif key == "inv_mask":
            inv_mask = batch.get("inv_mask")
            if inv_mask is not None:
                mask = 1.0 - inv_mask
        else:
            mask = batch.get("hint")
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return mask.float()

    def run_eval(split_name, loader):
        if loader is None or args.eval_batches <= 0:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            accelerator.print(f"[eval] Running {split_name} for {args.eval_batches} batches...")
        teacher_wrapper.base_model.eval()
        student_wrapper.base_model.eval()
        losses = []
        lcm_losses = []
        ffl_losses = []
        grad_losses = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= args.eval_batches:
                    break
                batch = {k: v for k, v in batch.items()}
                with autocast_context():
                    batch_size = batch["img"].shape[0]
                    latents = encode_img_and_masked_x(
                        batch, teacher_wrapper, device, non_blocking=non_blocking
                    )
                    timesteps = sample_timesteps(schedule, batch_size, device)
                    noise = torch.randn_like(latents)
                    noisy_latents = add_noise(latents, noise, timesteps, alphas_cumprod)

                    cond_batch = build_cond_batch(batch, device, non_blocking=non_blocking)
                    uncond_batch, uncond_text_info, uncond_text_emb = uncond_cache.get(
                        batch, teacher_wrapper, device
                    )
                    cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
                    cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)

                    noise_pred_cond = teacher_wrapper.forward(
                        noisy_latents, timesteps, cond_text_emb, cond_text_info, cond_batch["hint"]
                    )
                    noise_pred_uncond = teacher_wrapper.forward(
                        noisy_latents, timesteps, uncond_text_emb, uncond_text_info, uncond_batch["hint"]
                    )
                    noise_pred_teacher = apply_cfg(noise_pred_cond, noise_pred_uncond, args.cfg_scale)
                    target_x0 = predict_x0_from_eps(
                        noisy_latents, timesteps, noise_pred_teacher, alphas_cumprod
                    )

                    student_text_info = student_wrapper.prepare_text_info(cond_batch)
                    student_text_emb = student_wrapper.encode_text(cond_batch, student_text_info)
                    noise_pred_student = student_wrapper.forward(
                        noisy_latents, timesteps, student_text_emb, student_text_info, cond_batch["hint"]
                    )
                    pred_x0_student = predict_x0_from_eps(
                        noisy_latents, timesteps, noise_pred_student, alphas_cumprod
                    )
                    text_mask = get_text_mask(batch)
                    if text_mask is not None:
                        text_mask = text_mask.to(device, non_blocking=non_blocking)
                    loss, loss_dict = criterion(pred_x0_student, target_x0, mask=text_mask)
                losses.append(loss.detach().float().item())
                lcm_losses.append(loss_dict["lcm"].detach().float().item())
                ffl_losses.append(loss_dict["ffl"].detach().float().item())
                grad_losses.append(loss_dict["grad"].detach().float().item())

        if losses and accelerator.is_local_main_process:
            avg_loss = sum(losses) / len(losses)
            avg_lcm = sum(lcm_losses) / len(lcm_losses)
            avg_ffl = sum(ffl_losses) / len(ffl_losses)
            avg_grad = sum(grad_losses) / len(grad_losses)
            accelerator.log(
                {
                    f"{split_name}/loss": avg_loss,
                    f"{split_name}/lcm": avg_lcm,
                    f"{split_name}/ffl": avg_ffl,
                    f"{split_name}/grad": avg_grad,
                },
                step=global_step,
            )
            accelerator.print(
                f"[eval] {split_name} loss={avg_loss:.6f} lcm={avg_lcm:.6f} "
                f"ffl={avg_ffl:.6f} grad={avg_grad:.6f}"
            )
        teacher_wrapper.base_model.train()
        student_wrapper.base_model.train()
        accelerator.wait_for_everyone()

    resume_loaded = False
    if resume_path and args.resume_optimizer:
        state_path = os.path.join(resume_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            global_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", 0))
            ema_loss = state.get("ema_loss")
            optimizer_state = state.get("optimizer_state")
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            resume_loaded = True
            if accelerator.is_local_main_process:
                accelerator.print(f"[resume] Loaded optimizer state from {state_path}")
                accelerator.print(
                    f"[resume] global_step={global_step}, total_updates={total_updates}, "
                    f"remaining={max(total_updates - global_step, 0)}"
                )

    if resume_path and not resume_loaded:
        resume_source = None
        if resume_step_hint:
            global_step = resume_step_hint
            resume_source = "checkpoint name"
        else:
            inferred_step = _infer_latest_step(args.output_dir, log_image_dir)
            if inferred_step:
                global_step = inferred_step
                resume_source = "output_dir/image_log"
        if resume_source and accelerator.is_local_main_process:
            accelerator.print(
                "[resume] training_state.pt not found; "
                f"using step={global_step} inferred from {resume_source}."
            )
            accelerator.print(
                f"[resume] global_step={global_step}, total_updates={total_updates}, "
                f"remaining={max(total_updates - global_step, 0)}"
            )

    resume_add_steps = int(args.resume_add_steps)
    if resume_path and resume_add_steps <= 0 and args.max_epochs <= 0 and args.max_train_steps > 0:
        resume_add_steps = max(int(args.max_train_steps) - global_step, 0)

    if resume_path and resume_add_steps > 0:
        total_updates = global_step + resume_add_steps
        required_epochs = max(1, math.ceil(total_updates / steps_per_epoch))
        if required_epochs > max_epochs:
            if accelerator.is_local_main_process:
                accelerator.print(
                    f"[resume] Extending max_epochs from {max_epochs} to {required_epochs} "
                    f"for resume_add_steps={resume_add_steps}."
                )
            max_epochs = required_epochs
        if accelerator.is_local_main_process:
            accelerator.print(
                f"[resume] resume_add_steps={resume_add_steps}, total_updates={total_updates}, "
                f"remaining={max(total_updates - global_step, 0)}"
            )

    lr_schedule_enabled = args.lr_scheduler == "warmup_cosine"
    lr_base = float(args.learning_rate)
    lr_warmup_steps = int(args.lr_warmup_steps)
    lr_min_ratio = float(args.lr_min_ratio)

    progress_bar = tqdm(
        total=total_updates,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    if global_step > 0:
        progress_bar.update(min(global_step, total_updates))
        last_log_step = global_step
        last_log_time = time.perf_counter()

    def save_training_state(save_dir, epoch_idx):
        state = {
            "global_step": global_step,
            "epoch": epoch_idx,
            "ema_loss": ema_loss,
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(state, os.path.join(save_dir, "training_state.pt"))

    for epoch in range(start_epoch, max_epochs):
        if accelerator.is_local_main_process:
            progress_bar.set_description(f"Training (epoch {epoch + 1}/{max_epochs})")
        for batch in train_loader:
            with accelerator.accumulate(student):
                batch_size = batch["img"].shape[0]

                with autocast_context():
                    with torch.inference_mode():
                        latents = encode_img_and_masked_x(
                            batch, teacher_wrapper, device, non_blocking=non_blocking
                        )
                    latents = latents.detach().clone()
                    batch["masked_x"] = batch["masked_x"].detach().clone()

                    timesteps = sample_timesteps(schedule, batch_size, device)
                    noise = torch.randn_like(latents)
                    noisy_latents = add_noise(latents, noise, timesteps, alphas_cumprod)

                    cond_batch = build_cond_batch(batch, device, non_blocking=non_blocking)
                    with torch.inference_mode():
                        uncond_batch, uncond_text_info, uncond_text_emb = uncond_cache.get(
                            batch, teacher_wrapper, device
                        )
                        cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
                        cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)

                        noise_pred_cond = teacher_wrapper.forward(
                            noisy_latents, timesteps, cond_text_emb, cond_text_info, cond_batch["hint"]
                        )
                        noise_pred_uncond = teacher_wrapper.forward(
                            noisy_latents, timesteps, uncond_text_emb, uncond_text_info, uncond_batch["hint"]
                        )
                        noise_pred_teacher = apply_cfg(noise_pred_cond, noise_pred_uncond, args.cfg_scale)
                        target_x0 = predict_x0_from_eps(
                            noisy_latents, timesteps, noise_pred_teacher, alphas_cumprod
                        )
                    target_x0 = target_x0.detach().clone()

                    student_text_info = student_wrapper.prepare_text_info(cond_batch)
                    student_text_emb = student_wrapper.encode_text(cond_batch, student_text_info)
                    noise_pred_student = student_wrapper.forward(
                        noisy_latents, timesteps, student_text_emb, student_text_info, cond_batch["hint"]
                    )
                    pred_x0_student = predict_x0_from_eps(
                        noisy_latents, timesteps, noise_pred_student, alphas_cumprod
                    )
                    pred_x0_detached = pred_x0_student.detach()

                    text_mask = get_text_mask(batch)
                    if text_mask is not None:
                        text_mask = text_mask.to(device, non_blocking=non_blocking)
                    loss, loss_dict = criterion(pred_x0_student, target_x0, mask=text_mask)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if lr_schedule_enabled:
                        lr_scale = warmup_cosine_scale(
                            global_step, total_updates, lr_warmup_steps, lr_min_ratio
                        )
                        lr = lr_base * lr_scale
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % args.logging_steps == 0 and accelerator.is_local_main_process:
                        loss_val = loss.detach().float().item()
                        lcm_val = loss_dict["lcm"].detach().float().item()
                        ffl_val = loss_dict["ffl"].detach().float().item()
                        grad_val = loss_dict["grad"].detach().float().item()
                        ema_loss = loss_val if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_val)
                        now = time.perf_counter()
                        step_delta = max(global_step - last_log_step, 1)
                        time_delta = max(now - last_log_time, 1e-6)
                        it_s = step_delta / time_delta
                        lr = optimizer.param_groups[0]["lr"]
                        postfix = {
                            "loss": f"{loss_val:.4f}",
                            "ema": f"{ema_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "it/s": f"{it_s:.2f}",
                            "epoch": f"{epoch + 1}/{max_epochs}",
                        }
                        if torch.cuda.is_available():
                            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                            postfix["mem_gb"] = f"{mem_gb:.1f}"
                            torch.cuda.reset_peak_memory_stats(device)
                        progress_bar.set_postfix(postfix, refresh=True)
                        progress_pct = (global_step / total_updates) * 100.0
                        progress_bar.write(
                            "Training (epoch {epoch}/{total}): {step}/{total_steps} ({pct:.1f}%) "
                            "loss={loss} ema={ema} lr={lr} it/s={it_s}{mem}".format(
                                epoch=epoch + 1,
                                total=max_epochs,
                                step=global_step,
                                total_steps=total_updates,
                                pct=progress_pct,
                                loss=postfix["loss"],
                                ema=postfix["ema"],
                                lr=postfix["lr"],
                                it_s=postfix["it/s"],
                                mem=f" mem_gb={postfix.get('mem_gb', 'n/a')}",
                            )
                        )
                        accelerator.log(
                            {
                                "train/loss": loss_val,
                                "train/loss_ema": ema_loss,
                                "train/lcm": lcm_val,
                                "train/ffl": ffl_val,
                                "train/grad": grad_val,
                                "train/lr": lr,
                                "train/it_s": it_s,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )
                        last_log_time = now
                        last_log_step = global_step

                    if args.log_image_steps > 0 and global_step % args.log_image_steps == 0:
                        if accelerator.is_local_main_process:
                            log_train_images(
                                global_step,
                                batch,
                                pred_x0_detached,
                                student_wrapper,
                                log_image_dir,
                                max_samples=args.log_image_samples,
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        if accelerator.is_local_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            unwrapped = accelerator.unwrap_model(student)
                            unwrapped.save_pretrained(save_path)
                            save_training_state(save_path, epoch)
                        run_eval("val", val_loader)

                if global_step >= total_updates:
                    break

            if global_step >= total_updates:
                break

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            if accelerator.is_local_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                unwrapped = accelerator.unwrap_model(student)
                unwrapped.save_pretrained(save_path)
                save_training_state(save_path, epoch + 1)
            run_eval("val", val_loader)

        if global_step >= total_updates:
            break

    if accelerator.is_local_main_process:
        save_path = os.path.join(args.output_dir, "checkpoint-final")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(save_path)
        save_training_state(save_path, max_epochs)
    run_eval("val", val_loader)
    run_eval("test", test_loader)


if __name__ == "__main__":
    main()
