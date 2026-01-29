import argparse
import math
import os
import time
from datetime import datetime
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torchvision
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, PeftModel, get_peft_model
try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    Conv1D = None
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.model import create_model, load_state_dict
from student_model_v2.dataset_anytext_v2 import (
    AnyTextMockDataset,
    RealAnyTextDataset,
    collate_fn_anytext,
)
from student_model_v2.lcm_utils_v2 import (
    make_lcm_schedule,
    ddim_step,
    predict_eps_from_model_output,
)
from student_model_v3.lcm_solver import (
    DDIMSolver,
    extract_into_tensor,
    predicted_origin,
    scalings_for_boundary_conditions,
)
from student_model_v3.wrappers import AnyText2ForwardWrapper
from student_model_v3.losses import HighFreqTextLoss


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


def disable_checkpointing(model):
    if hasattr(model, "use_checkpoint"):
        model.use_checkpoint = False
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False


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


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def add_noise(x0, noise, timesteps, alphas, sigmas):
    alpha_t = extract_into_tensor(alphas, timesteps, x0.shape)
    sigma_t = extract_into_tensor(sigmas, timesteps, x0.shape)
    return alpha_t * x0 + sigma_t * noise


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


def build_cond_batch(batch, device, non_blocking=False):
    return {
        "img": batch["img"].to(device, non_blocking=non_blocking),
        "masked_img": batch["masked_img"].to(device, non_blocking=non_blocking),
        "hint": batch["hint"].to(device, non_blocking=non_blocking),
        "glyphs": [g.to(device, non_blocking=non_blocking) for g in batch["glyphs"]],
        "gly_line": [g.to(device, non_blocking=non_blocking) for g in batch["gly_line"]],
        "positions": [p.to(device, non_blocking=non_blocking) for p in batch["positions"]],
        "masked_x": batch["masked_x"].to(device, non_blocking=non_blocking),
        "img_caption": batch["img_caption"],
        "text_caption": batch["text_caption"],
        "texts": batch["texts"],
        "n_lines": batch["n_lines"].to(device, non_blocking=non_blocking),
        "font_hint": batch["font_hint"].to(device, non_blocking=non_blocking),
        "color": [c.to(device, non_blocking=non_blocking) for c in batch["color"]],
        "language": batch["language"],
        "inv_mask": batch["inv_mask"].to(device, non_blocking=non_blocking),
    }


def build_uncond_batch(cond_batch):
    batch_size = cond_batch["img"].shape[0]
    uncond_batch = dict(cond_batch)
    uncond_batch["img_caption"] = [""] * batch_size
    uncond_batch["text_caption"] = [""] * batch_size
    return uncond_batch


def get_cond_cache(batch, wrapper, device, non_blocking=False):
    cache = batch.get("_cond_cache")
    wrapper_id = id(wrapper.base_model)
    if cache and cache.get("wrapper_id") == wrapper_id:
        return cache["hint"], cache["text_info"], cache["text_emb"]
    cond_batch = build_cond_batch(batch, device, non_blocking=non_blocking)
    text_info = wrapper.prepare_text_info(cond_batch)
    text_emb = wrapper.encode_text(cond_batch, text_info)
    cache = {
        "wrapper_id": wrapper_id,
        "hint": cond_batch["hint"],
        "text_info": text_info,
        "text_emb": text_emb,
    }
    batch["_cond_cache"] = cache
    return cache["hint"], cache["text_info"], cache["text_emb"]


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


def _slice_batch_for_log(batch, max_samples):
    if max_samples <= 0:
        return batch
    if not batch or "img" not in batch:
        return batch
    if batch["img"].shape[0] <= max_samples:
        return batch

    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[:max_samples]
        elif isinstance(value, list):
            if not value:
                sliced[key] = value
            elif torch.is_tensor(value[0]):
                sliced[key] = [v[:max_samples] for v in value]
            elif isinstance(value[0], (list, tuple)):
                sliced[key] = [list(v[:max_samples]) for v in value]
            else:
                sliced[key] = value[:max_samples]
        else:
            sliced[key] = value
    return sliced


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


def log_train_images_infer(
    step,
    batch,
    wrapper,
    output_dir,
    max_samples,
    num_inference_steps,
    cfg_scale,
    use_cfg,
    alphas_cumprod,
    parameterization,
    device,
    autocast_context,
    non_blocking=False,
):
    if max_samples <= 0:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = _slice_batch_for_log(batch, max_samples)
    was_training = wrapper.base_model.training
    wrapper.base_model.eval()
    try:
        with torch.no_grad():
            with autocast_context():
                cond_batch = build_cond_batch(batch, device, non_blocking=non_blocking)
                uncond_batch = build_uncond_batch(cond_batch)

                text_info = wrapper.prepare_text_info(cond_batch)
                text_emb = wrapper.encode_text(cond_batch, text_info)
                uncond_text_info = wrapper.prepare_text_info(uncond_batch)
                uncond_text_emb = wrapper.encode_text(uncond_batch, uncond_text_info)
                hint = cond_batch["hint"]
                batch_size = hint.shape[0]
                latent_shape = batch["masked_x"].shape[1:]
                dtype = next(wrapper.base_model.parameters()).dtype

                latents = torch.randn((batch_size, *latent_shape), device=device, dtype=dtype)
                schedule = make_lcm_schedule(
                    num_inference_steps, num_train_timesteps=alphas_cumprod.shape[0]
                )
                use_cfg = use_cfg or cfg_scale > 1.0
                for i, t in enumerate(schedule):
                    ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
                    if use_cfg:
                        # CFG for visualization: combine cond/uncond predictions.
                        eps_cond = wrapper.forward(latents, ts, text_emb, text_info, hint)
                        eps_uncond = wrapper.forward(latents, ts, uncond_text_emb, uncond_text_info, hint)
                        model_output = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    else:
                        model_output = wrapper.forward(latents, ts, text_emb, text_info, hint)
                    eps = predict_eps_from_model_output(
                        latents, ts, model_output, alphas_cumprod, parameterization
                    )
                    t_prev = schedule[i + 1] if i + 1 < len(schedule) else 0
                    t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                    latents = ddim_step(latents, ts, t_prev_tensor, eps, alphas_cumprod)

                pred_img = wrapper.base_model.decode_first_stage(latents)
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
    finally:
        if was_training:
            wrapper.base_model.train()


def get_prediction_type(base_model):
    param = getattr(base_model, "parameterization", "eps")
    if param == "eps":
        return "epsilon"
    if param == "v":
        return "v_prediction"
    if param == "x0":
        return "x0"
    return "epsilon"


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


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _maybe_sync(enable):
    if enable and torch.cuda.is_available():
        torch.cuda.synchronize()


def report_lmdb_status(dataset, accelerator):
    datasets = []
    if isinstance(dataset, ConcatDataset):
        datasets = list(dataset.datasets)
    else:
        datasets = [dataset]
    enabled = 0
    font_hint = 0
    total = 0
    for ds in datasets:
        if not hasattr(ds, "_lmdb_enabled"):
            continue
        total += 1
        if getattr(ds, "_lmdb_enabled", False):
            enabled += 1
        if getattr(ds, "_lmdb_use_font_hint", False):
            font_hint += 1
    if total == 0:
        return
    accelerator.print(
        f"[lmdb] enabled {enabled}/{total} datasets, font_hint_base {font_hint}/{total}"
    )


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA distillation for AnyText2 (v3, official-style)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--output_dir", type=str, default="student_model_v3/checkpoints")
    parser.add_argument("--dataset_json", type=str, nargs="+", default=["demodataset/annotations/demo_data.json"])
    parser.add_argument("--lmdb_path", type=str, default="")
    parser.add_argument("--max_lines", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=20)
    parser.add_argument("--font_path", type=str, default="./font/Arial_Unicode.ttf")
    parser.add_argument("--font_hint_prob", type=float, default=0.8)
    parser.add_argument("--font_hint_randaug", type=_parse_bool, default=True)
    parser.add_argument("--color_prob", type=float, default=1.0)
    parser.add_argument("--glyph_scale", type=float, default=1.0)
    parser.add_argument("--mask_img_prob", type=float, default=0.5)
    parser.add_argument("--fix_masked_img_bug", type=_parse_bool, default=True)
    parser.add_argument("--use_mock_dataset", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    parser.add_argument("--num_ddim_timesteps", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--w_min", type=float, default=5.0)
    parser.add_argument("--w_max", type=float, default=15.0)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "huber"])
    parser.add_argument("--huber_c", type=float, default=0.001)
    parser.add_argument("--loss_teacher_x0_weight", type=float, default=0.0)
    parser.add_argument("--loss_ffl_weight", type=float, default=0.0)
    parser.add_argument("--loss_grad_weight", type=float, default=0.0)
    parser.add_argument("--ffl_alpha", type=float, default=1.0)
    parser.add_argument("--ffl_patch_factor", type=int, default=1)
    parser.add_argument("--ffl_ave_spectrum", action="store_true", default=False)
    parser.add_argument("--ffl_log_matrix", action="store_true", default=False)
    parser.add_argument("--ffl_batch_matrix", action="store_true", default=False)
    parser.add_argument("--loss_mask_key", type=str, default="hint", choices=["hint", "positions", "inv_mask"])
    parser.add_argument("--loss_text_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_image_steps", type=int, default=0)
    parser.add_argument("--log_image_samples", type=int, default=4)
    parser.add_argument("--log_image_infer_steps", type=int, default=4)
    parser.add_argument("--timing_steps", type=int, default=0)
    parser.add_argument("--timing_cuda_sync", type=_parse_bool, default=False)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--mp_context", type=str, default="")
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
    parser.add_argument("--cast_teacher_unet", action="store_true")
    args = parser.parse_args()

    if args.w_min > args.w_max:
        raise ValueError("w_min must be <= w_max")
    if args.num_ddim_timesteps <= 1:
        raise ValueError("num_ddim_timesteps must be > 1")

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
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "This v3 training script is single-GPU only. "
            "Run without multi-GPU accelerate and set CUDA_VISIBLE_DEVICES to a single GPU."
        )

    run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    if output_dir.name.startswith("train_"):
        run_dir = output_dir
    else:
        run_dir = output_dir / f"train_{run_suffix}"
    args.output_dir = str(run_dir)

    config_path = Path(args.config)
    ckpt_path = Path(args.teacher_ckpt)
    if not config_path.is_absolute():
        config_path = (Path(__file__).parent.parent / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).parent.parent / ckpt_path).resolve()

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.init_trackers("anytext2_lcm_v3", config=sanitize_hparams(vars(args)))

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
        cache_dir = args.cache_dir.strip() if args.cache_dir else ""
        if cache_dir:
            cache_dir = str((repo_root / cache_dir).resolve()) if not os.path.isabs(cache_dir) else cache_dir
        datasets = [
            RealAnyTextDataset(
                json_path=path,
                max_lines=args.max_lines,
                max_chars=args.max_chars,
                resolution=args.resolution,
                font_path=args.font_path,
                font_hint_prob=args.font_hint_prob,
                font_hint_randaug=args.font_hint_randaug,
                color_prob=args.color_prob,
                glyph_scale=args.glyph_scale,
                mask_img_prob=args.mask_img_prob,
                fix_masked_img_bug=args.fix_masked_img_bug,
                wm_thresh=args.wm_thresh,
                streaming=args.streaming,
                streaming_threshold_mb=args.streaming_threshold_mb,
                cache_dir=cache_dir or None,
                lmdb_path=args.lmdb_path or None,
            )
            for path in json_paths
        ]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        report_lmdb_status(dataset, accelerator)

    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    mp_context = args.mp_context if args.num_workers > 0 and args.mp_context else None

    worker_init_fn = None
    if args.num_workers > 0:
        worker_init_fn = partial(_worker_init_fn, args.worker_num_threads, args.cv2_num_threads)

    train_loader = DataLoader(
        dataset,
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

    student, optimizer, train_loader = accelerator.prepare(student, optimizer, train_loader)
    device = accelerator.device
    teacher.to(device)

    teacher_wrapper = AnyText2ForwardWrapper(teacher, device)
    student_wrapper = AnyText2ForwardWrapper(student, device)

    if args.cast_teacher_unet:
        if args.mixed_precision == "fp16":
            teacher_wrapper.base_model.to(dtype=torch.float16)
        elif args.mixed_precision == "bf16":
            teacher_wrapper.base_model.to(dtype=torch.bfloat16)

    alphas_cumprod = teacher_wrapper.base_model.alphas_cumprod.to(device)
    alpha_schedule = torch.sqrt(alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1.0 - alphas_cumprod).to(device)
    num_train_timesteps = int(alphas_cumprod.shape[0])
    solver = DDIMSolver(
        alphas_cumprod.detach().cpu().numpy(),
        timesteps=num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    ).to(device)

    prediction_type = get_prediction_type(teacher_wrapper.base_model)
    parameterization = getattr(teacher_wrapper.base_model, "parameterization", "eps")
    log_image_dir = os.path.join(args.output_dir, "train_img")
    non_blocking = bool(args.pin_memory)

    high_freq_loss = None
    if args.loss_ffl_weight > 0 or args.loss_grad_weight > 0:
        high_freq_loss = HighFreqTextLoss(
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

    def _normalize_text_mask(mask, target_shape):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() == 4 and mask.shape[1] > 1:
            mask = mask.max(dim=1, keepdim=True).values
        if mask.shape[-2:] != target_shape[-2:]:
            mask = F.interpolate(mask, size=target_shape[-2:], mode="nearest")
        return (mask > 0.5).float()

    def _loss_elementwise(pred, target):
        if args.loss_type == "l2":
            return (pred - target) ** 2
        return torch.sqrt((pred - target) ** 2 + args.huber_c**2) - args.huber_c

    def compute_mask_weighted_loss(pred, target, mask):
        err = _loss_elementwise(pred, target)
        if mask is None:
            return err.mean(), None
        mask_lat = _normalize_text_mask(mask, pred.shape)
        if mask_lat is None:
            return err.mean(), None
        # Emphasize text regions in the main LCM consistency loss.
        weight = 1.0 + (args.loss_text_weight - 1.0) * mask_lat
        weighted = (err * weight).mean() / (weight.mean() + 1e-8)
        return weighted, mask_lat

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

    lr_schedule_enabled = args.lr_scheduler == "warmup_cosine"
    lr_base = float(args.learning_rate)
    lr_warmup_steps = int(args.lr_warmup_steps)
    lr_min_ratio = float(args.lr_min_ratio)

    global_step = 0
    last_log_time = time.perf_counter()
    last_log_step = 0
    sanity_checked = False
    ema_loss = None

    progress_bar = tqdm(
        total=total_updates,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    timing_enabled = args.timing_steps > 0
    timing_sync = bool(args.timing_cuda_sync)
    timing_stats = {
        "data": 0.0,
        "encode": 0.0,
        "cond": 0.0,
        "student": 0.0,
        "teacher": 0.0,
        "target": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "opt": 0.0,
        "step": 0.0,
    }
    timing_count = 0
    last_step_end = time.perf_counter()

    for epoch in range(max_epochs):
        if accelerator.is_local_main_process:
            progress_bar.set_description(f"Training (epoch {epoch + 1}/{max_epochs})")
        for batch in train_loader:
            if timing_enabled:
                step_start = time.perf_counter()
                data_time = step_start - last_step_end
            else:
                data_time = 0.0
            with accelerator.accumulate(student):
                batch_size = batch["img"].shape[0]
                with autocast_context():
                    with torch.no_grad():
                        _maybe_sync(timing_sync)
                        t0 = time.perf_counter()
                        latents = encode_img_and_masked_x(batch, teacher_wrapper, device, non_blocking=args.pin_memory)
                        _maybe_sync(timing_sync)
                        encode_time = time.perf_counter() - t0 if timing_enabled else 0.0
                    latents = latents.detach().clone()
                    batch["masked_x"] = batch["masked_x"].detach().clone()

                    noise = torch.randn_like(latents)
                    topk = num_train_timesteps // args.num_ddim_timesteps
                    index = torch.randint(0, args.num_ddim_timesteps, (batch_size,), device=device).long()
                    start_timesteps = solver.ddim_timesteps[index]
                    timesteps = start_timesteps - topk
                    timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                    c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                    c_skip_start = append_dims(c_skip_start, latents.ndim).to(latents.dtype)
                    c_out_start = append_dims(c_out_start, latents.ndim).to(latents.dtype)
                    c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                    c_skip = append_dims(c_skip, latents.ndim).to(latents.dtype)
                    c_out = append_dims(c_out, latents.ndim).to(latents.dtype)

                    noisy_latents = add_noise(latents, noise, start_timesteps, alpha_schedule, sigma_schedule)

                    _maybe_sync(timing_sync)
                    t0 = time.perf_counter()
                    cond_batch = build_cond_batch(batch, device, non_blocking=args.pin_memory)
                    uncond_batch = build_uncond_batch(cond_batch)

                    if not sanity_checked and accelerator.is_local_main_process:
                        if not torch.equal(cond_batch["hint"], uncond_batch["hint"]):
                            raise RuntimeError("Uncond hint diverged from cond hint; controls must match.")
                        sanity_checked = True

                    cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
                    uncond_text_info = teacher_wrapper.prepare_text_info(uncond_batch)
                    cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)
                    uncond_text_emb = teacher_wrapper.encode_text(uncond_batch, uncond_text_info)
                    hint = cond_batch["hint"]
                    _maybe_sync(timing_sync)
                    cond_time = time.perf_counter() - t0 if timing_enabled else 0.0

                    _maybe_sync(timing_sync)
                    t0 = time.perf_counter()
                    noise_pred = student_wrapper.forward(
                        noisy_latents, start_timesteps, cond_text_emb, cond_text_info, hint
                    )
                    student_pred_x0 = predicted_origin(
                        noise_pred,
                        start_timesteps,
                        noisy_latents,
                        prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    model_pred = c_skip_start * noisy_latents + c_out_start * student_pred_x0
                    _maybe_sync(timing_sync)
                    student_time = time.perf_counter() - t0 if timing_enabled else 0.0

                    with torch.no_grad():
                        _maybe_sync(timing_sync)
                        t0 = time.perf_counter()
                        cond_teacher_output = teacher_wrapper.forward(
                            noisy_latents, start_timesteps, cond_text_emb, cond_text_info, hint
                        )
                        uncond_teacher_output = teacher_wrapper.forward(
                            noisy_latents, start_timesteps, uncond_text_emb, uncond_text_info, hint
                        )
                        cond_pred_x0 = predicted_origin(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_x0 = predicted_origin(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        w = (args.w_max - args.w_min) * torch.rand((batch_size,), device=device) + args.w_min
                        w = w.reshape(batch_size, 1, 1, 1).to(noisy_latents.dtype)

                        teacher_guided_pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                        x_prev = solver.ddim_step(teacher_guided_pred_x0, pred_noise, index)
                        x_prev = x_prev.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
                        _maybe_sync(timing_sync)
                        teacher_time = time.perf_counter() - t0 if timing_enabled else 0.0

                    with torch.no_grad():
                        _maybe_sync(timing_sync)
                        t0 = time.perf_counter()
                        target_noise_pred = student_wrapper.forward(
                            x_prev, timesteps, cond_text_emb, cond_text_info, hint
                        )
                        pred_x0_target = predicted_origin(
                            target_noise_pred,
                            timesteps,
                            x_prev,
                            prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        target = c_skip * x_prev + c_out * pred_x0_target
                        _maybe_sync(timing_sync)
                        target_time = time.perf_counter() - t0 if timing_enabled else 0.0

                    _maybe_sync(timing_sync)
                    t0 = time.perf_counter()
                    text_mask = get_text_mask(batch)
                    if text_mask is not None:
                        text_mask = text_mask.to(device, non_blocking=non_blocking)
                    loss_base_raw = _loss_elementwise(model_pred.float(), target.float()).mean()
                    loss_base, mask_lat = compute_mask_weighted_loss(
                        model_pred.float(), target.float(), text_mask
                    )
                    loss = loss_base
                    loss_teacher_x0 = loss_base.new_tensor(0.0)
                    if args.loss_teacher_x0_weight > 0:
                        if mask_lat is None:
                            weight = None
                        else:
                            weight = 1.0 + (args.loss_text_weight - 1.0) * mask_lat
                        l1_err = (student_pred_x0 - teacher_guided_pred_x0).abs()
                        if weight is None:
                            loss_teacher_x0 = l1_err.mean()
                        else:
                            loss_teacher_x0 = (l1_err * weight).mean() / (weight.mean() + 1e-8)
                        loss = loss + args.loss_teacher_x0_weight * loss_teacher_x0
                    loss_ffl = loss_base.new_tensor(0.0)
                    loss_grad = loss_base.new_tensor(0.0)
                    if high_freq_loss is not None:
                        hf_total, hf_dict = high_freq_loss(teacher_guided_pred_x0, pred_x0_target, mask=text_mask)
                        loss = loss + hf_total
                        loss_ffl = hf_dict["ffl"]
                        loss_grad = hf_dict["grad"]
                    _maybe_sync(timing_sync)
                    loss_time = time.perf_counter() - t0 if timing_enabled else 0.0

                _maybe_sync(timing_sync)
                t0 = time.perf_counter()
                accelerator.backward(loss)
                _maybe_sync(timing_sync)
                backward_time = time.perf_counter() - t0 if timing_enabled else 0.0
                if accelerator.sync_gradients:
                    _maybe_sync(timing_sync)
                    t0 = time.perf_counter()
                    accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    if lr_schedule_enabled:
                        scale = warmup_cosine_scale(global_step, total_updates, lr_warmup_steps, lr_min_ratio)
                        for group in optimizer.param_groups:
                            group["lr"] = lr_base * scale
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    _maybe_sync(timing_sync)
                    opt_time = time.perf_counter() - t0 if timing_enabled else 0.0
                else:
                    opt_time = 0.0

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if accelerator.is_local_main_process:
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(student)
                        unwrapped.save_pretrained(save_path)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    loss_val = loss.detach().float().item()
                    loss_base_val = loss_base.detach().float().item()
                    loss_base_raw_val = loss_base_raw.detach().float().item()
                    mask_ratio = mask_lat.mean().detach().float().item() if mask_lat is not None else None
                    loss_teacher_x0_val = loss_teacher_x0.detach().float().item()
                    ffl_val = None
                    grad_val = None
                    if high_freq_loss is not None:
                        ffl_val = loss_ffl.detach().float().item()
                        grad_val = loss_grad.detach().float().item()
                    ema_loss = loss_val if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_val)
                    it_s = (global_step - last_log_step) / max(now - last_log_time, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    postfix = {
                        "loss": f"{loss_val:.4f}",
                        "lcm": f"{loss_base_val:.4f}",
                        "lcm_raw": f"{loss_base_raw_val:.4f}",
                        "ema": f"{ema_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "it/s": f"{it_s:.2f}",
                        "epoch": f"{epoch + 1}/{max_epochs}",
                    }
                    if mask_ratio is not None:
                        postfix["mask"] = f"{mask_ratio:.3f}"
                    if args.loss_teacher_x0_weight > 0:
                        postfix["x0"] = f"{loss_teacher_x0_val:.4f}"
                    if ffl_val is not None and grad_val is not None:
                        postfix["ffl"] = f"{ffl_val:.4f}"
                        postfix["grad"] = f"{grad_val:.4f}"
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
                            "train/lcm": loss_base_val,
                            "train/lcm_raw": loss_base_raw_val,
                            "train/mask_ratio": mask_ratio if mask_ratio is not None else 0.0,
                            "train/teacher_x0": loss_teacher_x0_val,
                            "train/lr": lr,
                            "train/it_s": it_s,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                    if ffl_val is not None and grad_val is not None:
                        accelerator.log(
                            {
                                "train/ffl": ffl_val,
                                "train/grad": grad_val,
                            },
                            step=global_step,
                        )
                    last_log_step = global_step
                    last_log_time = now

                if args.log_image_steps > 0 and global_step % args.log_image_steps == 0:
                    if accelerator.is_local_main_process:
                        log_train_images_infer(
                            global_step,
                            batch,
                            student_wrapper,
                            log_image_dir,
                            max_samples=args.log_image_samples,
                            num_inference_steps=args.log_image_infer_steps,
                            cfg_scale=args.cfg_scale,
                            use_cfg=args.use_cfg,
                            alphas_cumprod=alphas_cumprod,
                            parameterization=parameterization,
                            device=device,
                            autocast_context=autocast_context,
                            non_blocking=non_blocking,
                        )

            if timing_enabled:
                step_time = time.perf_counter() - step_start
                timing_stats["data"] += data_time
                timing_stats["encode"] += encode_time
                timing_stats["cond"] += cond_time
                timing_stats["student"] += student_time
                timing_stats["teacher"] += teacher_time
                timing_stats["target"] += target_time
                timing_stats["loss"] += loss_time
                timing_stats["backward"] += backward_time
                timing_stats["opt"] += opt_time
                timing_stats["step"] += step_time
                timing_count += 1

                if accelerator.sync_gradients and timing_count > 0 and global_step % args.timing_steps == 0:
                    avg = {k: v / timing_count for k, v in timing_stats.items()}
                    progress_bar.write(
                        "Timing(avg): data={data:.3f}s encode={encode:.3f}s cond={cond:.3f}s "
                        "student={student:.3f}s teacher={teacher:.3f}s target={target:.3f}s "
                        "loss={loss:.3f}s backward={backward:.3f}s opt={opt:.3f}s step={step:.3f}s".format(
                            **avg
                        )
                    )
                    for k in timing_stats:
                        timing_stats[k] = 0.0
                    timing_count = 0

            if timing_enabled:
                last_step_end = time.perf_counter()

            if global_step >= total_updates:
                break

        if global_step >= total_updates:
            break

    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "checkpoint-final")
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(final_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
