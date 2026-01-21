import argparse
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
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
from student_model_v3.lcm_solver import (
    DDIMSolver,
    extract_into_tensor,
    predicted_origin,
    scalings_for_boundary_conditions,
)
from student_model_v3.wrappers import AnyText2ForwardWrapper


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


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA distillation for AnyText2 (v3, official-style)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--output_dir", type=str, default="student_model_v3/checkpoints")
    parser.add_argument("--dataset_json", type=str, nargs="+", default=["demodataset/annotations/demo_data.json"])
    parser.add_argument("--lmdb_path", type=str, default="")
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
    parser.add_argument("--w_min", type=float, default=5.0)
    parser.add_argument("--w_max", type=float, default=15.0)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "huber"])
    parser.add_argument("--huber_c", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
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
                resolution=args.resolution,
                wm_thresh=args.wm_thresh,
                streaming=args.streaming,
                streaming_threshold_mb=args.streaming_threshold_mb,
                cache_dir=cache_dir or None,
                lmdb_path=args.lmdb_path or None,
            )
            for path in json_paths
        ]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

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

    progress_bar = tqdm(
        total=total_updates,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(max_epochs):
        if accelerator.is_local_main_process:
            progress_bar.set_description(f"Training (epoch {epoch + 1}/{max_epochs})")
        for batch in train_loader:
            with accelerator.accumulate(student):
                batch_size = batch["img"].shape[0]
                with autocast_context():
                    with torch.no_grad():
                        latents = encode_img_and_masked_x(batch, teacher_wrapper, device, non_blocking=args.pin_memory)
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

                    noise_pred = student_wrapper.forward(
                        noisy_latents, start_timesteps, cond_text_emb, cond_text_info, hint
                    )
                    pred_x0 = predicted_origin(
                        noise_pred,
                        start_timesteps,
                        noisy_latents,
                        prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    model_pred = c_skip_start * noisy_latents + c_out_start * pred_x0

                    with torch.no_grad():
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

                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                    with torch.no_grad():
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

                    if args.loss_type == "l2":
                        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        loss = torch.mean(
                            torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                        )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    if lr_schedule_enabled:
                        scale = warmup_cosine_scale(global_step, total_updates, lr_warmup_steps, lr_min_ratio)
                        for group in optimizer.param_groups:
                            group["lr"] = lr_base * scale
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

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
                    it_s = (global_step - last_log_step) / max(now - last_log_time, 1e-6)
                    accelerator.log({"train/loss": loss.detach().item(), "train/it_s": it_s}, step=global_step)
                    last_log_step = global_step
                    last_log_time = now

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
