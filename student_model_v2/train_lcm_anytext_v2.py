import argparse
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
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
from student_model_v2.lcm_utils_v2 import (
    add_noise,
    apply_cfg,
    compute_lcm_loss,
    make_lcm_schedule,
    predict_x0_from_eps,
    prepare_conditional_batch,
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


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA distillation for AnyText2 (v2)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--output_dir", type=str, default="student_model_v2/checkpoints")
    parser.add_argument("--dataset_json", type=str, nargs="+", default=["demodataset/annotations/demo_data.json"])
    parser.add_argument("--use_mock_dataset", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--eval_batches", type=int, default=1)
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
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn_anytext,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )
    if test_set is not None and len(test_set) > 0:
        test_loader = DataLoader(
            test_set,
            batch_size=args.train_batch_size,
            shuffle=False,
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

    alphas_cumprod = teacher_wrapper.base_model.alphas_cumprod.to(device)
    schedule = make_lcm_schedule(args.num_inference_steps, num_train_timesteps=alphas_cumprod.shape[0])
    autocast_context = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    global_step = 0
    last_log_time = time.perf_counter()
    last_log_step = 0
    ema_loss = None

    non_blocking = bool(args.pin_memory)

    def encode_masked_x(batch, wrapper):
        if "masked_img" not in batch:
            masked_img = batch["img"]
        else:
            masked_img = batch["masked_img"]
        masked_img_nchw = masked_img.permute(0, 3, 1, 2).to(device, non_blocking=non_blocking)
        with torch.no_grad():
            masked_dist = wrapper.base_model.first_stage_model.encode(masked_img_nchw)
            masked_latents = masked_dist.sample() * wrapper.base_model.scale_factor
        batch["masked_x"] = masked_latents

    def run_eval(split_name, loader):
        if loader is None:
            return
        teacher_wrapper.base_model.eval()
        student_wrapper.base_model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.eval_batches:
                    break
                batch = {k: v for k, v in batch.items()}
                with torch.no_grad():
                    with autocast_context():
                        encode_masked_x(batch, teacher_wrapper)
                        batch_size = batch["img"].shape[0]
                        img_nchw = batch["img"].permute(0, 3, 1, 2).to(
                            device, non_blocking=non_blocking
                        )
                        latent_dist = teacher_wrapper.base_model.first_stage_model.encode(img_nchw)
                        latents = latent_dist.sample() * teacher_wrapper.base_model.scale_factor
                        timesteps = sample_timesteps(schedule, batch_size, device)
                        noise = torch.randn_like(latents)
                        noisy_latents = add_noise(latents, noise, timesteps, alphas_cumprod)

                        cond_batch, uncond_batch = prepare_conditional_batch(
                            batch, device, non_blocking=non_blocking
                        )
                        cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
                        uncond_text_info = teacher_wrapper.prepare_text_info(uncond_batch)
                        cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)
                        uncond_text_emb = teacher_wrapper.encode_text(uncond_batch, uncond_text_info)
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
                        loss = compute_lcm_loss(pred_x0_student, target_x0, loss_type="huber")
                losses.append(loss.detach().float().item())

        if losses and accelerator.is_local_main_process:
            avg_loss = sum(losses) / len(losses)
            accelerator.log({f"{split_name}/loss": avg_loss}, step=global_step)
        teacher_wrapper.base_model.train()
        student_wrapper.base_model.train()

    for epoch in range(1000):
        for batch in train_loader:
            with accelerator.accumulate(student):
                batch_size = batch["img"].shape[0]

                with autocast_context():
                    encode_masked_x(batch, teacher_wrapper)
                    img_nchw = batch["img"].permute(0, 3, 1, 2).to(
                        device, non_blocking=non_blocking
                    )
                    with torch.no_grad():
                        latent_dist = teacher_wrapper.base_model.first_stage_model.encode(img_nchw)
                        latents = latent_dist.sample() * teacher_wrapper.base_model.scale_factor

                    timesteps = sample_timesteps(schedule, batch_size, device)
                    noise = torch.randn_like(latents)
                    noisy_latents = add_noise(latents, noise, timesteps, alphas_cumprod)

                    cond_batch, uncond_batch = prepare_conditional_batch(
                        batch, device, non_blocking=non_blocking
                    )
                    cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
                    uncond_text_info = teacher_wrapper.prepare_text_info(uncond_batch)
                    cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)
                    uncond_text_emb = teacher_wrapper.encode_text(uncond_batch, uncond_text_info)

                    with torch.no_grad():
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

                    loss = compute_lcm_loss(pred_x0_student, target_x0, loss_type="huber")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % args.logging_steps == 0 and accelerator.is_local_main_process:
                        loss_val = loss.detach().float().item()
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
                        }
                        if torch.cuda.is_available():
                            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                            postfix["mem_gb"] = f"{mem_gb:.1f}"
                            torch.cuda.reset_peak_memory_stats(device)
                        progress_bar.set_postfix(postfix, refresh=True)
                        accelerator.log({"train/loss": loss_val, "train/loss_ema": ema_loss}, step=global_step)
                        last_log_time = now
                        last_log_step = global_step

                    if global_step % args.save_steps == 0 and accelerator.is_local_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unwrapped = accelerator.unwrap_model(student)
                        unwrapped.save_pretrained(save_path)
                        run_eval("val", val_loader)

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    if accelerator.is_local_main_process:
        save_path = os.path.join(args.output_dir, "checkpoint-final")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(save_path)
        run_eval("val", val_loader)
        run_eval("test", test_loader)


if __name__ == "__main__":
    main()
