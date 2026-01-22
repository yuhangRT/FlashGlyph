import argparse
from pathlib import Path

import torch
import torchvision

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.model import create_model, load_state_dict
from student_model_v2.dataset_anytext_v2 import RealAnyTextDataset, collate_fn_anytext
from student_model_v2.lcm_utils_v2 import (
    make_lcm_schedule,
    ddim_step,
    predict_eps_from_model_output,
)
from student_model_v3.wrappers import AnyText2ForwardWrapper


def _find_latest_run_dir(base_dir):
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("train_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_lora_path(path_str):
    path = Path(path_str).expanduser()
    if path.exists():
        if path.is_dir() and (path / "adapter_config.json").exists():
            return path
        # If a run dir was provided, try checkpoint-final inside it.
        if path.is_dir() and path.name.startswith("train_"):
            candidate = path / "checkpoint-final"
            if (candidate / "adapter_config.json").exists():
                return candidate
        # If a checkpoints dir is provided, pick latest run/checkpoint-final.
        if path.is_dir() and path.name == "checkpoints":
            latest = _find_latest_run_dir(path)
            if latest:
                candidate = latest / "checkpoint-final"
                if (candidate / "adapter_config.json").exists():
                    return candidate
        return path

    # Path doesn't exist: try to resolve under checkpoints/train_*/checkpoint-final.
    checkpoints_dir = (Path(__file__).parent / "checkpoints").resolve()
    if path.name.startswith("checkpoint") or path.name == "checkpoint-final":
        base = path.parent
        if base.name == "checkpoints":
            latest = _find_latest_run_dir(base)
            if latest:
                candidate = latest / path.name
                if (candidate / "adapter_config.json").exists():
                    return candidate
    # Fallback to student_model_v3/checkpoints if base not found.
    if checkpoints_dir.exists():
        latest = _find_latest_run_dir(checkpoints_dir)
        if latest:
            candidate = latest / "checkpoint-final"
            if (candidate / "adapter_config.json").exists():
                return candidate
    return path


def encode_img_and_masked_x(batch, wrapper, device):
    img = batch["img"]
    masked_img = batch.get("masked_img", img)
    img_nchw = img.permute(0, 3, 1, 2).to(device)
    masked_nchw = masked_img.permute(0, 3, 1, 2).to(device)
    stacked = torch.cat([img_nchw, masked_nchw], dim=0)
    latent_dist = wrapper.base_model.first_stage_model.encode(stacked)
    latents = latent_dist.sample() * wrapper.base_model.scale_factor
    latents_img, latents_masked = latents.chunk(2, dim=0)
    batch["masked_x"] = latents_masked
    return latents_img


def build_cond_batch(batch, device):
    return {
        "img": batch["img"].to(device),
        "masked_img": batch["masked_img"].to(device),
        "hint": batch["hint"].to(device),
        "glyphs": [g.to(device) for g in batch["glyphs"]],
        "gly_line": [g.to(device) for g in batch["gly_line"]],
        "positions": [p.to(device) for p in batch["positions"]],
        "masked_x": batch["masked_x"].to(device),
        "img_caption": batch["img_caption"],
        "text_caption": batch["text_caption"],
        "texts": batch["texts"],
        "n_lines": batch["n_lines"].to(device),
        "font_hint": batch["font_hint"].to(device),
        "color": [c.to(device) for c in batch["color"]],
        "language": batch["language"],
        "inv_mask": batch["inv_mask"].to(device),
    }


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


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA inference for AnyText2 (v3)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--student_lora_path", type=str, default="student_model_v3/checkpoints/checkpoint-final")
    parser.add_argument("--dataset_json", type=str, nargs="+", default=["demodataset/annotations/demo_data.json"])
    parser.add_argument("--lmdb_path", type=str, default="")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--output", type=str, default="student_model_v3/preview.png")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.teacher_ckpt)
    if not config_path.is_absolute():
        config_path = (Path(__file__).parent.parent / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).parent.parent / ckpt_path).resolve()

    model = create_model(str(config_path))
    state_dict = load_state_dict(str(ckpt_path), location="cpu")
    model.load_state_dict(state_dict, strict=False)

    if args.student_lora_path:
        from peft import PeftModel
        lora_path = resolve_lora_path(args.student_lora_path)
        if not (Path(lora_path) / "adapter_config.json").exists():
            raise ValueError(
                f"Can't find 'adapter_config.json' at '{lora_path}'. "
                "Pass a specific checkpoint directory or the checkpoints root."
            )
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=False)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = AnyText2ForwardWrapper(model, device)

    dataset = RealAnyTextDataset(
        json_path=args.dataset_json[0],
        resolution=args.resolution,
        lmdb_path=args.lmdb_path or None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.max_samples,
        shuffle=False,
        collate_fn=collate_fn_anytext,
    )

    batch = next(iter(loader))
    encode_img_and_masked_x(batch, wrapper, device)
    cond_batch = build_cond_batch(batch, device)

    text_info = wrapper.prepare_text_info(cond_batch)
    text_emb = wrapper.encode_text(cond_batch, text_info)
    hint = cond_batch["hint"]

    alphas_cumprod = wrapper.base_model.alphas_cumprod.to(device)
    parameterization = getattr(wrapper.base_model, "parameterization", "eps")

    batch_size = hint.shape[0]
    latent_shape = cond_batch["masked_x"].shape[1:]
    dtype = next(wrapper.base_model.parameters()).dtype

    latents = torch.randn((batch_size, *latent_shape), device=device, dtype=dtype)
    schedule = make_lcm_schedule(args.num_inference_steps, num_train_timesteps=alphas_cumprod.shape[0])
    for i, t in enumerate(schedule):
        ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_output = wrapper.forward(latents, ts, text_emb, text_info, hint)
        eps = predict_eps_from_model_output(latents, ts, model_output, alphas_cumprod, parameterization)
        t_prev = schedule[i + 1] if i + 1 < len(schedule) else 0
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
        latents = ddim_step(latents, ts, t_prev_tensor, eps, alphas_cumprod)

    with torch.no_grad():
        pred_img = wrapper.base_model.decode_first_stage(latents)

    grid = make_preview_grid(batch["img"], batch["masked_img"], batch["hint"], pred_img, max_samples=args.max_samples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, out_path)
    print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    main()
