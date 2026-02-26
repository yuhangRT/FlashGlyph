import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.model import create_model, load_state_dict
from student_model_v2.dataset_anytext_v2 import RealAnyTextDataset, collate_fn_anytext
from student_model_v2.lcm_utils_v2 import (
    ddim_step,
    make_lcm_schedule,
    predict_eps_from_model_output,
)
from student_model_v3.infer_lcm_anytext_v3 import (
    build_cond_batch,
    build_uncond_batch,
    encode_img_and_masked_x,
    resolve_lora_path,
)
from student_model_v3.wrappers import AnyText2ForwardWrapper


def _img_key_from_entry(entry, idx):
    img_name = str(entry.get("img_name", ""))
    stem = Path(img_name).stem
    if stem:
        return stem
    return f"idx_{idx:06d}"


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_once(
    wrapper,
    num_steps,
    batch_size,
    latent_shape,
    dtype,
    device,
    alphas_cumprod,
    parameterization,
    use_cfg,
    cfg_scale,
    text_emb,
    text_info,
    uncond_text_emb,
    uncond_text_info,
    hint,
    seed,
):
    if device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(int(seed))

    latents = torch.randn((batch_size, *latent_shape), device=device, dtype=dtype, generator=gen)
    schedule = make_lcm_schedule(num_steps, num_train_timesteps=alphas_cumprod.shape[0])

    for i, t in enumerate(schedule):
        ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
        if use_cfg:
            eps_cond = wrapper.forward(latents, ts, text_emb, text_info, hint)
            eps_uncond = wrapper.forward(latents, ts, uncond_text_emb, uncond_text_info, hint)
            model_output = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            model_output = wrapper.forward(latents, ts, text_emb, text_info, hint)
        eps = predict_eps_from_model_output(latents, ts, model_output, alphas_cumprod, parameterization)
        t_prev = schedule[i + 1] if i + 1 < len(schedule) else 0
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
        latents = ddim_step(latents, ts, t_prev_tensor, eps, alphas_cumprod)

    return wrapper.base_model.decode_first_stage(latents)


def _save_tensor_image(img_tensor, out_path, jpeg_quality=95):
    img01 = ((img_tensor.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    arr = (img01.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(arr)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(out_path, quality=int(jpeg_quality))
    else:
        image.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Batch generation for evaluation images (AnyText-style naming).")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--student_lora_path", type=str, default="student_model_v3/checkpoints/checkpoint-final")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples_per_input", type=int, default=4)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=0, help="0 means until dataset end")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_ext", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--streaming", dest="streaming", action="store_true")
    parser.add_argument("--no_streaming", dest="streaming", action="store_false")
    parser.add_argument("--mask_img_prob", type=float, default=1.0)
    parser.add_argument("--font_hint_randaug", action="store_true", default=False)
    parser.add_argument("--log_every", type=int, default=20)
    parser.set_defaults(streaming=True)
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

    lora_path = None
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
        json_path=args.input_json,
        resolution=args.resolution,
        lmdb_path=args.lmdb_path or None,
        streaming=args.streaming,
        mask_img_prob=float(args.mask_img_prob),
        font_hint_randaug=bool(args.font_hint_randaug),
    )
    total = len(dataset)
    if total <= 0:
        raise RuntimeError("Dataset is empty.")

    start = max(0, int(args.start_idx))
    if start >= total:
        raise ValueError(f"start_idx={start} out of range (dataset size={total})")
    end = total if int(args.max_items) <= 0 else min(total, start + int(args.max_items))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas_cumprod = wrapper.base_model.alphas_cumprod.to(device)
    parameterization = getattr(wrapper.base_model, "parameterization", "eps")
    use_cfg = bool(args.use_cfg or args.cfg_scale > 1.0)
    dtype = next(wrapper.base_model.parameters()).dtype

    generated = 0
    skipped = 0
    t0 = time.time()

    for idx in range(start, end):
        entry = dataset.data_list[idx]
        img_key = _img_key_from_entry(entry, idx)

        targets = [out_dir / f"{img_key}_{k}.{args.image_ext}" for k in range(args.num_samples_per_input)]
        if args.skip_existing and all(p.exists() for p in targets):
            skipped += len(targets)
            continue

        _set_global_seed(args.seed + idx)
        sample = dataset[idx]
        batch = collate_fn_anytext([sample])
        encode_img_and_masked_x(batch, wrapper, device)
        cond_batch = build_cond_batch(batch, device)
        uncond_batch = build_uncond_batch(cond_batch)

        text_info = wrapper.prepare_text_info(cond_batch)
        text_emb = wrapper.encode_text(cond_batch, text_info)
        uncond_text_info = wrapper.prepare_text_info(uncond_batch)
        uncond_text_emb = wrapper.encode_text(uncond_batch, uncond_text_info)
        hint = cond_batch["hint"]
        batch_size = hint.shape[0]
        latent_shape = cond_batch["masked_x"].shape[1:]

        for k, out_path in enumerate(targets):
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            pred_img = _sample_once(
                wrapper=wrapper,
                num_steps=args.num_inference_steps,
                batch_size=batch_size,
                latent_shape=latent_shape,
                dtype=dtype,
                device=device,
                alphas_cumprod=alphas_cumprod,
                parameterization=parameterization,
                use_cfg=use_cfg,
                cfg_scale=args.cfg_scale,
                text_emb=text_emb,
                text_info=text_info,
                uncond_text_emb=uncond_text_emb,
                uncond_text_info=uncond_text_info,
                hint=hint,
                seed=args.seed * 1000003 + idx * 97 + k,
            )
            _save_tensor_image(pred_img[0], out_path, jpeg_quality=args.jpeg_quality)
            generated += 1

        if (idx - start + 1) % max(1, args.log_every) == 0:
            elapsed = time.time() - t0
            print(
                f"[{idx - start + 1}/{end - start}] "
                f"generated={generated}, skipped={skipped}, elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(
        f"Done. range=[{start}, {end}), generated={generated}, skipped={skipped}, "
        f"elapsed={elapsed:.1f}s, out_dir={out_dir}, lora={lora_path}"
    )


if __name__ == "__main__":
    main()
