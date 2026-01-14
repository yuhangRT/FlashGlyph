import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.cldm import ControlLDM
from cldm.model import create_model, load_state_dict

from student_model_v2.dataset_anytext_v2 import (
    AnyTextMockDataset,
    RealAnyTextDataset,
    collate_fn_anytext,
)
from student_model_v2.lcm_utils_v2 import (
    ddim_step,
    make_lcm_schedule,
    prepare_conditional_batch,
)


def unwrap_controlldm(model):
    if isinstance(model, ControlLDM):
        return model
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
        self.reset_state()

    def reset_state(self):
        self.base_model.control = None
        self.base_model.control_uncond = None
        self.base_model.is_uncond = False

    def encode_text(self, batch, text_info=None):
        cond = {
            "c_crossattn": [[batch["img_caption"], batch["text_caption"]]],
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


def save_image(tensor, output_path):
    image = tensor.clamp(-1, 1)
    image = (image + 1) / 2.0
    image = (image * 255).round().byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA inference for AnyText2 (v2)")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--dataset_json", type=str, default="demodataset/annotations/demo_data.json")
    parser.add_argument("--use_mock_dataset", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="student_model_v2/sample.png")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.teacher_ckpt)
    if not config_path.is_absolute():
        config_path = (Path(__file__).parent.parent / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).parent.parent / ckpt_path).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = create_model(str(config_path))
    base_sd = load_state_dict(str(ckpt_path), location="cpu")
    base_model.load_state_dict(base_sd, strict=False)
    base_model.eval()

    student = PeftModel.from_pretrained(base_model, args.lora_path)
    student.eval()

    wrapper = AnyText2ForwardWrapper(student, device)

    if args.use_mock_dataset:
        dataset = AnyTextMockDataset(size=10, resolution=args.resolution)
    else:
        dataset = RealAnyTextDataset(json_path=args.dataset_json, resolution=args.resolution)
    sample = dataset[args.sample_index % len(dataset)]
    batch = collate_fn_anytext([sample])
    if "masked_img" in batch:
        masked_img = batch["masked_img"]
    else:
        masked_img = batch["img"]
    masked_img_nchw = masked_img.to(device).permute(0, 3, 1, 2)
    with torch.no_grad():
        masked_dist = wrapper.base_model.first_stage_model.encode(masked_img_nchw)
        masked_latents = masked_dist.sample() * wrapper.base_model.scale_factor
    batch["masked_x"] = masked_latents
    cond_batch, _ = prepare_conditional_batch(batch, device)
    text_info = wrapper.prepare_text_info(cond_batch)
    text_emb = wrapper.encode_text(cond_batch, text_info)

    dtype = next(wrapper.base_model.parameters()).dtype
    latent_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)

    alphas_cumprod = wrapper.base_model.alphas_cumprod.to(device)
    schedule = make_lcm_schedule(args.num_inference_steps, num_train_timesteps=alphas_cumprod.shape[0])

    with torch.no_grad():
        for i, t in enumerate(schedule):
            t_tensor = torch.tensor([t], device=device)
            noise_pred = wrapper.forward(latents, t_tensor, text_emb, text_info, cond_batch["hint"])
            t_prev = schedule[i + 1] if i + 1 < len(schedule) else 0
            t_prev_tensor = torch.tensor([t_prev], device=device)
            latents = ddim_step(latents, t_tensor, t_prev_tensor, noise_pred, alphas_cumprod)

        decoded = wrapper.base_model.decode_first_stage(latents)
        save_image(decoded[0], args.output_path)

    print(f"Saved sample to: {args.output_path}")


if __name__ == "__main__":
    main()
