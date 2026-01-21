import torch

from cldm.cldm import ControlLDM


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
