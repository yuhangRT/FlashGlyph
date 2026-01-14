import torch
from peft import PeftModel

from cldm.ddim_hacked import DDIMSampler
from ms_wrapper import AnyText2Model
from student_model_v2.lcm_utils_v2 import ddim_step, make_lcm_schedule


class LCMSampler:
    def __init__(self, model):
        self.model = model
        self.num_timesteps = model.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        verbose=True,
        eta=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs,
    ):
        device = self.model.betas.device
        dtype = next(self.model.parameters()).dtype
        latents = torch.randn((batch_size, *shape), device=device, dtype=dtype)

        schedule = make_lcm_schedule(S, num_train_timesteps=self.num_timesteps)
        alphas_cumprod = self.model.alphas_cumprod.to(device)

        for i, t in enumerate(schedule):
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                model_output = self.model.apply_model(latents, ts, conditioning)
            else:
                model_t = self.model.apply_model(latents, ts, conditioning)
                model_uncond = self.model.apply_model(latents, ts, unconditional_conditioning)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            if self.model.parameterization == "v":
                eps = self.model.predict_eps_from_z_and_v(latents, ts, model_output)
            else:
                eps = model_output

            t_prev = schedule[i + 1] if i + 1 < len(schedule) else 0
            t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            latents = ddim_step(latents, ts, t_prev_tensor, eps, alphas_cumprod)

        return latents, {}


class AnyText2StudentModel(AnyText2Model):
    def __init__(
        self,
        model_dir,
        *args,
        student_lora_path=None,
        merge_student_lora=True,
        use_lcm_sampler=True,
        **kwargs,
    ):
        self.student_lora_path = student_lora_path
        self.merge_student_lora = merge_student_lora
        self.use_lcm_sampler = use_lcm_sampler
        self._student_lora_applied = False
        super().__init__(model_dir, *args, **kwargs)
        self._refresh_sampler()

    def _refresh_sampler(self):
        if self.use_lcm_sampler:
            self.ddim_sampler = LCMSampler(self.model)
        else:
            self.ddim_sampler = DDIMSampler(self.model)

    def _apply_student_lora(self):
        if not self.student_lora_path or self._student_lora_applied:
            return
        try:
            lora_model = PeftModel.from_pretrained(
                self.model,
                self.student_lora_path,
                is_trainable=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "LoRA injection failed. zero_convs are Conv2d modules and must be supported. "
                "Please upgrade peft/transformers to a version that supports Conv2d LoRA."
            ) from exc

        if self.merge_student_lora:
            if not hasattr(lora_model, "merge_and_unload"):
                raise RuntimeError(
                    "peft does not support merge_and_unload. Please upgrade peft to use Conv2d LoRA."
                )
            try:
                self.model = lora_model.merge_and_unload()
            except Exception as exc:
                raise RuntimeError(
                    "LoRA merge failed. Please upgrade peft/transformers to a version that supports Conv2d LoRA."
                ) from exc
        else:
            self.model = lora_model

        self.model.eval()
        self._student_lora_applied = True
        self._refresh_sampler()

    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        self._refresh_sampler()

    def load_weights(self):
        self._student_lora_applied = False
        super().load_weights()
        self._apply_student_lora()

    def load_base_model(self, model_path):
        self._student_lora_applied = False
        super().load_base_model(model_path)
        self._apply_student_lora()
