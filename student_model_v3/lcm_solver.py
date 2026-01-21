import numpy as np
import torch


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type in ("epsilon", "eps"):
        sigmas_t = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas_t = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x0 = (sample - sigmas_t * model_output) / alphas_t
        return pred_x0
    if prediction_type in ("v_prediction", "v"):
        sigmas_t = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas_t = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x0 = alphas_t * sample - sigmas_t * model_output
        return pred_x0
    if prediction_type == "x0":
        return model_output
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def scalings_for_boundary_conditions(timestep, sigma_data=0.5):
    # Matches LCMScheduler.get_scalings_for_boundary_condition_discrete in diffusers.
    scaled_t = timestep / 0.1
    c_skip = sigma_data**2 / (scaled_t**2 + sigma_data**2)
    c_out = scaled_t / (scaled_t**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )

        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
