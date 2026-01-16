import torch
import torch.nn.functional as F


def make_lcm_schedule(num_inference_steps, num_train_timesteps=1000):
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")
    if num_inference_steps >= num_train_timesteps:
        return list(range(num_train_timesteps - 1, -1, -1))

    step_ratio = (num_train_timesteps - 1) / (num_inference_steps - 1)
    timesteps = [
        int(round(num_train_timesteps - 1 - i * step_ratio))
        for i in range(num_inference_steps)
    ]
    timesteps[-1] = 0
    timesteps = sorted(set(timesteps), reverse=True)

    if len(timesteps) < num_inference_steps:
        for t in range(num_train_timesteps - 1, -1, -1):
            if t not in timesteps:
                timesteps.append(t)
            if len(timesteps) >= num_inference_steps:
                break
        timesteps = sorted(timesteps, reverse=True)[:num_inference_steps]

    return timesteps


def sample_timesteps(schedule, batch_size, device):
    schedule_tensor = torch.tensor(schedule, device=device)
    indices = torch.randint(0, schedule_tensor.shape[0], (batch_size,), device=device)
    return schedule_tensor[indices]


def extract_into_tensor(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def add_noise(x0, noise, timesteps, alphas_cumprod):
    sqrt_alpha = extract_into_tensor(torch.sqrt(alphas_cumprod), timesteps, x0.shape)
    sqrt_one_minus_alpha = extract_into_tensor(torch.sqrt(1.0 - alphas_cumprod), timesteps, x0.shape)
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    alpha_t = extract_into_tensor(alphas_cumprod, t, x_t.shape)
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
    return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha


def ddim_step(x_t, t, t_prev, eps, alphas_cumprod):
    if t_prev is None:
        return predict_x0_from_eps(x_t, t, eps, alphas_cumprod)

    alpha_t = extract_into_tensor(alphas_cumprod, t, x_t.shape)
    alpha_prev = extract_into_tensor(alphas_cumprod, t_prev, x_t.shape)
    pred_x0 = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)

    dir_xt = torch.sqrt(1.0 - alpha_prev) * eps
    x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
    return x_prev


def apply_cfg(noise_pred_cond, noise_pred_uncond, guidance_scale=7.5):
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


def compute_lcm_loss(pred_x0_student, target_x0_teacher, loss_type="huber"):
    if loss_type == "huber":
        return F.huber_loss(pred_x0_student, target_x0_teacher, delta=1.0)
    if loss_type == "mse":
        return F.mse_loss(pred_x0_student, target_x0_teacher)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def prepare_conditional_batch(batch, device, non_blocking=False):
    batch_size = batch["img"].shape[0]

    cond_batch = {
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

    null_glyphs = [torch.zeros_like(g, device=device) for g in batch["glyphs"]]
    null_positions = [torch.zeros_like(p, device=device) for p in batch["positions"]]
    null_hint = torch.zeros_like(batch["hint"], device=device)
    null_gly_line = [torch.zeros_like(g, device=device) for g in batch["gly_line"]]
    null_masked_x = torch.zeros_like(batch["masked_x"], device=device)
    null_font_hint = torch.zeros_like(batch["font_hint"], device=device)
    null_inv_mask = torch.ones_like(batch["inv_mask"], device=device)

    null_img_caption = [""] * batch_size
    null_text_caption = [""] * batch_size
    max_lines = len(batch["texts"])
    null_texts = [[""] * batch_size for _ in range(max_lines)]
    null_n_lines = torch.zeros(batch_size, dtype=batch["n_lines"].dtype, device=device)
    null_colors = [torch.ones(batch_size, 3, device=device) * 0.5 for _ in range(max_lines)]
    null_language = [""] * batch_size

    uncond_batch = {
        "img": batch["img"].to(device),
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

    return cond_batch, uncond_batch


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timesteps, sigma_data=0.5, timestep_scaling=10.0):
    # Match diffusers LCMScheduler boundary scaling (timestep / 0.1 -> timestep * 10).
    scaled_t = timesteps.float() * float(timestep_scaling)
    sigma_data = float(sigma_data)
    denom = scaled_t ** 2 + sigma_data ** 2
    c_skip = (sigma_data ** 2) / denom
    c_out = scaled_t / torch.sqrt(denom)
    return c_skip, c_out


def predict_x0_from_model_output(x_t, t, model_output, alphas_cumprod, parameterization="eps"):
    if parameterization == "eps":
        return predict_x0_from_eps(x_t, t, model_output, alphas_cumprod)
    if parameterization == "x0":
        return model_output
    if parameterization == "v":
        alpha_t = extract_into_tensor(alphas_cumprod, t, x_t.shape)
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
    raise ValueError(f"Unknown parameterization: {parameterization}")


def predict_eps_from_model_output(x_t, t, model_output, alphas_cumprod, parameterization="eps"):
    if parameterization == "eps":
        return model_output
    alpha_t = extract_into_tensor(alphas_cumprod, t, x_t.shape)
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
    if parameterization == "x0":
        return (x_t - sqrt_alpha * model_output) / sqrt_one_minus_alpha
    if parameterization == "v":
        return sqrt_alpha * model_output + sqrt_one_minus_alpha * x_t
    raise ValueError(f"Unknown parameterization: {parameterization}")


def sample_timestep_pairs(schedule, batch_size, device, step_stride=1):
    if step_stride < 1:
        raise ValueError("step_stride must be >= 1")
    if len(schedule) <= step_stride:
        raise ValueError("schedule length must be > step_stride")
    schedule_tensor = torch.tensor(schedule, device=device)
    max_index = schedule_tensor.shape[0] - step_stride
    indices = torch.randint(0, max_index, (batch_size,), device=device)
    t_start = schedule_tensor[indices]
    t_next = schedule_tensor[indices + step_stride]
    return t_start, t_next
