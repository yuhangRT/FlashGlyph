"""
LCM Utilities for AnyText2 Distillation

This module contains LCM-specific functions:
- Coarse timestep sampling for LCM training
- DDIM solver for teacher target computation
- Noise prediction to x0 prediction conversion
- Conditional batch preparation for CFG

Math Reference:
    LCM (Latent Consistency Model) distillation learns to predict x_0 directly
    from noisy latents x_t in a single step, using a teacher's multi-step predictions.

    DDIM Solver (for teacher target):
        Given noise prediction ε_θ(x_t, t), predict clean image x_0:
        x_0 = (x_t - sqrt(1-α_t) * ε_θ) / sqrt(α_t)

    Student learns:
        x_0 ≈ f_θ(x_t, t, cond)  [consistency property]

    Teacher provides:
        Multi-step refined x_0 as supervision signal
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple


# Coarse timestep schedules for different inference step counts
LCM_SCHEDULES = {
    4: [999, 599, 299, 50],
    6: [999, 799, 599, 399, 199, 50],
    8: [999, 899, 799, 699, 599, 499, 399, 50],
    16: [999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 499, 449, 399, 349, 299, 50],
}


def get_coarse_timesteps(
    batch_size: int,
    device: torch.device,
    num_inference_steps: int = 8,
    state: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample coarse timesteps for LCM training.

    LCM uses discrete timesteps from a coarse schedule rather than uniform sampling.
    This allows the student to learn consistency at key diffusion stages.

    Args:
        batch_size: Number of samples
        device: Device to create tensors on
        num_inference_steps: Target number of inference steps (4, 6, 8, or 16)
        state: Random state generator

    Returns:
        timesteps: (batch_size,) tensor of timesteps

    Example:
        >>> t = get_coarse_timesteps(4, device='cuda', num_inference_steps=8)
        >>> print(t)
        tensor([999, 799, 599, 399])  # Sampled from coarse schedule
    """
    if num_inference_steps not in LCM_SCHEDULES:
        raise ValueError(f"num_inference_steps must be one of {list(LCM_SCHEDULES.keys())}, got {num_inference_steps}")

    schedule = LCM_SCHEDULES[num_inference_steps]

    # Sample random timesteps from the schedule
    indices = torch.randint(0, len(schedule), (batch_size,), generator=state)
    timesteps = torch.tensor([schedule[i] for i in indices], device=device)

    return timesteps.long()


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """
    Extract values from a 1D tensor at indices t and reshape to match x_shape.

    Args:
        a: 1D tensor (e.g., alphas_cumprod)
        t: Indices to extract
        x_shape: Target shape for broadcasting

    Returns:
        Extracted values reshaped to x_shape
    """
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise_pred: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    scheduler_type: str = "ddim"
) -> torch.Tensor:
    """
    Convert noise prediction to x0 prediction using DDIM formula.

    Math:
        Given:
            x_t: Noisy latent at timestep t
            ε_θ: Predicted noise
            α_t: Cumproduct of alphas at timestep t

        DDIM x0 prediction:
            pred_x0 = (x_t - sqrt(1-α_t) * ε_θ) / sqrt(α_t)

    Args:
        x_t: Noisy latents (B, C, H, W)
        t: Timesteps (B,)
        noise_pred: Predicted noise (B, C, H, W)
        alphas_cumprod: Cumprod alphas (T,) from scheduler
        scheduler_type: Type of scheduler (default: ddim)

    Returns:
        pred_x0: Predicted clean image (B, C, H, W)
    """
    # Extract alpha values at timestep t
    alpha_t = extract_into_tensor(alphas_cumprod, t, x_t.shape)

    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

    # DDIM formula: x0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
    pred_x0 = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

    return pred_x0


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to samples using DDPM forward process.

    Math:
        q(x_t | x_0) = N(x_t; sqrt(α_t) * x_0, (1-α_t) * I)

        x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε

    Args:
        original_samples: Clean samples (B, C, H, W)
        noise: Noise to add (B, C, H, W)
        timesteps: Timesteps (B,)
        alphas_cumprod: Cumprod alphas (T,)

    Returns:
        noisy_samples: Samples with added noise (B, C, H, W)
    """
    sqrt_alpha_prod = extract_into_tensor(torch.sqrt(alphas_cumprod), timesteps, original_samples.shape)
    sqrt_one_minus_alpha_prod = extract_into_tensor(torch.sqrt(1.0 - alphas_cumprod), timesteps, original_samples.shape)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def prepare_conditional_batch(
    batch: dict,
    model,
    device: torch.device
) -> Tuple[dict, dict]:
    """
    Prepare conditional and unconditional batches for Classifier-Free Guidance (CFG).

    For CFG, we need:
    1. Conditional: Real text, real glyphs, real positions
    2. Unconditional: Null text, null glyphs, null positions

    Args:
        batch: Input batch with text, glyphs, positions, etc.
        model: AnyText2 model (for getting null conditioning)
        device: Device to create tensors on

    Returns:
        cond_batch: Conditional batch dictionary
        uncond_batch: Unconditional batch dictionary

    Example:
        >>> cond, uncond = prepare_conditional_batch(batch, model, device)
        >>> # cond['text_info']['glyphs'] contains real glyphs
        >>> # uncond['text_info']['glyphs'] contains null glyphs (zeros)
    """
    batch_size = batch['img'].shape[0]

    # Conditional batch: use real data
    cond_batch = {
        'img': batch['img'].to(device),
        'hint': batch['hint'].to(device),
        'glyphs': [g.to(device) for g in batch['glyphs']],
        'gly_line': [g.to(device) for g in batch['gly_line']],
        'positions': [p.to(device) for p in batch['positions']],
        'masked_x': batch['masked_x'].to(device),
        'img_caption': batch['img_caption'],
        'text_caption': batch['text_caption'],
        'texts': batch['texts'],
        'n_lines': batch['n_lines'],
        'font_hint': batch['font_hint'].to(device),
        'color': [c.to(device) for c in batch['color']],
        'language': batch['language'],
        'inv_mask': batch['inv_mask'].to(device),
    }

    # Unconditional batch: null/empty values
    # Null glyphs: zero tensors
    null_glyphs = [torch.zeros_like(g).to(device) for g in batch['glyphs']]

    # Null positions: zero tensors
    null_positions = [torch.zeros_like(p).to(device) for p in batch['positions']]

    # Null hint: zero tensor
    null_hint = torch.zeros_like(batch['hint']).to(device)

    # Null gly_line: zero tensors
    null_gly_line = [torch.zeros_like(g).to(device) for g in batch['gly_line']]

    # Null masked_x: zeros
    null_masked_x = torch.zeros_like(batch['masked_x']).to(device)

    # Null font_hint: zeros
    null_font_hint = torch.zeros_like(batch['font_hint']).to(device)

    # Null inv_mask: ones (fully unmasked)
    null_inv_mask = torch.ones_like(batch['inv_mask']).to(device)

    # Null captions: empty strings
    batch_size = batch['img'].shape[0]
    null_img_caption = [""] * batch_size
    null_text_caption = [""] * batch_size

    # Null texts: empty strings for each line
    max_lines = len(batch['texts'])
    null_texts = [[""] * batch_size for _ in range(max_lines)]

    # Null n_lines: zeros
    null_n_lines = torch.zeros(batch_size, dtype=batch['n_lines'].dtype).to(device)

    # Null colors: gray (0.5, 0.5, 0.5)
    null_colors = [torch.ones(batch_size, 3).to(device) * 0.5 for _ in range(max_lines)]

    # Null language: empty string
    null_language = [""] * batch_size

    uncond_batch = {
        'img': batch['img'].to(device),  # Same image
        'hint': null_hint,
        'glyphs': null_glyphs,
        'gly_line': null_gly_line,
        'positions': null_positions,
        'masked_x': null_masked_x,
        'img_caption': null_img_caption,
        'text_caption': null_text_caption,
        'texts': null_texts,
        'n_lines': null_n_lines,
        'font_hint': null_font_hint,
        'color': null_colors,
        'language': null_language,
        'inv_mask': null_inv_mask,
    }

    return cond_batch, uncond_batch


def apply_cfg(
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    guidance_scale: float = 7.5
) -> torch.Tensor:
    """
    Apply Classifier-Free Guidance (CFG).

    Math:
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    Args:
        noise_pred_cond: Conditional noise prediction (B, C, H, W)
        noise_pred_uncond: Unconditional noise prediction (B, C, H, W)
        guidance_scale: Guidance scale (default: 7.5)

    Returns:
        noise_pred: CFG-combined noise prediction (B, C, H, W)
    """
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


def compute_lcm_loss(
    pred_x0_student: torch.Tensor,
    target_x0_teacher: torch.Tensor,
    loss_type: str = "huber"
) -> torch.Tensor:
    """
    Compute LCM distillation loss.

    Args:
        pred_x0_student: Student's predicted x0 (B, C, H, W)
        target_x0_teacher: Teacher's target x0 (B, C, H, W)
        loss_type: Type of loss ('huber' or 'mse')

    Returns:
        loss: Scalar loss tensor
    """
    if loss_type == "huber":
        # Huber loss is more robust to outliers
        loss = F.huber_loss(pred_x0_student, target_x0_teacher, delta=1.0)
    elif loss_type == "mse":
        loss = F.mse_loss(pred_x0_student, target_x0_teacher)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


class LCMScheduler:
    """
    Simple scheduler for LCM training with DDIM noise schedule.

    Wraps alpha values needed for noise addition and x0 prediction.
    """

    def __init__(self, alphas_cumprod: torch.Tensor, device: torch.device):
        """
        Args:
            alphas_cumprod: Cumprod alphas from diffusion scheduler (T,)
            device: Device to keep tensors on
        """
        self.device = device  # Set device first before register_buffer
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))

        # Precompute common values
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer (like nn.Module)."""
        setattr(self, name, tensor.to(self.device))

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples."""
        return add_noise(original_samples, noise, timesteps, self.alphas_cumprod)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """Predict x0 from noise prediction."""
        return predict_x0(x_t, t, noise_pred, self.alphas_cumprod)


def create_lcm_scheduler_from_anytext(model) -> LCMScheduler:
    """
    Create LCMScheduler from AnyText2 model.

    Args:
        model: ControlLDM model instance

    Returns:
        LCMScheduler instance
    """
    # Extract alphas_cumprod from model
    alphas_cumprod = model.alphas_cumprod

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return LCMScheduler(alphas_cumprod, device)


# Example usage
if __name__ == "__main__":
    # Test coarse timestep sampling
    print("Testing coarse timestep sampling...")
    t = get_coarse_timesteps(8, device='cpu', num_inference_steps=8)
    print(f"Timesteps: {t}")

    # Test noise addition and x0 prediction
    print("\nTesting noise addition and x0 prediction...")

    # Mock alphas_cumprod (1000 timesteps, linear schedule)
    alphas = torch.linspace(0.0001, 0.02, 1000)
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Create scheduler
    scheduler = LCMScheduler(alphas_cumprod, device='cpu')

    # Mock data
    B, C, H, W = 2, 4, 64, 64
    x0 = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)
    t = torch.tensor([999, 500])

    # Add noise
    x_t = scheduler.add_noise(x0, noise, t)
    print(f"x0 shape: {x0.shape}")
    print(f"x_t shape: {x_t.shape}")

    # Predict x0 from noise
    pred_x0 = scheduler.predict_x0(x_t, t, noise)
    print(f"pred_x0 shape: {pred_x0.shape}")
    print(f"x0 prediction error: {F.mse_loss(pred_x0, x0).item():.6f}")

    print("\n✓ All tests passed!")
