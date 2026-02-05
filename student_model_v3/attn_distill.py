import logging
from typing import Iterable, List, Optional

from ldm.modules.attention import SpatialTransformer, CrossAttention


logger = logging.getLogger(__name__)


def collect_attn_modules(unet) -> List[CrossAttention]:
    modules: List[CrossAttention] = []
    for module in unet.modules():
        if isinstance(module, SpatialTransformer):
            for block in module.transformer_blocks:
                if hasattr(block, "attn2") and isinstance(block.attn2, CrossAttention):
                    modules.append(block.attn2)
                if hasattr(block, "attn2x") and isinstance(block.attn2x, CrossAttention):
                    modules.append(block.attn2x)
    return modules


def set_attn_recording(
    modules: Iterable[CrossAttention],
    token_mask_spec: Optional[dict],
    hw_allowlist: Optional[Iterable[int]] = None,
    record: bool = True,
) -> None:
    allow = set(hw_allowlist) if hw_allowlist else None
    for module in modules:
        module._record_attn = bool(record)
        module._token_mask_spec = token_mask_spec
        module._attn_hw_allowlist = allow
        module._last_mass = None


def gather_attn_mass(modules: Iterable[CrossAttention]) -> List[Optional[object]]:
    masses: List[Optional[object]] = []
    for module in modules:
        if getattr(module, "_record_attn", False):
            masses.append(module._last_mass)
            module._last_mass = None
    return masses


def resolve_unet_for_attn(model):
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        return model.model.diffusion_model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(
        model.base_model.model, "diffusion_model"
    ):
        return model.base_model.model.diffusion_model
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model
    logger.warning("Falling back to the provided model for attention traversal.")
    return model


def resolve_control_for_attn(model):
    if hasattr(model, "control_model"):
        return model.control_model
    if hasattr(model, "base_model") and hasattr(model.base_model, "control_model"):
        return model.base_model.control_model
    if hasattr(model, "model") and hasattr(model.model, "control_model"):
        return model.model.control_model
    logger.warning("Cannot resolve control_model for attn traversal, fallback to provided model.")
    return model
