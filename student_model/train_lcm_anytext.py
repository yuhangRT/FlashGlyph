"""
LCM-LoRA Distillation Training for AnyText2 - FIXED VERSION

This script implements LCM (Latent Consistency Model) distillation with LoRA for the AnyText2 model.
The student model learns to accelerate both background generation (UNet) and text generation
(ControlNet/Glyph branches) using LoRA-efficient fine-tuning.

Key Features:
- Teacher-student distillation with CFG
- LoRA injection for both UNet and ControlNet
- Coarse timestep sampling for LCM training
- 4-8 step inference capability

CRITICAL FIXES in this version:
1. Physical Isolation: Separate model instances for teacher/student (no deepcopy)
2. State Surgery: Forced cache clearing before/after every forward
3. Recursive Detach: Deep graph isolation for nested data structures
4. Explicit State Management: Reset is_uncond flag to prevent phase misalignment

Usage:
    python train_lcm_anytext.py \\
        --config models_yaml/anytext2_sd15.yaml \\
        --teacher_ckpt models/anytext_v2.0.ckpt \\
        --output_dir ./student_model/checkpoints

Reference: "Latent Consistency Models: Image Synthesis in a Few Steps"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Accelerate for distributed training
from accelerate import Accelerator
from accelerate.utils import set_seed

# PEFT for LoRA
from peft import LoraConfig, get_peft_model

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# AnyText2 imports
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM

# Local imports
from student_model.dataset_anytext import AnyTextMockDataset, RealAnyTextDataset, collate_fn_anytext
from student_model.lcm_utils import (
    get_coarse_timesteps,
    prepare_conditional_batch,
    apply_cfg,
    compute_lcm_loss,
    create_lcm_scheduler_from_anytext,
)

# =========================================================================
# Helper: Recursive Detach (Crucial for Graph Isolation)
# =========================================================================
def detach_recursive(obj):
    """
    Recursively detach and clone objects to break computation graphs.
    Crucial for AnyText's nested dict/list structures.

    This function handles:
    - Tensors: detach() + clone() to break graph
    - Dicts: recursively process all values
    - Lists: recursively process all items
    - Tuples: recursively process and convert to tuple
    - Other types: preserve as-is (str, int, float, None, etc.)
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_recursive(v) for v in obj)
    else:
        # Preserve non-tensor objects (str, int, float, None, etc.)
        return obj

# =========================================================================
# Wrapper Class with STATE SURGERY
# =========================================================================
class AnyText2ForwardWrapper:
    """
    Wrapper to simplify AnyText2 forward pass for LCM training.
    Includes aggressive state cleaning to prevent graph leaks in ControlLDM.
    """

    def __init__(self, model: ControlLDM, device: torch.device):
        self.model = model
        self.model.to(device)
        self.device = device

        # [SURGERY] Disable AnyText's internal caching mechanisms
        # Based on cldm.py source code analysis
        if hasattr(self.model, 'control_model'):
            self.model.control_model.fast_control = False  # Disable optimization that relies on cache

        # [CRITICAL FIX] Monkey-patch apply_model to completely prevent caching
        # This prevents the graph leak caused by ControlNet's caching mechanism
        self._patch_apply_model()

        # [EXTRA AGGRESSIVE] Also clear all control-related state immediately
        self.model.control = None
        self.model.control_uncond = None
        self.model.is_uncond = False

    def reset_state(self):
        """
        [CRITICAL FIX] Force reset all internal buffers in ControlLDM to prevent graph leaks.
        Call this BEFORE every forward pass.

        Based on cldm/cldm.py:409-411, these are the critical state variables:
        - self.control: Caches conditional control signal
        - self.control_uncond: Caches unconditional control signal
        - self.is_uncond: Flag that toggles between cond/uncond mode
        """
        self.model.control = None
        self.model.control_uncond = None
        self.model.is_uncond = False  # Reset flag to prevent phase misalignment

    def _patch_apply_model(self):
        """
        [CRITICAL FIX] Monkey-patch ControlLDM.apply_model to completely bypass caching.

        The original caching mechanism at cldm/cldm.py:524-538 is incompatible with
        teacher-student distillation because it causes graph leaks.

        This patch forces recomputation of control tensors on EVERY forward pass,
        preventing any graph connections from lingering across calls.
        """
        # Handle PEFT wrapping: after LoRA injection, model might be wrapped in PeftModel
        # We need to get the base ControlLDM model
        # Structure:
        # - Teacher (no PEFT): ControlLDM
        # - Student (with PEFT): PeftModel -> base_model -> ControlLDM -> model -> DiffusionWrapper
        # KEY: We want ControlLDM, NOT DiffusionWrapper!

        from cldm.cldm import ControlLDM

        # Step 1: Check if self.model is already ControlLDM (teacher case)
        if isinstance(self.model, ControlLDM):
            # Teacher model - no PEFT wrapping
            base_model = self.model
        elif hasattr(self.model, 'base_model'):
            # Student model - PEFT wrapped, need to unwrap further
            # PEFT structure: PeftModel -> LoraModel -> ControlLDM
            base_model = self.model.base_model
            while hasattr(base_model, 'base_model') or hasattr(base_model, 'model'):
                if hasattr(base_model, 'model') and isinstance(base_model.model, ControlLDM):
                    base_model = base_model.model
                    break
                elif hasattr(base_model, 'base_model'):
                    base_model = base_model.base_model
                else:
                    break
        else:
            # Unknown structure
            raise TypeError(f"Model structure not recognized: {type(self.model)}")

        # Step 2: Verify we have ControlLDM
        if not isinstance(base_model, ControlLDM):
            raise TypeError(f"Model unwrapping failed: expected ControlLDM, got {type(base_model)}")

        # Store reference and get the unbound method from the actual ControlLDM class
        model_instance = base_model
        original_method = ControlLDM.apply_model  # Get unbound method from ControlLDM class

        def patched_apply_model(self, x_noisy, t, cond, *args, **kwargs):
            """
            Wrapper that forces recomputation of control by clearing cache
            before and after the original apply_model call.
            """
            # CRITICAL: Clear all cached control tensors BEFORE forward
            # This forces apply_model to recompute instead of using cache
            model_instance.control = None
            model_instance.control_uncond = None
            model_instance.is_uncond = False

            # Call original apply_model with explicit self
            eps = original_method(model_instance, x_noisy, t, cond, *args, **kwargs)

            # CRITICAL: Clear cache AGAIN AFTER forward
            # This prevents the graph from being retained in self.control
            model_instance.control = None
            model_instance.control_uncond = None
            model_instance.is_uncond = False

            return eps

        # Bind the patched function to the base model instance
        import types
        model_instance.apply_model = types.MethodType(patched_apply_model, model_instance)

    def encode_text(self, batch: dict, text_info: dict = None) -> dict:
        """
        Encode text captions using CLIP encoder.

        Args:
            batch: Input batch with captions
            text_info: Optional text_info dict for embedding_manager

        Returns:
            Encoded conditioning dict
        """
        # Prepare conditioning for AnyText2
        cond = {
            'c_crossattn': [[batch['img_caption'], batch['text_caption']]],
            'text_info': text_info  # Pass text_info so embedding_manager can be initialized
        }

        # Get learned conditioning
        with torch.no_grad():
            c = self.model.get_learned_conditioning(cond)

        return c

    def prepare_text_info(self, batch: dict) -> dict:
        """
        Prepare text_info dict for AnyText2 forward.

        Args:
            batch: Input batch

        Returns:
            text_info dict
        """
        text_info = {
            'glyphs': batch['glyphs'],
            'positions': batch['positions'],
            'colors': batch['color'],
            'n_lines': batch['n_lines'],
            'language': batch['language'],
            'texts': batch['texts'],
            'img': batch['img'],  # (B, H, W, 3) NHWC
            'masked_x': batch['masked_x'],
            'gly_line': batch['gly_line'],
            'inv_mask': batch['inv_mask'],
            'font_hint': batch['font_hint'],
        }
        return text_info

    def forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        text_emb: dict,
        text_info: dict,
        hint: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through AnyText2 model with automatic state cleaning.

        Args:
            latents: Noisy latents (B, C, H, W)
            t: Timesteps (B,)
            text_emb: Encoded text embeddings from encode_text()
            text_info: Text info dict from prepare_text_info()
            hint: Control hint (B, H, W, C) - position mask for text rendering

        Returns:
            noise_pred: Predicted noise (B, C, H, W)
        """
        # Prepare conditioning dict for apply_model
        cond = {
            'c_concat': [hint],  # Add hint as list format for ControlNet
            'c_crossattn': text_emb['c_crossattn'],
            'text_info': text_info
        }

        # Forward pass
        # NOTE: If model is PEFT-wrapped, access the base ControlLDM's apply_model
        # Use the same unwrapping logic as in _patch_apply_model

        from cldm.cldm import ControlLDM

        if isinstance(self.model, ControlLDM):
            # Teacher model - already ControlLDM
            target_model = self.model
        elif hasattr(self.model, 'base_model'):
            # Student model - PEFT wrapped, unwrap to get ControlLDM
            target_model = self.model.base_model
        else:
            # Fallback - use as is
            target_model = self.model

        noise_pred = target_model.apply_model(latents, t, cond)

        return noise_pred

# =========================================================================
# Training Step
# =========================================================================
def training_step(
    teacher_wrapper: AnyText2ForwardWrapper,
    student_wrapper: AnyText2ForwardWrapper,
    batch: dict,
    scheduler,
    accelerator: Accelerator,
    cfg_scale: float = 7.5,
    num_inference_steps: int = 8
) -> Dict[str, torch.Tensor]:
    """
    Perform one LCM distillation training step with RECURSIVE DETACH.

    Args:
        teacher_wrapper: Teacher model wrapper
        student_wrapper: Student model wrapper
        batch: Input batch
        scheduler: LCMScheduler
        accelerator: Accelerator instance
        cfg_scale: CFG scale (default: 7.5)
        num_inference_steps: Target inference steps

    Returns:
        Dictionary with loss and metrics
    """
    batch_size = batch['img'].shape[0]
    device = accelerator.device

    # =========================================================================
    # A. Teacher Phase (Generate Targets & Conditions) - NO GRAD
    # =========================================================================
    with torch.no_grad():
        # 1. Encode Images to Latents
        img_nhwc = batch['img']
        img_nchw = img_nhwc.permute(0, 3, 1, 2)
        latent_dist = teacher_wrapper.model.first_stage_model.encode(img_nchw)
        latents = latent_dist.sample() * teacher_wrapper.model.scale_factor

        # 2. Sample Timesteps & Add Noise
        t = get_coarse_timesteps(batch_size, device, num_inference_steps)
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, t)

        # 3. Prepare Conditions (The source of the graph leak)
        cond_batch, uncond_batch = prepare_conditional_batch(batch, teacher_wrapper.model, device)

        # 3a. Prepare Text Info (Complex Dicts)
        cond_text_info = teacher_wrapper.prepare_text_info(cond_batch)
        uncond_text_info = teacher_wrapper.prepare_text_info(uncond_batch)

        # 3b. Encode Text (Returns Complex Dict of Tensors)
        cond_text_emb = teacher_wrapper.encode_text(cond_batch, cond_text_info)
        uncond_text_emb = teacher_wrapper.encode_text(uncond_batch, uncond_text_info)

        # 3c. Hints
        cond_hint = cond_batch['hint']
        uncond_hint = uncond_batch['hint']

        # 4. Teacher Forward (Predict Target)
        noise_pred_cond = teacher_wrapper.forward(
            noisy_latents, t, cond_text_emb, cond_text_info, cond_hint
        )
        noise_pred_uncond = teacher_wrapper.forward(
            noisy_latents, t, uncond_text_emb, uncond_text_info, uncond_hint
        )
        noise_pred_teacher = apply_cfg(noise_pred_cond, noise_pred_uncond, cfg_scale)
        target_x0 = scheduler.predict_x0(noisy_latents, t, noise_pred_teacher)

    # =========================================================================
    # B. The Firewall: Recursive Detach (Fixing the Root Cause)
    # =========================================================================
    # We package everything needed for the Student into a clean dictionary
    # and run detach_recursive on it. This creates deep copies of all tensors.
    student_inputs = {
        'noisy_latents': detach_recursive(noisy_latents),
        't': detach_recursive(t),
        'cond_text_emb': detach_recursive(cond_text_emb),   # <--- 关键点：深度切断复杂字典
        'cond_text_info': detach_recursive(cond_text_info), # <--- 关键点：深度切断复杂字典
        'hint': detach_recursive(cond_hint),
        'target_x0': detach_recursive(target_x0)
    }

    # =========================================================================
    # C. Student Phase - ENABLE GRAD
    # =========================================================================
    # Explicitly enable grad ensures we are building the student's graph correctly
    with torch.set_grad_enabled(True):
        noise_pred_student = student_wrapper.forward(
            student_inputs['noisy_latents'],
            student_inputs['t'],
            student_inputs['cond_text_emb'],
            student_inputs['cond_text_info'],
            student_inputs['hint']
        )

        pred_x0_student = scheduler.predict_x0(
            student_inputs['noisy_latents'],
            student_inputs['t'],
            noise_pred_student
        )

        loss = compute_lcm_loss(
            pred_x0_student,
            student_inputs['target_x0'],
            loss_type="huber"
        )

    return {
        'loss': loss,
        'pred_x0_student': pred_x0_student,
        'target_x0': student_inputs['target_x0'],
    }

# =========================================================================
# Utilities (Config, Freeze, etc.)
# =========================================================================
def create_lora_config(
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """
    Create LoRA configuration for AnyText2.

    Args:
        rank: LoRA rank (default: 64)
        alpha: LoRA alpha (default: 64)
        dropout: LoRA dropout (default: 0.0)
        target_modules: List of target module names

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default target modules for AnyText2
        # These should match the output from inspect_modules.py
        target_modules = []

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="DIFFUSION",  # PEFT will handle this
    )


def freeze_model(model: torch.nn.Module):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="LCM-LoRA training for AnyText2")

    # Model arguments
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml",
                       help="Path to model config")
    parser.add_argument("--teacher_ckpt", type=str, default="models/anytext_v2.0.ckpt",
                       help="Path to teacher checkpoint")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    parser.add_argument("--target_modules_file", type=str, default="student_model/target_modules_list.txt",
                       help="Path to target modules list from inspect_modules.py")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./student_model/checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Image resolution")
    parser.add_argument("--train_batch_size", type=int, default=2,
                       help="Train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"], help="Mixed precision")

    # LCM arguments
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=8,
                       choices=[4, 6, 8, 16], help="Target inference steps")

    # Dataset arguments
    parser.add_argument("--use_mock_dataset", action="store_true",
                       help="Use mock dataset for testing (default: use real dataset)")
    parser.add_argument("--use_real_dataset", action="store_true",
                       help="Use real demodataset for training")
    parser.add_argument("--dataset_json", type=str,
                       default="demodataset/annotations/demo_data.json",
                       help="Path to dataset JSON file for real dataset")
    parser.add_argument("--dataset_size", type=int, default=100,
                       help="Size of mock dataset")
    parser.add_argument("--max_lines", type=int, default=5,
                       help="Maximum number of text lines per image")
    parser.add_argument("--max_chars", type=int, default=20,
                       help="Maximum number of characters per text line")
    parser.add_argument("--font_path", type=str, default="./font/Arial_Unicode.ttf",
                       help="Path to font file for text rendering")
    parser.add_argument("--font_hint_prob", type=float, default=0.8,
                       help="Probability of using font hints")
    parser.add_argument("--font_hint_area", type=float, nargs=2, default=[0.7, 1.0],
                       help="Range of area to preserve for font hints [min, max]")
    parser.add_argument("--color_prob", type=float, default=1.0,
                       help="Probability of using color labels")
    parser.add_argument("--wm_thresh", type=float, default=1.0,
                       help="Watermark filtering threshold (1.0 = filter all)")
    parser.add_argument("--glyph_scale", type=float, default=0.7,
                       help="Scale factor for glyph rendering")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save checkpoint every N steps")

    args = parser.parse_args()

    # =========================================================================
    # Path Resolution
    # =========================================================================
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent

    config_path = Path(args.config)
    ckpt_path = Path(args.teacher_ckpt)
    target_modules_path = Path(args.target_modules_file)

    # If path is not absolute, make it relative to parent directory
    if not config_path.is_absolute():
        config_path = (parent_dir / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (parent_dir / ckpt_path).resolve()
    if not target_modules_path.is_absolute():
        target_modules_path = (parent_dir / target_modules_path).resolve()

    # =========================================================================
    # Accelerator Setup
    # =========================================================================
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    set_seed(args.seed)

    # =========================================================================
    # Load Target Modules
    # =========================================================================
    if os.path.exists(target_modules_path):
        with open(target_modules_path, 'r') as f:
            content = f.read()
            # Parse the list (simple approach)
            target_modules = eval(content[content.find('['):])
        print(f"Loaded {len(target_modules)} target modules from {target_modules_path}")
    else:
        print(f"Warning: {args.target_modules_file} not found!")
        print("Please run inspect_modules.py first to generate target modules list.")
        print("Using default empty list - please specify target_modules manually.")
        target_modules = []

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 1. Load Teacher (Physical Instance 1)
    # =========================================================================
    print(f"\n{'='*80}\nLoading Teacher (Instance 1)...\n{'='*80}")
    teacher = create_model(str(config_path))
    teacher_sd = load_state_dict(str(ckpt_path), location='cpu')
    missing, unexpected = teacher.load_state_dict(teacher_sd, strict=False)
    if missing or unexpected:
        print(f"  Warning: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    freeze_model(teacher)
    teacher.eval()

    # Disable gradient checkpointing
    if hasattr(teacher, 'use_checkpoint'):
        teacher.use_checkpoint = False
    for module in teacher.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False

    print("✓ Teacher loaded")

    # =========================================================================
    # 2. Load Student (Physical Instance 2 - CRITICAL: No deepcopy)
    # =========================================================================
    print(f"\n{'='*80}\nLoading Student (Instance 2)...\n{'='*80}")
    # CRITICAL FIX: Create fresh instance instead of deepcopy
    # This ensures complete isolation between teacher and student
    student = create_model(str(config_path))
    student_sd = load_state_dict(str(ckpt_path), location='cpu')
    student.load_state_dict(student_sd, strict=False)

    freeze_model(student)
    student.train()

    # Disable gradient checkpointing
    if hasattr(student, 'use_checkpoint'):
        student.use_checkpoint = False
    for module in student.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False

    print("✓ Student loaded (physically isolated from teacher)")

    # =========================================================================
    # 3. Inject LoRA into Student Only
    # =========================================================================
    print(f"\n{'='*80}\nInjecting LoRA...\n{'='*80}")
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=target_modules
    )

    try:
        student = get_peft_model(student, lora_config)
        print("✓ LoRA injected successfully")
    except Exception as e:
        print(f"⚠ Warning: PEFT get_peft_model failed: {e}")
        print("Falling back to manual LoRA injection (not implemented)")
        raise

    # Count trainable parameters
    trainable_params = count_trainable_parameters(student)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # =========================================================================
    # 4. Create Optimizer
    # =========================================================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.learning_rate,
    )

    # =========================================================================
    # 5. Create Dataset
    # =========================================================================
    print(f"\n{'='*80}\nCreating dataset...\n{'='*80}")
    if args.use_mock_dataset:
        dataset = AnyTextMockDataset(
            size=args.dataset_size,
            resolution=args.resolution
        )
        print(f"✓ Mock dataset created: {args.dataset_size} samples")
    elif args.use_real_dataset:
        dataset = RealAnyTextDataset(
            json_path=args.dataset_json,
            max_lines=args.max_lines,
            max_chars=args.max_chars,
            resolution=args.resolution,
            font_path=args.font_path,
            font_hint_prob=args.font_hint_prob,
            font_hint_area=args.font_hint_area,
            color_prob=args.color_prob,
            wm_thresh=args.wm_thresh,
            glyph_scale=args.glyph_scale,
            font_hint_randaug=True,
        )
        print(f"✓ Real dataset created: {len(dataset)} samples")
        print(f"  - JSON path: {args.dataset_json}")
        print(f"  - Max lines: {args.max_lines}, Max chars: {args.max_chars}")
        print(f"  - Font hint prob: {args.font_hint_prob}, Color prob: {args.color_prob}")
    else:
        # Default to mock dataset if neither specified
        print("⚠ No dataset specified, using mock dataset for testing")
        print("  Use --use_real_dataset to train with real demodataset")
        dataset = AnyTextMockDataset(
            size=args.dataset_size,
            resolution=args.resolution
        )
        print(f"✓ Mock dataset created: {args.dataset_size} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_anytext,
        num_workers=4,
    )

    # =========================================================================
    # 6. Prepare for Accelerate
    # =========================================================================
    print(f"\n{'='*80}\nPreparing training...\n{'='*80}")

    # Note: We only prepare student, optimizer, dataloader
    # Teacher is kept on CPU or moved manually as needed
    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)

    device = accelerator.device
    teacher.to(device)  # Manual move for teacher

    # Create wrappers
    teacher_wrapper = AnyText2ForwardWrapper(teacher, device)
    student_wrapper = AnyText2ForwardWrapper(student, device)

    # Create scheduler
    scheduler = create_lcm_scheduler_from_anytext(teacher)

    print(f"✓ Training setup complete")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Target inference steps: {args.num_inference_steps}")

    # =========================================================================
    # 7. Training Loop (Manual Accumulation)
    # =========================================================================
    print(f"\n{'='*80}\nStarting Training (Manual Accumulation Mode)...\n{'='*80}\n")

    # 1. Ensure Student is in training mode
    student.train()

    # 2. Initial gradient clear
    optimizer.zero_grad()

    global_step = 0
    total_batch_steps = 0  # Track total batch steps

    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(100):  # Epoch Loop
        for batch in dataloader:
            # ---------------------------------------------------------
            # A. Forward & Loss Calculation
            # ---------------------------------------------------------
            # Call training_step (pure function, no side effects)
            outputs = training_step(
                teacher_wrapper=teacher_wrapper,
                student_wrapper=student_wrapper,
                batch=batch,
                scheduler=scheduler,
                accelerator=accelerator,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
            )

            # Get Loss and scale for accumulation
            loss = outputs['loss']
            loss_scaled = loss / args.gradient_accumulation_steps

            # ---------------------------------------------------------
            # B. Backward (No Graph Retention)
            # ---------------------------------------------------------
            # At this point, backward only accumulates gradients
            # Use accelerator.backward for mixed precision support
            accelerator.backward(loss_scaled, retain_graph=False)

            total_batch_steps += 1

            # ---------------------------------------------------------
            # C. Optimizer Step - Only when accumulation is complete
            # ---------------------------------------------------------
            if total_batch_steps % args.gradient_accumulation_steps == 0:
                # Gradient clipping (optional, prevents explosion)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # Complete clear

                # Update global progress
                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % args.logging_steps == 0:
                    if accelerator.is_local_main_process:
                        # Restore loss value for display
                        current_loss = loss.detach().item()
                        accelerator.log({"train/loss": current_loss}, step=global_step)
                        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

                # Checkpointing
                if global_step % args.save_steps == 0:
                    if accelerator.is_local_main_process:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{timestamp}-step{global_step}")
                        # Unwrap and save LoRA
                        unwrapped = accelerator.unwrap_model(student)
                        unwrapped.save_pretrained(save_path)
                        print(f"\n✓ Saved: {save_path}")

            # ---------------------------------------------------------
            # D. Cleanup
            # ---------------------------------------------------------
            # Force release temporary variables to prevent graph leaks
            del outputs, loss, loss_scaled

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Final Save
    if accelerator.is_local_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.output_dir, f"checkpoint-{timestamp}-final")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(save_path)
        print(f"\n✓ Final checkpoint saved: {save_path}")

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()
