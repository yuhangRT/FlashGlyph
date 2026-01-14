"""
AnyText2 Model Inspector for LoRA Target Module Identification

This script analyzes the AnyText2 model architecture and prints all Linear and Conv2D layers
that should receive LoRA injection for LCM-LoRA distillation.

Usage:
    python inspect_modules.py --config models_yaml/anytext2_sd15.yaml --ckpt models/anytext_v2.0.ckpt

Output:
    - Prints all target modules grouped by component (ControlNet, UNet)
    - Saves target_modules list to target_modules_list.txt
"""

import argparse
import sys
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to import from cldm
sys.path.insert(0, str(Path(__file__).parent.parent))

from cldm.model import create_model, load_state_dict


def inspect_layer(module, prefix="", target_modules=None):
    """
    Recursively inspect a module and collect Linear/Conv2d layers.

    Args:
        module: PyTorch module to inspect
        prefix: Current name prefix
        target_modules: Dictionary to store found modules

    Returns:
        Dictionary of layer_name -> (layer_type, shape)
    """
    if target_modules is None:
        target_modules = {}

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this is a target layer type
        if isinstance(child, nn.Linear):
            target_modules[full_name] = ("Linear", child.weight.shape)
        elif isinstance(child, nn.Conv2d):
            target_modules[full_name] = ("Conv2d", child.weight.shape)
        else:
            # Recursively inspect child modules
            inspect_layer(child, full_name, target_modules)

    return target_modules


def categorize_modules(all_modules):
    """
    Categorize modules into ControlNet and UNet groups.

    Returns:
        Dictionary with keys: 'controlnet_zero_convs', 'controlnet_attention',
                             'unet_attention', 'unet_output_blocks'
    """
    categorized = {
        'controlnet_zero_convs': [],
        'controlnet_attention': [],
        'unet_input_blocks': [],
        'unet_middle_block': [],
        'unet_output_blocks': [],
        'other': []
    }

    for name, (layer_type, shape) in all_modules.items():
        # ControlNet zero_convs
        if 'control_model.zero_convs' in name:
            categorized['controlnet_zero_convs'].append((name, layer_type, shape))
        # ControlNet attention layers
        elif 'control_model' in name and ('attn1.' in name or 'attn2.' in name or 'attn1x.' in name or 'attn2x.' in name):
            categorized['controlnet_attention'].append((name, layer_type, shape))
        # UNet input blocks
        elif 'model.diffusion_model.input_blocks' in name:
            categorized['unet_input_blocks'].append((name, layer_type, shape))
        # UNet middle block
        elif 'model.diffusion_model.middle_block' in name:
            categorized['unet_middle_block'].append((name, layer_type, shape))
        # UNet output blocks
        elif 'model.diffusion_model.output_blocks' in name:
            categorized['unet_output_blocks'].append((name, layer_type, shape))
        else:
            categorized['other'].append((name, layer_type, shape))

    return categorized


def print_module_summary(categorized_modules):
    """Print a formatted summary of all target modules."""

    total_count = 0

    print("\n" + "="*80)
    print("ANYTEXT2 LORA TARGET MODULE INSPECTION")
    print("="*80)

    # ControlNet zero_convs
    print("\n[1] ControlNet Zero Convolutions (Conv2D)")
    print("-" * 80)
    print(f"{'Count':<6} {'Layer Type':<10} {'Shape':<30} {'Module Name'}")
    print("-" * 80)
    for i, (name, layer_type, shape) in enumerate(categorized_modules['controlnet_zero_convs']):
        shape_str = f"({shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})"
        print(f"{i+1:<6} {layer_type:<10} {shape_str:<30} {name}")
    total_count += len(categorized_modules['controlnet_zero_convs'])
    print(f"\nSubtotal: {len(categorized_modules['controlnet_zero_convs'])} zero_conv layers")

    # ControlNet attention
    print("\n[2] ControlNet Attention Projections (Linear)")
    print("-" * 80)
    print(f"{'Count':<6} {'Layer Type':<10} {'Shape':<20} {'Module Name'}")
    print("-" * 80)
    attn_count = 0
    for name, layer_type, shape in categorized_modules['controlnet_attention']:
        if any(proj in name for proj in ['to_q', 'to_k', 'to_v', 'to_out']):
            shape_str = f"({shape[0]}, {shape[1]})"
            print(f"{attn_count+1:<6} {layer_type:<10} {shape_str:<20} {name}")
            attn_count += 1
    total_count += attn_count
    print(f"\nSubtotal: {attn_count} attention projection layers")

    # UNet input blocks
    print("\n[3] UNet Input Blocks - Attention Projections (Linear)")
    print("-" * 80)
    print(f"{'Count':<6} {'Layer Type':<10} {'Shape':<20} {'Module Name'}")
    print("-" * 80)
    attn_count = 0
    for name, layer_type, shape in categorized_modules['unet_input_blocks']:
        if any(proj in name for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            shape_str = f"({shape[0]}, {shape[1]})"
            print(f"{attn_count+1:<6} {layer_type:<10} {shape_str:<20} {name}")
            attn_count += 1
    total_count += attn_count
    print(f"\nSubtotal: {attn_count} attention projection layers")

    # UNet middle block
    print("\n[4] UNet Middle Block - Attention Projections (Linear)")
    print("-" * 80)
    print(f"{'Count':<6} {'Layer Type':<10} {'Shape':<20} {'Module Name'}")
    print("-" * 80)
    attn_count = 0
    for name, layer_type, shape in categorized_modules['unet_middle_block']:
        if any(proj in name for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            shape_str = f"({shape[0]}, {shape[1]})"
            print(f"{attn_count+1:<6} {layer_type:<10} {shape_str:<20} {name}")
            attn_count += 1
    total_count += attn_count
    print(f"\nSubtotal: {attn_count} attention projection layers")

    # UNet output blocks
    print("\n[5] UNet Output Blocks - Attention Projections (Linear)")
    print("-" * 80)
    print(f"{'Count':<6} {'Layer Type':<10} {'Shape':<20} {'Module Name'}")
    print("-" * 80)
    attn_count = 0
    for name, layer_type, shape in categorized_modules['unet_output_blocks']:
        if any(proj in name for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            shape_str = f"({shape[0]}, {shape[1]})"
            print(f"{attn_count+1:<6} {layer_type:<10} {shape_str:<20} {name}")
            attn_count += 1
    total_count += attn_count
    print(f"\nSubtotal: {attn_count} attention projection layers")

    print("\n" + "="*80)
    print(f"TOTAL TARGET MODULES: {total_count}")
    print("="*80)

    return total_count


def generate_target_modules_list(categorized_modules, output_file):
    """
    Generate a Python-formatted list of target modules for PEFT LoRA config.

    Saves to output_file for easy copy-paste into training script.
    """
    target_modules = []

    # Add ControlNet zero_convs (Conv2D)
    for name, layer_type, _ in categorized_modules['controlnet_zero_convs']:
        # Remove .weight suffix for PEFT
        module_name = name.replace('.weight', '')
        target_modules.append(module_name)

    # Add attention projections (both ControlNet and UNet)
    all_attention = (categorized_modules['controlnet_attention'] +
                    categorized_modules['unet_input_blocks'] +
                    categorized_modules['unet_middle_block'] +
                    categorized_modules['unet_output_blocks'])

    for name, layer_type, _ in all_attention:
        if any(proj in name for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']):
            # Remove .weight or .bias suffix
            module_name = name.replace('.weight', '').replace('.bias', '')
            target_modules.append(module_name)

    # Save to file
    with open(output_file, 'w') as f:
        f.write("# Target modules for PEFT LoRA Configuration\n")
        f.write("# Generated by inspect_modules.py\n\n")
        f.write("target_modules = [\n")
        for module in sorted(target_modules):
            f.write(f"    \"{module}\",\n")
        f.write("]\n")

    print(f"\n✓ Target modules list saved to: {output_file}")
    print(f"  Total modules: {len(target_modules)}")

    return target_modules


def main():
    parser = argparse.ArgumentParser(description="Inspect AnyText2 model for LoRA target modules")
    parser.add_argument("--config", type=str, default="models_yaml/anytext2_sd15.yaml",
                       help="Path to model config file")
    parser.add_argument("--ckpt", type=str, default="models/anytext_v2.0.ckpt",
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="student_model/target_modules_list.txt",
                       help="Output file for target modules list")

    args = parser.parse_args()

    # Convert relative paths to absolute paths from parent directory
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent

    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt)

    # If path is not absolute, make it relative to parent directory
    if not config_path.is_absolute():
        config_path = (parent_dir / config_path).resolve()
    if not ckpt_path.is_absolute():
        ckpt_path = (parent_dir / ckpt_path).resolve()

    print(f"Loading model from {config_path}...")
    model = create_model(str(config_path))

    print(f"Loading checkpoint from {ckpt_path}...")
    state_dict = load_state_dict(str(ckpt_path), location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("✓ Model loaded successfully")
    if missing:
        print(f"  Warning: {len(missing)} missing keys (likely due to version upgrade)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (likely due to version upgrade)")

    # Inspect all modules
    print("\nInspecting model architecture...")
    all_modules = inspect_layer(model)

    # Categorize modules
    categorized = categorize_modules(all_modules)

    # Print summary
    total = print_module_summary(categorized)

    # Generate target modules list
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_modules = generate_target_modules_list(categorized, args.output)

    print("\n✓ Inspection complete!")
    print(f"\nNext steps:")
    print(f"1. Review the target modules list in: {args.output}")
    print(f"2. Use this list in the LoraConfig 'target_modules' parameter")
    print(f"3. Run training with: python train_lcm_anytext.py")


if __name__ == "__main__":
    main()
