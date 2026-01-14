"""
AnyText2 模型检查工具（简化版）

这是一个不实际加载模型的版本，直接从配置文件中推导出目标模块。
适用于环境不兼容或只想快速查看目标模块的情况。
"""

import argparse
from pathlib import Path


def generate_anytext2_target_modules():
    """
    根据 AnyText2 架构生成 LoRA 目标模块列表。

    基于 cldm/cldm.py 的已知架构：
    - ControlNet 有 13 个 zero_conv (Conv2d)
    - ControlNet 和 UNet 都有 SpatialTransformer (带注意力投影层)
    - 注意力投影层包括 to_q, to_k, to_v, to_out.0

    Returns:
        list: 目标模块名称列表
    """
    target_modules = []

    # ControlNet zero_convs (Conv2D)
    # ControlNet 有 13 个 zero_conv（对应 13 个 control 输出）
    for i in range(13):
        target_modules.append(f"control_model.zero_convs.{i}.0")

    # ControlNet input_blocks 中的注意力投影
    # ControlNet 结构类似 UNet，有 input_blocks 和 middle_block
    # 假设有 channel_mult = [1, 2, 4, 4]，每个有 2 个 res_blocks

    # input_blocks (从索引 1 开始，0 是卷积层)
    # 结构：input_blocks[1-12] 包含 Transformer
    input_block_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for block_idx in input_block_indices:
        # 每个块有 1 个 SpatialTransformer (transformer_blocks.0)
        # 包含 attn1 (self-attn) 和 attn2 (cross-attn)
        for attn_type in ['attn1', 'attn2']:
            for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']:
                target_modules.append(
                    f"control_model.input_blocks.{block_idx}.1.transformer_blocks.0.{attn_type}.{proj}"
                )

    # middle_block 中的注意力投影
    for attn_type in ['attn1', 'attn2']:
        for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']:
            target_modules.append(
                f"control_model.middle_block.0.{attn_type}.{proj}"
            )

    # UNet (diffusion_model) 的注意力投影
    # UNet 有 input_blocks, middle_block, output_blocks

    # UNet input_blocks
    for block_idx in input_block_indices:
        # SpatialTransformer with attn1, attn2, and可选的 attn1x, attn2x
        for attn_type in ['attn1', 'attn2', 'attn1x', 'attn2x']:
            for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']:
                target_modules.append(
                    f"model.diffusion_model.input_blocks.{block_idx}.1.transformer_blocks.0.{attn_type}.{proj}"
                )

    # UNet middle_block
    for attn_type in ['attn1', 'attn2', 'attn1x', 'attn2x']:
        for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']:
            target_modules.append(
                f"model.diffusion_model.middle_block.0.{attn_type}.{proj}"
            )

    # UNet output_blocks
    # output_blocks 数量类似 input_blocks
    output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for block_idx in output_block_indices:
        # 每个块有 1 个 SpatialTransformer
        for attn_type in ['attn1', 'attn2', 'attn1x', 'attn2x']:
            for proj in ['to_q', 'to_k', 'to_v', 'to_out.0']:
                target_modules.append(
                    f"model.diffusion_model.output_blocks.{block_idx}.1.transformer_blocks.0.{attn_type}.{proj}"
                )

    return target_modules


def categorize_modules(target_modules):
    """将模块分类以便更好地展示"""
    categories = {
        'ControlNet Zero Convs (Conv2D)': [],
        'ControlNet Attention (Linear)': [],
        'UNet Input Blocks Attention (Linear)': [],
        'UNet Middle Block Attention (Linear)': [],
        'UNet Output Blocks Attention (Linear)': [],
    }

    for module in target_modules:
        if 'zero_convs' in module:
            categories['ControlNet Zero Convs (Conv2D)'].append(module)
        elif 'control_model.input_blocks' in module or 'control_model.middle_block' in module:
            categories['ControlNet Attention (Linear)'].append(module)
        elif 'model.diffusion_model.input_blocks' in module:
            categories['UNet Input Blocks Attention (Linear)'].append(module)
        elif 'model.diffusion_model.middle_block' in module:
            categories['UNet Middle Block Attention (Linear)'].append(module)
        elif 'model.diffusion_model.output_blocks' in module:
            categories['UNet Output Blocks Attention (Linear)'].append(module)

    return categories


def print_summary(categories):
    """打印模块摘要"""
    print("\n" + "="*80)
    print("AnyText2 LoRA 目标模块（基于架构推导）")
    print("="*80)

    total = 0
    for category, modules in categories.items():
        print(f"\n{category}:")
        print(f"  数量: {len(modules)}")
        if modules:
            print(f"  示例: {modules[0]}")
            if len(modules) > 1:
                print(f"        {modules[-1]}")
        total += len(modules)

    print("\n" + "="*80)
    print(f"总计: {total} 个目标模块")
    print("="*80)

    # 详细统计
    print("\n详细统计:")
    print(f"  - ControlNet Zero Convs (Conv2D): {len(categories['ControlNet Zero Convs (Conv2D)'])}")
    print(f"  - ControlNet Attention (Linear): {len(categories['ControlNet Attention (Linear)'])}")
    print(f"  - UNet Input Blocks (Linear): {len(categories['UNet Input Blocks Attention (Linear)'])}")
    print(f"  - UNet Middle Block (Linear): {len(categories['UNet Middle Block Attention (Linear)'])}")
    print(f"  - UNet Output Blocks (Linear): {len(categories['UNet Output Blocks Attention (Linear)'])}")


def save_target_modules(target_modules, output_file):
    """保存目标模块列表到文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# AnyText2 LoRA 目标模块列表\n")
        f.write("# 基于架构推导生成，无需加载完整模型\n\n")
        f.write("target_modules = [\n")
        for module in sorted(target_modules):
            f.write(f"    \"{module}\",\n")
        f.write("]\n")

    print(f"\n✓ 目标模块列表已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 AnyText2 LoRA 目标模块列表（简化版）")
    parser.add_argument("--output", type=str, default="student_model/target_modules_list.txt",
                       help="输出文件路径")

    args = parser.parse_args()

    print("\n正在生成 AnyText2 LoRA 目标模块列表...")
    print("基于已知架构推导，无需加载模型\n")

    # 生成目标模块
    target_modules = generate_anytext2_target_modules()

    # 分类
    categories = categorize_modules(target_modules)

    # 打印摘要
    print_summary(categories)

    # 保存到文件
    save_target_modules(target_modules, args.output)

    print("\n✓ 生成完成！")
    print(f"\n下一步:")
    print(f"1. 查看生成的文件: {args.output}")
    print(f"2. 使用这个列表训练: python train_lcm_anytext.py")


if __name__ == "__main__":
    main()
