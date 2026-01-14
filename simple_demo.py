#!/usr/bin/env python
"""
AnyText2 简单推理脚本 - 绕过Gradio UI问题
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("AnyText2 简单推理脚本")
print("=" * 60)

# 检查CUDA
print(f"\nCUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")

# 导入模型
try:
    from ms_wrapper import AnyText2Model
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 加载模型
print("\n正在加载模型...")
try:
    model = AnyText2Model(
        model_dir='./models/iic/cv_anytext2',
        use_fp16=True,
        use_translator=False
    ).cuda(0)
    print("✓ 模型加载成功！")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

# 示例1: 文本生成
print("\n" + "=" * 60)
print("示例1: 文本生成")
print("=" * 60)

try:
    result, code, warning, debug = model.forward(
        {
            'text_prompt': 'A photo of a coffee shop menu',
            'texts': ['Cafe Menu'],
            'draw_pos': None,
            'mode': 'text-generation',
            'image_count': 1,
            'ddim_steps': 20,
            'image_width': 512,
            'image_height': 512,
            'cfg_scale': 9.0,
            'seed': 42
        },
        mode='text-generation',
        sort_priority='↔',
        revise_pos=False,
        base_model_path='',
        lora_path_ratio='',
        f1='No Font(不指定字体)', f2='No Font(不指定字体)',
        f3='No Font(不指定字体)', f4='No Font(不指定字体)', f5='No Font(不指定字体)',
        m1=None, m2=None, m3=None, m4=None, m5=None,
        c1='black', c2='black', c3='black', c4='black', c5='black',
        show_debug=False,
        draw_img=None, ref_img=None, ori_img=None,
        img_count=1, ddim_steps=20, w=512, h=512, strength=1.0,
        attnx_scale=1.0, font_hollow=None, cfg_scale=9.0, seed=42, eta=0.0,
        a_prompt='best quality, extremely detailed, 4k, HD, supper legible text, clear text edges',
        n_prompt='low-res, bad anatomy, extra digit, cropped, worst quality, low quality, watermark'
    )

    if result is not None and len(result) > 0:
        img = result[0]
        output_path = 'output_text_generation.png'
        Image.fromarray(img).save(output_path)
        print(f"✓ 文本生成完成！图像已保存到: {output_path}")
        if warning:
            print(f"警告: {warning}")
    else:
        print("✗ 生成失败")

except Exception as e:
    print(f"✗ 文本生成出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("使用说明")
print("=" * 60)
print("""
模型已成功加载！您可以使用以下API:

1. 文本生成:
   result = model.forward(
       {...},
       mode='text-generation',
       texts=['要生成的文字'],
       text_prompt='图像描述'
   )

2. 文本编辑:
   需要提供:
   - ori_image: 原始图像 (numpy数组)
   - draw_pos: 编辑区域 (mask图像)

详细参数请参考 ms_wrapper.py 中的 forward 函数。
""")
