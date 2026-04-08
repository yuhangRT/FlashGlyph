#!/usr/bin/env python
"""
AnyText2 实验脚本 - 用于快速测试和实验
可以修改参数来测试不同的效果
"""
import os
import sys
import numpy as np
from PIL import Image
import cv2

print("=" * 70)
print("AnyText2 实验脚本")
print("=" * 70)

# ==================== 配置区域 ====================
# 在这里修改你的实验参数

# 实验模式
EXPERIMENT_TYPE = "text_generation"  # 选项: "text_generation", "text_editing"

# 提示词配置
IMG_PROMPT = 'A cartoon cat holding a sign with words on it'
TEXT_PROMPT = 'that reads "Hello" "World"'

# 生成配置
SEED = 42
IMAGE_COUNT = 4           # 生成图片数量
DDIM_STEPS = 20           # 采样步数
IMAGE_WIDTH = 512         # 图片宽度
IMAGE_HEIGHT = 512        # 图片高度
CFG_SCALE = 9.0           # CFG 强度
STRENGTH = 1.0            # ControlNet 控制强度
ATTNX_SCALE = 1.0         # AttnX 注意力层强度

# 字体和颜色配置 (最多5行文字)
# 字体选项: 'No Font(不指定字体)', 'Mimic From Image(模仿图中字体)', 或其他字体名
FONTS = [
    'IndieFlower',        # 第1行字体
    'No Font(不指定字体)', # 第2行字体
    'No Font(不指定字体)', # 第3行字体
    'No Font(不指定字体)', # 第4行字体
    'No Font(不指定字体)', # 第5行字体
]

# 颜色 (RGB值, 500,500,500 表示随机颜色)
COLORS = [
    '0,0,0',              # 第1行颜色 (黑色)
    '500,500,500',        # 第2行颜色 (随机)
    '500,500,500',        # 第3行颜色 (随机)
    '500,500,500',        # 第4行颜色 (随机)
    '500,500,500',        # 第5行颜色 (随机)
]

# 输出目录
OUTPUT_DIR = 'experiment_results'

# ================================================


def setup_model():
    """加载模型"""
    print("\n[1/4] 正在加载模型...")
    try:
        from ms_wrapper import AnyText2Model
        
        model = AnyText2Model(
            model_dir='./models/iic/cv_anytext2',
            use_fp16=True,
            use_translator=False,  # 设为 True 如果需要中文翻译
            font_path='font/Arial_Unicode.ttf'
        ).cuda(0)
        
        print("✓ 模型加载成功")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("\n请确保:")
        print("  1. 已安装所有依赖: conda env create -f environment.yaml")
        print("  2. 已下载模型: python -c \"from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')\"")
        sys.exit(1)


def create_position_map(width=512, height=512):
    """
    创建位置图 - 用于指定文字位置
    白色区域表示文字位置
    """
    pos_img = np.zeros((height, width, 1), dtype=np.uint8)
    
    # 示例: 创建两个矩形区域
    # 第一个文字区域 (左上)
    cv2.rectangle(pos_img, (50, 100), (250, 180), 255, -1)
    # 第二个文字区域 (右下)
    cv2.rectangle(pos_img, (280, 300), (480, 380), 255, -1)
    
    return pos_img


def run_text_generation(model):
    """运行文本生成实验"""
    print("\n[2/4] 准备文本生成任务...")
    
    # 创建位置图 (可选，None 表示模型自动决定位置)
    draw_pos = create_position_map(IMAGE_WIDTH, IMAGE_HEIGHT)
    cv2.imwrite(f'{OUTPUT_DIR}/position_map.png', draw_pos)
    print(f"✓ 位置图已保存到 {OUTPUT_DIR}/position_map.png")
    
    # 准备输入数据
    input_data = {
        'img_prompt': IMG_PROMPT,
        'text_prompt': TEXT_PROMPT,
        'seed': SEED,
        'draw_pos': draw_pos,
        'ori_image': None
    }
    
    # 准备参数
    params = {
        'mode': 'text-generation',
        'sort_priority': '↕',
        'show_debug': True,
        'revise_pos': False,
        'image_count': IMAGE_COUNT,
        'ddim_steps': DDIM_STEPS,
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'strength': STRENGTH,
        'attnx_scale': ATTNX_SCALE,
        'font_hollow': True,
        'cfg_scale': CFG_SCALE,
        'seed': SEED,
        'eta': 0.0,
        'a_prompt': 'best quality, extremely detailed,4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks',
        'n_prompt': 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture',
        'base_model_path': '',
        'lora_path_ratio': '',
        'glyline_font_path': FONTS,
        'font_hint_image': [None] * 5,
        'font_hint_mask': [None] * 5,
        'text_colors': ' '.join(COLORS)
    }
    
    print("\n[3/4] 开始生成...")
    print(f"  图像提示词: {IMG_PROMPT}")
    print(f"  文字提示词: {TEXT_PROMPT}")
    print(f"  生成数量: {IMAGE_COUNT}")
    print(f"  采样步数: {DDIM_STEPS}")
    print(f"  CFG强度: {CFG_SCALE}")
    print(f"  控制强度: {STRENGTH}")
    print(f"  AttnX强度: {ATTNX_SCALE}")
    print(f"  种子: {SEED}")
    
    # 执行推理
    results, rtn_code, rtn_warning, debug_info = model.forward(input_data, **params)
    
    # 保存结果
    print("\n[4/4] 保存结果...")
    if rtn_code >= 0 and results is not None:
        for i, img in enumerate(results):
            if i < IMAGE_COUNT:
                # 生成的结果图像
                output_path = f'{OUTPUT_DIR}/result_{i:02d}.png'
                Image.fromarray(img).save(output_path)
                print(f"✓ 生成图像已保存: {output_path}")
            else:
                # 调试信息图像 (glyph图等)
                if i == IMAGE_COUNT:
                    output_path = f'{OUTPUT_DIR}/debug_glyph.png'
                    Image.fromarray(img).save(output_path)
                    print(f"✓ Debug glyph图已保存: {output_path}")
                elif i == IMAGE_COUNT + 1:
                    output_path = f'{OUTPUT_DIR}/debug_font_hint.png'
                    Image.fromarray(img).save(output_path)
                    print(f"✓ Font hint图已保存: {output_path}")
        
        print(f"\n{'=' * 70}")
        print("✓ 生成完成！")
        print(f"{'=' * 70}")
        print(f"\n结果保存在: {OUTPUT_DIR}/")
        print(f"调试信息:\n{debug_info}")
        
        if rtn_warning:
            print(f"\n警告: {rtn_warning}")
    else:
        print(f"✗ 生成失败: {rtn_warning}")


def run_batch_experiments(model):
    """运行批量实验 - 测试不同参数"""
    print("\n" + "=" * 70)
    print("批量实验模式")
    print("=" * 70)
    
    # 实验 1: 测试不同的控制强度
    print("\n实验 1: 测试不同的 ControlNet 控制强度")
    strengths = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for s in strengths:
        print(f"\n--- 测试 strength={s} ---")
        input_data = {
            'img_prompt': IMG_PROMPT,
            'text_prompt': TEXT_PROMPT,
            'seed': SEED,
            'draw_pos': None,
            'ori_image': None
        }
        
        params = {
            'mode': 'text-generation',
            'sort_priority': '↕',
            'show_debug': False,
            'revise_pos': False,
            'image_count': 1,
            'ddim_steps': DDIM_STEPS,
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'strength': s,
            'attnx_scale': ATTNX_SCALE,
            'font_hollow': True,
            'cfg_scale': CFG_SCALE,
            'seed': SEED,
            'eta': 0.0,
            'a_prompt': 'best quality, extremely detailed',
            'n_prompt': 'low-res, bad anatomy',
            'base_model_path': '',
            'lora_path_ratio': '',
            'glyline_font_path': FONTS,
            'font_hint_image': [None] * 5,
            'font_hint_mask': [None] * 5,
            'text_colors': ' '.join(COLORS)
        }
        
        results, code, warning, _ = model.forward(input_data, **params)
        if results is not None:
            output_path = f'{OUTPUT_DIR}/exp1_strength_{s:.2f}.png'
            Image.fromarray(results[0]).save(output_path)
            print(f"✓ 已保存: {output_path}")


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    model = setup_model()
    
    # 根据实验类型运行
    if EXPERIMENT_TYPE == "text_generation":
        run_text_generation(model)
    elif EXPERIMENT_TYPE == "batch":
        run_batch_experiments(model)
    else:
        print(f"未知实验类型: {EXPERIMENT_TYPE}")
        print("可选: 'text_generation' 或 'batch'")
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
