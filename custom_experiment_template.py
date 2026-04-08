#!/usr/bin/env python
"""
AnyText2 自定义实验模板
复制此文件并根据需要修改参数来进行实验
"""
import os
import numpy as np
from PIL import Image
import cv2

def experiment_1_basic_generation(model):
    """实验1: 基础文本生成"""
    print("\n" + "="*60)
    print("实验1: 基础文本生成")
    print("="*60)
    
    input_data = {
        'img_prompt': 'A beautiful coffee shop sign',
        'text_prompt': 'that reads "Coffee"',
        'seed': 42,
        'draw_pos': None,  # 自动决定位置
        'ori_image': None
    }
    
    params = {
        'mode': 'text-generation',
        'sort_priority': '↕',
        'show_debug': False,
        'revise_pos': False,
        'image_count': 4,
        'ddim_steps': 20,
        'image_width': 512,
        'image_height': 512,
        'strength': 1.0,
        'attnx_scale': 1.0,
        'font_hollow': True,
        'cfg_scale': 9.0,
        'seed': 42,
        'eta': 0.0,
        'a_prompt': 'best quality, extremely detailed',
        'n_prompt': 'low-res, bad anatomy',
        'base_model_path': '',
        'lora_path_ratio': '',
        'glyline_font_path': ['IndieFlower'] + ['No Font(不指定字体)']*4,
        'font_hint_image': [None]*5,
        'font_hint_mask': [None]*5,
        'text_colors': ' '.join(['500,500,500']*5)
    }
    
    results, code, warning, debug = model.forward(input_data, **params)
    
    if results:
        for i, img in enumerate(results):
            path = f'output/exp1_basic_{i}.png'
            Image.fromarray(img).save(path)
            print(f'✓ 保存: {path}')


def experiment_2_custom_position(model):
    """实验2: 自定义文字位置"""
    print("\n" + "="*60)
    print("实验2: 自定义文字位置")
    print("="*60)
    
    # 创建自定义位置图
    pos_img = np.zeros((512, 512, 1), dtype=np.uint8)
    
    # 位置1: 左上角
    cv2.rectangle(pos_img, (30, 80), (250, 150), 255, -1)
    # 位置2: 右下角  
    cv2.rectangle(pos_img, (280, 350), (490, 420), 255, -1)
    
    # 保存位置图用于查看
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/position_map.png', pos_img)
    
    input_data = {
        'img_prompt': 'A vintage poster design',
        'text_prompt': 'that says "Vintage" and "Poster"',
        'seed': 123,
        'draw_pos': pos_img,
        'ori_image': None
    }
    
    params = {
        'mode': 'text-generation',
        'sort_priority': '↕',
        'show_debug': True,
        'revise_pos': False,
        'image_count': 2,
        'ddim_steps': 20,
        'image_width': 512,
        'image_height': 512,
        'strength': 1.0,
        'attnx_scale': 1.0,
        'font_hollow': True,
        'cfg_scale': 9.0,
        'seed': 123,
        'eta': 0.0,
        'a_prompt': 'best quality, extremely detailed',
        'n_prompt': 'low-res, bad anatomy',
        'base_model_path': '',
        'lora_path_ratio': '',
        'glyline_font_path': ['站酷快乐体2016修订版', 'IndieFlower'] + ['No Font(不指定字体)']*3,
        'font_hint_image': [None]*5,
        'font_hint_mask': [None]*5,
        'text_colors': '255,0,0 0,0,255 500,500,500 500,500,500 500,500,500'
    }
    
    results, code, warning, debug = model.forward(input_data, **params)
    
    if results:
        for i, img in enumerate(results):
            path = f'output/exp2_position_{i}.png'
            Image.fromarray(img).save(path)
            print(f'✓ 保存: {path}')


def experiment_3_strength_comparison(model):
    """实验3: 比较不同控制强度"""
    print("\n" + "="*60)
    print("实验3: 控制强度比较")
    print("="*60)
    
    strengths = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for strength in strengths:
        print(f'\n测试 strength={strength}...')
        
        input_data = {
            'img_prompt': 'A book cover with title',
            'text_prompt': 'that reads "My Book"',
            'seed': 42,
            'draw_pos': None,
            'ori_image': None
        }
        
        params = {
            'mode': 'text-generation',
            'sort_priority': '↕',
            'show_debug': False,
            'revise_pos': False,
            'image_count': 1,
            'ddim_steps': 20,
            'image_width': 512,
            'image_height': 512,
            'strength': strength,
            'attnx_scale': 1.0,
            'font_hollow': True,
            'cfg_scale': 9.0,
            'seed': 42,
            'eta': 0.0,
            'a_prompt': 'best quality, extremely detailed',
            'n_prompt': 'low-res, bad anatomy',
            'base_model_path': '',
            'lora_path_ratio': '',
            'glyline_font_path': ['Arial_Unicode'] + ['No Font(不指定字体)']*4,
            'font_hint_image': [None]*5,
            'font_hint_mask': [None]*5,
            'text_colors': ' '.join(['500,500,500']*5)
        }
        
        results, code, warning, _ = model.forward(input_data, **params)
        
        if results:
            strength_str = str(strength).replace('.', '_')
            path = f'output/exp3_strength_{strength_str}.png'
            Image.fromarray(results[0]).save(path)
            print(f'✓ 保存: {path}')


def experiment_4_seed_variation(model):
    """实验4: 不同随机种子"""
    print("\n" + "="*60)
    print("实验4: 随机种子变化")
    print("="*60)
    
    seeds = [42, 123, 456, 789, 1234]
    
    for seed in seeds:
        print(f'\n测试 seed={seed}...')
        
        input_data = {
            'img_prompt': 'A restaurant menu board',
            'text_prompt': 'that says "Menu"',
            'seed': seed,
            'draw_pos': None,
            'ori_image': None
        }
        
        params = {
            'mode': 'text-generation',
            'sort_priority': '↕',
            'show_debug': False,
            'revise_pos': False,
            'image_count': 1,
            'ddim_steps': 20,
            'image_width': 512,
            'image_height': 512,
            'strength': 1.0,
            'attnx_scale': 1.0,
            'font_hollow': True,
            'cfg_scale': 9.0,
            'seed': seed,
            'eta': 0.0,
            'a_prompt': 'best quality, extremely detailed',
            'n_prompt': 'low-res, bad anatomy',
            'base_model_path': '',
            'lora_path_ratio': '',
            'glyline_font_path': ['Pacifico'] + ['No Font(不指定字体)']*4,
            'font_hint_image': [None]*5,
            'font_hint_mask': [None]*5,
            'text_colors': ' '.join(['500,500,500']*5)
        }
        
        results, code, warning, _ = model.forward(input_data, **params)
        
        if results:
            path = f'output/exp4_seed_{seed}.png'
            Image.fromarray(results[0]).save(path)
            print(f'✓ 保存: {path}')


def main():
    """主函数 - 选择要运行的实验"""
    print("AnyText2 自定义实验模板")
    print("="*60)
    
    # 加载模型
    print("\n正在加载模型...")
    try:
        from ms_wrapper import AnyText2Model
        
        model = AnyText2Model(
            model_dir='./models/iic/cv_anytext2',
            use_fp16=True,
            use_translator=False,
            font_path='font/Arial_Unicode.ttf'
        ).cuda(0)
        
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 选择要运行的实验
    # 注释掉不需要的实验
    
    experiment_1_basic_generation(model)
    experiment_2_custom_position(model)
    experiment_3_strength_comparison(model)
    experiment_4_seed_variation(model)
    
    print("\n" + "="*60)
    print("✓ 所有实验完成！")
    print("="*60)
    print("\n结果保存在 output/ 目录")


if __name__ == "__main__":
    main()
