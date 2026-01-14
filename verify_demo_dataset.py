#!/usr/bin/env python3
"""
验证演示数据集是否正确创建

使用方法:
    python verify_demo_dataset.py --dataset_dir ./demodataset
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def verify_dataset(dataset_dir):
    """验证数据集完整性"""

    dataset_dir = Path(dataset_dir)

    print("="*60)
    print("演示数据集验证")
    print("="*60)

    # 1. 检查目录结构
    print("\n1. 检查目录结构...")
    required_dirs = ["imgs", "annotations"]
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - 不存在")
            return False

    # 2. 检查标注文件
    print("\n2. 检查标注文件...")
    json_path = dataset_dir / "annotations" / "demo_data.json"
    if not json_path.exists():
        print(f"  ❌ demo_data.json 不存在")
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  ✅ 加载标注文件成功")

    # 3. 检查数据格式
    print("\n3. 检查数据格式...")
    required_keys = ['data_root', 'data_list']
    for key in required_keys:
        if key not in data:
            print(f"  ❌ 缺少字段: {key}")
            return False
        print(f"  ✅ {key}: {type(data[key]).__name__}")

    # 4. 统计信息
    print("\n4. 数据集统计...")
    total_samples = len(data['data_list'])
    print(f"  总样本数: {total_samples}")

    if total_samples == 0:
        print(f"  ❌ 数据集为空")
        return False

    # 统计文本数量
    total_texts = sum(len(item.get('annotations', [])) for item in data['data_list'])
    print(f"  总文本数: {total_texts}")
    print(f"  平均每张图文本数: {total_texts / total_samples:.2f}")

    # 5. 检查图片文件
    print("\n5. 检查图片文件...")
    images_dir = dataset_dir / "imgs"
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"  图片数量: {len(image_files)}")

    if len(image_files) == 0:
        print(f"  ❌ 没有找到图片文件")
        return False

    # 6. 验证图片和标注的对应关系
    print("\n6. 验证图片和标注的对应关系...")
    missing_images = []
    for i, item in enumerate(data['data_list'][:10]):  # 只检查前10个
        img_name = item.get('img_name', '')
        img_path = images_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)

    if missing_images:
        print(f"  ⚠️  部分图片缺失: {missing_images[:5]}...")
    else:
        print(f"  ✅ 前10个样本的图片都存在")

    # 7. 显示第一个样本的详细信息
    print("\n7. 第一个样本详细信息...")
    if total_samples > 0:
        first_sample = data['data_list'][0]
        print(f"  图片名: {first_sample['img_name']}")
        print(f"  标注数: {len(first_sample.get('annotations', []))}")

        if first_sample.get('annotations'):
            first_annotation = first_sample['annotations'][0]
            print(f"  文本: {first_annotation.get('text', 'N/A')}")
            print(f"  语言: {first_annotation.get('language', 'N/A')}")
            print(f"  置信度: {first_annotation.get('rec_score', 'N/A')}")

    # 8. 可视化第一个样本（如果有图片）
    print("\n8. 可视化第一个样本...")
    if total_samples > 0:
        try:
            first_sample = data['data_list'][0]
            img_name = first_sample['img_name']
            img_path = images_dir / img_name

            if img_path.exists():
                img = Image.open(img_path)

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(img)

                # 绘制文本框
                for ann in first_sample.get('annotations', []):
                    polygon = np.array(ann['polygon'])
                    poly = patches.Polygon(
                        polygon,
                        closed=True,
                        fill=False,
                        edgecolor='red',
                        linewidth=2,
                        label=ann.get('text', '')
                    )
                    ax.add_patch(poly)

                    # 添加文本标签
                    if 'text' in ann:
                        text_pos = polygon[0]
                        ax.text(
                            text_pos[0],
                            text_pos[1] - 5,
                            ann['text'],
                            color='red',
                            fontsize=12,
                            bbox=dict(
                                facecolor='white',
                                edgecolor='none',
                                alpha=0.7
                            )
                        )

                ax.axis('off')
                ax.set_title(f"样本可视化: {img_name}")

                # 保存图像
                viz_path = dataset_dir / "sample_visualization.png"
                plt.savefig(viz_path, bbox_inches='tight', dpi=150)
                plt.close()

                print(f"  ✅ 可视化已保存: {viz_path}")
        except Exception as e:
            print(f"  ⚠️  可视化失败: {e}")

    # 9. 检查数据集信息文件
    print("\n9. 检查数据集信息文件...")
    info_path = dataset_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        print(f"  ✅ dataset_info.json")
        print(f"    总样本: {info.get('total_samples', 'N/A')}")
        print(f"    训练集: {info.get('train_split', 'N/A')}")
        print(f"    验证集: {info.get('val_split', 'N/A')}")
    else:
        print(f"  ⚠️  dataset_info.json 不存在")

    print("\n" + "="*60)
    print("✅ 数据集验证完成!")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="验证演示数据集"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./demodataset",
        help="数据集目录 (默认: ./demodataset)"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        return 1

    success = verify_dataset(args.dataset_dir)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
