#!/usr/bin/env python3
"""
从完整 AnyWord-3M 数据集中抽取 1000 张图片及其标注
使用流式读取处理大型 JSON 文件，避免 OOM

使用方法:
    python create_demo_dataset.py --num_samples 1000 --output_dir ./demodataset
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import copy
from decimal import Decimal

try:
    import ijson
except ImportError:
    ijson = None

def reservoir_sampling(iterable, k, seed=None):
    """
    蓄水池抽样算法 - 从大数据流中随机抽取 k 个样本

    Args:
        iterable: 可迭代的数据流
        k: 抽取的样本数量
        seed: 随机种子

    Returns:
        抽取的样本列表
    """
    if seed is not None:
        random.seed(seed)

    reservoir = []
    for i, item in enumerate(iterable):
        if i < k:
            # 前k个元素直接放入蓄水池
            reservoir.append(item)
        else:
            # 对于第i个元素（i >= k），以 k/i 的概率替换
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir


def stream_json_objects(json_path, num_samples):
    """
    流式读取大型 JSON 文件，逐项读取 data_list

    Args:
        json_path: JSON 文件路径
        num_samples: 目标样本数（用于进度条）

    Yields:
        data_list 中的每一项
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        # 读取开头的 {
        char = f.read(1)
        if char != '{':
            raise ValueError("Invalid JSON format: expected '{'")

        # 查找 "data_list" 字段
        buffer = char
        found_data_list = False

        while not found_data_list:
            char = f.read(1)
            if not char:
                raise ValueError("EOF reached before finding data_list")

            buffer += char

            if '"data_list"' in buffer:
                # 找到了 data_list 字段，接下来查找 [
                # 继续读取直到找到 [
                while True:
                    char = f.read(1)
                    buffer += char
                    if char == '[':
                        found_data_list = True
                        break

        # 现在我们到了 data_list 的开始位置
        # 开始流式读取数组元素，只在完整对象闭合时解析
        obj_buffer = ""
        brace_count = 0
        in_string = False
        escape = False

        pbar = tqdm(total=num_samples, desc="  流式读取JSON")

        while True:
            char = f.read(1)
            if not char:
                break

            # 处理字符串和转义
            if escape:
                if brace_count > 0:
                    obj_buffer += char
                escape = False
                continue

            if char == '\\':
                if brace_count > 0:
                    obj_buffer += char
                escape = True
                continue

            if char == '"':
                if brace_count > 0:
                    obj_buffer += char
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                    obj_buffer += char
                    continue
                if char == '}' and brace_count > 0:
                    brace_count -= 1
                    obj_buffer += char
                    if brace_count == 0:
                        try:
                            obj = json.loads(obj_buffer)
                            pbar.update(1)
                            yield obj
                        except json.JSONDecodeError as e:
                            print(f"\n警告: JSON 解析错误: {e}")
                            print(f"缓冲区内容: {obj_buffer[:200]}...")
                        obj_buffer = ""
                    continue
                if char == ']' and brace_count == 0:
                    break

                if brace_count == 0:
                    # 跳过对象之间的逗号/空白
                    continue

            if brace_count > 0:
                obj_buffer += char

        pbar.close()


def stream_json_ijson(json_path, num_samples):
    """
    使用 ijson 进行快速流式读取 data_list

    Args:
        json_path: JSON 文件路径
        num_samples: 目标样本数（用于进度条）

    Yields:
        data_list 中的每一项
    """
    if ijson is None:
        raise RuntimeError("ijson not installed")

    with open(json_path, 'rb') as f:
        pbar = tqdm(total=num_samples, desc="  ijson 读取JSON")
        for obj in ijson.items(f, 'data_list.item'):
            pbar.update(1)
            yield obj
        pbar.close()


def stream_json_simple(json_path):
    """
    简化的流式读取 - 更稳健的流式读取 data_list

    Args:
        json_path: JSON 文件路径

    Yields:
        data_list 中的每一项
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        # 流式查找 "data_list" 字段，避免一次性读入大块内容
        key = '"data_list"'
        key_idx = 0
        found_key = False
        in_string = False
        escape = False

        while True:
            char = f.read(1)
            if not char:
                break

            if escape:
                escape = False
                continue

            if char == '\\':
                escape = True
                continue

            if char == '"':
                in_string = not in_string

            if in_string:
                continue

            if char == key[key_idx]:
                key_idx += 1
                if key_idx == len(key):
                    found_key = True
                    break
            else:
                key_idx = 0

        if not found_key:
            return

        # 查找 data_list 后面的 '['
        while True:
            char = f.read(1)
            if not char:
                return
            if char == '[':
                break

        # 现在开始读取对象
        obj_buffer = ""
        brace_count = 0
        in_string = False
        escape = False

        while True:
            char = f.read(1)
            if not char:
                break

            if escape:
                obj_buffer += char
                escape = False
                continue

            if char == '\\':
                obj_buffer += char
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                obj_buffer += char
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == ']' and brace_count == 0:
                    # 数组结束
                    break

            obj_buffer += char

            # 当读取完一个对象
            if brace_count == 0 and obj_buffer.strip():
                try:
                    obj = json.loads(obj_buffer)
                    yield obj
                    obj_buffer = ""
                except json.JSONDecodeError:
                    # 可能是数组分隔符或不完整缓冲，继续读取
                    continue


def create_demo_dataset(args):
    """
    从完整数据集中抽取样本（流式处理，避免 OOM）
    """

    # 设置随机种子
    random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "imgs"
    images_dir.mkdir(exist_ok=True)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    print(f"创建演示数据集: {output_dir}")
    print(f"目标样本数: {args.num_samples}")
    print(f"使用流式读取，避免加载整个 JSON 到内存")
    print("="*60)

    # 数据源配置
    dataset_root = Path(args.dataset_root)

    # 定义数据源及其对应的 JSON 文件
    data_sources = [
        {
            "name": "laion",
            "image_dirs": [
                dataset_root / "laion" / "laion_p1" / "imgs",
                dataset_root / "laion" / "laion_p2" / "imgs",
                dataset_root / "laion" / "laion_p3" / "imgs",
            ],
            "json_file": dataset_root / "anytext2_json_files" / "anytext2_json_files" / "laion_word" / "data_v1.2b.json",
            "max_samples": int(args.num_samples * 0.6),  # 60% 来自 LAION
        },
        {
            "name": "wukong",
            "image_dirs": [
                dataset_root / "wukong_1of5" / "wukong_1of5" / "imgs",
                dataset_root / "wukong_2of5" / "wukong_2of5" / "imgs",
                dataset_root / "wukong_3of5" / "wukong_3of5" / "imgs",
            ],
            "json_file": dataset_root / "anytext2_json_files" / "anytext2_json_files" / "wukong_word" / "data_v1.2b.json",
            "max_samples": int(args.num_samples * 0.4),  # 40% 来自 Wukong
        },
    ]

    all_samples = []
    total_collected = 0

    # 遍历每个数据源
    for source in data_sources:
        print(f"\n处理数据源: {source['name']}")
        print(f"目标样本数: {source['max_samples']}")

        # 检查 JSON 文件
        json_path = source['json_file']
        if not json_path.exists():
            print(f"  ⚠️  JSON 文件不存在: {json_path}")
            continue

        # 检查文件大小
        json_size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"  JSON 文件大小: {json_size_mb:.1f} MB")
        print(f"  使用流式读取（蓄水池抽样）...")

        # 使用蓄水池抽样算法从数据流中抽取样本
        try:
            if ijson is not None:
                print(f"  使用 ijson 加速读取...")
                sampled_data = reservoir_sampling(
                    stream_json_ijson(json_path, source['max_samples']),
                    source['max_samples'],
                    seed=args.seed
                )
            else:
                sampled_data = reservoir_sampling(
                    stream_json_objects(json_path, source['max_samples']),
                    source['max_samples'],
                    seed=args.seed
                )
        except Exception as e:
            print(f"  ⚠️  流式读取失败: {e}")
            print(f"  尝试使用内置流式读取...")
            try:
                sampled_data = reservoir_sampling(
                    stream_json_objects(json_path, source['max_samples']),
                    source['max_samples'],
                    seed=args.seed
                )
            except Exception as e2:
                print(f"  ⚠️  内置流式读取失败: {e2}")
                print(f"  尝试使用简化流式读取...")
                try:
                    sampled_data = reservoir_sampling(
                        stream_json_simple(json_path),
                        source['max_samples'],
                        seed=args.seed
                    )
                except Exception as e3:
                    print(f"  ❌ 所有方法都失败: {e3}")
                    continue

        print(f"  ✅ 成功抽取: {len(sampled_data)} 个样本")

        # 收集样本（复制图片）
        collected = 0
        for item in tqdm(sampled_data, desc=f"  复制 {source['name']} 图片"):
            img_name = item.get('img_name', '')
            if not img_name:
                continue

            # 在所有图像目录中查找图片
            img_found = False
            for img_dir in source['image_dirs']:
                if not img_dir.exists():
                    continue

                img_path = img_dir / img_name
                if img_path.exists():
                    # 复制图片
                    dst_img_path = images_dir / img_name

                    # 检查是否已存在（避免重复）
                    if not dst_img_path.exists():
                        shutil.copy2(img_path, dst_img_path)

                    # 更新 data_root 为输出目录
                    new_item = copy.deepcopy(item)
                    new_item['img_name'] = img_name

                    all_samples.append(new_item)
                    collected += 1
                    img_found = True
                    break

            if collected >= source['max_samples']:
                break

        print(f"  ✅ 成功收集: {collected} 张图片")
        total_collected += collected

    print(f"\n{'='*60}")
    print(f"总计收集: {total_collected} 张图片")

    # 保存新的 JSON 标注文件
    output_json = {
        "data_root": str(images_dir),
        "data_list": all_samples
    }

    def _json_default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    json_output_path = annotations_dir / "demo_data.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"✅ 标注文件已保存: {json_output_path}")

    # 创建数据集信息文件
    info = {
        "total_samples": total_collected,
        "train_split": int(total_collected * 0.8),
        "val_split": int(total_collected * 0.2),
        "sources": {
            "laion": int(total_collected * 0.6),
            "wukong": int(total_collected * 0.4),
        }
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据集信息已保存: {info_path}")

    # 生成训练配置示例
    config_example = f"""# 训练配置示例 (train.py)

# 数据集路径
dataset_json: "{json_output_path}"

# 训练参数
batch_size: 4  # 小数据集可以使用较大的 batch size
grad_accum: 1
learning_rate: 1e-4

# 训练轮次
max_epochs: 10  # 小数据集可以训练更多轮

# 数据集划分
train_split: {info['train_split']}
val_split: {info['val_split']}

# 保存路径
output_dir: "./experiments/demo_distill"
"""

    config_path = output_dir / "config_example.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_example)
    print(f"✅ 配置示例已保存: {config_path}")

    print(f"\n{'='*60}")
    print(f"✅ 演示数据集创建完成!")
    print(f"")
    print(f"数据集位置: {output_dir}")
    print(f"  - 图片: {images_dir}")
    print(f"  - 标注: {json_output_path}")
    print(f"  - 信息: {info_path}")
    print(f"")
    print(f"统计信息:")
    print(f"  - 总样本数: {total_collected}")
    print(f"  - 训练集: {info['train_split']}")
    print(f"  - 验证集: {info['val_split']}")
    print(f"  - 磁盘占用: ~{total_collected * 0.5:.1f} MB (估算)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="从 AnyWord-3M 数据集抽取演示数据集（流式处理，避免 OOM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
    # 抽取 1000 张样本
    python create_demo_dataset.py --num_samples 1000

    # 抽取 500 张样本并指定输出目录
    python create_demo_dataset.py --num_samples 500 --output_dir ./test_dataset

    # 使用完整数据集路径
    python create_demo_dataset.py --dataset_root /path/to/dataset --num_samples 1000
        """
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./dataset",
        help="完整数据集根目录 (默认: ./dataset)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="抽取的样本总数 (默认: 1000)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demodataset",
        help="输出目录 (默认: ./demodataset)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 检查数据集目录
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_root}")
        return 1

    print(f"数据集根目录: {dataset_root}")

    # 创建演示数据集
    create_demo_dataset(args)

    return 0


if __name__ == "__main__":
    exit(main())
