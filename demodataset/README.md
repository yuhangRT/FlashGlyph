# 演示数据集创建指南

本指南说明如何从完整的 AnyWord-3M 数据集（200GB）中抽取一个小的演示数据集，用于测试蒸馏训练脚本。

## 📋 脚本说明

`create_demo_dataset.py` 会从完整数据集中随机抽取指定数量的图片及其标注文件。

### 功能特性

- ✅ 随机抽样，确保数据多样性
- ✅ 自动处理 LAION 和 Wukong 数据源
- ✅ 复制图片和重新生成标注文件
- ✅ 生成数据集统计信息
- ✅ 创建训练配置示例

### 数据源分布

默认采样比例：
- **LAION**: 60% (英文数据)
- **Wukong**: 40% (中文数据)

## 🚀 使用方法

### 基本用法

```bash
# 激活环境
conda activate anytext2

# 抽取 1000 张样本（默认）
python create_demo_dataset.py --num_samples 1000

# 抽取 500 张样本
python create_demo_dataset.py --num_samples 500

# 指定输出目录
python create_demo_dataset.py --num_samples 1000 --output_dir ./my_test_dataset
```

### 完整参数

```bash
python create_demo_dataset.py \
    --dataset_root ./dataset \      # 完整数据集路径
    --num_samples 1000 \              # 抽取样本数
    --output_dir ./demodataset \      # 输出目录
    --seed 42                         # 随机种子
```

## 📁 输出结构

运行后会生成以下目录结构：

```
demodataset/
├── imgs/                           # 图片目录
│   ├── 000000006.jpg
│   ├── 000000012.jpg
│   └── ...
├── annotations/                    # 标注目录
│   └── demo_data.json             # 演示数据集标注
├── dataset_info.json              # 数据集统计信息
└── config_example.yaml           # 训练配置示例
```

### 文件说明

#### 1. `demo_data.json`
```json
{
  "data_root": "/path/to/demodataset/imgs",
  "data_list": [
    {
      "img_name": "000000006.jpg",
      "annotations": [
        {
          "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
          "text": "Sample Text",
          "language": "Latin",
          "rec_score": 0.99,
          "valid": true
        }
      ]
    }
  ]
}
```

#### 2. `dataset_info.json`
```json
{
  "total_samples": 1000,
  "train_split": 800,
  "val_split": 200,
  "sources": {
    "laion": 600,
    "wukong": 400
  }
}
```

## 📊 资源占用

| 样本数 | 磁盘占用 | 抽取时间 |
|--------|---------|---------|
| 500    | ~250 MB | ~2 分钟 |
| 1000   | ~500 MB | ~4 分钟 |
| 2000   | ~1 GB   | ~8 分钟 |

## 🔧 在训练脚本中使用

### 方法 1: 直接使用 JSON 文件

修改 `train.py` 中的数据集路径：

```python
# train.py

dataset_config = {
    'json_path': './demodataset/annotations/demo_data.json',
    'train_split': 0.8,
    'val_split': 0.2,
}

# 创建数据集
from t3_dataset import T3Dataset
train_dataset = T3Dataset(
    json_path=dataset_config['json_path'],
    split='train',
    train_ratio=dataset_config['train_split'],
)
```

### 方法 2: 使用配置文件

创建 `configs/demo_dataset.yaml`：

```yaml
dataset:
  type: "T3Dataset"
  json_path: "./demodataset/annotations/demo_data.json"
  train_split: 0.8
  val_split: 0.2

training:
  batch_size: 4
  grad_accum: 1
  learning_rate: 1e-4
  max_epochs: 10

model:
  name: "ControlLDM"
  checkpoint: "./models/iic/cv_anytext2/anytext_v2.0.ckpt"
```

## 📝 使用示例

### 示例 1: 测试训练脚本

```bash
# 1. 创建演示数据集
python create_demo_dataset.py --num_samples 1000

# 2. 使用演示数据集训练
python train.py \
    --dataset_json ./demodataset/annotations/demo_data.json \
    --batch_size 4 \
    --max_epochs 10
```

### 示例 2: 测试 LCM-LoRA 蒸馏

```bash
# 1. 创建小数据集（快速迭代）
python create_demo_dataset.py --num_samples 500

# 2. 运行蒸馏训练
python student_model/train_lcm_anytext.py \
    --dataset_json ./demodataset/annotations/demo_data.json \
    --lcm_steps 4 \
    --batch_size 8
```

### 示例 3: 验证数据加载

```python
# test_dataset.py
import json
from PIL import Image

# 加载标注
with open('./demodataset/annotations/demo_data.json', 'r') as f:
    data = json.load(f)

# 检查第一个样本
sample = data['data_list'][0]
img_path = f"./demodataset/imgs/{sample['img_name']}"

# 加载图片
img = Image.open(img_path)
print(f"图片: {img.size}")
print(f"标注: {sample['annotations']}")

img.show()
```

## ⚠️ 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间存储输出文件
2. **随机性**: 使用相同 seed 会得到相同的样本集合
3. **数据平衡**: 脚本自动平衡 LAION 和 Wukong 的比例
4. **路径问题**: 确保 `dataset_root` 指向正确的解压后的数据集目录

## 🐛 常见问题

### 问题 1: 找不到图片文件

```
错误: 图片文件不存在
```

**解决**: 检查数据集是否已正确解压，确保目录结构为：
```
dataset/
├── laion/laion_p1/imgs/*.jpg
├── laion/laion_p2/imgs/*.jpg
└── ...
```

### 问题 2: JSON 格式错误

```
错误: JSON 格式不正确，缺少 data_list
```

**解决**: 确保使用的是 AnyText2 v2.0 的标注文件（data_v1.2b.json）

### 问题 3: 权限错误

```
错误: Permission denied
```

**解决**: 检查输出目录的写权限
```bash
chmod +w ./demodataset
```

## 📚 相关文档

- [AnyText2 训练指南](./AnyText2_项目全面解析.md)
- [LCM-LoRA 蒸馏教程](./student_model/train_lcm_anytext.py)
- [数据集格式说明](./t3_dataset.py)

---

**创建时间**: 2026-01-06
**脚本版本**: v1.0
**作者**: AnyText2 Team
