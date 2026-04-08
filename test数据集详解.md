# Test 数据集详解

## 📋 两个命令中的 test 数据集是同一个吗？

### ✅ **是的，必须是同一个！**

```bash
# 步骤1: 生成图像
python eval/anytext2_singleGPU.py \
  --input_json <test数据集.json> \    # ← JSON A
  --output_dir student_model_v3/experiments/generated/teacher50_cn

# 步骤2: OCR评估
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/teacher50_cn \
  --input_json <test数据集.json> \    # ← 必须是同一个 JSON A
  --out_json student_model_v3/experiments/results/teacher50_cn.json
```

**原因**: 
- 步骤1 根据 JSON 中的 `annotations` 生成图像
- 步骤2 需要同一个 JSON 来对比 **生成的图像中的文字** vs **JSON 中的 ground truth 文字**
- 如果用不同的 JSON，OCR 评估就会拿错误的标准答案对比，结果完全无效！

---

## 📦 Test 数据集是什么样的？

### 文件结构

```
test数据集目录/
├── test1k.json              ← 标注文件（核心）
└── imgs/                    ← 原始图像（可选，评估时不一定需要）
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

### JSON 格式详解

**test1k.json** 结构如下：

```json
{
  "data_root": "/path/to/images",
  "data_list": [
    {
      "img_name": "000000001.jpg",
      "caption": "A photo of a street with text",
      "annotations": [
        {
          "polygon": [[10, 20], [200, 20], [200, 60], [10, 60]],
          "text": "Main Street",
          "valid": true,
          "pos": 0
        },
        {
          "polygon": [[50, 100], [300, 100], [300, 140], [50, 140]],
          "text": "Coffee Shop",
          "valid": true,
          "pos": 1
        }
      ]
    },
    {
      "img_name": "000000002.jpg",
      "caption": "A book cover",
      "annotations": [
        {
          "polygon": [[30, 50], [250, 50], [250, 90], [30, 90]],
          "text": "My Book",
          "valid": true,
          "pos": 0
        }
      ]
    }
  ]
}
```

### 关键字段说明

| 字段 | 类型 | 作用 | 示例 |
|------|------|------|------|
| `img_name` | string | 图像文件名 | `"000000001.jpg"` |
| `caption` | string | 图像描述（用于图像提示词） | `"A photo of a street"` |
| `annotations` | array | 文字标注列表 | - |
| `annotations[].polygon` | array | 文字区域的多边形坐标 | `[[x1,y1], [x2,y2], ...]` |
| `annotations[].text` | string | **Ground truth 文字**（用于OCR对比） | `"Main Street"` |
| `annotations[].valid` | boolean | 是否有效标注 | `true` / `false` |
| `annotations[].pos` | int | 位置索引 | `0`, `1`, `2`... |

---

## 🆚 两种 test 数据集

### 1. **中文测试集** (wukong_word/test1k.json)

- **来源**: 悟空数据集（中文图文数据）
- **语言**: 中文文字
- **样本数**: 1000 张图
- **用途**: 评估中文字符识别准确率

```bash
# 在代码中引用
json_path = '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json'
```

### 2. **英文测试集** (laion_word/test1k.json)

- **来源**: LAION 数据集（英文图文数据）
- **语言**: 英文文字
- **样本数**: 1000 张图
- **用途**: 评估英文字符识别准确率

```bash
# 在代码中引用
json_path = '/data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k-sample.json'
```

### 论文中的用法

从 `eval/gen_imgs_anytext2.sh` 可以看到：

```bash
# 中文评估
python eval/anytext2_multiGPUs.py \
  --json_path /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json \
  --output_dir ./anytext2_wukong_generated

# 英文评估
python eval/anytext2_multiGPUs.py \
  --json_path /data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k.json \
  --output_dir ./anytext2_laion_generated
```

---

## ❓ DemoDataset 可以作为 test 数据集吗？

### ✅ **可以，但需要满足条件！**

### 检查你的 demodataset 结构

```bash
ls demodataset/
# 应该看到:
# ├── imgs/
# │   ├── img001.jpg
# │   ├── img002.jpg
# │   └── ...
# └── annotations/
#     └── demo_data.json    ← 这个文件
```

### 查看 demo_data.json 的格式

```bash
cat demodataset/annotations/demo_data.json
```

**必须包含的字段**:

```json
{
  "data_root": "demodataset/imgs",
  "data_list": [
    {
      "img_name": "img001.jpg",
      "caption": "图像描述",
      "annotations": [
        {
          "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
          "text": "文字内容",
          "valid": true
        }
      ]
    }
  ]
}
```

### ✅ 如果 demo_data.json 有上述字段 → 可以直接使用

### ❌ 如果缺少 `annotations` 字段 → 不能直接使用

---

## 🔧 如何将 demodataset 转换为 test 数据集

### 情况1: `create_demo_dataset.py` 生成的数据

`create_demo_dataset.py` 从完整数据集中抽取样本，**已经包含了正确的标注格式**：

```python
# 从 create_demo_dataset.py 可以看到
output_json = {
    "data_root": str(images_dir),     # "demodataset/imgs"
    "data_list": all_samples          # 包含 annotations 字段
}
```

**所以可以直接使用！**

### 使用方法

```bash
# 步骤1: 生成图像（使用 demodataset）
python eval/anytext2_singleGPU.py \
  --input_json demodataset/annotations/demo_data.json \
  --output_dir student_model_v3/experiments/generated/demo_test \
  --ddim_steps 50

# 步骤2: OCR评估（使用同一个 demodataset）
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/demo_test \
  --input_json demodataset/annotations/demo_data.json \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/demo_test.json
```

---

## ⚠️ 重要注意事项

### 1. **图像必须存在**

OCR 评估脚本会在 `--img_dir` 中查找图像：

```python
# 脚本会查找这种命名
img_path = Path(img_dir) / f"{img_key}_{sidx}.jpg"
# 例如: student_model_v3/experiments/generated/demo_test/000000001_0.jpg
```

**如果图像不存在**:
```
images_missing: 1000  ← 全部缺失，评估失败！
```

### 2. **样本数量对应**

```json
{
  "data_list": [
    {...},  // 第1张图
    {...},  // 第2张图
    ...
    {...}   // 第1000张图
  ]
}
```

- 如果 JSON 中有 1000 张图
- 步骤1 会生成 `1000 × num_samples` 张图像
- 步骤2 会查找同样数量的图像

### 3. **Polygon 坐标用于裁剪**

```json
"polygon": [[10, 20], [200, 20], [200, 60], [10, 60]]
```

- OCR 评估脚本使用 polygon 在生成的图像中**裁剪出文字区域**
- 然后识别该区域的文字
- 对比识别结果与 `text` 字段

### 4. **demodataset 的样本量**

```python
# 从 create_demo_dataset.py 默认参数
--num_samples 1000  # 默认抽取 1000 张
```

**建议**:
- 用于测试：100-200 张图足够
- 用于论文报告：需要 1000 张以上

```bash
# 生成小型测试集
python create_demo_dataset.py --num_samples 200 --output_dir ./demodataset_small
```

---

## 📝 完整示例：使用 demodataset 做评估

### 步骤0: 准备 demodataset

```bash
# 生成演示数据集（如果还没有）
python create_demo_dataset.py \
  --dataset_root /path/to/full_dataset \
  --num_samples 500 \
  --output_dir ./demodataset

# 检查输出
ls demodataset/annotations/
# 应该看到: demo_data.json
```

### 步骤1: 验证 JSON 格式

```python
import json

with open('demodataset/annotations/demo_data.json') as f:
    data = json.load(f)

print(f"样本数量: {len(data['data_list'])}")
print(f"第一张图: {data['data_list'][0]['img_name']}")
print(f"标注数量: {len(data['data_list'][0]['annotations'])}")
print(f"第一个标注: {data['data_list'][0]['annotations'][0]['text']}")
```

**期望输出**:
```
样本数量: 500
第一张图: 000000001.jpg
标注数量: 3
第一个标注: 你好世界
```

### 步骤2: 生成图像

```bash
python eval/anytext2_singleGPU.py \
  --input_json demodataset/annotations/demo_data.json \
  --output_dir student_model_v3/experiments/generated/demo_test \
  --ddim_steps 50 \
  --num_samples 4

# 检查输出
ls student_model_v3/experiments/generated/demo_test/ | head
# 应该看到:
# 000000001_0.jpg
# 000000001_1.jpg
# 000000001_2.jpg
# 000000001_3.jpg
# ...
```

### 步骤3: OCR评估

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/demo_test \
  --input_json demodataset/annotations/demo_data.json \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/demo_test.json

# 查看结果
cat student_model_v3/experiments/results/demo_test.json
```

### 步骤4: 解读结果

```json
{
  "avg": {
    "word_acc": 0.752,   // 词准确率 75.2%
    "char_acc": 0.891,   // 字符准确率 89.1%
    "cer": 0.109,        // 字符错误率 10.9%
    "wer": 0.248         // 词错误率 24.8%
  },
  "images_found": 1987,      // 找到 1987 张图
  "images_missing": 13,      // 缺失 13 张
  "input_items": 500,        // 输入 500 张图
  "num_samples_per_input": 4 // 每张图生成 4 个样本
}
```

---

## 🎯 总结对比表

| 维度 | 官方 test1k.json | demodataset/demo_data.json |
|------|------------------|---------------------------|
| **来源** | 从完整数据集精心挑选的1000张 | 随机抽取的N张 |
| **样本数** | 固定 1000 张 | 可配置 (默认1000) |
| **格式** | `{"data_root":..., "data_list":[...]}` | 相同格式 ✅ |
| **包含 annotations** | ✅ 是 | ✅ 是 |
| **可以直接使用** | ✅ 是 | ✅ 是（如果是用 create_demo_dataset.py 生成的） |
| **用于论文报告** | ✅ 推荐 | ⚠️ 需要验证样本代表性 |
| **用于测试调试** | ✅ 可以 | ✅ 推荐（样本少更快） |

---

## ✅ 快速检查清单

在使用 demodataset 作为 test 数据集前，确认：

- [ ] `demo_data.json` 包含 `data_list` 字段
- [ ] 每个条目包含 `img_name`, `caption`, `annotations`
- [ ] `annotations` 中每个条目包含 `polygon`, `text`, `valid`
- [ ] `demodataset/imgs/` 目录中有对应的图像文件
- [ ] JSON 中的 `img_name` 与实际文件名完全匹配

如果以上都满足，就可以直接使用 demodataset 作为 test 数据集！

---

## 💡 建议做法

### 用于调试/测试

```bash
# 使用 demodataset (快速)
python create_demo_dataset.py --num_samples 100 --output_dir ./demodataset_small

# 运行评估
python eval/anytext2_singleGPU.py \
  --input_json demodataset_small/annotations/demo_data.json \
  --output_dir student_model_v3/experiments/generated/debug_test \
  --ddim_steps 20  # 用较少的步数快速测试
```

### 用于论文/正式报告

```bash
# 使用官方 test1k.json (如果有的话)
python eval/anytext2_singleGPU.py \
  --input_json /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json \
  --output_dir student_model_v3/experiments/generated/teacher50_cn \
  --ddim_steps 50
```
