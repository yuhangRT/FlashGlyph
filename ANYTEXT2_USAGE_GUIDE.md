# AnyText2 原始模型使用指南

本文档将教你如何不使用自定义的 student_model，而是使用原始的 AnyText2 模型进行实验。

## 📋 目录
1. [环境准备](#环境准备)
2. [模型下载与准备](#模型下载与准备)
3. [推理使用](#推理使用)
4. [训练流程](#训练流程)
5. [实验建议](#实验建议)

---

## 环境准备

### 1. 创建并激活 conda 环境

```bash
conda env create -f environment.yaml
conda activate anytext2
```

### 2. 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 模型下载与准备

### 方案一：使用官方预训练模型（推荐用于推理）

#### 1. 下载官方 AnyText2 模型

```bash
# 使用 modelscope 下载官方模型
python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"
```

这会将模型下载到 `./models/iic/cv_anytext2` 目录，包含：
- `anytext_v2.0.ckpt` - 主要模型检查点
- `clip-vit-large-patch14/` - CLIP 文本编码器
- `nlp_csanmt_translation_zh2en/` - 中英翻译模型（可选）

#### 2. 验证模型文件

```bash
ls -la models/iic/cv_anytext2/
```

应该看到：
- `anytext_v2.0.ckpt`
- `clip-vit-large-patch14/`
- 其他辅助文件

---

### 方案二：从头准备训练模型（用于训练）

如果你想从头训练或微调模型，需要先创建一个基础检查点：

#### 1. 下载 Stable Diffusion 1.5 基础模型

你需要先下载 SD1.5 模型文件（例如 `v1-5-pruned.ckpt`），可以从以下来源获取：
- HuggingFace: `runwayml/stable-diffusion-v1-5`
- 其他来源

假设下载后保存在：`/path/to/v1-5-pruned.ckpt`

#### 2. 运行模型合并脚本

```bash
# 语法：python tool_add_anytext.py [SD1.5模型路径] [输出路径]
python tool_add_anytext.py /path/to/v1-5-pruned.ckpt ./models/anytext2_sd15_scratch.ckpt
```

这个脚本会：
- 加载 SD1.5 的基础权重
- 添加 AnyText2 的 ControlNet 架构（WriteNet）
- 添加 AttnX 层的权重
- 添加 OCR 识别器权重（用于训练时的辅助损失和字体风格编码）

输出文件 `anytext2_sd15_scratch.ckpt` 将用于训练。

---

## 推理使用

### 方法一：使用 Gradio Web 界面（交互式使用）

这是最直观的方式：

```bash
# 启动 Gradio 界面
python demo.py

# 可选参数：
python demo.py --use_fp32          # 使用 fp32 而非 fp16
python demo.py --no_translator     # 禁用中文翻译器（节省 ~4GB 显存）
python demo.py --font_path /path/to/font.ttf  # 指定默认字体
python demo.py --model_path /path/to/custom.ckpt  # 加载自定义检查点
```

启动后：
1. 打开浏览器访问显示的地址（通常是 `http://localhost:7860`）
2. 选择模式：Text Generation（文本生成）或 Text Editing（文本编辑）
3. 输入图像提示词和文字提示词
4. 用画刷指定文字位置
5. 点击 "Run" 生成结果

---

### 方法二：使用简单推理脚本（编程方式）

查看 `simple_demo.py`，这是一个简化的推理示例：

```bash
# 运行简单推理脚本
python simple_demo.py
```

这个脚本会：
1. 加载模型
2. 执行一次文本生成任务
3. 保存结果图像

---

### 方法三：自定义 Python 脚本（灵活控制）

创建一个自定义推理脚本 `my_inference.py`：

```python
import os
import numpy as np
from PIL import Image
from ms_wrapper import AnyText2Model

# 1. 加载模型
model = AnyText2Model(
    model_dir='./models/iic/cv_anytext2',  # 模型目录
    use_fp16=True,                         # 使用半精度
    use_translator=False,                  # 是否使用翻译器
    font_path='font/Arial_Unicode.ttf'     # 字体路径
).cuda(0)

# 2. 准备输入
input_data = {
    'img_prompt': 'A photo of a coffee shop',  # 图像提示词
    'text_prompt': 'with a sign that reads "Coffee Shop"',  # 文字提示词
    'seed': 42,                                 # 随机种子
    'draw_pos': None,                           # 位置图（None=自动生成）
    'ori_image': None                           # 原图（编辑模式需要）
}

# 3. 设置参数
params = {
    'mode': 'text-generation',                  # 模式：text-generation 或 text-editing
    'sort_priority': '↕',                       # 位置排序优先级
    'show_debug': True,                         # 显示调试信息
    'revise_pos': False,                        # 修正位置
    'image_count': 4,                           # 生成图片数量
    'ddim_steps': 20,                           # 采样步数
    'image_width': 512,                         # 图像宽度
    'image_height': 512,                        # 图像高度
    'strength': 1.0,                            # 控制强度
    'attnx_scale': 1.0,                         # AttnX 缩放
    'font_hollow': True,                        # 使用空心字体
    'cfg_scale': 9.0,                           # CFG 强度
    'seed': 42,                                 # 种子
    'eta': 0.0,                                 # DDIM eta
    'a_prompt': 'best quality, extremely detailed',  # 正向提示词
    'n_prompt': 'low-res, bad anatomy',         # 负向提示词
    'base_model_path': '',                      # 基础模型路径（可选）
    'lora_path_ratio': '',                      # LoRA 路径和比例（可选）
    'glyline_font_path': ['No Font(不指定字体)']*5,  # 每行字体
    'font_hint_image': [None]*5,                # 字体风格参考图像
    'font_hint_mask': [None]*5,                 # 字体风格掩码
    'text_colors': ' '.join(['500,500,500']*5)  # 每行文字颜色
}

# 4. 执行推理
results, rtn_code, rtn_warning, debug_info = model.forward(input_data, **params)

# 5. 保存结果
if rtn_code >= 0 and results is not None:
    for i, img in enumerate(results):
        output_path = f'output_{i}.png'
        Image.fromarray(img).save(output_path)
        print(f'保存图像到: {output_path}')
else:
    print(f'生成失败: {rtn_warning}')
```

运行：
```bash
python my_inference.py
```

---

### 方法四：文本编辑模式（Editing）

文本编辑模式需要在已有图像上修改文字：

```python
import cv2
import numpy as np

# 加载原图
ori_image = cv2.imread('your_image.jpg')[..., ::-1]  # BGR -> RGB

# 准备编辑区域掩码（白色表示要编辑的区域）
draw_pos = np.zeros((512, 512, 1), dtype=np.uint8)
# 在 draw_pos 上绘制白色区域表示要编辑的位置
cv2.rectangle(draw_pos, (100, 200), (400, 250), 255, -1)

input_data = {
    'img_prompt': 'A book cover',
    'text_prompt': 'with title "My Book"',
    'seed': 123,
    'draw_pos': draw_pos,        # 编辑区域
    'ori_image': ori_image       # 原始图像
}

params = {
    'mode': 'text-editing',      # 切换到编辑模式
    # ... 其他参数同上
}

results, code, warning, debug = model.forward(input_data, **params)
```

---

## 训练流程

### 1. 准备数据集

AnyText2 使用 AnyWord-3M 数据集。你需要：

#### 选项 A：使用官方数据集

1. 下载数据集（参考 `download_dataset.py`）
2. 解压到指定目录
3. 确保目录结构正确

#### 选项 B：创建自己的小数据集

```python
# 参考 create_demo_dataset.py 创建小型测试数据集
python create_demo_dataset.py
```

### 2. 配置训练参数

编辑 `train.py`，关键参数：

```python
# 训练配置
batch_size = 3              # 批次大小
grad_accum = 2              # 梯度累积
learning_rate = 2e-5        # 学习率
max_epochs = 15             # 最大训练轮数

# 数据集路径（需要修改为你的实际路径）
# 第 71-85 行：完整数据集
# 第 87-101 行：200K 子集
```

### 3. 开始训练

#### 从头训练

```bash
# 使用准备好的模型
python train.py
```

#### 从检查点恢复训练

```python
# 在 train.py 中设置
ckpt_path = './models/anytext2_sd15_scratch.ckpt'  # 或之前的检查点
```

然后运行：
```bash
python train.py
```

### 4. 监控训练

训练日志会保存在 `logs/` 目录，可以使用 TensorBoard 查看：

```bash
tensorboard --logdir logs/
```

---

## 实验建议

### 实验 1：不同提示词的效果

```python
# 测试不同的 img_prompt 和 text_prompt 组合
prompts = [
    ('A coffee cup with text', '"Hello World"'),
    ('A street sign', '"Main Street"'),
    ('A book cover', '"My Novel"'),
]

for img_p, text_p in prompts:
    # 运行推理...
```

### 实验 2：控制强度的影响

```python
# 测试不同的 strength 值
for strength in [0.5, 0.75, 1.0, 1.25, 1.5]:
    params['strength'] = strength
    # 运行推理...
```

### 实验 3：AttnX 缩放的影响

```python
# 测试不同的 attnx_scale 值
for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
    params['attnx_scale'] = scale
    # 运行推理...
```

### 实验 4：字体和颜色控制

```python
# 测试不同字体和颜色
fonts = ['Arial_Unicode', '站酷快乐体2016修订版', '阿里妈妈东方大楷']
colors = ['255,0,0', '0,255,0', '0,0,255']

params['glyline_font_path'] = [fonts[0], 'No Font(不指定字体)', ...]
params['text_colors'] = colors[0] + ' 500,500,500 500,500,500 ...'
```

### 实验 5：步数和 CFG 的影响

```python
# 测试不同的 ddim_steps 和 cfg_scale
for steps in [10, 20, 30, 50]:
    for cfg in [5.0, 7.5, 9.0, 12.0]:
        params['ddim_steps'] = steps
        params['cfg_scale'] = cfg
        # 运行推理...
```

---

## 常见问题

### Q1: 显存不足怎么办？

```bash
# 方案 1: 使用 fp16
python demo.py --use_fp32  # 不要加这个参数就是默认使用 fp16

# 方案 2: 禁用翻译器
python demo.py --no_translator  # 节省 ~4GB

# 方案 3: 减小图像尺寸
# 在参数中设置 image_width=384, image_height=384

# 方案 4: 减小 batch_size
# 在 train.py 中减小 batch_size
```

### Q2: 如何更换字体？

1. 将字体文件放入 `font/lang_font/` 目录
2. 在 `demo.py` 的 `font_path` 字典中添加：
   ```python
   font_path = {
       "我的字体": "font/lang_font/my_font.ttf",
       # ...
   }
   ```

### Q3: 如何评估模型质量？

项目提供了多种评估脚本：

```bash
# OCR 准确率评估
bash eval/eval_ocr.sh

# CLIP 分数评估
bash eval/eval_clip.sh

# FID 分数评估
bash eval/eval_fid.sh
```

### Q4: 如何使用 LoRA？

```python
params['base_model_path'] = '/path/to/base_model.ckpt'
params['lora_path_ratio'] = '/path/to/lora1.pth 0.7 /path/to/lora2.pth 0.3'
```

---

## 文件结构参考

```
FlashGlyph/
├── models/                          # 模型目录
│   └── iic/cv_anytext2/            # 官方模型
│       ├── anytext_v2.0.ckpt       # 主模型
│       └── clip-vit-large-patch14/ # CLIP 编码器
├── models_yaml/
│   └── anytext2_sd15.yaml          # 模型架构配置
├── cldm/                            # ControlNet 实现
├── ldm/                             # Latent Diffusion 实现
├── ms_wrapper.py                    # 模型推理封装
├── demo.py                          # Gradio Web 界面
├── simple_demo.py                   # 简单推理脚本
├── train.py                         # 训练脚本
├── tool_add_anytext.py             # 模型合并工具
├── t3_dataset.py                    # 数据集类
├── font/
│   └── lang_font/                  # 字体文件目录
└── ocr_weights/
    └── ppv3_rec.pth                # OCR 权重
```

---

## 快速开始清单

✅ **用于推理实验：**
1. `conda env create -f environment.yaml && conda activate anytext2`
2. 下载官方模型：`python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"`
3. 运行 Gradio：`python demo.py` 或自定义脚本

✅ **用于训练实验：**
1. 完成上述步骤
2. 下载 SD1.5 基础模型
3. 运行 `python tool_add_anytext.py` 创建训练用检查点
4. 编辑 `train.py` 设置数据集路径
5. 运行 `python train.py`

---

## 相关文档

- `CLAUDE.md` - 项目详细文档
- `AnyText2_项目全面解析.md` - 中文项目解析
- `md_explain/` - 详细的技术文档目录
