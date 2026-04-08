# AnyText2 快速开始指南

## 🚀 5分钟快速上手

### 1️⃣ 安装环境 (2分钟)

```bash
# 创建并激活 conda 环境
conda env create -f environment.yaml
conda activate anytext2
```

### 2️⃣ 下载模型 (2分钟)

```bash
# 下载官方 AnyText2 预训练模型
python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"
```

### 3️⃣ 运行Demo (1分钟)

```bash
# 启动 Gradio Web 界面
python demo.py

# 等待显示 URL 后，在浏览器中打开 (通常是 http://localhost:7860)
```

---

## 📖 三种使用方式

### 方式一: Gradio Web 界面 (最简单)

```bash
python demo.py
```

**优点**: 
- 可视化界面
- 可以实时调整参数
- 支持中文翻译

**使用步骤**:
1. 选择模式 (Text Generation 或 Text Editing)
2. 输入图像提示词 (描述你想要的图片)
3. 输入文字提示词 (用双引号包裹要生成的文字)
4. 用画刷指定文字位置
5. 点击 "Run" 生成

---

### 方式二: 简单推理脚本 (适合测试)

```bash
# 运行已有的简单脚本
python simple_demo.py
```

---

### 方式三: 实验脚本 (适合研究)

```bash
# 1. 编辑实验脚本
vim experiment_anytext2.py  # 或用你喜欢的编辑器

# 2. 修改顶部的配置参数
# EXPERIMENT_TYPE = "text_generation"
# IMG_PROMPT = 'A coffee cup with text'
# TEXT_PROMPT = 'that reads "Coffee Time"'
# SEED = 42
# ...

# 3. 运行实验
python experiment_anytext2.py

# 4. 查看结果
ls experiment_results/
```

---

## 🎯 快速实验: 测试不同参数

### 实验 1: 修改提示词

编辑 `experiment_anytext2.py` 顶部:

```python
IMG_PROMPT = 'A street sign with text'
TEXT_PROMPT = 'that reads "Main Street"'
SEED = 123  # 换种随机种子
```

运行:
```bash
python experiment_anytext2.py
```

---

### 实验 2: 测试控制强度

```bash
# 运行批量实验，自动测试不同强度
# 在 experiment_anytext2.py 中修改:
EXPERIMENT_TYPE = "batch"
```

然后运行:
```bash
python experiment_anytext2.py
```

这会生成 5 张图，分别使用 strength = [0.5, 0.75, 1.0, 1.25, 1.5]

---

### 实验 3: 使用不同字体

```python
# 在 experiment_anytext2.py 中修改:
FONTS = [
    '站酷快乐体2016修订版',  # 使用中文字体
    'No Font(不指定字体)',
    'No Font(不指定字体)',
    'No Font(不指定字体)',
    'No Font(不指定字体)',
]

COLORS = [
    '255,0,0',  # 红色
    '500,500,500',
    '500,500,500',
    '500,500,500',
    '500,500,500',
]
```

---

## 🔧 常见问题

### Q: 显存不够怎么办？

```bash
# 方法1: 禁用翻译器 (节省 ~4GB)
python demo.py --no_translator

# 方法2: 使用更小的图片
# 在脚本中设置:
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384

# 方法3: 减少生成数量
IMAGE_COUNT = 1
```

### Q: 如何生成中文？

**方式1**: 使用翻译器 (需要更多显存)
```python
model = AnyText2Model(
    model_dir='./models/iic/cv_anytext2',
    use_translator=True,  # 启用翻译
)

# 输入中文
TEXT_PROMPT = '上面写着"你好" "世界"'
```

**方式2**: 直接输入英文引号包裹的中文
```python
# 即使不启用翻译器，也可以用Unicode
TEXT_PROMPT = 'that reads "你好" "世界"'
```

### Q: 如何指定文字位置？

**在 Gradio 中**: 用画刷在 Draw Position 面板绘制白色区域

**在脚本中**: 创建位置图
```python
import cv2
import numpy as np

pos_img = np.zeros((512, 512, 1), dtype=np.uint8)
# 绘制白色矩形表示文字位置
cv2.rectangle(pos_img, (50, 100), (250, 180), 255, -1)

input_data['draw_pos'] = pos_img
```

---

## 📊 理解输出

运行实验后，你会在 `experiment_results/` 目录看到:

```
experiment_results/
├── result_00.png          # 第1张生成图
├── result_01.png          # 第2张生成图
├── result_02.png          # 第3张生成图
├── result_03.png          # 第4张生成图
├── debug_glyph.png        # Debug: glyph位置图
├── debug_font_hint.png    # Debug: 字体风格提示图
└── position_map.png       # 你定义的位置图
```

---

## 🎓 进阶: 训练自己的模型

如果你想训练或微调:

```bash
# 1. 下载 SD1.5 基础模型
# (需要从 HuggingFace 或其他来源下载 v1-5-pruned.ckpt)

# 2. 创建 AnyText2 训练用检查点
python tool_add_anytext.py /path/to/v1-5-pruned.ckpt ./models/anytext2_sd15_scratch.ckpt

# 3. 编辑 train.py 设置数据集路径

# 4. 开始训练
python train.py
```

详细训练流程请参考 `ANYTEXT2_USAGE_GUIDE.md`

---

## 📚 更多文档

- `ANYTEXT2_USAGE_GUIDE.md` - 完整使用指南
- `CLAUDE.md` - 项目技术文档
- `md_explain/` - 架构和原理详解

---

## 💡 快速参考: 关键参数说明

| 参数 | 作用 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| `ddim_steps` | 采样步数，越多质量越好但越慢 | 15-30 | 20 |
| `cfg_scale` | CFG强度，越高越遵循提示词 | 7-12 | 9.0 |
| `strength` | ControlNet控制强度 | 0.7-1.3 | 1.0 |
| `attnx_scale` | AttnX注意力层强度 | 0.7-1.3 | 1.0 |
| `seed` | 随机种子，-1表示随机 | -1~99999999 | -1 |
| `IMAGE_COUNT` | 一次生成的图片数量 | 1-12 | 4 |

---

## 🎉 开始你的实验！

现在你可以:
1. 运行 `python demo.py` 进行交互式使用
2. 编辑并运行 `python experiment_anytext2.py` 进行批量实验
3. 阅读 `ANYTEXT2_USAGE_GUIDE.md` 了解更多高级功能

祝实验顺利！
