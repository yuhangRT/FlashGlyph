# AnyText2 推理流程

## 概述

AnyText2 推理涉及通过多阶段流程处理多个输入模态（文本、字体、颜色、位置）以生成带有文本的图像。本文档解释从输入到输出的完整推理流程。

## 入口点

### 1. 演示界面 (Gradio)

**文件**: [demo.py](../demo.py)

基于 Web 的界面，有两种模式：
- **文本生成**：从头开始生成文本
- **文本编辑**：编辑图像中的现有文本

### 2. 模型包装器

**文件**: [ms_wrapper.py](../ms_wrapper.py)

`AnyText2Model` 类包装模型用于推理：
```python
class AnyText2Model:
    def __init__(self, model_dir, use_fp16, use_translator, font_path):
        # 加载模型
        # 加载 LoRA 权重
        # 初始化 OCR
        # 设置翻译器（可选）

    def process(self, mode, img_prompt, text_prompt, ...):
        # 主推理入口点
```

## 完整推理流程

```
用户输入
    ↓
1. 输入处理
    - 解析文本、字体、颜色、位置
    - 翻译中文 → 英文（可选）
    ↓
2. 多模态编码
    - 文本编码 (BERT/CLIP)
    - 字体编码（字形渲染）
    - 颜色编码（RGB → 嵌入）
    - 位置编码（掩码处理）
    ↓
3. 嵌入融合
    - EmbeddingManager 组合模态
    - 替换占位符标记 (*)
    ↓
4. 条件准备
    - CLIP 文本编码用于提示
    - 控制信号生成
    ↓
5. DDIM 采样循环
    - 应用 ControlNet 控制
    - AttnX 调制的注意力
    - 迭代去噪
    ↓
6. 图像解码
    - VAE 解码器：latent → image
    - 后处理
    ↓
输出：生成的图像
```

## 步骤 1：输入处理

### 文本解析

**文件**: [ms_wrapper.py](../ms_wrapper.py) - `modify_prompt()`

```python
# 从 text_prompt 中提取引用的字符串
# 示例：'"Any" "Text" "2"' → ['Any', 'Text', '2']
texts = re.findall(r'"([^"]*)"', text_prompt)
```

### 翻译（可选）

```python
if use_translator:
    # 检测中文字符
    # 翻译成英文以获得更好的提示
    img_prompt = translate(img_prompt)  # 中文 → 英文
```

### 位置图生成

**文件**: [ms_wrapper.py](../ms_wrapper.py) - `separate_pos_imgs()`

对于**文本生成**：
```python
# 用户在绘图板上绘制（黑底白色）
position_map = sketchpad_to_mask(user_drawing)
# 连通分量 → 分离的文本行掩码
```

对于**文本编辑**：
```python
# 从参考图像创建位置图
position_map = extract_from_reference(image)
```

### 字体处理

```python
if font == "Mimic From Image":
    # 从参考图像裁剪文本区域
    # 通过 OCR 生成字体提示
    font_hint = crop_region(image, position)
else:
    # 使用指定的字体文件
    font_hint = load_font(font_path)
```

### 颜色处理

```python
# 将颜色字符串转换为 RGB
# 示例："red" → (255, 0, 0)
# 示例："#FF0000" → (255, 0, 0)
color_rgb = parse_color(color_string)
```

## 步骤 2：多模态编码

### 文本编码

**文件**: [bert_tokenizer.py](../bert_tokenizer.py)

```python
# 1. BERT 分词
tokens = bert_tokenize(text_prompt)

# 2. CLIP 文本编码（用于 img_prompt）
img_emb = clip_encoder.encode(img_prompt)

# 3. 文本提示编码（带有占位符）
text_emb = clip_encoder.encode(text_prompt)  # 包含 * 占位符
```

### 字形/字体编码

**文件**: [ms_wrapper.py](../ms_wrapper.py) - `draw_glyph()`, `draw_glyph2()`

```python
# 为每行文本生成字形图像
for text_line, font, color, position:
    # 将文本渲染为 512x80 灰度图像
    glyph = render_text(
        text=text_line,
        font=font,
        color=color,
        size=optimal_font_size(position)
    )
```

### 颜色编码

**文件**: [cldm/embedding_manager.py](../cldm/embedding_manager.py)

```python
# 选项 1：直接线性投影
color_emb = linear_layer(rgb_color)  # (3,) → (token_dim,)

# 选项 2：傅里叶编码（如果 color_fourier_encode=True）
color_emb = fourier_encode(rgb_color)  # 更高维度
```

### 位置编码

```python
# 处理位置掩码
for position_mask in positions:
    # 1. 通过轮廓检测提取多边形
    polygon = find_contours(position_mask)

    # 2. 生成位置提示
    # 所有位置掩码的总和 → 单个提示图像
```

## 步骤 3：嵌入融合

**文件**: [cldm/embedding_manager.py](../cldm/embedding_manager.py) - `forward()`

```python
# EmbeddingManager 用多模态嵌入替换 * 标记

def forward(self, text_tokens, text_info):
    # 1. 文本内容的 OCR 编码
    ocr_emb = self.recognizer.encode(glyphs)

    # 2. 位置编码
    pos_emb = self.position_encoder(positions)

    # 3. 风格（字体）编码
    style_emb = self.style_encoder(font_hints)

    # 4. 颜色编码
    color_emb = self.color_encoder(colors)

    # 5. 替换占位符标记 (*)
    for each * in text_tokens:
        text_tokens[*] = fuse(ocr_emb, pos_emb, style_emb, color_emb)

    return text_tokens
```

### 融合策略

```
输入文本："A photo with * and * text"
                  ↓        ↓
            嵌入 1      嵌入 2
                  ↓        ↓
融合后："A photo with [emb1] and [emb2] text"
```

每个嵌入包含：
- 文本内容（OCR 特征）
- 位置（空间位置）
- 风格（字体特征）
- 颜色（RGB 属性）

## 步骤 4：条件准备

**文件**: [cldm/cldm.py](../cldm/cldm.py) - `get_learned_conditioning()`

```python
cond = model.get_learned_conditioning({
    'c_concat': [hint],                    # 位置提示（1 通道）
    'c_crossattn': [
        [img_prompt + ', ' + a_prompt],    # 场景描述
        [text_prompt]                       # 带有 * 替换的文本
    ],
    'text_info': {
        'glyphs': glyphs,                   # 字形图像
        'positions': positions,             # 位置掩码
        'colors': colors,                   # RGB 颜色
        'languages': languages,             # 每行的语言
        'n_lines': n_lines                  # 文本行数
    }
})
```

### 控制信号生成

**文件**: [cldm/cldm.py](../cldm/cldm.py) - `ControlNet.forward()`

```python
# 1. 编码字形
glyph_features = self.glyph_block(glyphs)  # → 多尺度特征

# 2. 编码位置
pos_features = self.position_block(positions)  # → 多尺度特征

# 3. 融合并生成控制信号
control = self.control_blocks(glyph_features, pos_features)

# 输出：13 个不同尺度的控制张量
```

## 步骤 5：DDIM 采样

**文件**: [cldm/ddim_hacked.py](../cldm/ddim_hacked.py) - `DDIMSampler`

集成 ControlNet 的改进 DDIM：

### 采样循环

```python
# 初始化
x_T = torch.randn(batch_size, 4, 64, 64)  # 随机噪声

# 迭代去噪
for i, t in enumerate(timesteps):  # 默认：20 步
    # 1. 模型预测
    model_output = model.apply_model(
        x=x_t,
        t=t,
        cond=cond,
        control=control  # ControlNet 信号
    )

    # 2. DDIM 去噪步骤
    pred_x0 = (x_t - sqrt_one_minus_at * model_output) / sqrt_at
    x_prev = sqrt_at_prev * pred_x0 + sqrt_one_minus_at_prev * model_output

    # 3. 无分类器引导
    if cfg_scale > 1:
        x_prev = x_prev + (cfg_scale - 1) * (x_prev_cond - x_prev_uncond)

    x_t = x_prev

# 输出：去噪的潜在表示
return x_t
```

### 与标准 DDIM 的关键区别

| 特性 | 标准 DDIM | AnyText2 DDIM |
|---------|--------------|---------------|
| 控制 | 无 | ControlNet 提示 |
| 注意力 | 标准 | AttnX 调制 |
| 引导 | 仅 CFG | CFG + 控制比例 |
| 参数 | eta, steps | eta, steps, attnx_scale |

### 采样期间的 AttnX

```python
# 在 BasicTransformerBlock 中
if attnx:
    # 带文本条件的并行注意力
    x = attn1x(norm1(x), context) * attnx_scale + x
    x = attn2x(norm2(x), context) * attnx_scale + x
```

`attnx_scale` 控制文本渲染强度：
- `attnx_scale = 0`：无文本条件
- `attnx_scale = 1.0`：标准强度
- `attnx_scale > 1.0`：更强的文本（可能降低质量）

## 步骤 6：图像解码

### VAE 解码

```python
# 将潜在表示解码到像素空间
x_samples = model.decode_first_stage(latents)

# 后处理
x_samples = rearrange(x_samples, 'b c h w -> b h w c')
x_samples = (x_samples * 127.5 + 127.5).clip(0, 255)  # [-1,1] → [0,255]
x_samples = x_samples.astype(np.uint8)
```

### 调试可视化（可选）

```python
if show_debug:
    # 包含字形和位置提示
    debug_img = concat([
        generated_image,
        glyph_images,
        position_hints
    ])
    return debug_img
```

## 推理参数

### 核心参数

| 参数 | 默认值 | 范围 | 描述 |
|-----------|---------|-------|-------------|
| `ddim_steps` | 20 | 10-50 | 去噪步数 |
| `cfg_scale` | 7.5 | 1-15 | 无分类器引导 |
| `eta` | 0.0 | 0-1 | DDIM 随机性 |
| `attnx_scale` | 1.0 | 0-2 | 文本注意力强度 |
| `strength` | 1.0 | 0-1 | 控制强度（编辑） |

### 文本参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `font_hollow` | False | 使用空心字体渲染 |
| `revise_pos` | True | 根据字形大小自动调整位置 |
| `sort_priority` | vertical | 文本行排序（↔ 或 ↕） |

### 质量参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `a_prompt` | (空) | 正面质量提示 |
| `n_prompt` | (空) | 负面提示 |
| `use_fp16` | True | 使用半精度 |

## 推理模式

### 1. 文本生成模式

```python
mode = 'gen'
input:
    - img_prompt: "A photo of coffee cup"
    - text_prompt: '"Starbucks"'
    - position: 用户在空白画布上绘制
    - font: "Arial_Unicode"
    - color: "white"

output: 带有生成文本的新图像
```

### 2. 文本编辑模式

```python
mode = 'edit'
input:
    - img_prompt: (来自参考图像)
    - text_prompt: '"New" "Text"'
    - reference_image: 带有旧文本的图像
    - position: (来自参考图像)
    - font: "Mimic From Image"
    - color: (来自参考图像)

output: 带有新文本的编辑图像
```

## 内存需求

| 配置 | 内存 |
|---------------|--------|
| FP16，无翻译器 | ~6 GB |
| FP16，带翻译器 | ~10 GB |
| FP32，无翻译器 | ~12 GB |
| FP32，带翻译器 | ~16 GB |

## 推理示例

### 示例 1：简单生成

```python
result = inference.process(
    mode='gen',
    img_prompt='photo of caramel macchiato coffee',
    text_prompt='"Any" "Text"',
    font=['Arial_Unicode', 'Arial_Unicode'],
    color=['#FFFFFF', '#FFFFFF'],
    position=[pos1, pos2],  # 用户绘制
    ddim_steps=20,
    cfg_scale=7.5
)
```

### 示例 2：字体模仿

```python
result = inference.process(
    mode='edit',
    img_prompt='',
    text_prompt='"New" "Text"',
    reference_image='coffee_with_logo.jpg',
    font=['Mimic From Image', 'Mimic From Image'],
    color=['#FFFFFF', '#FFFFFF'],
    ddim_steps=20,
    strength=0.8  # 编辑时使用较低的值
)
```

### 示例 3：多风格生成

```python
result = inference.process(
    mode='gen',
    img_prompt='neon sign at night',
    text_prompt='"Open" "24" "Hours"',
    font=['NeonFont', 'NeonFont', 'NeonFont'],
    color=['#FF0000', '#00FF00', '#0000FF'],  # RGB
    position=[...],
    sort_priority='horizontal'  # 从左到右
)
```

## 推理关键文件

| 组件 | 文件 |
|-----------|------|
| 演示界面 | [demo.py](../demo.py) |
| 模型包装器 | [ms_wrapper.py](../ms_wrapper.py) |
| 采样 | [cldm/ddim_hacked.py](../cldm/ddim_hacked.py) |
| 文本分词器 | [bert_tokenizer.py](../bert_tokenizer.py) |
| 嵌入管理器 | [cldm/embedding_manager.py](../cldm/embedding_manager.py) |
| 工具函数 | [util.py](../util.py) |

## 总结

AnyText2 推理流程：
1. **输入处理** - 解析和翻译用户输入
2. **多模态编码** - 编码文本、字体、颜色、位置
3. **嵌入融合** - 通过 EmbeddingManager 组合模态
4. **条件** - 准备 ControlNet 信号
5. **DDIM 采样** - 使用 AttnX 迭代去噪
6. **解码** - VAE 解码器 + 后处理

该流程能够精确控制带有可定制属性的文本生成，同时保持视觉质量和文本准确性。
