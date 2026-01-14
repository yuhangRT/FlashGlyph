# AnyText2 模型架构

## 概述

AnyText2 实现了一个复杂的文本到图像生成架构，将 Stable Diffusion 与自定义文本渲染能力相结合。该架构通过多模态条件系统实现对文本属性（字体、颜色、位置）的精确控制。

## 架构层次

```
ControlLDM (主模型)
├── LatentDiffusion (基础扩散模型)
│   ├── DDPM (去噪扩散概率模型)
│   ├── AutoencoderKL (VAE 编码器/解码器)
│   └── FrozenCLIPEmbedderT3 (文本编码器)
├── ControlNet (文本渲染控制)
│   ├── glyph_block (字形/文本编码器)
│   └── position_block (位置编码器)
├── ControlledUnetModel (改进的 UNet)
│   └── WriteNet+AttnX 扩展
└── EmbeddingManager (多模态文本嵌入)
    ├── TextRecognizer (OCR 组件)
    └── EncodeNet (位置/风格编码器)
```

## 核心组件

### 1. ControlLDM (主模型)

**文件**: [cldm/cldm.py](../cldm/cldm.py) (第 390-745 行)

`ControlLDM` 是协调所有组件的主模型类：

```python
class ControlLDM(LatentDiffusion):
    def __init__(self, ...):
        # 继承自 LatentDiffusion
        # 添加:
        # - control_model: 用于空间控制的 ControlNet
        # - embedding_manager: 多模态文本嵌入
        # - cn_recognizer: OCR 文本识别器
```

**主要职责**:
- 管理潜在扩散过程
- 集成 ControlNet 控制信号
- 协调多模态文本嵌入
- 处理训练和推理的前向传播

### 2. ControlNet

**文件**: [cldm/cldm.py](../cldm/cldm.py) (第 75-388 行)

`ControlNet` 提供对文本生成的空间控制：

```python
class ControlNet(nn.Module):
    def __init__(self, ...):
        self.glyph_block = [...]    # 编码文本字形
        self.position_block = [...]  # 编码位置信息
        self.control_blocks = [...]  # 多尺度控制信号
```

**架构**:
- **glyph_block**: 处理文本字形图像（1 通道，512x80）
- **position_block**: 处理位置掩码（1 通道）
- **control_blocks**: 在多个尺度上生成控制信号

**前向传播** (第 354-387 行):
1. 分别编码字形和位置
2. 与掩码图像区域融合
3. 为 UNet 生成多尺度控制信号

### 3. ControlledUnetModel

**文件**: [cldm/cldm.py](../cldm/cldm.py) (第 36-73 行)

`ControlledUnetModel` 扩展了基础 UNet，集成了 ControlNet：

```python
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps, context, control, attnx_scale, ...):
        # 标准 UNet 前向传播，带有:
        # - 在跳跃连接处注入控制信号
        # - attnx_scale 用于注意力调制
```

**关键特性**:
- 接收来自 ControlNet 的控制信号
- 通过跳跃连接在多个尺度注入控制
- 支持 `attnx_scale` 进行细粒度注意力控制

### 4. WriteNet+AttnX 架构

**位置**: [ldm/modules/attention.py](../ldm/modules/attention.py) (第 264-284 行)

**AttnX** 机制创建并行注意力路径：

```python
class BasicTransformerBlock:
    def forward(self, x, context, attnx_scale):
        if self.attnx:
            # 并行注意力流
            x = self.attn1x(self.norm1(x), context=context) * attnx_scale + x
            x = self.attn2x(self.norm2(x), context=context) * attnx_scale + x
```

**作用**:
- `attn1x`: 带文本条件的空间自注意力
- `attn2x`: 用于文本内容注入的交叉注意力
- `attnx_scale`: 控制文本渲染注意力的强度

### 5. EmbeddingManager

**文件**: [cldm/embedding_manager.py](../cldm/embedding_manager.py) (第 113-340 行)

`EmbeddingManager` 管理多模态文本嵌入：

```python
class EmbeddingManager(nn.Module):
    def __init__(self, ...):
        self.recognizer = TextRecognizer(...)     # 基于 OCR 的文本编码
        self.position_encoder = EncodeNet(...)     # 位置编码
        self.style_encoder = EncodeNet(...)        # 字体风格编码
        self.color_encoder = [...]                 # 颜色编码
```

**嵌入类型**:

| 模态 | 编码器 | 用途 |
|----------|---------|---------|
| 文本 (OCR) | TextRecognizer | 字符级嵌入 |
| 位置 | EncodeNet | 空间位置嵌入 |
| 风格 (字体) | TextRecognizer/EncodeNet | 字体风格嵌入 |
| 颜色 | Linear/Fourier | RGB 颜色嵌入 |

**前向传播**:
1. 从文本、位置、风格、颜色提取嵌入
2. 替换文本提示中的占位符标记（`*`）
3. 返回用于条件控制的融合嵌入

### 6. TextRecognizer (OCR 组件)

**文件**: [cldm/recognizer.py](../cldm/recognizer.py) (第 128-262 行)

`TextRecognizer` 提供基于 OCR 的文本编码：

```python
class TextRecognizer:
    def __init__(self, ...):
        self.predictor = create_predictor(...)  # PP-OCRv3 模型
        # MobileNetV1 骨干网络 + RNN 头部
```

**特性**:
- 基于 PP-OCRv3 (PaddleOCR)
- 输出字符预测和颈部特征
- 支持多种语言（中文、英文等）
- 提供用于训练监督的 CTC 损失

### 7. EncodeNet

**文件**: [cldm/embedding_manager.py](../cldm/embedding_manager.py) (第 87-111 行)

`EncodeNet` 编码空间特征（位置/风格）：

```python
class EncodeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 4 层卷积下采样器
        # AdaptiveAvgPool2d 用于全局池化
```

**架构**:
- 4 个卷积层，步长为 2 进行下采样
- 无论输入空间大小如何，都输出固定大小的嵌入

## 模型配置

**配置文件**: [models_yaml/anytext2_sd15.yaml](../models_yaml/anytext2_sd15.yaml)

### ControlNet 配置
```yaml
control_stage_config:
  model_channels: 320
  glyph_channels: 1
  position_channels: 1
  attention_resolutions: [4, 2, 1]
  channel_mult: [1, 2, 4, 4]
  num_heads: 8
  context_dim: 768  # CLIP 嵌入大小
```

### Embedding Manager 配置
```yaml
embedding_manager_config:
  emb_type: ocr           # 基于 OCR 的嵌入
  add_pos: true           # 添加位置编码
  add_style_ocr: true     # 添加 OCR 风格编码
  add_color: true         # 添加颜色编码
  style_ocr_trainable: true
```

### UNet 配置
```yaml
unet_config:
  input_attnx: False
  mid_attnx: True
  output_attnx: True
  output_attnx_level: [2, 1]  # 哪些输出块使用 AttnX
```

## 前向传播流程

### 训练前向传播

```
输入: images, text_info (glyphs, positions, colors, languages)
    ↓
1. get_input() - 准备输入
    - 处理文本信息
    - 准备控制信号
    ↓
2. get_learned_conditioning() - 编码提示
    - CLIP 文本编码器用于图像/文本标题
    - EmbeddingManager 用于多模态文本嵌入
    ↓
3. apply_model() - UNet 前向传播
    - ControlledUnetModel 带 ControlNet 信号
    - AttnX 注意力调制
    ↓
4. p_losses() - 计算损失
    - 扩散损失 (MSE)
    - OCR 感知损失 (可选)
    - CTC 损失 (可选)
    ↓
输出: loss, loss_dict
```

### 推理前向传播

```
输入: prompts, text, glyphs, positions, colors
    ↓
1. 预处理输入
    - 分词提示
    - 从文本生成字形
    - 准备位置掩码
    ↓
2. 编码条件
    - CLIP 文本编码器
    - EmbeddingManager 多模态融合
    ↓
3. DDIM 采样循环
    - 应用 ControlNet 控制信号
    - AttnX 调制的注意力
    - 逐步去噪
    ↓
4. 解码潜在表示
    - VAE 解码器: latent → image
    ↓
输出: 生成的图像
```

## 关键文件位置

| 组件 | 文件 | 行数 |
|-----------|------|-------|
| ControlLDM | [cldm/cldm.py](../cldm/cldm.py) | 390-745 |
| ControlNet | [cldm/cldm.py](../cldm/cldm.py) | 75-388 |
| ControlledUnetModel | [cldm/cldm.py](../cldm/cldm.py) | 36-73 |
| EmbeddingManager | [cldm/embedding_manager.py](../cldm/embedding_manager.py) | 113-340 |
| TextRecognizer | [cldm/recognizer.py](../cldm/recognizer.py) | 128-262 |
| EncodeNet | [cldm/embedding_manager.py](../cldm/embedding_manager.py) | 87-111 |
| AttnX | [ldm/modules/attention.py](../ldm/modules/attention.py) | 264-284 |
| LatentDiffusion | [ldm/models/diffusion/ddpm.py](../ldm/models/diffusion/ddpm.py) | 554-1100 |

## 模型参数

### 关键维度

| 参数 | 值 | 描述 |
|-----------|-------|-------------|
| `context_dim` | 768 | CLIP 文本嵌入维度 |
| `model_channels` | 320 | 基础 UNet 通道数 |
| `channels` | 4 | 潜在空间通道数 (VAE) |
| `image_size` | 64 | 潜在空间分辨率 (512x512 → 64x64) |
| `glyph_channels` | 1 | 字形图像通道数（灰度） |
| `position_channels` | 1 | 位置掩码通道数 |

### 注意力配置

| 参数 | 值 | 描述 |
|-----------|-------|-------------|
| `num_heads` | 8 | 注意力头数 |
| `transformer_depth` | 1 | 每个块的 Transformer 层数 |
| `attention_resolutions` | [4,2,1] | 带注意力的分辨率 |
| `input_attnx` | False | 输入块中的 AttnX |
| `mid_attnx` | True | 中间块中的 AttnX |
| `output_attnx` | True | 输出块中的 AttnX |

## 总结

AnyText2 的架构结合了：
1. **ControlNet** 用于对文本放置的精确空间控制
2. **WriteNet+AttnX** 通过并行注意力增强文本渲染
3. **EmbeddingManager** 用于多模态文本属性编码
4. **基于 OCR 的监督** 用于训练期间的文本准确性

该架构能够生成具有可定制属性（字体、颜色、位置）的文本图像，同时保持视觉真实感和文本准确性。
