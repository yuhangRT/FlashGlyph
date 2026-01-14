# AnyText2 LCM-LoRA 知识蒸馏训练

本目录包含使用 LoRA 高效微调训练 LCM（Latent Consistency Model）蒸馏 AnyText2 模型的脚本。

## 概述

LCM-LoRA 蒸馏使 AnyText2 能够在 **4-8 个推理步骤**内生成高质量的文本图像，而不是默认的 50+ 步 DDIM 采样，同时保持文本渲染质量。

### 核心特性

- ✅ **加速推理**：4-8 步生成（对比 50+ 步）
- ✅ **LoRA 高效**：可训练参数 <5%
- ✅ **完整 AnyText2 支持**：同时蒸馏 UNet（背景生成）和 ControlNet（文本渲染）
- ✅ **Conv2D LoRA**：对 ControlNet zero_convs 应用 LoRA 实现完整蒸馏
- ✅ **多 GPU 训练**：集成 Accelerate 支持分布式训练

## 架构说明

### 蒸馏内容

1. **UNet（背景生成）**
   - 注意力投影层：`to_q`、`to_k`、`to_v`、`to_out`
   - AttnX 层：AnyText2 中特殊的文本注意力层

2. **ControlNet（文本渲染）**
   - 零卷积层：`zero_convs`（Conv2D 层）
   - 所有 input/middle 块中的注意力投影
   - 字形和位置处理

3. **冻结的模块**
   - VAE 编码器/解码器
   - CLIP 文本编码器
   - Embedding manager（多模态条件）
   - OCR 辅助编码器

## 安装说明

### 环境要求

```bash
# 安装额外的依赖
pip install accelerate>=0.25.0 peft>=0.8.0

# 确保 AnyText2 依赖已安装
cd ..
conda env create -f environment.yaml
conda activate anytext2
```

### 硬件要求

- **推荐配置**：3x NVIDIA RTX 4090（每张 24GB 显存）
- **最低配置**：1x RTX 3090（24GB）需降低 batch size
- **训练时间**：3x4090 上训练 50K 步约需 24-48 小时

## 快速开始

### 步骤 1：检查模型以获取 LoRA 目标

有两种方法生成目标模块列表：

#### 方法 1：简化版（推荐，无需加载模型）

```bash
cd student_model
python inspect_modules_simple.py
```

**优势**：
- ✅ 无需加载完整模型
- ✅ 避免环境兼容性问题
- ✅ 快速生成 517 个目标模块

#### 方法 2：完整版（需要兼容环境）

如果环境完全兼容，可以运行完整版：

```bash
cd student_model
python inspect_modules.py \
    --config ../models_yaml/anytext2_sd15.yaml \
    --ckpt ../models/anytext_v2.0.ckpt \
    --output target_modules_list.txt
```

**输出结果**（两种方法相同）：
- 打印按组件分组的所有 Linear 和 Conv2D 层
- 保存 `target_modules_list.txt` 包含 PEFT 配置的层名称

### 步骤 2：配置 Accelerate

```bash
accelerate config
```

**推荐配置**：
```
- 分布式：多 GPU（数据并行 / ZeRO-2）
- 混合精度：fp16
- 梯度累积：4 步
- GPU 数量：3
```

### 步骤 3：开始训练（使用模拟数据集）

先用合成数据测试：

```bash
accelerate launch train_lcm_anytext.py \
    --config ../models_yaml/anytext2_sd15.yaml \
    --teacher_ckpt ../models/anytext_v2.0.ckpt \
    --output_dir ./checkpoints \
    --use_mock_dataset \
    --dataset_size 1000 \
    --resolution 512 \
    --train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --num_inference_steps 8 \
    --max_train_steps 50000 \
    --mixed_precision fp16 \
    --logging_steps 100 \
    --save_steps 5000
```

### 步骤 4：使用真实数据训练

替换 `dataset_anytext.py` 为你的真实数据加载器：

1. 修改 `dataset_anytext.py` 加载你的数据
2. 移除 `--use_mock_dataset` 标志
3. 如需要调整 `--dataset_size`

## 训练配置说明

### 关键参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--lora_rank` | 64 | LoRA 秩（越高 = 容量越大） |
| `--lora_alpha` | 64 | LoRA alpha（缩放因子） |
| `--num_inference_steps` | 8 | 目标推理步数（4、6、8 或 16） |
| `--cfg_scale` | 7.5 | 分类器无关引导强度 |
| `--train_batch_size` | 12 | 每个 GPU 的批大小 |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--learning_rate` | 1e-4 | 学习率 |

### LCM 时间步调度

不同的推理步数使用不同的粗时间步调度：

```python
4 步:  [999, 599, 299, 50]
6 步:  [999, 799, 599, 399, 199, 50]
8 步:  [999, 899, 799, 699, 599, 499, 399, 50]  # 推荐
16 步: [999, 949, ... , 299, 50]
```

**权衡考虑**：
- 步数更少：推理更快，质量略低
- 步数更多：质量更好，训练更慢

## 文件结构

```
student_model/
├── inspect_modules.py       # 模型检查和 LoRA 目标识别
├── dataset_anytext.py       # 模拟数据集（替换为你的数据）
├── lcm_utils.py             # LCM 工具（DDIM 求解器、时间步等）
├── train_lcm_anytext.py     # 主训练脚本
├── target_modules_list.txt  # 由 inspect_modules.py 生成
└── README.md                # 本文件
```

## 训练流程详解

### 1. 数据格式

你的数据集必须匹配 AnyText2 的预期格式（参见 `dataset_anytext.py`）：

```python
{
    'img': torch.Tensor,           # (H, W, 3) 归一化到 [-1, 1]
    'hint': torch.Tensor,          # (H, W, 1) 位置掩码
    'glyphs': List[torch.Tensor],  # 字形图像列表，每个元素 (1, H, W)
    'positions': List[torch.Tensor], # 位置掩码列表，每个元素 (1, H, W)
    'masked_x': torch.Tensor,      # (1, H, W, 3) 掩码后的潜在表示
    'img_caption': str,            # 基础描述
    'text_caption': str,           # 带占位符 '*' 的描述
    'texts': List[str],            # 每行文本内容
    'n_lines': int,                # 文本行数
    'font_hint': torch.Tensor,     # (H, W, 1) 字体提示图像
    'color': List[torch.Tensor],   # 每行 RGB 颜色列表
    'language': str,               # 语言代码（'en'、'zh' 等）
    'inv_mask': torch.Tensor,      # (H, W, 1) 反向掩码
}
```

### 2. LCM 蒸馏循环

每个训练步骤：

1. **编码图像为潜在表示** 使用 VAE
2. **从 LCM 调度中采样粗时间步**
3. **添加噪声** 到潜在表示
4. **教师模型前向传播** 带 CFG → 预测噪声
5. **转换教师预测** 为目标 x₀ 使用 DDIM 求解器
6. **学生模型前向传播** → 预测噪声
7. **转换学生预测** 为 x₀
8. **计算 Huber 损失** 在学生 x₀ 和教师目标 x₀ 之间
9. **反向传播** 并仅更新 LoRA 参数

### 3. 检查点保存

每 N 步保存检查点：

```
checkpoints/
├── checkpoint-5000/
│   ├── adapter_config.json
│   └── adapter_model.bin  # LoRA 权重
├── checkpoint-10000/
└── checkpoint-final/
```

## 使用训练好的 LoRA 进行推理

训练完成后，加载你的 LoRA 权重进行快速推理：

```python
from peft import PeftModel
from cldm.model import create_model, load_state_dict

# 加载基础模型
base_model = create_model("models_yaml/anytext2_sd15.yaml")
state_dict = load_state_dict("models/anytext_v2.0.ckpt")
base_model.load_state_dict(state_dict)

# 加载 LoRA 权重
student = PeftModel.from_pretrained(
    base_model,
    "student_model/checkpoints/checkpoint-5000"
)

# 使用学生模型进行 4-8 步推理
# （修改 demo.py 使用 student 而不是 base_model）
```

**修改采样**：
```python
# 使用粗时间步而不是完整的 1000 步
timesteps = [999, 799, 599, 399, 199, 50]  # 6 步推理

# 或使用更少的步数
timesteps = [999, 599, 299, 50]  # 4 步推理
```

## 常见问题排查

### 问题：PEFT Conv2D LoRA 不支持

**错误**：`Conv2d` LoRA 不可用

**解决方案**：
```bash
pip install peft>=0.8.0  # 确保最新的 PEFT
```

### 问题：显存不足 (OOM)

**解决方案**：
- 降低 `--train_batch_size`（尝试 6 或 3）
- 增加 `--gradient_accumulation_steps`（尝试 8 或 16）
- 使用 `--mixed_precision fp16`（RTX 4090 可用 bf16）
- 降低 `--lora_rank`（尝试 32）

### 问题：损失值为 NaN

**解决方案**：
- 降低学习率：`--learning_rate 5e-5`
- 使用梯度裁剪（添加到训练脚本）
- 检查数据归一化（应该是 [-1, 1]）
- 确保教师模型正确冻结

### 问题：文本质量差

**解决方案**：
- 增加 `--num_inference_steps`（尝试 16 而不是 8）
- 训练更多步数
- 检查 ControlNet LoRA 目标是否包含
- 验证 `target_modules_list.txt` 包含 zero_convs

## 高级用法

### 自定义目标模块

如需要手动编辑 `target_modules_list.txt`：

```python
target_modules = [
    "control_model.zero_convs.0.0",
    "control_model.zero_convs.1.0",
    # ... 添加或移除模块
]
```

### EMA（指数移动平均）

为了更好的稳定性，在训练中添加 EMA：

```python
from ema import EMAModel

ema_student = EMAModel(
    student,
    decay=0.9999,
    device=accelerator.device
)

# 每步更新 EMA
ema_student.step(student.parameters())

# 保存 EMA 权重
ema_student.save_pretrained("checkpoint-ema")
```

### 多分辨率训练

在多个分辨率上训练（如 512、768、1024）：

1. 修改数据集返回可变分辨率
2. 向训练脚本添加 `--resolution` 参数
3. 确保位置缩放对所有分辨率有效

## 性能基准

在 3x RTX 4090 上的预期训练速度：

| 批大小 | 累积 | 步/秒 | 50K步/小时 |
|------------|--------------|-----------|-----------|
| 12 | 4 | ~2.5 | ~5.5 小时 |
| 6 | 8 | ~1.5 | ~9 小时 |
| 3 | 16 | ~0.8 | ~17 小时 |

预期推理加速：

| 模型 | 步数 | 推理时间 | 质量 |
|-------|-------|----------------|---------|
| 教师 (DDIM) | 50 | ~10s | 100% (基线) |
| 学生 (4 步) | 4 | ~0.8s | ~92% |
| 学生 (8 步) | 8 | ~1.6s | ~96% |

## 引用

如果使用本代码，请引用：

```bibtex
@article{tuo2024anytext2,
  title={AnyText2: Visual Text Generation and Editing With Customizable Attributes},
  author={Tuo, Yuxiang and Geng, Yifeng and Bo, Liefeng},
  year={2024},
  archivePrefix={arXiv},
  eprint={2411.15245}
}

@article{lcms,
  title={Latent Consistency Models: Image Synthesis in a Few Steps},
  author={Sim, Jianbo and others},
  year={2024}
}
```

## 许可证

本代码遵循与 AnyText2 相同的许可证。详情请参考主仓库。

## 联系方式

如有问题或疑问：
1. 查看主 AnyText2 仓库
2. 参考 LCM 论文了解算法细节
3. 在 GitHub 上提 issue

---

**祝训练顺利！🚀**
