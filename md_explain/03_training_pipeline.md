# AnyText2 训练流程

## 概述

AnyText2 使用 PyTorch Lightning 作为训练基础设施。训练过程对预处理的 Stable Diffusion 检查点进行微调，添加用于文本生成的 ControlNet 组件。

## 训练脚本

**文件**: [train.py](../train.py)

## 训练配置

### 核心参数

**文件**: [train.py](../train.py) (第 20-43 行)

```python
# 模型检查点
ckpt_path = None           # 从特定检查点恢复
resume_path = './models/anytext2_sd15_scratch.ckpt'  # 基础检查点
config_path = './models_yaml/anytext2_sd15.yaml'

# 训练超参数
batch_size = 3             # 每个 GPU 的批次大小
grad_accum = 2             # 梯度累积步数
learning_rate = 2e-5       # 学习率
max_epochs = 15            # 最大训练轮数

# 检查点保存
save_steps = None          # 每 N 步保存（None = 使用轮次）
save_epochs = 1            # 每 N 轮保存
save_ckpt_top = 3          # 保留前 K 个检查点
root_dir = './models'      # 检查点目录

# 数据增强
mask_ratio = 0             # 修复/文本编辑比率（0 = 禁用）
font_hint_prob = 0.8       # 使用字体提示的概率
color_prob = 1.0           # 使用颜色信息的概率
font_hint_area = [0.7, 1]  # 字体提示的保留区域
wm_thresh = 1.0            # 水印过滤阈值

# 字体设置
rand_font = True           # 随机字体选择
font_hint_randaug = True   # 字体提示的随机增强
```

### 模型配置

**文件**: [train.py](../train.py) (第 55-61 行)

```python
model.learning_rate = learning_rate
model.sd_locked = True         # 冻结主扩散模型
model.only_mid_control = False # 使用所有控制块
model.unlockQKV = False        # 保持 QKV 投影冻结
```

## 优化与冻结的组件

### 冻结组件（不更新）

| 组件 | 状态 | 原因 |
|-----------|--------|--------|
| 主 U-Net 骨干网络 | `sd_locked = True` | 保留预训练的 SD 知识 |
| QKV 投影 | `unlockQKV = False` | 防止灾难性遗忘 |
| CLIP 文本编码器 | 冻结 | 稳定的文本条件 |
| VAE 编码器/解码器 | 冻结 | 稳定的潜在空间 |

### 优化组件

| 组件 | 状态 | 用途 |
|-----------|--------|---------|
| ControlNet 参数 | 可训练 | 学习文本渲染控制 |
| AttnX 注意力 (attn1x, attnx2) | 可训练 | 文本内容注入 |
| EmbeddingManager | 可训练（如果 `cond_stage_trainable: true`） | 多模态嵌入 |
| 风格 OCR 编码器 | 可训练（如果 `style_ocr_trainable: true`） | 字体风格编码 |

## 数据集配置

### 数据集路径

**文件**: [train.py](../train.py) (第 71-85 行)

```python
json_paths = [
    # OCR 数据集（中文）
    '/data/.../Art/data_v1.2b.json',
    '/data/.../COCO_Text/data_v1.2b.json',
    '/data/.../icdar2017rctw/data_v1.2b.json',
    '/data/.../LSVT/data_v1.2b.json',
    '/data/.../mlt2019/data_v1.2b.json',
    '/data/.../MTWI2018/data_v1.2b.json',
    '/data/.../ReCTS/data_v1.2b.json',

    # 大规模数据集
    '/data/.../laion_word/data_v1.2b.json',        # 英文
    '/data/.../wukong_word/wukong_1of5/data_v1.2b.json',  # 中文
    '/data/.../wukong_word/wukong_2of5/data_v1.2b.json',
    '/data/.../wukong_word/wukong_3of5/data_v1.2b.json',
    '/data/.../wukong_word/wukong_4of5/data_v1.2b.json',
    '/data/.../wukong_word/wukong_5of5/data_v1.2b.json',
]
```

### 数据集：T3DataSet

**文件**: [t3_dataset.py](../t3_dataset.py)

数据集提供：
- **图像**：来自各种来源的包含文本的图像
- **文本注释**：最多 5 行文本，每行 20 个字符
- **字体提示**：用于风格迁移的字体图像
- **位置掩码**：每行文本的空间位置
- **颜色标签**：每行文本的 RGB 颜色
- **标题**：图像和文本描述

## 数据增强

### 空间增强

**文件**: [t3_dataset.py](../t3_dataset.py)

```python
def random_augment(image, rot=(-10, 10), trans=(-5, 5), scale=(0.9, 1.1)):
    image = random_rotate(image, rot)      # 随机旋转
    image = random_translate(image, trans)  # 随机平移
    image = random_scale(image, scale)     # 随机缩放
    return image
```

应用于：
- 输入图像
- 位置掩码
- 字形图像

### 字体随机化

```python
if rand_font:
    font = random.choice(font_list)  # 每行随机字体
else:
    font = default_font
```

### 掩码比率（修复）

```python
mask_ratio = 0  # 0 = 生成模式
mask_ratio = 0.5  # 0.5 = 编辑模式（掩码 50% 的图像）
```

当 `mask_ratio > 0` 时：
- 随机掩码图像区域
- 模型学习在掩码区域重新生成文本

## 训练模式

### 1. 生成模式 (`mask_ratio = 0`)

从头开始完整图像重建：
- 输入：噪声潜在表示 + 文本条件
- 输出：带有生成文本的清晰图像
- 用例：文本生成

### 2. 编辑模式 (`mask_ratio = 0.5`)

用于文本编辑的部分修复：
- 输入：噪声潜在表示 + 掩码图像 + 文本条件
- 输出：带有编辑文本的清晰图像
- 用例：文本编辑

## 训练循环

### PyTorch Lightning 设置

**文件**: [train.py](../train.py) (第 47-103 行)

```python
# 创建模型
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = True

# 设置训练器
trainer = pl.Trainer(
    accelerator='gpu',
    devices=8,  # 多 GPU
    precision=16,  # 混合精度
    accumulate_grad_batches=grad_accum,
    max_epochs=max_epochs,
    callbacks=[checkpoint_callback, ImageLogger(...)],
    ...
)

# 创建数据加载器
train_data = T3DataSet(json_paths, ...)
train_loader = DataLoader(train_data, batch_size=batch_size, ...)

# 训练
trainer.fit(model, train_loader)
```

### 训练步骤

**文件**: [ldm/models/diffusion/ddpm.py](../ldm/models/diffusion/ddpm.py)

```python
def training_step(self, batch, batch_idx):
    # 1. 获取输入
    x_start = batch['img']  # 图像
    cond = {
        'c_crossattn': [img_caption, text_caption],
        'c_concat': [hint],  # 位置提示
        'text_info': {...}  # 字形、位置、颜色、语言
    }

    # 2. 采样时间步
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # 3. 计算损失
    loss, loss_dict = self.p_losses(x_start, cond, t)

    # 4. 记录
    self.log_dict(loss_dict, prog_bar=True, ...)

    return loss
```

### 优化器

**文件**: [ldm/models/diffusion/ddpm.py](../ldm/models/diffusion/ddpm.py)

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    return optimizer
```

## 日志和监控

### 图像记录器

**文件**: [cldm/logger.py](../cldm/logger.py)

训练期间记录生成的图像：
- 带有文本的输入图像
- 生成/重建的图像
- 字形和位置提示
- OCR 预测（用于调试）

### 损失跟踪

记录的指标：
- `t/sim` - 简单扩散损失
- `t/ocr` - OCR 感知损失
- `t/ctc` - CTC 文本损失
- `t/loss` - 总损失
- 相应的验证指标 (`v/*`)

### 检查点

**文件**: [train.py](../train.py) (第 63-69 行)

```python
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=save_steps,
    every_n_epochs=save_epochs,
    save_top_k=save_ckpt_top,
    monitor="global_step",
    mode="max",
)
```

## 训练工作流

```
1. 环境设置
   conda activate anytext2
   # 从 environment.yaml 安装依赖

2. 模型准备
   python tool_add_anytext.py sd_v1.5.ckpt models/anytext2_sd15_scratch.ckpt
   # 创建 AnyText2 就绪的检查点

3. 数据集准备
   # 下载 AnyWord-3M 数据集
   # 解压 anytext2_json_files.zip
   # 在 train.py 中更新 json_paths

4. 启动训练
   python train.py

5. 监控训练
   # 检查 models/image_log/train/ 中的日志
   # 监控 TensorBoard/WandB 指标

6. 评估
   # 训练后，运行评估脚本
   ./eval/eval_ocr.sh
   ./eval/eval_clip.sh
   ./eval/eval_fid.sh
```

## 多 GPU 训练

**配置**: [train.py](../train.py)

```python
NUM_NODES = 1  # 节点数
# 使用 PyTorch Lightning DDP 进行多 GPU

trainer = pl.Trainer(
    devices=8,  # 8 个 GPU
    accelerator='gpu',
    strategy='ddp',  # 分布式数据并行
    ...
)
```

### 有效批次大小

```
effective_batch_size = batch_size × grad_accum × num_gpus
                     = 3 × 2 × 8
                     = 48
```

## 训练技巧

### 1. 内存管理

- 使用 8 个 GPU 的 `batch_size=3`（24 有效批次大小）
- 启用梯度检查点：`use_checkpoint=True`
- 混合精度 (FP16) 减少约 50% 内存

### 2. 收敛性

- 典型训练：15 轮
- 学习率：2e-5（微调的保守值）
- 监控验证损失以进行早停

### 3. 损失权重

- 从 `loss_alpha=0`、`loss_beta=0` 开始
- 收敛后启用 OCR 损失（`loss_alpha=0.003`）
- CTC 损失（`loss_beta`）可选，用于准确性关键的应用

### 4. 数据质量

- 过滤水印：`wm_thresh=1.0`（全部移除）
- 调整 `font_hint_prob` 以控制字体条件强度
- 使用 `color_prob=1.0` 进行颜色条件

## 常见问题

### 问题 1：内存不足

**解决方案**：
- 减少 `batch_size`
- 启用梯度检查点
- 使用更少但内存更大的 GPU

### 问题 2：文本质量差

**解决方案**：
- 增加 `loss_alpha`（感知 OCR 损失）
- 检查字体提示质量
- 验证数据集注释

### 问题 3：训练不稳定

**解决方案**：
- 降低学习率
- 增加梯度累积
- 检查 NaN 损失（减少 `loss_alpha`/`loss_beta`）

### 问题 4：收敛缓慢

**解决方案**：
- 增加有效批次大小
- 使用学习率预热
- 验证数据集质量和平衡

## 训练配置摘要

| 参数 | 默认值 | 推荐范围 | 描述 |
|-----------|---------|-------------------|-------------|
| `batch_size` | 3 | 1-6 | 每个 GPU 的批次大小 |
| `grad_accum` | 2 | 1-4 | 梯度累积 |
| `learning_rate` | 2e-5 | 1e-5 - 5e-5 | AdamW 学习率 |
| `max_epochs` | 15 | 10-30 | 训练轮数 |
| `mask_ratio` | 0 | 0-0.5 | 编辑模式比率 |
| `font_hint_prob` | 0.8 | 0.5-1.0 | 字体提示概率 |
| `color_prob` | 1.0 | 0.5-1.0 | 颜色信息概率 |
| `loss_alpha` | 0 | 0-0.01 | OCR 损失权重 |
| `loss_beta` | 0 | 0-0.001 | CTC 损失权重 |

## 总结

AnyText2 训练流程：
1. **基础模型**：预处理的 Stable Diffusion 检查点
2. **优化**：ControlNet + AttnX + EmbeddingManager
3. **冻结**：主 U-Net + QKV + CLIP 编码器
4. **数据集**：AnyWord-3M（300 万+ 图像，多语言）
5. **损失**：多组件（扩散 + OCR + CTC）
6. **基础设施**：PyTorch Lightning 支持多 GPU
