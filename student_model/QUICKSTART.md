# 快速开始指南

## ✅ 环境检查

当前版本已提供简化工具，可以在环境不完全兼容的情况下运行。

## 🚀 三步开始训练

### 步骤 1：生成 LoRA 目标模块列表

```bash
# 从项目根目录运行
python ./student_model/inspect_modules_simple.py
```

**输出**：
```
总计: 517 个目标模块
✓ 目标模块列表已保存到: student_model/target_modules_list.txt
```

### 步骤 2：配置 Accelerate（首次使用）

```bash
accelerate config
```

**推荐配置**：
```
分布式：多 GPU（数据并行 / ZeRO-2）
混合精度：fp16
梯度累积：4 步
GPU 数量：3
```

### 步骤 3：开始训练（使用模拟数据测试）

```bash
accelerate launch student_model/train_lcm_anytext.py \
    --config models_yaml/anytext2_sd15.yaml \
    --teacher_ckpt models/anytext_v2.0.ckpt \
    --output_dir ./student_model/checkpoints \
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

## 📊 预期输出

训练开始后会看到：
```
================================================================================
Loading teacher model...
================================================================================
✓ Teacher loaded: models/anytext_v2.0.ckpt

================================================================================
Creating student model...
================================================================================
✓ Student base model frozen

================================================================================
Injecting LoRA into student model...
================================================================================
✓ LoRA injected successfully

Trainable parameters: 25,000,000
Total parameters: 860,000,000
Trainable %: 2.91%

================================================================================
Creating dataset...
================================================================================
✓ Mock dataset created: 1000 samples

================================================================================
Preparing training...
================================================================================
✓ Training setup complete
  Device: cuda:0
  Batch size: 12
  Gradient accumulation: 4
  Effective batch size: 48
  Mixed precision: fp16
  Target inference steps: 8

================================================================================
Starting training...
================================================================================

Training:   0%|          | 0/50000 [00:00<?, ?it/s]
```

## 📝 参数说明

### 基础参数
- `--config`: 模型配置文件路径
- `--teacher_ckpt`: 教师模型检查点路径
- `--output_dir`: 检查点保存目录

### LoRA 参数
- `--lora_rank 64`: LoRA 秩（推荐 32-128）
- `--lora_alpha 64`: LoRA alpha（通常等于 rank）

### 训练参数
- `--train_batch_size 12`: 每个 GPU 的批大小
- `--gradient_accumulation_steps 4`: 梯度累积步数
- `--learning_rate 1e-4`: 学习率
- `--max_train_steps 50000`: 总训练步数

### LCM 参数
- `--num_inference_steps 8`: 目标推理步数（4/6/8/16）
- `--cfg_scale 7.5`: CFG 强度（默认 7.5）

### 数据集参数
- `--use_mock_dataset`: 使用模拟数据集（测试用）
- `--dataset_size 1000`: 模拟数据集大小

## ⚡ 性能优化

### 显存不足？

降低批大小：
```bash
--train_batch_size 6 --gradient_accumulation_steps 8
```

或：
```bash
--train_batch_size 3 --gradient_accumulation_steps 16
```

### 训练太慢？

增加批大小（如果有足够显存）：
```bash
--train_batch_size 18 --gradient_accumulation_steps 2
```

### 质量不够好？

增加推理步数：
```bash
--num_inference_steps 16  # 从 8 改为 16
```

## 📈 监控训练

### TensorBoard

在另一个终端运行：
```bash
tensorboard --logdir student_model/checkpoints/logs
```

然后访问 http://localhost:6006

### 检查点

检查点保存在：
```
student_model/checkpoints/
├── checkpoint-5000/
├── checkpoint-10000/
├── checkpoint-15000/
└── checkpoint-final/
```

每个检查点包含：
- `adapter_config.json`: LoRA 配置
- `adapter_model.bin`: LoRA 权重（~50-100MB）

## 🔧 常见问题

### Q: 如何使用真实数据集？

A: 修改 `student_model/dataset_anytext.py`，替换模拟数据生成为你的数据加载逻辑，然后移除 `--use_mock_dataset` 参数。

### Q: 如何调整推理速度？

A: 修改 `--num_inference_steps`：
- 4 步：最快，质量略降
- 8 步：平衡（推荐）
- 16 步：最慢，质量最好

### Q: 如何恢复训练？

A: 训练会自动保存检查点。如需从中断处继续，可以修改 `train_lcm_anytext.py` 添加 `--resume_from_checkpoint` 参数。

### Q: 显存不足怎么办？

A:
1. 降低 `--train_batch_size`
2. 增加 `--gradient_accumulation_steps`
3. 降低 `--lora_rank`（从 64 降到 32）
4. 使用 `--mixed_precision fp16`（已默认）

## 📚 更多信息

详细文档请参考：
- [README.md](README.md) - 完整文档
- [SUMMARY.md](SUMMARY.md) - 实现总结
- [ENVIRONMENT_FIX.md](ENVIRONMENT_FIX.md) - 环境问题解决
- [PATH_FIX.md](PATH_FIX.md) - 路径修复说明

---

**准备好了？开始训练吧！** 🚀
