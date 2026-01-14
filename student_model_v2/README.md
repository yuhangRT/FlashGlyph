# student_model_v2（中文版）

面向新手的 AnyText2 LCM-LoRA 蒸馏训练与推理示例。目标是 **先跑通全链路**，后续再逐步优化效果与速度。

## 这是什么

`student_model_v2` 是一个最小可运行版本，包含：
- **训练脚本**：用 Teacher 引导蒸馏 Student（LoRA 微调）。
- **数据管线**：复用 AnyText2 的输入格式（图像、glyph、位置、颜色等）。
- **推理脚本**：从验证集选一个样本进行重建。

它不追求“最佳性能”，而是追求：
1) 代码可读  
2) 依赖清晰  
3) 一键跑通训练与推理

## 关键设计（面向初学者）

1) **只训练 UNet + ControlNet（LoRA）**  
不训练文本编码器与 OCR/Embedding 模块，避免语义漂移与图结构风险。

2) **Teacher 用 CFG，Student 不用 CFG**  
Teacher 同时跑 Cond + Uncond，再混合；Student 只跑 Cond。  
这样 Student 学到的结果已经“内化了引导”，推理时不需要 CFG。

3) **目标是 x0 回归（一次 Teacher forward）**  
Teacher 只跑一次，得到噪声预测，再按 DDIM 公式换算为 x0 作为监督目标。  
显存更省，更适合 AnyText2 这种大模型。

4) **时间步只在固定离散点采样**  
这符合 LCM 训练逻辑：让 Student 精准学习少数关键步骤。

## 目录结构

```
student_model_v2/
  dataset_anytext_v2.py      # 数据集与 collate（含 masked_img 逻辑）
  lcm_utils_v2.py            # 时间步采样与 x0 计算
  train_lcm_anytext_v2.py    # 训练脚本
  train_lcm_anytext_v2_2.py  # 训练脚本（优化版）
  infer_lcm_anytext_v2.py    # 推理脚本（从数据集中取样）
  README.md                  # 使用说明
```

## 环境准备

建议使用你现有的 AnyText2 环境（含 torch、diffusers、peft、accelerate）。  
确保以下文件可用：
- `models/anytext_v2.0.ckpt`
- `models_yaml/anytext2_sd15.yaml`
- `demodataset/annotations/demo_data.json`

## 训练（使用 demodataset）

```bash
python student_model_v2/train_lcm_anytext_v2.py \
  --config models_yaml/anytext2_sd15.yaml \
  --teacher_ckpt models/anytext_v2.0.ckpt \
  --dataset_json demodataset/annotations/demo_data.json \
  --output_dir student_model_v2/checkpoints \
  --num_inference_steps 50
```

## 训练（优化版，速度优先）

优化点：
1) 合并 VAE 编码：一次 encode 同时生成 `img` 与 `masked_img` 的 latents  
2) 缓存 Uncond：复用同形状 batch 的 Uncond embedding  
3) Teacher 使用 `torch.inference_mode()` 降低开销  

```bash
python student_model_v2/train_lcm_anytext_v2_2.py \
  --config models_yaml/anytext2_sd15.yaml \
  --teacher_ckpt models/anytext_v2.0.ckpt \
  --dataset_json demodataset/annotations/demo_data.json \
  --output_dir student_model_v2/checkpoints \
  --num_inference_steps 50
```

说明：`launch_from_yaml.py` 会读取 YAML 中的 `train.use_optimized`，为 `true` 时调用 `train_lcm_anytext_v2_2.py`。

## 训练（从 YAML 配置启动）

使用 `student_model_v2/train_config_template.yaml` 作为模板，在 YAML 里修改参数：

```bash
accelerate launch --multi_gpu student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template.yaml
```

### 断点续训

保存 checkpoint 时会同时写入 `training_state.pt`（包含 optimizer 与步数）。  
恢复训练时设置：

```yaml
train:
  resume_path: student_model_v2/checkpoints/checkpoint-2000
  resume_optimizer: true
```

说明：会恢复 LoRA 权重与 optimizer 状态，`global_step` 会延续；数据加载从 epoch 起点重新开始。

### Epoch 驱动与可视化日志

- `train.max_epochs > 0` 时按 **epoch** 训练（优先级高于 `max_train_steps`）。
- `train.save_epochs > 0` 时按 **epoch** 保存 checkpoint。
- `train.log_image_steps > 0` 会在 `output_dir/image_log/train` 输出预览图  
  （每行依次为：原图 / masked_img / hint / 预测重建）。

## 内存保护（避免系统 OOM）

如果担心系统内存被打爆，可用 `oom_guard.py` 在内存接近阈值时自动结束训练：

```bash
python student_model_v2/oom_guard.py --min-available-gb 8 -- \
  accelerate launch --multi_gpu student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template.yaml
```

可用 `--print_args` 查看解析后的 CLI 参数：

```bash
python student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template.yaml \
  --print_args
```

默认会自动划分 train/val/test：  
- 训练：0.9  
- 验证：0.05  
- 测试：0.05  

你可以这样自定义：

```bash
python student_model_v2/train_lcm_anytext_v2.py \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 快速跑通（使用 Mock 数据）

```bash
python student_model_v2/train_lcm_anytext_v2.py --use_mock_dataset
```

## 推理（从数据集中取一个样本）

```bash
python student_model_v2/infer_lcm_anytext_v2.py \
  --config models_yaml/anytext2_sd15.yaml \
  --teacher_ckpt models/anytext_v2.0.ckpt \
  --lora_path student_model_v2/checkpoints/checkpoint-final \
  --dataset_json demodataset/annotations/demo_data.json \
  --num_inference_steps 4 \
  --sample_index 0 \
  --output_path student_model_v2/sample.png
```

**说明**：  
推理阶段默认从数据集中拿一个样本作为输入（这适合 AnyText2 的复杂条件输入）。  
无需 free-form prompt，降低工程复杂度。

## 推理（完整功能：Gradio UI）

此版本保留原 AnyText2 的全部推理功能（文字生成、文字编辑、位置控制、字体与颜色等）。  
使用方式与原 `demo.py` 完全一致，只是替换为 Student LoRA。

```bash
python student_model_v2/demo_v2.py \
  --student_lora_path student_model_v2/checkpoints/checkpoint-final \
  --model_path models/anytext_v2.0.ckpt
```

如需关闭 LCM、回退到原始 DDIM 采样，可加参数：

```bash
python student_model_v2/demo_v2.py --use_ddim_sampler
```

**LCM 模式默认行为**：
- 自动关闭 CFG（只跑 Cond）
- 步数限制在 4～8 之间（避免误用）
- 若需要原始 CFG/更大步数，请使用 `--use_ddim_sampler`

## masked_img 说明（与 t3_dataset.py 一致）

当前 `masked_img` 逻辑与 `t3_dataset.py` 保持一致：
- 由 `mask_img_prob` 控制是否进入编辑模式  
- 编辑模式下：`masked_img = (img - mask * 10).clip(-1, 1)`  
- 非编辑模式下：返回全 -1 的图片（可视为“纯遮罩”）

训练与推理时会 **用 VAE 编码 masked_img 得到 masked_x**，保证分布一致。

## 常见问题

**Q1: 为什么我的显存占用很高？**  
AnyText2 本身很大，Teacher+Student 同时跑会吃显存。可以尝试：
- `--train_batch_size 1`
- `--gradient_accumulation_steps 4`
- `--mixed_precision fp16`

**Q2: LoRA 注入报错怎么办？**  
本版本要求 `zero_convs` 也必须 LoRA，若报错请升级 `peft` 和 `transformers`。

**Q3: 想用真实数据集怎么办？**  
直接替换 `--dataset_json` 为你的 JSON 即可，格式需与 AnyText2 数据集一致。

---

如果你希望我继续：
- 增加更严格的评估（比如 OCR 识别率）
- 做自由 prompt 的推理 pipeline
- 引入真正多步一致性蒸馏  
告诉我即可。
