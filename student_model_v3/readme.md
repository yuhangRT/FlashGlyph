## FlashGlyph v3（方案 B 主线）

FlashGlyph 主方法是**三重约束**：

- Alignment：attention alignment distillation
- Semantics：OCR-CTC supervision
- Topology：soft-skeleton + clDice

默认主配置是 `student_model_v3/configs/lcm_v3.yaml`：

- 开启：`loss_attn_weight`、`loss_ocr_weight`、`loss_cldice_weight`
- 关闭：`loss_ffl_weight=0`、`loss_grad_weight=0`

`student_model_v3/configs/lcm_v3_gl.yaml` 仅作为 **optional polish / ablation**（FFL + Grad 抛光项），不是主方法默认配置。

## 训练

单卡（默认主线配置）：

```bash
python3 student_model_v3/launch_single_gpu.py \
  --config student_model_v3/configs/lcm_v3.yaml \
  --gpu 0
```

内存保护：

```bash
python3 student_model_v3/launch_single_gpu.py \
  --config student_model_v3/configs/lcm_v3.yaml \
  --gpu 0 \
  --min-available-gb 4
```

续训：在 yaml 的 `model` 段添加 `resume_path`。

## 推理

主线模型（推荐）：

```bash
python3 student_model_v3/infer_lcm_anytext_v3.py \
  --student_lora_path student_model_v3/checkpoints/<main_run>/checkpoint-final \
  --output student_model_v3/preview_main.png
```

可选 GL 抛光模型：

```bash
python3 student_model_v3/infer_lcm_anytext_v3.py \
  --student_lora_path student_model_v3/checkpoints/<gl_run>/checkpoint-final \
  --output student_model_v3/preview_gl.png
```

批量生成评测集（供 OCR/FID 表格评测）：

```bash
python3 student_model_v3/generate_eval_images.py \
  --student_lora_path student_model_v3/checkpoints/<main_run>/checkpoint-final \
  --input_json /path/to/test1k.json \
  --output_dir student_model_v3/experiments/generated/flashglyph_main \
  --num_samples_per_input 4
```

## 训练/测试识别器解耦

- 训练 OCR 监督：`cldm/recognizer.py`（PP-OCRv3 CTC，冻结）
- 测试 OCR 评测：`eval/eval_parseq.py` + `eval/eval_trocr.py`

## TensorBoard

查看所有日志：

```bash
tensorboard --logdir student_model_v3/checkpoints/logs --port 6006 --bind_all
```

查看特定 run：

```bash
tensorboard --logdir student_model_v3/checkpoints/train_20260130_072325/logs --port 6006 --bind_all
```

查看所有历史训练：

```bash
tensorboard --logdir student_model_v3/checkpoints --port 6006 --bind_all
```
