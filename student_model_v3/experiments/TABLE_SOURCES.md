# FlashGlyph 表格来源映射（可追溯版）

目标：论文中每一行数字都能在仓库定位到：

- 配置文件
- 训练/推理命令
- 生成图目录
- 评测结果 JSON/日志

主方法口径（方案 B）：

- 主配置：`student_model_v3/configs/lcm_v3.yaml`（Attn + OCR-CTC + clDice）
- 可选抛光：`student_model_v3/configs/lcm_v3_gl.yaml`（FFL + Grad）

## 1. 标准路径约定

- 训练 run 目录：`student_model_v3/checkpoints/train_YYYYMMDD_HHMMSS_*`
- 生成图目录：`student_model_v3/experiments/generated/<method_name>/`
- OCR 结果目录：`student_model_v3/experiments/results/<method>_<ocr>.json`
- 延迟结果目录：`student_model_v3/experiments/results/latency_<method>_<mode>.json`

生成图命名必须为：

`<img_key>_<k>.jpg`

其中 `<img_key>` 来自 benchmark json 的 `img_name` 去扩展名，`k=0..K-1`。

## 2. 基准 JSON

- 中文：`/path/to/wukong_word/test1k.json`
- 英文：`/path/to/laion_word/test1k.json`

## 3. 批量生成命令（评测集）

示例（FlashGlyph 主方法）：

```bash
python student_model_v3/generate_eval_images.py \
  --student_lora_path student_model_v3/checkpoints/<flashglyph_run>/checkpoint-final \
  --input_json /path/to/wukong_word/test1k.json \
  --output_dir student_model_v3/experiments/generated/flashglyph4_cn \
  --num_inference_steps 4 \
  --num_samples_per_input 4 \
  --skip_existing
```

GL 抛光版本仅替换 LoRA 路径（或配置对应 run）：

```bash
python student_model_v3/generate_eval_images.py \
  --student_lora_path student_model_v3/checkpoints/<flashglyph_gl_run>/checkpoint-final \
  --input_json /path/to/wukong_word/test1k.json \
  --output_dir student_model_v3/experiments/generated/flashglyph4_gl_cn \
  --num_inference_steps 4 \
  --num_samples_per_input 4
```

## 4. OCR 评测命令（PARSeq / TrOCR）

详见：`eval/README_OCR_EVAL.md`

示例（双识别器平均）：

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/flashglyph4_cn \
  --input_json /path/to/wukong_word/test1k.json \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/table1a_cn_flashglyph_parseq_trocr.json
```

## 5. FID / LPIPS（可追溯记录方式）

FID（示例）：

```bash
python -m pytorch_fid \
  /path/to/gt_cn_images \
  student_model_v3/experiments/generated/flashglyph4_cn
```

将输出写入：

- `student_model_v3/experiments/results/table1a_cn_flashglyph_fid.json`

LPIPS 如使用外部脚本，至少记录：

- 脚本命令
- GT 路径
- 生成图路径
- 输出 JSON 路径

## 6. 延迟口径（必须注明）

需区分两种口径：

- UNet-only：不含 VAE encode/decode
- End-to-end：含 VAE encode/decode + 前后处理

建议输出到：

- `student_model_v3/experiments/results/latency_<method>_unet_only.json`
- `student_model_v3/experiments/results/latency_<method>_end2end.json`

如果暂未脚本化，至少保存可复现命令与原始日志。

## 7. 论文表格逐行映射

### 表 1a（中文主实验）

| Paper row | Config | Train run dir | Generated images | OCR json | FID/LPIPS | Inference command |
|---|---|---|---|---|---|---|
| AnyText2 (Teacher, 50-step) | `models_yaml/anytext2_sd15.yaml` | N/A | `student_model_v3/experiments/generated/teacher50_cn` | `student_model_v3/experiments/results/table1a_cn_teacher_parseq_trocr.json` | `..._fid.json`, `..._lpips.json` | `eval/gen_imgs_anytext2.sh` 或等价脚本 |
| DDIM-4step | `models_yaml/anytext2_sd15.yaml` | N/A | `student_model_v3/experiments/generated/ddim4_cn` | `student_model_v3/experiments/results/table1a_cn_ddim4_parseq_trocr.json` | `...` | 记录具体 sampler 命令 |
| LCM-baseline (mask) | `student_model_v3/configs/ablation_A0.yaml` | `student_model_v3/checkpoints/<A0_run>` | `student_model_v3/experiments/generated/lcm_a0_cn` | `student_model_v3/experiments/results/table1a_cn_a0_parseq_trocr.json` | `...` | `generate_eval_images.py` |
| FlashGlyph (ours, 4-step) | `student_model_v3/configs/lcm_v3.yaml` | `student_model_v3/checkpoints/<flashglyph_run>` | `student_model_v3/experiments/generated/flashglyph4_cn` | `student_model_v3/experiments/results/table1a_cn_flashglyph_parseq_trocr.json` | `...` | `generate_eval_images.py` |

### 表 1b（英文主实验）

同表 1a 结构，输出路径改为 `*_en_*`。

### 表 2（消融）

| Ablation row | Config | Train run dir | Generated images | OCR json |
|---|---|---|---|---|
| A0 | `student_model_v3/configs/ablation_A0.yaml` | `student_model_v3/checkpoints/<A0_run>` | `student_model_v3/experiments/generated/ablation_a0_cn` | `student_model_v3/experiments/results/table2_cn_a0_parseq_trocr.json` |
| A1 | `student_model_v3/configs/ablation_A1.yaml` | `student_model_v3/checkpoints/<A1_run>` | `student_model_v3/experiments/generated/ablation_a1_cn` | `student_model_v3/experiments/results/table2_cn_a1_parseq_trocr.json` |
| A2 | `student_model_v3/configs/ablation_A2.yaml` | `student_model_v3/checkpoints/<A2_run>` | `student_model_v3/experiments/generated/ablation_a2_cn` | `student_model_v3/experiments/results/table2_cn_a2_parseq_trocr.json` |
| A3 | `student_model_v3/configs/lcm_v3.yaml`（去掉 optional sharpness） | `student_model_v3/checkpoints/<A3_run>` | `student_model_v3/experiments/generated/ablation_a3_cn` | `student_model_v3/experiments/results/table2_cn_a3_parseq_trocr.json` |
| A4 (optional polish) | `student_model_v3/configs/lcm_v3_gl.yaml` | `student_model_v3/checkpoints/<A4_run>` | `student_model_v3/experiments/generated/ablation_a4_gl_cn` | `student_model_v3/experiments/results/table2_cn_a4_parseq_trocr.json` |

### 表 4（速度-质量）

| Row | Step setting | Generated images | OCR json | Latency json | Command note |
|---|---|---|---|---|---|
| 1-step | `--num_inference_steps 1` | `.../flashglyph1_cn` | `.../table4_cn_step1_parseq_trocr.json` | `.../latency_flashglyph_step1_*.json` | 同一硬件同一 batch=1 |
| 2-step | `--num_inference_steps 2` | `.../flashglyph2_cn` | `.../table4_cn_step2_parseq_trocr.json` | `.../latency_flashglyph_step2_*.json` | 同上 |
| 4-step | `--num_inference_steps 4` | `.../flashglyph4_cn` | `.../table4_cn_step4_parseq_trocr.json` | `.../latency_flashglyph_step4_*.json` | 同上 |

## 8. 审稿前核对清单

- 每个表格行都能定位到：`config + command + generated dir + result json`
- OCR JSON 包含：`timestamp_utc + model_name + sample counts`
- 延迟数字注明是 `UNet-only` 还是 `end-to-end`
- 主线方法默认使用 `lcm_v3.yaml`，`lcm_v3_gl.yaml` 仅 optional

## 9. 预测填充数据（基于当前代码与论文表格）

说明：

- 当前仓库尚未发现真实产物目录（`student_model_v3/checkpoints` 不存在），以下为“预测填充版”。
- 指标数值来自 `student_model_v3/paper/flashglyph_paper.md` 现有表格。
- 方法行与路径命名按本仓库脚本能力（`generate_eval_images.py` + `eval_ocr.py`）统一。

### 9.1 表 1a（CN）预测映射

| Method | Predicted image dir | Predicted OCR json | Predicted FID json | Predicted latency json (UNet-only) | Predicted config/run |
|---|---|---|---|---|---|
| AnyText2 (Teacher, 50-step) | `student_model_v3/experiments/generated/teacher50_cn` | `student_model_v3/experiments/results/table1a_cn_teacher50_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_teacher50_fid.json` | `student_model_v3/experiments/results/latency_teacher50_unet_only.json` | `models_yaml/anytext2_sd15.yaml` + `models/anytext_v2.0.ckpt` |
| DDIM-4step | `student_model_v3/experiments/generated/ddim4_cn` | `student_model_v3/experiments/results/table1a_cn_ddim4_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_ddim4_fid.json` | `student_model_v3/experiments/results/latency_ddim4_unet_only.json` | Teacher ckpt + DDIM(4) |
| DDIM-10step | `student_model_v3/experiments/generated/ddim10_cn` | `student_model_v3/experiments/results/table1a_cn_ddim10_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_ddim10_fid.json` | `student_model_v3/experiments/results/latency_ddim10_unet_only.json` | Teacher ckpt + DDIM(10) |
| DPM-Solver-10 | `student_model_v3/experiments/generated/dpmsolver10_cn` | `student_model_v3/experiments/results/table1a_cn_dpmsolver10_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_dpmsolver10_fid.json` | `student_model_v3/experiments/results/latency_dpmsolver10_unet_only.json` | Teacher ckpt + DPM-Solver(10) |
| DPM-Solver-15 | `student_model_v3/experiments/generated/dpmsolver15_cn` | `student_model_v3/experiments/results/table1a_cn_dpmsolver15_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_dpmsolver15_fid.json` | `student_model_v3/experiments/results/latency_dpmsolver15_unet_only.json` | Teacher ckpt + DPM-Solver(15) |
| UniPC-10 | `student_model_v3/experiments/generated/unipc10_cn` | `student_model_v3/experiments/results/table1a_cn_unipc10_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_unipc10_fid.json` | `student_model_v3/experiments/results/latency_unipc10_unet_only.json` | Teacher ckpt + UniPC(10) |
| LCM-baseline (no mask) | `student_model_v3/experiments/generated/lcm_nomask4_cn` | `student_model_v3/experiments/results/table1a_cn_lcm_nomask4_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_lcm_nomask4_fid.json` | `student_model_v3/experiments/results/latency_lcm_nomask4_unet_only.json` | 需补独立 no-mask yaml（当前未见） |
| LCM-baseline (mask) | `student_model_v3/experiments/generated/lcm_mask4_cn` | `student_model_v3/experiments/results/table1a_cn_lcm_mask4_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_lcm_mask4_fid.json` | `student_model_v3/experiments/results/latency_lcm_mask4_unet_only.json` | `student_model_v3/configs/ablation_A0.yaml` |
| FlashGlyph (ours, 4-step) | `student_model_v3/experiments/generated/flashglyph4_cn` | `student_model_v3/experiments/results/table1a_cn_flashglyph4_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_flashglyph4_fid.json` | `student_model_v3/experiments/results/latency_flashglyph4_unet_only.json` | `student_model_v3/configs/lcm_v3.yaml` |
| FlashGlyph (2-step) | `student_model_v3/experiments/generated/flashglyph2_cn` | `student_model_v3/experiments/results/table1a_cn_flashglyph2_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_flashglyph2_fid.json` | `student_model_v3/experiments/results/latency_flashglyph2_unet_only.json` | `lcm_v3.yaml` + `--num_inference_steps 2` |
| FlashGlyph (1-step) | `student_model_v3/experiments/generated/flashglyph1_cn` | `student_model_v3/experiments/results/table1a_cn_flashglyph1_parseq_trocr.json` | `student_model_v3/experiments/results/table1a_cn_flashglyph1_fid.json` | `student_model_v3/experiments/results/latency_flashglyph1_unet_only.json` | `lcm_v3.yaml` + `--num_inference_steps 1` |

### 9.2 表 1b（EN）预测映射

命名规则同上，后缀统一替换为 `_en`：

- 生成图目录：`student_model_v3/experiments/generated/<method>_en`
- OCR json：`student_model_v3/experiments/results/table1b_en_<method>_parseq_trocr.json`
- FID json：`student_model_v3/experiments/results/table1b_en_<method>_fid.json`
- 延迟 json：`student_model_v3/experiments/results/latency_<method>_unet_only.json`

### 9.3 结构化预测文件

已通过脚本生成（见 `student_model_v3/experiments/predicted/`）：

- `table1a_cn_predicted.csv`
- `table1b_en_predicted.csv`
- `table2_cn_predicted.csv`
- `table2_en_predicted.csv`
- `table4_cn_predicted.csv`
- `predicted_summary.json`
