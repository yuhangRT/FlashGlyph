# OCR 可读性评测说明（PARSeq / TrOCR）

本目录提供与论文口径一致的外部 OCR 评测脚本：

- `eval/eval_ocr.py`：主入口（读 GT json + 读生成图 + 计算 CharAcc/WordAcc/CER/WER）
- `eval/eval_parseq.py`：PARSeq 推理封装
- `eval/eval_trocr.py`：TrOCR 推理封装

## 1. 依赖

最小依赖（建议单独环境）：

```bash
pip install torch torchvision transformers accelerate pillow opencv-python
```

可选依赖：

- `evaluate`（若你后续接入 HuggingFace evaluate）
- `python-Levenshtein`（可选加速编辑距离；当前脚本内置了纯 Python 实现）

## 2. 输入数据与命名规则

GT 标注 JSON 支持 AnyText-benchmark / AnyWord 风格，核心字段：

- `data_list[*].img_name`
- `data_list[*].annotations[*].polygon`
- `data_list[*].annotations[*].text`

生成图命名规则：

```text
<img_key>_<k>.jpg
```

- `<img_key>` = `img_name` 去掉扩展名
- `<k>` = 第 k 个采样（`0..num_samples-1`）

脚本会自动回退尝试 `.png`。

## 3. 命令示例

PARSeq：

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/flashglyph_main_cn \
  --input_json /path/to/wukong_word/test1k.json \
  --backend parseq \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/flashglyph_cn_parseq.json
```

TrOCR：

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/flashglyph_main_cn \
  --input_json /path/to/wukong_word/test1k.json \
  --backend trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/flashglyph_cn_trocr.json
```

PARSeq + TrOCR 平均（推荐论文口径）：

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/flashglyph_main_cn \
  --input_json /path/to/wukong_word/test1k.json \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/flashglyph_cn_parseq_trocr_avg.json
```

## 4. 输出结果

推荐统一保存到：

`student_model_v3/experiments/results/*.json`

输出 JSON 包含：

- 指标：`word_acc` / `char_acc` / `cer` / `wer`
- 样本计数：`input_items` / `images_found` / `lines`
- 模型信息：`model_name` / `backend`
- 追溯信息：`timestamp_utc` / `img_dir` / `input_json` / `num_samples_per_input`

## 5. 训练/测试识别器解耦

- 训练阶段 OCR 监督：`cldm/recognizer.py`（PP-OCRv3，冻结）
- 测试阶段评测 OCR：`eval/eval_parseq.py` + `eval/eval_trocr.py`

请勿使用训练识别器直接作为测试指标，以避免同构偏置。
