# AnyText2 原始结果 vs student_model_v3/predicted 的区别

## 📊 核心区别

### 1. **`student_model_v3/experiments/predicted/` 中的结果**

**性质**: **预测值/预估值** (Predicted)，不是实际测量值

**来源**: 从论文草稿 (`flashglyph_paper.md`) 中提取的数字，用于**预填充表格**

**内容**:
```
predicted_summary.json 明确说明:
"source": "Predicted from flashglyph_paper.md (current in-repo tables)"
"note": "Not measured from real artifacts yet; use for TABLE_SOURCES pre-fill only."
```

**包含的表格**:
- `table1a_cn_predicted.csv` - 中文Table 1a的预测数据
- `table1b_en_predicted.csv` - 英文Table 1b的预测数据
- `table2_cn_predicted.csv` - 中文Table 2的预测数据
- `table2_en_predicted.csv` - 英文Table 2的预测数据
- `table4_cn_predicted.csv` - 中文Table 4的预测数据

**每行包含的指标**:
- `method`: 方法名称 (AnyText2 Teacher, DDIM, DPM-Solver, LCM, FlashGlyph等)
- `steps`: 采样步数
- `latency_ms`: 延迟（毫秒）
- `speedup`: 加速比
- `char_acc`: 字符准确率
- `word_acc`: 词准确率
- `cer`: 字符错误率
- `wer`: 词错误率
- `fid`: FID分数（图像质量）
- `lpips`: LPIPS感知相似度
- `config`: 使用的配置文件
- `image_dir`: 生成图像的目录
- `ocr_json`: OCR评估结果文件

---

### 2. **AnyText2 原始评估脚本的结果**

**性质**: **实际测量值** (Measured)，通过运行真实评估得到

**来源**: 运行 `eval/` 目录下的评估脚本

**评估流程**:

#### 步骤1: 生成图像
```bash
# 使用 AnyText2 原始模型生成图像
bash eval/gen_imgs_anytext2.sh
# 或
python eval/anytext2_singleGPU.py --input_json <test.json> --output_dir <output_dir>
```

这会在输出目录生成实际图像文件，例如:
```
eval/gen_imgs_test/
├── img_000000_0.png
├── img_000000_1.png
├── img_000001_0.png
└── ...
```

#### 步骤2: OCR 评估
```bash
# 使用 PARSeq + TrOCR 评估 OCR 准确率
python eval/eval_ocr.py \
  --img_dir <生成的图像目录> \
  --input_json <test1k.json> \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json <输出JSON>
```

#### 步骤3: 其他评估
```bash
# CLIP 分数
bash eval/eval_clip.sh

# FID 分数
bash eval/eval_fid.sh
```

**实际输出**: JSON 文件，包含每个样本的详细评估结果

---

## 🔍 具体区别对比

| 维度 | predicted/ 中的结果 | AnyText2 原始评估结果 |
|------|---------------------|----------------------|
| **数据来源** | 从论文表格手动提取/预估 | 实际运行模型测量 |
| **可靠性** | ⚠️ 仅供参考，非实测 | ✅ 真实可靠 |
| **用途** | 论文草稿预填充 | 论文最终数据/实验报告 |
| **更新方式** | 手动编辑 CSV | 运行评估脚本自动生成 |
| **包含图像** | ❌ 只有路径引用 | ✅ 有实际图像文件 |
| **包含 OCR 原文** | ❌ 只有汇总指标 | ✅ 有每个样本的详细OCR结果 |
| **时间戳** | ❌ 无 | ✅ 有评估时间 |

---

## ❓ 能否覆盖？

### **不能直接覆盖！原因如下:**

#### 1. **数据性质不同**

```
predicted/ 中的数据:
  方法名 → 预估值（从论文复制）
  
AnyText2 评估的数据:
  方法名 → 实际测量值（从模型运行）
```

#### 2. **格式不同**

**predicted CSV 格式**:
```csv
method,steps,latency_ms,char_acc,fid,config,image_dir,ocr_json
AnyText2 (Teacher),50,10440,94.1,11.8,models_yaml/anytext2_sd15.yaml,...
```

**AnyText2 评估 JSON 格式** (示例):
```json
{
  "img_000000_0.png": {
    "gt_text": "你好世界",
    "pred_text_parseq": "你好世界",
    "pred_text_trocr": "你好世界",
    "char_acc": 100.0,
    "cer": 0.0
  },
  "img_000000_1.png": { ... },
  ...
  "summary": {
    "avg_char_acc": 94.1,
    "avg_cer": 5.9
  }
}
```

#### 3. **目录结构不同**

```
student_model_v3/experiments/
├── predicted/              ← 预估值（CSV表格）
│   ├── table1a_cn_predicted.csv
│   ├── predicted_summary.json
│   └── ...
├── generated/              ← 实际生成的图像（需要运行生成）
│   ├── teacher50_cn/      ← AnyText2原始模型生成
│   ├── flashglyph4_cn/    ← FlashGlyph学生模型生成
│   └── ...
└── results/                ← OCR评估结果（需要运行评估）
    ├── table1a_cn_teacher50_parseq_trocr.json
    └── ...
```

---

## ✅ 正确的替换流程

如果你想用 AnyText2 原始模型的实测结果替换 predicted 中的预估值:

### 步骤1: 用 AnyText2 原始模型生成图像

```bash
# 准备测试数据集（例如 test1k.json）
# 运行生成脚本
python eval/anytext2_singleGPU.py \
  --input_json /path/to/test1k.json \
  --output_dir student_model_v3/experiments/generated/anytext2_original_cn \
  --ckpt_path ./models/iic/cv_anytext2/anytext_v2.0.ckpt \
  --config_yaml ./models_yaml/anytext2_sd15.yaml \
  --num_samples 4 \
  --ddim_steps 50  # 或其他步数
```

### 步骤2: 运行 OCR 评估

```bash
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/anytext2_original_cn \
  --input_json /path/to/test1k.json \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/anytext2_original_parseq_trocr.json
```

### 步骤3: 从评估结果提取汇总指标

```python
import json

# 读取评估结果
with open('student_model_v3/experiments/results/anytext2_original_parseq_trocr.json') as f:
    results = json.load(f)

# 提取摘要
summary = results.get('summary', {})
char_acc = summary.get('avg_char_acc', 0)
cer = summary.get('avg_cer', 0)
# ... 其他指标
```

### 步骤4: 更新 predicted CSV

手动或脚本更新 `table1a_cn_predicted.csv`:

```csv
method,steps,latency_ms,char_acc,fid,...
AnyText2 (Original),50,<实测延迟>,<实测准确率>,<实测FID>,...
```

### 步骤5: 更新 predicted_summary.json

```json
{
  "source": "Measured from actual AnyText2 evaluation",
  "note": "Updated from real artifacts",
  "measured": true
}
```

---

## 🎯 建议做法

### 方案A: 保留 predicted，新增 measured 目录

```
student_model_v3/experiments/
├── predicted/           ← 保留论文预估值
├── measured/            ← ← 新增实测值
│   ├── table1a_cn_measured.csv
│   └── ...
└── comparison/          ← 对比分析
    └── predicted_vs_measured.md
```

**优点**: 
- 保留原始预测用于对比
- 可以分析预测与实际的差距

### 方案B: 直接更新 predicted

```bash
# 1. 备份原始 predicted
cp -r student_model_v3/experiments/predicted student_model_v3/experiments/predicted_backup

# 2. 运行 AnyText2 评估（步骤1-3）

# 3. 用实测数据更新 CSV

# 4. 更新 predicted_summary.json
{
  "source": "Measured from AnyText2 evaluation scripts",
  "note": "Updated with real measurements",
  "measured": true,
  "update_date": "2026-04-07"
}
```

---

## 📋 完整实验清单

要获得 AnyText2 原始模型的完整实测数据，需要运行:

### 1. 生成图像 (不同配置)

```bash
# AnyText2 Teacher - 50 steps (中文)
python eval/anytext2_singleGPU.py \
  --input_json <中文test.json> \
  --output_dir student_model_v3/experiments/generated/teacher50_cn \
  --ddim_steps 50

# AnyText2 Teacher - 20 steps (中文)
python eval/anytext2_singleGPU.py \
  --input_json <中文test.json> \
  --output_dir student_model_v3/experiments/generated/teacher20_cn \
  --ddim_steps 20

# 英文同理
```

### 2. OCR 评估

```bash
# 对每个生成的图像目录运行
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/teacher50_cn \
  --input_json <中文test.json> \
  --backend parseq+trocr \
  --out_json student_model_v3/experiments/results/teacher50_cn_parseq_trocr.json
```

### 3. FID/CLIP 评估

```bash
bash eval/eval_fid.sh
bash eval/eval_clip.sh
```

### 4. 汇总结果

从所有 JSON 结果文件中提取指标，填入 CSV 表格。

---

## ⚠️ 注意事项

1. **测试数据集必须一致**
   - predicted 中引用的数据来自 `test1k.json` 或类似测试集
   - 实测时必须使用相同的数据集才能对比

2. **随机种子**
   - predicted 中的数字可能基于特定种子
   - 实测时应该多次运行取平均

3. **环境差异**
   - 延迟 (`latency_ms`) 高度依赖硬件
   - 在不同GPU上运行会有差异

4. **模型版本**
   - 确保使用相同版本的 AnyText2 模型
   - 检查 checkpoint 文件是否一致

---

## 🤔 总结

| 问题 | 答案 |
|------|------|
| predicted 中的数据是什么？ | 从论文表格预填的预估值，非实测 |
| AnyText2 原始评估能生成什么？ | 实际运行模型得到的真实测量值 |
| 能直接覆盖吗？ | ❌ 不能，格式和来源都不同 |
| 应该如何更新？ | 运行评估脚本 → 提取指标 → 手动/脚本更新CSV |
| 建议做法？ | 保留 predicted 作为对比基准，新增 measured 目录存放实测数据 |

---

## 🚀 快速开始实测

```bash
# 1. 确保模型已下载
python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"

# 2. 准备测试数据集
# (需要 test1k.json 或类似标注文件)

# 3. 生成图像
python eval/anytext2_singleGPU.py \
  --input_json <你的test.json> \
  --output_dir student_model_v3/experiments/generated/anytext2实测 \
  --ddim_steps 50

# 4. OCR评估
python eval/eval_ocr.py \
  --img_dir student_model_v3/experiments/generated/anytext2实测 \
  --input_json <你的test.json> \
  --backend parseq+trocr \
  --out_json student_model_v3/experiments/results/anytext2实测.json

# 5. 从JSON提取指标更新CSV
# (需要写脚本或手动更新)
```
