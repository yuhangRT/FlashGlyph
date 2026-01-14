# 路径修复说明

## ✅ 已修复的问题

### 问题描述
原始代码在使用相对路径时无法正确解析文件位置，导致 `FileNotFoundError`。

### 修复方案
在 `inspect_modules.py` 和 `train_lcm_anytext.py` 中添加了智能路径解析逻辑：

```python
# 获取脚本目录和父目录
script_dir = Path(__file__).parent.resolve()
parent_dir = script_dir.parent

# 如果路径不是绝对路径，则相对于父目录解析
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = (parent_dir / config_path).resolve()
```

### 修复的文件
1. ✅ `student_model/inspect_modules.py` - 修复了配置和检查点路径
2. ✅ `student_model/train_lcm_anytext.py` - 修复了所有输入路径
3. ✅ `student_model/test_paths.py` - 新增路径测试工具

## 使用方法

### 从任意目录运行

现在可以从任意目录运行脚本，路径会自动正确解析：

```bash
# 从项目根目录运行
cd /home/zyh/AnyText2
python ./student_model/inspect_modules.py --config models_yaml/anytext2_sd15.yaml --ckpt models/anytext_v2.0.ckpt

# 从 student_model 目录运行
cd /home/zyh/AnyText2/student_model
python inspect_modules.py --config ../models_yaml/anytext2_sd15.yaml --ckpt ../models/anytext_v2.0.ckpt

# 使用绝对路径（也支持）
python ./student_model/inspect_modules.py --config /home/zyh/AnyText2/models_yaml/anytext2_sd15.yaml --ckpt /home/zyh/AnyText2/models/anytext_v2.0.ckpt
```

### 测试路径解析

运行测试脚本验证路径是否正确：

```bash
conda run -n anytext2 python ./student_model/test_paths.py
```

预期输出：
```
✓ 所有路径解析正确！可以运行 inspect_modules.py
```

## 完整运行流程

### 1. 激活环境并测试路径
```bash
conda activate anytext2
cd /home/zyh/AnyText2
python ./student_model/test_paths.py
```

### 2. 运行模型检查
```bash
python ./student_model/inspect_modules.py \
    --config models_yaml/anytext2_sd15.yaml \
    --ckpt models/anytext_v2.0.ckpt \
    --output student_model/target_modules_list.txt
```

### 3. 开始训练（使用模拟数据）
```bash
accelerate launch student_model/train_lcm_anytext.py \
    --config models_yaml/anytext2_sd15.yaml \
    --teacher_ckpt models/anytext_v2.0.ckpt \
    --output_dir ./student_model/checkpoints \
    --use_mock_dataset \
    --train_batch_size 12 \
    --num_inference_steps 8
```

## 技术细节

### 路径解析逻辑

1. **获取脚本位置**：使用 `__file__` 获取脚本所在目录
2. **解析为绝对路径**：使用 `.resolve()` 消除符号链接和相对路径
3. **检查路径类型**：判断输入路径是绝对还是相对
4. **组合路径**：如果是相对路径，与父目录组合后解析

### 优势

- ✅ 支持从任意目录运行
- ✅ 支持相对路径和绝对路径
- ✅ 自动处理符号链接
- ✅ 跨平台兼容（Linux/Mac/Windows）

## 更新日期
2025-01-06
