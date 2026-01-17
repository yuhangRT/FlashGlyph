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

3) **v2_2：x0 回归（一次 Teacher forward）**  
Teacher 只跑一次，得到噪声预测，再按 DDIM 公式换算为 x0 作为监督目标。  
显存更省，更适合 AnyText2 这种大模型。

4) **v2_4：轨迹一致性（贴近官方 LCM）**  
Teacher 在更高噪声步预测并通过 DDIM 推进到下一步；Student 学习一致性目标（带边界缩放）。  
效果更稳，但显存/算力开销更高。

5) **时间步只在固定离散点采样**  
这符合 LCM 训练逻辑：让 Student 精准学习少数关键步骤。

## 目录结构

```
student_model_v2/
  dataset_anytext_v2.py      # 数据集与 collate（含 masked_img 逻辑）
  lcm_utils_v2.py            # 时间步采样与 x0 计算
  train_lcm_anytext_v2.py    # 训练脚本
  train_lcm_anytext_v2_2.py  # 训练脚本（优化版）
  train_lcm_anytext_v2_3.py  # 训练脚本（多域损失版）
  train_lcm_anytext_v2_4.py  # 训练脚本（轨迹一致性版）
  losses.py                 # FFL + Masked Grad 组合损失
  train_config_template_v3.yaml  # v3 训练配置模板
  train_config_template_v4.yaml  # v4 训练配置模板
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
也可以在 YAML 中设置 `train_script` 指定训练脚本（如 v3）：

```yaml
train:
  train_script: student_model_v2/train_lcm_anytext_v2_3.py
```

## 训练（v2.4 轨迹一致性，贴近官方 LCM）

v2.4 使用 Teacher 轨迹推进 + 边界缩放的一致性目标，训练更稳但显存更高。
推荐直接使用 v4 模板启动：

```bash
CUDA_VISIBLE_DEVICES=1,2 python student_model_v2/oom_guard.py --min-available-gb 4 \
  accelerate launch --num_processes 2 student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template_v4.yaml
```

可调参数：`w_min/w_max`（随机 CFG）、`lcm_step`（步长）、`target_from_teacher`（更严格但更耗显存）、`log_image_infer_steps`（预览推理步数）。

## 离线 LMDB 缓存（可选，加速数据管线）

用于缓存 `glyphs/gly_line/font_hint_base` 等重 CPU 特征，训练时只做轻量随机化。

构建 LMDB：

```bash
python student_model_v2/build_lmdb_cache.py \
  --dataset_json dataset/anytext2_json_files/anytext2_json_files/ocr_data/Art/data_v1.2b.json \
  --output_lmdb dataset_cache/anytext2_lmdb \
  --resolution 512 \
  --max_chars 20 \
  --font_path ./font/Arial_Unicode.ttf \
  --num_workers 4 \
  --map_size_gb 256
```

启用 LMDB（YAML `data` 段）：

```yaml
data:
  lmdb_path: dataset_cache/anytext2_lmdb
```

注意事项：
1) LMDB 的 `meta` 必须匹配 `resolution/max_chars/font_path/glyph_scale/vert_ang`  
2) `font_hint_randaug=true` 时会回退在线计算（不使用缓存 font_hint）  
3) 建议 `num_workers=4~6`，避免 64G 内存压力


## 训练（多域损失 v3：LCM + FFL + Masked Grad）

新增的 v3 脚本引入频域约束与文字区域梯度约束，提升文字边缘与笔画连贯性。
推荐使用 v3 模板直接启动：

```bash
CUDA_VISIBLE_DEVICES=1,2 python student_model_v2/oom_guard.py --min-available-gb 4 \
  accelerate launch --num_processes 2 student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template_v3.yaml
```

v3 模板里通过 `train_script` 指定训练脚本，便于扩展新版本。

## 训练（从 YAML 配置启动）

使用 `student_model_v2/train_config_template.yaml`、`student_model_v2/train_config_template_v3.yaml` 或 `student_model_v2/train_config_template_v4.yaml` 作为模板，在 YAML 里修改参数：

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

v3 模板示例：

```bash
python student_model_v2/oom_guard.py --min-available-gb 8 -- \
  accelerate launch --multi_gpu student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template_v3.yaml
```

v4 模板示例：

```bash
python student_model_v2/oom_guard.py --min-available-gb 8 -- \
  accelerate launch --multi_gpu student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template_v4.yaml
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

# 版本更替说明
## v2版本（是对v1的bug修改）
v2_3是添加了ffl的版本（没有用过，由于baseline效果不好，这个肯定达不到图片清楚文字清晰的效果）  
虽然能跑通baseline，但是效果很不好。训练结果保存在/checkpoints/train_20260115_090713_v2中。
核心缺点在于：**代码实现的数学逻辑与 LCM 官方原理存在偏差**，以及**训练配置中的超参数设置失误**。

仔细对比了 LCM-LoRA 官方论文/Demo 代码与你当前使用的 `train_lcm_anytext_v2_2.py`，发现了几处关键差异。正是这些差异导致了模型“只学到了皮毛，没学到灵魂”。

以下是深度分析报告：

---

### 第一部分：核心差异分析（Your Code vs. Official LCM-LoRA）

LCM-LoRA 能发顶刊的核心在于其独特的 Loss 设计和 Solver 逻辑。你的代码在“形”上模仿了，但在“神”上（数学逻辑）有偏差。

#### 差异 1：Target 计算逻辑（最致命的差异）

*   **Official LCM-LoRA (Demo/Paper)**:
    真正的 LCM 蒸馏使用的是 **Consistency Loss**。
    Target 不是 Teacher 在当前时刻 $t$ 的预测，而是 **Teacher 模拟 ODE 轨迹走到 $t_{next}$ (或 $t-k$) 后的预测**。
    $$ \text{Target} = \text{TeacherSolver}(z_t, t, t-k) $$
    它强迫 Student 在 $t$ 时刻直接预测出 Teacher 在 $t-k$ 时刻的结果。

*   **Your Code (`train_lcm_anytext_v2_2.py`)**:
    ```python
    # 代码逻辑简化版
    noise_pred_teacher = ... # Teacher 预测当前 t 的噪声
    target_x0 = predict_x0_from_eps(noisy_latents, timesteps, noise_pred_teacher, ...)
    loss = F.huber_loss(pred_x0_student, target_x0)
    ```
    **你的逻辑是“单步 x0 回归”**。你让 Student 去拟合 Teacher 在当前 $t$ 时刻认为的 $x_0$。
    **后果**：Student 学会了“去噪”，但没学会“加速”。在噪声很大的早期（如 t=900），Teacher 预测的 $x_0$ 本身就是极其模糊、充满噪声的（因为 Teacher 也猜不到最终结果）。Student 拼命去拟合这个模糊的目标，结果就是你看到的**“图形完全看不清，全是伪影”**。

#### 差异 2：Huber Loss 的 Delta 设置

*   **Official LCM**: 通常使用 `F.mse_loss` 或者 `delta` 较小的 Huber Loss。
*   **Your Code**: `loss = compute_lcm_loss(..., loss_type="huber")` (通常默认 delta=1.0)。
    对于 Latent Space 的 $x_0$ 预测，数值范围通常在 $[-1, 1]$ 之间。如果 Delta 设得太大，Huber Loss 就退化成了 MSE。这本身问题不大，但在高噪声区间，Student 容易为了平均误差而输出“灰色”图像。

#### 差异 3：CFG Guidance 的“烘焙”方式

*   **Official LCM**: 明确要求 Teacher 使用很大的 Guidance Scale (e.g., $w=8.0$)，并且 Loss 计算是在 $x_0$ 空间进行的。
*   **Your Code**: 实现了 CFG，这点是做对了。但由于 Target 是单步 $x_0$，Teacher 的 CFG 效果被高噪声掩盖了。

---

### 第二部分：为什么 28500 步效果依然很差？（归因分析）

除了上述算法逻辑差异，你的配置文件中还有两个**“隐形杀手”**：

#### 1. 学习率过高 (Learning Rate Too High)
*   **你的设置**: `learning_rate: 5e-4`
*   **标准 LoRA 训练**: 通常是 `1e-4` 或 `5e-5`。
*   **后果**：LCM 蒸馏对参数非常敏感。`5e-4` 对于 LoRA 来说太大了，会导致权重在最优解附近剧烈震荡，无法收敛到细腻的纹理。这解释了为什么你的图看起来充满了高频噪声和伪影（像过曝的照片）。

#### 2. Batch Size 过小且 Accumulation 策略存疑
*   **你的设置**: `bs=4`, `accum=2` -> `Effective BS = 8`。
*   **问题**：对于扩散模型的蒸馏，Batch Size 过小会导致梯度的方差极大。模型一会往左走，一会往右走，很难学到通用的去噪规律。
*   **建议**：Effective BS 至少要达到 **32** 甚至 **64**。

#### 3. 缺乏“高频约束”
你之前的尝试中去掉了 FFL 和 Gradient Loss，回到了纯 Huber Loss。
*   在“单步 x0 回归”的错误逻辑下，Huber Loss 倾向于生成模糊的平均值。
*   AnyText 的文字部分如果没有额外的边缘约束，在模糊的目标下就会变成一团墨迹。

---

### 第三部分：修正方案（Action Plan）

为了达到 Demo 那样的效果，必须把代码逻辑改回 **真正的 LCM**，并调整超参。

#### 步骤 1：修改 `lcm_utils_v2.py` (引入 Solver 逻辑)

你需要在 `lcm_utils_v2.py` 中实现一个 DDIM Solver Step，用于计算 Teacher 的“下一步”。

```python
# 在 lcm_utils_v2.py 中添加
def ddim_step(noise_pred, timesteps, latents, alphas_cumprod, eta=0.0):
    """
    模拟 DDIM 往前走一步，计算 z_{t-1} (Next Latent) 而不是 x0
    """
    # 1. 获取 alpha_t, alpha_prev
    # 注意：这里需要处理 timesteps 的索引映射
    # 简单起见，假设 timesteps 是实际的时间步
    
    # ... (需要实现完整的 DDIM update 公式: x_t -> x_t-1) ...
    # 核心公式: x_prev = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev) * pred_epsilon
    pass
```
修正后，写出了v3版本

## v2_4版本（在v2_2上修改核心算法，遵循lcm-lora）
训练命令：
```
CUDA_VISIBLE_DEVICES=1,2 python student_model_v2/oom_guard.py --min-available-gb 4 \
  accelerate launch --num_processes 2 student_model_v2/launch_from_yaml.py \
  --config student_model_v2/train_config_template_v4.yaml
```
