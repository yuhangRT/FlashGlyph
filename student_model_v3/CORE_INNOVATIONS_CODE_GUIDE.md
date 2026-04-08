# FlashGlyph v3 核心创新点与代码教学指南

本文档面向刚接触 `student_model_v3` 的新同学，目标是两件事：
1. 先把“论文里真正的创新点”拆清楚（按代码事实，不按口号）。
2. 再给出一条从配置到训练循环的代码级学习路径，确保你能独立改 loss、做消融、定位问题。

---

## 1. 一页结论：这篇工作的核心创新到底是什么

从当前代码看，FlashGlyph v3 的创新不是单一 loss，而是“**LCM-LoRA 主干 + 文本结构增强约束栈**”。

主创新可分为 6 层（按默认主线 `lcm_v3.yaml` 和完整消融栈）：

1. **LoRA 目标模块重设计（针对 AnyText2）**
   - 不是全量微调，而是只训练 LoRA；
   - 只选扩散主干和 control 分支关键层，并跳过 `glyph_block/position_block`；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:74`

2. **文本区域加权的一致性蒸馏（Mask-Weighted LCM）**
   - 在 LCM 一致性损失上对文本区域加权，避免背景淹没文字监督；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:904`、`student_model_v3/train_lcm_anytext_v3.py:934`

3. **Attention Mass Distillation（注意力质量蒸馏）**
   - 记录 cross-attention 中文本 token 质量分布，对学生/教师做 KL 对齐；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:1046`、`student_model_v3/train_lcm_anytext_v3.py:1205`
   - 支撑模块：`student_model_v3/attn_distill.py:10`、`ldm/modules/attention.py:165`

4. **OCR-CTC 语义约束**
   - 在训练中解码学生预测图，裁剪文本框后用 recognizer 计算 CTC loss；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:1221`

5. **clDice 拓扑约束**
   - 用软骨架化的 clDice 约束文字结构连通性；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:1235`
   - 支撑模块：`student_model_v3/topology_loss.py:28`

6. **SPFD / Sharpness 分支（FFL + Grad）**
   - 频域 FFL + 梯度约束，且新版本加入 residual 视角和 soft window；
   - 代码：`student_model_v3/train_lcm_anytext_v3.py:1267`
   - 核心实现：`student_model_v3/losses.py:93`

一句话总结：  
**v3 把“少步推理带来的文字结构退化”拆成可控约束：区域权重（哪里学）+ 注意力对齐（看哪里）+ OCR（读什么）+ 拓扑（结构不断）+ 频域锐化（边界不糊）。**

---

## 2. 先看入口：从 YAML 到训练循环

### 2.1 启动路径

推荐启动命令：
```bash
python3 student_model_v3/launch_single_gpu.py --config student_model_v3/configs/lcm_v3.yaml --gpu 0
```

调用链是：
1. `launch_single_gpu.py` 负责设 `CUDA_VISIBLE_DEVICES` 和可选 OOM guard。
2. `launch_from_yaml.py` 读取 YAML，并把 `model/data/train` 三段转成 CLI 参数。
3. `train_lcm_anytext_v3.py` 真正执行训练。

关键代码：
- YAML 合并：`student_model_v3/launch_from_yaml.py:25`
- bool/list 参数转 CLI：`student_model_v3/launch_from_yaml.py:38`
- `disable_xformers` 环境变量处理：`student_model_v3/launch_from_yaml.py:99`

注意：`train_script` 和 `disable_xformers` 是 launcher 层字段，不是 `train_lcm_anytext_v3.py` 的 argparse 参数本体。

### 2.2 参数落地点

训练脚本参数定义从：
- `student_model_v3/train_lcm_anytext_v3.py:543`

你要改创新行为，核心看这些参数组：
- 蒸馏主干：`loss_type/loss_text_weight/loss_mask_key`
- 注意力蒸馏：`loss_attn_weight` 及 `attn_*`
- OCR：`loss_ocr_weight/ocr_every`
- 拓扑：`loss_cldice_weight/cldice_iters/stroke_*`
- 频域锐化：`loss_ffl_weight/loss_grad_weight/ffl_*`

---

## 3. 代码级教学：训练循环里每一步在干什么

### Step A：构造 teacher/student + LoRA

1. 加载 teacher 和 student 同权重初始点  
   - `student_model_v3/train_lcm_anytext_v3.py:696`
2. 冻结 teacher 全参、冻结 student 基座  
   - `student_model_v3/train_lcm_anytext_v3.py:703`
3. 只在目标模块注入 LoRA  
   - 目标模块筛选：`student_model_v3/train_lcm_anytext_v3.py:74`
   - 注入逻辑：`student_model_v3/train_lcm_anytext_v3.py:723`

教学重点：  
`build_lora_target_modules()` 这段是 AnyText2 适配的关键工程创新，特别是：
- 跳过 `glyph_block/position_block`（`line 79`）；
- 覆盖 attention 投影 + control zero conv（`line 87`、`line 90`）。

### Step B：把样本编码成潜变量并采样时间步

1. 图像与 masked 图拼接后走 VAE encode  
   - `student_model_v3/train_lcm_anytext_v3.py:113`
2. 随机采样 DDIM 索引 `index`，得到 `start_timesteps` 和 `timesteps`  
   - `student_model_v3/train_lcm_anytext_v3.py:1012`
3. LCM 边界条件系数 `c_skip/c_out`  
   - `student_model_v3/lcm_solver.py:27`
   - 调用点：`student_model_v3/train_lcm_anytext_v3.py:1019`

### Step C：teacher-student 一致性蒸馏主干（LCM）

1. student 在 `x_t` 上预测 `student_pred_x0`  
   - `student_model_v3/train_lcm_anytext_v3.py:1084`
2. teacher 条件/无条件预测后，采样随机 `w` 做 CFG 组合  
   - `student_model_v3/train_lcm_anytext_v3.py:1130`
   - `w` 采样：`student_model_v3/train_lcm_anytext_v3.py:1150`
3. 通过 `solver.ddim_step` 得到 `x_prev`，再让 student 在 `t-Δ` 预测 target  
   - `student_model_v3/train_lcm_anytext_v3.py:1155`
   - `student_model_v3/train_lcm_anytext_v3.py:1163`

这部分就是论文里的“一致性蒸馏主方程”来源。

### Step D：文本区域加权（最先做、最稳的改进）

1. 取文本 mask（`hint` / `positions` / `inv_mask`）  
   - `student_model_v3/train_lcm_anytext_v3.py:904`
2. 用 `1 + (text_weight-1)*mask` 加权误差  
   - `student_model_v3/train_lcm_anytext_v3.py:942`
3. 得到 `lcm_loss` 作为总 loss 基础项  
   - `student_model_v3/train_lcm_anytext_v3.py:1188`

如果你是新人，先只看这层，做 `A0_nomask` vs `A0` 就能感受“文字区域重加权”的收益。

### Step E：可选增强约束（逐层叠加）

#### E1. Attention Mass Distill

1. 计算 token mask（占位符 token）并按 sigma/timestep gate  
   - `student_model_v3/train_lcm_anytext_v3.py:1055`
2. 开启 attention 记录，分别抓 student/teacher 的 mass  
   - `student_model_v3/train_lcm_anytext_v3.py:1078`
   - `student_model_v3/train_lcm_anytext_v3.py:1106`
3. 在文本区域内算 KL  
   - `student_model_v3/train_lcm_anytext_v3.py:1205`
   - 函数：`student_model_v3/train_lcm_anytext_v3.py:254`

底层修改在 `CrossAttention`：
- 新增 `_record_attn/_token_mask_spec/_last_mass`：`ldm/modules/attention.py:165`
- softmax 后提取 mass：`ldm/modules/attention.py:250`

#### E2. OCR-CTC

1. 把 `student_pred_x0` decode 成图  
2. 按位置掩码裁剪文本框（`_extract_text_crops`）  
3. 调 recognizer 做 CTC  
   - 主逻辑：`student_model_v3/train_lcm_anytext_v3.py:1221`
   - 文本裁剪函数：`student_model_v3/train_lcm_anytext_v3.py:300`

#### E3. Topology（clDice）

1. 用 `pred_img - masked_img` 与 `gt - masked_img` 得到编辑差分；
2. 下采样到 128，sigmoid 成笔画概率；
3. 计算 clDice。  
   - 调用：`student_model_v3/train_lcm_anytext_v3.py:1235`
   - 算法：`student_model_v3/topology_loss.py:17`

#### E4. SPFD（FFL+Grad）

1. 入口在 `HighFreqTextLoss`  
   - 调用：`student_model_v3/train_lcm_anytext_v3.py:1267`
2. FFL 频域损失：`student_model_v3/losses.py:10`
3. Grad 边缘损失：`student_model_v3/losses.py:161`
4. 新版强化点：
   - residual 模式：`pred_x0 - masked_x` 后再对齐（`line 201`）
   - soft window：mask 膨胀 + 高斯平滑降低频谱振铃（`line 149`）

这两点解释了为什么当前 sharpness 分支比“纯 FFL + Sobel”更稳。

---

## 4. 消融配置怎么对应创新点

当前 `student_model_v3/configs/` 中可直接映射创新层级：

1. `ablation_A0_nomask.yaml`  
   - 只有基础 LCM，`loss_text_weight=1.0`（不做文本重加权）。

2. `ablation_A0.yaml`  
   - 打开文本重加权（`loss_text_weight=5.0`），仍不加其他正则。

3. `ablation_A1.yaml`  
   - A0 + attention distill（`loss_attn_weight=0.1`）。

4. `ablation_A2.yaml`  
   - A1 + OCR（`loss_ocr_weight=0.01`）。

5. `ablation_A3.yaml`  
   - A2 + topology（`loss_cldice_weight=0.1`）。

6. `ablation_A4.yaml`  
   - A3 + SPFD（`loss_ffl_weight=0.05`、`loss_grad_weight=0.05`）。

7. `lcm_v3.yaml`  
   - 当前主线配置（Attn + OCR + clDice，默认 FFL/Grad 关闭）。

8. `lcm_v3_gl.yaml`  
   - 主线上加 sharpness（FFL+Grad）抛光版。

9. `lcm_v3_12b.yaml`  
   - 轻量快速试验版（1k steps，便于 smoke test）。

---

## 5. 新人最快上手路线（建议按这个顺序）

1. **先跑一个最小 smoke**
   - 配置：`student_model_v3/configs/lcm_v3_12b.yaml`
   - 目的：验证环境、数据路径、checkpoint 加载都通。

2. **理解核心创新最小闭环**
   - 对比：`ablation_A0_nomask.yaml` vs `ablation_A0.yaml`
   - 只看“文本重加权”带来的变化。

3. **逐层加约束做认知分离**
   - A1 → A2 → A3 → A4
   - 每次只多一个约束，避免“堆料后不知道谁生效”。

4. **回到主线配置**
   - 跑 `lcm_v3.yaml`
   - 若你需要更锐利边缘，再试 `lcm_v3_gl.yaml`。

5. **用推理脚本做可视化验证**
   - `student_model_v3/infer_lcm_anytext_v3.py`
   - 同一 batch 对比 teacher 50-step vs student 4-step。

---

## 6. 常见误区与排错

1. **`loss_attn_weight>0` 但没关 xformers**
   - 会直接报错；
   - 检查 `launch_from_yaml.py` 是否生效设置 `DISABLE_XFORMERS=1`。

2. **文字 mask 选错导致“看似没学到字”**
   - `loss_mask_key` 默认建议 `inv_mask`；
   - 若用 `hint`，注意阈值逻辑在 `wm_thresh` + fallback 0.5。

3. **OCR loss 不稳定**
   - 先确认 `_extract_text_crops` 裁剪到有效框；
   - `ocr_every` 可先调大降低波动。

4. **FFL 开了但训练发散或伪纹理增强**
   - 先减 `loss_ffl_weight`；
   - 保留 grad 分支，逐步回加 FFL；
   - 检查 residual + soft window 是否打开（当前默认是开）。

5. **以为“配置写了就生效”**
   - `train_script`、`disable_xformers` 是 launcher 字段；
   - 直接运行 `train_lcm_anytext_v3.py` 时，这两个字段不会自动处理。

---

## 7. 一张代码地图（按阅读顺序）

建议阅读顺序（每个文件只看关键段）：

1. `student_model_v3/launch_from_yaml.py:25`  
   - 理解 YAML 如何变成 CLI。

2. `student_model_v3/train_lcm_anytext_v3.py:74`  
   - LoRA 目标模块选择逻辑。

3. `student_model_v3/train_lcm_anytext_v3.py:991`  
   - 主训练循环入口。

4. `student_model_v3/train_lcm_anytext_v3.py:1188`  
   - 基础一致性 + mask reweight。

5. `student_model_v3/train_lcm_anytext_v3.py:1197`、`student_model_v3/train_lcm_anytext_v3.py:1221`、`student_model_v3/train_lcm_anytext_v3.py:1235`、`student_model_v3/train_lcm_anytext_v3.py:1267`  
   - 四个增强约束挂载点。

6. `student_model_v3/losses.py:93`  
   - SPFD 具体实现。

7. `student_model_v3/attn_distill.py:10` + `ldm/modules/attention.py:165`  
   - 注意力蒸馏的 hook 与 mass 记录。

8. `student_model_v3/topology_loss.py:28`  
   - clDice 拓扑损失实现。

---

## 8. 给论文撰写同学的技术口径建议

如果你要把“创新点”写进论文，建议用这套口径：

1. 主创新：**文本结构退化的分层治理框架**，而非单一新 loss。
2. 第一性改进：**文本区域重加权一致性蒸馏**（泛化稳定、可解释）。
3. 结构增强：attention / OCR / topology / SPFD 分层叠加，各自有开关与消融。
4. 工程贡献：LoRA 目标模块选择 + AnyText2 wrapper 状态重置，保障可训练性与稳定性。

这套口径和代码结构是一一对应的，不会出现“论文说了但代码没有”的断层。

