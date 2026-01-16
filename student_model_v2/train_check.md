本文件是对每次项目进行大改时候的评审文件。
# v2_4版本
## 核心风险
显存不足  
当前方案中，Target 的计算逻辑是：
```python
# Teacher at t_prev (CFG)
eps_cond_prev = ...
eps_uncond_prev = ...
eps_teacher_prev = eps_uncond_prev + w * (...)
```
这意味着在一个训练步内，Teacher 需要执行 **4次 Forward**（$t$ 时刻 Cond/Uncond + $t_{prev}$ 时刻 Cond/Uncond），Student 执行 **1次 Forward**。
**总计：5次 UNet Forward。**

**风险判定**：
*   在 24GB 显存（RTX 4090）上，AnyText2（本身参数量大+ControlNet+OCR Encoder）跑 5 次 Forward，即使 `Batch Size=1`，也**处于 OOM 的边缘**。
*   **如果爆显存，请立即执行以下降级策略（Fallback）**：
    在计算 $t_{prev}$ 的 Target 时，**放弃 CFG，只跑 Cond**。
    即：`eps_teacher_prev = teacher_wrapper.forward(z_prev, t_prev, cond_emb, ...)`
    *代价：Target 的质量稍微下降（引导性变弱），但能省下 1 次 Teacher Forward，确保能跑起来。*

### 代码实现细节修正

在落地 `v2_4` 时，请注意以下两个细节修正，直接写入代码：

#### A. `lcm_utils_v2.py` 的 `sample_lcm_timesteps`
你的草案中直接在函数里转 tensor，建议把 `device` 处理得更稳健些。

```python
def sample_lcm_timesteps(schedule, batch_size, k, device):
    # 确保 k 不会导致越界
    max_idx = len(schedule) - k - 1
    # 随机采样索引
    idx = torch.randint(0, max_idx, (batch_size,), device=device).long()
    
    # 转换为 Tensor 并在 device 上索引
    if not torch.is_tensor(schedule):
        schedule = torch.tensor(schedule, device=device)
    else:
        schedule = schedule.to(device)
        
    t = schedule[idx]
    t_prev = schedule[idx + k]
    return t, t_prev
```

#### B. Huber Delta 的选择
你建议 `huber_delta: 0.001`。
*   **指正**：在 Latent Space 下，数值通常在 -4 到 4 之间。`delta=0.001` 实际上让 Huber Loss 退化成了 **L1 Loss (MAE)**。
*   **建议**：这没问题，L1 Loss 对异常值（Outliers）更鲁棒，生成的边缘往往更锐利。但如果发现 Loss 不下降或震荡，请改回 `1.0` (MSE behavior)。

**如果遇到 OOM，优先砍掉 $t_{prev}$ 的 Uncond 计算。**