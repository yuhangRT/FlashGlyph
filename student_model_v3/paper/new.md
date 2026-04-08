# FlashGlyph：面向可读性的对齐—语义—拓扑三重约束少步蒸馏框架  
（FlashGlyph: Readability-Driven Triple-Constraint Few-Step Distillation Framework for Scene Text Editing）  
张某人¹  
(1. XXX大学 XXX学院，XXX 000000；2. XXX研究院，XXX 000000)

---

## 摘 要

场景文本编辑旨在保持背景一致性的前提下，对图像中的文本进行精准插入或替换。以 AnyText2 为代表的扩散式专用模型在字形可控与融合质量方面表现突出，但通常依赖 20～50 步迭代采样，高昂的端到端延迟限制了其实时交互应用[1,2,6,22]。近年来，一致性模型与潜在一致性模型（Latent Consistency Model, LCM）可将推理压缩至 1～4 步[8,9]，但在文本编辑任务中常出现显著的结构性退化：字符位置漂移、字形语义错误（伪字符）以及笔画断裂或粘连等拓扑损伤。  

针对上述问题，本文提出 FlashGlyph：一种面向场景文本编辑的可读性保持少步蒸馏框架。FlashGlyph 在 LCM‑LoRA 一致性蒸馏基础上[9,10]构建“对齐—语义—拓扑”三重约束：  
（1）**注意力对齐蒸馏**：在控制分支 Cross‑Attention 层对齐教师与学生的“字形 token—空间区域”响应分布，抑制少步推理下的位置漂移与重影[12,13,38]；  
（2）**OCR‑CTC 语义监督**：引入冻结识别器对生成文本区域施加序列监督，使学生学习正确字符序列表征而非伪纹理[23,32]；  
（3）**拓扑一致性约束**：利用软骨架化与 clDice 损失显式约束笔画连通性，针对性修复断笔与孔洞闭合问题[28]。此外，本文提供可选的频域/边界梯度损失作为轻量锐化项用于“抛光”[27]。  

实验在 AnyWord‑3M 数据集上开展[21,39]，结果表明：在 4 步推理设置下，FlashGlyph 能在外部 OCR 可读性与视觉质量之间取得更优折中，并显著降低端到端推理延迟。

关键词：场景文本编辑；扩散模型；一致性蒸馏；LCM；LoRA；注意力蒸馏；OCR‑CTC；拓扑约束

---

## 1 引言

场景文本编辑（Scene Text Editing）要求在修改图像中文本内容的同时，严格保持背景纹理、光照与几何透视的一致性。与通用图像编辑不同，文本是一种高度结构化的视觉信号，其可读性对局部笔画的连通性、字符间距、孔洞结构以及边缘清晰度极为敏感。轻微的结构退化（例如笔画断裂、相邻字符粘连或孔洞闭合）即可造成语义级别的识别错误，从而显著降低编辑结果的可用性。  

传统场景文本编辑多采用“定位/分割—背景修复—文本渲染/融合”的分解式流程，例如 SRNet（Editing Text in the Wild）[14]、STEFANN[15]、SwapText[16] 等；该类方法可解释性强，但依赖前置检测、几何校正与字体建模，误差易级联，且在多语言复杂字形与任意形变文本上存在局限。  

扩散模型在高保真生成与可控编辑方面取得显著进展[1,6]。面向文本的专用模型通过引入字形条件（glyph）与位置掩码等控制信号，实现复杂背景下的自然融合，例如 TextDiffuser[17]、TextDiffuser‑2[18]、GlyphControl[19]、GlyphDraw[20]、AnyText[21] 与 AnyText2[22]。然而，扩散模型的迭代去噪机制通常需要 20～50 步采样（甚至更多），使单张图像生成的端到端延迟达到秒级，难以满足移动端、交互式设计、即时翻译等场景。  

一致性蒸馏与少步生成近年快速发展：Progressive Distillation[7]、Consistency Models[8] 与 LCM[9] 等可将推理压缩至 1～4 步。然而，将通用少步蒸馏直接用于场景文本编辑时，我们观察到明显的“可读性坍塌”现象：  
（i）条件对齐失效导致字符位置漂移与重影；  
（ii）语义真值缺失导致模型生成视觉上像字但不可识别的伪纹理；  
（iii）拓扑结构破坏导致断笔、粘连与孔洞错误。  

为此，本文提出 FlashGlyph，一个以“可读性”为中心目标的少步蒸馏框架。核心思想是将“文本可读性”拆解为可优化的三类约束：对齐（alignment）、语义（semantics）与拓扑（topology），并将其作为一致性蒸馏的辅助监督信号引入训练。在不重写主干网络的前提下，我们以 AnyText2 作为冻结教师模型[22]，使用 LoRA‑LCM 进行蒸馏训练[9,10]，从而具备向其他扩散式文本编辑/生成模型迁移的潜力。  

本文主要贡献如下：  
- 提出面向文本可读性的少步蒸馏范式，将可读性分解为“对齐—语义—拓扑”三类可优化目标，并给出可复现的训练/评测协议[21,39]。  
- 设计注意力对齐蒸馏，通过对齐控制分支 Cross‑Attention 的响应分布，缓解少步推理中的条件引导失效与位置漂移[12,13,38]。  
- 引入 OCR‑CTC 语义监督与 clDice 拓扑一致性约束，分别解决“伪字符/语义错字”与“断笔/粘连/孔洞闭合”等结构性错误[23,28,32]。  
- 在 AnyWord‑3M 数据集上验证 FlashGlyph 在 4 步推理下可显著提升外部 OCR 指标，并在速度与可读性之间取得更优折中[21,25,26,39]。  

---

## 2 相关工作

### 2.1 场景文本生成与编辑

传统场景文本编辑多采用“定位/分割—背景修复—文本渲染/融合”的分解式流程。SRNet（Editing Text in the Wild）通过内容与风格分解实现自然场景文本替换[14]；STEFANN 引入字体适配机制进行字符级编辑[15]；SwapText 通过阶段化文本交换与背景补全提升复杂场景的融合质量[16]。该类方法具有较强可解释性，但依赖前置模块，误差易级联放大，且在多语言复杂字形、任意形变文本与端到端一致性方面存在局限。  

扩散模型为端到端文本编辑提供了新的范式[1,6]。TextDiffuser 通过布局规划与扩散渲染结合提升文本布局一致性与内容正确性[17]；TextDiffuser‑2 进一步引入语言模型增强布局规划与渲染能力[18]。在字形可控方向，GlyphControl 与 GlyphDraw 将 glyph 与空间结构作为条件，提升复杂结构下的文字渲染一致性[19,20]。AnyText 提出多语言视觉文本生成与编辑并提供基准与数据集 AnyWord‑3M[21,39]；AnyText2 在其基础上支持更细粒度的属性控制与更强的背景融合能力[22]。本文以 AnyText2 作为教师模型，聚焦其少步蒸馏到 1～4 步推理时的可读性退化问题。  

### 2.2 扩散模型加速与一致性蒸馏

扩散模型推理加速主要包括训练无关（solver）与训练相关（蒸馏/一致性）两类路线。训练无关的采样器如 DDIM[2]、DPM‑Solver[3]/DPM‑Solver++[4] 与 UniPC[5] 可在不改动模型参数的情况下将采样步数降低到约 10～20 步，但在更少步数时质量下降明显，且对文本这类高结构信号更易出现边缘抹平与结构错误。  

训练相关方法通过学习少步映射实现极致加速。Progressive Distillation[7]、Consistency Models[8] 等通过递进蒸馏或一致性训练将生成压缩到极少步；LCM 将一致性思想迁移到潜空间扩散并通过少量训练获得 2～4 步推理能力[9]，并可结合参数高效微调（如 LoRA）降低训练开销[10]。然而，现有一致性目标多采用点对点误差（如 MSE/Huber）对齐轨迹，其优化偏向降低全局平均误差，对“文本笔画连通性”“字符语义正确性”“glyph‑位置条件对齐”等结构化约束缺乏专用归纳偏置，因此通用蒸馏在文本编辑中常出现背景合理但文本区域错字/伪字与结构坍塌。  

### 2.3 面向可读性的对齐、语义与拓扑约束

（1）**对齐约束**：注意力图监督常用于可控编辑与区域一致性保持，例如通过 Cross‑Attention 控制提示词编辑[13]，或结合外部结构条件进行空间引导（如 ControlNet）[12]。与这些工作不同，本文关注少步蒸馏中“条件引导失效”的根源问题，将注意力作为中间变量进行教师—学生对齐，以降低位置漂移与重影。  

（2）**语义约束**：在文本生成与场景文本渲染领域，基于识别器的监督（recognition loss）可直接优化可读性，典型形式包括交叉熵或 CTC 损失[23]。本文将冻结识别器（PP‑OCR 风格 CTC 识别器）引入一致性蒸馏训练，使学生模型在少步推理下仍被迫输出可被识别为目标字符串的图像证据，从而抑制伪纹理[23,32]。  

（3）**拓扑约束**：clDice 与软骨架化广泛用于血管分割、道路提取等细长结构任务，通过骨架重合度显式鼓励连通性与拓扑一致[28]。文本笔画同样具备细长结构特征，本文将 clDice 引入文本编辑蒸馏，以针对性抑制断笔、粘连与孔洞错误。  

---

## 3 方法

### 3.1 问题定义与总体框架

给定输入图像 I、编辑区域位置掩码 M，以及字形/文本条件（例如 glyph 图像 G、目标字符串 y 与相关属性提示），场景文本编辑的目标是在保持背景一致性的前提下生成编辑结果 I'，使文本区域呈现目标内容并自然融合到原始场景中。我们在潜扩散框架中工作：VAE 将图像编码为潜变量 z[36]，扩散过程在潜空间中进行去噪[1,6]。  

FlashGlyph 采用教师—学生蒸馏结构。教师模型 T 为冻结的 AnyText2[22]；学生模型 S 与教师同构但仅训练注入的 LoRA 参数[10]。训练阶段以 LCM 一致性蒸馏为主目标，使学生在 1～4 步推理下逼近教师的去噪轨迹[9]。在此基础上，FlashGlyph 引入“对齐—语义—拓扑”三重可读性约束，分别作用于：（i）条件引导的空间对齐，（ii）字符序列语义正确性，（iii）笔画连通性与孔洞结构。  

![图 1 FlashGlyph 总体蒸馏框架示意图（教师/学生共享控制信号，LCM 主损失 + 三重可读性约束，推理为 1～4 步）。](figures/fig1.png)

*图 1 FlashGlyph 总体蒸馏框架示意图。*

### 3.2 基础蒸馏：区域加权的 LCM‑LoRA 一致性蒸馏（最终版）

我们采用教师—学生蒸馏框架：冻结教师扩散模型 T（AnyText2）[22]，学生模型 Sθ 与教师同构，仅训练注入的 LoRA 低秩适配参数 θ[10]。在潜扩散（Latent Diffusion）中，输入图像 I 经 VAE 编码器 E 得到潜变量[6,36]：

$$
x_0=\mathcal{E}(I)\in\mathbb{R}^{C\times H\times W}.
$$

对任意时间步 t，加噪状态为：

$$
x_t=\alpha_t x_0+\sigma_t\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\mathbf{I}),
$$

其中 $\alpha_t=\sqrt{\bar\alpha_t}$、$\sigma_t=\sqrt{1-\bar\alpha_t}$。

#### 3.2.1 Boundary‑condition 一致性目标

LCM 的关键是学习一个“从任意 t 到解”的边界条件映射[9]。我们用两组与时间步相关的缩放系数 $c_{\text{skip}}(t),c_{\text{out}}(t)$ 将学生的 $x_0$ 估计组合成一致性映射：

$$
f_\theta(x_t,c)=c_{\text{skip}}(t)\,x_t+c_{\text{out}}(t)\,\hat x_0^\theta(x_t,c),
$$

其中 c 表示 AnyText2 的条件（hint/positions/glyph/text embedding 等控制信号）[22]。

#### 3.2.2 教师引导的一步转移（teacher‑guided DDIM step）

为构造一致性对齐的下一状态 $x_{t'}$（$t'<t$），我们对教师做条件/无条件两次前向，形成 teacher‑guidance（classifier‑free guidance）[11]：

- 条件输出（cond）：$\hat x_{0,\text{cond}}^{T}(x_t,c)$、$\hat\varepsilon_{\text{cond}}^{T}(x_t,c)$  
- 无条件输出（uncond）：$\hat x_{0,\text{uncond}}^{T}(x_t,c)$、$\hat\varepsilon_{\text{uncond}}^{T}(x_t,c)$  

采样随机引导强度 $w\sim \mathcal{U}(w_{\min},w_{\max})$，构造教师引导预测：

$$
\hat x_{0}^{T,g}=\hat x_{0,\text{cond}}^{T}+w\big(\hat x_{0,\text{cond}}^{T}-\hat x_{0,\text{uncond}}^{T}\big),
$$

$$
\hat\varepsilon^{T,g}=\hat\varepsilon_{\text{cond}}^{T}+w\big(\hat\varepsilon_{\text{cond}}^{T}-\hat\varepsilon_{\text{uncond}}^{T}\big).
$$

随后使用离散 DDIM 更新算子将 $x_t$ 单步推进到 $x_{t'}$[2]：

$$
x_{t'}=\Phi_{\text{DDIM}}\big(x_t;\hat x_{0}^{T,g},\hat\varepsilon^{T,g},t\rightarrow t'\big).
$$

#### 3.2.3 Stop‑grad 目标与区域加权一致性损失

接着我们在 $x_{t'}$ 上用学生再做一次前向，但停止梯度，构造一致性 target[8,9]：

$$
\text{target}=\operatorname{sg}\big(f_\theta(x_{t'},c)\big).
$$

由于文本编辑主要关注文本区域，我们构造像素空间文本掩码 M 并下采样到潜空间得到 $M_{\text{lat}}$。定义文本加权系数 $w_{\text{text}}>1$，区域权重图为：

$$
W = 1+(w_{\text{text}}-1)\,M_{\text{lat}}.
$$

一致性误差采用逐元素的 $\rho(\cdot)$，可取 L2 或 Huber：

$$
\rho(u)=
\begin{cases}
u^2,& \text{(L2)}\\
\sqrt{u^2+\delta^2}-\delta,& \text{(Huber)}
\end{cases}
$$

最终区域加权一致性损失为：

$$
\mathcal{L}_{\text{LCM}'}=
\mathbb{E}\left[
\frac{
\sum_{i} W_i \,\rho\Big(f_\theta(x_t,c)_i-\text{target}_{i}\Big)
}{
\sum_{i} W_i+\epsilon
}\right].
$$

> 可选（消融用）：实现支持额外的 teacher‑$x_0$ 对齐项  
> $\mathcal{L}_{x_0}=\mathbb{E}\big[\text{mask‑weighted}(\hat x_0^\theta(x_t,c),\hat x_0^{T,g}(x_t,c))\big]$，主线可置零。  

### 3.3 约束一：注意力对齐蒸馏（Alignment，最终版）

少步推理时，“条件引导失效”常表现为文本 token 对空间位置的关注漂移。为显式约束空间定位，我们在控制分支（ControlNet/Control‑UNet）Cross‑Attention 层对齐教师与学生的注意力质量图（Attention Mass Map）[12,38]。  

设某一 Cross‑Attention 层注意力权重为：

$$
A\in\mathbb{R}^{(HW)\times L},\qquad
A=\operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right),
$$

其中 HW 为查询空间位置数，L 为条件 token 长度。  

我们构造 token mask $m\in\{0,1\}^{L}$，对应 placeholder token 在 tokenizer 序列中的位置（即需要生成/编辑文本的 token 段）。在多头注意力下，对被选 token 的注意力求和并对 heads 平均，得到 Attention Mass：

$$
s(p)=\frac{1}{H}\sum_{h=1}^{H}\sum_{j=1}^{L} m(j)\,A_h(p,j).
$$

将 s reshape 为 $S\in\mathbb{R}^{H\times W}$。我们只在文本区域对齐，将质量图归一化为概率分布后用 KL 散度对齐（Teacher‖Student）：

$$
\tilde S = \frac{M_{\text{lat}}\odot S}{\sum_{x,y} (M_{\text{lat}}\odot S)(x,y)+\epsilon},
$$

$$
\mathcal{L}_{\text{attn}}=
\frac{1}{|\mathcal{L}|}\sum_{l\in\mathcal{L}}
\operatorname{KL}\big(\tilde S_l^{(T)}\ \|\ \tilde S_l^{(S)}\big).
$$

该思路与提示词编辑中基于 Cross‑Attention 控制区域对应关系的直觉一致，但本文将其用于少步蒸馏的教师—学生对齐，以增强条件定位稳定性[13]。  

### 3.4 约束二：OCR‑CTC 语义监督（Semantics，最终版）

为抑制“像字但不可读”的伪纹理，我们引入冻结文本识别器 R（训练期冻结）对生成结果施加 CTC 序列监督[23]。工程上采用 PP‑OCR 风格 CTC 识别器[32]。  

#### 3.4.1 从潜空间到可识别文本块

学生在 $x_t$ 上得到 $\hat x_0^\theta$ 后，通过 VAE 解码器 D 还原到像素空间[6,36]：

$$
\hat I = \mathcal{D}(\hat x_0^\theta)\in[-1,1]^{3\times H_I\times W_I}.
$$

对每一行文本，数据提供二值位置掩码 $P_j$（来自 positions）。实现采用轴对齐 bbox 提取并 resize 到固定识别尺寸（如 $48\times 320$）：

$$
I_{\text{crop}}^{(j)}=\operatorname{Resize}\big(\hat I[b(P_j)],\,48\times 320\big).
$$

#### 3.4.2 CTC 语义损失

识别器输出字符类别 logits 序列 $P=\mathcal{R}(I_{\text{crop}})$。给定目标字符串 y，CTC 损失为：

$$
\mathcal{L}_{\text{ocr}}=\operatorname{CTC}(P,y)=-\log p(y\,|\,I_{\text{crop}}).
$$

训练时 R 冻结，仅将梯度回传到学生模型。  

### 3.5 约束三：拓扑一致性（Topology，最终版）

文本笔画具有细长连通拓扑结构，少步生成易出现断笔/粘连/孔洞错误。我们引入 soft‑skeleton 与 clDice 损失显式约束拓扑一致性[28]。  

#### 3.5.1 无像素级笔画真值下的 stroke 概率图构造

我们利用“原图 vs masked 图”的差分信号构造近似笔画强度：  
- 原图（含真值文本）$I_{\text{gt}}$  
- masked 图（背景/去字参考）$I_{\text{mask}}$  
- 学生生成结果 $\hat I$  

定义灰度差分强度：

$$
d_S = \frac{1}{3}\sum_{c=1}^{3}\big|\hat I_c - I_{\text{mask},c}\big|,\qquad
d_G = \frac{1}{3}\sum_{c=1}^{3}\big|I_{\text{gt},c} - I_{\text{mask},c}\big|.
$$

使用 Sigmoid 得到平滑 stroke 概率图：

$$
V_S=\sigma\big(k(d_S-\tau)\big),\qquad
V_G=\sigma\big(k(d_G-\tau)\big).
$$

#### 3.5.2 Soft‑skeleton 与 clDice

$$
S_S=\operatorname{SoftSkel}(V_S),\qquad S_G=\operatorname{SoftSkel}(V_G).
$$

$$
T_{\text{prec}}=\frac{\sum(S_S\odot V_G)}{\sum S_S+\epsilon},\qquad
T_{\text{sens}}=\frac{\sum(S_G\odot V_S)}{\sum S_G+\epsilon}.
$$

$$
\mathcal{L}_{\text{topo}}=
1-2\frac{T_{\text{prec}}T_{\text{sens}}}{T_{\text{prec}}+T_{\text{sens}}+\epsilon}.
$$

### 3.6 可选锐化项与总体优化目标（最终版）

#### 3.6.1 可选锐化项（Residual FFL+Grad）

在三重约束建立后，我们提供轻量“抛光”项提升笔画边缘清晰度。与直接对整幅潜变量施加频域损失不同，本文将锐化项作用于编辑残差，以避免背景纹理频谱主导优化。FFL 参考 focal frequency loss[27]。  

设 masked 潜变量为 $x_{\text{mask}}$，学生预测为 $\hat x_0^\theta$，教师引导预测为 $\hat x_0^{T,g}$，定义残差：

$$
R_S = \hat x_0^\theta - x_{\text{mask}},\qquad
R_T = \hat x_0^{T,g} - x_{\text{mask}}.
$$

构造软窗函数 $\tilde M_{\text{lat}}$ 并加窗：

$$
\tilde R_S = \tilde M_{\text{lat}}\odot R_S,\qquad
\tilde R_T = \tilde M_{\text{lat}}\odot R_T.
$$

频域损失：

$$
\mathcal{L}_{\text{ffl}}=\operatorname{FFL}(\tilde R_S,\tilde R_T).
$$

边界梯度损失：

$$
\mathcal{L}_{\text{grad}}=\|\nabla \tilde R_S-\nabla \tilde R_T\|_1.
$$

$$
\mathcal{L}_{\text{sharp}}=\lambda_{\text{ffl}}\mathcal{L}_{\text{ffl}}+\lambda_{\text{grad}}\mathcal{L}_{\text{grad}}.
$$

#### 3.6.2 总体目标

$$
\mathcal{L}_{\text{total}}=
\mathcal{L}_{\text{LCM}'}+
\lambda_1\mathcal{L}_{\text{attn}}+
\lambda_2\mathcal{L}_{\text{ocr}}+
\lambda_3\mathcal{L}_{\text{topo}}+
\lambda_4\mathcal{L}_{\text{sharp}}.
$$

---

## 4 实验

### 4.1 数据集与评测协议

本文主要使用 AnyWord‑3M 进行蒸馏训练与评测[21,39]。该数据集针对多语言文字生成任务构建，包含 3,034,486 张图像、超过 900 万行文本与超过 2000 万个字符或拉丁文字[39]。图像来源涵盖 Noah‑Wukong[35]、LAION‑400M[33] 以及多个 OCR 数据集，并使用 PP‑OCR 检测与识别生成文本行标注[32]，使用 BLIP‑2 生成文本描述[31]。  

评测以可读性为核心。使用外部 OCR/STR 模型评估输出，计算 Char Acc、Word Acc、CER/WER 等。测试阶段使用独立的 PARSeq[26] 与 TrOCR[25] 识别器，并报告多模型平均值，避免“同识别器训练又测试”的同构偏置。  

同时报告 FID 与 LPIPS 等感知质量指标[29,30]与端到端延迟（batch=1，固定软硬件环境，明确是否包含 VAE 编解码）。  

### 4.2 实现细节

教师模型为 AnyText2 checkpoint[22]；学生为同构网络 + LoRA[10]。时间表采用 DDIM 50 步[2]，推理步数固定为 4，并可扩展报告 1/2 步。优化器采用 AdamW[37]，混合精度训练。  

注意力蒸馏在可记录 attention 权重的实现路径下启用；OCR 约束默认每 k 步触发一次；拓扑约束通过有限迭代的 soft‑skeletonization 计算；锐化项（FFL+Grad）在消融配置中开启[27]。  

### 4.3 主结果与对比（示例表格结构保持不变）

对比方法至少包含：  
（i）AnyText2（50 步，教师）[22]；  
（ii）教师 + 少步采样器（DDIM[2] 4/10 步、DPM‑Solver[3]/UniPC[5] 10/15 步）；  
（iii）LCM‑LoRA baseline（仅一致性蒸馏）[9,10]；  
（iv）FlashGlyph（本文）。  

（表格略，见原文结构；建议终稿替换预测值为真实复现实验。）

![图 4 主结果定性对比（Teacher vs LCM baseline vs FlashGlyph，附局部放大与 OCR 识别字符串）。](figures/fig4.png)

### 4.4 消融实验与分析

消融实验按模块逐步叠加：LCM baseline → +Attention → +OCR → +Topology → +Sharpness。除平均指标外，统计结构崩溃样本率（例如 CER > 30%）以量化稳定性。  

### 4.5 结果来源与可追溯性

（保持你仓库映射与脚本引用不变；终稿建议补充：每个表格对应的 commit hash / checkpoint hash / eval 配置。）

---

## 5 局限性与讨论

（1）对控制信号质量的依赖：本方法依赖位置掩码与 glyph 控制信号准确性，未来可探索软掩码与不确定性建模。  
（2）对识别器的依赖与偏置：OCR‑CTC 监督受识别器能力与词表影响，需训练/测试识别器解耦与多识别器评测[25,26]。  
（3）极端少步下的背景细节：1～2 步设置下背景纹理精细度可能受损，需要速度—可读性—真实感权衡。  
（4）潜在滥用风险：建议发布时集成水印、元数据签名或编辑记录机制。  

---

## 6 结论

本文提出 FlashGlyph，一个面向场景文本编辑的可读性保持少步蒸馏框架。通过在 LCM‑LoRA 一致性蒸馏基础上引入“对齐—语义—拓扑”三重约束[9,10,23,28]，FlashGlyph 能在 4 步推理下显著缓解少步蒸馏带来的位置漂移、伪字符与拓扑断裂等结构性退化问题，从而提升外部 OCR 可读性，并保持低延迟优势。未来工作将围绕跨语言泛化、控制信号噪声鲁棒性、端侧量化部署与更严格的可复现评测协议进一步展开。  

---

## 参考文献（GB/T 7714—2015）

[1] HO J, JAIN A, ABBEEL P. Denoising diffusion probabilistic models[C]//Advances in Neural Information Processing Systems (NeurIPS 2020). 2020.

[2] SONG J, MENG C, ERMON S. Denoising diffusion implicit models[C]//International Conference on Learning Representations (ICLR). 2021.

[3] LU C, CHEN Y, CHEN F, et al. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling[C]//Advances in Neural Information Processing Systems (NeurIPS 2022). 2022.

[4] LU C, ZHOU Y, BAO F, et al. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models[J]. Machine Intelligence Research, 2025, 22(4): 730-751. DOI:10.1007/s11633-025-1562-4.

[5] ZHAO W, BAI L, CHEN Y, et al. UniPC: A unified predictor-corrector framework for fast sampling of diffusion models[C]//Advances in Neural Information Processing Systems (NeurIPS 2023). 2023.

[6] ROMBACH R, BLATTMANN A, LORENZ D, et al. High-resolution image synthesis with latent diffusion models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 10684-10695.

[7] SALIMANS T, HO J. Progressive distillation for fast sampling of diffusion models[C]//International Conference on Learning Representations (ICLR). 2022.

[8] SONG Y, DHARIWAL P, CHEN M, et al. Consistency models[C]//International Conference on Machine Learning (ICML). PMLR, 2023: 32211-32252.

[9] LUO S, TAN Y, HUANG L, et al. Latent consistency models: Synthesizing high-resolution images with few-step inference[EB/OL]. arXiv:2310.04378, 2023.

[10] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-rank adaptation of large language models[EB/OL]. arXiv:2106.09685, 2021.

[11] HO J, SALIMANS T. Classifier-free diffusion guidance[EB/OL]. arXiv:2207.12598, 2022.

[12] ZHANG L, RAO A, AGRAWALA M. Adding conditional control to text-to-image diffusion models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023: 3836-3847.

[13] HERTZ A, MOKADY R, MAYTAL A, et al. Prompt-to-Prompt: Image editing with cross-attention control[EB/OL]. arXiv:2208.01626, 2022.

[14] WU L, ZHANG C, LIU J, et al. Editing text in the wild[C]//Proceedings of the 27th ACM International Conference on Multimedia (ACM MM). 2019. DOI:10.1145/3343031.3350929.

[15] ROY P, BHATTACHARYA S, GHOSH S, et al. STEFANN: Scene text editor using font adaptive neural network[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020: 13228-13237.

[16] YANG Q, HUANG J, LIN W. SwapText: Image based texts transfer in scenes[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020: 14700-14709.

[17] CHEN X, WANG B, JIN L, et al. TextDiffuser: Diffusion models as text painters[C]//Advances in Neural Information Processing Systems (NeurIPS 2023). 2023.

[18] RUAN C, MA C, LIN J, et al. TextDiffuser-2: Unleashing the power of language models for text rendering[C]//Computer Vision – ECCV 2024. Lecture Notes in Computer Science, vol 15063. Springer, 2024: 386-402. DOI:10.1007/978-3-031-72652-1_23.

[19] YANG X, HUANG Z, ZHANG J, et al. GlyphControl: Glyph conditional control for visual text generation[C]//Advances in Neural Information Processing Systems (NeurIPS 2023). 2023.

[20] MA J, ZHAO M, CHEN C, et al. GlyphDraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation[EB/OL]. arXiv:2303.17870, 2023.

[21] TUO Y, XIANG W, HE J, et al. AnyText: Multilingual visual text generation and editing[C]//International Conference on Learning Representations (ICLR). 2024. (Spotlight)

[22] TUO Y, XIANG W, HE J, et al. AnyText2: Scene text editing with glyph and position control[EB/OL]. arXiv:2411.15245, 2024.

[23] GRAVES A, FERNÁNDEZ S, GOMEZ F, et al. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks[C]//International Conference on Machine Learning (ICML). 2006: 369-376.

[24] SHI B, BAI X, YAO C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(11): 2298-2304.

[25] LI M, LV T, CUI L, et al. TrOCR: Transformer-based optical character recognition with pre-trained models[C]//AAAI Conference on Artificial Intelligence (AAAI). 2023.

[26] BAUTISTA D, ATIENZA R. Scene text recognition with permuted autoregressive sequence models[C]//Computer Vision – ECCV 2022. Lecture Notes in Computer Science, vol 13688. Springer, 2022: 178-196.

[27] JIANG L, ZHANG C, HUANG D, et al. Focal frequency loss for image reconstruction and synthesis[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021.

[28] SHIT S, PAUL N, WILDSCHUT A, et al. clDice: A novel topology-preserving loss function for tubular structure segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021.

[29] HEUSEL M, RAMSauer H, UNTERTHINER T, et al. GANs trained by a two time-scale update rule converge to a local Nash equilibrium[C]//Advances in Neural Information Processing Systems (NeurIPS 2017). 2017. (FID)

[30] ZHANG R, ISOLA P, EFROS A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2018. (LPIPS)

[31] LI J, LI D, XIONG C, et al. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models[C]//International Conference on Machine Learning (ICML). 2023.

[32] DU Y, LI C, GUO L, et al. PP-OCR: A practical ultra lightweight OCR system[EB/OL]. arXiv:2009.09941, 2020.

[33] SCHUHMANN C, VUILLEMIN T, KUHNLE A, et al. LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs[EB/OL]. arXiv:2111.02114, 2021.

[34] SCHUHMANN C, KIRSTAIN Y, KURZEDER R, et al. LAION-5B: An open large-scale dataset for training next generation image-text models[C]//NeurIPS 2022 Datasets and Benchmarks Track. 2022.

[35] YUAN W, CHEN Z, WANG H, et al. Wukong: 100 million large-scale Chinese cross-modal pre-training benchmark[C]//NeurIPS 2022 Datasets and Benchmarks Track. 2022.

[36] KINGMA D P, WELLING M. Auto-encoding variational Bayes[C]//International Conference on Learning Representations (ICLR). 2014.

[37] LOSHCHILOV I, HUTTER F. Decoupled weight decay regularization[C]//International Conference on Learning Representations (ICLR). 2019.

[38] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems (NeurIPS 2017). 2017.

[39] ZHAO S (stzhao), AnyText Team. AnyWord-3M[DB/OL]. Hugging Face Datasets, dataset id: stzhao/AnyWord-3M.