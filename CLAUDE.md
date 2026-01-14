# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AnyText2 is a research project for Visual Text Generation and Editing with Customizable Attributes. It's a deep learning system built on PyTorch that combines Stable Diffusion with custom text rendering capabilities to generate and edit text in natural scene images with precise control over text attributes like font, color, and position.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate anytext2

# Download model checkpoint (requires modelscope)
python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"
```

## Common Development Commands

### Demo/Inference
- **Launch Gradio web interface**: `python demo.py`
  - Requires GPU with 8GB+ memory
  - Supports both text generation and editing
  - Includes customizable text attributes (font, color, position)
- **Optional parameters**: `python demo.py [--use_fp32] [--no_translator] [--font_path PATH] [--model_path PATH]`
  - `--use_fp32`: Use fp32 precision instead of fp16
  - `--no_translator`: Disable Chinese-to-English translator (saves ~4GB VRAM)
  - `--model_path`: Load custom checkpoint (default: `models/anytext_v2.0.ckpt`)

### Model Preparation
- **Create AnyText2-ready checkpoint**: `python tool_add_anytext.py [input_model] [output_model]`
  - Merges OCR components into base Stable Diffusion model
  - Default input: SD1.5 checkpoint (download separately)
  - Default output: `./models/anytext2_sd15_scratch.ckpt`
  - Required before training or inference with custom base models

### Training
- **Start training**: `python train.py`
  - Uses PyTorch Lightning framework
  - Key parameters in train.py: `batch_size=3`, `grad_accum=2`, `learning_rate=2e-5`
  - Dataset paths are hardcoded in train.py (lines 71-85) - modify these for your data
  - Checkpoint saving: configured via `save_steps` or `save_epochs` (one must be None)
  - Resume training: set `ckpt_path` to continue from checkpoint

### Evaluation
- **OCR accuracy**: `eval/eval_ocr.sh`
  - Supports Chinese (wukong) and English (laion) evaluation datasets
  - Long caption evaluation: change `test1k.json` to `test1k_long.json`
- **CLIP scores**: `eval/eval_clip.sh`
- **FID scores**: `eval/eval_fid.sh`
- **Generate evaluation images**: `eval/gen_imgs_anytext2.sh`
- **Multi-GPU evaluation**: `python eval/anytext2_multiGPUs.py`

## Architecture Overview

### Core Components

1. **Latent Diffusion Model (`/ldm/`)**: Modified Stable Diffusion backbone
   - Attention mechanisms in `ldm/modules/attention.py`
   - Diffusion components in `ldm/modules/diffusionmodules/`
   - Text encoders in `ldm/modules/encoders/modules.py` (CLIP ViT-L + BERT)
   - Autoencoder for latent space compression in `ldm/models/autoencoder.py`

2. **ControlNet Integration (`/cldm/`)**: Text rendering control system
   - `cldm.py`: Main `ControlledUnetModel` implementation extending UNetModel
   - `cldm/cldm.py`: Contains `ControlledUnetModel` with AttnX cross-attention injection
   - `ddim_hacked.py`: Modified DDIM sampling algorithm for AnyText2
   - `model.py`: Main `AnyText2` model class with PyTorch Lightning integration
   - `embedding_manager.py`: Multi-modal text embedding management (text, font, color)
   - `recognizer.py`: OCR/text recognition components (SVTR architecture)

3. **Text Processing Pipeline**
   - `bert_tokenizer.py`: BERT-based text tokenization with custom vocab
   - Font handling in `/font/lang_font/` (multilingual support)
   - OCR weights: `./ocr_weights/ppv3_rec.pth` (PP-OCR v3 recognition model)

4. **Model Wrapper**
   - `ms_wrapper.py`: `AnyText2Model` class - high-level inference wrapper
   - Handles model loading, preprocessing, and generation pipeline
   - Manages optional translator module for Chinese→English conversion

### Key Architectural Patterns

- **WriteNet+AttnX**: Custom architecture for injecting text rendering capabilities
  - WriteNet (ControlNet branch): Generates text rendering guidance from glyph/position control maps
  - AttnX (attn1x/attn2x): Modified cross-attention layers that inject text features at precise spatial locations
  - Implemented in `cldm/cldm.py:36-100` via `ControlledUnetModel`
- **Multi-modal Conditioning**: Combines text embeddings, font hints, and color attributes
  - Separate encoding paths for different text attributes via `EmbeddingManager`
  - Allows per-line customization of font and color (up to 5 lines)
  - Font hints generated using OCR encoder (shared weights with text recognizer)
- **Glyph Control Maps**: Position maps rendered as text images to guide generation
  - Generated in dataset using PIL with random fonts
  - Scaled by `glyph_scale` parameter (default 0.7, controlled in training config)
- **Dual OCR Encoders**: Shared OCR weights for two purposes
  1. Text recognizer for training supervision (auxiliary loss)
  2. Font hint encoder for style extraction (in embedding_manager)
  - Both use `ppv3_rec.pth` initialized via `tool_add_anytext.py`
- **ControlNet**: Standard ControlNet architecture with zero convolution
  - Spatial control over text placement through glyph/position maps
  - Injected into UNet at multiple resolutions
- **Embedding Manager** (`embedding_manager.py`):
  - Manages learnable embeddings for text content, font hints, and colors
  - Font predictor: OCR encoder shared with text recognizer
  - Color encoder: Simple MLP for color attribute encoding
  - Supports both training and inference modes

### Data Flow

1. **Input Processing**:
   - Text prompt → BERT tokenizer (`bert_tokenizer.py`) → text embeddings
   - Font hints → Optional images → OCR encoder (if provided) or random sampling
   - Color attributes → RGB values → color encoder MLP
   - Position/revamping info → Control map generation (glyph images + position masks)
   - Optional translator: Chinese prompt → English (using translation model, adds ~4GB VRAM)

2. **Model Forward Pass** (training):
   - Images → VAE encoder → latents
   - Text/attributes → EmbeddingManager → conditioning embeddings
   - Control maps → ControlNet (WriteNet) → control features
   - Noisy latents + control → UNet with AttnX → predicted noise
   - OCR loss: Recognizer predicts text from generated images (auxiliary supervision)

3. **Sampling** (inference):
   - Text prompt → BERT → text embeddings
   - Control maps → WriteNet → guidance features
   - DDIM iteration with AttnX injection → denoised latents
   - VAE decoder → final image with rendered text

4. **Output**:
   - Generated images with rendered text
   - Training: OCR-based auxiliary loss for better text fidelity

## Configuration

- **Model config**: `./models_yaml/anytext2_sd15.yaml`
  - Defines architecture parameters (UNet channels, attention layers)
  - Configures embedding manager settings (dimensions, trainable parameters)
  - Controls loss function weights (reconstruction, OCR loss)
  - Specifies tokenizer and encoder paths

- **Training config**: Parameters defined in `train.py` (lines 20-43)
  - `ckpt_path`: Resume from checkpoint (None = start from scratch)
  - `resume_path`: Base checkpoint for fine-tuning (output of `tool_add_anytext.py`)
  - `batch_size`, `grad_accum`, `learning_rate`: Standard training params
  - `mask_ratio`: Inpainting/text editing ratio (0 = disable, default 0.5)
  - `font_hint_prob`: Probability of using font hints (0.8 default)
  - `color_prob`: Probability of using color labels (1.0 default)
  - `wm_thresh`: Watermark filtering threshold (1.0 = filter all)
  - `save_steps`/`save_epochs`: Checkpoint frequency (one must be None)
  - Dataset paths: Lines 71-85 (full dataset) or 87-101 (200K subset)

## Model Requirements

- Base checkpoint must be processed with `tool_add_anytext.py` before use
- OCR weights: `./ocr_weights/ppv3_rec.pth`
- Font files: `./font/lang_font/` (multiple languages supported)
- Font dictionary: `lang_font_dict.npy` - for font hint generation
- Final model location: `./models/` directory

## Dataset

AnyText2 uses the **AnyWord-3M** dataset (3M+ images, 9M+ text lines):
- **Sources**: Noah-Wukong, LAION-400M, OCR datasets (ArT, COCO-Text, RCTW, LSVT, MLT, MTWI, ReCTS)
- **Languages**: ~1.6M Chinese, ~1.39M English, ~10K other (Japanese, Korean, Arabic, Bengali, Hindi)
- **AnyText2-specific annotations**: Extract `anytext2_json_files.zip` and replace json files in dataset
- **Evaluation benchmark**: [AnyText-benchmark](https://modelscope.cn/datasets/iic/AnyText-benchmark/) with 1K images each for Chinese/English

**Dataset Structure**:
- LAION data: Split into 5 independent zip files (`laion_p[1-5].zip`) - extract all to `imgs/` folder
- JSON files: Contain text annotations, font hints, color labels, and long captions

## Development Notes

- GPU memory requirement: 8GB+ for inference, higher for training
- Supports both Chinese and English text generation
- Uses PyTorch Lightning for training infrastructure
- Gradio interface provides comprehensive demo capabilities
- Model supports both text generation and editing workflows
- Training configuration uses hardcoded paths in train.py - modify for local dataset location

### Working with the Codebase

**Adding Custom Fonts**:
1. Place font files in `font/lang_font/`
2. Update font dictionary in [demo.py](demo.py:21-38) for UI access
3. Regenerate `lang_font_dict.npy` for training compatibility (used by dataset)

**Modifying Training Configuration**:
- Edit parameters directly in [train.py](train.py:20-43)
- Dataset paths: Lines 71-85 (full dataset) or 87-101 (200K subset)
- Key training parameters:
  - `mask_ratio`: Controls inpainting/text editing ratio (0 = disable)
  - `font_hint_prob`: Probability of using font hints (0.8 default)
  - `color_prob`: Probability of using color labels (1.0 default)
  - `wm_thresh`: Watermark filtering threshold (1.0 = filter all)
- Note: Dataset paths are hardcoded and must be updated for local environment

**Model Checkpoint Files**:
- `anytext_v2.0.ckpt`: Official inference checkpoint (from ModelScope)
- `anytext2_sd15_scratch.ckpt`: Base checkpoint for training (create with `tool_add_anytext.py`)
- Checkpoint loading uses `load_state_dict()` with `location='cpu'` for safety in [train.py:57](train.py:57)

**Understanding AttnX Implementation**:
- AttnX layers are modified cross-attention layers in the UNet
- Look for `attn1x` and `attn2x` references in [cldm/cldm.py](cldm/cldm.py:36-100)
- These layers receive additional conditioning from the embedding manager
- The `attnx_scale` parameter controls the strength of this conditioning (default 1.0)

**Dataset Format** (AnyWord-3M):
Each JSON entry contains:
- `gt_text`: Ground truth text content
- `font_name`: Font used for rendering (optional)
- `text_color`: RGB color values (optional)
- `points`: Quadrilateral coordinates for text position
- `long_caption`: Extended caption for training (v2.0 addition)

**Inference Wrapper Usage**:
```python
from ms_wrapper import AnyText2Model

model = AnyText2Model(
    model_dir='./models/iic/cv_anytext2',
    use_fp16=True,
    use_translator=True,  # Chinese→English translation
    font_path='font/Arial_Unicode.ttf'
).cuda()

# Call model.process() for generation
# See demo.py:83-100 for usage examples
```

## Testing and Evaluation

The project includes comprehensive evaluation scripts for OCR accuracy, CLIP scores, and FID metrics. Evaluation scripts support both single-GPU and multi-GPU configurations with separate datasets for Chinese and English evaluation.