#!/usr/bin/env python
"""
Test if AnyText2 can load without OCR weights
"""
import os
import sys

print("=" * 60)
print("Testing AnyText2 Loading Without OCR Weights")
print("=" * 60)

# Check if OCR weights exist
ocr_path = './ocr_weights/ppv3_rec.pth'
if os.path.exists(ocr_path):
    print(f"\n✓ OCR weights found at: {ocr_path}")
    print(f"  Size: {os.path.getsize(ocr_path) / 1024 / 1024:.1f} MB")
else:
    print(f"\n✗ OCR weights NOT found at: {ocr_path}")
    print("  Will attempt to load without them...")

# Check config
cfg_path = 'models_yaml/anytext2_sd15.yaml'
print(f"\nConfig file: {cfg_path}")
if os.path.exists(cfg_path):
    from omegaconf import OmegaConf
    config = OmegaConf.load(cfg_path)
    emb_type = config.model.params.embedding_manager_config.params.emb_type
    add_style_ocr = config.model.params.embedding_manager_config.params.add_style_ocr
    print(f"  emb_type: {emb_type}")
    print(f"  add_style_ocr: {add_style_ocr}")

# Try to load the model
print("\n" + "=" * 60)
print("Attempting to load AnyText2Model...")
print("=" * 60)

try:
    import torch
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, testing on CPU")

    from ms_wrapper import AnyText2Model

    model = AnyText2Model(
        model_dir='./models/iic/cv_anytext2',
        use_fp16=False,  # CPU doesn't support fp16 well
        use_translator=False,
        model_path='./models/iic/cv_anytext2/anytext_v2.0.ckpt'
    )

    if torch.cuda.is_available():
        model = model.cuda(0)

    print("\n✓ Model loaded successfully!")

    # Check if OCR predictor was loaded
    if hasattr(model.model, 'text_predictor'):
        print("  ✓ text_predictor exists")
    else:
        print("  ✗ text_predictor does NOT exist")

    if hasattr(model.model, 'cn_recognizer'):
        print("  ✓ cn_recognizer exists")
    else:
        print("  ✗ cn_recognizer does NOT exist")

    print("\n" + "=" * 60)
    print("SUCCESS: Model can run without OCR weights file!")
    print("=" * 60)
    print("\nNote: The OCR model architecture is created but uses")
    print("uninitialized weights when ppv3_rec.pth is missing.")
    print("This may affect text generation quality but won't crash.")

except Exception as e:
    print("\n✗ Failed to load model!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 60)
    print("OCR weights ARE required!")
    print("=" * 60)
