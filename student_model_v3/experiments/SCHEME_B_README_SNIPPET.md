## ✅ Scheme B (paper-aligned): Triple-constraint FlashGlyph is the *mainline*

In this repo, the paper mainline should treat the **triple constraints** as the core contribution:

- **Alignment**: attention-mass distillation (teacher ↔ student)  
- **Semantics**: OCR-CTC supervision (train OCR frozen; test OCR decoupled)  
- **Topology**: soft-skeleton + clDice (stroke connectivity)

### Main training config (paper mainline)

- **`student_model_v3/configs/lcm_v3.yaml`**  
  - enables: `loss_attn_weight`, `loss_ocr_weight`, `loss_cldice_weight`  
  - disables (default): `loss_ffl_weight`, `loss_grad_weight` (sharpness polishing moved to ablation/optional)

### Optional sharpness polishing (ablation / appendix)

- **`student_model_v3/configs/lcm_v3_gl.yaml`**  
  Frequency/gradient sharpening (FFL + Grad). Use this as an optional ablation or “polish” stage.

### Reproducible table numbers (paper provenance)

See: `student_model_v3/experiments/TABLE_SOURCES.md`

### External OCR evaluation scripts (PARSeq + TrOCR)

- `eval/eval_ocr.py` (driver)
- `eval/eval_parseq.py` (PARSeq backend)
- `eval/eval_trocr.py` (TrOCR backend)

Example:

```bash
python eval/eval_ocr.py \
  --img_dir <generated_images_dir> \
  --input_json <test1k.json> \
  --backend parseq+trocr \
  --num_samples 4 \
  --out_json student_model_v3/experiments/results/ocr_report.json
```
