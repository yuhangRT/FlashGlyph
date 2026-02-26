# Predicted Table Bundle

Generated from in-repo paper numbers (pre-fill only).

## Files
- `table1a_cn_predicted.csv`
- `table1b_en_predicted.csv`
- `table2_cn_predicted.csv`
- `table2_en_predicted.csv`
- `table4_cn_predicted.csv`

## Table1a CN (first 5 rows)

| method | steps | latency_ms | char_acc | fid | image_dir | ocr_json |
|---|---:|---:|---:|---:|---|---|
| AnyText2 (Teacher) | 50 | 8700 | 94.1 | 11.8 | student_model_v3/experiments/generated/teacher50_cn | student_model_v3/experiments/results/table1a_cn_teacher50_parseq_trocr.json |
| DDIM-4step | 4 | 700 | 58.3 | 52.3 | student_model_v3/experiments/generated/ddim4_cn | student_model_v3/experiments/results/table1a_cn_ddim4_parseq_trocr.json |
| DDIM-10step | 10 | 1750 | 71.2 | 34.7 | student_model_v3/experiments/generated/ddim10_cn | student_model_v3/experiments/results/table1a_cn_ddim10_parseq_trocr.json |
| DPM-Solver-10 | 10 | 780 | 73.5 | 31.2 | student_model_v3/experiments/generated/dpmsolver10_cn | student_model_v3/experiments/results/table1a_cn_dpmsolver10_parseq_trocr.json |
| DPM-Solver-15 | 15 | 1170 | 78.9 | 24.8 | student_model_v3/experiments/generated/dpmsolver15_cn | student_model_v3/experiments/results/table1a_cn_dpmsolver15_parseq_trocr.json |
