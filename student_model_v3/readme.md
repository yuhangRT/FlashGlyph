### 用yaml跑v3的命令
单卡  
python3 student_model_v3/launch_single_gpu.py --config student_model_v3/configs/lcm_v3.yaml --gpu 0

### 内存保护
python3 student_model_v3/launch_single_gpu.py --config student_model_v3/configs/lcm_v3.yaml --gpu 0 --min-available-gb 4


### 续训
在yaml文件的model中，添加 resume_path

# 利用student模型推理

python3 student_model_v3/infer_lcm_anytext_v3.py \
  --student_lora_path student_model_v3/checkpoints/<l2_run>/checkpoint-final \
  --output student_model_v3/preview_l2.png

python3 student_model_v3/infer_lcm_anytext_v3.py \
  --student_lora_path student_model_v3/checkpoints/<gl_run>/checkpoint-final \
  --output student_model_v3/preview_gl.png
