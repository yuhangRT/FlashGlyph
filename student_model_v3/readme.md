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

# 利用 TensorBoard 查看log日志

方法1：查看所有日志
```
tensorboard --logdir student_model_v3/checkpoints/logs --port 6006 --bind_all

```

方法2：查看特定训练的日志

```
tensorboard --logdir student_model_v3/checkpoints/train_20260130_072325/logs --port 6006 --bind_all

```

方法3：查看所有历史训练
```
tensorboard --logdir student_model_v3/checkpoints --port 6006 --bind_all

```