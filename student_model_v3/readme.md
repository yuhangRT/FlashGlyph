### 用yaml跑v3的命令
单卡  
python3 student_model_v3/launch_single_gpu.py --config student_model_v3/configs/lcm_v3.yaml --gpu 0

### 内存保护
python3 student_model_v3/launch_single_gpu.py --config student_model_v3/configs/lcm_v3.yaml --gpu 0 --min-available-gb 4
