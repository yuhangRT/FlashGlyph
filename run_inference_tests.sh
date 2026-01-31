#!/bin/bash
# LCM AnyText2 推理测试脚本
# 测试三个检查点，每个检查点分别测试 CFG=7.5 和 CFG=1.0

# 遇到错误继续执行，不立即退出

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/zyh/AnyText2:$PYTHONPATH

echo "==================================="
echo "LCM AnyText2 推理测试"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "测试数据集: demodataset/annotations/demo_data.json"
echo "==================================="

CHECKPOINTS=(
  "student_model_v3/checkpoints/train_A0_maskLCM/checkpoint-final"
  "student_model_v3/checkpoints/train_A1_maskLCM_x0/checkpoint-final"
  "student_model_v3/checkpoints/train_A2_maskLCM_x0_fflgrad/checkpoint-final"
)

CFG_VALUES=(7.5 1.0)
PYTHON_BIN="/home/zyh/anaconda3/envs/anytext2/bin/python3"

for CKPT in "${CHECKPOINTS[@]}"; do
  NAME=$(basename $(dirname $CKPT))
  echo ""
  echo "-----------------------------------"
  echo "Testing: $NAME"
  echo "Checkpoint: $CKPT"
  echo "-----------------------------------"

  for CFG in "${CFG_VALUES[@]}"; do
    echo "  Running with CFG scale: $CFG"

    if [ "$CFG" = "7.5" ]; then
      USE_CFG="--use_cfg"
    else
      USE_CFG=""
    fi

    $PYTHON_BIN student_model_v3/infer_lcm_anytext_v3.py \
      --config models_yaml/anytext2_sd15.yaml \
      --teacher_ckpt models/anytext_v2.0.ckpt \
      --student_lora_path $CKPT \
      --dataset_json demodataset/annotations/demo_data.json \
      --cfg_scale $CFG \
      $USE_CFG \
      --num_inference_steps 4 \
      --max_samples 4 \
      --output ${CKPT%/*}/test_cfg${CFG}.png

    if [ $? -eq 0 ]; then
      echo "  ✓ Saved: ${CKPT%/*}/test_cfg${CFG}.png"
    else
      echo "  ✗ Failed: ${CKPT%/*}/test_cfg${CFG}.png"
    fi
  done
done

echo ""
echo "==================================="
echo "所有测试完成!"
echo "==================================="
echo ""
echo "生成的图像文件:"
ls -lh student_model_v3/checkpoints/train_A*/test_cfg*.png 2>/dev/null || echo "未找到输出文件"
