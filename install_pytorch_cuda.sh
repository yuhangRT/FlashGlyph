#!/bin/bash
# PyTorch CUDA 11.8 安装脚本
# 适用于CUDA 12.6环境（向下兼容）

echo "============================================================"
echo "PyTorch CUDA版本安装"
echo "============================================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate anytext2

echo ""
echo "当前环境信息:"
echo "  PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA驱动: 13.0"
echo "  CUDA Toolkit: 12.6"
echo ""

echo "============================================================"
echo "卸载当前CPU版本PyTorch"
echo "============================================================"

pip uninstall -y torch torchvision

echo ""
echo "============================================================"
echo "安装PyTorch with CUDA 11.8"
echo "============================================================"

# 安装PyTorch 2.1.0 with CUDA 11.8
# 使用pip安装，兼容CUDA 12.6
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "============================================================"
echo "验证安装"
echo "============================================================"

python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠ CUDA仍不可用，可能需要重启终端'
"

echo ""
echo "============================================================"
echo "完成！"
echo "============================================================"
