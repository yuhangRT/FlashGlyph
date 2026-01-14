#!/bin/bash
# for Chinese:
python eval/anytext2_multiGPUs.py \
        --model_path models/anytext_v2.0.ckpt \
        --json_path /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json \
        --output_dir ./anytext2_wukong_generated \
        --gpus 0,1,2,3,4,5,6,7
# for English:  change json_path to .../laion_word/test1k.json
# for long caption evaluation:  change .../test1k.json to .../test1k_long.json
