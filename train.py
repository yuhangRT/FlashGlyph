'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil


USING_DLC = False
NUM_NODES = 1
# Configs
ckpt_path = None  # if not None, continue training task, will not load "resume_path"
resume_path = './models/anytext2_sd15_scratch.ckpt'  # finetune from scratch, run tool_add_anytext.py to get this ckpt
config_path = './models_yaml/anytext2_sd15.yaml'
grad_accum = 2  # default 1
batch_size = 3  # default 6
logger_freq = 1000
learning_rate = 2e-5  # default 2e-5
mask_ratio = 0  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
wm_thresh = 1.0  # perentage of skip images with watermark from training
save_ckpt_top = 3

root_dir = './models'  # path for save checkpoints
dataset_percent = 1
save_steps = None  # step frequency of saving checkpoints
save_epochs = 1  # epoch frequency of saving checkpoints
max_epochs = 15  # default 60
# font
rand_font = True
font_hint_prob = 0.8  # set 0 will disable font hint
color_prob = 1.0
font_hint_area = [0.7, 1]  # reserved area on each line of font hint
font_hint_randaug = True

assert (save_steps is None) != (save_epochs is None)


if __name__ == '__main__':
    log_img = os.path.join(root_dir, 'image_log/train')
    if os.path.exists(log_img):
        try:
            shutil.rmtree(log_img)
        except OSError:
            pass
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config_path).cpu()
    if ckpt_path is None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=True)
    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False
    model.unlockQKV = False

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=save_steps,
        every_n_epochs=save_epochs,
        save_top_k=save_ckpt_top,
        monitor="global_step",
        mode="max",
    )
    #  full dataset
    json_paths = [
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/Art/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/COCO_Text/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/icdar2017rctw/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/LSVT/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/mlt2019/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/MTWI2018/data_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/ReCTS/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/laion_word/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_1of5/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_2of5/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_3of5/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_4of5/data_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_5of5/data_v1.2b.json',
    ]
    # 200k dataset, for ablation study
    json_paths = [
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/Art/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/COCO_Text/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/icdar2017rctw/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/LSVT/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/mlt2019/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/MTWI2018/data20w_v1.2b.json',
        r'/data/vdb/yuxiang.tyx/AIGC/data/ocr_data/ReCTS/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/laion_word/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_1of5/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_2of5/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_3of5/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_4of5/data20w_v1.2b.json',
        '/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/wukong_5of5/data20w_v1.2b.json',
    ]
    if USING_DLC:
        json_paths = [i.replace('/data/vdb', '/mnt/data', 1) for i in json_paths]
    glyph_scale = model.control_model.glyph_scale
    dataset = T3DataSet(json_paths, max_lines=5, max_chars=20, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=glyph_scale,
                        percent=dataset_percent, debug=False, using_dlc=USING_DLC, wm_thresh=wm_thresh, render_glyph=True,
                        trunc_cap=128, rand_font=rand_font, font_hint_prob=font_hint_prob, font_hint_area=font_hint_area,
                        font_hint_randaug=font_hint_randaug, color_prob=color_prob)
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=-1, precision=32, max_epochs=max_epochs, num_nodes=NUM_NODES, accumulate_grad_batches=grad_accum, callbacks=[logger, checkpoint_callback], default_root_dir=root_dir, strategy='ddp')

    # Train!
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
