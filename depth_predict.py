import glob

import torch
import os
CODE_SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction

import cv2
from tqdm import tqdm
import numpy as np
import time

cfg_large = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
model_large = get_configured_monodepth_model(cfg_large, )
model_large, _, _, _ = load_ckpt('./weight/metric_depth_vit_large_800k.pth', model_large, strict_match=False)
model_large.eval()

device = "cuda"

model = model_large.to(device)
cfg = cfg_large

img_dir= './inputs/rgb_inpainting'
vis_save_dir = "./outputs/vis_test_results"
npy_save_dir = './outputs/npy_test_results'
os.makedirs(vis_save_dir, exist_ok=True)
os.makedirs(npy_save_dir, exist_ok=True)
img_list = glob.glob("./inputs/rgb_inpainting/*/*.png")
total_num = len(img_list)
print(f'image directory : {img_dir}')
start = time.time()
for scene in tqdm(sorted(os.listdir(img_dir))):
    scene_dir = os.path.join(img_dir, scene)
    print(f'\nprocessing scene {scene}, {len(os.listdir(scene_dir))} images in total')
    for img_name in tqdm(sorted(os.listdir(scene_dir))):
        img_path = os.path.join(scene_dir, img_name)
        cv_image = cv2.imread(img_path)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        intrinsic = [1000.0, 1000.0, img.shape[1] / 2, img.shape[0] / 2]
        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic,
                                                                                              cfg.data_basic)
        rgb_input = rgb_input.to(device)
        new_cam_models_stacks = []
        for cam_model in cam_models_stacks:
            new_cam_models_stacks.append(cam_model.to(device))
        cam_models_stacks = new_cam_models_stacks

        with torch.no_grad():
            pred_depth, pred_depth_scale, scale, output = get_prediction(
                model=model,
                input=rgb_input,
                cam_model=cam_models_stacks,
                pad_info=pad,
                scale_info=label_scale_factor,
                gt_depth=None,
                normalize_scale=cfg.data_basic.depth_range[1],
                ori_shape=[img.shape[0], img.shape[1]],
            )

            pred_normal = output['normal_out_list'][0][:, :3, :, :]
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]

        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_depth[pred_depth < 0] = 0

        os.makedirs(os.path.join(npy_save_dir,scene),exist_ok=True)
        os.makedirs(os.path.join(vis_save_dir, scene),exist_ok=True)
        np.save(os.path.join(npy_save_dir, scene, img_name.split('.')[0] + '.npy'), pred_depth)
        cv2.imwrite(os.path.join(vis_save_dir, scene, img_name.split('.')[0] + '.png'), pred_depth * 255)

end = time.time()
aver_time = (end-start)/total_num
print(f"average time:{aver_time}")

