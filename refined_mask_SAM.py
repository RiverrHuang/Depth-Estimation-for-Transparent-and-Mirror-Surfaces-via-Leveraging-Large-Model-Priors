import cv2
import numpy as np
import torch
from torch import autocast

import os
import glob
import time

from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry['vit_h'](checkpoint='/data/huangyachuan/code/SAM_SDXL_Inpainting/models/sam_vit_h_4b8939.pth').to("cuda")
mobile_sam.eval()
mobile_predictor = SamPredictor(mobile_sam)
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]


def segmentation(img, sel_pix):
    # print("segmentation")
    # online show seg mask
    points = []
    labels = []
    for p, l in sel_pix:
        points.append(p)
        labels.append(l)
    mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
    with torch.no_grad():
        with autocast("cuda"):
            masks, _, _ = mobile_predictor.predict(point_coords=np.array(points), point_labels=np.array(labels),
                                                   multimask_output=False)

    output_mask = np.ones((masks.shape[1], masks.shape[2], 3)) * 255
    for i in range(3):
        output_mask[masks[0] == True, i] = 0.0

    mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        mask_all[masks[0] == True, i] = color_mask[i]
    masked_img = img / 255 * 0.3 + mask_all * 0.7
    # masked_img = img / 255 * 1.0 + mask_all * 0.7
    masked_img = masked_img * 255
    ## draw points
    for point, label in sel_pix:
        cv2.drawMarker(masked_img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    #invert mask
    output_mask = np.logical_not(output_mask)
    output_mask = output_mask.astype(np.uint8)*255
    return masked_img, output_mask


def find_largest_region(mask):
    # 寻找连通区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的连通区域
    largest_contour = max(contours, key=cv2.contourArea)

    # 创建一个与输入mask相同大小的空白图像
    largest_region_mask = np.zeros_like(mask)

    # 在空白图像上绘制最大的连通区域
    cv2.drawContours(largest_region_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return largest_region_mask
def sample_points(coarse_mask, num_sample):
    points = []
    if len(coarse_mask.shape) > 2 and coarse_mask.shape[2] == 3:
        coarse_mask = coarse_mask[:, :, 0]
    indices = np.argwhere(coarse_mask > 0)
    for _ in range(num_sample):
        random_index = np.random.choice(len(indices))
        # 返回选取的坐标，并且交换x和y轴
        points.append(tuple(indices[random_index][::-1]))
    return points

img_root_dir = "./inputs/test_mono_nogt"
img_list = glob.glob("./inputs/rgb_inpainting/*/*.png")
total_num = len(img_list)
start = time.time()
for scene in tqdm(sorted(os.listdir(img_root_dir))):
    # scene = 'Window3'
    img_dir = f"./inputs/test_mono_nogt/{scene}/camera_00"
    mask_dir = f"./masks/filtered_mask/{scene}/"
    save_dir = "./masks/refined_mask/"
    os.makedirs(f'{save_dir}/{scene}',exist_ok=True)
    for img_name in tqdm(sorted(os.listdir(img_dir))):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        coarse_mask = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        try:
            coarse_mask = find_largest_region(coarse_mask)
            points = sample_points(coarse_mask, 3)

            point_list = []
            for point in points:
                point_list.append((point, 1))
            # print(point_list)
            masked_img, output_mask = segmentation(img, point_list)
        except:
            output_mask = coarse_mask

        cv2.imwrite(f'./masks/refined_mask/{scene}/'+img_name, output_mask)

end = time.time()
aver_time = (end - start) / total_num
print(f"average time:{aver_time}")
