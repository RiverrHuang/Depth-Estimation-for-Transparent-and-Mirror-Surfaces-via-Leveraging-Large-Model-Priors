import os
import numpy as np
import cv2
from scipy.stats import mode
from tqdm import tqdm
import glob
import time
def most_frequent_color(image, mask):
    # 获取mask区域的像素值
    masked_pixels = image[mask > 0]
    if len(masked_pixels)==0:
        return (0,0,0)

    # 统计RGB三元组出现的次数
    color_counts = {}
    for pixel in masked_pixels:
        color_key = tuple(pixel)
        if color_key in color_counts:
            color_counts[color_key] += 1
        else:
            color_counts[color_key] = 1

    # 获取出现频率最高的颜色
    most_frequent_color = max(color_counts, key=color_counts.get)

    # print(most_frequent_color)

    return most_frequent_color


def replace_mask_color(image, mask):
    # 获取最频繁的颜色
    color = most_frequent_color(image, mask)

    # 将mask区域的颜色替换为最频繁的颜色
    image[mask > 0] = color

    return image.astype(np.uint8)
def main():
    mask_root = "masks/refined_mask"
    rgb_root = "inputs/test_mono_nogt/"
    vis_save_root = "inputs/rgb_inpainting/"
    img_list = glob.glob("./inputs/rgb_inpainting/*/*.png")
    total_num = len(img_list)
    start = time.time()
    for scene in tqdm(sorted(os.listdir(mask_root))):
        mask_dir = f"{mask_root}/{scene}"
        rgb_dir = f"{rgb_root}/{scene}/camera_00"
        vis_save_dir = f"{vis_save_root}/{scene}"
        os.makedirs(vis_save_dir, exist_ok=True)

        for rgb_name in tqdm(sorted(os.listdir(rgb_dir))):
            rgb = cv2.imread(os.path.join(rgb_dir, rgb_name))
            mask_name = rgb_name

            mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            # for channel in range(3):
            #     rgb[:,:,channel][mask>0] = mode(rgb[:,:,channel][mask>0]).mode
            rgb = replace_mask_color(rgb,mask)
            # 保存修改后的RGB
            cv2.imwrite(os.path.join(vis_save_dir, mask_name), rgb)
    end = time.time()
    aver_time = (end - start) / total_num
    print(f"average time:{aver_time}")

if __name__ == '__main__':
    main()

