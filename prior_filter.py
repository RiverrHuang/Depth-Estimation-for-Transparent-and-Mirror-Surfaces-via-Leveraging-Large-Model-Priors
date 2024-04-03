import cv2
import numpy as np
from tqdm import tqdm
from joblib import load
import os
def find_largest_connected_component(mask):
    # 寻找二值掩码中最大的连通域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component = np.zeros_like(mask)
    largest_component[labels == largest_label] = 1
    return largest_component
def prior_filter(rectangularity,eccentricity):
    prior_filter = load("./data_info/prior_filter_0.94.joblib")
    return prior_filter.predict(np.array([[rectangularity,eccentricity]]))
def compute_shape_regularities(mask):
    # 计算连通域的形状规则性
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = contours[0]
    # 计算连通域的外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    # 计算连通域的椭圆拟合
    ellipse = cv2.fitEllipse(contour)
    # 计算外接矩形和椭圆的规则性
    rectangularity = w / h  # 长方形的规则性为宽度和高度的比例
    _, (minor_axis, major_axis), _ = ellipse
    eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))  # 椭圆的偏心率
    return rectangularity, eccentricity



coarse_mask_root = "./masks/coarse_mask"
filtered_mask_save_root = "./masks/filtered_mask"

for scene in tqdm(sorted(os.listdir(coarse_mask_root))):
    mask_dir = f"{coarse_mask_root}/{scene}"
    filtered_mask_save_dir = f"{filtered_mask_save_root}/{scene}"
    os.makedirs(filtered_mask_save_dir, exist_ok=True)

    for name in tqdm(sorted(os.listdir(mask_dir))):

        mask = cv2.imread(os.path.join(mask_dir,name),cv2.IMREAD_GRAYSCALE)
        largest_component = find_largest_connected_component(mask)
        rectangularity, eccentricity = compute_shape_regularities(largest_component)
        if prior_filter(rectangularity,eccentricity)==0:
            mask = np.zeros_like(mask)

        cv2.imwrite(os.path.join(filtered_mask_save_dir, name), mask)
        print("Rectangularity:", rectangularity)
        print("Eccentricity:", eccentricity)
