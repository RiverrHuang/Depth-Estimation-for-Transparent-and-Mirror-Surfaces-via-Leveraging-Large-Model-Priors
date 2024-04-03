# NTIRE 2024: HR Depth from Images of Specular and Transparent Surfaces

## Depth Estimation for Transparent and Mirror Surfaces via Leveraging Large Model Priors
Yachuan Huang (email: yachuan@hust.edu.cn), Jiaqi Li, Junrui Zhang, Yiran Wang, Zihao Huang, Tianqi Liu, Zhiguo Cao
### Test
#### To test our depth estimation, follow the instructions below
1.install requirements
```
pip install -r requirements.txt
```

2.download checkpoints at [here](https://huggingface.co/spaces/JUGGHM/Metric3D/blob/main/weight/metric_depth_vit_large_800k.pth)
and put it at weight/metric_depth_vit_large_800k.pth

3.then you can run depth_predict.py
```
python depth_predict.py
```

### Preprocess
We have already provided images after preprocessing in inputs/rgb_inpainting.
If you still want to redo this process again, follow the instructions below.

1.Download weights for GDNet at [GDNet.pth](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)

2.Download weights for SAM at [sam_vit_h_4b8939.pth](https://huggingface.co/spaces/ameerazam08/SAM_SDXL_Inpainting/blob/main/models/sam_vit_h_4b8939.pth)

3.run CVPR2020_GDNet/coarse_mask.py to get coarse mask

!!!you should edit several lines in CVPR2020_GDNet/coarse_mask.py to successfully run it.
```
ckpt_path = "path to GDNet Checkpoints"
img_dir = "path to test dataset"
save_dir = "path to save coarse mask"

```
4.run prior_fileter.py to get filtered mask

!!!you should edit several lines in prior_fileter.py to successfully run it.
```
coarse_mask_root = "path to coarse_mask"
filtered_mask_save_root = "path to save filtered mask"
```

5.run refined_mask_SAM.py to get refined mask

!!!you should edit several lines in prior_fileter.py to successfully run it.
```
mobile_sam = sam_model_registry['vit_h'](checkpoint='path to sam_vit_h_4b8939.pth').to("cuda")
img_root_dir = "path to test_mono_nogt"
img_dir = f" path to /test_mono_nogt/{scene}/camera_00"
mask_dir = f"path to /filtered_mask/{scene}/"
save_dir = "path to save refined_mask"
```

6.run inpaint_mask_region_rgb.py to get inpainted rgb
```
mask_root = "path to refined_mask"
rgb_root = "path to test_mono_nogt/"
vis_save_root = "path to rgb_inpainting/"
```