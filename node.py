import sys
import os
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

def show_anns(anns, image_shape):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((image_shape[0], image_shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # 将带有标注的numpy图像转化为torch张量
    annotated_img_tensor = torch.from_numpy(img)

    return annotated_img_tensor

class AutomaticMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', ),
                "image": ("IMAGE", ),
                "points_per_side": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "pred_iou_thresh": ("FLOAT", {
                    "default": 0.86,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "stability_score_thresh": ("FLOAT", {
                    "default": 0.92,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "crop_n_layers": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "crop_n_points_downscale_factor": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "min_mask_region_area": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "MK/segment_anything"
    RETURN_TYPES = ("IMAGE",)

    def main(self, sam_model, image, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
        )

        image_res = []
        for item in image:
            image_shape = (item.shape[0], item.shape[1])
            print(image_shape)
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            image_np = np.array(item)
            image_np_rgb = image_np[..., :3]


            # 生成蒙版
            masks = mask_generator.generate(image_np_rgb)
            annotated_image_tensor = show_anns(masks, image_shape)

            image_res.append(annotated_image_tensor)

        return (image_res,)