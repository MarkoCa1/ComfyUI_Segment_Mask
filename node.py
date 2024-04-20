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
                "mask": ('MASK', ),
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

    FUNCTION = "main"
    CATEGORY = "MK/segment_anything"
    RETURN_TYPES = ("IMAGE","MASK","IMAGE")
    RETURN_NAMES = ("Image","Mask","Segment Image")

    def main(self, sam_model, image, mask, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
        )

        original_image = image[0].clone()
        image = image[0]
        source_mask = mask[0].to(torch.uint8)

        image_shape = (image.shape[0], image.shape[1])
        image = Image.fromarray(np.clip(255. * image.clone().cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
        image_np = np.array(image)
        image_np_rgb = image_np[..., :3]

        masks = mask_generator.generate(image_np_rgb)
        source_mask_np_array = np.array(source_mask)
        for mask_item in masks:
            segmentation = torch.from_numpy(mask_item["segmentation"])
            segmentation = segmentation.clone().to(torch.uint8)
            segmentation_np_array = np.array(segmentation)

            if source_mask_np_array.shape != segmentation_np_array.shape:
                print("The size of the mask is different and cannot be compared")
            else:
                overlap = np.sum(source_mask_np_array * segmentation_np_array) > 0
                if overlap :
                    source_mask = (source_mask | segmentation).type(torch.uint8)

        annotated_image_tensor = show_anns(masks, image_shape)

        transparent_image_tensor = mask_to_transparent(original_image, source_mask)

        return ([annotated_image_tensor],source_mask, [transparent_image_tensor])

def mask_to_transparent(original_image, source_mask):
    original_image_np = original_image.numpy()
    source_mask_np = source_mask.numpy().astype(np.uint8)
    source_mask_np = source_mask_np * 255

    height, width = original_image_np.shape[0], original_image_np.shape[1]
    transparent_image = np.zeros((height, width, 4), dtype=np.float64)

    transparent_image[..., :3] = original_image_np
    transparent_image[..., 3] = source_mask_np
    transparent_image_tensor = torch.from_numpy(transparent_image)

    return transparent_image_tensor

