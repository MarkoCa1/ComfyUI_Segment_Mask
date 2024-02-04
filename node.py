import sys
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import folder_paths

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
                "image": ("IMAGE",),
            },
        }

    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "segment_anything"
    RETURN_TYPES = ("IMAGE",)

    def main(self, image):
        sam_checkpoint = folder_paths.get_full_path('sams', 'sam_vit_h_4b8939.pth')
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)

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