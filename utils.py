import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from typing import Tuple,List
def plot_mammography(path: str, mask_path: str, bboxes: List[Tuple[float, float, float, float]],resize_mask=True):
    img = (cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    clahe=cv2.createCLAHE(clipLimit=50)
    img=clahe.apply(img)
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    if resize_mask:
        mask=cv2.resize(mask,[img.shape[1],img.shape[0]])
    h, w = img.shape
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(mask, cmap='gray')
    for bbox in bboxes:
        bbox_x = int(bbox[0] * w)
        bbox_y = int(bbox[1] * h)
        bbox_w = int(bbox[2] * w)
        bbox_h = int(bbox[3] * h)
        axs[0].add_patch(patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, fill=False, linewidth=1, color="black"))
    axs[0].axis("off")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
    
