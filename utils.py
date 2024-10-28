import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, List


def plot_mammography(path: str, mask_path: str, bboxes: List[Tuple[float, float, float, float]], resize_mask=False):
    img = (cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    clahe = cv2.createCLAHE(clipLimit=50)
    img = clahe.apply(img)
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    if resize_mask:
        mask = cv2.resize(mask, [img.shape[1], img.shape[0]])
    h, w = img.shape
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(mask, cmap='gray')
    for bbox in bboxes:
        bbox_x = int(bbox[0] * w)
        bbox_y = int(bbox[1] * h)
        bbox_w = int(bbox[2] * w)
        bbox_h = int(bbox[3] * h)
        axs[0].add_patch(patches.Rectangle(
            (bbox_x, bbox_y), bbox_w, bbox_h, fill=False, linewidth=1, color="black"))
    axs[0].axis("off")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_feature_sizes_of_efficient_net(model: nn.Module):
    modules = [
        "features.1.3.block.2.1",
        "features.2.6.block.3.1",
        "features.3.6.block.3.1",
        "features.4.9.block.3.1",
        "features.5.9.block.3.1",
        "features.6.12.block.3.1",
        "features.7.3.block.3.1",
        "features.8.1",
    ]
    features = []
    for name, module in model.named_modules():
        if name in modules:
            features.append(module.num_features)
    log_features = np.log1p(features)
    fig = plt.figure(figsize=(8, 10))
    ax = plt.gca()
    fig.suptitle("Batch Norm Sizes")
    sns.barplot(x=modules, y=features, palette="rocket_r",
                ax=ax, hue=log_features)
    ax.get_legend().remove()
    for i, feature in enumerate(features):
        ax.text(i, feature + 1, str(feature), ha='center',
                va='bottom', fontsize=10, color='black')
    ax.set_xlabel("Modules")
    ax.set_ylabel("Num Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()


def plot_ddsm():
    with open("saved/assesments.pickle", "rb") as assesment_f:
        assesment_scores: List[int] = pickle.load(assesment_f)
    with open("saved/x_sizes.pickle", "rb") as x_sizes_f:
        x_sizes: List[int] = pickle.load(x_sizes_f)
    with open("saved/y_sizes.pickle", "rb") as y_sizes_f:
        y_sizes: List[int] = pickle.load(y_sizes_f)
    fig, axs = plt.subplots(1, 3, figsize=(16, 20))
    fig.suptitle("DATASET ATTRIBUTES")
    sns.histplot(assesment_scores, ax=axs[0], palette="mako")
    sns.boxplot(assesment_scores, ax=axs[1], palette="mako")
    data = pd.DataFrame({"x_sizes": x_sizes, "y_sizes": y_sizes})
    print(data.describe())
    data_long = data.melt(var_name="type", value_name="size")
    sns.violinplot(data=data_long, x="type", y="size",
                   ax=axs[2], split=True, palette="mako", hue="type", inner="quart", scale="width", fill=False)

    axs[0].set_xlabel("Assesment scores")
    axs[0].set_ylabel("Frequency")

    axs[1].set_xlabel("")
    axs[1].set_ylabel("scores")

    axs[2].set_xlabel("Size Type")
    axs[2].set_ylabel("Size")

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)

    plt.show()
