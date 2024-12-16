import numpy as np
import math

import torch.nn as nn
import torch
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import cv2


def mask_to_tensor(sparse_list, final_size=(256, 256)):
    mask = np.zeros((640, 640))  # original size
    indices = np.asarray(sparse_list, np.int32)
    indices = indices.reshape((int(indices.size/2), 2))

    cv2.fillPoly(mask, [indices], 1)
    return nn.functional.interpolate(torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=final_size, mode='bilinear', align_corners=False).squeeze(0)


def iou(y1, y2):
    return 2 * (y1 * y2).sum() / (y1 + y2).sum()


def show_img(tensor):  # grayscale
    plt.imshow(np.asarray(F.to_pil_image(tensor.detach())))


def show_img_grid(l, nx=4, ny=None):
    n = len(l)

    if ny == None:
        ny = math.ceil(n/nx)
    elif ny * nx < n:
        raise ValueError("Grid is too small. ny*nx must be larger than len(l)")
    for idx, tensor in enumerate(l):
        plt.subplot(ny, nx, idx + 1)
        plt.axis('off')
        show_img(tensor)
    plt.show()
