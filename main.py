import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from SegmentationDataset import SegmentationDataset

import torchvision.transforms.functional as F
import cv2

import matplotlib.pyplot as plt


def mask_to_tensor(sparse_list, final_size=(640, 640)):
    mask = np.zeros(final_size)
    indices = np.asarray(sparse_list, np.int32)
    indices = indices.reshape((int(indices.size/2), 2))

    cv2.fillPoly(mask, [indices], 1)
    return torch.tensor(mask)


def show_img(tensor):  # grayscale
    plt.imshow(np.asarray(F.to_pil_image(tensor.detach())))


df = pd.read_json("_annotation.json")
ds = SegmentationDataset("data", annotations=df,
                         target_transform=mask_to_tensor)

dl = DataLoader(ds, 10)

x, y = next(iter(dl))

plt.subplot(2, 1, 1)
show_img(x[0])
plt.subplot(2, 1, 2)
show_img(y[0])
plt.show()
