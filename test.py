import pandas as pd
import numpy as np

import torch
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F
import cv2

import matplotlib.pyplot as plt
import os


def mask_to_tensor(sparse_list, final_size=(640, 640)):
    mask = np.zeros(final_size)
    indices = np.asarray(sparse_list, np.int32)
    indices = indices.reshape((int(indices.size/2), 2))

    cv2.fillPoly(mask, [indices], 1)
    return torch.tensor(mask)

    # out = torch.zeros(final_size)
    # indices = torch.asarray(sparse_list, dtype=torch.long)

    # indices = indices.reshape((2, int(indices.size()[1]/2)))
    # print(indices[0])
    # print(indices[1])
    # out[indices[0], indices[1]] = 1
    # return out


def show_img(tensor):  # grayscale
    plt.imshow(np.asarray(F.to_pil_image(tensor.detach())))


df = pd.read_json("data/_annotation.json")


img1 = read_image("data/2.jpg", ImageReadMode.GRAY)
label1 = mask_to_tensor(df["2.jpg"]["segmentation"])

plt.subplot(2, 1, 1)
show_img(label1)
print(torch.sum(label1))
plt.subplot(2, 1, 2)
show_img(img1)
plt.show()
