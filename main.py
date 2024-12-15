import pandas as pd
import numpy as np
import math

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from SegmentationDataset import SegmentationDataset
from UNet import UNet
from loss import ComboLoss

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


def train(model, dataloader, optimizer, logging=False):
    model.train()
    gpu = torch.cuda.device(0)
    loss_history = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(gpu), batch[1].to(gpu)
        optimizer.zero_grad()
        outputs = model(x)
        loss = ComboLoss()(outputs, y)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
        if i % 10 == 0 and logging:
            print(i, loss.item())
    return model, np.average(loss_history)


df = pd.read_json("_annotation.json")
ds = SegmentationDataset("data", annotations=df,
                         target_transform=mask_to_tensor)

dl = DataLoader(ds, 12)

"""x, y = next(iter(dl))

show_img_grid(x)
"""

unet_model = UNet(1, 1)
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)
train_loss = []
for e in 1:
    unet_model, loss = train(unet_model, dl, optimizer, True)
    train_loss.append(loss)
    torch.save(unet_model.state_dict(), "last.pt")
