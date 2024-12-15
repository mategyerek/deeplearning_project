import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from SegmentationDataset import SegmentationDataset
from UNet import UNet
from loss import ComboLoss, SoftDiceLoss
import torch.nn as nn
import gc
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import cv2
import matplotlib.pyplot as plt



def mask_to_tensor(sparse_list, final_size=(256, 256)):
    mask = np.zeros((640, 640)) #original size
    indices = np.asarray(sparse_list, np.int32)
    indices = indices.reshape((int(indices.size/2), 2))

    cv2.fillPoly(mask, [indices], 1)
    return nn.functional.interpolate(torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=final_size, mode='bilinear', align_corners=False).squeeze(0)


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

    loss_history = []
    for i, batch in enumerate(tqdm(dataloader)):
        x, y = batch[0].to(device='cuda'), batch[1].to(device='cuda')
        optimizer.zero_grad()
        outputs = model(x)
        loss = ComboLoss()(outputs, y)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
        if i % 10 == 0 and logging:
            print(i, loss.item().detach())
        del x, y, loss, outputs
    return torch.mean(torch.tensor(loss_history))

def validate(model, dataloader, logging=False):
    model.eval()

    loss_history = []
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x, y = batch[0].to(device='cuda'), batch[1].to(device='cuda')
            outputs = model(x)
            loss = ComboLoss()(outputs, y)
            loss_history.append(loss)
            if i % 10 == 0 and logging:
                print(i, loss.item().detach())
            del x, y, loss, outputs
    return torch.mean(torch.tensor(loss_history))

if __name__ == '__main__':
    SEED = 42
    BATCH = 64
    EPOCHS = 50
    load = False
    LR = 0.001
    df = pd.read_json("_annotation.json")
    ds = SegmentationDataset("data", annotations=df,
                            target_transform=mask_to_tensor)
    
    trainset, valset, testset = random_split(ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(trainset, BATCH, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True)
    val_loader = DataLoader(valset, BATCH, num_workers=1, pin_memory=True, prefetch_factor=5)
    test_loader = DataLoader(testset, BATCH, num_workers=1, pin_memory=True, prefetch_factor=5)

    x, y = next(iter(train_loader))

    
    unet_model = UNet(1, 1).to(device="cuda", dtype=torch.float32)
    if load:
        unet_model.load_state_dict(torch.load("last.pt"))
    #unet_model.eval()
    #show_img_grid([x[1], unet_model(x)[1], y[1]])

    optimizer = optim.Adam(unet_model.parameters(), lr=LR)
    train_loss = []
    val_loss = []
    for e in range(EPOCHS):
        print("Epoch ", e)
        print("Training...")
        loss = train(unet_model, train_loader, optimizer)
        train_loss.append(loss)
        torch.save(unet_model.state_dict(), "last.pt")
        print("Training loss: ", loss)
        gc.collect()
        torch.cuda.empty_cache()
        print("Validating...")
        loss = validate(unet_model, val_loader)
        val_loss.append(loss)
        print("Validation loss: ", loss)
    plt.plot(range(EPOCHS), train_loss)
    plt.plot(range(EPOCHS), val_loss)
    plt.show()
