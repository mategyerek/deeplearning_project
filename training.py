import pandas as pd

from tqdm import tqdm
import gc

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as tf

from SegmentationDataset import SegmentationDataset
from UNet import UNet
from loss import ComboLoss, SoftDiceLoss, MLoss
from utils import mask_to_tensor, iou
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, loss_fn, logging=False):
    model.train()
    iou_history = []
    loss_history = []
    for i, batch in enumerate(tqdm(dataloader)):
        x, y = batch[0].to(device='cuda'), batch[1].to(device='cuda')
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        iou_ = iou(outputs, y)
        loss_history.append(loss)
        iou_history.append(iou_)
        loss.backward()
        optimizer.step()
        if i % 10 == 0 and logging:
            print(i, loss.item().detach())
        del x, y, loss, outputs
    return torch.mean(torch.tensor(loss_history)), torch.mean(torch.tensor(iou_history))


def validate(model, dataloader, loss_fn, logging=False):
    model.eval()
    iou_history = []
    loss_history = []
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x, y = batch[0].to(device='cuda'), batch[1].to(device='cuda')
            outputs = model(x)
            loss = loss_fn(outputs, y)
            iou_ = iou(outputs, y)
            loss_history.append(loss)
            iou_history.append(iou_)
            if i % 10 == 0 and logging:
                print(i, loss.item().detach())
            del x, y, loss, outputs
    return torch.mean(torch.tensor(loss_history)), torch.mean(torch.tensor(iou_history))


if __name__ == '__main__':
    SEED = 42
    BATCH = 64
    EPOCHS = 150
    loss_fn = MLoss()
    load = False
    LR = 0.01
    df = pd.read_json("_annotation.json")
    ds = SegmentationDataset("data", annotations=df,
                             target_transform=mask_to_tensor)

    trainset, valset, testset = random_split(
        ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(trainset, BATCH, num_workers=1,
                              pin_memory=True, prefetch_factor=10, persistent_workers=True)
    val_loader = DataLoader(valset, BATCH, num_workers=1,
                            pin_memory=True, prefetch_factor=5)
    test_loader = DataLoader(
        testset, BATCH, num_workers=1, pin_memory=True, prefetch_factor=5)

    x, y = next(iter(train_loader))

    unet_model = UNet(1, 1).to(device="cuda", dtype=torch.float32)
    if load:
        unet_model.load_state_dict(torch.load("last.pt"))

    optimizer = optim.Adam(unet_model.parameters(), lr=LR)
    train_loss = []
    val_loss = []
    train_iou = []
    val_iou = []
    for e in range(EPOCHS):
        print("Epoch ", e)
        print("Training...")
        loss, iou_ = train(unet_model, train_loader, optimizer, loss_fn)
        train_loss.append(loss)
        train_iou.append(iou_)
        torch.save(unet_model.state_dict(), "last.pt")
        print("Training loss: ", loss)
        print("Training iou: ", iou_)
        gc.collect()
        torch.cuda.empty_cache()
        print("Validating...")
        loss, iou_ = validate(unet_model, val_loader, loss_fn)
        val_loss.append(loss)
        val_iou.append(iou_)
        print("Validation loss: ", loss)
        print("Validation iou: ", iou_)
    plt.subplot(1, 2, 1)
    plt.title("Loss against epoch")
    plt.plot(range(EPOCHS), train_loss, label="training")
    plt.plot(range(EPOCHS), val_loss, label="validation")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("IOU against epoch")
    plt.plot(range(EPOCHS), train_iou, label="training")
    plt.plot(range(EPOCHS), val_iou, label="validation")
    plt.legend()
    plt.show()
