import pandas as pd

from tqdm import tqdm
import gc

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as tf

from SegmentationDataset import SegmentationDataset
# from UNet import UNet_small
from loss import CombinedLoss
from utils import mask_to_tensor, iou_score
import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet





def train(model, dataloader, optimizer, loss_fn, logging=False):
    model.train()
    iou_history = []
    loss_history = []
    for i, batch in enumerate(tqdm(dataloader)):
        x, y = batch[0].to(device=device), batch[1].to(device=device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        iou_ = iou_score(outputs, y)
        loss_history.append(loss)
        iou_history.append(iou_)
        loss.backward()
        optimizer.step()
        if i % 10 == 0 and logging:
            print(i, loss.item().detach())

    return torch.mean(torch.tensor(loss_history)), torch.mean(torch.tensor(iou_history))


def validate(model, dataloader, loss_fn, logging=False):
    model.eval()
    iou_history = []
    loss_history = []
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x, y = batch[0].to(device=device), batch[1].to(device=device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            iou_ = iou_score(outputs, y)
            loss_history.append(loss)
            iou_history.append(iou_)
            if i % 10 == 0 and logging:
                print(i, loss.item().detach())

    return torch.mean(torch.tensor(loss_history)), torch.mean(torch.tensor(iou_history))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
if __name__ == '__main__':
    SEED = 42
    BATCH = 64
    EPOCHS = 10
    
    load = False
    lr = 0.001
    wd = 0
    alpha = 1
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
    
    # lr in [0.01, 0.001, 0.0001]
    # wd in [0.001, 0.0001, 0.00001]
    # alpha in [0.5, 0.75, 1, 1.25, 1.5]
    for nothing in range(1):
        loss_fn = CombinedLoss(alpha=alpha)
        unet_model = Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation="sigmoid",
        ).to(device, dtype=torch.float32)


        if load:
            unet_model.load_state_dict(torch.load("512lr0.001 wd0 alpha1 epochs15.pt"))

        optimizer = optim.Adam(unet_model.parameters(), lr=lr, weight_decay=wd)
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
            torch.save(unet_model.state_dict(), f'512lr{lr} wd{wd} alpha{alpha} epochs{EPOCHS}.pt')
            torch.save(unet_model.state_dict(), 'last.pt')
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
        plt.figure()
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
        plt.savefig(f'512lr{lr} wd{wd} alpha{alpha} epochs{EPOCHS}.png')
