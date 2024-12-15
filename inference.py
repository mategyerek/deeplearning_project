from main import show_img_grid, mask_to_tensor
from UNet import UNet
from SegmentationDataset import SegmentationDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split


df = pd.read_json("_annotation.json")
ds = SegmentationDataset("data", annotations=df,
target_transform=mask_to_tensor)
trainset, valset, testset = random_split(ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(trainset, 1)
val_loader = DataLoader(valset, 1)
test_loader = DataLoader(testset, 1)
x, y = next(iter(train_loader))


unet_model = UNet(1, 1)
unet_model.load_state_dict(torch.load("last.pt"))
unet_model.eval()
show_img_grid([x[0], unet_model(x)[0], y[0]])