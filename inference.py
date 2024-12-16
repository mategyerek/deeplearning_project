from utils import show_img_grid, mask_to_tensor, iou_score
from segmentation_models_pytorch import Unet
from SegmentationDataset import SegmentationDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from itertools import chain
from loss import CombinedLoss

torch.manual_seed(0)
device = torch.device('cpu')
def test(model, dataloader, loss_fn, logging=False):
    model.eval()
    iou_history = []
    loss_history = []
    for i, batch in enumerate(dataloader):
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

df = pd.read_json("_annotation.json")
ds = SegmentationDataset("data", annotations=df,
                         target_transform=mask_to_tensor)
trainset, valset, testset = random_split(
    ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))
batch = 10
train_loader = DataLoader(trainset, batch)
val_loader = DataLoader(valset, batch)
test_loader = DataLoader(testset, batch)
x, y = next(iter(test_loader))


unet_model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation="sigmoid",
    ).to(device)
unet_model.load_state_dict(torch.load("lr0.001 wd0 alpha1 epochs15.pt"))
unet_model.eval()
print(test(unet_model, test_loader, CombinedLoss()))
show_img_grid(list(chain.from_iterable([[x[i], unet_model(x)[i], y[i]] for i in range(4, batch)])), 3)
