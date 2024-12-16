from torchvision.datasets import VisionDataset
import torch
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype
import torch.nn as nn


class SegmentationDataset(VisionDataset):
    def __init__(self, *args, annotations, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotations = annotations
        self.samples = next(os.walk(self.root))[2]

    def __getitem__(self, idx):
        x = self.samples[idx]  # filename
        y = self.annotations[x]["segmentation"]
        return nn.functional.interpolate(convert_image_dtype(read_image(os.path.join(self.root, x), ImageReadMode.GRAY), dtype=torch.float32).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0), self.target_transform(y)

    def __len__(self):
        return len(self.samples)
