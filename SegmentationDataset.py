from torchvision.datasets import VisionDataset
import os
from torchvision.io import read_image, ImageReadMode


class SegmentationDataset(VisionDataset):
    def __init__(self, *args, annotations, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotations = annotations
        self.samples = next(os.walk(self.root))[2]

    def __getitem__(self, idx):
        x = self.samples[idx]  # filename
        y = self.annotations[x]["segmentation"]
        return read_image(os.path.join(self.root, x), ImageReadMode.GRAY), self.target_transform(y)

    def __len__(self):
        return len(self.samples)
