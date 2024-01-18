from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose(
            [tv.transforms.ToPILImage(),  # Use only if the input image is not a PIL image
             tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=[train_mean], std=[train_std])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        sample = self.data.iloc[index]
        relative_image_path = sample["filename"]
        crack_label = sample["crack"]
        inactive_label = sample["inactive"]

        # Open image.
        script_directory = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_directory, relative_image_path)

        # Transform image from gray scale to RGB.
        image = imread(image_path)
        image = gray2rgb(image)

        # Perform transformations on the image.
        image = self.transform(image)

        # Stack labels into torch.tensor.
        labels = torch.tensor([crack_label, inactive_label])

        return image, labels

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transformations):
        self._transform = transformations
