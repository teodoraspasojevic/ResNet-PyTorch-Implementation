from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # if torch.is_tensor(index):
        #     index = index.to_list()

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
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomApply([tv.transforms.RandomRotation((90, 90)),
                                           tv.transforms.RandomRotation((180, 180)),
                                           tv.transforms.RandomRotation((270, 270))], p=0.5),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.RandomErasing()
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        image = self._transform(image)
        # image = image.unsqueeze(0)                  # add the batch size

        # Stack labels into torch.tensor.
        labels = torch.tensor([crack_label, inactive_label])

        return image, labels

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transformations):
        self._transform = transformations

    def add_samples(self, indices):
        additional_data = self.data.iloc[indices]
        self.data = pd.concat([self.data, additional_data])

    def get_labels(self):
        labels = []
        for i in range(len(self.data)):
            sample = self.data.iloc[i]
            crack_label = sample["crack"]
            inactive_label = sample["inactive"]
            label = [crack_label, inactive_label]
            labels.append(label)

        return labels

    def oversample_unbalanced_classes(self):
        labels = self.get_labels()

        unique, counts = np.unique(labels, axis=0, return_counts=True)

        desired_freq = max(counts)

        # Oversample underrepresented combinations
        oversampled_data = []
        for label_combination in unique:
            indices = np.where((labels == label_combination).all(axis=1))[0]
            oversample_count = desired_freq - counts[unique.tolist().index(list(label_combination))]
            oversampled_data.extend(np.random.choice(indices, size=oversample_count, replace=True))

        self.add_samples(oversampled_data)
