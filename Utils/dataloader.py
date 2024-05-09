# data_utils.py

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import ToTensor

target_size = (256, 256)
ratio_min_dist = 0.2
range_vignette = (0.2, 0.8)


class AddVignette(object):
    def __init__(self, ratio_min_dist=0.2, range_vignette=(0.2, 0.8), random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, image):


        h, w = image.shape[1:]
        min_dist = np.array([h, w]) / 2 * self.ratio_min_dist * np.random.random()

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)

        # Apply vignette separately to each color channel
        for c in range(image.shape[0]):
            sign = 2 * (np.random.random() < 0.5) * self.random_sign - 1
            image[c] = image[c] * (1 + sign * vignette)


        return image



image_transforms = [transforms.Resize(target_size),
              AddVignette(ratio_min_dist, range_vignette),
              ToTensor()]


class GetTrainingPairs(Dataset):
    def __init__(self, root, dataset_name, transforms=image_transforms):
        self.filesA, self.filesB = self.get_file_paths(root, dataset_name)
        self.len = min(len(self.filesA), len(self.filesB))
        self.transforms = transforms

    def __getitem__(self, index):
        img_input_RGB = Image.open(self.filesA[index % self.len]).convert('RGB')
        img_target_RGB = Image.open(self.filesB[index % self.len]).convert('RGB')


        if self.transforms:
            # Apply resize transformation
            img_input_RGB = np.array(self.transforms[0](img_input_RGB))
            img_target_RGB = np.array(self.transforms[0](img_target_RGB))


            # Apply vignette transformation
            img_input_RGB = self.transforms[1](img_input_RGB)

            # Apply ToTensor transformation
            img_input_RGB = self.transforms[2](img_input_RGB)
            img_target_RGB = self.transforms[2](img_target_RGB)

        return {
            "input_rgb": img_input_RGB,
            "target_rgb": img_target_RGB,
        }

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        filesA, filesB = [], []

        if dataset_name == 'EUVP':
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))

        return filesA, filesB