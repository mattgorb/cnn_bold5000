from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py

class Bold5000(Dataset):

    def __init__(self, fmri_dir,imagedir, transformm):
        self.data={}

        with h5py.File(fmri_dir, 'r') as f:
            keys = list(f.keys())
            print(keys)
            for i in keys:
                self.data[i] = list(f[i])

        self.root_dir = fmri_dir
        self.imagedir=imagedir
        self.transform = transform

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, 'stimulus' + str(idx + 1) + str('.tif'))
        image = io.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)),transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

filename = './ROIs/CSI1/h5/CSI1_ROIs_TR1.h5'
image_folder='./BOLD5000_Stimuli/Scene_Stimuli/Original_Images/ImageNet'

hmit = Bold5000(fmri_dir=filename,
            imagedir=image_folder,
            transform=to_tensor)
hmit_dl = DataLoader(hmit, batch_size=2, shuffle=True)




