import os
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

class Bold5000(Dataset):

    def __init__(self, fmri_dir,imagedir,stim_list_dir, transform):
        self.data={}

        with h5py.File(fmri_dir, 'r') as f:
            keys = list(f.keys())
            #print(keys)
            for i in keys:
                self.data[i] = list(f[i])

        self.fmri_dir = fmri_dir
        self.imagedir=imagedir
        self.transform = transform

        self.target_data = self.data['RHPPA']

        f = open(stim_list_dir, 'r')
        self.CSI01_stim_lists = f.read().splitlines()
        f.close()
        self.imagenet_idxs=[i for i, x in enumerate(self.CSI01_stim_lists) if x.startswith('n0') or x.startswith('n1')]


    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.imagedir, self.CSI01_stim_lists[idx])
        image = io.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        target=self.target_data[idx]

        return sample,torch.from_numpy(target), idx

def get_bold5000_dataset():
    to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)),transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    filename = './ROIs/CSI1/h5/CSI1_ROIs_TR1.h5'
    image_folder='./BOLD5000_Stimuli/Scene_Stimuli/Original_Images/ImageNet'
    stim_list='./ROIs/stim_lists/CSI01_stim_lists.txt'

    bold5000 = Bold5000(fmri_dir=filename,
                imagedir=image_folder,
                stim_list_dir=stim_list,
                transform=to_tensor)
    #hmit_dl = DataLoader(bold5000, batch_size=2, shuffle=True)
    return bold5000




