import os
import torch
from skimage import io,color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import torch.nn as nn
import numpy as np
from torch import FloatTensor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler


def get_classes(CSI01_stim_lists):
    parent_list = []
    child_list = []
    with open("dataloaders/wordnet.is_a.txt") as f:
        for line in f:
            parent, child = line.split()
            parent_list.append(parent)
            child_list.append(child)
    f.close()

    def get_word_net_class(find):
        while True:
                idx = child_list.index(find)
                temp = parent_list[idx]
                if temp in ['n00001740','n00001930','n00002137','n00002684' ,'n00020827', 'n00024264', 'n00033020',
                            'n00003553','n00019613', 'n00020090', 'n00027807', 'n03892891', 'n06791372'
                            'n06793231' ,'n09287968' ,'n15046900']:
                    break
                else:
                    find=temp
        return find

    stim_list_idx=[]
    class_dict={}

    for i in CSI01_stim_lists:
        if i.startswith('n0') or i.startswith('n1'):
            temp=i.split('_')[0]
            wn_class=get_word_net_class(temp)
            if wn_class=='n00004258':
                class_dict[i]=0
                stim_list_idx.append(CSI01_stim_lists.index(i))
            elif wn_class=='n00021939':
                class_dict[i] = 1
                stim_list_idx.append(CSI01_stim_lists.index(i))

    return stim_list_idx, class_dict



class Bold5000(Dataset):
    def __init__(self, fmri_dir,imagedir,stim_list_dir, transform, batch_size):
        self.data={}

        with h5py.File(fmri_dir, 'r') as f:
            keys = list(f.keys())
            #print(keys)
            for i in keys:
                self.data[i] = list(f[i])

        self.fmri_dir = fmri_dir
        self.imagedir=imagedir
        self.transform = transform
        self.batch_size=batch_size


        self.brain_target_data = self.data['RHPPA']
        #self.brain_target_data = torch.tensor(self.brain_target_data)

        quantile_transformer = QuantileTransformer(random_state=0, output_distribution='normal')#,n_quantiles=92
        #quantile_transformer = PowerTransformer(method='yeo-johnson')#,n_quantiles=92
        #quantile_transformer = StandardScaler()#,n_quantiles=92
        self.brain_target_data=torch.tensor(quantile_transformer.fit_transform(self.brain_target_data))


        #self.brain_target_data=(self.brain_target_data - self.brain_target_data.min())/(self.brain_target_data.max()-self.brain_target_data.min())
        #self.brain_target_data=torch.tensor(self.brain_target_data)
        #self.brain_target_data= (self.brain_target_data - self.brain_target_data.mean()) / self.brain_target_data.std()

        #print(self.brain_target_data[0])

        '''self.norm=nn.BatchNorm1d(200, affine=False)
        self.brain_target_data=torch.tensor(self.data['RHPPA'])
        self.brain_target_data=self.brain_target_data.type(torch.float32)
        self.brain_target_data=self.norm(self.brain_target_data)'''

        f = open(stim_list_dir, 'r')
        self.CSI01_stim_lists = f.read().splitlines()
        f.close()

        #self.imagenet_idxs=[i for i, x in enumerate(self.CSI01_stim_lists) if x.startswith('n0') or x.startswith('n1')]

        self.imagenet_idxs, self.binary_class_data=get_classes(self.CSI01_stim_lists)

        self.train=self.imagenet_idxs[:int(len(self.imagenet_idxs)*.75)]
        self.test=self.imagenet_idxs[int(len(self.imagenet_idxs)*.75):]


    def __len__(self):
        return len(self.imagenet_idxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.imagedir, self.CSI01_stim_lists[idx])
        image = io.imread(img_name)

        if len(image.shape)==2:
            color.gray2rgb(image)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        target=self.brain_target_data[idx]
        return sample,torch.tensor(target), idx

    def get_random_idxs(self, test):
        if test:
            return np.random.choice(self.train, size=(self.batch_size))
        else:
            return np.random.choice(self.test, size=(self.batch_size))

    def get_batch(self,test=False):
        sample_list = []
        target_list = []
        for i in self.get_random_idxs(test):
            sample, target, _ = self[i]
            sample_list.append(sample)
            target_list.append(target)
        return torch.stack(sample_list),torch.stack(target_list)


def get_bold5000_dataset(batch_size):
    to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)),transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])

    filename = './ROIs/CSI1/h5/CSI1_ROIs_TR34.h5'
    image_folder='./BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/ImageNet'
    stim_list='./ROIs/stim_lists/CSI01_stim_lists.txt'

    bold5000 = Bold5000(fmri_dir=filename,
                imagedir=image_folder,
                stim_list_dir=stim_list,
                transform=to_tensor, batch_size=batch_size)
    return bold5000





x=get_bold5000_dataset(50)
