import h5py
import os
from skimage import io
from torchvision import transforms, models
import torch

filename = './ROIs/CSI1/h5/CSI1_ROIs_TR1.h5'

image_folder='./BOLD5000_Stimuli/Scene_Stimuli/Original_Images/ImageNet'

data={}

with h5py.File(filename, 'r') as f:
    keys = list(f.keys())
    print(keys)
    for i in keys:
        data[i] = list(f[i])


data=data['RHPPA']


f = open('./ROIs/stim_lists/CSI01_stim_lists.txt', 'r')
CSI01_stim_lists = f.read().splitlines()
f.close()

#first image
image_path_full1=os.path.join(image_folder, CSI01_stim_lists[0])
image = io.imread(image_path_full1)
to_tensor = transforms.Compose([transforms.ToTensor()])
image1=to_tensor(image)


#second image
image_path_full1=os.path.join(image_folder, CSI01_stim_lists[1])
image = io.imread(image_path_full1)
to_tensor = transforms.Compose([transforms.ToTensor()])
image2=to_tensor(image)


alexnet = models.alexnet(pretrained=True)

alexnet.features[:2]#cnn1
alexnet.features[:5]#cnn2
alexnet.features[:8]#cnn3
alexnet.features[:10]#cnn4
alexnet.features[:12]#cnn5

out=alexnet.features(image1.unsqueeze(dim=0))
out=alexnet.avgpool(out)
out = torch.flatten(out, 1)
out=alexnet.classifier[:3](out)#fc6

out=alexnet.features(image1.unsqueeze(dim=0))
out=alexnet.avgpool(out)
out = torch.flatten(out, 1)
out=alexnet.classifier[:6](out)#fc7


