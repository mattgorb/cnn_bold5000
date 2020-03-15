import h5py
import os
from skimage import io
from torchvision import transforms

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
CSI01_stim_lists = f.readlines()
f.close()

print(CSI01_stim_lists[1])
print(CSI01_stim_lists[2])

image_path_full1=os.path.join(image_folder, CSI01_stim_lists[0])
image = io.imread(image_path_full1)
sample = image
to_tensor = transforms.Compose([transforms.ToTensor()])

sample=to_tensor(sample)
print(sample)