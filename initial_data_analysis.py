import h5py
import os
from skimage import io
from torchvision import transforms, models
import torch
import audtorch

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

to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((375,375)),transforms.ToTensor()])

#load first image
image_path_full1=os.path.join(image_folder, CSI01_stim_lists[0])
image = io.imread(image_path_full1)
image1=to_tensor(image)

#load second image
image_path_full1=os.path.join(image_folder, CSI01_stim_lists[1])
image = io.imread(image_path_full1)
image2=to_tensor(image)


alexnet = models.alexnet(pretrained=True)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

#brain_roi cosine similarity, first two inputs:
brain1=torch.from_numpy(data[0])
brain2=torch.from_numpy(data[1])
brain_similarity=cos(brain1,brain2)
brain_similarity=audtorch.metrics.functional.pearsonr(brain1, brain2).squeeze(dim=0)
print(brain_similarity)

out=alexnet.features[:2](image1.unsqueeze(dim=0))#cnn1
out1=torch.flatten(out)#out.view(-1, out.detach().numpy().shape[1]* out.detach().numpy().shape[2]* out.detach().numpy().shape[3])

out=alexnet.features[:2](image2.unsqueeze(dim=0))#cnn1
out2=torch.flatten(out)#out.view(-1, out.detach().numpy().shape[1]* out.detach().numpy().shape[2]* out.detach().numpy().shape[3])

output = cos(out1, out2)

alexnet.features[:5](image1.unsqueeze(dim=0))#cnn2
alexnet.features[:8](image1.unsqueeze(dim=0))#cnn3
alexnet.features[:10](image1.unsqueeze(dim=0))#cnn4
alexnet.features[:12](image1.unsqueeze(dim=0))#cnn5

out=alexnet.features(image1.unsqueeze(dim=0))
out=alexnet.avgpool(out)
out = torch.flatten(out, 1)
out=alexnet.classifier[:3](out)#fc6

out=alexnet.features(image1.unsqueeze(dim=0))
out=alexnet.avgpool(out)
out = torch.flatten(out, 1)
out=alexnet.classifier[:6](out)#fc7




