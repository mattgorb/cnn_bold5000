from torch import optim
import torch.nn as nn
from dataloaders.bold5000_2 import *

from models.autoencoder import *
from autoencoder_trainer import AutoencoderTrainer
import torchvision

batch_size = 25
data={}


get_bold5000_dataset = get_bold5000_dataset(batch_size)
data['fmri_data'] = get_bold5000_dataset

# regularize_layer={1,2,3,4,fc1,fc2}
regularize_layer = 'fc1'

random=False
from resnet import *
#model = ResNet(3, 3, 1, 8, need_sigmoid=True, act_fun='LeakyReLU')
model=Autoencoder()

'''print(model)

x=torch.rand(1,3,32,32)
out=model(x)
print(out.size())
sys.exit()'''

weight_file='model_weights/autoencoder.pth'


# Check for cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

optimizer =optim.Adam(model.parameters())
loss = nn.BCELoss()

# Define trainer
trainer = AutoencoderTrainer(model, optimizer, loss, data, weight_file, use_cuda=use_cuda,regularize_layer=regularize_layer)

trainer.train()