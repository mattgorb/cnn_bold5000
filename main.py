from torch import optim
import torch.nn as nn
from dataloaders.bold5000_dataset import *
from dataloaders.cifar10_dataloaders import *

from models.resnet import *
from trainer import *

batch_size = 50
epochs = 100

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)
get_bold5000_dataset=get_bold5000_dataset(batch_size)

data={}
data['train_main']=cifar10_train_loader
data['test_main']=cifar10_test_loader
data['brain_data']=get_bold5000_dataset

weight_file='cifar10_resnet50.pth'


model = resnet18()

if use_cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()

# Define trainer
trainer = Trainer(model, optimizer,loss,data, weight_file,use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train()
