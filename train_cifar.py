from dataloaders.cifar10_dataloaders import *
from models.resnet import *
from trainer import *
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from dataloaders.bold5000_dataset import *
from dataloaders.cifar10_dataloaders import *

from models.resnet import *
from trainer import *
from fmri_only_trainer import *


batch_size = 50

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)

data={}
data['train_main'] = cifar10_train_loader
data['test_main'] = cifar10_test_loader

'''
weight_file='model_weights/resnet50_fmri_only_layer_4_random_False.pth'
loss_file="results/test_losses_fmri_false_layer_4.txt"
accuracy_file="results/test_losses_fmri_false_layer_4.txt"

weight_file='model_weights/resnet50_fmri_only_layer_4_random_True.pth'
loss_file="results/test_losses_fmri_true_layer_4.txt"
accuracy_file="results/test_losses_fmri_true_layer_4.txt"'''

weight_file='model_weights/resnet50_fmri_only_layer_2_random_False.pth'
loss_file="results/test_losses_fmri_false_layer_2.txt"
accuracy_file="results/test_losses_fmri_false_layer_2.txt"

weight_file='model_weights/resnet50_fmri_only_layer_2_random_True.pth'
loss_file="results/test_losses_fmri_true_layer_2.txt"
accuracy_file="results/test_losses_fmri_true_layer_2.txt"

model = resnet18()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

network_state_dict = torch.load(weight_file)
model.load_state_dict(network_state_dict)

optimizer =optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=.0005)# optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()



# Define trainer
trainer = Trainer(model, optimizer,loss,data, weight_file,
                  regularize_with_fmri_data=False,use_cuda=use_cuda,
                  alpha_factor=0,regularize_layer=None,
                  fmri_weight_file_names={'loss_file': loss_file,'accuracy_file':accuracy_file})

# Train model for 250 epochs
trainer.train()