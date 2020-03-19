from torch import optim
import torch.nn as nn
from dataloaders.bold5000_dataset import *
from dataloaders.cifar10_dataloaders import *

from models.resnet import *
from trainer import *
from fmri_only_trainer import *

batch_size = 50
data={}

only_fmri=True

if only_fmri:
    '''
    this trains only on FMRI data, not using CIFAR10 data at all.  Want to apply this to transfer learning
    '''
    get_bold5000_dataset = get_bold5000_dataset(batch_size)
    data['fmri_data'] = get_bold5000_dataset

    # regularize_layer={1,2,3,4}
    regularize_layer = 2

    random=False


    model = resnet18(regularize_layer=regularize_layer)
    weight_file = 'model_weights/resnet50_fmri_only_layer_' + str(regularize_layer) + '_random_'+str(random)+'.pth'

    # Check for cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()


    optimizer =optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=.0005)# optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    # Define trainer
    trainer = FMRIOnlyTrainer(model, optimizer,loss,data, weight_file,
                      use_cuda=use_cuda,
                      regularize_layer=regularize_layer,
                              random=random)

else:
    '''
    regularize CIFAR10 model with FMRI data. trying to replicate similar to
      https://papers.nips.cc/paper/9149-learning-from-brains-how-to-regularize-machines
    '''
    # Load data
    cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)
    data['train_main']=cifar10_train_loader
    data['test_main']=cifar10_test_loader

    #Train with fmri data
    regularize_with_fmri_data=True

    if regularize_with_fmri_data:
        get_bold5000_dataset = get_bold5000_dataset(batch_size)
        data['fmri_data']=get_bold5000_dataset

        # Main Variables
        alpha = 0.2
        #regularize_layer={1,2,3,4}
        regularize_layer = 2

        model = resnet18(regularize_layer=regularize_layer)
        weight_file='model_weights/cifar10_resnet50_fmri_layer'+str(regularize_layer)+'_alpha'+str(alpha)+'.pth'
    else:
        #train normally
        weight_file='model_weights/cifar10_resnet50.pth'
        model = resnet18()



    # Check for cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()


    optimizer =optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=.0005)# optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    # Define trainer
    trainer = Trainer(model, optimizer,loss,data, weight_file,
                      regularize_with_fmri_data=regularize_with_fmri_data,use_cuda=use_cuda,
                      alpha_factor=alpha,regularize_layer=regularize_layer)

# Train model for 250 epochs
trainer.train()

