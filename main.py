from torch import optim

from dataloaders.cifar10_dataloaders import *
from fmri_only_trainer import FMRIOnlyTrainer

from models.resnet import *
from models.vgg import *

#from old_files.trainer import *
#from old_files.fmri_only_trainer import *
from fmri_direct_trainer import FMRIDirectTrainer

batch_size = 10
data={}

only_fmri=True

if only_fmri:
    '''
    this trains only on FMRI data, not using CIFAR10 data at all.  Want to apply this to transfer learning
    '''

    #from dataloaders.bold5000_traintest import *
    #from dataloaders.bold5000_dataset import *
    from dataloaders.bold5000_wordnet_mods import *
    get_bold5000_dataset = get_bold5000_dataset(batch_size)
    data['fmri_data'] = get_bold5000_dataset

    # regularize_layer={1,2,3,4,fc1,fc2}
    regularize_layer = 'fc1'

    random=False

    model = resnet18(regularize_layer=regularize_layer)
    model=vgg16(num_classes=200)
    model=models.vgg16(pretrained=True, num_classes=8)

    weight_file = 'model_weights/pca.pth'

    # Check for cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    optimizer=optim.Adam(model.parameters(), lr=1e-3)#optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=.0005)#optim.Adam(model.parameters())#
    loss = nn.CrossEntropyLoss()
    #F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # Define trainer
    '''trainer = FMRIOnlyTrainer(model, optimizer,loss,data, weight_file,
                      use_cuda=use_cuda,
                      regularize_layer=regularize_layer,
                      random=random)'''

    trainer = FMRIDirectTrainer(model, optimizer, loss, data, weight_file,
                              use_cuda=use_cuda,
                              regularize_layer=regularize_layer,
                              random=random)

else:
    '''
    regularize CIFAR10 model with FMRI data. trying to replicate similar to
      https://papers.nips.cc/paper/9149-learning-from-brains-how-to-regularize-machines
    '''
    from old_files.bold5000_dataset import *
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
        alpha = 20
        #regularize_layer={1,2,3,4}
        regularize_layer = 'fc1'

        model = resnet18(regularize_layer=regularize_layer)
        weight_file='model_weights/cifar10_resnet50_fmri_layer'+str(regularize_layer)+'_alpha'+str(alpha)+'.pth'
    else:
        #train normally
        weight_file='model_weights/cifar10_resnet50.pth'
        model = resnet18()

        #remove
        get_bold5000_dataset = get_bold5000_dataset(batch_size)
        data['fmri_data'] = get_bold5000_dataset
        regularize_layer ='fc1'
        model = resnet18(regularize_layer=regularize_layer)

        alpha=0


    # Check for cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()


    optimizer =optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=.0005)# optim.Adam(model.parameters())
    #loss = nn.CrossEntropyLoss()

    # Define trainer
    '''trainer = Trainer(model, optimizer,loss,data, weight_file,
                      regularize_with_fmri_data=regularize_with_fmri_data,use_cuda=use_cuda,
                      alpha_factor=alpha,regularize_layer=regularize_layer)'''

# Train model for 250 epochs
trainer.train()

