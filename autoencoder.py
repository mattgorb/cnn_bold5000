import torch
import audtorch
import torch.nn as nn
#import torch.nn.functional as F
from torch.nn import functional as F
import torch
import audtorch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloaders.bold5000_wordnet_mods import *
from torch import optim

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 200))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


batch_size=25
get_bold5000_dataset = get_bold5000_dataset(batch_size)
model=Autoencoder()
optimizer=optim.Adam(model.parameters())
weight_file='model_weights/ae.pth'
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.1)
best=1e7

def loss_fmri( fmri_out1, fmri_target):
    c=nn.MSELoss()
    loss=c(fmri_out1, fmri_target)

    #loss = F.binary_cross_entropy(fmri_out1, fmri_target, reduction='sum')
    return loss

def train():
    for epoch in range(25000):
        mean_epoch_loss = train_epoch(epoch)
        print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1, batch_size * mean_epoch_loss))
        #self.test_epoch(epoch)
        #scheduler.step()

def train_epoch( epoch):
    global best
    epoch_loss = 0.
    model.train()
    for batch_idx in range(int(len(get_bold5000_dataset.imagenet_idxs)/batch_size)):

        fmri_data, fmri_target = get_bold5000_dataset.get_batch()
        #if self.use_cuda:
            #fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()

        fmri_target = fmri_target.type(torch.float32)
        fmri_out1=model(fmri_target)


        loss = loss_fmri( fmri_out1, fmri_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        epoch_loss += train_loss

    #if epoch_loss<best:
        #print('best: '+str(epoch_loss))
        #best=epoch_loss
    torch.save(model.state_dict(), weight_file)


    # Return mean epoch loss
    return epoch_loss / len(get_bold5000_dataset.imagenet_idxs)

def test():
    epoch_loss = 0.
    model.eval()

    network_state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(network_state_dict)
    fmri_data, fmri_target = get_bold5000_dataset.get_batch()
    fmri_target = fmri_target.type(torch.float32)
    fmri_out1 = model(fmri_target)

    print(fmri_out1[0][:10])
    print(fmri_target[0][:10])






#train()
test()