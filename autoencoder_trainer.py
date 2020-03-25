import torch
import audtorch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

class AutoencoderTrainer():
    def __init__(self, model, optimizer, loss, data, weight_file,  print_loss_every=100, epochs=25,
                 use_cuda=False, regularize_layer=None, random=False):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma=0.1)

        self.print_loss_every = print_loss_every
        self.epochs = epochs
        self.use_cuda = use_cuda

        self.accuracy = 0

        self.weight_file = weight_file

        self.fmri_data = data['fmri_data']
        self.batch_size = self.fmri_data.batch_size

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        self.fmri_loss = []

        self.criterion = loss
        if self.use_cuda:
            self.model.cuda()

    def loss_fmri(self, fmri_out1, fmri_target,log_fmri_corr=False):
        c=nn.BCELoss()
        BCE=c(fmri_out1, fmri_target)
        #BCE=self.loss(fmri_out1, fmri_target)
        #BCE = F.binary_cross_entropy(fmri_out1, fmri_target, reduction='sum')
        #print(BCE)
        return BCE

    def train(self):
        for epoch in range(self.epochs):
            mean_epoch_loss = self.train_epoch(epoch)
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1, self.batch_size * mean_epoch_loss))
            #self.test_epoch(epoch)
            #self.scheduler.step()

    def train_epoch(self, epoch):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.train()


        for batch_idx in range(int(len(self.fmri_data.imagenet_idxs)/self.batch_size)):

            fmri_data, fmri_target = self.fmri_data.get_batch()
            if self.use_cuda:
                fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()


            fmri_out1 = self.model(fmri_data)



            loss = self.criterion( fmri_out1, fmri_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            epoch_loss += train_loss
            print_every_loss += train_loss

            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx ,int(len(self.fmri_data.imagenet_idxs)/self.batch_size), mean_loss))
                print_every_loss = 0.


        torch.save(self.model.state_dict(), self.weight_file)

        if epoch>4:
            fmri_data, fmri_target = self.fmri_data.get_batch()
            if self.use_cuda:
                fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()

            fmri_out1 = self.model(fmri_data)

            plt.clf()
            plt.imshow(fmri_out1[0].detach().permute(1, 2, 0))
            plt.savefig(str(epoch)+'.png')

            plt.clf()
            plt.imshow(fmri_data[0].detach().permute(1, 2, 0))
            plt.savefig(str(epoch+1000)+'.png')


        # Return mean epoch loss
        return epoch_loss / len(self.fmri_data.imagenet_idxs)

    def test_epoch(self, epoch):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.eval()

        fmri_data, fmri_target = self.fmri_data.get_batch()



        if self.use_cuda:
            fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()

        fmri_out1 = self.model(fmri_data)

        plt.clf()
        plt.imshow(fmri_out1[0].detach().numpy().permute(1, 2, 0))
        plt.savefig('test.png')
        plt.clf()
        plt.imshow(fmri_data[0].detach().numpy().permute(1, 2, 0))
        plt.savefig('test2.png')


