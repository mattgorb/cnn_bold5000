import imageio
import numpy as np
import torch
from torch.nn import functional as F

class Trainer():
    def __init__(self, model, optimizer,loss,data,weight_file,with_brain_data=False ,print_loss_every=5, epochs=100,
                 use_cuda=False):

        self.model = model
        self.optimizer = optimizer
        self.loss=loss

        self.print_loss_every = print_loss_every
        self.epochs = epochs
        self.use_cuda = use_cuda

        self.accuracy=0
        self.with_brain_data=-with_brain_data
        self.weight_file=weight_file

        self.train_main=data['train_main']
        self.test_main=data['test_main']
        if data['brain_data']:
            self.brain_data=data['brain_data']


        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.losses = {'loss': []}

    def train(self):
        self.batch_size = self.train_main.batch_size
        self.model.train()

        for epoch in range(self.epochs):
            mean_epoch_loss = self.train_epoch()
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,
                                                          self.batch_size *  mean_epoch_loss))
            self.test_epoch()



    def train_epoch(self):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_main):

            if self.use_cuda:
                data = data.cuda()

            self.optimizer.zero_grad()

            output = self.model(data)

            if  self.with_brain_data:
                print('here')

            loss = self.loss(output, target)
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
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(self.train_main.dataset),mean_loss))
                print_every_loss = 0.
        # Return mean epoch loss
        return epoch_loss / len(self.train_main.dataset)


    def test_epoch(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_main):

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        if correct/total>self.accuracy:
            self.accuracy=(correct/total)
            torch.save(self.model.state_dict(),self.weight_file)
