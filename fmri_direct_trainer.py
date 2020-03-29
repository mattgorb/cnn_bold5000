import torch
import audtorch
import torch.nn as nn
#import torch.nn.functional as F
from torch.nn import functional as F

class FMRIDirectTrainer():
    def __init__(self, model, optimizer, loss, data, weight_file,  print_loss_every=100, epochs=250,
                 use_cuda=False, regularize_layer=None, random=False):

        self.model = model
        self.optimizer = optimizer
        self.loss = nn.MSELoss()


        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma=0.1)

        self.print_loss_every = print_loss_every
        self.epochs = epochs
        self.use_cuda = use_cuda

        self.best = 1e6

        self.weight_file = weight_file

        self.fmri_data = data['fmri_data']
        self.batch_size = self.fmri_data.batch_size

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        #self.regularize_layer = regularize_layer
        self.fmri_loss = []

        self.random=random
        #self.normalize = nn.BatchNorm1d(200)

        if self.use_cuda:
            self.model.cuda()

    def loss_fmri(self, fmri_out1, fmri_target,log_fmri_corr=False):

        def atanh(x, eps=1e-5):
            return 0.5 * torch.log((1 + x + eps) / (1 - x + eps))

        #fmri_target=fmri_target.type(torch.float32)
        #fmri_target=self.normalize(fmri_target)

        # cosine similarity
        #criterion = nn.MSELoss()
        #fmri_loss =criterion(fmri_out1.double(), fmri_target.double())


        #fmri_loss=1- self.cos(fmri_out1, fmri_target).mean()

        #backup
        fmri_target=fmri_target.type(torch.float32)
        #fmri_loss=F.binary_cross_entropy(fmri_out1,fmri_target, reduction='sum')

        fmri_loss=self.loss(fmri_out1,fmri_target)
        #fmri_loss=F.binary_cross_entropy(fmri_out1, fmri_target, reduction='sum')

        #fmri_loss = (self.cos(fmri_out1, fmri_target))#.pow(2).mean()
        #print(fmri_loss)
        #sys.exit()

        #similarity from paper https://papers.nips.cc/paper/9149-learning-from-brains-how-to-regularize-machines.pdf
        #fmri_loss =(atanh(model_sim)-atanh(fmri_sim)).pow(2).sum()

        # 1-pearson correlation
        #fmri_loss = 1 - audtorch.metrics.functional.pearsonr(model_sim, fmri_sim).squeeze(dim=0)
        #fmri_loss = ((model_sim) - (fmri_sim)).pow(2).mean()


        if log_fmri_corr:
            self.fmri_loss.append(str(fmri_loss.item()))

        return fmri_loss

    def train(self):
        for epoch in range(self.epochs):
            mean_epoch_loss = self.train_epoch(epoch)
            #print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1, self.batch_size * mean_epoch_loss))
            #self.test_epoch(epoch)
            self.scheduler.step()



    def train_epoch(self, epoch):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.train()

        for batch_idx in range(int(len(self.fmri_data.imagenet_idxs)/self.batch_size)):
        #for batch_idx in range(int(len(self.fmri_data.train)/self.batch_size)):

            self.optimizer.zero_grad()

            fmri_data, fmri_target = self.fmri_data.get_batch()

            if self.random:
                fmri_target=torch.rand_like(fmri_target)


            if self.use_cuda:
                fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()

            fmri_out1 = self.model(fmri_data)

            loss = self.loss_fmri( fmri_out1, fmri_target,log_fmri_corr=True)

            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            epoch_loss += train_loss
            print_every_loss += train_loss

        print(epoch_loss)
        if epoch_loss<self.best:
            self.best=epoch_loss
            print('saving...loss='+str(epoch_loss))
            torch.save(self.model.state_dict(), self.weight_file)

        # Return mean epoch loss
        return epoch_loss / len(self.fmri_data.imagenet_idxs)

    def test_epoch(self, epoch):
        epoch_loss = 0.
        self.model.eval()

        for batch_idx in range(int(len(self.fmri_data.test)/self.batch_size)):
            fmri_data, fmri_target = self.fmri_data.get_batch(True)

            if self.random:
                fmri_target=torch.rand_like(fmri_target)

            if self.use_cuda:
                fmri_data, fmri_target = fmri_data.cuda(), fmri_target.cuda()

            fmri_out1 = self.model.forward_single_fmri(fmri_data)
            loss = self.loss_fmri( fmri_out1, fmri_target,log_fmri_corr=True)
            train_loss = loss.item()
            epoch_loss += train_loss

        fmri_loss_file="results/fmri_only_dissimilarity_layer_"+str(self.regularize_layer)+".txt"
        if epoch > 0:
            outF = open(fmri_loss_file, "a")
        else:
            outF = open(fmri_loss_file, "w")

        for line in self.fmri_loss:
            outF.write(line)
            outF.write("\n")
        outF.close()
        self.fmri_loss = []
        print(epoch_loss)
        if epoch_loss<self.best:
            self.best=epoch_loss
            print('saving...loss='+str(epoch_loss))
            torch.save(self.model.state_dict(), self.weight_file)