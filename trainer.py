import torch
import audtorch

class Trainer():
    def __init__(self, model, optimizer,loss,data,weight_file,with_fmri_data=False ,print_loss_every=5, epochs=100,
                 use_cuda=False):

        self.model = model
        self.optimizer = optimizer
        self.loss=loss

        self.print_loss_every = print_loss_every
        self.epochs = epochs
        self.use_cuda = use_cuda

        self.accuracy=0

        self.weight_file=weight_file

        self.train_main=data['train_main']
        self.test_main=data['test_main']
        self.batch_size = self.train_main.batch_size

        self.with_fmri_data = -with_fmri_data
        if self.with_fmri_data:
            self.fmri_data=data['fmri_data']
            self.cos=torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            self.alpha_factor=.5
            self.fmri_loss=[]

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.losses = {'loss': []}


    def loss_fmri(self,output, target, fmri_out1, fmri_target, fmri_out2, fmri_target2):
        loss_main=self.loss(output, target)

        #cosine similarity
        model_sim=self.cos(fmri_out1, fmri_out2)
        fmri_sim=self.cos(fmri_target,fmri_target2)

        #1-pearson correlation
        fmri_loss=1-audtorch.metrics.functional.pearsonr(model_sim, fmri_sim).squeeze(dim=0)
        self.fmri_loss.append(str(fmri_loss.item()))

        total_loss=loss_main+self.alpha_factor*fmri_loss
        return total_loss


    def train(self):

        for epoch in range(self.epochs):
            mean_epoch_loss = self.train_epoch()
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1, self.batch_size *  mean_epoch_loss))
            self.test_epoch()


    def train_epoch(self):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_main):
            if self.use_cuda:
                data = data.cuda()

            self.optimizer.zero_grad()

            if self.with_fmri_data:
                fmri_data, fmri_target=self.fmri_data.get_batch()
                fmri_data2, fmri_target2 = self.fmri_data.get_batch()

                output, fmri_out1, fmri_out2=self.model.forward_fmri(data, fmri_data, fmri_data2)
                loss=self.loss_fmri(output, target, fmri_out1, fmri_target, fmri_out2, fmri_target2)


            else:
                output = self.model(data)
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

        outF = open("fmri_losses.txt", "a")
        for line in self.fmri_loss:
            outF.write(line)
            outF.write("\n")
        outF.close()
        self.fmri_loss = []

        # Return mean epoch loss
        return epoch_loss / len(self.train_main.dataset)


    def test_epoch(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_main):

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        if correct/total>self.accuracy:
            print('Saving weights...')
            self.accuracy=(correct/total)
            torch.save(self.model.state_dict(),self.weight_file)
