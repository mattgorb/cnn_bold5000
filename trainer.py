import torch
import audtorch


class Trainer():
    def __init__(self, model, optimizer, loss, data, weight_file, regularize_with_fmri_data=False, print_loss_every=5, epochs=250,
                 use_cuda=False, alpha_factor=0.5,regularize_layer=None):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma=0.1)

        self.print_loss_every = print_loss_every
        self.epochs = epochs
        self.use_cuda = use_cuda

        self.accuracy = 0

        self.weight_file = weight_file

        self.train_main = data['train_main']
        self.test_main = data['test_main']
        self.batch_size = self.train_main.batch_size



        self.regularize_with_fmri_data = regularize_with_fmri_data
        if self.regularize_with_fmri_data:
            self.fmri_data = data['fmri_data']
            self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            self.alpha_factor = alpha_factor
            self.regularize_layer = regularize_layer

            self.fmri_loss = []

        if self.use_cuda:
            self.model.cuda()

    def loss_plus_fmri(self, output, target, fmri_out1, fmri_target, fmri_out2, fmri_target2,log_fmri_corr=False):

        def atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        loss_main = self.loss(output, target)

        # cosine similarity
        model_sim = self.cos(fmri_out1, fmri_out2)
        fmri_sim = self.cos(fmri_target, fmri_target2)


        #similarity from paper https://papers.nips.cc/paper/9149-learning-from-brains-how-to-regularize-machines.pdf
        fmri_loss =(atanh(model_sim)-atanh(fmri_sim)).pow(2).sum()

        #print(fmri_loss)

        # 1-pearson correlation
        #fmri_loss = 1 - audtorch.metrics.functional.pearsonr(model_sim, fmri_sim).squeeze(dim=0)

        if log_fmri_corr:
            self.fmri_loss.append(str(fmri_loss.item()))

        total_loss = loss_main + self.alpha_factor * fmri_loss
        return total_loss

    def train(self):
        for epoch in range(self.epochs):
            mean_epoch_loss = self.train_epoch(epoch)
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1, self.batch_size * mean_epoch_loss))
            self.test_epoch(epoch)
            self.scheduler.step()

    def train_epoch(self, epoch):
        epoch_loss = 0.
        print_every_loss = 0.
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_main):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()

            if self.regularize_with_fmri_data:
                fmri_data, fmri_target = self.fmri_data.get_batch()
                fmri_data2, fmri_target2 = self.fmri_data.get_batch()

                if self.use_cuda:
                    fmri_data, fmri_target,fmri_data2, fmri_target2 = fmri_data.cuda(), fmri_target.cuda(),fmri_data2.cuda(), fmri_target2.cuda()

                output, fmri_out1, fmri_out2 = self.model.forward_fmri(data, fmri_data, fmri_data2)
                loss = self.loss_plus_fmri(output, target, fmri_out1, fmri_target, fmri_out2, fmri_target2)


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
                                                   len(self.train_main.dataset), mean_loss))
                print_every_loss = 0.



        # Return mean epoch loss
        return epoch_loss / len(self.train_main.dataset)

    def test_epoch(self, epoch):
        correct = 0
        total = 0
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_main):
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)



                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if self.regularize_with_fmri_data:
                    fmri_data, fmri_target = self.fmri_data.get_batch()
                    fmri_data2, fmri_target2 = self.fmri_data.get_batch()

                    if self.use_cuda:
                        fmri_data, fmri_target, fmri_data2, fmri_target2 = fmri_data.cuda(), fmri_target.cuda(), fmri_data2.cuda(), fmri_target2.cuda()

                    output, fmri_out1, fmri_out2 = self.model.forward_fmri(data, fmri_data, fmri_data2)
                    loss = self.loss_fmri(output, target, fmri_out1, fmri_target, fmri_out2, fmri_target2, log_fmri_corr=True)

                    test_loss += self.loss(output, target).item()

                else:
                    test_loss += self.loss(output, target).item()

        if self.regularize_with_fmri_data:
            fmri_loss_file="results/fmri_dissimilarity_layer_"+str(self.regularize_layer)+'_alpha_'+str(self.alpha_factor)+".txt"
            if epoch > 0:
                outF = open(fmri_loss_file, "a")
            else:
                outF = open(fmri_loss_file, "w")

            for line in self.fmri_loss:
                outF.write(line)
                outF.write("\n")
            outF.close()
            self.fmri_loss = []



        if self.regularize_with_fmri_data:
            test_loss_file = "results/test_losses_fmri_layer_"+str(self.regularize_layer)+'_alpha_'+str(self.alpha_factor)+".txt"
            test_accuracy_file = "results/test_accuracy_fmri_layer_"+str(self.regularize_layer)+'_alpha_'+str(self.alpha_factor)+".txt"
        else:
            test_loss_file = "results/test_losses.txt"
            test_accuracy_file = "results/test_accuracy.txt"

        if epoch > 0:
            outF = open(test_loss_file, "a")
        else:
            outF = open(test_loss_file, "w")
        outF.write(str(test_loss))
        outF.write("\n")
        outF.close()

        if epoch > 0:
            outF = open(test_accuracy_file, "a")
        else:
            outF = open(test_accuracy_file, "w")
        outF.write(str(correct / total))
        outF.write("\n")
        outF.close()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        if correct / total > self.accuracy:
            print('Saving weights...')
            self.accuracy = (correct / total)
            torch.save(self.model.state_dict(), self.weight_file)

