from torch import optim
from dataloaders.bold5000_dataset import *
from dataloaders.cifar10_dataloaders import *
from models.vgg16 import *

batch_size = 64
lr = 5e-4
epochs = 100

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)

model = VGG()
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), 'example-model.pt')