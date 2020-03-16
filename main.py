import torch

from torch import optim
from vgg16 import *

batch_size = 64
lr = 5e-4
epochs = 100

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader, _ = get_mnist_dataloaders(batch_size=batch_size)
img_size = (3, 32, 32)

# Define latent spec and model
latent_spec = {'cont': 10, 'disc': [10]}
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