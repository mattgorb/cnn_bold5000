from dataloaders.cifar10_dataloaders import *
from models.resnet import *
from trainer import *

batch_size = 50

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)


model = resnet18()
use_cuda = torch.cuda.is_available()

weight_file='model_weights/cifar10_resnet50.pth'
if use_cuda:
    model.cuda()
    network_state_dict = torch.load(weight_file)
    model.load_state_dict(network_state_dict)
else:
    network_state_dict = torch.load(weight_file,map_location = torch.device('cpu'))
    model.load_state_dict(network_state_dict)

correct = 0
total = 0
test_loss = 0
model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(cifar10_test_loader):
        #add random noise
        data=data+10*torch.rand_like(data)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))