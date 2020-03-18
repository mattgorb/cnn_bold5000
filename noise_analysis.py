from dataloaders.cifar10_dataloaders import *
from models.resnet import *
from trainer import *
import matplotlib.pyplot as plt


batch_size = 50

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=batch_size)






def get_noise_on_test(weight_file, noise):
    model = resnet18()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        network_state_dict = torch.load(weight_file)
        model.load_state_dict(network_state_dict)
    else:
        network_state_dict = torch.load(weight_file,map_location = torch.device('cpu'))
        model.load_state_dict(network_state_dict)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(cifar10_test_loader):
            #add random noise
            data=data+noise*torch.rand_like(data)

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return (100 * correct / total)



weight_file='model_weights/cifar10_resnet50.pth'
weight_file_2='model_weights/cifar10_resnet50_fmri_layer2_alpha0.2.pth'
weight_file_4='model_weights/cifar10_resnet50_fmri_layer4_alpha0.2.pth'



normal=[]
layer2=[]
layer4=[]
noise=[0.0,0.1,0.2,0.5,1,2,5,10]

for n in noise:
    normal.append(get_noise_on_test(weight_file, n))

for n in noise:
    layer2.append(get_noise_on_test(weight_file_2, n))

for n in noise:
    layer4.append(get_noise_on_test(weight_file_4, n))


plt.plot([i for i in noise], [i for i in normal], linestyle='--', label='Normal')
plt.plot([i for i in noise], [i for i in layer2], linestyle='--', label='Layer 2 reg')
plt.plot([i for i in noise], [i for i in layer4], linestyle='--', label='Layer 4 reg')

plt.legend(loc='upper right', framealpha=1, frameon=True)
plt.savefig('visuals/robustness.png')



f = open('results/fmri_dissimilarity_layer_2_alpha_0.2.txt', 'r')
layer_2 = f.readlines()
f.close()
layer_2=[float(x[:-1]) for x in layer_2]


f = open('results/fmri_dissimilarity_layer_4_alpha_0.2.txt', 'r')
layer_4 = f.readlines()
f.close()
layer_4=[float(x[:-1]) for x in layer_4]

layer_2=layer_2[0::20]

layer_4=layer_4[0::20]

plt.plot([i for i in range(len(layer_2))], [i for i in layer_2],marker='.', linestyle='None', label='Layer 2 regularized')
plt.plot([i for i in range(len(layer_4))], [i for i in layer_4],marker='.', linestyle='None', label='Layer 4 regularized')

plt.legend(loc='upper right', framealpha=1, frameon=True)
plt.savefig('visuals/regularization.png')
#plt.show()