
from dataloaders.bold5000_dataset import *
from dataloaders.cifar10_dataloaders import *

from models.resnet import *
from trainer import *
from fmri_only_trainer import *


batch_size = 50


weight_file='model_weights/cifar10_resnet50.pth'


batch_size=50
fmri_dataset=get_bold5000_dataset(batch_size)


def loss_plus_fmri( fmri_out1, fmri_target, fmri_out2, fmri_target2):
    def atanh(x, eps=1e-5):
        return 0.5 * torch.log((1 + x + eps) / (1 - x + eps))

    cos=torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    model_sim =atanh(cos(fmri_out1, fmri_out2))
    fmri_sim =atanh( cos(fmri_target, fmri_target2))

    # similarity from paper https://papers.nips.cc/paper/9149-learning-from-brains-how-to-regularize-machines.pdf
    fmri_loss=audtorch.metrics.functional.pearsonr(model_sim, fmri_sim).squeeze(dim=0)

    return fmri_loss.item()

for reg in [1,2,3,4,'fc1','fc2']:
    # regularize_layer={1,2,3,4,fc1,fc2}
    model = resnet18(regularize_layer=reg)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    network_state_dict = torch.load(weight_file, map_location='cpu')
    model.load_state_dict(network_state_dict)

    model.eval()
    average=0
    for batch_idx in range(int(len(fmri_dataset.imagenet_idxs) / batch_size)):
        fmri_data, fmri_target = fmri_dataset.get_batch()
        fmri_data2, fmri_target2 = fmri_dataset.get_batch()

        fmri_out1, fmri_out2=model.forward_only_fmri(fmri_data,fmri_data2)

        average+=loss_plus_fmri( fmri_out1, fmri_target, fmri_out2, fmri_target2)

    print(average/batch_idx)