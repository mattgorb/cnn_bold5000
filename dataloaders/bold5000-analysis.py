import os
import torch
from skimage import io, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

airplane = ['n04552348','n02690373' ]
automobile = ['n04037443', 'n04285008','n03100240']
bird = ['n01531178','n01532829','n01534433','n01537544',
        'n01558993','n01560419','n01580077','n01582220','n01592084','n01601694',
        'n01833805','n01806143','n01806567','n01807496',
        'n01818515','n01819313','n01820546','n01824575','n01828970','n01829413',
        'n01833805','n01843065','n01843383','n01847000','n01855032']
cat = ['n02123045','n02123159','n02123394','n02123597','n02124075' ]
dog = ['n02116738','n02085620','n02085782','n02085936','n02086079','n02086240','n02086646','n02086910',
       'n02087046','n02087394','n02088094','n02088238','n02088364','n02088466','n02088632','n02089078','n02089867',
       'n02089973','n02090379','n02090622','n02090721','n02091032','n02091134','n02091244','n02091467',
        'n02092339','n02092002','n02091831','n02091635', 'n02093256','n02093428','n02093647','n02093754',
        'n02093859','n02093991','n02094114','n02094258', 'n02094433','n02095314','n02095570','n02095889',
       'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209',
       'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429',
       'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006',
       'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029',
        'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855',
        'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574',
       'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047',
       'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958',
       'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706',
       'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978']

frog = ['n01641577','n01644373','n01644900' ]
ship = ['n03095699','n03947888','n03344393','n03662601','n03873416','n04273569','n02951358','n02981792' ]
truck = ['n04461696','n03345487','n03417042','n03930630', 'n04467665']

class Bold5000(Dataset):
    def __init__(self, fmri_dir, imagedir, stim_list_dir, transform, batch_size):
        self.data = {}

        with h5py.File(fmri_dir, 'r') as f:
            keys = list(f.keys())
            # print(keys)
            for i in keys:
                self.data[i] = list(f[i])

        self.fmri_dir = fmri_dir
        self.imagedir = imagedir
        self.transform = transform
        self.batch_size = batch_size

        self.target_data = self.data['RHPPA']

        f = open(stim_list_dir, 'r')
        self.CSI01_stim_lists = f.read().splitlines()
        f.close()

        all=airplane+automobile+bird+cat+dog+frog+truck+ship
        self.imagenet_idxs=[]
        self.index_targets={}

        def add(list, target):
            for j in list:
                if j in x:
                    self.imagenet_idxs.append(i)
                    self.index_targets[i]=target


        for i, x in enumerate(self.CSI01_stim_lists):
            add(airplane, 0)
            add(automobile,0)
            add(bird, 1)
            add(cat,1)
            add(dog, 1)
            add(frog, 1)
            add(truck, 0)
            add(ship, 0)


    def __len__(self):
        return len(self.imagenet_idxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.imagedir, self.CSI01_stim_lists[idx])
        image = io.imread(img_name)

        if len(image.shape) == 2:
            color.gray2rgb(image)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        target = self.target_data[idx]

        return sample, torch.from_numpy(target), idx

    def get_random_idxs(self):
        return np.random.choice(self.imagenet_idxs, size=(self.batch_size))

    def get_batch(self):
        sample_list = []
        target_list = []
        for i in self.get_random_idxs():
            sample, target, _ = self[i]
            sample_list.append(sample)
            target_list.append(target)
        return torch.stack(sample_list), torch.stack(target_list)


def get_bold5000_dataset(batch_size):
    to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    filename = './ROIs/CSI1/h5/CSI1_ROIs_TR34.h5'
    image_folder = './BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/ImageNet'
    stim_list = './ROIs/stim_lists/CSI01_stim_lists.txt'

    bold5000 = Bold5000(fmri_dir=filename,
                        imagedir=image_folder,
                        stim_list_dir=stim_list,
                        transform=to_tensor, batch_size=batch_size)
    return bold5000


x = get_bold5000_dataset(50)

all_brain_data=[]
brain_targets=[]
for i in x.imagenet_idxs:
    all_brain_data.append(x.target_data[i])
    brain_targets.append(x.index_targets[i])







#norm=nn.BatchNorm1d(200)
#tensor=torch.tensor(all_brain_data).type(torch.float32)

#tensor=norm(tensor)

#all_brain_data=tensor.detach().numpy()


X = np.array(all_brain_data)
X_embedded = TSNE(n_components=2).fit_transform(X)

palette = sns.color_palette("bright", 2)
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.clf()
sns.scatterplot(X_embedded[:,0], X_embedded[:,1],hue=np.array(brain_targets), legend='full', palette=palette)


plt.show()

'''
estimator = GaussianMixture(n_components=8)
#estimator =KMeans(init='k-means++', n_clusters=10, n_init=10)
#pca = PCA(n_components=10).fit(x)
#estimator =KMeans(init=pca.components_, n_clusters=10, n_init=1)

preds=estimator.fit_predict(X)


print(metrics.homogeneity_score(np.array(brain_targets),preds))
print(metrics.completeness_score(np.array(brain_targets), preds))
print(metrics.v_measure_score(np.array(brain_targets), preds))
print(metrics.adjusted_rand_score(np.array(brain_targets),preds))
print(metrics.adjusted_mutual_info_score(np.array(brain_targets),preds))


'''