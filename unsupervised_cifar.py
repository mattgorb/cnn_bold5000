from torch import optim
from dataloaders.cifar10_dataloaders import *
import matplotlib.pyplot as plt
from models.resnet import *
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from dataloaders.bold5000_wordnet_mods import *
from collections import Counter
from sklearn.svm import SVC
from models.vgg import *

#batch_size = 50

# Load data
cifar10_train_loader, cifar10_test_loader = get_cifar_dataloaders(batch_size=10)


weight_file='model_weights/pca.pth'


model = resnet18()
model=vgg11(num_classes=200)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

model.eval()

network_state_dict = torch.load(weight_file)
model.load_state_dict(network_state_dict)

print('loaded '+str(weight_file))


b5000 = get_bold5000_dataset(1)
fmri_data=[]
fmri_class_data=[]
j=0
for i in b5000.imagenet_idxs:
    image, target, idx=b5000[i]
    out=model(image.unsqueeze(dim=0)).detach().numpy()
    #fmri_data.append(out[0])

    fmri_data.append(target.numpy())
    #print(out[0])
    #print(target.numpy())#.numpy()
    '''print(np.mean((out[0]-target.numpy())**2))
    #print(np.mean(np.abs((out[0]-target.numpy()))))
    j+=1
    if j>5:
        break'''

    fmri_class_data.append(b5000.binary_class_data[b5000.CSI01_stim_lists[i]])

#rom sklearn.decomposition import PCA
#pca = PCA(n_components=20)
#fmri_data=pca.fit_transform(fmri_data)
#print(pca.singular_values_)
#print(pca.explained_variance_ratio_)

objects=[0,1,8,9]
animals=[2,3,4,5,6,7]

cifar10_data=[]
cifar10_targets=[]
for batch_idx, (data, target) in enumerate(cifar10_test_loader):
    output = model(data).detach().numpy()
    for i, j in zip(output, target):
        cifar10_data.append(i)
        if j in objects:
            cifar10_targets.append(1)
        else:
            cifar10_targets.append(0)




def balance_classes(data,targets):
    revised_data=[]
    revised_targets=[]
    t=Counter(targets)
    even_num=min(t[0],t[1])
    a=0
    b=0
    for i in range(len(targets)):
        if targets[i]==0:
            if a<even_num:
                revised_data.append(data[i])
                revised_targets.append(targets[i])
                a+=1
        else:
            if b<even_num:
                revised_data.append(data[i])
                revised_targets.append(targets[i])
                b+=1
    return np.array(revised_data), np.array(revised_targets)

#cifar10_data,cifar10_targets=balance_classes(cifar10_data,cifar10_targets)
fmri_data,fmri_class_data=balance_classes(fmri_data,fmri_class_data)




def fit_svm(data,targets):
    clf = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')  #
    #clf.fit(fmri_data, np.array(fmri_class_data))
    clf = SVC(gamma='auto', kernel='rbf')
    clf.fit(fmri_data, fmri_class_data)
    y_pred = clf.predict(data)
    #y_pred = clf.decision_function(data)

    #print("Roc Score")
    #fpr, tpr, threshold = metrics.roc_curve(targets, y_pred)
    print(roc_auc_score(targets, y_pred))
    return
    coef = np.reshape(clf.coef_, (clf.coef_.shape[1], clf.coef_.shape[0]))
    test_values = []
    for h in range(len(data)):
        test_values.append(data[h].dot(coef) + clf.intercept_)

    i = [test_values[i] for i in range(len(test_values)) if targets[i] == 1 ]
    l2 = [test_values[i][0] for i in range(len(test_values)) if targets[i] == 0 ]


    plt.clf()

    plt.plot([j for j in range(len(i))], i, 'x', label='vehicles')
    plt.plot([j for j in range(len(l2))], l2, '+', label='animals')

    plt.legend(loc="lower right")
    plt.savefig('svm_cifar.png')

fit_svm(cifar10_data, cifar10_targets)
print(fmri_data.shape)
fit_svm(fmri_data,fmri_class_data)




'''X = np.array(data_)
X_embedded = TSNE(n_components=2).fit_transform(X)

palette = sns.color_palette("bright", 2)
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.clf()
sns.scatterplot(X_embedded[:,0], X_embedded[:,1],hue=np.array(targets), legend='full', palette=palette)

#plt.show()
plt.savefig('cifar10_tsne_2.png')
plt.clf()'''