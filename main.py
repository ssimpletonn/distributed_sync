import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
import modin.pandas as pd
import ray
import pandas
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

num_workers = 3
ray.init(address='auto',runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})

#paths=[]
#labels=[]
#for dirname, _, filenames in os.walk('./ILSVRC/Data/CLS-LOC/train'):
#    for filename in filenames:
#        if filename[-4:]=='JPEG':
#            paths+=[(os.path.join(dirname, filename))]
#            label=dirname.split('/')[-1]
#            labels+=[label]

#tpaths=[]
#tlabels=[]
#for dirname, _, filenames in os.walk('./content/imagenet-mini/val'):
#    for filename in filenames:
#        if filename[-4:]=='JPEG':
#            tpaths+=[(os.path.join(dirname, filename))]
#            label=dirname.split('/')[-1]
#            tlabels+=[label]


#class_names=sorted(set(labels))
#N=list(range(len(class_names)))
#normal_mapping=dict(zip(class_names, N))
#reverse_mapping=dict(zip(N, class_names))


#df=pd.DataFrame(columns=['path','label'])
#df['path']=paths
#df['label']=labels
#df['label']=df['label'].map(normal_mapping)

df = pd.read_pickle("imagenet.pkl")

#tdf=pd.DataFrame(columns=['path','label'])
#tdf['path']=tpaths
#tdf['label']=tlabels
#tdf['label']=tdf['label'].map(normal_mapping)
#tdf.to_pickle("test_imagenet.pkl")


#df.to_pickle("imagenet.pkl")

tdf = pd.read_pickle("test_imagenet.pkl")

print(df.head())
print('df saved')



class CustomDataset(Dataset):
    def __init__(self, dataframe):
        dataframe.reset_index(drop=True, inplace=True)
        self.dataframe = dataframe
        self.mean = [0.4871, 0.4636, 0.4174]
        self.std = [0.2252, 0.2216, 0.2226]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        path = self.dataframe.loc[index, 'path']
        label = self.dataframe.loc[index, 'label']
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
            ])
        image = transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.mean = [0.4871, 0.4636, 0.4174]
        self.std = [0.2252, 0.2216, 0.2226]

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        path = self.dataframe.loc[index, 'path']
        label = self.dataframe.loc[index, 'label']
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
            ])
        image = transform(image)
        return image, label

def split_dataframe_by_position(df, splits):
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split
    return dataframes

train_df = split_dataframe_by_position(df, num_workers)
train_ds = []
for i in range(num_workers):
    train_ds.append(CustomDataset(train_df[i]))

train_loader = []
train_loader_ref = []


for i in range(num_workers):
    train_loader_ref.append(ray.put(train_loader.append(DataLoader(train_ds[i],batch_size=2048, shuffle=True))))

test_ds=TestDataset(tdf)

test_loader=DataLoader(test_ds,batch_size=2048)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data, target
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

    

def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}

def set_weights(self, weights):
    self.load_state_dict(weights)

def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data
        grads.append(grad)
    return grads

def set_gradients(self, gradients):
    for g, p in zip(gradients, self.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)
=======
import argparse
from data import loadCIFAR10, loadFashionMNIST, loadImageNet, evaluate
from models import loadEfficientNet, loadLeNet
import sys
import config
from plot import plot_accuracy, plot_accuracy_workers, plot_loss_workers


num_workers = config.n_workers
iterations = config.iterations
lr=config.lr

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model')
parser.add_argument('-d','--dataset')
parser.add_argument('-p', '--pretrained')

args = parser.parse_args()
train_loader, test_loader = None, None


if(args.dataset == 'imagenet'):
    train_loader, test_loader = loadImageNet(config.imagenet_path, num_workers)
elif(args.dataset=='cifar10'):
    train_loader, test_loader = loadCIFAR10(num_workers)
elif(args.dataset=='fashionmnist'):
    train_loader, test_loader = loadFashionMNIST(num_workers)
else:
    sys.exit('Dataset is currently not supported')

load_model = None
if(args.model == 'efficientnet'):
    if(args.dataset == 'fashionmnist'):
        sys.exit('fashionmnist with efficientnet')
    load_model = loadEfficientNet
    print('EfficientNet')
elif(args.model == 'lenet'):
    if(args.dataset == 'imagenet'):
        sys.exit('imagenet with lenet')
    load_model = loadLeNet
    print('LeNet')
    
@ray.remote(resources={"ps" : 1})
class ParameterServer(object):
    def __init__(self, lr, model, momentum, weight_decay):
        self.model = model

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.loss_vals = []
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)#, momentum=momentum)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.model.train()
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()
    

@ray.remote(resources={"worker" : 1})
class DataWorker(object):
    def __init__(self, data_loader):
        self.model = load_model(classifier=True)

        self.train_loader = data_loader
        self.data_iterator = iter(self.train_loader)
        self.loss_vals = []
        self.acc_vals = []


    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        model.train()
        try:
            data, target = next(self.data_iterator)
        except:
            self.data_iterator = iter(self.train_loader)
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        self.loss_vals.append(loss.item())
        #acc = evaluate(self.model, test_loader)
        #self.acc_vals.append(acc)
        print(loss)
        loss.backward()
        return self.model.get_gradients()
    
    def get_loss_vals(self):
        return self.loss_vals
    
    def get_accuracy_vals(self):
        return self.acc_vals

model = load_model()

context = ray.init(address='auto')

iterations = 200
model = torchvision.models.efficientnet_b0(weights=None)

ps = ParameterServer.remote(lr=config.lr, model=model, momentum=0.9)
workers = [DataWorker.remote(train_loader[i]) for i in range(num_workers)]
current_weights = ps.get_weights.remote()


print("Запуск синхронного распределенного обучения с параметрическим сервером.")
accuracy = 0
accuracy_vals = []
if(config.until_80):
    c = 0
    while(accuracy < 80):
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)


ps = ParameterServer.remote(lr=0.08, model=model, momentum=0.9, weight_decay=1e-5)
workers = [DataWorker.remote(model, train_loader_ref[i]) for i in range(num_workers)]

print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
for i in range(iterations):
    gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(*gradients)
    
    # Evaluate the current model.
    model.set_weights(model, ray.get(current_weights))
    with open("model.pkl", "wb") as d:
        pickle.dump(model, f)
    accuracy = evaluate(model, test_loader)
    print("Iter {}: \taccuracy is {}".format(i + 1, accuracy))

print("Final accuracy is {:.1f}.".format(accuracy))
# Clean up Ray resources and processes before the next example.

        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {}".format(c + 1, accuracy))
else:
    for i in range(iterations):
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        accuracy_vals.append(accuracy)
        print("Iter {}: \taccuracy is {}".format(i + 1, accuracy))

print("Final accuracy is {}.".format(accuracy))


plot_accuracy(accuracy_vals, iterations)

#accuracy_workers = []
#for i in range(num_workers):
#    accuracy_workers.append([])

#for i in range(num_workers):
#    accuracy_workers[i].append(
#        [ray.get(workers[i].get_accuracy_vals.remote())])
#plot_accuracy_workers(accuracy_workers, iterations, num_workers)

loss_workers = []
for i in range(num_workers):
    loss_workers.append([])

for i in range(num_workers):
    loss_workers[i].append(
        ray.get(workers[i].get_loss_vals.remote()))

plot_loss_workers(loss_workers, iterations, num_workers)

ray.shutdown()
