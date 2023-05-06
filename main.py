import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
import pandas as pd
import ray
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

paths=[]
labels=[]
for dirname, _, filenames in os.walk('./content/imagenet-mini/train'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            paths+=[(os.path.join(dirname, filename))]
            label=dirname.split('/')[-1]
            labels+=[label]

tpaths=[]
tlabels=[]
for dirname, _, filenames in os.walk('./content/imagenet-mini/val'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            tpaths+=[(os.path.join(dirname, filename))]
            label=dirname.split('/')[-1]
            tlabels+=[label]


class_names=sorted(set(labels))
N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names, N))
reverse_mapping=dict(zip(N, class_names))


df=pd.DataFrame(columns=['path','label'])
df['path']=paths
df['label']=labels
df['label']=df['label'].map(normal_mapping)

tdf=pd.DataFrame(columns=['path','label'])
tdf['path']=tpaths
tdf['label']=tlabels
tdf['label']=tdf['label'].map(normal_mapping)

class CustomDataset(Dataset):
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

train_ds=CustomDataset(df)
test_ds=TestDataset(tdf)


train_loader=DataLoader(train_ds,batch_size=100,shuffle=True)
test_loader=DataLoader(test_ds,batch_size=100)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 10000:
                break
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
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

def set_gradients(self, gradients):
    for g, p in zip(gradients, self.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)

@ray.remote(resources={"ps" : 1})
class ParameterServer(object):
    def __init__(self, lr, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(self.model, summed_gradients)
        self.optimizer.step()
        return self.model.get_weights(self.model)

    def get_weights(self):
        return self.model.get_weights(self.model)
    

@ray.remote(resources={"worker" : 1})
class DataWorker(object):
    def __init__(self, model):
        self.model = model 
        self.data_iterator = iter(train_loader)

    def compute_gradients(self, weights):
        self.model.set_weights(self.model, weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(train_loader)
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        return self.model.get_gradients(self.model)
    

iterations = 200
num_workers = 15
model = torchvision.models.efficientnet_b0()

model.get_weights = get_weights
model.set_weights = set_weights
model.get_gradients = get_gradients
model.set_gradients = set_gradients

ray.init(address='auto')
ps = ParameterServer.remote(0.1, model)
workers = [DataWorker.remote(model) for i in range(num_workers)]



print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
for i in range(iterations):
    gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(*gradients)

    # Evaluate the current model.
    model.set_weights(model, ray.get(current_weights))
    accuracy = evaluate(model, test_loader)
    print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

print("Final accuracy is {:.1f}.".format(accuracy))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
