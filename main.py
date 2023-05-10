import os
import types
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
import pickle

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
        transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()#transforms.Compose([
            #transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            #transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
            #])
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
        transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()#transforms.Compose([
            #transforms.Resize((224, 224)),
            #transforms.ToTensor(),
            #transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
            #])
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


def get_data_loader(train_ds, test_ds, num_workers):
    with FileLock(os.path.expanduser('./data.lock')):
        train_loader = []
        for i in range(num_workers):
            train_loader.append(DataLoader(train_ds[i],batch_size=64, shuffle=True))
        test_loader = DataLoader(test_ds, batch_size=64)
    return train_loader, test_loader


test_ds = TestDataset(tdf)

train_loader, test_loader = get_data_loader(train_ds, test_ds, num_workers) #=DataLoader(test_ds,batch_size=2048)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data, target
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100.0 * correct / total 

def get_weights(self):
    return {k: v for k, v in self.state_dict().items()}

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
        else:
            print("Something is wrong!")

@ray.remote(resources={"ps" : 1})
class ParameterServer(object):
    def __init__(self, lr, momentum, weight_decay):
        self.model = torchvision.models.efficientnet_b0(weights=None)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)#, momentum=momentum)
        self.model.set_gradients = types.MethodType(set_gradients, self.model)
        self.model.get_weights = types.MethodType(get_weights, self.model)
        for params in self.model.parameters():
            params.requires_grad = True

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
        self.model = torchvision.models.efficientnet_b0(weights=None) 
        self.train_loader = data_loader
        self.data_iterator = iter(self.train_loader)
        self.model.set_weights = types.MethodType(set_weights, self.model)
        self.model.get_gradients = types.MethodType(get_gradients, self.model)
        for params in self.model.parameters():
            params.requires_grad = True

    def compute_gradients(self, weights):
        self.model.train()
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except:  
            self.data_iterator = iter(self.train_loader)
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = nn.CrossEntropyLoss()
        out = loss(output, target)
        print(out)
        out.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        return self.model.get_gradients()

iterations = 200
model = torchvision.models.efficientnet_b0(weights=None)

model.get_weights = types.MethodType(get_weights, model)
model.set_weights = types.MethodType(set_weights, model)
model.get_gradients = types.MethodType(get_gradients, model)
model.set_gradients = types.MethodType(set_gradients, model)

for params in model.parameters():
    params.requires_grad = True

ps = ParameterServer.remote(lr=0.01, momentum=0.9, weight_decay=1e-5)
workers = [DataWorker.remote(train_loader[i]) for i in range(num_workers)]

print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
for i in range(iterations):
    gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
    current_weights = ps.apply_gradients.remote(*gradients)
    model.set_weights(ray.get(current_weights))
    with open(f'{i}_model.pkl', "wb") as d:
        pickle.dump(model, d)
    accuracy = evaluate(model, test_loader)
    print("Iter {}: \taccuracy is {}".format(i + 1, accuracy))

print("Final accuracy is {:.1f}.".format(accuracy))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
