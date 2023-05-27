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
import pickle as pkl

num_workers = 15
ray.init(address='auto',runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
print('inited')
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

#df = pd.read_pickle("imagenet.pkl")

#tdf=pd.DataFrame(columns=['path','label'])
#tdf['path']=tpaths
#tdf['label']=tlabels
#tdf['label']=tdf['label'].map(normal_mapping)
#tdf.to_pickle("test_imagenet.pkl")


#df.to_pickle("imagenet.pkl")

#tdf = pd.read_pickle("test_imagenet.pkl")

#print(df.head())
#print('df saved')



#class CustomDataset(Dataset):
#    def __init__(self, dataframe):
#        dataframe.reset_index(drop=True, inplace=True)
#        self.dataframe = dataframe
#        self.mean = [0.4871, 0.4636, 0.4174]
#        self.std = [0.2252, 0.2216, 0.2226]
#
#    def __len__(self):
#        return len(self.dataframe)
#
#    def __getitem__(self, index):
#        path = self.dataframe.loc[index, 'path']
#        label = self.dataframe.loc[index, 'label']
#        image = Image.open(path).convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize((224, 224)),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
#            ])
#        image = transform(image)
#        return image, label

#class TestDataset(Dataset):
#    def __init__(self, dataframe):
#        self.dataframe = dataframe
#        self.mean = [0.4871, 0.4636, 0.4174]
#        self.std = [0.2252, 0.2216, 0.2226]
#
#    def __len__(self):
#        return len(self.dataframe)
#    
#    def __getitem__(self, index):
#        path = self.dataframe.loc[index, 'path']
#        label = self.dataframe.loc[index, 'label']
#        image = Image.open(path).convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize((224, 224)),
#            transforms.ToTensor(),
#            transforms.Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))
#            ])
#        image = transform(image)
#        return image, label

#def split_dataframe_by_position(df, splits):
#    dataframes = []
#    index_to_split = len(df) // splits
#    start = 0
#    end = index_to_split
#    for split in range(splits):
#        temporary_df = df.iloc[start:end, :]
#        dataframes.append(temporary_df)
##        start += index_to_split
 #       end += index_to_split
#    return dataframes

#train_df = split_dataframe_by_position(df, num_workers)
#train_ds = []
#for i in range(num_workers):
#    train_ds.append(CustomDataset(train_df[i]))

#train_loader = []
#train_loader_ref = []


#for i in range(num_workers):
#    train_loader_ref.append(ray.put(train_loader.append(DataLoader(train_ds[i],batch_size=2048, shuffle=True))))

#test_ds=TestDataset(tdf)

#test_loader=DataLoader(test_ds,batch_size=2048)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])


train_path = './ILSVRC/Data/CLS-LOC/train'
print('creating dataset')
full_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=transform)
print('size')
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size
print('splitting full dataset')
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

lenght = len(train_dataset)//num_workers
print('splitting dataset')
train_loader = []
for i in range(num_workers):
    train_loader.append(torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, range(i*num_workers, i*num_workers + lenght))))

#train_loader = torch.utils.data.DataLoader(
#        train_dataset,
#        batch_size=256,
#        num_workers=0,
#        shuffle=True)
print('creating loader')
test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=0,
        shuffle=False)

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

@ray.remote(resources={"ps" : 1})
class ParameterServer(object):
    def __init__(self, lr, model, momentum, weight_decay):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) #momentum=momentum, weight_decay=weight_decay)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.model.train()
        self.optimizer.zero_grad()
        self.model.set_gradients(self.model, summed_gradients)
        self.optimizer.step()
        return self.model.get_weights(self.model)

    def get_weights(self):
        return self.model.get_weights(self.model)
    

@ray.remote(resources={"worker" : 1})
class DataWorker(object):
    def __init__(self, model, data_loader):
        self.model = torchvision.models.efficientnet.efficientnet_b0(weights=None) 
        self.train_loader = data_loader
        self.data_iterator = iter(self.train_loader)
        self.model.set_weights = set_weights
        self.model.get_gradients = get_gradients

    def compute_gradients(self, weights):
        self.model.set_weights(self.model, weights)
        model.train()
        try:
            data, target = next(self.data_iterator)
        except:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(self.train_loader)
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        print(loss)
        loss.backward()
        return self.model.get_gradients(self.model)


iterations = 200
model = torchvision.models.efficientnet_b0(weights=None)

model.get_weights = get_weights
model.set_weights = set_weights
model.get_gradients = get_gradients
model.set_gradients = set_gradients

ps = ParameterServer.remote(lr=0.001, model=model, momentum=0.9, weight_decay=1e-5)
workers = [DataWorker.remote(model, train_loader[i]) for i in range(num_workers)]

print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
c = 0
acc = 0
for i in range(iterations):
    gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(*gradients)
    c += 1
    # Evaluate the current model.
    model.set_weights(model, ray.get(current_weights))
    with open("model.pkl", "wb") as d:
        pkl.dump(model, d)
    print('evaluating')
    if len(train_loader[0]) == c:
        c = 0
        accuracy = evaluate(model, test_loader)
        acc = 0
        print("Iter {}: \taccuracy is {}".format(i + 1,  accuracy))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
