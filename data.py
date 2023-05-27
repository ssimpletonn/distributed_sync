from torchvision.datasets import CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from filelock import FileLock
import os

def evaluate(model, test_loader):
    '''Evaluating'''
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

def loadCIFAR10(num_workers):
    '''Loads CIFAR10'''
    with FileLock(os.path.expanduser("~/data.lock")):
        train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224,224], antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.Normalize((0.5074,0.4867,0.4411), (0.2011,0.1987,0.2025))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5074,0.4867,0.4411), (0.2011,0.1987,0.2025))
        ])
        
        train_data = CIFAR10(download=True, root="./data", transform=train_transforms)
        test_data = CIFAR10(root="./data", train=False, transform=test_transforms)
        lengths=[]
        for i in range(num_workers):
            lengths.append(1/num_workers)
        train_loader_list = []
        train_sets = torch.utils.data.random_split(train_data, lengths)
        for i in range(len(train_sets)):
            train_loader_list.append(DataLoader(train_sets[i], 64, shuffle=True))
        test_loader = DataLoader(test_data, 64, shuffle=True)
        return train_loader_list, test_loader

def loadImageNet(path, num_workers):
    '''Loads ImageNet data. Need to specify folder.'''
    with FileLock(os.path.expanduser("~/data.lock")):
        train_path = path
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])
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
            train_loader.append(torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, range(i*num_workers, i*num_workers + lenght)), batch_size=128))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False)
        return train_loader, test_loader

def loadFashionMNIST(num_workers):
    '''Loads FashionMNIST.'''
    with FileLock(os.path.expanduser("~/data.lock")):
        train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                      transforms.Compose([transforms.ToTensor()]))
        train_set_list = []

        for i in range(num_workers):
            train_set_list.append(torch.utils.data.Subset(train_set, range(i*len(train_set)//num_workers, (i + 1)*len(train_set)//num_workers)))
        test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                                      transforms.Compose([transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=128, shuffle=True)
        train_loader_list = []
        for i in train_set_list:
            train_loader_list.append(torch.utils.data.DataLoader(i, batch_size=256, shuffle = True))
    return train_loader_list, test_loader