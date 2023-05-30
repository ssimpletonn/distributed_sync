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
    def __init__(self, lr, model, momentum):
        self.model = model
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
