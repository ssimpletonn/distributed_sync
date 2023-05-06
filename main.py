import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np

import ray
from ConvNet import ConvNet, evaluate, get_data_loader


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_vals = []
        self.data_iterator = iter(get_data_loader()[0])


    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        self.loss_vals.append(loss.item())
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def get_loss_vals(self):
        return self.loss_vals


@ray.remote
class Worker(object):
    def __init__(self):
        self.model = ConvNet()
        self.data_iterator = iter(get_data_loader()[0])
        self.test_loader = get_data_loader()[1]
        self.loss_vals = []
        self.acc_vals = []

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.loss_vals.append(loss.item())
        self.acc_vals.append(evaluate(self.model, self.test_loader))
        return self.model.get_gradients()

    def get_accuracy_vals(self):
        return self.acc_vals

    def get_loss_vals(self):
        return self.loss_vals



if __name__ = '__main__':
    iterations = 600
    num_workers = 10
    lr = 0.001
    accuracy_vals = []

    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(lr)
    workers = [Worker.remote() for i in range(num_workers)]

    model = ConvNet()
    test_loader = get_data_loader()[1]

    print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    for i in range(iterations):
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % 10 == 0:
            # Evaluate the current model.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            accuracy_vals.append(accuracy)
            print("Epoch {}: \taccuracy is {:.1f}".format(i / 10 + 1, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    # Clean up Ray resources and processes before the next example.

    epochs = range(1, 61)
    worker1loss = []
    worker2loss = []
    worker1acc = []
    worker2acc = []
    workerss = [[], [], [], [], [], [], [], [], [], []]

    for i in range(num_workers):
        workerss[i].append(
            [ray.get(workers[i].get_accuracy_vals.remote())[::10], ray.get(workers[i].get_loss_vals.remote())[::10]])

    for i in range(num_workers):
        plt.plot(epochs, workerss[i][0][0])

    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Local ccuracy")

    plt.legend()
    plt.show()

    for i in range(num_workers):
        plt.plot(epochs, workerss[i][0][1])

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Local loss")

    plt.legend()
    plt.show()

    plt.plot(epochs, accuracy_vals)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Global accuracy")

    plt.legend()
    plt.show()

    plt.plot(epochs, ray.get(ps.get_loss_vals.remote())[::10])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Global loss")
    plt.legend()
    plt.show()