import matplotlib.pyplot as plt
import numpy as np
def plot_accuracy(accuracy_vals, iterations):
    data = np.squeeze(accuracy_vals)
    plt.plot(data)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Global accuracy")
    plt.legend()
    plt.savefig('accuracy.png')
    plt.clf()

def plot_accuracy_workers(accuracy_workers, iterations, num_workers):
    for i in range(num_workers):
        plt.plot(iterations, accuracy_workers[i])
    plt.xlabel("iter")
    plt.ylabel("accuracy")
    plt.title("Local accuracy")

    plt.legend()
    plt.savefig('localaccuracy.png')

    plt.clf()

def plot_loss_workers(loss_workers, iterations, num_workers):
    for i in range(num_workers):
        plt.plot(iterations, loss_workers[i])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("Local loss")
    plt.legend()
    plt.savefig('localloss.png')
    plt.clf()
