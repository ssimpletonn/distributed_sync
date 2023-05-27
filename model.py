import torch
import torchvision


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

def get_model():
    model = torchvision.models.efficientnet_b0(weights=None)
    model.getWeights()


def get_model_from_pkl(path='model.pkl'):

