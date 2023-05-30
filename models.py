import torchvision
import torch
import torch.nn as nn
import types

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

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

def loadEfficientNet(pretrained=True, classifier=True, num_classes=10):
    model = torchvision.models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
    if(classifier):
        model.classifier=nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                  nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True))
    model.get_weights = types.MethodType(get_weights, model)
    model.set_weights = types.MethodType(set_weights, model)
    model.get_gradients = types.MethodType(get_gradients, model)
    model.set_gradients = types.MethodType(set_gradients, model)
    return model

def loadLeNet():
    model = ConvNet()
    return model
