from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(model, train_loader, epochs, lr, device, proximal_mu=0.0, global_params=None):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    model.parameters(), global_params
                ):
                    proximal_term += (
                        (local_param - global_param.to(device)) ** 2
                    ).sum()
                loss += (proximal_mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()


def test(model, test_loader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item() * labels.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return loss / total, correct / total
