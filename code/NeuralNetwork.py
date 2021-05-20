import torch
import torch.nn as nn
import numpy as np
import math


# Sigmoid activation function
def sigmoid(input):
    new_input = np.zeros(input.shape)
    for i in range(len(input)):
        new_input[i] = 1 / (1 + math.exp(-1 * input[i]))
    return new_input


# tanh activation function
def tanh(input):
    return np.tanh(input)


class NN():
    def __init__(self, D_in, H, D_out, N):
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.N = N

    def feed_forward(self, x, y):
        model = nn.Sequential(
            nn.Linear(self.D_in, self.H),
            nn.ReLU(),
            nn.Linear(self.H, self.D_out),
            nn.Sigmoid())
        loss_fn = nn.MSELoss(reduction='sum')
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(50):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            print('epoch: ', epoch, ' loss: ', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return


class SimpleCNN(nn.Module):
    def __init__(self, N, D_in, D_out, D2_in, D2_out, D3_out):
        super(SimpleCNN, self).__init__()
        self.N = N
        self.D_in = D_in
        self.D_out = D_out
        self.D2_in = D2_in
        self.D2_out = D2_out
        self.D3_out = D3_out
        self.layer1 = nn.Sequential(nn.Conv2d(self.D_in, self.D_out, kernel_size=3, stride=1),
                                    nn.BatchNorm2d(self.D_out),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(self.D2_in, self.D2_out, kernel_size=3, stride=1),
                                    nn.BatchNorm2d(self.D2_out),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(D3_out, N)

    def feed_forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        model = SimpleCNN(self.N, self.D_in, self.D_out, self.D2_in, self.D2_out, self.D3_out)
        loss_fn = nn.CrossEntropyLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(50):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            print('epoch: ', epoch, ' loss: ', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return out
