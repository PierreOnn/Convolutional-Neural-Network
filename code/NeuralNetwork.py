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


class CNN():
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
            optimizer.zero_grad
            loss.backward()
            optimizer.step()
        return
