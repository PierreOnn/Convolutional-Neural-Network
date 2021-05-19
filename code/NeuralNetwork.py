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
    def __init__(self, Wxh, Why, Whh):
        self.Wxh = Wxh
        self.Why = Why
        self.Whh = Whh

    def feed_forward(self, x):
        h_x = np.dot(x, self.Wxh)
        h = tanh(h_x)
        y = np.dot(h, self.Why)
        return y