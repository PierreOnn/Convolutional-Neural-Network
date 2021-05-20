from NeuralNetwork import *


def main():
    D_in, H, D_out, N = 10, 5, 1, 10
    cnn = CNN(D_in, H, D_out, N)
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    print(cnn.feed_forward(x, y))

if __name__ == '__main__':
    main()
