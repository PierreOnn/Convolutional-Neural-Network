from NeuralNetwork import *


def main():
    # D_in, H, D_out, N = 10, 5, 1, 10
    # nn = NN(D_in, H, D_out, N)
    # x = torch.randn(N, D_in)
    # y = torch.randn(N, D_out)
    # print(nn.feed_forward(x, y))

    N, D_in, D_out, D2_in, D2_out, D3_out = ()
    cnn = SimpleCNN(N, D_in, D_out, D2_in, D2_out, D3_out)
    x = torch.randn()
    y = torch.randn()
    print(cnn.feed_forward(x, y))


if __name__ == '__main__':
    main()
