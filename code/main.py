from ConvNeuralNetwork import *
import wandb


def main():
    run = wandb.init(project='MU-CV-Assignment2',
                     config={
                         "learning_rate": 0.01,
                         "epochs": 10,
                         "batch_size": 64,
                         "loss_function": "sparse_categorical_crossentropy",
                         "architecture": "CNN",
                         "dataset": "FER-2013"
                     },
                     entity="mu_cv_cnn")
    config = wandb.config
    cnn = CNN(config.learning_rate, config.loss_function, config.epochs, config.batch_size)

    accuracy = cnn.evaluate()
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})
    run.join()


if __name__ == '__main__':
    main()
