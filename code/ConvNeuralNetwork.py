import tensorflow as tf
from tensorflow.keras import layers, models
from wandb.keras import WandbCallback
from preprocess import *
import matplotlib.pyplot as plt
import os


os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(42)
tf.random.set_seed(42)


# TensorFlow implementation CNN
class CNN():
    def __init__(self, lr, lf, epoch, batch):
        super(CNN, self).__init__()
        self.lr = lr
        self.lf = lf
        self.epoch = epoch
        self.batch = batch

    def build(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(7))
        model.summary()
        return model

    def feed_forward(self):
        model = CNN.build(self)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        optimizer = tf.keras.optimizers.Adam(self.lr)
        model.compile(optimizer, self.lf, metrics=['acc'])
        return model

    def evaluate(self):
        model = CNN.feed_forward(self)
        history = model.fit(X_train, y_train, epochs=self.epoch,
                            validation_data=(X_valid, y_valid),
                            batch_size=self.batch,
                            callbacks=[WandbCallback(data_type="image", labels=y_train)])
        loss, accuracy = model.evaluate(X_valid, y_valid)

        plt.plot(history.history['acc'], label='accuracy')
        plt.plot(history.history['val_acc'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        return accuracy
