import os.path

import keras
import matplotlib.image as img
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import *
from keras.src.utils import to_categorical

from core.NeuralData import ActivationFuncs, Metrics
from core.NeuralData import Optimizer
from core.NeuralData import ValuesTypes

leaky_relu: str = ActivationFuncs.LeakyRelu.value
softmax: str = ActivationFuncs.Softmax.value
adam: str = Optimizer.Adam.value
float32: str = ValuesTypes.Float32.value
crossentropy: str = Optimizer.Loss.value
accuracy: str = Metrics.Accuracy.value


class Network:
    model: Sequential
    save_model_path: str = '../core/saveModel/save.keras'
    one_image_to_path: str

    def __init__(self, test_mode=False, one_image_to_recognize: str = None):
        if test_mode:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            self.x_train = self.x_train.reshape((self.x_train.shape[0], 28 * 28)).astype(np.float32) / 255
            self.x_test = self.x_test.reshape((self.x_test.shape[0], 28 * 28)).astype(np.float32) / 255
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
        else:
            self.one_image_to_path = one_image_to_recognize
            self.x_image = img.imread(self.one_image_to_path)
            self.x_image = self.x_image.reshape((1, 3136)).astype(np.float32) / 255

    def __init_model(self):
        self.model.add(Dense(784, input_shape=(28 * 28,)))
        self.model.add(Conv2D(512, activation=leaky_relu))
        self.model.add(Dense(256, activation=leaky_relu))
        self.model.add(Dense(10, activation=softmax))
        self.model.compile(optimizer=adam, loss=crossentropy,
                           metrics=[accuracy])
        return self.model

    def save_model(self):
        self.model.save(self.save_model_path)

    def load_model(self):
        if os.path.exists(self.save_model_path):
            with open(self.save_model_path, 'r'):
                self.model = tf.keras.models.load_model(self.save_model_path)
        else:
            self.model = Sequential()
            self.__init_model()
        return self.model

    def train_model(self, model):
        print('evaluate train: ')
        model.fit(self.x_test, self.y_test, epochs=10)
        model.evaluate(self.x_test, self.y_test)

    def work_model(self, model):
        print('evaluate work: ')
        model.fit(self.x_train, self.y_train, epochs=10)
        model.evaluate(self.x_train, self.y_train)

    def predict_number(self):
        if self.model is not None:
            print(self.model.predict(x=self.x_image))
        else:
            print('Please, load model')
