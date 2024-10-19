import os.path

import keras
import matplotlib.image as img
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import *
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical

from core.NeuralData import ActivationFuncs, Metrics, NeuroArc
from core.NeuralData import Optimizer
from core.NeuralData import ValuesTypes

leaky_relu: str = ActivationFuncs.LeakyRelu.value
softmax: str = ActivationFuncs.Softmax.value
sigmoid: str = ActivationFuncs.Sigmoid.value
tahn: str = ActivationFuncs.Tahn.value

float32: str = ValuesTypes.Float32.value
crossentropy: str = Optimizer.Loss.value
accuracy: str = Metrics.Accuracy.value
opt = Adam(learning_rate=0.02)


class Network:
    model: Sequential = ...
    save_model_path: str = '../core/saveModel/save.keras'
    one_image_to_path: str = ...

    def __init__(self, test_mode: bool = False, one_image_to_recognize: str = None):
        if test_mode and one_image_to_recognize is None:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            self.x_train = self.x_train.reshape((self.x_train.shape[0], 28 * 28)).astype(np.float32) / 255
            self.x_test = self.x_test.reshape((self.x_test.shape[0], 28 * 28)).astype(np.float32) / 255
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
        else:
            self.one_image_to_path = one_image_to_recognize
            self.x_image = img.imread(self.one_image_to_path)
            self.x_image = self.x_image.sum(axis=2)
            self.x_image = self.x_image.reshape((1, 28 * 28)).astype(np.float32) / 255

    def init_model(self, model_arc: NeuroArc):
        """:param model_arc:  dense | conv | flatten
        """
        if model_arc == NeuroArc.dense:
            self.model.add(Dense(784, input_shape=(28 * 28,), activation=tahn))
            self.model.add(Dense(512, activation=leaky_relu))
            self.model.add(Dense(256, activation=leaky_relu))
            self.model.add(Dense(10, activation=softmax))
            self.model.compile(optimizer=opt, loss=crossentropy,
                               metrics=[accuracy])

        elif model_arc == NeuroArc.conv:
            self.model.add(Dense(784, input_shape=(28 * 28,), activation=sigmoid))
            self.model.add(Conv2D(512, kernel_size=(1, 28 * 28), activation=leaky_relu))
            self.model.add(Conv2D(256, kernel_size=(1, 28 * 28), activation=leaky_relu))
            self.model.add(Dense(10, activation=softmax))
            self.model.compile(optimizer=opt, loss=crossentropy,
                               metrics=[accuracy])

        elif model_arc == NeuroArc.flatten:
            self.model.add(Flatten(784, input_shape=(28 * 28,), activation=sigmoid))
            self.model.add(Dense(512, activation=leaky_relu))
            self.model.add(Dense(256, activation=leaky_relu))
            self.model.add(Dense(10, activation=softmax))
            self.model.compile(optimizer=opt, loss=crossentropy,
                               metrics=[accuracy])
        else:
            raise RuntimeError("model type not specified")

    def save_model(self):
        self.model.save(self.save_model_path)

    def load_model(self):
        if os.path.exists(self.save_model_path):
            with open(self.save_model_path, 'r'):
                self.model = tf.keras.models.load_model(self.save_model_path)
        else:
            self.model = Sequential(name="nnetwork")
            self.init_model(model_arc=NeuroArc.dense)

    def train_model(self, after_train_save=False):
        print('evaluate train: ')
        self.model.fit(self.x_test, self.y_test, epochs=10)
        self.model.evaluate(self.x_test, self.y_test)
        if after_train_save:
            self.save_model()

    def work_model(self, after_work_save=False):
        print('evaluate work: ')
        self.model.fit(self.x_train, self.y_train, epochs=10)
        self.model.evaluate(self.x_train, self.y_train)
        if after_work_save:
            self.save_model()

    def predict_number(self):
        if self.model is not None and self.x_image is not None:
            value = self.model.predict(x=self.x_image, batch_size=1)[0]
            num = 0
            for _ in value:
                print('#' + num.__str__(), _)
                num = num + 1
        else:
            print('Please, load model')
