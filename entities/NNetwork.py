import os.path

import keras as keras
import matplotlib.image as img
import numpy as np
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
    model_name: str
    save_model_path: str = '../core/saveModel/'
    ext: str = '.keras'
    one_image_to_path: str = ...
    model_arc: NeuroArc

    def __init__(self):
        self.image = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None

    @classmethod
    def init_test_data(cls, test_mode: bool):
        if test_mode:
            (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = keras.datasets.mnist.load_data()
            cls.x_train = cls.x_train.reshape((cls.x_train.shape[0], 28 * 28, 1)).astype(np.float32) / 255
            cls.x_test = cls.x_test.reshape((cls.x_test.shape[0], 28 * 28, 1)).astype(np.float32) / 255
            cls.y_train = to_categorical(cls.y_train)
            cls.y_test = to_categorical(cls.y_test)
        else:
            print('No init data')

    @classmethod
    def create_model(cls, model_arc: NeuroArc = NeuroArc.dense):
        """:param model_arc:  dense | conv | flatten
        """
        cls.model_arc = model_arc
        cls.model_name = model_arc.value
        cls.model = Sequential(name="nnetwork")  # keras.src.models.Functional
        if model_arc == NeuroArc.dense:
            cls.model.add(Input((28, 28, 1)))
            # self.model.add(Dense(784, input_shape=(28 * 28,), activation=tahn))
            cls.model.add(Dense(512, activation=leaky_relu))
            cls.model.add(Dense(256, activation=leaky_relu))
            cls.model.add(Dense(10, activation=softmax))
            cls.model.compile(optimizer=opt, loss=crossentropy,
                              metrics=[accuracy])

        elif model_arc == NeuroArc.conv:
            cls.model.add(Input((28, 28, 1)))
            # self.model.add(Dense(784, input_shape=(28, 28, 1), activation=sigmoid))
            cls.model.add(Conv2D(512, kernel_size=(1, 1), activation=leaky_relu))
            cls.model.add(Conv2D(256, kernel_size=(1, 1), activation=leaky_relu))
            cls.model.add(Conv2D(128, kernel_size=(1, 1), activation=leaky_relu))
            cls.model.add(Dense(10, activation=softmax))
            cls.model.compile(optimizer=opt, loss=crossentropy,
                              metrics=[accuracy])

        elif model_arc == NeuroArc.flatten:
            cls.model.add(Flatten(input_shape=(28 * 28,), data_format='channels_last'))
            cls.model.add(Dense(512, activation=leaky_relu))
            cls.model.add(Dense(256, activation=leaky_relu))
            cls.model.add(Dense(10, activation=softmax))
            cls.model.compile(optimizer=opt, loss=crossentropy,
                              metrics=[accuracy])
        else:
            raise RuntimeError("model type not specified")

    @classmethod
    def save_model(cls):
        cls.model.save(cls.save_model_path + cls.model_name + cls.ext)

    @classmethod
    def load_model(cls, neuro_arc=NeuroArc.dense):
        if os.path.exists(cls.save_model_path) and os.listdir(cls.save_model_path).__len__() != 0:
            list_models = os.listdir(cls.save_model_path)
            if list_models.count(neuro_arc.value) != 0:
                val_to_load = cls.save_model_path + neuro_arc.value + cls.ext
                with open(val_to_load, 'r'):
                    cls.model = keras.models.load_model(val_to_load)
            else:
                cls.create_model(neuro_arc)
        else:
            cls.create_model(neuro_arc)

    @classmethod
    def train_model(cls, after_train_save=False, epochs_to_train=10):
        print('evaluate train: ')
        cls.model.fit(cls.x_test, cls.y_test, epochs=epochs_to_train)
        cls.model.evaluate(cls.x_test, cls.y_test)
        if after_train_save:
            cls.save_model()

    @classmethod
    def work_model(cls, after_work_save=False, epochs_to_work=10):
        print('evaluate work: ')
        cls.model.fit(cls.x_train, cls.y_train, epochs=epochs_to_work)
        cls.model.evaluate(cls.x_train, cls.y_train)
        if after_work_save:
            cls.save_model()

    @classmethod
    def predict_number(cls, image_to_recognize):
        if image_to_recognize is not None:
            cls.one_image_to_path = image_to_recognize
            cls.image = img.imread(cls.one_image_to_path)
            cls.image = cls.image.sum(axis=2)
            cls.image = cls.image.reshape((1, 28 * 28)).astype(np.float32) / 255

            if cls.model is not None and cls.image is not None:
                value = cls.model.predict(x=cls.image, batch_size=1)[0]
                num = 0
                for _ in value:
                    print('#' + num.__str__(), _)
                    num = num + 1
            else:
                print('Please, load model')
        else:
            print('Please specify the image')
