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


class tensorflow_nn:
    model: Sequential
    model_name: str
    model_arc: NeuroArc

    save_model_path: str = '../core/saveModel/'
    ext: str = '.keras'

    @classmethod
    def __init__(cls):
        cls.image_to_recognize = None
        cls.y_test = None
        cls.y_train = None
        cls.x_test = None
        cls.x_train = None

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
            cls.model.add(Dense(784, input_shape=(28 * 28,), activation=sigmoid))
            cls.model.add(Dense(256, activation=leaky_relu))
            cls.model.add(Dense(128, activation=leaky_relu))
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
            # надо чтобы была фигура target.shape=(None, 10)
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
    def save_model(cls) -> None:
        cls.model.save(cls.save_model_path + cls.model_name + cls.ext)

    @classmethod
    def load_model(cls, neuro_arc=NeuroArc.dense) -> None:
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
    def train_model(cls, after_process_save=False, epochs_to_train=10) -> None:
        if (cls.x_train is not None) and (cls.y_train is not None):
            print('evaluate train: ')

            cls.model.fit(cls.x_train, cls.y_train, epochs=epochs_to_train)
            cls.model.evaluate(cls.x_train, cls.y_train)
            if after_process_save:
                cls.save_model()
        else:
            print('please, init test data')

    @classmethod
    def work_model(cls, after_process_save=False, epochs_to_work=10) -> None:
        if (cls.x_test is not None) and (cls.y_test is not None):
            print('evaluate work: ')
            cls.model.fit(cls.x_test, cls.y_test, epochs=epochs_to_work)
            cls.model.evaluate(cls.x_test, cls.y_test)
            if after_process_save:
                cls.save_model()
        else:
            print('please, init work data')

    @classmethod
    def predict_number(cls, image_to_recognize, batch_size=1) -> None:
        weights_before = cls.model.get_weights()  # get weights before recognize
        print('freeze weights')
        try:
            if image_to_recognize is not None:
                cls.image_to_recognize = img.imread(image_to_recognize)
                cls.image_to_recognize = cls.image_to_recognize.sum(axis=2)
                cls.image_to_recognize = cls.image_to_recognize.reshape((1, 28 * 28)).astype(np.float32) / 255

                if cls.model is not None and cls.image_to_recognize is not None:
                    value = cls.model.predict(x=cls.image_to_recognize, batch_size=batch_size)[0]
                    key = 0
                    results = dict()
                    for _ in value:
                        results[key] = _
                        key += 1
                    cls.get_maximum_from_dict(results)

                    print('please enter "yes" or "y" if prediction is right else "no" or "n" if not: ', end='')
                    user_prompt = input().lower()
                    if user_prompt == 'yes' or user_prompt == 'y':
                        cls.model.set_weights(weights_before)  # set frozen weights after recognize
                        print('weights returned')
                    else:
                        print('weights changed')
                else:
                    print('Please, load model')
            else:
                print('Please specify the image')
        except FileNotFoundError as e:
            print('please, make sure that file with image exists')

    @staticmethod
    def get_maximum_from_dict(dictionary: dict) -> None:
        value_tmp: float = 0.0
        key_tmp: int = 0
        for key, val in dictionary.items():
            if value_tmp < val:
                value_tmp = val
                key_tmp = key
        print(f'Neuro net thinks that number from the image is {key_tmp}.')
