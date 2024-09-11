from enum import Enum


class ActivationFuncs(Enum):
    Sigmoid = 'sigmoid'
    Tahn = 'tahn'
    Relu = 'relu'
    LeakyRelu = 'Leaky ReLU'
    ELU = 'elu'
    Softmax = 'softmax'


class Optimizer(Enum):
    Adam = 'adam'
    Rmsprop = "rmsprop"
    Loss = 'categorical_crossentropy'


class ValuesTypes(Enum):
    Float32 = 'float32'

class Metrics(Enum):
    Accuracy = 'accuracy'