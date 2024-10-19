from enum import Enum


class ActivationFuncs(Enum):
    Sigmoid = 'sigmoid'
    Tahn = 'tanh'
    Relu = 'relu'
    LeakyRelu = 'leaky_relu'
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


class NeuroArc(Enum):
    conv = 'conv'
    flatten = 'flatten'
    dense = 'dense'
    resNet = 'resnet'
    EfficientNet = 'efficientnet'
    DenseNet = 'densenet'
