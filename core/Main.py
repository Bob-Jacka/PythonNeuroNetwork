from core.NeuralData import NeuroArc
from entities.NNetwork import Network

path_to_image: str = '../core/data/TwoRemaster2.png'
network = Network()
network.load_model(NeuroArc.conv)
network.init_test_data(test_mode=True)
network.train_model(after_train_save=True)
# network.work_model()
# network.predict_number()
