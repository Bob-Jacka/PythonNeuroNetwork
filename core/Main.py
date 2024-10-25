from core.NeuralData import NeuroArc
from entities.NNetwork import Network

path_to_image: str = '../core/data/three.png'
network = Network()
network.load_model(NeuroArc.dense)
network.init_test_data(test_mode=True)
#
# network.train_model(after_process_save=True)
# network.work_model(after_process_save=True)
network.predict_number(image_to_recognize=path_to_image)
