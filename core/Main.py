from entities.NNetwork import Network

path_to_file_to_save: str = '../core/data/TwoRemaster2.png'
network = Network(test_mode=False, one_image_to_recognize=path_to_file_to_save)
network.load_model()
# network.train_model(after_train_save=True)
# network.work_model()
network.predict_number()
