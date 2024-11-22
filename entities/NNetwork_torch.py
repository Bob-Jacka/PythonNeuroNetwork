import torch.cuda
import torchvision
from torch import T
from torch.nn import Sequential, Linear, ReLU, Dropout, Sigmoid, BCELoss
from torch.optim import Optimizer


class torch_nn:
    model: Sequential
    optimizer: Optimizer
    loss_fn: BCELoss
    data_loader: tuple
    model_name: str

    transform_func = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])])
    save_model_path: str = '../core/saveModel/'
    ext: str = '.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.create_model()
        self.create_optim()

    @classmethod
    def create_model(cls, dropout_rate=0.25):
        cls.model = Sequential(
            Linear(28 * 28, 256),
            ReLU(),  # D
            Linear(256, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 1),
            Dropout(p=dropout_rate),
            Sigmoid()).to(cls.device)

    @classmethod
    def create_optim(cls):
        learning_rate = 0.01
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=learning_rate)
        cls.loss_fn = BCELoss()

    @classmethod
    def get_data_loaders(cls, batch_size=64, shuffle_sets=True):
        """
        first binary_train_loader
        second binary_test_loader
        :return:
        """
        train_set = torchvision.datasets.FashionMNIST(  # A
            root=".",  # B
            train=True,  # C
            download=True,  # D
            transform=cls.transform_func)  # E
        test_set = torchvision.datasets.FashionMNIST(root=".",
                                                     train=False, download=True, transform=cls.transform_func)
        binary_train_set = [x for x in train_set if x[1] in [0, 9]]
        binary_test_set = [x for x in test_set if x[1] in [0, 9]]
        binary_train_loader = torch.utils.data.DataLoader(
            binary_train_set,
            batch_size=batch_size,
            shuffle=shuffle_sets)
        binary_test_loader = torch.utils.data.DataLoader(
            binary_test_set,
            batch_size=batch_size, shuffle=shuffle_sets)
        cls.data_loader = (binary_train_loader, binary_test_loader)

    @classmethod
    def train(cls, epochs=50):
        for i in range(epochs):  # A
            tloss = 0
            for n, (imgs, labels) in enumerate(cls.data_loader[0]):
                imgs = imgs.reshape(-1, 28 * 28)
                imgs = imgs.to(cls.device)
                labels = torch.FloatTensor(
                    [x if x == 0 else 1 for x in labels])
                labels = labels.reshape(-1, 1).to(cls.device)
                preds = cls.model(imgs)
                loss = cls.loss_fn(preds, labels)
                cls.optimizer.zero_grad()
                loss.backward()
                cls.optimizer.step()
                tloss += loss
            tloss = tloss / n
            print(f"at epoch {i}, loss is {tloss}")

    @classmethod
    def save_model(cls):
        scripted = torch.jit.script(cls.model)
        scripted.save(cls.model_name)

    @classmethod
    def load_model(cls):
        scripted = torch.jit.load(cls.save_model_path + cls.model_name, map_location=cls.device)
        scripted.eval()
