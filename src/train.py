from src.model import VanillaCNN
from src.data import load_canoncial_vision_datasets


class Trainer():
    def __init__(self, config):
        self.model = VanillaCNN(**config['model'])
        self.train_dataloader, self.test_dataloader, self.classes = load_canoncial_vision_datasets(**config['data'])